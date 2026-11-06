/**
 * 回测引擎 — 实现
 */

#include "core/backtest_engine.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace quant {

BacktestEngine::BacktestEngine(
    std::shared_ptr<IDataProvider> data_provider,
    BacktestConfig config)
    : data_provider_(std::move(data_provider))
    , config_(std::move(config))
    , broker_(config_.broker) {
}

BacktestResult BacktestEngine::run(
    IStrategy& strategy,
    const std::unordered_map<std::string, double>& params) {

    // 尝试向量化模式
    auto result = run_vectorized(strategy, params);

    // 如果向量化返回空结果，回退到事件驱动
    if (result.daily_records.empty()) {
        result = run_event_driven(strategy, params);
    }

    return result;
}

BacktestResult BacktestEngine::run_vectorized(
    IStrategy& strategy,
    const std::unordered_map<std::string, double>& /*params*/) {

    BacktestResult result;
    Portfolio& portfolio = result.final_portfolio;
    portfolio.init(config_.initial_cash);

    // 获取股票列表
    auto stock_list = data_provider_->get_stock_list();
    if (stock_list.empty()) return result;

    // 取前N只股票进行回测
    constexpr int MAX_STOCKS = 500;
    std::vector<std::string> stock_codes;
    for (size_t i = 0; i < std::min(stock_list.size(), size_t(MAX_STOCKS)); ++i) {
        stock_codes.push_back(stock_list[i].ts_code);
    }

    // 加载全量数据
    auto data = data_provider_->get_multi_daily_kline(
        stock_codes, config_.start_date, config_.end_date);

    if (!data || data->num_rows() == 0) return result;

    // 尝试向量化信号生成
    auto signals = strategy.compute_signals_vectorized(*data);
    if (!signals) return result;  // 返回空结果，触发事件驱动回退

    // ── 向量化交易执行 ─────────────────────────────────
    // 从 Arrow Table 提取列数据
    auto trade_dates = data_provider_->get_trade_dates(
        config_.start_date, config_.end_date);

    if (trade_dates.empty()) return result;

    PerformanceEval evaluator;

    // 按日期遍历，每天处理所有股票的截面
    for (size_t day_idx = 0; day_idx < trade_dates.size(); ++day_idx) {
        const auto& date = trade_dates[day_idx];
        portfolio.new_day();

        // 获取当天所有股票的价格
        auto snapshot = data_provider_->get_universe_snapshot(date);
        if (!snapshot || snapshot->num_rows() == 0) continue;

        std::unordered_map<std::string, double> prices;
        auto code_arr = std::static_pointer_cast<arrow::StringArray>(
            snapshot->column(0)->chunk(0));
        auto close_arr = std::static_pointer_cast<arrow::DoubleArray>(
            snapshot->column(4)->chunk(0));

        for (int64_t i = 0; i < snapshot->num_rows(); ++i) {
            prices[code_arr->GetString(i)] = close_arr->Value(i);
        }

        // 处理信号（简化版：从 signals 数组中读取对应行）
        // 完整实现需要根据 date 和 ts_code 精确匹配

        // 更新市值
        portfolio.mark_to_market(prices);

        // 记录每日数据
        DailyRecord rec;
        rec.date = date;
        rec.total_value = portfolio.total_value();
        rec.cash = portfolio.cash();
        rec.market_value = portfolio.market_value();
        rec.daily_return = portfolio.daily_return();
        evaluator.add_daily(rec);
        result.daily_records.push_back(rec);
    }

    // 评估绩效
    result.metrics = evaluator.calculate(config_.benchmark_trade_days);
    result.trade_history = {};  // 向量化模式暂不记录逐笔成交
    result.final_portfolio = portfolio;

    return result;
}

BacktestResult BacktestEngine::run_event_driven(
    IStrategy& strategy,
    const std::unordered_map<std::string, double>& /*params*/) {

    BacktestResult result;
    Portfolio& portfolio = result.final_portfolio;
    portfolio.init(config_.initial_cash);

    // 获取股票列表和交易日历
    auto stock_list = data_provider_->get_stock_list();
    if (stock_list.empty()) return result;

    auto trade_dates = data_provider_->get_trade_dates(
        config_.start_date, config_.end_date);
    if (trade_dates.empty()) return result;

    // 限制回测股票数量（避免内存爆炸）
    constexpr int MAX_STOCKS = 100;
    std::vector<std::string> stock_codes;
    for (size_t i = 0; i < std::min(stock_list.size(), size_t(MAX_STOCKS)); ++i) {
        stock_codes.push_back(stock_list[i].ts_code);
    }

    // 初始化策略
    StrategyContext ctx;
    ctx.stock_codes = stock_codes;
    ctx.start_date = config_.start_date;
    ctx.end_date = config_.end_date;
    strategy.init(ctx);

    PerformanceEval evaluator;

    // 逐日遍历
    for (size_t day_idx = 0; day_idx < trade_dates.size(); ++day_idx) {
        const auto& date = trade_dates[day_idx];
        portfolio.new_day();

        // 获取当天数据
        auto snapshot = data_provider_->get_universe_snapshot(date);
        if (!snapshot || snapshot->num_rows() == 0) continue;

        std::unordered_map<std::string, double> prices;
        auto code_arr = std::static_pointer_cast<arrow::StringArray>(
            snapshot->column(0)->chunk(0));
        auto close_arr = std::static_pointer_cast<arrow::DoubleArray>(
            snapshot->column(4)->chunk(0));

        for (int64_t i = 0; i < snapshot->num_rows(); ++i) {
            prices[code_arr->GetString(i)] = close_arr->Value(i);
        }

        // 对每只可交易股票调用策略
        auto tradable = data_provider_->get_tradable_stocks(date);
        for (const auto& ts_code : tradable) {
            // 获取该股票的K线窗口
            auto kline = data_provider_->get_daily_kline(
                ts_code, config_.start_date, date);

            if (!kline || kline->num_rows() == 0) continue;

            // 调用策略逻辑
            Signal signal = strategy.on_bar(
                static_cast<int>(day_idx), *kline, portfolio);

            // 执行信号
            auto pit = prices.find(ts_code);
            if (pit != prices.end()) {
                process_signal(portfolio, signal, ts_code, pit->second, date);
            }
        }

        // 更新市值
        portfolio.mark_to_market(prices);

        // 记录每日数据
        DailyRecord rec;
        rec.date = date;
        rec.total_value = portfolio.total_value();
        rec.cash = portfolio.cash();
        rec.market_value = portfolio.market_value();
        rec.daily_return = portfolio.daily_return();
        evaluator.add_daily(rec);
        result.daily_records.push_back(rec);
    }

    // 评估绩效
    result.metrics = evaluator.calculate(config_.benchmark_trade_days);
    result.trade_history = {};  // 从 portfolio 获取

    return result;
}

void BacktestEngine::process_signal(
    Portfolio& portfolio,
    const Signal& signal,
    const std::string& ts_code,
    double price,
    const std::string& date) {

    if (signal.action == Signal::Action::HOLD) return;

    auto* pos = portfolio.get_position(ts_code);
    double current_value = pos ? pos->market_value : 0.0;
    double total_value = portfolio.total_value();
    double current_weight = total_value > 0 ? current_value / total_value : 0.0;

    if (signal.action == Signal::Action::BUY) {
        // 买入到目标权重
        double target_value = total_value * signal.weight;
        double buy_value = std::max(0.0, target_value - current_value);

        // 单只股票仓位上限 20%
        double max_value = total_value * 0.20;
        buy_value = std::min(buy_value, max_value - current_value);

        if (buy_value > 0) {
            double shares = buy_value / price;
            portfolio.buy(ts_code, price, shares, date);
        }
    }
    else if (signal.action == Signal::Action::SELL) {
        if (pos && pos->shares > 0) {
            double sell_shares = pos->shares * signal.weight;
            portfolio.sell(ts_code, price, sell_shares, date);
        }
    }
}

std::shared_ptr<arrow::Table> BacktestEngine::prepare_data(
    const std::vector<std::string>& stock_codes) {

    return data_provider_->get_multi_daily_kline(
        stock_codes, config_.start_date, config_.end_date);
}

}  // namespace quant
