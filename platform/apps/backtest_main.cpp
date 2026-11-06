/**
 * 单策略回测入口
 *
 * Usage:
 *   ./backtest_main --data ../data --strategy macross
 *   ./backtest_main --data ../data --strategy macross --fast 5 --slow 20
 *   ./backtest_main --data ../data --strategy meanrev --bb 20 --k 2.0
 */

#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "core/backtest_engine.h"
#include "core/duckdb_provider.h"
#include "strategy/ma_cross_strategy.h"
#include "strategy/mean_reversion_strategy.h"

using namespace quant;

int main(int argc, char* argv[]) {
    CLI::App app{"Quant System - Single Strategy Backtest"};

    // ── CLI 参数 ──────────────────────────────────────
    std::string data_dir = "../data";
    std::string db_path = ":memory:";
    std::string strategy_name = "macross";
    std::string start_date = "2020-01-01";
    std::string end_date = "2024-12-31";
    double initial_cash = 1'000'000.0;

    // 策略参数
    int fast_period = 5;
    int slow_period = 20;
    int bb_period = 20;
    double bb_k = 2.0;

    app.add_option("--data", data_dir, "数据目录路径")->required();
    app.add_option("--db", db_path, "DuckDB 数据库路径 (默认 :memory:)");
    app.add_option("--strategy", strategy_name, "策略名称 (macross, meanrev)");
    app.add_option("--start", start_date, "回测起始日期");
    app.add_option("--end", end_date, "回测结束日期");
    app.add_option("--cash", initial_cash, "初始资金");
    app.add_option("--fast", fast_period, "快线周期 (双均线)");
    app.add_option("--slow", slow_period, "慢线周期 (双均线)");
    app.add_option("--bb", bb_period, "布林带周期 (均值回归)");
    app.add_option("--k", bb_k, "布林带标准差倍数 (均值回归)");

    CLI11_PARSE(app, argc, argv);

    // ── 启动 ──────────────────────────────────────────
    fmt::print("╔══════════════════════════════════════════╗\n");
    fmt::print("║   Quant System - 本地量化回测系统        ║\n");
    fmt::print("╚══════════════════════════════════════════╝\n\n");

    try {
        // 创建数据提供者
        auto provider = std::make_shared<DuckDBProvider>(db_path, data_dir);
        fmt::print("📊 数据源: {}\n", data_dir);

        // 配置回测引擎
        BacktestConfig config;
        config.start_date = start_date;
        config.end_date = end_date;
        config.initial_cash = initial_cash;
        config.benchmark_trade_days = 252;

        BacktestEngine engine(provider, config);

        // 创建策略
        std::unique_ptr<IStrategy> strategy;

        if (strategy_name == "macross") {
            strategy = std::make_unique<MACrossStrategy>(fast_period, slow_period);
            fmt::print("📈 策略: 双均线交叉 (fast={}, slow={})\n",
                       fast_period, slow_period);
        } else if (strategy_name == "meanrev") {
            strategy = std::make_unique<MeanReversionStrategy>(bb_period, bb_k);
            fmt::print("📈 策略: 均值回归 (BB period={}, k={})\n",
                       bb_period, bb_k);
        } else {
            fmt::print("❌ 未知策略: {}\n", strategy_name);
            fmt::print("   可用策略: macross, meanrev\n");
            return 1;
        }

        fmt::print("📅 回测区间: {} ~ {}\n", start_date, end_date);
        fmt::print("💰 初始资金: {:,.0f}\n\n", initial_cash);

        // 运行回测
        fmt::print("⏳ 正在回测...\n");
        auto result = engine.run(*strategy);

        // ── 输出结果 ──────────────────────────────────
        fmt::print("\n");
        fmt::print("┌─────────────────────────────────────────┐\n");
        fmt::print("│              回测结果                    │\n");
        fmt::print("├─────────────────────────────────────────┤\n");

        const auto& m = result.metrics;
        fmt::print("│ 总收益率:     {:>8.2f}%                  │\n",
                   m.total_return * 100);
        fmt::print("│ 年化收益率:   {:>8.2f}%                  │\n",
                   m.annual_return * 100);
        fmt::print("│ 最大回撤:     {:>8.2f}%                  │\n",
                   m.max_drawdown * 100);
        fmt::print("│ 年化波动率:   {:>8.2f}%                  │\n",
                   m.annual_volatility * 100);
        fmt::print("│ 夏普比率:     {:>8.3f}                   │\n",
                   m.sharpe_ratio);
        fmt::print("│ 卡玛比率:     {:>8.3f}                   │\n",
                   m.calmar_ratio);
        fmt::print("│ 索提诺比率:   {:>8.3f}                   │\n",
                   m.sortino_ratio);
        fmt::print("│ 胜率:         {:>8.1f}%                  │\n",
                   m.win_rate * 100);
        fmt::print("│ 总交易次数:   {:>8d}                     │\n",
                   m.total_trades);
        fmt::print("│ 盈亏比:       {:>8.2f}                   │\n",
                   m.profit_factor);
        fmt::print("└─────────────────────────────────────────┘\n");

        // 最终资金
        if (!result.daily_records.empty()) {
            double final_value = result.daily_records.back().total_value;
            fmt::print("\n💰 最终资金: {:,.2f} (初始: {:,.2f})\n",
                       final_value, initial_cash);
        }

    } catch (const std::exception& e) {
        fmt::print("❌ 回测失败: {}\n", e.what());
        return 1;
    }

    return 0;
}
