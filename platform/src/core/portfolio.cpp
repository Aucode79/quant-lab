/**
 * 投资组合管理 — 实现
 */

#include "core/portfolio.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <fmt/format.h>

namespace quant {

void Portfolio::init(double initial_cash,
                     double commission_rate,
                     double stamp_tax_rate,
                     double slippage_rate) {
    cash_ = initial_cash;
    total_value_ = initial_cash;
    market_value_ = 0.0;
    initial_cash_ = initial_cash;
    prev_total_value_ = initial_cash;
    daily_return_ = 0.0;

    commission_rate_ = commission_rate;
    stamp_tax_rate_ = stamp_tax_rate;
    slippage_rate_ = slippage_rate;

    positions_.clear();
    daily_trades_.clear();
    trade_history_.clear();
}

const Position* Portfolio::get_position(const std::string& ts_code) const {
    auto it = positions_.find(ts_code);
    if (it != positions_.end()) {
        return &it->second;
    }
    return nullptr;
}

double Portfolio::buy(const std::string& ts_code, double price,
                      double shares, const std::string& date) {
    if (shares <= 0) return 0.0;

    // 整手数对齐（A股 100股/手）
    int lots = static_cast<int>(shares / 100.0);
    if (lots == 0) return 0.0;
    shares = lots * 100.0;

    // 含滑点的成交价
    double exec_price = price * (1.0 + slippage_rate_);

    // 计算成本
    double gross_cost = exec_price * shares;
    double commission = std::max(gross_cost * commission_rate_, 5.0);  // 最低5元
    double total_cost = gross_cost + commission;

    // 检查资金
    if (total_cost > cash_) {
        // 资金不足，按最大可买股数调整
        double max_shares = cash_ / (exec_price * (1.0 + commission_rate_));
        lots = static_cast<int>(max_shares / 100.0);
        if (lots == 0) return 0.0;
        shares = lots * 100.0;

        gross_cost = exec_price * shares;
        commission = std::max(gross_cost * commission_rate_, 5.0);
        total_cost = gross_cost + commission;
    }

    // 更新现金
    cash_ -= total_cost;

    // 更新持仓
    auto& pos = positions_[ts_code];
    double old_cost = pos.avg_cost * pos.shares;
    pos.shares += shares;
    pos.avg_cost = (old_cost + total_cost) / pos.shares;
    pos.ts_code = ts_code;

    // 记录成交
    Trade trade;
    trade.ts_code = ts_code;
    trade.date = date;
    trade.price = exec_price;
    trade.shares = shares;
    trade.commission = commission;
    trade.tax = 0.0;
    trade.slippage = gross_cost - price * shares;
    trade.cost = total_cost;
    daily_trades_.push_back(trade);
    trade_history_.push_back(trade);

    return shares;
}

double Portfolio::sell(const std::string& ts_code, double price,
                       double shares, const std::string& date) {
    auto it = positions_.find(ts_code);
    if (it == positions_.end()) return 0.0;

    shares = std::min(shares, it->second.shares);
    if (shares <= 0) return 0.0;

    // 整手数对齐
    int lots = static_cast<int>(shares / 100.0);
    if (lots == 0) return 0.0;
    shares = lots * 100.0;

    // 含滑点的成交价
    double exec_price = price * (1.0 - slippage_rate_);

    // 计算成本和税费
    double gross_proceeds = exec_price * shares;
    double commission = std::max(gross_proceeds * commission_rate_, 5.0);
    double stamp_tax = gross_proceeds * stamp_tax_rate_;  // A股卖方印花税
    double net_proceeds = gross_proceeds - commission - stamp_tax;

    // 更新现金
    cash_ += net_proceeds;

    // 更新持仓
    it->second.shares -= shares;
    if (it->second.shares < 1.0) {
        positions_.erase(it);  // 清仓
    }

    // 记录成交
    Trade trade;
    trade.ts_code = ts_code;
    trade.date = date;
    trade.price = exec_price;
    trade.shares = -shares;
    trade.commission = commission;
    trade.tax = stamp_tax;
    trade.slippage = price * shares - gross_proceeds;
    trade.cost = commission + stamp_tax + trade.slippage;
    daily_trades_.push_back(trade);
    trade_history_.push_back(trade);

    return net_proceeds;
}

void Portfolio::rebalance(
    const std::unordered_map<std::string, double>& target_weights,
    const std::unordered_map<std::string, double>& prices,
    const std::string& date) {

    double total = total_value_;

    // 第一步：卖出需要减仓的股票
    for (auto& [ts_code, pos] : positions_) {
        double target_w = 0.0;
        auto wit = target_weights.find(ts_code);
        if (wit != target_weights.end()) {
            target_w = wit->second;
        }

        double current_val = pos.market_value;
        double target_val = total * target_w;

        if (current_val > target_val && current_val > 0) {
            auto pit = prices.find(ts_code);
            if (pit != prices.end()) {
                double sell_value = current_val - target_val;
                double sell_shares = sell_value / pit->second;
                sell(ts_code, pit->second, sell_shares, date);
            }
        }
    }

    // 第二步：买入需要加仓的股票
    for (auto& [ts_code, target_w] : target_weights) {
        if (target_w <= 0) continue;

        auto pit = prices.find(ts_code);
        if (pit == prices.end()) continue;

        double current_val = 0.0;
        auto pos_it = positions_.find(ts_code);
        if (pos_it != positions_.end()) {
            current_val = pos_it->second.market_value;
        }

        double target_val = total * target_w;
        if (target_val > current_val) {
            double buy_value = target_val - current_val;
            double buy_shares = buy_value / pit->second;
            buy(ts_code, pit->second, buy_shares, date);
        }
    }
}

void Portfolio::mark_to_market(
    const std::unordered_map<std::string, double>& prices) {

    market_value_ = 0.0;
    for (auto& [ts_code, pos] : positions_) {
        auto pit = prices.find(ts_code);
        if (pit != prices.end()) {
            pos.market_value = pos.shares * pit->second;
            market_value_ += pos.market_value;
        }
    }

    total_value_ = cash_ + market_value_;

    // 更新持仓权重
    if (total_value_ > 0) {
        for (auto& [ts_code, pos] : positions_) {
            pos.weight = pos.market_value / total_value_;
        }
    }

    // 计算日收益率
    if (prev_total_value_ > 0) {
        daily_return_ = total_value_ / prev_total_value_ - 1.0;
    }
}

void Portfolio::new_day() {
    prev_total_value_ = total_value_;
    daily_trades_.clear();
    daily_return_ = 0.0;
}

}  // namespace quant
