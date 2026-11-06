#pragma once
/**
 * 投资组合管理 (Portfolio)
 *
 * 追踪现金、持仓、总资产，提供仓位计算接口。
 * 支持多股票组合，考虑佣金、印花税、滑点成本。
 */

#include <string>
#include <unordered_map>
#include <vector>

namespace quant {

/// 单只股票持仓
struct Position {
    std::string ts_code;
    double shares = 0.0;        // 持仓股数
    double avg_cost = 0.0;      // 持仓均价
    double market_value = 0.0;  // 当前市值
    double weight = 0.0;        // 占总资产权重
};

/// 成交记录
struct Trade {
    std::string ts_code;
    std::string date;
    double price = 0.0;
    double shares = 0.0;        // 正=买入, 负=卖出
    double commission = 0.0;    // 佣金
    double tax = 0.0;           // 印花税（仅卖出）
    double slippage = 0.0;      // 滑点成本
    double cost = 0.0;          // 总成本
};

class Portfolio {
public:
    Portfolio() = default;

    /// 初始化
    /// @param initial_cash 初始资金
    /// @param commission_rate 佣金比率 (默认万3)
    /// @param stamp_tax_rate 印花税比率 (默认千1，仅卖出收取)
    /// @param slippage_rate 滑点比率 (默认千1)
    void init(double initial_cash,
              double commission_rate = 0.0003,
              double stamp_tax_rate = 0.001,
              double slippage_rate = 0.001);

    // ── 状态查询 ───────────────────────────────────────────

    double cash() const { return cash_; }
    double total_value() const { return total_value_; }
    double market_value() const { return market_value_; }
    double initial_cash() const { return initial_cash_; }
    double daily_return() const { return daily_return_; }

    /// 获取某只股票的持仓
    const Position* get_position(const std::string& ts_code) const;

    /// 获取所有持仓
    const std::unordered_map<std::string, Position>& positions() const {
        return positions_;
    }

    /// 获取当日成交记录
    const std::vector<Trade>& daily_trades() const { return daily_trades_; }

    // ── 交易操作 ───────────────────────────────────────────

    /// 买入
    /// @param ts_code 股票代码
    /// @param price 成交价（已含滑点）
    /// @param shares 买入股数（整手数 = 100的倍数）
    /// @param date 交易日期
    /// @return 实际买入股数（资金不足时可能减少）
    double buy(const std::string& ts_code, double price,
               double shares, const std::string& date);

    /// 卖出
    /// @param ts_code 股票代码
    /// @param price 成交价（已含滑点）
    /// @param shares 卖出股数（正数）
    /// @param date 交易日期
    /// @return 实际卖出的金额
    double sell(const std::string& ts_code, double price,
                double shares, const std::string& date);

    /// 按目标权重调仓（核心方法）
    /// @param target_weights {ts_code → target_weight}
    /// @param prices {ts_code → current_price}
    /// @param date 交易日期
    void rebalance(const std::unordered_map<std::string, double>& target_weights,
                   const std::unordered_map<std::string, double>& prices,
                   const std::string& date);

    /// 更新市值（按最新价格）
    /// @param prices {ts_code → current_price}
    void mark_to_market(const std::unordered_map<std::string, double>& prices);

    /// 新交易日重置（清空当日成交记录等）
    void new_day();

private:
    double cash_ = 0.0;
    double total_value_ = 0.0;
    double market_value_ = 0.0;
    double initial_cash_ = 0.0;
    double daily_return_ = 0.0;
    double prev_total_value_ = 0.0;

    double commission_rate_ = 0.0003;
    double stamp_tax_rate_ = 0.001;
    double slippage_rate_ = 0.001;

    std::unordered_map<std::string, Position> positions_;
    std::vector<Trade> daily_trades_;
    std::vector<Trade> trade_history_;  // 全部历史成交
};

}  // namespace quant
