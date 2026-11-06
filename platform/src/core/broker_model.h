#pragma once
/**
 * 交易成本模型 (Broker Model)
 *
 * 模拟真实交易中的各项成本。
 * A股交易成本:
 *   - 佣金: 万2.5 ~ 万3（双边，最低5元）
 *   - 印花税: 千1（仅卖出，2023年8月起减半为千0.5）
 *   - 过户费: 万0.1（双边）
 *   - 滑点: 视市场流动性而定
 */

#include <cmath>
#include <string>

namespace quant {

struct BrokerConfig {
    double commission_rate = 0.00025;   // 万2.5 佣金
    double min_commission = 5.0;        // 最低佣金 5元
    double stamp_tax_rate = 0.0005;     // 千0.5 印花税（2023减半后）
    double transfer_fee = 0.00001;     // 万0.1 过户费
    double slippage_rate = 0.001;      // 千1 滑点
    double min_slippage = 0.01;        // 最小滑点 1分钱
};

/// 交易方向
enum class OrderSide { BUY, SELL };

/// 订单执行结果
struct ExecutionResult {
    double exec_price = 0.0;      // 实际成交价
    double commission = 0.0;      // 佣金
    double stamp_tax = 0.0;       // 印花税
    double transfer_fee = 0.0;    // 过户费
    double slippage_cost = 0.0;   // 滑点成本
    double total_cost = 0.0;      // 总交易成本

    /// 总费率（成本 / 成交金额）
    double total_rate() const {
        double gross = exec_price * std::abs(total_cost);
        if (gross < 1e-9) return 0.0;
        return total_cost / (total_cost + gross - total_cost);
    }
};

class BrokerModel {
public:
    explicit BrokerModel(const BrokerConfig& config = BrokerConfig{})
        : config_(config) {}

    /// 计算模拟成交价
    /// @param signal_price 信号价格（通常是收盘价）
    /// @param side 买卖方向
    /// @return 含滑点的模拟成交价
    double exec_price(double signal_price, OrderSide side) const {
        double slippage = signal_price * config_.slippage_rate;
        slippage = std::max(slippage, config_.min_slippage);

        if (side == OrderSide::BUY) {
            return signal_price + slippage;
        } else {
            return signal_price - slippage;
        }
    }

    /// 计算完整执行结果
    ExecutionResult calculate(double signal_price, double shares,
                              OrderSide side) const;

    /// 更新配置
    void set_config(const BrokerConfig& config) { config_ = config; }
    const BrokerConfig& config() const { return config_; }

private:
    BrokerConfig config_;
};

// ── Inline 实现 ───────────────────────────────────────────

inline ExecutionResult BrokerModel::calculate(
    double signal_price, double shares, OrderSide side) const {

    ExecutionResult result;
    result.exec_price = exec_price(signal_price, side);

    double gross = result.exec_price * shares;

    // 佣金（双边）
    result.commission = std::max(gross * config_.commission_rate,
                                  config_.min_commission);

    // 印花税（仅卖出）
    result.stamp_tax = (side == OrderSide::SELL)
                       ? gross * config_.stamp_tax_rate
                       : 0.0;

    // 过户费（双边）
    result.transfer_fee = gross * config_.transfer_fee;

    // 滑点成本
    result.slippage_cost = std::abs(result.exec_price - signal_price) * shares;

    // 总成本
    result.total_cost = result.commission + result.stamp_tax
                        + result.transfer_fee + result.slippage_cost;

    return result;
}

}  // namespace quant
