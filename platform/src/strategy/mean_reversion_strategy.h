#pragma once
/**
 * 均值回归策略 (Mean Reversion Strategy)
 *
 * 基于布林带:
 *   - 收盘价跌破下轨 → 超卖 → 买入
 *   - 收盘价突破上轨 → 超买 → 卖出
 *
 * 可优化参数:
 *   - bb_period: 布林带周期 (默认20)
 *   - bb_k: 标准差倍数 (默认2.0)
 */

#include "strategy/strategy_base.h"

namespace quant {

class MeanReversionStrategy : public IStrategy {
public:
    explicit MeanReversionStrategy(int bb_period = 20, double bb_k = 2.0)
        : bb_period_(bb_period), bb_k_(bb_k) {}

    std::string name() const override {
        return "MeanReversion";
    }

    void init(const StrategyContext& context) override;
    Signal on_bar(int bar_idx, const arrow::Table& market_data,
                  const Portfolio& portfolio) override;
    std::vector<ParamDef> params() const override;

private:
    int bb_period_;
    double bb_k_;
};

}  // namespace quant
