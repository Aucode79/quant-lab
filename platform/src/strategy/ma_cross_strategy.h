#pragma once
/**
 * 双均线交叉策略 (MA Cross Strategy)
 *
 * 经典趋势跟踪策略:
 *   - 快线上穿慢线 → 全仓买入
 *   - 快线下穿慢线 → 全部卖出
 *
 * 可优化参数:
 *   - fast_period: 快线周期 (默认5)
 *   - slow_period: 慢线周期 (默认20)
 */

#include "strategy/strategy_base.h"

namespace quant {

class MACrossStrategy : public IStrategy {
public:
    explicit MACrossStrategy(int fast_period = 5, int slow_period = 20)
        : fast_period_(fast_period), slow_period_(slow_period) {}

    std::string name() const override {
        return "MACross";
    }

    void init(const StrategyContext& context) override;
    Signal on_bar(int bar_idx, const arrow::Table& market_data,
                  const Portfolio& portfolio) override;
    std::vector<ParamDef> params() const override;

private:
    int fast_period_;
    int slow_period_;
};

}  // namespace quant
