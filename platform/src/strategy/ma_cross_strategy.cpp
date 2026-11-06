/**
 * 双均线交叉策略 — 实现
 */

#include "strategy/ma_cross_strategy.h"

#include <arrow/api.h>           // NOLINT
#include <arrow/compute/api.h>   // NOLINT

#include "indicators/ma.h"
#include "utils/arrow_utils.h"

namespace quant {

void MACrossStrategy::init(const StrategyContext& context) {
    ctx_ = context;
}

Signal MACrossStrategy::on_bar(int bar_idx,
                               const arrow::Table& kline,
                               const Portfolio& /*portfolio*/) {
    Signal sig;
    sig.action = Signal::Action::HOLD;

    // 需要至少 slow_period_ + 1 根K线
    int min_bars = slow_period_ + 1;
    if (kline.num_rows() < min_bars || bar_idx < min_bars) {
        return sig;
    }

    // 提取收盘价序列
    auto close_vec = util::table_column_to_eigen(kline, "close");

    // 计算当前和前一天的均线
    int n = static_cast<int>(close_vec.size());
    auto ma_fast = indicator::sma(close_vec.head(n), fast_period_);
    auto ma_slow = indicator::sma(close_vec.head(n), slow_period_);

    if (std::isnan(ma_fast(n - 1)) || std::isnan(ma_slow(n - 1)) ||
        std::isnan(ma_fast(n - 2)) || std::isnan(ma_slow(n - 2))) {
        return sig;
    }

    // 金叉：快线从下方上穿慢线
    bool golden_cross = (ma_fast(n - 2) <= ma_slow(n - 2))
                     && (ma_fast(n - 1) > ma_slow(n - 1));

    // 死叉：快线从上方下穿慢线
    bool dead_cross = (ma_fast(n - 2) >= ma_slow(n - 2))
                   && (ma_fast(n - 1) < ma_slow(n - 1));

    if (golden_cross) {
        sig.action = Signal::Action::BUY;
        sig.weight = 0.95;  // 95%仓位
        sig.confidence = 0.7;
    } else if (dead_cross) {
        sig.action = Signal::Action::SELL;
        sig.weight = 1.0;   // 清仓
        sig.confidence = 0.7;
    }

    return sig;
}

std::vector<ParamDef> MACrossStrategy::params() const {
    return {
        {"fast_period", 3.0, 20.0, 1.0, 5.0},
        {"slow_period", 10.0, 60.0, 1.0, 20.0},
    };
}

}  // namespace quant
