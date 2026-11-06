/**
 * 均值回归策略 — 实现
 */

#include "strategy/mean_reversion_strategy.h"

#include "indicators/bollinger.h"
#include "indicators/rsi.h"
#include "utils/arrow_utils.h"

namespace quant {

void MeanReversionStrategy::init(const StrategyContext& context) {
    ctx_ = context;
}

Signal MeanReversionStrategy::on_bar(int bar_idx,
                                     const arrow::Table& kline,
                                     const Portfolio& /*portfolio*/) {
    Signal sig;
    sig.action = Signal::Action::HOLD;

    int min_bars = bb_period_ + 1;
    if (kline.num_rows() < min_bars || bar_idx < min_bars) {
        return sig;
    }

    // 提取收盘价序列
    auto close_vec = util::table_column_to_eigen(kline, "close");
    int n = static_cast<int>(close_vec.size());

    // 计算布林带
    auto bb = indicator::bollinger(close_vec.head(n), bb_period_, bb_k_);

    // 计算 RSI（辅助确认）
    auto rsi_vals = indicator::rsi(close_vec.head(n), 14);

    double last_close = close_vec(n - 1);
    double last_lower = bb.lower(n - 1);
    double last_upper = bb.upper(n - 1);
    double last_rsi = rsi_vals(n - 1);

    if (std::isnan(last_lower) || std::isnan(last_upper)) {
        return sig;
    }

    // 超卖信号：价格触及下轨 + RSI < 30
    if (last_close <= last_lower * 1.02 && last_rsi < 35) {
        sig.action = Signal::Action::BUY;
        sig.weight = 0.20;  // 轻仓试探
        sig.confidence = 0.6;
    }
    // 加仓信号：价格还在下轨附近 + RSI开始回升
    else if (last_close <= (last_lower + bb.upper(n - 1)) * 0.5
             && rsi_vals(n - 1) > rsi_vals(n - 2)
             && last_rsi < 40) {
        sig.action = Signal::Action::BUY;
        sig.weight = 0.15;
        sig.confidence = 0.5;
    }
    // 超买信号：价格突破上轨 + RSI > 70
    else if (last_close >= last_upper * 0.98 && last_rsi > 65) {
        sig.action = Signal::Action::SELL;
        sig.weight = 0.5;  // 减半仓
        sig.confidence = 0.6;
    }
    // 强力卖出：价格远离上轨 + RSI极高
    else if (last_close >= last_upper * 1.05 && last_rsi > 75) {
        sig.action = Signal::Action::SELL;
        sig.weight = 1.0;  // 清仓
        sig.confidence = 0.8;
    }

    return sig;
}

std::vector<ParamDef> MeanReversionStrategy::params() const {
    return {
        {"bb_period", 10.0, 50.0, 2.0, 20.0},
        {"bb_k", 1.5, 3.0, 0.25, 2.0},
    };
}

}  // namespace quant
