#pragma once
/**
 * RSI (相对强弱指标)
 *
 * RSI = 100 - 100 / (1 + RS)
 * RS = 平均涨幅 / 平均跌幅 (over N periods)
 *
 * 使用 Wilder's smoothing 方法:
 *   首期简单平均，后续使用平滑公式
 */

#include <Eigen/Core>

namespace quant::indicator {

/// 计算 RSI
/// @param data 收盘价序列
/// @param period 周期 (默认14)
/// @return RSI 序列
inline Eigen::VectorXd rsi(const Eigen::VectorXd& data, int period = 14) {
    Eigen::VectorXd result(data.size());
    result.setConstant(std::numeric_limits<double>::quiet_NaN());

    if (data.size() <= period) return result;

    // 计算涨跌幅
    Eigen::VectorXd changes = data.tail(data.size() - 1)
                              - data.head(data.size() - 1);

    // Wilder's smoothing
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    // 初始周期简单平均
    for (int i = 0; i < period; ++i) {
        if (changes(i) > 0) avg_gain += changes(i);
        else avg_loss -= changes(i);  // changes(i)为负，取绝对值
    }
    avg_gain /= period;
    avg_loss /= period;

    // 首次 RSI
    int idx = period;
    if (avg_loss < 1e-12) {
        result(idx) = 100.0;
    } else {
        double rs = avg_gain / avg_loss;
        result(idx) = 100.0 - 100.0 / (1.0 + rs);
    }

    // 后续采用平滑
    for (int i = period + 1; i < data.size(); ++i) {
        double change = changes(i - 1);

        double gain = (change > 0) ? change : 0.0;
        double loss = (change < 0) ? -change : 0.0;

        avg_gain = (avg_gain * (period - 1) + gain) / period;
        avg_loss = (avg_loss * (period - 1) + loss) / period;

        if (avg_loss < 1e-12) {
            result(i) = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            result(i) = 100.0 - 100.0 / (1.0 + rs);
        }
    }

    return result;
}

}  // namespace quant::indicator
