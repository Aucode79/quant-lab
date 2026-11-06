#pragma once
/**
 * 布林带 (Bollinger Bands)
 *
 * 中轨 = MA(period)
 * 上轨 = MA + k * stddev
 * 下轨 = MA - k * stddev
 *
 * 带宽 = (上轨 - 下轨) / 中轨
 * %B = (close - 下轨) / (上轨 - 下轨)
 */

#include <Eigen/Core>

#include "indicators/ma.h"

namespace quant::indicator {

struct BollingerResult {
    Eigen::VectorXd upper;   // 上轨
    Eigen::VectorXd middle;  // 中轨 (MA)
    Eigen::VectorXd lower;   // 下轨
    Eigen::VectorXd bandwidth; // 带宽百分比
    Eigen::VectorXd percent_b; // %B 指标
};

/// 计算布林带
/// @param data 收盘价序列
/// @param period 周期 (默认20)
/// @param k 标准差倍数 (默认2)
inline BollingerResult bollinger(const Eigen::VectorXd& data,
                                  int period = 20,
                                  double k = 2.0) {
    BollingerResult result;
    result.middle = sma(data, period);

    result.upper.resize(data.size());
    result.lower.resize(data.size());
    result.bandwidth.resize(data.size());
    result.percent_b.resize(data.size());

    for (int i = 0; i < data.size(); ++i) {
        if (i < period - 1) {
            result.upper(i) = std::numeric_limits<double>::quiet_NaN();
            result.lower(i) = std::numeric_limits<double>::quiet_NaN();
            result.bandwidth(i) = std::numeric_limits<double>::quiet_NaN();
            result.percent_b(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        // 计算窗口标准差
        double sum = 0.0, sum_sq = 0.0;
        for (int j = 0; j < period; ++j) {
            double val = data(i - j);
            sum += val;
            sum_sq += val * val;
        }
        double mean = sum / period;
        double variance = (sum_sq / period) - (mean * mean);
        double stddev = (variance > 0) ? std::sqrt(variance) : 0.0;

        double mid = result.middle(i);
        result.upper(i) = mid + k * stddev;
        result.lower(i) = mid - k * stddev;

        // 带宽
        double band_range = result.upper(i) - result.lower(i);
        if (mid > 1e-9) {
            result.bandwidth(i) = band_range / mid;
        } else {
            result.bandwidth(i) = std::numeric_limits<double>::quiet_NaN();
        }

        // %B
        if (band_range > 1e-9) {
            result.percent_b(i) = (data(i) - result.lower(i)) / band_range;
        } else {
            result.percent_b(i) = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return result;
}

}  // namespace quant::indicator
