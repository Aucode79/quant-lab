#pragma once
/**
 * 横截面因子 (Cross-Sectional Factors)
 *
 * 常用选股因子（截面标准化后的 Z-Score 形式）:
 *   - momentum: N日动量 = close / close.shift(N) - 1
 *   - volatility: N日波动率 = std(log_return) * sqrt(252)
 *   - turnover: 换手率因子
 *   - volume_ratio: 量比
 */

#include <Eigen/Core>

namespace quant::indicator {

/// N日动量因子
/// @param data 收盘价矩阵 [rows=时间, cols=股票数]
/// @param period 回看天数
/// @return 动量因子矩阵 [rows=时间, cols=股票数]
inline Eigen::MatrixXd momentum(const Eigen::MatrixXd& data, int period = 20) {
    Eigen::MatrixXd result(data.rows(), data.cols());
    result.setConstant(std::numeric_limits<double>::quiet_NaN());

    for (int col = 0; col < data.cols(); ++col) {
        for (int row = period; row < data.rows(); ++row) {
            double prev = data(row - period, col);
            if (prev > 1e-9) {
                result(row, col) = data(row, col) / prev - 1.0;
            }
        }
    }

    return result;
}

/// N日波动率因子（年化）
/// @param data 收盘价矩阵
/// @param period 回看天数 (默认20)
inline Eigen::MatrixXd volatility(const Eigen::MatrixXd& data, int period = 20) {
    Eigen::MatrixXd result(data.rows(), data.cols());
    result.setConstant(std::numeric_limits<double>::quiet_NaN());

    for (int col = 0; col < data.cols(); ++col) {
        for (int row = period; row < data.rows(); ++row) {
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;

            for (int j = 1; j <= period; ++j) {
                double prev = data(row - j, col);
                double curr = data(row - j + 1, col);
                if (prev > 1e-9) {
                    double log_ret = std::log(curr / prev);
                    sum += log_ret;
                    sum_sq += log_ret * log_ret;
                    count++;
                }
            }

            if (count > 1) {
                double variance = (sum_sq / count) - std::pow(sum / count, 2.0);
                result(row, col) = std::sqrt(std::max(0.0, variance) * 252.0);
            }
        }
    }

    return result;
}

/// Z-Score 标准化（横截面）
/// 将每行的因子值标准化为均值0、标准差1的分布
/// @param factor_values 因子值矩阵 [rows=时间, cols=股票数]
inline Eigen::MatrixXd zscore_cross_section(const Eigen::MatrixXd& factor_values) {
    Eigen::MatrixXd result = factor_values;

    for (int row = 0; row < factor_values.rows(); ++row) {
        double sum = 0.0, sum_sq = 0.0;
        int valid_count = 0;

        for (int col = 0; col < factor_values.cols(); ++col) {
            if (!std::isnan(factor_values(row, col))) {
                sum += factor_values(row, col);
                sum_sq += factor_values(row, col) * factor_values(row, col);
                valid_count++;
            }
        }

        if (valid_count < 2) continue;

        double mean = sum / valid_count;
        double variance = (sum_sq / valid_count) - (mean * mean);
        double stddev = std::sqrt(std::max(0.0, variance));

        if (stddev < 1e-9) {
            for (int col = 0; col < factor_values.cols(); ++col) {
                result(row, col) = 0.0;
            }
        } else {
            for (int col = 0; col < factor_values.cols(); ++col) {
                if (!std::isnan(factor_values(row, col))) {
                    result(row, col) = (factor_values(row, col) - mean) / stddev;
                }
            }
        }
    }

    return result;
}

}  // namespace quant::indicator
