#pragma once
/**
 * 移动平均线族 (Moving Averages)
 *
 * 向量化实现 — 对 Eigen::VectorXd 进行批量计算。
 * 所有函数针对连续内存布局优化，支持 SIMD 自动向量化。
 *
 * 包含:
 *   - SMA:  简单移动平均
 *   - EMA:  指数移动平均
 *   - WMA:  加权移动平均
 *   - MACD: MACD 指标
 */

#include <Eigen/Core>

namespace quant::indicator {

/// 简单移动平均 (SMA)
/// @param data 输入序列
/// @param period 窗口期数
/// @return 与输入等长的 SMA 序列（前 period-1 个值为 NaN）
inline Eigen::VectorXd sma(const Eigen::VectorXd& data, int period) {
    if (period <= 1) return data;

    Eigen::VectorXd result(data.size());
    double sum = 0.0;

    for (int i = 0; i < data.size(); ++i) {
        sum += data(i);
        if (i >= period) {
            sum -= data(i - period);
        }
        if (i >= period - 1) {
            result(i) = sum / period;
        } else {
            result(i) = std::numeric_limits<double>::quiet_NaN();
        }
    }

    return result;
}

/// 指数移动平均 (EMA)
/// @param data 输入序列
/// @param period 窗口期数
/// @return 与输入等长的 EMA 序列
inline Eigen::VectorXd ema(const Eigen::VectorXd& data, int period) {
    Eigen::VectorXd result(data.size());
    double alpha = 2.0 / (period + 1.0);

    // 初始值 = 第一个有效值
    result(0) = data(0);

    for (int i = 1; i < data.size(); ++i) {
        result(i) = alpha * data(i) + (1.0 - alpha) * result(i - 1);
    }

    return result;
}

/// 加权移动平均 (WMA)
/// @param data 输入序列
/// @param period 窗口期数
inline Eigen::VectorXd wma(const Eigen::VectorXd& data, int period) {
    Eigen::VectorXd result(data.size());
    double denominator = period * (period + 1) / 2.0;

    for (int i = 0; i < data.size(); ++i) {
        if (i < period - 1) {
            result(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        double weighted_sum = 0.0;
        for (int j = 0; j < period; ++j) {
            weighted_sum += data(i - j) * (period - j);
        }
        result(i) = weighted_sum / denominator;
    }

    return result;
}

/// MACD 指标
struct MACDResult {
    Eigen::VectorXd dif;   // DIF = EMA(fast) - EMA(slow)
    Eigen::VectorXd dea;   // DEA = EMA(DIF, signal)
    Eigen::VectorXd hist;  // MACD柱 = 2 * (DIF - DEA)
};

/// 计算 MACD
/// @param data 收盘价序列
/// @param fast 快线周期 (默认12)
/// @param slow 慢线周期 (默认26)
/// @param signal 信号线周期 (默认9)
inline MACDResult macd(const Eigen::VectorXd& data,
                        int fast = 12,
                        int slow = 26,
                        int signal = 9) {
    MACDResult result;
    result.dif = ema(data, fast) - ema(data, slow);
    result.dea = ema(result.dif, signal);
    result.hist = 2.0 * (result.dif - result.dea);
    return result;
}

/// 多列并行计算 SMA（用于批量处理多只股票）
/// @param data 矩阵 [rows=时间, cols=股票数]
/// @param period 窗口期数
inline Eigen::MatrixXd sma_matrix(const Eigen::MatrixXd& data, int period) {
    Eigen::MatrixXd result(data.rows(), data.cols());
    for (int col = 0; col < data.cols(); ++col) {
        result.col(col) = sma(data.col(col), period);
    }
    return result;
}

/// 多列并行计算 EMA
inline Eigen::MatrixXd ema_matrix(const Eigen::MatrixXd& data, int period) {
    Eigen::MatrixXd result(data.rows(), data.cols());
    for (int col = 0; col < data.cols(); ++col) {
        result.col(col) = ema(data.col(col), period);
    }
    return result;
}

}  // namespace quant::indicator
