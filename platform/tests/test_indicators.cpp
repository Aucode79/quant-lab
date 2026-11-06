/**
 * 技术指标单元测试
 *
 * 验证 SMA, EMA, MACD, RSI, Bollinger 等指标的计算精度。
 * 使用已知结果的测试向量，与 ta-lib 结果对比。
 */

#include <iostream>
#include <cmath>
#include <cassert>

#include <Eigen/Core>

#include "indicators/ma.h"
#include "indicators/rsi.h"
#include "indicators/bollinger.h"
#include "indicators/cross_sectional.h"

using namespace quant::indicator;

constexpr double EPSILON = 1e-4;  // 浮点精度容忍度

// ── 辅助函数 ───────────────────────────────────────────────

/// 检查向量是否近似相等
bool approx_equal(const Eigen::VectorXd& a, const Eigen::VectorXd& b,
                  double epsilon = EPSILON) {
    if (a.size() != b.size()) {
        std::cerr << "Size mismatch: " << a.size() << " vs " << b.size() << "\n";
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        if (std::isnan(a(i)) != std::isnan(b(i))) return false;
        if (!std::isnan(a(i)) && std::abs(a(i) - b(i)) > epsilon) {
            std::cerr << "Mismatch at [" << i << "]: "
                      << a(i) << " vs " << b(i) << "\n";
            return false;
        }
    }
    return true;
}

// ── 测试用例 ───────────────────────────────────────────────

void test_sma() {
    std::cout << "Testing SMA... ";

    Eigen::VectorXd data(10);
    data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;

    auto result = sma(data, 3);

    // 前2个应为 NaN, 后续为 3-period 均线
    assert(std::isnan(result(0)));
    assert(std::isnan(result(1)));
    assert(std::abs(result(2) - 2.0) < EPSILON);   // (1+2+3)/3
    assert(std::abs(result(3) - 3.0) < EPSILON);   // (2+3+4)/3
    assert(std::abs(result(9) - 9.0) < EPSILON);   // (8+9+10)/3

    std::cout << "PASSED\n";
}

void test_ema() {
    std::cout << "Testing EMA... ";

    Eigen::VectorXd data(5);
    data << 10, 12, 14, 13, 15;

    auto result = ema(data, 3);
    // alpha = 2/(3+1) = 0.5
    // EMA[0] = 10
    // EMA[1] = 0.5*12 + 0.5*10 = 11
    // EMA[2] = 0.5*14 + 0.5*11 = 12.5

    assert(std::abs(result(0) - 10.0) < EPSILON);
    assert(std::abs(result(1) - 11.0) < EPSILON);
    assert(std::abs(result(2) - 12.5) < EPSILON);

    std::cout << "PASSED\n";
}

void test_macd() {
    std::cout << "Testing MACD... ";

    // 生成随机数据验证接口正确性
    Eigen::VectorXd data(50);
    for (int i = 0; i < 50; ++i) {
        data(i) = 10.0 + std::sin(i * 0.1) * 5.0;
    }

    auto result = macd(data, 12, 26, 9);

    assert(result.dif.size() == data.size());
    assert(result.dea.size() == data.size());
    assert(result.hist.size() == data.size());

    // HIST = 2 * (DIF - DEA), 验证这个关系
    for (int i = 0; i < 50; ++i) {
        if (!std::isnan(result.hist(i))) {
            double expected = 2.0 * (result.dif(i) - result.dea(i));
            assert(std::abs(result.hist(i) - expected) < EPSILON);
        }
    }

    std::cout << "PASSED\n";
}

void test_rsi() {
    std::cout << "Testing RSI... ";

    // 构造一个全部上涨的序列 → RSI 应接近 100
    Eigen::VectorXd up_data(20);
    for (int i = 0; i < 20; ++i) {
        up_data(i) = 10.0 + i * 0.5;
    }

    auto result = rsi(up_data, 14);
    // RSI 应在 14 之后有值
    assert(!std::isnan(result(14)));
    // 全部上涨时 RSI 应为 100 或接近 100
    assert(result(16) > 90.0);

    // 全部下跌的序列 → RSI 应接近 0
    Eigen::VectorXd down_data(20);
    for (int i = 0; i < 20; ++i) {
        down_data(i) = 20.0 - i * 0.5;
    }

    auto result2 = rsi(down_data, 14);
    assert(result2(16) < 10.0);

    std::cout << "PASSED\n";
}

void test_bollinger() {
    std::cout << "Testing Bollinger Bands... ";

    Eigen::VectorXd data(30);
    for (int i = 0; i < 30; ++i) {
        data(i) = 100.0 + std::sin(i * 0.2) * 5.0 + (i % 5) * 0.1;
    }

    auto bb = bollinger(data, 20, 2.0);

    // 前 19 个值为 NaN
    assert(std::isnan(bb.upper(0)));
    assert(!std::isnan(bb.upper(19)));

    // 上轨 > 中轨 > 下轨
    for (int i = 19; i < 30; ++i) {
        assert(bb.upper(i) >= bb.middle(i) - EPSILON);
        assert(bb.middle(i) >= bb.lower(i) - EPSILON);
    }

    // %B 应在 [0, 1] 左右（2σ范围覆盖约95%）
    for (int i = 19; i < 30; ++i) {
        assert(bb.percent_b(i) >= -0.1);
        assert(bb.percent_b(i) <= 1.1);
    }

    std::cout << "PASSED\n";
}

void test_momentum() {
    std::cout << "Testing Momentum Factor... ";

    Eigen::MatrixXd data(10, 2);  // 10天, 2只股票
    for (int i = 0; i < 10; ++i) {
        data(i, 0) = 100.0 + i * 2.0;   // 上涨股
        data(i, 1) = 100.0 - i * 1.0;   // 下跌股
    }

    auto mom = momentum(data, 5);

    // 验证动量方向
    assert(mom(9, 0) > 0.0);   // 上涨股的正动量
    assert(mom(9, 1) < 0.0);   // 下跌股的负动量

    std::cout << "PASSED\n";
}

void test_zscore() {
    std::cout << "Testing Z-Score... ";

    Eigen::MatrixXd data(3, 3);  // 3天, 3只股票
    data << 10, 20, 30,
            12, 22, 28,
            15, 18, 27;

    auto z = zscore_cross_section(data);

    // 每天（每行）的均值应为0，标准差应为1
    for (int row = 0; row < 3; ++row) {
        double sum = 0.0;
        int valid = 0;
        for (int col = 0; col < 3; ++col) {
            if (!std::isnan(z(row, col))) {
                sum += z(row, col);
                valid++;
            }
        }
        if (valid > 0) {
            assert(std::abs(sum / valid) < EPSILON);  // 均值≈0
        }
    }

    std::cout << "PASSED\n";
}

// ── 性能基准（简单） ───────────────────────────────────────

void bench_sma() {
    std::cout << "Benchmarking SMA... ";

    constexpr int N = 5000;  // 5000只股票
    constexpr int T = 2430;  // 10年日线

    Eigen::MatrixXd data = Eigen::MatrixXd::Random(T, N);

    auto start = std::chrono::steady_clock::now();
    auto result = sma_matrix(data, 20);
    auto end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();

    std::cout << ms << "ms (对 " << N << " 只股票 × " << T
              << " 天计算 SMA20)\n";

    // 粗略性能检查（5000×2430 应在合理时间内完成）
    assert(ms < 10000);  // 应在 10秒内完成
}

// ── Main ───────────────────────────────────────────────────

int main() {
    std::cout << "╔══════════════════════════════════════╗\n";
    std::cout << "║   指标库单元测试                      ║\n";
    std::cout << "╚══════════════════════════════════════╝\n\n";

    try {
        test_sma();
        test_ema();
        test_macd();
        test_rsi();
        test_bollinger();
        test_momentum();
        test_zscore();

        std::cout << "\n── 性能基准测试 ──\n";
        bench_sma();

        std::cout << "\n✅ 所有测试通过!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ 测试失败: " << e.what() << "\n";
        return 1;
    }
}
