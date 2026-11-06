/**
 * 绩效评估 — 实现
 */

#include "core/performance_eval.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace quant {

void PerformanceEval::add_daily(const DailyRecord& record) {
    daily_records_.push_back(record);
}

PerformanceMetrics PerformanceEval::calculate(int trading_days_per_year) const {
    PerformanceMetrics m;

    if (daily_records_.empty()) return m;

    const size_t n = daily_records_.size();

    // ── 收益率计算 ──────────────────────────────────────

    std::vector<double> returns;
    returns.reserve(n);
    std::vector<double> equity;
    equity.reserve(n);

    for (const auto& rec : daily_records_) {
        returns.push_back(rec.daily_return);
        equity.push_back(rec.total_value);
    }

    // 总收益
    double initial = daily_records_.front().total_value;
    double final = daily_records_.back().total_value;
    if (initial > 0) {
        m.total_return = (final - initial) / initial;
    }

    // 年化收益
    double years = static_cast<double>(n) / trading_days_per_year;
    m.annual_return = std::pow(1.0 + m.total_return, 1.0 / years) - 1.0;

    // ── 风险指标 ──────────────────────────────────────

    // 最大回撤
    m.max_drawdown = max_drawdown(equity);

    // 年化波动率
    if (n > 1) {
        double sum = 0.0, sum_sq = 0.0;
        for (auto r : returns) {
            sum += r;
            sum_sq += r * r;
        }
        double mean = sum / n;
        double variance = (sum_sq / n) - (mean * mean);
        if (variance > 0) {
            m.annual_volatility = std::sqrt(variance * trading_days_per_year);
        }

        // 下行波动率
        double down_sum_sq = 0.0;
        int down_count = 0;
        double daily_rf = risk_free_rate_ / trading_days_per_year;
        for (auto r : returns) {
            if (r < daily_rf) {
                double diff = r - daily_rf;
                down_sum_sq += diff * diff;
                down_count++;
            }
        }
        if (down_count > 0) {
            m.downside_volatility = std::sqrt(
                down_sum_sq / down_count * trading_days_per_year);
        }
    }

    // ── 风险调整收益 ─────────────────────────────────────

    // 夏普比率（假设无风险利率）
    double excess = m.annual_return - risk_free_rate_;
    if (m.annual_volatility > 1e-9) {
        m.sharpe_ratio = excess / m.annual_volatility;
    }

    // 卡玛比率
    if (m.max_drawdown > 1e-9) {
        m.calmar_ratio = m.annual_return / m.max_drawdown;
    }

    // 索提诺比率
    if (m.downside_volatility > 1e-9) {
        m.sortino_ratio = excess / m.downside_volatility;
    }

    return m;
}

std::vector<double> PerformanceEval::equity_curve() const {
    std::vector<double> curve;
    curve.reserve(daily_records_.size());
    for (const auto& rec : daily_records_) {
        curve.push_back(rec.total_value);
    }
    return curve;
}

std::vector<double> PerformanceEval::drawdown_series() const {
    std::vector<double> dd;
    if (daily_records_.empty()) return dd;

    dd.reserve(daily_records_.size());
    double peak = daily_records_[0].total_value;

    for (const auto& rec : daily_records_) {
        peak = std::max(peak, rec.total_value);
        if (peak > 0) {
            dd.push_back((peak - rec.total_value) / peak);
        } else {
            dd.push_back(0.0);
        }
    }
    return dd;
}

double PerformanceEval::annualize_return(double total_return, double years) {
    if (years <= 0) return 0.0;
    return std::pow(1.0 + total_return, 1.0 / years) - 1.0;
}

double PerformanceEval::max_drawdown(const std::vector<double>& equity) {
    if (equity.empty()) return 0.0;

    double peak = equity[0];
    double max_dd = 0.0;

    for (double val : equity) {
        peak = std::max(peak, val);
        if (peak > 0) {
            double dd = (peak - val) / peak;
            max_dd = std::max(max_dd, dd);
        }
    }

    return max_dd;
}

std::unordered_map<std::string, double> PerformanceMetrics::to_map() const {
    return {
        {"total_return", total_return},
        {"annual_return", annual_return},
        {"excess_return", excess_return},
        {"max_drawdown", max_drawdown},
        {"annual_volatility", annual_volatility},
        {"downside_volatility", downside_volatility},
        {"sharpe_ratio", sharpe_ratio},
        {"calmar_ratio", calmar_ratio},
        {"sortino_ratio", sortino_ratio},
        {"information_ratio", information_ratio},
        {"total_trades", static_cast<double>(total_trades)},
        {"winning_trades", static_cast<double>(winning_trades)},
        {"win_rate", win_rate},
        {"avg_win", avg_win},
        {"avg_loss", avg_loss},
        {"profit_factor", profit_factor},
        {"avg_hold_days", avg_hold_days},
        {"turnover_rate", turnover_rate},
    };
}

}  // namespace quant
