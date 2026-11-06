#pragma once
/**
 * 绩效评估 (Performance Evaluation)
 *
 * 计算回测的核心绩效指标:
 *   - 总收益 / 年化收益
 *   - 最大回撤
 *   - 夏普比率 / 卡玛比率 / 索提诺比率
 *   - 胜率 / 盈亏比
 *   - 换手率 / 持仓集中度
 */

#include <string>
#include <unordered_map>
#include <vector>

namespace quant {

/// 每日净值记录
struct DailyRecord {
    std::string date;
    double total_value = 0.0;
    double cash = 0.0;
    double market_value = 0.0;
    double daily_return = 0.0;
};

/// 完整回测绩效指标
struct PerformanceMetrics {
    // 收益指标
    double total_return = 0.0;        // 总收益率
    double annual_return = 0.0;       // 年化收益率
    double excess_return = 0.0;       // 超额收益 (vs 基准)

    // 风险指标
    double max_drawdown = 0.0;        // 最大回撤
    double annual_volatility = 0.0;   // 年化波动率
    double downside_volatility = 0.0; // 下行波动率

    // 风险调整收益
    double sharpe_ratio = 0.0;        // 夏普比率
    double calmar_ratio = 0.0;        // 卡玛比率 (= 年化收益/最大回撤)
    double sortino_ratio = 0.0;       // 索提诺比率
    double information_ratio = 0.0;   // 信息比率

    // 交易统计
    int total_trades = 0;            // 总交易次数
    int winning_trades = 0;          // 盈利次数
    double win_rate = 0.0;           // 胜率
    double avg_win = 0.0;            // 平均盈利
    double avg_loss = 0.0;           // 平均亏损
    double profit_factor = 0.0;      // 盈亏比 (总盈利/总亏损)
    double avg_hold_days = 0.0;      // 平均持仓天数

    // 年度统计
    double turnover_rate = 0.0;      // 年化换手率

    // 将结果转为 key-value 映射（便于保存到数据库）
    std::unordered_map<std::string, double> to_map() const;
};

class PerformanceEval {
public:
    PerformanceEval() = default;

    /// 设置无风险利率
    void set_risk_free_rate(double rate) { risk_free_rate_ = rate; }

    /// 添加每日记录
    void add_daily(const DailyRecord& record);

    /// 计算全部绩效指标
    PerformanceMetrics calculate(int trading_days_per_year = 252) const;

    /// 获取每日记录
    const std::vector<DailyRecord>& daily_records() const {
        return daily_records_;
    }

    /// 获取净值曲线数据（用于绘图）
    std::vector<double> equity_curve() const;

    /// 获取回撤序列
    std::vector<double> drawdown_series() const;

    /// 静态工具方法：计算年化收益
    static double annualize_return(double total_return, double years);

    /// 静态工具方法：计算最大回撤
    static double max_drawdown(const std::vector<double>& equity);

private:
    std::vector<DailyRecord> daily_records_;
    double risk_free_rate_ = 0.03;  // 默认 3%
};

}  // namespace quant
