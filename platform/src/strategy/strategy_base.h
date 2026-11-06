#pragma once
/**
 * 策略基类接口
 *
 * 支持两种模式:
 *   1. 事件驱动: 逐bar调用 on_bar() → Signal（灵活，适合分钟线/复杂逻辑）
 *   2. 向量化: 重写 compute_signals_vectorized()（极速，适合日线策略）
 *
 * 策略通过 params() 声明可优化参数，供 ParameterOptimizer 使用。
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>  // NOLINT

namespace quant {

// ── 数据结构 ───────────────────────────────────────────────

/// 交易信号
struct Signal {
    enum class Action : int8_t {
        HOLD = 0,
        BUY = 1,
        SELL = 2,
    };

    Action action = Action::HOLD;

    /// 目标权重 (0.0 ~ 1.0)
    /// BUY时: weight = 目标仓位占比
    /// SELL时: weight = 减仓比例 (1.0 = 全部卖出)
    double weight = 0.0;

    /// 信号强度/置信度 (0.0 ~ 1.0, 可选)
    double confidence = 0.0;
};

/// 可优化参数定义
struct ParamDef {
    std::string name;
    double min_val;
    double max_val;
    double step;
    double default_val;

    /// 生成该参数的所有网格点
    std::vector<double> grid_points() const {
        std::vector<double> points;
        for (double v = min_val; v <= max_val + step * 0.5; v += step) {
            points.push_back(v);
        }
        return points;
    }
};

/// 策略上下文（初始化时传入）
struct StrategyContext {
    std::vector<std::string> stock_codes;  // 全市场股票列表
    std::string start_date;
    std::string end_date;
    int lookback_days = 60;               // 策略需要的回溯天数
};

// ── 策略基类 ───────────────────────────────────────────────

class IStrategy {
public:
    virtual ~IStrategy() = default;

    /// 策略名称
    virtual std::string name() const = 0;

    /// 初始化策略
    /// @param context 策略上下文（股票列表、日期范围等）
    virtual void init(const StrategyContext& context) {
        ctx_ = context;
    }

    /// 事件驱动接口: 处理单根K线
    /// @param bar_idx 当前bar索引（在整个时间序列中的位置）
    /// @param market_data 当前可用的市场数据窗口
    ///   schema: [ts_code, trade_date, open, high, low, close,
    ///            pre_close, volume, amount, adj_factor]
    /// @param portfolio 当前持仓状态
    /// @return 交易信号
    virtual Signal on_bar(
        int bar_idx,
        const arrow::Table& market_data,
        const class Portfolio& portfolio) = 0;

    /// 向量化接口: 批量生成信号（可选重写以实现极致性能）
    /// @param market_data 全量市场数据
    ///   schema: [ts_code, trade_date, open, ..., close, ...]
    /// @return Arrow Array of Signal encodings
    ///         0=HOLD, 1=BUY, 2=SELL per row
    virtual std::shared_ptr<arrow::Array> compute_signals_vectorized(
        const arrow::Table& market_data);

    /// 声明可优化参数
    /// @return 参数网格定义，空 = 无参数
    virtual std::vector<ParamDef> params() const { return {}; }

protected:
    StrategyContext ctx_;
};

// 默认实现：抛异常，子类必须重写其中一个
inline std::shared_ptr<arrow::Array>
IStrategy::compute_signals_vectorized(const arrow::Table& /*market_data*/) {
    return nullptr;  // 表示未实现，BacktestEngine 会回退到事件驱动
}

}  // namespace quant
