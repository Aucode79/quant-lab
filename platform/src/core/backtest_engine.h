#pragma once
/**
 * 回测引擎 (Backtesting Engine)
 *
 * 核心引擎，支持两种回测模式:
 *   1. 向量化回测 (Vectorized): 最快，适合不需要逐bar复杂逻辑的策略
 *   2. 事件驱动回测 (Event-driven): 逐bar调用 on_bar()，灵活但较慢
 *
 * 自动选择: 如果策略实现了 compute_signals_vectorized() 就用向量化模式，
 *           否则回退到事件驱动模式。
 */

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <arrow/api.h>  // NOLINT

#include "core/broker_model.h"
#include "core/data_provider.h"
#include "core/performance_eval.h"
#include "core/portfolio.h"
#include "strategy/strategy_base.h"

namespace quant {

/// 回测配置
struct BacktestConfig {
    std::string start_date;
    std::string end_date;
    double initial_cash = 1'000'000.0;
    int benchmark_trade_days = 252;
    BrokerConfig broker;  // 交易成本配置
};

/// 回测结果
struct BacktestResult {
    PerformanceMetrics metrics;
    std::vector<DailyRecord> daily_records;
    Portfolio final_portfolio;
    std::vector<Trade> trade_history;
};

class BacktestEngine {
public:
    /// 构造函数
    /// @param data_provider 数据提供者
    /// @param config 回测配置
    BacktestEngine(std::shared_ptr<IDataProvider> data_provider,
                   BacktestConfig config);

    /// 运行回测
    /// @param strategy 策略实例
    /// @param params 策略参数 {参数名 -> 参数值}
    /// @return 回测结果
    BacktestResult run(IStrategy& strategy,
                       const std::unordered_map<std::string, double>& params = {});

    /// 向量化回测（日线级别，极致性能）
    BacktestResult run_vectorized(
        IStrategy& strategy,
        const std::unordered_map<std::string, double>& params);

    /// 事件驱动回测（逐bar调用 on_bar）
    BacktestResult run_event_driven(
        IStrategy& strategy,
        const std::unordered_map<std::string, double>& params);

    /// 获取数据提供者
    std::shared_ptr<IDataProvider> data_provider() const {
        return data_provider_;
    }

private:
    /// 准备回测数据（加载所有需要的股票K线）
    std::shared_ptr<arrow::Table> prepare_data(
        const std::vector<std::string>& stock_codes);

    /// 处理信号并执行交易
    void process_signal(Portfolio& portfolio,
                        const Signal& signal,
                        const std::string& ts_code,
                        double price,
                        const std::string& date);

    std::shared_ptr<IDataProvider> data_provider_;
    BacktestConfig config_;
    BrokerModel broker_;
};

}  // namespace quant
