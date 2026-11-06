#pragma once
/**
 * 参数优化器 (Parameter Optimizer)
 *
 * 支持两种优化方法:
 *   1. 网格搜索 (Grid Search): 暴力枚举所有参数组合
 *   2. 遗传算法 (Genetic Algorithm): 启发式搜索
 *
 * 并行策略:
 *   - 网格搜索: OpenMP parallel for，每组参数独立回测
 *   - 遗传算法: 每代内并行评估种群
 *
 * 中间结果保护:
 *   - 每完成一轮即保存结果到 DuckDB
 *   - 防止长时间运行过程丢失数据
 */

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/backtest_engine.h"
#include "strategy/strategy_base.h"

namespace quant {

/// 一组参数回测的结果
struct ParamResult {
    std::unordered_map<std::string, double> params;
    double sharpe_ratio = 0.0;
    double annual_return = 0.0;
    double max_drawdown = 0.0;
    double calmar_ratio = 0.0;
    double total_return = 0.0;
    double win_rate = 0.0;

    /// 组合评分（默认按夏普比率）
    double score() const { return sharpe_ratio; }
};

/// 优化结果
struct OptimizationResult {
    /// Top N 结果（按分数降序排列）
    std::vector<ParamResult> top_results;

    /// 全部参数组合的结果
    std::vector<ParamResult> all_results;

    /// 总参数组合数
    int total_combinations = 0;

    /// 实际完成数
    int completed = 0;

    /// 耗时（秒）
    double elapsed_seconds = 0.0;
};

/// 进度回调类型
using ProgressCallback = std::function<void(int done, int total)>;

class ParameterOptimizer {
public:
    ParameterOptimizer() = default;

    // ── 网格搜索 ───────────────────────────────────────────

    /// 网格搜索（核心功能）
    /// @param engine 回测引擎
    /// @param strategy 策略实例（会多次调用 run() 修改参数）
    /// @param param_grid 参数网格定义
    /// @param top_n 保留前N个最优参数
    /// @param n_threads 并行线程数（0 = 自动检测）
    /// @param progress_cb 进度回调（可选）
    OptimizationResult grid_search(
        BacktestEngine& engine,
        IStrategy& strategy,
        const std::vector<ParamDef>& param_grid,
        int top_n = 20,
        int n_threads = 0,
        ProgressCallback progress_cb = nullptr);

    // ── 遗传算法 ───────────────────────────────────────────

    /// 遗传算法优化
    /// @param engine 回测引擎
    /// @param strategy 策略实例
    /// @param param_ranges 参数范围
    /// @param population_size 种群大小 (默认100)
    /// @param generations 代数 (默认50)
    /// @param mutation_rate 变异率 (默认0.1)
    /// @param crossover_rate 交叉率 (默认0.8)
    /// @param n_threads 并行线程数
    OptimizationResult genetic_optimize(
        BacktestEngine& engine,
        IStrategy& strategy,
        const std::vector<ParamDef>& param_ranges,
        int population_size = 100,
        int generations = 50,
        double mutation_rate = 0.1,
        double crossover_rate = 0.8,
        int n_threads = 0);

    // ── 结果保存 ───────────────────────────────────────────

    /// 将优化结果保存到 DuckDB 文件
    void save_results(const OptimizationResult& result,
                      const std::string& output_path);

    /// 从 DuckDB 文件加载优化结果
    static OptimizationResult load_results(const std::string& path);

private:
    /// 生成参数网格的所有组合
    static std::vector<std::unordered_map<std::string, double>>
    generate_grid(const std::vector<ParamDef>& param_defs);

    /// 运行单次回测并评估
    ParamResult evaluate_params(
        BacktestEngine& engine,
        IStrategy& strategy,
        const std::unordered_map<std::string, double>& params);

    /// 按分数排序
    static void sort_by_score(std::vector<ParamResult>& results);
};

/// 遗传算法的个体
struct Individual {
    std::unordered_map<std::string, double> genes;
    double fitness = -std::numeric_limits<double>::infinity();
};

}  // namespace quant
