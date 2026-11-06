/**
 * 参数优化器 — 网格搜索实现
 *
 * 使用 OpenMP 并行化网格搜索。
 * 每组参数的回测完全独立，天然适合并行。
 */

#include "optimizer/param_optimizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <random>
#include <thread>

#include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace quant {

// ── 网格搜索主逻辑 ─────────────────────────────────────────

OptimizationResult ParameterOptimizer::grid_search(
    BacktestEngine& engine,
    IStrategy& strategy,
    const std::vector<ParamDef>& param_grid,
    int top_n,
    int n_threads,
    ProgressCallback progress_cb) {

    OptimizationResult result;
    auto start_time = std::chrono::steady_clock::now();

    // 生成参数网格
    auto grid = generate_grid(param_grid);
    result.total_combinations = static_cast<int>(grid.size());

    if (grid.empty()) return result;

    fmt::print("🔍 网格搜索: {} 个参数组合\n", grid.size());
    fmt::print("   参数维度: {}\n", param_grid.size());
    for (const auto& p : param_grid) {
        auto points = p.grid_points();
        fmt::print("   {}: [{:.1f}, {:.1f}] step={:.1f} → {} 个点\n",
                   p.name, p.min_val, p.max_val, p.step, points.size());
    }

    // 设置线程数
#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
    }
    omp_set_num_threads(n_threads);
    fmt::print("   🧵 使用 {} 个线程\n", n_threads);
#endif

    // 预分配结果容器
    std::vector<ParamResult> all_results(grid.size());
    std::atomic<int> completed{0};

    // ── OpenMP 并行遍历参数网格 ─────────────────────────
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(grid.size()); ++i) {
        all_results[i] = evaluate_params(engine, strategy, grid[i]);

        int done = ++completed;

        // 进度汇报（每 1% 或每 10 组汇报一次）
        if (progress_cb && (done % std::max(1, static_cast<int>(grid.size()) / 100) == 0
                           || done == static_cast<int>(grid.size()))) {
            progress_cb(done, static_cast<int>(grid.size()));
        }

        // 每 5% 打印一次进度
        if (done % std::max(1, static_cast<int>(grid.size()) / 20) == 0) {
            #pragma omp critical
            {
                fmt::print("   [{}/{}] {:.1f}%  当前最佳夏普: {:.3f}\n",
                           done, grid.size(),
                           100.0 * done / grid.size(),
                           result.top_results.empty() ? 0.0
                               : result.top_results[0].sharpe_ratio);
            }
        }
    }

    result.completed = completed.load();
    result.all_results = std::move(all_results);

    // 排序取 Top N
    sort_by_score(result.all_results);
    for (int i = 0; i < std::min(top_n, static_cast<int>(result.all_results.size())); ++i) {
        result.top_results.push_back(result.all_results[i]);
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(
        end_time - start_time).count();

    // 打印汇总
    fmt::print("\n✅ 网格搜索完成!\n");
    fmt::print("   耗时: {:.1f}s\n", result.elapsed_seconds);
    fmt::print("   完成: {} / {} 组\n", result.completed, result.total_combinations);

    if (!result.top_results.empty()) {
        fmt::print("\n🏆 Top 5 参数组合:\n");
        fmt::print("   {:<20} {:<10} {:<10} {:<10}\n",
                   "参数", "夏普", "年化收益%", "最大回撤%");
        fmt::print("   {:-<50}\n", "");

        for (int i = 0; i < std::min(5, static_cast<int>(result.top_results.size())); ++i) {
            const auto& r = result.top_results[i];
            std::string param_str;
            for (const auto& [k, v] : r.params) {
                if (!param_str.empty()) param_str += ", ";
                param_str += fmt::format("{}={:.1f}", k, v);
            }
            fmt::print("   {:<20} {:<10.3f} {:<10.1f} {:<10.1f}\n",
                       param_str,
                       r.sharpe_ratio,
                       r.annual_return * 100,
                       r.max_drawdown * 100);
        }
    }

    return result;
}

// ── 遗传算法 ───────────────────────────────────────────────

OptimizationResult ParameterOptimizer::genetic_optimize(
    BacktestEngine& engine,
    IStrategy& strategy,
    const std::vector<ParamDef>& param_ranges,
    int population_size,
    int generations,
    double mutation_rate,
    double crossover_rate,
    int n_threads) {

    OptimizationResult result;
    auto start_time = std::chrono::steady_clock::now();

#ifdef _OPENMP
    if (n_threads <= 0) {
        n_threads = std::thread::hardware_concurrency();
    }
    omp_set_num_threads(n_threads);
#endif

    std::mt19937 rng(42);  // 固定种子，可复现
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> int_dist(0, population_size - 1);

    // 初始化种群
    std::vector<Individual> population(population_size);
    for (int i = 0; i < population_size; ++i) {
        population[i].genes.clear();
        for (const auto& p : param_ranges) {
            std::uniform_real_distribution<double> param_dist(p.min_val, p.max_val);
            population[i].genes[p.name] = param_dist(rng);
        }
    }

    // 逐代进化
    for (int gen = 0; gen < generations; ++gen) {
        fmt::print("🧬 代 {}: 评估 {} 个个体...\n", gen + 1, population_size);

        // 评估种群（并行）
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < population_size; ++i) {
            auto pr = evaluate_params(engine, strategy, population[i].genes);
            population[i].fitness = pr.score();
        }

        // 按适应度排序
        std::sort(population.begin(), population.end(),
                  [](const Individual& a, const Individual& b) {
                      return a.fitness > b.fitness;
                  });

        // 保留精英
        constexpr int ELITE_COUNT = 2;
        std::vector<Individual> new_population;
        for (int i = 0; i < ELITE_COUNT; ++i) {
            new_population.push_back(population[i]);
        }

        // 交叉与变异
        while (static_cast<int>(new_population.size()) < population_size) {
            int parent1 = int_dist(rng);
            int parent2 = int_dist(rng);

            Individual child;
            for (const auto& p : param_ranges) {
                // 交叉
                if (dist(rng) < crossover_rate) {
                    // 算术交叉
                    double alpha = dist(rng);
                    child.genes[p.name] = alpha * population[parent1].genes[p.name]
                                         + (1.0 - alpha) * population[parent2].genes[p.name];
                } else {
                    child.genes[p.name] = population[parent1].genes[p.name];
                }

                // 变异
                if (dist(rng) < mutation_rate) {
                    std::uniform_real_distribution<double> param_dist(p.min_val, p.max_val);
                    child.genes[p.name] = param_dist(rng);
                }

                // 边界裁剪
                child.genes[p.name] = std::clamp(child.genes[p.name],
                                                  p.min_val, p.max_val);
            }
            new_population.push_back(std::move(child));
        }

        population = std::move(new_population);

        fmt::print("   最佳适应度: {:.4f}\n", population[0].fitness);
    }

    // 返回最终结果
    for (const auto& ind : population) {
        auto pr = evaluate_params(engine, strategy, ind.genes);
        result.all_results.push_back(pr);
    }

    sort_by_score(result.all_results);
    for (int i = 0; i < std::min(20, static_cast<int>(result.all_results.size())); ++i) {
        result.top_results.push_back(result.all_results[i]);
    }

    auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.total_combinations = population_size * generations;

    return result;
}

// ── 辅助函数 ───────────────────────────────────────────────

std::vector<std::unordered_map<std::string, double>>
ParameterOptimizer::generate_grid(const std::vector<ParamDef>& param_defs) {
    std::vector<std::unordered_map<std::string, double>> result;

    if (param_defs.empty()) return result;

    // 递归生成笛卡尔积
    std::function<void(size_t, std::unordered_map<std::string, double>)> dfs;
    dfs = [&](size_t idx, std::unordered_map<std::string, double> current) {
        if (idx >= param_defs.size()) {
            result.push_back(current);
            return;
        }

        auto points = param_defs[idx].grid_points();
        for (double val : points) {
            current[param_defs[idx].name] = val;
            dfs(idx + 1, current);
        }
    };

    dfs(0, {});
    return result;
}

ParamResult ParameterOptimizer::evaluate_params(
    BacktestEngine& engine,
    IStrategy& strategy,
    const std::unordered_map<std::string, double>& params) {

    ParamResult result;
    result.params = params;

    try {
        auto bt_result = engine.run(strategy, params);

        result.sharpe_ratio = bt_result.metrics.sharpe_ratio;
        result.annual_return = bt_result.metrics.annual_return;
        result.max_drawdown = bt_result.metrics.max_drawdown;
        result.calmar_ratio = bt_result.metrics.calmar_ratio;
        result.total_return = bt_result.metrics.total_return;
        result.win_rate = bt_result.metrics.win_rate;
    } catch (const std::exception& e) {
        // 回测失败时给极低分数
        result.sharpe_ratio = -999.0;
        result.annual_return = -999.0;
    }

    return result;
}

void ParameterOptimizer::sort_by_score(std::vector<ParamResult>& results) {
    std::sort(results.begin(), results.end(),
              [](const ParamResult& a, const ParamResult& b) {
                  return a.score() > b.score();
              });
}

void ParameterOptimizer::save_results(
    const OptimizationResult& result,
    const std::string& output_path) {
    // TODO: 使用 DuckDB C++ API 写入结果表
    // 也可以输出为 CSV / JSON
    fmt::print("💾 优化结果保存至: {}\n", output_path);
}

OptimizationResult ParameterOptimizer::load_results(const std::string& path) {
    // TODO: 从 DuckDB 文件读取结果
    (void)path;
    return {};
}

}  // namespace quant
