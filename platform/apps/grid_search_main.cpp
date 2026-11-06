/**
 * 参数网格搜索入口
 *
 * 核心功能: 对策略的参数空间进行暴力网格搜索，找到最优参数组合。
 *
 * Usage:
 *   ./grid_search_main --data ../data --strategy macross
 *   ./grid_search_main --data ../data --strategy meanrev --threads 16
 *   ./grid_search_main --data ../data --strategy macross --output results/opt.duckdb
 */

#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <fmt/chrono.h>

#include "core/backtest_engine.h"
#include "core/duckdb_provider.h"
#include "optimizer/param_optimizer.h"
#include "strategy/ma_cross_strategy.h"
#include "strategy/mean_reversion_strategy.h"

using namespace quant;

int main(int argc, char* argv[]) {
    CLI::App app{"Quant System - Parameter Grid Search"};

    // ── CLI 参数 ──────────────────────────────────────
    std::string data_dir = "../data";
    std::string db_path = ":memory:";
    std::string strategy_name = "macross";
    std::string start_date = "2020-01-01";
    std::string end_date = "2024-12-31";
    std::string output_path;
    double initial_cash = 1'000'000.0;

    // 网格参数覆盖（不指定则使用策略默认值）
    std::vector<double> fast_range;
    std::vector<double> slow_range;
    std::vector<double> bb_range;
    std::vector<double> k_range;

    int n_threads = 0;  // 0 = 自动
    int top_n = 20;
    bool use_genetic = false;

    app.add_option("--data", data_dir, "数据目录路径")->required();
    app.add_option("--db", db_path, "DuckDB 数据库路径");
    app.add_option("--strategy", strategy_name, "策略名称 (macross, meanrev)");
    app.add_option("--start", start_date, "回测起始日期");
    app.add_option("--end", end_date, "回测结束日期");
    app.add_option("--cash", initial_cash, "初始资金");
    app.add_option("--output", output_path, "优化结果输出路径");
    app.add_option("--threads", n_threads, "并行线程数 (0=自动)");
    app.add_option("--top", top_n, "保留前N个最优参数");

    // 双均线参数范围
    app.add_option("--fast-range", fast_range, "快线参数范围 [min, max, step]")->expected(3);
    app.add_option("--slow-range", slow_range, "慢线参数范围 [min, max, step]")->expected(3);

    // 均值回归参数范围
    app.add_option("--bb-range", bb_range, "布林带周期范围 [min, max, step]")->expected(3);
    app.add_option("--k-range", k_range, "布林带K值范围 [min, max, step]")->expected(3);

    app.add_flag("--genetic", use_genetic, "使用遗传算法（默认网格搜索）");

    CLI11_PARSE(app, argc, argv);

    // ── 启动 ──────────────────────────────────────────
    fmt::print("╔══════════════════════════════════════════╗\n");
    fmt::print("║   Quant System - 参数网格优化            ║\n");
    fmt::print("╚══════════════════════════════════════════╝\n\n");

    try {
        // 创建数据提供者
        auto provider = std::make_shared<DuckDBProvider>(db_path, data_dir);

        // 配置回测引擎
        BacktestConfig config;
        config.start_date = start_date;
        config.end_date = end_date;
        config.initial_cash = initial_cash;

        BacktestEngine engine(provider, config);

        // 创建策略
        std::unique_ptr<IStrategy> strategy;
        std::vector<ParamDef> param_grid;

        if (strategy_name == "macross") {
            strategy = std::make_unique<MACrossStrategy>();
            param_grid = strategy->params();

            // 命令行参数覆盖
            if (fast_range.size() == 3) {
                param_grid[0] = {"fast_period", fast_range[0],
                                 fast_range[1], fast_range[2], 5.0};
            }
            if (slow_range.size() == 3) {
                param_grid[1] = {"slow_period", slow_range[0],
                                 slow_range[1], slow_range[2], 20.0};
            }
        } else if (strategy_name == "meanrev") {
            strategy = std::make_unique<MeanReversionStrategy>();
            param_grid = strategy->params();

            if (bb_range.size() == 3) {
                param_grid[0] = {"bb_period", bb_range[0],
                                 bb_range[1], bb_range[2], 20.0};
            }
            if (k_range.size() == 3) {
                param_grid[1] = {"bb_k", k_range[0],
                                 k_range[1], k_range[2], 2.0};
            }
        } else {
            fmt::print("❌ 未知策略: {}\n", strategy_name);
            return 1;
        }

        // 预估计算量
        int total = 1;
        for (const auto& p : param_grid) {
            total *= static_cast<int>(p.grid_points().size());
        }
        fmt::print("📊 参数网格: {} 个参数维度, {} 个组合\n",
                   param_grid.size(), total);

        // 确认运行
        if (total > 10000) {
            fmt::print("⚠️  参数组合数 > 10000，可能需要较长时间。是否继续？[y/N] ");
            std::string confirm;
            std::getline(std::cin, confirm);
            if (confirm != "y" && confirm != "Y") {
                fmt::print("已取消\n");
                return 0;
            }
        }

        // 进度回调
        auto progress = [total](int done, int /*total*/) {
            // 进度由 grid_search 内部打印
        };

        // 运行优化
        ParameterOptimizer optimizer;
        OptimizationResult result;

        if (use_genetic) {
            fmt::print("🧬 使用遗传算法\n");
            result = optimizer.genetic_optimize(
                engine, *strategy, param_grid,
                100, 50, 0.1, 0.8, n_threads);
        } else {
            result = optimizer.grid_search(
                engine, *strategy, param_grid,
                top_n, n_threads, progress);
        }

        // 保存结果
        if (!output_path.empty()) {
            optimizer.save_results(result, output_path);
        }

        // 输出CSV格式结果（方便导入Python分析）
        fmt::print("\n📋 CSV格式结果 (可导入Python/Pandas分析):\n");
        fmt::print("sharpe,annual_return,max_drawdown");
        for (const auto& p : param_grid) {
            fmt::print(",{}", p.name);
        }
        fmt::print("\n");

        for (const auto& r : result.top_results) {
            fmt::print("{:.4f},{:.4f},{:.4f}",
                       r.sharpe_ratio,
                       r.annual_return,
                       r.max_drawdown);
            for (const auto& p : param_grid) {
                auto it = r.params.find(p.name);
                if (it != r.params.end()) {
                    fmt::print(",{:.1f}", it->second);
                }
            }
            fmt::print("\n");
        }

    } catch (const std::exception& e) {
        fmt::print("❌ 优化失败: {}\n", e.what());
        return 1;
    }

    return 0;
}
