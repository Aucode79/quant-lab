# platform — 数据管道与回测框架

## 演进：从聚宽在线回测到本地自建工作流

本项目最初在聚宽（JoinQuant）平台做在线回测（`scripts/jq_backtest*.py` 为三版对标脚本留档），后迁移到本地全链路自建。迁移动机：

1. **数据规模**：策略研究需要 9 年分钟级数据的自由因子计算，在线平台无法满足；
2. **寻优规模**：网格搜索需要万组参数级别的并行寻优（1 万组 ≈18 秒 @16 核）；
3. **可复现与可审计**：自建数据校验与无未来函数自检，每个结论可追溯到数据层。

迁移原则：**本地回测结果必须与聚宽口径对齐**——复权基准差异已知并可解释（见 [`../docs/data-pipeline.md`](../docs/data-pipeline.md)），动量滚仓策略的本地实现与聚宽平台结果完成对标（收益/回撤/夏普一致）。

## Python 工具链（`scripts/`）

| 脚本 | 用途 |
|---|---|
| `run_import.py` | 分钟线 CSV → Parquet 主导入（DuckDB 原生 COPY） |
| `fetch_stock_basic.py` / `fetch_kline_daily.py` | akshare / baostock 免费数据源（替代商业数据） |
| `update_daily.py` | 每日增量更新 |
| `_verify_data.py` / `validate_data.py` | 全量数据质量校验 |
| `quick_backtest.py` | 快速回测示例 |
| `jq_backtest*.py` | 与聚宽（JoinQuant）平台结果对标脚本 |
| `grid_search_main.cpp` / `apps/` | 网格寻优入口（C++） |

网格寻优另有 Python 实现（multiprocessing），1 万组参数约 18 秒 @16 核。

## C++ 回测引擎（`src/`）— 开发中，如实说明

- 分层：`core/`（数据提供、撮合、组合、绩效）、`strategy/`（均线交叉、均值回归）、`indicators/`（MA/MACD/RSI/BOLL，header-only）、`optimizer/`（网格搜索）。
- 数据层走 DuckDB C++ API 直读 Parquet。
- **当前状态：框架代码已写完，但尚未通过编译验证**（CMake + MSYS2/mingw64 环境已备）。构建配置仍在调整中，暂不能保证开箱即用。

## 数据说明

分钟线数据为商业授权来源，**不随本仓库分发**；`data/` 已被 `.gitignore` 排除。复现请使用 akshare/baostock 免费源或自有数据，并在脚本中修改数据路径。复权口径说明见 [`../docs/data-pipeline.md`](../docs/data-pipeline.md)。
