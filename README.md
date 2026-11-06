# Quant Lab — 本地 A 股量化研究平台

个人量化研究项目：本地行情数据管道 + 回测/参数寻优框架 + 多策略研究。始于 2026-07，持续开发中。

> ⚠️ 仅供学习与研究，不构成任何投资建议。历史回测不代表未来表现。

## 结构

```
quant-lab/
├── platform/                  # 数据管道与回测框架
│   ├── scripts/               # 导入、校验、回测、寻优（Python / DuckDB）
│   ├── src/ apps/ tests/      # C++ 回测引擎（开发中，尚未编译验证）
│   └── results/               # 回测输出样例
├── strategies/
│   ├── momentum-rolling/      # 动量滚仓：周度 ETF 动量轮动（含聚宽平台对标）
│   └── cycle-etf-rotation/    # 周期 ETF 三门轮动：完整研究流程（审计/稳健性/安慰剂）
└── docs/
    └── data-pipeline.md       # 28.4 亿行分钟线数据管道的实现记录
```

另有隔夜持股策略尚在探索期（仅本地日志），暂未入库。

## 数据管道（platform）

- 数据：9 年 A 股 + ETF 1 分钟线（2018–2026H1），约 **28.4 亿行 / 38.9 GB**。数据为商业授权来源，**不随本仓库分发**；`platform/scripts/` 内提供 akshare / baostock 免费数据源的下载脚本作为替代。
- 导入：DuckDB 原生 `COPY` → Hive 分区 Parquet，单年约 60–167 秒。
- 校验：全量零缺失、零 OHLC 异常、零重复，每交易日 240 bar，复权连续性通过。
- 寻优：Python 网格搜索（multiprocessing，1 万组参数约 18 秒 @16 核）。

实现细节与踩坑记录见 [`docs/data-pipeline.md`](docs/data-pipeline.md)。

## 策略研究

| 策略 | 方法 | 状态 |
|---|---|---|
| [momentum-rolling](strategies/momentum-rolling/) | 周度动量轮动（20 日收益/波动打分 + MA20 过滤 + 现金避险），成本建模，与聚宽平台结果对标 | V1 完成 |
| [cycle-etf-rotation](strategies/cycle-etf-rotation/) | 周期因子检验（多窗口动量/位置 z-score/波动比 × Spearman 多窗口 IC、PCA 共同驱动、方差比诊断）+ FFT 周期谱估计 + 行业领先滞后（置换检验）+ 三门轮动策略 | 研究中，**回测引擎指标存在已知 bug（见其 README）** |

## 环境

- Python 3.10+（开发环境 3.13）：duckdb / pandas / pyarrow / matplotlib / akshare / baostock
- C++ 引擎：CMake + MSYS2 (mingw64) + DuckDB C++ API

## License

[MIT](LICENSE)
