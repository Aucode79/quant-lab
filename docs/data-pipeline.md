# 数据管道实现记录（28.4 亿行分钟线）

本文记录本地分钟线数据管道的实现过程、性能与踩坑。数据为商业授权来源，不随仓库分发。

## 数据源

- A 股 + ETF 1 分钟线（2018 ~ 2026H1），每年一个压缩包（2~4 GB 压缩 / 12~20 GB 解压）
- CSV 列：`日期, 代码, 开盘价, 最高价, 最低价, 收盘价, 成交量（手）, 成交额（元）`（ETF 列名略异）
- 代码格式 `SH.600000 / SZ.000001 / BJ.920000`，统一转为 `600000.SH` 风格
- 每交易日精确 240 根 bar（09:31~15:00）

## 最终数据量

| 类型 | 年份范围 | 行数 | Parquet |
|---|---|---|---|
| 股票 | 2018–2026H1 | 22.5 亿（3,575 → 5,542 只） | 31 GB |
| ETF | 2018–2026H1 | 5.85 亿（783 → 2,395 只） | 7.9 GB |
| **合计** | | **28.4 亿** | **38.9 GB** |

存储为 Hive 分区 Parquet（`year=YYYY/month=M/`，SNAPPY），DuckDB 可直接通配符查询：

```sql
SELECT * FROM 'kline_1min.parquet/**/*.parquet' WHERE year = 2024;
```

## 导入方案：DuckDB 原生 COPY

```sql
COPY (
    WITH raw AS (
        SELECT * FROM read_csv('2018/*.csv', header=true, encoding='utf-8',
                  all_varchar=true, ignore_errors=true)
    ),
    clean AS (
        SELECT
            CASE WHEN "代码" LIKE 'SH.%' THEN ... END AS ts_code,
            CAST("日期" AS TIMESTAMP) AS trade_time,
            CAST("开盘价" AS DOUBLE) AS open, ...
        FROM raw WHERE CAST("收盘价" AS DOUBLE) > 0 ...
    )
    SELECT * FROM clean
) TO 'output' (FORMAT PARQUET, PARTITION_BY (year, month), COMPRESSION SNAPPY)
```

性能：1 年约 60–167 秒（9 年全量约 18 分钟）。

**踩坑教训**：先用 pandas 分批 + DuckDB INSERT 的方案反复失败（concat 慢、表越大 INSERT 越慢、分批覆盖数据）。结论：**DuckDB 的 INSERT 不要处理超过 5000 万行的场景，大批量导入必须用 COPY**。

## 质量校验（全量通过）

- 零 null / 零 OHLC 异常 / 零重复 (ts_code, trade_time)
- 每交易日精确 240 根 bar
- 复权连续性通过；新股上市与退市自然覆盖

## 与聚宽（JoinQuant）数据的差异

同一标的同日开盘价，本地数据与聚宽可差 2–5%（通富微电 +2.4%、巨化股份 -3.5%），原因为**复权基准日不同**——每个数据源都有自己的基准日，属正常现象。结论：回测与实盘使用同一数据源即可，不要混用。ETF 基本无分红，差异 <0.5%。

## 环境

- Python 3.13（Windows，`py -3` 调用）；duckdb / pandas / pyarrow / matplotlib / akshare / baostock / tqdm
- 免费替代数据源：akshare（股票列表）、baostock（日K），见 `platform/scripts/fetch_*.py`
