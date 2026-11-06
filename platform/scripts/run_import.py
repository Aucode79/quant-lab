#!/usr/bin/env python3
"""DuckDB 原生导入——读CSV直接写Parquet，无pandas，极速"""
import glob, os, re, sys, io, time
from pathlib import Path
import duckdb

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_ROOT = Path(__file__).parent.parent.parent / "数据"
PARQUET_DIR = Path(__file__).parent.parent / "data" / "market"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = PARQUET_DIR / "import_stock.log"

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def import_year(data_type, year):
    """DuckDB 原生：CSV → 清洗 → Parquet"""
    import shutil
    subdir = "股票" if data_type == "stock" else "etf"
    output_name = "kline_1min.parquet" if data_type == "stock" else "kline_1min_etf.parquet"
    output_path = PARQUET_DIR / output_name

    csv_dir = DATA_ROOT / subdir
    src = csv_dir / str(year)
    if not src.exists():
        candidates = [d for d in csv_dir.iterdir() if d.name.startswith(str(year))]
        src = candidates[0] if candidates else None
    if not src or not src.exists():
        log(f"SKIP {data_type} {year}: dir not found")
        return

    csv_pattern = str(src / "*.csv").replace("\\", "/")
    log(f"[{data_type.upper()} {year}] {csv_pattern}")

    sql = f"""
    COPY (
        WITH raw AS (
            SELECT * FROM read_csv(
                '{csv_pattern}',
                header=true, encoding='utf-8',
                all_varchar=true,
                ignore_errors=true
            )
        ),
        clean AS (
            SELECT
                CASE
                    WHEN "代码" LIKE 'SH.%' THEN SUBSTRING("代码", 4, 6) || '.SH'
                    WHEN "代码" LIKE 'SZ.%' THEN SUBSTRING("代码", 4, 6) || '.SZ'
                    WHEN "代码" LIKE 'BJ.%' THEN SUBSTRING("代码", 4, 6) || '.BJ'
                    ELSE "代码"
                END AS ts_code,
                CAST("日期" AS TIMESTAMP) AS trade_time,
                CAST("日期" AS DATE) AS trade_date,
                CAST("开盘价" AS DOUBLE) AS open,
                CAST("最高价" AS DOUBLE) AS high,
                CAST("最低价" AS DOUBLE) AS low,
                CAST("收盘价" AS DOUBLE) AS close,
                CAST("成交量（手）" AS DOUBLE) AS volume,
                CAST("成交额（元）" AS DOUBLE) AS amount,
                1.0 AS adj_factor,
                CAST(strftime(CAST("日期" AS TIMESTAMP), '%Y') AS INTEGER) AS year,
                CAST(strftime(CAST("日期" AS TIMESTAMP), '%m') AS INTEGER) AS month,
            FROM raw
            WHERE CAST("收盘价" AS DOUBLE) > 0
              AND CAST("最高价" AS DOUBLE) >= CAST("开盘价" AS DOUBLE) - 0.001
              AND CAST("最高价" AS DOUBLE) >= CAST("收盘价" AS DOUBLE) - 0.001
              AND CAST("最低价" AS DOUBLE) <= CAST("开盘价" AS DOUBLE) + 0.001
              AND CAST("最低价" AS DOUBLE) <= CAST("收盘价" AS DOUBLE) + 0.001
        )
        SELECT * FROM clean
    ) TO '{output_path}' (
        FORMAT PARQUET,
        PARTITION_BY (year, month),
        COMPRESSION SNAPPY,
        OVERWRITE_OR_IGNORE
    )
    """

    t0 = time.time()
    con = duckdb.connect(":memory:")
    # 允许更多内存加速
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=8")

    log(f"[{data_type.upper()} {year}] Processing...")
    con.execute(sql)
    con.close()

    elapsed = time.time() - t0
    log(f"[{data_type.upper()} {year}] DONE in {elapsed:.0f}s")

    # Verify
    try:
        vcon = duckdb.connect(":memory:")
        v = vcon.execute(f"""
            SELECT COUNT(*) as n, COUNT(DISTINCT ts_code) as stk,
                   MIN(trade_date) as d0, MAX(trade_date) as d1
            FROM read_parquet('{output_path}/**/*.parquet', hive_partitioning=true)
            WHERE year = {year}
        """).fetchone()
        if v and v[0]:
            log(f"[{data_type.upper()} {year}] VERIFY: {v[0]:,}rows, {v[1]}stocks, {v[2]}~{v[3]} [OK]")
        vcon.close()
    except Exception as e:
        log(f"[{data_type.upper()} {year}] VERIFY FAILED: {e}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--type", default="stock", choices=["stock","etf"])
    p.add_argument("--year", type=int, nargs="*")
    args = p.parse_args()

    if args.year:
        years = args.year
    else:
        subdir = "股票" if args.type == "stock" else "etf"
        csv_dir = DATA_ROOT / subdir
        years = sorted(int(d.name[:4]) for d in csv_dir.iterdir()
                       if d.is_dir() and d.name[:4].isdigit())

    log(f"=== {args.type.upper()} Import {len(years)} years: {years} ===")
    t0 = time.time()
    for y in years:
        import_year(args.type, y)
    log(f"=== ALL DONE in {(time.time()-t0)/60:.1f}min ===")
