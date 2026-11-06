#!/usr/bin/env python3
"""
导入1分钟线数据 → Parquet (针对淘宝数据定制优化)

数据格式: CSV, GBK编码, 逗号分隔
  列: 日期,代码,开盘价,最高价,最低价,收盘价,成交量（手）,成交额（元）
  代码: SH.600000 / SZ.000001 / BJ.920000
  时间: YYYY-MM-DD HH:MM:SS

用法:
  python import_min_data.py                     # 处理所有年份
  python import_min_data.py --year 2024         # 单一年份
  python import_min_data.py --test              # 测试: 1只股票
  python import_min_data.py --delete-csv        # 导入完自动删CSV释放空间
"""

import argparse
import glob
import io
import os
import re
import sys
import time
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── 路径 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_BASE = PROJECT_ROOT.parent / "数据" / "股票"
PARQUET_PATH = PROJECT_ROOT / "data" / "market" / "kline_1min.parquet"
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 列名映射 ──────────────────────────────────────────────
COLUMN_RENAME = {
    "日期": "trade_time",
    "代码": "ts_code_raw",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量（手）": "volume",
    "成交额（元）": "amount",
}


def normalize_code(raw: str) -> str:
    """SH.600000 → 600000.SH"""
    m = re.match(r'(SH|SZ|BJ)\.(\d{6})', str(raw), re.IGNORECASE)
    if m:
        return f"{m.group(2)}.{m.group(1).upper()}"
    return str(raw)


def parse_batch(filepaths: list[str]) -> pd.DataFrame:
    """批量解析 CSV 文件"""
    frames = []
    for fp in filepaths:
        try:
            df = pd.read_csv(fp, encoding="utf-8", dtype=str,
                             on_bad_lines="skip")
            if df.empty:
                continue

            df = df.rename(columns=COLUMN_RENAME)
            required = ["ts_code_raw", "trade_time",
                        "open", "high", "low", "close"]
            if not all(c in df.columns for c in required):
                continue

            df["ts_code"] = df["ts_code_raw"].apply(normalize_code)

            # 数值转换
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 时间
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
            df = df.dropna(subset=["trade_time", "close"])
            df = df[df["close"] > 0]

            # OHLC 逻辑
            o, h, l, c = df["open"], df["high"], df["low"], df["close"]
            df = df[(h >= o - 0.001) & (h >= c - 0.001)
                    & (l <= o + 0.001) & (l <= c + 0.001)]

            if df.empty:
                continue

            df["trade_date"] = df["trade_time"].dt.date
            df["year"] = df["trade_time"].dt.year
            df["month"] = df["trade_time"].dt.month
            df["adj_factor"] = 1.0

            df = df.drop_duplicates(subset=["ts_code", "trade_time"], keep="last")

            out = ["ts_code", "trade_time", "trade_date",
                   "open", "high", "low", "close",
                   "volume", "amount", "adj_factor", "year", "month"]
            frames.append(df[out])

        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def get_years() -> dict:
    """扫描年份目录"""
    years = {}
    for d in sorted(CSV_BASE.iterdir()):
        if d.is_dir() and re.match(r'^\d{4}', d.name):
            y = int(d.name[:4])
            files = sorted(glob.glob(os.path.join(str(d), "*.csv")))
            years[y] = {"path": str(d), "files": files, "count": len(files)}
    return years


def process_year(year: int, info: dict, batch_size: int = 150):
    """处理一年数据"""
    files = info["files"]
    total = len(files)
    all_data = []
    rows = 0
    errs = 0
    t0 = time.time()

    print(f"\n{'='*55}")
    print(f"  [{year}年] {total} 个文件")
    print(f"{'='*55}")

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        df = parse_batch(batch)
        n = i + len(batch)

        if not df.empty:
            all_data.append(df)
            rows += len(df)
        else:
            errs += len(batch)

        elapsed = time.time() - t0
        if elapsed > 0:
            rate = n / elapsed
            eta = (total - n) / rate
            eta_str = f"{eta/60:.0f}min" if eta > 60 else f"{eta:.0f}s"
        else:
            eta_str = "..."

        print(f"  [{year}] {n:4d}/{total} ({100*n//total:3d}%) "
              f"| {rows/1e6:.1f}M行 | {elapsed:.0f}s | ETA {eta_str}")

    if not all_data:
        print(f"  [{year}] 无数据!")
        return 0

    print(f"  [{year}] 合并...")
    merged = pd.concat(all_data, ignore_index=True)
    n_stocks = merged["ts_code"].nunique()
    print(f"  [{year}] {len(merged):,}行, {n_stocks} 只股票, 写入Parquet...")

    table = pa.Table.from_pandas(merged)
    pq.write_to_dataset(
        table, root_path=str(PARQUET_PATH),
        partition_cols=["year", "month"],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )

    elapsed = time.time() - t0
    print(f"  [{year}] 完成! {elapsed/60:.1f}min")
    return len(merged)


def verify():
    """验证 Parquet"""
    print(f"\n{'='*55}")
    print(f"  数据验证")
    print(f"{'='*55}")
    try:
        con = duckdb.connect(":memory:")
        s = con.execute(f"""
            SELECT COUNT(*) as rows, COUNT(DISTINCT ts_code) as stocks,
                   MIN(trade_date) as d0, MAX(trade_date) as d1,
                   COUNT(DISTINCT trade_date) as days
            FROM '{PARQUET_PATH}/**/*.parquet'
        """).fetchone()
        print(f"  总行数:   {s[0]:,}")
        print(f"  股票数:   {s[1]}")
        print(f"  时间范围: {s[2]} ~ {s[3]}")
        print(f"  交易日数: {s[4]}")
        print(f"\n  年度明细:")
        for r in con.execute(f"""
            SELECT year, COUNT(*) as n, COUNT(DISTINCT ts_code) as stk
            FROM '{PARQUET_PATH}/**/*.parquet'
            GROUP BY year ORDER BY year
        """).fetchall():
            print(f"    {r[0]}: {r[1]:,}行, {r[2]}只")
        con.close()
    except Exception as e:
        print(f"  [WARN] {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, nargs="*")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--delete-csv", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    years = get_years()
    if not years:
        print("[ERROR] 未找到数据: " + str(CSV_BASE))
        return

    print(f"\n  CSV路径: {CSV_BASE}")
    print(f"  输出:    {PARQUET_PATH}")
    for y in sorted(years):
        print(f"    {y}年: {years[y]['count']} 文件")
    print(f"  共 {sum(v['count'] for v in years.values())} 个CSV")

    if args.test:
        # 只处理2026(1-6)的第一个文件
        info = years.get(2026) or list(years.values())[-1]
        info["files"] = info["files"][:1]
        info["count"] = 1
        years = {9999: info}
        print("  [TEST] 只处理1只股票")

    targets = args.year or sorted(years.keys())

    if args.skip_existing:
        existing = set()
        if PARQUET_PATH.exists():
            try:
                con = duckdb.connect(":memory:")
                for r in con.execute(f"""
                    SELECT DISTINCT year FROM '{PARQUET_PATH}/**/*.parquet'
                """).fetchall():
                    existing.add(r[0])
                con.close()
            except Exception:
                pass
        targets = [y for y in targets if y not in existing]
        print(f"  跳过已有: {sorted(existing)}")

    if not targets:
        print("  无任务")
        return

    print(f"  处理: {targets}\n")

    t0 = time.time()
    for y in targets:
        if y in years:
            process_year(y, years[y], args.batch_size)

        if args.delete_csv and y in years:
            import shutil
            d = years[y]["path"]
            sz = sum(os.path.getsize(f) for f in years[y]["files"])
            print(f"  删除CSV: {d} ({sz/1024/1024/1024:.1f}GB)")
            shutil.rmtree(d)
            print(f"  已释放 {sz/1024/1024/1024:.1f}GB")

    print(f"\n  总耗时: {(time.time()-t0)/60:.1f} 分钟")

    verify()

    total = sum(f.stat().st_size for f in PARQUET_PATH.rglob("*.parquet"))
    print(f"\n  Parquet总大小: {total/1024/1024/1024:.2f} GB")
    print(f"  [Done]")


if __name__ == "__main__":
    main()
