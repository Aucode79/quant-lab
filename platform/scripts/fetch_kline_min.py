#!/usr/bin/env python3
"""
获取全A股5分钟K线数据（前复权），直接调用东方财富 API。

数据源: 东方财富 push2his.eastmoney.com (免费, 无需token)
频率: 5分钟
输出: data/market/kline_5min.parquet (按年份分区)

特点:
  - 直接 HTTP 请求，绕开 akshare 的代理问题
  - 每次请求约 1 年数据（API 单次最大返回约 10,000 条）
  - 全市场 5527只 x 5年 ≈ 3 小时下载

Usage:
    python fetch_kline_min.py --year 2024
    python fetch_kline_min.py --start 2020 --end 2024
    python fetch_kline_min.py --codes 600000.SH 000001.SZ
"""

import argparse
import io
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# ── 路径 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "market"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = DATA_DIR / "kline_5min.parquet"

# 东方财富 API 参数
EM_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EM_PERIOD = 5  # 5分钟
# 上交所: secid=1.{code}, 深交所: secid=0.{code}


def get_stock_list() -> list[dict]:
    """从 Parquet 获取全部股票代码和名称"""
    stock_basic = DATA_DIR.parent / "reference" / "stock_basic.parquet"
    if not stock_basic.exists():
        print("[ERROR] 请先运行 fetch_stock_basic.py")
        sys.exit(1)
    con = duckdb.connect(":memory:")
    rows = con.execute(f"""
        SELECT ts_code, name, exchange
        FROM '{stock_basic}'
        WHERE exchange IN ('SSE', 'SZSE')
        ORDER BY ts_code
    """).fetchall()
    con.close()
    return [{"ts_code": r[0], "name": r[1], "exchange": r[2]} for r in rows]


def get_completed_stocks(year: int, output_path: Path) -> set[str]:
    """已有数据的股票集合"""
    if not output_path.exists() or not list(output_path.glob("*.parquet")):
        return set()
    try:
        con = duckdb.connect(":memory:")
        result = con.execute(f"""
            SELECT DISTINCT ts_code
            FROM '{output_path}/**/*.parquet'
            WHERE year = {year}
        """).fetchall()
        con.close()
        return {r[0] for r in result}
    except Exception:
        return set()


def em_secid(ts_code: str) -> str:
    """600000.SH → 1.600000, 000001.SZ → 0.000001"""
    parts = ts_code.split(".")
    code = parts[0]
    market = "1" if parts[1] == "SH" else "0"
    return f"{market}.{code}"


def fetch_one_stock(ts_code: str, year: int) -> pd.DataFrame | None:
    """
    获取单只股票单年的 5 分钟线（前复权）。
    返回 DataFrame 或 None。
    """
    secid = em_secid(ts_code)

    # 构造请求参数
    # beg/end 是序号而非日期，所以用大范围然后用日期过滤
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": str(EM_PERIOD),
        "fqt": "1",  # 前复权
        "secid": secid,
        "beg": "0",
        "end": "15000",  # 足够覆盖一年
    }

    try:
        r = requests.get(EM_URL, params=params, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("rc") != 0:
            return None
        klines = data.get("data", {}).get("klines", [])
        if not klines:
            return None
    except Exception:
        return None

    # 解析 K 线数据
    # 格式: "时间,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率"
    rows = []
    for line in klines:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        try:
            trade_time = parts[0]  # "2024-01-02 09:35"
            dt = pd.to_datetime(trade_time)

            # 过滤到目标年份
            if dt.year != year:
                continue

            rows.append({
                "ts_code": ts_code,
                "trade_time": trade_time,
                "open": float(parts[1]),
                "close": float(parts[2]),
                "high": float(parts[3]),
                "low": float(parts[4]),
                "volume": float(parts[5]),
                "amount": float(parts[6]),
                "adj_factor": 1.0,
                "year": year,
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return None

    return pd.DataFrame(rows)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """保存分区 Parquet"""
    if df.empty:
        return
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=str(path),
        partition_cols=["year"],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )


def main():
    parser = argparse.ArgumentParser(description="下载A股5分钟K线 (东方财富源)")
    parser.add_argument("--year", type=int, help="下载单一年份")
    parser.add_argument("--start", type=int, help="起始年份")
    parser.add_argument("--end", type=int, help="结束年份")
    parser.add_argument("--codes", type=str, nargs="*", help="指定股票代码")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有数据的股票")
    args = parser.parse_args()

    output_path = Path(args.output)
    this_year = date.today().year

    if args.year:
        years = [args.year]
    elif args.start and args.end:
        years = list(range(args.start, args.end + 1))
    else:
        years = list(range(this_year - 5, this_year + 1))

    if args.codes:
        stocks = [{"ts_code": c, "name": c, "exchange": c[-2:]} for c in args.codes]
    else:
        stocks = get_stock_list()

    # 计算任务量
    total_tasks = 0
    year_tasks = {}
    for y in years:
        if args.skip_existing:
            completed = get_completed_stocks(y, output_path)
            todo = [s for s in stocks if s["ts_code"] not in completed]
        else:
            todo = stocks
        year_tasks[y] = todo
        total_tasks += len(todo)

    print(f"[配置] {len(stocks)} 只股票 x {len(years)} 年 = {total_tasks} 个任务")
    print(f"[预估] 每个任务约 0.5-1 秒, 总耗时约 {total_tasks * 0.75 / 3600:.1f} 小时")
    print(f"[输出] {output_path}")
    print()

    overall_start = time.time()
    total_rows = 0
    all_failed = []

    for y in years:
        todo = year_tasks[y]
        print(f"{'='*50}")
        print(f"  {y}年: {len(todo)} 只股票")
        print(f"{'='*50}")

        if not todo:
            print("  [跳过] 该年份已完整")
            continue

        batch_data = []
        stock_ok = 0
        stock_fail = 0
        batch_sz = 200  # 每 200 只保存一次

        pbar = tqdm(todo, desc=f"  {y}年 5min", unit="stock")
        for stock in pbar:
            df = fetch_one_stock(stock["ts_code"], y)
            if df is not None and not df.empty:
                batch_data.append(df)
                stock_ok += 1
            else:
                stock_fail += 1
                all_failed.append((stock["ts_code"], y))

            pbar.set_postfix({"ok": stock_ok, "fail": stock_fail})

            # 批量保存
            if len(batch_data) >= batch_sz:
                merged = pd.concat(batch_data, ignore_index=True)
                save_parquet(merged, output_path)
                total_rows += len(merged)
                batch_data = []

            # 轻微延迟防止被限速
            time.sleep(0.05)

        # 保存剩余
        if batch_data:
            merged = pd.concat(batch_data, ignore_index=True)
            save_parquet(merged, output_path)
            total_rows += len(merged)

        print(f"  {y}年完成: 成功 {stock_ok}, 失败 {stock_fail}")

    elapsed = time.time() - overall_start
    avg_sec = elapsed / max(total_tasks, 1)

    print(f"\n{'='*50}")
    print(f"[完成] 总耗时: {elapsed/3600:.1f} 小时 ({elapsed/60:.0f} 分钟)")
    print(f"[完成] 总行数: {total_rows:,}")
    print(f"[完成] 平均每只: {avg_sec:.1f} 秒")
    print(f"[完成] 失败任务: {len(all_failed)}")
    print(f"{'='*50}")

    if all_failed:
        print(f"\n[失败] 前20:")
        for code, y in all_failed[:20]:
            print(f"  {code} ({y}年)")


if __name__ == "__main__":
    main()
