#!/usr/bin/env python3
"""
获取全A股日K线数据（前复权），写入 Parquet 文件。

数据源: baostock (免费，无需token，稳定)
输出: data/market/kline_daily.parquet (按年月分区)

特性:
  - 自动跳过已下载的日期范围（增量更新）
  - 前复权OHLCV + 复权因子
  - 数据完整性校验

Usage:
    python fetch_kline_daily.py                        # 下载最近10年
    python fetch_kline_daily.py --start 2015-01-01     # 指定起始日期
    python fetch_kline_daily.py --update               # 增量更新最近30天
"""

import argparse
import sys
import io
import time
from datetime import date, datetime, timedelta
from pathlib import Path

# Fix GBK encoding issue on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import baostock as bs
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "market"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "kline_daily.parquet"
PARTITION_COLUMN = "year"


def get_stock_list(data_dir: Path) -> list[str]:
    """从 stock_basic.parquet 获取全部股票代码列表。"""
    stock_basic_path = data_dir.parent / "reference" / "stock_basic.parquet"
    if stock_basic_path.exists():
        con = duckdb.connect(":memory:")
        codes = con.execute(
            f"SELECT ts_code FROM '{stock_basic_path}' ORDER BY ts_code"
        ).fetchall()
        con.close()
        return [row[0] for row in codes]
    else:
        # Fallback: 使用 baostock 获取股票列表
        print("[fallback] 从 baostock 获取股票列表...")
        bs.login()
        rs = bs.query_stock_basic()
        data = []
        while (rs.error_code == '0') & rs.next():
            data.append(rs.get_row_data())
        bs.logout()
        df = pd.DataFrame(data, columns=rs.fields)
        # baostock code 格式: sh.600000 → 600000.SH
        codes = []
        for _, row in df.iterrows():
            if row["type"] == "1":  # 股票
                parts = row["code"].split(".")
                if len(parts) == 2:
                    codes.append(f"{parts[1]}.{parts[0].upper()}")
        return codes


def bao_code(ts_code: str) -> str:
    """转换 '600000.SH' → 'sh.600000' """
    parts = ts_code.split(".")
    return f"{parts[1].lower()}.{parts[0]}"


def fetch_single_stock(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    获取单只股票的日K线数据。
    返回 DataFrame，失败返回 None。
    """
    try:
        rs = bs.query_history_k_data_plus(
            bao_code(ts_code),
            "date,open,high,low,close,preclose,volume,amount,"
            "adjustflag,turn,tradestatus,pctChg,isST",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"  # 前复权
        )

        if rs.error_code != '0':
            return None

        data = []
        while (rs.error_code == '0') & rs.next():
            data.append(rs.get_row_data())

        if not data:
            return None

        df = pd.DataFrame(data, columns=rs.fields)

        # 类型转换
        numeric_cols = ["open", "high", "low", "close", "preclose",
                        "volume", "amount", "turn", "pctChg"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 过滤无效数据
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        if df.empty:
            return None

        # 标准化列名和代码
        df["ts_code"] = ts_code
        df["trade_date"] = pd.to_datetime(df["date"]).dt.date
        df["is_st"] = df["isST"].apply(lambda x: x == "1")
        df["is_suspend"] = df["tradestatus"].apply(lambda x: x == "0")

        # baostock 前复权模式下，adjfactor 固定为 1
        # 如需后复权因子，需单独查询 adjustflag=1 计算
        df["adj_factor"] = 1.0

        # 选择最终列
        result = df[[
            "ts_code", "trade_date", "open", "high", "low", "close",
            "preclose", "volume", "amount", "adj_factor", "is_st", "is_suspend"
        ]].copy()

        return result

    except Exception as e:
        # 静默处理单只股票的错误
        return None


def fetch_all_stocks(
    stock_codes: list[str],
    start_date: str,
    end_date: str,
    existing_dates: set[str] | None = None,
) -> pd.DataFrame:
    """
    遍历所有股票获取K线数据。
    existing_dates: 已完成下载的日期集合（用于增量更新跳过）。
    """
    all_data = []
    failed = []

    bs.login()

    pbar = tqdm(stock_codes, desc="下载日K线", unit="stock")
    for ts_code in pbar:
        df = fetch_single_stock(ts_code, start_date, end_date)
        if df is not None:
            if existing_dates:
                df = df[~df["trade_date"].astype(str).isin(existing_dates)]
            if not df.empty:
                all_data.append(df)

            pbar.set_postfix({
                "ok": len(all_data),
                "fail": len(failed),
                "code": ts_code[:6]
            })
        else:
            failed.append(ts_code)

    bs.logout()

    if failed:
        print(f"\n[WARN] {len(failed)} 只股票数据获取失败")

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    result["year"] = pd.to_datetime(result["trade_date"]).dt.year

    return result


def save_parquet_partitioned(df: pd.DataFrame, base_path: Path) -> None:
    """
    保存为分区Parquet文件。
    使用 pyarrow 的 partition_cols 按年份分区。
    """
    if df.empty:
        print("[WARN] 无新数据需要保存")
        return

    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(
        table,
        root_path=str(base_path),
        partition_cols=["year"],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )
    print(f"[OK] 已保存 {len(df)} 条日K线数据到: {base_path}")


def verify_data(path: Path) -> None:
    """使用 DuckDB 验证数据完整性。"""
    con = duckdb.connect(":memory:")
    try:
        stats = con.execute(f"""
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT ts_code) AS stocks,
                MIN(trade_date) AS start_date,
                MAX(trade_date) AS end_date,
                COUNT(DISTINCT trade_date) AS trade_days
            FROM '{path}/**/*.parquet'
        """).fetchone()

        print(f"\n[数据概览]")
        print(f"   总K线条数: {stats[0]:,}")
        print(f"   覆盖股票:  {stats[1]:,} 只")
        print(f"   时间范围:  {stats[2]} ~ {stats[3]}")
        print(f"   交易日数:  {stats[4]} 天")

        # 检查每天的股票数量（粗略完整性检查）
        daily_check = con.execute(f"""
            SELECT trade_date, COUNT(*) AS cnt
            FROM '{path}/**/*.parquet'
            GROUP BY trade_date
            ORDER BY trade_date
        """).fetchdf()
        if not daily_check.empty:
            median_cnt = daily_check["cnt"].median()
            print(f"   日均股票数: {median_cnt:.0f} 只")
    finally:
        con.close()


def get_existing_dates(path: Path) -> set[str]:
    """获取已有数据的日期列表，用于增量更新。"""
    if not path.exists() or not list(path.glob("*.parquet")):
        return set()

    try:
        con = duckdb.connect(":memory:")
        result = con.execute(f"""
            SELECT DISTINCT CAST(trade_date AS VARCHAR) AS d
            FROM '{path}/**/*.parquet'
        """).fetchall()
        con.close()
        dates = {row[0] for row in result}
        print(f"[已有数据] {len(dates)} 个交易日")
        return dates
    except Exception:
        return set()


def main():
    parser = argparse.ArgumentParser(description="获取A股日K线数据")
    parser.add_argument("--start", type=str, default=None,
                        help="起始日期 (默认: 10年前)")
    parser.add_argument("--end", type=str, default=None,
                        help="结束日期 (默认: 今天)")
    parser.add_argument("--update", action="store_true",
                        help="增量更新模式（只下载最近30天）")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"输出路径 (默认: {OUTPUT_PATH})")
    parser.add_argument("--codes", type=str, nargs="*",
                        help="指定股票代码列表 (不指定则全部)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过数据验证")
    args = parser.parse_args()

    output_path = Path(args.output)

    # 确定日期范围
    today = date.today()
    if args.end:
        end_date = args.end
    else:
        end_date = today.strftime("%Y-%m-%d")

    if args.update:
        # 增量更新：最近30天
        start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")  # 60天覆盖周末
        print(f"[更新模式] {start_date} ~ {end_date}")
    elif args.start:
        start_date = args.start
    else:
        start_date = (today - timedelta(days=365 * 10 + 30)).strftime("%Y-%m-%d")
        print(f"[默认下载10年] {start_date} ~ {end_date}")

    # 获取股票列表
    if args.codes:
        stock_codes = args.codes
    else:
        stock_codes = get_stock_list(DATA_DIR.parent)
        if not stock_codes:
            print("[ERROR] 无法获取股票列表，请先运行 fetch_stock_basic.py")
            sys.exit(1)

    print(f"[待下载] {len(stock_codes)} 只股票")
    print(f"[日期范围] {start_date} ~ {end_date}")

    # 获取已有日期（增量模式）
    existing_dates = get_existing_dates(output_path) if not args.update else set()

    start_time = time.time()

    # 批量下载
    df = fetch_all_stocks(stock_codes, start_date, end_date, existing_dates)

    elapsed = time.time() - start_time
    print(f"\n[耗时] {elapsed:.1f} 秒")

    # 保存
    save_parquet_partitioned(df, output_path)

    # 验证
    if not args.skip_verify and not df.empty:
        verify_data(output_path)

    print("\n[Done] 完成！")


if __name__ == "__main__":
    main()
