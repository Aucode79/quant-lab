#!/usr/bin/env python3
"""
获取全A股股票基本信息，写入 Parquet 文件。

数据源: akshare (免费，无需token)
输出: data/reference/stock_basic.parquet

Usage:
    python fetch_stock_basic.py
    python fetch_stock_basic.py --output ../data/reference/stock_basic.parquet
"""

import argparse
import sys
import io
from pathlib import Path

# Fix GBK encoding issue on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import akshare as ak
import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reference"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "stock_basic.parquet"


def fetch_stock_basic() -> pd.DataFrame:
    """
    从 akshare 获取沪深京三市A股基本信息。
    返回标准化的 DataFrame，列名与 schema 一致。
    """
    print("[1/3] 获取沪深A股列表...")
    df_sh_sz = ak.stock_info_a_code_name()
    # stock_info_a_code_name 返回列: code, name
    # code 示例: 'sh600000', 'sz000001'

    print(f"      沪深A股共 {len(df_sh_sz)} 只")

    print("[2/3] 获取北交所股票列表...")
    try:
        # Try different API names depending on akshare version
        try:
            df_bj = ak.stock_info_bj_a_code_name()
        except AttributeError:
            try:
                df_bj = ak.stock_bj_a_code_name()
            except AttributeError:
                df_bj = ak.stock_info_bj_code_name()
        print(f"      北交所股票共 {len(df_bj)} 只")
    except Exception as e:
        print(f"      北交所获取失败（暂不支持）: {e}")
        df_bj = pd.DataFrame(columns=["code", "name"])

    # 合并
    df_all = pd.concat([df_sh_sz, df_bj], ignore_index=True)

    # 标准化代码格式: 纯数字 → 数字.交易所后缀
    def normalize_code(raw: str) -> str:
        raw = str(raw).strip()
        # 如果已有前缀格式 'sh600000' → '600000.SH'
        if raw.startswith("sh"):
            return raw[2:] + ".SH"
        elif raw.startswith("sz"):
            return raw[2:] + ".SZ"
        elif raw.startswith("bj"):
            return raw[2:] + ".BJ"

        # 纯数字格式：根据编码规则判断交易所
        # 深交所主板: 000xxx, 001xxx, 002xxx, 003xxx
        # 深交所创业板: 300xxx, 301xxx
        # 上交所主板: 600xxx, 601xxx, 603xxx, 605xxx
        # 上交所科创板: 688xxx, 689xxx
        # 北交所: 4xxxxx, 8xxxxx, 9xxxxx
        if raw.startswith(("000", "001", "002", "003")):
            return raw + ".SZ"
        elif raw.startswith(("300", "301")):
            return raw + ".SZ"
        elif raw.startswith(("600", "601", "603", "605")):
            return raw + ".SH"
        elif raw.startswith(("688", "689")):
            return raw + ".SH"
        elif raw.startswith(("4", "8", "9")):
            return raw + ".BJ"
        else:
            return raw + ".UNKNOWN"

    df_all["ts_code"] = df_all["code"].apply(normalize_code)
    df_all["name"] = df_all["name"].str.strip()

    # 尝试获取行业分类（申万一级）
    print("[3/3] 获取行业分类...")
    try:
        industry_df = ak.stock_board_industry_name_em()
        # 这个返回的是行业板块，需要反向映射——简化处理，先设为UNKNOWN
        # 后续可通过 tushare 的 stock_basic 接口补充
        df_all["industry"] = "UNKNOWN"
        print("      行业分类暂设为 UNKNOWN（后续可通过tushare补充）")
    except Exception:
        df_all["industry"] = "UNKNOWN"

    # 交易所
    def get_exchange(code: str) -> str:
        if code.endswith(".SH"):
            return "SSE"
        elif code.endswith(".SZ"):
            return "SZSE"
        elif code.endswith(".BJ"):
            return "BSE"
        return "UNKNOWN"

    df_all["exchange"] = df_all["ts_code"].apply(get_exchange)

    # 添加默认日期字段（后续可用 tushare 精确补充）
    df_all["list_date"] = None
    df_all["delist_date"] = None

    # 最终列顺序
    result = df_all[["ts_code", "name", "industry", "list_date", "delist_date", "exchange"]]
    return result


def save_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """保存为 Parquet 文件，使用 Snappy 压缩。"""
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        str(path),
        compression="snappy",
        # 按 exchange 分区有助于后续查询优化
    )
    print(f"\n[OK] 已保存 {len(df)} 只股票信息到: {path}")


def verify_with_duckdb(path: Path) -> None:
    """使用 DuckDB 验证 Parquet 文件完整性。"""
    con = duckdb.connect(":memory:")
    result = con.execute(f"""
        SELECT
            COUNT(*) AS total_stocks,
            COUNT(DISTINCT ts_code) AS unique_codes,
            COUNT(DISTINCT exchange) AS exchanges
        FROM '{path}'
    """).fetchone()
    print(f"\n[验证] 总计 {result[0]} 只, "
          f"唯一代码 {result[1]} 个, "
          f"覆盖 {result[2]} 个交易所")

    # 按交易所统计
    exchange_stats = con.execute(f"""
        SELECT exchange, COUNT(*) AS cnt
        FROM '{path}'
        GROUP BY exchange
        ORDER BY cnt DESC
    """).fetchall()
    print("   交易所分布:")
    for row in exchange_stats:
        print(f"     {row[0]}: {row[1]} 只")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="获取A股股票基本信息")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help=f"输出路径 (默认: {OUTPUT_PATH})")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过 DuckDB 验证")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = fetch_stock_basic()
        save_to_parquet(df, output_path)

        if not args.skip_verify:
            verify_with_duckdb(output_path)

        print("\n[Done] 完成！")
    except Exception as e:
        print(f"\n[ERROR] 错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
