"""快速验证 Parquet 数据质量"""
import sys, io, duckdb
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import argparse

p = argparse.ArgumentParser()
p.add_argument("--type", default="etf", choices=["stock","etf"])
p.add_argument("--year", type=int, default=0)
args = p.parse_args()

fname = "kline_1min_etf.parquet" if args.type == "etf" else "kline_1min.parquet"
con = duckdb.connect(":memory:")

# Get years
if args.year:
    years = [args.year]
else:
    try:
        rows = con.execute(f"SELECT DISTINCT year FROM 'data/market/{fname}/**/*.parquet' ORDER BY year").fetchall()
        years = [r[0] for r in rows]
    except:
        print(f"No data for {args.type}")
        con.close()
        sys.exit(0)

for y in years:
    try:
        r = con.execute(f"""
            SELECT COUNT(*), COUNT(DISTINCT ts_code),
                   MIN(trade_date), MAX(trade_date)
            FROM 'data/market/{fname}/**/*.parquet'
            WHERE year = {y}
        """).fetchone()
        total, stks, d0, d1 = r

        # Quality checks
        bad = con.execute(f"""
            SELECT
                SUM(CASE WHEN close IS NULL OR open IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN high + 0.001 < open OR high + 0.001 < close THEN 1 ELSE 0 END),
                SUM(CASE WHEN low - 0.001 > open OR low - 0.001 > close THEN 1 ELSE 0 END)
            FROM 'data/market/{fname}/**/*.parquet'
            WHERE year = {y}
        """).fetchone()
        nulls, bad_high, bad_low = bad

        ok = (nulls == 0 and bad_high == 0 and bad_low == 0)
        status = "[OK]" if ok else "[BAD]"
        print(f"{args.type.upper()} {y}: {total:>12,} rows | {stks:>5} stocks | {str(d0)[:10]} ~ {str(d1)[:10]} | null={nulls} hi={bad_high} lo={bad_low} {status}")
    except Exception as e:
        print(f"{args.type.upper()} {y}: ERROR - {str(e)[:80]}")

con.close()
