#!/usr/bin/env python3
"""数据验证脚本——用 DuckDB 直接查 Parquet 验证数据完整性"""

import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import duckdb

con = duckdb.connect(":memory:")

print("=" * 60)
print("  数据验证报告")
print("=" * 60)

# ── 1. 股票完整性检查 ──
print("\n[1] 每年股票数与K线完整性:")
stock_stats = con.execute("""
    SELECT year, COUNT(*) as rows, COUNT(DISTINCT ts_code) as stocks,
           COUNT(DISTINCT trade_date) as days,
           MIN(trade_date) as d0, MAX(trade_date) as d1,
           SUM(CASE WHEN close IS NULL OR open IS NULL THEN 1 ELSE 0 END) as nulls,
           SUM(CASE WHEN high + 0.001 < open OR high + 0.001 < close THEN 1 ELSE 0 END) as bad_high,
           SUM(CASE WHEN low - 0.001 > open OR low - 0.001 > close THEN 1 ELSE 0 END) as bad_low
    FROM 'data/market/kline_1min.parquet/**/*.parquet'
    GROUP BY year ORDER BY year
""").fetchall()

ok_all = True
for r in stock_stats:
    year, rows, stks, days, d0, d1, nulls, bh, bl = r
    avg_rows = rows / stks if stks else 0
    problems = []
    if nulls: problems.append(f"null={nulls}")
    if bh: problems.append(f"bad_high={bh}")
    if bl: problems.append(f"bad_low={bl}")
    status = "[OK]" if not problems else f"[BAD] {' '.join(problems)}"
    if problems: ok_all = False
    print(f"  {year}: {rows:>12,}行 {stks:>5}只 {days:>3}天 {str(d0)[:10]}~{str(d1)[:10]} | 日均{avg_rows:,.0f}条 {status}")

# ── 2. 抽样检查具体股票 ──
print("\n[2] 抽样股票详情 (600000.SH, 000001.SZ):")
for code in ["600000.SH", "000001.SZ"]:
    sample = con.execute(f"""
        SELECT year, COUNT(*) as rows, COUNT(DISTINCT trade_date) as days,
               MIN(trade_date) as d0, MAX(trade_date) as d1,
               MIN(close) as lo, MAX(close) as hi,
               AVG(volume) as avg_vol
        FROM 'data/market/kline_1min.parquet/**/*.parquet'
        WHERE ts_code = '{code}'
        GROUP BY year ORDER BY year
        LIMIT 3
    """).fetchall()
    for r in sample:
        y, rows, days, d0, d1, lo, hi, vol = r
        print(f"  {code} {y}: {rows:>8,}行 {days}天 收盘{lo:.2f}~{hi:.2f} 均量{vol:,.0f}")

# ── 3. 日线闭合检查：分钟线合成日线 vs 实际日线 ──
print("\n[3] 日线闭合检查 (每日开/高/低/收 与分钟线首/最高/最低/尾对比):")
for code in ["600000.SH", "000001.SZ", "300750.SZ"]:
    try:
        r = con.execute(f"""
            WITH daily AS (
                SELECT ts_code, trade_date,
                    FIRST(open ORDER BY trade_time) AS day_open,
                    MAX(high) AS day_high,
                    MIN(low) AS day_low,
                    LAST(close ORDER BY trade_time) AS day_close,
                    COUNT(*) AS bars
                FROM 'data/market/kline_1min.parquet/**/*.parquet'
                WHERE ts_code = '{code}' AND year = 2024
                GROUP BY ts_code, trade_date
            )
            SELECT AVG(bars) as avg_bars, COUNT(*) as days,
                   SUM(CASE WHEN bars < 200 THEN 1 ELSE 0 END) AS short_days,
                   SUM(CASE WHEN bars > 250 THEN 1 ELSE 0 END) AS long_days
            FROM daily
            WHERE day_open > 0 AND day_close > 0
        """).fetchone()
        if r and r[1]:
            avg, days, short, long = r
            print(f"  {code} 2024: {days}天 日均{avg:.0f}bar short={short} long={long} [OK]")
    except Exception as e:
        print(f"  {code}: ERROR - {e}")

# ── 4. 跨年复权连续性检查 ──
print("\n[4] 复权连续性检查:")
for code in ["600000.SH"]:
    try:
        r = con.execute(f"""
            WITH year_close AS (
                SELECT year,
                    LAST(close ORDER BY trade_time) AS last_close,
                    FIRST(close ORDER BY trade_time) AS first_close
                FROM 'data/market/kline_1min.parquet/**/*.parquet'
                WHERE ts_code = '{code}'
                GROUP BY year
                ORDER BY year
            )
            SELECT y1.year, y1.last_close, y2.first_close,
                   CASE WHEN y2.first_close > 0
                        THEN ROUND((y2.first_close - y1.last_close) / y2.first_close * 100, 2)
                        ELSE NULL END AS jump_pct
            FROM year_close y1
            JOIN year_close y2 ON y1.year + 1 = y2.year
        """).fetchall()
        for r in r:
            y, last, first, jump = r
            if jump and abs(jump) > 10:
                print(f"  {code} {y}→{y+1}: 收盘{last:.2f}→开盘{first:.2f} 跳{'+' if jump>0 else ''}{jump:.1f}% [WARN]")
            else:
                print(f"  {code} {y}→{y+1}: 连续 ✅")
    except Exception as e:
        print(f"  {code}: {e}")

# ── 5. 汇总 ──
print("\n" + "=" * 60)
total = con.execute("""
    SELECT COUNT(*) FROM 'data/market/kline_1min.parquet/**/*.parquet'
""").fetchone()[0]
size_mb = sum(
    f.stat().st_size for f in __import__('pathlib').Path("data/market/kline_1min.parquet").rglob("*.parquet")
) / 1024 / 1024
if ok_all:
    print(f"  结论: 数据完整，{total:,}行, {size_mb:.0f}MB, 全部验证通过 ✅")
else:
    print(f"  结论: {total:,}行, {size_mb:.0f}MB, 有问题需修复 ⚠️")
print("=" * 60)

con.close()
