"""
抓取 baostock 参考数据：证监会行业分类 + 上市退市日期
=====================================================
- query_stock_industry(): 证监会行业分类 (updateDate, code, name, industry)
- query_stock_basic():     上市日/退市日/状态
- 代码格式转换: sh.600000 -> 600000.SH
- 存 parquet 到 quant-system/data/reference/
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import baostock as bs
import pandas as pd

OUT_DIR = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"

lg = bs.login()
print("login:", lg.error_code, lg.error_msg)

# ---- 1. 行业分类 (证监会) ----
t0 = time.time()
rs = bs.query_stock_industry()
rows = []
while (rs.error_code == "0") and rs.next():
    rows.append(rs.get_row_data())
ind = pd.DataFrame(rows, columns=rs.fields)
print(f"行业分类: {len(ind)} 行  ({time.time()-t0:.0f}s)")
ind["updateDate"] = pd.to_datetime(ind["updateDate"], errors="coerce")


def conv(c):
    # sh.600000 -> 600000.SH ; sz.000001 -> 000001.SZ ; bj.920000 -> 920000.BJ
    if "." not in c:
        return c
    ex, num = c.split(".")
    return f"{num}.{ex.upper()}"


ind["ts_code"] = ind["code"].map(conv)
ind = ind.rename(columns={"code_name": "name"})
n_empty = ind["industry"].eq("").sum()
print(f"  行业字段为空: {n_empty} / {len(ind)}  (退市股无证监会分类)")
print(f"  行业大类数: {ind['industry'].str[:3].nunique()}")
ind.to_parquet(f"{OUT_DIR}/csrc_industry.parquet", index=False)
print("  [OK] csrc_industry.parquet")

# ---- 2. 基本信息 (上市/退市日期) ----
t0 = time.time()
rs = bs.query_stock_basic()
rows = []
while (rs.error_code == "0") and rs.next():
    rows.append(rs.get_row_data())
basic = pd.DataFrame(rows, columns=rs.fields)
print(f"\n基本信息: {len(basic)} 行  ({time.time()-t0:.0f}s)")
print(f"  字段: {list(basic.columns)}")
basic["ts_code"] = basic["code"].map(conv)
basic = basic.rename(columns={"code_name": "name"})
basic["ipoDate"] = pd.to_datetime(basic["ipoDate"], errors="coerce")
basic["outDate"] = pd.to_datetime(basic["outDate"], errors="coerce")
n_delist = basic["outDate"].notna().sum()
n_2018 = ((basic["ipoDate"] < "2018-01-01")).sum()
print(f"  有退市日期: {n_delist} 只;  2018年前上市: {n_2018} 只")
basic.to_parquet(f"{OUT_DIR}/stock_basic_bs.parquet", index=False)
print("  [OK] stock_basic_bs.parquet")

# ---- 3. 周期相关大类预览 ----
CYC_PREFIX = ["B06", "B07", "B08", "B09", "C25", "C26", "C28", "C30", "C31", "C32"]
ind3 = ind[ind["industry"].str[:3].isin(CYC_PREFIX)]
print(f"\n周期相关证监会大类 (B06煤/B07油气开采/B08黑矿/B09有色矿/C25炼焦/"
      f"C26化学原料/C28化纤/C30非金属矿物/C31钢铁冶炼/C32有色冶炼):")
print(f"  覆盖 {len(ind3)} 只 (行业非空的)")
print(ind3.groupby(ind3["industry"].str[:3]).size().to_string())

bs.logout()
print("\n[完成]")
