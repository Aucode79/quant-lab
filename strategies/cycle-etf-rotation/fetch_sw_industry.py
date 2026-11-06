"""
抓取申万三级行业成分 (335个指数) — 全市场细分行业标签
=====================================================
产物: reference/sw_industry_map.parquet
      ts_code / sw一级 / sw二级 / sw三级
然后回填 终审1/标的池 增强版CSV的 细分行业 列。
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd

BASE = "C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股"
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"

idx = ak.sw_index_third_info()
print(f"申万三级指数: {len(idx)} 个")

# 二级映射: 上级行业列是二级名
third = idx[["行业代码", "行业名称", "上级行业"]].copy()
third.columns = ["sw_code", "sw三级", "sw二级"]

rows = []
t0 = time.time()
fails = 0
for i, r in enumerate(third.itertuples()):
    try:
        cons = ak.sw_index_third_cons(symbol=r.sw_code)
        col = "股票代码" if "股票代码" in cons.columns else cons.columns[0]
        for c in cons[col]:
            rows.append({"num": str(c), "sw三级": r.sw三级, "sw二级": r.sw二级})
    except Exception:
        fails += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(third)}  累计{len(rows)}行  ({time.time()-t0:.0f}s)  失败{fails}", flush=True)
    time.sleep(0.1)

m = pd.DataFrame(rows).drop_duplicates("num", keep="first")
print(f"\n成分抓取完成: {len(m)} 只覆盖, 失败指数 {fails} 个 ({time.time()-t0:.0f}s)")

# 一级分类: 用二级→一级映射再抓一次太重, 从三级信息推不出来, 抓二级列表对齐
try:
    sec = ak.sw_index_second_info()
    sec2first = dict(zip(sec["行业名称"], sec["上级行业"]))
    m["sw一级"] = m["sw二级"].map(sec2first).fillna("")
except Exception as e:
    sys.stderr.write(f"一级映射FAIL: {e!r}\n")
    m["sw一级"] = ""

m["ts_code"] = m["num"].map(
    lambda x: f"{x}.SH" if x.startswith(("6", "9")) else
              f"{x}.SZ" if x.startswith(("0", "2", "3")) else f"{x}.BJ")
m[["ts_code", "sw一级", "sw二级", "sw三级"]].to_parquet(
    f"{REF}/sw_industry_map.parquet", index=False)
print(f"[已保存] {REF}/sw_industry_map.parquet")

# ============ 回填增强版 CSV ============
sw = pd.read_parquet(f"{REF}/sw_industry_map.parquet")
for fn, tag in [("标的池_分层清单_增强版.csv", "pool"),
                ("终审1_入池但非官方周期_增强版.csv", "review")]:
    try:
        df = pd.read_csv(f"{BASE}/{fn}")
    except FileNotFoundError:
        continue
    df = df.drop(columns=["申万三级"], errors="ignore")
    df = df.merge(sw.rename(columns={"sw三级": "申万三级",
                                     "sw二级": "申万二级"}),
                  on="ts_code", how="left")
    # cninfo行业为空的用申万补
    if "cninfo行业" in df.columns:
        df["cninfo行业"] = df["cninfo行业"].fillna(df["申万二级"])
    df.to_csv(f"{BASE}/{fn}", index=False, encoding="utf-8-sig")
    print(f"[回填] {fn}: 申万覆盖 {df['申万三级'].notna().sum()}/{len(df)}")

rev = pd.read_csv(f"{BASE}/终审1_入池但非官方周期_增强版.csv")
print("\n[终审1 最终形态示例] (按建议判定分组, 各3只)")
for v in rev["建议判定"].unique():
    sub = rev[rev["建议判定"] == v].head(3)
    print(f"\n  {v}:")
    for _, r in sub.iterrows():
        print(f"    {r['ts_code']} {r['名称']:<6} [{str(r['申万三级'])[:8]:<8}] "
              f"{str(r['主营业务'])[:40]}")
