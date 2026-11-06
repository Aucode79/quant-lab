"""
注册表 v2 全量概况抓取 — 补齐新增类别的公司资料
=====================================================
只抓注册表中缺概况的股票, 增量合并进 profile_cninfo.parquet
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd

BASE = "C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股"
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"

reg = pd.read_csv(f"{BASE}/标的池_注册表_v2_20260913.csv")
try:
    prof = pd.read_parquet(f"{REF}/profile_cninfo.parquet")
except FileNotFoundError:
    prof = pd.DataFrame(columns=["ts_code", "公司全称", "cninfo行业", "主营业务", "经营范围"])
have = set(prof["ts_code"])
todo = [c for c in reg["ts_code"] if c not in have]
print(f"注册表 {len(reg)} 只, 已有概况 {len(have)}, 待抓 {len(todo)}")

rows = []
t0 = time.time()
fails = []
for i, code in enumerate(todo):
    try:
        p = ak.stock_profile_cninfo(symbol=code.split(".")[0])
        if p is not None and len(p):
            r = p.iloc[0]
            rows.append({"ts_code": code,
                         "公司全称": r.get("公司名称", ""),
                         "cninfo行业": r.get("所属行业", ""),
                         "主营业务": str(r.get("主营业务", ""))[:300],
                         "经营范围": str(r.get("经营范围", ""))[:300]})
    except Exception:
        fails.append(code)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(todo)}  累计{len(rows)}  失败{len(fails)}  ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.25)

new = pd.concat([prof, pd.DataFrame(rows)], ignore_index=True)
new.to_parquet(f"{REF}/profile_cninfo.parquet", index=False)
print(f"\n[完成] 概况库共 {len(new)} 只; 本次成功 {len(rows)}, 失败 {len(fails)}")
if fails:
    print("失败名单(多为退市/重组股):", fails[:20])
