"""
标的池增强 — 抓主营业务/细分行业/市值, 自动预判, 减轻人工终审负担
=====================================================
只读增强，不改变池子结构，不构成投资建议。

数据源 (akshare, 2026-09 网络已恢复):
  1. cninfo 公司概况(逐只): 主营业务/经营范围/所属行业
  2. 东财行业板块成分(86板块×成分): stock → 细分行业标签
  3. 东财全A快照(1次): 总市值/换手率
产物:
  reference/profile_cninfo.parquet / em_industry_map.parquet / em_spot.parquet
  标的池_分层清单_增强版.csv
  终审1_入池但非官方周期_增强版.csv  (含"建议判定"列)
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import akshare as ak
import pandas as pd
import numpy as np

BASE = "C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股"
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"

pool = pd.read_csv(f"{BASE}/标的池_分层清单.csv")
targets = pool[pool["层级"].isin(["L1核心", "L2扩展", "漂移"])].copy()
codes = targets["ts_code"].tolist()
print(f"[目标] L1+L2+漂移 共 {len(codes)} 只")

# ============ 1. cninfo 公司概况 ============
def num2ts(c):
    ex = c.split(".")[1]
    return c.split(".")[0] + (".SZ" if ex == "SZ" else ".SH" if ex == "SH" else ".BJ")


rows = []
t0 = time.time()
for i, code in enumerate(codes):
    sym = code.split(".")[0]
    prof = None
    for attempt in range(2):
        try:
            prof = ak.stock_profile_cninfo(symbol=sym)
            break
        except Exception:
            time.sleep(1.0)
    if prof is not None and len(prof):
        r = prof.iloc[0]
        rows.append({
            "ts_code": code,
            "公司全称": r.get("公司名称", ""),
            "cninfo行业": r.get("所属行业", ""),
            "主营业务": str(r.get("主营业务", ""))[:300],
            "经营范围": str(r.get("经营范围", ""))[:300],
        })
    if (i + 1) % 40 == 0:
        print(f"  cninfo {i+1}/{len(codes)}  ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.25)
prof_df = pd.DataFrame(rows)
prof_df.to_parquet(f"{REF}/profile_cninfo.parquet", index=False)
print(f"[1/3] cninfo概况: {len(prof_df)}/{len(codes)}  ({time.time()-t0:.0f}s)")

# ============ 2. 东财行业板块成分 ============
try:
    boards = ak.stock_board_industry_name_em()
    maps = []
    for bi, bname in enumerate(boards["板块名称"]):
        try:
            cons = ak.stock_board_industry_cons_em(symbol=bname)
            for c in cons["代码"]:
                maps.append({"num": str(c), "东财细分行业": bname})
        except Exception:
            pass
        if (bi + 1) % 20 == 0:
            print(f"  em板块 {bi+1}/{len(boards)}", flush=True)
        time.sleep(0.15)
    em_map = pd.DataFrame(maps).drop_duplicates("num", keep="first")
    em_map.to_parquet(f"{REF}/em_industry_map.parquet", index=False)
    print(f"[2/3] 东财行业成分: {len(em_map)} 只覆盖")
except Exception as e:
    sys.stderr.write(f"东财行业FAIL: {e!r}\n")
    em_map = pd.DataFrame(columns=["num", "东财细分行业"])

# ============ 3. 东财全A快照 ============
try:
    spot = ak.stock_zh_a_spot_em()
    spot = spot[["代码", "名称", "总市值", "换手率"]].copy()
    spot.columns = ["num", "em名称", "总市值", "换手率"]
    spot["num"] = spot["num"].astype(str)
    spot.to_parquet(f"{REF}/em_spot.parquet", index=False)
    print(f"[3/3] 东财快照: {len(spot)} 只")
except Exception as e:
    sys.stderr.write(f"东财快照FAIL: {e!r}\n")
    spot = pd.DataFrame(columns=["num", "em名称", "总市值", "换手率"])

# ============ 合并进清单 ============
targets["num"] = targets["ts_code"].str.split(".").str[0]
targets = targets.merge(prof_df, on="ts_code", how="left")
targets = targets.merge(em_map, on="num", how="left")
targets = targets.merge(spot, on="num", how="left")
targets["总市值亿"] = (pd.to_numeric(targets["总市值"], errors="coerce") / 1e8).round(1)
targets = targets.drop(columns=["num", "总市值"], errors="ignore")

# ============ 自动预判 (终审1: 入池但非官方周期) ============
EXCL_KW = ["银行", "证券", "保险", "信托", "白酒", "啤酒", "影视", "游戏", "广告",
           "教育", "旅游", "免税", "软件", "互联网", "电商平台", "房地产经纪",
           "物业", "医院", "口腔", "眼科", "疫苗", "创新药"]
KEEP_KW = ["铜", "铝", "锌", "铅", "镍", "锡", "稀土", "黄金", "铁矿", "煤炭", "焦炭",
           "钢", "冶炼", "采选", "矿山", "水泥", "浮法玻璃", "化肥", "磷", "钾",
           "纯碱", "氯碱", "农药", "染料", "涤纶", "粘胶", "炼化", "石油", "化工",
           "化纤", "航运", "造船", "工程机械", "租赁", "玻璃纤维"]
BOTH_KW = ["房地产", "建筑", "港口", "装修", "电力"]


def suggest(r):
    txt = f"{r.get('主营业务','')} {r.get('经营范围','')} {r.get('东财细分行业','')}"
    if not txt.strip():
        return "缺资料"
    has_ex = any(k in txt for k in EXCL_KW)
    has_kp = any(k in txt for k in KEEP_KW)
    has_bo = any(k in txt for k in BOTH_KW)
    if has_ex and not has_kp:
        return "建议剔除(非周期业务)"
    if has_kp and not has_ex:
        return "建议保留(周期业务)"
    if has_ex and has_kp:
        return "混合-人工判"
    if has_bo:
        return "地产建筑链-人工判"
    return "未匹配-人工判"


targets["建议判定"] = targets.apply(suggest, axis=1)

# ============ 输出 ============
cols_pool = [c for c in targets.columns if c not in
             ("公司全称", "经营范围", "em名称", "换手率")]
targets[cols_pool].sort_values(["层级", "max暴露"], ascending=[True, False]).to_csv(
    f"{BASE}/标的池_分层清单_增强版.csv", index=False, encoding="utf-8-sig")

rev = targets[targets["层级"].isin(["L1核心", "L2扩展"]) &
              (targets["周期分类"] != "Y")].copy()
rev_cols = ["ts_code", "名称", "层级", "最强组", "max暴露", "达标窗口数",
            "东财细分行业", "总市值亿", "主营业务", "建议判定"]
rev[rev_cols].sort_values("max暴露", ascending=False).to_csv(
    f"{BASE}/终审1_入池但非官方周期_增强版.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 70)
print("终审1 自动预判分布 (原158只, 含漂移层被移出的不计)")
print("=" * 70)
print(rev["建议判定"].value_counts().to_string())
need_human = rev[~rev["建议判定"].str.startswith(("建议保留", "建议剔除"))]
print(f"\n→ 需要人工看的只剩 {len(need_human)} 只")
print("\n[建议保留] 示例:")
for _, r in rev[rev["建议判定"].str.startswith("建议保留")].head(6).iterrows():
    print(f"  {r['ts_code']} {r['名称']:<6} {str(r['东财细分行业'])[:10]:<10} "
          f"{str(r['主营业务'])[:44]}")
print("\n[建议剔除] 示例:")
for _, r in rev[rev["建议判定"].str.startswith("建议剔除")].head(8).iterrows():
    print(f"  {r['ts_code']} {r['名称']:<6} {str(r['东财细分行业'])[:10]:<10} "
          f"{str(r['主营业务'])[:44]}")
print("\n[人工判] 示例:")
for _, r in need_human.head(8).iterrows():
    print(f"  {r['ts_code']} {r['名称']:<6} {str(r['东财细分行业'])[:10]:<10} "
          f"{str(r['主营业务'])[:44]}")
print("\n[已保存] 标的池_分层清单_增强版.csv / 终审1_..._增强版.csv")
