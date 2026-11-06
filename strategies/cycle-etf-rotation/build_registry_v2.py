"""
标的注册表 v2 — 收集优先架构 (只怕少不怕多)
=====================================================
v1教训: 暴露阈值(0.398)+三窗稳定性在收集阶段当门槛, 挡掉502/689只
证监会周期类股票 — 包括万华化学/恒力石化/荣盛石化/三一重工/巨化股份/
多氟多。行为分被当成了门槛, 而它只该是注释。

v2架构 (三层分离):
  收集层: 证监会类目全量入册 + 名称关键词兜底 — 无任何行为门槛
  证据层: 暴露分数/窗口稳定性 只作注释列, 供后续人工与AI参考
  决策层: 保留/剔除 全部留给人工终审与AI研判 (后续环节)

类别: 煤炭/钢铁/有色/建材/化工/石化 + 新增 橡胶塑料/生猪养殖/航运港口/造纸
输出: 标的池_注册表_v2_20260913.csv
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

BASE = Path("C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股")
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"

ind = pd.read_parquet(f"{REF}/csrc_industry.parquet")
ind["prefix"] = ind["industry"].fillna("").str[:3]
basic = pd.read_parquet(f"{REF}/stock_basic_bs.parquet")
old = pd.read_csv(BASE / "标的池_分层清单.csv")

# ============ 收集层: 类目定义 (全量, 无门槛) ============
CLASS_MAP = {
    "煤炭": ["B06"],
    "钢铁": ["B08", "C31"],
    "有色": ["B09", "C32"],
    "建材": ["C30"],
    "化工": ["C26", "C28"],
    "石化": ["B07", "C25"],
    "橡胶塑料": ["C29"],          # 含塑料, 子类标注区分
    "生猪养殖": ["A03"],           # 畜牧业
    "航运港口": ["G55"],
    "造纸": ["C22"],
}
NAME_ADD = {   # 名称关键词兜底 (跨类目)
    "橡胶塑料": ["橡胶", "轮胎"],
    "生猪养殖": ["饲料", "猪", "牧原", "温氏", "农牧", "养殖"],
}

rows = []
seen = set()
for cat, prefixes in CLASS_MAP.items():
    sub = ind[ind["prefix"].isin(prefixes)]
    for _, r in sub.iterrows():
        rows.append({"ts_code": r["ts_code"], "名称": r["name"], "行业类别": cat,
                     "来源": "CSRC类目", "证监会行业": r["industry"]})
        seen.add(r["ts_code"])
for cat, kws in NAME_ADD.items():
    pat = "|".join(kws)
    sub = ind[ind["name"].str.contains(pat, na=False)]
    for _, r in sub.iterrows():
        if r["ts_code"] in seen:
            # 已有类目, 只补来源标记
            for row in rows:
                if row["ts_code"] == r["ts_code"]:
                    row["来源"] += "+名称关键词"
                    break
            continue
        rows.append({"ts_code": r["ts_code"], "名称": r["name"], "行业类别": cat,
                     "来源": "名称关键词", "证监会行业": r["industry"]})
        seen.add(r["ts_code"])

reg = pd.DataFrame(rows)
print(f"[收集层] CSRC+关键词: {len(reg)} 只")

# v1数据行为入池但CSRC类目未覆盖的股票 — 全部保留 (类别=最强暴露组)
old_pool = old[old["层级"].isin(["L1核心", "L2扩展"])]
added = 0
for _, r in old_pool.iterrows():
    if r["ts_code"] in seen:
        continue
    csrc_row = ind[ind["ts_code"] == r["ts_code"]]
    rows.append({"ts_code": r["ts_code"], "名称": r["名称"],
                 "行业类别": r["最强组"], "来源": "数据行为(v1入池)",
                 "证监会行业": csrc_row.iloc[0]["industry"] if len(csrc_row) else "(无分类)"})
    seen.add(r["ts_code"])
    added += 1
print(f"[补录] v1行为入池未被类目覆盖: {added} 只")
reg = pd.DataFrame(rows)
print(f"[收集层] 注册表 v2: {len(reg)} 只 (v1池 345 只)")
print(reg["行业类别"].value_counts().to_string())

# ============ 证据层: 合并暴露分数 (只注释, 不筛) ============
exp_cols = ["暴露_煤炭", "暴露_钢铁", "暴露_有色", "暴露_建材", "暴露_化工", "暴露_石化"]
keep = ["ts_code", "层级", "最强组", "max暴露", "达标窗口数",
        "全期达标", "早期达标", "晚期达标", "中位日成交额(万)"] + exp_cols
old_s = old[keep].copy()
reg = reg.merge(old_s, on="ts_code", how="left")

# v2层级语义: 收集层全量在册; v1的L1/L2保留原义, 其余统称L3候选
reg["层级"] = reg["层级"].fillna("L3候选")
reg.loc[reg["层级"] == "-", "层级"] = "L3候选"
reg.loc[reg["层级"] == "漂移", "层级"] = "L3候选(漂移)"
reg.loc[reg["层级"] == "L3边界(流动性不足)", "层级"] = "L3候选(流动性)"

# 新类别尚未算暴露
reg["暴露状态"] = "已评分"
reg.loc[reg["行业类别"].isin(["橡胶塑料", "生猪养殖", "航运港口", "造纸"]), "暴露状态"] = \
    reg.loc[reg["行业类别"].isin(["橡胶塑料", "生猪养殖", "航运港口", "造纸"]), "暴露状态"] \
    .where(reg.loc[reg["行业类别"].isin(["橡胶塑料", "生猪养殖", "航运港口", "造纸"]),
                  "max暴露"].notna(), "待评分(需定种子)")

# ============ 注释: 子类标签 ============
def subtype(r):
    n, cat = str(r["名称"]), r["行业类别"]
    tags = []
    if cat == "有色":
        if "黄金" in n or n.endswith("金") or "金业" in n: tags.append("贵金属")
        if any(k in n for k in ["锂", "钴", "镍"]): tags.append("能源金属(锂钴镍)")
        if "稀土" in n: tags.append("稀土")
    if cat == "橡胶塑料":
        if "轮胎" in n: tags.append("轮胎")
        elif "橡胶" in n: tags.append("橡胶")
        else: tags.append("塑料/制品")
    if cat == "生猪养殖":
        if "饲料" in n: tags.append("饲料")
        elif any(k in n for k in ["种", "育", "农牧"]): tags.append("种业/养殖链")
        else: tags.append("养殖")
    if "氟" in n: tags.append("氟化工")
    if "巨化" in n or "制冷" in n: tags.append("制冷剂")
    if any(k in n for k in ["锂", "钴", "镍"]) and cat == "化工": tags.append("新能源链化工")
    return ";".join(tags) if tags else ""


reg["子类注释"] = reg.apply(subtype, axis=1)
reg = reg.merge(basic[["ts_code", "ipoDate", "outDate"]], on="ts_code", how="left")

# ============ 输出 ============
out_fn = BASE / "标的池_注册表_v2_20260913.csv"
reg.sort_values(["行业类别", "层级", "max暴露"],
                ascending=[True, True, False]).to_csv(out_fn, index=False,
                                                      encoding="utf-8-sig")
print(f"\n[层级分布]")
print(reg["层级"].value_counts().to_string())
print(f"\n[重点个案确认]")
for code in ["600160.SH", "002407.SZ", "600309.SH", "002714.SZ", "601118.SH",
             "601058.SH", "600018.SH"]:
    r = reg[reg["ts_code"] == code]
    if len(r):
        r = r.iloc[0]
        print(f"  {code} {r['名称']:<6} 类别={r['行业类别']:<6} 层级={r['层级']:<12} "
              f"暴露={r['max暴露'] if pd.notna(r['max暴露']) else '待评分'} "
              f"子类={r['子类注释']}")
print(f"\n[已保存] {out_fn}")
