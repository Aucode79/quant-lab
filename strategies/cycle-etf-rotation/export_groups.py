"""
分组清单导出 — 供人工核对
=====================================================
只读，输出每个关键词分支的完整成分 + 已知问题 + 漏召参考清单。

产物:
  分组清单_待人工核对.csv   —— 可加一列"人工判定"直接改
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import pandas as pd

BASIC = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/reference/"
         "stock_basic.parquet")

KW = {
    "煤炭": ["煤"],
    "钢铁": ["钢"],
    "有色": ["铜", "铝", "锌", "镍", "稀土", "矿业", "黄金"],
    "建材": ["水泥", "玻璃"],
    "化工": ["化工", "化学", "化纤"],
    "石化": ["石化", "石油", "油气"],
}

con = duckdb.connect(":memory:")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}' ORDER BY ts_code").fetchdf()
con.close()
basic["name"] = basic["name"].str.replace(r"\s+", "", regex=True)

rows = []
for g, ks in KW.items():
    pat = "|".join(ks)
    hit = basic[basic["name"].str.contains(pat, na=False)]
    for _, r in hit.iterrows():
        rows.append({"分组": g, "ts_code": r["ts_code"], "名称": r["name"],
                     "命中关键词": ",".join([k for k in ks if k in r["name"]]),
                     "人工判定": ""})
df = pd.DataFrame(rows)
df.to_csv("分组清单_待人工核对.csv", index=False, encoding="utf-8-sig")

# ---- 打印 ----
print("=" * 76)
print("关键词分组完整清单 (共 %d 只)" % len(df))
print("=" * 76)
for g in KW:
    sub = df[df["分组"] == g].sort_values("ts_code")
    print(f"\n【{g}】{len(sub)} 只")
    for i in range(0, len(sub), 4):
        chunk = sub.iloc[i:i + 4]
        line = "  " + " | ".join(f"{r.ts_code} {r.名称:<6}" for r in chunk.itertuples())
        print(line)

# ---- 跨组重复 ----
dup = df[df.duplicated("ts_code", keep=False)].sort_values("ts_code")
print("\n" + "=" * 76)
print("跨组重复 (同一只被多个关键词命中, 需人工决定归哪组)")
print("=" * 76)
if len(dup):
    for c, grp in dup.groupby("ts_code"):
        print(f"  {c} {grp.iloc[0].名称}: {', '.join(grp['分组'])}")
else:
    print("  无")

# ---- 已知问题 (我人工识别的) ----
print("\n" + "=" * 76)
print("已知问题 — 需要你重点核对")
print("=" * 76)
FLAG = [
    ("600985.SH", "淮北矿业", "误分类", "煤炭公司, 因名字含'矿业'被归入【有色】"),
    ("600660.SH", "福耀玻璃", "误分类", "汽车玻璃, 属汽车零部件, 被归入【建材】"),
    ("600019.SH", "宝钢股份", "待确认", "'钢'命中但为钢铁龙头, 正确"),
]
for code, nm, kind, note in FLAG:
    m = basic[basic["ts_code"] == code]
    if len(m) == 0:
        continue
    got = df[df["ts_code"] == code]
    g = got.iloc[0]["分组"] if len(got) else "(不在任何组)"
    print(f"  [{kind}] {code} {nm}  当前={g}")
    print(f"          {note}")

# ---- 漏召参考清单 (域知识构造, 必须人工确认) ----
MISS = {
    "煤炭": ["601088.SH", "600188.SH", "601699.SH", "002128.SZ", "601898.SH",
             "601001.SH", "600123.SH", "600508.SH", "601918.SH", "000937.SZ",
             "600997.SH", "601101.SH", "600157.SH", "600971.SH"],
    "有色": ["603993.SH", "000060.SZ", "600362.SH", "000878.SZ", "600497.SH",
             "601168.SH", "600549.SH", "000630.SZ", "002460.SZ", "600111.SH"],
    "钢铁": ["600282.SH", "000708.SZ", "002110.SZ", "600507.SH", "000959.SZ",
             "600581.SH", "601005.SH", "000898.SZ"],
    "建材": ["600176.SH", "000786.SZ", "002271.SZ", "600801.SH", "002080.SZ",
             "600585.SH", "000012.SZ", "002233.SZ"],
    "化工": ["600426.SH", "601233.SH", "600309.SH", "002064.SZ", "600143.SH",
             "002648.SZ", "600096.SH", "000830.SZ", "600409.SH"],
    "石化": ["601808.SH", "600028.SH", "601857.SH", "600346.SH", "002493.SZ",
             "000301.SZ", "600688.SH", "601216.SH"],
}
print("\n" + "=" * 76)
print("漏召参考清单 (我按域知识列出的知名周期股, 名称不含关键词 → 当前被漏)")
print("=" * 76)
miss_rows = []
for g, codes in MISS.items():
    have = set(df[df["分组"] == g]["ts_code"])
    print(f"\n【{g}】建议核对 {len(codes)} 只")
    for c in codes:
        m = basic[basic["ts_code"] == c]
        if len(m) == 0:
            continue
        nm = m.iloc[0]["name"]
        already = c in set(df["ts_code"])
        tag = "已在其他组" if already else "当前未收录"
        print(f"  {c} {nm:<6} [{tag}]")
        miss_rows.append({"建议分组": g, "ts_code": c, "名称": nm,
                          "当前状态": tag, "人工判定": ""})
pd.DataFrame(miss_rows).to_csv("漏召参考清单.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 76)
print("已导出两个 CSV (utf-8-sig, Excel可直接打开):")
print("  分组清单_待人工核对.csv  —— 现有成员, 填'人工判定'列")
print("  漏召参考清单.csv         —— 建议补入, 填'人工判定'列")
print("=" * 76)
