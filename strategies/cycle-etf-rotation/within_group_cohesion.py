"""
组内"特立独行"个股识别 — 概念炒作/误分类探测器
=====================================================
只读诊断，不构成投资建议。

动机
----
个股可能被概念炒作驱动(金牛化工的"牛"、多氟多的液冷…),
使其价格与所属行业脱钩。频率检验显示这类噪声在**组均值层面**会被抵消,
但在**个股层面**依然存在, 且会污染:
  - 数据驱动聚类 (概念股会被错聚到一起)
  - 组内选股排序

方法
----
对每个周期组, 算每只股票与其他成员的**平均相关**(组内凝聚力),
并结合其**特异波动**。两者结合可以标出:
  - 低凝聚 + 高特异波动 → 疑似概念炒作 或 误分类
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

PQ_STOCK = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
            "kline_1min.parquet")
PQ_ETF = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
          "kline_1min_etf.parquet")
BASIC = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/reference/"
         "stock_basic.parquet")
MKT = "510300.SH"

KW = {
    "煤炭": ["煤"],
    "钢铁": ["钢"],
    "有色": ["铜", "铝", "锌", "镍", "稀土", "矿业", "黄金"],
    "建材": ["水泥", "玻璃"],
    "化工": ["化工", "化学", "化纤"],
    "石化": ["石化", "石油", "油气"],
}
EXCLUDE = {"002678.SZ", "300329.SZ", "300895.SZ", "600237.SH", "688548.SH",
           "300797.SZ", "300226.SZ", "601968.SH", "600660.SH", "600916.SH",
           "601117.SH", "000852.SZ", "603353.SH", "002541.SZ", "002743.SZ",
           "600477.SH", "600496.SH", "688119.SH", "000928.SH", "601068.SH",
           "920019.BJ"}
REASSIGN = {"600985.SH": "煤炭", "000923.SZ": "有色"}
ADD = {"煤炭": ["601088.SH", "600188.SH", "601699.SH", "002128.SZ",
                "600123.SH", "600508.SH", "601918.SH", "000937.SZ"],
       "有色": ["603993.SH", "000060.SZ", "600549.SH", "002460.SZ"],
       "建材": ["600176.SH", "000786.SZ", "002271.SZ", "600801.SH",
                "002080.SZ", "000012.SZ", "002233.SZ"],
       "化工": ["600426.SH", "601233.SH", "600143.SH", "600096.SH"],
       "石化": ["601808.SH", "000301.SZ"]}
# 用户点名的疑似概念股 (+多氟多, 名称不含关键词但常被概念驱动)
SPECIAL = {"600722.SH": "金牛化工(化工组)", "002407.SZ": "多氟多(不在组内)"}

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}'").fetchdf()
basic["name"] = basic["name"].str.replace(r"\s+", "", regex=True)
NMAP = dict(zip(basic.ts_code, basic["name"]))

groups = {}
for g, ks in KW.items():
    groups[g] = [c for c in basic[basic["name"].str.contains("|".join(ks), na=False)]["ts_code"]]
for g in groups:
    groups[g] = [c for c in groups[g] if c not in EXCLUDE]
for c, g in REASSIGN.items():
    for gg in groups:
        if c in groups[gg] and gg != g:
            groups[gg].remove(c)
    if c not in groups[g]:
        groups[g].append(c)
for g, codes in ADD.items():
    for c in codes:
        if c not in groups[g]:
            groups[g].append(c)

allc = sorted(set().union(*groups.values()) | set(SPECIAL))
cs = ", ".join(f"'{c}'" for c in allc)
yr = ",".join(str(y) for y in range(2018, 2027))

t0 = time.time()
sd = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_STOCK}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code IN ({cs})
    GROUP BY ts_code, trade_date
""").fetchdf()
ed = con.execute(f"""
    SELECT trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_ETF}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code = '{MKT}'
    GROUP BY trade_date
""").fetchdf()
con.close()
for d in (sd, ed):
    d["trade_date"] = pd.to_datetime(d["trade_date"])
PX = sd.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
mkt = np.log(ed.set_index("trade_date")["c"]).diff().reindex(PX.index)
r = np.log(PX).diff().loc[mkt.notna()]
mkt = mkt.dropna()
vm = mkt.var()
res = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
for c in r.columns:
    s = r[c]
    ok = s.notna()
    if ok.sum() < 300:
        continue
    b = np.cov(s[ok], mkt[ok])[0, 1] / vm
    res.loc[ok, c] = s[ok] - b * mkt[ok]
res = res.dropna(how="all", axis=1)
print("=" * 78)
print(f"组内'特立独行'个股识别  面板 {res.shape[0]}日 x {res.shape[1]}只  "
      f"({time.time()-t0:.0f}s)")
print("=" * 78)

ann = np.sqrt(242)
rows = []
for g, v in groups.items():
    cols = [c for c in v if c in res.columns]
    if len(cols) < 3:
        continue
    cm = res[cols].corr(min_periods=200).to_numpy(dtype=float)
    for i, c in enumerate(cols):
        others = np.delete(cm[i], i)
        others = others[~np.isnan(others)]
        if not len(others):
            continue
        vol = res[c].std() * ann
        rows.append({"分组": g, "代码": c, "名称": NMAP.get(c, "?"),
                     "组内平均相关": float(np.mean(others)),
                     "特异波动(年化)": float(vol)})
D = pd.DataFrame(rows)

print("\n" + "=" * 78)
print("各组'凝聚力最低'的 5 只 (疑似概念炒作 / 误分类)")
print("=" * 78)
for g in KW:
    sub = D[D["分组"] == g].sort_values("组内平均相关")
    if not len(sub):
        continue
    print(f"\n【{g}】组均值={sub['组内平均相关'].mean():+.3f}  "
          f"共{len(sub)}只")
    gmean = sub["组内平均相关"].mean()
    for _, r_ in sub.head(5).iterrows():
        flag = "  ← 显著离群" if r_["组内平均相关"] < gmean - 0.15 else ""
        print(f"  {r_['代码']} {r_['名称']:<7} 组内相关={r_['组内平均相关']:+.3f}  "
              f"特异波动={r_['特异波动(年化)']:6.1%}{flag}")

print("\n" + "=" * 78)
print("各组'凝聚力最高'的 3 只 (最纯正的行业代表)")
print("=" * 78)
for g in KW:
    sub = D[D["分组"] == g].sort_values("组内平均相关", ascending=False)
    if not len(sub):
        continue
    top = " | ".join(f"{r_['名称']}({r_['组内平均相关']:+.2f})"
                     for _, r_ in sub.head(3).iterrows())
    print(f"  【{g}】{top}")

print("\n" + "=" * 78)
print("用户点名个股")
print("=" * 78)
for c, lab in SPECIAL.items():
    if c not in res.columns:
        print(f"  {lab}: 数据不足")
        continue
    best, bv = None, -9
    for g, v in groups.items():
        cc = [x for x in v if x in res.columns and x != c]
        if len(cc) < 3:
            continue
        cm = res[cc + [c]].corr(min_periods=200)
        val = cm.loc[c, cc].mean()
        if val > bv:
            best, bv = g, val
    print(f"  {c} {lab}")
    print(f"    特异波动 = {res[c].std()*ann:.1%}/年   "
          f"最相关组 = {best}({bv:+.3f})")
    if c in res.columns and len(D[D['代码'] == c]):
        row = D[D["代码"] == c].iloc[0]
        print(f"    在【{row.分组}】组内平均相关 = {row.组内平均相关:+.3f}")

D.sort_values(["分组", "组内平均相关"]).to_csv(
    "组内凝聚力_个股.csv", index=False, encoding="utf-8-sig")
print("\n[已保存] 组内凝聚力_个股.csv")
print("判读: 组内相关低 + 特异波动高 = 该股走得跟同行不一样,")
print("      可能是概念炒作标的, 也可能是我们分组本来就错了。")
