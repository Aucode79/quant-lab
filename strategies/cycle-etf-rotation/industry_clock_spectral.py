"""
行业时钟检验 + 周期长度谱估计 + 相关regime
=====================================================
只读诊断，不构成投资建议。

三个此前未检验的结构:
  T1. 子行业领先滞后(行业时钟): 煤炭领先钢铁? 有色领先化工?
      周频残差收益, lag ±8周, 非对称度, 分半稳定性, 循环移位置换零假设
  T2. 周期长度谱估计: 等权周期篮子去趋势后FFT周期图,
      1.5-8年频段峰值, 逐年jackknife, 诚实报告分辨率极限
  T3. 相关regime: 滚动60日子行业间平均相关 vs 未来收益
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
rng = np.random.default_rng(42)

KW = {
    "煤炭": ["煤"], "钢铁": ["钢"],
    "有色": ["铜", "铝", "锌", "镍", "稀土", "矿业", "黄金"],
    "建材": ["水泥", "玻璃"], "化工": ["化工", "化学", "化纤"],
    "石化": ["石化", "石油", "油气"],
}
EXCLUDE = {"002678.SZ", "300329.SZ", "300895.SZ", "600237.SH", "688548.SH",
           "300797.SZ", "300226.SZ", "601968.SH", "600660.SH", "600916.SH",
           "601117.SH", "000852.SZ", "603353.SH", "002541.SZ", "002743.SZ",
           "600477.SH", "600496.SH", "688119.SH", "000928.SZ", "601068.SH",
           "920019.BJ"}
REASSIGN = {"600985.SH": "煤炭", "000923.SZ": "有色"}
ADD = {"煤炭": ["601088.SH", "600188.SH", "601699.SH", "002128.SZ", "600123.SH",
                "600508.SH", "601918.SH", "000937.SZ"],
       "有色": ["603993.SH", "000060.SZ", "600549.SH", "002460.SZ"],
       "建材": ["600176.SH", "000786.SZ", "002271.SZ", "600801.SH", "002080.SH",
                "000012.SZ", "002233.SZ"],
       "化工": ["600426.SH", "601233.SH", "600143.SH", "600096.SH"],
       "石化": ["601808.SH", "000301.SZ"]}

print("=" * 78)
print("行业时钟检验 + 周期长度谱估计")
print("=" * 78)

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}'").fetchdf()
basic["name"] = basic["name"].str.replace(r"\s+", "", regex=True)

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

allc = sorted(set().union(*groups.values()))
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
print(f"[面板] {res.shape[0]}日 x {res.shape[1]}只  ({time.time()-t0:.0f}s)")

GN = [g for g in KW]
# 周频篮子
wk = res.resample("W-FRI").sum(min_count=1)
baskets = pd.DataFrame({g: wk[[c for c in groups[g] if c in wk.columns]]
                         .mean(axis=1) for g in GN}).dropna(how="all")
print(f"[周频] {len(baskets)}周  ({baskets.index[0].date()} ~ {baskets.index[-1].date()})")

# ================= T1 行业时钟 =================
print("\n" + "=" * 78)
print("T1. 子行业领先滞后 (lag=±8周, 非对称度 A(i→j)=corr(i_t,j_{t+2})-corr(j_t,i_{t+2}))")
print("=" * 78)


def pair_lag(bi, bj, k):
    """corr(bi(t), bj(t+k)) — bj 前移k期"""
    d = pd.concat([bi, bj.shift(-k)], axis=1).dropna()
    if len(d) < 40:
        return np.nan
    return float(np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1])


def asymmetry(bdf, i, j, k=2):
    a = pair_lag(bdf[i], bdf[j], k)   # i 领先 j
    b = pair_lag(bdf[j], bdf[i], k)   # j 领先 i
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a - b


def half(bdf, tag):
    mid = bdf.index[len(bdf) // 2]
    return bdf[bdf.index <= mid], bdf[bdf.index > mid]


h1, h2 = half(baskets, "")
print(f"\n  全期({len(baskets)}周) / 前半({len(h1)}) / 后半({len(h2)}) 的非对称度:")
print(f"  {'对':<12}{'全期A':>8}{'前半A':>8}{'后半A':>8}{'同号?':>6}")
pairs = [(GN[i], GN[j]) for i in range(len(GN)) for j in range(i + 1, len(GN))]
asym_all, asym_h1, asym_h2 = {}, {}, {}
for i, j in pairs:
    a = asymmetry(baskets, i, j)
    a1 = asymmetry(h1, i, j)
    a2 = asymmetry(h2, i, j)
    asym_all[(i, j)], asym_h1[(i, j)], asym_h2[(i, j)] = a, a1, a2
    same = (not np.isnan(a1)) and (not np.isnan(a2)) and (np.sign(a1) == np.sign(a2))
    print(f"  {i}→{j:<6}{a:>8.3f}{a1:>8.3f}{a2:>8.3f}{'  Y' if same else '':>6}")

n_same = sum(1 for k_ in asym_all if not np.isnan(asym_h1[k_]) and not np.isnan(asym_h2[k_])
             and np.sign(asym_h1[k_]) == np.sign(asym_h2[k_]))
n_valid = sum(1 for k_ in asym_all if not np.isnan(asym_h1[k_]) and not np.isnan(asym_h2[k_]))
print(f"\n  前后半同号: {n_same}/{n_valid} 对   (随机期望≈50%)")

# 置换零假设: 循环移位(保留各自自相关, 破坏领先滞后)
print("\n  置换检验(循环移位500次): 全期非对称度的零分布")
NS = 500
sig_pairs = []
for (i, j) in pairs:
    if np.isnan(asym_all[(i, j)]):
        continue
    d = pd.concat([baskets[i], baskets[j]], axis=1).dropna()
    if len(d) < 60:
        continue
    x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    null = []
    for _ in range(NS):
        s = rng.integers(1, len(x))
        # x领先y: corr(x_t, y_{t+2}) 用移位后的x
        x2 = np.roll(x, s)
        c1 = np.corrcoef(x2[:-2], y[2:])[0, 1]
        c2 = np.corrcoef(y[:-2], x2[2:])[0, 1]
        null.append(c1 - c2)
    null = np.array(null)
    p = float(np.mean(np.abs(null) >= abs(asym_all[(i, j)])))
    star = " **" if p < 0.05 else ""
    if p < 0.05:
        sig_pairs.append((i, j, asym_all[(i, j)], p))
    print(f"    {i}→{j:<6}A={asym_all[(i,j)]:+.3f}  null_sd={null.std():.3f}  p={p:.3f}{star}")
print(f"\n  15对中置换显著(p<0.05)的: {len(sig_pairs)}对  "
      f"(多重比较下随机期望≈0.75对)")

# ================= T2 谱估计 =================
print("\n" + "=" * 78)
print("T2. 周期长度谱估计 (等权周期篮子, 剥beta, 减250日MA, 月度)")
print("=" * 78)
bd = res[[c for c in res.columns if c in set().union(*groups.values())]]
basket_daily = bd.mean(axis=1)          # 等权周期篮子日度残差收益
idx = basket_daily.cumsum()             # 对数"超额净值"
cyc = idx - idx.rolling(250, min_periods=120).mean()
cyc = cyc.dropna()
mo = cyc.resample("ME").last().dropna()
mo = mo - mo.mean()
print(f"  月度点数 N={len(mo)}  ({mo.index[0].date()} ~ {mo.index[-1].date()})")

X = np.fft.rfft(mo.values)
P = (np.abs(X) ** 2)
freqs = np.fft.rfftfreq(len(mo))
periods = np.where(freqs > 0, 1.0 / freqs, np.inf)  # 月
band = (periods >= 18) & (periods <= 96)
print(f"  频率分辨率: Δf=1/{len(mo)} — 4年周期处相邻可分辨周期约 "
      f"{1/(1/48+1/len(mo)):.0f}~{1/(1/48-1/len(mo)):.0f}月")
if band.any():
    pb = periods[band]
    Pb = P[band]
    top = np.argsort(Pb)[::-1][:5]
    print("  1.5~8年频段功率前5峰:")
    for t in top:
        print(f"    周期≈{pb[t]:5.0f}月({pb[t]/12:.1f}年)  功率占比={Pb[t]/P.sum():.1%}")

# jackknife: 逐年去掉12个月
print("\n  逐年jackknife(每次去掉一年重算1.5-8年频段主峰):")
peaks = []
for y in range(2019, 2027):
    m2 = mo[~((mo.index.year == y))]
    if len(m2) < 40:
        continue
    X2 = np.fft.rfft(m2.values - m2.values.mean())
    P2 = np.abs(X2) ** 2
    f2 = np.fft.rfftfreq(len(m2))
    pr2 = np.where(f2 > 0, 1.0 / f2, np.inf)
    b2 = (pr2 >= 18) & (pr2 <= 96)
    if b2.any():
        pk = pr2[b2][np.argmax(P2[b2])]
        peaks.append(pk)
        print(f"    去{y}: 主峰≈{pk:4.0f}月({pk/12:.1f}年)")
if peaks:
    print(f"  峰位范围: {min(peaks):.0f}~{max(peaks):.0f}月  "
          f"→ {'稳定' if (max(peaks)-min(peaks)) < 12 else '不稳定, 谱峰不可信'}")

# ================= T3 相关regime =================
print("\n" + "=" * 78)
print("T3. 相关regime: 滚动60日子行业间平均相关 vs 未来60日篮子收益")
print("=" * 78)
bdaily = pd.DataFrame({g: res[[c for c in groups[g] if c in res.columns]]
                       .mean(axis=1) for g in GN})
cors = []
for i in range(len(GN)):
    for j in range(i + 1, len(GN)):
        cors.append(bdaily[GN[i]].rolling(60).corr(bdaily[GN[j]]))
avgc = pd.concat(cors, axis=1).mean(axis=1).dropna()
bidx = bdaily.mean(axis=1).cumsum()
fwd = bidx.shift(-60) - bidx
d = pd.concat([avgc, fwd], axis=1).dropna()
d.columns = ["corr", "fwd"]
d2 = d.iloc[::60]  # 非重叠
if len(d2) > 20:
    ic = float(np.corrcoef(d2.iloc[:, 0], d2.iloc[:, 1])[0, 1])
    q = pd.qcut(d2["corr"], 3, labels=["低相关", "中", "高相关"], duplicates="drop")
    gm = d2.groupby(q, observed=True)["fwd"].mean()
    print(f"  非重叠样本 n={len(d2)}  IC={ic:+.3f}")
    for k_, v in gm.items():
        print(f"    {k_}: 未来60日超额收益均值 = {v:+.4f}")
else:
    print("  样本不足")

print("\n" + "=" * 78)
print("判读: T1看领先滞后是否稳定+置换显著; T2看谱峰是否逐年稳定;")
print("      T3看'相关升高'是否对应未来收益变化( contagion/风险信号 )。")
print("      8年样本的所有结论都要打折 — 这是结构性限制。")
print("=" * 78)
