"""
周期因子有效性测试 — 动量 vs 反转, 以及"周期时钟"是否存在
=====================================================
只读诊断，不构成投资建议。

三个问题
--------
Q1. CYC 板块内部是否有共同驱动("周期时钟")？
    → 相关矩阵 / PCA 第一主成分占比 / 两两平均相关
Q2. "周期位置反转"成立吗？(低位置→未来涨)
    → 长周期 z 位置 vs 未来收益的 IC
Q3. 动量 vs 反转, 哪个才是 CYC 的有效方向？
    → 对不同 horizon 的信号计算 pooled IC, 分组对比
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

PARQUET = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
           "kline_1min_etf.parquet")

CYC = ["159930.SZ", "512400.SH", "515210.SH", "159870.SZ", "159745.SZ", "560280.SH"]
DEF = ["512170.SH", "512010.SH", "159928.SZ", "159611.SZ"]
BENCH = ["159915.SZ"]
ALL = CYC + DEF + BENCH
codes_str = ", ".join(f"'{c}'" for c in ALL)

con = duckdb.connect(":memory:")
daily = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PARQUET}/**/*.parquet'
    WHERE ts_code IN ({codes_str})
    GROUP BY ts_code, trade_date ORDER BY ts_code, trade_date
""").fetchdf()
con.close()
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
close = daily.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
print("=" * 78)
print("周期因子有效性测试")
print("=" * 78)
print(f"面板: {close.shape[0]} 交易日 x {close.shape[1]} ETF "
      f"({close.index[0].date()} ~ {close.index[-1].date()})")

# ---------------- Q1: 共同驱动 / 周期时钟 ----------------
print("\n" + "=" * 78)
print("Q1. CYC 内部是否有共同驱动(周期时钟)?")
print("=" * 78)
ret = np.log(close).diff()


def block_stats(codes, label, min_overlap=200):
    sub = ret[[c for c in codes if c in ret.columns]]
    cols = list(sub.columns)
    # 两两平均相关 (至少 min_overlap 共同样本)
    cs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = sub[[cols[i], cols[j]]].dropna()
            if len(pair) >= min_overlap:
                cs.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    # 用 pairwise-complete 相关矩阵做 PCA(避免被短历史ETF砍掉样本)
    cm = sub.corr(min_periods=min_overlap)
    cm = np.array(sub.corr(min_periods=min_overlap).fillna(0.0), dtype=float)
    np.fill_diagonal(cm, 1.0)
    if len(cols) >= 2:
        eig = np.linalg.eigvalsh(cm)[::-1]
        pc1 = eig[0] / eig.sum()
        cond = eig[0] / eig[-1]
    else:
        pc1, cond = np.nan, np.nan
    avg_c = float(np.mean(cs)) if cs else np.nan
    print(f"  {label:8s} 两两平均相关={avg_c:+.3f} (n对={len(cs):2d})  "
          f"PC1占比={pc1:.1%}  特征值比={cond:5.2f}")
    return avg_c, float(pc1)


c_cyc = block_stats(CYC, "CYC")
c_def = block_stats(DEF, "DEF")
b, _ = block_stats(BENCH, "BENCH")

# 子期间稳定性: CYC 相关性是否被某一段主导?
print("\n  --- CYC 两两平均相关的子期间稳定性 ---")
for lo, hi in [("2018-01-01", "2020-12-31"), ("2021-01-01", "2022-12-31"),
               ("2023-01-01", "2026-06-30")]:
    sl = ret.loc[(ret.index >= lo) & (ret.index <= hi), CYC]
    cs = []
    cols = list(sl.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pair = sl[[cols[i], cols[j]]].dropna()
            if len(pair) >= 100:
                cs.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    print(f"    {lo[:7]}~{hi[:7]}: 平均相关={np.mean(cs):+.3f} (n对={len(cs)})")

# 篮子层面: CYC vs DEF vs BENCH
cyc_b = ret[[c for c in CYC if c in ret.columns]].mean(axis=1)
def_b = ret[[c for c in DEF if c in ret.columns]].mean(axis=1)
ben_b = ret[BENCH].mean(axis=1)
print("\n  --- 篮子层面相关 (等权) ---")
for (n1, s1), (n2, s2) in [(("CYC", cyc_b), ("DEF", def_b)),
                           (("CYC", cyc_b), ("BENCH", ben_b)),
                           (("DEF", def_b), ("BENCH", ben_b))]:
    p = pd.concat([s1, s2], axis=1).dropna()
    print(f"    {n1} vs {n2}: 相关={p.iloc[:, 0].corr(p.iloc[:, 1]):+.3f}  (n={len(p)})")

print("\n  --- 领先滞后 (CYC 领先 DEF ?) corr(CYC_t, DEF_{t+k}) ---")
for k in [-5, -2, -1, 0, 1, 2, 5]:
    p = pd.concat([cyc_b, def_b.shift(-k)], axis=1).dropna()
    print(f"    k={k:+3d}日: {p.iloc[:, 0].corr(p.iloc[:, 1]):+.3f}")

# ---------------- Q2 & Q3: IC 测试 ----------------
print("\n" + "=" * 78)
print("Q2/Q3. 动量 vs 反转 — 各 horizon 的 pooled IC (Spearman)")
print("=" * 78)


def make_signals(px):
    lr = np.log(px)
    sig = {}
    # 动量类
    sig["mom20"] = lr - lr.shift(20)
    sig["mom60"] = lr - lr.shift(60)
    sig["mom120"] = lr - lr.shift(120)
    sig["mom250"] = lr - lr.shift(250)
    # 位置类 (相对长期中枢的位置)
    m250 = lr.rolling(250, min_periods=120).mean()
    s250 = lr.rolling(250, min_periods=120).std()
    sig["pos250"] = (lr - m250) / s250
    m500 = lr.rolling(500, min_periods=250).mean()
    s500 = lr.rolling(500, min_periods=250).std()
    sig["pos500"] = (lr - m500) / s500
    # 波动状态
    r = px.pct_change()
    v20 = r.rolling(20).std()
    v252 = r.rolling(252, min_periods=120).std()
    sig["volratio"] = v20 / v252.replace(0, np.nan)
    # 量价背离代理: 价格位置 vs 成交额位置 (用价格动量代替成交额不可得处)
    return sig


def spearman(a, b):
    """rank 后算 pearson = Spearman (免 scipy 依赖)"""
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    return float(np.corrcoef(ra.values, rb.values)[0, 1])


def ic_table(codes, label):
    sub = close[[c for c in codes if c in close.columns]]
    sig = make_signals(sub)
    out = {}
    for hor in [20, 60, 120, 250]:
        fwd = np.log(sub).shift(-hor) - np.log(sub)
        col = {}
        for name, s in sig.items():
            sv = s.stack()
            fv = fwd.stack()
            common = pd.concat([sv, fv], axis=1).dropna()
            col[name] = spearman(common.iloc[:, 0], common.iloc[:, 1]) \
                if len(common) >= 50 else np.nan
        out[f"fwd{hor}"] = col
    df = pd.DataFrame(out)
    print(f"\n  --- {label} ---")
    print(df.round(4).to_string())
    return df


ic_cyc = ic_table(CYC, "CYC 周期组")
ic_def = ic_table(DEF, "DEF 防御组")

print("\n" + "=" * 78)
print("解读")
print("=" * 78)
print("  IC>0 = 该信号越大未来越涨 (动量有效)")
print("  IC<0 = 该信号越大未来越跌 (反转有效)")
print("  pos250/pos500 若 IC 显著为负 → 周期位置反转成立")
print("  mom60/mom120 若 IC 显著为正 → 中期动量成立")
