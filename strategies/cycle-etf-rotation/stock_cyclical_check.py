"""
用股票数据检验"周期共同因子" — 剥离市场 beta + 流动性匹配对照
=====================================================
只读诊断，不构成投资建议。

为什么要剥离市场因子
--------------------
单只股票的相关性被大盘 beta 主导(单年测试: 随机对照组内相关也高达 0.42)。
真正该问的是: 剥离市场后, 周期股是否还有"共同的非市场因子"?

方法
----
1. 市场因子 m = 沪深300ETF(510300.SH) 日收益
2. 每只股票对 m 回归: r_i = a + b*m + e_i, 用残差 e_i 算组内相关
3. 对照: (a) 普通随机组   (b) 流动性匹配组(按中位成交额取TopN)
   —— 排除"周期股都是大盘股所以更相关"的混淆
4. 子期间稳定性: 2018-2021 vs 2022-2026

用法: py -3 stock_cyclical_check.py [起始年] [结束年]
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

YEARS = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (2018, 2026)
MIN_DAYS = 400
MIN_OVERLAP = 200
N_RANDOM_POOL = 800        # 随机池(大), 供构造两个对照组

KW = {
    "煤炭": ["煤"],
    "钢铁": ["钢"],
    "有色": ["铜", "铝", "锌", "镍", "稀土", "矿业", "黄金"],
    "建材": ["水泥", "玻璃"],
    "化工": ["化工", "化学", "化纤"],
    "石化": ["石化", "石油", "油气"],
}
rng = np.random.default_rng(42)

print("=" * 78)
print(f"股票数据检验周期共同因子 (剥离市场beta + 流动性匹配)  {YEARS[0]}-{YEARS[1]}")
print("=" * 78)

con = duckdb.connect(":memory:")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}'").fetchdf()
kw_groups = {g: list(basic[basic["name"].str.contains("|".join(ks), na=False)]["ts_code"])
             for g, ks in KW.items()}
matched = set().union(*kw_groups.values())
pool = [c for c in basic["ts_code"] if c not in matched]
rand_pool = list(rng.choice(pool, size=min(N_RANDOM_POOL, len(pool)), replace=False))
for g, v in kw_groups.items():
    print(f"  {g:6s}: {len(v):4d} 只")
print(f"  随机池 : {len(rand_pool)} 只 (待筛出对照组)")

all_codes = sorted(set().union(*kw_groups.values()) | set(rand_pool))
cs = ", ".join(f"'{c}'" for c in all_codes)
yr = ",".join(str(y) for y in range(YEARS[0], YEARS[1] + 1))

print(f"\n[查询] {len(all_codes)} 只股票日线 + 市场基准 {MKT} ...")
t0 = time.time()
daily = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c,
           SUM(amount) AS amt
    FROM '{PQ_STOCK}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code IN ({cs})
    GROUP BY ts_code, trade_date
""").fetchdf()
mkt = con.execute(f"""
    SELECT trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_ETF}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code = '{MKT}'
    GROUP BY trade_date
""").fetchdf()
con.close()
print(f"[查询] {len(daily):,} 行, 耗时 {time.time()-t0:.1f}s")

daily["trade_date"] = pd.to_datetime(daily["trade_date"])
mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
close = daily.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
amt = daily.pivot(index="trade_date", columns="ts_code", values="amt").sort_index()
ret = np.log(close).diff()
m = np.log(mkt.set_index("trade_date")["c"]).diff().reindex(ret.index)
ret = ret.loc[m.notna()]
m = m.dropna()
print(f"[面板] {ret.shape[0]} 交易日 x {ret.shape[1]} 只股票 "
      f"({ret.index[0].date()} ~ {ret.index[-1].date()})")

# ---- 剥离市场 beta ----
var_m = m.var()
resid = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
for c in ret.columns:
    s = ret[c]
    ok = s.notna()
    if ok.sum() < MIN_DAYS:
        continue
    b = np.cov(s[ok], m[ok])[0, 1] / var_m
    resid.loc[ok, c] = s[ok] - b * m[ok]
resid = resid.dropna(how="all", axis=1)
print(f"[残差] 有效股票 {resid.shape[1]} 只")

# ---- 构造两个对照组 ----
N = sum(len(v) for v in kw_groups.values())
rp = [c for c in rand_pool if c in amt.columns]
med_amt = amt[rp].median().dropna()
ctrl_rand = list(rng.choice(med_amt.index, size=min(N, len(med_amt)), replace=False))
ctrl_liq = list(med_amt.sort_values(ascending=False).head(N).index)
# 周期股的流动性水平(对照用)
cyc_med = med_amt.reindex([])  # placeholder
kw_all = sorted(set().union(*kw_groups.values()))
cyc_liq_med = amt[kw_all].median().median()
print(f"[对照] 普通随机={len(ctrl_rand)}只  流动性匹配Top{len(ctrl_liq)}只")
print(f"[流动性] 周期股中位日成交额={cyc_liq_med/1e8:.2f}亿  全样本随机池中位="
      f"{med_amt.median()/1e8:.2f}亿")


def avg_corr(df, codes, min_ov=MIN_OVERLAP, idx=None):
    sub = df[[c for c in codes if c in df.columns]]
    if idx is not None:
        sub = sub.loc[idx]
    sub = sub.loc[:, sub.notna().sum() >= min(MIN_DAYS, len(sub) // 3)]
    cols = list(sub.columns)
    cs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            p = sub[[cols[i], cols[j]]].dropna()
            if len(p) >= min_ov:
                cs.append(p.iloc[:, 0].corr(p.iloc[:, 1]))
    return (float(np.mean(cs)) if cs else np.nan), len(cs), len(cols)


def pc1(df, codes, min_ov=MIN_OVERLAP):
    sub = df[[c for c in codes if c in df.columns]]
    sub = sub.loc[:, sub.notna().sum() >= MIN_DAYS]
    if sub.shape[1] < 2:
        return np.nan
    cm = np.array(sub.corr(min_periods=min_ov).fillna(0.0), dtype=float)
    np.fill_diagonal(cm, 1.0)
    eig = np.linalg.eigvalsh(cm)[::-1]
    return float(eig[0] / eig.sum())


print("\n" + "=" * 78)
print("H1. 组内残差相关 — 周期各组 vs 两个对照组")
print("=" * 78)
print(f"  {'组':<10}{'残差相关':>10}{'股票数':>8}{'配对数':>8}")
cyc_res = []
for g, v in kw_groups.items():
    a, n, k = avg_corr(resid, v)
    cyc_res.append(a)
    print(f"  {g:<10}{a:>10.3f}{k:>8}{n:>8}")
for lab, cc in [("普通随机", ctrl_rand), ("流动性匹配", ctrl_liq)]:
    a, n, k = avg_corr(resid, cc)
    print(f"  {lab:<10}{a:>10.3f}{k:>8}{n:>8}   ← 对照")

cyc_mean = float(np.nanmean(cyc_res))
a_r, _, _ = avg_corr(resid, ctrl_rand)
a_l, _, _ = avg_corr(resid, ctrl_liq)
print(f"\n  周期6组均值 = {cyc_mean:+.3f}")
print(f"  vs 普通随机   {a_r:+.3f}  → 差 {cyc_mean - a_r:+.3f}")
print(f"  vs 流动性匹配 {a_l:+.3f}  → 差 {cyc_mean - a_l:+.3f}  (关键: 排除了市值混淆)")

print("\n" + "=" * 78)
print("H2. 子期间稳定性 (周期组均值 残差相关)")
print("=" * 78)
for lo, hi, lab in [("2018-01-01", "2021-12-31", "2018-2021"),
                    ("2022-01-01", "2026-06-30", "2022-2026")]:
    idx = resid.index[(resid.index >= lo) & (resid.index <= hi)]
    vals = [avg_corr(resid, v, min_ov=120, idx=idx)[0] for v in kw_groups.values()]
    cr = avg_corr(resid, ctrl_rand, min_ov=120, idx=idx)[0]
    cl = avg_corr(resid, ctrl_liq, min_ov=120, idx=idx)[0]
    print(f"  {lab}: 周期={np.nanmean(vals):+.3f}  普通随机={cr:+.3f}  "
          f"流动性匹配={cl:+.3f}")

print("\n" + "=" * 78)
print("H3. 子行业篮子 & PC1")
print("=" * 78)
bk = {g: resid[[c for c in v if c in resid.columns]].mean(axis=1)
      for g, v in kw_groups.items()}
bdf = pd.DataFrame(bk)
print(bdf.corr().round(3).to_string())
off = bdf.corr().values[np.triu_indices(len(kw_groups), 1)]
print(f"\n  子行业间平均残差相关 = {np.mean(off):+.3f}")
print(f"  周期子行业篮子(6个) PC1 = {pc1(bdf, list(kw_groups)):.1%}")
print(f"  周期个股面板 PC1 = {pc1(resid, kw_all):.1%}  "
      f"/ 随机个股面板 PC1 = {pc1(resid, ctrl_rand):.1%}")
print(f"  [参照] 6只周期ETF 65.5%")

print("\n" + "=" * 78)
