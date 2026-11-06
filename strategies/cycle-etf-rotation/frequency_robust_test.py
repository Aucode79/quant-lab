"""
概念炒作噪声检验 — 频率与稳健性
=====================================================
只读诊断，不构成投资建议。

问题
----
个股常被"概念/题材"炒作驱动(金牛化工的"牛"、多氟多的液冷…),
这类个体噪声在**日频**收益里占比最大, 会污染行业共同因子的估计。
而行业周期是**低频**现象。因此:

  H1. 降频(日→周→月)后, 周期组的共同相关是否相对随机组变强?
  H2. 日频下去极值/秩相关, 能否削弱概念噪声?

分组
----
使用 6 个关键词组, 但**已剔除明显误召**并**补入漏召龙头**。
(剔除/补入清单在下方 EXCLUDE / ADD, 可自行修改)
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
# 明显误召 → 剔除
EXCLUDE = {"002678.SZ", "300329.SZ",   # 珠江钢琴/海伦钢琴 ("钢琴"含"钢")
           "300895.SZ", "600237.SH",   # 铜牛信息(IT)/铜峰电子(电容)
           "688548.SH", "300797.SZ",   # 广钢气体/钢研纳克
           "300226.SZ", "601968.SH",   # 上海钢联/宝钢包装
           "600660.SH",                # 福耀玻璃(汽车玻璃)
           "600916.SH", "601117.SH",   # 中国黄金(零售)/中国化学(工程)
           "000852.SZ", "603353.SH",   # 石化机械/和顺石油
           "002541.SZ", "002743.SZ", "600477.SH", "600496.SH",  # 钢结构
           "688119.SH", "000928.SH", "601068.SH", "920019.BJ"}  # 耐火/工程
# 改判
REASSIGN = {"600985.SH": "煤炭",    # 淮北矿业 是煤炭
            "000923.SZ": "有色"}    # 河钢资源 是铜矿
# 补入漏召龙头
ADD = {"煤炭": ["601088.SH", "600188.SH", "601699.SH", "002128.SZ",
                "600123.SH", "600508.SH", "601918.SH", "000937.SZ"],
       "有色": ["603993.SH", "000060.SZ", "600549.SH", "002460.SZ"],
       "建材": ["600176.SH", "000786.SZ", "002271.SZ", "600801.SH",
                "002080.SZ", "000012.SZ", "002233.SZ"],
       "化工": ["600426.SH", "601233.SH", "600143.SH", "600096.SH"],
       "石化": ["601808.SH", "000301.SZ"]}
N_RAND = 500
rng = np.random.default_rng(42)

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}'").fetchdf()
basic["name"] = basic["name"].str.replace(r"\s+", "", regex=True)

groups = {}
for g, ks in KW.items():
    pat = "|".join(ks)
    groups[g] = [c for c in basic[basic["name"].str.contains(pat, na=False)]["ts_code"]]
# 剔除误召
for g in groups:
    groups[g] = [c for c in groups[g] if c not in EXCLUDE]
# 改判
for c, g in REASSIGN.items():
    for gg in groups:
        if c in groups[gg] and gg != g:
            groups[gg].remove(c)
    if c not in groups[g]:
        groups[g].append(c)
# 补入
for g, codes in ADD.items():
    for c in codes:
        if c not in groups[g]:
            groups[g].append(c)

matched = set().union(*groups.values())
pool = [c for c in basic["ts_code"] if c not in matched]
ctrl = list(rng.choice(pool, size=N_RAND, replace=False))
groups["随机对照"] = ctrl

print("=" * 76)
print("概念炒作噪声检验 — 频率与稳健性")
print("=" * 76)
for g, v in groups.items():
    print(f"  {g:6s}: {len(v):4d} 只")

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
print(f"\n[查询] {len(sd):,} 行, {time.time()-t0:.1f}s")

PX = sd.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
mkt = np.log(ed.set_index("trade_date")["c"]).diff().reindex(PX.index)
r = np.log(PX).diff()
r = r.loc[mkt.notna()]
mkt = mkt.dropna()

# 残差化
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
print(f"[面板] {res.shape[0]} 日 x {res.shape[1]} 只\n")


def agg(res, rule):
    """按周/月聚合对数收益 (求和即复利)"""
    if rule is None:
        return res
    return res.resample(rule).sum(min_count=1)


def _mean_offdiag(cm):
    """相关矩阵上三角均值"""
    n = cm.shape[0]
    if n < 2:
        return np.nan
    iu = np.triu_indices(n, 1)
    v = cm[iu]
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else np.nan


def avg_corr(df, codes, min_ov):
    """用 pandas C 实现算 pairwise-complete 相关矩阵, 再取上三角均值"""
    sub = df[[c for c in codes if c in df.columns]]
    sub = sub.loc[:, sub.notna().sum() >= max(min_ov, 20)]
    if sub.shape[1] < 2:
        return np.nan
    cm = sub.corr(min_periods=min_ov).to_numpy(dtype=float)
    return _mean_offdiag(cm)


def avg_corr_rank(df, codes, min_ov):
    """秩相关 = 先对每列排名再算 Pearson"""
    sub = df[[c for c in codes if c in df.columns]]
    sub = sub.loc[:, sub.notna().sum() >= max(min_ov, 20)]
    if sub.shape[1] < 2:
        return np.nan
    cm = sub.rank().corr(min_periods=min_ov).to_numpy(dtype=float)
    return _mean_offdiag(cm)


cyc = [g for g in KW]
wc = agg(res, "W-FRI")
mc = agg(res, "ME")
# 去极值: 每股在自身 1%/99% 分位截断
w = res.copy()
for c in w.columns:
    lo, hi = w[c].quantile(0.01), w[c].quantile(0.99)
    w[c] = w[c].clip(lo, hi)

tests = [
    ("日频 Pearson", res, 150, avg_corr),
    ("日频 Spearman", res, 150, avg_corr_rank),
    ("日频 去极值(1/99)", w, 150, avg_corr),
    ("周频 Pearson", wc, 60, avg_corr),
    ("月频 Pearson", mc, 30, avg_corr),
]

print("=" * 76)
print("周期6组均值 vs 随机对照  (残差相关)")
print("=" * 76)
print(f"  {'处理':<18}{'周期均值':>10}{'随机对照':>10}{'差值':>10}{'倍数':>8}")
for lab, df, mo, fn in tests:
    cv = np.nanmean([fn(df, groups[g], mo) for g in cyc])
    rv = fn(df, groups["随机对照"], mo)
    print(f"  {lab:<18}{cv:>10.3f}{rv:>10.3f}{cv-rv:>+10.3f}{cv/max(rv,1e-6):>8.2f}")

print("\n" + "=" * 76)
print("结论提示")
print("=" * 76)
print("  若'差值'随频率下降(日→周→月)而**扩大**, 说明概念噪声确实主导日频,")
print("  低频才是行业共同因子的正确观测尺度。")
print("  若 Spearman / 去极值 明显高于原始 Pearson, 说明极端值(涨停等)在污染估计。")
