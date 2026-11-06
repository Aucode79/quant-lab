"""
周期标的池构建 — 双源交叉法
=====================================================
只读诊断 + 产出标的池清单，不构成投资建议。

方法
----
源1 (官方): 证监会行业分类的周期大类 (B06/B07/B08/B09/C25/C26/C28/C30/C31/C32)
            → 提供候选广度 (689只) 与种子
源2 (数据): 种子锚定扩散 — 每组取组内凝聚力top5做种子,
            全市场每只股票(含无分类退市股)与6组种子的残差相关 = 周期暴露
阈值: 数据自定 — 随机对照(非周期分类股)暴露分布的95分位 = 入池下限
窗口: 全期/2018-2021/2022-2026 三窗 → 稳定 vs 漂移
过滤: 近252有效日中位成交额 >= 2000万
输出: 标的池_分层清单.csv (L1核心/L2扩展/L3边界 + 漂移标记)
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
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"
MKT = "510300.SH"
OUT = "C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股"
MIN_AMT = 20_000_000

# 证监会大类 → 6 个周期种子组
GROUP_DEF = {
    "煤炭": ["B06"],
    "钢铁": ["B08", "C31"],
    "有色": ["B09", "C32"],
    "建材": ["C30"],
    "化工": ["C26", "C28"],
    "石化": ["B07", "C25"],
}
ALL_CYC_PREFIX = [p for v in GROUP_DEF.values() for p in v]

print("=" * 78)
print("周期标的池构建 — 证监会分类 × 种子锚定扩散 双源交叉")
print("=" * 78)

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")

ind = con.execute(f"SELECT ts_code, name, industry FROM '{REF}/csrc_industry.parquet'").fetchdf()
basic = con.execute(f"SELECT ts_code, ipoDate, outDate FROM '{REF}/stock_basic_bs.parquet'").fetchdf()
ind["prefix"] = ind["industry"].fillna("").str[:3]
ind = ind.merge(basic, on="ts_code", how="left")
n_all = len(ind)
cyc_mask = ind["prefix"].isin(ALL_CYC_PREFIX)
print(f"全市场 {n_all} 只; 证监会周期大类 {cyc_mask.sum()} 只; "
      f"无分类(多为退市) {ind['industry'].eq('').sum()} 只")

# ---- 全市场日线 (一次查询) ----
print("\n[查询] 全市场日线聚合 (约1-3分钟)...")
t0 = time.time()
daily = con.execute(f"""
    SELECT ts_code, trade_date,
           LAST(close ORDER BY trade_time) AS c,
           SUM(amount) AS amt
    FROM '{PQ_STOCK}/**/*.parquet'
    WHERE year BETWEEN 2018 AND 2026
    GROUP BY ts_code, trade_date
""").fetchdf()
mkt = con.execute(f"""
    SELECT trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_ETF}/**/*.parquet'
    WHERE year BETWEEN 2018 AND 2026 AND ts_code = '{MKT}'
    GROUP BY trade_date
""").fetchdf()
con.close()
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
mkt["trade_date"] = pd.to_datetime(mkt["trade_date"])
print(f"[查询] {len(daily):,} 行  {time.time()-t0:.0f}s")

close = daily.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
amt = daily.pivot(index="trade_date", columns="ts_code", values="amt").sort_index()
codes_in_data = set(close.columns)
print(f"[面板] {close.shape[0]}日 x {close.shape[1]}只  "
      f"({close.index[0].date()} ~ {close.index[-1].date()})")

mret = np.log(mkt.set_index("trade_date")["c"]).diff()
mret = mret.reindex(close.index)


def resid_window(lo, hi):
    """窗口内残差收益矩阵 (剥市场beta), NaN填0前先按列标准化记录有效天数"""
    idx = close.index[(close.index >= lo) & (close.index <= hi)]
    C = np.log(close.loc[idx])
    R = C.diff()
    m = mret.loc[idx]
    R = R.loc[m.notna()]
    m = m.dropna()
    vm = m.var()
    out = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    for c in R.columns:
        s = R[c]
        ok = s.notna()
        if ok.sum() < 100:
            continue
        b = np.cov(s[ok], m[ok])[0, 1] / vm
        out.loc[ok, c] = s[ok] - b * m[ok]
    # 有效天数
    valid = (out != 0.0).sum(axis=0)
    # 列标准化 (在有效值上)
    Z = out.copy()
    for c in Z.columns:
        v = Z[c].values
        nz = v != 0.0
        if nz.sum() > 30 and v[nz].std() > 0:
            mu, sd = v[nz].mean(), v[nz].std()
            Z[c] = np.where(nz, (v - mu) / sd, 0.0)
    return Z, valid


WINDOWS = {
    "full": ("2018-01-01", "2026-12-31"),
    "early": ("2018-01-01", "2021-12-31"),
    "late": ("2022-01-01", "2026-12-31"),
}

# ---- 第一步: 组内凝聚力选种子 (用全期窗口) ----
print("\n[种子] 证监会周期大类内凝聚力 top5:")
Zf, validf = resid_window(*WINDOWS["full"])
seeds = {}
for g, prefixes in GROUP_DEF.items():
    members = [c for c in ind[(ind["prefix"].isin(prefixes))]["ts_code"]
               if c in Zf.columns and validf.get(c, 0) >= 800]
    if len(members) < 6:
        continue
    sub = Zf[members]
    cm = np.array((sub.T @ sub) / len(sub), dtype=float)  # 可写副本
    np.fill_diagonal(cm, 0)
    coh = pd.Series(cm.mean(axis=1), index=members).sort_values(ascending=False)
    seeds[g] = list(coh.head(5).index)
    print(f"  {g}: {' '.join(seeds[g])}  (组内{len(members)}只)")

all_seeds = [s for v in seeds.values() for s in v]
Sf = Zf[all_seeds]

# ---- 第二步: 全市场暴露 (三窗口) ----
print("\n[暴露] 全市场 × 6组种子, 三窗口:")
exposure = {}
valid_all = {}
for wname, (lo, hi) in WINDOWS.items():
    Z, valid = resid_window(lo, hi)
    Sv = Z[all_seeds]
    E = (Z.T @ Sv)  # stock × seed 点积
    # 除以每只股票自己的有效天数(而非窗口长度):
    # 避免短历史/退市股的暴露被系统性稀释
    vd = valid.clip(lower=1)
    E = E.div(vd, axis=0)
    # 组平均
    G = pd.DataFrame(index=E.index, columns=list(seeds.keys()), dtype=float)
    for g, ss in seeds.items():
        G[g] = E[ss].mean(axis=1)
    exposure[wname] = G
    valid_all[wname] = valid
    print(f"  {wname}: {Z.shape[0]}日  暴露矩阵 {G.shape}")

# ---- 第三步: 阈值 (随机对照 95 分位) ----
noncyc = [c for c in ind[~cyc_mask & ind["industry"].ne("")]["ts_code"]
          if c in exposure["full"].index]
rng = np.random.default_rng(42)
ctrl_codes = list(rng.choice(noncyc, size=min(800, len(noncyc)), replace=False))
ctrl_max = exposure["full"].loc[ctrl_codes].max(axis=1)
th_pool = float(ctrl_max.quantile(0.95))
th_core = float(ctrl_max.quantile(0.99))
print(f"\n[阈值] 随机对照(n={len(ctrl_codes)}) max暴露分布: "
      f"中位={ctrl_max.median():.3f}  95分位={th_pool:.3f}  99分位={th_core:.3f}")
print(f"  → 入池下限(扩展层)={th_pool:.3f}  核心层={th_core:.3f}")

# ---- 第四步: 汇总每只股票 ----
name_map = dict(zip(ind["ts_code"], ind["name"]))
csrc_map = dict(zip(ind["ts_code"], ind["industry"].fillna("")))


def cyc_map_code(code):
    row = ind[ind["ts_code"] == code]
    return len(row) > 0 and row.iloc[0]["prefix"] in ALL_CYC_PREFIX


rows = []
for code in close.columns:
    if valid_all["full"].get(code, 0) < 200:
        continue
    # 主窗口: 取该股票有效数据最多的窗口 (退市股用早期窗, 新股用晚期窗)
    main_w = max(["full", "early", "late"],
                 key=lambda w: valid_all[w].get(code, 0))
    ef = exposure[main_w].loc[code]
    em = ef
    g_best, v_best = em.idxmax(), em.max()
    # 三窗达标
    hit = {}
    for w in ["full", "early", "late"]:
        if valid_all[w].get(code, 0) >= 200:
            hit[w] = bool(exposure[w].loc[code].max() >= th_pool)  # 显式转Python bool
        else:
            hit[w] = None
    n_hit = sum(1 for v in hit.values() if v is True)
    n_avail = sum(1 for v in hit.values() if v is not None)
    # 流动性: 最后252个有效日中位成交额
    a = amt[code].dropna()
    liq = float(a.tail(252).median()) if len(a) else 0.0
    rows.append({
        "ts_code": code, "名称": name_map.get(code, "?"),
        "证监会行业": csrc_map.get(code, ""),
        "周期分类": "Y" if cyc_map_code(code) else
                   ("无分类" if csrc_map.get(code, "") == "" else "N"),
        "主窗口": main_w,
        "暴露_煤炭": round(ef.get("煤炭", np.nan), 3),
        "暴露_钢铁": round(ef.get("钢铁", np.nan), 3),
        "暴露_有色": round(ef.get("有色", np.nan), 3),
        "暴露_建材": round(ef.get("建材", np.nan), 3),
        "暴露_化工": round(ef.get("化工", np.nan), 3),
        "暴露_石化": round(ef.get("石化", np.nan), 3),
        "max暴露": round(v_best, 3), "最强组": g_best,
        "全期达标": hit["full"], "早期达标": hit["early"], "晚期达标": hit["late"],
        "达标窗口数": f"{n_hit}/{n_avail}",
        "中位日成交额(万)": round(liq / 1e4, 1),
    })


def cyc_map_code(code, ind_df, prefixes):
    row = ind_df[ind_df["ts_code"] == code]
    return len(row) > 0 and row.iloc[0]["prefix"] in prefixes


pool = pd.DataFrame(rows)

# ---- 第五步: 分层 ----
def tier(r):
    if r["max暴露"] < th_pool:
        return "-"
    liquid = r["中位日成交额(万)"] >= MIN_AMT / 1e4
    n_hit, n_av = map(int, r["达标窗口数"].split("/"))
    stable = (n_hit == n_av)  # 所有有数据的窗口都达标
    if r["max暴露"] >= th_core and stable and liquid:
        return "L1核心"
    if stable and liquid:
        return "L2扩展"
    if not stable:
        return "漂移"
    return "L3边界(流动性不足)"


pool["层级"] = pool.apply(tier, axis=1)

# ---- 输出 ----
print("\n" + "=" * 78)
print("标的池分层结果")
print("=" * 78)
print(pool["层级"].value_counts().to_string())
for t in ["L1核心", "L2扩展", "漂移"]:
    sub = pool[pool["层级"] == t].sort_values("max暴露", ascending=False)
    print(f"\n【{t}】{len(sub)} 只 — 按最强组分布:")
    print(sub["最强组"].value_counts().to_string())

# 来源交叉: 官方分类 vs 数据验证
print("\n[来源交叉] L1+L2 的官方分类构成:")
inpool = pool[pool["层级"].isin(["L1核心", "L2扩展"])]
print(inpool["周期分类"].value_counts().to_string())
n_csrc_cyc = len(ind[ind["prefix"].isin(ALL_CYC_PREFIX) &
                     ind["ts_code"].isin(set(inpool["ts_code"]))])
print(f"  证监会周期大类 {int((ind['prefix'].isin(ALL_CYC_PREFIX)).sum())} 只中 "
      f"{n_csrc_cyc} 只通过数据验证入池")

# 漂移股示例
drift = pool[pool["层级"] == "漂移"].sort_values("max暴露", ascending=False)
if len(drift):
    print(f"\n【漂移】{len(drift)} 只 — 示例(早期/晚期暴露不一致):")
    print(drift.head(12)[["ts_code", "名称", "证监会行业", "max暴露",
                          "全期达标", "早期达标", "晚期达标"]].to_string(index=False))

pool.sort_values(["层级", "max暴露"], ascending=[True, False]).to_csv(
    f"{OUT}/标的池_分层清单.csv", index=False, encoding="utf-8-sig")
print(f"\n[已保存] {OUT}/标的池_分层清单.csv  (共{len(pool)}只)")

# 与关键词清单交叉
kw_codes = set()
try:
    kw = pd.read_csv(f"{OUT}/分组清单_待人工核对.csv")
    kw_codes = set(kw["ts_code"])
    inpool = pool[(pool["层级"].isin(["L1核心", "L2扩展"])) & pool["ts_code"].isin(kw_codes)]
    print(f"\n[交叉] 关键词清单148只中 {len(inpool)} 只进入L1/L2; "
          f"未入池 {len(kw_codes - set(inpool['ts_code']))} 只"
          f" (多为概念股/误分类 → 被数据过滤)")
except Exception:
    pass
