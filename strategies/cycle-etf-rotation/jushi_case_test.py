"""
个案检验: 中国巨石(600176.SH) 是"建材/周期"还是"电子/AI"?
=====================================================
只读诊断，不构成投资建议。

背景: 中国巨石是全球玻纤龙头, 但同时也是"电子布一哥"。
      2025-26 因 AI/PCB 带动电子布涨价, 电子纱及电子布收入占比
      从 ~18% 升到 24%, 市场按"电子/AI"逻辑重估。
      → 静态行业标签在某个时点上必然是错的。

方法: 剥离沪深300 beta 后, 比较 中国巨石 残差收益与
      「建材/化工/钢铁 股票篮子」 vs 「半导体/芯片/科技 ETF篮子」
      的相关性, 并分早期(2018-2021)/晚期(2022-2026)看漂移。
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
TARGET = "600176.SH"          # 中国巨石
MKT = "510300.SH"
TECH = ["512480.SH", "159995.SZ", "512760.SH", "515000.SH",
        "512720.SH", "515880.SH", "159801.SZ", "159939.SZ"]
# 电子布的直接下游是覆铜板(CCL)/PCB, 不是半导体
PCB = ["600183.SH", "002463.SZ", "002916.SZ", "002938.SZ", "300476.SZ"]

KW = {
    "建材": ["水泥", "玻璃"],
    "化工": ["化工", "化学", "化纤"],
    "钢铁": ["钢"],
}
YRS = (2018, 2026)
yr = ",".join(str(y) for y in range(YRS[0], YRS[1] + 1))

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")
basic = con.execute(f"SELECT ts_code, name FROM '{BASIC}'").fetchdf()
grp = {g: [c for c in basic[basic["name"].str.contains("|".join(ks), na=False)]["ts_code"]]
       for g, ks in KW.items()}
stocks = sorted(set().union(*grp.values()) | {TARGET} | set(PCB))
cs = ", ".join(f"'{c}'" for c in stocks)
ecs = ", ".join(f"'{c}'" for c in TECH + [MKT])

print("=" * 74)
print(f"中国巨石({TARGET}) 归属检验 — 建材/周期 vs 电子/科技")
print("=" * 74)
t0 = time.time()
sd = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_STOCK}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code IN ({cs})
    GROUP BY ts_code, trade_date
""").fetchdf()
ed = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_ETF}/**/*.parquet'
    WHERE year IN ({yr}) AND ts_code IN ({ecs})
    GROUP BY ts_code, trade_date
""").fetchdf()
con.close()
for d in (sd, ed):
    d["trade_date"] = pd.to_datetime(d["trade_date"])
print(f"[查询] stock {len(sd):,}行 + etf {len(ed):,}行, {time.time()-t0:.1f}s")

S = sd.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
E = ed.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
# 合并面板
PX = S.join(E, how="outer").sort_index()
ret = np.log(PX).diff()
m = np.log(PX[MKT]).diff()
ret = ret.loc[m.notna()]
m = m.dropna()

# 残差化(剥离沪深300)
res = pd.DataFrame(index=ret.index, columns=ret.columns, dtype=float)
vm = m.var()
for c in ret.columns:
    s = ret[c]
    ok = s.notna()
    if ok.sum() < 100:
        continue
    b = np.cov(s[ok], m[ok])[0, 1] / vm
    res.loc[ok, c] = s[ok] - b * m[ok]

# 篮子
def basket(cols):
    cc = [c for c in cols if c in res.columns]
    return res[cc].mean(axis=1)

buckets = {f"{g}篮子": basket(v) for g, v in grp.items()}
buckets["PCB/覆铜板篮子"] = basket(PCB)
buckets["电子/科技ETF篮子"] = basket(TECH)

tg = res[TARGET]
print(f"\n[基准] {TARGET} 残差波动 = {tg.std()*np.sqrt(242):.2%}/年")


def corr_by(idx, lab):
    print(f"\n--- {lab} (n={len(idx)}) ---")
    out = {}
    for k, b in buckets.items():
        p = pd.concat([tg.loc[idx], b.loc[idx]], axis=1).dropna()
        if len(p) < 60:
            print(f"  {k:<18} 样本不足")
            continue
        cv = float(np.corrcoef(p.iloc[:, 0], p.iloc[:, 1])[0, 1])
        out[k] = cv
        print(f"  {k:<18} 残差相关 = {cv:+.3f}")
    return out


allc = corr_by(res.index, "全期 2018-2026")
early = corr_by(res.index[res.index <= "2021-12-31"], "早期 2018-2021")
late = corr_by(res.index[res.index >= "2022-01-01"], "晚期 2022-2026")

print("\n" + "=" * 74)
print("漂移对比")
print("=" * 74)
print(f"  {'篮子':<18}{'早期':>10}{'晚期':>10}{'变化':>10}")
for k in allc:
    print(f"  {k:<18}{early.get(k, np.nan):>10.3f}{late.get(k, np.nan):>10.3f}"
          f"{late.get(k, np.nan) - early.get(k, np.nan):>+10.3f}")
print("\n说明: 相关为剥离沪深300后的残差相关; 2025-26 电子布涨价期间")
print("      应在'电子'一列看到抬升 —— 静态行业标签无法刻画这种漂移。")
