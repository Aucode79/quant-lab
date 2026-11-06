"""
周期择时信号测试 — 篮子层面
=====================================================
只读诊断，不构成投资建议。

测试"什么时候该持有周期板块"这个核心问题:
  T1. 周期篮子的时序动量/反转: 篮子自身收益 → 未来篮子收益
  T2. 周期篮子的长期位置 z → 未来篮子收益 (反转 or 延续?)
  T3. 现策略的门控信号 cyc_rs(CYC相对DEF强弱) → 未来 CYC 超额
  T4. 子期间稳定性 (2018-2021 vs 2022-2026)

注意: CYC 篮子成分随时间变化(2018 只有煤炭+有色), 因此
      主线用"核心篮子"(两只全程 ETF)保证成分恒定, 另附 5 只篮子。
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

PARQUET = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
           "kline_1min_etf.parquet")
CORE = ["159930.SZ", "512400.SH"]                 # 全程可用, 成分恒定
CYC5 = ["159930.SZ", "512400.SH", "515210.SH", "159870.SZ", "159745.SZ"]
DEF = ["512170.SH", "512010.SH", "159928.SZ", "159611.SZ"]
ALL = sorted(set(CORE + CYC5 + DEF))
cs = ", ".join(f"'{c}'" for c in ALL)

con = duckdb.connect(":memory:")
daily = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PARQUET}/**/*.parquet'
    WHERE ts_code IN ({cs})
    GROUP BY ts_code, trade_date ORDER BY ts_code, trade_date
""").fetchdf()
con.close()
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
close = daily.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
ret = np.log(close).diff()

core_b = ret[CORE].mean(axis=1)
cyc5_b = ret[CYC5].mean(axis=1)
def_b = ret[[c for c in DEF if c in ret.columns]].mean(axis=1)

print("=" * 78)
print("周期择时信号测试 (篮子层面)")
print("=" * 78)


def fwd_sum(r, h):
    return r.rolling(h).sum().shift(-h)


def ts_ic(sig, basket, hor, lo=None, hi=None, step=1):
    """时序 IC: 信号 vs 未来篮子收益"""
    f = fwd_sum(basket, hor)
    d = pd.concat([sig, f], axis=1).dropna()
    if lo:
        d = d[(d.index >= lo) & (d.index <= hi)]
    if len(d) < 60:
        return np.nan, len(d)
    # 用非重叠采样降低自相关 (每 hor 取一个点)
    d2 = d.iloc[::max(hor, 1)]
    if len(d2) < 20:
        d2 = d
    return float(np.corrcoef(d2.iloc[:, 0], d2.iloc[:, 1])[0, 1]), len(d2)


def run_block(basket, name):
    lr = np.log(close)
    ts_log = np.log((1 + basket).cumprod())      # 篮子对数净值
    sig = {}
    for h in [20, 60, 120, 250]:
        sig[f"mom{h}"] = ts_log - ts_log.shift(h)
    m250 = ts_log.rolling(250, min_periods=120).mean()
    s250 = ts_log.rolling(250, min_periods=120).std()
    sig["pos250"] = (ts_log - m250) / s250
    m500 = ts_log.rolling(500, min_periods=250).mean()
    s500 = ts_log.rolling(500, min_periods=250).std()
    sig["pos500"] = (ts_log - m500) / s500
    # 现策略门控: CYC 相对 DEF 的 60 日相对强弱 z
    rs_c = ts_log - np.log((1 + def_b).cumprod())
    sig["cyc_rs60"] = rs_c - rs_c.rolling(60).mean()

    print(f"\n--- {name} ---")
    print(f"{'信号':<12}" + "".join(f"{'fwd'+str(h):>10}" for h in [20, 60, 120, 250]))
    for sn, s in sig.items():
        row = f"{sn:<12}"
        for h in [20, 60, 120, 250]:
            ic, n = ts_ic(s, basket, h)
            row += f"{ic:>10.3f}"
        print(row)

    print(f"  [子期间] 信号 mom120 / pos250 的 fwd120 IC")
    for lo, hi, lab in [("2018-01-01", "2021-12-31", "2018-2021"),
                        ("2022-01-01", "2026-06-30", "2022-2026")]:
        i1, _ = ts_ic(sig["mom120"], basket, 120, lo, hi)
        i2, _ = ts_ic(sig["pos250"], basket, 120, lo, hi)
        print(f"    {lab}: mom120={i1:+.3f}  pos250={i2:+.3f}")
    return sig


print("\n" + "=" * 78)
print("T1/T2. 周期篮子自身的时序信号")
print("=" * 78)
run_block(core_b, "核心篮子 (煤炭+有色, 2018-2026 成分恒定)")
run_block(cyc5_b, "5只周期篮子 (2020+ 才有全成分)")

print("\n" + "=" * 78)
print("T3. 现策略门控 cyc_rs60 → 未来 CYC 超额收益 (CYC - DEF)")
print("=" * 78)
ts_log = np.log((1 + cyc5_b).cumprod())
rs_c = ts_log - np.log((1 + def_b).cumprod())
gate = rs_c - rs_c.rolling(60).mean()
exc = cyc5_b - def_b
for h in [20, 60, 120]:
    f = fwd_sum(exc, h)
    d = pd.concat([gate, f], axis=1).dropna()
    d2 = d.iloc[::h]
    ic = float(np.corrcoef(d2.iloc[:, 0], d2.iloc[:, 1])[0, 1]) if len(d2) > 20 else np.nan
    # 分组均值
    q = pd.qcut(d.iloc[:, 0], 3, labels=["low", "mid", "high"], duplicates="drop")
    gm = d.groupby(q, observed=True)[d.columns[1]].mean()
    print(f"  fwd{h:>3}: IC={ic:+.3f} | 门控分3组未来超额均值 "
          f"low={gm.get('low', np.nan):+.4f} mid={gm.get('mid', np.nan):+.4f} "
          f"high={gm.get('high', np.nan):+.4f}")

print("\n" + "=" * 78)
print("说明: IC 基于非重叠采样降低自相关; 样本仅 8 年≈1.5个周期, 谨慎解读")
print("=" * 78)
