"""
周期特征诊断 — K线能否区分"周期 vs 非周期"？
=====================================================
只读诊断脚本，不产生回测，不构成投资建议。

核心问题：把"行业周期性"翻译成可从 OHLCV 计算的统计量，
先用数据检验 CYC(周期) 与 DEF(防御) 两组 ETF 是否真的可分。

度量定义
--------
1. ER   (Kaufman效率比) = |P_t - P_{t-n}| / Σ|ΔP|
   → 接近1: 单边趋势; 接近0: 来回震荡(周期性特征)
2. VR(q) (方差比) = Var(q日收益) / (q * Var(1日收益))
   → <1 均值回归(周期); ≈1 随机漫步; >1 趋势延续(成长)
3. AC(h) = corr(r_t, r_{t-h})  长滞后自相关
   → 负 = 长周期反转(周期的统计指纹)
4. MA250穿越次数/年 → 价格围绕长期中枢摆动的频率
5. 回撤特征 → 深且反复的回撤 = 周期股
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd

PARQUET = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
           "kline_1min_etf.parquet")

CYC = {
    "159930.SZ": "能源/煤炭",
    "512400.SH": "有色金属",
    "515210.SH": "钢铁",
    "159870.SZ": "化工",
    "159745.SZ": "建材",
    "560280.SH": "机械",
}
DEF = {
    "512170.SH": "医疗",
    "512010.SH": "医药",
    "159928.SZ": "消费",
    "159611.SZ": "公用事业",
}
BENCH = {"159915.SZ": "创业板指"}
ALL = {**CYC, **DEF, **BENCH}
codes_str = ", ".join(f"'{c}'" for c in ALL)

print("=" * 78)
print("周期特征诊断 — CYC(周期) vs DEF(防御) 的 K线统计指纹")
print("=" * 78)

con = duckdb.connect(":memory:")
daily = con.execute(f"""
    SELECT ts_code, trade_date,
           FIRST(open ORDER BY trade_time)  AS o,
           LAST(close ORDER BY trade_time)  AS c,
           MAX(high) AS h, MIN(low) AS l,
           SUM(volume) AS v, SUM(amount) AS amt
    FROM '{PARQUET}/**/*.parquet'
    WHERE ts_code IN ({codes_str})
    GROUP BY ts_code, trade_date
    ORDER BY ts_code, trade_date
""").fetchdf()
con.close()
daily["trade_date"] = pd.to_datetime(daily["trade_date"])
print(f"日线总行数: {len(daily):,}")


def efficiency_ratio(px, n):
    """Kaufman 效率比，滚动 n 日，返回均值"""
    d = px.diff().abs().rolling(n).sum()
    net = (px - px.shift(n)).abs()
    er = (net / d.replace(0, np.nan)).dropna()
    return float(er.mean()) if len(er) else np.nan


def variance_ratio(r, q):
    """VR(q) = Var(q期收益)/(q*Var(1期收益))"""
    r = r.dropna()
    if len(r) < q * 10:
        return np.nan
    var1 = r.var()
    if var1 == 0:
        return np.nan
    rq = (1 + r).rolling(q).apply(np.prod, raw=True) - 1
    rq = rq.dropna()
    if len(rq) < 10:
        return np.nan
    return float(rq.var() / (q * var1))


def autocorr(r, h):
    r = r.dropna()
    if len(r) < h * 5:
        return np.nan
    return float(r.autocorr(lag=h))


rows = []
for code, name in ALL.items():
    sub = daily[daily.ts_code == code].sort_values("trade_date")
    if len(sub) < 120:
        continue
    px = sub["c"].astype(float).reset_index(drop=True)
    r = np.log(px).diff()
    grp = "CYC" if code in CYC else ("DEF" if code in DEF else "BENCH")

    # MA250 穿越次数 / 年
    ma250 = px.rolling(250, min_periods=60).mean()
    above = (px > ma250).astype(int)
    cross = (above.diff().abs() > 0).sum()
    years = len(px) / 242
    cross_py = cross / years if years > 0 else np.nan

    # 回撤
    cummax = px.cummax()
    ddser = px / cummax - 1
    max_dd = float(ddser.min())
    # 深度回撤(<-20%)的次数
    deep = int((ddser < -0.20).sum())

    # 总涨幅 vs 路径长度
    total_ret = float(px.iloc[-1] / px.iloc[0] - 1)

    rows.append({
        "组": grp, "代码": code, "名称": name, "天数": len(px),
        "起始": str(sub.trade_date.iloc[0].date()),
        "年化波动": float(r.std() * np.sqrt(242)),
        "累计收益": total_ret,
        "ER20": efficiency_ratio(px, 20),
        "ER60": efficiency_ratio(px, 60),
        "VR5": variance_ratio(np.exp(r) - 1, 5),
        "VR20": variance_ratio(np.exp(r) - 1, 20),
        "VR60": variance_ratio(np.exp(r) - 1, 60),
        "AC1": autocorr(r, 1),
        "AC20": autocorr(r, 20),
        "AC60": autocorr(r, 60),
        "AC250": autocorr(r, 250),
        "MA250穿越/年": cross_py,
        "最大回撤": max_dd,
        "深回撤天数": deep,
    })

df = pd.DataFrame(rows).sort_values(["组", "代码"])

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:8.4f}")

print("\n--- 每个 ETF ---")
show = ["组", "代码", "名称", "天数", "年化波动", "累计收益",
        "ER20", "ER60", "VR5", "VR20", "VR60",
        "AC1", "AC20", "AC60", "AC250", "MA250穿越/年", "最大回撤"]
print(df[show].to_string(index=False))

print("\n--- 分组均值 (关键: 看 CYC 与 DEF 是否可分) ---")
agg = df.groupby("组")[["年化波动", "ER20", "ER60", "VR5", "VR20", "VR60",
                         "AC1", "AC20", "AC60", "AC250",
                         "MA250穿越/年", "最大回撤"]].mean()
print(agg.to_string())

print("\n--- 解读要点 ---")
print("  ER  低 = 震荡/周期性;  高 = 单边趋势")
print("  VR  <1 = 长周期反转(周期指纹);  >1 = 趋势延续(成长指纹)")
print("  AC60/AC250 为负 = 长周期均值回归")
print("  MA250穿越/年 高 = 反复围绕长期中枢摆动")

df.to_csv("factor_diagnosis.csv", index=False, encoding="utf-8-sig")
print("\n[已保存] factor_diagnosis.csv")
