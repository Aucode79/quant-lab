"""
新类别种子评分 — 生猪/航运/造纸/橡胶塑料
=====================================================
沿用工业六组的暴露方法: 剥离沪深300的日残差收益 与
新类别种子篮子的相关 (全期/早期/晚期三窗口)。

只做证据层, 不做门槛 (注册表v2原则):
  - 对注册表全部1057只算 暴露_生猪/航运/造纸/橡胶 四列
  - 附: 新类别组内共同因子强度 vs 匹配随机对照 (仅注释)
输出: 更新 标的池_注册表_v2_20260913.csv
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股")
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"
PQ_STOCK = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
            "kline_1min.parquet")
PQ_ETF = ("C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
          "kline_1min_etf.parquet")
MKT = "510300.SH"

SEEDS = {
    "生猪": ["002714.SZ", "300498.SZ", "000876.SZ"],       # 牧原/温氏/新希望
    "航运": ["601919.SH", "601872.SH", "600026.SH"],       # 中远海控/招商轮船/中远海能
    "造纸": ["002078.SZ", "600567.SH", "000488.SZ"],       # 太阳纸业/山鹰国际/晨鸣纸业
    "橡胶": ["601118.SH", "601058.SH", "601966.SH"],       # 海南橡胶/赛轮/玲珑
}

reg = pd.read_csv(BASE / "标的池_注册表_v2_20260913.csv")
all_codes = sorted(set(reg["ts_code"]) | {s for v in SEEDS.values() for s in v} | {MKT})
print(f"注册表 {len(reg)} 只; 面板查询 {len(all_codes)} 只")

con = duckdb.connect(":memory:")
con.execute("SET enable_progress_bar=false")
cs = ", ".join(f"'{c}'" for c in all_codes)
t0 = time.time()
sd = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_STOCK}/**/*.parquet'
    WHERE year BETWEEN 2018 AND 2026 AND ts_code IN ({cs})
    GROUP BY ts_code, trade_date
""").fetchdf()
ed = con.execute(f"""
    SELECT trade_date, LAST(close ORDER BY trade_time) AS c
    FROM '{PQ_ETF}/**/*.parquet'
    WHERE year BETWEEN 2018 AND 2026 AND ts_code = '{MKT}'
    GROUP BY trade_date
""").fetchdf()
con.close()
for d in (sd, ed):
    d["trade_date"] = pd.to_datetime(d["trade_date"])
print(f"[查询] {len(sd):,}行 ({time.time()-t0:.0f}s)")

PX = sd.pivot(index="trade_date", columns="ts_code", values="c").sort_index()
mret = np.log(ed.set_index("trade_date")["c"]).diff().reindex(PX.index)
R = np.log(PX).diff().loc[mret.notna()]
m = mret.dropna()

WINDOWS = {"full": ("2018-01-01", "2027-01-01"),
           "early": ("2018-01-01", "2022-01-01"),
           "late": ("2022-01-01", "2027-01-01")}


def resid_window(lo, hi):
    idx = R.index[(R.index >= lo) & (R.index < hi)]
    Rw = R.loc[idx]
    mw = m.loc[idx]
    vm = mw.var()
    Z = pd.DataFrame(0.0, index=Rw.index, columns=Rw.columns)
    for c in Rw.columns:
        s = Rw[c]
        ok = s.notna()
        if ok.sum() < 100:
            continue
        b = np.cov(s[ok], mw[ok])[0, 1] / vm
        Z.loc[ok, c] = s[ok] - b * mw[ok]
    # 列标准化 (有效值上)
    valid = (Z != 0.0).sum(axis=0)
    for c in Z.columns:
        v = Z[c].values
        nz = v != 0.0
        if nz.sum() > 30 and v[nz].std() > 0:
            Z[c] = np.where(nz, (v - v[nz].mean()) / v[nz].std(), 0.0)
    return Z, valid


exposure_cols = {f"暴露_{g}": {} for g in SEEDS}
stab_cols = {f"{g}窗口": {} for g in SEEDS}
Zw = {}
for wname, (lo, hi) in WINDOWS.items():
    Z, valid = resid_window(lo, hi)
    Zw[wname] = Z
    for g, seeds in SEEDS.items():
        ss = [s for s in seeds if s in Z.columns and valid.get(s, 0) > 200]
        if not ss:
            print(f"[警告] {g} 种子不可用: {seeds}")
            continue
        b = Z[ss].mean(axis=1)          # 种子篮子 (已标准化列均值)
        bs = b.std()
        if bs > 0:
            b = (b - b.mean()) / bs
        # 暴露 = corr(stock, basket) ≈ Σ z·b / 重叠交易日数 (逐股重叠, 防短史稀释)
        n = len(Z)
        E = (Z.T @ b)
        overlap = (Z != 0).T.astype(float) @ (b != 0).astype(float)
        E = E / overlap.clip(lower=1)
        if wname == "full":
            for c in E.index:
                exposure_cols[f"暴露_{g}"][c] = float(E[c])
        else:
            for c in E.index:
                stab_cols[f"{g}窗口"].setdefault(c, {})[wname] = float(E[c])
    print(f"[{wname}] {Z.shape[0]}日 完成")

# 合并进注册表
for col, d in exposure_cols.items():
    reg[col] = reg["ts_code"].map(d).round(3)
for g in SEEDS:
    vals = []
    for _, r in reg.iterrows():
        w = stab_cols[f"{g}窗口"].get(r["ts_code"], {})
        if len(w) < 2:
            vals.append(np.nan)
        else:
            same = np.sign(w.get("early", 0)) == np.sign(w.get("late", 0)) \
                and abs(w.get("early", 0)) > 0.15
            vals.append("稳" if same else "不稳")
    reg[f"{g}稳定性"] = vals

# ============ 注释: 新类别组内共同因子 vs 匹配对照 ============
print("\n[注释] 新类别组内平均残差相关 vs 匹配随机对照 (全期, 仅参考):")
Zf = Zw["full"]
csrc = pd.read_parquet(f"{REF}/csrc_industry.parquet")
csrc["prefix"] = csrc["industry"].fillna("").str[:3]
CYC = {"B06","B07","B08","B09","C25","C26","C28","C30","C31","C32",
       "C29","A03","C22"}
CAT_CLS = {"生猪": ["A03"], "航运": ["G55"], "造纸": ["C22"],
           "橡胶": ["C29"]}
rng = np.random.default_rng(42)


def avg_corr(cols):
    cols = [c for c in cols if c in Zf.columns]
    cols = [c for c in cols if (Zf[c] != 0).sum() >= 400]
    if len(cols) < 3:
        return np.nan, 0
    cm = np.array(Zf[cols].corr(min_periods=200).fillna(0.0), dtype=float)
    np.fill_diagonal(cm, 0)
    iu = np.triu_indices(len(cols), 1)
    v = cm[iu]
    v = v[~np.isnan(v)]
    return (float(v.mean()) if len(v) else np.nan), len(cols)


noncyc = csrc[~csrc["prefix"].isin(CYC)]["ts_code"].tolist()
for g, cls in CAT_CLS.items():
    members = reg[(reg["行业类别"].isin(
        {"生猪": ["生猪养殖"], "航运": ["航运港口"], "造纸": ["造纸"],
         "橡胶": ["橡胶塑料"]}[g]))]["ts_code"].tolist()
    mv, npair = avg_corr(members)
    ctrl = list(rng.choice(noncyc, size=len(members), replace=False))
    cv, _ = avg_corr(ctrl)
    print(f"  {g}: 组内相关={mv:.3f} (n对={npair})  匹配对照={cv:.3f}  差={mv-cv:+.3f}")

# ============ 输出 ============
reg.to_csv(BASE / "标的池_注册表_v2_20260913.csv", index=False,
           encoding="utf-8-sig")
print(f"\n[已更新] 注册表 (新增4列暴露+4列稳定性)")

print("\n[新类别成员 暴露排名] (每类前5)")
for g, cat in [("生猪", "生猪养殖"), ("航运", "航运港口"),
               ("造纸", "造纸"), ("橡胶", "橡胶塑料")]:
    sub = reg[reg["行业类别"] == cat].sort_values(f"暴露_{g}",
                                                  ascending=False).head(5)
    names = " | ".join(f"{r['名称']}({r[f'暴露_{g}']})"
                       for _, r in sub.iterrows())
    print(f"  {g}: {names}")

print("\n[跨界发现] 非本类别但新暴露高的股票 (每类前3)")
for g in SEEDS:
    sub = reg[(reg["行业类别"] != {"生猪": "生猪养殖", "航运": "航运港口",
                                   "造纸": "造纸", "橡胶": "橡胶塑料"}[g]) &
              reg[f"暴露_{g}"].notna()].sort_values(f"暴露_{g}", ascending=False).head(3)
    names = " | ".join(f"{r['名称']}[{r['行业类别']}]({r[f'暴露_{g}']})"
                       for _, r in sub.iterrows())
    print(f"  {g}: {names}")
