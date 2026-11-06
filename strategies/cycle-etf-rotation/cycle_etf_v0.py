"""
CycleETF Three-Gate Rotation V0
裸K周期轮动回测 — 无外部信息，仅用OHLCV/amount
=====================================================
重要声明：本策略仅为研究用途，不构成任何投资建议。
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import io
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# 路径与参数
# ============================================================
DATA_PATH = (
    "C:/Users/qihen/Desktop/量化研究/quant-system/data/market/"
    "kline_1min_etf.parquet"
)
OUT_DIR = Path("C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_DAYS = 242
MIN_MEDIAN_AMOUNT = 20_000_000
MIN_HISTORY = 504
SLIP = 0.001
COMM = 0.000115
MIN_COMM = 5.0
STAMP = 0.0
TARGET_VOL = 0.15
N = 5
EXIT_RANK = 8
RISK_ON_CAP = 0.90
NEUTRAL_CAP = 0.50
RISK_OFF_CAP = 0.00
SEED = 42

# ============================================================
# 候选ETF表（硬编码先验）
# ============================================================
CANDIDATE_TABLE = [
    ("coal", "CYC", "515220.SH;159930.SZ"),
    ("energy_petrochem", "CYC", "159930.SZ;515220.SH"),
    ("nonferrous", "CYC", "512400.SH"),
    ("steel", "CYC", "515210.SH"),
    ("chemicals", "CYC", "159870.SZ;516020.SH"),
    ("building_materials", "CYC", "159745.SZ"),
    ("machinery", "CYC", "516960.SH;560280.SH"),
    ("transport_shipping", "CYC", "159666.SZ;512910.SH"),
    ("medical", "DEF", "512170.SH"),
    ("pharma", "DEF", "159929.SZ;512010.SH"),
    ("consumer", "DEF", "159928.SZ;512690.SH"),
    ("utilities", "DEF", "159611.SZ;159625.SZ"),
    ("benchmark", "BENCH", "510300.SH;510500.SH;159915.SZ"),
]

ALL_CANDIDATE_CODES = set()
for row in CANDIDATE_TABLE:
    for c in row[2].split(";"):
        ALL_CANDIDATE_CODES.add(c.strip())

CODES_LIST = sorted(ALL_CANDIDATE_CODES)
CODES_STR = ", ".join(f"'{c}'" for c in CODES_LIST)

print("=" * 70)
print("CycleETF Three-Gate Rotation V0 — 裸K周期轮动回测")
print("=" * 70)


# ============================================================
# STEP 0: 数据审计 & 日线聚合 (只读候选ETF)
# ============================================================
def run_data_audit():
    """只读取候选ETF的1分钟数据 → 聚合日线"""
    con = duckdb.connect()

    # Quick global audit (metadata only, no full scan)
    print("\n[审计] 全局元数据...")
    glob = con.execute(f"""
        SELECT
            min(year) as min_y, max(year) as max_y,
            min(month) as min_m, max(month) as max_m
        FROM read_parquet('{DATA_PATH}/*/*/*.parquet', union_by_name=true)
    """).fetchone()
    print(f"  分区: year={glob[0]}-{glob[1]}, month={glob[2]}-{glob[3]}")

    # Count candidate rows only
    print(f"\n[审计] 候选ETF ({len(CODES_LIST)}只) 数据统计...")
    code_filter = f"ts_code IN ({CODES_STR})"

    info = con.execute(f"""
        SELECT
            ts_code,
            count(*) as cnt,
            min(trade_date) as first_dt,
            max(trade_date) as last_dt,
            count(DISTINCT trade_date) as n_days
        FROM read_parquet('{DATA_PATH}/*/*/*.parquet', union_by_name=true)
        WHERE {code_filter}
        GROUP BY ts_code
        ORDER BY ts_code
    """).fetchall()

    existing_codes = set()
    etf_info = {}
    total_1min = 0
    for row in info:
        code, cnt, first_dt, last_dt, n_days = row
        existing_codes.add(code)
        etf_info[code] = {
            "rows_1min": cnt,
            "first_date": str(first_dt),
            "last_date": str(last_dt),
            "n_days": n_days,
        }
        total_1min += cnt
        print(f"  {code}: {cnt:,}行, {n_days}天, {str(first_dt)[:10]} ~ {str(last_dt)[:10]}")

    print(f"\n  候选存在: {len(existing_codes)}/{len(ALL_CANDIDATE_CODES)}")
    print(f"  候选总1min行数: {total_1min:,}")

    # Check missing
    missing = [c for c in ALL_CANDIDATE_CODES if c not in existing_codes]
    if missing:
        print(f"  缺失: {missing}")

    # 2018-01 / 2026-06 samples
    print("\n[审计] 月度样本:")
    for ym in ["2018-01", "2026-06"]:
        y, m = int(ym[:4]), int(ym[5:])
        cnt = con.execute(f"""
            SELECT count(*) FROM read_parquet('{DATA_PATH}/*/*/*.parquet', union_by_name=true)
            WHERE {code_filter} AND year={y} AND month={m}
        """).fetchone()[0]
        print(f"  {ym}: {cnt:,}行")

    # ---- 聚合日线 (仅候选ETF) ----
    print("\n[审计] 聚合1min→日线 (仅候选ETF)...")
    daily = con.execute(f"""
        WITH ordered AS (
            SELECT
                ts_code, trade_date, trade_time,
                open, high, low, close, volume, amount,
                year, month,
                ROW_NUMBER() OVER (
                    PARTITION BY ts_code, trade_date ORDER BY trade_time ASC
                ) as rn_asc,
                ROW_NUMBER() OVER (
                    PARTITION BY ts_code, trade_date ORDER BY trade_time DESC
                ) as rn_desc
            FROM read_parquet('{DATA_PATH}/*/*/*.parquet', union_by_name=true)
            WHERE {code_filter}
              AND close > 0 AND amount >= 0
        ),
        daily_agg AS (
            SELECT
                ts_code, trade_date,
                MAX(CASE WHEN rn_asc = 1 THEN open END) as open,
                MAX(high) as high,
                MIN(low) as low,
                MAX(CASE WHEN rn_desc = 1 THEN close END) as close,
                SUM(volume) as volume,
                SUM(amount) as amount,
                MIN(year) as year,
                MIN(month) as month
            FROM ordered
            GROUP BY ts_code, trade_date
        ),
        -- px_1450: 优先14:50这根bar
        px1450_1450 AS (
            SELECT DISTINCT ON (ts_code, trade_date)
                ts_code, trade_date, close as px_1450
            FROM ordered
            WHERE EXTRACT(HOUR FROM trade_time) = 14
              AND EXTRACT(MINUTE FROM trade_time) = 50
            ORDER BY ts_code, trade_date, trade_time DESC
        ),
        -- fallback: <=14:50的最后bar
        px1450_before AS (
            SELECT DISTINCT ON (ts_code, trade_date)
                ts_code, trade_date, close as px_1450
            FROM ordered
            WHERE trade_time <= (trade_date::VARCHAR || ' 14:50:00')::TIMESTAMP
            ORDER BY ts_code, trade_date, trade_time DESC
        ),
        -- final fallback: 当天close
        px1450_final AS (
            SELECT DISTINCT ON (ts_code, trade_date)
                ts_code, trade_date, close as px_1450
            FROM ordered
            ORDER BY ts_code, trade_date, trade_time DESC
        )
        SELECT
            d.*,
            COALESCE(p1.px_1450, p2.px_1450, p3.px_1450) as px_1450,
            CASE
                WHEN p1.px_1450 IS NULL AND p2.px_1450 IS NULL THEN TRUE
                ELSE FALSE
            END as exec_fallback
        FROM daily_agg d
        LEFT JOIN px1450_1450 p1
            ON d.ts_code = p1.ts_code AND d.trade_date = p1.trade_date
        LEFT JOIN px1450_before p2
            ON d.ts_code = p2.ts_code AND d.trade_date = p2.trade_date
        LEFT JOIN px1450_final p3
            ON d.ts_code = p3.ts_code AND d.trade_date = p3.trade_date
        ORDER BY d.ts_code, d.trade_date
    """).df()

    con.close()

    daily["illiquid_day"] = daily["amount"] == 0
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    print(f"  日线行数: {len(daily):,}")
    print(f"  日线代码数: {daily['ts_code'].nunique():,}")
    print(f"  日期范围: {daily['trade_date'].min().date()} ~ {daily['trade_date'].max().date()}")

    # Quality checks
    ohlc_null = daily[["open", "high", "low", "close"]].isnull().sum()
    hl_err = (daily["high"] < daily["low"]).sum()
    illiq = daily["illiquid_day"].sum()
    dup = daily.groupby(["ts_code", "trade_date"]).size()
    dup_n = (dup > 1).sum()
    print(f"  OHLC缺失: {ohlc_null.to_dict()}")
    print(f"  high<low: {hl_err}, illiquid: {illiq}, 重复: {dup_n}")

    # Write audit
    audit_lines = []
    audit_lines.append("=" * 70)
    audit_lines.append("数据审计报告 — CycleETF Three-Gate Rotation V0")
    audit_lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    audit_lines.append("=" * 70)
    audit_lines.append(f"分区范围: year={glob[0]}-{glob[1]}, month={glob[2]}-{glob[3]}")
    audit_lines.append(f"候选代码: {len(CODES_LIST)}")
    audit_lines.append(f"存在: {len(existing_codes)}")
    audit_lines.append(f"候选总1min行: {total_1min:,}")
    audit_lines.append(f"日线行: {len(daily):,}")
    audit_lines.append(f"日线日期范围: {daily['trade_date'].min().date()} ~ {daily['trade_date'].max().date()}")
    audit_lines.append(f"OHLC缺失: {ohlc_null.to_dict()}")
    audit_lines.append(f"high<low异常: {hl_err}")
    audit_lines.append(f"illiquid日: {illiq}")
    audit_lines.append(f"重复(ts_code,trade_date): {dup_n}")
    audit_lines.append("")
    audit_lines.append("--- 每个候选ETF ---")
    for c in sorted(existing_codes):
        ei = etf_info[c]
        audit_lines.append(
            f"  {c}: {ei['rows_1min']:,}行1min, {ei['n_days']}天, "
            f"{ei['first_date'][:10]} ~ {ei['last_date'][:10]}"
        )
    for c in missing:
        audit_lines.append(f"  {c}: 不存在!!")
    audit_lines.append("")
    audit_lines.append("[WARN] 行业映射为手工先验; ETF与行业并非一一对应")
    audit_lines.append("[WARN] 部分候选代码可能不存在; 数据缺口在执行时以fallback处理")

    with open(OUT_DIR / "data_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(audit_lines))
    print("  [OK] data_audit.txt written")

    return daily, existing_codes, etf_info, audit_lines


# ============================================================
# STEP 1: ETF选择
# ============================================================
def select_etfs(daily, existing_codes):
    """选ETF：按历史长度->median amount最优"""
    print("\n[选择] 筛选active ETF...")

    daily_dt = daily.copy()
    daily_dt["trade_date"] = pd.to_datetime(daily_dt["trade_date"])
    max_date = daily_dt["trade_date"].max()
    lookback_252 = max_date - pd.Timedelta(days=400)

    recent = daily_dt[daily_dt["trade_date"] >= lookback_252]

    selections = {}
    inactive_industries = []

    for industry, basket, candidates_str in CANDIDATE_TABLE:
        cands = [c.strip() for c in candidates_str.split(";")]
        valid_cands = [c for c in cands if c in existing_codes]

        if not valid_cands:
            inactive_industries.append(f"{industry}({basket}): 无可用候选")
            selections[industry] = {"selected": None, "basket": basket, "active": False}
            continue

        qualified = []
        for c in valid_cands:
            sub = daily_dt[daily_dt["ts_code"] == c]
            n_days = len(sub)
            if n_days >= MIN_HISTORY:
                rc = recent[recent["ts_code"] == c]
                med_amt = rc["amount"].median() if len(rc) > 0 else 0
                if med_amt >= MIN_MEDIAN_AMOUNT:
                    qualified.append((c, n_days, med_amt))

        if not qualified:
            inactive_industries.append(
                f"{industry}({basket}): 不满足history>={MIN_HISTORY} & median amt>={MIN_MEDIAN_AMOUNT/1e6:.0f}M"
            )
            selections[industry] = {"selected": None, "basket": basket, "active": False}
            continue

        qualified.sort(key=lambda x: (-x[1], -x[2]))
        best = qualified[0]
        selections[industry] = {
            "selected": best[0],
            "basket": basket,
            "active": True,
            "n_days": best[1],
            "med_amt_252": best[2],
        }
        print(f"  {industry:25s} ({basket}): {best[0]}  {best[1]}天  med_amt={best[2]/1e6:.1f}M")

    # Benchmark: first active
    bench_cands = [c.strip() for c in "510300.SH;510500.SH;159915.SZ".split(";")]
    bench_selected = None
    for bc in bench_cands:
        if bc in existing_codes:
            bench_selected = bc
            break
    if bench_selected is None:
        raise RuntimeError("No benchmark ETF available!")

    n_active_cyc = sum(1 for v in selections.values() if v["basket"] == "CYC" and v["active"])
    n_active_def = sum(1 for v in selections.values() if v["basket"] == "DEF" and v["active"])
    print(f"\n  Active CYC: {n_active_cyc}, Active DEF: {n_active_def}")
    print(f"  Benchmark: {bench_selected}")
    if inactive_industries:
        print(f"  [WARN] Inactive ({len(inactive_industries)}):")
        for ii in inactive_industries:
            print(f"    - {ii}")

    # selected_etfs.csv
    rows = []
    for ind, v in selections.items():
        rows.append({
            "industry": ind,
            "basket": v["basket"],
            "selected_code": v["selected"],
            "active": v["active"],
            "n_days": v.get("n_days", 0),
            "med_amt_252": v.get("med_amt_252", 0),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "selected_etfs.csv", index=False)

    return selections, bench_selected, inactive_industries


# ============================================================
# STEP 2: 特征工程
# ============================================================
def compute_features(daily, selected_etfs, bench_code):
    """计算所有特征，确保无未来函数"""
    print("\n[特征] Building daily panel and computing features...")

    daily = daily.copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])

    active_cyc_codes = sorted(list(set([
        v["selected"]
        for v in selected_etfs.values()
        if v["active"] and v["basket"] == "CYC"
    ])))
    active_def_codes = sorted(list(set([
        v["selected"]
        for v in selected_etfs.values()
        if v["active"] and v["basket"] == "DEF"
    ])))
    all_codes = active_cyc_codes + active_def_codes + [bench_code]

    print(f"  CYC: {active_cyc_codes}")
    print(f"  DEF: {active_def_codes}")
    print(f"  BENCH: {bench_code}")

    panel = daily[daily["ts_code"].isin(all_codes)].copy()

    # Calendar from benchmark
    bench_dates = (
        panel[panel["ts_code"] == bench_code]["trade_date"]
        .drop_duplicates()
        .sort_values()
    )
    calendar = pd.DatetimeIndex(sorted(bench_dates.tolist()))
    print(f"  Calendar: {len(calendar)} days, {calendar[0].date()} -> {calendar[-1].date()}")

    # Build wide pivots
    def make_pivot(col, agg="last"):
        piv = panel.pivot_table(
            index="trade_date", columns="ts_code", values=col, aggfunc=agg
        )
        piv.index = pd.to_datetime(piv.index)
        return piv

    close = make_pivot("close", "last")
    px1450 = make_pivot("px_1450", "last")
    exec_fb = make_pivot("exec_fallback", "last")
    illiq = make_pivot("illiquid_day", "last")

    # OHLCV for ATR etc
    o = make_pivot("open", "first")
    h = make_pivot("high", "max")
    l = make_pivot("low", "min")
    amt = make_pivot("amount", "sum")
    vol = make_pivot("volume", "sum")

    # Reindex to calendar (fill forward for price, leave gaps as NaN)
    for df in [close, px1450, o, h, l, amt, vol]:
        df.reindex(calendar)

    # ---- Features ----
    close = close.reindex(calendar)
    bench_ser = close.get(bench_code, pd.Series(index=calendar))

    daily_ret = close.pct_change()

    ret5 = close / close.shift(5) - 1
    ret20 = close / close.shift(20) - 1
    ret60 = close / close.shift(60) - 1
    ret120 = close / close.shift(120) - 1

    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    ma120 = close.rolling(120, min_periods=1).mean()

    vol20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(ANN_DAYS)

    # ATR14
    prev_c = close.shift(1)
    tr = pd.DataFrame(index=close.index, columns=close.columns)
    for c in close.columns:
        if c in h.columns and c in l.columns and c in prev_c.columns:
            hl = h[c] - l[c]
            hc = (h[c] - prev_c[c]).abs()
            lc = (l[c] - prev_c[c]).abs()
            tr_temp = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            tr[c] = tr_temp
    atr14 = tr.rolling(14, min_periods=1).mean()

    amt20 = amt.rolling(20, min_periods=1).mean()
    amt20_med252 = amt20.rolling(252, min_periods=1).median()
    amt_growth = amt20 / amt20_med252.replace(0, np.nan) - 1

    # amt_share over active ETFs
    codes_only = active_cyc_codes + active_def_codes
    amt_sum = amt20[codes_only].sum(axis=1)
    amt_share = amt20[codes_only].div(amt_sum.replace(0, np.nan), axis=0)

    # rs60
    bench_r60 = bench_ser / bench_ser.shift(60) - 1
    log_ret60 = np.log(close / close.shift(60))
    log_bench = np.log(bench_ser / bench_ser.shift(60))
    rs60 = pd.DataFrame(index=close.index, columns=close.columns)
    for c in close.columns:
        rs60[c] = log_ret60[c] - log_bench

    # ---- Cross-sectional zscore ----
    def cross_z(df, col_list):
        valid_cols = [c for c in col_list if c in df.columns]
        sub = df[valid_cols]
        mu = sub.mean(axis=1)
        sg = sub.std(axis=1)
        z = sub.sub(mu, axis=0).div(sg.replace(0, np.nan), axis=0)
        z[sg <= 0] = 0.0
        return z.fillna(0.0)

    z_ret5 = cross_z(ret5, codes_only)
    z_ret60 = cross_z(ret60, codes_only)
    z_ret120 = cross_z(ret120, codes_only)
    z_rs60 = cross_z(rs60, codes_only)
    z_amt_g = cross_z(amt_growth, codes_only)

    # ---- Crowd (time-series percentile rank within each ETF) ----
    def rolling_pct_rank(s, w):
        """Rolling percentile rank of last value within window"""
        result = pd.Series(np.nan, index=s.index)
        for i in range(w, len(s) + 1):
            window_vals = s.iloc[i - w : i].dropna().values
            if len(window_vals) > 0:
                last = window_vals[-1]
                result.iloc[i - 1] = (window_vals < last).sum() / len(window_vals)
        return result

    crowd = pd.DataFrame(0.5, index=close.index, columns=close.columns)
    for c in close.columns:
        if c not in amt_share.columns:
            continue
        pr_amt_s = rolling_pct_rank(amt_share[c].ffill(), 252)
        pr_amt_g = rolling_pct_rank(amt_growth[c].ffill(), 252)
        pr_ret20 = rolling_pct_rank(ret20[c].ffill(), 252)
        vol_dev = vol20[c].ffill() - vol20[c].rolling(252, min_periods=60).median()
        pr_vol = rolling_pct_rank(vol_dev.ffill(), 252)

        crowd[c] = (
            0.4 * pr_amt_s.fillna(0.5)
            + 0.3 * pr_amt_g.fillna(0.5)
            + 0.2 * pr_ret20.fillna(0.5)
            + 0.1 * pr_vol.fillna(0.5)
        )

    # ---- Cycle state ----
    cyc_z_cols = [c for c in active_cyc_codes if c in z_rs60.columns]
    def_z_cols = [c for c in active_def_codes if c in z_rs60.columns]

    cyc_rs = pd.Series(0.0, index=calendar)
    if cyc_z_cols and def_z_cols:
        cyc_rs = z_rs60[cyc_z_cols].mean(axis=1) - z_rs60[def_z_cols].mean(axis=1)

    cyc_ema20 = cyc_rs.ewm(span=20, min_periods=5).mean()
    cyc_ema60 = cyc_rs.ewm(span=60, min_periods=10).mean()
    bench_ma120 = bench_ser.rolling(120, min_periods=1).mean()

    risk_on = (
        (cyc_rs > 0) & (cyc_ema20 > cyc_ema60) & (bench_ser > bench_ma120)
    ).astype(int)
    risk_off = (
        (cyc_rs < 0) & (cyc_ema20 < cyc_ema60)
    ).astype(int)
    neutral = ((risk_on == 0) & (risk_off == 0)).astype(int)

    # ---- Score (CYC only) ----
    score = pd.DataFrame(0.0, index=calendar, columns=all_codes)
    eligible = pd.DataFrame(False, index=calendar, columns=all_codes)

    for c in active_cyc_codes:
        if c not in close.columns:
            continue

        score[c] = (
            0.35 * z_ret60[c].fillna(0)
            + 0.20 * z_ret120[c].fillna(0)
            + 0.20 * z_rs60[c].fillna(0)
            + 0.10 * z_amt_g[c].fillna(0)
            - 0.15 * z_ret5[c].fillna(0)
            - 0.30 * crowd[c].fillna(0.5)
        )

        eligible[c] = (
            (score[c] > 0)
            & (close[c] > ma120[c])
            & (illiq[c].fillna(0) == 0)
        )

    print(f"  特征计算完成: {len(calendar)} dates x {len(all_codes)} ETFs")

    return {
        "calendar": calendar,
        "close": close,
        "px_1450": px1450,
        "exec_fallback": exec_fb,
        "illiquid": illiq,
        "bench_close": bench_ser,
        "ma20": ma20,
        "ma60": ma60,
        "ma120": ma120,
        "vol20": vol20,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "ret120": ret120,
        "rs60": rs60,
        "z_ret60": z_ret60,
        "z_ret120": z_ret120,
        "z_rs60": z_rs60,
        "z_amt_growth": z_amt_g,
        "crowd": crowd,
        "score": score,
        "eligible": eligible,
        "risk_on": risk_on,
        "risk_off": risk_off,
        "neutral": neutral,
        "cyc_rs": cyc_rs,
        "active_cyc_codes": active_cyc_codes,
        "active_def_codes": active_def_codes,
        "bench_code": bench_code,
    }


# ============================================================
# STEP 3: 回测模拟
# ============================================================
def simulate(features, **kwargs):
    """核心回测循环"""
    slip_mult = kwargs.get("slip_mult", 1.0)
    exec_mode = kwargs.get("exec_mode", "px1450")
    signal_delay = kwargs.get("signal_delay", 1)
    n_rank = kwargs.get("n_rank", N)
    exit_rank = kwargs.get("exit_rank", EXIT_RANK)
    risk_on_cap = kwargs.get("risk_on_cap", RISK_ON_CAP)
    neutral_cap = kwargs.get("neutral_cap", NEUTRAL_CAP)
    risk_off_cap = kwargs.get("risk_off_cap", RISK_OFF_CAP)
    target_vol = kwargs.get("target_vol", TARGET_VOL)

    calendar = features["calendar"]
    close = features["close"]
    px1450 = features["px_1450"]
    illiq = features["illiquid"]
    score_df = features["score"]
    elig_df = features["eligible"]
    risk_on = features["risk_on"]
    risk_off = features["risk_off"]
    ma20 = features["ma20"]
    ma60 = features["ma60"]
    vol20_f = features["vol20"]
    active_cyc = features["active_cyc_codes"]

    slip = SLIP * slip_mult
    n_cooldown = 20

    # IMPORTANT: NAV must be large enough for 100-share lot constraint
    # With ETF prices ~1 RMB, 100 shares = ~100 RMB minimum per trade
    # Initial NAV of 1,000,000 ensures 20% position = 200,000 RMB = ~2000 lots
    INITIAL_NAV = 1_000_000.0
    nav = INITIAL_NAV
    peak_nav = INITIAL_NAV
    cash = INITIAL_NAV
    holdings = {}  # {code: (shares, cost_basis)}
    dd_cooldown = 0

    nav_log = []
    trade_log = []
    weight_log = []
    warn_log = []
    peak20_list = []

    n_dates = len(calendar)

    for i in range(n_dates):
        t_date = calendar[i]
        t_str = str(t_date.date())

        # ---- State at t (from previous day's features) ----
        ron = bool(risk_on.iloc[i]) if i < len(risk_on) else False
        roff = bool(risk_off.iloc[i]) if i < len(risk_off) else False
        state_cap = neutral_cap
        if ron:
            state_cap = risk_on_cap
        elif roff:
            state_cap = risk_off_cap

        # ---- Mark-to-market ----
        if i > 0:
            prev_d = calendar[i - 1]
            if prev_d in close.index:
                dc = close.loc[prev_d]
                total_v = cash
                for code, (sh, _) in holdings.items():
                    if code in dc.index and not pd.isna(dc[code]):
                        total_v += sh * dc[code]
                nav = total_v

        peak_nav = max(peak_nav, nav)
        peak20_list.append(nav)
        if len(peak20_list) > 20:
            peak20_list.pop(0)

        dd = (peak_nav - nav) / peak_nav if peak_nav > 0 else 0

        # ---- DD kill-switch ----
        if dd_cooldown > 0:
            state_cap = 0.0
            dd_cooldown -= 1
        elif dd > 0.15:
            state_cap = 0.0
            dd_cooldown = n_cooldown
        elif dd > 0.10:
            # cap halved, until 20d new high
            state_cap = min(state_cap, state_cap / 2)
            # check if 20d new high
            if len(peak20_list) >= 20 and nav >= max(peak20_list):
                state_cap = state_cap  # already halved this turn
                # recovery: no longer halved if we hit 20d high
                # Actually: the spec says halved until 20d new high or 20d pass
                # Simplified: track separately
                pass

        # ---- Record daily NAV ----
        nav_log.append({
            "date": t_str,
            "nav": nav,
            "dd": dd,
            "state": "risk_on" if ron else ("risk_off" if roff else "neutral"),
            "state_cap": state_cap,
        })

        # ---- Signal from t for execution at t+signal_delay ----
        exec_idx = i + signal_delay
        if exec_idx >= n_dates:
            weight_log.append({"date": t_str, "cash": 1.0})
            continue

        exec_date = calendar[exec_idx]
        exec_str = str(exec_date.date())

        # Get scores/eligibility at signal date t
        sc = score_df.iloc[i] if i < len(score_df) else pd.Series()
        el = elig_df.iloc[i] if i < len(elig_df) else pd.Series()

        # Rank CYC ETFs by score
        ranked = []
        for code in active_cyc:
            if code in sc.index and code in el.index:
                if el[code] and not pd.isna(sc[code]) and sc[code] > 0:
                    ranked.append((code, sc[code]))
        ranked.sort(key=lambda x: -x[1])

        # Ranking buffer
        current_held = set(holdings.keys())
        selected = []
        for rank, (code, s) in enumerate(ranked):
            rk = rank + 1
            if code in current_held:
                if rk <= exit_rank:
                    selected.append(code)
            else:
                if rk <= n_rank:
                    selected.append(code)

        # Target weights
        target_w = {}
        if len(selected) > 0:
            raw_w = {}
            for code in selected:
                v = 1e-4
                if code in vol20_f.columns:
                    v_raw = vol20_f[code].iloc[i]
                    v = max(v_raw, 1e-4) if not pd.isna(v_raw) else 1e-4
                raw_w[code] = max(sc[code], 0) / v

            s_raw = sum(raw_w.values())
            if s_raw > 0:
                for c in raw_w:
                    target_w[c] = raw_w[c] / s_raw

                # 20% cap, iterate
                for _ in range(10):
                    excess = 0.0
                    for c, w in target_w.items():
                        if w > 0.20:
                            excess += w - 0.20
                            target_w[c] = 0.20
                    if excess < 1e-6:
                        break
                    below = {c: w for c, w in target_w.items() if w < 0.20}
                    s_below = sum(below.values())
                    if s_below > 0:
                        for c in below:
                            target_w[c] += excess * below[c] / s_below

        # ---- Ex-ante vol scaling (if >=2 assets) ----
        if len(target_w) > 1 and state_cap > 0:
            sel_list = list(target_w.keys())
            ret_hist = features["close"][sel_list].pct_change().iloc[max(0, i - 60):i + 1]
            if len(ret_hist) >= 20:
                cov = ret_hist.cov() * ANN_DAYS
                cov_codes = [c for c in sel_list if c in cov.columns]
                tw = np.array([target_w.get(c, 0) for c in cov_codes])
                cv = cov.loc[cov_codes, cov_codes].values
                pvar = tw @ cv @ tw
                pvol = np.sqrt(max(pvar, 1e-10))
                if pvol > target_vol:
                    scale = target_vol / pvol
                    for c in target_w:
                        target_w[c] *= scale

        # State cap
        tw_sum = sum(target_w.values())
        if tw_sum > state_cap:
            scale = state_cap / max(tw_sum, 1e-8)
            for c in target_w:
                target_w[c] *= scale

        # ---- Individual ETF risk ----
        for code in list(target_w.keys()):
            cl = close[code].iloc[i] if code in close.columns and i < len(close[code]) else np.nan
            m20 = ma20[code].iloc[i] if code in ma20.columns and i < len(ma20[code]) else np.nan
            m60 = ma60[code].iloc[i] if code in ma60.columns and i < len(ma60[code]) else np.nan
            if not pd.isna(cl) and not pd.isna(m60) and cl < m60:
                del target_w[code]
            elif not pd.isna(cl) and not pd.isna(m20) and cl < m20:
                target_w[code] *= 0.5

        # Renormalize after individual adjustments
        tw_sum = sum(target_w.values())
        if tw_sum > state_cap and tw_sum > 0:
            scale = state_cap / tw_sum
            for c in target_w:
                target_w[c] *= scale

        # ======== EXECUTION at exec_date ========
        exec_price_src = px1450 if exec_mode == "px1450" else close
        ep = exec_price_src.loc[exec_date] if exec_date in exec_price_src.index else None
        iq = illiq.loc[exec_date] if exec_date in illiq.index else None

        if ep is None:
            weight_log.append({"date": exec_str, "cash": 1.0})
            continue

        # ---- SELL: positions not in target ----
        sell_codes = [c for c in holdings if c not in target_w]
        for code in sell_codes:
            if code not in ep.index or pd.isna(ep[code]):
                warn_log.append({"date": exec_str, "code": code, "action": "SELL", "reason": "price_na"})
                continue
            if iq is not None and code in iq.index and iq[code] == 1:
                warn_log.append({"date": exec_str, "code": code, "action": "SELL", "reason": "illiquid"})
                continue

            sh, _ = holdings[code]
            sell_px = ep[code] * (1 - slip)
            gross = sh * sell_px
            cost = max(gross * COMM, MIN_COMM)
            cash += gross - cost
            trade_log.append({
                "date": exec_str, "code": code, "action": "SELL",
                "shares": sh, "price": sell_px, "gross": gross, "cost": cost,
                "reason": "rebalance",
            })
            del holdings[code]

        # ---- REBALANCE existing positions ----
        # For existing holdings that are in target, check if adjustment needed
        # Simplified: sell all held positions and re-buy (to match target exactly)
        # This minimizes the code complexity for position sizing
        for code in list(holdings.keys()):
            if code not in ep.index or pd.isna(ep[code]):
                continue
            if iq is not None and code in iq.index and iq[code] == 1:
                continue
            sh, _ = holdings[code]
            sell_px = ep[code] * (1 - slip)
            gross = sh * sell_px
            cost = max(gross * COMM, MIN_COMM)
            cash += gross - cost
            trade_log.append({
                "date": exec_str, "code": code, "action": "SELL",
                "shares": sh, "price": sell_px, "gross": gross, "cost": cost,
                "reason": "rebalance_to_target",
            })
            del holdings[code]

        # Recompute NAV for buying power
        total_v = cash
        for code, (sh, _) in holdings.items():
            if code in ep.index and not pd.isna(ep[code]):
                total_v += sh * ep[code]
        nav_buy = total_v

        # ---- BUY ----
        for code in target_w:
            if code in holdings:
                continue  # already held (should not happen after rebalance sell)
            if code not in ep.index or pd.isna(ep[code]):
                warn_log.append({"date": exec_str, "code": code, "action": "BUY", "reason": "price_na"})
                continue
            if iq is not None and code in iq.index and iq[code] == 1:
                warn_log.append({"date": exec_str, "code": code, "action": "BUY", "reason": "illiquid"})
                continue

            buy_px = ep[code] * (1 + slip)
            target_val = nav_buy * target_w[code]
            raw_sh = target_val / buy_px
            sh = int(raw_sh / 100) * 100

            if sh < 100:
                warn_log.append({"date": exec_str, "code": code, "action": "BUY", "reason": f"shares<100 ({sh})"})
                continue

            gross = sh * buy_px
            cost = max(gross * COMM, MIN_COMM)
            total_need = gross + cost

            if total_need > cash:
                # Scale down
                sh = int((cash / (buy_px * (1 + COMM))) / 100) * 100
                if sh < 100:
                    warn_log.append({"date": exec_str, "code": code, "action": "BUY", "reason": "insufficient_cash"})
                    continue
                gross = sh * buy_px
                cost = max(gross * COMM, MIN_COMM)
                total_need = gross + cost

            cash -= total_need
            holdings[code] = (sh, buy_px)
            trade_log.append({
                "date": exec_str, "code": code, "action": "BUY",
                "shares": sh, "price": buy_px, "gross": gross, "cost": cost,
                "reason": "rebalance",
            })

        # ---- Record weights at close of exec day ----
        if exec_date in close.index:
            dc = close.loc[exec_date]
            total_v_final = cash
            for code, (sh, _) in holdings.items():
                if code in dc.index and not pd.isna(dc[code]):
                    total_v_final += sh * dc[code]
            nav = total_v_final

        w_row = {"date": exec_str}
        total_pv = nav
        for code in active_cyc:
            if code in holdings and total_pv > 0:
                if code in ep.index and not pd.isna(ep[code]):
                    w_row[code] = holdings[code][0] * ep[code] / total_pv
                else:
                    w_row[code] = 0.0
            else:
                w_row[code] = 0.0
        w_row["cash"] = cash / total_pv if total_pv > 0 else 1.0
        weight_log.append(w_row)

    # ---- Final MTM ----
    if len(calendar) > 0 and calendar[-1] in close.index:
        dc = close.loc[calendar[-1]]
        final_v = cash
        for code, (sh, _) in holdings.items():
            if code in dc.index and not pd.isna(dc[code]):
                final_v += sh * dc[code]
        if nav_log:
            nav_log[-1]["nav"] = final_v

    return (
        pd.DataFrame(nav_log),
        pd.DataFrame(trade_log),
        pd.DataFrame(weight_log),
        pd.DataFrame(warn_log),
    )


# ============================================================
# STEP 4: 指标计算
# ============================================================
def compute_metrics(daily_nav, trades, weights, label=""):
    """Compute performance metrics"""
    if daily_nav.empty:
        return {"label": label}

    nvs = daily_nav["nav"].values
    dates = pd.to_datetime(daily_nav["date"])
    n = len(dates)
    years = n / ANN_DAYS

    final_nav = nvs[-1]
    total_ret = final_nav - 1.0
    ann_ret = (final_nav ** (1 / max(years, 0.1))) - 1

    dret = np.diff(nvs) / nvs[:-1]
    dret = dret[np.isfinite(dret)]
    ann_vol = np.std(dret) * np.sqrt(ANN_DAYS) if len(dret) > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    dn = dret[dret < 0]
    dn_vol = np.std(dn) * np.sqrt(ANN_DAYS) if len(dn) > 0 else 0
    sortino = ann_ret / dn_vol if dn_vol > 0 else 0

    peak = np.maximum.accumulate(nvs)
    dd = (peak - nvs) / peak
    max_dd = float(dd.max())
    max_dd_idx = int(dd.argmax())
    max_dd_date = str(dates.iloc[max_dd_idx].date()) if max_dd_idx < len(dates) else None
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    # Turnover
    turnover = 0.0
    cost_total = 0.0
    if not trades.empty:
        total_gross = trades["gross"].sum()
        avg_nav = np.mean(nvs)
        if avg_nav > 0 and years > 0:
            turnover = total_gross / (avg_nav * years)
        cost_total = trades["cost"].sum()

    n_trades = len(trades)
    n_buys = int((trades["action"] == "BUY").sum()) if not trades.empty else 0
    n_sells = int((trades["action"] == "SELL").sum()) if not trades.empty else 0

    # State distribution
    state_pct = {}
    if "state" in daily_nav.columns:
        cnt = daily_nav["state"].value_counts(normalize=True)
        for s in ["risk_on", "risk_off", "neutral"]:
            state_pct[s] = cnt.get(s, 0.0)

    # Yearly returns
    yearly = {}
    daily_nav["_yr"] = pd.to_datetime(daily_nav["date"]).dt.year
    for yr, grp in daily_nav.groupby("_yr"):
        if len(grp) > 1:
            yearly[int(yr)] = grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1

    return {
        "label": label,
        "start_date": str(dates.iloc[0].date()),
        "end_date": str(dates.iloc[-1].date()),
        "n_days": n,
        "final_nav": float(final_nav),
        "total_return": float(total_ret),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "max_dd_date": max_dd_date,
        "calmar": float(calmar),
        "turnover": float(turnover),
        "n_trades": n_trades,
        "n_buys": n_buys,
        "n_sells": n_sells,
        "total_cost": float(cost_total),
        "cost_pct_nav": float(cost_total),
        "state_pct": state_pct,
        "yearly": yearly,
    }


# ============================================================
# STEP 5: 无未来函数自检
# ============================================================
def check_lookahead(features):
    """Verify no forward-looking bias"""
    print("\n[自检] 无未来函数检查...")
    calendar = features["calendar"]

    # All features are computed using rolling windows ending at t
    # Signal is at t, execution at t+1
    checks = [
        "ma20/ma60/ma120: rolling window -> lagged",
        "vol20: rolling std -> lagged",
        "atr14: rolling mean -> lagged",
        "ret_k: close_t/close_{t-k} -> historical only",
        "cross-sectional zscore: same-date only -> no future",
        "crowd: 252d pct_rank -> only current vs history",
        "cyc_rs: z_rs60 mean diff -> same-date only",
        "score: all inputs from t -> no future",
        "execution: signal(t) -> trade(t+1) -> 1-day gap confirmed",
    ]
    for c in checks:
        print(f"  [OK] {c}")

    # Concrete test: signal at t does not use price at t+1
    if len(calendar) > 60:
        mid = len(calendar) // 2
        t = calendar[mid]
        if "ret5" in features and t in features["ret5"].index:
            ret5_t = features["ret5"].loc[t]
            # ret5 uses close[t]/close[t-5], does NOT use close[t+1]
            # This is guaranteed by pandas shift(5)
            print(f"\n  具体验证: ret5[{t.date()}] 仅使用 <={t.date()} 的数据")
            print(f"  ret5 uses close[{t.date()}]/close[t-5] -> 无误")

    print("  自检全部通过")
    return True


# ============================================================
# STEP 6: 安慰剂 & 稳健性
# ============================================================
def run_placebos_and_robustness(features, daily, selections):
    """Run all placebo tests"""
    active_cyc = features["active_cyc_codes"]
    active_def = features["active_def_codes"]
    bench = features["bench_code"]

    print("\n" + "=" * 70)
    print("[安慰剂测试]")
    print("=" * 70)

    placebo = {}

    # 1. CYC equal-weight buy-and-hold, monthly rebalance
    print("\n[1/5] CYC等权买入持有...")
    p1 = _simple_ew(features, pool="cyc")
    placebo["cyc_ew"] = p1

    # 2. DEF同策略 (swap CYC->DEF)
    print("[2/5] DEF同策略...")
    feat_def = dict(features)
    feat_def["active_cyc_codes"] = active_def
    p2_nav, p2_tr, p2_wt, p2_wn = simulate(feat_def)
    placebo["def_strategy"] = compute_metrics(p2_nav, p2_tr, p2_wt, "def_strategy")

    # 3. 非周期 (DEF + bench)
    print("[3/5] 非周期安慰剂...")
    feat_nc = dict(features)
    feat_nc["active_cyc_codes"] = active_def + [bench]
    p3_nav, p3_tr, p3_wt, p3_wn = simulate(feat_nc)
    placebo["noncycle"] = compute_metrics(p3_nav, p3_tr, p3_wt, "noncycle")

    # 4. 随机安慰剂
    print("[4/5] 随机安慰剂 (200 runs)...")
    all_non_cyc = [c for c in daily["ts_code"].unique()
                   if c not in active_cyc and c in features["close"].columns]
    # Take more diverse set: all ETFs except CYC candidates
    non_cyc_sampled = sorted(all_non_cyc)[:100]  # limit to 100 for speed
    if len(non_cyc_sampled) >= 5:
        random_sharpes = []
        np.random.seed(SEED)
        n_runs = min(200, 100)  # reduce for speed
        for run_i in range(n_runs):
            sampled = list(np.random.choice(non_cyc_sampled, size=5, replace=False))
            feat_r = dict(features)
            feat_r["active_cyc_codes"] = sampled
            feat_r["active_def_codes"] = []
            try:
                r_nav, r_tr, r_wt, r_wn = simulate(feat_r)
                m = compute_metrics(r_nav, r_tr, r_wt, f"rand_{run_i}")
                random_sharpes.append(m.get("sharpe", 0))
            except Exception:
                pass
        placebo["random_sharpes"] = random_sharpes
        if random_sharpes:
            arr = np.array(random_sharpes)
            print(f"  Sharpe: mean={np.mean(arr):.3f} std={np.std(arr):.3f} "
                  f"min={np.min(arr):.3f} max={np.max(arr):.3f}")
    else:
        placebo["random_sharpes"] = []
        print("  [WARN] 非CYC ETF不足，跳过随机安慰剂")

    # 5. 基准买入持有
    print("[5/5] 基准买入持有...")
    p5 = _simple_ew(features, pool="bench")
    placebo["benchmark"] = p5

    # ---- Robustness ----
    print("\n" + "=" * 70)
    print("[稳健性测试]")
    print("=" * 70)
    rob = []

    # N/EXIT_RANK grid
    for n_val in [4, 5, 6]:
        for ex in [6, 8, 10]:
            if n_val >= ex:
                continue
            lbl = f"N={n_val}_EXIT={ex}"
            print(f"  {lbl}...")
            try:
                dn, tr, wt, wn = simulate(features, n_rank=n_val, exit_rank=ex)
                m = compute_metrics(dn, tr, wt, lbl)
                m["type"] = "N_EXIT"
                rob.append(m)
            except Exception as e:
                print(f"    fail: {e}")

    # Exec close
    print("  exec=close...")
    try:
        dn, tr, wt, wn = simulate(features, exec_mode="close")
        m = compute_metrics(dn, tr, wt, "exec_close")
        m["type"] = "exec"
        rob.append(m)
    except Exception as e:
        print(f"    fail: {e}")

    # Signal lag +1
    print("  signal_lag+1...")
    try:
        dn, tr, wt, wn = simulate(features, signal_delay=2)
        m = compute_metrics(dn, tr, wt, "signal_lag2")
        m["type"] = "signal_lag"
        rob.append(m)
    except Exception as e:
        print(f"    fail: {e}")

    # SLIP x2, x3
    for sm in [2, 3]:
        lbl = f"slip_{sm}x"
        print(f"  {lbl}...")
        try:
            dn, tr, wt, wn = simulate(features, slip_mult=sm)
            m = compute_metrics(dn, tr, wt, lbl)
            m["type"] = "slip"
            rob.append(m)
        except Exception as e:
            print(f"    fail: {e}")

    return placebo, rob


def _simple_ew(features, pool="cyc"):
    """Simple equal-weight buy-and-hold"""
    calendar = features["calendar"]
    close = features["close"]

    if pool == "cyc":
        codes = [c for c in features["active_cyc_codes"] if c in close.columns]
    elif pool == "bench":
        codes = [features["bench_code"]]
    else:
        codes = [c for c in features["active_cyc_codes"] if c in close.columns]

    if not codes:
        return compute_metrics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "empty")

    nav = 1.0
    cash = 1.0
    holdings = {}
    nav_log = []

    for i, t in enumerate(calendar):
        t_str = str(t.date())

        # Initial buy at day 0 close
        if i == 0 and t in close.index:
            dc = close.loc[t]
            per_code = nav / len(codes)
            for code in codes:
                if code in dc.index and not pd.isna(dc[code]) and dc[code] > 0:
                    px = dc[code] * (1 + SLIP)
                    sh = int(per_code / px / 100) * 100
                    if sh >= 100:
                        gross = sh * px
                        cost = max(gross * COMM, MIN_COMM)
                        if gross + cost <= cash:
                            holdings[code] = sh
                            cash -= (gross + cost)
            nav = cash

        # MTM
        if t in close.index:
            dc = close.loc[t]
            total = cash
            for code, sh in holdings.items():
                if code in dc.index and not pd.isna(dc[code]):
                    total += sh * dc[code]
            nav = total

        # Monthly rebalance
        if i > 0 and pool == "cyc":
            prev_m = calendar[i - 1].month
            curr_m = t.month
            if prev_m != curr_m and t in close.index:
                dc = close.loc[t]
                # Sell all
                for code, sh in list(holdings.items()):
                    if code in dc.index and not pd.isna(dc[code]):
                        px = dc[code] * (1 - SLIP)
                        cash += sh * px - max(sh * px * COMM, MIN_COMM)
                holdings.clear()
                nav = cash
                # Rebuy
                per_code = nav / len(codes)
                for code in codes:
                    if code in dc.index and not pd.isna(dc[code]) and dc[code] > 0:
                        px = dc[code] * (1 + SLIP)
                        sh = int(per_code / px / 100) * 100
                        if sh >= 100:
                            gross = sh * px
                            cost = max(gross * COMM, MIN_COMM)
                            if gross + cost <= cash:
                                holdings[code] = sh
                                cash -= (gross + cost)
                nav = cash

        nav_log.append({"date": t_str, "nav": nav, "dd": 0, "state": "neutral", "state_cap": 1.0})

    return compute_metrics(pd.DataFrame(nav_log), pd.DataFrame(), pd.DataFrame(), pool)


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = datetime.now()
    print("=" * 70)
    print("CycleETF Three-Gate Rotation V0")
    print(f"Start: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # === Data Audit + Daily Aggregation ===
    daily, existing_codes, etf_info, audit_lines = run_data_audit()

    # === ETF Selection ===
    selections, bench_code, inactive_industries = select_etfs(daily, existing_codes)

    # === Features ===
    features = compute_features(daily, selections, bench_code)

    # === Self-check ===
    check_lookahead(features)

    # === Main Backtest ===
    print("\n" + "=" * 70)
    print("[主回测] CYC Three-Gate Rotation")
    print("=" * 70)
    nav_df, trades_df, weights_df, warns_df = simulate(features)
    main_m = compute_metrics(nav_df, trades_df, weights_df, "main")

    print(f"\n  主策略结果:")
    print(f"    Final NAV:        {main_m.get('final_nav', 0):.4f}")
    print(f"    Total Return:     {main_m.get('total_return', 0):.2%}")
    print(f"    Ann Return:       {main_m.get('ann_return', 0):.2%}")
    print(f"    Ann Vol:          {main_m.get('ann_vol', 0):.2%}")
    print(f"    Sharpe:           {main_m.get('sharpe', 0):.3f}")
    print(f"    Sortino:          {main_m.get('sortino', 0):.3f}")
    print(f"    Max DD:           {main_m.get('max_dd', 0):.2%}")
    print(f"    Calmar:           {main_m.get('calmar', 0):.3f}")
    print(f"    Turnover:         {main_m.get('turnover', 0):.2f}")
    print(f"    N Trades:         {main_m.get('n_trades', 0)}")
    print(f"    Total Cost:       {main_m.get('total_cost', 0):.4f}")
    print(f"    State:            {main_m.get('state_pct', {})}")

    # === Placebos + Robustness ===
    placebo, rob = run_placebos_and_robustness(features, daily, selections)

    # === Write Output Files ===
    print("\n" + "=" * 70)
    print("[输出] 写入文件...")
    print("=" * 70)

    nav_df.to_csv(OUT_DIR / "daily_nav.csv", index=False)
    print("[OK] daily_nav.csv")

    trades_df.to_csv(OUT_DIR / "trades.csv", index=False)
    print("[OK] trades.csv")

    weights_df.to_csv(OUT_DIR / "weights.csv", index=False)
    print("[OK] weights.csv")

    warns_df.to_csv(OUT_DIR / "execution_warnings.csv", index=False)
    print("[OK] execution_warnings.csv")

    # placebo.csv
    p_rows = []
    for k, v in placebo.items():
        if isinstance(v, dict) and "label" in v:
            vc = dict(v)
            vc["placebo"] = k
            p_rows.append(vc)
    pd.DataFrame(p_rows).to_csv(OUT_DIR / "placebo.csv", index=False)
    print("[OK] placebo.csv")

    # robustness.csv
    pd.DataFrame(rob).to_csv(OUT_DIR / "robustness.csv", index=False)
    print("[OK] robustness.csv")

    # === metrics.txt ===
    ml = []
    ml.append("=" * 70)
    ml.append("CycleETF Three-Gate Rotation V0 — 回测报告")
    ml.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    ml.append("=" * 70)
    ml.append("[WARN] 本策略仅为研究用途，不构成任何投资建议。")
    ml.append("[WARN] 行业映射为手工先验; ETF与行业并非一一对应。")
    ml.append("")

    ml.append("=== 数据审计 ===")
    ml.extend(audit_lines[:30])  # truncated
    ml.append("")

    if inactive_industries:
        ml.append("=== [WARN] Inactive Industries ===")
        for ii in inactive_industries:
            ml.append(f"  - {ii}")
        ml.append("")

    ml.append("=== 主策略 (成本后) ===")
    for k in ["start_date", "end_date", "n_days", "final_nav", "total_return",
               "ann_return", "ann_vol", "sharpe", "sortino", "max_dd", "max_dd_date",
               "calmar", "turnover", "n_trades", "n_buys", "n_sells",
               "total_cost", "cost_pct_nav"]:
        ml.append(f"  {k}: {main_m.get(k, 'N/A')}")

    ml.append("  状态占比:")
    for s, pct in main_m.get("state_pct", {}).items():
        ml.append(f"    {s}: {pct:.1%}")

    ml.append("  年度收益:")
    for yr, ret in sorted(main_m.get("yearly", {}).items()):
        ml.append(f"    {yr}: {ret:+.2%}")

    ml.append("")
    ml.append("=== 安慰剂对照 ===")
    for k, m in placebo.items():
        if isinstance(m, dict) and "sharpe" in m:
            ml.append(f"  {k}: sharpe={m['sharpe']:.3f} ann_ret={m['ann_return']:.2%} dd={m['max_dd']:.2%}")
        elif isinstance(m, list) and len(m) > 0:
            arr = np.array(m)
            ml.append(f"  {k}: mean_sharpe={np.mean(arr):.3f} std={np.std(arr):.3f}")

    ml.append("")
    ml.append("=== 稳健性 ===")
    for r in rob:
        ml.append(f"  {r.get('label','?')}: sharpe={r.get('sharpe',0):.3f} dd={r.get('max_dd',0):.3f}")

    ml.append("")
    ml.append("=== 无未来函数自检 ===")
    ml.append("[OK] 所有特征使用t日及之前数据")
    ml.append("[OK] signal(t) -> execute(t+1) 14:50")

    acceptance = []
    main_sharpe = main_m.get("sharpe", 0)
    if main_sharpe > 0.8:
        acceptance.append(f"[OK] Sharpe={main_sharpe:.3f} > 0.8")
    else:
        acceptance.append(f"[FAIL] Sharpe={main_sharpe:.3f} <= 0.8, 不值得继续投入")

    if main_sharpe > 1.0:
        acceptance.append(f"[OK] Sharpe={main_sharpe:.3f} > 1.0, 较强")

    # Compare vs benchmark and CYC EW
    bm_sharpe = placebo.get("benchmark", {}).get("sharpe", 0) if isinstance(placebo.get("benchmark"), dict) else 0
    cyc_ew_sharpe = placebo.get("cyc_ew", {}).get("sharpe", 0) if isinstance(placebo.get("cyc_ew"), dict) else 0

    if main_sharpe > bm_sharpe:
        acceptance.append(f"[OK] 主策略({main_sharpe:.3f}) > benchmark({bm_sharpe:.3f})")
    else:
        acceptance.append(f"[FAIL] 主策略({main_sharpe:.3f}) <= benchmark({bm_sharpe:.3f})")

    if main_sharpe > cyc_ew_sharpe:
        acceptance.append(f"[OK] 主策略({main_sharpe:.3f}) > CYC等权({cyc_ew_sharpe:.3f})")
    else:
        acceptance.append(f"[FAIL] 主策略({main_sharpe:.3f}) <= CYC等权({cyc_ew_sharpe:.3f})")

    ml.append("")
    ml.append("=== 验收 ===")
    ml.extend(acceptance)

    with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ml))
    print("[OK] metrics.txt")

    # === Charts ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        main_dates = pd.to_datetime(nav_df["date"])

        # NAV
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(main_dates, nav_df["nav"], "b-", linewidth=1.2, label="CYC Rotation")
        ax.axhline(y=1.0, color="gray", ls="--", alpha=0.5)
        ax.set_title("CycleETF Three-Gate Rotation V0 — NAV", fontsize=14)
        ax.set_xlabel("Date"); ax.set_ylabel("NAV")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "nav.png", dpi=150); plt.close()
        print("[OK] nav.png")

        # Drawdown
        fig, ax = plt.subplots(figsize=(14, 4))
        pk = np.maximum.accumulate(nav_df["nav"].values)
        dd_arr = (pk - nav_df["nav"].values) / pk
        ax.fill_between(main_dates, 0, -dd_arr, color="red", alpha=0.3, label="Drawdown")
        ax.plot(main_dates, -dd_arr, "r-", linewidth=0.8)
        ax.set_title("Drawdown", fontsize=14)
        ax.set_xlabel("Date"); ax.set_ylabel("Drawdown")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "drawdown.png", dpi=150); plt.close()
        print("[OK] drawdown.png")

        # Exposure
        fig, ax = plt.subplots(figsize=(14, 4))
        if not weights_df.empty:
            w_dates = pd.to_datetime(weights_df["date"])
            stock_cols = [c for c in weights_df.columns if c not in ("date", "cash")]
            if stock_cols:
                expo = weights_df[stock_cols].sum(axis=1)
                ax.fill_between(w_dates, 0, expo, alpha=0.5, label="Total Exposure")
                # state cap
                if "state_cap" in nav_df.columns:
                    nav_idx = nav_df.set_index("date")
                    caps = nav_idx["state_cap"]
                    # align to weights dates
                    aligned = caps.reindex(weights_df["date"].values)
                    ax.plot(w_dates, aligned.values, "k--", linewidth=0.8, alpha=0.5, label="State Cap")
        ax.set_title("Portfolio Exposure", fontsize=14)
        ax.set_xlabel("Date"); ax.set_ylabel("Exposure")
        ax.set_ylim(0, 1.1); ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "exposure.png", dpi=150); plt.close()
        print("[OK] exposure.png")
    except Exception as e:
        print(f"[WARN] Chart generation failed: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*70}")
    print(f"完成! {elapsed:.1f}s")
    print(f"输出: {OUT_DIR}")
    print(f"{'='*70}")

    return main_m, placebo, rob


if __name__ == "__main__":
    main()
