"""Debug v2: find root cause of zero trades"""
import duckdb, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_PATH = 'C:/Users/qihen/Desktop/量化研究/quant-system/data/market/kline_1min_etf.parquet'
CODES = ['159745.SZ','159870.SZ','159930.SZ','512400.SH','515210.SH','560280.SH',
         '159611.SZ','159928.SZ','512010.SH','512170.SH','510300.SH']
CODES_STR = ', '.join(f"'{c}'" for c in CODES)

con = duckdb.connect()
daily = con.execute(f"""
    WITH ordered AS (
        SELECT ts_code, trade_date, trade_time,
            open, high, low, close, volume, amount, year, month,
            ROW_NUMBER() OVER (PARTITION BY ts_code, trade_date ORDER BY trade_time ASC) as rn_asc,
            ROW_NUMBER() OVER (PARTITION BY ts_code, trade_date ORDER BY trade_time DESC) as rn_desc
        FROM read_parquet('{DATA_PATH}/*/*/*.parquet', union_by_name=true)
        WHERE ts_code IN ({CODES_STR}) AND close > 0 AND amount >= 0
    ),
    daily_agg AS (
        SELECT ts_code, trade_date,
            MAX(CASE WHEN rn_asc = 1 THEN open END) as open,
            MAX(high) as high, MIN(low) as low,
            MAX(CASE WHEN rn_desc = 1 THEN close END) as close,
            SUM(volume) as volume, SUM(amount) as amount,
            MIN(year) as year, MIN(month) as month
        FROM ordered GROUP BY ts_code, trade_date
    ),
    px1450 AS (
        SELECT DISTINCT ON (ts_code, trade_date)
            ts_code, trade_date, close as px_1450
        FROM ordered
        WHERE EXTRACT(HOUR FROM trade_time) = 14 AND EXTRACT(MINUTE FROM trade_time) = 50
        ORDER BY ts_code, trade_date, trade_time DESC
    ),
    px_before AS (
        SELECT DISTINCT ON (ts_code, trade_date)
            ts_code, trade_date, close as px_1450
        FROM ordered
        WHERE trade_time <= (trade_date::VARCHAR || ' 14:50:00')::TIMESTAMP
        ORDER BY ts_code, trade_date, trade_time DESC
    ),
    px_final AS (
        SELECT DISTINCT ON (ts_code, trade_date)
            ts_code, trade_date, close as px_1450
        FROM ordered ORDER BY ts_code, trade_date, trade_time DESC
    )
    SELECT d.*, COALESCE(p1.px_1450, p2.px_1450, p3.px_1450) as px_1450,
           CASE WHEN p1.px_1450 IS NULL AND p2.px_1450 IS NULL THEN TRUE ELSE FALSE END as exec_fallback
    FROM daily_agg d
    LEFT JOIN px1450 p1 ON d.ts_code=p1.ts_code AND d.trade_date=p1.trade_date
    LEFT JOIN px_before p2 ON d.ts_code=p2.ts_code AND d.trade_date=p2.trade_date
    LEFT JOIN px_final p3 ON d.ts_code=p3.ts_code AND d.trade_date=p3.trade_date
    ORDER BY d.ts_code, d.trade_date
""").df()
con.close()

daily['trade_date'] = pd.to_datetime(daily['trade_date'])
daily['illiquid_day'] = daily['amount'] == 0

close = daily.pivot_table(index='trade_date', columns='ts_code', values='close', aggfunc='last')
close.index = pd.to_datetime(close.index)
amount = daily.pivot_table(index='trade_date', columns='ts_code', values='amount', aggfunc='sum')
amount.index = pd.to_datetime(amount.index)
px1450_df = daily.pivot_table(index='trade_date', columns='ts_code', values='px_1450', aggfunc='last')
px1450_df.index = pd.to_datetime(px1450_df.index)
illiq = daily.pivot_table(index='trade_date', columns='ts_code', values='illiquid_day', aggfunc='last')
illiq.index = pd.to_datetime(illiq.index)

bench_dates = daily[daily['ts_code']=='510300.SH']['trade_date'].drop_duplicates().sort_values()
calendar = pd.DatetimeIndex(sorted(bench_dates))

close = close.reindex(calendar)
amount = amount.reindex(calendar)
px1450_df = px1450_df.reindex(calendar)
illiq = illiq.reindex(calendar)

cyc_codes = ['159745.SZ','159870.SZ','159930.SZ','512400.SH','515210.SH','560280.SH']
def_codes = ['159611.SZ','159928.SZ','512010.SH','512170.SH']
all_codes = cyc_codes + def_codes + ['510300.SH']
ANN_DAYS = 242

# Features (same as main)
ret5 = close / close.shift(5) - 1
ret20 = close / close.shift(20) - 1
ret60 = close / close.shift(60) - 1
ret120 = close / close.shift(120) - 1
ma120 = close.rolling(120, min_periods=1).mean()
daily_ret = close.pct_change()
vol20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(ANN_DAYS)
amt20 = amount.rolling(20, min_periods=1).mean()
amt20_med252 = amt20.rolling(252, min_periods=1).median()
amt_growth = amt20 / amt20_med252.replace(0, np.nan) - 1

codes_only = cyc_codes + def_codes
amt_sum = amt20[codes_only].sum(axis=1)
amt_share = amt20[codes_only].div(amt_sum.replace(0, np.nan), axis=0)

bench = close['510300.SH']
log_ret60 = np.log(close / close.shift(60))
log_bench = np.log(bench / bench.shift(60))
rs60 = pd.DataFrame(index=close.index, columns=close.columns)
for c in close.columns:
    if c in log_ret60.columns:
        rs60[c] = log_ret60[c] - log_bench

def cross_z(df, col_list):
    valid = [c for c in col_list if c in df.columns]
    sub = df[valid]
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

crowd = pd.DataFrame(0.5, index=close.index, columns=close.columns)

score = pd.DataFrame(0.0, index=calendar, columns=all_codes)
for c in cyc_codes:
    if c not in close.columns: continue
    score[c] = (0.35 * z_ret60[c].fillna(0) + 0.20 * z_ret120[c].fillna(0)
                + 0.20 * z_rs60[c].fillna(0) + 0.10 * z_amt_g[c].fillna(0)
                - 0.15 * z_ret5[c].fillna(0) - 0.30 * crowd[c].fillna(0.5))

eligible = pd.DataFrame(False, index=calendar, columns=all_codes)
for c in cyc_codes:
    if c not in close.columns or c not in ma120.columns: continue
    iq_chk = illiq[c].fillna(0) == 0 if c in illiq.columns else True
    eligible[c] = (score[c] > 0) & (close[c] > ma120[c]) & iq_chk

# Check first date with any eligible
print("First 5 dates with eligible ETFs:")
n_found = 0
for i in range(len(calendar)):
    el = eligible.iloc[i]
    elig_codes = [c for c in cyc_codes if c in el.index and el[c]]
    if len(elig_codes) > 0 and n_found < 5:
        print(f"  {calendar[i].date()}: {len(elig_codes)} eligible: {elig_codes}")
        n_found += 1
    if n_found >= 5:
        break

# Check: is vol20 non-zero?
print(f"\nVol20 stats for 512400.SH:")
v = vol20['512400.SH']
vn = v.dropna()
print(f"  non-NaN: {len(vn)}, min: {vn.min():.6f}, mean: {vn.mean():.6f}")

# Check px1450 availability
print(f"\npx1450 check for 512400.SH - first 10 dates:")
px = px1450_df['512400.SH']
n = 0
for i in range(len(calendar)):
    v = px.iloc[i]
    if not pd.isna(v):
        print(f"  {calendar[i].date()}: px1450={v:.4f}")
        n += 1
        if n >= 5:
            break

# ===== Minimal simulate with debug =====
print("\n=== MINIMAL SIMULATE ===")
N = 5
EXIT_RANK = 8
SLIP = 0.001
COMM = 0.000115
MIN_COMM = 5.0
nav = 1.0
cash = 1.0
holdings = {}
trades = 0

found_signal = False
for i in range(len(calendar)):
    exec_idx = i + 1
    if exec_idx >= len(calendar):
        continue
    exec_date = calendar[exec_idx]

    sc = score.iloc[i]
    el = eligible.iloc[i]

    ranked = []
    for c in cyc_codes:
        if c in el.index and c in sc.index:
            is_elig = bool(el[c])
            is_pos = bool(sc[c] > 0)
            if is_elig and is_pos:
                ranked.append((c, sc[c]))
    ranked.sort(key=lambda x: -x[1])

    if len(ranked) > 0 and not found_signal:
        found_signal = True
        t_date = calendar[i]
        print(f"First ranked at signal_date={t_date.date()}, exec_date={exec_date.date()}:")
        for c, s in ranked[:5]:
            print(f"  {c}: score={s:.4f}, vol20={vol20[c].iloc[i]:.4f}, px1450={px1450_df[c].loc[exec_date] if exec_date in px1450_df.index else 'MISSING'}")

    # Build selected
    current = set(holdings.keys())
    selected = []
    for rank, (c, s) in enumerate(ranked):
        rk = rank + 1
        if c in current:
            if rk <= EXIT_RANK:
                selected.append(c)
        else:
            if rk <= N:
                selected.append(c)

    if len(selected) == 0:
        continue

    # Check execution price
    ep = px1450_df.loc[exec_date] if exec_date in px1450_df.index else None
    if ep is None:
        continue

    # Check that selected ETF has px1450
    all_have_px = True
    for c in selected:
        if c not in ep.index or pd.isna(ep[c]):
            all_have_px = False
            if not found_signal:
                print(f"  MISSING px1450 for {c} at {exec_date.date()}")

    if not all_have_px:
        continue

    # Try to trade
    print(f"TRADE at signal={calendar[i].date()}, exec={exec_date.date()}")
    print(f"  Selected: {[(c, round(float(sc[c]),4)) for c in selected]}")

    # Simple equal weight
    w = 0.9 / len(selected)
    nav_local = cash
    for c, sh in holdings.items():
        if c in ep.index and not pd.isna(ep[c]):
            nav_local += sh * ep[c]

    for c in selected:
        if c in holdings:
            continue
        buy_px = ep[c] * (1 + SLIP)
        target = nav_local * w
        sh = int(target / buy_px / 100) * 100
        if sh >= 100:
            gross = sh * buy_px
            cost = max(gross * COMM, MIN_COMM)
            if gross + cost <= cash:
                cash -= (gross + cost)
                holdings[c] = sh
                trades += 1
                print(f"  BUY {c}: {sh} shares @ {buy_px:.4f} = {gross:.2f}")

    if trades > 0:
        break

print(f"\nTrades: {trades}, Holdings: {len(holdings)}")
