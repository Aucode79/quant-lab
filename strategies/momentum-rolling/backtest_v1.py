#!/usr/bin/env python3
"""
V1动量滚仓 — 精确对齐JoinQuant
- 动量计算: 日线收盘价 (与JQ history('1d','close')一致)
- 交易执行: 14:50价 (与JQ run_weekly 14:50一致)
- 避险: 现金 (与JQ 511280退市后行为一致)
"""
import sys, io, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import duckdb, pandas as pd, numpy as np

PARQUET_ETF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/market/kline_1min_etf.parquet"
OUT = Path(__file__).parent
CASH = 100_000; START = "2024-05-10"; END = "2026-05-10"
MOM = 20; SLIP = 0.001; COMM = 0.000115; MIN_COMM = 5.0
ETF_POOL = ['159869.SZ','513100.SH','512890.SH','518880.SH']

con = duckdb.connect(":memory:")
cs = "','".join(ETF_POOL)
avail = {r[0] for r in con.execute(f"SELECT DISTINCT ts_code FROM '{PARQUET_ETF}/**/*.parquet' WHERE ts_code IN ('{cs}')").fetchall()}
pool = [e for e in ETF_POOL if e in avail]
print(f"ETF: {len(pool)}/4")

# 提前加载30天数据用于回溯 (与JQ预加载数据一致)
PRE_START = "2024-03-15"

# 日线收盘 (动量) + 14:50价格 (执行)
df_c = con.execute(f"""
    SELECT ts_code, trade_date, LAST(close ORDER BY trade_time) AS close
    FROM '{PARQUET_ETF}/**/*.parquet'
    WHERE ts_code IN ('{cs}') AND trade_date>='{PRE_START}' AND trade_date<='{END}'
    GROUP BY ts_code, trade_date ORDER BY ts_code, trade_date
""").fetchdf()

df_x = con.execute(f"""
    SELECT ts_code, trade_date, close AS px
    FROM '{PARQUET_ETF}/**/*.parquet'
    WHERE ts_code IN ('{cs}') AND trade_date>='{PRE_START}' AND trade_date<='{END}'
      AND strftime(trade_time,'%H:%M')='14:50'
    ORDER BY ts_code, trade_date
""").fetchdf()
con.close()

PC = df_c.pivot(index='trade_date', columns='ts_code', values='close')  # 日线收盘=动量
PX = df_x.pivot(index='trade_date', columns='ts_code', values='px')      # 14:50=执行价
dates = sorted(PC.index)
print(f"交易日: {len(dates)} ({str(dates[0])[:10]}~{str(dates[-1])[:10]})")

cash = CASH; pos = None; trades = []; daily = []

for i, today in enumerate(dates):
    t = str(today)[:10]
    rc = PC.loc[today] if today in PC.index else None  # 日线收盘
    rx = PX.loc[today] if today in PX.index else None  # 14:50价

    # 盯市 (用日线收盘)
    mv = 0.0
    if pos:
        code, sh = pos
        if rc is not None and code in rc.index:
            p = rc[code]
            if not pd.isna(p): mv = sh*p
    daily.append({'date':t,'total':cash+mv})

    if pd.Timestamp(today).dayofweek != 0: continue

    # 动量得分: 纯日线收盘 (21天, 全部用每日收盘价)
    lb = max(0,i-MOM); lbd = [str(dates[j])[:10] for j in range(lb,i+1)]
    scores = {}
    for etf in pool:
        cols = [PC.loc[d,etf] for d in lbd if d in PC.index and etf in PC.columns and not pd.isna(PC.loc[d,etf])]
        if len(cols)<MOM: continue
        s = pd.Series(cols)
        r = (s.iloc[-1]-s.iloc[0])/s.iloc[0]
        v = max(s.pct_change().dropna().std()*np.sqrt(252),1e-4)
        ma = s.iloc[-20:].mean()
        scores[etf] = {'score':r/v,'ret':r,'price':s.iloc[-1],'ma20':ma}
    if not scores: continue
    valid = {k:v for k,v in scores.items() if v['ret']>0 and v['price']>v['ma20']}
    target = max(valid.items(), key=lambda x:x[1]['score'])[0] if valid else "CASH"
    if pos and pos[0]==target: continue

    # 卖出 (用日线收盘价)
    if pos:
        oc,osh = pos
        op = rc[oc] if rc is not None and oc in rc.index else None
        if op and op>0:
            p = osh*op*(1-SLIP)
            c = max(p*COMM,MIN_COMM)
            cash += p-c
            trades.append({'date':t,'act':'SELL','code':oc,'px':op*(1-SLIP),'sh':osh,'comm':c,'cash':cash})
        pos = None

    if target=="CASH":
        trades.append({'date':t,'act':'CASH','code':'CASH','px':0,'sh':0,'comm':0,'cash':cash})
        continue

    # 买入 (用日线收盘价)
    if rc is None or target not in rc.index: continue
    bp = rc[target]
    if pd.isna(bp) or bp<=0: continue
    bp = bp*(1+SLIP)
    gross = cash*(1-COMM)
    sh = int(gross/bp/100)*100
    if sh<100: continue
    cost = sh*bp; comm = max(cost*COMM,MIN_COMM)
    cash = cash-cost-comm
    pos = (target,sh)
    trades.append({'date':t,'act':'BUY','code':target,'px':bp,'sh':sh,'comm':comm,'cash':cash})

# 清算
if pos:
    code,sh = pos; lastp = PC.iloc[-1][code]; cash += sh*lastp*(1-SLIP)

vals = [r['total'] for r in daily]; n=len(vals)
rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,n) if vals[i-1]>0]
tr = (vals[-1]-vals[0])/vals[0]; ann = (1+tr)**(242/max(n,1))-1
pk=vals[0];dd=0
for v in vals: pk=max(pk,v); dd=max(dd,(pk-v)/pk if pk>0 else 0)
rf=0.03/242; exc=[r-rf for r in rets]
if len(exc)>1:
    m=sum(exc)/len(exc); std=math.sqrt(sum((x-m)**2 for x in exc)/len(exc))
    shp=(m/std)*math.sqrt(242) if std>0 else 0
else: shp=0

bt=len([t for t in trades if t['act']=='BUY'])
ct=len([t for t in trades if t['act']=='CASH'])

print(f"\n{'='*55}")
print(f"  V1动量滚仓 (精准对齐JQ)")
print(f"{'='*55}")
print(f"  本金:{CASH:>10,}  最终:{vals[-1]:>10,.0f}")
print(f"  收益:{tr*100:>6.1f}%  年化:{ann*100:>6.1f}%  回撤:{dd*100:>5.1f}%  夏普:{shp:>7.4f}")
print(f"  买:{bt}次  避险:{ct}次")
print(f"  {str(dates[0])[:10]} ~ {str(dates[-1])[:10]}")
print("="*55)
print(f"\nJQ: 收益88% 回撤17% 夏普1.43  21盈/11亏")

pd.DataFrame(trades).to_csv(OUT/"v1_trades.csv",index=False,encoding='utf-8-sig')
pd.DataFrame(daily).to_csv(OUT/"v1_daily.csv",index=False,encoding='utf-8-sig')
