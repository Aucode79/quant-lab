#!/usr/bin/env python3
"""
快速回测验证脚本 — 随机选股买入持有，与 JoinQuant 对标

策略：
  1. 随机选取 N 只股票
  2. 每只在随机日期买入（开盘价），持有随机天数后卖出（收盘价）
  3. 等权重分配资金，循环交易
  4. 计算 总收益/年化收益/最大回撤/夏普比率/胜率/盈亏比

用法：
  python quick_backtest.py                    # 默认: 100只, 2024年
  python quick_backtest.py --seed 42          # 固定随机种子（可复现）
  python quick_backtest.py --stocks 50 --year 2023

输出：
  1. 控制台摘要（与 JoinQuant 对比用）
  2. backtest_trades.csv（每笔交易明细）
  3. backtest_daily.csv（每日净值）
"""

import sys, io, random, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from datetime import date, timedelta
from pathlib import Path
import math
import duckdb
import pandas as pd
import numpy as np

# ── 参数 ──────────────────────────────────────────────────
import argparse
p = argparse.ArgumentParser()
p.add_argument("--stocks",  type=int, default=100,   help="随机选股数")
p.add_argument("--trades",  type=int, default=200,   help="总交易次数")
p.add_argument("--year",    type=int, default=2024,  help="回测年份")
p.add_argument("--max-hold",type=int, default=20,    help="最大持仓天数")
p.add_argument("--min-hold",type=int, default=5,     help="最小持仓天数")
p.add_argument("--seed",    type=int, default=None,  help="随机种子(固定则复现)")
p.add_argument("--cash",    type=float, default=1_000_000, help="初始资金")
args = p.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

PARQUET_PATH = "data/market/kline_1min.parquet"
con = duckdb.connect(":memory:")

print("=" * 60)
print("  快速回测 — 随机选股买入持有")
print("=" * 60)
print(f"  年份: {args.year}")
print(f"  选股: {args.stocks} 只")
print(f"  交易: {args.trades} 笔")
print(f"  持仓: {args.min_hold}~{args.max_hold} 天")
print(f"  种子: {args.seed}")
print(f"  资金: {args.cash:,.0f}")
print()

# ── 1. 获取候选股票池 ──
print("[1] 加载股票池...")
t0 = time.time()

stocks = con.execute(f"""
    SELECT DISTINCT ts_code
    FROM '{PARQUET_PATH}/**/*.parquet'
    WHERE year = {args.year}
""").fetchall()
all_codes = sorted([r[0] for r in stocks])
# 随机选 N 只
selected = random.sample(all_codes, min(args.stocks, len(all_codes)))
print(f"    全市场 {len(all_codes)} 只 → 随机选 {len(selected)} 只")

# ── 2. 获取交易日历 ──
trade_dates = con.execute(f"""
    SELECT DISTINCT trade_date
    FROM '{PARQUET_PATH}/**/*.parquet'
    WHERE year = {args.year} AND ts_code IN ('{selected[0]}')
    ORDER BY trade_date
""").fetchall()
dates = [r[0] for r in trade_dates]
print(f"    交易日: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

# ── 3. 生成随机交易计划 ──
print(f"\n[2] 生成 {args.trades} 笔随机交易...")

trades = []
for _ in range(args.trades):
    code = random.choice(selected)
    buy_idx = random.randint(0, len(dates) - args.max_hold - 1)
    buy_date = str(dates[buy_idx])
    hold_days = random.randint(args.min_hold, args.max_hold)
    sell_idx = min(buy_idx + hold_days, len(dates) - 1)
    sell_date = str(dates[sell_idx])
    trades.append({
        "ts_code": code,
        "buy_date": buy_date,
        "sell_date": sell_date,
        "hold_days": sell_idx - buy_idx,
    })

print(f"    持仓天数: {min(t['hold_days'] for t in trades)}~{max(t['hold_days'] for t in trades)}")

# ── 4. 执行回测 ──
print(f"\n[3] 执行回测...")

# 4a. 预加载所有需要的开盘/收盘价
price_cache = {}  # {(ts_code, date): (open, close)}
codes_needed = list(set(t["ts_code"] for t in trades))
print(f"    涉及 {len(codes_needed)} 只股票, 预加载价格...")

# 批量查询所有需要的价格
code_list = "','".join(codes_needed)
price_data = con.execute(f"""
    SELECT ts_code, trade_date,
        FIRST(open ORDER BY trade_time) AS day_open,
        LAST(close ORDER BY trade_time) AS day_close
    FROM '{PARQUET_PATH}/**/*.parquet'
    WHERE ts_code IN ('{code_list}')
      AND year = {args.year}
    GROUP BY ts_code, trade_date
""").fetchall()

for row in price_data:
    price_cache[(row[0], str(row[1]))] = (row[2], row[3])

print(f"    价格缓存: {len(price_cache)} 条 (day_open, day_close)")

# 4b. 按日期模拟
# 先构建所有交易的买入/卖出事件
events = []
for i, t in enumerate(trades):
    buy_price = price_cache.get((t["ts_code"], t["buy_date"]))
    sell_price = price_cache.get((t["ts_code"], t["sell_date"]))
    if buy_price is None or sell_price is None:
        continue
    if buy_price[0] <= 0 or sell_price[1] <= 0:
        continue
    events.append({
        "trade_id": i,
        "ts_code": t["ts_code"],
        "date": t["buy_date"],
        "type": "BUY",
        "price": buy_price[0],
    })
    events.append({
        "trade_id": i,
        "ts_code": t["ts_code"],
        "date": t["sell_date"],
        "type": "SELL",
        "price": sell_price[1],
    })

events.sort(key=lambda e: e["date"])

# 4c. 逐日模拟
cash = args.cash
positions = {}  # {trade_id: {shares, buy_price, code}}
daily_values = []  # [{date, cash, market_value, total_value}]
trade_results = []  # [{trade_id, code, buy_date, sell_date, buy_p, sell_p, shares, return, hold_days}]

event_idx = 0
for d_idx, d in enumerate(dates):
    d_str = str(d)

    # 处理当天事件
    while event_idx < len(events) and events[event_idx]["date"] == d_str:
        ev = events[event_idx]

        if ev["type"] == "BUY":
            # 等权重分配: 每笔交易用总资金的 1/N
            trade_cash = cash / max(1, args.trades - len(positions))
            shares = int(trade_cash / ev["price"] / 100) * 100  # 整手
            if shares >= 100:
                cost = shares * ev["price"]
                # 佣金万3 + 滑点千1
                commission = max(cost * 0.0003, 5.0)
                slippage = cost * 0.001
                total_cost = cost + commission + slippage
                if total_cost <= cash:
                    cash -= total_cost
                    positions[ev["trade_id"]] = {
                        "shares": shares,
                        "buy_price": ev["price"],
                        "code": ev["ts_code"],
                        "buy_date": d_str,
                        "cost": total_cost,
                    }
        elif ev["type"] == "SELL":
            if ev["trade_id"] in positions:
                pos = positions.pop(ev["trade_id"])
                proceeds = pos["shares"] * ev["price"]
                commission = max(proceeds * 0.0003, 5.0)
                tax = proceeds * 0.001  # 印花税
                slippage = proceeds * 0.001
                net = proceeds - commission - tax - slippage
                cash += net
                ret = (net - pos["cost"]) / pos["cost"]
                trade_results.append({
                    "ts_code": pos["code"],
                    "buy_date": pos["buy_date"],
                    "sell_date": d_str,
                    "buy_price": pos["buy_price"],
                    "sell_price": ev["price"],
                    "shares": pos["shares"],
                    "return_pct": ret * 100,
                    "hold_days": (pd.to_datetime(d_str) - pd.to_datetime(pos["buy_date"])).days,
                })

        event_idx += 1

    # 盯市：计算持仓市值
    market_val = 0.0
    for tid, pos in positions.items():
        price = price_cache.get((pos["code"], d_str))
        if price:
            market_val += pos["shares"] * price[1]  # 用收盘价
        else:
            market_val += pos["shares"] * pos["buy_price"]  # fallback

    daily_values.append({
        "date": d_str,
        "cash": cash,
        "market_value": market_val,
        "total_value": cash + market_val,
        "positions": len(positions),
    })

# ── 5. 计算绩效 ──
print(f"\n[4] 计算绩效指标...")

dv = pd.DataFrame(daily_values)
dv["daily_return"] = dv["total_value"].pct_change()
initial = dv["total_value"].iloc[0]
final = dv["total_value"].iloc[-1]
total_return = (final - initial) / initial

# 年化收益
years = len(dv) / 242  # A股约242交易日/年
annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

# 最大回撤
dv["peak"] = dv["total_value"].cummax()
dv["drawdown"] = (dv["peak"] - dv["total_value"]) / dv["peak"]
max_dd = dv["drawdown"].max()

# 夏普
rf_daily = 0.03 / 242
returns = dv["daily_return"].dropna()
excess = returns - rf_daily
sharpe = (excess.mean() / excess.std()) * math.sqrt(242) if excess.std() > 0 else 0

# 胜率
tr = pd.DataFrame(trade_results)
if len(tr) > 0:
    win_rate = (tr["return_pct"] > 0).mean()
    avg_win = tr[tr["return_pct"] > 0]["return_pct"].mean() if (tr["return_pct"] > 0).any() else 0
    avg_loss = tr[tr["return_pct"] < 0]["return_pct"].mean() if (tr["return_pct"] < 0).any() else 0
    profit_factor = abs(tr[tr["return_pct"] > 0]["return_pct"].sum() / tr[tr["return_pct"] < 0]["return_pct"].sum()) if (tr["return_pct"] < 0).any() else float('inf')
    avg_hold = tr["hold_days"].mean()
else:
    win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0

# ── 6. 输出 ──
print()
print("=" * 60)
print("  回测结果 (请与 JoinQuant 对比)")
print("=" * 60)
print(f"  初始资金:     {args.cash:>12,.0f}")
print(f"  最终资金:     {final:>12,.0f}")
print(f"  总收益率:     {total_return*100:>11.2f}%")
print(f"  年化收益率:   {annual_return*100:>11.2f}%")
print(f"  最大回撤:     {max_dd*100:>11.2f}%")
print(f"  夏普比率:     {sharpe:>11.4f}")
print(f"  胜率:         {win_rate*100:>11.1f}%")
print(f"  平均盈利:     {avg_win:>11.2f}%")
print(f"  平均亏损:     {avg_loss:>11.2f}%")
print(f"  盈亏比:       {profit_factor:>11.2f}")
print(f"  平均持仓:     {avg_hold:>11.1f} 天")
print(f"  总交易数:     {len(trade_results):>11}")
print(f"  完成交易:     {len(trade_results):>11} / {args.trades}")

# 现金/持仓状态
last_positions = dv["positions"].iloc[-1]
last_cash_pct = dv["cash"].iloc[-1] / dv["total_value"].iloc[-1] * 100
print(f"  末持仓数:     {last_positions:>11}")
print(f"  末现金占比:   {last_cash_pct:>11.1f}%")

print()
print(f"  ℹ 随机种子 = {args.seed}, 在 JoinQuant 中使用相同种子复现")
print(f"  ℹ JoinQuant 设置:")
print(f"     - 股票池: 同上 {len(selected)} 只")
print(f"     - 交易: Buy(date=open) Hold(days) Sell(date=close)")
print(f"     - 佣金: 万3, 印花税: 千1(卖), 滑点: 千1")

# 输出 CSV 供对比
tr.to_csv("backtest_trades.csv", index=False, encoding="utf-8-sig")
dv.to_csv("backtest_daily.csv", index=False, encoding="utf-8-sig")
print(f"\n  交易明细: backtest_trades.csv")
print(f"  每日净值: backtest_daily.csv")

elapsed = time.time() - t0
print(f"\n  耗时: {elapsed:.1f}s")
print("=" * 60)

con.close()
