# -*- coding: utf-8 -*-
"""
JoinQuant 对标回测脚本
直接粘贴到 JoinQuant 策略编辑器运行

配置:
  - seed=42, 2024年, 100只随机股票, 200笔随机交易
  - 买入=开盘价, 卖出=收盘价, 持仓5~20天
  - 万3佣金, 千1印花税(卖), 千1滑点
"""

import random
import numpy as np
import pandas as pd
import math
import datetime

# ── 参数 ──
SEED = 42
YEAR = 2024
CASH = 1_000_000
TRADES = 200
MIN_HOLD = 5
MAX_HOLD = 20
COMMISSION = 0.0003
STAMP_TAX = 0.001
SLIPPAGE = 0.001

random.seed(SEED)
np.random.seed(SEED)

# ── 本地系统 100 只股票池 (seed=42 随机选出) ──
# 去掉北交所 (92xxxx)，JoinQuant 不支持
STOCK_POOL = [
    '002397.XSHE',  # 002397.SZ
    '000659.XSHE',  # 000659.SZ
    '300781.XSHE',  # 300781.SZ
    '300522.XSHE',  # 300522.SZ
    '300337.XSHE',  # 300337.SZ
    '002643.XSHE',  # 002643.SZ
    '002323.XSHE',  # 002323.SZ
    '603986.XSHG',  # 603986.SH
    '002193.XSHE',  # 002193.SZ
    '688283.XSHG',  # 688283.SH
    '600732.XSHG',  # 600732.SH
    '000738.XSHE',  # 000738.SZ
    '000718.XSHE',  # 000718.SZ
    '002249.XSHE',  # 002249.SZ
    '300294.XSHE',  # 300294.SZ
    '300420.XSHE',  # 300420.SZ
    '603325.XSHG',  # 603325.SH
    '688390.XSHG',  # 688390.SH
    '000681.XSHE',  # 000681.SZ
    '688006.XSHG',  # 688006.SH
    '300125.XSHE',  # 300125.SZ
    '603982.XSHG',  # 603982.SH
    '600711.XSHG',  # 600711.SH
    '300310.XSHE',  # 300310.SZ
    '601022.XSHG',  # 601022.SH
    '688272.XSHG',  # 688272.SH
    '300807.XSHE',  # 300807.SZ
    '000089.XSHE',  # 000089.SZ
    '002823.XSHE',  # 002823.SZ
    '600738.XSHG',  # 600738.SH
    '301387.XSHE',  # 301387.SZ
    '300805.XSHE',  # 300805.SZ
    '002786.XSHE',  # 002786.SZ
    '300265.XSHE',  # 300265.SZ
    '301348.XSHE',  # 301348.SZ
    '002321.XSHE',  # 002321.SZ
    '002241.XSHE',  # 002241.SZ
    '600295.XSHG',  # 600295.SH
    '002276.XSHE',  # 002276.SZ
    '600079.XSHG',  # 600079.SH
    '301498.XSHE',  # 301498.SZ
    '688416.XSHG',  # 688416.SH
    '300687.XSHE',  # 300687.SZ
    '000898.XSHE',  # 000898.SZ
    '601375.XSHG',  # 601375.SH
    '603859.XSHG',  # 603859.SH
    '002518.XSHE',  # 002518.SZ
    '600280.XSHG',  # 600280.SH
    '002124.XSHE',  # 002124.SZ
    '605155.XSHG',  # 605155.SH
    '300933.XSHE',  # 300933.SZ
    '688728.XSHG',  # 688728.SH
    '688608.XSHG',  # 688608.SH
    '600108.XSHG',  # 600108.SH
    '688159.XSHG',  # 688159.SH
    '300069.XSHE',  # 300069.SZ
    '002044.XSHE',  # 002044.SZ
    '000923.XSHE',  # 000923.SZ
    '300380.XSHE',  # 300380.SZ
    '300900.XSHE',  # 300900.SZ
    '002132.XSHE',  # 002132.SZ
    '300422.XSHE',  # 300422.SZ
    '002311.XSHE',  # 002311.SZ
    '600297.XSHG',  # 600297.SH
    '300806.XSHE',  # 300806.SZ
    '601137.XSHG',  # 601137.SH
    '600137.XSHG',  # 600137.SH
    '002850.XSHE',  # 002850.SZ
    '600193.XSHG',  # 600193.SH
    '600035.XSHG',  # 600035.SH
    '300218.XSHE',  # 300218.SZ
    '300709.XSHE',  # 300709.SZ
    '002059.XSHE',  # 002059.SZ
    '688510.XSHG',  # 688510.SH
    '002925.XSHE',  # 002925.SZ
    '603818.XSHG',  # 603818.SH
    '300521.XSHE',  # 300521.SZ
    '002857.XSHE',  # 002857.SZ
    '601600.XSHG',  # 601600.SH
    '600289.XSHG',  # 600289.SH
    '300736.XSHE',  # 300736.SZ
    '605333.XSHG',  # 605333.SH
    '300303.XSHE',  # 300303.SZ
    '301219.XSHE',  # 301219.SZ
    '001236.XSHE',  # 001236.SZ
    '300390.XSHE',  # 300390.SZ
    '000750.XSHE',  # 000750.SZ
    '301131.XSHE',  # 301131.SZ
    '600529.XSHG',  # 600529.SH
    '300716.XSHE',  # 300716.SZ
    '002016.XSHE',  # 002016.SZ
    '300230.XSHE',  # 300230.SZ
    '688063.XSHG',  # 688063.SH
]

# ── 2024 交易日历 (从本地1分钟数据提取, 242天) ──
TRADE_DATES_STR = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12', '2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19', '2024-01-22', '2024-01-23', '2024-01-24', '2024-01-25', '2024-01-26', '2024-01-29', '2024-01-30', '2024-01-31', '2024-02-01', '2024-02-02', '2024-02-05', '2024-02-06', '2024-02-07', '2024-02-08', '2024-02-19', '2024-02-20', '2024-02-21', '2024-02-22', '2024-02-23', '2024-02-26', '2024-02-27', '2024-02-28', '2024-02-29', '2024-03-01', '2024-03-04', '2024-03-05', '2024-03-06', '2024-03-07', '2024-03-08', '2024-03-11', '2024-03-12', '2024-03-13', '2024-03-14', '2024-03-15', '2024-03-18', '2024-03-19', '2024-03-20', '2024-03-21', '2024-03-22', '2024-03-25', '2024-03-26', '2024-03-27', '2024-03-28', '2024-03-29', '2024-04-01', '2024-04-02', '2024-04-03', '2024-04-08', '2024-04-09', '2024-04-10', '2024-04-11', '2024-04-12', '2024-04-15', '2024-04-16', '2024-04-17', '2024-04-18', '2024-04-19', '2024-04-22', '2024-04-23', '2024-04-24', '2024-04-25', '2024-04-26', '2024-04-29', '2024-04-30', '2024-05-06', '2024-05-07', '2024-05-08', '2024-05-09', '2024-05-10', '2024-05-13', '2024-05-14', '2024-05-15', '2024-05-16', '2024-05-17', '2024-05-20', '2024-05-21', '2024-05-22', '2024-05-23', '2024-05-24', '2024-05-27', '2024-05-28', '2024-05-29', '2024-05-30', '2024-05-31', '2024-06-03', '2024-06-04', '2024-06-05', '2024-06-06', '2024-06-07', '2024-06-11', '2024-06-12', '2024-06-13', '2024-06-14', '2024-06-17', '2024-06-18', '2024-06-19', '2024-06-20', '2024-06-21', '2024-06-24', '2024-06-25', '2024-06-26', '2024-06-27', '2024-06-28', '2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05', '2024-07-08', '2024-07-09', '2024-07-10', '2024-07-11', '2024-07-12', '2024-07-15', '2024-07-16', '2024-07-17', '2024-07-18', '2024-07-19', '2024-07-22', '2024-07-23', '2024-07-24', '2024-07-25', '2024-07-26', '2024-07-29', '2024-07-30', '2024-07-31', '2024-08-01', '2024-08-02', '2024-08-05', '2024-08-06', '2024-08-07', '2024-08-08', '2024-08-09', '2024-08-12', '2024-08-13', '2024-08-14', '2024-08-15', '2024-08-16', '2024-08-19', '2024-08-20', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-26', '2024-08-27', '2024-08-28', '2024-08-29', '2024-08-30', '2024-09-02', '2024-09-03', '2024-09-04', '2024-09-05', '2024-09-06', '2024-09-09', '2024-09-10', '2024-09-11', '2024-09-12', '2024-09-13', '2024-09-18', '2024-09-19', '2024-09-20', '2024-09-23', '2024-09-24', '2024-09-25', '2024-09-26', '2024-09-27', '2024-09-30', '2024-10-08', '2024-10-09', '2024-10-10', '2024-10-11', '2024-10-14', '2024-10-15', '2024-10-16', '2024-10-17', '2024-10-18', '2024-10-21', '2024-10-22', '2024-10-23', '2024-10-24', '2024-10-25', '2024-10-28', '2024-10-29', '2024-10-30', '2024-10-31', '2024-11-01', '2024-11-04', '2024-11-05', '2024-11-06', '2024-11-07', '2024-11-08', '2024-11-11', '2024-11-12', '2024-11-13', '2024-11-14', '2024-11-15', '2024-11-18', '2024-11-19', '2024-11-20', '2024-11-21', '2024-11-22', '2024-11-25', '2024-11-26', '2024-11-27', '2024-11-28', '2024-11-29', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05', '2024-12-06', '2024-12-09', '2024-12-10', '2024-12-11', '2024-12-12', '2024-12-13', '2024-12-16', '2024-12-17', '2024-12-18', '2024-12-19', '2024-12-20', '2024-12-23', '2024-12-24', '2024-12-25', '2024-12-26', '2024-12-27', '2024-12-30', '2024-12-31']

def initialize(context):
    """JoinQuant 初始化"""
    g.stock_pool = STOCK_POOL
    g.cash = CASH
    g.trades_planned = TRADES
    g.min_hold = MIN_HOLD
    g.max_hold = MAX_HOLD
    g.commission = COMMISSION
    g.stamp_tax = STAMP_TAX
    g.slippage = SLIPPAGE
    g.seed = SEED

    # 设置手续费
    set_order_cost(OrderCost(
        open_tax=0,
        close_tax=g.stamp_tax,
        open_commission=g.commission,
        close_commission=g.commission,
        close_today_commission=0,
        min_commission=5.0
    ), type='stock')
    set_slippage(FixedSlippage(g.slippage * 0.5))  # JoinQuant 是双边滑点

    # 随机生成交易计划（种子固定，保证可复现）
    random.seed(g.seed)
    np.random.seed(g.seed)

    # 2024交易日历 (从本地数据提取, 共242天)
    trade_dates = [datetime.date(*(int(x) for x in d.split('-')))
                   for d in TRADE_DATES_STR]

    g.trade_plan = []
    for _ in range(g.trades_planned):
        code = random.choice(g.stock_pool)
        buy_idx = random.randint(0, len(trade_dates) - g.max_hold - 1)
        buy_date = trade_dates[buy_idx]
        hold_days = random.randint(g.min_hold, g.max_hold)
        sell_idx = min(buy_idx + hold_days, len(trade_dates) - 1)
        sell_date = trade_dates[sell_idx]

        g.trade_plan.append({
            "code": code,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "executed": False,
        })

    g.trade_idx = 0
    g.trade_results = []
    g.daily_records = []
    log.info(f"交易计划: {len(g.trade_plan)} 笔")


def handle_data(context, data):
    """每日运行"""
    current_dt = context.current_dt.date()

    # 执行买入
    while g.trade_idx < len(g.trade_plan):
        plan = g.trade_plan[g.trade_idx]
        if plan["buy_date"] > current_dt:
            break
        if plan["executed"]:
            g.trade_idx += 1
            continue

        code = plan["code"]
        try:
            price = data[code].open
            if price is None or price <= 0:
                g.trade_idx += 1
                continue

            # 等权重分配
            n_trades = min(g.trades_planned, len(g.trade_plan))
            trade_cash = context.portfolio.total_value / n_trades
            if trade_cash <= 0:
                g.trade_idx += 1
                continue

            shares = int(trade_cash / price / 100) * 100
            if shares >= 100:
                order_target_value(code, shares * price)
                plan["shares"] = shares
                plan["buy_price"] = price
                plan["executed"] = True
            else:
                plan["executed"] = True  # 标记跳过，不重复尝试
        except Exception as e:
            log.warn(f"买入异常 {code}: {e}")
            plan["executed"] = True
        g.trade_idx += 1

    # 执行卖出
    to_sell = [plan for plan in g.trade_plan
               if plan["executed"] and plan.get("shares", 0) > 0
               and plan["sell_date"] == current_dt]
    for plan in to_sell:
        code = plan["code"]
        try:
            price = data[code].close
            if price is None or price <= 0:
                continue
            order_target_value(code, 0)
            ret = (price - plan["buy_price"]) / plan["buy_price"]
            g.trade_results.append({
                "code": code,
                "buy_date": str(plan["buy_date"]),
                "sell_date": str(current_dt),
                "buy_price": plan["buy_price"],
                "sell_price": price,
                "return": ret,
                "hold_days": (plan["sell_date"] - plan["buy_date"]).days,
            })
            plan["shares"] = 0
        except Exception as e:
            log.warn(f"卖出异常 {code}: {e}")

    # 记录每日净值
    g.daily_records.append({
        "date": str(current_dt),
        "total_value": context.portfolio.total_value,
        "cash": context.portfolio.cash,
        "positions": len(context.portfolio.positions),
    })


def after_trading_end(context):
    """收盘后——最后一天计算指标"""
    if context.current_dt.date() < datetime.date(YEAR, 12, 30):
        return

    n = len(g.daily_records)
    if n < 2:
        return

    # 计算指标
    values = [r["total_value"] for r in g.daily_records]
    returns = [(values[i] - values[i-1]) / values[i-1]
               for i in range(1, n)]

    total_return = (values[-1] - values[0]) / values[0]
    annual_return = (1 + total_return) ** (252 / n) - 1

    peak = values[0]
    max_dd = 0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    rf_daily = 0.03 / 252
    excess = [r - rf_daily for r in returns]
    if len(excess) > 1:
        sharpe = (np.mean(excess) / np.std(excess)) * math.sqrt(252)
    else:
        sharpe = 0

    trades = g.trade_results
    wins = [t for t in trades if t["return"] > 0]
    losses = [t for t in trades if t["return"] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t["return"] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t["return"] for t in losses]) * 100 if losses else 0

    log.info("=" * 50)
    log.info("  回测结果 (JoinQuant)")
    log.info("=" * 50)
    log.info(f"  初始资金:   {CASH:,.0f}")
    log.info(f"  最终资金:   {values[-1]:,.0f}")
    log.info(f"  总收益率:   {total_return*100:.2f}%")
    log.info(f"  年化收益率: {annual_return*100:.2f}%")
    log.info(f"  最大回撤:   {max_dd*100:.2f}%")
    log.info(f"  夏普比率:   {sharpe:.4f}")
    log.info(f"  胜率:       {win_rate*100:.1f}%")
    log.info(f"  平均盈利:   {avg_win:.2f}%")
    log.info(f"  平均亏损:   {avg_loss:.2f}%")
    log.info(f"  完成交易:   {len(trades)} / {TRADES}")
    log.info("=" * 50)

    # 输出本地对标值
    log.info("")
    log.info("  本地系统结果 (应与此接近):")
    log.info(f"  总收益 1.68%, 夏普 -0.622, 回撤 2.23%, 胜率 47.3%")
    log.info(f"  交易 184/200, 盈亏比 1.40")
