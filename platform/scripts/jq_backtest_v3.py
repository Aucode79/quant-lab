# -*- coding: utf-8 -*-
"""
JoinQuant 回测 v3 — 使用原始 order() 命令
每笔交易独立执行，买按开盘价，卖按收盘价
"""
import math

# 前15笔交易（先跑通再补全）
TRADES = [
    ("001236.XSHE", "2024-05-08", "2024-05-23"),
    ("002857.XSHE", "2024-07-15", "2024-08-07"),
    ("300521.XSHE", "2024-07-01", "2024-07-12"),
    ("300805.XSHE", "2024-02-28", "2024-03-15"),
    ("600529.XSHG", "2024-08-06", "2024-08-23"),
    ("600529.XSHG", "2024-08-14", "2024-09-09"),
    ("300218.XSHE", "2024-06-07", "2024-07-02"),
    ("300807.XSHE", "2024-02-28", "2024-03-27"),
    ("600732.XSHG", "2024-10-24", "2024-11-01"),
    ("002249.XSHE", "2024-03-05", "2024-03-19"),
    ("600280.XSHG", "2024-05-31", "2024-06-28"),
    ("300806.XSHE", "2024-04-11", "2024-04-18"),
    ("300337.XSHE", "2024-03-11", "2024-03-18"),
    ("300900.XSHE", "2024-05-17", "2024-06-12"),
    ("688283.XSHG", "2024-03-11", "2024-03-26"),
]

CASH = 1_000_000

def initialize(context):
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
        open_commission=0.0003, close_commission=0.0003,
        close_today_commission=0, min_commission=5.0), type='stock')
    set_slippage(FixedSlippage(0.001))

    g.buy_orders = {}   # date -> [(code, shares)]
    g.sell_orders = {}  # date -> [(code, shares)]
    g.trade_results = []
    g.daily_vals = []

    # 构建买卖指令：每笔交易固定10万资金
    per_trade_cash = CASH / len(TRADES)
    for code, buy_d, sell_d in TRADES:
        if buy_d not in g.buy_orders:
            g.buy_orders[buy_d] = []
        g.buy_orders[buy_d].append((code, per_trade_cash))

        if sell_d not in g.sell_orders:
            g.sell_orders[sell_d] = []
        g.sell_orders[sell_d].append(code)

    g.total_trades = len(TRADES)
    g.completed = 0

def handle_data(context, data):
    today = context.current_dt.strftime('%Y-%m-%d')

    # 卖出
    if today in g.sell_orders:
        for code in g.sell_orders[today]:
            pos = context.portfolio.positions.get(code)
            if pos and pos.closeable_amount > 0:
                order(code, -pos.closeable_amount)
                g.completed += 1

    # 买入
    if today in g.buy_orders:
        for code, cash_alloc in g.buy_orders[today]:
            try:
                price = data[code].close  # 用收盘价（简化，避免开盘价不可得）
                if price and price > 0:
                    shares = int(cash_alloc / price / 100) * 100
                    if shares >= 100:
                        order(code, shares)
            except:
                pass

    g.daily_vals.append(context.portfolio.total_value)

def after_trading_end(context):
    if context.current_dt.strftime('%Y-%m-%d') != '2024-12-31':
        return

    vals = g.daily_vals
    n = len(vals)
    rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,n) if vals[i-1]>0]
    total_ret = (vals[-1]-vals[0])/vals[0] if vals[0]>0 else 0
    annual = (1+total_ret)**(252/max(n,1))-1

    peak = vals[0]; max_dd = 0
    for v in vals:
        peak = max(peak,v)
        max_dd = max(max_dd, (peak-v)/peak if peak>0 else 0)

    rf = 0.03/252
    exc = [r-rf for r in rets]
    if len(exc)>1:
        m = sum(exc)/len(exc)
        std = math.sqrt(sum((x-m)**2 for x in exc)/len(exc))
        sharpe = (m/std)*math.sqrt(252) if std>0 else 0
    else:
        sharpe = 0

    log.info("=" * 45)
    log.info(f"  总收益: {total_ret*100:.2f}%   (本地: 1.68%)")
    log.info(f"  年化:   {annual*100:.2f}%   (本地: 1.68%)")
    log.info(f"  回撤:   {max_dd*100:.2f}%   (本地: 2.23%)")
    log.info(f"  夏普:   {sharpe:.4f}    (本地: -0.622)")
    log.info(f"  交易:   {g.completed} 笔       (本地: 184笔)")
    log.info("=" * 45)
