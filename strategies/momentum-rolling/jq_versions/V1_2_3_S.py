# 策略版本：V1.2.3_S (滑点测试版)
# 核心逻辑：加入 0.1% 的动态滑点摩擦，模拟真实的盘口吃单损失
from jqdata import *
import numpy as np
import pandas as pd
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 模拟真实市场摩擦：设置千分之一的价格滑点
    # 这是区分“纸上富贵”和“实盘提款机”的试金石
    set_slippage(PriceRelatedSlippage(0.001))
    g.etf_pool = ['159869.XSHE', '513100.XSHG', '512890.XSHG', '518880.XSHG']
    g.bond_etf = '511280.XSHG'
    g.momentum_days = 20
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.000115, close_commission=0.000115, 
        min_commission=5
    ), type='fund')
    run_weekly(rotate_etf, weekday=1, time='14:50')
def rotate_etf(context):
    all_targets = g.etf_pool + [g.bond_etf]
    hist = history(g.momentum_days + 1, '1d', 'close', all_targets)
    scores = {}
    for target in g.etf_pool:
        closes = hist[target]
        if len(closes) < g.momentum_days + 1 or pd.isna(closes.iloc[-1]):
            continue
        ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        daily_rets = closes.pct_change().dropna()
        volatility = max(daily_rets.std() * np.sqrt(252), 0.0001)
        scores[target] = {
            'score': ret / volatility, 
            'ret': ret,
            'current_price': closes.iloc[-1],
            'ma20': closes.iloc[-20:].mean()
        }
    if not scores:
        return
    valid_candidates = {k: v for k, v in scores.items() if v['ret'] > 0 and v['current_price'] > v['ma20']}
    if not valid_candidates:
        log.info("【宏观避险】风险资产全线破位，本周全仓退守国债: %s" % g.bond_etf)
        target_to_hold = g.bond_etf
    else:
        target_to_hold = max(valid_candidates.items(), key=lambda x: x[1]['score'])[0]
    current_positions = list(context.portfolio.positions.keys())
    for security in current_positions:
        if security != target_to_hold:
            order_target(security, 0)
            log.info("--- 平仓调出: %s" % security)
    if target_to_hold and target_to_hold not in current_positions:
        order_target_value(target_to_hold, context.portfolio.total_value)
        log.info("+++ 满仓调入: %s" % target_to_hold)
