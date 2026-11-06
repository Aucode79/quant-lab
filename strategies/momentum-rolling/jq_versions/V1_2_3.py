# 策略版本：V1.2.3 (国债回归版)
# 核心逻辑：四大金刚标的池 + 夏普动量排序 + 20日均线防呆 + 周二全仓国债避险
from jqdata import *
import numpy as np
import pandas as pd
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 四大金刚风险资产
    g.etf_pool = ['159869.XSHE', '513100.XSHG', '512890.XSHG', '518880.XSHG']
    # 绝对避险底仓：重新引入国债 ETF
    g.bond_etf = '511280.XSHG'
    g.momentum_days = 20
    # 设置佣金万分之 1.15
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.000115, close_commission=0.000115, 
        min_commission=5
    ), type='fund')
    # 周二 14:50 执行轮动
    run_weekly(rotate_etf, weekday=1, time='14:50')
def rotate_etf(context):
    # 拉取包含国债在内的所有数据
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
    # 防呆过滤：收益必须为正，且站上 MA20
    valid_candidates = {k: v for k, v in scores.items() if v['ret'] > 0 and v['current_price'] > v['ma20']}
    # ================= 核心变动 =================
    if not valid_candidates:
        # 如果风险资产全灭，目标标的设定为国债 ETF
        log.info("【宏观避险】风险资产全线破位，本周全仓退守国债: %s" % g.bond_etf)
        target_to_hold = g.bond_etf
    else:
        target_to_hold = max(valid_candidates.items(), key=lambda x: x[1]['score'])[0]
        best_ret = round(valid_candidates[target_to_hold]['ret'] * 100, 2)
        log.info("【强者恒强】本周调入标的: %s | 20日涨幅: %s%%" % (target_to_hold, best_ret))
    # 极简调仓执行
    current_positions = list(context.portfolio.positions.keys())
    for security in current_positions:
        if security != target_to_hold:
            order_target(security, 0)
            log.info("--- 平仓调出: %s" % security)
    if target_to_hold and target_to_hold not in current_positions:
        order_target_value(target_to_hold, context.portfolio.total_value)
        log.info("+++ 满仓调入: %s" % target_to_hold)
