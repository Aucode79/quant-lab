# 策略版本：V1.2.2 (大类资产截面轮动 - 化繁为简终极版)
# 核心逻辑：四大金刚标的池 + 夏普动量排序 + 20日均线防呆 + 周二轮动空仓避险
from jqdata import *
import numpy as np
import pandas as pd
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 资产池重构：回归正交性，剔除高度相关的多余 Beta
    g.etf_pool = [
        '159869.XSHE', # 游戏 ETF (本土高弹主攻)
        '513100.XSHG', # 纳指 ETF (海外科技对冲)
        '512890.XSHG', # 红利低波 ETF (本土价值防守)
        '518880.XSHG'  # 黄金 ETF (抗通胀独立避险)
    ]
    # 动量计算周期：20个交易日 (约1个月)
    g.momentum_days = 20
    # 真实费率设置：万分之1.15 (ETF免印花税)
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.000115, close_commission=0.000115, 
        min_commission=5
    ), type='fund')
    # 定时任务：每周二下午 14:50 执行 (避开周四砸盘与周五周末博弈)
    run_weekly(rotate_etf, weekday=1, time='14:50')
def rotate_etf(context):
    # 1. 批量拉取数据 (20天均线 + 20天波动率所需)
    hist = history(g.momentum_days + 1, '1d', 'close', g.etf_pool)
    scores = {}
    for target in g.etf_pool:
        closes = hist[target]
        if len(closes) < g.momentum_days + 1 or pd.isna(closes.iloc[-1]):
            continue
        # 计算动量因子 (20日绝对收益率)
        ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        # 计算风险因子 (20日年化波动率)
        daily_rets = closes.pct_change().dropna()
        volatility = daily_rets.std() * np.sqrt(252)
        # 计算风险调整后得分 (夏普评分)
        volatility = max(volatility, 0.0001) # 避免除零错误
        score = ret / volatility
        scores[target] = {
            'score': score, 
            'ret': ret,
            'current_price': closes.iloc[-1],
            'ma20': closes.iloc[-20:].mean()
        }
    if not scores:
        return
    # 2. 筛选满足“防呆条件”的资产：(收益必须 > 0) 且 (当前价格必须站上 MA20 均线)
    valid_candidates = {k: v for k, v in scores.items() if v['ret'] > 0 and v['current_price'] > v['ma20']}
    target_to_hold = None
    if not valid_candidates:
        # 如果四大金刚全线熄火或破位，执行绝对空仓
        log.info("【宏观避险】风险资产全线破位，本周清仓躲避 (实盘将自动吃逆回购利息)")
        target_to_hold = None
    else:
        # 在有效的资产中选出夏普评分最高（涨得又快又稳）的一只
        target_to_hold = max(valid_candidates.items(), key=lambda x: x[1]['score'])[0]
        best_ret = round(valid_candidates[target_to_hold]['ret'] * 100, 2)
        log.info("【强者恒强】本周调入标的: %s | 20日涨幅: %s%%" % (target_to_hold, best_ret))
    # 3. 极简调仓执行逻辑
    current_positions = list(context.portfolio.positions.keys())
    # 第一步：卖出不再持有的标的 (腾出资金)
    for security in current_positions:
        if security != target_to_hold:
            order_target(security, 0)
            log.info("--- 弱者淘汰，平仓调出: %s" % security)
    # 第二步：满仓买入目标标的 (集中火力，规避最低5元手续费摩擦)
    if target_to_hold and target_to_hold not in current_positions:
        order_target_value(target_to_hold, context.portfolio.total_value)
        log.info("+++ 满仓出击，调入: %s" % target_to_hold)
