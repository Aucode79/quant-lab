# 策略版本：V1.2.1 (截面动量轮动 - 周二 14:50 版)
# 核心逻辑：夏普动量排序 + 20日均线防呆 + 空仓现金避险 (模拟逆回购)
from jqdata import *
import numpy as np
import pandas as pd
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 资产池重构：覆盖本土、海外、红利、黄金与中概龙头
    g.etf_pool = [
        '159869.XSHE', # 游戏 ETF (本土高弹)
        '513100.XSHG', # 纳指 ETF (全球科技)
        '513050.XSHG', # 中概互联 ETF (阿里腾讯等)
        '512880.XSHG', # 证券 ETF (牛市旗手) - 代替红利增加攻击性，也可换回 512890
        '518880.XSHG'  # 黄金 ETF (避险/抗通胀)
    ]
    # 动量计算周期：20个交易日
    g.momentum_days = 20
    # 万 1.15 佣金设置 (ETF免印花税)
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.000115, close_commission=0.000115, 
        min_commission=5
    ), type='fund')
    # 定时任务：修改为每周二下午 14:50 (weekday=1)
    run_weekly(rotate_etf, weekday=1, time='14:50')
def rotate_etf(context):
    # 1. 批量拉取数据 (20天均线 + 20天波动率所需)
    hist = history(g.momentum_days + 1, '1d', 'close', g.etf_pool)
    scores = {}
    for target in g.etf_pool:
        closes = hist[target]
        if len(closes) < g.momentum_days + 1 or pd.isna(closes.iloc[-1]):
            continue
        # 计算动量因子 (20日收益率)
        ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        # 计算风险因子 (20日年化波动率)
        daily_rets = closes.pct_change().dropna()
        volatility = daily_rets.std() * np.sqrt(252)
        # 计算风险调整后得分 (夏普评分)
        volatility = max(volatility, 0.0001) # 避免除零
        score = ret / volatility
        scores[target] = {
            'score': score, 
            'ret': ret,
            'current_price': closes.iloc[-1],
            'ma20': closes.iloc[-20:].mean()
        }
    if not scores:
        return
    # 2. 筛选满足“防呆条件”的资产：(收益>0) 且 (价格>MA20)
    valid_candidates = {k: v for k, v in scores.items() if v['ret'] > 0 and v['current_price'] > v['ma20']}
    target_to_hold = None
    if not valid_candidates:
        # 如果全场熄火，执行“空仓避险”
        log.info("【宏观避险】无标的满足均线防呆条件，执行空仓 (实盘将自动执行逆回购)")
        target_to_hold = None
    else:
        # 在有效的资产中选出夏普评分最高的一只
        target_to_hold = max(valid_candidates.items(), key=lambda x: x[1]['score'])[0]
        best_ret = round(valid_candidates[target_to_hold]['ret'] * 100, 2)
        log.info("【强者恒强】调入标的: %s | 20日涨幅: %s%%" % (target_to_hold, best_ret))
    # 3. 调仓执行
    current_positions = list(context.portfolio.positions.keys())
    # 卖出不再持有的标的
    for security in current_positions:
        if security != target_to_hold:
            order_target(security, 0)
            log.info("--- 淘汰清仓: %s" % security)
    # 买入或继续持有目标标的
    if target_to_hold and target_to_hold not in current_positions:
        order_target_value(target_to_hold, context.portfolio.total_value)
        log.info("+++ 全仓出击: %s" % target_to_hold)
