# ==============================================================================
V1.2
# 导入聚宽函数库
from jqdata import *
import numpy as np
import pandas as pd
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    # 核心鱼池：弱相关大类资产配置
    g.etf_pool = [
        '159869.XSHE', # 游戏 ETF (高弹性进攻)
        '513100.XSHG', # 纳指 ETF (美股科技对冲)
        '512890.XSHG', # 红利低波 ETF (A股防守)
        '518880.XSHG'  # 黄金 ETF (抗通胀/避险)
    ]
    # 绝对避险底仓
    g.bond_etf = '511280.XSHG' # 十年期国债 ETF
    # 动量参考周期
    g.momentum_days = 20
    set_order_cost(OrderCost(
        open_tax=0, close_tax=0, 
        open_commission=0.00025, close_commission=0.00025, 
        close_today_commission=0, min_commission=5
    ), type='fund')
    # 每周五 14:50 执行轮动
    run_weekly(rotate_etf, weekday=5, time='14:50')
def rotate_etf(context):
    all_etfs = g.etf_pool + [g.bond_etf]
    # 拉取过去 21 天的数据用于计算涨幅和波动率
    hist = history(g.momentum_days + 1, '1d', 'close', all_etfs)
    scores = {}
    for target in g.etf_pool:
        closes = hist[target]
        if len(closes) < g.momentum_days + 1 or pd.isna(closes.iloc[-1]):
            continue
        # 1. 计算 20 日收益率
        ret = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
        # 2. 计算 20 日年化波动率
        daily_rets = closes.pct_change().dropna()
        volatility = daily_rets.std() * np.sqrt(252)
        # 3. 计算风险调整后动量得分 (类似夏普比率)，避免0除
        if volatility == 0:
            volatility = 0.0001
        score = ret / volatility
        scores[target] = {
            'score': score, 
            'ret': ret,
            'current_price': closes.iloc[-1],
            'ma20': closes.iloc[-20:].mean()
        }
    if not scores:
        return
    # 选出得分最高（风险调整后动量最强）的资产
    best_etf = max(scores.items(), key=lambda x: x[1]['score'])[0]
    best_data = scores[best_etf]
    target_to_hold = None
    # --- V7.0 核心风控：双重一票否决 ---
    # 条件1：近20天必须是赚钱的 (绝对收益 > 0)
    # 条件2：当前价格必须站在20日均线之上 (一票否决急跌坠落)
    if best_data['ret'] <= 0 or best_data['current_price'] < best_data['ma20']:
        log.info("【宏观避险】最强标的 (%s) 趋势破位或收益为负，本周空仓避险至: 国债 (511280)" % best_etf)
        target_to_hold = g.bond_etf
    else:
        target_to_hold = best_etf
        log.info("【强者恒强】本周轮动王: %s | 收益率: %s%%" % (best_etf, round(best_data['ret']*100, 2)))
    # 执行调仓指令
    current_positions = list(context.portfolio.positions.keys())
    for security in current_positions:
        if security != target_to_hold:
            order_target(security, 0)
            log.info("--- 平仓调出: %s" % security)
    if target_to_hold not in current_positions:
        # 资金依然满仓单只最强标的，以对抗 5 元最低手续费
        order_target_value(target_to_hold, context.portfolio.total_value)
        log.info("+++ 满仓调入: %s" % target_to_hold)
