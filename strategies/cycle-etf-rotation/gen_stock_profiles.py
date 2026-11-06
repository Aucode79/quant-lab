"""
个股资料库生成 v2 — 十类别 + 注册表全量
=====================================================
数据源: 标的池_注册表_v2_20260913.csv (1057只)
        profile_cninfo.parquet (1109只概况)
        stock_basic_bs.parquet (上市/退市)
输出:   个股资料/<类别>/<名称>_<代码>.txt  (每只一个)
"""
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

BASE = Path("C:/Users/qihen/Desktop/量化研究/量化策略/3.周期股")
REF = "C:/Users/qihen/Desktop/量化研究/quant-system/data/reference"
OUT = BASE / "个股资料"
GEN_DATE = "2026-09-13"

reg = pd.read_csv(BASE / "标的池_注册表_v2_20260913.csv")
prof = pd.read_parquet(f"{REF}/profile_cninfo.parquet")
basic = pd.read_parquet(f"{REF}/stock_basic_bs.parquet")

prof = prof.drop_duplicates("ts_code", keep="last")
reg = reg.merge(prof[["ts_code", "公司全称", "cninfo行业", "主营业务", "经营范围"]],
                on="ts_code", how="left")
basic["ipoDate"] = pd.to_datetime(basic["ipoDate"], errors="coerce")
basic["outDate"] = pd.to_datetime(basic["outDate"], errors="coerce")
# 注册表CSV里已含ipoDate/outDate, 先删避免列名冲突(_x/_y)
reg = reg.drop(columns=["ipoDate", "outDate"], errors="ignore")
reg = reg.merge(basic[["ts_code", "ipoDate", "outDate"]], on="ts_code", how="left")

GROUPS = ["煤炭", "钢铁", "有色", "建材", "化工", "石化",
          "橡胶塑料", "生猪养殖", "航运港口", "造纸"]
LAYER_NOTE = {
    "L1核心": "数据双确认(官方分类+行为暴露≥0.452+三窗稳定+流动性)",
    "L2扩展": "数据确认(暴露达标+稳定+流动性, 未到核心线)",
    "L3候选": "官方分类入册, 行为分未达标 — 收集层默认保留, 待人工/AI研判",
    "L3候选(漂移)": "行为暴露不稳定(类巨化案例: 可能是新周期或退潮), 重点研判对象",
    "L3候选(流动性)": "暴露达标但流动性不足",
}


def sanitize(name):
    return re.sub(r'[*:?<>|/\\"]', "", str(name)).strip()


def fmt3(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return None


def one(v):
    s = str(v)
    return "-" if s in ("nan", "None", "") or pd.isna(v) else s


def status(r):
    if pd.notna(r.get("outDate")):
        return f"已退市 ({pd.Timestamp(r['outDate']).date()})"
    return "在市"


# 清空旧库重建 (v1只有6类345只, v2结构不同)
import shutil
if OUT.exists():
    shutil.rmtree(OUT)

n_files = 0
for g in GROUPS:
    (OUT / g).mkdir(parents=True, exist_ok=True)
    sub = reg[reg["行业类别"] == g].sort_values(
        "max暴露", ascending=False, na_position="last")
    for _, r in sub.iterrows():
        exp_cells = []
        for x in GROUPS[:6]:
            v = fmt3(r.get(f"暴露_{x}"))
            exp_cells.append(f"{x} {v if v else '—'}")
        exp_str = " | ".join(exp_cells)
        new_cells = []
        for x in ["生猪", "航运", "造纸", "橡胶"]:
            v = fmt3(r.get(f"暴露_{x}"))
            if v:
                new_cells.append(f"{x} {v}")
        new_str = " | ".join(new_cells) if new_cells else "(未覆盖)"
        stab_new = []
        for x in ["生猪", "航运", "造纸", "橡胶"]:
            s = r.get(f"{x}稳定性")
            if isinstance(s, str) and s.strip():
                stab_new.append(f"{x}{s}")
        stab_new_str = " ".join(stab_new) if stab_new else ""
        mx = fmt3(r.get("max暴露"))
        mv = one(r.get("总市值亿"))
        mv = "(未获取)" if mv == "-" else f"{mv}亿"
        layer = r.get("层级", "L3候选")
        layer_note = LAYER_NOTE.get(layer, "")
        scored = "是" if pd.notna(r.get("max暴露")) else "否(历史不足)"
        new_cat = g in ("橡胶塑料", "生猪养殖", "航运港口", "造纸")
        exp_caption = ("六维工业周期暴露(对新类别仅供参考)" if new_cat
                       else "六维周期暴露")

        txt = f"""{'='*56}
{r['名称']} ({r['ts_code']})
{'='*56}
生成: {GEN_DATE} | 数据截至: 2026-06-30 | 仅供研究

【归属】
  行业类别: {g}
  子类注释: {one(r.get('子类注释'))}
  池层级: {layer}
    ({layer_note})
  入册来源: {one(r.get('来源'))}
  证监会行业: {one(r.get('证监会行业'))}
  巨潮行业: {one(r.get('cninfo行业'))}
  公司全称: {one(r.get('公司全称'))}
  上市日期: {pd.Timestamp(r['ipoDate']).date() if pd.notna(r.get('ipoDate')) else '未知'}   状态: {status(r)}

【市场】
  总市值: {mv}
  中位日成交额(近252有效日): {one(r.get('中位日成交额(万)'))}万

【周期暴露】({exp_caption}, 已剥离沪深300)
  {exp_str}
  最强: {one(r.get('最强组'))} {mx if mx else '—'}
  已评分: {scored}
  窗口稳定性: {one(r.get('达标窗口数'))} (全期{'√' if str(r.get('全期达标'))=='True' else '×'} 早期{'√' if str(r.get('早期达标'))=='True' else '×'} 晚期{'√' if str(r.get('晚期达标'))=='True' else '×'})

【新类别暴露】(生猪/航运/造纸/橡胶种子篮子, 2026-09-13 增评)
  {new_str}{('   窗口: ' + stab_new_str) if stab_new_str else ''}

【主营业务】
  {one(r.get('主营业务'))}

【经营范围】
  {one(r.get('经营范围'))}

【备注】
  暴露 = 该股日残差收益与对应行业种子篮子的相关系数; 分数只作证据注释,
  不构成入册门槛 (注册表v2: 收集层全量入册, 去留由人工/AI后置研判)。
  稳定性低 ≠ 剔除: 可能是概念噪声, 也可能是新周期前段 (如制冷剂2023后启动)。
"""
        fn = OUT / g / f"{sanitize(r['名称'])}_{r['ts_code'].replace('.','-')}.txt"
        fn.write_text(txt, encoding="utf-8")
        n_files += 1

readme = f"""个股资料库 v2 ({GEN_DATE} 生成)
{'='*50}

结构: 个股资料/<行业类别>/<名称>_<代码>.txt
范围: 标的池注册表 v2 全量 {n_files} 只 (数据截至 2026-06-30)

类别 ({len(GROUPS)}): {' / '.join(GROUPS)}
  六大工业类为已建池类别; 橡胶塑料/生猪养殖/航运港口/造纸为收集期新增(尚未评种子)。

层级语义 (v2 — 收集/证据/决策三层分离):
  L1核心 184 | L2扩展 161 | L3候选(含漂移/流动性) 约712
  L1/L2 = 数据双确认; L3 = 官方分类入册待研判。去留由后续人工+AI决定。

分类决策记录:
  1. 半导体、新能源(光伏/新能源车/锂电中下游)剔除: 成长性大于周期性 (用户 {GEN_DATE})
  2. 收集阶段宁多勿少 — "只怕少不怕多, 后续慢慢删" (用户 {GEN_DATE})
     暴露分数/稳定性只作注释, 绝不当门槛 (v1教训: 阈值+稳定门挡掉502/689只,
     含万华/恒力/巨化/多氟多; 巨化0.409过阈值仍被"三窗稳定"误杀)
  3. 橡胶/生猪按用户要求入册; 独立产品周期(猪肉/黄金/锂钴/航运)不与工业周期混池
  4. 白酒不纳入

重建脚本: gen_stock_profiles.py (跑一次即全量重建)
"""
(OUT / "00_README.txt").write_text(readme, encoding="utf-8")

print(f"[完成] {n_files} 只 → {OUT}")
for g in GROUPS:
    print(f"  {g}: {len(list((OUT / g).glob('*.txt')))} 只")
