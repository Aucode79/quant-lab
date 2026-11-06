#!/usr/bin/env python3
"""
回测结果可视化工具

支持:
  - 净值曲线 + 回撤图
  - 参数热力图（网格搜索结果）
  - 月度/年度收益分析

Usage:
    python visualize_results.py --results ../results/backtest_xxx.duckdb
    python visualize_results.py --grid-search ../results/opt_results.duckdb
"""

import argparse
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Seaborn 风格
sns.set_style("whitegrid")
sns.set_palette("muted")


def plot_equity_curve(daily_df: pd.DataFrame, output_path: str | None = None):
    """
    绘制净值曲线 + 回撤区域。
    daily_df 需要包含: trade_date, total_value
    """
    if "trade_date" not in daily_df.columns:
        daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])

    daily_df = daily_df.sort_values("trade_date")

    # 计算净值
    initial = daily_df["total_value"].iloc[0]
    daily_df["nav"] = daily_df["total_value"] / initial

    # 计算回撤
    daily_df["peak"] = daily_df["nav"].cummax()
    daily_df["drawdown"] = (daily_df["peak"] - daily_df["nav"]) / daily_df["peak"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # 上图：净值曲线
    ax1.plot(daily_df["trade_date"], daily_df["nav"],
             linewidth=1.2, color="#1f77b4", label="策略净值")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_ylabel("累计净值")
    ax1.set_title("策略净值曲线")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # 标注最高点和最终点
    max_idx = daily_df["nav"].idxmax()
    ax1.annotate(f"最高: {daily_df['nav'].iloc[max_idx]:.2f}",
                 xy=(daily_df["trade_date"].iloc[max_idx],
                     daily_df["nav"].iloc[max_idx]),
                 xytext=(10, 10), textcoords="offset points",
                 fontsize=9, color="green",
                 arrowprops=dict(arrowstyle="->", color="green", alpha=0.5))

    # 下图：回撤
    ax2.fill_between(daily_df["trade_date"], 0, daily_df["drawdown"] * 100,
                     color="#d62728", alpha=0.3, label="回撤")
    ax2.plot(daily_df["trade_date"], daily_df["drawdown"] * 100,
             linewidth=0.8, color="#d62728", alpha=0.8)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_xlabel("日期")

    max_dd = daily_df["drawdown"].max() * 100
    ax2.set_title(f"回撤曲线 (最大回撤: {max_dd:.2f}%)")
    ax2.invert_yaxis()  # 回撤向下

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 图表已保存: {output_path}")
    else:
        plt.show()


def plot_heatmap(results_df: pd.DataFrame, output_path: str | None = None):
    """
    绘制参数热力图（二维参数空间）。
    results_df 需要包含: param1, param2, metric (如 sharpe_ratio)
    """
    if len(results_df.columns) < 3:
        print("⚠️  热力图需要至少2个参数维度和1个指标列")
        return

    # 识别参数列和指标列
    metric_cols = ["sharpe_ratio", "annual_return", "max_drawdown", "calmar_ratio"]
    param_cols = [c for c in results_df.columns if c not in metric_cols]

    if len(param_cols) < 2:
        print("⚠️  一维参数空间，使用折线图代替")
        return

    param_x = param_cols[0]
    param_y = param_cols[1]
    metric = "sharpe_ratio"

    # 透视表
    pivot = results_df.pivot_table(
        values=metric,
        index=param_y,
        columns=param_x,
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, cbar_kws={"label": "夏普比率"})
    ax.set_title(f"参数热力图: {param_x} × {param_y}")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 热力图已保存: {output_path}")
    else:
        plt.show()


def plot_monthly_returns(daily_df: pd.DataFrame, output_path: str | None = None):
    """绘制月度收益率热力图。"""
    if "trade_date" not in daily_df.columns:
        daily_df["trade_date"] = pd.to_datetime(daily_df["trade_date"])

    daily_df = daily_df.set_index("trade_date")
    if "daily_return" not in daily_df.columns and "total_value" in daily_df.columns:
        daily_df["daily_return"] = daily_df["total_value"].pct_change()

    monthly = daily_df["daily_return"].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )

    # 透视表: 行=年份, 列=月份
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values
    })
    pivot = monthly_df.pivot_table(
        values="return", index="year", columns="month", aggfunc="sum"
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn",
                center=0, ax=ax, cbar_kws={"label": "收益率 (%)"})
    ax.set_title("月度收益率热力图 (%)")
    ax.set_ylabel("年份")
    ax.set_xlabel("月份")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"📊 月度收益热力图已保存: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="回测结果可视化")
    parser.add_argument("--results", type=str, help="回测结果 DuckDB 文件路径")
    parser.add_argument("--grid-search", type=str, help="网格搜索结果路径 (CSV)")
    parser.add_argument("--output", type=str, help="图表输出路径")
    parser.add_argument("--type", type=str,
                        choices=["equity", "heatmap", "monthly", "all"],
                        default="all", help="图表类型")
    args = parser.parse_args()

    if not args.results and not args.grid_search:
        print("❌ 请指定 --results 或 --grid-search", file=sys.stderr)
        sys.exit(1)

    # ── 回测结果图表 ──
    if args.results:
        con = duckdb.connect(args.results, read_only=True)

        # 尝试读取每日记录
        try:
            daily_df = con.execute("SELECT * FROM backtest_daily ORDER BY trade_date").fetchdf()
            if not daily_df.empty:
                if args.type in ("equity", "all"):
                    out = args.output or None
                    plot_equity_curve(daily_df, out)
                if args.type in ("monthly", "all"):
                    monthly_out = args.output.replace(".png", "_monthly.png") if args.output else None
                    plot_monthly_returns(daily_df, monthly_out)
        except Exception as e:
            print(f"⚠️  读取每日数据失败: {e}")

        con.close()

    # ── 网格搜索热力图 ──
    if args.grid_search:
        csv_path = Path(args.grid_search)
        if csv_path.exists():
            results_df = pd.read_csv(csv_path)
            if not results_df.empty and args.type in ("heatmap", "all"):
                heatmap_out = args.output if args.output else None
                plot_heatmap(results_df, heatmap_out)
        else:
            print(f"⚠️  文件不存在: {csv_path}")


if __name__ == "__main__":
    main()
