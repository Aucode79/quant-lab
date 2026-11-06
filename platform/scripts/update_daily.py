#!/usr/bin/env python3
"""
每日增量更新数据管线。

一键运行所有更新：
    python update_daily.py

也可以分步运行：
    python update_daily.py --step kline      # 只更新日K线
    python update_daily.py --step basic       # 只更新股票列表

建议通过 cron / 计划任务每天 17:00（收盘后）自动运行。
"""

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def run_script(script_name: str, extra_args: list[str] | None = None) -> bool:
    """运行同目录下的Python脚本，返回是否成功。"""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"🚀 运行: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="每日数据更新")
    parser.add_argument("--step", type=str, choices=["all", "basic", "kline"],
                        default="all", help="更新步骤 (默认: all)")
    args = parser.parse_args()

    today = date.today()
    print(f"📅 更新日期: {today}")

    success = True

    if args.step in ("all", "basic"):
        # 更新股票列表（每月运行一次即可，但每天跑也不会有问题）
        if not run_script("fetch_stock_basic.py"):
            success = False

    if args.step in ("all", "kline"):
        # 增量更新最近30天的日K线
        end_date = today.strftime("%Y-%m-%d")
        start_date = (today - timedelta(days=60)).strftime("%Y-%m-%d")
        if not run_script("fetch_kline_daily.py",
                          ["--start", start_date, "--end", end_date]):
            success = False

    if success:
        print(f"\n✅ 全部更新完成 ({today})")
    else:
        print(f"\n⚠️  部分更新失败，请检查日志 ({today})")
        sys.exit(1)


if __name__ == "__main__":
    main()
