"""依次运行四个教学案例。

从项目根目录执行：
    python run_all.py
"""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "01_markowitz_1952" / "markowitz_demo.py",
    ROOT / "02_black_litterman_1992" / "black_litterman_demo.py",
    ROOT / "03_boyd_multiperiod_2017" / "multi_period_demo.py",
    ROOT / "04_industry_workflow" / "industry_factor_optimizer.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n{'=' * 72}\n运行：{script.relative_to(ROOT)}")
        subprocess.run(
            [sys.executable, script.name],
            cwd=script.parent,
            check=True,
        )
    print("\n全部案例运行成功。请查看各文件夹中的 outputs/。")


if __name__ == "__main__":
    main()
