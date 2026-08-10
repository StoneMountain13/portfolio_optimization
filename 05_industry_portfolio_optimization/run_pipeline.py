"""一键运行：生成数据 -> PCA/Barra 风险 -> 优化 -> 独立检查 -> 可视化。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data_generation import generate_market, save_data
from src.risk_models import barra_style_covariance, pca_covariance
from src.optimizer import build_report, solve_active_portfolio


def run_one(model: str, universe: pd.DataFrame, returns: pd.DataFrame, output_dir: Path):
    """对一个风险模型执行全流程并保存可审计输出。"""
    if model == "pca":
        covariance, diagnostic = pca_covariance(returns, n_components=5)
    else:
        covariance, diagnostic = barra_style_covariance(returns, universe)
    weights, status = solve_active_portfolio(universe, covariance)
    report, checks, sectors = build_report(universe, weights, covariance)
    prefix = model
    report.to_csv(output_dir / f"{prefix}_portfolio_report.csv", index=False)
    checks.to_csv(output_dir / f"{prefix}_constraint_check.csv", index=False)
    sectors.to_csv(output_dir / f"{prefix}_sector_exposure.csv", index=False)
    np.save(output_dir / f"{prefix}_covariance.npy", covariance)

    # 易读图：交易前、基准、交易后权重。只显示绝对权重最大的 20 个名称避免图像拥挤。
    view = report.assign(abs_active=lambda x: x.active_weight.abs()).nlargest(20, "abs_active").sort_values("active_weight")
    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(len(view))
    ax.barh(y - 0.22, view["benchmark_weight"], height=0.22, label="benchmark")
    ax.barh(y, view["old_weight"], height=0.22, label="old")
    ax.barh(y + 0.22, view["target_weight"], height=0.22, label="target")
    ax.set_yticks(y, view["ticker"])
    ax.set_xlabel("weight")
    ax.set_title(f"{diagnostic['model']} risk model: top active positions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_weights.png", dpi=160)
    plt.close(fig)

    result = {"risk_model": diagnostic["model"], "solver_status": status}
    result.update(dict(zip(checks.metric, checks.value)))
    if model == "pca":
        result["pca_explained_variance_top5"] = float(np.sum(diagnostic["explained_ratio"]))
    return result


def main():
    parser = argparse.ArgumentParser(description="运行业界风格组合优化教学管线")
    parser.add_argument("--risk-model", choices=["pca", "barra", "both"], default="both")
    args = parser.parse_args()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    universe, returns = generate_market()
    save_data(universe, returns, ROOT / "data")
    models = ["pca", "barra"] if args.risk_model == "both" else [args.risk_model]
    summary = [run_one(model, universe, returns, output_dir) for model in models]
    pd.DataFrame(summary).to_csv(output_dir / "model_comparison.csv", index=False)
    print("完成。请查看 outputs/model_comparison.csv、各风险模型的约束检查 CSV 和权重图。")


if __name__ == "__main__":
    main()
