"""Markowitz (1952) 均值—方差组合优化教学示例。

特点：
1. 不需要联网；
2. 固定随机种子，任何人运行都能复现；
3. 只依赖 numpy、pandas、scipy 和 matplotlib；
4. 每个关键步骤都有中文注释。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 在没有图形界面的服务器上也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ------------------------------
# 1. 基础设置
# ------------------------------
RANDOM_SEED = 20260800
TRADING_DAYS = 252
N_OBSERVATIONS = 126  # 约半年日频数据
RISK_FREE_RATE = 0.02  # 年化无风险利率，仅用于最大夏普组合
ASSETS = ["US_Equity", "Intl_Equity", "Bonds", "Gold", "REITs"]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_synthetic_returns() -> pd.DataFrame:
    """生成带有现实感相关结构的模拟日收益。

    这里先指定年化预期收益、年化波动率和相关系数，再换算为日频参数。
    模拟数据只用于教学，不代表对这些资产的真实预测。
    """

    annual_mean = np.array([0.09, 0.08, 0.035, 0.05, 0.075])
    annual_vol = np.array([0.18, 0.20, 0.07, 0.16, 0.17])

    correlation = np.array(
        [
            [1.00, 0.75, -0.10, 0.05, 0.60],
            [0.75, 1.00, -0.05, 0.10, 0.55],
            [-0.10, -0.05, 1.00, 0.15, 0.05],
            [0.05, 0.10, 0.15, 1.00, 0.10],
            [0.60, 0.55, 0.05, 0.10, 1.00],
        ]
    )

    # covariance_ij = correlation_ij * vol_i * vol_j
    annual_covariance = correlation * np.outer(annual_vol, annual_vol)
    daily_mean = annual_mean / TRADING_DAYS
    daily_covariance = annual_covariance / TRADING_DAYS

    rng = np.random.default_rng(RANDOM_SEED)
    simulated = rng.multivariate_normal(
        mean=daily_mean,
        cov=daily_covariance,
        size=N_OBSERVATIONS,
    )
    return pd.DataFrame(simulated, columns=ASSETS)


def annualized_estimates(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """由日收益估计年化均值和协方差。"""

    mean = returns.mean().to_numpy() * TRADING_DAYS
    covariance = returns.cov().to_numpy() * TRADING_DAYS
    return mean, covariance


def portfolio_statistics(
    weights: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float, float]:
    """返回组合的年化收益、波动率和夏普比率。"""

    expected_return = float(mean @ weights)
    variance = float(weights @ covariance @ weights)
    volatility = np.sqrt(max(variance, 0.0))
    sharpe = (expected_return - RISK_FREE_RATE) / volatility
    return expected_return, volatility, sharpe


def solve_minimum_variance(
    mean: np.ndarray,
    covariance: np.ndarray,
    target_return: float | None = None,
) -> np.ndarray:
    """求长仓满仓条件下的最小方差组合。

    如果 target_return 为 None，就是全局最小方差组合；
    否则再加入组合收益不低于 target_return 的约束。
    """

    n_assets = len(mean)
    initial = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, 1.0)] * n_assets

    constraints: list[dict] = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    if target_return is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda w: mean @ w - target_return}
        )

    result = minimize(
        fun=lambda w: w @ covariance @ w,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"最小方差优化失败：{result.message}")
    return result.x


def solve_maximum_sharpe(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """求长仓满仓条件下的最大夏普组合。"""

    n_assets = len(mean)
    initial = np.repeat(1.0 / n_assets, n_assets)

    def negative_sharpe(weights: np.ndarray) -> float:
        _, _, sharpe = portfolio_statistics(weights, mean, covariance)
        return -sharpe

    result = minimize(
        fun=negative_sharpe,
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"最大夏普优化失败：{result.message}")
    return result.x


def build_efficient_frontier(
    mean: np.ndarray,
    covariance: np.ndarray,
) -> pd.DataFrame:
    """对多个目标收益重复优化，得到有效前沿上的点。"""

    # 长仓组合的收益不会低于最小单资产均值，也不会高于最大单资产均值。
    target_returns = np.linspace(mean.min(), mean.max() * 0.995, 60)
    rows = []
    for target in target_returns:
        weights = solve_minimum_variance(mean, covariance, target)
        ret, vol, sharpe = portfolio_statistics(weights, mean, covariance)
        rows.append(
            {
                "target_return": target,
                "realized_expected_return": ret,
                "volatility": vol,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    returns = make_synthetic_returns()
    mean, covariance = annualized_estimates(returns)

    equal_weight = np.repeat(1.0 / len(ASSETS), len(ASSETS))
    global_minimum_variance = solve_minimum_variance(mean, covariance)
    maximum_sharpe = solve_maximum_sharpe(mean, covariance)

    portfolios = {
        "Equal_Weight": equal_weight,
        "Global_Min_Variance": global_minimum_variance,
        "Maximum_Sharpe": maximum_sharpe,
    }

    weights_table = pd.DataFrame(portfolios, index=ASSETS)
    weights_table.index.name = "asset"
    weights_table.to_csv(OUTPUT_DIR / "portfolio_weights.csv", float_format="%.8f")

    metric_rows = []
    for name, weights in portfolios.items():
        ret, vol, sharpe = portfolio_statistics(weights, mean, covariance)
        metric_rows.append(
            {
                "portfolio": name,
                "expected_return": ret,
                "volatility": vol,
                "sharpe": sharpe,
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("portfolio")
    metrics.to_csv(OUTPUT_DIR / "portfolio_metrics.csv", float_format="%.8f")

    frontier = build_efficient_frontier(mean, covariance)
    frontier.to_csv(OUTPUT_DIR / "efficient_frontier.csv", index=False)

    # 绘制有效前沿，并标出三个容易理解的组合。
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        frontier["volatility"],
        frontier["realized_expected_return"],
        linewidth=2.2,
        label="Efficient frontier",
    )
    for name, row in metrics.iterrows():
        ax.scatter(row["volatility"], row["expected_return"], s=70, label=name)
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Annualized expected return")
    ax.set_title("Markowitz Mean-Variance Efficient Frontier")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "efficient_frontier.png", dpi=180)
    plt.close(fig)

    print("\n样本估计的年化收益：")
    print(pd.Series(mean, index=ASSETS).round(4))
    print("\n三个组合的权重：")
    print(weights_table.round(4))
    print("\n组合指标：")
    print(metrics.round(4))


if __name__ == "__main__":
    main()
