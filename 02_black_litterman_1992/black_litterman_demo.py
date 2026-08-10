"""Black–Litterman (1992) 教学示例。

示例展示：
1. 从市场权重反推出均衡预期超额收益；
2. 用观点矩阵 P、观点收益 Q 和不确定性 Omega 融合观点；
3. 用后验预期收益做长仓组合优化；
4. 检查观点置信度变化如何改变权重。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ASSETS = ["US_Equity", "Intl_Equity", "Aggregate_Bonds", "Gold"]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# 风险厌恶越高，同样的预期收益下越不愿意承担方差。
RISK_AVERSION = 2.5

# tau 表示对均衡收益先验的不确定程度。
# 这里选择 0.05 只用于教学；实际中要与 Omega 的构造一起校准。
TAU = 0.05


def example_market_inputs() -> tuple[np.ndarray, np.ndarray]:
    """返回市场权重和年化协方差矩阵。"""

    market_weights = np.array([0.48, 0.24, 0.20, 0.08])
    annual_volatility = np.array([0.18, 0.20, 0.07, 0.16])
    correlation = np.array(
        [
            [1.00, 0.78, -0.10, 0.05],
            [0.78, 1.00, -0.05, 0.10],
            [-0.10, -0.05, 1.00, 0.15],
            [0.05, 0.10, 0.15, 1.00],
        ]
    )
    covariance = correlation * np.outer(annual_volatility, annual_volatility)
    return market_weights, covariance


def implied_equilibrium_returns(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """反向优化：Pi = delta * Sigma * w_market。"""

    return risk_aversion * covariance @ market_weights


def view_uncertainty_from_confidence(
    p: np.ndarray,
    covariance: np.ndarray,
    tau: float,
    confidence: np.ndarray,
) -> np.ndarray:
    """把 0~1 的直观置信度映射为对角观点误差矩阵 Omega。

    基准观点方差为 diag(P * tau*Sigma * P')。
    使用 (1-c)/c 作为缩放：
    - c 越高，Omega 越小，观点越有话语权；
    - c 越低，Omega 越大，结果越接近市场先验。

    这是一种透明的教学映射，不是 Black–Litterman 唯一规定的做法。
    """

    if np.any((confidence <= 0.0) | (confidence >= 1.0)):
        raise ValueError("confidence 必须严格位于 0 和 1 之间。")
    base_variance = np.diag(p @ (tau * covariance) @ p.T)
    scaled_variance = base_variance * (1.0 - confidence) / confidence
    return np.diag(scaled_variance)


def black_litterman_posterior(
    prior_mean: np.ndarray,
    covariance: np.ndarray,
    tau: float,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 BL 后验均值及“均值估计的不确定性协方差”。

    这里用解线性方程代替显式矩阵求逆，数值稳定性通常更好。
    """

    prior_covariance = tau * covariance
    prior_precision = np.linalg.inv(prior_covariance)
    view_precision = np.linalg.inv(omega)

    posterior_precision = prior_precision + p.T @ view_precision @ p
    right_hand_side = prior_precision @ prior_mean + p.T @ view_precision @ q
    posterior_mean = np.linalg.solve(posterior_precision, right_hand_side)
    posterior_mean_uncertainty = np.linalg.inv(posterior_precision)
    return posterior_mean, posterior_mean_uncertainty


def mean_variance_utility_weights(
    mean: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    """最大化 mu'w - delta/2 * w'Sigma w，且长仓、满仓。"""

    n_assets = len(mean)
    initial = np.repeat(1.0 / n_assets, n_assets)

    def negative_utility(weights: np.ndarray) -> float:
        expected_return = mean @ weights
        risk_penalty = 0.5 * risk_aversion * (weights @ covariance @ weights)
        return float(-expected_return + risk_penalty)

    result = minimize(
        negative_utility,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_assets,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"组合优化失败：{result.message}")
    return result.x


def posterior_for_confidence(
    prior: np.ndarray,
    covariance: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    """辅助函数：由一组置信度直接得到 BL 后验均值。"""

    omega = view_uncertainty_from_confidence(p, covariance, TAU, confidence)
    posterior, _ = black_litterman_posterior(
        prior, covariance, TAU, p, q, omega
    )
    return posterior


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    market_weights, covariance = example_market_inputs()
    prior = implied_equilibrium_returns(covariance, market_weights, RISK_AVERSION)

    # 观点 1：美国股票比国际股票多赚 1%。
    # 观点 2：黄金比债券多赚 1%。
    p = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    q = np.array([0.01, 0.01])
    confidence = np.array([0.50, 0.55])

    posterior = posterior_for_confidence(
        prior, covariance, p, q, confidence
    )

    # 先验优化用于验证：在没有长仓边界绑定时，应接近市场组合。
    prior_weights = mean_variance_utility_weights(
        prior, covariance, RISK_AVERSION
    )
    posterior_weights = mean_variance_utility_weights(
        posterior, covariance, RISK_AVERSION
    )

    expected_returns = pd.DataFrame(
        {
            "equilibrium_prior": prior,
            "bl_posterior": posterior,
            "change": posterior - prior,
        },
        index=ASSETS,
    )
    expected_returns.index.name = "asset"
    expected_returns.to_csv(
        OUTPUT_DIR / "bl_expected_returns.csv", float_format="%.8f"
    )

    weights = pd.DataFrame(
        {
            "market": market_weights,
            "prior_optimized": prior_weights,
            "bl_posterior_optimized": posterior_weights,
        },
        index=ASSETS,
    )
    weights.index.name = "asset"
    weights.to_csv(OUTPUT_DIR / "bl_weights.csv", float_format="%.8f")

    # 只改变第一条观点的置信度，观察组合怎样平滑偏离市场。
    sensitivity_rows = []
    for first_confidence in [0.25, 0.50, 0.75, 0.95]:
        current_confidence = np.array([first_confidence, 0.55])
        current_posterior = posterior_for_confidence(
            prior, covariance, p, q, current_confidence
        )
        current_weights = mean_variance_utility_weights(
            current_posterior, covariance, RISK_AVERSION
        )
        for asset, weight in zip(ASSETS, current_weights):
            sensitivity_rows.append(
                {
                    "first_view_confidence": first_confidence,
                    "asset": asset,
                    "weight": weight,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        OUTPUT_DIR / "confidence_sensitivity.csv", index=False, float_format="%.8f"
    )

    ax = weights.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Black-Litterman: Market Prior and Posterior Portfolio")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "bl_weights.png", dpi=180)
    plt.close(fig)

    print("\n均衡先验与 BL 后验预期超额收益：")
    print(expected_returns.round(4))
    print("\n市场、先验优化、观点融合后的权重：")
    print(weights.round(4))


if __name__ == "__main__":
    main()
