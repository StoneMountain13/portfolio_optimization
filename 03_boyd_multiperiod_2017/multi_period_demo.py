"""Boyd et al. (2017) 多周期组合优化的初学者版本。

我们用“买入变量 + 卖出变量”表示交易，从而显式计算线性换手成本，
并比较只看一期的 myopic 策略与向前规划四期的 MPO 策略。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ASSETS = ["US_Equity", "Intl_Equity", "Bonds", "Gold", "REITs"]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# 每期最大持仓和最大双边换手。
MAX_WEIGHT = 0.45
MAX_TURNOVER = 0.35

# 目标函数参数。收益和协方差均按“月”计。
RISK_AVERSION = 6.0
LINEAR_COST = np.array([0.0008, 0.0010, 0.0003, 0.0009, 0.0011])
IMPACT_COEFFICIENT = np.array([0.018, 0.022, 0.008, 0.020, 0.026])


def example_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回当前权重、未来四期收益预测、未来四期协方差。"""

    current_weights = np.array([0.32, 0.18, 0.28, 0.10, 0.12])

    # 每行是一月，每列是一个资产的月度预期收益。
    forecasts = np.array(
        [
            [0.010, 0.006, 0.0025, 0.004, 0.008],
            [0.009, 0.007, 0.0025, 0.004, 0.007],
            [0.004, 0.008, 0.0025, 0.005, 0.005],
            [0.003, 0.009, 0.0025, 0.005, 0.004],
        ]
    )

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
    monthly_covariance = (
        correlation * np.outer(annual_vol, annual_vol) / 12.0
    )

    # 让远期风险略有变化，模拟风险预测随期限变化。
    covariance_path = np.stack(
        [monthly_covariance * scale for scale in [1.00, 1.05, 1.10, 1.08]]
    )
    return current_weights, forecasts, covariance_path


def unpack_decision(
    decision: np.ndarray,
    horizon: int,
    n_assets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把一维求解变量还原成 H×N 的买入和卖出矩阵。"""

    split = horizon * n_assets
    buys = decision[:split].reshape(horizon, n_assets)
    sells = decision[split:].reshape(horizon, n_assets)
    return buys, sells


def weights_from_trades(
    current_weights: np.ndarray,
    buys: np.ndarray,
    sells: np.ndarray,
) -> np.ndarray:
    """由每期交易递推未来权重。"""

    net_trades = buys - sells
    return current_weights + np.cumsum(net_trades, axis=0)


def optimize_plan(
    current_weights: np.ndarray,
    forecasts: np.ndarray,
    covariance_path: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """求解给定预测期的交易计划。"""

    horizon, n_assets = forecasts.shape
    n_variables = 2 * horizon * n_assets
    initial = np.zeros(n_variables)  # 不交易是天然可行的起点

    def calculate_path(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buys, sells = unpack_decision(decision, horizon, n_assets)
        weights = weights_from_trades(current_weights, buys, sells)
        return buys, sells, weights

    def objective(decision: np.ndarray) -> float:
        buys, sells, weights = calculate_path(decision)
        net_trades = buys - sells

        total_value = 0.0
        for t in range(horizon):
            expected_return = forecasts[t] @ weights[t]
            variance_penalty = (
                0.5
                * RISK_AVERSION
                * (weights[t] @ covariance_path[t] @ weights[t])
            )
            linear_transaction_cost = LINEAR_COST @ (buys[t] + sells[t])
            market_impact = IMPACT_COEFFICIENT @ np.square(net_trades[t])
            total_value += (
                expected_return
                - variance_penalty
                - linear_transaction_cost
                - market_impact
            )
        return float(-total_value)  # scipy 做最小化，所以取负号

    def budget_constraint(decision: np.ndarray) -> np.ndarray:
        """每一期权重之和必须等于 1。"""

        _, _, weights = calculate_path(decision)
        return weights.sum(axis=1) - 1.0

    def lower_weight_constraint(decision: np.ndarray) -> np.ndarray:
        """长仓：所有权重必须不小于 0。"""

        _, _, weights = calculate_path(decision)
        return weights.ravel()

    def upper_weight_constraint(decision: np.ndarray) -> np.ndarray:
        """集中度控制：所有权重不超过 MAX_WEIGHT。"""

        _, _, weights = calculate_path(decision)
        return (MAX_WEIGHT - weights).ravel()

    def turnover_constraint(decision: np.ndarray) -> np.ndarray:
        """每期双边换手 sum(|trade|) 不超过上限。"""

        buys, sells, _ = calculate_path(decision)
        return MAX_TURNOVER - (buys + sells).sum(axis=1)

    constraints = [
        {"type": "eq", "fun": budget_constraint},
        {"type": "ineq", "fun": lower_weight_constraint},
        {"type": "ineq", "fun": upper_weight_constraint},
        {"type": "ineq", "fun": turnover_constraint},
    ]

    # 每个买入/卖出变量非负，单资产单期最多交易 50% 的组合净值。
    bounds = [(0.0, 0.50)] * n_variables
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 4_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"多周期优化失败：{result.message}")

    buys, sells, weights = calculate_path(result.x)
    trades = buys - sells

    # 求解后自行检查约束残差。生产系统不能只看 result.success。
    max_budget_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    max_turnover = float(np.max((buys + sells).sum(axis=1)))
    if max_budget_error > 1e-6 or weights.min() < -1e-6:
        raise RuntimeError("求解器返回的解没有通过独立约束检查。")

    return {
        "weights": weights,
        "trades": trades,
        "turnover": (buys + sells).sum(axis=1),
        "objective": -float(result.fun),
        "max_budget_error": max_budget_error,
        "max_turnover": max_turnover,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    current_weights, forecasts, covariance_path = example_inputs()

    # 单周期只使用第一行预测；MPO 使用全部四行预测。
    single_period = optimize_plan(
        current_weights, forecasts[:1], covariance_path[:1]
    )
    multi_period = optimize_plan(current_weights, forecasts, covariance_path)

    period_names = [f"Month_{i + 1}" for i in range(len(forecasts))]
    weight_table = pd.DataFrame(
        multi_period["weights"], index=period_names, columns=ASSETS
    )
    weight_table.index.name = "period"
    weight_table.to_csv(
        OUTPUT_DIR / "multi_period_weights.csv", float_format="%.8f"
    )

    trade_table = pd.DataFrame(
        multi_period["trades"], index=period_names, columns=ASSETS
    )
    trade_table["two_sided_turnover"] = multi_period["turnover"]
    trade_table.index.name = "period"
    trade_table.to_csv(
        OUTPUT_DIR / "multi_period_trades.csv", float_format="%.8f"
    )

    comparison = pd.DataFrame(
        {
            "current_weight": current_weights,
            "myopic_first_weight": single_period["weights"][0],
            "mpo_first_weight": multi_period["weights"][0],
            "myopic_first_trade": single_period["trades"][0],
            "mpo_first_trade": multi_period["trades"][0],
        },
        index=ASSETS,
    )
    comparison.index.name = "asset"
    comparison.to_csv(
        OUTPUT_DIR / "first_trade_comparison.csv", float_format="%.8f"
    )

    ax = weight_table.plot(marker="o", figsize=(10, 6))
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Multi-Period Planned Portfolio Weights")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "multi_period_plan.png", dpi=180)
    plt.close(fig)

    print("\n当前权重、单周期第一步与多周期第一步：")
    print(comparison.round(4))
    print("\n多周期计划权重：")
    print(weight_table.round(4))
    print("\n多周期每期双边换手：")
    print(pd.Series(multi_period["turnover"], index=period_names).round(4))


if __name__ == "__main__":
    main()

