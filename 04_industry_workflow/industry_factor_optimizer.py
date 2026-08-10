"""教学版业界多因子组合优化器。

目标：最大化 alpha - 主动风险惩罚 - 线性交易成本 - 二次冲击。
约束：满仓、长仓、个股上限、换手、行业主动权重、风格主动暴露。

所有样例数值均为教学用途，不构成任何证券观点。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

FACTOR_NAMES = ["beta", "size", "value", "momentum", "quality"]

# 示例假定一亿美元组合；该数值只用于把交易权重和 ADV 连接起来。
PORTFOLIO_NAV_USD = 100_000_000.0
RISK_AVERSION = 5.0
MAX_TURNOVER = 0.25
SECTOR_ACTIVE_BOUND = 0.04

# 不同风格因子的主动暴露上下限。
# 动量是本例 alpha 的重要来源，所以限额较宽；Beta 限额较严。
STYLE_ACTIVE_BOUNDS = {
    "beta": 0.05,
    "size": 0.10,
    "value": 0.10,
    "momentum": 0.25,
    "quality": 0.12,
}


def load_and_validate_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取数据并进行最基本的生产前校验。"""

    universe = pd.read_csv(DATA_DIR / "sample_universe.csv")
    factor_covariance = pd.read_csv(
        DATA_DIR / "factor_covariance.csv", index_col="factor"
    )

    required_columns = {
        "ticker",
        "sector",
        "benchmark_weight",
        "current_weight",
        "alpha_annual",
        "adv_usd",
        "cost_bps",
        "specific_vol_annual",
        *FACTOR_NAMES,
    }
    missing = required_columns - set(universe.columns)
    if missing:
        raise ValueError(f"sample_universe.csv 缺少字段：{sorted(missing)}")
    if universe["ticker"].duplicated().any():
        raise ValueError("ticker 必须唯一。")
    if abs(universe["benchmark_weight"].sum() - 1.0) > 1e-10:
        raise ValueError("benchmark_weight 之和必须等于 1。")
    if abs(universe["current_weight"].sum() - 1.0) > 1e-10:
        raise ValueError("current_weight 之和必须等于 1。")

    # 强制按指定顺序排列，防止“矩阵数值正确但证券/因子顺序错位”。
    factor_covariance = factor_covariance.loc[FACTOR_NAMES, FACTOR_NAMES]
    covariance_values = factor_covariance.to_numpy(dtype=float)
    if not np.allclose(covariance_values, covariance_values.T, atol=1e-12):
        raise ValueError("因子协方差矩阵必须对称。")
    if np.linalg.eigvalsh(covariance_values).min() < -1e-10:
        raise ValueError("因子协方差矩阵不是半正定矩阵。")
    return universe, factor_covariance


def build_stock_covariance(
    universe: pd.DataFrame,
    factor_covariance: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 Sigma = B F B' + D，并返回 Sigma 与暴露矩阵 B。"""

    exposures = universe[FACTOR_NAMES].to_numpy(dtype=float)
    factor_cov = factor_covariance.to_numpy(dtype=float)
    specific_variance = np.square(
        universe["specific_vol_annual"].to_numpy(dtype=float)
    )
    stock_covariance = (
        exposures @ factor_cov @ exposures.T + np.diag(specific_variance)
    )

    # 浮点计算可能产生极小非对称误差，显式对称化。
    stock_covariance = 0.5 * (stock_covariance + stock_covariance.T)
    if np.linalg.eigvalsh(stock_covariance).min() <= 0.0:
        raise ValueError("股票协方差矩阵应为正定；请检查特异风险。")
    return stock_covariance, exposures


def optimize_portfolio(
    universe: pd.DataFrame,
    stock_covariance: np.ndarray,
    exposures: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """求解单期、多因子、相对基准的组合优化问题。"""

    n_assets = len(universe)
    benchmark = universe["benchmark_weight"].to_numpy(dtype=float)
    current = universe["current_weight"].to_numpy(dtype=float)
    alpha = universe["alpha_annual"].to_numpy(dtype=float)

    # 单边线性成本：bps 转换为收益小数。
    linear_cost = universe["cost_bps"].to_numpy(dtype=float) / 10_000.0

    # 简化二次冲击系数：组合越大、证券 ADV 越小，冲击越大。
    # 生产环境需要用真实成交数据校准，而不是照抄这个公式。
    adv = universe["adv_usd"].to_numpy(dtype=float)
    impact_coefficient = 0.04 * PORTFOLIO_NAV_USD / adv

    # 决策变量前 N 个为 buys，后 N 个为 sells，均非负。
    initial = np.zeros(2 * n_assets)

    def unpack(decision: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        buys = decision[:n_assets]
        sells = decision[n_assets:]
        weights = current + buys - sells
        return buys, sells, weights

    def objective(decision: np.ndarray) -> float:
        buys, sells, weights = unpack(decision)
        trades = buys - sells
        active = weights - benchmark

        expected_alpha = alpha @ weights
        active_variance = active @ stock_covariance @ active
        transaction_cost = linear_cost @ (buys + sells)
        market_impact = impact_coefficient @ np.square(trades)

        utility = (
            expected_alpha
            - 0.5 * RISK_AVERSION * active_variance
            - transaction_cost
            - market_impact
        )
        return float(-utility)

    def budget(decision: np.ndarray) -> float:
        return float(unpack(decision)[2].sum() - 1.0)

    def long_only(decision: np.ndarray) -> np.ndarray:
        return unpack(decision)[2]

    # 个股上限同时考虑绝对集中度和相对基准的放大倍数。
    asset_upper = np.minimum(
        0.18, 2.0 * benchmark + 0.03
    )

    def asset_upper_constraint(decision: np.ndarray) -> np.ndarray:
        return asset_upper - unpack(decision)[2]

    def turnover_constraint(decision: np.ndarray) -> float:
        buys, sells, _ = unpack(decision)
        return float(MAX_TURNOVER - np.sum(buys + sells))

    constraints: list[dict] = [
        {"type": "eq", "fun": budget},
        {"type": "ineq", "fun": long_only},
        {"type": "ineq", "fun": asset_upper_constraint},
        {"type": "ineq", "fun": turnover_constraint},
    ]

    # 行业约束：每个行业相对基准的主动权重在 ±4% 内。
    for sector in sorted(universe["sector"].unique()):
        mask = (universe["sector"].to_numpy() == sector).astype(float)

        def sector_upper_constraint(
            decision: np.ndarray,
            mask: np.ndarray = mask,
        ) -> float:
            active = unpack(decision)[2] - benchmark
            return float(SECTOR_ACTIVE_BOUND - mask @ active)

        def sector_lower_constraint(
            decision: np.ndarray,
            mask: np.ndarray = mask,
        ) -> float:
            active = unpack(decision)[2] - benchmark
            return float(SECTOR_ACTIVE_BOUND + mask @ active)

        constraints.extend(
            [
                {"type": "ineq", "fun": sector_upper_constraint},
                {"type": "ineq", "fun": sector_lower_constraint},
            ]
        )

    # 风格约束：B' * active_weight 必须落在给定区间内。
    for factor_index, factor_name in enumerate(FACTOR_NAMES):
        factor_exposure = exposures[:, factor_index].copy()
        bound = STYLE_ACTIVE_BOUNDS[factor_name]

        def style_upper_constraint(
            decision: np.ndarray,
            factor_exposure: np.ndarray = factor_exposure,
            bound: float = bound,
        ) -> float:
            active = unpack(decision)[2] - benchmark
            return float(bound - factor_exposure @ active)

        def style_lower_constraint(
            decision: np.ndarray,
            factor_exposure: np.ndarray = factor_exposure,
            bound: float = bound,
        ) -> float:
            active = unpack(decision)[2] - benchmark
            return float(bound + factor_exposure @ active)

        constraints.extend(
            [
                {"type": "ineq", "fun": style_upper_constraint},
                {"type": "ineq", "fun": style_lower_constraint},
            ]
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 0.30)] * (2 * n_assets),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 6_000, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"组合优化失败：{result.message}")

    buys, sells, weights = unpack(result.x)
    return {
        "weights": weights,
        "trades": buys - sells,
        "turnover": float(np.sum(buys + sells)),
        "utility": -float(result.fun),
        "asset_upper": asset_upper,
    }


def style_active_exposures(
    weights: np.ndarray,
    benchmark: np.ndarray,
    exposures: np.ndarray,
) -> np.ndarray:
    """计算 B' * (w - w_benchmark)。"""

    return exposures.T @ (weights - benchmark)


def active_risk_breakdown(
    weights: np.ndarray,
    benchmark: np.ndarray,
    exposures: np.ndarray,
    factor_covariance: np.ndarray,
    specific_variance: np.ndarray,
) -> dict[str, float]:
    """把主动方差分为因子方差和特异方差。"""

    active = weights - benchmark
    factor_active = exposures.T @ active
    factor_variance = float(
        factor_active @ factor_covariance @ factor_active
    )
    specific_var = float(np.sum(np.square(active) * specific_variance))
    total_variance = factor_variance + specific_var
    return {
        "factor_variance": factor_variance,
        "specific_variance": specific_var,
        "total_active_variance": total_variance,
        "tracking_error": np.sqrt(max(total_variance, 0.0)),
    }


def build_diagnostics(
    universe: pd.DataFrame,
    optimized: np.ndarray,
    trades: np.ndarray,
    asset_upper: np.ndarray,
    exposures: np.ndarray,
) -> pd.DataFrame:
    """独立于优化器重新计算约束，避免只相信求解状态。"""

    benchmark = universe["benchmark_weight"].to_numpy(dtype=float)
    active = optimized - benchmark
    rows = [
        {
            "check": "budget_error",
            "value": abs(optimized.sum() - 1.0),
            "limit": 1e-6,
            "passed": abs(optimized.sum() - 1.0) <= 1e-6,
        },
        {
            "check": "minimum_weight_violation",
            "value": max(0.0, -float(optimized.min())),
            "limit": 1e-6,
            "passed": optimized.min() >= -1e-6,
        },
        {
            "check": "maximum_weight_violation",
            "value": max(0.0, float(np.max(optimized - asset_upper))),
            "limit": 1e-6,
            "passed": np.max(optimized - asset_upper) <= 1e-6,
        },
        {
            "check": "two_sided_turnover",
            "value": float(np.sum(np.abs(trades))),
            "limit": MAX_TURNOVER + 1e-6,
            "passed": np.sum(np.abs(trades)) <= MAX_TURNOVER + 1e-6,
        },
    ]

    for sector in sorted(universe["sector"].unique()):
        mask = universe["sector"].to_numpy() == sector
        exposure = float(active[mask].sum())
        rows.append(
            {
                "check": f"sector_active::{sector}",
                "value": abs(exposure),
                "limit": SECTOR_ACTIVE_BOUND + 1e-6,
                "passed": abs(exposure) <= SECTOR_ACTIVE_BOUND + 1e-6,
            }
        )

    styles = exposures.T @ active
    for name, exposure in zip(FACTOR_NAMES, styles):
        bound = STYLE_ACTIVE_BOUNDS[name]
        rows.append(
            {
                "check": f"style_active::{name}",
                "value": abs(float(exposure)),
                "limit": bound + 1e-6,
                "passed": abs(exposure) <= bound + 1e-6,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    universe, factor_covariance_df = load_and_validate_inputs()
    stock_covariance, exposures = build_stock_covariance(
        universe, factor_covariance_df
    )
    result = optimize_portfolio(universe, stock_covariance, exposures)

    benchmark = universe["benchmark_weight"].to_numpy(dtype=float)
    current = universe["current_weight"].to_numpy(dtype=float)
    optimized = np.asarray(result["weights"])
    trades = np.asarray(result["trades"])

    portfolio_table = universe[
        ["ticker", "sector", "alpha_annual", "benchmark_weight", "current_weight"]
    ].copy()
    portfolio_table["optimized_weight"] = optimized
    portfolio_table["trade"] = trades
    portfolio_table["active_weight"] = optimized - benchmark
    portfolio_table.to_csv(
        OUTPUT_DIR / "optimized_portfolio.csv", index=False, float_format="%.8f"
    )

    style_table = pd.DataFrame(
        {
            "pretrade_active_exposure": style_active_exposures(
                current, benchmark, exposures
            ),
            "optimized_active_exposure": style_active_exposures(
                optimized, benchmark, exposures
            ),
            "absolute_limit": [STYLE_ACTIVE_BOUNDS[x] for x in FACTOR_NAMES],
        },
        index=FACTOR_NAMES,
    )
    style_table.index.name = "factor"
    style_table.to_csv(
        OUTPUT_DIR / "active_exposures.csv", float_format="%.8f"
    )

    sector_rows = []
    for sector in sorted(universe["sector"].unique()):
        mask = universe["sector"].to_numpy() == sector
        sector_rows.append(
            {
                "sector": sector,
                "pretrade_active_weight": float((current - benchmark)[mask].sum()),
                "optimized_active_weight": float((optimized - benchmark)[mask].sum()),
                "absolute_limit": SECTOR_ACTIVE_BOUND,
            }
        )
    pd.DataFrame(sector_rows).to_csv(
        OUTPUT_DIR / "sector_active_weights.csv", index=False, float_format="%.8f"
    )

    specific_variance = np.square(
        universe["specific_vol_annual"].to_numpy(dtype=float)
    )
    risk_summary = pd.DataFrame(
        {
            "pretrade": active_risk_breakdown(
                current,
                benchmark,
                exposures,
                factor_covariance_df.to_numpy(dtype=float),
                specific_variance,
            ),
            "optimized": active_risk_breakdown(
                optimized,
                benchmark,
                exposures,
                factor_covariance_df.to_numpy(dtype=float),
                specific_variance,
            ),
        }
    )
    risk_summary.to_csv(OUTPUT_DIR / "risk_summary.csv", float_format="%.8f")

    diagnostics = build_diagnostics(
        universe,
        optimized,
        trades,
        np.asarray(result["asset_upper"]),
        exposures,
    )
    diagnostics.to_csv(
        OUTPUT_DIR / "constraint_diagnostics.csv", index=False, float_format="%.8f"
    )
    if not diagnostics["passed"].all():
        failed = diagnostics.loc[~diagnostics["passed"], "check"].tolist()
        raise RuntimeError(f"独立约束检查失败：{failed}")

    # 上图比较三组持仓，下图展示交易方向和大小。
    x = np.arange(len(universe))
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].bar(x - width, benchmark, width, label="Benchmark")
    axes[0].bar(x, current, width, label="Current")
    axes[0].bar(x + width, optimized, width, label="Optimized")
    axes[0].set_ylabel("Weight")
    axes[0].set_title("Industry-Style Factor Portfolio Construction")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    colors = np.where(trades >= 0.0, "#2b8a3e", "#c92a2a")
    axes[1].bar(x, trades, color=colors)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Trade weight")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(universe["ticker"], rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "portfolio_and_trades.png", dpi=180)
    plt.close(fig)

    print("\n优化后的持仓与交易：")
    print(
        portfolio_table[
            ["ticker", "benchmark_weight", "current_weight", "optimized_weight", "trade"]
        ].round(4)
    )
    print("\n风格主动暴露：")
    print(style_table.round(4))
    print("\n主动风险分解：")
    print(risk_summary.round(6))
    print(f"\n双边换手：{result['turnover']:.4f}")
    print("全部独立约束检查通过。")


if __name__ == "__main__":
    main()

