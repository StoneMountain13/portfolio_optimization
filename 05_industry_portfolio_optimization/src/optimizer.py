"""将 alpha、风险、成本和投资政策约束交给优化器。

生产环境常用 CVXPY/MOSEK、Gurobi 或 OSQP 建模并求解凸问题；为保证这个教学包在
普通 Python 环境无需商业/额外锥求解器也能运行，代码用 SciPy 的 SLSQP 直接求解同一目标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def solve_active_portfolio(universe: pd.DataFrame, covariance: np.ndarray, risk_aversion: float = 7.0,
                           linear_cost_aversion: float = 1.0, impact_aversion: float = 1.0):
    """求相对基准的 long-only 主动组合。

    风格、行业约束均对 active weights = w - benchmark 施加；这样能控制产品相对基准的风险，
    而不会把基准自身的行业/风格特征误当作主动押注。
    """
    n = len(universe)
    b = universe["benchmark_weight"].to_numpy()
    w_old = universe["old_weight"].to_numpy()
    alpha = universe["alpha"].to_numpy()
    linear_cost = universe["linear_cost"].to_numpy()
    impact_cost = universe["impact_cost"].to_numpy()

    # scipy 最小化，故取目标函数的相反数。L1 换手项在 0 处不可微，SLSQP 仍可处理本教学规模；
    # 生产中会引入辅助变量并交给锥/二次规划求解器，以得到更强的数值与最优性保障。
    def loss(w: np.ndarray) -> float:
        active, trade = w - b, w - w_old
        utility = (alpha @ active - risk_aversion * (active @ covariance @ active)
                   - linear_cost_aversion * (linear_cost @ np.abs(trade))
                   - impact_aversion * np.sum(impact_cost * trade ** 2))
        return -float(utility)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                   {"type": "ineq", "fun": lambda w: 0.35 - np.sum(np.abs(w - w_old))}]
    # 以两个不等式表达 |a| <= limit。
    for sector in sorted(universe["sector"].unique()):
        mask = (universe["sector"].to_numpy() == sector).astype(float)
        constraints += [
            {"type": "ineq", "fun": lambda w, m=mask: 0.05 - m @ (w - b)},
            {"type": "ineq", "fun": lambda w, m=mask: 0.05 + m @ (w - b)},
        ]
    for factor in ["size", "value", "momentum", "quality"]:
        exposure = universe[factor].to_numpy()
        constraints += [
            {"type": "ineq", "fun": lambda w, e=exposure: 0.10 - e @ (w - b)},
            {"type": "ineq", "fun": lambda w, e=exposure: 0.10 + e @ (w - b)},
        ]
    # active 单票限额与 long-only / 单票上限可统一成 bounds。
    lower = np.maximum(0.0, b - 0.03)
    upper = np.minimum(0.06, b + 0.03)
    result = minimize(loss, x0=w_old, method="SLSQP", bounds=list(zip(lower, upper)),
                      constraints=constraints, options={"maxiter": 3_000, "ftol": 1e-11, "disp": False})
    if not result.success:
        raise RuntimeError(f"优化失败，SLSQP 信息：{result.message}")
    return result.x, f"SLSQP optimal ({result.message})"


def build_report(universe: pd.DataFrame, weights: np.ndarray, covariance: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """独立于 cvxpy 约束对象重新计算组合与约束报表。"""
    report = universe.copy()
    report["target_weight"] = weights
    report["active_weight"] = report["target_weight"] - report["benchmark_weight"]
    report["trade_weight"] = report["target_weight"] - report["old_weight"]
    report["estimated_linear_cost"] = report["linear_cost"] * report["trade_weight"].abs()
    report["estimated_impact_cost"] = report["impact_cost"] * report["trade_weight"] ** 2
    active = report["active_weight"].to_numpy()
    metrics = {
        "budget_error": report["target_weight"].sum() - 1,
        "min_weight": report["target_weight"].min(),
        "max_weight": report["target_weight"].max(),
        "max_abs_active_stock": report["active_weight"].abs().max(),
        "one_way_turnover": report["trade_weight"].abs().sum() / 2,
        "gross_turnover": report["trade_weight"].abs().sum(),
        "annualized_tracking_error": float(np.sqrt(active @ covariance @ active)),
        "estimated_linear_cost": report["estimated_linear_cost"].sum(),
        "estimated_impact_cost": report["estimated_impact_cost"].sum(),
    }
    for sector, group in report.groupby("sector"):
        metrics[f"sector_active_{sector}"] = group["active_weight"].sum()
    for factor in ["size", "value", "momentum", "quality"]:
        metrics[f"style_active_{factor}"] = float(report[factor].to_numpy() @ active)
    checks = pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())})
    sector_report = report.groupby("sector", as_index=False).agg(
        benchmark_weight=("benchmark_weight", "sum"), target_weight=("target_weight", "sum"),
        active_weight=("active_weight", "sum"),
    )
    return report, checks, sector_report
