"""生成可复现的合成横截面数据与历史收益。

这是教学数据：为了让模型可检验，收益由一组隐藏的风格和行业因子生成。
真实项目应替换为严格点时的市场、财务和指数数据。
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


STYLE_COLUMNS = ["size", "value", "momentum", "quality"]
SECTORS = ["Finance", "Industrial", "Technology"]


def _zscore(x: np.ndarray) -> np.ndarray:
    """将截面描述子标准化为均值 0、标准差 1。"""
    return (x - x.mean()) / x.std(ddof=0)


def generate_market(n_assets: int = 40, n_days: int = 252, seed: int = 42):
    """生成一个股票池和历史日收益。

    Returns
    -------
    universe : DataFrame
        一行一只股票，包含基准权重、旧权重、风格暴露、行业、成本和 alpha。
    returns : DataFrame
        行为日期、列为股票代码的历史日收益，供风险模型在调仓日前估计。
    """
    rng = np.random.default_rng(seed)
    tickers = [f"STK{i:03d}" for i in range(1, n_assets + 1)]
    sector = np.repeat(SECTORS, [14, 13, 13])[:n_assets]
    rng.shuffle(sector)

    # 用 log 市值产生自然的市值分布，并构造市值加权基准。
    market_cap = np.exp(rng.normal(10.5, 1.0, n_assets))
    # 教学产品设为“较分散的指数”；否则极端大权重基准与 6% 单票上限会天然不可行。
    # 实务中应由产品规则决定：若跟踪集中指数，单票上限必须与指数豁免条款一起设计。
    benchmark = market_cap ** 0.25
    benchmark /= benchmark.sum()
    old_weight = np.clip(benchmark + rng.normal(0, 0.003, n_assets), 0.001, None)
    old_weight /= old_weight.sum()

    styles = np.column_stack([
        _zscore(np.log(market_cap)),
        _zscore(rng.normal(0, 1, n_assets)),
        _zscore(rng.normal(0, 1, n_assets)),
        _zscore(rng.normal(0, 1, n_assets)),
    ])
    sector_onehot = np.column_stack([(sector == s).astype(float) for s in SECTORS])

    # 流动性：市值大的一般更便宜；本例单位均为“权重改变 1 所对应的单期成本近似”。
    liquidity = _zscore(np.log(market_cap) + rng.normal(0, 0.5, n_assets))
    linear_cost = 0.0007 + 0.0010 * (liquidity.max() - liquidity) / (liquidity.max() - liquidity.min())
    impact_cost = 0.035 + 0.085 * (liquidity.max() - liquidity) / (liquidity.max() - liquidity.min())

    # 隐藏因子收益产生历史收益。第一项模拟市场共同波动；后面是风格和行业。
    exposure = np.column_stack([styles, sector_onehot[:, :2]])  # 省略一个行业避免完全共线
    true_factor_cov = np.array([
        [0.00010, 0.00001, 0, 0, 0, 0],
        [0.00001, 0.00008, 0, 0, 0, 0],
        [0, 0, 0.00009, 0, 0, 0],
        [0, 0, 0, 0.00007, 0, 0],
        [0, 0, 0, 0, 0.00008, 0],
        [0, 0, 0, 0, 0, 0.00007],
    ])
    factor_returns = rng.multivariate_normal(np.zeros(exposure.shape[1]), true_factor_cov, size=n_days)
    specific_vol = 0.007 + 0.004 * rng.random(n_assets)
    daily_returns = factor_returns @ exposure.T + rng.normal(0, specific_vol, size=(n_days, n_assets))
    dates = pd.bdate_range("2025-08-01", periods=n_days)
    returns = pd.DataFrame(daily_returns, index=dates, columns=tickers)

    # 教学 alpha：它与未来预期相关但故意加入噪声；行业效应经中性化以避免把行业 beta 当选股能力。
    raw_alpha = 0.006 * styles[:, 1] + 0.005 * styles[:, 2] + 0.003 * styles[:, 3] + rng.normal(0, 0.0025, n_assets)
    alpha = np.empty(n_assets)
    for s in SECTORS:
        mask = sector == s
        alpha[mask] = raw_alpha[mask] - raw_alpha[mask].mean()
    # 这里的 alpha 是一个月度期望收益风格的数字，风险会被从日频年化/转月频后匹配。
    universe = pd.DataFrame({
        "ticker": tickers, "sector": sector, "market_cap": market_cap,
        "benchmark_weight": benchmark, "old_weight": old_weight,
        "size": styles[:, 0], "value": styles[:, 1], "momentum": styles[:, 2], "quality": styles[:, 3],
        "linear_cost": linear_cost, "impact_cost": impact_cost, "alpha": alpha,
    })
    return universe, returns


def save_data(universe: pd.DataFrame, returns: pd.DataFrame, data_dir: Path) -> None:
    """将输入快照落盘，方便复现实验。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    universe.to_csv(data_dir / "universe_snapshot.csv", index=False)
    returns.to_csv(data_dir / "historical_returns.csv", index_label="date")
