"""PCA 与 Barra 风格的协方差估计。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _nearest_psd(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """通过抬升特征值使协方差半正定，避免数值误差破坏凸优化。"""
    eigvals, eigvecs = np.linalg.eigh((matrix + matrix.T) / 2)
    return (eigvecs * np.maximum(eigvals, floor)) @ eigvecs.T


def pca_covariance(returns: pd.DataFrame, n_components: int = 5, periods_per_year: int = 252):
    """用“前 K 个主成分 + 对角特异风险”估计年化协方差。

    特征分解而非直接使用样本协方差，便于明确保留与丢弃哪些风险方向。
    被 PCA 因子解释后的剩余方差逐股票保留在对角项，避免低秩模型错误地产生零风险套利。
    """
    sample_cov = np.cov(returns.values, rowvar=False, ddof=1) * periods_per_year
    eigenvalues, eigenvectors = np.linalg.eigh(sample_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    k = min(n_components, len(eigenvalues))
    loadings = eigenvectors[:, :k]
    factor_cov = np.diag(eigenvalues[:k])
    systematic = loadings @ factor_cov @ loadings.T
    residual_var = np.clip(np.diag(sample_cov - systematic), 1e-6, None)
    covariance = _nearest_psd(systematic + np.diag(residual_var))
    diagnostics = {
        "model": "PCA", "loadings": loadings,
        "factor_cov": factor_cov, "specific_var": residual_var,
        "explained_ratio": eigenvalues[:k] / eigenvalues.sum(),
    }
    return covariance, diagnostics


def barra_style_covariance(returns: pd.DataFrame, universe: pd.DataFrame, ridge: float = 1e-5,
                           periods_per_year: int = 252):
    """估计可解释的 Barra 风格协方差：Sigma = X F X' + D。

    每日横截面回归的因变量是该日股票收益，解释变量为风格与行业暴露。
    加一列截距并在回归后移除截距，不把“全市场平均回报”误记为某个行业因子。
    岭项只为教学样本中的数值稳定；生产中更常见的是加权稳健回归与更严格的行业约束。
    """
    styles = universe[["size", "value", "momentum", "quality"]].to_numpy()
    sector = pd.get_dummies(universe["sector"]).reindex(columns=sorted(universe["sector"].unique()), fill_value=0).to_numpy()
    # 少放一个行业因子，以截距作为基准行业，避免 dummy-variable trap。
    factor_names = ["size", "value", "momentum", "quality"] + sorted(universe["sector"].unique())[:-1]
    x = np.column_stack([np.ones(len(universe)), styles, sector[:, :-1]])
    xtx_inv_xt = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T)
    # 逐日横截面回归。factors 含截距，residuals 是每只股票的时间序列残差。
    beta_all = (xtx_inv_xt @ returns.to_numpy().T).T
    fitted = beta_all @ x.T
    residuals = returns.to_numpy() - fitted
    factor_returns = beta_all[:, 1:]
    factor_cov = np.cov(factor_returns, rowvar=False, ddof=1) * periods_per_year
    specific_var = np.clip(np.var(residuals, axis=0, ddof=1) * periods_per_year, 1e-6, None)
    exposures = x[:, 1:]
    covariance = _nearest_psd(exposures @ factor_cov @ exposures.T + np.diag(specific_var))
    diagnostics = {
        "model": "Barra-style", "loadings": exposures, "factor_cov": factor_cov,
        "specific_var": specific_var, "factor_names": factor_names,
        "factor_returns": factor_returns,
    }
    return covariance, diagnostics
