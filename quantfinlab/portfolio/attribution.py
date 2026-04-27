from __future__ import annotations

import numpy as np
import pandas as pd


def effective_number_of_holdings(weights: pd.DataFrame | pd.Series) -> float | pd.Series:
    """Effective N = 1 / HHI for one weight vector or each row of a weight panel."""
    w = weights.astype(float)
    if isinstance(w, pd.Series):
        hhi = float((w.fillna(0.0) ** 2).sum())
        return float(1.0 / hhi) if hhi > 0 else np.nan
    hhi = (w.fillna(0.0) ** 2).sum(axis=1)
    return (1.0 / hhi.replace(0.0, np.nan)).rename("Effective N")


def max_weight(weights: pd.DataFrame | pd.Series) -> float | pd.Series:
    w = weights.astype(float).fillna(0.0)
    if isinstance(w, pd.Series):
        return float(w.max()) if not w.empty else np.nan
    return w.max(axis=1).rename("Max Weight")


def concentration(weights: pd.DataFrame | pd.Series) -> float | pd.Series:
    """HHI concentration for one vector or each row."""
    w = weights.astype(float).fillna(0.0)
    if isinstance(w, pd.Series):
        return float((w**2).sum()) if not w.empty else np.nan
    return (w**2).sum(axis=1).rename("HHI")


def risk_contribution(weights, cov_ann, tickers=None) -> pd.Series:
    """Volatility risk contribution for a single portfolio weight vector."""
    if isinstance(weights, pd.Series):
        labels = weights.index
        w = weights.to_numpy(dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        labels = pd.Index(tickers if tickers is not None else [f"a{i}" for i in range(len(w))])
    cov = np.asarray(cov_ann, dtype=float)
    sigma_w = cov @ w
    port_var = float(w @ sigma_w)
    port_vol = float(np.sqrt(max(port_var, 1e-18)))
    return pd.Series((w * sigma_w) / port_vol, index=labels, dtype=float)


def turnover_summary(turnover: pd.Series | pd.DataFrame) -> pd.Series:
    vals = turnover.astype(float)
    if isinstance(vals, pd.DataFrame):
        vals = vals.sum(axis=1)
    return pd.Series(
        {
            "Avg Turnover": float(vals.mean()) if len(vals) else 0.0,
            "Total Turnover": float(vals.sum()) if len(vals) else 0.0,
        }
    )


__all__ = [
    "concentration",
    "effective_number_of_holdings",
    "max_weight",
    "risk_contribution",
    "turnover_summary",
]
