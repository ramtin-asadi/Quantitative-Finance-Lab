from __future__ import annotations

import numpy as np
import pandas as pd


def effective_number_of_holdings(weights: pd.DataFrame | pd.Series) -> float | pd.Series:
    """Compute the effective number of holdings implied by portfolio weights.

    The effective number of holdings is defined as ``1 / sum(w_i**2)``. It is
    the reciprocal of the Herfindahl-Hirschman concentration index and can be
    interpreted as the number of equally weighted positions that would produce
    the same concentration as the supplied portfolio.

    Parameters
    ----------
    weights : pandas.Series or pandas.DataFrame
        Portfolio weights. A Series is treated as a single weight vector. A
        DataFrame is treated as a panel of weight vectors, with one portfolio
        per row and assets in columns. Missing weights are treated as zero.

    Returns
    -------
    float or pandas.Series
        Effective number of holdings. Returns a scalar for a Series input and
        a Series indexed like the input rows for a DataFrame input. If all
        weights are zero, the result is ``NaN``.

    Notes
    -----
    The calculation does not require weights to sum exactly to one. However,
    the interpretation as an effective number of positions is most meaningful
    for normalized long-only weights.
    """

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
    """Compute Herfindahl-Hirschman weight concentration.

    The concentration measure is ``sum(w_i**2)``. It is low for diversified
    portfolios and high when capital is concentrated in a small number of
    positions. The reciprocal of this value is the effective number of holdings.

    Parameters
    ----------
    weights : pandas.Series or pandas.DataFrame
        Portfolio weights. A Series is treated as one portfolio. A DataFrame is
        treated as a time series or panel of portfolios, with portfolios in rows
        and assets in columns. Missing values are filled with zero.

    Returns
    -------
    float or pandas.Series
        Concentration value. Returns a scalar for a Series input and a Series
        indexed like the input rows for a DataFrame input.

    Notes
    -----
    For a fully invested equal-weight portfolio of ``n`` assets, the
    concentration is ``1 / n``. For a single-name portfolio, the concentration
    is ``1``.
    """

    w = weights.astype(float).fillna(0.0)
    if isinstance(w, pd.Series):
        return float((w**2).sum()) if not w.empty else np.nan
    return (w**2).sum(axis=1).rename("HHI")


def risk_contribution(weights, cov_ann, tickers=None) -> pd.Series:
    """Compute volatility risk contributions for a single portfolio.

    The function decomposes annualized portfolio volatility into additive
    asset-level contributions using ``w_i * (Sigma w)_i / sigma_p``, where
    ``Sigma`` is the annualized covariance matrix and ``sigma_p`` is portfolio
    volatility.

    Parameters
    ----------
    weights : array-like or pandas.Series
        Portfolio weights. If a Series is supplied, its index is used as the
        output labels. Otherwise, labels are taken from ``tickers`` or generated
        as ``a0``, ``a1``, ...
    cov_ann : array-like
        Annualized covariance matrix with shape ``(n_assets, n_assets)``.
    tickers : sequence of str, optional
        Asset labels to use when ``weights`` is array-like rather than a Series.

    Returns
    -------
    pandas.Series
        Asset-level volatility risk contributions, indexed by asset label. The
        sum of the returned values is approximately the portfolio volatility.

    Notes
    -----
    Risk contributions are expressed in volatility units, not percentages. To
    obtain fractional risk contributions, divide the result by the sum of the
    returned Series.
    """

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
