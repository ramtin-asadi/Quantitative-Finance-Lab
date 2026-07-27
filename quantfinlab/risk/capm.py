from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.utils import (
    DEFAULT_ANNUALIZATION,
    _align_pair,
    _coerce_objects,
    _excess_returns,
    _to_datetime_if_possible,
    _to_numeric_series,
)


def capm_ols(
    y_excess: pd.Series | Sequence[float] | np.ndarray,
    x_excess: pd.Series | Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    """Estimate a one-factor CAPM regression by ordinary least squares.

    The regression is ``y_excess = alpha + beta * x_excess + error`` after aligning
    the two input series and dropping missing values.

    Parameters
    ----------
    y_excess : array-like
        Asset or strategy excess return series.
    x_excess : array-like
        Market excess return series.

    Returns
    -------
    tuple of float
        ``(alpha, beta, r2)``. All three values are ``NaN`` if fewer than three
        aligned observations are available.

    Notes
    -----
    Returns are assumed to be one-period returns in decimal units. ``alpha`` is in
    the same one-period unit as the input returns.
    """

    y = _to_numeric_series(y_excess, name="y_excess")
    x = _to_numeric_series(x_excess, name="x_excess")
    z = _align_pair(y, x)
    if len(z) < 3:
        return float("nan"), float("nan"), float("nan")
    xv = z["x"].to_numpy(dtype=float)
    yv = z["y"].to_numpy(dtype=float)
    xmat = np.column_stack([np.ones(len(xv), dtype=float), xv])
    coef = np.linalg.lstsq(xmat, yv, rcond=None)[0]
    alpha = float(coef[0])
    beta = float(coef[1])
    yhat = xmat @ coef
    ssr = float(np.sum((yv - yhat) ** 2))
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ssr / sst if sst > 1e-12 else float("nan")
    return alpha, beta, float(r2)

def rolling_beta_corr(
    returns: pd.Series | Sequence[float] | np.ndarray,
    market_ret: pd.Series | Sequence[float] | np.ndarray,
    *,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling market beta and correlation.

    Parameters
    ----------
    returns : array-like
        Asset or strategy return series.
    market_ret : array-like
        Market return series.
    window : int
        Rolling window length. Must be at least five observations.

    Returns
    -------
    tuple of pandas.Series
        ``(beta, corr)`` where beta is rolling covariance divided by rolling market
        variance and corr is rolling Pearson correlation.

    Raises
    ------
    InputError
        If ``window < 5``.
    """

    if int(window) < 5:
        raise InputError("window must be >= 5.")
    r = _to_numeric_series(returns, name="returns")
    m = _to_numeric_series(market_ret, name="market_ret")
    z = _align_pair(r, m)
    beta = z["y"].rolling(int(window)).cov(z["x"]) / z["x"].rolling(int(window)).var()
    corr = z["y"].rolling(int(window)).corr(z["x"])
    beta.name = f"beta_{int(window)}"
    corr.name = f"corr_{int(window)}"
    return beta, corr

def _normalize_windows(rolling: int | Sequence[int] | None) -> list[int]:
    if rolling is None:
        return [126, 252]
    if isinstance(rolling, int):
        vals = [126, int(rolling)]
    else:
        vals = [int(v) for v in rolling if int(v) > 1]
        if not vals:
            vals = [126, 252]
    out = sorted(set(vals))
    return out

def capm_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    market_ret: pd.Series | Sequence[float] | np.ndarray,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    rolling: int | Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build CAPM and active-risk diagnostics for one or more return series.

    For each object, the function estimates alpha, beta, R-squared, tracking error,
    information ratio, up/down capture, and systematic variance share. Optional
    rolling beta/correlation tables can also be produced.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects to analyze.
    market_ret : array-like
        Market return series.
    rf_daily : float or pandas.Series, default=0.0
        One-period risk-free rate subtracted from returns before CAPM regression.
    annualization : float, default=252.0
        Annualization factor.
    rolling : int or sequence of int, optional
        Rolling windows for beta and correlation diagnostics.

    Returns
    -------
    tuple
        ``(capm, roll)`` where ``capm`` is a summary table indexed by object name
        and ``roll`` maps object names to rolling diagnostic DataFrames.

    Notes
    -----
    Tracking error and information ratio are computed from raw active returns
    relative to the market, while alpha and beta use excess returns.
    """

    obj = _coerce_objects(objects)
    m = _to_numeric_series(market_ret, name="market_ret")
    m = _to_datetime_if_possible(m)
    windows = _normalize_windows(rolling)
    rows: list[dict[str, Any]] = []
    roll: dict[str, pd.DataFrame] = {}
    ann = float(annualization)

    for name, r in obj.items():
        y = _to_datetime_if_possible(r)
        y_ex = _excess_returns(y, rf_daily)
        m_ex = _excess_returns(m, rf_daily)
        z_ex = _align_pair(y_ex, m_ex)
        alpha_d, beta, r2 = capm_ols(z_ex["y"], z_ex["x"])
        alpha_ann = (1.0 + alpha_d) ** ann - 1.0 if alpha_d > -0.999 else float("nan")

        z_raw = _align_pair(y, m)
        active = z_raw["y"] - z_raw["x"]
        has_var = len(active) > 1 and active.std(ddof=1) > 1e-12
        te = float(active.std(ddof=1) * math.sqrt(ann)) if has_var else float("nan")
        ir = float(active.mean() / active.std(ddof=1) * math.sqrt(ann)) if has_var else float("nan")

        up = z_raw["x"] > 0
        dn = z_raw["x"] < 0
        up_den = float(z_raw.loc[up, "x"].mean()) if up.any() else float("nan")
        dn_den = float(z_raw.loc[dn, "x"].mean()) if dn.any() else float("nan")
        up_cap = (
            float(z_raw.loc[up, "y"].mean() / up_den)
            if up.sum() > 10 and np.isfinite(up_den) and abs(up_den) > 1e-12
            else float("nan")
        )
        dn_cap = (
            float(z_raw.loc[dn, "y"].mean() / dn_den)
            if dn.sum() > 10 and np.isfinite(dn_den) and abs(dn_den) > 1e-12
            else float("nan")
        )

        vy = float(np.var(z_ex["y"].to_numpy(dtype=float), ddof=1)) if len(z_ex) > 1 else float("nan")
        vm = float(np.var(z_ex["x"].to_numpy(dtype=float), ddof=1)) if len(z_ex) > 1 else float("nan")
        sys_share = ((beta**2) * vm / vy) if np.isfinite(vy) and vy > 1e-12 and np.isfinite(vm) else float("nan")

        rows.append(
            {
                "object": name,
                "alpha_daily": alpha_d,
                "alpha_ann": alpha_ann,
                "beta": beta,
                "r2": r2,
                "tracking_error": te,
                "information_ratio": ir,
                "up_capture": up_cap,
                "down_capture": dn_cap,
                "systematic_var_share": sys_share,
            }
        )

        roll_cols: dict[str, pd.Series] = {}
        for w in windows:
            b, c = rolling_beta_corr(y_ex, m_ex, window=int(w))
            roll_cols[f"beta_{w}"] = b
            roll_cols[f"corr_{w}"] = c
        roll[name] = pd.DataFrame(roll_cols)

    capm = pd.DataFrame(rows).set_index("object").sort_index()
    return capm, roll


def capm_regression(
    y_excess: pd.Series | Sequence[float] | np.ndarray,
    x_excess: pd.Series | Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    return capm_ols(y_excess, x_excess)


def rolling_beta(
    returns: pd.Series | Sequence[float] | np.ndarray,
    market_ret: pd.Series | Sequence[float] | np.ndarray,
    *,
    window: int,
) -> pd.Series:
    beta, _ = rolling_beta_corr(returns, market_ret, window=window)
    return beta


__all__ = [
    "capm_ols",
    "capm_regression",
    "capm_table",
    "rolling_beta",
    "rolling_beta_corr",
]
