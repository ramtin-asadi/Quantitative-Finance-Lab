from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.risk.utils import (
    VAR_BACKTEST_METHODS,
    _coerce_objects,
    _normalize_alpha,
    _to_numeric_series,
)


def hist_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute historical VaR and expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.

    Returns
    -------
    tuple of float
        ``(VaR, ES)`` as positive loss values.
    """

    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    q = float(r.quantile(a))
    tail = r[r <= q]
    es = float(tail.mean()) if len(tail) else q
    return -q, -es

def cf_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_sim: int = 70_000,
    seed: int = 7,
) -> tuple[float, float]:
    """Compute Cornish-Fisher VaR and expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.
    n_sim : int, default=70000
        Number of simulated transformed-normal draws used to approximate ES.
    seed : int, default=7
        Random seed.

    Returns
    -------
    tuple of float
        ``(VaR, ES)`` as positive loss values.

    Notes
    -----
    The VaR quantile uses the Cornish-Fisher expansion with sample skewness and
    excess kurtosis. ES is approximated by simulation from the transformed normal
    distribution.
    """

    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    if len(r) < 10:
        return float("nan"), float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan"), float("nan")
    s = float(r.skew())
    k = float(r.kurt())
    z = NormalDist().inv_cdf(a)
    zc = z + (z**2 - 1.0) * s / 6.0 + (z**3 - 3.0 * z) * k / 24.0 - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    q = mu + sd * zc

    rng = np.random.default_rng(int(seed))
    zs = rng.standard_normal(int(n_sim))
    za = (
        zs
        + (zs**2 - 1.0) * s / 6.0
        + (zs**3 - 3.0 * zs) * k / 24.0
        - (2.0 * zs**3 - 5.0 * zs) * (s**2) / 36.0
    )
    rs = mu + sd * za
    tail = rs[rs <= q]
    es = float(np.mean(tail)) if len(tail) else float(q)
    return -float(q), -float(es)

def fhs_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.94,
) -> tuple[float, float]:
    """Compute filtered-historical-simulation VaR and expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.
    lam : float, default=0.94
        EWMA volatility decay parameter.

    Returns
    -------
    tuple of float
        ``(VaR, ES)`` as positive loss values.

    Raises
    ------
    InputError
        If ``lam`` is not in ``(0, 1)``.
    """

    a = _normalize_alpha(alpha)
    if not (0.0 < float(lam) < 1.0):
        raise InputError("lam must be in (0, 1).")
    r = _to_numeric_series(returns, name="returns")
    if len(r) < 10:
        return float("nan"), float("nan")
    mu = float(r.mean())
    e = r - mu
    sig = np.zeros(len(e), dtype=float)
    sig[0] = max(float(e.std(ddof=1)), 1e-6)
    ev = e.to_numpy(dtype=float)
    for t in range(1, len(e)):
        sig[t] = math.sqrt(float(lam) * sig[t - 1] ** 2 + (1.0 - float(lam)) * ev[t - 1] ** 2)
    z = ev / np.where(sig > 1e-12, sig, np.nan)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("nan"), float("nan")
    qz = float(np.quantile(z, a))
    tail = z[z <= qz]
    ez = float(np.mean(tail)) if len(tail) else qz
    sn = float(sig[-1])
    return -(mu + sn * qz), -(mu + sn * ez)

def _rolling_var_quantile(
    returns: pd.Series,
    *,
    alpha: float,
    lookback: int,
    method: Literal["hist", "cf", "fhs"],
    cf_n_sim: int = 15_000,
    cf_seed: int = 7,
    fhs_lambda: float = 0.94,
) -> pd.Series:
    if lookback < 20:
        raise InputError("lookback must be at least 20.")
    r = _to_numeric_series(returns, name="returns")
    if len(r) < lookback + 1:
        return pd.Series(dtype=float)
    method_norm = str(method).strip().lower()
    if method_norm not in set(VAR_BACKTEST_METHODS):
        raise InputError("method must be one of {'hist', 'cf', 'fhs'}.")
    if method_norm == "hist":
        # One-step-ahead VaR forecast (no look-ahead): estimate at t-1, test at t.
        return r.rolling(int(lookback), min_periods=int(lookback)).quantile(float(alpha)).shift(1)

    idx = r.index
    q = pd.Series(np.nan, index=idx, dtype=float)
    # One-step-ahead VaR forecast (no look-ahead): estimate from [t-lookback, t-1].
    for i in range(int(lookback), len(r)):
        window = r.iloc[i - int(lookback) : i]
        if method_norm == "cf":
            v, _ = cf_var_es(window, alpha=alpha, n_sim=cf_n_sim, seed=cf_seed)
        elif method_norm == "fhs":
            v, _ = fhs_var_es(window, alpha=alpha, lam=fhs_lambda)
        q.iloc[i] = -float(v) if np.isfinite(v) else np.nan
    return q

def rolling_var(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lookback: int = 252,
    method: Literal["hist", "cf", "fhs"] = "hist",
    cf_n_sim: int = 15_000,
    cf_seed: int = 7,
    fhs_lambda: float = 0.94,
) -> pd.Series:
    """Compute rolling one-period VaR forecasts.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.
    lookback : int, default=252
        Rolling estimation window.
    method : {"hist", "cf", "fhs"}, default="hist"
        VaR estimation method.
    cf_n_sim : int, default=15000
        Number of simulations used by the Cornish-Fisher method.
    cf_seed : int, default=7
        Random seed for Cornish-Fisher simulation.
    fhs_lambda : float, default=0.94
        EWMA decay parameter for filtered historical simulation.

    Returns
    -------
    pandas.Series
        Rolling VaR quantile in return units. Breaches occur when realized returns
        are below this negative threshold.
    """

    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    return _rolling_var_quantile(
        r,
        alpha=a,
        lookback=int(lookback),
        method=method,
        cf_n_sim=int(cf_n_sim),
        cf_seed=int(cf_seed),
        fhs_lambda=float(fhs_lambda),
    )

def var_es_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    alpha: float = 0.05,
    methods: Sequence[str] = ("hist", "cf", "fhs"),
) -> pd.DataFrame:
    """Build VaR and ES tables for multiple objects and methods.

    Parameters
    ----------
    objects : mapping or pandas.DataFrame
        Return objects.
    alpha : float, default=0.05
        Lower-tail probability.
    methods : sequence of str, default=("hist", "cf", "fhs")
        VaR/ES methods to include.

    Returns
    -------
    pandas.DataFrame
        Table indexed by object with method-specific VaR and ES columns.

    Raises
    ------
    InputError
        If no methods are supplied or an unknown method is requested.
    """

    a = _normalize_alpha(alpha)
    obj = _coerce_objects(objects)
    methods_norm = [str(m).strip().lower() for m in methods]
    if not methods_norm:
        raise InputError("methods must contain at least one method.")
    valid = {"hist", "cf", "fhs"}
    unknown = [m for m in methods_norm if m not in valid]
    if unknown:
        raise InputError(f"Unknown VaR/ES method(s): {unknown}")
    p = round(a * 100)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        row: dict[str, Any] = {"object": name}
        if "hist" in methods_norm:
            v, e = hist_var_es(r, alpha=a)
            row[f"hist_var{p}"] = v
            row[f"hist_es{p}"] = e
        if "cf" in methods_norm:
            v, e = cf_var_es(r, alpha=a)
            row[f"cf_var{p}"] = v
            row[f"cf_es{p}"] = e
        if "fhs" in methods_norm:
            v, e = fhs_var_es(r, alpha=a)
            row[f"fhs_var{p}"] = v
            row[f"fhs_es{p}"] = e
        rows.append(row)
    return pd.DataFrame(rows).set_index("object").sort_index()


def historical_var(returns: pd.Series | Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> float:
    return hist_var_es(returns, alpha=alpha)[0]


def cornish_fisher_var(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_sim: int = 70_000,
    seed: int = 7,
) -> float:
    return cf_var_es(returns, alpha=alpha, n_sim=n_sim, seed=seed)[0]


def filtered_historical_var(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.94,
) -> float:
    return fhs_var_es(returns, alpha=alpha, lam=lam)[0]


__all__ = [
    "cf_var_es",
    "cornish_fisher_var",
    "fhs_var_es",
    "filtered_historical_var",
    "hist_var_es",
    "historical_var",
    "rolling_var",
    "var_es_table",
]
