from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


def constraints_feasible(
    n_assets: int,
    *,
    w_min: float | None = None,
    w_max: float | None = None,
    long_only: bool = True,
) -> bool:
    """Check whether simple box constraints can hold a fully invested portfolio.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    w_min : float, optional
        Minimum weight per asset. For long-only portfolios, missing lower bounds
        are treated as zero.
    w_max : float, optional
        Maximum weight per asset. If omitted, no finite upper bound is applied.
    long_only : bool, default=True
        Whether negative weights are disallowed.

    Returns
    -------
    bool
        True if the bounds can contain at least one portfolio whose weights sum
        to one; otherwise False.

    Notes
    -----
    This is a feasibility pre-check for simple per-asset bounds. It does not
    verify more complex constraints such as turnover, active limits, or sector
    constraints.
    """

    n = int(n_assets)
    if n <= 0:
        return False
    w_min_eff = 0.0 if long_only else (-np.inf if w_min is None else float(w_min))
    w_max_eff = np.inf if w_max is None else float(w_max)
    if np.isfinite(w_max_eff) and w_max_eff * n < 1.0 - 1e-12:
        return False
    return not (np.isfinite(w_min_eff) and w_min_eff * n > 1.0 + 1e-12)


def normalize_weights(
    weights,
    index: pd.Index | Sequence[str] | None = None,
    *,
    w_min: float | None = None,
    w_max: float | None = None,
    long_only: bool = True,
    n_rounds: int = 3,
    as_series: bool | None = None,
) -> np.ndarray | pd.Series | None:
    """Normalize a weight vector under simple long-only and box constraints.

    The function clips weights to the requested lower and upper bounds, enforces
    non-negativity for long-only portfolios, and repeatedly renormalizes the
    vector to sum to one.

    Parameters
    ----------
    weights : array-like, pandas.Series, or mapping
        Raw portfolio weights.
    index : pandas.Index or sequence, optional
        Labels to use when returning a Series or when array-like weights need
        labels.
    w_min : float, optional
        Minimum weight per asset.
    w_max : float, optional
        Maximum weight per asset.
    long_only : bool, default=True
        Whether negative weights are clipped to zero.
    n_rounds : int, default=3
        Number of clip-and-renormalize passes.
    as_series : bool, optional
        Force Series output. If omitted, Series output is used when labels are
        available.

    Returns
    -------
    numpy.ndarray, pandas.Series, or None
        Normalized weights. Returns ``None`` if the input is empty, non-finite, or
        cannot be normalized.

    Raises
    ------
    InputError
        If Series output is requested but no index is available.

    Notes
    -----
    This helper is intentionally simple. For tight caps, repeated clipping can
    only approximate feasibility; optimization routines should still validate
    their final solutions.
    """

    labels = None
    if isinstance(weights, pd.Series):
        labels = weights.index
        arr = weights.to_numpy(dtype=float)
    else:
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if index is not None:
            labels = pd.Index(index)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        return None
    out = arr.copy()
    for _ in range(max(int(n_rounds), 1)):
        if long_only:
            out = np.maximum(out, 0.0)
        if w_min is not None:
            out = np.maximum(out, float(w_min))
        if w_max is not None:
            out = np.minimum(out, float(w_max))
        total = float(out.sum())
        if (not np.isfinite(total)) or total <= 0:
            return None
        out = out / total
    want_series = bool(as_series) if as_series is not None else labels is not None
    if want_series:
        if labels is None:
            raise InputError("index is required when as_series=True.")
        return pd.Series(out, index=labels, dtype=float)
    return out


def long_only_box_constraints(
    n_assets: int,
    *,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
) -> list[tuple[float, float]]:
    """Return scipy-style bounds for a long-only box-constrained portfolio."""
    if not constraints_feasible(n_assets, w_min=w_min, w_max=w_max, long_only=True):
        raise InputError("Constraints are infeasible for the number of assets.")
    lo = 0.0 if w_min is None else max(0.0, float(w_min))
    hi = 1.0 if w_max is None else float(w_max)
    return [(lo, hi) for _ in range(int(n_assets))]


def coerce_prev_weights(w_prev, n_assets: int) -> np.ndarray:
    """Normalize previous weights or fall back to equal weight."""
    n = int(n_assets)
    if n <= 0:
        raise InputError("n_assets must be positive.")
    if w_prev is None:
        return np.ones(n, dtype=float) / n
    arr = np.asarray(w_prev, dtype=float).reshape(-1)
    if arr.size != n:
        raise InputError("w_prev length must match number of assets.")
    out = normalize_weights(arr, long_only=False, as_series=False)
    if out is None:
        return np.ones(n, dtype=float) / n
    return np.asarray(out, dtype=float)


__all__ = [
    "coerce_prev_weights",
    "constraints_feasible",
    "long_only_box_constraints",
    "normalize_weights",
]
