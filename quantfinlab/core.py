from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

# ----------------------------
# Common callable types
# ----------------------------

DFCallable = Callable[[np.ndarray], np.ndarray]


# ----------------------------
# Exceptions (minimal, but useful)
# ----------------------------

class QuantFinLabError(Exception):
    """Base exception for the library."""


class InputError(QuantFinLabError):
    """Raised when inputs are missing, malformed, or inconsistent."""


class ModelError(QuantFinLabError):
    """Raised when a model fit/solve fails (optimizer, calibration, etc.)."""



# ----------------------------
# Public dataclasses (small)
# ----------------------------

@dataclass(frozen=True)
class CurvePillars:
    """Bootstrapped pillars (T, par yields, discount factors)."""
    asof: pd.Timestamp | None
    labels: list[str]
    T: np.ndarray          # years
    par: np.ndarray        # decimals (e.g., 0.045)
    dfs: np.ndarray        # discount factors at T

    # Optional train/test split fields used by rmse_backtest
    labels_test: list[str] | None = None
    T_test: np.ndarray | None = None
    par_test: np.ndarray | None = None


@dataclass(frozen=True)
class Curve:
    """A curve object that provides df(t), plus grid diagnostics."""
    method: str
    name: str
    grid: np.ndarray
    df_grid: np.ndarray
    z_grid: np.ndarray
    fwd_grid: np.ndarray
    df: Callable[[np.ndarray | float], np.ndarray]


@dataclass(frozen=True)
class IssuedBond:
    """Bond issued at issue_date, with cashflows defined in time-from-issue."""
    issue_date: pd.Timestamp
    maturity_years: int
    coupon: float
    freq: int
    times: np.ndarray  # from issue, years
    cfs: np.ndarray    # cashflows per unit notional


@dataclass(frozen=True)
class IssuanceBook:
    """Synthetic book grouped by maturity bucket."""
    maturities: list[int]        # e.g. [2,5,10,30]
    freq: int
    by_maturity: dict[int, list[IssuedBond]]


@dataclass(frozen=True)
class BookMetrics:
    """
    Container returned by book_metrics (kept simple for plotting).

    total_pv:  index=date, columns=method
    bucket_pv: index=date, columns MultiIndex (method, maturity)
    risk:      index=date, columns MultiIndex (method, metric) metric in {"pv01","convexity"}
    """
    total_pv: pd.DataFrame
    bucket_pv: pd.DataFrame
    risk: pd.DataFrame


@dataclass(frozen=True)
class Bond:
    """Plain fixed-rate bullet bond."""
    coupon: float                 # annual coupon rate in decimals
    maturity_years: float         # maturity in years
    freq: int = 2                 # payments per year
    face: float = 1.0             # notional
    day_count: str = "30/360"     # kept for metadata; accrual uses time-in-years below



# ----------------------------
# Tiny helpers (only what Project 1 needs)
# ----------------------------

def as_timestamp(x: pd.Timestamp | str | None) -> pd.Timestamp | None:
    """Convert to Timestamp (or keep None)."""
    if x is None:
        return None
    return pd.Timestamp(x)


def as_1d_float_array(x, *, name: str = "array") -> np.ndarray:
    """Convert to 1D float ndarray and validate finiteness."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise InputError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains NaN/inf.")
    return arr


def validate_sorted_strictly_increasing(T: np.ndarray, *, name: str = "T") -> None:
    """Ensure T is strictly increasing."""
    if np.any(np.diff(T) <= 0):
        raise InputError(f"{name} must be strictly increasing.")
