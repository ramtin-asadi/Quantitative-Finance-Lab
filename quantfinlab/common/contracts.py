from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .errors import InputError
from .types import DFCallable


@dataclass(frozen=True)
class CurvePillars:
    """Bootstrapped curve pillars for a single valuation date.

    The object stores the observed tenor labels, maturities, par yields, and
    bootstrapped discount factors used as inputs to curve-smoothing routines. It
    can also carry an optional holdout set for cross-validation or out-of-sample
    curve-fit diagnostics.

    Attributes
    ----------
    asof : pandas.Timestamp or None
        Valuation date of the curve row. ``None`` is allowed when the pillars are
        built from anonymous or synthetic inputs.
    labels : list of str
        Tenor labels corresponding to the fitted pillar points, such as ``"6M"``,
        ``"2Y"``, or ``"10Y"``.
    T : numpy.ndarray
        Pillar maturities in years, sorted in increasing order.
    par : numpy.ndarray
        Par yields at the fitted pillar maturities, expressed as decimals.
    dfs : numpy.ndarray
        Bootstrapped discount factors at the fitted pillar maturities.
    labels_test : list of str, optional
        Optional labels for holdout tenors excluded from curve fitting.
    T_test : numpy.ndarray, optional
        Optional holdout maturities in years.
    par_test : numpy.ndarray, optional
        Optional holdout par yields, expressed as decimals.

    Notes
    -----
    The class is a lightweight data container. It does not validate curve
    monotonicity, positivity, or no-arbitrage conditions by itself; those checks
    should be handled by the construction or fitting routines that create it.
    """

    asof: pd.Timestamp | None
    labels: list[str]
    T: np.ndarray
    par: np.ndarray
    dfs: np.ndarray
    labels_test: list[str] | None = None
    T_test: np.ndarray | None = None
    par_test: np.ndarray | None = None


@dataclass(frozen=True)
class Curve:
    """Interpolated fixed-income curve and its evaluation grids.

    The object stores a fitted curve method, display name, common evaluation grid,
    discount-factor grid, zero-rate grid, forward-rate grid, and a callable
    discount-factor function. It is designed to pass a complete curve
    representation between pricing, risk, plotting, and backtesting routines.

    Attributes
    ----------
    method : str
        Stable method key used for programmatic identification, such as
        ``"loglinear"``, ``"pchip"``, ``"nss"``, or ``"qp"``.
    name : str
        Human-readable curve name suitable for tables and plot legends.
    grid : numpy.ndarray
        Maturity grid in years.
    df_grid : numpy.ndarray
        Discount factors on ``grid``.
    z_grid : numpy.ndarray
        Continuously compounded zero rates on ``grid``.
    fwd_grid : numpy.ndarray
        Instantaneous forward rates on ``grid``.
    df : callable
        Function that maps maturities in years to discount factors.

    Notes
    -----
    The discount-factor callable should accept scalar-like or array-like maturities
    and return finite positive discount factors for supported maturities. Downstream
    risk and pricing functions assume maturities are measured in years and rates
    are in decimal units.
    """

    method: str
    name: str
    grid: np.ndarray
    df_grid: np.ndarray
    z_grid: np.ndarray
    fwd_grid: np.ndarray
    df: Callable[[np.ndarray | float], np.ndarray]


@dataclass(frozen=True)
class IssuedBond:
    """Synthetic bond issued into a maturity bucket.

    Attributes
    ----------
    issue_date : pandas.Timestamp
        Date on which the synthetic bond is issued.
    maturity_years : int
        Original maturity bucket of the bond in years.
    coupon : float
        Annual coupon rate expressed as a decimal.
    freq : int
        Coupon payment frequency per year.
    times : numpy.ndarray
        Scheduled cash-flow times from issue date, in years.
    cfs : numpy.ndarray
        Cash-flow amounts per unit face value at each scheduled payment time.

    Notes
    -----
    The object is intended for synthetic issuance and ladder/backtest workflows.
    It stores cash-flow times relative to the issue date, not absolute payment
    dates.
    """

    issue_date: pd.Timestamp
    maturity_years: int
    coupon: float
    freq: int
    times: np.ndarray
    cfs: np.ndarray


@dataclass(frozen=True)
class IssuanceBook:
    """Collection of synthetic bonds grouped by maturity bucket.

    Attributes
    ----------
    maturities : list of int
        Maturity buckets included in the book.
    freq : int
        Coupon payment frequency used for the issued bonds.
    by_maturity : dict[int, list[IssuedBond]]
        Mapping from maturity bucket to the list of bonds issued in that bucket.

    Notes
    -----
    This container is designed for fixed-income valuation and risk examples where
    new par bonds are issued repeatedly through time. It does not represent a live
    trading book with transaction records, cash balances, or settlement logic.
    """

    maturities: list[int]
    freq: int
    by_maturity: dict[int, list[IssuedBond]]


@dataclass(frozen=True)
class BookMetrics:
    """Valuation and risk tables for a fixed-income book.

    Attributes
    ----------
    total_pv : pandas.DataFrame
        Total present value by valuation date and curve method.
    bucket_pv : pandas.DataFrame
        Present value by valuation date, curve method, and maturity bucket.
    risk : pandas.DataFrame
        Risk metrics, typically including PV01 and convexity, by valuation date
        and curve method.

    Notes
    -----
    The object is a compact transport container for book-level analytics. It does
    not prescribe a specific column layout beyond the conventions used by the
    valuation and risk routines that create it.
    """

    total_pv: pd.DataFrame
    bucket_pv: pd.DataFrame
    risk: pd.DataFrame


@dataclass(frozen=True)
class Bond:
    """Plain fixed-coupon bond specification.

    Attributes
    ----------
    coupon : float
        Annual coupon rate expressed as a decimal.
    maturity_years : float
        Original maturity of the bond in years.
    freq : int, default 2
        Coupon payment frequency per year.
    face : float, default 1.0
        Face value used to scale cash flows.
    day_count : str, default "30/360"
        Informational day-count label. Pricing routines using this object apply
        their own simplified timing assumptions and do not currently implement a
        full day-count engine.

    Notes
    -----
    The class stores bond terms only. It does not include issue date, settlement
    date, accrued-interest schedule, holiday calendars, or ex-coupon treatment.
    """

    coupon: float
    maturity_years: float
    freq: int = 2
    face: float = 1.0
    day_count: str = "30/360"


@dataclass(frozen=True)
class PortfolioState:
    """Prepared portfolio inputs for an optimization or backtest step.

    Attributes
    ----------
    tickers : list of str
        Ordered asset universe.
    mu_excess_ann : pandas.Series
        Annualized expected excess returns indexed by ticker.
    cov_ann_map : dict[str, numpy.ndarray]
        Mapping from covariance-model name to annualized covariance matrix.
    avg_dollar_volume : pandas.Series, optional
        Average dollar volume by ticker, typically used for liquidity filters or
        capacity diagnostics.
    metadata : Mapping[str, Any], optional
        Additional information about the state, such as estimation windows,
        selected models, or data-quality flags.

    Methods
    -------
    as_dict()
        Return a dictionary representation preserving the stored pandas and NumPy
        objects.

    Notes
    -----
    The order of ``tickers`` should match the order of the covariance matrices.
    This class does not enforce dimensional consistency; callers should validate
    the state before optimization.
    """

    tickers: list[str]
    mu_excess_ann: pd.Series
    cov_ann_map: dict[str, np.ndarray]
    avg_dollar_volume: pd.Series | None = None
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tickers": list(self.tickers),
            "mu_excess_ann": self.mu_excess_ann,
            "cov_ann_map": self.cov_ann_map,
        }
        if self.avg_dollar_volume is not None:
            out["avg_dollar_volume"] = self.avg_dollar_volume
        if self.metadata is not None:
            out["metadata"] = dict(self.metadata)
        return out


@dataclass(frozen=True)
class BacktestResult:
    """Result container for a gross/net portfolio backtest.

    Attributes
    ----------
    gross_values : pandas.Series
        Portfolio value path before trading costs or other net adjustments.
    net_values : pandas.Series
        Portfolio value path after costs and net adjustments.
    gross_returns : pandas.Series
        Gross period returns.
    net_returns : pandas.Series
        Net period returns.
    weights : pandas.DataFrame
        Portfolio weights through time, usually indexed by rebalance or valuation
        date.
    turnover : pandas.Series
        Portfolio turnover by period.
    costs : pandas.Series
        Trading or implementation costs by period.
    fallbacks : int, default 0
        Number of optimization or construction fallbacks used during the backtest.
    metadata : Mapping[str, Any], optional
        Additional diagnostics or configuration values.

    Methods
    -------
    as_dict()
        Return the result as a dictionary.
    __getitem__(key)
        Dictionary-like access to stored fields.

    Notes
    -----
    The class intentionally keeps the object lightweight and dictionary-compatible
    for notebook analysis. It does not perform performance attribution or metric
    calculation by itself.
    """

    gross_values: pd.Series
    net_values: pd.Series
    gross_returns: pd.Series
    net_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    fallbacks: int = 0
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "gross_values": self.gross_values,
            "net_values": self.net_values,
            "gross_returns": self.gross_returns,
            "net_returns": self.net_returns,
            "weights": self.weights,
            "turnover": self.turnover,
            "costs": self.costs,
            "fallbacks": int(self.fallbacks),
        }
        if self.metadata is not None:
            out["metadata"] = dict(self.metadata)
        return out

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


@dataclass(frozen=True)
class StrategyBuildResult:
    """Container for a multi-strategy portfolio construction run.

    Attributes
    ----------
    prices : pandas.DataFrame
        Price panel used by the strategy builder.
    volumes : pandas.DataFrame
        Volume panel aligned to ``prices`` where available.
    returns : pandas.DataFrame
        Return panel derived from the input prices.
    rebal_dates : list of pandas.Timestamp
        Rebalance dates used by the strategy builder.
    cache : Mapping[pandas.Timestamp, Mapping[str, Any]]
        Per-date cache of prepared inputs, fitted models, or optimization states.
    results : Mapping[str, BacktestResult]
        Backtest results keyed by strategy name.
    cov_key_for_rc : Mapping[str, str]
        Mapping from strategy name to the covariance model used for risk
        contribution reporting.
    metadata : Mapping[str, Any], optional
        Additional run-level metadata.

    Methods
    -------
    as_dict()
        Return a dictionary representation of the build result.
    __getitem__(key)
        Dictionary-like access to stored fields.

    Notes
    -----
    This object is intended to bundle data preparation, model cache, and backtest
    outputs into a single return value. It does not guarantee that all strategies
    share the same rebalance schedule unless the builder created them that way.
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    returns: pd.DataFrame
    rebal_dates: list[pd.Timestamp]
    cache: Mapping[pd.Timestamp, Mapping[str, Any]]
    results: Mapping[str, BacktestResult]
    cov_key_for_rc: Mapping[str, str]
    metadata: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "prices": self.prices,
            "volumes": self.volumes,
            "returns": self.returns,
            "rebal_dates": list(self.rebal_dates),
            "cache": dict(self.cache),
            "results": dict(self.results),
            "cov_key_for_rc": dict(self.cov_key_for_rc),
        }
        if self.metadata is not None:
            out["metadata"] = dict(self.metadata)
        return out

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


@dataclass(frozen=True)
class FundamentalReportArtifacts:
    """Tables, figures, series, and text generated by a fundamental report.

    Parameters
    ----------
    tables : Mapping[str, pandas.DataFrame]
        Named report tables.
    figures : Mapping[str, list[Any]]
        Named groups of figure objects or figure-like handles.
    series : Mapping[str, Any], optional
        Additional named series or time-indexed report outputs.
    text : Mapping[str, list[str]], optional
        Named report notes, warnings, or summary bullets.
    """

    tables: Mapping[str, pd.DataFrame]
    figures: Mapping[str, list[Any]]
    series: Mapping[str, Any] | None = None
    text: Mapping[str, list[str]] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tables": dict(self.tables),
            "figures": dict(self.figures),
        }
        if self.series is not None:
            out["series"] = dict(self.series)
        if self.text is not None:
            out["text"] = dict(self.text)
        return out

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


@dataclass(frozen=True)
class RiskReportArtifacts:
    """Tables, figures, series, and text generated by a risk report.

    Attributes
    ----------
    tables : Mapping[str, pandas.DataFrame]
        Named report tables.
    figures : Mapping[str, list[Any]]
        Named groups of figure objects or figure-like handles.
    series : Mapping[str, Any], optional
        Additional named series, arrays, or time-indexed objects used by the
        report.
    text : Mapping[str, list[str]], optional
        Named narrative notes, warnings, or bullet lists.

    Methods
    -------
    as_dict()
        Return the artifacts as a dictionary.
    __getitem__(key)
        Dictionary-like access to stored artifact groups.

    Notes
    -----
    The figure values are typed as ``Any`` to allow matplotlib figures, axes,
    plotly objects, or other notebook-display objects.
    """

    tables: Mapping[str, pd.DataFrame]
    figures: Mapping[str, list[Any]]
    series: Mapping[str, Any] | None = None
    text: Mapping[str, list[str]] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "tables": dict(self.tables),
            "figures": dict(self.figures),
        }
        if self.series is not None:
            out["series"] = dict(self.series)
        if self.text is not None:
            out["text"] = dict(self.text)
        return out

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


def as_timestamp(x: pd.Timestamp | str | None) -> pd.Timestamp | None:
    """Convert a nullable date-like value to ``pandas.Timestamp``.

    Parameters
    ----------
    x : pandas.Timestamp, str, or None
        Date-like value to convert. ``None`` is preserved.

    Returns
    -------
    pandas.Timestamp or None
        Converted timestamp, or ``None`` when ``x`` is ``None``.

    Notes
    -----
    This helper is intentionally thin and follows ``pandas.Timestamp`` parsing
    rules for strings and timestamp-like inputs.
    """

    if x is None:
        return None
    return pd.Timestamp(x)


def as_1d_float_array(x, *, name: str = "array") -> np.ndarray:
    """Convert an input to a finite one-dimensional float array.

    Parameters
    ----------
    x : array-like
        Input values to convert.
    name : str, default "array"
        Name used in validation error messages.

    Returns
    -------
    numpy.ndarray
        One-dimensional float array.

    Raises
    ------
    InputError
        If the converted array is empty or contains NaN or infinite values.

    Notes
    -----
    The input is flattened with ``reshape(-1)``. Shape information from the
    original object is not preserved.
    """

    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise InputError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains NaN/inf.")
    return arr


def validate_sorted_strictly_increasing(T: np.ndarray, *, name: str = "T") -> None:
    """Validate that a numeric array is strictly increasing.

    Parameters
    ----------
    T : numpy.ndarray
        Array to check.
    name : str, default "T"
        Name used in validation error messages.

    Returns
    -------
    None
        The function returns ``None`` when validation passes.

    Raises
    ------
    InputError
        If any adjacent difference is less than or equal to zero.

    Notes
    -----
    The function assumes ``T`` is already numeric. Use a finite-array validator
    before calling this function when NaN or infinite values are possible.
    """

    if np.any(np.diff(T) <= 0):
        raise InputError(f"{name} must be strictly increasing.")


__all__ = [
    "BacktestResult",
    "Bond",
    "BookMetrics",
    "Curve",
    "CurvePillars",
    "DFCallable",
    "FundamentalReportArtifacts",
    "IssuanceBook",
    "IssuedBond",
    "PortfolioState",
    "RiskReportArtifacts",
    "StrategyBuildResult",
    "as_1d_float_array",
    "as_timestamp",
    "validate_sorted_strictly_increasing",
]
