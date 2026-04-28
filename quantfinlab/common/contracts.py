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
    method: str
    name: str
    grid: np.ndarray
    df_grid: np.ndarray
    z_grid: np.ndarray
    fwd_grid: np.ndarray
    df: Callable[[np.ndarray | float], np.ndarray]


@dataclass(frozen=True)
class IssuedBond:
    issue_date: pd.Timestamp
    maturity_years: int
    coupon: float
    freq: int
    times: np.ndarray
    cfs: np.ndarray


@dataclass(frozen=True)
class IssuanceBook:
    maturities: list[int]
    freq: int
    by_maturity: dict[int, list[IssuedBond]]


@dataclass(frozen=True)
class BookMetrics:
    total_pv: pd.DataFrame
    bucket_pv: pd.DataFrame
    risk: pd.DataFrame


@dataclass(frozen=True)
class Bond:
    coupon: float
    maturity_years: float
    freq: int = 2
    face: float = 1.0
    day_count: str = "30/360"


@dataclass(frozen=True)
class PortfolioState:
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
class RiskReportArtifacts:
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
    if x is None:
        return None
    return pd.Timestamp(x)


def as_1d_float_array(x, *, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise InputError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise InputError(f"{name} contains NaN/inf.")
    return arr


def validate_sorted_strictly_increasing(T: np.ndarray, *, name: str = "T") -> None:
    if np.any(np.diff(T) <= 0):
        raise InputError(f"{name} must be strictly increasing.")


__all__ = [
    "BacktestResult",
    "Bond",
    "BookMetrics",
    "Curve",
    "CurvePillars",
    "DFCallable",
    "IssuanceBook",
    "IssuedBond",
    "PortfolioState",
    "RiskReportArtifacts",
    "StrategyBuildResult",
    "as_1d_float_array",
    "as_timestamp",
    "validate_sorted_strictly_increasing",
]
