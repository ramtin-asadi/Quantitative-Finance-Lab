from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimpleBacktestResult:
    """Compact result container for strategy backtests.

    Attributes
    ----------
    nav : pandas.Series
        Net asset value or wealth path.
    returns : pandas.Series
        Period return series aligned to ``nav``.
    weights : pandas.DataFrame, optional
        Portfolio weights through time.
    trades : pandas.DataFrame, optional
        Executed trades or simulated transaction log.
    costs : pandas.Series or pandas.DataFrame, optional
        Trading costs, financing costs, or implementation costs.
    cashflows : pandas.DataFrame, optional
        Cash-flow, carry, coupon, or attribution components.
    diagnostics : dict[str, Any], optional
        Additional result-specific diagnostics.

    Methods
    -------
    as_dict()
        Return all stored fields as a dictionary.
    __getitem__(key)
        Dictionary-like access to stored fields.

    Notes
    -----
    The container is intentionally permissive so different strategy engines can
    return a consistent object without forcing a rigid schema for diagnostics.
    """

    nav: pd.Series
    returns: pd.Series
    weights: pd.DataFrame | None = None
    trades: pd.DataFrame | None = None
    costs: pd.Series | pd.DataFrame | None = None
    cashflows: pd.DataFrame | None = None
    diagnostics: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav,
            "returns": self.returns,
            "weights": self.weights,
            "trades": self.trades,
            "costs": self.costs,
            "cashflows": self.cashflows,
            "diagnostics": self.diagnostics,
        }

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


__all__ = ["SimpleBacktestResult"]
