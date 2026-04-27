from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimpleBacktestResult:
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
