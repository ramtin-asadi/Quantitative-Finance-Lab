from __future__ import annotations

from ..core import InputError, ModelError, QuantFinLabError


class DataError(QuantFinLabError):
    """Raised when input data is unavailable or inconsistent."""


class BacktestError(QuantFinLabError):
    """Raised when a backtest cannot be constructed or evaluated."""


__all__ = [
    "BacktestError",
    "DataError",
    "InputError",
    "ModelError",
    "QuantFinLabError",
]
