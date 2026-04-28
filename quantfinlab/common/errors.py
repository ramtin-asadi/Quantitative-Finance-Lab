from __future__ import annotations


class QuantFinLabError(Exception):
    """Base exception for the library."""


class InputError(QuantFinLabError):
    """Raised when inputs are missing, malformed, or inconsistent."""


class ModelError(QuantFinLabError):
    """Raised when a model fit or solve fails."""


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
