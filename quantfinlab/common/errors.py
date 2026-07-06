from __future__ import annotations


class QuantFinLabError(Exception):
    """Base exception for all package-specific errors.

    Notes
    -----
    Catching this class captures the package's custom error types while leaving
    unrelated Python, NumPy, pandas, or optimization exceptions untouched.
    """


class InputError(QuantFinLabError):
    """Error raised when user-supplied inputs are missing or malformed.

    Notes
    -----
    Use this exception for invalid shapes, missing columns, empty arrays, unsupported
    options, inconsistent indices, or impossible parameter combinations detected
    before model fitting or backtesting starts.
    """


class ModelError(QuantFinLabError):
    """Error raised when a model fit, numerical solve, or calibration fails.

    Notes
    -----
    Use this exception when inputs are structurally valid but the requested model
    cannot be estimated or solved under the supplied data and configuration.
    """


class DataError(QuantFinLabError):
    """Error raised when required data are unavailable or internally inconsistent.

    Notes
    -----
    Use this exception for missing source files, unusable market data, schema
    mismatches, or data-quality failures that are not merely argument-validation
    problems.
    """


class BacktestError(QuantFinLabError):
    """Error raised when a backtest cannot be constructed or evaluated.

    Notes
    -----
    Use this exception for failures such as missing contiguous date blocks,
    unavailable curves, invalid trading state, or other conditions that prevent a
    strategy simulation from producing a meaningful result.
    """


class MissingKernelsError(QuantFinLabError):
    """Error raised when optional compiled kernels are requested but unavailable.

    Notes
    -----
    Use this exception when a function explicitly requires compiled numerical
    kernels and the extension cannot be imported or does not provide the required
    kernel.
    """


__all__ = [
    "BacktestError",
    "DataError",
    "InputError",
    "MissingKernelsError",
    "ModelError",
    "QuantFinLabError",
]
