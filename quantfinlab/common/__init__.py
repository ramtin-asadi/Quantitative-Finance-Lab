from __future__ import annotations

from .contracts import (
    BacktestResult,
    Bond,
    BookMetrics,
    Curve,
    CurvePillars,
    IssuanceBook,
    IssuedBond,
    PortfolioState,
    RiskReportArtifacts,
    StrategyBuildResult,
    as_1d_float_array,
    as_timestamp,
    validate_sorted_strictly_increasing,
)
from .dates import align_to_previous_available, month_end_dates, previous_available_date, yearfrac
from .errors import BacktestError, DataError, InputError, ModelError, QuantFinLabError
from .results import SimpleBacktestResult
from .types import ArrayLike, DFCallable, SeriesOrFrame
from .validation import (
    normalize_weights,
    require_columns,
    require_finite_array,
    require_monotonic_index,
    require_non_empty_frame,
)

__all__ = [
    "ArrayLike",
    "BacktestError",
    "BacktestResult",
    "Bond",
    "BookMetrics",
    "Curve",
    "CurvePillars",
    "DFCallable",
    "DataError",
    "InputError",
    "IssuanceBook",
    "IssuedBond",
    "ModelError",
    "PortfolioState",
    "QuantFinLabError",
    "RiskReportArtifacts",
    "SeriesOrFrame",
    "SimpleBacktestResult",
    "StrategyBuildResult",
    "align_to_previous_available",
    "as_1d_float_array",
    "as_timestamp",
    "month_end_dates",
    "normalize_weights",
    "previous_available_date",
    "require_columns",
    "require_finite_array",
    "require_monotonic_index",
    "require_non_empty_frame",
    "validate_sorted_strictly_increasing",
    "yearfrac",
]
