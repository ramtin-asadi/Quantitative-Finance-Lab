from __future__ import annotations

from importlib import import_module
from types import ModuleType

from .common.contracts import (
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
)
from .common.errors import (
    BacktestError,
    DataError,
    InputError,
    MissingKernelsError,
    ModelError,
    QuantFinLabError,
)

__version__ = "0.5.0"

_PUBLIC_MODULES = {
    "backtest",
    "common",
    "dataio",
    "fixed_income",
    "hedging",
    "macro",
    "options",
    "portfolio",
    "reports",
    "risk",
    "volatility",
}

_LAZY_MODULES = _PUBLIC_MODULES | {"calibration", "ml", "numerics", "plotting"}


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_MODULES)


__all__ = [
    "BacktestError",
    "BacktestResult",
    "Bond",
    "BookMetrics",
    "Curve",
    "CurvePillars",
    "DataError",
    "InputError",
    "IssuanceBook",
    "IssuedBond",
    "MissingKernelsError",
    "ModelError",
    "PortfolioState",
    "QuantFinLabError",
    "RiskReportArtifacts",
    "StrategyBuildResult",
    "__version__",
    "backtest",
    "common",
    "dataio",
    "fixed_income",
    "hedging",
    "macro",
    "options",
    "portfolio",
    "reports",
    "risk",
    "volatility",
]
