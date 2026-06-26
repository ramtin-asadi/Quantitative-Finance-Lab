from __future__ import annotations

# Folder-based Project 3 risk API.
#
# The old flat quantfinlab/risk.py implementation was moved out of the import
# path during the library migration. This package is now the canonical home for
# atomic risk analytics; selected names are re-exported for the old
# ``import quantfinlab.risk as rk`` notebook style.
from . import (
    capm,
    contributions,
    correlation,
    diagnostics,
    distribution,
    drawdown,
    es,
    performance,
    stress,
    utils,
    var,
    var_backtesting,
)
from .capm import capm_ols, capm_regression, capm_table, rolling_beta, rolling_beta_corr
from .contributions import (
    attribution_tables,
    portfolio_contribution_snapshot,
    scenario_es_contribution,
    vol_contribution,
)
from .correlation import corr_matrix, rolling_corr
from .distribution import tail_ratio, tail_shape_table, worst_returns_summary
from .drawdown import (
    avg_recovery_time,
    drawdown_episodes,
    drawdown_episodes_table,
    drawdown_series,
    drawdown_summary_table,
    max_drawdown,
    ulcer_index,
)
from .es import cornish_fisher_es, filtered_historical_es, historical_es
from .performance import (
    make_returns_panel,
    nav_series,
    performance_table,
    rolling_volatility,
    sortino_ratio,
    total_return,
)
from .stress import historical_stress_table, stress_table
from .utils import DEFAULT_ANNUALIZATION, VAR_BACKTEST_METHODS
from .var import (
    cf_var_es,
    cornish_fisher_var,
    fhs_var_es,
    filtered_historical_var,
    hist_var_es,
    historical_var,
    rolling_var,
    var_es_table,
)
from .var_backtesting import (
    best_var_methods,
    breach_stats,
    christoffersen_independence,
    kupiec_test,
    longest_true_streak,
    quantile_loss,
    var_backtest_details,
    var_backtest_table,
)


def risk_report(*args, **kwargs):
    """Backward-compatible wrapper for ``quantfinlab.reports.risk_report``."""
    from quantfinlab.reports.risk_report import risk_report as _risk_report

    return _risk_report(*args, **kwargs)


def executive_bullets(*args, **kwargs):
    """Backward-compatible wrapper for report-specific executive bullets."""
    from quantfinlab.reports.risk_report import executive_bullets as _executive_bullets

    return _executive_bullets(*args, **kwargs)


__all__ = [
    "DEFAULT_ANNUALIZATION",
    "VAR_BACKTEST_METHODS",
    "attribution_tables",
    "avg_recovery_time",
    "best_var_methods",
    "breach_stats",
    "capm",
    "capm_ols",
    "capm_regression",
    "capm_table",
    "cf_var_es",
    "christoffersen_independence",
    "contributions",
    "cornish_fisher_es",
    "cornish_fisher_var",
    "corr_matrix",
    "correlation",
    "diagnostics",
    "distribution",
    "drawdown",
    "drawdown_episodes",
    "drawdown_episodes_table",
    "drawdown_series",
    "drawdown_summary_table",
    "es",
    "executive_bullets",
    "fhs_var_es",
    "filtered_historical_es",
    "filtered_historical_var",
    "hist_var_es",
    "historical_es",
    "historical_stress_table",
    "historical_var",
    "kupiec_test",
    "longest_true_streak",
    "make_returns_panel",
    "max_drawdown",
    "nav_series",
    "performance",
    "performance_table",
    "portfolio_contribution_snapshot",
    "quantile_loss",
    "risk_report",
    "rolling_beta",
    "rolling_beta_corr",
    "rolling_corr",
    "rolling_var",
    "rolling_volatility",
    "scenario_es_contribution",
    "sortino_ratio",
    "stress",
    "stress_table",
    "tail_ratio",
    "tail_shape_table",
    "total_return",
    "ulcer_index",
    "utils",
    "var",
    "var_backtest_details",
    "var_backtest_table",
    "var_backtesting",
    "var_es_table",
    "vol_contribution",
    "worst_returns_summary",
]
