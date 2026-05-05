from __future__ import annotations

from . import (
    attribution,
    black_litterman,
    confidence,
    constraints,
    costs,
    covariance,
    expected_returns,
    optimizers,
    selection,
    universe,
    views,
)
from .attribution import concentration, effective_number_of_holdings, max_weight, risk_contribution
from .constraints import constraints_feasible, long_only_box_constraints, normalize_weights
from .costs import apply_transaction_costs, portfolio_turnover, transaction_cost_from_turnover
from .covariance import (
    cov_estimate,
    estimate_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    make_psd,
    oas_covariance,
    sample_covariance,
)
from .expected_returns import (
    bayes_stein_momentum_mu,
    bayes_stein_mu,
    build_mu_excess_ann,
    build_scaled_mu_from_raw,
    momentum_mu,
    momentum_score_from_returns,
    mu_diagnostics,
    mu_momentum,
    scale_mu_to_target_sharpe,
    winsorize_signal,
    zscore_signal,
)
from .optimizers import (
    equal_weight,
    max_sharpe_frontier_grid,
    max_sharpe_slsqp,
    mean_variance,
    minimum_variance,
    ridge_mean_variance,
    weights_equal,
    weights_maxsharpe_frontier_grid,
    weights_maxsharpe_slsqp,
    weights_minvar,
    weights_mv,
    weights_ridge_mv,
)
from .selection import (
    best_strategy_by_sharpe,
    calc_drawdown,
    performance_metrics,
    result_sharpe,
    summarize_results,
)
from .universe import (
    build_liquid_universe_by_date,
    clean_close_volume_panels,
    make_rebalance_dates,
    prices_to_returns,
    select_liquid_universe,
)


def backtest(*args, **kwargs):
    from quantfinlab.backtest.portfolio import run_rebalanced_portfolio_backtest

    return run_rebalanced_portfolio_backtest(*args, **kwargs)


def run_rebalanced_portfolio_backtest(*args, **kwargs):
    from quantfinlab.backtest.portfolio import run_rebalanced_portfolio_backtest as _run

    return _run(*args, **kwargs)

DEFAULT_ANNUALIZATION = covariance.DEFAULT_ANNUALIZATION
DEFAULT_SOLVER_ORDER = optimizers.DEFAULT_SOLVER_ORDER

__all__ = [
    "DEFAULT_ANNUALIZATION",
    "DEFAULT_SOLVER_ORDER",
    "apply_transaction_costs",
    "attribution",
    "backtest",
    "bayes_stein_momentum_mu",
    "bayes_stein_mu",
    "best_strategy_by_sharpe",
    "black_litterman",
    "build_liquid_universe_by_date",
    "build_mu_excess_ann",
    "build_scaled_mu_from_raw",
    "calc_drawdown",
    "clean_close_volume_panels",
    "confidence",
    "concentration",
    "constraints",
    "constraints_feasible",
    "costs",
    "cov_estimate",
    "covariance",
    "effective_number_of_holdings",
    "equal_weight",
    "estimate_covariance",
    "ewma_covariance",
    "expected_returns",
    "ledoit_wolf_covariance",
    "long_only_box_constraints",
    "make_psd",
    "make_rebalance_dates",
    "max_sharpe_frontier_grid",
    "max_sharpe_slsqp",
    "max_weight",
    "mean_variance",
    "minimum_variance",
    "momentum_mu",
    "momentum_score_from_returns",
    "mu_diagnostics",
    "mu_momentum",
    "normalize_weights",
    "oas_covariance",
    "optimizers",
    "performance_metrics",
    "portfolio_turnover",
    "prices_to_returns",
    "result_sharpe",
    "ridge_mean_variance",
    "risk_contribution",
    "run_rebalanced_portfolio_backtest",
    "sample_covariance",
    "scale_mu_to_target_sharpe",
    "select_liquid_universe",
    "selection",
    "summarize_results",
    "transaction_cost_from_turnover",
    "universe",
    "views",
    "walkforward",
    "weights_equal",
    "weights_maxsharpe_frontier_grid",
    "weights_maxsharpe_slsqp",
    "weights_minvar",
    "weights_mv",
    "weights_ridge_mv",
    "winsorize_signal",
    "zscore_signal",
]
