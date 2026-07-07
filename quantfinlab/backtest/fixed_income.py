from __future__ import annotations

import pandas as pd

from ..common.results import SimpleBacktestResult
from ..fixed_income import duration_overlay, ladder


def run_bond_ladder_backtest(
    par_yields: pd.DataFrame,
    *,
    duration_target: float | None = None,
    duration_target_by_date: pd.Series | dict | None = None,
    overlay_fn=None,
    **kwargs,
) -> SimpleBacktestResult:
    """Run a bond-ladder backtest with optional duration targeting.

    This wrapper delegates the fixed-income ladder mechanics to the ladder engine
    and automatically supplies the standard duration-switch overlay when a duration
    target is requested and no overlay function is provided.

    Parameters
    ----------
    par_yields : pandas.DataFrame
        Par-yield curve panel indexed by date with tenor columns in decimal yield
        units.
    duration_target : float, optional
        Static target duration.
    duration_target_by_date : pandas.Series or dict, optional
        Time-varying target duration. The latest available target on or before each
        rebalance date is used by the ladder engine.
    overlay_fn : callable, optional
        Custom duration overlay function. If omitted and a duration target is
        supplied, the default duration-switch overlay is used.
    **kwargs
        Additional keyword arguments passed to the ladder backtest engine.

    Returns
    -------
    SimpleBacktestResult
        Fixed-income backtest result containing NAV, returns, weights/trades/costs
        when available, cashflows, and diagnostic tables.

    Notes
    -----
    The wrapper does not change yield-curve fitting or cashflow conventions; those
    are controlled by the underlying ladder engine and keyword arguments.
    """

    if (duration_target is not None or duration_target_by_date is not None) and overlay_fn is None:
        overlay_fn = duration_overlay.duration_switch_overlay
    return ladder.run_ladder_backtest(
        par_yields,
        duration_target=duration_target,
        duration_target_by_date=duration_target_by_date,
        overlay_fn=overlay_fn,
        **kwargs,
    )


def combine_ladder_with_return_overlay(
    base_result: SimpleBacktestResult,
    overlay_returns: pd.Series,
    *,
    strategy_name: str = "ladder_with_overlay",
    overlay_costs: pd.Series | None = None,
    diagnostics: dict | None = None,
) -> SimpleBacktestResult:
    overlay = pd.Series(overlay_returns, dtype=float).dropna().sort_index()
    idx = base_result.returns.index.intersection(overlay.index)
    base = base_result.returns.reindex(idx).astype(float).fillna(0.0)
    overlay = overlay.reindex(idx).astype(float).fillna(0.0)
    returns = (base + overlay).rename(strategy_name)
    start_nav = float(base_result.nav.reindex(idx).dropna().iloc[0]) if len(idx) else 1.0
    nav = (start_nav * (1.0 + returns).cumprod()).rename(strategy_name)

    costs = None
    if overlay_costs is not None:
        costs = pd.Series(overlay_costs, dtype=float).reindex(idx).fillna(0.0)

    diag = {"base_result": base_result, "overlay_returns": overlay}
    if diagnostics:
        diag.update(diagnostics)

    weights = base_result.weights.reindex(idx) if base_result.weights is not None else None
    return SimpleBacktestResult(
        nav=nav,
        returns=returns,
        weights=weights,
        trades=base_result.trades,
        costs=costs,
        cashflows=None,
        diagnostics=diag,
    )


__all__ = ["combine_ladder_with_return_overlay", "run_bond_ladder_backtest"]
