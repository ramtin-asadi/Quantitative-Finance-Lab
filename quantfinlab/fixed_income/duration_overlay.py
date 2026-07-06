from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..common.errors import InputError
from . import risk
from .bond_pricing import bond_position_value, position_values_by_bucket
from .tenors import DEFAULT_ISSUE_MATURITIES


def compute_duration_gap(portfolio_duration: float, target_duration: float) -> float:
    """Compute the difference between current and target duration.

    Parameters
    ----------
    portfolio_duration : float
        Current portfolio duration.
    target_duration : float
        Desired target duration.

    Returns
    -------
    float
        ``portfolio_duration - target_duration``.

    Notes
    -----
    A positive value indicates the portfolio is longer than the target; a negative
    value indicates it is shorter.
    """

    return float(portfolio_duration - target_duration)


def duration_overlay_trade(
    *,
    effective_duration: float,
    target_duration: float,
    nav: float,
    duration_sell: float,
    duration_buy: float,
    sell_value_available: float,
) -> float:
    """Compute the notional value to switch between two duration buckets.

    Parameters
    ----------
    effective_duration : float
        Current effective duration of the portfolio.
    target_duration : float
        Desired portfolio duration.
    nav : float
        Portfolio net asset value.
    duration_sell : float
        Duration of the bucket or instrument to be sold.
    duration_buy : float
        Duration of the bucket or instrument to be bought.
    sell_value_available : float
        Maximum market value available to sell.

    Returns
    -------
    float
        Trade value capped by available sell value. Returns zero when inputs are
        non-finite or the buy/sell durations are effectively identical.

    Notes
    -----
    The calculation uses a linear duration-gap approximation and does not account
    for convexity, transaction costs, or the effect of the trade on cash carry.
    """

    if not np.isfinite(effective_duration):
        return 0.0
    if not np.isfinite(duration_sell) or not np.isfinite(duration_buy):
        return 0.0
    if abs(duration_buy - duration_sell) < 1e-8:
        return 0.0
    trade_value = abs(float(nav) * (float(target_duration) - float(effective_duration)) / (duration_buy - duration_sell))
    return float(min(trade_value, max(float(sell_value_available), 0.0)))


def duration_switch_overlay(
    positions: dict[int, dict],
    cash: float,
    date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    target_duration: float,
    duration_band: float,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    short_maturity: int | None = None,
    long_maturity: int | None = None,
    apply_trade_fn: Callable | None = None,
    bump_bp: float = 1.0,
) -> tuple[dict[int, dict], float, list[dict], float]:
    """Adjust a ladder portfolio toward a target duration by switching buckets.

    Parameters
    ----------
    positions : dict[int, dict]
        Current synthetic bond positions by maturity bucket.
    cash : float
        Current cash balance.
    date : pandas.Timestamp
        Valuation and trade date.
    df_func : callable
        Discount-factor function used to value positions and estimate duration.
    target_duration : float
        Desired effective duration.
    duration_band : float
        No-trade band around the target duration.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Maturity buckets considered in the overlay.
    short_maturity : int or None, optional
        Short bucket used when duration needs to be reduced or increased. Defaults
        to the shortest bucket.
    long_maturity : int or None, optional
        Long bucket used when duration needs to be reduced or increased. Defaults
        to the longest bucket.
    apply_trade_fn : callable or None, optional
        Trade-execution callback used to modify positions and cash.
    bump_bp : float, default 1.0
        Bump size in basis points used for duration estimates.

    Returns
    -------
    tuple[dict[int, dict], float, list[dict], float]
        Updated positions, updated cash balance, trade records, and new effective
        duration.

    Raises
    ------
    InputError
        If a target is requested but no trade-application function is supplied.

    Notes
    -----
    If the current duration is within the band, no trade is placed. The function
    sells the long bucket and buys the short bucket when duration is too high, and
    does the reverse when duration is too low.
    """

    if target_duration is None:
        return positions, float(cash), [], np.nan
    if apply_trade_fn is None:
        raise InputError("apply_trade_fn is required for duration_switch_overlay.")

    buckets_l = [int(x) for x in buckets]
    short_maturity = int(short_maturity or min(buckets_l))
    long_maturity = int(long_maturity or max(buckets_l))

    eff_dur = risk.portfolio_parallel_risk(
        positions,
        cash,
        date,
        df_func,
        buckets=buckets_l,
        bump_bp=bump_bp,
    )["effective_duration"]
    if not np.isfinite(eff_dur) or abs(eff_dur - target_duration) <= duration_band:
        return positions, float(cash), [], float(eff_dur)

    bucket_values = position_values_by_bucket(positions, date, df_func, buckets=buckets_l)
    nav = float(cash) + sum(bucket_values.values())

    if eff_dur > target_duration:
        sell_maturity, buy_maturity = long_maturity, short_maturity
    else:
        sell_maturity, buy_maturity = short_maturity, long_maturity

    sell_value_now = bond_position_value(positions.get(sell_maturity), date, df_func)
    if sell_value_now <= 0:
        return positions, float(cash), [], float(eff_dur)

    dur_sell = risk.bond_modified_duration(positions.get(sell_maturity), date, df_func, bump_bp=bump_bp)
    dur_buy = risk.bond_modified_duration(positions.get(buy_maturity), date, df_func, bump_bp=bump_bp)

    trade_value = duration_overlay_trade(
        effective_duration=eff_dur,
        target_duration=target_duration,
        nav=nav,
        duration_sell=dur_sell,
        duration_buy=dur_buy,
        sell_value_available=sell_value_now,
    )
    if trade_value <= 1e-12:
        return positions, float(cash), [], float(eff_dur)

    trade_rows: list[dict] = []
    cash = apply_trade_fn(
        positions,
        cash,
        date,
        df_func,
        sell_maturity,
        -trade_value,
        trade_rows,
        "duration_overlay",
    )
    cash = apply_trade_fn(
        positions,
        cash,
        date,
        df_func,
        buy_maturity,
        +trade_value,
        trade_rows,
        "duration_overlay",
    )

    new_eff_dur = risk.portfolio_parallel_risk(
        positions,
        cash,
        date,
        df_func,
        buckets=buckets_l,
        bump_bp=bump_bp,
    )["effective_duration"]
    return positions, float(cash), trade_rows, float(new_eff_dur)


def apply_duration_targeting(*args, **kwargs):
    """Alias for duration-switch overlay execution.

    Parameters
    ----------
    *args
        Positional arguments forwarded to the duration overlay routine.
    **kwargs
        Keyword arguments forwarded to the duration overlay routine.

    Returns
    -------
    tuple
        Return value of the duration overlay routine.

    Notes
    -----
    This wrapper preserves an alternate public name for the same duration-targeting
    operation.
    """

    return duration_switch_overlay(*args, **kwargs)


__all__ = [
    "apply_duration_targeting",
    "compute_duration_gap",
    "duration_overlay_trade",
    "duration_switch_overlay",
]
