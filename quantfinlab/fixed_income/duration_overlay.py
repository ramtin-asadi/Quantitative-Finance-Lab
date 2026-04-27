from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..core import InputError
from . import risk
from .bond_pricing import bond_position_value, position_values_by_bucket
from .tenors import DEFAULT_ISSUE_MATURITIES


def compute_duration_gap(portfolio_duration: float, target_duration: float) -> float:
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
    return duration_switch_overlay(*args, **kwargs)


__all__ = [
    "apply_duration_targeting",
    "compute_duration_gap",
    "duration_overlay_trade",
    "duration_switch_overlay",
]
