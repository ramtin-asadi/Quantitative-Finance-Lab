from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..common.dates import yearfrac
from ..common.errors import BacktestError, InputError
from ..common.results import SimpleBacktestResult
from . import bootstrap, smoothers
from .bond_pricing import (
    bond_cashflows_between,
    bond_position_value,
    make_synthetic_bond,
    position_values_by_bucket,
    remaining_maturity,
)
from .discounting import curve_date_for
from .risk import strategy_risk_timeseries
from .tenors import DEFAULT_ISSUE_MATURITIES, nearest_tenor_label


def clone_positions(positions: dict[int, dict]) -> dict[int, dict]:
    return copy.deepcopy(positions)


def split_contiguous_blocks(
    dates,
    *,
    max_gap_days: int = 45,
) -> list[pd.DatetimeIndex]:
    dates_idx = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if len(dates_idx) == 0:
        return []

    blocks = []
    start = 0
    for i in range(1, len(dates_idx)):
        if (dates_idx[i] - dates_idx[i - 1]).days > max_gap_days:
            blocks.append(dates_idx[start:i])
            start = i
    blocks.append(dates_idx[start : len(dates_idx)])
    return [pd.DatetimeIndex(b) for b in blocks if len(b) > 0]


def choose_backtest_block(
    dates,
    *,
    max_gap_days: int = 45,
    min_len: int = 60,
) -> pd.DatetimeIndex:
    blocks = split_contiguous_blocks(dates, max_gap_days=max_gap_days)
    if not blocks:
        return pd.DatetimeIndex([])

    eligible = [b for b in blocks if len(b) >= min_len]
    if not eligible:
        return pd.DatetimeIndex([])
    return max(eligible, key=len)


def gap_safe_frame(obj: pd.Series | pd.DataFrame, *, max_gap_days: int = 45):
    out = obj.copy()
    if len(out.index) == 0:
        return out

    gap_mask = out.index.to_series().diff().dt.days.gt(max_gap_days)
    gap_dates = gap_mask[gap_mask].index
    if isinstance(out, pd.Series):
        out = out.astype(float)
        out.loc[gap_dates] = np.nan
    else:
        out = out.astype(float)
        out.loc[gap_dates, :] = np.nan
    return out


def ladder_returns(strategy_df: pd.DataFrame) -> pd.Series:
    return strategy_df["ret"].copy()


def ladder_nav(strategy_df: pd.DataFrame) -> pd.Series:
    return strategy_df["nav"].copy()


def ladder_performance_table(strategy_df: pd.DataFrame) -> pd.DataFrame:
    rets = strategy_df["ret"].dropna()
    if len(rets) == 0:
        return pd.DataFrame()

    wealth = (1.0 + rets).cumprod()
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    ann_return = wealth.iloc[-1] ** (12.0 / len(rets)) - 1.0
    ann_vol = rets.std(ddof=1) * np.sqrt(12.0) if len(rets) > 1 else np.nan
    max_dd = drawdown.min()

    return pd.DataFrame(
        [
            {
                "final_nav": strategy_df["nav"].iloc[-1],
                "annualized_return": ann_return,
                "annualized_vol": ann_vol,
                "max_drawdown": max_dd,
            }
        ],
        index=[strategy_df["strategy"].iloc[0]],
    )


def _target_weights_series(
    target_weights: dict[int, float] | None,
    buckets: list[int],
) -> pd.Series:
    if target_weights is None:
        target_weights = {m: 1.0 / len(buckets) for m in buckets}
    w = pd.Series({int(k): float(v) for k, v in target_weights.items()}, dtype=float).reindex(buckets)
    if w.isna().any():
        raise InputError("target_weights must provide every ladder bucket.")
    total = float(w.sum())
    if total <= 0:
        raise InputError("target_weights must sum to a positive value.")
    return w / total


def _issue_labels_for(par_yields: pd.DataFrame, buckets: list[int]) -> dict[int, str]:
    cols = [str(c) for c in par_yields.columns]
    return {m: nearest_tenor_label(cols, target_maturity_years=m) for m in buckets}


def _get_curve_builder(
    par_yields: pd.DataFrame,
    *,
    curve_method: str,
    tenor_cols: list[str],
    freq: int,
    short_end: str,
    min_df: float,
):
    curve_cache: dict[pd.Timestamp, Any] = {}

    def get_curve_for(date: pd.Timestamp):
        cd = curve_date_for(par_yields.index, pd.Timestamp(date))
        if cd is None:
            return None
        if cd in curve_cache:
            return curve_cache[cd]
        row = par_yields.loc[cd]
        try:
            pillars = bootstrap.bootstrap_pillars(
                row,
                asof=cd,
                tenor_cols=tenor_cols,
                freq=freq,
                short_end=short_end,
                min_df=min_df,
            )
            curves = smoothers.fit_curves(pillars, methods=(curve_method,), freq=freq, min_df=min_df)
            curve_cache[cd] = curves[curve_method]
        except Exception:
            curve_cache[cd] = None
        return curve_cache[cd]

    return get_curve_for


def make_curve_lookup(
    par_yields: pd.DataFrame,
    *,
    curve_method: str,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    short_end: str = "continuous",
    min_df: float = 1e-12,
):
    """
    Build the cached primary-curve lookup used by the ladder notebook cells.
    """
    cols = tenor_cols if tenor_cols is not None else [str(c) for c in par_yields.columns]
    return _get_curve_builder(
        par_yields[cols],
        curve_method=curve_method,
        tenor_cols=cols,
        freq=freq,
        short_end=short_end,
        min_df=min_df,
    )


def _cash_rate_lookup(
    par_yields: pd.DataFrame,
    *,
    cash_tenor_label: str,
    fallback_labels: tuple[str, ...],
) -> Callable[[pd.Timestamp], float]:
    def get_cash_rate(date: pd.Timestamp) -> float:
        curve_date = curve_date_for(par_yields.index, pd.Timestamp(date))
        if curve_date is None:
            return 0.0
        labels = (cash_tenor_label, *fallback_labels)
        for label in labels:
            if label in par_yields.columns:
                r = float(par_yields.loc[curve_date, label])
                if np.isfinite(r):
                    return r
        return 0.0

    return get_cash_rate


def initialize_ladder(
    start_date: pd.Timestamp,
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    target_weights: dict[int, float] | None = None,
    initial_nav: float = 100.0,
    coupon_lookup: Callable[[pd.Timestamp, int], float],
    freq: int = 2,
) -> tuple[dict[int, dict], float]:
    buckets_l = [int(x) for x in buckets]
    weights = _target_weights_series(target_weights, buckets_l)
    positions: dict[int, dict] = {}
    cash = float(initial_nav)
    for maturity in buckets_l:
        coupon = coupon_lookup(start_date, maturity)
        units = float(initial_nav) * float(weights.loc[maturity])
        positions[maturity] = make_synthetic_bond(
            start_date,
            maturity,
            coupon,
            units=units,
            freq=freq,
        )
        cash -= units
    return positions, float(cash)


def _cost_bps_for_maturity(trading_cost_bps, maturity: int) -> float:
    if callable(trading_cost_bps):
        return float(trading_cost_bps(int(maturity)))
    if isinstance(trading_cost_bps, dict):
        if int(maturity) in trading_cost_bps:
            return float(trading_cost_bps[int(maturity)])
        if str(maturity) in trading_cost_bps:
            return float(trading_cost_bps[str(maturity)])
    return float(trading_cost_bps)


def apply_trade(
    positions: dict[int, dict],
    cash: float,
    date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    maturity: int,
    target_value_delta: float,
    trade_rows: list[dict],
    reason: str,
    *,
    coupon_lookup: Callable[[pd.Timestamp, int], float],
    trading_cost_bps: float = 1.0,
    freq: int = 2,
) -> float:
    maturity = int(maturity)
    tc = _cost_bps_for_maturity(trading_cost_bps, maturity) / 10000.0

    if maturity not in positions or positions[maturity] is None:
        coupon = coupon_lookup(date, maturity)
        positions[maturity] = make_synthetic_bond(date, maturity, coupon, units=0.0, freq=freq)

    bond = positions[maturity]
    price = bond_position_value(bond, date, df_func) / bond["units"] if bond["units"] > 0 else 1.0

    if price <= 0:
        return float(cash)

    if target_value_delta < -1e-12:
        sell_value = min(-target_value_delta, bond_position_value(bond, date, df_func))
        sell_units = sell_value / price
        cost = sell_value * tc
        bond["units"] -= sell_units
        cash += sell_value - cost
        trade_rows.append(
            {
                "date": date,
                "maturity": maturity,
                "side": "sell",
                "notional": sell_value,
                "price": price,
                "units": sell_units,
                "cost": cost,
                "reason": reason,
            }
        )
        if bond["units"] <= 1e-12:
            positions.pop(maturity, None)
    elif target_value_delta > 1e-12:
        buy_value = min(target_value_delta, cash / (1.0 + tc))
        buy_units = buy_value / price
        cost = buy_value * tc
        bond["units"] += buy_units
        cash -= buy_value + cost
        trade_rows.append(
            {
                "date": date,
                "maturity": maturity,
                "side": "buy",
                "notional": buy_value,
                "price": price,
                "units": buy_units,
                "cost": cost,
                "reason": reason,
            }
        )

    return float(cash)


def roll_bucket_positions(
    positions: dict[int, dict],
    cash: float,
    date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    bucket_floor: dict[int, float] | None = None,
    trading_cost_bps: float = 1.0,
) -> tuple[dict[int, dict], float, list[dict]]:
    buckets_l = [int(x) for x in buckets]
    if bucket_floor is None:
        bucket_floor = {2: 1.5, 5: 3.5, 10: 7.5, 30: 20.0}
    trade_rows: list[dict] = []

    for maturity in buckets_l:
        bond = positions.get(maturity)
        if bond is None:
            continue

        rem = remaining_maturity(bond, date)
        value = bond_position_value(bond, date, df_func)
        if rem <= 1e-10:
            positions.pop(maturity, None)
            continue

        floor = float(bucket_floor.get(maturity, 0.0))
        if rem < floor:
            tc = _cost_bps_for_maturity(trading_cost_bps, maturity) / 10000.0
            cost = value * tc
            cash += value - cost
            trade_rows.append(
                {
                    "date": date,
                    "maturity": maturity,
                    "side": "sell",
                    "notional": value,
                    "price": value / max(bond["units"], 1e-12),
                    "units": bond["units"],
                    "cost": cost,
                    "reason": "roll",
                }
            )
            positions.pop(maturity, None)

    return positions, float(cash), trade_rows


def rebalance_ladder_to_buckets(
    positions: dict[int, dict],
    cash: float,
    date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    target_weights: dict[int, float],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    rebalance_band: float = 0.05,
    coupon_lookup: Callable[[pd.Timestamp, int], float],
    trading_cost_bps: float = 1.0,
    freq: int = 2,
    reason: str = "rebalance",
) -> tuple[dict[int, dict], float, list[dict]]:
    buckets_l = [int(x) for x in buckets]
    weights = _target_weights_series(target_weights, buckets_l)
    trade_rows: list[dict] = []

    bucket_values = position_values_by_bucket(positions, date, df_func, buckets=buckets_l)
    nav = float(cash) + sum(bucket_values.values())
    if nav <= 0:
        return positions, float(cash), trade_rows

    current_weights = {m: bucket_values[m] / nav for m in buckets_l}
    need_rebalance = any(abs(current_weights[m] - weights.loc[m]) > rebalance_band for m in buckets_l)
    missing_bucket = any((m not in positions) or (bucket_values[m] <= 1e-12) for m in buckets_l)
    if not need_rebalance and not missing_bucket:
        return positions, float(cash), trade_rows

    target_values = {m: nav * weights.loc[m] for m in buckets_l}
    deltas = {m: target_values[m] - bucket_values[m] for m in buckets_l}

    for maturity in buckets_l:
        if deltas[maturity] < -1e-12:
            cash = apply_trade(
                positions,
                cash,
                date,
                df_func,
                maturity,
                deltas[maturity],
                trade_rows,
                reason,
                coupon_lookup=coupon_lookup,
                trading_cost_bps=trading_cost_bps,
                freq=freq,
            )

    bucket_values = position_values_by_bucket(positions, date, df_func, buckets=buckets_l)
    nav = float(cash) + sum(bucket_values.values())
    target_values = {m: nav * weights.loc[m] for m in buckets_l}
    deltas = {m: target_values[m] - bucket_values[m] for m in buckets_l}

    for maturity in buckets_l:
        if deltas[maturity] > 1e-12:
            cash = apply_trade(
                positions,
                cash,
                date,
                df_func,
                maturity,
                deltas[maturity],
                trade_rows,
                reason,
                coupon_lookup=coupon_lookup,
                trading_cost_bps=trading_cost_bps,
                freq=freq,
            )

    return positions, float(cash), trade_rows


def rebalance_to_targets(*args, **kwargs):
    return rebalance_ladder_to_buckets(*args, **kwargs)


def run_ladder_backtest(
    par_yields: pd.DataFrame,
    *,
    strategy_name: str = "bond_ladder",
    curve_method: str = "pchip",
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    target_weights: dict[int, float] | None = None,
    bucket_floor: dict[int, float] | None = None,
    rebalance_band: float = 0.05,
    trading_cost_bps: float = 1.0,
    initial_nav: float = 100.0,
    cash_tenor_label: str = "1M",
    cash_fallback_labels: tuple[str, ...] = ("3M", "6M", "1Y"),
    duration_target: float | None = None,
    duration_target_by_date: pd.Series | dict | None = None,
    duration_band: float | None = None,
    overlay_fn: Callable | None = None,
    freq: int = 2,
    short_end: str = "continuous",
    min_df: float = 1e-12,
    max_gap_days: int = 45,
    min_block_len: int = 60,
    tenor_cols: list[str] | None = None,
    curve_lookup: Callable[[pd.Timestamp], Any] | None = None,
    risk_bucket_bounds: dict[int, tuple[float, float]] | None = None,
) -> SimpleBacktestResult:
    if par_yields.empty:
        raise InputError("par_yields is empty.")

    py = par_yields.copy().sort_index()
    py.index = pd.to_datetime(py.index)
    buckets_l = [int(x) for x in buckets]
    weights = _target_weights_series(target_weights, buckets_l)
    tenor_cols = tenor_cols if tenor_cols is not None else [str(c) for c in py.columns]
    issue_labels = _issue_labels_for(py[tenor_cols], buckets_l)
    month_end_curve = py[tenor_cols].resample("ME").last().dropna(how="all")
    issue_dates = month_end_curve.index

    def coupon_lookup(date: pd.Timestamp, maturity: int) -> float:
        label = issue_labels[int(maturity)]
        y = float(month_end_curve.loc[pd.Timestamp(date), label])
        if not np.isfinite(y):
            raise BacktestError(f"Missing par yield for {label} on {pd.Timestamp(date).date()}.")
        return y

    get_curve_for = curve_lookup or make_curve_lookup(
        py[tenor_cols],
        curve_method=curve_method,
        tenor_cols=tenor_cols,
        freq=freq,
        short_end=short_end,
        min_df=min_df,
    )
    get_cash_rate = _cash_rate_lookup(
        py,
        cash_tenor_label=cash_tenor_label,
        fallback_labels=cash_fallback_labels,
    )

    def grow_cash(cash_value: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> float:
        dt = yearfrac(start_date, end_date)
        r = get_cash_rate(start_date)
        return float(cash_value * math.exp(r * dt))

    all_valid_dates = []
    needed_cols = list(issue_labels.values())
    if cash_tenor_label in month_end_curve.columns:
        needed_cols.append(cash_tenor_label)
    for d in issue_dates:
        curve = get_curve_for(d)
        if curve is not None and month_end_curve.loc[d, needed_cols].notna().all():
            all_valid_dates.append(d)

    valid_dates = choose_backtest_block(
        all_valid_dates,
        max_gap_days=max_gap_days,
        min_len=min_block_len,
    )
    if len(valid_dates) == 0:
        raise BacktestError("No contiguous valid block available for the chosen curve method.")

    start_date = pd.Timestamp(valid_dates[0])
    positions, cash = initialize_ladder(
        start_date,
        buckets=buckets_l,
        target_weights=weights.to_dict(),
        initial_nav=initial_nav,
        coupon_lookup=coupon_lookup,
        freq=freq,
    )

    rows: list[dict] = []
    bucket_rows: list[dict] = []
    trade_rows: list[dict] = []
    carry_rows: list[dict] = []
    snapshots: dict[pd.Timestamp, dict] = {}

    start_curve = get_curve_for(start_date)
    df_start = start_curve.df
    start_bucket_values = position_values_by_bucket(positions, start_date, df_start, buckets=buckets_l)
    start_nav = float(cash) + sum(start_bucket_values.values())

    rows.append(
        {
            "date": start_date,
            "strategy": strategy_name,
            "nav": start_nav,
            "cash": cash,
            "ret": np.nan,
            "coupon_income": 0.0,
            "principal_income": 0.0,
            "cash_carry": 0.0,
            "curve_move_pnl": 0.0,
            "carry_roll_pnl": 0.0,
            "turnover": 0.0,
        }
    )
    for maturity in buckets_l:
        bucket_rows.append(
            {
                "date": start_date,
                "strategy": strategy_name,
                "maturity": maturity,
                "pv": start_bucket_values[maturity],
                "weight": start_bucket_values[maturity] / start_nav if start_nav > 0 else np.nan,
            }
        )
    snapshots[start_date] = {"positions": clone_positions(positions), "cash": float(cash)}

    target_series = None
    if duration_target_by_date is not None:
        target_series = pd.Series(duration_target_by_date, dtype=float).sort_index()
        target_series.index = pd.to_datetime(target_series.index)

    def duration_target_for(date: pd.Timestamp) -> float | None:
        if target_series is not None:
            history = target_series.loc[target_series.index <= pd.Timestamp(date)].dropna()
            if len(history) > 0:
                return float(history.iloc[-1])
        return None if duration_target is None else float(duration_target)

    if (duration_target is not None or target_series is not None) and overlay_fn is None:
        from .duration_overlay import duration_switch_overlay

        overlay_fn = duration_switch_overlay

    def apply_trade_bound(
        positions_arg,
        cash_arg,
        date_arg,
        df_func_arg,
        maturity_arg,
        target_delta_arg,
        trade_rows_arg,
        reason_arg,
    ) -> float:
        return apply_trade(
            positions_arg,
            cash_arg,
            date_arg,
            df_func_arg,
            maturity_arg,
            target_delta_arg,
            trade_rows_arg,
            reason_arg,
            coupon_lookup=coupon_lookup,
            trading_cost_bps=trading_cost_bps,
            freq=freq,
        )

    for i in range(1, len(valid_dates)):
        prev_date = pd.Timestamp(valid_dates[i - 1])
        date = pd.Timestamp(valid_dates[i])
        gap_days = (date - prev_date).days
        if gap_days > max_gap_days:
            raise BacktestError(
                f"Gap of {gap_days} days detected between {prev_date.date()} and {date.date()}."
            )

        prev_curve = get_curve_for(prev_date)
        current_curve = get_curve_for(date)
        if prev_curve is None or current_curve is None:
            continue
        df_prev = prev_curve.df
        df_now = current_curve.df

        positions_start = clone_positions(positions)
        cash_start = float(cash)
        nav_start = cash_start + sum(position_values_by_bucket(positions_start, prev_date, df_prev, buckets=buckets_l).values())

        cash = grow_cash(cash, prev_date, date)
        cash_carry = cash - cash_start

        coupon_income = 0.0
        principal_income = 0.0
        for maturity in list(positions.keys()):
            gross, coupon, principal = bond_cashflows_between(positions[maturity], prev_date, date)
            cash += gross
            coupon_income += coupon
            principal_income += principal

        same_curve_positions = sum(position_values_by_bucket(positions_start, date, df_prev, buckets=buckets_l).values())
        same_curve_nav = cash_start + cash_carry + coupon_income + principal_income + same_curve_positions
        actual_positions_pretrade = sum(position_values_by_bucket(positions, date, df_now, buckets=buckets_l).values())
        actual_nav_pretrade = cash + actual_positions_pretrade

        carry_roll_pnl = same_curve_nav - nav_start
        curve_move_pnl = actual_nav_pretrade - same_curve_nav

        positions, cash, roll_trades = roll_bucket_positions(
            positions,
            cash,
            date,
            df_now,
            buckets=buckets_l,
            bucket_floor=bucket_floor,
            trading_cost_bps=trading_cost_bps,
        )
        period_trades = list(roll_trades)

        positions, cash, rebalance_trades = rebalance_ladder_to_buckets(
            positions,
            cash,
            date,
            df_now,
            weights.to_dict(),
            buckets=buckets_l,
            rebalance_band=rebalance_band,
            coupon_lookup=coupon_lookup,
            trading_cost_bps=trading_cost_bps,
            freq=freq,
        )
        period_trades.extend(rebalance_trades)

        period_duration_target = duration_target_for(date)
        if period_duration_target is not None and overlay_fn is not None:
            positions, cash, overlay_trades, _ = overlay_fn(
                positions,
                cash,
                date,
                df_now,
                target_duration=period_duration_target,
                duration_band=float(duration_band or 0.0),
                buckets=buckets_l,
                apply_trade_fn=apply_trade_bound,
            )
            period_trades.extend(overlay_trades)

        bucket_values = position_values_by_bucket(positions, date, df_now, buckets=buckets_l)
        end_nav = cash + sum(bucket_values.values())
        turnover = sum(tr["notional"] for tr in period_trades) / nav_start if nav_start > 0 else np.nan

        rows.append(
            {
                "date": date,
                "strategy": strategy_name,
                "nav": end_nav,
                "cash": cash,
                "ret": (end_nav / nav_start - 1.0) if nav_start > 0 else np.nan,
                "coupon_income": coupon_income,
                "principal_income": principal_income,
                "cash_carry": cash_carry,
                "curve_move_pnl": curve_move_pnl,
                "carry_roll_pnl": carry_roll_pnl,
                "turnover": turnover,
            }
        )

        for maturity in buckets_l:
            bucket_rows.append(
                {
                    "date": date,
                    "strategy": strategy_name,
                    "maturity": maturity,
                    "pv": bucket_values[maturity],
                    "weight": bucket_values[maturity] / end_nav if end_nav > 0 else np.nan,
                }
            )

        for tr in period_trades:
            tr["strategy"] = strategy_name
            trade_rows.append(tr)

        coupon_carry_pnl = coupon_income + cash_carry
        roll_pnl = carry_roll_pnl - coupon_carry_pnl
        nav_denom = nav_start if nav_start > 0 else np.nan
        carry_rows.append(
            {
                "date": date,
                "strategy": strategy_name,
                "coupon_carry_pnl": coupon_carry_pnl,
                "roll_pnl": roll_pnl,
                "curve_move_pnl": curve_move_pnl,
                "coupon_carry_ret": coupon_carry_pnl / nav_denom if nav_start > 0 else np.nan,
                "roll_ret": roll_pnl / nav_denom if nav_start > 0 else np.nan,
                "curve_move_ret": curve_move_pnl / nav_denom if nav_start > 0 else np.nan,
                "explained_ret": (coupon_carry_pnl + roll_pnl + curve_move_pnl) / nav_denom
                if nav_start > 0
                else np.nan,
            }
        )
        snapshots[date] = {"positions": clone_positions(positions), "cash": float(cash)}

    strategy_df = pd.DataFrame(rows).set_index("date").sort_index()
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["date", "maturity"])
    trade_df = pd.DataFrame(trade_rows)
    carry_df = pd.DataFrame(carry_rows).set_index("date").sort_index() if carry_rows else pd.DataFrame()
    weights_df = bucket_df.pivot(index="date", columns="maturity", values="weight").sort_index()
    costs = (
        trade_df.groupby("date")["cost"].sum().reindex(strategy_df.index).fillna(0.0)
        if not trade_df.empty
        else pd.Series(0.0, index=strategy_df.index, name="cost")
    )

    risk_df, krd_df = strategy_risk_timeseries(
        strategy_df,
        snapshots,
        get_curve_for,
        buckets=buckets_l,
        bucket_bounds=risk_bucket_bounds,
    )

    diagnostics = {
        "strategy": strategy_df,
        "buckets": bucket_df,
        "bucket_values": bucket_df.pivot(index="date", columns="maturity", values="pv").sort_index(),
        "weights": weights_df,
        "trades": trade_df,
        "carry": carry_df,
        "snapshots": snapshots,
        "risk": risk_df,
        "krd": krd_df,
        "duration": risk_df["effective_duration"] if "effective_duration" in risk_df else pd.Series(dtype=float),
        "performance": ladder_performance_table(strategy_df),
        "issue_labels": issue_labels,
        "curve_method": curve_method,
        "valid_dates": valid_dates,
        "duration_target_by_date": target_series,
    }

    return SimpleBacktestResult(
        nav=ladder_nav(strategy_df),
        returns=ladder_returns(strategy_df),
        weights=weights_df,
        trades=trade_df,
        costs=costs,
        cashflows=carry_df,
        diagnostics=diagnostics,
    )


def prepare_secondary_curve_market(
    par_yields: pd.DataFrame,
    *,
    selected_tenors: list[str] | None = None,
    min_len: int = 60,
    max_gap_days: int = 45,
):
    from .bootstrap import build_zero_curve_panel_from_par_yields

    monthly_all = par_yields.sort_index().resample("ME").last()
    if selected_tenors is None:
        selected_tenors = [str(c) for c in monthly_all.columns]
    monthly = monthly_all[selected_tenors].dropna()
    curve_dates = choose_backtest_block(monthly.index, min_len=min_len, max_gap_days=max_gap_days)
    monthly = monthly.loc[curve_dates].copy()
    zero_rates = build_zero_curve_panel_from_par_yields(
        monthly,
        method="pchip",
        tenors=selected_tenors,
        as_continuous=True,
    )
    zero_rates = zero_rates.reindex(monthly.index).astype(float)
    zero_rates.columns = zero_rates.columns.astype(float)
    return monthly, zero_rates, pd.DatetimeIndex(monthly.index)


def build_duration_reference_ladders(
    par_yields: pd.DataFrame,
    duration_targets: dict[str, float],
    *,
    neutral_name: str = "neutral duration",
    strategy_prefix: str = "",
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    target_weights: dict[int, float] | None = None,
    trading_cost_bps: float | dict | Callable = 1.0,
    duration_band: float = 0.20,
    tenor_cols: list[str] | None = None,
    risk_bucket_bounds: dict[int, tuple[float, float]] | None = None,
    min_block_len: int = 60,
    **kwargs,
):
    out = {}
    for name, target in duration_targets.items():
        out[name] = run_ladder_backtest(
            par_yields,
            strategy_name=f"{strategy_prefix}{name}".strip(),
            buckets=buckets,
            target_weights=target_weights,
            trading_cost_bps=trading_cost_bps,
            duration_target=float(target),
            duration_band=float(duration_band),
            tenor_cols=tenor_cols,
            risk_bucket_bounds=risk_bucket_bounds,
            min_block_len=min_block_len,
            **kwargs,
        )
    if neutral_name not in out and 5.0 in [float(v) for v in duration_targets.values()]:
        neutral_key = next(k for k, v in duration_targets.items() if float(v) == 5.0)
        out[neutral_name] = out[neutral_key]
    return out


def forward_carry_roll_panel(
    reference_results: dict[str, SimpleBacktestResult],
    par_yields: pd.DataFrame,
    curve_dates,
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    tenor_cols: list[str] | None = None,
    curve_method: str = "pchip",
    cash_tenor_label: str = "3M",
    freq: int = 2,
    short_end: str = "continuous",
):
    curve_dates = pd.DatetimeIndex(curve_dates)
    buckets_l = [int(x) for x in buckets]
    curve_lookup = make_curve_lookup(
        par_yields,
        curve_method=curve_method,
        tenor_cols=tenor_cols,
        freq=freq,
        short_end=short_end,
    )

    def one_return(result, date):
        loc = curve_dates.get_loc(pd.Timestamp(date))
        if loc >= len(curve_dates) - 1:
            return np.nan
        next_date = curve_dates[loc + 1]
        snapshot = result.diagnostics["snapshots"].get(pd.Timestamp(date))
        curve = curve_lookup(date)
        if snapshot is None or curve is None:
            return np.nan
        positions = clone_positions(snapshot["positions"])
        cash = float(snapshot["cash"])
        df_func = curve.df
        nav_start = cash + sum(position_values_by_bucket(positions, date, df_func, buckets=buckets_l).values())
        cash_rate = float(par_yields.loc[pd.Timestamp(date), cash_tenor_label]) if cash_tenor_label in par_yields else 0.0
        cash_next = cash * math.exp(cash_rate * yearfrac(date, next_date))
        income = 0.0
        for maturity in list(positions):
            gross, _, _ = bond_cashflows_between(positions[maturity], date, next_date)
            income += gross
        value_next = sum(position_values_by_bucket(positions, next_date, df_func, buckets=buckets_l).values())
        return (cash_next + income + value_next) / max(nav_start, 1e-12) - 1

    return pd.DataFrame(
        {
            name: pd.Series({date: one_return(result, date) for date in curve_dates[:-1]}, name=name)
            for name, result in reference_results.items()
        }
    )


def target_score_table(
    date,
    model_name: str,
    *,
    target_names: list[str],
    duration_targets: dict[str, float],
    reference_krd: dict[str, pd.DataFrame],
    reference_carry: pd.DataFrame,
    model_view_func: Callable,
    key_maturities,
    previous_target: str,
    neutral_name: str = "neutral duration",
    risk_penalty: float = 0.15,
):
    expected_change, cov = model_view_func(model_name, date, key_maturities)
    rows = []
    base_krd = reference_krd[neutral_name].loc[:date].iloc[-1].to_numpy(float)
    for target_name in target_names:
        krd = reference_krd[target_name].loc[:date].iloc[-1].to_numpy(float)
        active_krd = krd - base_krd
        carry_diff = float(reference_carry.loc[date, target_name] - reference_carry.loc[date, neutral_name])
        curve_return = -float(active_krd @ expected_change)
        active_risk = float(np.sqrt(max(active_krd @ cov @ active_krd, 1e-12)))
        transition_cost = 0.0 if target_name == previous_target else 0.00020 + 0.00005 * abs(duration_targets[target_name] - duration_targets[previous_target])
        expected_active = carry_diff + curve_return - transition_cost
        score = (expected_active - float(risk_penalty) * active_risk) / active_risk
        rows.append(
            {
                "target": target_name,
                "target duration": duration_targets[target_name],
                "expected active return": expected_active,
                "active risk": active_risk,
                "score": score,
                "carry difference": carry_diff,
                "curve view return": curve_return,
                "transition cost": transition_cost,
            }
        )
    out = pd.DataFrame(rows).set_index("target")
    out.insert(0, "date", pd.Timestamp(date))
    out.insert(1, "model", model_name)
    return out


def select_dynamic_duration_targets(
    model_target_by_date: pd.DataFrame,
    model_returns: pd.DataFrame,
    baseline_returns: pd.Series,
    *,
    validation_months: int = 36,
    neutral_duration: float = 5.0,
):
    model_index = model_target_by_date.dropna(how="all").index
    rows = []
    for i in range(int(validation_months), len(model_index)):
        decision_date = model_index[i]
        train_end = model_index[i - 1]
        train = model_returns.loc[:train_end].tail(int(validation_months))
        train_base = baseline_returns.reindex(train.index)
        active = train.sub(train_base, axis=0)
        vol = active.std(ddof=1).replace(0, np.nan)
        score = active.mean() / vol
        selected_model = score.idxmax()
        target_duration = float(model_target_by_date.loc[decision_date, selected_model])
        rows.append(
            {
                "date": decision_date,
                "selected model": selected_model,
                "target duration": target_duration,
                "validation score": score.loc[selected_model],
            }
        )
    log = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    target_by_date = pd.Series(float(neutral_duration), index=model_target_by_date.index, name="dynamic target duration")
    if not log.empty:
        target_by_date.loc[log.index] = log["target duration"]
    return log, target_by_date


def active_returns_against_baseline(returns, baseline_returns: pd.Series):
    data = pd.DataFrame(returns).copy()
    base = baseline_returns.reindex(data.index)
    return data.sub(base, axis=0)


__all__ = [
    "active_returns_against_baseline",
    "apply_trade",
    "build_duration_reference_ladders",
    "choose_backtest_block",
    "clone_positions",
    "forward_carry_roll_panel",
    "gap_safe_frame",
    "initialize_ladder",
    "ladder_nav",
    "ladder_performance_table",
    "ladder_returns",
    "make_curve_lookup",
    "prepare_secondary_curve_market",
    "rebalance_ladder_to_buckets",
    "rebalance_to_targets",
    "roll_bucket_positions",
    "run_ladder_backtest",
    "select_dynamic_duration_targets",
    "split_contiguous_blocks",
    "target_score_table",
]
