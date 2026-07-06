from __future__ import annotations

import numpy as np
import pandas as pd


def zero_rate_at(zero_rates: pd.DataFrame, date, times, *, shift: float = 0.0):
    """Interpolate zero rates at selected maturities for one date.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel with maturity columns in years.
    date : date-like
        Date to select from the panel.
    times : array-like
        Maturities in years at which to interpolate rates.
    shift : float, default 0.0
        Additive rate shift in decimal units.

    Returns
    -------
    numpy.ndarray
        Interpolated zero rates plus the optional shift.

    Notes
    -----
    Rates outside the maturity grid are flat-extrapolated from the nearest endpoint.
    """

    grid = zero_rates.columns.to_numpy(float)
    values = zero_rates.loc[pd.Timestamp(date)].to_numpy(float)
    return np.interp(np.asarray(times, dtype=float), grid, values, left=values[0], right=values[-1]) + float(shift)


def discount_at(zero_rates: pd.DataFrame, date, times, *, shift: float = 0.0):
    """Compute discount factors from a zero-rate panel for one date.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel with maturity columns in years.
    date : date-like
        Date to select from the panel.
    times : array-like
        Maturities in years.
    shift : float, default 0.0
        Additive zero-rate shift in decimal units.

    Returns
    -------
    numpy.ndarray
        Discount factors ``exp(-r(t) * t)`` at the requested maturities.
    """

    t = np.asarray(times, dtype=float)
    return np.exp(-zero_rate_at(zero_rates, date, t, shift=shift) * t)


def swap_schedule(remaining_tenor, *, fixed_freq: int = 2):
    """Build a fixed-leg swap payment schedule.

    Parameters
    ----------
    remaining_tenor : float
        Remaining swap tenor in years.
    fixed_freq : int, default 2
        Fixed-leg payment frequency per year.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Payment times in years and accrual fractions between adjacent payments.

    Notes
    -----
    The schedule uses regular intervals of ``1 / fixed_freq`` and appends the exact
    remaining tenor when needed to include a final stub.
    """

    step = 1 / int(fixed_freq)
    remaining = max(float(remaining_tenor), step)
    times = np.arange(step, remaining + 1e-10, step)
    if len(times) == 0 or abs(times[-1] - remaining) > 1e-8:
        times = np.r_[times, remaining]
    accruals = np.diff(np.r_[0.0, times])
    return times, accruals


def swap_annuity(zero_rates: pd.DataFrame, date, remaining_tenor, *, shift: float = 0.0, fixed_freq: int = 2):
    """Compute fixed-leg swap annuity from a zero-rate panel.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    date : date-like
        Valuation date.
    remaining_tenor : float
        Remaining swap tenor in years.
    shift : float, default 0.0
        Additive zero-rate shift in decimal units.
    fixed_freq : int, default 2
        Fixed-leg payment frequency per year.

    Returns
    -------
    float
        Present value of one unit of fixed-leg coupon rate.

    Notes
    -----
    The annuity is the sum of accrual fractions multiplied by corresponding
    discount factors.
    """

    times, accruals = swap_schedule(remaining_tenor, fixed_freq=fixed_freq)
    return float(np.sum(accruals * discount_at(zero_rates, date, times, shift=shift)))


def par_swap_rate(zero_rates: pd.DataFrame, date, tenor, *, fixed_freq: int = 2):
    """Compute the par fixed rate of a plain fixed-for-floating swap.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    date : date-like
        Valuation date.
    tenor : float
        Swap tenor in years.
    fixed_freq : int, default 2
        Fixed-leg payment frequency per year.

    Returns
    -------
    float
        Par swap fixed rate in decimal units.

    Notes
    -----
    The floating leg is approximated as ``1 - DF(T)`` and the fixed leg uses the
    computed annuity.
    """

    annuity = swap_annuity(zero_rates, date, tenor, fixed_freq=fixed_freq)
    maturity_df = float(discount_at(zero_rates, date, [tenor])[0])
    return (1.0 - maturity_df) / max(annuity, 1e-12)


def swap_value(
    zero_rates: pd.DataFrame,
    date,
    remaining_tenor,
    fixed_rate,
    *,
    notional: float = 1.0,
    side: str = "receiver",
    shift: float = 0.0,
    fixed_freq: int = 2,
):
    """Value a plain fixed-for-floating interest-rate swap.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    date : date-like
        Valuation date.
    remaining_tenor : float
        Remaining swap tenor in years.
    fixed_rate : float
        Fixed coupon rate of the swap in decimal units.
    notional : float, default 1.0
        Swap notional.
    side : {"receiver", "payer"}, default "receiver"
        Receiver side receives fixed and pays floating. Any other value is treated
        as payer.
    shift : float, default 0.0
        Additive zero-rate shift in decimal units.
    fixed_freq : int, default 2
        Fixed-leg payment frequency per year.

    Returns
    -------
    float
        Swap value from the selected side's perspective.

    Notes
    -----
    The function uses a simple single-curve approximation with fixed-leg annuity
    and terminal floating-leg value ``1 - DF(T)``.
    """

    annuity = swap_annuity(zero_rates, date, remaining_tenor, shift=shift, fixed_freq=fixed_freq)
    maturity_df = float(discount_at(zero_rates, date, [remaining_tenor], shift=shift)[0])
    fixed_value = float(fixed_rate) * annuity * float(notional)
    floating_value = (1.0 - maturity_df) * float(notional)
    receiver_value = fixed_value - floating_value
    return receiver_value if str(side).lower() == "receiver" else -receiver_value


def swap_pv01(zero_rates: pd.DataFrame, date, tenor, *, side: str = "receiver", bump: float = 1e-4, fixed_freq: int = 2):
    """Compute PV01 of a par swap by symmetric curve shifts.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    date : date-like
        Valuation date.
    tenor : float
        Swap tenor in years.
    side : {"receiver", "payer"}, default "receiver"
        Swap side for valuation.
    bump : float, default 1e-4
        Additive zero-rate bump in decimal units.
    fixed_freq : int, default 2
        Fixed-leg payment frequency per year.

    Returns
    -------
    float
        Symmetric finite-difference PV01 of the par swap.

    Notes
    -----
    The fixed rate is first set to the par swap rate at the base curve.
    """

    fixed_rate = par_swap_rate(zero_rates, date, tenor, fixed_freq=fixed_freq)
    pv_up = swap_value(zero_rates, date, tenor, fixed_rate, side=side, shift=bump, fixed_freq=fixed_freq)
    pv_down = swap_value(zero_rates, date, tenor, fixed_rate, side=side, shift=-bump, fixed_freq=fixed_freq)
    return float((pv_down - pv_up) / 2)


def swap_overlay_signal_from_duration_target(target_duration, *, neutral_duration: float = 5.0, neutral_band: float = 0.5):
    """Convert a target duration into a swap overlay signal.

    Parameters
    ----------
    target_duration : float
        Desired duration.
    neutral_duration : float, default 5.0
        Neutral duration level.
    neutral_band : float, default 0.5
        No-signal band around the neutral level.

    Returns
    -------
    int
        ``1`` for duration extension, ``-1`` for duration reduction, and ``0`` for
        no overlay.

    Notes
    -----
    The signal is based only on the target duration relative to the neutral band.
    """

    target = float(target_duration)
    if target > float(neutral_duration) + float(neutral_band):
        return 1
    if target < float(neutral_duration) - float(neutral_band):
        return -1
    return 0


def _overlay_target(
    zero_rates,
    date,
    signal,
    nav,
    base_duration,
    *,
    tenor,
    duration_budget,
    dv01_fraction_cap,
):
    if int(signal) == 0:
        return {"side": "flat", "notional": 0.0, "fixed rate": np.nan, "pv01": 0.0, "base duration": base_duration}
    base_dv01 = float(base_duration) * float(nav) * 1e-4
    target_pv01 = int(signal) * min(float(duration_budget) * float(nav) * 1e-4, float(dv01_fraction_cap) * abs(base_dv01))
    side = "receiver" if target_pv01 > 0 else "payer"
    pv01_unit = swap_pv01(zero_rates, date, tenor, side=side)
    notional = abs(target_pv01 / max(abs(pv01_unit), 1e-12))
    return {
        "side": side,
        "notional": notional,
        "fixed rate": par_swap_rate(zero_rates, date, tenor),
        "pv01": pv01_unit * notional,
        "base duration": base_duration,
    }


def run_synthetic_swap_overlay(
    base_result,
    zero_rates: pd.DataFrame,
    target_log: pd.DataFrame,
    *,
    tenor: float = 10.0,
    neutral_duration: float = 5.0,
    neutral_band: float = 0.5,
    duration_budget: float = 1.5,
    dv01_fraction_cap: float = 0.40,
    slippage_bp: float = 0.5,
    start_date=None,
    label: str = "curve-implied synthetic swap overlay",
):
    """Simulate a simple swap overlay on top of a base strategy.

    Parameters
    ----------
    base_result : object
        Base strategy result with returns and effective-duration diagnostics.
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    target_log : pandas.DataFrame
        Target-duration decision log indexed by decision date.
    tenor : float, default 10.0
        Overlay swap tenor in years.
    neutral_duration : float, default 5.0
        Neutral duration used to derive overlay signal.
    neutral_band : float, default 0.5
        No-trade band around neutral duration.
    duration_budget : float, default 1.5
        Maximum duration adjustment budget.
    dv01_fraction_cap : float, default 0.40
        Cap on overlay DV01 as a fraction of base duration exposure.
    slippage_bp : float, default 0.5
        Transaction slippage in basis points of annuity notional change.
    start_date : date-like or None, optional
        Optional first date to include.
    label : str, default "curve-implied synthetic swap overlay"
        Name assigned to the overlay return series.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Overlay trade/performance log and overlay component return series.

    Notes
    -----
    The overlay is re-evaluated on decision dates, held to the next curve date, and
    combined with base strategy returns through a simple NAV recursion. The routine
    is a stylized overlay simulation, not a full swap valuation or collateral model.
    """

    curve_dates = pd.DatetimeIndex(zero_rates.index)
    base_returns = base_result.returns
    base_duration = base_result.diagnostics["risk"]["effective_duration"]
    rows = []
    nav = 1.0
    previous_signed_notional = 0.0
    first_date = pd.Timestamp(start_date) if start_date is not None else curve_dates.min()

    for date, row in target_log.iterrows():
        decision_date = pd.Timestamp(date)
        if decision_date not in curve_dates:
            continue
        loc = curve_dates.get_loc(decision_date)
        if loc >= len(curve_dates) - 1:
            continue
        next_date = curve_dates[loc + 1]
        if next_date < first_date or next_date not in base_returns.index:
            continue

        nav_start = nav
        signal = swap_overlay_signal_from_duration_target(
            row["target duration"],
            neutral_duration=neutral_duration,
            neutral_band=neutral_band,
        )
        target = _overlay_target(
            zero_rates,
            decision_date,
            signal,
            nav_start,
            float(base_duration.loc[:decision_date].iloc[-1]),
            tenor=tenor,
            duration_budget=duration_budget,
            dv01_fraction_cap=dv01_fraction_cap,
        )
        side = target["side"]
        notional = float(target["notional"])
        signed_notional = notional if side == "receiver" else -notional if side == "payer" else 0.0
        annuity = swap_annuity(zero_rates, decision_date, tenor)
        cost = abs(signed_notional - previous_signed_notional) * annuity * float(slippage_bp) / 10000

        if side == "flat":
            overlay_pnl = 0.0
        else:
            holding_years = max((next_date - decision_date).days / 365.25, 1 / 365.25)
            remaining_tenor = max(float(tenor) - holding_years, 0.25)
            overlay_pnl = swap_value(
                zero_rates,
                next_date,
                remaining_tenor,
                target["fixed rate"],
                notional=notional,
                side=side,
            )
        overlay_return = overlay_pnl / max(nav_start, 1e-12)
        cost_return = cost / max(nav_start, 1e-12)
        component_return = overlay_return - cost_return
        nav = nav_start * (1 + float(base_returns.loc[next_date]) + component_return)
        rows.append(
            {
                "date": next_date,
                "decision date": decision_date,
                "label": label,
                "side": side,
                "signal": signal,
                "notional": notional,
                "signed notional": signed_notional,
                "fixed rate": target["fixed rate"],
                "overlay pv01": target["pv01"],
                "base duration": target["base duration"],
                "target duration": row["target duration"],
                "nav start": nav_start,
                "overlay return": overlay_return,
                "cost return": cost_return,
                "overlay component return": component_return,
                "nav with base": nav,
            }
        )
        previous_signed_notional = signed_notional

    log = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    overlay_returns = (
        log["overlay component return"].rename(label)
        if not log.empty
        else pd.Series(dtype=float, name=label)
    )
    return log, overlay_returns


__all__ = [
    "discount_at",
    "par_swap_rate",
    "run_synthetic_swap_overlay",
    "swap_annuity",
    "swap_overlay_signal_from_duration_target",
    "swap_pv01",
    "swap_schedule",
    "swap_value",
    "zero_rate_at",
]
