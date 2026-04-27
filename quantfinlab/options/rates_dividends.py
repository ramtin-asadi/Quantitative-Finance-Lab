from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.fixed_income import discounting


def attach_rates_to_options(
    quotes: pd.DataFrame,
    curve_panel: pd.DataFrame | None = None,
    rates: pd.Series | pd.DataFrame | None = None,
    constant_rate: float | None = None,
    date_col: str = "date",
    tau_col: str = "tau",
    rate_col: str = "rate",
    method: str = "previous",
    interpolation: str = "linear",
    input_compounding: str = "continuous",
) -> pd.DataFrame:
    """Attach continuous zero rates while delegating curve math to fixed_income."""
    out = quotes.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").astype("datetime64[ns]")

    sources = [curve_panel is not None, rates is not None, constant_rate is not None]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of curve_panel, rates, or constant_rate.")

    if curve_panel is not None:
        out[rate_col] = discounting.map_curve_rates_to_dates_and_taus(
            curve_panel,
            dates=out[date_col],
            taus=out[tau_col],
            method=method,
            interpolation=interpolation,
        )
        return out

    if rates is not None:
        if isinstance(rates, pd.DataFrame):
            rate_frame = rates.copy()
            if date_col in rate_frame.columns:
                rate_frame[date_col] = pd.to_datetime(rate_frame[date_col], errors="coerce").astype("datetime64[ns]")
                rate_frame = rate_frame.set_index(date_col)
            if rate_col in rate_frame.columns:
                rate_series = rate_frame[rate_col]
            else:
                numeric_cols = rate_frame.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("rates DataFrame must contain a numeric rate column.")
                rate_series = rate_frame[numeric_cols[0]]
        else:
            rate_series = rates.copy()
        rate_series.index = pd.DatetimeIndex(pd.to_datetime(rate_series.index, errors="coerce")).astype("datetime64[ns]")
        rate_series = pd.to_numeric(rate_series.sort_index(), errors="coerce").dropna()
        left = out[[date_col]].reset_index().sort_values(date_col).rename(columns={"index": "_row"})
        right = rate_series.rename(rate_col).reset_index().rename(columns={"index": date_col})
        matched = pd.merge_asof(left, right.sort_values(date_col), on=date_col, direction="backward")
        out.loc[matched["_row"], rate_col] = matched[rate_col].to_numpy(dtype=float)
        if input_compounding == "simple":
            out[rate_col] = discounting.continuous_rate_from_simple_rate(out[rate_col], out[tau_col])
        return out

    out[rate_col] = discounting.constant_rate_series(
        out[tau_col],
        rate=float(constant_rate),
        input_compounding=input_compounding,
    )
    return out


def add_discount_factors(
    quotes: pd.DataFrame,
    rate_col: str = "rate",
    tau_col: str = "tau",
    out_col: str = "discount_factor",
) -> pd.DataFrame:
    """Attach discount factors using Project 1 fixed_income helpers."""
    out = quotes.copy()
    out[out_col] = discounting.discount_factor_from_rate(out[rate_col], out[tau_col])
    return out


def infer_dividend_yield_from_forward(
    spot: float | np.ndarray | pd.Series,
    forward: float | np.ndarray | pd.Series,
    rate: float | np.ndarray | pd.Series,
    tau: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Infer continuous dividend yield q from F = S exp((r - q)T)."""
    spot_arr, fwd_arr, rate_arr, tau_arr = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(forward, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(tau, dtype=float),
    )
    out = np.full_like(spot_arr, np.nan, dtype=float)
    mask = (tau_arr > 0) & (spot_arr > 0) & (fwd_arr > 0)
    out[mask] = rate_arr[mask] - np.log(fwd_arr[mask] / spot_arr[mask]) / tau_arr[mask]
    if isinstance(spot, pd.Series):
        return pd.Series(out, index=spot.index)
    if np.isscalar(spot) and np.isscalar(forward) and np.isscalar(rate) and np.isscalar(tau):
        return float(out.reshape(-1)[0])
    return out


def infer_carry_from_forward(
    spot: float | np.ndarray | pd.Series,
    forward: float | np.ndarray | pd.Series,
    tau: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Infer continuous carry log(F/S)/T from an observed forward."""
    spot_arr, fwd_arr, tau_arr = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(forward, dtype=float),
        np.asarray(tau, dtype=float),
    )
    out = np.full_like(spot_arr, np.nan, dtype=float)
    mask = (tau_arr > 0) & (spot_arr > 0) & (fwd_arr > 0)
    out[mask] = np.log(fwd_arr[mask] / spot_arr[mask]) / tau_arr[mask]
    if isinstance(spot, pd.Series):
        return pd.Series(out, index=spot.index)
    if np.isscalar(spot) and np.isscalar(forward) and np.isscalar(tau):
        return float(out.reshape(-1)[0])
    return out


__all__ = [
    "add_discount_factors",
    "attach_rates_to_options",
    "infer_carry_from_forward",
    "infer_dividend_yield_from_forward",
]
