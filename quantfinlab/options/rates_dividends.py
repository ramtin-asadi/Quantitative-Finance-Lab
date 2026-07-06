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
    """Attach continuous zero rates to option quotes from exactly one rate source.

    Rates can be supplied as a zero-curve panel, a dated rate series/table, or a
    constant rate. Curve-panel inputs are mapped by quote date and maturity using the
    latest available curve on or before the quote date, avoiding future information.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    curve_panel : pandas.DataFrame, optional
        Date-indexed zero-curve panel with tenor columns in years or tenor labels.
    rates : pandas.Series or pandas.DataFrame, optional
        Dated rate series or table. Values are matched by previous available date.
    constant_rate : float, optional
        Constant annualized rate applied to all rows.
    date_col : str, default='date'
        Quote date column.
    tau_col : str, default='tau'
        Time-to-expiry column in years.
    rate_col : str, default='rate'
        Output rate column.
    method : {'previous'}, default='previous'
        Date-matching method for curve-panel rates.
    interpolation : {'linear'}, default='linear'
        Maturity interpolation method for curve-panel rates.
    input_compounding : {'continuous', 'simple'}, default='continuous'
        Compounding convention for supplied scalar or dated rates. Simple rates are
        converted to continuous rates over each option horizon.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with a continuous-rate column.

    Raises
    ------
    ValueError
        If zero or multiple rate sources are supplied, or if a rate table has no
        usable numeric column.
    """

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


def attach_rates(quotes: pd.DataFrame, curve_panel: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
    """Short alias for attaching rates to option quotes.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    curve_panel : pandas.DataFrame, optional
        Zero-curve panel passed to the rate attachment routine.
    **kwargs
        Additional rate-attachment options. The compatibility keyword ``out_col`` is
        mapped to ``rate_col`` when present.

    Returns
    -------
    pandas.DataFrame
        Quote table with attached rate column.
    """

    if "out_col" in kwargs and "rate_col" not in kwargs:
        kwargs["rate_col"] = kwargs.pop("out_col")
    return attach_rates_to_options(quotes, curve_panel=curve_panel, **kwargs)


def add_discount_factors(
    quotes: pd.DataFrame,
    rate_col: str = "rate",
    tau_col: str = "tau",
    out_col: str = "discount_factor",
) -> pd.DataFrame:
    """Attach discount factors computed from continuous rates and option maturities.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    rate_col : str, default='rate'
        Continuous annualized rate column.
    tau_col : str, default='tau'
        Time-to-expiry column in years.
    out_col : str, default='discount_factor'
        Output discount-factor column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with discount factors.
    """

    out = quotes.copy()
    out[out_col] = discounting.discount_factor_from_rate(out[rate_col], out[tau_col])
    return out


def infer_dividend_yield_from_forward(
    spot: float | np.ndarray | pd.Series | pd.DataFrame,
    forward: float | np.ndarray | pd.Series | None = None,
    rate: float | np.ndarray | pd.Series | None = None,
    tau: float | np.ndarray | pd.Series | None = None,
    *,
    spot_col: str = "spot",
    forward_col: str = "forward",
    rate_col: str = "rate",
    tau_col: str = "tau",
    carry_col: str = "implied_carry",
    out_col: str | None = None,
) -> float | np.ndarray | pd.Series | pd.DataFrame:
    """Infer continuous dividend yield from spot, forward, rate, and maturity.

    The calculation rearranges ``F = S * exp((r - q) * T)`` to obtain
    ``q = r - log(F / S) / T``. DataFrame input is supported for table-level use.

    Parameters
    ----------
    spot : scalar, array-like, pandas.Series, or pandas.DataFrame
        Spot price input, or a DataFrame containing spot, forward, rate, and maturity
        columns.
    forward : scalar or array-like, optional
        Forward price. Required unless ``spot`` is a DataFrame.
    rate : scalar or array-like, optional
        Continuous annualized rate. Required unless ``spot`` is a DataFrame.
    tau : scalar or array-like, optional
        Time to expiry in years. Required unless ``spot`` is a DataFrame.
    spot_col, forward_col, rate_col, tau_col : str
        Column names used for DataFrame input.
    carry_col : str, default='implied_carry'
        Carry column used as an alternative to forward when DataFrame input already
        contains implied carry.
    out_col : str, optional
        Output column name for DataFrame input. Defaults to
        ``'implied_dividend_yield'``.

    Returns
    -------
    float, numpy.ndarray, pandas.Series, or pandas.DataFrame
        Implied continuous dividend yield, preserving the input style where possible.

    Raises
    ------
    ValueError
        If required scalar/array inputs are missing.
    """

    if isinstance(spot, pd.DataFrame):
        out = spot.copy()
        if carry_col in out.columns:
            q = pd.to_numeric(out[rate_col], errors="coerce") - pd.to_numeric(out[carry_col], errors="coerce")
        else:
            q = infer_dividend_yield_from_forward(out[spot_col], out[forward_col], out[rate_col], out[tau_col])
        out[out_col or "implied_dividend_yield"] = q
        return out
    if forward is None or rate is None or tau is None:
        raise ValueError("forward, rate, and tau are required unless spot is a DataFrame.")
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
    spot: float | np.ndarray | pd.Series | pd.DataFrame,
    forward: float | np.ndarray | pd.Series | None = None,
    tau: float | np.ndarray | pd.Series | None = None,
    *,
    spot_col: str = "spot",
    forward_col: str = "forward",
    tau_col: str = "tau",
    out_col: str | None = None,
) -> float | np.ndarray | pd.Series | pd.DataFrame:
    """Infer continuous carry from spot, forward, and maturity.

    The carry is defined as ``log(F / S) / T`` and is therefore equal to
    risk-free rate minus dividend yield under the continuous-carry model.

    Parameters
    ----------
    spot : scalar, array-like, pandas.Series, or pandas.DataFrame
        Spot price input, or a DataFrame containing spot, forward, and maturity
        columns.
    forward : scalar or array-like, optional
        Forward price. Required unless ``spot`` is a DataFrame.
    tau : scalar or array-like, optional
        Time to expiry in years. Required unless ``spot`` is a DataFrame.
    spot_col, forward_col, tau_col : str
        Column names used for DataFrame input.
    out_col : str, optional
        Output column name for DataFrame input. Defaults to ``'implied_carry'``.

    Returns
    -------
    float, numpy.ndarray, pandas.Series, or pandas.DataFrame
        Implied continuous carry.

    Raises
    ------
    ValueError
        If required scalar/array inputs are missing.
    """

    if isinstance(spot, pd.DataFrame):
        out = spot.copy()
        out[out_col or "implied_carry"] = infer_carry_from_forward(out[spot_col], out[forward_col], out[tau_col])
        return out
    if forward is None or tau is None:
        raise ValueError("forward and tau are required unless spot is a DataFrame.")
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
    "attach_rates",
    "attach_rates_to_options",
    "infer_carry_from_forward",
    "infer_dividend_yield_from_forward",
]
