from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..core import Curve, InputError
from .tenors import DEFAULT_METHODS, TENOR_PATTERN, tenor_to_years


def par_from_df(df_func: Callable[[np.ndarray], np.ndarray], T: np.ndarray, *, freq: int = 2) -> np.ndarray:
    """
    Compute par yield for maturity T from a discount factor function.
    Uses standard coupon bond equation with coupon dates 1/f,2/f,...,T.
    """
    T = np.array(T, dtype=float)
    out = np.full_like(T, np.nan, dtype=float)
    for i, Ti in enumerate(T):
        if Ti <= 0:
            continue
        if Ti < 1.0:
            # short end: return continuous zero implied by DF(T)
            d = float(df_func(np.array([Ti], dtype=float))[0])
            out[i] = -math.log(max(d, 1e-16)) / Ti
            continue

        n = round(Ti * freq)
        times = np.array([k / freq for k in range(1, n + 1)], dtype=float)
        dfs = df_func(times)
        denom = float(np.sum(dfs))
        if denom <= 0:
            continue
        dT = float(dfs[-1])
        out[i] = freq * (1.0 - dT) / denom
    return out


def curve_value_table(
    curves: dict[str, Curve],
    *,
    value: Literal["par", "zero", "df", "forward"] = "zero",
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
    freq: int = 2,
) -> pd.DataFrame:
    """
    Build a common-grid table of curve values by method.

    value:
    - "par": par yields (decimal)
    - "zero": zero rates (decimal)
    - "df": discount factors
    - "forward": instantaneous forward rates (decimal)
    """
    if not curves:
        raise InputError("curves is empty.")

    if grid is None:
        grid_arr = np.linspace(max(1 / 12, t_min), t_max, int(points))
    else:
        grid_arr = np.asarray(grid, dtype=float).reshape(-1)
        if grid_arr.size == 0:
            raise InputError("grid is empty.")

    out: dict[str, np.ndarray] = {}
    for method, curve in curves.items():
        if value == "par":
            vals = par_from_df(curve.df, grid_arr, freq=freq)
        elif value == "zero":
            vals = np.interp(grid_arr, curve.grid, curve.z_grid)
        elif value == "df":
            vals = np.interp(grid_arr, curve.grid, curve.df_grid)
        elif value == "forward":
            vals = np.interp(grid_arr, curve.grid, curve.fwd_grid)
        else:
            raise InputError(f"Unsupported curve value type: {value!r}.")
        out[method] = np.asarray(vals, dtype=float)

    return pd.DataFrame(out, index=grid_arr)


def zero_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
) -> pd.DataFrame:
    return curve_value_table(curves, value="zero", grid=grid, t_min=t_min, t_max=t_max, points=points)


def par_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
    freq: int = 2,
) -> pd.DataFrame:
    return curve_value_table(
        curves,
        value="par",
        grid=grid,
        t_min=t_min,
        t_max=t_max,
        points=points,
        freq=freq,
    )


def discount_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
) -> pd.DataFrame:
    return curve_value_table(curves, value="df", grid=grid, t_min=t_min, t_max=t_max, points=points)


def shifted_df_func(df_func: Callable[[np.ndarray], np.ndarray], shift_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Apply a continuous-rate shift: DF_shifted(t) = DF(t) * exp(-shift(t)*t)
    where shift(t) is in absolute rate units (e.g. 0.0001 for 1bp).
    """
    def _f(t: np.ndarray) -> np.ndarray:
        tt = np.array(t, dtype=float)
        return df_func(tt) * np.exp(-shift_func(tt) * tt)
    return _f


def curve_date_for(index: pd.Index, d: pd.Timestamp) -> pd.Timestamp | None:
    """Return the most recent curve date <= d (like your notebook)."""
    d = pd.Timestamp(d)
    if d in index:
        return d
    pos = index.searchsorted(d, side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(index[pos])


def short_rate_from_first_tenor(
    curve_row: pd.Series | dict,
    *,
    tenor_cols: list[str] | None = None,
    default_rate: float = 0.02,
) -> float:
    """
    Extract a short rate from the first available tenor in a par-yield row.
    Input yields are expected in decimal form.
    """
    if isinstance(curve_row, dict):
        curve_row = pd.Series(curve_row)

    cols = tenor_cols
    if cols is None:
        cols = [
            c
            for c in curve_row.index.astype(str)
            if TENOR_PATTERN.fullmatch(str(c).strip().replace(" ", "").upper())
        ]
    if not cols:
        return float(default_rate)

    cols = sorted(cols, key=tenor_to_years)
    vals = pd.to_numeric(curve_row[cols], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if finite.sum() == 0:
        return float(default_rate)
    first_idx = int(np.where(finite)[0][0])
    return float(vals[first_idx])


def zero_rate_from_df(df: np.ndarray | float, tau: np.ndarray | float, *, default_rate: float = 0.02) -> np.ndarray | float:
    """
    Convert discount factors to continuously-compounded zero rates.
    Uses default_rate where tau <= 0.
    """
    df_arr = np.asarray(df, dtype=float)
    tau_arr = np.asarray(tau, dtype=float)
    safe_df = np.clip(df_arr, 1e-16, None)
    out = np.full_like(safe_df, float(default_rate), dtype=float)
    pos = tau_arr > 0
    out[pos] = -np.log(safe_df[pos]) / tau_arr[pos]
    if np.isscalar(df) and np.isscalar(tau):
        return float(np.asarray(out).reshape(-1)[0])
    return out


def _first_pandas_index(*values: Any) -> pd.Index | None:
    for value in values:
        if isinstance(value, pd.Series):
            return value.index
    return None


def _wrap_like(value: np.ndarray, *templates: Any) -> float | np.ndarray | pd.Series:
    index = _first_pandas_index(*templates)
    if index is not None:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(index), float(arr), dtype=float)
        return pd.Series(arr, index=index)
    if all(np.isscalar(t) for t in templates if t is not None):
        return float(np.asarray(value, dtype=float).reshape(-1)[0])
    return np.asarray(value, dtype=float)


def discount_factor_from_rate(
    rate: float | np.ndarray | pd.Series,
    tau: float | np.ndarray | pd.Series,
    compounding: Literal["continuous", "simple", "annual"] = "continuous",
) -> float | np.ndarray | pd.Series:
    """
    Convert annualized rates into discount factors.

    Project 4 routes option discounting through this Project 1 fixed-income
    helper so options code does not duplicate rate-conversion logic.
    """
    rate_arr, tau_arr = np.broadcast_arrays(np.asarray(rate, dtype=float), np.asarray(tau, dtype=float))
    comp = str(compounding).lower().strip()

    if comp == "continuous":
        out = np.exp(-rate_arr * tau_arr)
    elif comp == "simple":
        out = 1.0 / (1.0 + rate_arr * tau_arr)
    elif comp == "annual":
        out = 1.0 / np.power(1.0 + rate_arr, tau_arr)
    else:
        raise InputError("compounding must be one of {'continuous', 'simple', 'annual'}.")

    out = np.where(tau_arr <= 0, 1.0, out)
    out = np.where(np.isfinite(out) & (out > 0), out, np.nan)
    return _wrap_like(out, tau, rate)


def continuous_rate_from_discount_factor(
    df: float | np.ndarray | pd.Series,
    tau: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Convert discount factors to continuously compounded zero rates."""
    df_arr, tau_arr = np.broadcast_arrays(np.asarray(df, dtype=float), np.asarray(tau, dtype=float))
    out = np.full_like(df_arr, np.nan, dtype=float)
    mask = (tau_arr > 0) & (df_arr > 0) & np.isfinite(df_arr) & np.isfinite(tau_arr)
    out[mask] = -np.log(df_arr[mask]) / tau_arr[mask]
    return _wrap_like(out, tau, df)


def continuous_rate_from_simple_rate(
    simple_rate: float | np.ndarray | pd.Series,
    tau: float | np.ndarray | pd.Series,
) -> float | np.ndarray | pd.Series:
    """Convert simple annualized rates over horizon tau into continuous rates."""
    rate_arr, tau_arr = np.broadcast_arrays(
        np.asarray(simple_rate, dtype=float),
        np.asarray(tau, dtype=float),
    )
    out = np.full_like(rate_arr, np.nan, dtype=float)
    gross = 1.0 + rate_arr * tau_arr
    mask = (tau_arr > 0) & (gross > 0) & np.isfinite(gross)
    out[mask] = np.log(gross[mask]) / tau_arr[mask]
    return _wrap_like(out, tau, simple_rate)


def constant_rate_series(
    tau: float | np.ndarray | pd.Series,
    rate: float = 0.06,
    input_compounding: Literal["continuous", "simple"] = "continuous",
) -> float | np.ndarray | pd.Series:
    """
    Return a horizon-aligned continuously compounded rate series.

    The NIFTY Project 4 workflow uses this for its temporary 6 percent India
    proxy rate; replacing that proxy with a curve later only changes the rate
    source, not the option-pricing code.
    """
    tau_arr = np.asarray(tau, dtype=float)
    comp = str(input_compounding).lower().strip()
    if comp == "continuous":
        out = np.full_like(tau_arr, float(rate), dtype=float)
    elif comp == "simple":
        converted = continuous_rate_from_simple_rate(float(rate), tau)
        if isinstance(converted, pd.Series):
            return converted
        out = np.asarray(converted, dtype=float)
    else:
        raise InputError("input_compounding must be one of {'continuous', 'simple'}.")
    return _wrap_like(out, tau)


def _maturity_years_from_labels(labels: Iterable[Any]) -> np.ndarray:
    vals: list[float] = []
    for label in labels:
        if isinstance(label, (int, float, np.integer, np.floating)):
            vals.append(float(label))
            continue
        text = str(label).strip()
        try:
            vals.append(float(text))
            continue
        except ValueError:
            pass
        compact = text.replace(" ", "").upper()
        if TENOR_PATTERN.fullmatch(compact):
            vals.append(float(tenor_to_years(compact)))
        else:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def rate_from_zero_curve(
    zero_curve: pd.Series | pd.DataFrame | dict[str, float],
    tau: float | np.ndarray | pd.Series,
    method: Literal["linear"] = "linear",
) -> float | np.ndarray | pd.Series:
    """Interpolate a continuous zero curve at option maturity tau."""
    if method != "linear":
        raise InputError("Only method='linear' is currently supported.")

    if isinstance(zero_curve, pd.DataFrame):
        if zero_curve.empty:
            raise InputError("zero_curve DataFrame is empty.")
        row = zero_curve.iloc[0]
    elif isinstance(zero_curve, dict):
        row = pd.Series(zero_curve)
    else:
        row = zero_curve

    maturities = _maturity_years_from_labels(row.index)
    rates = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(maturities) & np.isfinite(rates)
    if mask.sum() == 0:
        out = np.full_like(np.asarray(tau, dtype=float), np.nan, dtype=float)
        return _wrap_like(out, tau)

    maturities = maturities[mask]
    rates = rates[mask]
    order = np.argsort(maturities)
    maturities = maturities[order]
    rates = rates[order]

    tau_arr = np.asarray(tau, dtype=float)
    out = np.interp(tau_arr, maturities, rates, left=rates[0], right=rates[-1])
    out = np.where(tau_arr <= 0, rates[0], out)
    return _wrap_like(out, tau)


def map_curve_rates_to_dates_and_taus(
    curve_panel: pd.DataFrame,
    dates: pd.Series | pd.Index | np.ndarray | list,
    taus: pd.Series | np.ndarray | list | float,
    method: Literal["previous"] = "previous",
    interpolation: Literal["linear"] = "linear",
    return_source_dates: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    Map quote dates and maturities to zero rates from a panel of zero curves.

    For each quote date, the latest curve date on or before the quote date is
    used. This prevents options workflows from leaking future rate information.
    """
    if method != "previous":
        raise InputError("Only method='previous' is currently supported.")
    if curve_panel.empty:
        raise InputError("curve_panel is empty.")

    panel = curve_panel.copy()
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    date_index = dates.index if isinstance(dates, pd.Series) else None
    quote_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    if date_index is None and isinstance(taus, pd.Series):
        date_index = taus.index
    if date_index is None:
        date_index = pd.RangeIndex(len(quote_dates))
    quote_dates.index = date_index

    tau_series = pd.Series(taus, index=date_index, dtype=float)
    out = pd.Series(np.nan, index=date_index, dtype=float, name="rate")
    source = pd.Series(pd.NaT, index=date_index, dtype="datetime64[ns]", name="curve_date")

    for qdate, idx in quote_dates.groupby(quote_dates).groups.items():
        if pd.isna(qdate):
            continue
        curve_date = curve_date_for(panel.index, pd.Timestamp(qdate))
        if curve_date is None:
            continue
        tau_vals = tau_series.loc[idx]
        out.loc[idx] = rate_from_zero_curve(panel.loc[curve_date], tau_vals, method=interpolation)
        source.loc[idx] = curve_date

    if return_source_dates:
        return out, source
    return out


def make_discount_lookup(
    par_yields: pd.DataFrame,
    *,
    tenor_cols: list[str] | None = None,
    curve_method: str = "loglinear",
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    short_end_policy: Literal["first_tenor_exp", "curve_only"] = "first_tenor_exp",
    min_df: float = 1e-12,
    default_rate: float = 0.02,
) -> dict[str, Any]:
    """
    Build a cached lookup for option-horizon discount factors by trade date.

    Returns a dict with:
    - get_df(date, tau): discount factor(s)
    - get_rate(date, tau): zero rate(s)
    - resolve_date(date): latest available curve date <= date (or first date)
    - curve_mode: dict[curve_date -> mode string]
    - r0_by_date: dict[curve_date -> short rate]
    - tau_min: first tenor in years

    Notes:
    - Designed as an optional convenience layer; existing APIs are unchanged.
    - For tau < first tenor, short_end_policy='first_tenor_exp' uses exp(-r0*tau).
    """
    from .bootstrap import bootstrap_pillars
    from .smoothers import fit_curves

    if par_yields.empty:
        raise InputError("par_yields is empty.")

    py = par_yields.copy().sort_index()
    py.index = pd.to_datetime(py.index)

    cols = tenor_cols if tenor_cols is not None else [str(c) for c in py.columns]
    cols = sorted(cols, key=tenor_to_years)
    t_years = np.array([tenor_to_years(c) for c in cols], dtype=float)
    tau_min = float(np.nanmin(t_years))

    df_cache: dict[pd.Timestamp, Callable[[np.ndarray], np.ndarray]] = {}
    curve_mode: dict[pd.Timestamp, str] = {}
    r0_by_date: dict[pd.Timestamp, float] = {}

    def resolve_date(d: pd.Timestamp) -> pd.Timestamp:
        dd = pd.Timestamp(d).normalize()
        cd = curve_date_for(py.index, dd)
        if cd is not None:
            return cd
        return pd.Timestamp(py.index[0])

    def _fallback_df_func(row: pd.Series) -> Callable[[np.ndarray], np.ndarray]:
        vals = pd.to_numeric(row[cols], errors="coerce").to_numpy(dtype=float)
        good = np.isfinite(vals)
        if good.sum() == 0:
            return lambda x: np.exp(-float(default_rate) * np.asarray(x, dtype=float))
        tt = t_years[good]
        yy = vals[good]
        ord_idx = np.argsort(tt)
        tt = tt[ord_idx]
        yy = yy[ord_idx]
        log_df = np.log(np.clip(np.exp(-yy * tt), min_df, None))

        def _f(x: np.ndarray) -> np.ndarray:
            xx = np.asarray(x, dtype=float)
            xx = np.clip(xx, 1e-12, None)
            return np.exp(np.interp(xx, tt, log_df, left=log_df[0], right=log_df[-1]))

        return _f

    def _build_df_func(cdate: pd.Timestamp) -> Callable[[np.ndarray], np.ndarray]:
        if cdate in df_cache:
            return df_cache[cdate]

        row_obj = py.loc[cdate]
        row = row_obj.iloc[-1] if isinstance(row_obj, pd.DataFrame) else row_obj
        r0 = short_rate_from_first_tenor(row, tenor_cols=cols, default_rate=default_rate)
        r0_by_date[cdate] = float(r0)

        try:
            pillars = bootstrap_pillars(
                row,
                asof=cdate,
                tenor_cols=cols,
                freq=freq,
                short_end=short_end,
                min_df=min_df,
            )
            curves = fit_curves(pillars, methods=(curve_method,), freq=freq, min_df=min_df)
            base_df = curves[curve_method].df
            curve_mode[cdate] = f"qfi_{curve_method}"
        except Exception:
            base_df = _fallback_df_func(row)
            curve_mode[cdate] = "fallback_loglinear"

        def _df_func(tau: np.ndarray) -> np.ndarray:
            tt = np.asarray(tau, dtype=float)
            out = np.ones_like(tt, dtype=float)
            pos = tt > 0
            if pos.sum() == 0:
                return out
            tpos = tt[pos]
            if short_end_policy == "first_tenor_exp":
                short = tpos < tau_min
                o = np.empty_like(tpos, dtype=float)
                o[short] = np.exp(-float(r0) * tpos[short])
                o[~short] = np.asarray(base_df(tpos[~short]), dtype=float)
                out[pos] = o
            else:
                out[pos] = np.asarray(base_df(tpos), dtype=float)
            out = np.clip(out, min_df, 1.0)
            return out

        df_cache[cdate] = _df_func
        return _df_func

    def get_df(date: pd.Timestamp, tau: np.ndarray | float) -> np.ndarray | float:
        cdate = resolve_date(date)
        df_func = _build_df_func(cdate)
        tau_arr = np.asarray(tau, dtype=float)
        vals = np.asarray(df_func(tau_arr), dtype=float)
        if np.isscalar(tau):
            return float(vals.reshape(-1)[0])
        return vals

    def get_rate(date: pd.Timestamp, tau: np.ndarray | float) -> np.ndarray | float:
        cdate = resolve_date(date)
        _ = _build_df_func(cdate)
        r0 = float(r0_by_date.get(cdate, default_rate))
        vals_df = get_df(date, tau)
        return zero_rate_from_df(vals_df, tau, default_rate=r0)

    return {
        "get_df": get_df,
        "get_rate": get_rate,
        "resolve_date": resolve_date,
        "curve_mode": curve_mode,
        "r0_by_date": r0_by_date,
        "tau_min": tau_min,
    }


def attach_discount_columns(
    data: pd.DataFrame,
    lookup: dict[str, Any],
    *,
    date_col: str,
    tau_col: str,
    df_col: str = "df",
    rate_col: str = "r_short",
) -> pd.DataFrame:
    """
    Attach discount factor and short rate columns to a table using a lookup
    returned by make_discount_lookup.
    """
    out = data.copy()
    out[df_col] = np.nan
    out[rate_col] = np.nan

    if len(out) == 0:
        return out

    get_df = lookup["get_df"]
    get_rate = lookup["get_rate"]
    groups = out.groupby(date_col, sort=False).groups
    for d, idx in groups.items():
        tau_vals = out.loc[idx, tau_col].to_numpy(dtype=float)
        out.loc[idx, df_col] = np.asarray(get_df(d, tau_vals), dtype=float)
        out.loc[idx, rate_col] = np.asarray(get_rate(d, tau_vals), dtype=float)
    return out


def curves_by_valuation_date(
    valuation_dates: pd.Index | list[pd.Timestamp],
    par_yields: pd.DataFrame,
    *,
    methods: Iterable[str] = DEFAULT_METHODS,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
    tenor_cols: list[str] | None = None,
) -> dict[pd.Timestamp, dict[str, Curve]]:
    """
    Build fitted curves for each valuation date using the latest available
    market curve date <= valuation date.
    """
    from .bootstrap import bootstrap_pillars, normalize_methods
    from .smoothers import fit_curves

    methods_l = normalize_methods(methods)
    cols = tenor_cols if tenor_cols is not None else [str(c) for c in par_yields.columns]
    curve_cache: dict[pd.Timestamp, dict[str, Curve] | None] = {}
    out: dict[pd.Timestamp, dict[str, Curve]] = {}

    for d in list(valuation_dates):
        vd = pd.Timestamp(d)
        cd = curve_date_for(par_yields.index, vd)
        if cd is None:
            continue
        if cd not in curve_cache:
            row = par_yields.loc[cd]
            try:
                pillars = bootstrap_pillars(
                    row,
                    asof=cd,
                    tenor_cols=cols,
                    freq=freq,
                    short_end=short_end,
                    min_df=min_df,
                )
                curve_cache[cd] = fit_curves(pillars, methods=methods_l, freq=freq, min_df=min_df)
            except Exception:
                curve_cache[cd] = None
        curves_d = curve_cache[cd]
        if curves_d is not None:
            out[vd] = curves_d
    return out


def resolve_asof(index: pd.Index, asof: pd.Timestamp | str | None = None) -> pd.Timestamp:
    """
    Resolve an analysis date against available curve dates.
    - If asof is None: use latest date in index.
    - Otherwise: use the latest available date <= asof.
    """
    if len(index) == 0:
        raise InputError("Date index is empty.")
    if asof is None:
        return pd.Timestamp(index[-1])
    resolved = curve_date_for(index, pd.Timestamp(asof))
    if resolved is None:
        raise InputError(f"No available curve date on or before {pd.Timestamp(asof).date()}.")
    return resolved

__all__ = [
    "attach_discount_columns",
    "constant_rate_series",
    "continuous_rate_from_discount_factor",
    "continuous_rate_from_simple_rate",
    "curve_date_for",
    "curve_value_table",
    "curves_by_valuation_date",
    "discount_curve_table",
    "discount_factor_from_rate",
    "make_discount_lookup",
    "map_curve_rates_to_dates_and_taus",
    "par_curve_table",
    "par_from_df",
    "rate_from_zero_curve",
    "resolve_asof",
    "shifted_df_func",
    "short_rate_from_first_tenor",
    "zero_curve_table",
    "zero_rate_from_df",
]
