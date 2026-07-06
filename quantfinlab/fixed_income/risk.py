from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..common.contracts import Bond, BookMetrics, Curve
from .bond_pricing import (
    _book_pv,
    bond_cashflows,
    bond_position_value,
    bond_price,
    book_pv_timeseries,
    position_values_by_bucket,
    price_bond_from_issue,
    remaining_cashflow_arrays,
)
from .discounting import curves_by_valuation_date, shifted_df_func
from .tenors import DEFAULT_ISSUE_MATURITIES


def key_bump_func(keys: list[int], key: int, *, bump_bp: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Create a piecewise-linear key-rate bump function.

    Parameters
    ----------
    keys : list of int
        Key tenors in years.
    key : int
        Tenor to bump.
    bump_bp : float, default 1.0
        Bump size in basis points.

    Returns
    -------
    callable
        Function mapping maturities in years to rate shifts in decimal units.

    Notes
    -----
    The selected key receives the full bump and all other key nodes receive zero.
    Shifts between key nodes are linearly interpolated.
    """

    values = np.zeros(len(keys), dtype=float)
    k_idx = keys.index(key)
    values[k_idx] = bump_bp / 10000.0

    def shift(t: np.ndarray) -> np.ndarray:
        tt = np.array(t, dtype=float)
        return np.interp(tt, np.array(keys, float), values, left=0.0, right=0.0)

    return shift


def bucket_bump_func(
    key: int,
    *,
    bucket_bounds: dict[int, tuple[float, float]] | None = None,
    bump_bp: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a bucket-local rate bump function.

    Parameters
    ----------
    key : int
        Bucket key whose maturity interval should be shocked.
    bucket_bounds : dict[int, tuple[float, float]] or None, optional
        Mapping from key to lower and upper maturity bounds in years. Defaults to
        broad 2Y, 5Y, 10Y, and 30Y buckets.
    bump_bp : float, default 1.0
        Bump size in basis points.

    Returns
    -------
    callable
        Function mapping maturities in years to bucket shocks in decimal units.

    Notes
    -----
    Maturities inside the selected bucket receive the full bump; maturities outside
    the bucket receive zero.
    """

    if bucket_bounds is None:
        bucket_bounds = {
            2: (0.0, 3.5),
            5: (3.5, 7.5),
            10: (7.5, 20.0),
            30: (20.0, 30.0),
        }
    lo, hi = bucket_bounds[int(key)]
    bump = bump_bp / 10000.0

    def shift(t: np.ndarray) -> np.ndarray:
        tt = np.array(t, dtype=float)
        out = np.zeros_like(tt, dtype=float)
        out[(tt >= lo) & (tt <= hi + 1e-12)] = bump
        return out

    return shift


def book_parallel_risk_timeseries(
    book,
    curves_for_dates: dict[pd.Timestamp, dict[str, Curve]],
    *,
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """Compute book-level PV01 and convexity through time.

    Parameters
    ----------
    book : object
        Synthetic issuance book.
    curves_for_dates : dict[pandas.Timestamp, dict[str, Curve]]
        Nested mapping from valuation date to curve method to fitted curve.
    bump_bp : float, default 1.0
        Symmetric parallel bump size in basis points.

    Returns
    -------
    pandas.DataFrame
        MultiIndex-column table of PV01 and convexity by date and curve method.

    Notes
    -----
    PV01 is estimated from symmetric up/down parallel shifts. Convexity is computed
    as the second-order price response scaled by base PV and squared bump size.
    """

    bump = bump_bp / 10000.0
    risk_records: list[dict] = []

    for vd in sorted(curves_for_dates):
        curves_d = curves_for_dates[vd]
        for method, curve in curves_d.items():
            df0 = curve.df
            pv0, _ = _book_pv(book, vd, df0, cutoff_date=vd)

            up = shifted_df_func(df0, lambda t, b=bump: np.full_like(np.array(t, float), +b))
            dn = shifted_df_func(df0, lambda t, b=bump: np.full_like(np.array(t, float), -b))
            pv_up, _ = _book_pv(book, vd, up, cutoff_date=vd)
            pv_dn, _ = _book_pv(book, vd, dn, cutoff_date=vd)

            pv01_val = (pv_dn - pv_up) / 2.0
            convexity_val = (pv_up + pv_dn - 2.0 * pv0) / (pv0 * (bump**2)) if pv0 != 0 else np.nan
            risk_records.append({"date": vd, "method": method, "pv01": pv01_val, "convexity": convexity_val})

    risk = (
        pd.DataFrame(risk_records)
        .pivot_table(index="date", columns="method", values=["pv01", "convexity"])
        .sort_index(axis=1)
    )
    risk.columns = pd.MultiIndex.from_tuples(
        [(m, metric) for metric, m in risk.columns], names=["method", "metric"]
    )
    return risk


def book_krd_timeseries(
    book,
    curves_for_dates: dict[pd.Timestamp, dict[str, Curve]],
    *,
    keys: list[int] | tuple[int, ...] | None = None,
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """Compute book-level key-rate duration time series.

    Parameters
    ----------
    book : object
        Synthetic issuance book.
    curves_for_dates : dict[pandas.Timestamp, dict[str, Curve]]
        Nested mapping from valuation date to curve method to fitted curve.
    keys : list of int, tuple of int, or None, optional
        Key tenors in years. If omitted, book maturity buckets are used.
    bump_bp : float, default 1.0
        Key-rate bump size in basis points.

    Returns
    -------
    pandas.DataFrame
        Pivot table indexed by valuation date with method/key columns and KRD
        values.

    Notes
    -----
    Each key-rate duration is estimated by bumping one key-tenor node through a
    piecewise-linear shift function and revaluing the book.
    """

    bump = bump_bp / 10000.0
    keys_l = [int(k) for k in (keys if keys is not None else book.maturities)]
    krd_records: list[dict] = []

    for vd in sorted(curves_for_dates):
        curves_d = curves_for_dates[vd]
        for method, curve in curves_d.items():
            df0 = curve.df
            pv0, _ = _book_pv(book, vd, df0, cutoff_date=vd)
            for key in keys_l:
                shift = key_bump_func(keys_l, key, bump_bp=bump_bp)
                df_b = shifted_df_func(df0, shift)
                pv_b, _ = _book_pv(book, vd, df_b, cutoff_date=vd)
                krd = (pv0 - pv_b) / bump
                krd_records.append({"date": vd, "method": method, "key": key, "krd": krd})

    return (
        pd.DataFrame(krd_records)
        .pivot_table(index="date", columns=["method", "key"], values="krd")
        .sort_index()
    )


def bond_price_and_risk(
    bond: Bond,
    curves: dict[str, Curve],
    *,
    bump_bp: float = 1.0,
    key_tenors: list[int] | tuple[int, ...] | None = None,
    settle: float = 0.0,
) -> pd.DataFrame:
    """Compute bond price and rate-risk measures under multiple curves.

    Parameters
    ----------
    bond : Bond
        Fixed-coupon bond specification.
    curves : dict[str, Curve]
        Mapping from curve method to fitted curve.
    bump_bp : float, default 1.0
        Bump size in basis points for PV01, convexity, and KRD.
    key_tenors : list of int, tuple of int, or None, optional
        Key tenors used for key-rate duration. Defaults to standard maturity
        buckets.
    settle : float, default 0.0
        Years since the last coupon date.

    Returns
    -------
    pandas.DataFrame
        Table indexed by curve method with clean price, PV01, convexity, and
        key-rate duration columns.

    Notes
    -----
    Clean price is computed first, then simple accrued interest is added back to
    form the dirty-price base for risk calculations.
    """

    if key_tenors is None:
        key_tenors = list(DEFAULT_ISSUE_MATURITIES)
    bump = bump_bp / 10000.0
    rows = []
    for method, curve in curves.items():
        p0 = bond_price(bond, curve, settle=settle, clean=True)
        df0 = curve.df

        up = shifted_df_func(df0, lambda t: np.full_like(np.array(t, float), +bump))
        dn = shifted_df_func(df0, lambda t: np.full_like(np.array(t, float), -bump))

        times, cfs = bond_cashflows(bond.coupon, bond.maturity_years, freq=bond.freq, face=bond.face)
        pv_up = price_bond_from_issue(up, times, cfs, age=settle)
        pv_dn = price_bond_from_issue(dn, times, cfs, age=settle)

        dirty = p0 + bond.coupon * bond.face * settle
        pv01_val = (pv_dn - pv_up) / 2.0
        convexity_val = (pv_up + pv_dn - 2.0 * dirty) / (dirty * (bump**2)) if p0 != 0 else np.nan

        krd_vals = {}
        for k in key_tenors:
            shift = key_bump_func(list(key_tenors), int(k), bump_bp=bump_bp)
            df_b = shifted_df_func(df0, shift)
            pv_b = price_bond_from_issue(df_b, times, cfs, age=settle)
            krd_vals[f"krd_{k}Y"] = (dirty - pv_b) / bump

        rows.append(
            {
                "method": method,
                "clean_price": p0,
                "pv01": pv01_val,
                "convexity": convexity_val,
                **krd_vals,
            }
        )

    return pd.DataFrame(rows).set_index("method").sort_index()


def pv01(
    bond: Bond,
    curve: Curve,
    *,
    bump_bp: float = 1.0,
    settle: float = 0.0,
) -> float:
    """Compute PV01 of a fixed-coupon bond under one curve.

    Parameters
    ----------
    bond : Bond
        Fixed-coupon bond specification.
    curve : Curve
        Fitted discount curve.
    bump_bp : float, default 1.0
        Bump size in basis points.
    settle : float, default 0.0
        Years since the last coupon date.

    Returns
    -------
    float
        Symmetric finite-difference PV01.

    Notes
    -----
    The curve method label is taken from the curve object and used internally to
    build a one-curve risk table.
    """

    table = bond_price_and_risk(bond, {curve.method: curve}, bump_bp=bump_bp, settle=settle)
    return float(table.iloc[0]["pv01"])


def dv01(
    bond: Bond,
    curve: Curve,
    *,
    bump_bp: float = 1.0,
    settle: float = 0.0,
) -> float:
    """Compute DV01 of a fixed-coupon bond under one curve.

    Parameters
    ----------
    bond : Bond
        Fixed-coupon bond specification.
    curve : Curve
        Fitted discount curve.
    bump_bp : float, default 1.0
        Bump size in basis points.
    settle : float, default 0.0
        Years since the last coupon date.

    Returns
    -------
    float
        DV01 value, implemented as an alias of PV01.

    Notes
    -----
    The function is provided for naming convenience; it returns the same value as
    the PV01 helper.
    """

    return pv01(bond, curve, bump_bp=bump_bp, settle=settle)


def price_from_ytm(y: float, times: np.ndarray, cfs: np.ndarray, *, freq: int = 2) -> float:
    """Price cash flows using a flat yield-to-maturity.

    Parameters
    ----------
    y : float
        Annual yield-to-maturity in decimal units.
    times : numpy.ndarray
        Cash-flow times in years.
    cfs : numpy.ndarray
        Cash-flow amounts.
    freq : int, default 2
        Compounding frequency per year.

    Returns
    -------
    float
        Present value of the cash flows under periodic compounding.
    """

    return float(np.sum(cfs / np.power(1.0 + y / freq, freq * times)))


def solve_bond_ytm(price: float, times: np.ndarray, cfs: np.ndarray, *, freq: int = 2) -> float:
    """Solve the yield-to-maturity that matches a bond price.

    Parameters
    ----------
    price : float
        Target bond price.
    times : numpy.ndarray
        Remaining cash-flow times in years.
    cfs : numpy.ndarray
        Remaining cash-flow amounts.
    freq : int, default 2
        Compounding frequency per year.

    Returns
    -------
    float
        Yield-to-maturity in decimal units, or NaN when no valid bracket is found.

    Notes
    -----
    The solver uses bracketing and bisection. The upper bracket is expanded when
    needed up to a finite limit.
    """

    if price <= 0 or len(times) == 0:
        return np.nan

    lo, hi = -0.05, 0.50

    def err(y: float) -> float:
        return price_from_ytm(y, times, cfs, freq=freq) - price

    e_lo = err(lo)
    e_hi = err(hi)
    while e_lo * e_hi > 0 and hi < 5.0:
        hi *= 2.0
        e_hi = err(hi)

    if e_lo * e_hi > 0:
        return np.nan

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        e_mid = err(mid)
        if abs(e_mid) < 1e-12:
            return mid
        if e_lo * e_mid <= 0:
            hi = mid
            e_hi = e_mid
        else:
            lo = mid
            e_lo = e_mid
    return float(0.5 * (lo + hi))


def bond_modified_duration(
    bond: dict | None,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    bump_bp: float = 1.0,
) -> float:
    """Estimate modified duration of a synthetic bond position.

    Parameters
    ----------
    bond : dict or None
        Synthetic bond record.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Discount-factor function used to value the position.
    bump_bp : float, default 1.0
        Yield bump size in basis points.

    Returns
    -------
    float
        Modified duration. Returns zero for missing or matured positions and NaN
        when yield-to-maturity cannot be solved.

    Notes
    -----
    The function first solves a per-unit yield-to-maturity from the current market
    value, then computes symmetric finite-difference duration under yield bumps.
    """

    t_rem, cf_rem = remaining_cashflow_arrays(bond, valuation_date)
    if len(t_rem) == 0:
        return 0.0

    price = bond_position_value(bond, valuation_date, df_func)
    units = max(float(bond["units"]), 1e-12)
    price_per_unit = price / units
    cfs_per_unit = cf_rem / units
    ytm = solve_bond_ytm(price_per_unit, t_rem, cfs_per_unit, freq=int(bond["freq"]))
    if not np.isfinite(ytm):
        return np.nan

    dy = bump_bp / 10000.0
    p_up = price_from_ytm(ytm + dy, t_rem, cfs_per_unit, freq=int(bond["freq"]))
    p_dn = price_from_ytm(ytm - dy, t_rem, cfs_per_unit, freq=int(bond["freq"]))
    return float((p_dn - p_up) / (2.0 * price_per_unit * dy))


def portfolio_parallel_risk(
    positions: dict[int, dict],
    cash: float,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    bump_bp: float = 1.0,
) -> dict[str, float]:
    """Compute parallel rate risk for a synthetic bond portfolio.

    Parameters
    ----------
    positions : dict[int, dict]
        Synthetic bond positions by maturity bucket.
    cash : float
        Cash balance included in NAV but not rate-sensitive.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Base discount-factor function.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Maturity buckets included in the portfolio.
    bump_bp : float, default 1.0
        Parallel bump size in basis points.

    Returns
    -------
    dict[str, float]
        Dictionary with ``pv01``, ``effective_duration``, and ``convexity``.

    Notes
    -----
    Cash is included in base NAV but does not change under rate shifts. If base NAV
    is non-positive, all metrics are returned as NaN.
    """

    dy = bump_bp / 10000.0
    base_bucket = position_values_by_bucket(positions, valuation_date, df_func, buckets=buckets)
    base = float(cash) + sum(base_bucket.values())
    if base <= 0:
        return {"pv01": np.nan, "effective_duration": np.nan, "convexity": np.nan}

    shift_up = shifted_df_func(df_func, lambda t: np.full_like(np.array(t, dtype=float), dy))
    shift_dn = shifted_df_func(df_func, lambda t: np.full_like(np.array(t, dtype=float), -dy))

    pv_up = float(cash) + sum(position_values_by_bucket(positions, valuation_date, shift_up, buckets=buckets).values())
    pv_dn = float(cash) + sum(position_values_by_bucket(positions, valuation_date, shift_dn, buckets=buckets).values())

    return {
        "pv01": float((pv_dn - pv_up) / 2.0),
        "effective_duration": float((pv_dn - pv_up) / (2.0 * base * dy)),
        "convexity": float((pv_up + pv_dn - 2.0 * base) / (base * dy**2)),
    }


def portfolio_modified_duration(
    positions: dict[int, dict],
    cash: float,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    bump_bp: float = 1.0,
) -> float:
    """Compute value-weighted modified duration of a synthetic bond portfolio.

    Parameters
    ----------
    positions : dict[int, dict]
        Synthetic bond positions by maturity bucket.
    cash : float
        Cash balance.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Discount-factor function.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Buckets included in the calculation.
    bump_bp : float, default 1.0
        Yield bump size used in individual-bond duration estimates.

    Returns
    -------
    float
        Value-weighted modified duration. Returns NaN when total NAV is
        non-positive.

    Notes
    -----
    Only buckets with positive market value and finite individual modified duration
    contribute to the result.
    """

    bucket_values = position_values_by_bucket(positions, valuation_date, df_func, buckets=buckets)
    total_nav = float(cash) + sum(bucket_values.values())
    if total_nav <= 0:
        return np.nan

    out = 0.0
    for maturity in buckets:
        value = bucket_values[int(maturity)]
        if value <= 0:
            continue
        dur_i = bond_modified_duration(
            positions.get(int(maturity)),
            valuation_date,
            df_func,
            bump_bp=bump_bp,
        )
        if np.isfinite(dur_i):
            out += (value / total_nav) * dur_i
    return float(out)


def portfolio_key_rate_risk(
    positions: dict[int, dict],
    cash: float,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    bucket_bounds: dict[int, tuple[float, float]] | None = None,
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """Compute bucket key-rate risk for a synthetic bond portfolio.

    Parameters
    ----------
    positions : dict[int, dict]
        Synthetic bond positions by maturity bucket.
    cash : float
        Cash balance.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Base discount-factor function.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Key buckets to shock and value.
    bucket_bounds : dict[int, tuple[float, float]] or None, optional
        Maturity intervals assigned to each key bucket.
    bump_bp : float, default 1.0
        Symmetric bump size in basis points.

    Returns
    -------
    pandas.DataFrame
        Table with ``key``, ``krd``, and ``key_rate_pv01`` columns.

    Notes
    -----
    The function applies bucket-local up and down shifts and scales the resulting
    price sensitivity by base NAV and bump size.
    """

    base_bucket = position_values_by_bucket(positions, valuation_date, df_func, buckets=buckets)
    base = float(cash) + sum(base_bucket.values())
    if base <= 0:
        return pd.DataFrame(columns=["key", "krd", "key_rate_pv01"])

    rows = []
    dy = bump_bp / 10000.0
    for key in buckets:
        shift_up = shifted_df_func(
            df_func,
            bucket_bump_func(int(key), bucket_bounds=bucket_bounds, bump_bp=+bump_bp),
        )
        shift_dn = shifted_df_func(
            df_func,
            bucket_bump_func(int(key), bucket_bounds=bucket_bounds, bump_bp=-bump_bp),
        )

        pv_up = float(cash) + sum(position_values_by_bucket(positions, valuation_date, shift_up, buckets=buckets).values())
        pv_dn = float(cash) + sum(position_values_by_bucket(positions, valuation_date, shift_dn, buckets=buckets).values())

        rows.append(
            {
                "key": int(key),
                "krd": float((pv_dn - pv_up) / (2.0 * base * dy)),
                "key_rate_pv01": float((pv_dn - pv_up) / 2.0),
            }
        )

    return pd.DataFrame(rows)


def strategy_risk_timeseries(
    strategy_df: pd.DataFrame,
    snapshots: dict[pd.Timestamp, dict],
    curve_lookup: Callable[[pd.Timestamp], Curve | Callable[[np.ndarray], np.ndarray] | None],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
    bucket_bounds: dict[int, tuple[float, float]] | None = None,
    bump_bp: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute risk and key-rate diagnostics for a strategy through time.

    Parameters
    ----------
    strategy_df : pandas.DataFrame
        Strategy table indexed by date with at least ``strategy`` and ``nav``
        columns.
    snapshots : dict[pandas.Timestamp, dict]
        Stored position/cash snapshots by date.
    curve_lookup : callable
        Function returning a curve object or discount-factor function for a date.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Buckets included in risk calculations.
    bucket_bounds : dict[int, tuple[float, float]] or None, optional
        Bucket bounds for key-rate shocks.
    bump_bp : float, default 1.0
        Bump size in basis points.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Strategy-level risk table and long-form key-rate risk table.

    Notes
    -----
    Dates without a curve or without a stored snapshot are skipped. The first
    returned table is indexed by date; the second contains one row per date/key.
    """

    risk_rows = []
    krd_rows = []

    for date in strategy_df.index:
        curve = curve_lookup(pd.Timestamp(date))
        if curve is None or date not in snapshots:
            continue

        df_func = curve.df if hasattr(curve, "df") else curve
        positions = snapshots[date]["positions"]
        cash = float(snapshots[date]["cash"])

        parallel = portfolio_parallel_risk(
            positions,
            cash,
            date,
            df_func,
            buckets=buckets,
            bump_bp=bump_bp,
        )
        mod_dur = portfolio_modified_duration(
            positions,
            cash,
            date,
            df_func,
            buckets=buckets,
            bump_bp=bump_bp,
        )

        risk_rows.append(
            {
                "date": date,
                "strategy": strategy_df.loc[date, "strategy"],
                "nav": strategy_df.loc[date, "nav"],
                "pv01": parallel["pv01"],
                "modified_duration": mod_dur,
                "effective_duration": parallel["effective_duration"],
                "convexity": parallel["convexity"],
            }
        )

        krd_tmp = portfolio_key_rate_risk(
            positions,
            cash,
            date,
            df_func,
            buckets=buckets,
            bucket_bounds=bucket_bounds,
            bump_bp=bump_bp,
        )
        if len(krd_tmp) > 0:
            krd_tmp["date"] = date
            krd_tmp["strategy"] = strategy_df.loc[date, "strategy"]
            krd_rows.append(krd_tmp)

    risk_df = pd.DataFrame(risk_rows).set_index("date").sort_index()
    krd_df = pd.concat(krd_rows, axis=0, ignore_index=True) if krd_rows else pd.DataFrame()
    return risk_df, krd_df


def krd_pivot(krd_df: pd.DataFrame, *, value: str = "krd") -> pd.DataFrame:
    """Pivot long-form key-rate risk into a date-by-key table.

    Parameters
    ----------
    krd_df : pandas.DataFrame
        Long-form key-rate risk table containing ``date``, ``key``, and the
        selected value column.
    value : str, default "krd"
        Column to pivot.

    Returns
    -------
    pandas.DataFrame
        Pivoted table indexed by date with key columns. Returns an empty DataFrame
        if the input is empty or the requested value column is unavailable.
    """

    if krd_df is None or krd_df.empty or value not in krd_df.columns:
        return pd.DataFrame()
    return krd_df.pivot(index="date", columns="key", values=value).sort_index()


def latest_krd_table(krd_df: pd.DataFrame, *, date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return key-rate risk rows for the latest or selected date.

    Parameters
    ----------
    krd_df : pandas.DataFrame
        Long-form key-rate risk table.
    date : pandas.Timestamp or None, optional
        Date to select. If omitted, the latest available date is used.

    Returns
    -------
    pandas.DataFrame
        Table with ``key``, ``krd``, and ``key_rate_pv01`` columns sorted by key.
        Empty input returns an empty table with those columns.
    """

    if krd_df is None or krd_df.empty:
        return pd.DataFrame(columns=["key", "krd", "key_rate_pv01"])
    use_date = pd.Timestamp(date) if date is not None else pd.Timestamp(krd_df["date"].max())
    return krd_df[krd_df["date"] == use_date][["key", "krd", "key_rate_pv01"]].sort_values("key")


def duration_sanity_table(risk_df: pd.DataFrame, krd_df: pd.DataFrame) -> pd.DataFrame:
    """Compare aggregate key-rate duration with portfolio duration measures.

    Parameters
    ----------
    risk_df : pandas.DataFrame
        Strategy risk table containing effective and/or modified duration.
    krd_df : pandas.DataFrame
        Long-form key-rate risk table.

    Returns
    -------
    pandas.DataFrame
        Table with KRD sum, effective duration, and modified duration aligned by
        date. Returns an empty DataFrame when ``risk_df`` is empty.

    Notes
    -----
    The table is intended as a diagnostic check, not a formal identity; the match
    depends on the key-rate bucket design.
    """

    if risk_df is None or risk_df.empty:
        return pd.DataFrame()
    krd_sum = krd_pivot(krd_df, value="krd").sum(axis=1).reindex(risk_df.index)
    cols = {
        "krd_sum": krd_sum,
        "effective_duration": risk_df.get("effective_duration"),
        "modified_duration": risk_df.get("modified_duration"),
    }
    return pd.DataFrame(cols)


def pv01_sanity_table(risk_df: pd.DataFrame, krd_df: pd.DataFrame) -> pd.DataFrame:
    """Compare aggregate key-rate PV01 with parallel PV01.

    Parameters
    ----------
    risk_df : pandas.DataFrame
        Strategy risk table containing ``pv01``.
    krd_df : pandas.DataFrame
        Long-form key-rate risk table containing ``key_rate_pv01``.

    Returns
    -------
    pandas.DataFrame
        Table with summed key-rate PV01 and parallel PV01 aligned by date.

    Notes
    -----
    Differences can arise because key-rate shocks and parallel shocks are not the
    same perturbation unless the bucket design fully spans the curve shift.
    """

    if risk_df is None or risk_df.empty:
        return pd.DataFrame()
    pv01_sum = krd_pivot(krd_df, value="key_rate_pv01").sum(axis=1).reindex(risk_df.index)
    return pd.DataFrame({"key_rate_pv01_sum": pv01_sum, "pv01": risk_df.get("pv01")})


def make_book_metrics(
    total_pv: pd.DataFrame,
    bucket_pv: pd.DataFrame,
    risk: pd.DataFrame,
) -> BookMetrics:
    """Create a book-metrics container from valuation and risk tables.

    Parameters
    ----------
    total_pv : pandas.DataFrame
        Total present-value table.
    bucket_pv : pandas.DataFrame
        Bucket present-value table.
    risk : pandas.DataFrame
        Risk metric table.

    Returns
    -------
    BookMetrics
        Container holding the supplied tables.
    """

    return BookMetrics(total_pv=total_pv, bucket_pv=bucket_pv, risk=risk)


def book_metrics(
    book,
    valuation_dates: pd.Index | list[pd.Timestamp],
    par_yields: pd.DataFrame,
    *,
    methods=("loglinear", "pchip", "nss", "qp"),
    holdouts: list[str] | None = None,
    freq: int = 2,
    short_end: str = "continuous",
    min_df: float = 1e-12,
    bump_bp: float = 1.0,
    tenor_cols: list[str] | None = None,
) -> tuple[BookMetrics, pd.DataFrame]:
    """Build valuation and rate-risk metrics for a synthetic issuance book.

    Parameters
    ----------
    book : object
        Synthetic issuance book.
    valuation_dates : pandas.Index or list of pandas.Timestamp
        Dates on which to value the book.
    par_yields : pandas.DataFrame
        Date-indexed par-yield curve panel.
    methods : iterable of str, default ("loglinear", "pchip", "nss", "qp")
        Curve-fitting methods to use.
    holdouts : list of str or None, optional
        Accepted for compatibility and ignored by this routine.
    freq : int, default 2
        Coupon frequency used in bootstrapping.
    short_end : str, default "continuous"
        Short-end bootstrap convention.
    min_df : float, default 1e-12
        Discount-factor floor.
    bump_bp : float, default 1.0
        Bump size used in risk calculations.
    tenor_cols : list of str or None, optional
        Tenor columns to use.

    Returns
    -------
    tuple[BookMetrics, pandas.DataFrame]
        Book-metrics container and key-rate duration table.

    Notes
    -----
    The function fits curves for each valuation date, computes total and bucket PV,
    parallel PV01/convexity, and key-rate duration across book maturities.
    """

    _ = holdouts
    curves_for_dates = curves_by_valuation_date(
        valuation_dates,
        par_yields,
        methods=methods,
        freq=freq,
        short_end=short_end,
        min_df=min_df,
        tenor_cols=tenor_cols,
    )
    total_pv, bucket_pv = book_pv_timeseries(book, curves_for_dates)
    risk_df = book_parallel_risk_timeseries(book, curves_for_dates, bump_bp=bump_bp)
    krd_df = book_krd_timeseries(book, curves_for_dates, keys=book.maturities, bump_bp=bump_bp)
    return make_book_metrics(total_pv, bucket_pv, risk_df), krd_df


__all__ = [
    "bond_modified_duration",
    "bond_price_and_risk",
    "book_krd_timeseries",
    "book_metrics",
    "book_parallel_risk_timeseries",
    "bucket_bump_func",
    "duration_sanity_table",
    "dv01",
    "key_bump_func",
    "krd_pivot",
    "latest_krd_table",
    "make_book_metrics",
    "portfolio_key_rate_risk",
    "portfolio_modified_duration",
    "portfolio_parallel_risk",
    "price_from_ytm",
    "pv01",
    "pv01_sanity_table",
    "shifted_df_func",
    "solve_bond_ytm",
    "strategy_risk_timeseries",
]
