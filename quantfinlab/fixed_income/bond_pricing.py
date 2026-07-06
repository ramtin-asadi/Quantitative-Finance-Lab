from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from ..common.contracts import Bond, Curve, IssuanceBook, IssuedBond
from ..common.dates import yearfrac
from .tenors import DEFAULT_ISSUE_MATURITIES, nearest_tenor_label


def synthetic_issuance_book(
    month_end_curve: pd.DataFrame,
    *,
    maturities: list[int] | tuple[int, ...] | None = None,
    freq: int = 2,
    col_map: dict[int, str] | None = None,
) -> IssuanceBook:
    """Build a synthetic par-bond issuance book from month-end curves.

    Parameters
    ----------
    month_end_curve : pandas.DataFrame
        Date-indexed par-yield curve panel. Each row is treated as an issuance
        date, and each selected maturity is issued at the par yield available on
        that date.
    maturities : list of int, tuple of int, or None, optional
        Maturity buckets to issue. If ``None``, the default issuance maturities are
        used.
    freq : int, default 2
        Coupon payment frequency per year.
    col_map : dict[int, str] or None, optional
        Mapping from maturity bucket to the curve column used as the coupon source.
        If omitted, maturities map to labels such as ``2 -> "2Y"``.

    Returns
    -------
    IssuanceBook
        Synthetic book grouped by maturity bucket.

    Notes
    -----
    Each issued bond has face value one and a coupon equal to the observed par
    yield for its maturity bucket on the issue date. Rows with missing or non-finite
    coupon data are skipped for the affected maturity.
    """

    if maturities is None:
        maturities = list(DEFAULT_ISSUE_MATURITIES)
    maturities = [int(x) for x in maturities]
    col_map = col_map or {m: f"{m}Y" for m in maturities}
    by_mat: dict[int, list[IssuedBond]] = {m: [] for m in maturities}

    for d in month_end_curve.index:
        row = month_end_curve.loc[d]
        for m in maturities:
            col = col_map[m]
            c = float(row.get(col, np.nan))
            if not np.isfinite(c):
                continue
            times, cfs = bond_cashflows(c, float(m), freq=freq, face=1.0)
            by_mat[m].append(IssuedBond(
                issue_date=pd.Timestamp(d), maturity_years=m, coupon=c, freq=freq, times=times, cfs=cfs
            ))

    return IssuanceBook(maturities=list(maturities), freq=freq, by_maturity=by_mat)


def bond_cashflows(coupon: float, maturity_years: float, *, freq: int = 2, face: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed-coupon bond cash-flow times and amounts.

    Parameters
    ----------
    coupon : float
        Annual coupon rate expressed as a decimal.
    maturity_years : float
        Maturity in years.
    freq : int, default 2
        Number of coupon payments per year.
    face : float, default 1.0
        Face value repaid at maturity.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Payment times in years and corresponding cash-flow amounts.

    Notes
    -----
    Coupon payments are level at ``coupon / freq * face`` and the final cash flow
    includes principal repayment. The schedule assumes regular coupon intervals and
    does not model stub periods or calendar adjustments.
    """

    times = np.arange(1 / freq, maturity_years + 1e-9, 1 / freq)
    cfs = np.full_like(times, (coupon / freq) * face, dtype=float)
    cfs[-1] += face
    return times.astype(float), cfs.astype(float)


def price_bond_from_issue(df_func: Callable[[np.ndarray], np.ndarray], times: np.ndarray, cfs: np.ndarray, age: float) -> float:
    """Price remaining cash flows of a bond after a given age.

    Parameters
    ----------
    df_func : callable
        Discount-factor function accepting remaining maturities in years.
    times : numpy.ndarray
        Original cash-flow times from issue date, in years.
    cfs : numpy.ndarray
        Cash-flow amounts corresponding to ``times``.
    age : float
        Years elapsed since issue date.

    Returns
    -------
    float
        Present value of all cash flows with payment time greater than ``age``.
        Returns zero if no future cash flows remain.

    Notes
    -----
    The function discounts remaining cash flows by time-to-payment, not by original
    payment time. Cash flows at or before the age cutoff are excluded.
    """

    mask = times > age + 1e-12
    if not np.any(mask):
        return 0.0
    t_rem = times[mask] - age
    cf_rem = cfs[mask]
    return float(np.sum(cf_rem * df_func(t_rem)))


def _book_pv(
    book: IssuanceBook,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    cutoff_date: pd.Timestamp,
) -> tuple[float, dict[int, float]]:
    total = 0.0
    buckets: dict[int, float] = {}

    for m in book.maturities:
        pv_m = 0.0
        for b in book.by_maturity[m]:
            if b.issue_date > cutoff_date:
                break  # issued list is chronological
            age = yearfrac(b.issue_date, valuation_date)
            if age >= b.times[-1] - 1e-12:
                continue
            pv_m += price_bond_from_issue(df_func, b.times, b.cfs, age)
        buckets[m] = float(pv_m)
        total += float(pv_m)

    return float(total), buckets


def book_pv_timeseries(
    book: IssuanceBook,
    curves_for_dates: dict[pd.Timestamp, dict[str, Curve]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute total and maturity-bucket present values for a synthetic book.

    Parameters
    ----------
    book : IssuanceBook
        Synthetic issuance book to value.
    curves_for_dates : dict[pandas.Timestamp, dict[str, Curve]]
        Nested mapping from valuation date to curve method to fitted curve.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Total present value table indexed by date with curve methods as columns,
        and bucket present value table indexed by date with a method/maturity
        column MultiIndex.

    Notes
    -----
    Each valuation uses only bonds issued before or on the valuation date. The
    curve's discount-factor function is used directly for present-value
    calculation.
    """

    pv_records: list[dict] = []
    bucket_records: list[dict] = []

    for vd in sorted(curves_for_dates):
        curves_d = curves_for_dates[vd]
        for method, curve in curves_d.items():
            pv0, buckets = _book_pv(book, vd, curve.df, cutoff_date=vd)
            pv_records.append({"date": vd, "method": method, "pv": pv0})
            for m in book.maturities:
                bucket_records.append(
                    {"date": vd, "method": method, "maturity": m, "pv": buckets.get(m, 0.0)}
                )

    total_pv = pd.DataFrame(pv_records).pivot(index="date", columns="method", values="pv").sort_index()
    bucket_pv = (
        pd.DataFrame(bucket_records)
        .pivot_table(index="date", columns=["method", "maturity"], values="pv")
        .sort_index()
    )
    return total_pv, bucket_pv


def bond_from_par_curve_row(
    row: pd.Series,
    *,
    maturity_years: float,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    face: float = 1.0,
) -> tuple[Bond, str]:
    """Create a fixed-coupon bond from the nearest tenor in a par-curve row.

    Parameters
    ----------
    row : pandas.Series
        One row of a par-yield curve panel.
    maturity_years : float
        Desired bond maturity in years.
    tenor_cols : list of str or None, optional
        Candidate tenor columns. If omitted, all row labels are considered.
    freq : int, default 2
        Coupon payment frequency per year.
    face : float, default 1.0
        Bond face value.

    Returns
    -------
    tuple[Bond, str]
        Bond whose coupon is the nearest-tenor par yield, and the tenor label used.

    Notes
    -----
    The input row is assumed to contain yields in decimal units. No interpolation
    is performed; the coupon is taken from the nearest available tenor label.
    """

    cols = tenor_cols if tenor_cols is not None else [str(c) for c in row.index]
    tenor_label = nearest_tenor_label(cols, target_maturity_years=maturity_years)
    coupon = float(row[tenor_label])
    return Bond(coupon=coupon, maturity_years=float(maturity_years), freq=freq, face=face), tenor_label


def bond_price(
    bond: Bond,
    curve: Curve,
    *,
    settle: float = 0.0,   # years since last coupon date (0 means on coupon date)
    clean: bool = True,
) -> float:
    """Price a fixed-coupon bond from a fitted discount curve.

    Parameters
    ----------
    bond : Bond
        Bond specification.
    curve : Curve
        Fitted curve providing a discount-factor function.
    settle : float, default 0.0
        Years since the last coupon date. A value of zero represents settlement on
        a coupon date.
    clean : bool, default True
        If ``True``, subtract simple accrued interest from the dirty price.

    Returns
    -------
    float
        Clean or dirty bond price depending on ``clean``.

    Notes
    -----
    Accrued interest is approximated as ``coupon * face * settle``. The function
    uses a simplified regular coupon schedule and does not apply full market
    settlement conventions.
    """

    times, cfs = bond_cashflows(bond.coupon, bond.maturity_years, freq=bond.freq, face=bond.face)
    dirty = price_bond_from_issue(curve.df, times, cfs, age=settle)
    if not clean:
        return dirty
    accrued = bond.coupon * bond.face * settle  # simple time-based accrued interest
    return float(dirty - accrued)


def make_synthetic_bond(
    issue_date: pd.Timestamp,
    maturity_years: float,
    coupon: float,
    *,
    units: float = 0.0,
    freq: int = 2,
) -> dict:
    """Create a dictionary representation of a synthetic fixed-coupon bond.

    Parameters
    ----------
    issue_date : pandas.Timestamp
        Bond issue date.
    maturity_years : float
        Original maturity in years.
    coupon : float
        Annual coupon rate expressed as a decimal.
    units : float, default 0.0
        Position size in face-value units.
    freq : int, default 2
        Coupon payment frequency per year.

    Returns
    -------
    dict
        Synthetic bond record containing issue date, original maturity, coupon,
        relative cash-flow times, cash-flow amounts, payment dates, units, and
        frequency.

    Notes
    -----
    Payment dates are approximated by adding rounded month offsets to the issue
    date. The representation is optimized for backtesting synthetic ladders rather
    than exact bond-settlement accounting.
    """

    times, cfs = bond_cashflows(coupon, maturity_years, freq=freq)
    payment_dates = pd.to_datetime(
        [pd.Timestamp(issue_date) + pd.DateOffset(months=round(12 * t)) for t in times]
    )
    return {
        "issue_date": pd.Timestamp(issue_date),
        "original_maturity": float(maturity_years),
        "coupon": float(coupon),
        "times": times.astype(float),
        "cfs": cfs.astype(float),
        "payment_dates": payment_dates,
        "units": float(units),
        "freq": int(freq),
    }


def remaining_maturity(bond: dict | None, valuation_date: pd.Timestamp) -> float:
    """Compute remaining maturity of a synthetic bond.

    Parameters
    ----------
    bond : dict or None
        Synthetic bond record.
    valuation_date : pandas.Timestamp
        Date at which remaining maturity is measured.

    Returns
    -------
    float
        Remaining maturity in years. Returns zero for ``None`` bonds or bonds that
        have fully matured.

    Notes
    -----
    The elapsed time is computed using the package year-fraction convention.
    """

    if bond is None:
        return 0.0
    delta = yearfrac(bond["issue_date"], valuation_date)
    return float(max(bond["times"][-1] - delta, 0.0))


def remaining_cashflow_arrays(
    bond: dict | None,
    valuation_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return remaining cash-flow times and amounts for a synthetic bond position.

    Parameters
    ----------
    bond : dict or None
        Synthetic bond record with cash-flow arrays and units.
    valuation_date : pandas.Timestamp
        Valuation date.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Remaining times to payment in years and position-scaled cash-flow amounts.
        Empty arrays are returned for missing, zero-unit, or fully matured bonds.

    Notes
    -----
    Only cash flows strictly after the valuation date are retained.
    """

    if bond is None or bond.get("units", 0.0) <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    delta = yearfrac(bond["issue_date"], valuation_date)
    mask = bond["times"] > delta + 1e-12
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    t_rem = bond["times"][mask] - delta
    cf_rem = bond["cfs"][mask] * bond["units"]
    return t_rem, cf_rem


def bond_position_value(
    bond: dict | None,
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
) -> float:
    """Value a synthetic bond position using a discount-factor function.

    Parameters
    ----------
    bond : dict or None
        Synthetic bond record with units, cash-flow times, and cash-flow amounts.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Discount-factor function accepting maturities in years.

    Returns
    -------
    float
        Present value of the position. Returns zero for missing or zero-unit bonds.

    Notes
    -----
    The function prices one unit of the bond from its original cash-flow schedule
    and scales by the current position units.
    """

    if bond is None or bond.get("units", 0.0) <= 0:
        return 0.0
    delta = yearfrac(bond["issue_date"], valuation_date)
    return float(bond["units"] * price_bond_from_issue(df_func, bond["times"], bond["cfs"], delta))


def position_values_by_bucket(
    positions: dict[int, dict],
    valuation_date: pd.Timestamp,
    df_func: Callable[[np.ndarray], np.ndarray],
    *,
    buckets: list[int] | tuple[int, ...] = DEFAULT_ISSUE_MATURITIES,
) -> dict[int, float]:
    """Value synthetic bond positions by maturity bucket.

    Parameters
    ----------
    positions : dict[int, dict]
        Mapping from maturity bucket to synthetic bond record.
    valuation_date : pandas.Timestamp
        Valuation date.
    df_func : callable
        Discount-factor function.
    buckets : list of int or tuple of int, default DEFAULT_ISSUE_MATURITIES
        Buckets to include in the output.

    Returns
    -------
    dict[int, float]
        Present value for each requested bucket. Missing buckets receive zero.

    Notes
    -----
    The function preserves the requested bucket order through the returned mapping.
    """

    return {
        int(m): bond_position_value(positions.get(int(m)), valuation_date, df_func)
        for m in buckets
    }


def bond_cashflows_between(
    bond: dict | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[float, float, float]:
    """Sum bond cash flows paid within a date interval.

    Parameters
    ----------
    bond : dict or None
        Synthetic bond record with payment dates, cash flows, coupon, frequency,
        and units.
    start_date : pandas.Timestamp
        Start of the interval. Payments on this date are excluded.
    end_date : pandas.Timestamp
        End of the interval. Payments on this date are included.

    Returns
    -------
    tuple[float, float, float]
        Gross cash flow, coupon component, and principal component.

    Notes
    -----
    The coupon component is computed from the number of payment dates in the
    interval and the bond's coupon/frequency. The principal component is the
    residual gross amount after coupon income.
    """

    if bond is None or bond.get("units", 0.0) <= 0:
        return 0.0, 0.0, 0.0

    pay_mask = (bond["payment_dates"] > start_date) & (bond["payment_dates"] <= end_date)
    if not np.any(pay_mask):
        return 0.0, 0.0, 0.0

    gross = float(np.sum(bond["cfs"][pay_mask] * bond["units"]))
    n_pay = int(np.sum(pay_mask))
    coupon = float(n_pay * (bond["coupon"] / bond["freq"]) * bond["units"])
    principal = gross - coupon
    return gross, coupon, principal


__all__ = [
    "Bond",
    "bond_cashflows",
    "bond_cashflows_between",
    "bond_from_par_curve_row",
    "bond_position_value",
    "bond_price",
    "book_pv_timeseries",
    "make_synthetic_bond",
    "position_values_by_bucket",
    "price_bond_from_issue",
    "remaining_cashflow_arrays",
    "remaining_maturity",
    "synthetic_issuance_book",
    "yearfrac",
]
