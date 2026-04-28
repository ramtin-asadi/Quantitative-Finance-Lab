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
    """
    Build a synthetic issuance book:
    - For each month-end date, "issue" a par bond in each maturity bucket
    - coupon = par yield at that maturity for that date
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
    times = np.arange(1 / freq, maturity_years + 1e-9, 1 / freq)
    cfs = np.full_like(times, (coupon / freq) * face, dtype=float)
    cfs[-1] += face
    return times.astype(float), cfs.astype(float)


def price_bond_from_issue(df_func: Callable[[np.ndarray], np.ndarray], times: np.ndarray, cfs: np.ndarray, age: float) -> float:
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
    """
    Compute total PV and bucket PV by valuation date/method.
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
    if bond is None:
        return 0.0
    delta = yearfrac(bond["issue_date"], valuation_date)
    return float(max(bond["times"][-1] - delta, 0.0))


def remaining_cashflow_arrays(
    bond: dict | None,
    valuation_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
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
    return {
        int(m): bond_position_value(positions.get(int(m)), valuation_date, df_func)
        for m in buckets
    }


def bond_cashflows_between(
    bond: dict | None,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[float, float, float]:
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
