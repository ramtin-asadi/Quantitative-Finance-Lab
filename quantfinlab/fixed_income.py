from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from os import PathLike
from typing import Literal

import numpy as np
import pandas as pd

from .core import (
    Bond,
    BookMetrics,
    Curve,
    CurvePillars,
    InputError,
    IssuanceBook,
    IssuedBond,
)

TENOR_PATTERN = re.compile(r"^\d+[MY]$")
DEFAULT_METHODS = ("loglinear", "pchip", "nss", "qp")
DEFAULT_HOLDOUTS = ("6M", "2Y", "7Y", "20Y")
DEFAULT_ISSUE_MATURITIES = (2, 5, 10, 30)

_COLUMN_ALIASES = {
    "date": "date",
    "1 mo": "1M",
    "2 mo": "2M",
    "3 mo": "3M",
    "4 mo": "4M",
    "6 mo": "6M",
    "1 yr": "1Y",
    "2 yr": "2Y",
    "3 yr": "3Y",
    "5 yr": "5Y",
    "7 yr": "7Y",
    "10 yr": "10Y",
    "20 yr": "20Y",
    "30 yr": "30Y",
}

def normalize_par_yields(
    raw: pd.DataFrame,
    *,
    date_col: str | None = None,
    tenor_cols: list[str] | None = None,
    assume_percent: bool | None = None,
) -> pd.DataFrame:
    """
    Normalize a raw par-yield table:
    - harmonize common column names (e.g., '1 mo' -> '1M')
    - detect/parse date index
    - detect tenor columns and sort by maturity
    - convert yields to float and (optionally) percent -> decimal
    """
    if raw.empty:
        raise InputError("Input DataFrame is empty.")

    data = raw.copy()

    def _normalize_col_name(col: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(col).strip())
        key = cleaned.lower()
        if key in _COLUMN_ALIASES:
            return _COLUMN_ALIASES[key]
        compact = cleaned.replace(" ", "").upper()
        if TENOR_PATTERN.fullmatch(compact):
            return compact
        return cleaned

    data = data.rename(columns={c: _normalize_col_name(c) for c in data.columns})

    normalized_date_col = _normalize_col_name(date_col) if date_col is not None else None

    if normalized_date_col is not None and normalized_date_col in data.columns and normalized_date_col != "date":
        data = data.rename(columns={normalized_date_col: "date"})
    elif normalized_date_col is not None and normalized_date_col.lower() == "date" and "date" in data.columns:
        pass
    elif "date" not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data = data.rename(columns={data.columns[0]: "date"})

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).set_index("date")
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise InputError("Could not detect a date column/index.")

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if tenor_cols is None:
        detected = [c for c in data.columns if TENOR_PATTERN.fullmatch(str(c).strip().upper())]
    else:
        detected = [_normalize_col_name(c) for c in tenor_cols]

    if not detected:
        raise InputError("No tenor columns detected (expected labels like 6M, 2Y, 10Y).")

    detected = sorted(dict.fromkeys(detected), key=tenor_to_years)
    table = data[detected].apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()
    if table.empty:
        raise InputError("No usable tenor data after numeric coercion.")

    if assume_percent is None:
        med = float(np.nanmedian(table.to_numpy(dtype=float)))
        assume_percent = bool(np.isfinite(med) and med > 1.0)

    if assume_percent:
        table = table / 100.0

    return table


def load_par_yields_csv(
    path: str | PathLike[str],
    *,
    date_col: str | None = None,
    tenor_cols: list[str] | None = None,
    assume_percent: bool | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    raw = pd.read_csv(path, **read_csv_kwargs)
    return normalize_par_yields(
        raw,
        date_col=date_col,
        tenor_cols=tenor_cols,
        assume_percent=assume_percent,
    )


# ----------------------------
# Tenor parsing / extraction
# ----------------------------

def tenor_to_years(x: str | int | float) -> float:
    """
    Convert tenor labels like '6M', '2Y' to years.
    Also accepts numeric years (int/float).
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().upper()
    if s.endswith("M"):
        return float(int(s[:-1])) / 12.0
    if s.endswith("Y"):
        return float(int(s[:-1]))
    # allow '2' to mean 2Y
    if s.isdigit():
        return float(int(s))
    raise ValueError(f"Unsupported tenor label: {x!r}")


def extract_par_curve(
    row: pd.Series | dict,
    tenor_cols: list[str] | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Extract (labels, T, par) from a row-like object.
    - If tenor_cols is None, detect columns like '1M','6M','1Y','2Y',...
    - Returns par yields as decimals (assumes input already decimals).
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    if tenor_cols is None:
        tenor_cols = [
            c
            for c in row.index.astype(str)
            if TENOR_PATTERN.fullmatch(str(c).strip().replace(" ", "").upper())
        ]

    if not tenor_cols:
        raise InputError("No tenor columns detected. Pass tenor_cols explicitly.")

    y = row[tenor_cols].astype(float)
    mask = np.isfinite(y.values)
    labels = [tenor_cols[i] for i in range(len(tenor_cols)) if mask[i]]
    if not labels:
        raise InputError("All tenor values are NaN/non-finite for this row.")

    par = y.values[mask].astype(float)
    T = np.array([tenor_to_years(label) for label in labels], dtype=float)

    idx = np.argsort(T)
    T = T[idx]
    par = par[idx]
    labels = [labels[i] for i in idx]
    return labels, T, par


# ----------------------------
# Bootstrapping pillars
# ----------------------------

def bootstrap_pillars(
    par_curve_row: pd.Series | dict,
    *,
    asof: pd.Timestamp | None = None,
    tenor_cols: list[str] | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
) -> CurvePillars:
    """
    Bootstrap discount factors at observed tenors from a par-yield curve row.

    Convention:
    - For T < 1y: DF(T) = exp(-r*T) if short_end='continuous',
                 or 1/(1+r*T) if short_end='simple'
    - For T >= 1y: solve for DF(T) from par-bond equation with coupon=par yield,
      allowing log-linear interpolation between last known DF and the unknown DF(T).
    """
    labels, T, par = extract_par_curve(par_curve_row, tenor_cols=tenor_cols)
    dfs = bootstrap_from_inputs(
        T=T,
        par=par,
        labels=labels,
        date=asof,
        freq=freq,
        short_end=short_end,
        min_df=min_df,
    )["dfs"]
    return CurvePillars(asof=asof, labels=labels, T=T, par=par, dfs=dfs)


def bootstrap_from_inputs(
    *,
    T: np.ndarray,
    par: np.ndarray,
    labels: list[str],
    date: pd.Timestamp | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
) -> dict:
    d_map: dict[float, float] = {}

    for Ti, ri in zip(T, par, strict=True):
        Ti = float(Ti)
        ri = float(ri)

        if Ti < 1.0:
            d_T = _short_end_df(Ti, ri, short_end=short_end)
            d_map[Ti] = max(float(d_T), min_df)
            continue

        d_T = _solve_df_long_end(Ti, ri, d_map, freq=freq, min_df=min_df)
        if (not np.isfinite(d_T)) or (d_T <= 0):
            d_T = min_df
        d_map[Ti] = max(float(d_T), min_df)

    dfs = np.array([d_map[float(t)] for t in T], dtype=float)
    return {"date": date, "T": T, "par": par, "labels": labels, "dfs": dfs}


def _short_end_df(T: float, r: float, *, short_end: Literal["continuous", "simple"]) -> float:
    if short_end == "continuous":
        return math.exp(-r * T)
    return 1.0 / (1.0 + r * T)


def _price_error_loglinear(
    d_T: float,
    Ti: float,
    t_prev: float,
    d_prev: float,
    times_interp: np.ndarray,
    c: float,
    pv_known: float,
    *,
    freq: int,
    min_df: float,
) -> float:
    d_T = max(float(d_T), min_df)
    pv_interp = 0.0
    if len(times_interp) > 0:
        w = (times_interp - t_prev) / (Ti - t_prev)
        log_d = (1 - w) * np.log(d_prev) + w * np.log(d_T)
        d_interp = np.exp(log_d)
        pv_interp = float(np.sum((c / freq) * d_interp))
    return pv_known + pv_interp + d_T - 1.0


def _solve_df_long_end(
    Ti: float,
    ri: float,
    d_map: dict[float, float],
    *,
    freq: int,
    min_df: float,
) -> float:
    # If there are no prior pillars (e.g., curve starts at 1Y),
    # seed the first long-end DF with a continuous short-end proxy.
    if len(d_map) == 0:
        return float(max(math.exp(-ri * Ti), min_df))

    c = float(ri)
    n = int(round(Ti * freq))
    times = np.array([k / freq for k in range(1, n + 1)], dtype=float)

    known_T = np.array(sorted(d_map.keys()), dtype=float)
    known_D = np.array([d_map[t] for t in known_T], dtype=float)
    known_D = np.clip(known_D, min_df, None)

    t_prev = float(known_T[-1])
    d_prev = float(known_D[-1])

    times_known = times[times <= t_prev + 1e-12]
    times_interp = times[times > t_prev + 1e-12]

    pv_known = 0.0
    if len(times_known) > 0:
        log_known_D = np.log(known_D)
        log_df_known = np.interp(times_known, known_T, log_known_D)
        d_known = np.exp(log_df_known)
        pv_known = float(np.sum((c / freq) * d_known))

    lo = min_df
    hi = d_prev

    f_lo = _price_error_loglinear(lo, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)
    f_hi = _price_error_loglinear(hi, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)

    if f_lo * f_hi > 0:
        # fallback: assume coupon DFs can be obtained by log-linear interpolation from known pillars only
        log_known_D = np.log(known_D)
        log_df_cpn = np.interp(times[:-1], known_T, log_known_D, left=log_known_D[0], right=log_known_D[-1])
        d_cpn = np.exp(log_df_cpn)
        pv_coupons = float(np.sum((c / freq) * d_cpn))
        d_T = (1.0 - pv_coupons) / (1.0 + c / freq)
        return float(max(d_T, min_df))

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = _price_error_loglinear(mid, Ti, t_prev, d_prev, times_interp, c, pv_known, freq=freq, min_df=min_df)
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
        if abs(hi - lo) < 1e-12:
            break

    return float(max(0.5 * (lo + hi), min_df))


# ----------------------------
# Curve helpers (par/zero/forward)
# ----------------------------

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

        n = int(round(Ti * freq))
        times = np.array([k / freq for k in range(1, n + 1)], dtype=float)
        dfs = df_func(times)
        denom = float(np.sum(dfs))
        if denom <= 0:
            continue
        dT = float(dfs[-1])
        out[i] = freq * (1.0 - dT) / denom
    return out


def _curve_grid(T_min: float, T_max: float = 30.0, n: int = 1000) -> np.ndarray:
    return np.linspace(max(1 / 12, T_min), T_max, n)


def _curve_from_df_func(method: str, name: str, df_func: Callable[[np.ndarray], np.ndarray], *, T_min: float) -> Curve:
    grid = _curve_grid(T_min)
    df_grid = df_func(grid)
    df_grid = np.clip(df_grid, 1e-16, None)
    z_grid = -np.log(df_grid) / grid
    fwd_grid = -np.gradient(np.log(df_grid), grid)
    return Curve(method=method, name=name, grid=grid, df_grid=df_grid, z_grid=z_grid, fwd_grid=fwd_grid, df=df_func)


# ----------------------------
# 4 curve methods
# ----------------------------

def fit_curves(
    pillars: CurvePillars,
    *,
    methods: Iterable[str] = ("loglinear", "pchip", "nss", "qp"),
    freq: int = 2,
    min_df: float = 1e-12,
) -> dict[str, Curve]:
    """
    Build multiple curves from pillars using selected methods.
    Returns dict: method -> Curve
    """
    methods = list(methods)
    T = pillars.T
    par = pillars.par
    labels = pillars.labels
    dfs = pillars.dfs

    out: dict[str, Curve] = {}
    for m in methods:
        mm = m.lower().strip()
        if mm == "loglinear":
            out["loglinear"] = _loglinear_curve(T, dfs, min_df=min_df)
        elif mm == "pchip":
            out["pchip"] = _pchip_curve(T, dfs, min_df=min_df)
        elif mm == "nss":
            out["nss"] = _nss_curve(T, par, min_df=min_df, freq=freq)
        elif mm == "qp":
            out["qp"] = _qp_curve(labels, par, freq=freq, min_df=min_df)
        else:
            raise ValueError(f"Unknown curve method: {m!r}")
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


def forward_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
) -> pd.DataFrame:
    return curve_value_table(curves, value="forward", grid=grid, t_min=t_min, t_max=t_max, points=points)


def _loglinear_curve(T: np.ndarray, dfs: np.ndarray, *, min_df: float) -> Curve:
    log_dfs = np.log(np.clip(dfs, min_df, None))

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        log_df = np.interp(tt, T, log_dfs, left=log_dfs[0], right=log_dfs[-1])
        return np.exp(log_df)

    return _curve_from_df_func("loglinear", "Log-linear DF", df_func, T_min=float(T.min()))


def _pchip_curve(T: np.ndarray, dfs: np.ndarray, *, min_df: float) -> Curve:
    try:
        from scipy.interpolate import PchipInterpolator
    except Exception as e:  # pragma: no cover
        raise ImportError("PCHIP requires scipy. Install scipy to use method='pchip'.") from e

    zeros = -np.log(np.clip(dfs, min_df, None)) / T
    z_spline = PchipInterpolator(T, zeros, extrapolate=True)

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        z = z_spline(tt)
        return np.exp(-z * tt)

    return _curve_from_df_func("pchip", "PCHIP zero", df_func, T_min=float(T.min()))


def _nss_zero(T: np.ndarray, b0: float, b1: float, b2: float, b3: float, tau1: float, tau2: float) -> np.ndarray:
    T = np.array(T, dtype=float)
    T = np.clip(T, 1e-12, None)

    x1 = T / max(tau1, 1e-12)
    x2 = T / max(tau2, 1e-12)

    f1 = (1 - np.exp(-x1)) / x1
    f2 = f1 - np.exp(-x1)
    g1 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)

    return b0 + b1 * f1 + b2 * f2 + b3 * g1


def _nss_curve(T: np.ndarray, par: np.ndarray, *, min_df: float, freq: int) -> Curve:
    try:
        from scipy.optimize import minimize
    except Exception as e:  # pragma: no cover
        raise ImportError("NSS requires scipy. Install scipy to use method='nss'.") from e

    T = np.array(T, dtype=float)
    par = np.array(par, dtype=float)

    def obj(theta: np.ndarray) -> float:
        b0, b1, b2, b3, tau1, tau2 = theta
        z = _nss_zero(T, b0, b1, b2, b3, tau1, tau2)
        dfs_p = np.exp(-z * T)
        log_d = np.log(np.clip(dfs_p, min_df, None))

        def df_func_p(tt: np.ndarray) -> np.ndarray:
            ttt = np.array(tt, dtype=float)
            log_df = np.interp(ttt, T, log_d, left=log_d[0], right=log_d[-1])
            return np.exp(log_df)

        par_fit = par_from_df(df_func_p, T, freq=freq)
        err = par_fit - par
        return float(np.mean(err**2))

    b0_0 = float(np.nanmedian(par[-3:])) if len(par) >= 3 else float(np.nanmedian(par))
    x0 = np.array([b0_0, -0.02, 0.02, 0.01, 1.5, 5.0], dtype=float)

    res = minimize(obj, x0, method="L-BFGS-B")
    b0, b1, b2, b3, tau1, tau2 = res.x

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        z = _nss_zero(tt, b0, b1, b2, b3, tau1, tau2)
        return np.exp(-z * tt)

    return _curve_from_df_func("nss", "NSS", df_func, T_min=float(T.min()))


def _qp_curve(labels: list[str], par_mkt: np.ndarray, *, freq: int, min_df: float) -> Curve:
    """
    QP smoothing approach (matches your notebook design):
    - decision variables are discount factors on a time grid (coupon grid + observed maturities)
    - constraints enforce exact par pricing at observed maturities (linear constraints)
    - objective smooths second differences of DFs plus a small pull to a prior curve
    """
    try:
        import cvxpy  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("QP method requires cvxpy. Install cvxpy to use method='qp'.") from e

    T_obs = np.array([tenor_to_years(label) for label in labels], dtype=float)
    par_mkt = np.array(par_mkt, dtype=float)

    idx = np.argsort(T_obs)
    T_obs = T_obs[idx]
    par_mkt = par_mkt[idx]

    t_grid, grid_index = _qp_build_t_grid(T_obs, freq=freq)
    d, constraints = _qp_build_constraints(t_grid, grid_index, T_obs, par_mkt, freq=freq, min_df=min_df)
    d_sol, status = _qp_solve(t_grid, d, constraints, par_mkt, freq=freq, min_df=min_df)

    if status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"QP solve failed with status={status!r}")

    log_d = np.log(np.clip(d_sol, min_df, None))

    def df_func(t: np.ndarray | float) -> np.ndarray:
        tt = np.array(t, dtype=float)
        log_df = np.interp(tt, t_grid, log_d, left=log_d[0], right=log_d[-1])
        return np.exp(log_df)

    return _curve_from_df_func("qp", "QP DF", df_func, T_min=float(t_grid.min()))


def _qp_build_t_grid(T_obs: np.ndarray, *, freq: int) -> tuple[np.ndarray, dict[float, int]]:
    T_max = float(np.max(T_obs))
    n_grid = int(round(T_max * freq))
    base = np.array([i / freq for i in range(1, n_grid + 1)], dtype=float)
    t_grid = np.unique(np.concatenate([base, T_obs]))
    t_grid = np.array(sorted(t_grid), dtype=float)
    grid_index = {float(np.round(t, 10)): i for i, t in enumerate(t_grid)}
    return t_grid, grid_index


def _qp_build_constraints(
    t_grid: np.ndarray,
    grid_index: dict[float, int],
    T_obs: np.ndarray,
    par_mkt: np.ndarray,
    *,
    freq: int,
    min_df: float,
):
    import cvxpy as cp

    d = cp.Variable(len(t_grid))
    constraints = [d >= min_df, d[1:] <= d[:-1]]

    # Pin short-end nodes to exp(-y*T) when T<1
    for Tk, yk in zip(T_obs, par_mkt, strict=True):
        if Tk < 1.0:
            key = float(np.round(Tk, 10))
            if key in grid_index:
                i = grid_index[key]
                constraints.append(d[i] == float(np.exp(-yk * Tk)))

    # Par-bond constraints (linear): sum coupons + principal = 1
    for Tk, ck in zip(T_obs, par_mkt, strict=True):
        if Tk < 1.0:
            continue
        keyT = float(np.round(Tk, 10))
        if keyT not in grid_index:
            continue
        iT = grid_index[keyT]
        n = int(round(Tk * freq))

        coupon_idx = []
        for j in range(1, n + 1):
            key = float(np.round(j / freq, 10))
            coupon_idx.append(grid_index[key])

        constraints.append(cp.sum((ck / freq) * d[coupon_idx]) + d[iT] == 1.0)

    return d, constraints


def _qp_solve(
    t_grid: np.ndarray,
    d,
    constraints,
    par_mkt: np.ndarray,
    *,
    freq: int,
    min_df: float,
) -> tuple[np.ndarray, str]:
    import cvxpy as cp

    lam = 1e4
    eps = 1e-4
    prior_rate = float(np.nanmedian(par_mkt[-3:])) if len(par_mkt) >= 3 else float(np.nanmedian(par_mkt))
    d_prior = np.exp(-prior_rate * t_grid)

    d2 = d[2:] - 2 * d[1:-1] + d[:-2]
    obj = cp.Minimize(lam * cp.sum_squares(d2) + eps * cp.sum_squares(d - d_prior))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.OSQP)

    d_sol = np.array(d.value).astype(float)
    d_sol = np.clip(d_sol, min_df, None)
    return d_sol, str(prob.status)


# ----------------------------
# RMSE backtest (IS + OOS holdouts)
# ----------------------------

def rmse_backtest(
    par_yields: pd.DataFrame,
    *,
    methods: Iterable[str] = ("loglinear", "pchip", "nss", "qp"),
    holdouts: list[str] | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
    tenor_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Replicates your notebook logic:
    - For each date, optionally hold out a few tenors (if available and interior)
    - Fit curves on training set
    - Compute RMSE on training pillars (IS) and holdouts (OOS)
    """
    methods = [m.lower().strip() for m in methods]
    holdouts = holdouts or ["6M", "2Y", "7Y", "20Y"]
    curve_order = methods

    sse = {k: 0.0 for k in curve_order}
    cnt = {k: 0 for k in curve_order}
    n_dates = {k: 0 for k in curve_order}
    sse_oos = {k: 0.0 for k in curve_order}
    cnt_oos = {k: 0 for k in curve_order}
    n_dates_oos = {k: 0 for k in curve_order}

    failed: list[dict] = []

    for date in par_yields.index:
        row = par_yields.loc[date]
        try:
            pillars_full = bootstrap_pillars(
                row, asof=pd.Timestamp(date), tenor_cols=tenor_cols,
                freq=freq, short_end=short_end, min_df=min_df
            )
        except Exception as e:
            failed.append({"date": date, "method": "bootstrap", "error": str(e)})
            continue

        labels_full = pillars_full.labels
        T_full = pillars_full.T
        par_full = pillars_full.par

        holdout_idx = [labels_full.index(h) for h in holdouts if h in labels_full]
        holdout_idx = sorted(set([i for i in holdout_idx if 0 < i < len(labels_full) - 1]))

        min_train = 4
        if len(T_full) - len(holdout_idx) < min_train:
            holdout_idx = []

        if holdout_idx:
            mask_train = np.ones(len(T_full), dtype=bool)
            mask_train[holdout_idx] = False
            labels_tr = [labels_full[i] for i in range(len(labels_full)) if mask_train[i]]
            T_tr = T_full[mask_train]
            par_tr = par_full[mask_train]

            labels_te = [labels_full[i] for i in range(len(labels_full)) if not mask_train[i]]
            T_te = T_full[~mask_train]
            par_te = par_full[~mask_train]

            boot = bootstrap_from_inputs(
                T=T_tr, par=par_tr, labels=labels_tr, date=pd.Timestamp(date),
                freq=freq, short_end=short_end, min_df=min_df
            )
            pillars = CurvePillars(
                asof=pd.Timestamp(date), labels=labels_tr, T=T_tr, par=par_tr, dfs=boot["dfs"],
                labels_test=labels_te, T_test=T_te, par_test=par_te
            )
        else:
            pillars = pillars_full

        try:
            curves_d = fit_curves(pillars, methods=curve_order, freq=freq, min_df=min_df)
        except Exception as e:
            failed.append({"date": date, "method": "fit_curves", "error": str(e)})
            continue

        for k in curve_order:
            if k not in curves_d:
                continue
            c = curves_d[k]
            try:
                par_fit_tr = par_from_df(c.df, pillars.T, freq=freq)
                err_tr = par_fit_tr - pillars.par
                sse[k] += float(np.sum(err_tr**2))
                cnt[k] += int(len(err_tr))
                n_dates[k] += 1

                if pillars.T_test is not None and len(pillars.T_test) > 0:
                    par_fit_te = par_from_df(c.df, pillars.T_test, freq=freq)
                    err_te = par_fit_te - pillars.par_test
                    sse_oos[k] += float(np.sum(err_te**2))
                    cnt_oos[k] += int(len(err_te))
                    n_dates_oos[k] += 1
            except Exception as e:
                failed.append({"date": date, "method": k, "error": str(e)})

    rows = []
    for k in curve_order:
        if cnt[k] == 0:
            continue
        rmse_in = math.sqrt(sse[k] / cnt[k])
        rmse_out = math.sqrt(sse_oos[k] / cnt_oos[k]) if cnt_oos[k] > 0 else float("nan")
        rows.append({
            "method": k,
            "rmse": rmse_in,
            "rmse_oos": rmse_out,
            "n_obs": cnt[k],
            "n_obs_oos": cnt_oos[k],
            "n_dates": n_dates[k],
            "n_dates_oos": n_dates_oos[k],
            "n_failed": len(failed),
        })

    return pd.DataFrame(rows).set_index("method").sort_index()


# ----------------------------
# Synthetic issuance book + PV/risk
# ----------------------------

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


def yearfrac(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    return float((pd.Timestamp(t1) - pd.Timestamp(t0)).days) / 365.0


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


def shifted_df_func(df_func: Callable[[np.ndarray], np.ndarray], shift_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Apply a continuous-rate shift: DF_shifted(t) = DF(t) * exp(-shift(t)*t)
    where shift(t) is in absolute rate units (e.g. 0.0001 for 1bp).
    """
    def _f(t: np.ndarray) -> np.ndarray:
        tt = np.array(t, dtype=float)
        return df_func(tt) * np.exp(-shift_func(tt) * tt)
    return _f


def key_bump_func(keys: list[int], key: int, *, bump_bp: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Piecewise-linear key bump on the key-tenor grid (in continuous rate terms).
    Mimics your notebook: bump at one node, 0 elsewhere, interpolated linearly.
    """
    values = np.zeros(len(keys), dtype=float)
    k_idx = keys.index(key)
    values[k_idx] = bump_bp / 10000.0

    def shift(t: np.ndarray) -> np.ndarray:
        tt = np.array(t, dtype=float)
        return np.interp(tt, np.array(keys, float), values, left=0.0, right=0.0)

    return shift


def curve_date_for(index: pd.Index, d: pd.Timestamp) -> pd.Timestamp | None:
    """Return the most recent curve date <= d (like your notebook)."""
    d = pd.Timestamp(d)
    if d in index:
        return d
    pos = index.searchsorted(d, side="right") - 1
    if pos < 0:
        return None
    return pd.Timestamp(index[pos])


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


def book_parallel_risk_timeseries(
    book: IssuanceBook,
    curves_for_dates: dict[pd.Timestamp, dict[str, Curve]],
    *,
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """
    Compute PV01 and convexity time series with a symmetric parallel bump.
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

            pv01 = (pv_dn - pv_up) / 2.0
            convexity = (pv_up + pv_dn - 2.0 * pv0) / (pv0 * (bump**2)) if pv0 != 0 else np.nan
            risk_records.append({"date": vd, "method": method, "pv01": pv01, "convexity": convexity})

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
    book: IssuanceBook,
    curves_for_dates: dict[pd.Timestamp, dict[str, Curve]],
    *,
    keys: list[int] | tuple[int, ...] | None = None,
    bump_bp: float = 1.0,
) -> pd.DataFrame:
    """
    Compute key-rate duration time series by method and key tenor.
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


def make_book_metrics(
    total_pv: pd.DataFrame,
    bucket_pv: pd.DataFrame,
    risk: pd.DataFrame,
) -> BookMetrics:
    return BookMetrics(total_pv=total_pv, bucket_pv=bucket_pv, risk=risk)


def book_metrics(
    book: IssuanceBook,
    valuation_dates: pd.Index | list[pd.Timestamp],
    par_yields: pd.DataFrame,
    *,
    methods: Iterable[str] = ("loglinear", "pchip", "nss", "qp"),
    holdouts: list[str] | None = None,
    freq: int = 2,
    short_end: Literal["continuous", "simple"] = "continuous",
    min_df: float = 1e-12,
    bump_bp: float = 1.0,
    tenor_cols: list[str] | None = None,
) -> tuple[BookMetrics, pd.DataFrame]:
    """
    Full Project-1 book engine:
    - build curves for each valuation date (using latest available par curve <= date)
    - compute PV total + bucket PV
    - compute PV01, convexity
    - compute KRD for each key (book.maturities)
    Returns:
      BookMetrics(total_pv, bucket_pv, risk), krd_df (index=date, columns MultiIndex(method,key))
    """
    _ = holdouts  # retained for backward-compatible signature
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
    risk = book_parallel_risk_timeseries(book, curves_for_dates, bump_bp=bump_bp)
    krd_df = book_krd_timeseries(book, curves_for_dates, keys=book.maturities, bump_bp=bump_bp)
    return make_book_metrics(total_pv, bucket_pv, risk), krd_df


# ----------------------------
# Single-bond pricing + risk
# ----------------------------

def nearest_tenor_label(
    tenor_labels: list[str] | tuple[str, ...] | pd.Index,
    *,
    target_maturity_years: float,
) -> str:
    labels = [str(x) for x in tenor_labels]
    if not labels:
        raise InputError("tenor_labels is empty.")
    return min(labels, key=lambda c: abs(tenor_to_years(c) - float(target_maturity_years)))


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


def bond_price_and_risk(
    bond: Bond,
    curves: dict[str, Curve],
    *,
    bump_bp: float = 1.0,
    key_tenors: list[int] | tuple[int, ...] | None = None,
    settle: float = 0.0,
) -> pd.DataFrame:
    if key_tenors is None:
        key_tenors = list(DEFAULT_ISSUE_MATURITIES)
    bump = bump_bp / 10000.0
    rows = []
    for method, curve in curves.items():
        p0 = bond_price(bond, curve, settle=settle, clean=True)
        df0 = curve.df

        up = shifted_df_func(df0, lambda t: np.full_like(np.array(t, float), +bump))
        dn = shifted_df_func(df0, lambda t: np.full_like(np.array(t, float), -bump))

        # reuse cashflows; shift curve df
        times, cfs = bond_cashflows(bond.coupon, bond.maturity_years, freq=bond.freq, face=bond.face)
        pv_up = price_bond_from_issue(up, times, cfs, age=settle)
        pv_dn = price_bond_from_issue(dn, times, cfs, age=settle)

        pv01 = (pv_dn - pv_up) / 2.0
        convexity = (pv_up + pv_dn - 2.0 * (p0 + bond.coupon*bond.face*settle)) / ((p0 + bond.coupon*bond.face*settle) * (bump**2)) if p0 != 0 else np.nan

        # KRD by piecewise key shift
        krd_vals = {}
        for k in key_tenors:
            shift = key_bump_func(key_tenors, k, bump_bp=bump_bp)
            df_b = shifted_df_func(df0, shift)
            pv_b = price_bond_from_issue(df_b, times, cfs, age=settle)
            krd_vals[f"krd_{k}Y"] = ((p0 + bond.coupon*bond.face*settle) - pv_b) / bump

        row = {"method": method, "clean_price": p0, "pv01": pv01, "convexity": convexity, **krd_vals}
        rows.append(row)

    return pd.DataFrame(rows).set_index("method").sort_index()


# ----------------------------
# Topic-level orchestration helpers
# ----------------------------

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


def normalize_methods(methods: Iterable[str] = DEFAULT_METHODS) -> list[str]:
    vals = [m.lower().strip() for m in methods]
    if not vals:
        raise InputError("At least one curve method is required.")
    return list(dict.fromkeys(vals))


def sort_rmse_table(rmse: pd.DataFrame, *, methods: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Keep a stable method order (if provided) and sort by in-sample RMSE.
    """
    out = rmse.copy()
    if methods is not None:
        ordered = [m for m in normalize_methods(methods) if m in out.index]
        out = out.loc[ordered]
    if "rmse" in out.columns:
        out = out.sort_values("rmse")
    return out
