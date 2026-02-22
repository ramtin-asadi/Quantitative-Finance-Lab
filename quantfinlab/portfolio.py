from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .core import BacktestResult, InputError, ModelError, PortfolioState, StrategyBuildResult

try:  # optional dependency for shrinkage covariance estimators
    from sklearn.covariance import OAS, LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None
    OAS = None

DEFAULT_SOLVER_ORDER = ("OSQP", "ECOS", "SCS")
DEFAULT_ANNUALIZATION = 252.0


def _to_datetime_index(index_like: pd.Index | Sequence[pd.Timestamp | str]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index_like)
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        raise InputError("Date index is empty.")
    return idx.sort_values().unique()


def _sanitize_frame(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise InputError(f"{name} is empty.")
    out = df.copy()
    out = out.sort_index()
    out.index = pd.to_datetime(out.index)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _resolve_date_on_or_before(index: pd.DatetimeIndex, dt: pd.Timestamp | str) -> pd.Timestamp | None:
    d = pd.Timestamp(dt)
    if d in index:
        return d
    pos = int(index.searchsorted(d, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(index[pos])


def prices_to_returns(
    prices: pd.DataFrame,
    *,
    kind: Literal["simple", "log"] = "simple",
    drop_all_nan: bool = True,
    dtype: str | np.dtype = np.float64,
) -> pd.DataFrame:
    """
    Convert a price panel to daily returns.
    """
    px = _sanitize_frame(prices, name="prices")
    if kind == "simple":
        rets = px.pct_change(fill_method=None)
    elif kind == "log":
        rets = np.log(px / px.shift(1))
    else:
        raise InputError(f"Unsupported return kind: {kind!r}.")
    rets = rets.replace([np.inf, -np.inf], np.nan)
    if drop_all_nan:
        rets = rets.dropna(how="all")
    return rets.astype(dtype)


def make_rebalance_dates(
    index: pd.Index | Sequence[pd.Timestamp | str],
    *,
    freq: str = "M",
    min_history_days: int = 0,
) -> pd.DatetimeIndex:
    """
    Build rebalance dates from a trading-day index.
    """
    idx = _to_datetime_index(index)
    freq_norm = str(freq).upper().strip()
    freq_alias = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    grouped = pd.Series(idx, index=idx).groupby(pd.Grouper(freq=freq_alias.get(freq_norm, freq))).last().dropna()
    rebal = pd.DatetimeIndex(grouped.values).sort_values().unique()
    if min_history_days > 0:
        if min_history_days >= len(idx):
            raise InputError("min_history_days is larger than the index length.")
        rebal = rebal[rebal >= idx[int(min_history_days)]]
    return rebal


def _first_valid_date(close_prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.Series:
    first_close = close_prices.apply(pd.Series.first_valid_index)
    first_vol = volumes.apply(pd.Series.first_valid_index)
    return pd.concat([first_close, first_vol], axis=1).max(axis=1)


def select_liquid_universe(
    dt: pd.Timestamp | str,
    *,
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    top_n: int = 100,
    liq_lookback: int = 252,
    min_listing_days: int = 252,
    min_obs: int = 200,
    first_date: pd.Series | None = None,
) -> tuple[list[str], pd.Series]:
    """
    Select a liquidity-filtered stock universe at date `dt`.
    """
    if top_n <= 0:
        raise InputError("top_n must be positive.")
    if liq_lookback <= 0 or min_listing_days <= 0 or min_obs <= 0:
        raise InputError("liq_lookback, min_listing_days and min_obs must be positive.")

    cp = _sanitize_frame(close_prices, name="close_prices")
    vv = _sanitize_frame(volumes, name="volumes")

    common_idx = cp.index.intersection(vv.index)
    common_cols = cp.columns.intersection(vv.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        raise InputError("close_prices and volumes must overlap on index and columns.")
    cp = cp.loc[common_idx, common_cols]
    vv = vv.loc[common_idx, common_cols]

    idx = pd.DatetimeIndex(cp.index)
    d_eff = _resolve_date_on_or_before(idx, dt)
    if d_eff is None:
        return [], pd.Series(dtype=float)

    pos = int(idx.get_loc(d_eff))
    need = max(int(liq_lookback), int(min_listing_days))
    if pos < need:
        return [], pd.Series(dtype=float)

    fdate = first_date if first_date is not None else _first_valid_date(cp, vv)
    cutoff_date = idx[pos - int(min_listing_days)]
    seasoned = (fdate.notna()) & (fdate <= cutoff_date)
    seasoned = seasoned.reindex(cp.columns).fillna(False)
    cols = cp.columns[seasoned.values]
    if len(cols) == 0:
        return [], pd.Series(dtype=float)

    start = pos - int(liq_lookback)
    end = pos
    c = cp.iloc[start:end][cols]
    v = vv.iloc[start:end][cols]
    dv = c * v

    obs_ok = dv.notna().sum(axis=0) >= int(min_obs)
    pos_ok = (dv > 0).sum(axis=0) >= int(min_obs)
    selected = dv.columns[obs_ok & pos_ok]
    if len(selected) == 0:
        return [], pd.Series(dtype=float)

    adv = dv[selected].mean(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).dropna()
    if adv.empty:
        return [], pd.Series(dtype=float)
    top = adv.nlargest(min(int(top_n), len(adv)))
    return top.index.tolist(), top.astype(float)


def momentum_score_from_returns(
    ret_window: pd.DataFrame,
    *,
    mode: Literal["12-1", "6-1", "3-0"] = "6-1",
) -> np.ndarray:
    """
    Compute cross-sectional momentum scores from a return window.
    """
    R = _sanitize_frame(ret_window, name="ret_window").dropna(axis=0, how="any")
    T = len(R)
    if T < 20:
        raise InputError("ret_window has too few observations (<20).")

    if mode == "12-1":
        lookback, skip = 252, 21
    elif mode == "6-1":
        lookback, skip = 126, 21
    elif mode == "3-0":
        lookback, skip = 63, 0
    else:
        raise InputError(f"Unknown momentum mode: {mode!r}.")

    if T < lookback + skip + 5:
        lookback = min(lookback, max(20, T - skip - 1))

    use = R.iloc[-(lookback + skip) :]
    mom = use.iloc[:-skip] if skip > 0 else use
    score = ((1.0 + mom).prod(axis=0) - 1.0).to_numpy(dtype=float)
    return score


def winsorize_signal(x: np.ndarray, *, p: float = 0.05) -> np.ndarray:
    if not (0 <= p < 0.5):
        raise InputError("winsor percentile p must satisfy 0 <= p < 0.5.")
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        raise InputError("Signal is empty.")
    lo, hi = np.quantile(arr, [p, 1.0 - p])
    return np.clip(arr, lo, hi)


def zscore_signal(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    m = float(np.mean(arr))
    s = float(np.std(arr))
    if s <= eps:
        return np.zeros_like(arr)
    return (arr - m) / s


def make_psd(sigma: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """
    Project a square matrix to PSD space via eigenvalue flooring.
    """
    S = np.asarray(sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("sigma must be a square matrix.")
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, float(eps))
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def ewma_covariance(x: np.ndarray, *, lam: float = 0.94) -> np.ndarray:
    if not (0 < lam < 1):
        raise InputError("EWMA lambda must be in (0, 1).")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise InputError("x must be a 2D array.")
    T, n = arr.shape
    if T < 2 or n < 1:
        raise InputError("x must have shape (T, n) with T>=2.")
    arr = arr - arr.mean(axis=0, keepdims=True)
    S = np.zeros((n, n), dtype=float)
    a = 1.0 - lam
    for t in range(T):
        xt = arr[t][:, None]
        S = lam * S + a * (xt @ xt.T)
    scale = 1.0 - (lam ** max(T, 1))
    if scale > 1e-12:
        S = S / scale
    return S


def cov_estimate(
    window: pd.DataFrame,
    *,
    method: str = "ledoitwolf",
    annualization: float = DEFAULT_ANNUALIZATION,
    ewma_lambda: float = 0.94,
    psd: bool = True,
    ridge: float = 0.0,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Estimate annualized covariance from daily returns.
    """
    R = _sanitize_frame(window, name="window").dropna(axis=0, how="any")
    if R.shape[0] < 3 or R.shape[1] < 2:
        raise InputError("window must have at least 3 rows and 2 assets after cleaning.")
    x = R.to_numpy(dtype=float)

    m = method.strip().lower().replace(" ", "")
    aliases = {
        "sample": "samplecov",
        "samplecov": "samplecov",
        "lw": "ledoitwolf",
        "ledoitwolf": "ledoitwolf",
        "oas": "oas",
        "ewma": "ewma",
    }
    if m not in aliases:
        raise InputError(f"Unknown covariance method: {method!r}.")
    m = aliases[m]

    if m == "samplecov":
        cov_daily = np.cov(x, rowvar=False, ddof=1).astype(float)
    elif m in {"ledoitwolf", "oas"}:
        if LedoitWolf is None or OAS is None:
            raise ImportError("scikit-learn is required for method='ledoitwolf' or 'oas'.")
        if m == "ledoitwolf":
            cov_daily = LedoitWolf().fit(x).covariance_.astype(float)
        else:
            cov_daily = OAS().fit(x).covariance_.astype(float)
    else:
        cov_daily = ewma_covariance(x, lam=float(ewma_lambda))

    cov_ann = float(annualization) * cov_daily
    cov_ann = 0.5 * (cov_ann + cov_ann.T)
    if ridge > 0:
        cov_ann = cov_ann + float(ridge) * np.eye(cov_ann.shape[0])
    if psd:
        cov_ann = make_psd(cov_ann, eps=psd_eps)

    if return_df:
        return pd.DataFrame(cov_ann, index=R.columns, columns=R.columns)
    return cov_ann


def scale_mu_to_target_sharpe(
    mu_dir: np.ndarray,
    cov_ann: np.ndarray,
    *,
    target_sharpe_ann: float = 0.60,
    mu_cap_ann: float = 0.20,
) -> np.ndarray:
    mu = np.asarray(mu_dir, dtype=float).reshape(-1)
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_dir length.")
    if np.all(np.abs(mu) < 1e-12):
        return np.zeros_like(mu)
    A = S + 1e-8 * np.eye(S.shape[0])
    try:
        x = np.linalg.solve(A, mu)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, mu, rcond=None)[0]
    q = float(mu @ x)
    if (not np.isfinite(q)) or q <= 1e-18:
        return np.zeros_like(mu)
    scale = float(target_sharpe_ann) / math.sqrt(q)
    return np.clip(scale * mu, -float(mu_cap_ann), float(mu_cap_ann))


def mu_momentum(
    ret_window: pd.DataFrame,
    *,
    mode: Literal["12-1", "6-1", "3-0"] = "6-1",
    rf: float = 0.0,
    target_sharpe: float = 0.60,
    mu_cap: float = 0.20,
    winsor: float | None = 0.05,
    zscore: bool = True,
    cov_for_scaling: np.ndarray | None = None,
    cov_method: str = "ledoitwolf",
    annualization: float = DEFAULT_ANNUALIZATION,
    return_series: bool = False,
) -> np.ndarray | pd.Series:
    """
    Build expected annual excess returns from momentum scores.
    """
    R = _sanitize_frame(ret_window, name="ret_window").dropna(axis=0, how="any")
    score = momentum_score_from_returns(R, mode=mode)

    if winsor is not None:
        score = winsorize_signal(score, p=float(winsor))
    if zscore:
        score = zscore_signal(score)

    cov_ann = cov_for_scaling
    if cov_ann is None:
        cov_ann = cov_estimate(R, method=cov_method, annualization=annualization, psd=True, ridge=1e-10)
    mu = scale_mu_to_target_sharpe(
        score,
        np.asarray(cov_ann, dtype=float),
        target_sharpe_ann=float(target_sharpe),
        mu_cap_ann=float(mu_cap),
    )

    # Keep rf argument explicit for user workflows. Excess signal is centered, so no direct rf shift by default.
    _ = rf

    if return_series:
        return pd.Series(mu, index=R.columns, dtype=float)
    return mu


def _safe_normalize_weights(
    w: np.ndarray,
    *,
    w_min: float | None = None,
    w_max: float | None = None,
    long_only: bool = True,
    n_rounds: int = 3,
) -> np.ndarray | None:
    arr = np.asarray(w, dtype=float).reshape(-1)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        return None
    out = arr.copy()
    for _ in range(max(int(n_rounds), 1)):
        if long_only:
            out = np.maximum(out, 0.0)
        if w_min is not None:
            out = np.maximum(out, float(w_min))
        if w_max is not None:
            out = np.minimum(out, float(w_max))
        s = float(out.sum())
        if (not np.isfinite(s)) or s <= 0:
            return None
        out = out / s
    return out


def _constraints_feasible(
    n_assets: int,
    *,
    w_min: float | None,
    w_max: float | None,
    long_only: bool,
) -> bool:
    w_min_eff = 0.0 if long_only else (-np.inf if w_min is None else float(w_min))
    w_max_eff = np.inf if w_max is None else float(w_max)
    if np.isfinite(w_max_eff) and w_max_eff * n_assets < 1.0 - 1e-12:
        return False
    if np.isfinite(w_min_eff) and w_min_eff * n_assets > 1.0 + 1e-12:
        return False
    return True


def _coerce_prev_weights(w_prev: np.ndarray | None, n_assets: int) -> np.ndarray:
    if w_prev is None:
        return np.ones(n_assets, dtype=float) / n_assets
    arr = np.asarray(w_prev, dtype=float).reshape(-1)
    if arr.size != n_assets:
        raise InputError("w_prev length must match number of assets.")
    out = _safe_normalize_weights(arr, long_only=False)
    if out is None:
        return np.ones(n_assets, dtype=float) / n_assets
    return out


def _normalize_solver_order(order: Sequence[str] | None) -> list[str]:
    if order is None:
        return list(DEFAULT_SOLVER_ORDER)
    vals = [str(x).strip().upper() for x in order if str(x).strip()]
    if not vals:
        return list(DEFAULT_SOLVER_ORDER)
    return list(dict.fromkeys(vals))


def _solve_with_solvers(prob, var, solver_order: Sequence[str]) -> np.ndarray | None:
    for solver in solver_order:
        try:
            kwargs: dict[str, Any] = {"warm_start": True}
            if solver == "OSQP":
                kwargs["max_iter"] = 8000
            elif solver == "SCS":
                kwargs["max_iters"] = 10000
            elif solver == "ECOS":
                kwargs["max_iters"] = 10000
            prob.solve(solver=solver, **kwargs)
            if var.value is None:
                continue
            w = np.asarray(var.value, dtype=float).reshape(-1)
            if np.all(np.isfinite(w)):
                return w
        except Exception:
            continue
    return None


def weights_equal(
    assets: Sequence[str] | int,
    *,
    w_min: float | None = None,
    w_max: float | None = None,
    long_only: bool = True,
    as_series: bool = False,
) -> np.ndarray | pd.Series:
    """
    Equal-weight portfolio under bound constraints.
    """
    if isinstance(assets, int):
        n_assets = int(assets)
        labels = [f"a{i}" for i in range(n_assets)]
    else:
        labels = [str(x) for x in assets]
        n_assets = len(labels)
    if n_assets <= 0:
        raise InputError("assets must contain at least one asset.")
    if not _constraints_feasible(n_assets, w_min=w_min, w_max=w_max, long_only=long_only):
        raise InputError("Constraints are infeasible for equal weights.")
    w = np.ones(n_assets, dtype=float) / n_assets
    wn = _safe_normalize_weights(w, w_min=w_min, w_max=w_max, long_only=long_only)
    if wn is None:
        raise ModelError("Failed to normalize equal weights under constraints.")
    if as_series:
        return pd.Series(wn, index=labels, dtype=float)
    return wn


def weights_minvar(
    *,
    cov_ann: np.ndarray,
    w_prev: np.ndarray | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """
    Minimum-variance portfolio with turnover and ridge penalties.
    """
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("cov_ann must be a square matrix.")
    n = S.shape[0]
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if not _constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None

    wprev = _coerce_prev_weights(w_prev, n)
    kappa = (
        float(kappa_target_annual)
        if kappa_target_annual is not None
        else float(turnover_penalty_bps) / 10000.0
    )

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    w = cp.Variable(n)
    S_psd = cp.psd_wrap(make_psd(S, eps=1e-12))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))
    obj = cp.Minimize(
        cp.quad_form(w, S_psd)
        + 0.5 * kappa * cp.norm1(w - wprev)
        + 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    sol = _solve_with_solvers(prob, w, _normalize_solver_order(solver_order))
    if sol is None:
        if raise_on_fail:
            raise ModelError("MinVar solver failed to produce a feasible solution.")
        return None
    return _safe_normalize_weights(sol, w_min=w_min, w_max=w_max, long_only=long_only)


def weights_mv(
    *,
    mu_excess_ann: np.ndarray,
    cov_ann: np.ndarray,
    w_prev: np.ndarray | None = None,
    mv_lambda: float = 6.0,
    kappa_target_annual: float = 0.20,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """
    Mean-variance utility optimizer with L1 turnover and ridge penalties.
    """
    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = S.shape[0]
    if not _constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None

    wprev = _coerce_prev_weights(w_prev, n)
    base_kappa = float(turnover_penalty_bps) / 10000.0
    kappa = max(base_kappa, float(kappa_target_annual))

    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    w = cp.Variable(n)
    S_psd = cp.psd_wrap(make_psd(S, eps=1e-12))
    cons = [cp.sum(w) == 1]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))

    obj = cp.Maximize(
        mu @ w
        - 0.5 * float(mv_lambda) * cp.quad_form(w, S_psd)
        - 0.5 * kappa * cp.norm1(w - wprev)
        - 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    sol = _solve_with_solvers(prob, w, _normalize_solver_order(solver_order))
    if sol is None:
        if raise_on_fail:
            raise ModelError("MV solver failed to produce a feasible solution.")
        return None
    return _safe_normalize_weights(sol, w_min=w_min, w_max=w_max, long_only=long_only)


def weights_ridge_mv(
    *,
    mu_excess_ann: np.ndarray,
    cov_ann: np.ndarray,
    w_prev: np.ndarray | None = None,
    ridge: float = 1e-4,
    mv_lambda: float = 6.0,
    kappa_target_annual: float = 0.20,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """
    Ridge-regularized MV optimizer.
    """
    return weights_mv(
        mu_excess_ann=mu_excess_ann,
        cov_ann=cov_ann,
        w_prev=w_prev,
        mv_lambda=mv_lambda,
        kappa_target_annual=kappa_target_annual,
        w_min=w_min,
        w_max=w_max,
        long_only=long_only,
        turnover_penalty_bps=turnover_penalty_bps,
        ridge=float(ridge),
        solver_order=solver_order,
        raise_on_fail=raise_on_fail,
    )


def weights_maxsharpe_slsqp(
    *,
    mu_excess_ann: np.ndarray,
    cov_ann: np.ndarray,
    w_prev: np.ndarray | None = None,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    maxiter: int = 8000,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """
    Max-Sharpe optimization via SLSQP with turnover and ridge penalties.
    """
    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = len(mu)
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if not _constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover
        raise ImportError("scipy is required for SLSQP max-Sharpe optimization.") from exc

    wprev = _coerce_prev_weights(w_prev, n)
    cov_psd = make_psd(S, eps=1e-12)
    kappa = (
        float(kappa_target_annual)
        if kappa_target_annual is not None
        else float(turnover_penalty_bps) / 10000.0
    )

    lo = 0.0 if long_only else (-1.0 if w_min is None else float(w_min))
    hi = 1.0 if w_max is None else float(w_max)
    bounds = [(lo, hi) for _ in range(n)]
    x0 = _safe_normalize_weights(wprev, w_min=w_min, w_max=w_max, long_only=long_only)
    if x0 is None:
        x0 = np.ones(n, dtype=float) / n

    def obj(w: np.ndarray) -> float:
        ww = np.asarray(w, dtype=float).reshape(-1)
        if ww.size != n or np.any(~np.isfinite(ww)):
            return 1e12
        ret = float(mu @ ww)
        var = float(ww @ cov_psd @ ww)
        vol = math.sqrt(max(var, 1e-18))
        if vol <= 1e-12:
            return 1e12
        penalty = 0.5 * kappa * float(np.sum(np.abs(ww - wprev))) + 0.5 * float(ridge) * float(np.sum(ww**2))
        return -(ret / vol) + penalty

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},),
        options={"maxiter": int(maxiter)},
    )
    if (not bool(res.success)) or res.x is None:
        if raise_on_fail:
            raise ModelError(f"SLSQP max-Sharpe failed: {getattr(res, 'message', 'unknown error')}.")
        return None
    wn = _safe_normalize_weights(
        np.asarray(res.x, dtype=float),
        w_min=w_min,
        w_max=w_max,
        long_only=long_only,
    )
    if wn is None and raise_on_fail:
        raise ModelError("SLSQP max-Sharpe returned invalid/infeasible weights.")
    return wn


def _sharpe_from_weights(mu: np.ndarray, cov: np.ndarray, w: np.ndarray) -> float:
    ww = np.asarray(w, dtype=float).reshape(-1)
    r = float(np.dot(mu, ww))
    v2 = float(ww @ cov @ ww)
    if not np.isfinite(v2) or v2 <= 1e-18:
        return -np.inf
    return r / math.sqrt(v2)


def _greedy_max_return_weight(
    mu: np.ndarray,
    *,
    w_max: float | None,
    w_min: float | None,
    long_only: bool,
) -> np.ndarray | None:
    if not long_only:
        return None
    n = len(mu)
    cap = np.inf if w_max is None else float(w_max)
    floor = 0.0 if w_min is None else float(w_min)
    if cap <= 0:
        return None
    order = np.argsort(mu)[::-1]
    w = np.full(n, floor, dtype=float)
    rem = 1.0 - float(np.sum(w))
    if rem < -1e-10:
        return None
    rem = max(rem, 0.0)
    for i in order:
        if rem <= 1e-12:
            break
        room = cap - w[i]
        if room <= 0:
            continue
        add = min(room, rem)
        w[i] += add
        rem -= add
    if rem > 1e-8:
        return None
    return w


def weights_maxsharpe_frontier_grid(
    *,
    mu_excess_ann: np.ndarray,
    cov_ann: np.ndarray,
    w_prev: np.ndarray | None = None,
    grid_n: int = 25,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    turnover_penalty_bps: float = 10.0,
    kappa_target_annual: float | None = None,
    ridge: float = 1e-8,
    solver_order: Sequence[str] | None = None,
    raise_on_fail: bool = False,
) -> np.ndarray | None:
    """
    Approximate max-Sharpe portfolio by scanning target-return frontier points.
    """
    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    S = np.asarray(cov_ann, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1] or S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_excess_ann length.")
    n = len(mu)
    if n < 2:
        raise InputError("Need at least two assets for optimization.")
    if grid_n < 2:
        raise InputError("grid_n must be at least 2.")
    if not _constraints_feasible(n, w_min=w_min, w_max=w_max, long_only=long_only):
        if raise_on_fail:
            raise ModelError("Constraint set is infeasible.")
        return None

    wprev = _coerce_prev_weights(w_prev, n)
    w_minv = weights_minvar(
        cov_ann=S,
        w_prev=wprev,
        w_min=w_min,
        w_max=w_max,
        long_only=long_only,
        turnover_penalty_bps=turnover_penalty_bps,
        kappa_target_annual=kappa_target_annual,
        ridge=ridge,
        solver_order=solver_order,
        raise_on_fail=False,
    )
    if w_minv is None:
        w_minv = np.ones(n, dtype=float) / n

    w_maxr = _greedy_max_return_weight(mu, w_max=w_max, w_min=w_min, long_only=long_only)
    if w_maxr is None:
        if raise_on_fail:
            raise ModelError("Could not build max-return anchor under constraints.")
        return None

    r_lo = float(np.dot(mu, w_minv))
    r_hi = float(np.dot(mu, w_maxr))
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo + 1e-12:
        if raise_on_fail:
            raise ModelError("Degenerate frontier return range.")
        return None
    targets = np.linspace(r_lo, r_hi, int(grid_n))

    kappa = (
        float(kappa_target_annual)
        if kappa_target_annual is not None
        else float(turnover_penalty_bps) / 10000.0
    )
    try:
        import cvxpy as cp
    except Exception as exc:  # pragma: no cover
        raise ImportError("cvxpy is required for portfolio optimization.") from exc

    w = cp.Variable(n)
    r_target = cp.Parameter()
    S_psd = cp.psd_wrap(make_psd(S, eps=1e-12))
    cons = [cp.sum(w) == 1, mu @ w >= r_target]
    if long_only:
        cons.append(w >= 0)
    if w_min is not None:
        cons.append(w >= float(w_min))
    if w_max is not None:
        cons.append(w <= float(w_max))
    obj = cp.Minimize(
        cp.quad_form(w, S_psd)
        + kappa * cp.norm1(w - wprev)
        + 0.5 * float(ridge) * cp.sum_squares(w)
    )
    prob = cp.Problem(obj, cons)
    solver_list = _normalize_solver_order(solver_order)

    best_w: np.ndarray | None = None
    best_s = -np.inf
    for rt in targets:
        r_target.value = float(rt)
        sol = _solve_with_solvers(prob, w, solver_list)
        if sol is None:
            continue
        wn = _safe_normalize_weights(sol, w_min=w_min, w_max=w_max, long_only=long_only)
        if wn is None:
            continue
        sh = _sharpe_from_weights(mu, S, wn)
        if sh > best_s:
            best_s = sh
            best_w = wn
    if best_w is None and raise_on_fail:
        raise ModelError("Frontier-grid max-Sharpe solver did not find a feasible solution.")
    return best_w


def _state_as_mapping(state: Any) -> Mapping[str, Any]:
    if isinstance(state, PortfolioState):
        return state.as_dict()
    if isinstance(state, Mapping):
        return state
    raise InputError("Cache state must be a dict-like object or PortfolioState.")


def _weights_to_series(weights: np.ndarray | pd.Series | Sequence[float], tickers: Sequence[str]) -> pd.Series:
    labels = [str(x) for x in tickers]
    if isinstance(weights, pd.Series):
        w = weights.reindex(labels).fillna(0.0).astype(float)
    else:
        arr = np.asarray(weights, dtype=float).reshape(-1)
        if arr.size != len(labels):
            raise InputError("Weight vector length must match number of tickers.")
        w = pd.Series(arr, index=labels, dtype=float)
    return w


def backtest(
    returns: pd.DataFrame,
    rebal_dates: Sequence[pd.Timestamp | str],
    cache: Mapping[pd.Timestamp | str, Any],
    weight_fn: Callable[[pd.Timestamp, Mapping[str, Any], np.ndarray], np.ndarray | pd.Series | None],
    *,
    cost_bps: float = 10.0,
    fixed_fee: float = 0.0,
    fallback: Literal["equal", "previous", "none"] = "equal",
    blend: float = 0.0,
    w_min: float | None = 0.0,
    w_max: float | None = 0.25,
    long_only: bool = True,
    initial_value: float = 1.0,
    rf_daily: float = 0.0,
) -> BacktestResult:
    """
    Run a daily drift backtest with periodic rebalancing and trading costs.
    """
    if initial_value <= 0:
        raise InputError("initial_value must be positive.")

    R = _sanitize_frame(returns, name="returns").fillna(0.0)
    idx = pd.DatetimeIndex(R.index)
    if len(idx) == 0:
        raise InputError("returns index is empty.")
    rebal = pd.DatetimeIndex(pd.to_datetime(list(rebal_dates))).sort_values().unique()
    if len(rebal) == 0:
        raise InputError("rebal_dates is empty.")

    cache_norm: dict[pd.Timestamp, Any] = {pd.Timestamp(k): v for k, v in cache.items()}
    rebal = rebal[rebal.isin(pd.DatetimeIndex(cache_norm.keys()))]
    if len(rebal) == 0:
        raise InputError("No rebalance dates remain after intersecting with cache keys.")

    all_dates = idx[idx >= rebal[0]]
    rebal_set = set(rebal)

    gross_value = float(initial_value)
    net_value = float(initial_value)
    w = pd.Series(dtype=float)

    gross_values: list[float] = []
    net_values: list[float] = []
    gross_returns: list[float] = []
    weight_records: dict[pd.Timestamp, pd.Series] = {}
    turnover_vals: list[float] = []
    cost_vals: list[float] = []
    fallback_count = 0

    blend_eff = float(np.clip(blend, 0.0, 1.0))

    for dt in all_dates:
        if dt in rebal_set:
            state_raw = cache_norm[pd.Timestamp(dt)]
            state = _state_as_mapping(state_raw)

            tickers = [str(x) for x in state.get("tickers", [])]
            if len(tickers) >= 1:
                w_pre = w.reindex(tickers).fillna(0.0).astype(float)
                if float(w_pre.sum()) > 0:
                    w_pre = w_pre / float(w_pre.sum())
                else:
                    w_pre = pd.Series(np.ones(len(tickers), dtype=float) / len(tickers), index=tickers)

                w_tar_raw = None
                try:
                    w_tar_raw = weight_fn(pd.Timestamp(dt), state, w_pre.to_numpy(dtype=float))
                except Exception:
                    w_tar_raw = None

                if w_tar_raw is None:
                    fallback_count += 1
                    if fallback == "equal":
                        w_tar = pd.Series(
                            weights_equal(tickers, w_min=w_min, w_max=w_max, long_only=long_only),
                            index=tickers,
                            dtype=float,
                        )
                    elif fallback == "previous":
                        w_tar = w_pre.copy()
                    else:
                        w_tar = pd.Series(dtype=float)
                else:
                    try:
                        w_tar = _weights_to_series(w_tar_raw, tickers)
                    except Exception:
                        fallback_count += 1
                        if fallback == "equal":
                            w_tar = pd.Series(
                                weights_equal(tickers, w_min=w_min, w_max=w_max, long_only=long_only),
                                index=tickers,
                                dtype=float,
                            )
                        elif fallback == "previous":
                            w_tar = w_pre.copy()
                        else:
                            w_tar = pd.Series(dtype=float)

                if not w_tar.empty and blend_eff > 0:
                    w_tar = pd.Series(
                        (1.0 - blend_eff) * w_tar.to_numpy(dtype=float)
                        + blend_eff * w_pre.to_numpy(dtype=float),
                        index=tickers,
                        dtype=float,
                    )
                if not w_tar.empty:
                    wn = _safe_normalize_weights(
                        w_tar.to_numpy(dtype=float),
                        w_min=w_min,
                        w_max=w_max,
                        long_only=long_only,
                    )
                    if wn is None:
                        fallback_count += 1
                        w_tar = pd.Series(
                            np.ones(len(tickers), dtype=float) / len(tickers),
                            index=tickers,
                            dtype=float,
                        )
                    else:
                        w_tar = pd.Series(wn, index=tickers, dtype=float)

                if w_tar.empty:
                    turnover = 0.0
                    cost_value = 0.0
                else:
                    delta = w_tar.to_numpy(dtype=float) - w_pre.to_numpy(dtype=float)
                    turnover = 0.5 * float(np.sum(np.abs(delta)))
                    cost_rate = float(cost_bps) / 10000.0 * turnover
                    cost_value = float(net_value) * cost_rate
                    if fixed_fee > 0:
                        cost_value += float(fixed_fee) * float(np.count_nonzero(np.abs(delta) > 1e-12))
                    net_value = max(net_value - cost_value, 1e-12)
                    w = w_tar.copy()
                    weight_records[pd.Timestamp(dt)] = w_tar.astype(float)

                turnover_vals.append(turnover)
                cost_vals.append(cost_value)

        if w.empty:
            port_ret = 0.0
            w_next = pd.Series(dtype=float)
        else:
            r_today = R.loc[dt].reindex(w.index).fillna(0.0).astype(float)
            port_ret = float(np.dot(w.to_numpy(dtype=float), r_today.to_numpy(dtype=float)))
            grossed = w.to_numpy(dtype=float) * (1.0 + r_today.to_numpy(dtype=float))
            gross_sum = float(np.sum(grossed))
            if gross_sum > 0 and np.isfinite(gross_sum):
                w_next = pd.Series(grossed / gross_sum, index=w.index, dtype=float)
            else:
                w_next = pd.Series(dtype=float)

        gross_value *= 1.0 + port_ret
        net_value *= 1.0 + port_ret

        gross_values.append(float(gross_value))
        net_values.append(float(net_value))
        gross_returns.append(float(port_ret))
        w = w_next

    gross_values_s = pd.Series(gross_values, index=all_dates, name="gross_values")
    net_values_s = pd.Series(net_values, index=all_dates, name="net_values")
    gross_returns_s = pd.Series(gross_returns, index=all_dates, name="gross_returns")
    net_returns_s = net_values_s.pct_change().fillna(0.0)
    weights_df = pd.DataFrame.from_dict(weight_records, orient="index").fillna(0.0)
    turnover_s = (
        pd.Series(turnover_vals, index=weights_df.index, name="turnover")
        if len(weights_df)
        else pd.Series([], dtype=float, name="turnover")
    )
    costs_s = (
        pd.Series(cost_vals, index=weights_df.index, name="costs")
        if len(weights_df)
        else pd.Series([], dtype=float, name="costs")
    )

    return BacktestResult(
        gross_values=gross_values_s,
        net_values=net_values_s,
        gross_returns=gross_returns_s,
        net_returns=net_returns_s,
        weights=weights_df,
        turnover=turnover_s,
        costs=costs_s,
        fallbacks=int(fallback_count),
        metadata={
            "rf_daily": float(rf_daily),
            "cost_bps": float(cost_bps),
            "fixed_fee": float(fixed_fee),
            "blend": float(blend_eff),
        },
    )


def _clean_close_volume_panels(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2016-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cp = close_prices.copy()
    vv = volumes.copy()

    cp.columns = [str(c).strip() for c in cp.columns]
    vv.columns = [str(c).strip() for c in vv.columns]
    if cp.columns.duplicated().any():
        cp = cp.T.groupby(level=0).last().T
    if vv.columns.duplicated().any():
        vv = vv.T.groupby(level=0).last().T

    cp = _sanitize_frame(cp, name="close_prices")
    vv = _sanitize_frame(vv, name="volumes")
    cp = cp.where(cp > 0)
    vv = vv.where(vv >= 0)

    cp = cp[~cp.index.isna()].sort_index()
    vv = vv[~vv.index.isna()].sort_index()
    if cp.index.has_duplicates:
        cp = cp[~cp.index.duplicated(keep="last")]
    if vv.index.has_duplicates:
        vv = vv[~vv.index.duplicated(keep="last")]

    start_ts = pd.Timestamp(start)
    cp = cp.loc[cp.index >= start_ts]
    vv = vv.loc[vv.index >= start_ts]

    idx = cp.index.intersection(vv.index)
    cols = cp.columns.intersection(vv.columns)
    cp = cp.loc[idx, cols]
    vv = vv.loc[idx, cols]

    valid_cols = cp.notna().any(axis=0) & vv.notna().any(axis=0)
    cp = cp.loc[:, valid_cols]
    vv = vv.loc[:, valid_cols]

    # Deterministic column order improves reproducibility in tie cases.
    sorted_cols = sorted([str(c) for c in cp.columns])
    cp = cp.reindex(columns=sorted_cols)
    vv = vv.reindex(columns=sorted_cols)

    if cp.empty or vv.empty or cp.shape[1] < 2:
        raise InputError("Not enough valid assets after close/volume cleaning.")
    return cp.astype(float), vv.astype(float)


def build_all_portfolio_strategies(
    close_prices: pd.DataFrame,
    volumes: pd.DataFrame,
    *,
    start: str | pd.Timestamp = "2016-01-01",
    rf_annual: float = 0.04,
    rf_daily: float | None = None,
    annualization: float = DEFAULT_ANNUALIZATION,
    lookback_days: int = 252,
    universe_top_n: int = 100,
    liq_lookback: int = 252,
    min_listing_days: int = 252,
    min_obs: int = 252,
    min_window_rows: int = 251,
    cost_bps: float = 10.0,
    fallback: Literal["equal", "previous", "none"] = "equal",
    solver_order: Sequence[str] | None = None,
) -> StrategyBuildResult:
    """
    Build full Project-02 strategy set from close/volume panels.

    Returns cleaned data, state cache, and backtest results for:
    EW, MinVar (Sample/LW/OAS/EWMA), MV (Sample/LW/OAS/EWMA), Ridge-MV,
    MaxSharpe-SLSQP, MaxSharpe-Frontier.
    """
    ann = float(annualization)
    if ann <= 0:
        raise InputError("annualization must be positive.")
    rf_d = float(rf_daily) if rf_daily is not None else float((1.0 + float(rf_annual)) ** (1.0 / ann) - 1.0)

    prices, vols = _clean_close_volume_panels(close_prices, volumes, start=start)
    returns = prices_to_returns(prices)
    if returns.empty:
        raise InputError("returns is empty after cleaning.")

    rebal_dates = make_rebalance_dates(returns.index, freq="ME", min_history_days=int(lookback_days))
    first_date = _first_valid_date(prices, vols)
    idx_prices = pd.DatetimeIndex(prices.index)

    cache: dict[pd.Timestamp, dict[str, Any]] = {}
    for dt in rebal_dates:
        tickers, avg_dv = select_liquid_universe(
            dt,
            close_prices=prices,
            volumes=vols,
            top_n=int(universe_top_n),
            liq_lookback=int(liq_lookback),
            min_listing_days=int(min_listing_days),
            min_obs=int(min_obs),
            first_date=first_date,
        )
        if len(tickers) < 2:
            continue

        d_eff = _resolve_date_on_or_before(idx_prices, pd.Timestamp(dt))
        if d_eff is None:
            continue
        pos = int(idx_prices.get_loc(d_eff))
        if pos < int(lookback_days):
            continue

        close_for_model = prices.iloc[pos - int(lookback_days) : pos][tickers]
        if close_for_model.shape[0] < int(lookback_days):
            continue

        window = close_for_model.pct_change(fill_method=None).iloc[1:]
        window = window.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if window.shape[0] < int(min_window_rows) or window.shape[1] < 2:
            continue

        tickers = [str(c) for c in window.columns]
        mu_excess_ann = mu_momentum(
            window,
            mode="6-1",
            rf=float(rf_annual),
            target_sharpe=0.80,
            mu_cap=0.30,
            winsor=0.05,
            zscore=True,
            return_series=True,
        ).reindex(tickers).astype(float)

        cov_ann_map = {
            "sample": cov_estimate(window, method="samplecov", psd=True, ridge=1e-10),
            "lw": cov_estimate(window, method="ledoitwolf", psd=True, ridge=1e-10),
            "oas": cov_estimate(window, method="oas", psd=True, ridge=1e-10),
            "ewma": cov_estimate(window, method="ewma", ewma_lambda=0.94, psd=True, ridge=1e-10),
        }
        cache[pd.Timestamp(d_eff)] = {
            "tickers": tickers,
            "mu_excess_ann": mu_excess_ann,
            "cov_ann_map": cov_ann_map,
            "avg_dollar_volume": avg_dv.reindex(tickers).astype(float),
            "window": window,
        }

    usable_rebal_dates = [pd.Timestamp(d) for d in rebal_dates if pd.Timestamp(d) in cache]
    if len(usable_rebal_dates) == 0:
        raise InputError("No valid rebalance dates remain after state construction.")

    solve_order = _normalize_solver_order(solver_order)

    def ew(dt, st, w_prev):
        return weights_equal(st["tickers"], w_min=0.0, w_max=0.20, long_only=True)

    def make_minvar(cov_key: str):
        def _fn(dt, st, w_prev):
            return weights_minvar(
                cov_ann=st["cov_ann_map"][cov_key],
                w_prev=w_prev,
                w_min=0.0,
                w_max=0.25,
                long_only=True,
                turnover_penalty_bps=10.0,
                solver_order=solve_order,
            )
        return _fn

    def make_mv(cov_key: str):
        def _fn(dt, st, w_prev):
            return weights_mv(
                mu_excess_ann=st["mu_excess_ann"].values,
                cov_ann=st["cov_ann_map"][cov_key],
                w_prev=w_prev,
                mv_lambda=4.0,
                kappa_target_annual=0.20,
                w_min=0.0,
                w_max=0.25,
                long_only=True,
                turnover_penalty_bps=10.0,
                solver_order=solve_order,
            )
        return _fn

    def ridge_mv_lw(dt, st, w_prev):
        return weights_ridge_mv(
            mu_excess_ann=st["mu_excess_ann"].values,
            cov_ann=st["cov_ann_map"]["lw"],
            w_prev=w_prev,
            ridge=1e-4,
            mv_lambda=6.0,
            kappa_target_annual=0.30,
            w_min=0.0,
            w_max=0.25,
            long_only=True,
            turnover_penalty_bps=10.0,
            solver_order=solve_order,
        )

    def maxsharpe_slsqp_lw(dt, st, w_prev):
        return weights_maxsharpe_slsqp(
            mu_excess_ann=st["mu_excess_ann"].values,
            cov_ann=st["cov_ann_map"]["lw"],
            w_prev=w_prev,
            w_min=0.0,
            w_max=0.25,
            long_only=True,
            turnover_penalty_bps=10.0,
            kappa_target_annual=0.30,
        )

    def maxsharpe_frontier_lw(dt, st, w_prev):
        return weights_maxsharpe_frontier_grid(
            mu_excess_ann=st["mu_excess_ann"].values,
            cov_ann=st["cov_ann_map"]["lw"],
            w_prev=w_prev,
            grid_n=25,
            w_min=0.0,
            w_max=0.25,
            long_only=True,
            turnover_penalty_bps=10.0,
            solver_order=solve_order,
        )

    strategy_fns = {
        "ew": ew,
        "minvar_sample": make_minvar("sample"),
        "minvar_lw": make_minvar("lw"),
        "minvar_oas": make_minvar("oas"),
        "minvar_ewma": make_minvar("ewma"),
        "mv_sample": make_mv("sample"),
        "mv_lw": make_mv("lw"),
        "mv_oas": make_mv("oas"),
        "mv_ewma": make_mv("ewma"),
        "ridge_mv": ridge_mv_lw,
        "maxsharpe_slsqp": maxsharpe_slsqp_lw,
        "maxsharpe_frontier": maxsharpe_frontier_lw,
    }
    cov_key_for_rc = {
        "ew": "lw",
        "minvar_sample": "sample",
        "minvar_lw": "lw",
        "minvar_oas": "oas",
        "minvar_ewma": "ewma",
        "mv_sample": "sample",
        "mv_lw": "lw",
        "mv_oas": "oas",
        "mv_ewma": "ewma",
        "ridge_mv": "lw",
        "maxsharpe_slsqp": "lw",
        "maxsharpe_frontier": "lw",
    }

    results = {
        name: backtest(
            returns=returns,
            rebal_dates=usable_rebal_dates,
            cache=cache,
            weight_fn=fn,
            cost_bps=float(cost_bps),
            fallback=fallback,
            rf_daily=rf_d,
        )
        for name, fn in strategy_fns.items()
    }

    return StrategyBuildResult(
        prices=prices,
        volumes=vols,
        returns=returns,
        rebal_dates=usable_rebal_dates,
        cache=cache,
        results=results,
        cov_key_for_rc=cov_key_for_rc,
        metadata={
            "rf_annual": float(rf_annual),
            "rf_daily": rf_d,
            "annualization": ann,
            "lookback_days": int(lookback_days),
            "universe_top_n": int(universe_top_n),
            "n_strategies": len(strategy_fns),
        },
    )


def calc_drawdown(series: pd.Series) -> pd.Series:
    s = pd.Series(series, copy=False).astype(float)
    if s.empty:
        return s
    return s / s.cummax() - 1.0


def performance_metrics(
    net_returns: pd.Series,
    net_values: pd.Series,
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, float]:
    r = pd.Series(net_returns, copy=False).dropna().astype(float)
    v = pd.Series(net_values, copy=False).dropna().astype(float)
    if v.empty:
        return {
            "CAGR": np.nan,
            "AnnVol": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "Calmar": np.nan,
            "Sortino": np.nan,
        }
    years = len(r) / float(annualization) if len(r) > 0 else np.nan
    cagr = float(v.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(float(annualization))) if len(r) > 1 else np.nan
    ex = r - float(rf_daily)
    sharpe = (
        float(ex.mean() / r.std(ddof=1) * math.sqrt(float(annualization)))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan
    )
    dd = calc_drawdown(v)
    max_dd = float(dd.min()) if not dd.empty else np.nan
    calmar = float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else np.nan
    downside = r[r < 0]
    sortino = (
        float(ex.mean() / downside.std(ddof=1) * math.sqrt(float(annualization)))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else np.nan
    )
    return {
        "CAGR": cagr,
        "AnnVol": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "Sortino": sortino,
    }


def _as_result_obj(x: BacktestResult | Mapping[str, Any]) -> BacktestResult:
    if isinstance(x, BacktestResult):
        return x
    req = ["gross_values", "net_values", "gross_returns", "net_returns", "weights", "turnover", "costs"]
    missing = [k for k in req if k not in x]
    if missing:
        raise InputError(f"Result mapping is missing keys: {missing}")
    return BacktestResult(
        gross_values=pd.Series(x["gross_values"]),
        net_values=pd.Series(x["net_values"]),
        gross_returns=pd.Series(x["gross_returns"]),
        net_returns=pd.Series(x["net_returns"]),
        weights=pd.DataFrame(x["weights"]),
        turnover=pd.Series(x["turnover"]),
        costs=pd.Series(x["costs"]),
        fallbacks=int(x.get("fallbacks", 0)),
        metadata=dict(x.get("metadata", {})) if "metadata" in x else None,
    )


def result_sharpe(
    result: BacktestResult | Mapping[str, Any],
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    min_obs: int = 50,
) -> float:
    res = _as_result_obj(result)
    r = res.net_returns.dropna().astype(float)
    if len(r) < int(min_obs):
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    ex = r - float(rf_daily)
    return float(ex.mean() / sd * math.sqrt(float(annualization)))


def summarize_results(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build risk/return and trading summary tables for multiple strategies.
    """
    metrics_rows: list[dict[str, float | str]] = []
    trade_rows: list[dict[str, float | str]] = []

    for name, raw in results.items():
        res = _as_result_obj(raw)
        m = performance_metrics(
            res.net_returns,
            res.net_values,
            rf_daily=rf_daily,
            annualization=annualization,
        )
        metrics_rows.append({"Strategy": str(name), **m})

        if not res.weights.empty:
            hhi = (res.weights.astype(float) ** 2).sum(axis=1)
            avg_hhi = float(hhi.mean())
            eff_n = float(1.0 / avg_hhi) if avg_hhi > 0 else np.nan
        else:
            avg_hhi = np.nan
            eff_n = np.nan

        final_value = float(res.net_values.iloc[-1]) if not res.net_values.empty else np.nan
        total_cost = float(res.costs.sum()) if not res.costs.empty else 0.0
        trade_rows.append(
            {
                "Strategy": str(name),
                "Avg Turnover": float(res.turnover.mean()) if not res.turnover.empty else 0.0,
                "Total Turnover": float(res.turnover.sum()) if not res.turnover.empty else 0.0,
                "Total Costs": total_cost,
                "Cost % Final Value": (total_cost / final_value) if final_value and final_value > 0 else np.nan,
                "Avg HHI": avg_hhi,
                "Effective N": eff_n,
                "Fallbacks": int(res.fallbacks),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows).set_index("Strategy").sort_index()
    trade_df = pd.DataFrame(trade_rows).set_index("Strategy").sort_index()
    return metrics_df, trade_df


def best_strategy_by_sharpe(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    min_obs: int = 50,
) -> tuple[str, dict[str, float]]:
    sharpes = {
        name: result_sharpe(
            res,
            rf_daily=rf_daily,
            annualization=annualization,
            min_obs=min_obs,
        )
        for name, res in results.items()
    }
    best = max(sharpes, key=lambda k: -np.inf if np.isnan(sharpes[k]) else sharpes[k])
    return str(best), sharpes


__all__ = [
    "DEFAULT_SOLVER_ORDER",
    "DEFAULT_ANNUALIZATION",
    "prices_to_returns",
    "make_rebalance_dates",
    "select_liquid_universe",
    "momentum_score_from_returns",
    "winsorize_signal",
    "zscore_signal",
    "make_psd",
    "ewma_covariance",
    "cov_estimate",
    "scale_mu_to_target_sharpe",
    "mu_momentum",
    "weights_equal",
    "weights_minvar",
    "weights_mv",
    "weights_ridge_mv",
    "weights_maxsharpe_slsqp",
    "weights_maxsharpe_frontier_grid",
    "build_all_portfolio_strategies",
    "backtest",
    "calc_drawdown",
    "performance_metrics",
    "result_sharpe",
    "summarize_results",
    "best_strategy_by_sharpe",
]
