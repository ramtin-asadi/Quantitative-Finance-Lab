from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError

DEFAULT_ANNUALIZATION = 252.0
MuModel = Literal["Momentum", "BayesStein", "BayesSteinMomentum"]


def _sanitize_returns(ret_window: pd.DataFrame, *, min_rows: int = 0) -> pd.DataFrame:
    if not isinstance(ret_window, pd.DataFrame):
        raise InputError("ret_window must be a pandas DataFrame.")
    if ret_window.empty:
        return ret_window.astype(float)
    R = ret_window.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()
    R = R.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    R = R.dropna(axis=0, how="any")
    if min_rows and R.shape[0] < int(min_rows):
        raise InputError(f"ret_window has fewer than {min_rows} clean observations.")
    return R


def _as_cov_array(cov_ann: np.ndarray | pd.DataFrame, index: pd.Index | None = None) -> np.ndarray:
    if isinstance(cov_ann, pd.DataFrame):
        cov = cov_ann.reindex(index=index, columns=index).to_numpy(dtype=float) if index is not None else cov_ann.to_numpy(dtype=float)
    else:
        cov = np.asarray(cov_ann, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise InputError("cov_ann must be a square matrix.")
    return 0.5 * (cov + cov.T)


def _solve_linear_stable(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _return_mu(mu: np.ndarray, info: dict[str, float | int | str], index, return_info: bool, return_series: bool):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    if return_series and index is not None:
        out = pd.Series(mu, index=index, dtype=float)
    else:
        out = mu.astype(float)
    return (out, info) if return_info else out


def momentum_score_from_returns(
    ret_window: pd.DataFrame,
    *,
    mode: Literal["12-1", "6-1", "3-0"] = "6-1",
) -> np.ndarray:
    """
    Compute Notebook 2 momentum scores.

    Short clean windows fall back to the historical mean, matching the notebook.
    """
    R = _sanitize_returns(ret_window)
    T = len(R)
    if T == 0:
        return np.zeros(ret_window.shape[1] if isinstance(ret_window, pd.DataFrame) else 0, dtype=float)
    if T < 80:
        return R.mean().to_numpy(dtype=float)

    if mode == "12-1":
        lookback, skip = 252, 21
    elif mode == "6-1":
        lookback, skip = 126, 21
    elif mode == "3-0":
        lookback, skip = 63, 0
    else:
        raise InputError(f"Unknown momentum mode: {mode!r}.")

    if lookback + skip + 5 > T:
        lookback = min(lookback, max(63, T - skip - 1))

    R_use = R.iloc[-(lookback + skip) :]
    R_mom = R_use.iloc[:-skip] if skip > 0 else R_use
    return ((1.0 + R_mom).prod(axis=0) - 1.0).to_numpy(dtype=float)


def winsorize_signal(
    x: np.ndarray,
    *,
    p: float | None = None,
    p_lo: float = 0.05,
    p_hi: float = 0.95,
) -> np.ndarray:
    """Winsorize a finite-filled cross-sectional signal."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    fill_value = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill_value)
    if p is not None:
        p_lo, p_hi = float(p), 1.0 - float(p)
    if not (0.0 <= p_lo <= p_hi <= 1.0):
        raise InputError("winsor quantiles must satisfy 0 <= p_lo <= p_hi <= 1.")
    lo, hi = np.quantile(arr, [p_lo, p_hi])
    return np.clip(arr, lo, hi)


def zscore_signal(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Center and standardize a cross-sectional signal."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    fill_value = float(np.nanmedian(arr[finite]))
    arr = np.where(finite, arr, fill_value)
    arr = arr - float(arr.mean())
    sd = float(arr.std())
    if sd < eps:
        return np.zeros_like(arr)
    return arr / sd


def winsorize_and_zscore(x: np.ndarray, p_lo: float = 0.05, p_hi: float = 0.95) -> np.ndarray:
    return zscore_signal(winsorize_signal(x, p_lo=p_lo, p_hi=p_hi))


def scale_mu_to_target_sharpe(
    mu_dir: np.ndarray,
    cov_ann: np.ndarray | pd.DataFrame,
    *,
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Scale a direction vector so its unconstrained Sharpe equals the target."""
    mu = np.asarray(mu_dir, dtype=float).reshape(-1)
    if mu.size == 0 or np.all(np.abs(mu) < 1e-12):
        return np.zeros_like(mu)
    S = _as_cov_array(cov_ann)
    if S.shape[0] != mu.shape[0]:
        raise InputError("cov_ann shape must match mu_dir length.")
    A = S + float(ridge) * np.eye(len(mu))
    x = _solve_linear_stable(A, mu)
    q = float(mu @ x)
    if (not np.isfinite(q)) or q <= 1e-18:
        return np.zeros_like(mu)
    scale = float(target_sharpe_ann) / np.sqrt(q)
    return np.clip(scale * mu, -float(mu_cap_ann), float(mu_cap_ann))


def build_scaled_mu_from_raw(
    raw_mu: np.ndarray,
    cov_ann: np.ndarray | pd.DataFrame,
    *,
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    winsor_lo: float = 0.05,
    winsor_hi: float = 0.95,
    ridge: float = 1e-8,
) -> np.ndarray:
    """The canonical Notebook 2 raw-signal -> winsor/zscore -> scaled μ path."""
    mu_dir = winsorize_and_zscore(raw_mu, p_lo=winsor_lo, p_hi=winsor_hi)
    return scale_mu_to_target_sharpe(
        mu_dir,
        cov_ann,
        target_sharpe_ann=target_sharpe_ann,
        mu_cap_ann=mu_cap_ann,
        ridge=ridge,
    )


def sample_mean_excess_ann_from_returns(
    ret_window: pd.DataFrame,
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> np.ndarray:
    R = _sanitize_returns(ret_window)
    if R.shape[0] == 0:
        return np.zeros(R.shape[1], dtype=float)
    mu_daily = R.mean(axis=0).to_numpy(dtype=float) - float(rf_daily)
    return float(annualization) * mu_daily


def _return_mu_with_phi(mu: np.ndarray, phi: float, return_phi: bool):
    mu = np.asarray(mu, dtype=float).reshape(-1)
    return (mu, float(phi)) if return_phi else mu


def bayes_stein_mean_excess_ann(
    ret_window: pd.DataFrame,
    *,
    rf_daily: float = 0.0,
    ann_factor: float = DEFAULT_ANNUALIZATION,
    ridge: float = 1e-8,
    return_phi: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Jorion-style Bayes-Stein shrinkage of historical excess means."""
    R = _sanitize_returns(ret_window)
    n_cols = ret_window.shape[1] if isinstance(ret_window, pd.DataFrame) else R.shape[1]
    if R.shape[0] == 0:
        return _return_mu_with_phi(np.zeros(n_cols, dtype=float), np.nan, return_phi)

    x = R.to_numpy(dtype=float) - float(rf_daily)
    M, N = x.shape
    mu_hat = float(ann_factor) * x.mean(axis=0)

    if M <= N + 2:
        return _return_mu_with_phi(mu_hat, np.nan, return_phi)

    xc = x - x.mean(axis=0, keepdims=True)
    denom = M - N - 2
    if denom <= 0:
        return _return_mu_with_phi(mu_hat, np.nan, return_phi)

    sigma_hat = float(ann_factor) * ((xc.T @ xc) / float(denom))
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T) + float(ridge) * np.eye(N)

    ones = np.ones(N, dtype=float)
    inv_one = _solve_linear_stable(sigma_hat, ones)
    den_gmv = float(ones @ inv_one)
    if (not np.isfinite(den_gmv)) or den_gmv <= 1e-12:
        return _return_mu_with_phi(mu_hat, np.nan, return_phi)

    w_gmv = inv_one / den_gmv
    mu_min = float(mu_hat @ w_gmv)
    delta = mu_hat - mu_min * ones
    inv_delta = _solve_linear_stable(sigma_hat, delta)
    q = float(delta @ inv_delta)
    if (not np.isfinite(q)) or q < 0:
        return _return_mu_with_phi(mu_hat, np.nan, return_phi)

    phi = (N + 2.0) / ((N + 2.0) + M * q)
    phi = float(np.clip(phi, 0.0, 1.0))
    mu_bs = (1.0 - phi) * mu_hat + phi * (mu_min * ones)
    return _return_mu_with_phi(mu_bs, phi, return_phi)


def _gmv_scalar_prior(mu_excess_ann: np.ndarray, cov_ann: np.ndarray, ridge: float = 1e-8) -> float:
    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    n = len(mu)
    ones = np.ones(n, dtype=float)
    A = np.asarray(cov_ann, dtype=float) + float(ridge) * np.eye(n)
    inv_one = _solve_linear_stable(A, ones)
    den = float(ones @ inv_one)
    if (not np.isfinite(den)) or abs(den) < 1e-12:
        return float(np.nanmean(mu))
    return float(mu @ (inv_one / den))


def bayes_stein_shrink_mu(
    mu_excess_ann: np.ndarray,
    cov_ann: np.ndarray | pd.DataFrame,
    *,
    sample_size: int | float,
    ridge: float = 1e-8,
    return_phi: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Bayes-Stein-style shrinkage for an already-scaled μ vector."""
    mu = np.asarray(mu_excess_ann, dtype=float).reshape(-1)
    n = len(mu)
    if n == 0:
        return _return_mu_with_phi(mu, np.nan, return_phi)
    S = _as_cov_array(cov_ann)
    prior = _gmv_scalar_prior(mu, S, ridge=ridge)
    target = np.full(n, prior, dtype=float)
    diff = mu - target
    A = S + float(ridge) * np.eye(n)
    q = float(diff @ _solve_linear_stable(A, diff))
    if (not np.isfinite(q)) or q < 0:
        q = 0.0
    T = max(float(sample_size), 1.0)
    phi = (n + 2.0) / ((n + 2.0) + T * q)
    phi = float(np.clip(phi, 0.0, 1.0))
    return _return_mu_with_phi((1.0 - phi) * mu + phi * target, phi, return_phi)


def momentum_mu(
    ret_window: pd.DataFrame,
    *,
    cov_ann: np.ndarray | pd.DataFrame | None = None,
    mode: Literal["12-1", "6-1", "3-0"] = "6-1",
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    winsor_lo: float = 0.05,
    winsor_hi: float = 0.95,
    cov_method: str = "LedoitWolf",
    annualization: float = DEFAULT_ANNUALIZATION,
    return_info: bool = False,
    return_series: bool = True,
    **_,
):
    R = _sanitize_returns(ret_window)
    if cov_ann is None:
        from quantfinlab.portfolio import covariance

        cov_ann = covariance.estimate_covariance(R, method=cov_method, annualization=annualization)
    raw = momentum_score_from_returns(R, mode=mode)
    mu = build_scaled_mu_from_raw(
        raw,
        _as_cov_array(cov_ann, R.columns),
        target_sharpe_ann=target_sharpe_ann,
        mu_cap_ann=mu_cap_ann,
        winsor_lo=winsor_lo,
        winsor_hi=winsor_hi,
    )
    info = {
        "mu_model": "Momentum",
        "shrinkage_intensity": np.nan,
        "invalid_values": int((~np.isfinite(mu)).sum()),
    }
    mu = np.nan_to_num(mu, nan=0.0, posinf=mu_cap_ann, neginf=-mu_cap_ann)
    return _return_mu(mu, info, R.columns, return_info, return_series)


def bayes_stein_mu(
    ret_window: pd.DataFrame,
    *,
    cov_ann: np.ndarray | pd.DataFrame | None = None,
    rf_daily: float = 0.0,
    ann_factor: float = DEFAULT_ANNUALIZATION,
    annualization: float = DEFAULT_ANNUALIZATION,
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    winsor_lo: float = 0.05,
    winsor_hi: float = 0.95,
    ridge: float = 1e-8,
    cov_method: str = "LedoitWolf",
    return_info: bool = False,
    return_series: bool = True,
    **_,
):
    R = _sanitize_returns(ret_window)
    if cov_ann is None:
        from quantfinlab.portfolio import covariance

        cov_ann = covariance.estimate_covariance(R, method=cov_method, annualization=annualization)
    raw, phi = bayes_stein_mean_excess_ann(
        R,
        rf_daily=rf_daily,
        ann_factor=ann_factor,
        ridge=ridge,
        return_phi=True,
    )
    mu = build_scaled_mu_from_raw(
        raw,
        _as_cov_array(cov_ann, R.columns),
        target_sharpe_ann=target_sharpe_ann,
        mu_cap_ann=mu_cap_ann,
        winsor_lo=winsor_lo,
        winsor_hi=winsor_hi,
    )
    info = {
        "mu_model": "BayesStein",
        "shrinkage_intensity": float(phi) if np.isfinite(phi) else np.nan,
        "invalid_values": int((~np.isfinite(mu)).sum()),
    }
    mu = np.nan_to_num(mu, nan=0.0, posinf=mu_cap_ann, neginf=-mu_cap_ann)
    return _return_mu(mu, info, R.columns, return_info, return_series)


def bayes_stein_momentum_mu(
    ret_window: pd.DataFrame,
    *,
    cov_ann: np.ndarray | pd.DataFrame | None = None,
    mode: Literal["12-1", "6-1", "3-0"] = "6-1",
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    winsor_lo: float = 0.05,
    winsor_hi: float = 0.95,
    ridge: float = 1e-8,
    cov_method: str = "LedoitWolf",
    annualization: float = DEFAULT_ANNUALIZATION,
    return_info: bool = False,
    return_series: bool = True,
    **_,
):
    R = _sanitize_returns(ret_window)
    if cov_ann is None:
        from quantfinlab.portfolio import covariance

        cov_ann = covariance.estimate_covariance(R, method=cov_method, annualization=annualization)
    cov = _as_cov_array(cov_ann, R.columns)
    momentum_raw = momentum_score_from_returns(R, mode=mode)
    momentum_scaled = build_scaled_mu_from_raw(
        momentum_raw,
        cov,
        target_sharpe_ann=target_sharpe_ann,
        mu_cap_ann=mu_cap_ann,
        winsor_lo=winsor_lo,
        winsor_hi=winsor_hi,
    )
    mu, phi = bayes_stein_shrink_mu(
        momentum_scaled,
        cov,
        sample_size=len(R),
        ridge=ridge,
        return_phi=True,
    )
    info = {
        "mu_model": "BayesSteinMomentum",
        "shrinkage_intensity": float(phi) if np.isfinite(phi) else np.nan,
        "invalid_values": int((~np.isfinite(mu)).sum()),
    }
    mu = np.nan_to_num(mu, nan=0.0, posinf=mu_cap_ann, neginf=-mu_cap_ann)
    return _return_mu(mu, info, R.columns, return_info, return_series)


def normalize_mu_model(mu_model: str) -> MuModel:
    key = str(mu_model).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "momentum": "Momentum",
        "bayesstein": "BayesStein",
        "bs": "BayesStein",
        "bayessteinmomentum": "BayesSteinMomentum",
        "bsmomentum": "BayesSteinMomentum",
        "bsm": "BayesSteinMomentum",
    }
    if key not in aliases:
        raise InputError(f"Unknown mu model: {mu_model!r}.")
    return aliases[key]  # type: ignore[return-value]


def build_mu_excess_ann(
    window: pd.DataFrame | Mapping[str, object],
    cov_ann: np.ndarray | pd.DataFrame | None = None,
    mu_model: str = "Momentum",
    *,
    cov_key: str | None = None,
    return_info: bool = False,
    return_series: bool = True,
    **kwargs,
):
    """
    Build annualized excess expected returns using the canonical Notebook 2 pipeline.

    The first argument may be a returns window or a rebalance state containing
    ``R_mu`` and ``cov_ann_map``.
    """
    if isinstance(window, Mapping):
        state = window
        R_mu = state.get("R_mu")
        if not isinstance(R_mu, pd.DataFrame):
            raise InputError("state must contain a DataFrame under 'R_mu'.")
        if cov_ann is None:
            cov_map = state.get("cov_ann_map")
            if not isinstance(cov_map, Mapping):
                raise InputError("state must contain cov_ann_map when cov_ann is not supplied.")
            use_key = cov_key if cov_key is not None else next(iter(cov_map))
            cov_ann = cov_map[use_key]
        window = R_mu

    model = normalize_mu_model(mu_model)
    if model == "Momentum":
        return momentum_mu(window, cov_ann=cov_ann, return_info=return_info, return_series=return_series, **kwargs)
    if model == "BayesStein":
        return bayes_stein_mu(window, cov_ann=cov_ann, return_info=return_info, return_series=return_series, **kwargs)
    return bayes_stein_momentum_mu(
        window,
        cov_ann=cov_ann,
        return_info=return_info,
        return_series=return_series,
        **kwargs,
    )


def mu_diagnostics(
    cache: Mapping[pd.Timestamp, Mapping[str, object]],
    *,
    cov_key: str = "LedoitWolf",
    mu_models: tuple[str, ...] = ("Momentum", "BayesStein", "BayesSteinMomentum"),
    **kwargs,
) -> pd.DataFrame:
    """Summarize cross-sectional μ behavior and shrinkage diagnostics over a cache."""
    rows: list[dict[str, object]] = []
    rank_corrs: list[float] = []
    for dt, state in cache.items():
        mu_by_model: dict[str, np.ndarray] = {}
        for model in mu_models:
            try:
                mu, info = build_mu_excess_ann(
                    state,
                    mu_model=model,
                    cov_key=cov_key,
                    return_info=True,
                    return_series=False,
                    **kwargs,
                )
                invalid_rebalance = int(info["invalid_values"] > 0)
            except Exception:
                mu = np.array([], dtype=float)
                info = {"shrinkage_intensity": np.nan, "invalid_values": np.nan}
                invalid_rebalance = 1

            finite_mu = np.asarray(mu, dtype=float)[np.isfinite(mu)]
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "mu_model": normalize_mu_model(model),
                    "cross_sectional_std": float(np.std(finite_mu)) if finite_mu.size else np.nan,
                    "max_abs_mu": float(np.max(np.abs(finite_mu))) if finite_mu.size else np.nan,
                    "shrinkage_intensity": info.get("shrinkage_intensity", np.nan),
                    "invalid_rebalance": invalid_rebalance,
                }
            )
            mu_by_model[normalize_mu_model(model)] = np.asarray(mu, dtype=float)

        if {"Momentum", "BayesSteinMomentum"}.issubset(mu_by_model):
            a = pd.Series(mu_by_model["Momentum"])
            b = pd.Series(mu_by_model["BayesSteinMomentum"])
            rank_corrs.append(float(a.rank().corr(b.rank())))

    diag = pd.DataFrame(rows)
    if diag.empty:
        return pd.DataFrame(
            columns=[
                "mu_model",
                "avg_cross_sectional_std",
                "avg_max_abs_mu",
                "avg_shrinkage_intensity",
                "invalid_rebalances",
                "momentum_bsm_rank_corr",
            ]
        )
    summary = (
        diag.groupby("mu_model")
        .agg(
            avg_cross_sectional_std=("cross_sectional_std", "mean"),
            avg_max_abs_mu=("max_abs_mu", "mean"),
            avg_shrinkage_intensity=("shrinkage_intensity", "mean"),
            invalid_rebalances=("invalid_rebalance", "sum"),
        )
        .reset_index()
    )
    summary["momentum_bsm_rank_corr"] = np.nan
    if rank_corrs:
        mask = summary["mu_model"].eq("BayesSteinMomentum")
        summary.loc[mask, "momentum_bsm_rank_corr"] = float(pd.Series(rank_corrs).mean())
    return summary


raw_momentum_signal = momentum_score_from_returns
raw_bayes_stein_signal = bayes_stein_mean_excess_ann
raw_bayes_stein_momentum_signal = bayes_stein_momentum_mu
mu_momentum = momentum_mu


__all__ = [
    "bayes_stein_mean_excess_ann",
    "bayes_stein_momentum_mu",
    "bayes_stein_mu",
    "bayes_stein_shrink_mu",
    "build_mu_excess_ann",
    "build_scaled_mu_from_raw",
    "momentum_mu",
    "momentum_score_from_returns",
    "mu_diagnostics",
    "mu_momentum",
    "normalize_mu_model",
    "raw_bayes_stein_momentum_signal",
    "raw_bayes_stein_signal",
    "raw_momentum_signal",
    "sample_mean_excess_ann_from_returns",
    "scale_mu_to_target_sharpe",
    "winsorize_and_zscore",
    "winsorize_signal",
    "zscore_signal",
]
