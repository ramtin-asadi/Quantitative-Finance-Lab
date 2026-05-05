from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import BacktestResult
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio import confidence, covariance, selection, views

try:  # pragma: no cover - exercised in notebooks when cvxpy is installed
    import cvxpy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass
class BLSettings:
    rebalance_freq: str = "ME"
    cov_lookback: int = 756
    mu_lookback: int = 252
    annualization: float = 252.0
    risk_free_rate_annual: float = 0.04
    tau: float = 0.05
    ewma_lambda: float = 0.97
    delta_fallback: float = 2.5
    delta_min: float = 0.50
    delta_max: float = 8.00
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.40
    min_weight: float = 0.0
    active_weight_limit: float = 0.15
    active_weight_relaxed: float = 0.22
    max_selected_views: int = 5
    redundancy_similarity: float = 0.85
    max_same_direction: int = 3
    confidence_mode: str = "learned"
    use_learned_q: bool = True
    full_view_covariance: bool = True
    empirical_omega_rescale: bool = False
    posterior_mu_clip: float = 0.30
    te_gamma_fallback: float = 3.0


@dataclass
class MarketState:
    date: pd.Timestamp
    signal_table: pd.DataFrame
    values: dict[str, Any]
    returns: pd.DataFrame
    signal_returns: pd.DataFrame


@dataclass
class BLRun:
    weights: pd.DataFrame
    candidate_view_log: pd.DataFrame
    selected_view_log: pd.DataFrame
    selection_log: pd.DataFrame
    payoff_history: pd.DataFrame
    reliability_summary: pd.DataFrame
    posterior_log: pd.DataFrame
    confidence_log: pd.DataFrame
    state_log: pd.DataFrame
    prior_mu: pd.DataFrame
    posterior_mu: pd.DataFrame
    covariance_by_date: dict[pd.Timestamp, pd.DataFrame] = field(default_factory=dict)
    fallback_flags: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "candidate_view_log": self.candidate_view_log,
            "selected_view_log": self.selected_view_log,
            "selection_log": self.selection_log,
            "payoff_history": self.payoff_history,
            "reliability_summary": self.reliability_summary,
            "posterior_log": self.posterior_log,
            "confidence_log": self.confidence_log,
            "state_log": self.state_log,
            "prior_mu": self.prior_mu,
            "posterior_mu": self.posterior_mu,
            "covariance_by_date": self.covariance_by_date,
            "fallback_flags": self.fallback_flags,
            "metadata": self.metadata,
        }

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


def risk_free_daily(rate_annual: float, annualization: float = 252.0) -> float:
    return float((1.0 + float(rate_annual)) ** (1.0 / float(annualization)) - 1.0)


def _sanitize_returns(returns: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise InputError(f"{name} must be a non-empty DataFrame.")
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c) for c in out.columns]
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def _resolve_pos(index: pd.DatetimeIndex, dt: pd.Timestamp) -> int:
    pos = int(index.searchsorted(pd.Timestamp(dt), side="right")) - 1
    return pos


def _normalize_series(weights: pd.Series, caps: pd.Series, *, floor: float = 0.0) -> pd.Series:
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).reindex(caps.index).fillna(0.0)
    w = w.clip(lower=max(float(floor), 0.0), upper=caps)
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0, index=caps.index, dtype=float).clip(upper=caps)
    w = w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(caps), index=caps.index)
    for _ in range(100):
        over = w > caps + 1e-12
        if not bool(over.any()):
            break
        extra = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = (caps[~over] - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += extra * room / float(room.sum())
    w = w.clip(lower=max(float(floor), 0.0), upper=caps)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(w), index=w.index)


def prior_from_benchmark(
    cov_ann: pd.DataFrame,
    benchmark_weights: pd.Series,
    *,
    returns_window: pd.DataFrame | None = None,
    settings: BLSettings | None = None,
    delta: float | None = None,
) -> tuple[pd.Series, float]:
    settings = settings or BLSettings()
    assets = list(cov_ann.index)
    bench = pd.Series(benchmark_weights, dtype=float).reindex(assets).fillna(0.0)
    bench = bench / float(bench.sum()) if float(bench.sum()) > 1e-12 else pd.Series(1.0 / len(assets), index=assets)
    if delta is None:
        if returns_window is None or returns_window.empty:
            delta_t = float(settings.delta_fallback)
        else:
            window = returns_window.reindex(columns=assets).fillna(0.0)
            bench_ret = window @ bench
            ann_return = float(bench_ret.mean() * float(settings.annualization))
            ann_var = float(bench_ret.var(ddof=1) * float(settings.annualization))
            rf_ann = risk_free_daily(settings.risk_free_rate_annual, settings.annualization) * settings.annualization
            raw_delta = settings.delta_fallback if (not np.isfinite(ann_var) or ann_var <= 1e-10) else (ann_return - rf_ann) / ann_var
            delta_t = raw_delta if np.isfinite(raw_delta) else settings.delta_fallback
    else:
        delta_t = float(delta)
    delta_t = float(np.clip(delta_t, settings.delta_min, settings.delta_max))
    prior = pd.Series(delta_t * cov_ann.loc[assets, assets].values @ bench.loc[assets].values, index=assets, dtype=float)
    return prior, delta_t


def posterior_returns(
    prior_mu: pd.Series,
    cov_ann: pd.DataFrame,
    view_p: np.ndarray,
    view_q: np.ndarray,
    omega: np.ndarray,
    *,
    tau: float = 0.05,
    mu_clip: float = 0.30,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    prior_mu = pd.Series(prior_mu).astype(float)
    cov_ann = pd.DataFrame(cov_ann, index=prior_mu.index, columns=prior_mu.index).astype(float)
    diag = {
        "n_views": int(view_p.shape[0]),
        "avg_confidence": np.nan,
        "max_confidence": np.nan,
        "min_confidence": np.nan,
        "max_abs_q": float(np.max(np.abs(view_q))) if len(view_q) else 0.0,
        "max_abs_posterior_shift": 0.0,
        "condition_number": np.nan,
        "no_views": view_p.shape[0] == 0,
        "failure": False,
        "clipped": False,
    }
    if view_p.shape[0] == 0:
        return prior_mu.copy(), cov_ann.copy(), diag
    try:
        cov_arr = covariance.make_psd(cov_ann.values, eps=1e-10)
        tau_cov = float(tau) * cov_arr
        omega_arr = np.asarray(omega, dtype=float)
        omega_arr = 0.5 * (omega_arr + omega_arr.T) + np.eye(view_p.shape[0]) * 1e-8
        middle = view_p @ tau_cov @ view_p.T + omega_arr
        middle = 0.5 * (middle + middle.T) + np.eye(view_p.shape[0]) * 1e-10
        view_gap = np.asarray(view_q, dtype=float) - view_p @ prior_mu.values
        diag["condition_number"] = float(np.linalg.cond(middle))
        solved_gap = np.linalg.solve(middle, view_gap)
        post_mu_arr = prior_mu.values + tau_cov @ view_p.T @ solved_gap
        solved_cov = np.linalg.solve(middle, view_p @ tau_cov)
        posterior_uncertainty = tau_cov - tau_cov @ view_p.T @ solved_cov
        post_cov_arr = covariance.make_psd(cov_arr + posterior_uncertainty, eps=1e-10)
        clipped = np.any((post_mu_arr < -float(mu_clip)) | (post_mu_arr > float(mu_clip)))
        post_mu_arr = np.clip(post_mu_arr, -float(mu_clip), float(mu_clip))
        post_mu = pd.Series(post_mu_arr, index=prior_mu.index, dtype=float)
        post_cov = pd.DataFrame(post_cov_arr, index=prior_mu.index, columns=prior_mu.index)
        diag["clipped"] = bool(clipped)
        diag["max_abs_posterior_shift"] = float(np.max(np.abs(post_mu.values - prior_mu.values)))
        return post_mu, post_cov, diag
    except Exception:
        diag["failure"] = True
        return prior_mu.copy(), cov_ann.copy(), diag


def empirical_view_variance(view_family: str, history: pd.DataFrame | None, current_date: pd.Timestamp | str) -> tuple[float, int]:
    if history is None or len(history) == 0:
        return np.nan, 0
    sample = history[(history["view_family"] == view_family) & (pd.to_datetime(history["payoff_end_date"]) < pd.Timestamp(current_date))].copy()
    payoff_col = "payoff_ann" if "payoff_ann" in sample.columns else "payoff"
    x = pd.to_numeric(sample.get(payoff_col, pd.Series(dtype=float)), errors="coerce").dropna()
    if len(x) < 8:
        return np.nan, int(len(x))
    return float(x.var(ddof=1)), int(len(x))


def omega_from_confidence(
    view_p: np.ndarray,
    cov_ann: pd.DataFrame | np.ndarray,
    confidence_values: Sequence[float],
    *,
    tau: float = 0.05,
    active_view_rows: Sequence[Mapping[str, Any]] | None = None,
    history: pd.DataFrame | None = None,
    current_date: pd.Timestamp | str | None = None,
    empirical_rescale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if len(confidence_values) == 0 or view_p.shape[0] == 0:
        return np.empty((0, 0)), np.array([])
    conf = np.asarray(confidence_values, dtype=float)
    conf = np.where(np.isfinite(conf), conf, 0.50)
    conf = np.maximum(conf, 1e-4)
    cov_arr = np.asarray(cov_ann, dtype=float)
    view_cov = view_p @ cov_arr @ view_p.T
    view_cov = 0.5 * (view_cov + view_cov.T)
    view_var = np.maximum(np.diag(view_cov), 1e-8)
    scale = np.sqrt((1.0 - conf) / conf)
    base_omega = float(tau) * np.diag(scale) @ view_cov @ np.diag(scale)
    base_omega = 0.5 * (base_omega + base_omega.T) + np.eye(len(conf)) * 1e-8
    base_diag = np.maximum(np.diag(base_omega), 1e-10)
    target_diag = base_diag.copy()
    if empirical_rescale and active_view_rows is not None and history is not None and current_date is not None:
        for i, row in enumerate(active_view_rows):
            emp_var, n_obs = empirical_view_variance(str(row.get("view_family")), history, current_date)
            if np.isfinite(emp_var) and emp_var > 1e-10:
                sample_weight = min(0.50, n_obs / 48.0)
                target_diag[i] = (1.0 - sample_weight) * base_diag[i] + sample_weight * emp_var
    try:
        rescale = np.sqrt(np.maximum(target_diag, 1e-10) / base_diag)
        omega = np.diag(rescale) @ base_omega @ np.diag(rescale)
        omega = 0.5 * (omega + omega.T) + np.eye(len(conf)) * 1e-8
        if np.all(np.isfinite(omega)):
            return omega, view_var
    except Exception:
        pass
    return np.diag(np.maximum(target_diag, 1e-8)), view_var


def _cvxpy_bl_weights(
    mu: pd.Series,
    cov_ann: pd.DataFrame,
    benchmark: pd.Series,
    caps: pd.Series,
    *,
    w_prev: pd.Series | None,
    risk_aversion: float,
    settings: BLSettings,
    active_limit: float | None,
    sleeve_constraints: Mapping[str, Mapping[str, Any]] | None,
    te_gamma: float = 0.0,
) -> np.ndarray | None:
    if cp is None:
        return None
    index = list(mu.index)
    n = len(index)
    w_var = cp.Variable(n)
    cov_arr = covariance.make_psd(cov_ann.values + np.eye(n) * 1e-8, eps=1e-10)
    cap_arr = caps.reindex(index).fillna(settings.max_weight).to_numpy(dtype=float)
    bench = benchmark.reindex(index).fillna(0.0).to_numpy(dtype=float)
    prev = (w_prev.reindex(index).fillna(0.0).to_numpy(dtype=float) if w_prev is not None else bench)
    risk_aversion = float(settings.delta_fallback if risk_aversion is None or not np.isfinite(risk_aversion) else risk_aversion)
    turnover_penalty = float(settings.transaction_cost_bps) / 10000.0 * cp.norm1(w_var - prev)
    te_penalty = float(te_gamma) * cp.quad_form(w_var - bench, cov_arr) if te_gamma and te_gamma > 0 else 0.0
    objective = cp.Maximize(mu.values @ w_var - 0.5 * risk_aversion * cp.quad_form(w_var, cov_arr) - turnover_penalty - te_penalty)
    cons = [cp.sum(w_var) == 1.0, w_var >= settings.min_weight, w_var <= cap_arr]
    if active_limit is not None:
        cons += [w_var - bench <= float(active_limit), bench - w_var <= float(active_limit)]
    if sleeve_constraints:
        for spec in sleeve_constraints.values():
            locs = [index.index(asset) for asset in spec.get("assets", []) if asset in index]
            if locs:
                sleeve_weight = cp.sum(w_var[locs])
                cons += [sleeve_weight >= float(spec.get("min", 0.0)), sleeve_weight <= float(spec.get("max", 1.0))]
    problem = cp.Problem(objective, cons)
    for solver in ["CLARABEL", "OSQP", "ECOS", "SCS"]:
        try:
            problem.solve(solver=solver, verbose=False)
            if w_var.value is not None and problem.status in ["optimal", "optimal_inaccurate"]:
                return np.asarray(w_var.value, dtype=float)
        except Exception:
            continue
    return None


def posterior_weights(
    post_mu: pd.Series,
    post_cov: pd.DataFrame,
    benchmark_weights: pd.Series,
    *,
    previous_weights: pd.Series | None = None,
    risk_aversion: float | None = None,
    settings: BLSettings | None = None,
    constraints: Mapping[str, Any] | None = None,
) -> tuple[pd.Series, bool, dict[str, Any]]:
    settings = settings or BLSettings()
    constraints = dict(constraints or {})
    index = list(post_mu.index)
    caps = constraints.get("asset_caps")
    caps = pd.Series(caps, dtype=float).reindex(index).fillna(settings.max_weight) if caps is not None else pd.Series(settings.max_weight, index=index, dtype=float)
    bench = pd.Series(benchmark_weights, dtype=float).reindex(index).fillna(0.0)
    bench = _normalize_series(bench, caps, floor=settings.min_weight)
    w_prev = pd.Series(previous_weights, dtype=float).reindex(index).fillna(0.0) if previous_weights is not None else None
    sleeve_constraints = constraints.get("sleeve_constraints")
    attempts = [
        {"active_limit": settings.active_weight_limit, "sleeve_constraints": sleeve_constraints, "te_gamma": 0.0},
        {"active_limit": settings.active_weight_relaxed, "sleeve_constraints": sleeve_constraints, "te_gamma": 0.0},
        {"active_limit": settings.active_weight_limit, "sleeve_constraints": None, "te_gamma": 0.0},
        {"active_limit": settings.active_weight_relaxed, "sleeve_constraints": None, "te_gamma": 0.0},
        {"active_limit": None, "sleeve_constraints": None, "te_gamma": settings.te_gamma_fallback},
    ]
    for attempt in attempts:
        raw = _cvxpy_bl_weights(
            post_mu,
            post_cov,
            bench,
            caps,
            w_prev=w_prev,
            risk_aversion=settings.delta_fallback if risk_aversion is None else risk_aversion,
            settings=settings,
            **attempt,
        )
        if raw is not None:
            return _normalize_series(pd.Series(raw, index=index), caps, floor=settings.min_weight), False, dict(attempt)
    fallback = w_prev if w_prev is not None else bench
    return _normalize_series(fallback, caps, floor=settings.min_weight), True, {"active_limit": None, "sleeve_constraints": None, "te_gamma": np.nan}


def market_state(
    *,
    returns: pd.DataFrame,
    signal_returns: pd.DataFrame,
    date: pd.Timestamp | str,
    roles: Mapping[str, Any],
    settings: BLSettings | None = None,
    view_settings: views.ViewSettings | None = None,
) -> MarketState:
    settings = settings or BLSettings()
    view_settings = view_settings or views.ViewSettings()
    view_settings_use = replace(view_settings, annualization=settings.annualization)
    date = pd.Timestamp(date)
    ret_pos = _resolve_pos(pd.DatetimeIndex(returns.index), date)
    sig_pos = _resolve_pos(pd.DatetimeIndex(signal_returns.index), date)
    if ret_pos < 0 or sig_pos < 0:
        return MarketState(date, pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame())
    ret_hist = returns.iloc[: ret_pos + 1].copy()
    sig_hist = signal_returns.iloc[: sig_pos + 1].copy()
    signal_table, values = views.signal_table_from_returns(sig_hist, date, roles=roles, settings=view_settings_use)
    return MarketState(date=date, signal_table=signal_table, values=values, returns=ret_hist, signal_returns=sig_hist)


def _state_log_row(state: MarketState) -> dict[str, Any]:
    row: dict[str, Any] = {"date": state.date}
    for key, value in state.values.items():
        if isinstance(value, Mapping):
            continue
        row[key] = value
    return row


def _reliability_summary(candidate_log: pd.DataFrame, selected_log: pd.DataFrame, payoff_log: pd.DataFrame, confidence_log: pd.DataFrame) -> pd.DataFrame:
    if candidate_log.empty:
        return pd.DataFrame()
    candidate_counts = candidate_log.groupby("view_family").size().rename("candidate_count")
    display_name = candidate_log.groupby("view_family")["family_display_name"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]) if "family_display_name" in candidate_log.columns else pd.Series(dtype=object)
    selected_counts = selected_log.groupby("view_family").size().rename("selected_count") if not selected_log.empty else pd.Series(dtype=float, name="selected_count")
    payoff_stats = pd.DataFrame()
    if not payoff_log.empty:
        payoff_stats = payoff_log.groupby("view_family").agg(hit_rate=("hit", "mean"), avg_payoff=("payoff", "mean"), payoff_vol=("payoff", "std"))
        payoff_stats["payoff_ir"] = payoff_stats["avg_payoff"] / payoff_stats["payoff_vol"].replace(0.0, np.nan)
    conf_means = confidence_log.groupby("view_family")["confidence"].mean().rename("avg_confidence") if not confidence_log.empty else pd.Series(dtype=float)
    q_means = selected_log.groupby("view_family")["q_tilt_final"].mean().rename("avg_q") if not selected_log.empty and "q_tilt_final" in selected_log.columns else pd.Series(dtype=float)
    abs_q = selected_log.groupby("view_family")["q_tilt_final"].agg(lambda x: np.mean(np.abs(x))).rename("avg_abs_q") if not selected_log.empty and "q_tilt_final" in selected_log.columns else pd.Series(dtype=float)
    out = pd.concat([display_name.rename("display_name"), candidate_counts, selected_counts, payoff_stats, conf_means, q_means, abs_q], axis=1)
    out["selected_share"] = out["selected_count"] / out["candidate_count"].replace(0.0, np.nan)
    return out.sort_values(["selected_count", "candidate_count"], ascending=False)


def learned_confidence_bl(
    *,
    returns: pd.DataFrame,
    signal_returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    benchmark_weights: pd.Series | Mapping[str, float],
    roles: Mapping[str, Any],
    view_functions: Sequence[Callable[[MarketState, Mapping[str, Any], views.ViewSettings], views.View | Mapping[str, Any] | None]],
    view_settings: views.ViewSettings,
    settings: BLSettings | None = None,
    constraints: Mapping[str, Any] | None = None,
) -> BLRun:
    settings = settings or BLSettings()
    R = _sanitize_returns(returns, name="returns")
    S = _sanitize_returns(signal_returns, name="signal_returns")
    assets = [str(c) for c in R.columns]
    view_settings_use = replace(view_settings, assets=assets, annualization=settings.annualization)
    dates = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values().unique()
    if len(dates) < 2:
        raise InputError("At least two rebalance dates are required.")
    bench = pd.Series(benchmark_weights, dtype=float).reindex(assets).fillna(0.0)
    if float(bench.sum()) <= 1e-12:
        bench = pd.Series(1.0 / len(assets), index=assets, dtype=float)
    else:
        bench = bench / float(bench.sum())

    weight_rows: list[pd.Series] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    prior_rows: list[pd.Series] = []
    posterior_mu_rows: list[pd.Series] = []
    cov_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    fallback_rows: dict[pd.Timestamp, float] = {}
    previous_weights: pd.Series | None = None

    for dt_raw in dates[:-1]:
        dt = pd.Timestamp(dt_raw)
        pos = _resolve_pos(pd.DatetimeIndex(R.index), dt)
        if pos < max(int(settings.cov_lookback), int(settings.mu_lookback)) - 1:
            continue
        cov_window = R.iloc[max(0, pos - int(settings.cov_lookback) + 1) : pos + 1].reindex(columns=assets).fillna(0.0)
        mu_window = R.iloc[max(0, pos - int(settings.mu_lookback) + 1) : pos + 1].reindex(columns=assets).fillna(0.0)
        if len(cov_window) < max(30, settings.cov_lookback // 3):
            continue
        cov_ann = covariance.estimate_covariance(cov_window, method="EWMA", annualization=settings.annualization, ewma_lambda=settings.ewma_lambda, return_df=True)
        cov_ann = pd.DataFrame(cov_ann, index=assets, columns=assets).astype(float)
        prior_mu, delta_t = prior_from_benchmark(cov_ann, bench, returns_window=mu_window, settings=settings)
        cov_by_date[dt] = cov_ann
        prior_rows.append(prior_mu.rename(dt))

        state = market_state(returns=R, signal_returns=S, date=dt, roles=roles, settings=settings, view_settings=view_settings_use)
        state_rows.append(_state_log_row(state))
        active_views: list[dict[str, Any]] = []
        for fn in view_functions:
            row = fn(state, roles, view_settings_use)
            active_views.extend(views.view_rows([row]))
        for row in active_views:
            row["date"] = dt
            row["view_set"] = "candidate"
        payoff_now = confidence.payoff_history(
            pd.DataFrame(candidate_rows),
            R[assets],
            horizon=view_settings_use.view_horizon_days,
            state_log=pd.DataFrame(state_rows),
            annualization=settings.annualization,
        )
        if active_views:
            candidate_rows.extend(active_views)
        selected, scale_rows = confidence.select_views(
            active_views,
            payoff_now,
            dt,
            state.values,
            assets=assets,
            family_q_caps=view_settings_use.family_q_caps,
            confidence_mode=settings.confidence_mode,
            max_selected_views=settings.max_selected_views,
            redundancy_similarity=settings.redundancy_similarity,
            max_same_direction=settings.max_same_direction,
            protective_families=("correlation_stress", "quality_defensive", "defensive_rotation"),
        )
        selection_rows.extend(scale_rows)
        view_p, view_q, clean_table = confidence.view_matrix(
            selected,
            assets,
            prior_mu=prior_mu,
            history=payoff_now,
            current_date=dt,
            family_q_caps=view_settings_use.family_q_caps,
            q_strength_scale=view_settings_use.q_strength_scale,
            use_learned_q=settings.use_learned_q,
            protective_families=("correlation_stress", "quality_defensive", "defensive_rotation"),
        )
        if not clean_table.empty:
            selected_rows.extend(clean_table.assign(date=dt, view_set="selected", strategy="Learned-Confidence BL").to_dict("records"))
        row_records = clean_table.to_dict("records") if not clean_table.empty else []
        conf_values: list[float] = []
        for row in row_records:
            stats = confidence.family_reliability(row["view_family"], payoff_now, dt, protective_families=("correlation_stress", "quality_defensive", "defensive_rotation"))
            details = confidence.confidence_score(stats, row, state.values, confidence_mode=settings.confidence_mode, protective_families=("correlation_stress", "quality_defensive", "defensive_rotation"))
            conf = details["confidence"] if np.isfinite(details["confidence"]) else 0.50
            conf_values.append(float(conf))
            confidence_rows.append(
                {
                    "date": dt,
                    "view_set": "selected",
                    "strategy": "Learned-Confidence BL",
                    "view_family": row["view_family"],
                    "view_name": row["view_name"],
                    "economic_theme": row.get("family_display_name", row["view_family"]),
                    "confidence_mode": settings.confidence_mode,
                    "confidence": conf,
                    "historical_confidence": row.get("historical_confidence", np.nan),
                    "selected_score": row.get("selected_score", np.nan),
                    "confluence_score": row.get("confluence_score", np.nan),
                    "novelty_score": row.get("novelty_score", np.nan),
                    "economic_priority": row.get("economic_priority", np.nan),
                    "q_tilt": row.get("q_tilt", np.nan),
                    "hit_rate": details["hit_rate"],
                    "recent_hit_rate": details["recent_hit_rate"],
                    "payoff_ir": details["payoff_ir"],
                    "t_stat": details.get("t_stat", np.nan),
                    "info_coefficient": details.get("info_coefficient", np.nan),
                    "n_obs": details["n_obs"],
                    "haircut_multiplier": details.get("haircut_multiplier", 1.0),
                    "risk_orientation": row.get("risk_orientation", "neutral"),
                }
            )
        omega, view_var = omega_from_confidence(
            view_p,
            cov_ann,
            conf_values,
            tau=settings.tau,
            active_view_rows=row_records,
            history=payoff_now,
            current_date=dt,
            empirical_rescale=settings.empirical_omega_rescale,
        )
        post_mu, post_cov, diag = posterior_returns(prior_mu, cov_ann, view_p, view_q, omega, tau=settings.tau, mu_clip=settings.posterior_mu_clip)
        if conf_values:
            diag["avg_confidence"] = float(np.mean(conf_values))
            diag["max_confidence"] = float(np.max(conf_values))
            diag["min_confidence"] = float(np.min(conf_values))
        diag.update({"date": dt, "strategy": "Learned-Confidence BL", "confidence_mode": settings.confidence_mode, "risk_aversion": delta_t, "avg_view_variance": float(np.mean(view_var)) if len(view_var) else np.nan})
        posterior_rows.append(diag)
        posterior_mu_rows.append(post_mu.rename(dt))
        w_new, fallback, opt_info = posterior_weights(
            post_mu,
            post_cov,
            bench,
            previous_weights=previous_weights,
            risk_aversion=delta_t,
            settings=settings,
            constraints=constraints,
        )
        weight_rows.append(w_new.rename(dt))
        fallback_rows[dt] = float(bool(fallback or diag.get("failure", False)))
        previous_weights = w_new

    weights = pd.DataFrame(weight_rows).reindex(columns=assets).fillna(0.0)
    candidate_log = pd.DataFrame(candidate_rows)
    selected_log = pd.DataFrame(selected_rows)
    selection_log = pd.DataFrame(selection_rows)
    state_log = pd.DataFrame(state_rows)
    payoff_log = confidence.payoff_history(candidate_log, R[assets], horizon=view_settings_use.view_horizon_days, state_log=state_log, annualization=settings.annualization)
    confidence_log = pd.DataFrame(confidence_rows)
    posterior_log = pd.DataFrame(posterior_rows)
    prior_frame = pd.DataFrame(prior_rows).sort_index() if prior_rows else pd.DataFrame(columns=assets)
    posterior_mu_frame = pd.DataFrame(posterior_mu_rows).sort_index() if posterior_mu_rows else pd.DataFrame(columns=assets)
    reliability = _reliability_summary(candidate_log, selected_log, payoff_log, confidence_log)
    return BLRun(
        weights=weights,
        candidate_view_log=candidate_log,
        selected_view_log=selected_log,
        selection_log=selection_log,
        payoff_history=payoff_log,
        reliability_summary=reliability,
        posterior_log=posterior_log,
        confidence_log=confidence_log,
        state_log=state_log,
        prior_mu=prior_frame,
        posterior_mu=posterior_mu_frame,
        covariance_by_date=cov_by_date,
        fallback_flags=pd.Series(fallback_rows, dtype=float).sort_index(),
        metadata={
            "assets": assets,
            "settings": settings,
            "view_settings": view_settings_use,
            "benchmark_weights": bench,
        },
    )


def _as_result(result: BacktestResult | Mapping[str, Any]) -> BacktestResult:
    if isinstance(result, BacktestResult):
        return result
    return BacktestResult(
        gross_values=pd.Series(result.get("gross_values", result.get("gross_nav", pd.Series(dtype=float)))),
        net_values=pd.Series(result.get("net_values", result.get("nav", pd.Series(dtype=float)))),
        gross_returns=pd.Series(result.get("gross_returns", pd.Series(dtype=float))),
        net_returns=pd.Series(result.get("net_returns", result.get("returns", pd.Series(dtype=float)))),
        weights=pd.DataFrame(result.get("weights", pd.DataFrame())),
        turnover=pd.Series(result.get("turnover", pd.Series(dtype=float))),
        costs=pd.Series(result.get("costs", pd.Series(dtype=float))),
        fallbacks=int(result.get("fallbacks", result.get("fallback_count", 0))),
        metadata=dict(result.get("metadata", {})),
    )


def data_coverage_table(
    prices: pd.DataFrame,
    *,
    tradable_assets: Sequence[str],
    signal_assets: Sequence[str] | None = None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    if start is not None:
        px = px.loc[px.index >= pd.Timestamp(start)]
    tradable = set(str(x) for x in tradable_assets)
    signals = [str(x) for x in (signal_assets or [])]
    tickers = list(dict.fromkeys([*tradable_assets, *signals]))
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in px.columns:
            rows.append(
                {
                    "ticker": ticker,
                    "role": "tradable" if ticker in tradable else "signal",
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "observations": 0,
                    "missing_pct_after_first_valid": np.nan,
                    "included": False,
                }
            )
            continue
        series = px[ticker]
        valid = series.dropna()
        first = valid.index.min() if len(valid) else pd.NaT
        missing = float(series.loc[first:].isna().mean()) if pd.notna(first) else np.nan
        rows.append(
            {
                "ticker": ticker,
                "role": "tradable" if ticker in tradable else "signal",
                "first_date": first,
                "last_date": valid.index.max() if len(valid) else pd.NaT,
                "observations": int(len(valid)),
                "missing_pct_after_first_valid": missing,
                "included": bool(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def view_spec_table(specs: Sequence[Mapping[str, Any]], caps: Mapping[str, float], display_names: Mapping[str, str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    display_names = dict(display_names or {})
    for spec in specs:
        family = str(spec["family"])
        rows.append(
            {
                "family": family,
                "display_name": display_names.get(family, family.replace("_", " ").title()),
                "cap": float(caps.get(family, np.nan)),
                "function": spec.get("function", ""),
                "economic_idea": spec.get("economic_idea", ""),
                "typical_long": spec.get("typical_long", ""),
                "typical_short": spec.get("typical_short", ""),
                "main_signal": spec.get("main_signal", ""),
            }
        )
    return pd.DataFrame(rows)


def latest_view_table(view_rows: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(list(view_rows)) if not isinstance(view_rows, pd.DataFrame) else view_rows.copy()
    if df.empty:
        return pd.DataFrame(columns=["family", "name", "state", "strength", "raw_q", "final_q", "confidence", "long_assets", "short_assets"])
    return pd.DataFrame(
        {
            "family": df.get("view_family"),
            "name": df.get("view_name"),
            "state": df.get("view_state", df.get("risk_orientation")),
            "strength": df.get("raw_strength", df.get("signal_value")),
            "raw_q": df.get("raw_q", df.get("q_tilt")),
            "final_q": df.get("final_q", df.get("q_tilt_final", df.get("q_tilt"))),
            "confidence": df.get("confidence"),
            "long_assets": df.get("long_assets").map(lambda x: ", ".join(x) if isinstance(x, list) else x),
            "short_assets": df.get("short_assets").map(lambda x: ", ".join(x) if isinstance(x, list) else x),
        }
    )


def _active_metrics(strategy: BacktestResult, benchmark: BacktestResult, annualization: float) -> dict[str, float]:
    bench_ret = benchmark.net_returns.dropna().astype(float)
    ret = strategy.net_returns.reindex(bench_ret.index).fillna(0.0).astype(float)
    active = ret - bench_ret
    tracking_error = float(active.std(ddof=1) * np.sqrt(float(annualization))) if active.std(ddof=1) > 1e-12 else np.nan
    active_return = float(active.mean() * float(annualization)) if len(active) else np.nan
    info_ratio = active_return / tracking_error if np.isfinite(tracking_error) and tracking_error > 1e-12 else np.nan
    corr = float(ret.corr(bench_ret)) if ret.std(ddof=1) > 1e-12 and bench_ret.std(ddof=1) > 1e-12 else np.nan
    beta = float(ret.cov(bench_ret) / bench_ret.var(ddof=1)) if bench_ret.var(ddof=1) > 1e-12 else np.nan
    return {
        "Active return": active_return,
        "Tracking error": tracking_error,
        "Information ratio": info_ratio,
        "Hit rate vs benchmark": float((active > 0).mean()) if len(active) else np.nan,
        "Correlation to benchmark": corr,
        "Beta to benchmark": beta,
    }


def model_comparison_table(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    benchmark_name: str,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
) -> pd.DataFrame:
    objects = {name: _as_result(result) for name, result in results.items()}
    benchmark = objects[benchmark_name]
    rows: list[dict[str, Any]] = []
    for name, result in objects.items():
        perf = selection.performance_metrics(result.net_returns, result.net_values, rf_daily=rf_daily, annualization=annualization)
        active = {
            k: (0.0 if k in {"Active return", "Tracking error"} and name == benchmark_name else np.nan)
            for k in ["Active return", "Tracking error", "Information ratio", "Hit rate vs benchmark", "Correlation to benchmark", "Beta to benchmark"]
        }
        if name != benchmark_name:
            active = _active_metrics(result, benchmark, annualization)
        rows.append(
            {
                "Strategy": name,
                "Net CAGR": perf.get("CAGR"),
                "Volatility": perf.get("Vol"),
                "Sharpe": perf.get("Sharpe"),
                "Max Drawdown": perf.get("Max Drawdown"),
                "Calmar": perf.get("Calmar"),
                "Avg turnover": float(result.turnover.mean()) if not result.turnover.empty else 0.0,
                "Total turnover": float(result.turnover.sum()) if not result.turnover.empty else 0.0,
                "Cost drag": float(result.costs.sum() / result.net_values.iloc[-1]) if (not result.costs.empty and not result.net_values.empty and result.net_values.iloc[-1] > 0) else 0.0,
                **active,
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


def active_summary_table(
    strategy_result: BacktestResult | Mapping[str, Any],
    benchmark_result: BacktestResult | Mapping[str, Any],
    *,
    strategy_name: str = "Learned-Confidence BL",
    benchmark_name: str = "Benchmark",
    annualization: float = 252.0,
) -> pd.DataFrame:
    strategy = _as_result(strategy_result)
    benchmark = _as_result(benchmark_result)
    metrics = _active_metrics(strategy, benchmark, annualization)
    active = strategy.net_returns.reindex(benchmark.net_returns.index).fillna(0.0) - benchmark.net_returns.dropna()
    active_nav = (1.0 + active.fillna(0.0)).cumprod()
    metrics["Active max drawdown"] = float((active_nav / active_nav.cummax() - 1.0).min()) if not active_nav.empty else np.nan
    if not strategy.weights.empty and not benchmark.weights.empty:
        bw = benchmark.weights.reindex(strategy.weights.index).ffill().reindex(columns=strategy.weights.columns).fillna(0.0)
        metrics["Avg active weight distance"] = float(strategy.weights.subtract(bw, fill_value=0.0).abs().sum(axis=1).mean())
    metrics["Strategy"] = strategy_name
    metrics["Benchmark"] = benchmark_name
    return pd.DataFrame([metrics]).set_index("Strategy")


def view_reliability_table(bl_run: Any) -> pd.DataFrame:
    reliability = getattr(bl_run, "reliability_summary", pd.DataFrame()).copy()
    if reliability.empty:
        return reliability
    cols = [
        "display_name",
        "candidate_count",
        "selected_count",
        "selected_share",
        "hit_rate",
        "avg_payoff",
        "payoff_vol",
        "payoff_ir",
        "avg_q",
        "avg_abs_q",
        "avg_confidence",
    ]
    return reliability[[c for c in cols if c in reliability.columns]]


def selection_summary_table(selection_log: pd.DataFrame) -> pd.DataFrame:
    if selection_log is None or selection_log.empty:
        return pd.DataFrame()
    counts = selection_log.groupby("scale_reason").size().rename("count").sort_values(ascending=False).to_frame()
    counts["share"] = counts["count"] / counts["count"].sum()
    monthly = selection_log.groupby("date").agg(candidate_views=("view_family", "size"), selected_views=("kept", "sum"))
    average_rows = pd.DataFrame(
        {
            "count": [
                float(monthly["selected_views"].mean()) if not monthly.empty else 0.0,
                float(monthly["candidate_views"].mean()) if not monthly.empty else 0.0,
            ],
            "share": [0.0, 0.0],
        },
        index=["average selected views", "average candidate views"],
    )
    return pd.concat([counts, average_rows], axis=0).fillna({"share": 0.0})


def latest_weight_table(
    weights: pd.DataFrame,
    benchmark_weights: pd.Series | Mapping[str, float],
    *,
    date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    if weights.empty:
        return pd.DataFrame()
    dt = pd.Timestamp(date) if date is not None else pd.Timestamp(weights.index.max())
    row = weights.loc[:dt].iloc[-1].astype(float)
    bench = pd.Series(benchmark_weights, dtype=float).reindex(row.index).fillna(0.0)
    out = pd.DataFrame({"benchmark_weight": bench, "bl_weight": row})
    out["active_weight"] = out["bl_weight"] - out["benchmark_weight"]
    out["abs_active_weight"] = out["active_weight"].abs()
    return out.sort_values("abs_active_weight", ascending=False)


def stress_summary_table(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    benchmark_name: str,
    windows: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    objects = {name: _as_result(result) for name, result in results.items()}
    if windows is None:
        windows = {
            "2018 Q4 selloff": ("2018-10-01", "2018-12-31"),
            "COVID crash": ("2020-02-19", "2020-03-23"),
            "2022 rates/inflation": ("2022-01-03", "2022-10-14"),
            "2023 growth rebound": ("2023-01-03", "2023-07-31"),
            "2024-2025 cycle": ("2024-01-02", "2025-12-31"),
        }
    rows: list[dict[str, Any]] = []
    bench_ret = objects[benchmark_name].net_returns
    for window_name, (start, end) in windows.items():
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for name, result in objects.items():
            r = result.net_returns.loc[(result.net_returns.index >= start_ts) & (result.net_returns.index <= end_ts)].dropna()
            if r.empty:
                continue
            total_return = float((1.0 + r).prod() - 1.0)
            row = {"window": window_name, "strategy": name, "return": total_return}
            if name != benchmark_name:
                b = bench_ret.reindex(r.index).fillna(0.0)
                row["active_return"] = float(((1.0 + r).prod() - 1.0) - ((1.0 + b).prod() - 1.0))
            else:
                row["active_return"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "active_summary_table",
    "BLRun",
    "BLSettings",
    "MarketState",
    "data_coverage_table",
    "learned_confidence_bl",
    "latest_view_table",
    "latest_weight_table",
    "market_state",
    "model_comparison_table",
    "omega_from_confidence",
    "posterior_returns",
    "posterior_weights",
    "prior_from_benchmark",
    "risk_free_daily",
    "selection_summary_table",
    "stress_summary_table",
    "view_reliability_table",
    "view_spec_table",
]
