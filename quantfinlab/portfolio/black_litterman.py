from __future__ import annotations

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
    """Configuration container for the learned-confidence Black-Litterman workflow.

    The settings define the rebalance cadence, lookback windows, covariance and
    expected-return scaling conventions, Black-Litterman uncertainty parameters,
    transaction-cost assumptions, portfolio constraints, view-selection rules,
    and confidence/learning behavior used by the allocation engine.

    Attributes
    ----------
    rebalance_freq : str
        Rebalance frequency label.
    cov_lookback : int
        Number of return observations used for covariance estimation.
    mu_lookback : int
        Number of return observations used for expected-return estimation.
    annualization : float
        Annualization factor for daily or periodic returns.
    risk_free_rate_annual : float
        Annual risk-free rate used in risk-aversion and reporting calculations.
    tau : float
        Black-Litterman prior uncertainty scale.
    ewma_lambda : float
        Decay parameter used when EWMA covariance is selected.
    delta_fallback : float
        Fallback risk-aversion value when it cannot be inferred from benchmark
        returns.
    delta_min, delta_max : float
        Lower and upper bounds applied to inferred risk aversion.
    transaction_cost_bps : float
        Transaction-cost assumption in basis points.
    max_weight, min_weight : float
        Per-asset portfolio weight bounds.
    active_weight_limit : float
        Default absolute active-weight limit relative to the benchmark.
    active_weight_relaxed : float
        Relaxed active-weight limit used when the first optimization attempt is
        infeasible.
    max_selected_views : int
        Maximum number of views selected at each rebalance.
    redundancy_similarity : float
        Exposure-cosine similarity threshold used to reject redundant views.
    max_same_direction : int
        Maximum number of selected views with the same broad risk orientation.
    confidence_mode : str
        Confidence rule, such as learned, fixed, or haircut.
    use_learned_q : bool
        Whether view tilts are adjusted using historical payoff evidence.
    full_view_covariance : bool
        Whether the full view covariance is retained when constructing omega.
    empirical_omega_rescale : bool
        Whether view uncertainty is blended with empirical payoff variance.
    posterior_mu_clip : float
        Absolute annualized bound applied to posterior expected returns.
    te_gamma_fallback : float
        Tracking-error penalty used in the final relaxed optimizer fallback.

    Notes
    -----
    This object is designed to make the Black-Litterman workflow reproducible.
    Changing these fields changes both portfolio weights and diagnostics, so
    settings should be logged with any reported backtest.
    """

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
    """Snapshot of market information available at one rebalance date.

    A market state bundles the historical return window, signal-return window,
    current signal table, and derived scalar state values used by view-generation
    rules. It is passed to view functions so each rule evaluates information
    available up to the decision date only.

    Attributes
    ----------
    date : pandas.Timestamp
        Rebalance or decision date represented by this state.
    signal_table : pandas.DataFrame
        Cross-sectional signal table, typically containing momentum, trend,
        volatility, drawdown, and composite score fields.
    values : dict
        Derived market-state scalars, such as breadth, relative-performance
        spreads, volatility states, and correlation states.
    returns : pandas.DataFrame
        Asset return history through ``date``.
    signal_returns : pandas.DataFrame
        Return history for signal assets through ``date``.

    Notes
    -----
    The object is intentionally lightweight and immutable by convention. View
    functions should read from it but should not mutate it in place.
    """

    date: pd.Timestamp
    signal_table: pd.DataFrame
    values: dict[str, Any]
    returns: pd.DataFrame
    signal_returns: pd.DataFrame


@dataclass
class BLRun:
    """Container for a complete learned-confidence Black-Litterman run.

    The object stores portfolio weights, generated views, selected views,
    view-selection diagnostics, payoff history, reliability statistics,
    posterior-return diagnostics, confidence diagnostics, prior/posterior
    expected returns, date-indexed covariance matrices, fallback flags, and
    metadata needed to audit a run.

    Attributes
    ----------
    weights : pandas.DataFrame
        Rebalance-date portfolio weights.
    candidate_view_log : pandas.DataFrame
        All candidate views generated before selection.
    selected_view_log : pandas.DataFrame
        Views retained after confidence, redundancy, and crowding filters.
    selection_log : pandas.DataFrame
        View-selection diagnostics, including kept/rejected flags and reasons.
    payoff_history : pandas.DataFrame
        Historical payoff records used to learn view reliability.
    reliability_summary : pandas.DataFrame
        Aggregated reliability metrics by view family.
    posterior_log : pandas.DataFrame
        Diagnostics from posterior-return construction.
    confidence_log : pandas.DataFrame
        Per-view confidence and evidence diagnostics.
    state_log : pandas.DataFrame
        Market-state diagnostics at each rebalance date.
    prior_mu : pandas.DataFrame
        Prior expected returns by date and asset.
    posterior_mu : pandas.DataFrame
        Posterior expected returns by date and asset.
    covariance_by_date : dict
        Mapping from rebalance date to covariance matrix used on that date.
    fallback_flags : pandas.Series
        Indicator for optimizer or posterior fallback events by date.
    metadata : dict
        Additional run metadata, including assets, settings, view settings, and
        benchmark weights.

    Methods
    -------
    as_dict()
        Return the run contents as a plain dictionary.
    __getitem__(key)
        Dictionary-style access to stored artifacts.

    Notes
    -----
    The result object is intentionally richer than a simple weight table because
    the workflow is evidence-driven. The logs allow the user to explain why a
    view was selected, how confident the model was, and whether optimization
    fell back to a benchmark-like allocation.
    """

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
    """Infer Black-Litterman equilibrium prior returns from a benchmark portfolio.

    The prior is computed as ``delta * Sigma * w_benchmark``. If ``delta`` is not
    supplied, the function estimates risk aversion from benchmark excess return
    and variance over a historical return window, then clips the estimate to the
    configured bounds.

    Parameters
    ----------
    cov_ann : pandas.DataFrame
        Annualized covariance matrix indexed and columned by asset name.
    benchmark_weights : pandas.Series
        Benchmark or market-cap weights. The weights are reindexed to the
        covariance assets and normalized. If the supplied vector sums to zero,
        an equal-weight benchmark is used.
    returns_window : pandas.DataFrame, optional
        Historical asset returns used to infer benchmark risk aversion when
        ``delta`` is not provided.
    settings : BLSettings, optional
        Black-Litterman settings controlling annualization, risk-free rate,
        fallback risk aversion, and clipping bounds.
    delta : float, optional
        Explicit risk-aversion coefficient. When provided, no historical
        inference is performed.

    Returns
    -------
    prior : pandas.Series
        Annualized prior expected returns indexed by asset.
    delta : float
        Risk-aversion coefficient used to build the prior.

    Notes
    -----
    The prior is expressed in the same annualized return units as the covariance
    matrix. The function is robust to missing benchmark names by reindexing and
    filling absent assets with zero weight.
    """

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
    """Compute Black-Litterman posterior expected returns and covariance.

    The function combines a prior expected-return vector, prior covariance
    matrix, and linear views ``P mu = q`` with view uncertainty ``omega``. It
    returns posterior expected returns, posterior covariance, and diagnostics
    about view count, conditioning, clipping, and fallback status.

    Parameters
    ----------
    prior_mu : pandas.Series
        Annualized prior expected returns indexed by asset.
    cov_ann : pandas.DataFrame
        Annualized prior covariance matrix.
    view_p : numpy.ndarray
        View loading matrix with shape ``(n_views, n_assets)``.
    view_q : numpy.ndarray
        View target vector with shape ``(n_views,)``. Values are expressed in
        annualized return-spread units compatible with ``prior_mu``.
    omega : numpy.ndarray
        View uncertainty covariance matrix with shape ``(n_views, n_views)``.
    tau : float, default=0.05
        Prior uncertainty scaling parameter.
    mu_clip : float, default=0.30
        Absolute bound applied to posterior expected returns.

    Returns
    -------
    posterior_mu : pandas.Series
        Posterior annualized expected returns indexed like ``prior_mu``.
    posterior_cov : pandas.DataFrame
        Posterior covariance matrix indexed and columned like ``cov_ann``.
    diagnostics : dict
        Diagnostics including number of views, condition number, posterior shift,
        clipping flag, no-view flag, and failure flag.

    Notes
    -----
    If no views are supplied, the function returns the prior unchanged. If the
    linear solve fails, the function also returns the prior and marks the
    diagnostic failure flag rather than raising.
    """

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
    """Build a Black-Litterman view-uncertainty matrix from confidence scores.

    The function converts each confidence value into a view-variance scale and
    applies it to the covariance-implied variance of each view exposure. Higher
    confidence values produce lower uncertainty; lower confidence values produce
    higher uncertainty. Optionally, the diagonal can be blended with empirical
    view payoff variance.

    Parameters
    ----------
    view_p : numpy.ndarray
        View exposure matrix with shape ``(n_views, n_assets)``.
    cov_ann : pandas.DataFrame or numpy.ndarray
        Annualized asset covariance matrix.
    confidence_values : sequence of float
        Confidence values for each view. Values should be positive and are
        typically bounded between zero and one.
    tau : float, default=0.05
        Prior uncertainty scale used in Black-Litterman.
    active_view_rows : sequence of mappings, optional
        View metadata used for empirical variance rescaling.
    history : pandas.DataFrame, optional
        Historical view payoff table.
    current_date : pandas.Timestamp or str, optional
        Decision date used when selecting historical evidence.
    empirical_rescale : bool, default=False
        If True, blend the model-implied diagonal with empirical view payoff
        variance when enough historical observations are available.

    Returns
    -------
    omega : numpy.ndarray
        Positive-stabilized view uncertainty covariance matrix.
    view_var : numpy.ndarray
        Covariance-implied variance of each view exposure before confidence
        scaling.

    Notes
    -----
    The returned matrix is symmetrized and given a small diagonal jitter for
    numerical stability. If the full rescaling step fails, a diagonal fallback
    omega is returned.
    """

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
    """Optimize portfolio weights from posterior Black-Litterman inputs.

    The optimizer maximizes posterior mean-variance utility subject to benchmark
    active-weight controls, box constraints, optional sleeve constraints, and
    turnover costs. It attempts a sequence of increasingly relaxed optimization
    problems before falling back to the previous or benchmark weights.

    Parameters
    ----------
    post_mu : pandas.Series
        Posterior annualized expected returns.
    post_cov : pandas.DataFrame
        Posterior annualized covariance matrix.
    benchmark_weights : pandas.Series
        Benchmark weights used for active constraints and fallback.
    previous_weights : pandas.Series, optional
        Previous rebalance weights used for turnover penalties and fallback.
    risk_aversion : float, optional
        Risk-aversion coefficient. If omitted, the settings fallback is used.
    settings : BLSettings, optional
        Optimization and constraint settings.
    constraints : mapping, optional
        Additional constraints such as asset-specific caps or sleeve constraints.

    Returns
    -------
    weights : pandas.Series
        Normalized portfolio weights indexed by asset.
    fallback : bool
        True if all optimization attempts failed and fallback weights were used.
    info : dict
        Details of the successful attempt or fallback mode, including active
        limit, sleeve-constraint use, and tracking-error penalty.

    Notes
    -----
    The function is deliberately defensive. It relaxes active and sleeve
    constraints before abandoning the optimizer, which is useful in historical
    walk-forward runs where a small number of dates may be numerically difficult.
    """

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
    """Build the market-state object used by view-generation rules.

    The function truncates asset returns and signal returns through a decision
    date, computes the current signal table and derived state values, and returns
    a ``MarketState`` object. It never uses observations after the requested date.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel.
    signal_returns : pandas.DataFrame
        Return panel used to compute cross-asset signals and state variables.
    date : pandas.Timestamp or str
        Decision date. The latest observations on or before this date are used.
    roles : mapping
        Asset-role configuration used by signal and view functions.
    settings : BLSettings, optional
        Black-Litterman settings, mainly for annualization.
    view_settings : ViewSettings, optional
        Signal and view-generation settings.

    Returns
    -------
    MarketState
        State object containing the decision date, signal table, derived state
        values, and return histories through the decision date.

    Notes
    -----
    If the decision date is before either input history starts, an empty state is
    returned rather than raising.
    """

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
    """Run the learned-confidence Black-Litterman allocation workflow.

    The workflow generates candidate views at each rebalance date, scores their
    historical reliability, selects a diversified subset, converts selected views
    into Black-Litterman view matrices, builds confidence-scaled view uncertainty,
    computes posterior expected returns, and optimizes portfolio weights.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel used for covariance estimation, prior inference,
        payoff evaluation, and portfolio backtesting.
    signal_returns : pandas.DataFrame
        Return panel used by signal and view-generation rules.
    rebalance_dates : sequence of pandas.Timestamp or str
        Candidate rebalance dates.
    benchmark_weights : pandas.Series or mapping
        Benchmark weights used for prior construction, active constraints, and
        comparison.
    roles : mapping
        Asset-role definitions used by the view functions.
    view_functions : sequence of callable
        Functions that accept ``(MarketState, roles, ViewSettings)`` and return
        one or more view records.
    view_settings : ViewSettings
        Settings controlling signal construction, q caps, view horizon, and
        family display names.
    settings : BLSettings, optional
        Black-Litterman settings controlling covariance, priors, confidence,
        selection, posterior, and optimization behavior.
    constraints : mapping, optional
        Optional portfolio constraints such as sleeve constraints or asset caps.

    Returns
    -------
    BLRun
        Complete run artifact containing weights, view logs, confidence logs,
        payoff history, posterior diagnostics, covariance cache, fallback flags,
        and metadata.

    Raises
    ------
    InputError
        If fewer than two rebalance dates are supplied or input histories cannot
        support the requested workflow.

    Notes
    -----
    This function is intentionally audit-heavy. The returned logs are designed to
    explain why each view was generated, selected, scaled, or rejected, and how
    those views changed expected returns and portfolio weights.
    """

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
    """Format view records into a compact display table.

    Parameters
    ----------
    view_rows : sequence of mappings or pandas.DataFrame
        View records containing family, name, state, strength, q, confidence,
        and long/short asset lists.

    Returns
    -------
    pandas.DataFrame
        Human-readable table with family, view name, state, strength, raw and
        final q values, confidence, long assets, and short assets.

    Notes
    -----
    This helper is intended for reporting. It does not recompute views or check
    whether the views are active or selected.
    """

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
    """Compare strategy backtests against a named benchmark.

    The function combines standard performance metrics, turnover/cost statistics,
    and active metrics relative to the benchmark into a single comparison table.

    Parameters
    ----------
    results : mapping
        Mapping from strategy name to backtest result object or compatible
        dictionary.
    benchmark_name : str
        Name of the benchmark strategy in ``results``.
    rf_daily : float, default=0.0
        Daily risk-free rate used for Sharpe calculations.
    annualization : float, default=252.0
        Annualization factor for return and volatility metrics.

    Returns
    -------
    pandas.DataFrame
        Strategy-indexed table containing net CAGR, volatility, Sharpe, maximum
        drawdown, Calmar, turnover, cost drag, active return, tracking error,
        information ratio, hit rate versus benchmark, correlation, and beta.

    Notes
    -----
    Benchmark active return and tracking error are set to zero by construction.
    Other active metrics are computed only for non-benchmark strategies.
    """

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
    """Summarize one strategy's active performance versus a benchmark.

    Parameters
    ----------
    strategy_result : BacktestResult or mapping
        Strategy backtest artifact.
    benchmark_result : BacktestResult or mapping
        Benchmark backtest artifact.
    strategy_name : str, default="Learned-Confidence BL"
        Label used for the output row.
    benchmark_name : str, default="Benchmark"
        Label stored in the benchmark column.
    annualization : float, default=252.0
        Annualization factor for active-return and tracking-error metrics.

    Returns
    -------
    pandas.DataFrame
        One-row table containing active return, tracking error, information
        ratio, hit rate, correlation, beta, active drawdown, average active
        weight distance, and labels.

    Notes
    -----
    Active returns are computed by aligning strategy and benchmark net returns
    on the benchmark return index.
    """

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
    """Extract a compact view-family reliability table from a Black-Litterman run.

    Parameters
    ----------
    bl_run : object
        Object with a ``reliability_summary`` attribute, typically a ``BLRun``.

    Returns
    -------
    pandas.DataFrame
        Reliability table with available columns such as display name, candidate
        count, selected count, selected share, hit rate, average payoff, payoff
        volatility, payoff information ratio, average q, average absolute q, and
        average confidence.

    Notes
    -----
    If no reliability summary is available, an empty DataFrame is returned.
    """

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
