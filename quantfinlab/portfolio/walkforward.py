from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.backtest.portfolio import run_rebalanced_portfolio_backtest
from quantfinlab.common.contracts import BacktestResult
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio import (
    covariance,
    expected_returns,
    optimizers as optimizer_module,
    universe,
)


@dataclass
class WalkForwardGridResult:
    """Container for a complete walk-forward strategy-grid run.

    Attributes
    ----------
    results : pandas.DataFrame
        Strategy performance summary.
    nav : pandas.DataFrame
        Net asset value paths by strategy.
    returns : pandas.DataFrame
        Net return paths by strategy.
    weights : dict of pandas.DataFrame
        Strategy weight frames keyed by strategy name.
    turnover : pandas.DataFrame
        Turnover paths by strategy.
    costs : pandas.DataFrame
        Cost paths by strategy.
    diagnostics : pandas.DataFrame
        Strategy-level diagnostics such as optimizer family and fallback counts.
    cache : dict
        Rebalance-state cache used by the run.
    backtests : dict
        Raw backtest result objects by strategy.
    metadata : dict
        Run configuration and derived metadata.

    Methods
    -------
    as_dict()
        Return all stored artifacts as a plain dictionary.
    __getitem__(key)
        Dictionary-style access to stored artifacts.

    Notes
    -----
    The object stores both summary outputs and raw artifacts so users can inspect
    individual strategies without rerunning the grid.
    """

    results: pd.DataFrame
    nav: pd.DataFrame
    returns: pd.DataFrame
    weights: dict[str, pd.DataFrame]
    turnover: pd.DataFrame
    costs: pd.DataFrame
    diagnostics: pd.DataFrame
    cache: dict[pd.Timestamp, dict[str, Any]]
    backtests: dict[str, BacktestResult]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "results": self.results,
            "nav": self.nav,
            "returns": self.returns,
            "weights": self.weights,
            "turnover": self.turnover,
            "costs": self.costs,
            "diagnostics": self.diagnostics,
            "cache": self.cache,
            "backtests": self.backtests,
            "metadata": self.metadata,
        }

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


DEFAULT_BLEND_BY_OPTIMIZER = {
    "EW": 0.00,
    "MinVar": 0.20,
    "MV": 0.15,
    "RidgeMV": 0.15,
    "MaxSharpe": 0.10,
    "FrontierGrid": 0.10,
}


def rebalances_per_year(rebalance_dates: Sequence[pd.Timestamp | str]) -> float:
    idx = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates)))
    if len(idx) < 2:
        return 1.0
    per_year = pd.Series(1, index=idx).resample("YE").sum()
    return float(per_year.median())


def _resolve_pos(index: pd.DatetimeIndex, dt: pd.Timestamp) -> int | None:
    if dt in index:
        pos = index.get_loc(dt)
        if isinstance(pos, slice):
            return int(pos.stop - 1)
        return int(pos)
    pos = int(index.searchsorted(dt, side="right")) - 1
    return pos if pos >= 0 else None


def _call_cov_model(fn: Callable, window: pd.DataFrame, *, annualization: float, ewma_lambda: float):
    try:
        return fn(window, annualization=annualization, lam=ewma_lambda, return_df=False)
    except TypeError:
        try:
            return fn(window, annualization=annualization, return_df=False)
        except TypeError:
            return fn(window)


def _call_mu_model(
    fn: Callable,
    window: pd.DataFrame,
    *,
    cov_ann,
    model_name: str,
    mu_kwargs: dict[str, Any],
):
    try:
        out = fn(window, cov_ann=cov_ann, return_info=True, return_series=True, **mu_kwargs)
    except TypeError:
        out = expected_returns.build_mu_excess_ann(
            window,
            cov_ann=cov_ann,
            mu_model=model_name,
            return_info=True,
            return_series=True,
            **mu_kwargs,
        )
    if isinstance(out, tuple):
        mu, info = out
    else:
        mu, info = out, {"mu_model": model_name, "shrinkage_intensity": np.nan, "invalid_values": 0}
    if not isinstance(mu, pd.Series):
        mu = pd.Series(np.asarray(mu, dtype=float), index=window.columns, dtype=float)
    return mu.reindex(window.columns).astype(float), dict(info)


def build_rebalance_state_cache(
    *,
    returns: pd.DataFrame,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    rebalance_dates: Sequence[pd.Timestamp | str],
    universe_by_date: Mapping[pd.Timestamp, Any] | None = None,
    cov_models: Mapping[str, Callable] | None = None,
    mu_models: Mapping[str, Callable] | None = None,
    cov_lookback: int = 252,
    mu_lookback: int = 504,
    min_cov_observations: int | None = None,
    min_mu_observations: int = 252,
    top_n: int = 100,
    liquidity_lookback: int = 252,
    min_listing_days: int = 252,
    min_obs: int = 252,
    min_price: float | None = None,
    annualization: float = 252.0,
    ewma_lambda: float = 0.94,
    momentum_mode: str = "6-1",
    rf_daily: float | pd.Series = 0.0,
    target_sharpe_ann: float = 0.80,
    mu_cap_ann: float = 0.30,
    winsor_lo: float = 0.05,
    winsor_hi: float = 0.95,
) -> dict[pd.Timestamp, dict[str, Any]]:
    """Build per-rebalance model states for a walk-forward portfolio experiment.

    For each rebalance date, the function selects or receives an eligible
    universe, slices covariance and expected-return windows, estimates all
    configured covariance models, builds all configured expected-return models,
    and stores liquidity diagnostics.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel.
    close : pandas.DataFrame, optional
        Close-price panel used for universe selection and return-window
        construction.
    volume : pandas.DataFrame, optional
        Volume panel used for liquidity filtering.
    rebalance_dates : sequence of pandas.Timestamp or str
        Candidate rebalance dates.
    universe_by_date : mapping, optional
        Precomputed universe mapping. If omitted, close and volume panels are
        used to build one.
    cov_models : mapping, optional
        Mapping from covariance model name to callable.
    mu_models : mapping, optional
        Mapping from expected-return model name to callable.
    cov_lookback : int, default=252
        Lookback length for covariance estimation.
    mu_lookback : int, default=504
        Lookback length for expected-return estimation.
    min_cov_observations : int, optional
        Minimum clean observations required for covariance windows.
    min_mu_observations : int, default=252
        Minimum clean observations required for expected-return windows.
    top_n : int, default=100
        Number of liquid names selected when building universes internally.
    liquidity_lookback : int, default=252
        Average-dollar-volume lookback.
    min_listing_days : int, default=252
        Seasoning requirement for universe selection.
    min_obs : int, default=252
        Minimum valid dollar-volume observations.
    min_price : float, optional
        Minimum price filter for universe selection.
    annualization : float, default=252.0
        Annualization factor.
    ewma_lambda : float, default=0.94
        EWMA covariance decay parameter.
    momentum_mode : str, default="6-1"
        Momentum convention for expected-return models.
    rf_daily : float or pandas.Series, default=0.0
        Daily risk-free rate for expected-return estimation.
    target_sharpe_ann : float, default=0.80
        Target Sharpe used when scaling expected-return directions.
    mu_cap_ann : float, default=0.30
        Absolute cap for annualized expected returns.
    winsor_lo, winsor_hi : float
        Winsorization quantiles for expected-return scaling.

    Returns
    -------
    dict
        Mapping from rebalance date to state dictionaries containing tickers,
        covariance and expected-return windows, raw signals, expected-return
        maps, covariance maps, model diagnostics, and average dollar volume.

    Raises
    ------
    InputError
        If returns are empty or universe information cannot be built.

    Notes
    -----
    The cache is designed to separate expensive state construction from strategy
    evaluation. Multiple optimizers can reuse the same cache without recomputing
    covariance and expected-return estimates.
    """

    if returns.empty:
        raise InputError("returns is empty.")
    R_all = returns.copy()
    R_all.index = pd.to_datetime(R_all.index)
    R_all = R_all.sort_index()

    close_panel = close
    if close_panel is not None:
        close_panel = close_panel.copy()
        close_panel.index = pd.to_datetime(close_panel.index)
        close_panel = close_panel.sort_index()

    if cov_models is None:
        cov_models = {
            "Sample": covariance.sample_covariance,
            "LedoitWolf": covariance.ledoit_wolf_covariance,
            "OAS": covariance.oas_covariance,
            "EWMA": covariance.ewma_covariance,
        }
    if mu_models is None:
        mu_models = {
            "Momentum": expected_returns.momentum_mu,
            "BayesStein": expected_returns.bayes_stein_mu,
            "BayesSteinMomentum": expected_returns.bayes_stein_momentum_mu,
        }
    if universe_by_date is None:
        if close is None or volume is None:
            raise InputError("universe_by_date or close/volume panels are required.")
        universe_by_date = universe.build_liquid_universe_by_date(
            close=close,
            volume=volume,
            rebalance_dates=rebalance_dates,
            top_n=top_n,
            liquidity_lookback=liquidity_lookback,
            min_listing_days=min_listing_days,
            min_obs=min_obs,
            min_price=min_price,
        )

    min_cov = int(min_cov_observations if min_cov_observations is not None else cov_lookback - 1)
    mu_kwargs = {
        "mode": momentum_mode,
        "rf_daily": rf_daily,
        "target_sharpe_ann": target_sharpe_ann,
        "mu_cap_ann": mu_cap_ann,
        "winsor_lo": winsor_lo,
        "winsor_hi": winsor_hi,
        "annualization": annualization,
    }

    cache: dict[pd.Timestamp, dict[str, Any]] = {}
    idx_model = pd.DatetimeIndex(close_panel.index if close_panel is not None else R_all.index)

    for raw_dt in pd.to_datetime(list(rebalance_dates)):
        dt = pd.Timestamp(raw_dt)
        urec = universe_by_date.get(dt) if isinstance(universe_by_date, Mapping) else None
        if urec is None:
            continue
        tickers = list(urec.get("tickers", urec) if isinstance(urec, Mapping) else urec)
        if len(tickers) < 2:
            continue

        pos = _resolve_pos(idx_model, dt)
        if pos is None or pos < int(cov_lookback):
            continue

        cov_start = max(0, pos - int(cov_lookback))
        mu_start = max(0, pos - int(mu_lookback))
        if close_panel is not None:
            close_cov = close_panel[tickers].iloc[cov_start:pos]
            close_mu = close_panel[tickers].iloc[mu_start:pos]
            window_cov = close_cov.pct_change(fill_method=None).iloc[1:]
            window_mu = close_mu.pct_change(fill_method=None).iloc[1:]
        else:
            r_pos = _resolve_pos(pd.DatetimeIndex(R_all.index), dt)
            if r_pos is None or r_pos < int(cov_lookback):
                continue
            window_cov = R_all[tickers].iloc[max(0, r_pos - int(cov_lookback)) : r_pos]
            window_mu = R_all[tickers].iloc[max(0, r_pos - int(mu_lookback)) : r_pos]

        window_cov = window_cov.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        window_mu = window_mu.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if window_cov.shape[0] < min_cov or window_cov.shape[1] < 2:
            continue
        if window_mu.shape[0] < int(min_mu_observations) or window_mu.shape[1] < 2:
            continue

        active_tickers = window_cov.columns.tolist()
        window_mu = window_mu.reindex(columns=active_tickers)
        if window_mu.isna().any().any():
            window_mu = window_mu.dropna(axis=0, how="any")
        if window_mu.shape[0] < int(min_mu_observations):
            continue

        cov_ann_map: dict[str, np.ndarray] = {}
        for cov_key, cov_fn in cov_models.items():
            cov_label = covariance.normalize_covariance_method(cov_key)
            cov_ann_map[cov_label] = np.asarray(
                _call_cov_model(cov_fn, window_cov, annualization=annualization, ewma_lambda=ewma_lambda),
                dtype=float,
            )

        mu_raw_map = {
            "Momentum": expected_returns.momentum_score_from_returns(window_mu, mode=momentum_mode),
        }
        mu_ann_map: dict[str, dict[str, pd.Series]] = {}
        mu_info_map: dict[str, dict[str, dict[str, Any]]] = {}
        for cov_key, cov_ann in cov_ann_map.items():
            mu_ann_map[cov_key] = {}
            mu_info_map[cov_key] = {}
            for mu_key, mu_fn in mu_models.items():
                mu_label = expected_returns.normalize_mu_model(mu_key)
                mu, info = _call_mu_model(
                    mu_fn,
                    window_mu,
                    cov_ann=cov_ann,
                    model_name=mu_label,
                    mu_kwargs=mu_kwargs,
                )
                mu_ann_map[cov_key][mu_label] = mu.reindex(active_tickers).astype(float)
                mu_info_map[cov_key][mu_label] = info

        avg_dv = None
        if isinstance(urec, Mapping):
            avg_dv = urec.get("avg_dollar_volume")
        if avg_dv is None:
            avg_dv = pd.Series(dtype=float)
        avg_dv = pd.Series(avg_dv, dtype=float).reindex(active_tickers)

        cache[dt] = {
            "tickers": active_tickers,
            "R_cov": window_cov.astype(np.float32),
            "R_mu": window_mu.astype(np.float32),
            "mu_raw_map": mu_raw_map,
            "mu_ann_map": mu_ann_map,
            "mu_info_map": mu_info_map,
            "cov_ann_map": cov_ann_map,
            "avg_dollar_volume": avg_dv.astype(float),
        }

    return cache


def build_universe_state_cache(
    *,
    returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    universe_by_date: Mapping[pd.Timestamp, Any],
) -> dict[pd.Timestamp, dict[str, Any]]:
    """Build the ticker-only state needed by an equal-weight walk-forward run."""

    if returns.empty:
        raise InputError("returns is empty.")

    return_dates = pd.DatetimeIndex(pd.to_datetime(returns.index))
    return_tickers = {str(ticker) for ticker in returns.columns}
    universe = {pd.Timestamp(date): value for date, value in universe_by_date.items()}
    cache: dict[pd.Timestamp, dict[str, Any]] = {}

    for raw_date in pd.to_datetime(list(rebalance_dates)):
        date = pd.Timestamp(raw_date)
        if date not in return_dates or date not in universe:
            continue
        value = universe[date]
        tickers = value.get("tickers", value) if isinstance(value, Mapping) else value
        tickers = [
            str(ticker)
            for ticker in tickers
            if str(ticker) in return_tickers
        ]
        if len(tickers) >= 2:
            cache[date] = {"tickers": tickers}
    return cache


def _canonical_optimizer(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "ew": "EW",
        "equalweight": "EW",
        "minvar": "MinVar",
        "minimumvariance": "MinVar",
        "mv": "MV",
        "meanvariance": "MV",
        "ridgemv": "RidgeMV",
        "ridgemeanvariance": "RidgeMV",
        "maxsharpe": "MaxSharpe",
        "maxsharpeslsqp": "MaxSharpe",
        "frontiergrid": "FrontierGrid",
    }
    return aliases.get(key, str(name))


def build_strategy_grid(
    *,
    mu_models: Mapping[str, Callable],
    cov_models: Mapping[str, Callable],
    optimizers: Mapping[str, Callable],
    include_ridge: bool = True,
) -> list[dict[str, Any]]:
    """Build the default Cartesian strategy specification grid.

    The function combines expected-return models, covariance models, and
    optimizers into named strategy specifications used by the walk-forward
    engine.

    Parameters
    ----------
    mu_models : mapping
        Expected-return model callables keyed by model name.
    cov_models : mapping
        Covariance model callables keyed by model name.
    optimizers : mapping
        Optimizer callables keyed by optimizer name.
    include_ridge : bool, default=True
        Whether to include RidgeMV strategy variants.

    Returns
    -------
    list of dict
        Strategy specifications. Each specification contains name, optimizer
        family, covariance model, expected-return model, and optimizer function.

    Raises
    ------
    InputError
        If generated strategy names are not unique.

    Notes
    -----
    Equal-weight and minimum-variance strategies are added without expected
    return models. Mean-variance, RidgeMV, and MaxSharpe strategies are expanded
    across covariance and expected-return model combinations.
    """

    cov_keys = [covariance.normalize_covariance_method(k) for k in cov_models]
    mu_keys = [expected_returns.normalize_mu_model(k) for k in mu_models]
    specs: list[dict[str, Any]] = []
    opt_map = {_canonical_optimizer(k): v for k, v in optimizers.items()}

    if "EW" in opt_map:
        specs.append({"name": "EW", "optimizer": "EW", "cov_model": None, "mu_model": None, "fn": opt_map["EW"]})
    if "MinVar" in opt_map:
        for cov_key in cov_keys:
            specs.append({"name": f"MinVar ({cov_key})", "optimizer": "MinVar", "cov_model": cov_key, "mu_model": None, "fn": opt_map["MinVar"]})
    if "MV" in opt_map:
        for cov_key in cov_keys:
            for mu_key in mu_keys:
                specs.append({"name": f"MV ({cov_key}, {mu_key})", "optimizer": "MV", "cov_model": cov_key, "mu_model": mu_key, "fn": opt_map["MV"]})
    if include_ridge and "RidgeMV" in opt_map:
        for cov_key in cov_keys:
            for mu_key in mu_keys:
                specs.append({"name": f"RidgeMV ({cov_key}, {mu_key})", "optimizer": "RidgeMV", "cov_model": cov_key, "mu_model": mu_key, "fn": opt_map["RidgeMV"]})
    if "MaxSharpe" in opt_map:
        for cov_key in cov_keys:
            for mu_key in mu_keys:
                specs.append({"name": f"MaxSharpe ({cov_key}, {mu_key})", "optimizer": "MaxSharpe", "cov_model": cov_key, "mu_model": mu_key, "fn": opt_map["MaxSharpe"]})

    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise InputError("Strategy names must be unique.")
    return specs


def _default_strategy_name(family: str, cov_key: str | None, mu_key: str | None) -> str:
    if family == "EW":
        return "EW"
    if family == "MinVar":
        return f"MinVar ({cov_key})"
    if family == "FrontierGrid":
        return f"MaxSharpe (FrontierGrid) ({cov_key}, {mu_key})"
    if family in {"MV", "RidgeMV", "MaxSharpe"}:
        return f"{family} ({cov_key}, {mu_key})"
    return family


def _normalize_strategy_specs(
    strategy_specs: Sequence[Mapping[str, Any]],
    optimizers: Mapping[str, Callable],
) -> list[dict[str, Any]]:
    opt_map = {_canonical_optimizer(k): v for k, v in optimizers.items()}
    opt_map.setdefault("FrontierGrid", optimizer_module.max_sharpe_frontier_grid)

    specs: list[dict[str, Any]] = []
    for raw in strategy_specs:
        family = _canonical_optimizer(str(raw.get("optimizer", "")))
        if family not in opt_map:
            raise InputError(f"Unknown optimizer in strategy spec: {raw.get('optimizer')!r}.")

        cov_key = raw.get("cov_model")
        mu_key = raw.get("mu_model")
        cov_label = None if cov_key in (None, "") else covariance.normalize_covariance_method(str(cov_key))
        mu_label = None if mu_key in (None, "") else expected_returns.normalize_mu_model(str(mu_key))

        if family in {"MinVar", "MV", "RidgeMV", "MaxSharpe", "FrontierGrid"} and cov_label is None:
            raise InputError(f"Strategy spec for {family} requires cov_model.")
        if family in {"MV", "RidgeMV", "MaxSharpe", "FrontierGrid"} and mu_label is None:
            raise InputError(f"Strategy spec for {family} requires mu_model.")

        specs.append(
            {
                "name": str(raw.get("name") or _default_strategy_name(family, cov_label, mu_label)),
                "optimizer": family,
                "cov_model": cov_label,
                "mu_model": mu_label,
                "fn": raw.get("fn") or opt_map[family],
            }
        )

    names = [s["name"] for s in specs]
    if len(names) != len(set(names)):
        raise InputError("Strategy names must be unique.")
    return specs


def _with_metadata(result: BacktestResult, metadata: Mapping[str, Any]) -> BacktestResult:
    meta = dict(result.metadata or {})
    meta.update(metadata)
    return BacktestResult(
        gross_values=result.gross_values,
        net_values=result.net_values,
        gross_returns=result.gross_returns,
        net_returns=result.net_returns,
        weights=result.weights,
        turnover=result.turnover,
        costs=result.costs,
        fallbacks=result.fallbacks,
        metadata=meta,
    )


def _strategy_weight_fn(
    spec: Mapping[str, Any],
    *,
    optimizer_params: Mapping[str, Mapping[str, Any]],
    w_min: float | None,
    w_max: float | None,
    long_only: bool,
    turnover_penalty_bps: float,
    kappa_target_annual: float,
    solver_order: Sequence[str] | None,
):
    family = spec["optimizer"]
    fn = spec["fn"]
    cov_key = spec.get("cov_model")
    mu_key = spec.get("mu_model")
    params = dict(optimizer_params.get(family, {}))

    def _fn(dt: pd.Timestamp, state: Mapping[str, Any], w_prev: np.ndarray):
        tickers = [str(x) for x in state.get("tickers", [])]
        if family == "EW":
            return fn(tickers, w_min=w_min, w_max=w_max, long_only=long_only)

        cov_ann = state["cov_ann_map"][cov_key]
        common_kwargs = {
            "w_prev": w_prev,
            "w_min": w_min,
            "w_max": w_max,
            "long_only": long_only,
            "turnover_penalty_bps": turnover_penalty_bps,
            "kappa_target_annual": kappa_target_annual,
        }
        if family == "MinVar":
            return fn(cov_ann=cov_ann, solver_order=solver_order, **common_kwargs, **params)

        mu = state["mu_ann_map"][cov_key][mu_key].reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        if family in {"MV", "RidgeMV"}:
            return fn(
                mu_excess_ann=mu,
                cov_ann=cov_ann,
                solver_order=solver_order,
                **common_kwargs,
                **params,
            )
        if family in {"MaxSharpe", "FrontierGrid"}:
            try:
                return fn(
                    mu_excess_ann=mu,
                    cov_ann=cov_ann,
                    solver_order=solver_order,
                    **common_kwargs,
                    **params,
                )
            except TypeError:
                return fn(
                    mu_excess_ann=mu,
                    cov_ann=cov_ann,
                    **common_kwargs,
                    **params,
                )
        return None

    return _fn


def _collect_outputs(
    backtests: Mapping[str, BacktestResult],
    *,
    rf_daily: float | pd.Series,
    annualization: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from quantfinlab.portfolio import selection

    nav = pd.concat({name: res.net_values for name, res in backtests.items()}, axis=1) if backtests else pd.DataFrame()
    ret = pd.concat({name: res.net_returns for name, res in backtests.items()}, axis=1) if backtests else pd.DataFrame()
    turnover = pd.concat({name: res.turnover for name, res in backtests.items()}, axis=1) if backtests else pd.DataFrame()
    costs = pd.concat({name: res.costs for name, res in backtests.items()}, axis=1) if backtests else pd.DataFrame()
    summary = selection.build_strategy_summary(backtests, rf_daily=rf_daily, annualization=annualization)
    return summary, nav, ret, turnover, costs


def run_walkforward_grid(
    *,
    returns: pd.DataFrame,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    rebalance_dates: Sequence[pd.Timestamp | str],
    universe_by_date: Mapping[pd.Timestamp, Any] | None = None,
    mu_models: Mapping[str, Callable] | None = None,
    cov_models: Mapping[str, Callable] | None = None,
    optimizers: Mapping[str, Callable] | None = None,
    strategy_specs: Sequence[Mapping[str, Any]] | None = None,
    cache: dict[pd.Timestamp, dict[str, Any]] | None = None,
    cov_lookback: int = 252,
    mu_lookback: int = 504,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
    long_only: bool = True,
    trading_cost_bps: float = 10.0,
    turnover_penalty_bps: float = 10.0,
    fallback: str = "equal",
    solver_order: Sequence[str] | None = ("OSQP", "ECOS", "SCS"),
    optimizer_params: Mapping[str, Mapping[str, Any]] | None = None,
    blend_by_optimizer: Mapping[str, float] | None = None,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = 252.0,
    **cache_kwargs,
) -> WalkForwardGridResult:
    """Run a full walk-forward portfolio strategy grid.

    The function builds or accepts a rebalance-state cache, constructs strategy
    specifications, runs each strategy through a rebalanced portfolio backtest,
    collects performance, NAV, returns, turnover, costs, diagnostics, and raw
    backtest objects, and returns a single result container.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel used for backtesting and, when needed, state
        construction.
    close : pandas.DataFrame, optional
        Close-price panel used for universe selection and state construction.
    volume : pandas.DataFrame, optional
        Volume panel used for universe selection.
    rebalance_dates : sequence of pandas.Timestamp or str
        Candidate rebalance dates.
    universe_by_date : mapping, optional
        Precomputed liquid universes by date.
    mu_models : mapping, optional
        Expected-return model callables.
    cov_models : mapping, optional
        Covariance model callables.
    optimizers : mapping, optional
        Optimizer callables.
    strategy_specs : sequence of mappings, optional
        Explicit strategy specifications. If omitted, the default grid is built.
    cache : dict, optional
        Precomputed rebalance-state cache.
    cov_lookback : int, default=252
        Covariance lookback used when building the cache.
    mu_lookback : int, default=504
        Expected-return lookback used when building the cache.
    max_weight : float, default=0.25
        Per-asset upper bound.
    min_weight : float, default=0.0
        Per-asset lower bound.
    long_only : bool, default=True
        Whether strategies are long-only.
    trading_cost_bps : float, default=10.0
        Transaction cost in basis points.
    turnover_penalty_bps : float, default=10.0
        Turnover penalty used by optimizers.
    fallback : str, default="equal"
        Fallback rule used by the backtest engine when a strategy cannot produce
        valid weights.
    solver_order : sequence of str, optional
        Optimizer solver preference order.
    optimizer_params : mapping, optional
        Per-optimizer parameter overrides.
    blend_by_optimizer : mapping, optional
        Optional smoothing/blending strength by optimizer family.
    rf_daily : float or pandas.Series, default=0.0
        Daily risk-free rate for performance metrics.
    annualization : float, default=252.0
        Annualization factor.
    **cache_kwargs
        Additional keyword arguments passed to cache construction.

    Returns
    -------
    WalkForwardGridResult
        Complete walk-forward grid result, including summary tables, paths,
        weights, turnover, costs, diagnostics, cache, raw backtests, and metadata.

    Raises
    ------
    InputError
        If no valid rebalance dates remain after cache construction.

    Notes
    -----
    This function intentionally performs no plotting or data loading. It is the
    core reusable engine behind the notebook strategy grid.
    """

    if mu_models is None:
        mu_models = {
            "Momentum": expected_returns.momentum_mu,
            "BayesStein": expected_returns.bayes_stein_mu,
            "BayesSteinMomentum": expected_returns.bayes_stein_momentum_mu,
        }
    if cov_models is None:
        cov_models = {
            "Sample": covariance.sample_covariance,
            "LedoitWolf": covariance.ledoit_wolf_covariance,
            "OAS": covariance.oas_covariance,
            "EWMA": covariance.ewma_covariance,
        }
    if optimizers is None:
        optimizers = {
            "EW": optimizer_module.equal_weight,
            "MinVar": optimizer_module.minimum_variance,
            "MV": optimizer_module.mean_variance,
            "RidgeMV": optimizer_module.ridge_mean_variance,
            "MaxSharpe": optimizer_module.max_sharpe_slsqp,
            "FrontierGrid": optimizer_module.max_sharpe_frontier_grid,
        }
    if cache is None:
        cache = build_rebalance_state_cache(
            returns=returns,
            close=close,
            volume=volume,
            rebalance_dates=rebalance_dates,
            universe_by_date=universe_by_date,
            cov_models=cov_models,
            mu_models=mu_models,
            cov_lookback=cov_lookback,
            mu_lookback=mu_lookback,
            annualization=annualization,
            rf_daily=rf_daily,
            **cache_kwargs,
        )
    usable_rebalance_dates = [pd.Timestamp(d) for d in rebalance_dates if pd.Timestamp(d) in cache]
    if len(usable_rebalance_dates) == 0:
        raise InputError("No valid rebalance dates remain after state construction.")

    opt_params = {k: dict(v) for k, v in (optimizer_params or {}).items()}
    blend_map = dict(DEFAULT_BLEND_BY_OPTIMIZER)
    if blend_by_optimizer is not None:
        blend_map.update({_canonical_optimizer(k): float(v) for k, v in blend_by_optimizer.items()})
    rpy = rebalances_per_year(usable_rebalance_dates)
    kappa_target_annual = float(rpy) * (float(trading_cost_bps) + float(turnover_penalty_bps)) / 10000.0
    specs = (
        _normalize_strategy_specs(strategy_specs, optimizers)
        if strategy_specs is not None
        else build_strategy_grid(mu_models=mu_models, cov_models=cov_models, optimizers=optimizers)
    )

    backtests: dict[str, BacktestResult] = {}
    diag_rows: list[dict[str, Any]] = []
    for spec in specs:
        name = spec["name"]
        family = spec["optimizer"]
        weight_fn = _strategy_weight_fn(
            spec,
            optimizer_params=opt_params,
            w_min=min_weight,
            w_max=max_weight,
            long_only=long_only,
            turnover_penalty_bps=turnover_penalty_bps,
            kappa_target_annual=kappa_target_annual,
            solver_order=solver_order,
        )
        res = run_rebalanced_portfolio_backtest(
            returns=returns,
            rebal_dates=usable_rebalance_dates,
            cache=cache,
            weight_fn=weight_fn,
            cost_bps=trading_cost_bps,
            fallback=fallback,  # type: ignore[arg-type]
            blend=blend_map.get(family, 0.0),
            w_min=min_weight,
            w_max=max_weight,
            long_only=long_only,
            rf_daily=rf_daily,
        )
        res = _with_metadata(
            res,
            {
                "optimizer": family,
                "cov_model": spec.get("cov_model"),
                "mu_model": spec.get("mu_model"),
                "strategy_name": name,
            },
        )
        backtests[name] = res
        diag_rows.append(
            {
                "Strategy": name,
                "Optimizer": family,
                "Mu model": spec.get("mu_model"),
                "Covariance model": spec.get("cov_model"),
                "Fallbacks": int(res.fallbacks),
            }
        )

    summary, nav, ret, turnover, costs = _collect_outputs(backtests, rf_daily=rf_daily, annualization=annualization)
    diagnostics = pd.DataFrame(diag_rows).set_index("Strategy") if diag_rows else pd.DataFrame()
    return WalkForwardGridResult(
        results=summary,
        nav=nav,
        returns=ret,
        weights={name: res.weights for name, res in backtests.items()},
        turnover=turnover,
        costs=costs,
        diagnostics=diagnostics,
        cache=cache,
        backtests=backtests,
        metadata={
            "rebalance_dates": usable_rebalance_dates,
            "_returns_source": returns,
            "rebalances_per_year": rpy,
            "kappa_target_annual": kappa_target_annual,
            "mu_models": list(mu_models.keys()),
            "cov_models": [covariance.normalize_covariance_method(k) for k in cov_models],
            "optimizer_params": opt_params,
            "blend_by_optimizer": blend_map,
            "rf_daily": rf_daily,
            "annualization": annualization,
            "trading_cost_bps": trading_cost_bps,
            "turnover_penalty_bps": turnover_penalty_bps,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "long_only": long_only,
            "fallback": fallback,
            "solver_order": list(solver_order) if solver_order is not None else None,
        },
    )


def run_equal_weight_walkforward(
    *,
    returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    universe_by_date: Mapping[pd.Timestamp, Any],
    max_weight: float | None = 0.25,
    min_weight: float | None = 0.0,
    long_only: bool = True,
    trading_cost_bps: float = 10.0,
    rf_daily: float | pd.Series = 0.0,
) -> BacktestResult:
    """Run an equal-weight walk-forward backtest without estimating model state."""

    cache = build_universe_state_cache(
        returns=returns,
        rebalance_dates=rebalance_dates,
        universe_by_date=universe_by_date,
    )
    usable_dates = [
        pd.Timestamp(date)
        for date in rebalance_dates
        if pd.Timestamp(date) in cache
    ]
    if not usable_dates:
        raise InputError("No valid rebalance dates remain after universe alignment.")

    def weight_fn(
        date: pd.Timestamp,
        state: Mapping[str, Any],
        previous_weights: np.ndarray,
    ) -> np.ndarray:
        del date, previous_weights
        return np.asarray(
            optimizer_module.equal_weight(
                state["tickers"],
                w_min=min_weight,
                w_max=max_weight,
                long_only=long_only,
            ),
            dtype=float,
        )

    result = run_rebalanced_portfolio_backtest(
        returns=returns,
        rebal_dates=usable_dates,
        cache=cache,
        weight_fn=weight_fn,
        cost_bps=trading_cost_bps,
        fallback="equal",
        blend=0.0,
        w_min=min_weight,
        w_max=max_weight,
        long_only=long_only,
        rf_daily=rf_daily,
    )
    return _with_metadata(
        result,
        {
            "optimizer": "EW",
            "strategy_name": "EW",
            "rebalance_dates": usable_dates,
        },
    )


def append_frontiergrid_strategy(
    grid: WalkForwardGridResult,
    *,
    cov_model: str,
    mu_model: str,
    frontier_optimizer: Callable = optimizer_module.max_sharpe_frontier_grid,
    grid_n: int = 25,
    name: str | None = None,
) -> WalkForwardGridResult:
    """Append one FrontierGrid strategy for an explicit covariance/mu pair."""
    cov_key = covariance.normalize_covariance_method(cov_model)
    mu_key = expected_returns.normalize_mu_model(mu_model)
    strategy_name = name or f"MaxSharpe (FrontierGrid) ({cov_key}, {mu_key})"
    if strategy_name in grid.backtests:
        return grid

    metadata = dict(grid.metadata)
    opt_params = {k: dict(v) for k, v in metadata.get("optimizer_params", {}).items()}
    opt_params.setdefault("FrontierGrid", {})
    opt_params["FrontierGrid"].setdefault("grid_n", grid_n)
    spec = {
        "name": strategy_name,
        "optimizer": "FrontierGrid",
        "cov_model": cov_key,
        "mu_model": mu_key,
        "fn": frontier_optimizer,
    }
    weight_fn = _strategy_weight_fn(
        spec,
        optimizer_params=opt_params,
        w_min=metadata.get("min_weight", 0.0),
        w_max=metadata.get("max_weight", 0.25),
        long_only=bool(metadata.get("long_only", True)),
        turnover_penalty_bps=float(metadata.get("turnover_penalty_bps", 10.0)),
        kappa_target_annual=float(metadata.get("kappa_target_annual", 0.0)),
        solver_order=metadata.get("solver_order"),
    )
    res = run_rebalanced_portfolio_backtest(
        returns=_returns_source_from_grid(grid),
        rebal_dates=metadata["rebalance_dates"],
        cache=grid.cache,
        weight_fn=weight_fn,
        cost_bps=float(metadata.get("trading_cost_bps", 10.0)),
        fallback=metadata.get("fallback", "equal"),
        blend=metadata.get("blend_by_optimizer", DEFAULT_BLEND_BY_OPTIMIZER).get("FrontierGrid", 0.10),
        w_min=metadata.get("min_weight", 0.0),
        w_max=metadata.get("max_weight", 0.25),
        long_only=bool(metadata.get("long_only", True)),
        rf_daily=metadata.get("rf_daily", 0.0),
    )
    res = _with_metadata(
        res,
        {"optimizer": "FrontierGrid", "cov_model": cov_key, "mu_model": mu_key, "strategy_name": strategy_name},
    )

    new_backtests = dict(grid.backtests)
    new_backtests[strategy_name] = res
    summary, nav, ret, turnover, costs = _collect_outputs(
        new_backtests,
        rf_daily=metadata.get("rf_daily", 0.0),
        annualization=float(metadata.get("annualization", 252.0)),
    )
    diag = grid.diagnostics.copy()
    diag.loc[strategy_name, ["Optimizer", "Mu model", "Covariance model", "Fallbacks"]] = [
        "FrontierGrid",
        mu_key,
        cov_key,
        int(res.fallbacks),
    ]
    return WalkForwardGridResult(
        results=summary,
        nav=nav,
        returns=ret,
        weights={strategy: bt.weights for strategy, bt in new_backtests.items()},
        turnover=turnover,
        costs=costs,
        diagnostics=diag,
        cache=grid.cache,
        backtests=new_backtests,
        metadata=metadata,
    )


def append_frontiergrid_from_best_maxsharpe(
    grid: WalkForwardGridResult,
    *,
    frontier_optimizer: Callable = optimizer_module.max_sharpe_frontier_grid,
    metric: str = "Sharpe",
    grid_n: int = 25,
) -> WalkForwardGridResult:
    """Append one FrontierGrid strategy using the best existing MaxSharpe cov/mu pair."""
    from quantfinlab.portfolio import selection

    best = selection.select_best_maxsharpe_combination(grid.results, metric=metric)
    if best is None:
        return grid
    return append_frontiergrid_strategy(
        grid,
        cov_model=str(best["Covariance model"]),
        mu_model=str(best["Mu model"]),
        frontier_optimizer=frontier_optimizer,
        grid_n=grid_n,
    )


def _returns_source_from_grid(grid: WalkForwardGridResult) -> pd.DataFrame:
    src = grid.metadata.get("_returns_source")
    if isinstance(src, pd.DataFrame):
        return src
    # Net returns cannot reproduce drift weights; this branch is only a defensive guard.
    raise InputError("Original asset returns are missing from grid metadata; rerun run_walkforward_grid.")


__all__ = [
    "DEFAULT_BLEND_BY_OPTIMIZER",
    "WalkForwardGridResult",
    "append_frontiergrid_from_best_maxsharpe",
    "append_frontiergrid_strategy",
    "build_rebalance_state_cache",
    "build_strategy_grid",
    "build_universe_state_cache",
    "rebalances_per_year",
    "run_equal_weight_walkforward",
    "run_walkforward_grid",
]
