from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import BacktestResult, PortfolioState
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio.constraints import normalize_weights
from quantfinlab.portfolio.optimizers import equal_weight
from quantfinlab.risk.utils import _risk_free_metadata


def _sanitize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise InputError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise InputError("returns is empty.")
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def _state_as_mapping(state: Any) -> Mapping[str, Any]:
    if isinstance(state, PortfolioState):
        return state.as_dict()
    if isinstance(state, Mapping):
        return state
    raise InputError("Cache state must be a dict-like object or PortfolioState.")


def _weights_to_series(weights: np.ndarray | pd.Series | Sequence[float], tickers: Sequence[str]) -> pd.Series:
    labels = [str(x) for x in tickers]
    if isinstance(weights, pd.Series):
        return weights.reindex(labels).fillna(0.0).astype(float)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size != len(labels):
        raise InputError("Weight vector length must match number of tickers.")
    return pd.Series(arr, index=labels, dtype=float)


def _coerce_weight_frame(weights: pd.DataFrame | Mapping[Any, Any], columns: Sequence[str]) -> pd.DataFrame:
    if isinstance(weights, pd.DataFrame):
        out = weights.copy()
    elif isinstance(weights, Mapping):
        out = pd.DataFrame.from_dict(weights, orient="index")
    else:
        raise InputError("weights must be a pandas DataFrame or a date-keyed mapping.")
    if out.empty:
        raise InputError("weights is empty.")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c) for c in out.columns]
    cols = [str(c) for c in columns]
    out = out.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(how="all")
    if out.empty:
        raise InputError("weights has no usable rows after alignment.")
    return out.fillna(0.0)


def _bound_vector(bound: float | Mapping[str, float] | pd.Series | None, columns: Sequence[str], default: float | None) -> pd.Series:
    labels = [str(c) for c in columns]
    if bound is None:
        fill = np.inf if default is None else float(default)
        return pd.Series(fill, index=labels, dtype=float)
    if isinstance(bound, (Mapping, pd.Series)):
        s = pd.Series(bound, dtype=float).reindex(labels)
        fill = np.inf if default is None else float(default)
        return s.fillna(fill).astype(float)
    return pd.Series(float(bound), index=labels, dtype=float)


def _normalize_weight_series(
    weights: pd.Series,
    *,
    w_min: float | Mapping[str, float] | pd.Series | None,
    w_max: float | Mapping[str, float] | pd.Series | None,
    long_only: bool,
) -> pd.Series:
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if long_only:
        floor = _bound_vector(w_min, w.index, 0.0)
        cap = _bound_vector(w_max, w.index, None)
        w = w.clip(lower=floor, upper=cap)
        if float(w.sum()) <= 1e-12:
            room = (cap - floor).replace([np.inf, -np.inf], np.nan)
            if room.notna().any() and float(room.fillna(0.0).clip(lower=0.0).sum()) > 0:
                w = floor.copy()
                extra = 1.0 - float(w.sum())
                alloc_room = room.fillna(room[room.notna()].max()).clip(lower=0.0)
                w = w + max(extra, 0.0) * alloc_room / float(alloc_room.sum())
            else:
                w = pd.Series(1.0 / len(w), index=w.index, dtype=float)
        else:
            w = w / float(w.sum())
        for _ in range(25):
            over = w > cap + 1e-12
            if not bool(over.any()):
                break
            extra = float((w[over] - cap[over]).sum())
            w[over] = cap[over]
            room = (cap[~over] - w[~over]).clip(lower=0.0)
            if float(room.sum()) <= 1e-12:
                break
            w.loc[room.index] += extra * room / float(room.sum())
        w = w.clip(lower=floor, upper=cap)
        return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(w), index=w.index)
    total_abs = float(np.abs(w).sum())
    return w / total_abs if total_abs > 1e-12 else w


def run_weights_backtest(
    returns: pd.DataFrame,
    weights: pd.DataFrame | Mapping[Any, Any],
    *,
    cost_bps: float = 10.0,
    fixed_fee: float = 0.0,
    initial_value: float = 1.0,
    rf_daily: float | pd.Series = 0.0,
    w_min: float | Mapping[str, float] | pd.Series | None = 0.0,
    w_max: float | Mapping[str, float] | pd.Series | None = None,
    long_only: bool = True,
    normalize: bool = True,
    weight_timing: Literal["next_close", "same_day"] = "next_close",
    name: str | None = None,
) -> BacktestResult:
    """Backtest a precomputed rebalance-weight schedule with daily drift and costs.

    This engine is intended for strategies that already produce a date-indexed
    weight table. It aligns each decision-date weight vector to an effective trading
    date, applies transaction costs when the portfolio is rebalanced, drifts weights
    with realized returns between rebalances, and returns gross and net value
    paths.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel indexed by trading date. Values are one-period
        arithmetic returns in decimal units.
    weights : pandas.DataFrame or mapping
        Rebalance weight schedule indexed by decision date. Columns should match
        the return assets. Missing asset weights are filled with zero before
        optional normalization.
    cost_bps : float, default=10.0
        Proportional transaction cost in basis points applied to one-way turnover.
    fixed_fee : float, default=0.0
        Optional fixed fee applied per asset whose weight changes at a rebalance.
    initial_value : float, default=1.0
        Initial portfolio value.
    rf_daily : float or pandas.Series, default=0.0
        Stored in metadata for downstream reporting.
    w_min : float, mapping, pandas.Series, or None, default=0.0
        Minimum weight constraint used when normalizing.
    w_max : float, mapping, pandas.Series, or None, optional
        Maximum weight constraint used when normalizing.
    long_only : bool, default=True
        Whether negative weights are disallowed during normalization.
    normalize : bool, default=True
        If true, rebalance weights are normalized under the supplied constraints.
    weight_timing : {"next_close", "same_day"}, default="next_close"
        Timing convention for decision weights. ``"next_close"`` makes a decision
        dated ``t`` effective on the next available return row, avoiding use of the
        decision day's close-to-close return. ``"same_day"`` makes it effective on
        the first return row on or after the decision date.
    name : str, optional
        Strategy name stored in result metadata.

    Returns
    -------
    BacktestResult
        Result object containing gross/net value paths, gross/net returns,
        rebalance weights, turnover, costs, fallback count, and metadata.

    Raises
    ------
    InputError
        If ``initial_value`` is non-positive, timing is invalid, or no weights align
        to the return index.

    Notes
    -----
    Turnover is defined as half the L1 change in portfolio weights. Costs are
    subtracted from net value before the period return is applied on rebalance
    dates. Between rebalances, weights drift naturally with asset returns and are
    renormalized by the portfolio's grossed asset values.
    """

    if initial_value <= 0:
        raise InputError("initial_value must be positive.")
    if weight_timing not in {"next_close", "same_day"}:
        raise InputError("weight_timing must be 'next_close' or 'same_day'.")

    R = _sanitize_returns(returns)
    cols = [str(c) for c in R.columns]
    W = _coerce_weight_frame(weights, cols)
    idx = pd.DatetimeIndex(R.index)

    schedule: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
    for decision_dt, row in W.iterrows():
        side = "right" if weight_timing == "next_close" else "left"
        pos = int(idx.searchsorted(pd.Timestamp(decision_dt), side=side))
        if pos >= len(idx):
            continue
        effective_dt = pd.Timestamp(idx[pos])
        w_tar = pd.Series(row, index=cols, dtype=float).fillna(0.0)
        if normalize:
            w_tar = _normalize_weight_series(w_tar, w_min=w_min, w_max=w_max, long_only=long_only)
        schedule[effective_dt] = (pd.Timestamp(decision_dt), w_tar.astype(float))
    if not schedule:
        raise InputError("No weights align to the return index.")

    all_dates = idx[idx >= min(schedule)]
    gross_value = float(initial_value)
    net_value = float(initial_value)
    w = pd.Series(dtype=float)

    gross_values: list[float] = []
    net_values: list[float] = []
    gross_returns: list[float] = []
    weight_records: dict[pd.Timestamp, pd.Series] = {}
    turnover_records: dict[pd.Timestamp, float] = {}
    cost_records: dict[pd.Timestamp, float] = {}

    for dt in all_dates:
        if dt in schedule:
            decision_dt, w_tar = schedule[pd.Timestamp(dt)]
            w_pre = w.reindex(cols).fillna(0.0).astype(float)
            if float(w_pre.sum()) > 0 and long_only:
                w_pre = w_pre / float(w_pre.sum())
            delta = w_tar.reindex(cols).fillna(0.0).to_numpy(dtype=float) - w_pre.to_numpy(dtype=float)
            turnover = 0.5 * float(np.sum(np.abs(delta)))
            cost_value = float(net_value) * (float(cost_bps) / 10000.0) * turnover
            if fixed_fee > 0:
                cost_value += float(fixed_fee) * float(np.count_nonzero(np.abs(delta) > 1e-12))
            net_value = max(net_value - cost_value, 1e-12)
            w = w_tar[w_tar.abs() > 1e-14].copy()
            weight_records[decision_dt] = w_tar.astype(float)
            turnover_records[decision_dt] = turnover
            cost_records[decision_dt] = cost_value

        if w.empty:
            port_ret = 0.0
            w_next = pd.Series(dtype=float)
        else:
            r_today = R.loc[dt].reindex(w.index).fillna(0.0).astype(float)
            port_ret = float(np.dot(w.to_numpy(dtype=float), r_today.to_numpy(dtype=float)))
            grossed = w.to_numpy(dtype=float) * (1.0 + r_today.to_numpy(dtype=float))
            gross_sum = float(np.sum(grossed))
            w_next = (
                pd.Series(grossed / gross_sum, index=w.index, dtype=float)
                if gross_sum > 0 and np.isfinite(gross_sum)
                else pd.Series(dtype=float)
            )

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
    weights_df = (
        pd.DataFrame.from_dict(weight_records, orient="index")
        .fillna(0.0).sort_index()
    )
    turnover_s = pd.Series(turnover_records, name="turnover").sort_index()
    costs_s = pd.Series(cost_records, name="costs").sort_index()

    return BacktestResult(
        gross_values=gross_values_s,
        net_values=net_values_s,
        gross_returns=gross_returns_s,
        net_returns=net_returns_s,
        weights=weights_df,
        turnover=turnover_s,
        costs=costs_s,
        fallbacks=0,
        metadata={
            "strategy_name": name,
            "rf_daily": _risk_free_metadata(rf_daily),
            "cost_bps": float(cost_bps),
            "fixed_fee": float(fixed_fee),
            "weight_timing": weight_timing,
            "normalize": bool(normalize),
        },
    )


def run_many_weights_backtests(
    weights_by_strategy: Mapping[str, pd.DataFrame | Mapping[Any, Any]],
    *,
    returns: pd.DataFrame,
    **kwargs,
) -> dict[str, BacktestResult]:
    """Run several precomputed-weight backtests.

    Parameters
    ----------
    weights_by_strategy : mapping
        Mapping from strategy name to weight schedule.
    returns : pandas.DataFrame
        Asset return panel.
    **kwargs
        Additional keyword arguments passed to ``run_weights_backtest``.

    Returns
    -------
    dict
        Mapping from strategy name to ``BacktestResult``.
    """

    return {
        str(name): run_weights_backtest(returns=returns, weights=weights, name=str(name), **kwargs)
        for name, weights in weights_by_strategy.items()
    }


def run_rebalanced_portfolio_backtest(
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
    rf_daily: float | pd.Series = 0.0,
) -> BacktestResult:
    """Run a model-driven periodic rebalance backtest.

    This is the core model-agnostic portfolio backtest engine. At each rebalance
    date, it retrieves the precomputed model state from ``cache``, calls
    ``weight_fn`` to obtain target weights, applies fallback logic if the optimizer
    fails, optionally blends the target with previous weights, applies transaction
    costs, and then lets weights drift with daily returns until the next rebalance.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel indexed by trading date.
    rebal_dates : sequence of str or pandas.Timestamp
        Candidate rebalance dates. Only dates present in ``cache`` are used.
    cache : mapping
        Mapping from rebalance date to state object. Each state should provide
        ``tickers`` and whatever additional model inputs ``weight_fn`` requires.
    weight_fn : callable
        Function with signature ``weight_fn(date, state, previous_weights)``. It
        must return target weights as an array or Series, or ``None`` on failure.
    cost_bps : float, default=10.0
        Proportional transaction cost in basis points applied to one-way turnover.
    fixed_fee : float, default=0.0
        Optional fixed fee per changed asset.
    fallback : {"equal", "previous", "none"}, default="equal"
        Fallback behavior when ``weight_fn`` fails or returns invalid weights.
    blend : float, default=0.0
        Weight smoothing strength. ``0`` uses the new target; ``1`` would keep the
        previous weights.
    w_min : float or None, default=0.0
        Minimum weight for normalization.
    w_max : float or None, default=0.25
        Maximum weight for normalization.
    long_only : bool, default=True
        Whether negative weights are disallowed.
    initial_value : float, default=1.0
        Initial portfolio value.
    rf_daily : float or pandas.Series, default=0.0
        Stored in metadata for downstream reporting.

    Returns
    -------
    BacktestResult
        Result object with gross/net value paths, returns, weights, turnover,
        costs, fallback count, and metadata.

    Raises
    ------
    InputError
        If initial value is non-positive, rebalance dates are empty, or no rebalance
        dates remain after intersecting with cache keys.

    Notes
    -----
    The engine itself does not estimate expected returns, covariance matrices, or
    signals. Those belong in the cache and weight function. This separation is what
    makes the backtest reusable across mean-variance, risk-parity, Black-Litterman,
    ML, and custom allocation methods.

    Transaction costs are applied at rebalances before the day's return is applied.
    Weights then drift between rebalances according to realized asset returns.
    """

    if initial_value <= 0:
        raise InputError("initial_value must be positive.")

    R = _sanitize_returns(returns)
    idx = pd.DatetimeIndex(R.index)
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
    turnover_records: dict[pd.Timestamp, float] = {}
    cost_records: dict[pd.Timestamp, float] = {}
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

                try:
                    w_tar_raw = weight_fn(pd.Timestamp(dt), state, w_pre.to_numpy(dtype=float))
                except Exception:
                    w_tar_raw = None

                if w_tar_raw is None:
                    fallback_count += 1
                    if fallback == "equal":
                        w_tar = pd.Series(
                            equal_weight(tickers, w_min=w_min, w_max=w_max, long_only=long_only),
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
                                equal_weight(tickers, w_min=w_min, w_max=w_max, long_only=long_only),
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
                    wn = normalize_weights(
                        w_tar.to_numpy(dtype=float),
                        w_min=w_min,
                        w_max=w_max,
                        long_only=long_only,
                        as_series=False,
                    )
                    if wn is None:
                        fallback_count += 1
                        w_tar = pd.Series(
                            np.ones(len(tickers), dtype=float) / len(tickers),
                            index=tickers,
                            dtype=float,
                        )
                    else:
                        w_tar = pd.Series(np.asarray(wn, dtype=float), index=tickers, dtype=float)

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
                    turnover_records[pd.Timestamp(dt)] = turnover
                    cost_records[pd.Timestamp(dt)] = cost_value

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
    weights_df = (
        pd.DataFrame.from_dict(weight_records, orient="index")
        .fillna(0.0).sort_index()
    )
    turnover_s = pd.Series(turnover_records, name="turnover").sort_index()
    costs_s = pd.Series(cost_records, name="costs").sort_index()

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
            "rf_daily": _risk_free_metadata(rf_daily),
            "cost_bps": float(cost_bps),
            "fixed_fee": float(fixed_fee),
            "blend": float(blend_eff),
        },
    )


def run_strategy_backtest(*args, **kwargs) -> BacktestResult:
    """Alias with a notebook-friendly name."""
    return run_rebalanced_portfolio_backtest(*args, **kwargs)


def run_strategy_grid_backtests(
    strategy_fns: Mapping[str, Callable[[pd.Timestamp, Mapping[str, Any], np.ndarray], np.ndarray | pd.Series | None]],
    *,
    returns: pd.DataFrame,
    rebal_dates: Sequence[pd.Timestamp | str],
    cache: Mapping[pd.Timestamp | str, Any],
    **kwargs,
) -> dict[str, BacktestResult]:
    """Run several strategy functions through the same rebalance engine.

    Parameters
    ----------
    strategy_fns : mapping
        Mapping from strategy name to a ``weight_fn`` compatible with
        ``run_rebalanced_portfolio_backtest``.
    returns : pandas.DataFrame
        Asset return panel.
    rebal_dates : sequence
        Rebalance dates.
    cache : mapping
        Rebalance state cache.
    **kwargs
        Additional keyword arguments passed to
        ``run_rebalanced_portfolio_backtest``.

    Returns
    -------
    dict
        Mapping from strategy name to ``BacktestResult``.
    """

    return {
        name: run_rebalanced_portfolio_backtest(
            returns=returns,
            rebal_dates=rebal_dates,
            cache=cache,
            weight_fn=fn,
            **kwargs,
        )
        for name, fn in strategy_fns.items()
    }


backtest = run_rebalanced_portfolio_backtest


__all__ = [
    "backtest",
    "run_many_weights_backtests",
    "run_rebalanced_portfolio_backtest",
    "run_strategy_backtest",
    "run_strategy_grid_backtests",
    "run_weights_backtest",
]
