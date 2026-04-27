from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.core import BacktestResult, PortfolioState
from quantfinlab.portfolio.constraints import normalize_weights
from quantfinlab.portfolio.optimizers import equal_weight


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
    rf_daily: float = 0.0,
) -> BacktestResult:
    """
    Daily-drift, periodic-rebalance portfolio backtest with transaction costs.

    This is intentionally portfolio-specific but model-agnostic: all model and
    optimizer choices arrive through ``cache`` and ``weight_fn``.
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
    """Run several strategy functions through the same rebalance engine."""
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
    "run_rebalanced_portfolio_backtest",
    "run_strategy_backtest",
    "run_strategy_grid_backtests",
]
