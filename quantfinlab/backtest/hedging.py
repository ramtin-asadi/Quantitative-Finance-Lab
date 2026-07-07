from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


@dataclass(frozen=True)
class HedgeBacktestResult:
    """Container for hedge-book backtest results.

    The object stores the target return, gross and net hedged returns, turnover,
    transaction costs, traded beta path, and cumulative value paths. It also exposes
    portfolio-style aliases so hedge results can be consumed by shared reporting
    and scoring functions.

    Attributes
    ----------
    target_return : pandas.Series
        Unhedged target or spread return.
    gross_return : pandas.Series
        Hedged return before transaction costs.
    net_return : pandas.Series
        Hedged return after transaction costs.
    turnover : pandas.Series
        Period turnover of hedge ratios or hedge weights.
    cost : pandas.Series
        Period transaction-cost drag.
    beta : pandas.DataFrame
        Traded hedge beta path.
    gross_values : pandas.Series
        Cumulative gross value path.
    net_values : pandas.Series
        Cumulative net value path.
    target_values : pandas.Series
        Cumulative unhedged target value path.
    metadata : mapping, optional
        Strategy metadata such as target, hedges, cost assumptions, and beta lag.

    Properties
    ----------
    gross_returns, net_returns, costs, weights
        Compatibility aliases for portfolio and risk-report APIs.

    Methods
    -------
    as_dict()
        Return all fields as a dictionary.
    __getitem__(key)
        Dictionary-style field access.
    """

    target_return: pd.Series
    gross_return: pd.Series
    net_return: pd.Series
    turnover: pd.Series
    cost: pd.Series
    beta: pd.DataFrame
    gross_values: pd.Series
    net_values: pd.Series
    target_values: pd.Series
    metadata: Mapping[str, Any] | None = None

    @property
    def gross_returns(self) -> pd.Series:
        return self.gross_return

    @property
    def net_returns(self) -> pd.Series:
        return self.net_return

    @property
    def costs(self) -> pd.Series:
        return self.cost

    @property
    def weights(self) -> pd.DataFrame:
        if self.beta.empty:
            return pd.DataFrame(index=self.beta.index)
        w = -self.beta.copy()
        return w

    def as_dict(self) -> dict[str, Any]:
        out = {
            "target_return": self.target_return,
            "gross_return": self.gross_return,
            "net_return": self.net_return,
            "turnover": self.turnover,
            "cost": self.cost,
            "beta": self.beta,
            "gross_values": self.gross_values,
            "net_values": self.net_values,
            "target_values": self.target_values,
        }
        if self.metadata is not None:
            out["metadata"] = dict(self.metadata)
        return out

    def __getitem__(self, key: str) -> Any:
        data = self.as_dict()
        if key not in data:
            raise KeyError(key)
        return data[key]


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise InputError("returns must be a pandas DataFrame.")
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _clean_beta(beta: pd.DataFrame | None, hedges: list[str], index: pd.Index) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index)
    if beta is None:
        return pd.DataFrame(0.0, index=idx, columns=hedges, dtype=float)
    if not isinstance(beta, pd.DataFrame):
        raise InputError("beta must be a pandas DataFrame or None.")
    out = beta.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    out = out.reindex(columns=hedges)
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.reindex(idx.union(out.index)).sort_index().ffill().reindex(idx)


def run_hedge_backtest(
    returns: pd.DataFrame,
    beta: pd.DataFrame | None,
    *,
    target: str,
    hedges: list[str],
    cost_bps: float = 5.0,
    beta_lag: int = 1,
    name: str | None = None,
) -> HedgeBacktestResult:
    """Backtest a direct target-minus-beta hedge book.

    The hedge-book return is computed as the target return minus the lagged hedge
    beta times hedge returns. A positive beta therefore represents a short hedge
    position against the target. Transaction costs are applied to changes in traded
    beta exposure.

    Parameters
    ----------
    returns : pandas.DataFrame
        Return panel containing the target and all hedge assets. Column names are
        normalized to lowercase internally.
    beta : pandas.DataFrame or None
        Desired hedge beta path with one column per hedge asset. Betas are aligned
        to the relationship return panel and lagged before use.
    target : str
        Target asset column.
    hedges : list of str
        Hedge asset columns.
    cost_bps : float, default=5.0
        Transaction cost in basis points applied to beta turnover.
    beta_lag : int, default=1
        Number of rows by which the desired beta is lagged before trading. The
        default uses a beta known at date ``t`` starting in period ``t+1``.
    name : str, optional
        Strategy name stored in result metadata.

    Returns
    -------
    HedgeBacktestResult
        Hedge backtest result with gross/net returns, turnover, costs, traded beta,
        and cumulative value paths.

    Raises
    ------
    InputError
        If required return columns are missing, no complete return rows remain, or
        no rows remain after beta lagging.

    Notes
    -----
    The gross return formula is:

    ``target_return - sum(beta_h * hedge_return_h)``.

    Transaction cost for each period is:

    ``sum(abs(delta beta_h)) * cost_bps / 10000``.

    The first traded beta row is treated as opening turnover.
    """

    target = str(target).strip().lower()
    hedges = [str(h).strip().lower() for h in hedges]
    r = _clean_returns(returns)
    missing = [c for c in [target, *hedges] if c not in r.columns]
    if missing:
        raise InputError(f"Missing return columns: {missing}")

    panel = r[[target, *hedges]].dropna(how="any")
    if panel.empty:
        raise InputError("No complete relationship return rows remain.")

    b_raw = _clean_beta(beta, hedges, panel.index)
    b_used = b_raw.shift(int(beta_lag)).reindex(panel.index)
    data = pd.concat([panel, b_used.add_prefix("beta__")], axis=1).dropna(how="any")
    if data.empty:
        raise InputError("No rows remain after lagging hedge ratios.")

    beta_cols = [f"beta__{h}" for h in hedges]
    beta_used = data[beta_cols].copy()
    beta_used.columns = hedges

    target_ret = data[target].astype(float)
    hedge_ret = data[hedges].astype(float)
    gross = target_ret - (beta_used * hedge_ret).sum(axis=1)

    turnover = beta_used.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = beta_used.iloc[0].abs().sum()
    cost = turnover * (float(cost_bps) / 10000.0)
    net = gross - cost

    target_values = (1.0 + target_ret.fillna(0.0)).cumprod()
    gross_values = (1.0 + gross.fillna(0.0)).cumprod()
    net_values = (1.0 + net.fillna(0.0)).cumprod()

    return HedgeBacktestResult(
        target_return=target_ret.rename("target_return"),
        gross_return=gross.rename("gross_return"),
        net_return=net.rename("net_return"),
        turnover=turnover.rename("turnover"),
        cost=cost.rename("cost"),
        beta=beta_used,
        gross_values=gross_values.rename("gross_values"),
        net_values=net_values.rename("net_values"),
        target_values=target_values.rename("target_values"),
        metadata={
            "strategy_name": name,
            "target": target,
            "hedges": list(hedges),
            "cost_bps": float(cost_bps),
            "beta_lag": int(beta_lag),
        },
    )


def run_many_hedge_backtests(
    books: Mapping[str, Mapping[str, Any] | tuple[Any, ...]],
    *,
    returns: pd.DataFrame,
    cost_bps: float = 5.0,
    beta_lag: int = 1,
) -> dict[str, HedgeBacktestResult]:
    """Run several hedge-book backtests with shared return and cost settings.

    Parameters
    ----------
    books : mapping
        Mapping from strategy name to either a mapping with ``target``, ``hedges``,
        and optional ``beta`` keys, or a tuple ``(target, hedges, beta)``.
    returns : pandas.DataFrame
        Return panel.
    cost_bps : float, default=5.0
        Transaction cost in basis points.
    beta_lag : int, default=1
        Execution lag applied to all beta paths.

    Returns
    -------
    dict
        Mapping from strategy name to ``HedgeBacktestResult``.
    """

    out: dict[str, HedgeBacktestResult] = {}
    for name, spec in books.items():
        if isinstance(spec, Mapping):
            target = spec["target"]
            hedges = spec["hedges"]
            beta = spec.get("beta")
        else:
            target, hedges, beta = spec
        out[str(name)] = run_hedge_backtest(
            returns,
            beta,
            target=str(target),
            hedges=list(hedges),
            cost_bps=cost_bps,
            beta_lag=beta_lag,
            name=str(name),
        )
    return out


def run_spread_backtest(
    returns: pd.DataFrame,
    beta: pd.DataFrame,
    signal: pd.Series,
    *,
    target: str,
    hedge: str,
    cost_bps: float = 5.0,
    beta_lag: int = 1,
    name: str | None = None,
) -> HedgeBacktestResult:
    """Backtest a residual spread-trading signal.

    The spread return is ``target_return - beta * hedge_return``. The supplied
    signal scales exposure to that spread, and transaction costs are applied to
    changes in target and hedge weights implied by the signal and beta.

    Parameters
    ----------
    returns : pandas.DataFrame
        Return panel containing target and hedge columns.
    beta : pandas.DataFrame
        Time-varying residual beta table with a ``beta`` column.
    signal : pandas.Series
        Trading signal. Positive values are long the residual spread; negative
        values are short the residual spread.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    cost_bps : float, default=5.0
        Transaction cost in basis points.
    beta_lag : int, default=1
        Lag applied to the beta path before execution.
    name : str, optional
        Strategy name stored in metadata.

    Returns
    -------
    HedgeBacktestResult
        Spread backtest result.

    Raises
    ------
    InputError
        If required columns are missing, beta lacks a ``beta`` column, or alignment
        leaves no tradable rows.

    Notes
    -----
    The traded weights are ``signal`` in the target and ``-signal * beta`` in the
    hedge. Turnover is the sum of absolute changes in those weights.
    """

    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    r = _clean_returns(returns)
    missing = [c for c in [target, hedge] if c not in r.columns]
    if missing:
        raise InputError(f"Missing return columns: {missing}")
    if not isinstance(beta, pd.DataFrame) or "beta" not in beta.columns:
        raise InputError("beta must be a DataFrame with a 'beta' column.")

    panel = r[[target, hedge]].dropna(how="any")
    b = pd.to_numeric(beta["beta"], errors="coerce")
    b.index = pd.to_datetime(beta.index)
    b = b.sort_index().reindex(panel.index).ffill().shift(int(beta_lag))
    sig = pd.Series(signal, dtype=float)
    sig.index = pd.to_datetime(sig.index)
    sig = sig.sort_index().reindex(panel.index).fillna(0.0)

    data = pd.concat([panel, b.rename("beta"), sig.rename("signal")], axis=1).dropna(how="any")
    if data.empty:
        raise InputError("No rows remain after aligning residual beta and signal.")

    spread_ret = data[target] - data["beta"] * data[hedge]
    gross = data["signal"] * spread_ret
    weights = pd.DataFrame({target: data["signal"], hedge: -data["signal"] * data["beta"]})
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()
    cost = turnover * (float(cost_bps) / 10000.0)
    net = gross - cost
    beta_used = pd.DataFrame({hedge: data["beta"]}, index=data.index)

    return HedgeBacktestResult(
        target_return=spread_ret.rename("spread_return"),
        gross_return=gross.rename("gross_return"),
        net_return=net.rename("net_return"),
        turnover=turnover.rename("turnover"),
        cost=cost.rename("cost"),
        beta=beta_used,
        gross_values=(1.0 + gross.fillna(0.0)).cumprod().rename("gross_values"),
        net_values=(1.0 + net.fillna(0.0)).cumprod().rename("net_values"),
        target_values=(1.0 + spread_ret.fillna(0.0)).cumprod().rename("spread_values"),
        metadata={
            "strategy_name": name,
            "target": target,
            "hedges": [hedge],
            "cost_bps": float(cost_bps),
            "beta_lag": int(beta_lag),
        },
    )


__all__ = ["HedgeBacktestResult", "run_hedge_backtest", "run_many_hedge_backtests", "run_spread_backtest"]
