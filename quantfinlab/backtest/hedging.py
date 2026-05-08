from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


@dataclass(frozen=True)
class HedgeBacktestResult:
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
    """
    Direct hedge-book P&L with target return minus lagged beta times hedge returns.

    ``beta`` is interpreted as the desired hedge ratio known at its timestamp.
    The traded beta used in period ``t`` is lagged by ``beta_lag`` rows after
    alignment to the relationship's return history.
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
    """Run direct hedge-book backtests for several named beta schedules."""
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


__all__ = ["HedgeBacktestResult", "run_hedge_backtest", "run_many_hedge_backtests"]
