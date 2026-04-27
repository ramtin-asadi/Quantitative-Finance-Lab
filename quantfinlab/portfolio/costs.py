from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.backtest.costs import bps_cost


def portfolio_turnover(w_new, w_old=None) -> float:
    """One-way portfolio turnover: 0.5 * sum(abs(delta weights))."""
    new = pd.Series(w_new, dtype=float) if not isinstance(w_new, pd.Series) else w_new.astype(float)
    if w_old is None:
        old = pd.Series(0.0, index=new.index, dtype=float)
    else:
        old = pd.Series(w_old, dtype=float) if not isinstance(w_old, pd.Series) else w_old.astype(float)
    idx = new.index.union(old.index)
    delta = new.reindex(idx).fillna(0.0) - old.reindex(idx).fillna(0.0)
    return 0.5 * float(np.abs(delta.to_numpy(dtype=float)).sum())


def transaction_cost_from_turnover(turnover: float | pd.Series, bps: float, notional=1.0):
    """Convert turnover to a proportional or currency transaction cost."""
    return bps_cost(np.asarray(notional, dtype=float) * turnover, bps)


def apply_transaction_costs(nav: pd.Series, turnover: pd.Series, bps: float) -> pd.Series:
    """Apply proportional turnover costs to a NAV series at rebalance dates."""
    out = nav.astype(float).copy()
    if out.empty or turnover.empty:
        return out
    for dt, t in turnover.dropna().items():
        if dt in out.index:
            out.loc[dt:] *= max(1.0 - float(t) * float(bps) / 10000.0, 0.0)
    return out


__all__ = [
    "apply_transaction_costs",
    "portfolio_turnover",
    "transaction_cost_from_turnover",
]
