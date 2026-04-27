from __future__ import annotations

import numpy as np
import pandas as pd


def bps_cost(notional, bps: float):
    return np.asarray(notional, dtype=float) * (float(bps) / 10000.0)


def turnover_cost(trade_values, bps: float):
    if isinstance(trade_values, pd.Series):
        return trade_values.abs() * (float(bps) / 10000.0)
    if isinstance(trade_values, pd.DataFrame):
        return trade_values.abs().sum(axis=1) * (float(bps) / 10000.0)
    return np.sum(np.abs(np.asarray(trade_values, dtype=float))) * (float(bps) / 10000.0)


__all__ = ["bps_cost", "turnover_cost"]
