from __future__ import annotations

import numpy as np
import pandas as pd


def parallel_shift_scenarios(shocks_bp=(-100, -50, 50, 100), *, maturities=(2, 5, 10, 30)):
    maturities = [int(x) for x in maturities]
    rows = []
    for shock in shocks_bp:
        row = {"scenario": f"parallel {shock:+.0f} bp"}
        row.update({m: float(shock) / 10000.0 for m in maturities})
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def key_rate_shock_scenarios(shock_bp=50, *, maturities=(2, 5, 10, 30)):
    maturities = [int(x) for x in maturities]
    rows = []
    for key in maturities:
        for sign in (-1, 1):
            row = {"scenario": f"{key}y key {sign * float(shock_bp):+.0f} bp"}
            row.update({m: 0.0 for m in maturities})
            row[key] = sign * float(shock_bp) / 10000.0
            rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def scenario_quantiles(paths, quantiles=(0.05, 0.25, 0.50, 0.75, 0.95)):
    data = np.asarray(paths, dtype=float)
    q = np.quantile(data, list(quantiles), axis=0)
    return pd.DataFrame(q, index=list(quantiles))


def krd_approx_scenario_pnl(krd, shocks):
    exposure = pd.Series(krd, dtype=float)
    shock_frame = pd.DataFrame(shocks).astype(float)
    shock_frame = shock_frame.reindex(columns=exposure.index).fillna(0.0)
    return -shock_frame @ exposure


def strategy_scenario_summary(strategy_krd, scenarios):
    rows = []
    shock_frame = pd.DataFrame(scenarios).astype(float)
    for name, krd in strategy_krd.items():
        pnl = krd_approx_scenario_pnl(krd, shock_frame)
        for scenario, value in pnl.items():
            rows.append({"strategy": name, "scenario": scenario, "return": float(value)})
    return pd.DataFrame(rows).pivot(index="scenario", columns="strategy", values="return").reindex(shock_frame.index)


__all__ = [
    "key_rate_shock_scenarios",
    "krd_approx_scenario_pnl",
    "parallel_shift_scenarios",
    "scenario_quantiles",
    "strategy_scenario_summary",
]
