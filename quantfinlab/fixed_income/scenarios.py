from __future__ import annotations

import numpy as np
import pandas as pd


def parallel_shift_scenarios(shocks_bp=(-100, -50, 50, 100), *, maturities=(2, 5, 10, 30)):
    """Create parallel yield-curve shock scenarios.

    Parameters
    ----------
    shocks_bp : sequence of float, default (-100, -50, 50, 100)
        Parallel shocks in basis points.
    maturities : sequence of int, default (2, 5, 10, 30)
        Maturity columns to include.

    Returns
    -------
    pandas.DataFrame
        Scenario table indexed by scenario name with shocks in decimal rate units.

    Notes
    -----
    Each scenario applies the same shock to every maturity column.
    """

    maturities = [int(x) for x in maturities]
    rows = []
    for shock in shocks_bp:
        row = {"scenario": f"parallel {shock:+.0f} bp"}
        row.update({m: float(shock) / 10000.0 for m in maturities})
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def key_rate_shock_scenarios(shock_bp=50, *, maturities=(2, 5, 10, 30)):
    """Create single-key rate shock scenarios.

    Parameters
    ----------
    shock_bp : float, default 50
        Absolute shock size in basis points.
    maturities : sequence of int, default (2, 5, 10, 30)
        Key maturities to include.

    Returns
    -------
    pandas.DataFrame
        Scenario table indexed by scenario name with one positive and one negative
        shock for each key maturity.

    Notes
    -----
    Shocks are expressed in decimal rate units. Non-shocked maturities are zero.
    """

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
    """Compute cross-scenario quantiles for simulated paths.

    Parameters
    ----------
    paths : array-like
        Simulated path matrix. Quantiles are computed along axis 0.
    quantiles : sequence of float, default (0.05, 0.25, 0.50, 0.75, 0.95)
        Quantile levels to compute.

    Returns
    -------
    pandas.DataFrame
        Quantile table indexed by quantile level.

    Notes
    -----
    The function is a thin wrapper around ``numpy.quantile`` with DataFrame output.
    """

    data = np.asarray(paths, dtype=float)
    q = np.quantile(data, list(quantiles), axis=0)
    return pd.DataFrame(q, index=list(quantiles))


def krd_approx_scenario_pnl(krd, shocks):
    """Approximate scenario P&L from key-rate duration exposure.

    Parameters
    ----------
    krd : array-like or pandas.Series
        Key-rate duration or key-rate PV01 exposure indexed by maturity.
    shocks : array-like or pandas.DataFrame
        Scenario shock table with maturity columns matching the exposure index.
        Shocks should be expressed in decimal rate units.

    Returns
    -------
    pandas.Series
        Approximate scenario P&L or return for each scenario.

    Notes
    -----
    The approximation uses ``-shock @ exposure``. Missing shock columns are treated
    as zero after reindexing to the exposure index.
    """

    exposure = pd.Series(krd, dtype=float)
    shock_frame = pd.DataFrame(shocks).astype(float)
    shock_frame = shock_frame.reindex(columns=exposure.index).fillna(0.0)
    return -shock_frame @ exposure


def strategy_scenario_summary(strategy_krd, scenarios):
    """Apply key-rate shock scenarios to multiple strategy exposures.

    Parameters
    ----------
    strategy_krd : Mapping[str, array-like]
        Mapping from strategy name to key-rate exposure vector.
    scenarios : pandas.DataFrame or array-like
        Scenario shock table with maturity columns.

    Returns
    -------
    pandas.DataFrame
        Scenario-by-strategy table of approximate returns or P&L values.

    Notes
    -----
    Each strategy is evaluated with the same scenario table using the key-rate
    duration approximation.
    """

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
