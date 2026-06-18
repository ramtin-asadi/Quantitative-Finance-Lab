from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import BacktestResult, PortfolioState
from quantfinlab.risk.contributions import (
    attribution_tables,
    portfolio_contribution_snapshot,
    scenario_es_contribution,
    vol_contribution,
)
from quantfinlab.risk.stress import stress_table
from quantfinlab.risk.var_backtesting import (
    best_var_methods,
    breach_stats,
    christoffersen_independence,
    kupiec_test,
    longest_true_streak,
    quantile_loss,
    var_backtest_details,
    var_backtest_table,
)
from tests.synthetic.generators import return_panel


def test_var_backtesting_statistics_and_method_ranking_are_well_formed() -> None:
    panel = return_panel(n=95, assets=("AAA", "BBB"))
    returns = panel["AAA"]
    breaches = [False, True, True, False, True, False, False]

    lr_uc, p_uc = kupiec_test(breaches, alpha=0.25)
    lr_ind, p_ind = christoffersen_independence(breaches)
    stats = breach_stats(returns, alpha=0.10, lookback=20, method="hist")
    details = var_backtest_details(panel[["AAA", "BBB"]], alpha=0.10, lookback=20, method="hist")
    table = var_backtest_table(panel[["AAA", "BBB"]], alpha=0.10, lookback=20, methods=("hist", "fhs"))
    best = best_var_methods(table)

    assert longest_true_streak(breaches) == 2
    assert np.isfinite([lr_uc, p_uc, lr_ind, p_ind]).all()
    assert quantile_loss([0.01, -0.03], [-0.01, -0.02], alpha=0.10) >= 0
    assert {"count", "rate", "quantile_loss", "series"}.issubset(stats)
    assert set(details) == {"AAA", "BBB"}
    assert isinstance(table.index, pd.MultiIndex)
    assert set(best) == {"AAA", "BBB"}


def test_stress_windows_and_risk_contributions_use_synthetic_portfolio_state() -> None:
    panel = return_panel(n=80, assets=("AAA", "BBB", "CCC"))
    weights = pd.Series({"AAA": 0.50, "BBB": 0.30, "CCC": 0.20})
    cov = panel.cov().to_numpy() * 252.0
    port_ret = panel @ weights
    nav = (1.0 + port_ret).cumprod()
    weights_history = pd.DataFrame([weights.to_dict()] * len(panel), index=panel.index)
    result = BacktestResult(nav, nav, port_ret, port_ret, weights_history, turnover=port_ret.abs() * 0.0, costs=port_ret * 0.0)
    state = PortfolioState(
        list(weights.index),
        mu_excess_ann=panel.mean() * 252.0,
        cov_ann_map={"sample": cov},
        metadata={"R_cov": panel},
    )

    vol_rc = vol_contribution(weights, cov, index=weights.index)
    es_rc = scenario_es_contribution(panel, weights, alpha=0.10)
    snap_vol, snap_es = portfolio_contribution_snapshot({"backtest": result, "state_cache": {panel.index[-1]: state}, "cov_key": "sample"}, es_alpha=0.10)
    vol_table, es_table, overlap = attribution_tables({"demo": {"backtest": result, "state_cache": {panel.index[-1]: state}, "cov_key": "sample"}}, top_k=2)
    stress = stress_table(panel[["AAA", "BBB"]], windows={"early": (panel.index[0], panel.index[20]), "late": (panel.index[40], panel.index[-1])})

    assert np.isclose(vol_rc.sum(), np.sqrt(weights.to_numpy() @ cov @ weights.to_numpy()))
    assert np.isclose(es_rc.sum(), -np.mean((panel @ weights)[(panel @ weights) <= np.quantile(panel @ weights, 0.10)]))
    assert set(snap_vol.index) == set(weights.index)
    assert set(snap_es.index) == set(weights.index)
    assert vol_table.loc["demo"].notna().all()
    assert "top2_overlap_count" in overlap.columns
    assert set(stress.index) == {"AAA", "BBB"}
