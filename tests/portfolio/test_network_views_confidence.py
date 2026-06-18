from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.portfolio import confidence, views
from tests.synthetic.generators import return_panel


def _view_rows() -> list[dict[str, object]]:
    settings = views.ViewSettings(assets=["AAA", "BBB", "CCC"])
    roles = {"assets": ["AAA", "BBB", "CCC"]}
    made = views.make_view(
        "dual_momentum",
        "AAA over BBB",
        "synthetic momentum spread",
        ["AAA"],
        ["BBB"],
        1.5,
        "risk_on",
        roles=roles,
        settings=settings,
        priority=0.7,
    )
    assert made is not None
    rows = views.view_rows([made])
    rows[0]["date"] = pd.Timestamp("2024-01-31")
    return rows


def test_network_dependence_pipeline_builds_graph_weights_and_summary() -> None:
    pytest.importorskip("networkx")
    from quantfinlab.portfolio import network

    returns = return_panel(n=100, assets=("AAA", "BBB", "CCC", "DDD"))
    corr = network.shrink_corr(returns)
    dist = network.corr_distance(corr)
    pseudo = network.pseudo_observations(returns)
    rho = network.kendall_to_t_copula_corr(pseudo)
    nu = network.select_t_copula_nu(pseudo, nu_grid=(4, 8), max_pairs=4)
    tail = network.student_t_tail_dependence(rho, nu)
    graph = network.mst_network(tail, distance=network.dependence_to_distance(tail))
    centrality = network.centrality_table(graph)
    weights = network.network_diversifier_weights(
        network.network_score(1.0 - centrality["combined"], returns=returns, momentum_window=40, volatility_window=40, drawdown_window=50),
        returns=returns,
        n_stocks=3,
        max_weight=0.60,
    )
    pair_corr = network.pairwise_corr_for_weights(returns, weights.rename(returns.index[-1]).to_frame().T, lookback=40)
    summary = network.network_summary(corr=corr, tail=tail, centrality=centrality, nu=nu)

    assert np.allclose(np.diag(dist), 0.0)
    assert np.isfinite(network.student_t_copula_loglik(pseudo, rho, nu, pairs=[(0, 1)]))
    assert graph.number_of_nodes() == returns.shape[1]
    assert weights.sum() == pytest.approx(1.0)
    assert pair_corr.notna().all()
    assert summary["avg_tail"] >= 0.0


def test_view_helpers_create_clean_relative_views_and_signal_tables() -> None:
    returns = return_panel(n=230, assets=("AAA", "BBB", "CCC"))
    settings = views.ViewSettings(assets=["AAA", "BBB", "CCC"], min_signal_obs=20, trend_window=40)
    roles = {"assets": ["AAA", "BBB", "CCC"], "risky": ["AAA", "BBB"], "defensive": ["CCC"]}
    row = _view_rows()[0]
    p = views.p_series_from_assets(["AAA"], ["BBB"], ["AAA", "BBB", "CCC"])
    signal_table, values = views.signal_table_from_returns(returns, returns.index[-1], roles=roles, settings=settings)

    assert p is not None
    assert p.sum() == pytest.approx(0.0)
    assert row["q_tilt"] > 0.0
    assert np.isfinite(views.relative_basket_return(returns, ["AAA"], ["BBB"], lookback=20))
    assert np.isfinite(views.trailing_volatility(returns, "AAA", lookback=20))
    assert {"score", "mom_6_1", "vol_63"}.issubset(signal_table.columns)
    assert "risky_trend_breadth" in values


def test_confidence_scores_select_views_and_build_view_matrix() -> None:
    returns = return_panel(n=90, assets=("AAA", "BBB", "CCC"))
    rows = _view_rows()
    row = rows[0]
    history_log = pd.DataFrame(
        [
            {**row, "date": returns.index[i], "payoff_end_date": returns.index[i + 5], "payoff_ann": 0.03 + 0.001 * i, "payoff": 0.03 + 0.001 * i, "hit": True, "stress_state": False}
            for i in range(10, 25)
        ]
    )

    p = confidence.p_row_from_view(row, ["AAA", "BBB", "CCC"])
    payoff = confidence.payoff_history(pd.DataFrame([{**row, "date": returns.index[30]}]), returns, horizon=5)
    stats = confidence.family_reliability("dual_momentum", history_log, returns.index[40])
    score = confidence.confidence_score(stats, row, {"equity_stress": False})
    selected, log_rows = confidence.select_views(
        [row],
        history_log,
        returns.index[40],
        {"equity_stress": False},
        assets=["AAA", "BBB", "CCC"],
        family_q_caps=views.ViewSettings().family_q_caps,
    )
    view_p, view_q, clean = confidence.view_matrix(
        selected,
        ["AAA", "BBB", "CCC"],
        history=history_log,
        current_date=returns.index[40],
        family_q_caps=views.ViewSettings().family_q_caps,
    )

    assert p is not None
    assert not payoff.empty
    assert stats["n_obs"] >= 8
    assert 0.30 <= score["confidence"] <= 0.90
    assert len(selected) == 1
    assert log_rows[0]["kept"] is True
    assert view_p.shape == (1, 3)
    assert view_q.shape == (1,)
    assert clean.iloc[0]["q_tilt_final"] > 0.0
