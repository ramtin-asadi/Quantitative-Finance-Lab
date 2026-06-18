from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.hedging import policies, ratios, relations


def _return_panel() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=90)
    base = np.linspace(0.0, 4.0 * np.pi, len(idx))
    hedge_a = 0.001 + 0.006 * np.sin(base)
    hedge_b = -0.0002 + 0.004 * np.cos(base * 0.7)
    target = 0.0001 + 0.55 * hedge_a - 0.25 * hedge_b + 0.001 * np.sin(base * 1.7)
    return pd.DataFrame({"Target": target, "Hedge_A": hedge_a, "Hedge_B": hedge_b}, index=idx)


def test_relationship_helpers_normalize_tickers_and_filter_availability() -> None:
    rel_a = relations.rel("Pair", "Target", ["Hedge_A", "Hedge_B"], desc="synthetic", pair=("Target", "Hedge_A"))
    rel_b = relations.rel("Missing", "Target", ["Other"])

    tickers = relations.rel_tickers([rel_a, rel_b])
    kept, missing = relations.filter_rels([rel_a, rel_b], ["target", "hedge_a", "hedge_b"])
    table = relations.rel_table([rel_a, rel_b], columns=["target", "hedge_a", "hedge_b"])
    proxy = relations.hedge_proxy_ret(_return_panel(), rel_a)

    assert rel_a.assets == ["target", "hedge_a", "hedge_b"]
    assert tickers == ["target", "hedge_a", "hedge_b", "other"]
    assert kept == [rel_a]
    assert missing == {"missing": ["other"]}
    assert bool(table.loc[0, "included"]) is True
    assert proxy.name == "pair_hedge_proxy"
    assert proxy.notna().all()


def test_beta_estimators_return_labeled_paths_after_training_window() -> None:
    returns = _return_panel()
    hedge_rel = relations.rel("Pair", "Target", ["Hedge_A", "Hedge_B"])

    ols = ratios.ols_beta(returns, hedge_rel, n_train=30)
    rolling = ratios.roll_beta(returns, hedge_rel, win=20, n_train=30)
    ridge = ratios.ridge_beta(returns, hedge_rel, win=20, n_train=30, alpha=1.0)
    kalman = ratios.kf_beta(returns, hedge_rel, n_train=30, q=1e-5, r_mult=1.0)

    for beta in (ols, rolling, ridge, kalman):
        assert list(beta.columns) == ["hedge_a", "hedge_b"]
        assert beta.notna().any().all()

    assert ols.dropna().iloc[0]["hedge_a"] == pytest.approx(0.55, abs=0.15)
    assert np.isfinite(ols.dropna().iloc[0]["hedge_b"])


def test_policy_helpers_sample_band_and_convert_betas_to_weights() -> None:
    returns = _return_panel()
    hedge_rel = relations.rel("Pair", "Target", ["Hedge_A", "Hedge_B"])
    beta = pd.DataFrame(
        {"Hedge_A": [0.50, 0.52, 0.70, 0.71], "Hedge_B": [-0.20, -0.21, -0.22, -0.40]},
        index=returns.index[::20][:4],
    )

    rebalanced = policies.rebalance_beta(beta, returns.index, freq="W-FRI")
    banded = policies.band_beta(beta, band=0.05)
    target_weights = policies.target_w(returns.index, hedge_rel, ["target", "hedge_a", "hedge_b"])
    hedge_weights = policies.beta_to_w(banded, hedge_rel, ["target", "hedge_a", "hedge_b"])

    assert not rebalanced.empty
    assert banded.iloc[1]["hedge_a"] == pytest.approx(banded.iloc[0]["hedge_a"])
    assert banded.iloc[2]["hedge_a"] == pytest.approx(0.70)
    assert target_weights.iloc[0]["target"] == pytest.approx(1.0)
    assert hedge_weights.iloc[0]["target"] == pytest.approx(1.0)
    assert hedge_weights.iloc[0]["hedge_a"] == pytest.approx(-0.50)
