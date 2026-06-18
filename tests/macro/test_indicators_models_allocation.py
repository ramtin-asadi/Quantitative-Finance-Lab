from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.macro import allocation, indicators, models


def _macro_factors(n: int = 84) -> pd.DataFrame:
    idx = pd.date_range("2017-01-31", periods=n, freq="ME")
    t = np.linspace(0.0, 4.0 * np.pi, len(idx))
    return pd.DataFrame(
        {
            "cpi_core": 2.0 + 0.30 * np.sin(t) + np.linspace(0.0, 0.4, len(idx)),
            "ppi_goods": 1.5 + 0.20 * np.cos(t * 0.7),
            "inrt_policy": 1.0 + 0.50 * np.sin(t * 0.6),
            "gdp_growth": 2.0 + 0.40 * np.cos(t * 0.5),
            "inp_output": 1.0 + 0.30 * np.sin(t * 0.8),
            "unrt_rate": 5.0 + 0.20 * np.sin(t * 0.9),
            "injc_claims": 200.0 + 10.0 * np.cos(t),
            "nfp_jobs": 150.0 + 20.0 * np.sin(t * 0.4),
            "hsp_sales": 100.0 + 5.0 * np.cos(t * 0.6),
            "exp_goods": 80.0 + 3.0 * np.sin(t * 0.5),
            "trbn_balance": -20.0 + 2.0 * np.cos(t * 0.7),
            "cnci_confidence": 95.0 + 4.0 * np.sin(t * 0.8),
        },
        index=idx,
    )


def _signals_and_blocks() -> tuple[pd.DataFrame, pd.DataFrame]:
    factors = _macro_factors()
    signals = pd.concat(
        [
            indicators.inflation_level_pressure(factors, min_history=12),
            indicators.inflation_impulse(factors, min_history=12),
            indicators.inflation_acceleration(factors, min_history=12),
            indicators.inflation_diffusion(factors, min_history=12),
            indicators.policy_tightness(factors, min_history=12),
            indicators.policy_shock(factors, min_history=12),
            indicators.growth_momentum_stress(factors, min_history=12),
            indicators.growth_acceleration_stress(factors, min_history=12),
            indicators.growth_breadth_stress(factors, min_history=12),
            indicators.survey_warning(factors, min_history=12),
            indicators.labor_cooling(factors, min_history=12),
            indicators.sahm_pressure(factors, min_history=12),
            indicators.housing_impulse_stress(factors, min_history=12),
            indicators.external_demand_stress(factors, min_history=12),
            indicators.external_vulnerability(factors, min_history=12),
        ],
        axis=1,
    )
    signals["real_rate_squeeze"] = indicators.real_rate_squeeze(signals["policy_tightness"], signals["inflation_impulse"])
    signals["policy_catchup_risk"] = indicators.policy_catchup_risk(signals["inflation_level_pressure"], signals["policy_tightness"], min_history=12)
    signals["housing_rate_squeeze"] = indicators.housing_rate_squeeze(signals["housing_impulse_stress"], signals["policy_tightness"])
    signals["stress_breadth"] = indicators.stress_breadth(signals, min_history=12)
    signals["severe_stress_breadth"] = indicators.severe_stress_breadth(signals, min_history=12)
    signals["stagflation_pressure"] = indicators.stagflation_pressure(signals)
    signals["goldilocks_support"] = indicators.goldilocks_support(signals)
    blocks = indicators.condition_blocks(signals)
    return signals, blocks


def test_macro_indicators_build_signals_blocks_and_snapshots() -> None:
    factors = _macro_factors()
    signals, blocks = _signals_and_blocks()
    snapshot = indicators.macro_block_snapshot(blocks, dates=[blocks.index[-1]])

    assert indicators.prefix_columns(factors, ("cpi", "ppi")) == ["cpi_core", "ppi_goods"]
    assert indicators.expanding_zscore(factors["cpi_core"], min_history=12).notna().sum() > 0
    assert set(indicators.BLOCK_COLUMNS).issubset(blocks.columns)
    assert blocks.notna().any().all()
    assert snapshot.index[0] == blocks.index[-1]


def test_macro_fci_models_score_and_select_best_series() -> None:
    _, blocks = _signals_and_blocks()
    returns = pd.DataFrame(
        {"SPY": 0.01 * np.sin(np.linspace(0.0, 7.0, len(blocks))) + 0.002},
        index=blocks.index,
    )
    target = models.future_stress_target(returns, asset="SPY", min_history=12)
    econ = models.economic_fci(blocks, min_history=24)
    pca = models.pca_fci(blocks, min_history=24, min_blocks=4)
    pls = models.targeted_pls_fci(blocks, target["future_stress"], min_history=24, min_blocks=4, embargo_months=1)
    blended = models.blended_fci(pca, pca, econ, min_history=12)
    prob = models.stress_probability_fci(blocks, target["future_stress"], min_history=24, embargo_months=1)
    fci_models = pd.concat([econ, pca, pls, blended, prob["FCI_PROB_Z"]], axis=1)
    report = models.fci_quintile_report(blended, target)
    scores = models.fci_model_scores(fci_models, target)
    selected_name, selected = models.select_fci_model(fci_models, scores, min_observations=20)

    assert econ.notna().sum() > 0
    assert pca.notna().sum() > 0
    assert pls.notna().sum() > 0
    assert blended.name == "FCI_BLEND"
    assert {"FCI_PROB", "FCI_PROB_Z"}.issubset(prob.columns)
    assert models.fci_percentile(blended, min_history=20).notna().sum() > 0
    assert models.fci_change(blended, periods=3).name == "FCI_BLEND_3m_change"
    assert not report.empty
    assert "final_score" in scores.columns
    assert selected_name in fci_models.columns
    assert selected.name == selected_name


def test_macro_allocation_weights_are_normalized_and_explain_latest_decision() -> None:
    _, blocks = _signals_and_blocks()
    sectors = ["XLP", "XLF", "XLE", "XLK"]
    defensive = ["SHY", "GLD"]
    all_assets = [*sectors, *defensive]
    returns = pd.DataFrame(
        {
            asset: 0.004 + 0.01 * np.sin(np.linspace(0.0, 6.0, len(blocks)) + i)
            for i, asset in enumerate(all_assets)
        },
        index=blocks.index,
    )
    features = blocks.copy().ffill().fillna(0.0)
    features["best_fci_value"] = features.mean(axis=1)
    features["best_fci_percentile"] = features["best_fci_value"].rank(pct=True).fillna(0.5)
    features["best_fci_3m_change"] = features["best_fci_value"].diff(3).fillna(0.0)
    features["dominant_macro_block"] = features[indicators.BLOCK_COLUMNS].idxmax(axis=1)

    momentum = allocation.etf_momentum_score(returns, sectors)
    risky = allocation.fci_risky_weight(features["best_fci_percentile"], features["best_fci_3m_change"])
    defensive_w = allocation.defensive_weights(features.iloc[-1], defensive)
    equal = allocation.equal_sector_weights(features.index[-3:], sectors, all_assets=all_assets)
    sector_mom = allocation.momentum_sector_weights(returns, sectors, top_n=2, cap=0.60)
    macro_fit = allocation.sector_macro_fit(features, sectors)
    gated, gated_details = allocation.fci_gated_weights(features, sectors, defensive, return_details=True)
    allocated, details = allocation.fci_momentum_weights(returns, features, sectors, defensive, top_n=2, cap=0.60, return_details=True)
    latest = allocation.latest_decision_table(features, allocated, details, selected_fci_model="FCI_BLEND")

    assert list(momentum.columns) == sectors
    assert risky.between(0.5, 1.0).all()
    assert defensive_w.sum() == pytest.approx(1.0)
    assert np.allclose(equal.sum(axis=1), 1.0)
    assert sector_mom.sum(axis=1).max() <= 1.0 + 1e-12
    assert macro_fit.columns.tolist() == sectors
    assert np.allclose(gated.sum(axis=1), 1.0)
    assert "risky_weight" in gated_details
    assert np.allclose(allocated.sum(axis=1), 1.0)
    assert "final_score" in details
    assert set(latest["sector"]) == set(allocated.columns)
