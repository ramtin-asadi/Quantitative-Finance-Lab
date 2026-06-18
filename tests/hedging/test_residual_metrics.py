from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.backtest.hedging import run_hedge_backtest
from quantfinlab.hedging import metrics, residual
from quantfinlab.hedging.relations import rel


def _price_panel() -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=120)
    base = np.linspace(0.0, 6.0 * np.pi, len(idx))
    hedge_log = np.log(100.0) + np.cumsum(0.0005 + 0.004 * np.sin(base))
    spread = 0.02 * np.sin(base * 1.3)
    target_log = 0.15 + 1.15 * hedge_log + spread
    return pd.DataFrame({"Target": np.exp(target_log), "Hedge": np.exp(hedge_log)}, index=idx)


def _return_panel() -> pd.DataFrame:
    px = _price_panel()
    return px.pct_change(fill_method=None).dropna()


def test_residual_price_beta_spread_signal_and_gate() -> None:
    px = _price_panel()
    static = residual.price_ols_beta(px, "Target", "Hedge", n_train=40)
    rolling = residual.roll_price_beta(px, "Target", "Hedge", win=30, n_train=40)
    kalman = residual.kf_price_beta(px, "Target", "Hedge", n_train=40, q=1e-6, r_mult=1.0)
    spread = residual.log_spread(px, "Target", "Hedge", static)
    signal = residual.z_signal(spread, z_win=20, z_in=1.0, z_out=0.25, z_stop=4.0)
    weights = residual.spread_w(signal["signal"], static, "Target", "Hedge", ["target", "hedge"])
    gated = residual.resid_gate(
        [
            {
                "beta_source": "static",
                "eg_p": 0.02,
                "adf_p": 0.03,
                "half_life": 10.0,
                "spread_vol": 0.02,
                "trades": 5,
                "cost_drag": 0.01,
                "beta_turnover": 0.05,
            },
            {
                "beta_source": "static",
                "eg_p": 0.50,
                "adf_p": 0.40,
                "half_life": 100.0,
                "spread_vol": 0.001,
                "trades": 1,
                "cost_drag": 0.30,
                "beta_turnover": 0.50,
            },
        ]
    )

    assert static.notna().any().all()
    assert rolling.notna().any().all()
    assert kalman.notna().any().all()
    assert spread.notna().any()
    assert {"z", "signal"} == set(signal.columns)
    assert list(weights.columns) == ["target", "hedge"]
    assert gated["eligible"].tolist() == [True, False]


def test_residual_stationarity_helpers_skip_or_return_valid_pvalues() -> None:
    pytest.importorskip("statsmodels")
    px = _price_panel()
    spread = residual.log_spread(px, "Target", "Hedge", beta=1.15)

    eg_p = residual.eg_test(np.log(px["Target"]), np.log(px["Hedge"]))
    adf_p = residual.adf_test(spread.dropna())
    half_life = residual.half_life(spread.dropna())

    assert 0.0 <= eg_p <= 1.0 or np.isnan(eg_p)
    assert 0.0 <= adf_p <= 1.0 or np.isnan(adf_p)
    assert half_life > 0.0 or np.isinf(half_life)


def test_hedging_metrics_tables_score_and_select_models() -> None:
    returns = _return_panel().rename(columns={"Target": "target", "Hedge": "hedge"})
    hedge_rel = rel("pair", "target", ["hedge"], desc="synthetic pair")
    zero_beta = pd.DataFrame({"hedge": 0.0}, index=returns.index)
    active_beta = pd.DataFrame({"hedge": 0.7}, index=returns.index)
    base = run_hedge_backtest(returns, zero_beta, target="target", hedges=["hedge"], cost_bps=0.0, beta_lag=0)
    hedged = run_hedge_backtest(returns, active_beta, target="target", hedges=["hedge"], cost_bps=1.0, beta_lag=0)
    backtests = {"pair | target": base, "pair | ols": hedged}

    coverage = metrics.coverage_table(returns, [hedge_rel])
    diag = metrics.diag_table(returns, [hedge_rel], win=30)
    model = metrics.model_table(backtests, [hedge_rel], returns)
    quality = metrics.quality_table(model)
    scored = metrics.score_table(model)
    best = metrics.best_table(scored)
    robust = metrics.robust_table(scored)

    assert bool(coverage.loc[0, "included"]) is True
    assert diag.loc[0, "obs"] == len(returns)
    assert model.loc[0, "relationship"] == "pair"
    assert quality.loc[0, "model_count"] == 1
    assert "score" in scored.columns
    assert best.loc[0, "best_model"] == "ols"
    assert robust.loc[0, "model"] == "ols"
