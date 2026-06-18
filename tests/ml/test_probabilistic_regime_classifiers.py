from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from quantfinlab.ml import classifiers, probabilistic, regimes, uncertainty

pytestmark = pytest.mark.ml


def test_probabilistic_losses_conformal_offsets_and_calibration_tables() -> None:
    index = pd.RangeIndex(12)
    y = pd.Series(np.linspace(-1.0, 1.0, len(index)), index=index)
    q10 = y - 0.20
    q50 = y + 0.05
    q90 = y + 0.20
    ordered = probabilistic.enforce_quantile_order(q90, q50, q10)
    metrics = probabilistic.quantile_metrics(
        pd.DataFrame({"y": y, "q10": q10, "q50": q50, "q90": q90}),
        y_col="y",
        quantile_sets={"model": ("q10", "q50", "q90")},
    )
    nll = probabilistic.nll_metrics(
        pd.DataFrame({"y": y, "mu": q50, "sigma": 0.25}),
        y_col="y",
        mean_col="mu",
        sigma_col="sigma",
    )
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=220).repeat(2)})
    frame["y"] = np.sin(np.arange(len(frame)) / 8.0)
    frame["lo"] = frame["y"] - 0.10
    frame["hi"] = frame["y"] + 0.10
    offsets = probabilistic.rolling_conformal_offsets(
        frame,
        y_col="y",
        low_col="lo",
        high_col="hi",
        lookback_days=40,
        gap_days=5,
        min_obs=20,
    )
    adjusted = probabilistic.apply_rolling_conformal(
        frame.tail(10),
        y_col="y",
        low_col="lo",
        high_col="hi",
        lookback_days=40,
        gap_days=5,
        min_obs=20,
    )
    calibration = probabilistic.calibration_table(y, q10, q90, n_bins=3)

    assert probabilistic.pinball_loss(y, q50, 0.50) == pytest.approx(0.025)
    assert probabilistic.gaussian_nll(y, q50, pd.Series(0.25, index=index)) > 0.0
    assert ordered[0].le(ordered[1]).all() and ordered[1].le(ordered[2]).all()
    assert metrics.loc["model", "coverage_80"] == pytest.approx(1.0)
    assert nll.loc["mu", "avg_sigma"] == pytest.approx(0.25)
    assert probabilistic.conformal_offsets(y, q10, q90) == pytest.approx((0.0, 0.0))
    lo_c, hi_c = probabilistic.conformal_quantiles(q10, q90, offset_low=0.05, offset_high=0.03)
    assert lo_c.iloc[0] == pytest.approx(q10.iloc[0] - 0.05)
    assert hi_c.iloc[-1] == pytest.approx(q90.iloc[-1] + 0.03)
    assert offsets["calibration_n"].max() >= 20
    assert {"offset_low", "offset_high", "q_low_c", "q_high_c"}.issubset(adjusted.columns)
    assert probabilistic.interval_coverage(y, q10, q90) == pytest.approx(1.0)
    assert probabilistic.interval_width(q10, q90) == pytest.approx(0.40)
    assert calibration["count"].sum() == len(y)


def test_regime_profiles_transitions_and_quality_metrics() -> None:
    index = pd.bdate_range("2024-01-01", periods=20)
    labels = pd.Series([0] * 10 + [1] * 10, index=index, name="state")
    x = pd.DataFrame(
        {
            "risk_on_return": np.r_[np.ones(10), np.zeros(10)],
            "market_vol": np.r_[np.linspace(0.10, 0.20, 10), np.linspace(0.40, 0.50, 10)],
        },
        index=index,
    )
    outcomes = pd.Series(np.r_[np.full(10, 0.01), np.full(10, -0.01)], index=index, name="ret")
    proba = np.column_stack([np.linspace(0.80, 0.20, len(index)), np.linspace(0.20, 0.80, len(index))])

    profile = regimes.state_profile(x, labels, outcomes=outcomes)
    order = regimes.sort_states_by_profile(profile)
    remapped = regimes.remap_labels(labels, order)
    p_frame = regimes.proba_frame(proba, index)
    transitions = regimes.transition_table(labels)
    durations = regimes.duration_table(labels)
    quality = regimes.model_quality_row("toy", x, labels, x=x, proba=p_frame, outcomes=outcomes)

    assert profile.loc[0, "outcome_ret"] > profile.loc[1, "outcome_ret"]
    assert order[0] == 0
    assert remapped.iloc[0] == 0
    assert np.allclose(p_frame.sum(axis=1), 1.0)
    assert transitions.loc[0, 0] == pytest.approx(0.90)
    assert durations.loc[durations["state"].eq(1), "max_duration"].iloc[0] == 10
    assert regimes.posterior_confidence(p_frame).between(0.5, 1.0).all()
    assert regimes.posterior_entropy(p_frame).between(0.0, 1.0).all()
    assert regimes.regime_separation_score(outcomes, labels) > 0.0
    assert quality["states"] == 2
    assert quality["economic_separation"] > 0.0


def test_uncertainty_confidence_adjustments_and_classifier_importance_tables() -> None:
    predictions = pd.DataFrame({"rf": [0.01, 0.03, 0.02], "gbm": [0.02, 0.01, 0.02], "enet": [0.00, 0.02, 0.03]})
    disagreement = uncertainty.model_disagreement(predictions)
    width_conf = uncertainty.forecast_confidence(pd.Series([0.02, -0.01, 0.03]), pd.Series([0.05, 0.04, 0.06]))
    model_conf = uncertainty.disagreement_confidence(pd.Series(np.linspace(0.01, 0.05, 30)))
    adjusted = uncertainty.confidence_adjusted_mu(pd.Series([0.04, -0.02, 0.01]), width_conf, model_conf.tail(3).reset_index(drop=True))
    width = uncertainty.interval_width(pd.Series([0.1, 0.2]), pd.Series([0.4, 0.5]))

    x = pd.DataFrame(
        {
            "signal": np.r_[np.linspace(0.0, 1.0, 20), np.linspace(1.0, 2.0, 20)],
            "seasonal": np.sin(np.arange(40)),
        }
    )
    y = pd.Series([0] * 20 + [1] * 20)
    rf = RandomForestClassifier(n_estimators=30, random_state=7).fit(x, y)
    lr = LogisticRegression().fit(x, y)
    scores = classifiers.classifier_scores({"rf": rf, "lr": lr}, x, y)
    importance = classifiers.rf_importance(rf, x, y, n_repeats=2, random_state=7)

    assert disagreement.iloc[0] > 0.0
    assert width_conf.between(0.0, 1.0).all()
    assert model_conf.between(0.10, 1.0).all()
    assert adjusted.abs().le(pd.Series([0.04, 0.02, 0.01])).all()
    assert width.tolist() == pytest.approx([0.3, 0.3])
    assert scores.loc["rf", "accuracy"] == pytest.approx(1.0)
    assert scores["log_loss"].notna().all()
    assert importance.index[0] == "signal"
    assert {"importance", "permutation_importance", "permutation_std"}.issubset(importance.columns)
