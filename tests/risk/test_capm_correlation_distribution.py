from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.risk.capm import capm_ols, capm_table, rolling_beta, rolling_beta_corr
from quantfinlab.risk.correlation import corr_matrix, rolling_corr
from quantfinlab.risk.distribution import tail_ratio, tail_shape_table, worst_returns_summary
from tests.synthetic.generators import business_dates


def _capm_panel() -> pd.DataFrame:
    idx = business_dates(90)
    base = np.linspace(0.0, 2.0 * np.pi, len(idx))
    market = pd.Series(0.0002 + 0.008 * np.sin(base) + 0.003 * np.cos(2.0 * base), index=idx, name="MKT")
    alpha = pd.Series(0.00015 + 1.25 * market + 0.001 * np.sin(3.0 * base), index=idx, name="ALPHA")
    defensive = pd.Series(0.0001 + 0.55 * market - 0.001 * np.cos(2.5 * base), index=idx, name="DEF")
    return pd.concat([market, alpha, defensive], axis=1)


def test_capm_regression_and_rolling_beta_capture_known_synthetic_exposures() -> None:
    panel = _capm_panel()

    intercept, beta, r2 = capm_ols(panel["ALPHA"], panel["MKT"])
    beta_series, corr_series = rolling_beta_corr(panel["ALPHA"], panel["MKT"], window=20)
    beta_only = rolling_beta(panel["DEF"], panel["MKT"], window=20)
    table, rolling = capm_table(panel[["ALPHA", "DEF"]], market_ret=panel["MKT"], rolling=(20, 40))

    assert abs(intercept) < 0.001
    assert 1.15 < beta < 1.35
    assert r2 > 0.95
    assert beta_series.notna().sum() == len(panel) - 19
    assert corr_series.dropna().iloc[-1] > 0.95
    assert beta_only.dropna().median() < beta_series.dropna().median()
    assert table.loc["ALPHA", "beta"] > table.loc["DEF", "beta"]
    assert {"beta_20", "corr_40"}.issubset(rolling["ALPHA"].columns)


def test_correlation_and_distribution_tables_rank_tail_behavior() -> None:
    panel = _capm_panel()

    corr = corr_matrix(panel, min_periods=20)
    roll_corr = rolling_corr(panel["ALPHA"], panel["MKT"], window=15)
    shape = tail_shape_table(panel[["ALPHA", "DEF"]])
    worst = worst_returns_summary(panel[["ALPHA", "DEF"]], counts=(1, 3))

    assert corr.loc["MKT", "ALPHA"] > corr.loc["MKT", "DEF"]
    assert roll_corr.name == "corr_15"
    assert tail_ratio(panel["ALPHA"]) > 0
    assert {"skew", "excess_kurtosis", "tail_ratio_95_05"}.issubset(shape.columns)
    assert worst.loc["ALPHA", "worst_3d_avg"] >= worst.loc["ALPHA", "worst_1d_avg"] - 1e-12


def test_capm_table_accepts_aligned_risk_free_series() -> None:
    panel = _capm_panel()
    rf = pd.Series(np.linspace(0.0, 0.0002, len(panel)), index=panel.index)

    table, _ = capm_table(panel[["ALPHA"]], market_ret=panel["MKT"], rf_daily=rf)
    _, beta, _ = capm_ols(panel["ALPHA"] - rf, panel["MKT"] - rf)

    assert table.loc["ALPHA", "beta"] == pytest.approx(beta)
