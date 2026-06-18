from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantfinlab.common.contracts import RiskReportArtifacts
from quantfinlab.reports import executive_bullets, risk_report


def _report_returns() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=320)
    t = np.arange(len(index), dtype=float)
    alpha = 0.0002 + 0.0010 * np.sin(t / 10.0) + 0.0005 * np.cos(t / 17.0)
    defensive = 0.00005 + 0.4 * alpha
    return pd.DataFrame({"Alpha": alpha, "Defensive": defensive}, index=index)


def _close_report_figures(artifacts: RiskReportArtifacts) -> None:
    for figures in artifacts.figures.values():
        for fig in figures:
            plt.close(fig)


def test_risk_report_returns_tables_series_text_and_figures_without_displaying() -> None:
    returns = _report_returns()

    artifacts = risk_report(
        objects=returns,
        market_ret=returns["Alpha"] * 0.8,
        include={"attribution": False, "rolling_beta": False},
        var_settings={"methods": ["hist"], "alpha": 0.05},
        backtest_settings={"methods": ["hist"], "lookback": 60, "alpha": 0.05},
        stress_settings={"windows": {"sample": (returns.index[20], returns.index[80])}},
        output={
            "display_tables": False,
            "print_exec_bullets": False,
            "show_figures": False,
        },
    )

    try:
        assert isinstance(artifacts, RiskReportArtifacts)
        expected_tables = {
            "performance",
            "shape",
            "drawdown_summary",
            "drawdown_episodes",
            "var_es",
            "var_backtest",
            "stress",
            "capm",
            "corr",
        }
        assert expected_tables.issubset(artifacts.tables)
        assert {"Alpha", "Defensive"}.issubset(artifacts.tables["performance"].index)
        assert artifacts.tables["performance"].loc["Defensive", "sharpe"] > artifacts.tables["performance"].loc["Alpha", "sharpe"]
        assert not artifacts.tables["var_backtest"].empty
        assert set(artifacts.tables["stress"]["window"]) == {"sample"}
        assert {"var_backtest_detail", "var_backtest_best_method", "stress_full", "capm_roll"}.issubset(artifacts.series or {})
        assert {"drawdown_compare", "rolling_vol", "var_backtest", "stress", "capm_scatter", "correlation"}.issubset(artifacts.figures)
        assert all(artifacts.figures[key] for key in artifacts.figures)
        assert artifacts["tables"].keys() == artifacts.tables.keys()
    finally:
        _close_report_figures(artifacts)


def test_executive_bullets_summarize_report_tables_consistently() -> None:
    returns = _report_returns()
    artifacts = risk_report(
        objects=returns,
        market_ret=returns["Alpha"] * 0.8,
        include={"attribution": False, "rolling_beta": False},
        var_settings={"methods": ["hist"]},
        backtest_settings={"methods": ["hist"], "lookback": 60},
        stress_settings={"windows": {"sample": (returns.index[20], returns.index[80])}},
        output={"display_tables": False, "print_exec_bullets": False, "show_figures": False},
    )

    try:
        bullets = executive_bullets(
            perf_tbl=artifacts.tables["performance"],
            dd_tbl=artifacts.tables["drawdown_summary"],
            var_tbl=artifacts.tables["var_es"],
            capm_tbl=artifacts.tables["capm"],
            var_bt_tbl=artifacts.tables["var_backtest"],
        )

        assert bullets == artifacts.text["exec_bullets"]
        assert len(bullets) >= 4
        assert any("Sharpe" in bullet for bullet in bullets)
        assert any("beta" in bullet for bullet in bullets)
    finally:
        _close_report_figures(artifacts)
