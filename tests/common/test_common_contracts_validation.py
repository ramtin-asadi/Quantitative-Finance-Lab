from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.common import (
    BacktestResult,
    FundamentalReportArtifacts,
    InputError,
    PortfolioState,
    RiskReportArtifacts,
    SimpleBacktestResult,
    align_to_previous_available,
    as_1d_float_array,
    as_timestamp,
    month_end_dates,
    normalize_weights,
    previous_available_date,
    require_columns,
    require_finite_array,
    require_monotonic_index,
    require_non_empty_frame,
    validate_sorted_strictly_increasing,
    yearfrac,
)


def test_validation_helpers_accept_good_inputs_and_raise_precise_input_errors() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=pd.date_range("2024-01-01", periods=2))

    assert require_non_empty_frame(frame) is frame
    assert require_columns(frame, ["a", "b"]) is frame
    assert require_monotonic_index(frame) is frame
    np.testing.assert_allclose(require_finite_array([1.0, 2.0]), [1.0, 2.0])
    np.testing.assert_allclose(as_1d_float_array([[1.0], [2.0]], name="x"), [1.0, 2.0])
    validate_sorted_strictly_increasing(np.array([0.1, 0.2, 0.5]))

    with pytest.raises(InputError, match="missing required columns"):
        require_columns(frame, ["missing"])
    with pytest.raises(InputError, match="monotonic decreasing"):
        require_monotonic_index(frame, increasing=False)
    with pytest.raises(InputError, match="finite"):
        require_finite_array([1.0, np.nan])
    with pytest.raises(InputError, match="strictly increasing"):
        validate_sorted_strictly_increasing(np.array([0.1, 0.1]))


def test_date_helpers_align_to_observed_calendar() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-05", "2024-02-29"])

    assert as_timestamp("2024-01-02") == pd.Timestamp("2024-01-02")
    assert as_timestamp(None) is None
    assert yearfrac("2024-01-01", "2024-04-01") == 91.0 / 365.0
    assert previous_available_date(index, pd.Timestamp("2024-02-10")) == pd.Timestamp("2024-02-05")
    assert previous_available_date(index, pd.Timestamp("2023-12-31")) is None
    assert list(month_end_dates(index)) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert list(align_to_previous_available(index, ["2024-01-15", "2024-03-01"])) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-29")]
    with pytest.raises(InputError, match="ACT/365"):
        yearfrac("2024-01-01", "2024-02-01", basis="30/360")


def test_normalize_weights_and_result_containers_behave_like_mappings() -> None:
    weights = normalize_weights({"AAA": 2.0, "BBB": 1.0}, index=["AAA", "BBB", "CCC"])
    short = normalize_weights(pd.Series({"AAA": 0.6, "BBB": -0.2}), allow_short=True)
    dates = pd.date_range("2024-01-01", periods=3)
    returns = pd.Series([0.01, -0.02, 0.03], index=dates)
    nav = (1.0 + returns).cumprod()
    weight_frame = pd.DataFrame({"AAA": [0.6, 0.5, 0.4], "BBB": [0.4, 0.5, 0.6]}, index=dates)

    simple = SimpleBacktestResult(nav=nav, returns=returns, weights=weight_frame, diagnostics={"ok": True})
    backtest = BacktestResult(nav, nav, returns, returns, weight_frame, turnover=returns.abs(), costs=returns * 0.0, metadata={"name": "demo"})
    state = PortfolioState(["AAA", "BBB"], pd.Series({"AAA": 0.05, "BBB": 0.03}), {"sample": np.eye(2)}, metadata={"run": 1})
    artifacts = RiskReportArtifacts(tables={"summary": pd.DataFrame({"metric": ["vol"]})}, figures={"risk": []}, text={"notes": ["synthetic"]})

    np.testing.assert_allclose(weights.to_numpy(), [2.0 / 3.0, 1.0 / 3.0, 0.0])
    assert np.isclose(short.sum(), 1.0)
    assert simple["diagnostics"] == {"ok": True}
    assert backtest["fallbacks"] == 0
    assert state.as_dict()["metadata"] == {"run": 1}
    assert artifacts["text"] == {"notes": ["synthetic"]}
    with pytest.raises(KeyError):
        _ = simple["missing"]
    with pytest.raises(InputError, match="non-negative"):
        normalize_weights({"AAA": 1.0, "BBB": -0.5})


def test_fundamental_report_artifacts_copy_named_groups() -> None:
    table = pd.DataFrame({"score": [0.8]}, index=["AAA"])
    report = FundamentalReportArtifacts(
        tables={"snapshot": table},
        figures={"earnings": []},
        series={"history": pd.Series([0.7, 0.8])},
        text={"summary": ["stable profitability"]},
    )

    assert report["figures"] == {"earnings": []}
    assert report.as_dict()["text"] == {"summary": ["stable profitability"]}
    with pytest.raises(KeyError):
        _ = report["missing"]
