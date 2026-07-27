from __future__ import annotations

import pandas as pd
import pytest

from quantfinlab.dataio.rates import (
    load_par_yield_curve,
    risk_free_returns,
    tenor_first_valid,
    tenor_label_to_years,
)


def test_load_par_yield_curve_normalizes_tenors_percent_and_duplicates(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "1 mo": [5.10, 5.00, 5.05],
            "2 yr": [4.00, 3.90, 3.95],
            "10 yr": [4.25, 4.20, 4.21],
        }
    )
    path = tmp_path / "curve.csv"
    raw.to_csv(path, index=False)

    curve = load_par_yield_curve(path, column_map={"Date": "date"}, percent=True)

    assert list(curve.columns) == ["1M", "2Y", "10Y"]
    assert list(curve.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert curve.loc[pd.Timestamp("2024-01-02"), "1M"] == pytest.approx(0.0505)
    assert tenor_label_to_years("6M") == 0.5
    assert tenor_first_valid(curve)["10Y"] == pd.Timestamp("2024-01-02")


def test_risk_free_returns_uses_previous_quote_and_calendar_days() -> None:
    yields = pd.Series(
        [0.04, 0.05],
        index=pd.to_datetime(["2024-01-05", "2024-01-08"]),
    )
    dates = pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"])

    returns = risk_free_returns(yields, dates)

    assert pd.isna(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx((1.0 + 0.04 / 2.0) ** (2.0 * 3.0 / 365.25) - 1.0)
    assert returns.iloc[2] == pytest.approx((1.0 + 0.05 / 2.0) ** (2.0 / 365.25) - 1.0)
