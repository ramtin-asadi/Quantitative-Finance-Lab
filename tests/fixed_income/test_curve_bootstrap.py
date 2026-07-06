from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.bootstrap import (
    bootstrap_pillars,
    build_zero_curve_panel_from_par_yields,
    extract_par_curve,
    normalize_methods,
    normalize_par_yields,
    rmse_backtest,
    select_primary_curve,
)
from tests.synthetic.generators import yield_curve_panel


def test_normalize_and_bootstrap_par_curve_orders_tenors() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "2 yr": [4.2],
            "6 mo": [5.0],
            "10 yr": [4.5],
            "1 yr": [4.8],
        }
    )
    normalized = normalize_par_yields(raw, assume_percent=True)
    labels, maturities, par = extract_par_curve(normalized.iloc[0])
    pillars = bootstrap_pillars(normalized.iloc[0], asof=normalized.index[0])

    assert labels == ["6M", "1Y", "2Y", "10Y"]
    assert np.all(np.diff(maturities) > 0)
    assert np.allclose(par, [0.050, 0.048, 0.042, 0.045])
    assert np.all(np.isfinite(pillars.dfs))
    assert np.all((pillars.dfs > 0.0) & (pillars.dfs <= 1.0))
    assert np.diff(pillars.dfs).max() <= 1e-6


def test_zero_curve_panel_and_rmse_ranking_are_populated() -> None:
    par_yields = yield_curve_panel()
    zeros = build_zero_curve_panel_from_par_yields(
        par_yields,
        method="loglinear",
        tenors=["6M", "2Y", "5Y"],
    )
    rmse = rmse_backtest(par_yields, methods=("loglinear",), holdouts=[])
    method, display, ranked = select_primary_curve(rmse)

    assert list(zeros.columns) == [0.5, 2.0, 5.0]
    assert zeros.notna().all().all()
    assert method == "loglinear"
    assert "Log-linear" in display
    assert ranked.loc["loglinear", "rmse"] == pytest.approx(rmse.loc["loglinear", "rmse"])
    assert normalize_methods(["PCHIP", "pchip", "loglinear"]) == ["pchip", "loglinear"]
