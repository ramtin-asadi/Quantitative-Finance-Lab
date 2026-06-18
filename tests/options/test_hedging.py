from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.hedging import (
    build_delta_hedge_targets,
    build_delta_vega_hedge_targets,
    hedge_exposure_table,
    hedge_trade_from_band,
    hedging_summary_table,
    option_position_greeks,
    portfolio_greek_exposure,
    target_delta_hedge,
    target_vega_hedge,
)


def test_position_and_portfolio_exposures_scale_by_quantity_and_multiplier() -> None:
    row = pd.Series({"delta": 0.45, "gamma": 0.02, "vega": 8.0, "theta": -0.04, "rho": 0.12})
    exposure = option_position_greeks(row, quantity=3, multiplier=100)
    positions = pd.DataFrame(
        [
            {**row.to_dict(), "quantity": 3, "multiplier": 100},
            {**row.to_dict(), "quantity": -1, "multiplier": 50},
        ]
    )

    assert np.isclose(exposure["delta_exposure"], 135.0)
    assert np.isclose(portfolio_greek_exposure(positions)["vega_exposure"], 2000.0)
    assert target_delta_hedge(135.0, hedge_delta=1.0) == -135.0
    assert target_vega_hedge(2000.0, hedge_vega=25.0) == -80.0


def test_hedge_target_tables_and_band_trades() -> None:
    greek_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "delta": [0.50, 0.45],
            "vega": [9.0, 8.0],
            "underlying_units": [-45.0, -40.0],
        }
    )
    vega_hedge = pd.DataFrame({"date": greek_table["date"], "vega": [12.0, 10.0], "delta": [0.35, 0.30]})

    delta_targets = build_delta_hedge_targets(greek_table, option_quantity=100)
    dv_targets = build_delta_vega_hedge_targets(greek_table, vega_hedge, option_quantity=100)
    exposure = hedge_exposure_table(greek_table)

    assert np.allclose(delta_targets["target_underlying_units"], [-50.0, -45.0])
    assert "target_vega_contracts" in dv_targets.columns
    assert hedge_trade_from_band(current_units=-40, target_units=-50, exposure=11, band=10) == -10
    assert hedge_trade_from_band(current_units=-40, target_units=-50, exposure=9, band=10) == 0
    assert np.allclose(exposure["net_delta_exposure"], [-44.5, -39.55])
    assert hedging_summary_table({"summary": pd.DataFrame({"metric": ["turnover"]})}).loc[0, "metric"] == "turnover"
