from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.diagnostics import (
    choose_liquid_single_day_for_diagnostics,
    iv_solver_diagnostics,
    pricing_error_summary,
    realized_vol_forward_bsm_pricing_comparison,
)
from quantfinlab.options.iv import compute_iv_table
from tests.synthetic.generators import option_surface_quotes


def test_iv_solver_diagnostics_summarize_failures_iterations_and_errors() -> None:
    quotes = option_surface_quotes().query("k.abs() <= 0.20").copy()
    table = compute_iv_table(quotes, engine="python", solver="newton_bisection")
    table.loc[table.index[:2], "iv_mid_success"] = False
    if "iv_success" in table.columns:
        table.loc[table.index[:2], "iv_success"] = False

    diagnostics = iv_solver_diagnostics({"newton": table}, bins=np.array([-0.25, -0.05, 0.05, 0.25]))
    summary = pricing_error_summary(pd.DataFrame({"pricing_error": [-0.10, 0.0, 0.20]}))

    assert diagnostics["summary"].loc[0, "solver"] == "newton"
    assert 0.0 < diagnostics["summary"].loc[0, "failure_rate"] < 1.0
    assert not diagnostics["failure_by_log_moneyness"].empty
    assert not diagnostics["iterations_by_log_moneyness"].empty
    assert summary.loc[0, "max_abs_error"] == 0.20


def test_realized_vol_pricing_comparison_and_liquid_day_choice_are_stable() -> None:
    quotes = option_surface_quotes(dates=("2024-01-02", "2024-01-03")).query("k.abs() <= 0.12").copy()
    realized = pd.DataFrame(
        {"rv_30": [0.19, 0.21]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    )

    comparison = realized_vol_forward_bsm_pricing_comparison(quotes, realized, vol_window=30)
    chosen = choose_liquid_single_day_for_diagnostics(quotes, min_pairs=4, prefer_dte_range=(20, 80))

    assert comparison["table"]["realized_vol"].notna().all()
    assert {"pricing_error", "vega_scaled_pricing_error", "inside_spread_hit"}.issubset(comparison["table"].columns)
    assert comparison["summary"].loc[0, "n"] == len(quotes)
    assert chosen in set(pd.to_datetime(quotes["date"]).dt.normalize())
