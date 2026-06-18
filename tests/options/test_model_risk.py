from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.model_risk import (
    add_calibration_weights,
    balanced_model_quotes,
    calibration_quotes,
    choose_model_engine,
    choose_surface_date,
    common_model_quotes,
    market_summary,
    residual_entry_schedule,
    residual_scores,
    signal_dates,
)
from tests.synthetic.generators import option_surface_quotes


def test_calibration_quote_selection_and_date_choice_are_stable() -> None:
    quotes = option_surface_quotes(dates=("2024-01-02", "2024-01-03"))

    weighted = add_calibration_weights(quotes)
    selected = calibration_quotes(weighted, min_dte=7, max_dte=120, max_relative_spread=0.2)
    balanced = balanced_model_quotes(selected, min_quotes_per_expiry=5)
    common = common_model_quotes(selected, balanced, min_tail_count=4)
    chosen = choose_surface_date(selected, min_expiries=3, min_quotes=20)
    dates = signal_dates(selected, min_quotes=20, min_expiries=3, min_near_atm_quotes=4)

    assert choose_model_engine("numpy").loc[0, "engine_used"] == "numpy"
    assert weighted["obs_weight"].between(0.05, 20.0).all()
    assert not selected.empty
    assert not balanced.empty
    assert len(common) >= len(balanced)
    assert chosen in set(pd.to_datetime(selected["date"]).dt.normalize())
    assert len(dates) == 2


def test_residual_scores_entry_schedule_and_market_summary() -> None:
    quotes = option_surface_quotes().head(12).copy()
    fair = quotes.assign(
        ensemble_price=quotes["mid"] + np.linspace(-0.20, 0.30, len(quotes)),
        ensemble_price_residual=np.linspace(-0.20, 0.30, len(quotes)),
        model_disagreement=0.03,
        expected_exit_half_spread=quotes["half_spread"],
        fit_error=0.01,
        signal_direction=1.0,
    )

    scores = residual_scores(fair)
    schedule = residual_entry_schedule(scores, require_signal_direction=True, entry_spacing_days=1)
    summary = market_summary(
        "SYN",
        quotes,
        model_comparison=pd.DataFrame({"model": ["svi", "sabr"]}),
        validation=scores,
        hedge_comparison=pd.DataFrame({"run": ["base", "base"]}),
        engine="numpy",
    )

    assert scores["watchlist_candidate"].any()
    assert set(["entry_date", "contract_key", "quantity", "entry_z"]).issubset(schedule.columns)
    assert summary.loc[0, "asset"] == "SYN"
    assert summary.loc[0, "quotes"] == len(quotes)
