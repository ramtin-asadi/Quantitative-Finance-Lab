from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantfinlab.fixed_income.swaps import (
    discount_at,
    par_swap_rate,
    run_synthetic_swap_overlay,
    swap_annuity,
    swap_overlay_signal_from_duration_target,
    swap_pv01,
    swap_schedule,
    swap_value,
    zero_rate_at,
)
from tests.synthetic.generators import zero_rate_panel


def test_par_swap_has_near_zero_value_and_receiver_pv01_positive() -> None:
    zeros = zero_rate_panel(0.04)
    date = zeros.index[0]
    fixed = par_swap_rate(zeros, date, 5.0)

    times, accruals = swap_schedule(5.0, fixed_freq=2)
    value_at_par = swap_value(zeros, date, 5.0, fixed, side="receiver")

    assert len(times) == len(accruals)
    assert zero_rate_at(zeros, date, [1.0])[0] == pytest.approx(zeros.loc[date, 1.0])
    assert discount_at(zeros, date, [1.0])[0] < 1.0
    assert swap_annuity(zeros, date, 5.0) > 0.0
    assert value_at_par == pytest.approx(0.0, abs=1e-12)
    assert swap_pv01(zeros, date, 5.0, side="receiver") > 0.0


def test_swap_overlay_signal_respects_neutral_band() -> None:
    assert swap_overlay_signal_from_duration_target(6.0, neutral_duration=5.0, neutral_band=0.5) == 1
    assert swap_overlay_signal_from_duration_target(4.0, neutral_duration=5.0, neutral_band=0.5) == -1
    assert swap_overlay_signal_from_duration_target(5.2, neutral_duration=5.0, neutral_band=0.5) == 0


def test_synthetic_swap_overlay_builds_position_log_and_returns() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    cols = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    zero_rates = pd.DataFrame(
        [0.035 + 0.0005 * i + 0.0001 * cols for i in range(len(dates))],
        index=dates,
        columns=cols,
    )
    base_result = SimpleNamespace(
        returns=pd.Series(0.0002, index=dates[1:]),
        diagnostics={"risk": {"effective_duration": pd.Series(5.0, index=dates)}},
    )
    target_log = pd.DataFrame({"target duration": [6.5, 4.0, 5.1]}, index=dates[:3])

    log, overlay_returns = run_synthetic_swap_overlay(
        base_result,
        zero_rates,
        target_log,
        tenor=5.0,
        neutral_duration=5.0,
        neutral_band=0.25,
        duration_budget=0.5,
        slippage_bp=0.0,
    )

    assert not log.empty
    assert set(log["side"]).issubset({"receiver", "payer", "flat"})
    assert overlay_returns.index.equals(log.index)
    assert np.isfinite(log["overlay component return"]).all()
