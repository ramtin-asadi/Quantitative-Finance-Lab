from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options import parity
from tests.synthetic.generators import option_surface_quotes


def test_put_call_parity_infers_forward_from_long_quotes() -> None:
    quotes = option_surface_quotes()
    quotes = quotes.loc[quotes["date"].eq(pd.Timestamp("2024-01-02"))].copy()

    forwards = parity.infer_forwards_from_put_call_parity(quotes)
    with_forward = parity.infer_forwards_from_parity(quotes.drop(columns=["forward"]))
    errors = parity.parity_error_table(with_forward)

    assert len(forwards) == quotes["expiry"].nunique()
    assert forwards["n_pairs"].min() == 8
    np.testing.assert_allclose(forwards["forward"], forwards["spot"] * np.exp(forwards["implied_carry"] * forwards["tau"]), rtol=1e-12)
    assert errors["parity_residual"].abs().max() < 1e-10


def test_already_paired_quotes_use_median_forward_per_expiry() -> None:
    quotes = option_surface_quotes()
    pairs = quotes.pivot_table(
        index=["date", "expiry", "strike", "tau", "spot", "rate"],
        columns="option_type",
        values="mid",
        aggfunc="first",
    ).reset_index()
    pairs = pairs.rename(columns={"call": "c_mid", "put": "p_mid"})

    out = parity.infer_forwards_from_paired_quotes(pairs)

    assert out["n_pairs"].min() >= 8
    assert out["parity_error_mad"].max() < 1e-10
    assert parity.choose_liquid_single_day(quotes, min_pairs=8) == quotes["date"].min().normalize()
