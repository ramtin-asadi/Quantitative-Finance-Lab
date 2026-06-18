from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.dataio.rates import tenor_label_to_years
from quantfinlab.fixed_income.discounting import (
    attach_discount_columns,
    constant_rate_series,
    continuous_rate_from_discount_factor,
    discount_factor_from_rate,
    make_discount_lookup,
    map_curve_rates_to_dates_and_taus,
    rate_from_zero_curve,
)
from tests.synthetic.generators import yield_curve_panel


def test_discount_factor_round_trip_and_pandas_shape() -> None:
    tau = pd.Series([0.25, 1.0, 2.0], index=list("abc"))
    rates = pd.Series([0.03, 0.04, 0.05], index=tau.index)

    df = discount_factor_from_rate(rates, tau, compounding="continuous")
    restored = continuous_rate_from_discount_factor(df, tau)
    simple_as_cont = constant_rate_series(tau, rate=0.06, input_compounding="simple")

    assert isinstance(df, pd.Series)
    assert df.index.equals(tau.index)
    assert np.allclose(restored.to_numpy(), rates.to_numpy())
    assert simple_as_cont.loc["b"] == pytest.approx(np.log1p(0.06))
    assert discount_factor_from_rate(0.04, 0.0) == 1.0


def test_curve_rate_mapping_uses_previous_available_date_without_lookahead() -> None:
    curve_panel = pd.DataFrame(
        {"6M": [0.030, 0.050], "2Y": [0.035, 0.055]},
        index=pd.to_datetime(["2024-01-02", "2024-01-05"]),
    )
    quote_dates = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-05"]))
    taus = pd.Series([tenor_label_to_years("6M"), tenor_label_to_years("2Y")])

    rates, source_dates = map_curve_rates_to_dates_and_taus(
        curve_panel,
        quote_dates,
        taus,
        return_source_dates=True,
    )

    assert rates.iloc[0] == pytest.approx(0.030)
    assert rates.iloc[1] == pytest.approx(0.055)
    assert source_dates.iloc[0] == pd.Timestamp("2024-01-02")
    assert source_dates.iloc[1] == pd.Timestamp("2024-01-05")


def test_discount_lookup_attaches_finite_discount_factors() -> None:
    par_yields = yield_curve_panel()
    lookup = make_discount_lookup(par_yields, curve_method="loglinear")
    quotes = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-10")],
            "tau": [0.25, 2.0],
        }
    )

    out = attach_discount_columns(quotes, lookup, date_col="date", tau_col="tau")
    direct_rate = rate_from_zero_curve(par_yields.iloc[0], np.asarray([0.5, 2.0]))

    assert out["df"].between(0.0, 1.0).all()
    assert out["r_short"].notna().all()
    assert np.all(np.isfinite(direct_rate))
    assert lookup["resolve_date"](pd.Timestamp("2024-01-10")) == par_yields.index[-1]
