from __future__ import annotations

import pandas as pd

from quantfinlab.fixed_income.bond_pricing import make_synthetic_bond
from quantfinlab.fixed_income.risk import (
    duration_sanity_table,
    krd_pivot,
    latest_krd_table,
    portfolio_key_rate_risk,
    portfolio_parallel_risk,
    pv01_sanity_table,
)
from tests.synthetic.generators import flat_curve


def test_portfolio_key_rate_risk_tables_align_with_parallel_risk() -> None:
    date = pd.Timestamp("2024-01-31")
    curve = flat_curve(0.04)
    positions = {
        2: make_synthetic_bond(date, 2.0, 0.04, units=60.0),
        5: make_synthetic_bond(date, 5.0, 0.045, units=40.0),
    }

    parallel = portfolio_parallel_risk(positions, 5.0, date, curve.df, buckets=(2, 5))
    krd = portfolio_key_rate_risk(positions, 5.0, date, curve.df, buckets=(2, 5))
    krd_with_dates = krd.assign(date=date)
    risk_df = pd.DataFrame([parallel], index=[date])

    assert parallel["pv01"] > 0.0
    assert set(krd["key"]) == {2, 5}
    assert krd["key_rate_pv01"].sum() > 0.0
    assert latest_krd_table(krd_with_dates).shape[0] == 2
    assert not krd_pivot(krd_with_dates).empty
    assert "krd_sum" in duration_sanity_table(risk_df, krd_with_dates)
    assert "key_rate_pv01_sum" in pv01_sanity_table(risk_df, krd_with_dates)
