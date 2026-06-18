from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.common.contracts import Bond
from quantfinlab.fixed_income.bond_pricing import (
    bond_cashflows,
    bond_cashflows_between,
    bond_from_par_curve_row,
    bond_position_value,
    bond_price,
    make_synthetic_bond,
    remaining_cashflow_arrays,
    remaining_maturity,
)
from tests.synthetic.generators import flat_curve, yield_curve_panel


def test_bond_cashflows_and_flat_curve_price_are_consistent() -> None:
    curve = flat_curve(0.04)
    bond = Bond(coupon=0.04, maturity_years=2.0, freq=2, face=1.0)
    times, cfs = bond_cashflows(bond.coupon, bond.maturity_years, freq=bond.freq, face=bond.face)

    expected_dirty = float(np.sum(cfs * np.exp(-0.04 * times)))
    assert times.tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert cfs[-1] == pytest.approx(1.02)
    assert bond_price(bond, curve, clean=False) == pytest.approx(expected_dirty)
    assert bond_price(bond, curve, settle=0.25, clean=True) < bond_price(bond, curve, settle=0.25, clean=False)


def test_synthetic_bond_remaining_cashflows_and_coupon_split() -> None:
    par_yields = yield_curve_panel()
    issue = pd.Timestamp("2024-01-31")
    bond = make_synthetic_bond(issue, 2.0, 0.04, units=100.0, freq=2)
    curve = flat_curve(0.04)

    t_rem, cf_rem = remaining_cashflow_arrays(bond, issue)
    gross, coupon, principal = bond_cashflows_between(
        bond,
        issue,
        issue + pd.DateOffset(months=6),
    )
    contract, label = bond_from_par_curve_row(par_yields.iloc[0], maturity_years=4.0)

    assert remaining_maturity(bond, issue + pd.DateOffset(months=6)) == pytest.approx(1.5, abs=0.01)
    assert len(t_rem) == 4
    assert cf_rem[-1] == pytest.approx(102.0)
    assert gross == pytest.approx(2.0)
    assert coupon == pytest.approx(2.0)
    assert principal == pytest.approx(0.0)
    assert bond_position_value(bond, issue, curve.df) > 95.0
    assert contract.maturity_years == 4.0
    assert label == "5Y"
