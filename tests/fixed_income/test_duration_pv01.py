from __future__ import annotations

import numpy as np
import pytest

from quantfinlab.common.contracts import Bond
from quantfinlab.fixed_income.bond_pricing import bond_cashflows, bond_price, price_bond_from_issue
from quantfinlab.fixed_income.discounting import shifted_df_func
from quantfinlab.fixed_income.risk import (
    bond_price_and_risk,
    dv01,
    price_from_ytm,
    pv01,
    solve_bond_ytm,
)
from tests.synthetic.generators import flat_curve


def test_pv01_matches_symmetric_parallel_bump() -> None:
    curve = flat_curve(0.04)
    bond = Bond(coupon=0.045, maturity_years=5.0, freq=2, face=1.0)
    times, cfs = bond_cashflows(bond.coupon, bond.maturity_years, freq=bond.freq, face=bond.face)
    bump = 1.0 / 10000.0

    up = shifted_df_func(curve.df, lambda t: np.full_like(np.asarray(t, dtype=float), bump))
    down = shifted_df_func(curve.df, lambda t: np.full_like(np.asarray(t, dtype=float), -bump))
    manual = (price_bond_from_issue(down, times, cfs, 0.0) - price_bond_from_issue(up, times, cfs, 0.0)) / 2.0
    table = bond_price_and_risk(bond, {"flat": curve}, key_tenors=(2, 5, 10))

    assert pv01(bond, curve) == pytest.approx(manual)
    assert table.loc["flat", "clean_price"] == pytest.approx(bond_price(bond, curve))
    assert table.loc["flat", "pv01"] > 0.0
    assert table.filter(like="krd_").to_numpy().sum() > 0.0
    assert dv01(bond, curve) > 0.0


def test_ytm_solver_round_trips_bond_price() -> None:
    times, cfs = bond_cashflows(0.05, 3.0, freq=2)
    price = price_from_ytm(0.0475, times, cfs, freq=2)
    solved = solve_bond_ytm(price, times, cfs, freq=2)
    assert solved == pytest.approx(0.0475, abs=1e-8)
