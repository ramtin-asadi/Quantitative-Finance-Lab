from __future__ import annotations

import numpy as np
import pytest

from quantfinlab.fixed_income.bootstrap import bootstrap_pillars
from quantfinlab.fixed_income.discounting import (
    discount_curve_table,
    par_curve_table,
    zero_curve_table,
)
from quantfinlab.fixed_income.forwards import forward_curve_table
from quantfinlab.fixed_income.smoothers import fit_curves
from tests.synthetic.generators import yield_curve_panel


def test_loglinear_and_pchip_curves_reproduce_pillar_par_yields() -> None:
    par_yields = yield_curve_panel()
    pillars = bootstrap_pillars(par_yields.iloc[0])
    curves = fit_curves(pillars, methods=("loglinear", "pchip"))

    for name, curve in curves.items():
        dfs = curve.df(pillars.T)
        par_fit = par_curve_table({name: curve}, grid=pillars.T).iloc[:, 0].to_numpy()
        assert np.all(np.isfinite(dfs))
        assert np.all(dfs > 0.0)
        assert np.max(np.abs(par_fit - pillars.par)) < 8e-3


def test_curve_value_tables_share_explicit_grid() -> None:
    par_yields = yield_curve_panel()
    pillars = bootstrap_pillars(par_yields.iloc[0])
    curves = fit_curves(pillars, methods=("loglinear",))
    grid = np.asarray([0.5, 1.0, 2.0, 5.0])

    zero = zero_curve_table(curves, grid=grid)
    forward = forward_curve_table(curves, grid=grid)
    discount = discount_curve_table(curves, grid=grid)

    assert zero.index.tolist() == pytest.approx(grid.tolist())
    assert forward.shape == zero.shape == discount.shape
    assert discount["loglinear"].between(0.0, 1.0).all()
