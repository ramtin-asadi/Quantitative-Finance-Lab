from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.numerics.interpolation import (
    bspline_basis,
    bspline_knots,
    second_difference_matrix,
    slice_pchip_grid,
    tensor_spline_fit,
    tensor_spline_values,
)


def test_bspline_basis_partitions_unity_and_validates_knots() -> None:
    knots = bspline_knots(6, degree=3, x_min=-2.0, x_max=2.0)
    x = np.linspace(-1.8, 1.8, 9)
    basis = bspline_basis(x, knots, degree=3)
    derivative = bspline_basis(x, knots, degree=3, derivative=1)

    assert len(knots) == 10
    assert np.allclose(knots[:4], -2.0)
    assert np.allclose(knots[-4:], 2.0)
    np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-10)
    assert derivative.shape == basis.shape
    with pytest.raises(ValueError):
        bspline_knots(3, degree=3)


def test_tensor_spline_fit_recovers_smooth_surface_on_grid_and_points() -> None:
    x_grid = np.linspace(-1.0, 1.0, 7)
    y_grid = np.linspace(0.0, 1.0, 6)
    xx, yy = np.meshgrid(x_grid, y_grid)
    z = np.sin(xx) + 0.5 * yy**2

    fit = tensor_spline_fit(xx.ravel(), yy.ravel(), z.ravel(), n_x_basis=5, n_y_basis=5, degree=2, lambda_x=0.001, lambda_y=0.001)
    fitted_points = tensor_spline_values(fit, xx.ravel(), yy.ravel())
    fitted_grid = tensor_spline_values(fit, x_grid, y_grid, grid=True)

    assert fit["n_obs"] == xx.size
    assert fitted_grid.shape == xx.shape
    assert np.sqrt(np.mean((fitted_points - z.ravel()) ** 2)) < 0.03


def test_second_difference_and_slice_pchip_grid_handle_weighted_duplicate_x_values() -> None:
    rows = []
    for y in [0.0, 1.0, 2.0]:
        for x in [0.0, 1.0, 2.0, 3.0, 4.0]:
            rows.append({"slice": y, "x": x, "y": y, "z": x * x + y, "weight": 1.0})
        rows.append({"slice": y, "x": 2.0, "y": y, "z": 5.0 + y, "weight": 3.0})
    data = pd.DataFrame(rows)

    matrix = second_difference_matrix(5)
    grid, meta = slice_pchip_grid(
        data,
        x_col="x",
        y_col="y",
        z_col="z",
        x_grid=np.array([1.0, 2.0, 3.0]),
        y_grid=np.array([0.0, 1.0, 2.0]),
        weight_col="weight",
        slice_col="slice",
        min_x=5,
    )

    assert matrix.shape == (3, 5)
    np.testing.assert_allclose(matrix @ np.ones(5), 0.0)
    assert grid.shape == (3, 3)
    assert len(meta) == 3
    assert np.isfinite(grid).all()
    assert grid[1, 1] > 4.0
