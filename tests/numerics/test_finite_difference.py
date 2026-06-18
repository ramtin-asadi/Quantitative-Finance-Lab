from __future__ import annotations

import numpy as np

from quantfinlab.numerics.finite_difference import (
    complementarity_error,
    crank_nicolson_step,
    explicit_step,
    extract_boundary,
    implicit_step,
    payoff_values,
    pde_diagnostics,
    pde_grid,
    penalty_solve,
    psor_solve,
    rannacher_steps,
)


def test_grid_payoff_and_boundary_helpers_are_consistent() -> None:
    grid = pde_grid(spot=100.0, strike=105.0, s_steps=30)
    call = payoff_values(grid, 105.0, "call")
    put = payoff_values(grid, 105.0, "put")
    value = put + np.where(grid <= 90.0, 0.0, 0.25)

    assert len(grid) == 31
    assert grid[0] == 0.0
    assert grid[-1] >= 315.0
    assert call[0] == 0.0
    assert put[0] == 105.0
    assert extract_boundary(grid, value, put, "put") <= 90.0
    assert complementarity_error(value, put) == 0.0


def test_tridiagonal_steps_match_dense_linear_algebra() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.0, -0.1, -0.1, -0.1, 0.0])
    diag = np.array([1.0, 1.2, 1.2, 1.2, 1.0])
    upper = np.array([0.0, -0.1, -0.1, -0.1, 0.0])
    matrix = np.diag(diag) + np.diag(upper[:-1], 1) + np.diag(lower[1:], -1)

    explicit = explicit_step(v, lower, diag, upper)
    implicit = implicit_step(v, lower, diag, upper)
    cn = crank_nicolson_step(v, lower, diag, upper, lower, diag, upper)

    np.testing.assert_allclose(explicit[1:-1], lower[1:-1] * v[:-2] + diag[1:-1] * v[1:-1] + upper[1:-1] * v[2:])
    np.testing.assert_allclose(matrix @ implicit, v)
    assert cn.shape == v.shape


def test_psor_penalty_rannacher_and_diagnostics_preserve_exercise_constraint() -> None:
    rhs = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    payoff = np.array([1.0, 1.25, 1.15, 1.30, 1.0])
    lower = np.zeros_like(rhs)
    upper = np.zeros_like(rhs)
    diag = np.ones_like(rhs)

    value, residual, iterations = psor_solve(lower, diag, upper, rhs, payoff, omega=1.1, tol=1e-10)
    penalty_value = penalty_solve(lower, diag, upper, rhs, payoff, tol=1e-10)
    smoothed, residuals = rannacher_steps(rhs, lower, diag, upper, payoff, steps=2, omega=1.1, tol=1e-10)
    diagnostics = pde_diagnostics(value, payoff, np.array([residual]), iterations=np.array([iterations]))

    np.testing.assert_allclose(value, np.maximum(rhs, payoff))
    np.testing.assert_allclose(penalty_value, np.maximum(rhs, payoff))
    assert np.all(smoothed >= payoff)
    assert residuals.shape == (2,)
    assert diagnostics["complementarity_error"] < 1e-8
