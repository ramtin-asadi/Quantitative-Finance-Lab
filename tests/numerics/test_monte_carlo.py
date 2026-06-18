from __future__ import annotations

import numpy as np

from quantfinlab.numerics.monte_carlo import antithetic_normals, gbm_paths, payoff_paths


def test_antithetic_normals_are_paired_and_reproducible() -> None:
    z1 = antithetic_normals(paths=6, steps=4, seed=17)
    z2 = antithetic_normals(paths=6, steps=4, seed=17)

    assert z1.shape == (6, 4)
    np.testing.assert_allclose(z1, z2)
    np.testing.assert_allclose(z1[:3] + z1[3:], 0.0)


def test_gbm_paths_numpy_are_deterministic_and_payoffs_are_terminal_intrinsic_values() -> None:
    paths = gbm_paths(100.0, 0.03, 0.01, 0.20, 0.5, steps=6, paths=8, seed=21, engine="numpy")
    paths_again = gbm_paths(100.0, 0.03, 0.01, 0.20, 0.5, steps=6, paths=8, seed=21, engine="numpy")
    call_payoff = payoff_paths(paths[:, -1], 100.0, "call")
    put_payoff = payoff_paths(paths[:, -1], 100.0, "put")

    assert paths.shape == (8, 7)
    assert np.all(paths[:, 0] == 100.0)
    np.testing.assert_allclose(paths, paths_again)
    np.testing.assert_allclose(call_payoff, np.maximum(paths[:, -1] - 100.0, 0.0))
    np.testing.assert_allclose(put_payoff, np.maximum(100.0 - paths[:, -1], 0.0))
