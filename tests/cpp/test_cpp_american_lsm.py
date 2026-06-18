from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpp


def _kernels():
    return pytest.importorskip("quantfinlab._kernels")


def test_cpp_gbm_paths_are_antithetic_and_start_at_spot() -> None:
    kernels = _kernels()
    paths = kernels.gbm_paths_antithetic(100.0, 0.03, 0.01, 0.20, 1.0, 5, 6, 123)
    dt = 1.0 / 5.0
    drift = (0.03 - 0.01 - 0.5 * 0.20**2) * dt
    half = paths.shape[0] // 2

    assert paths.shape == (6, 6)
    np.testing.assert_allclose(paths[:, 0], 100.0)
    pair_log_steps = np.diff(np.log(paths[:half]), axis=1) + np.diff(np.log(paths[half:]), axis=1)
    np.testing.assert_allclose(pair_log_steps, 2.0 * drift, atol=1e-12)


def test_cpp_lsm_backward_policy_round_trip_is_finite_and_consistent() -> None:
    kernels = _kernels()
    paths = kernels.gbm_paths_antithetic(100.0, 0.03, 0.01, 0.20, 1.0, 40, 160, 321)

    fit = kernels.lsm_backward(paths, 100.0, 0.03, 1.0, -1, 2)
    evaluated = kernels.lsm_eval_policy(paths, 100.0, 0.03, 1.0, -1, fit["coefficients"])

    assert {"price", "exercise_time", "coefficients"} == set(fit)
    assert fit["coefficients"].shape == (41, 3)
    assert fit["exercise_time"].shape == (160,)
    assert fit["exercise_time"].min() >= 1
    assert fit["exercise_time"].max() <= 40
    assert np.isfinite(fit["price"])
    assert fit["price"] > 0.0
    assert evaluated["price"] == pytest.approx(fit["price"])
    np.testing.assert_array_equal(evaluated["exercise_time"], fit["exercise_time"])
