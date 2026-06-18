from __future__ import annotations

import numpy as np
import pytest

from quantfinlab.options.american import tree_batch
from quantfinlab.options.bsm import bsm_price

pytestmark = pytest.mark.cpp


def _kernels():
    return pytest.importorskip("quantfinlab._kernels")


def test_cpp_tree_batch_matches_numpy_tree_and_european_bsm_reference() -> None:
    kernels = _kernels()
    spot = np.array([100.0, 100.0, 100.0])
    strike = np.array([90.0, 100.0, 110.0])
    rate = np.full(3, 0.03)
    dividend = np.full(3, 0.01)
    sigma = np.full(3, 0.22)
    tau = np.full(3, 0.5)
    flags = np.array([1, -1, -1], dtype=np.int32)

    cpp_american = kernels.american_tree_batch(spot, strike, rate, dividend, sigma, tau, flags, 80, 0, True)
    numpy_american = tree_batch(spot, strike, rate, dividend, sigma, tau, flags, steps=80, engine="numpy")
    wrapper_cpp = tree_batch(spot, strike, rate, dividend, sigma, tau, flags, steps=80, engine="cpp")
    cpp_european = kernels.american_tree_batch(spot, strike, rate, dividend, sigma, tau, flags, 80, 0, False)
    closed = np.array(
        [
            bsm_price("call", 100.0, 90.0, 0.5, 0.22, rate=0.03, dividend_yield=0.01),
            bsm_price("put", 100.0, 100.0, 0.5, 0.22, rate=0.03, dividend_yield=0.01),
            bsm_price("put", 100.0, 110.0, 0.5, 0.22, rate=0.03, dividend_yield=0.01),
        ]
    )

    np.testing.assert_allclose(cpp_american, numpy_american, atol=1e-10)
    np.testing.assert_allclose(wrapper_cpp, cpp_american, atol=1e-12)
    np.testing.assert_allclose(cpp_european, closed, atol=0.04)
    assert cpp_american[1] >= cpp_european[1]


def test_cpp_tree_boundary_returns_consistent_shapes_and_price() -> None:
    kernels = _kernels()
    boundary = kernels.american_tree_boundary(100.0, 100.0, 0.03, 0.01, 0.22, 0.5, -1, 50, 0, True)
    batch_price = kernels.american_tree_batch(
        np.array([100.0]),
        np.array([100.0]),
        np.array([0.03]),
        np.array([0.01]),
        np.array([0.22]),
        np.array([0.5]),
        np.array([-1], dtype=np.int32),
        50,
        0,
        True,
    )[0]

    assert {"times", "boundary", "price"} == set(boundary)
    assert boundary["times"].shape == (51,)
    assert boundary["boundary"].shape == (51,)
    assert boundary["boundary"][-1] == pytest.approx(100.0)
    assert boundary["price"] == pytest.approx(batch_price)
