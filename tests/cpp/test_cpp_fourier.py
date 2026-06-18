from __future__ import annotations

import numpy as np
import pytest

from quantfinlab.options.bsm import bsm_price

pytestmark = pytest.mark.cpp


def _kernels():
    return pytest.importorskip("quantfinlab._kernels")


def test_cpp_direct_and_cos_bsm_prices_match_closed_form() -> None:
    kernels = _kernels()
    strikes = np.array([90.0, 100.0, 110.0])
    tau = np.full(3, 0.5)
    params = np.array([0.22])
    closed = bsm_price("call", 100.0, strikes, 0.5, 0.22, rate=0.03, dividend_yield=0.01)

    direct = kernels.direct_prices(0, params, strikes, tau, 100.0, 0.03, 0.01, 512, 120.0, 1)
    cos = kernels.cos_prices(0, params, strikes, tau, 100.0, 0.03, 0.01, 256, 10.0, 1)
    puts = kernels.direct_prices(0, params, strikes, tau, 100.0, 0.03, 0.01, 512, 120.0, -1)
    parity_residual = direct - puts - (100.0 * np.exp(-0.01 * tau) - strikes * np.exp(-0.03 * tau))

    np.testing.assert_allclose(direct, closed, atol=1e-8)
    np.testing.assert_allclose(cos, closed, atol=1e-8)
    np.testing.assert_allclose(parity_residual, 0.0, atol=1e-8)


def test_cpp_fft_returns_monotone_strikes_and_finite_nonnegative_prices() -> None:
    kernels = _kernels()
    out = kernels.fft_prices(0, np.array([0.22]), 100.0, 0.03, 0.01, 0.5, 1.5, 64, 0.25, 1)

    assert {"strikes", "prices"} == set(out)
    assert out["strikes"].shape == out["prices"].shape == (64,)
    assert np.all(np.diff(out["strikes"]) > 0.0)
    assert np.isfinite(out["prices"]).all()
    assert (out["prices"] >= 0.0).all()
