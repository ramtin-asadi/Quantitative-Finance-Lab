from __future__ import annotations

import numpy as np
import pytest

from quantfinlab.numerics.fourier import carr_madan_fft_numba, cos_price_numba, direct_price_numba
from quantfinlab.options.bsm import bsm_price

pytestmark = pytest.mark.numerics


def test_numba_direct_and_cos_black_scholes_kernels_match_closed_form() -> None:
    pytest.importorskip("numba")
    strikes = np.array([90.0, 100.0, 110.0])
    spot = np.full_like(strikes, 100.0)
    rate = np.full_like(strikes, 0.03)
    dividend = np.full_like(strikes, 0.01)
    tau = np.full_like(strikes, 0.5)
    flags = np.ones_like(strikes, dtype=np.int32)
    params = np.array([0.22])

    direct = direct_price_numba(0, params, spot, strikes, rate, dividend, tau, flags, n=256, u_max=100.0)
    cos = cos_price_numba(0, params, spot, strikes, rate, dividend, tau, flags, n_terms=96, truncation_width=10.0)
    closed = bsm_price("call", 100.0, strikes, 0.5, 0.22, rate=0.03, dividend_yield=0.01)

    np.testing.assert_allclose(direct, closed, atol=0.15)
    np.testing.assert_allclose(cos, closed, atol=0.20)


def test_numba_carr_madan_fft_returns_increasing_strike_grid() -> None:
    pytest.importorskip("numba")

    strikes, prices = carr_madan_fft_numba(0, np.array([0.20]), 100.0, 0.03, 0.0, 0.4, n=64, eta=0.30)

    assert strikes.shape == prices.shape == (64,)
    assert np.all(np.diff(strikes) > 0)
    assert np.isfinite(prices).all()
    assert (prices >= 0).all()
