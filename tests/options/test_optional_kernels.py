from __future__ import annotations

import numpy as np
import pytest

from quantfinlab import _optional
from quantfinlab.common.errors import MissingKernelsError
from quantfinlab.numerics.monte_carlo import gbm_paths
from quantfinlab.options.american import pde_price, tree_price
from quantfinlab.options.fourier import direct_price
from quantfinlab.options.rough import rough_heston_prices


def _hide_cpp_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import_module = _optional.import_module

    def fake_import_module(name: str):
        if name == "quantfinlab._kernels":
            raise ImportError("hidden compiled kernels")
        return original_import_module(name)

    monkeypatch.setattr(_optional, "import_module", fake_import_module)


def test_explicit_cpp_engine_raises_helpful_error_when_kernels_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_cpp_kernels(monkeypatch)

    with pytest.raises(MissingKernelsError, match="engine='cpp'"):
        tree_price(100.0, 100.0, 0.03, 0.0, 0.20, 0.5, "put", steps=8, engine="cpp")


def test_auto_engine_falls_back_without_compiled_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_cpp_kernels(monkeypatch)

    auto_tree = tree_price(100.0, 100.0, 0.03, 0.0, 0.20, 0.5, "put", steps=12, engine="auto")
    numpy_tree = tree_price(100.0, 100.0, 0.03, 0.0, 0.20, 0.5, "put", steps=12, engine="numpy")
    pde = pde_price(100.0, 100.0, 0.03, 0.0, 0.20, 0.25, "put", s_steps=30, t_steps=18, max_iter=400, engine="auto")
    paths = gbm_paths(100.0, 0.03, 0.0, 0.20, 0.25, steps=4, paths=6, seed=3, engine="auto")

    assert auto_tree == pytest.approx(numpy_tree, rel=2e-2, abs=2e-2)
    assert pde["price"] > 0.0
    assert pde["engine_used"] in {"numba", "numpy"}
    assert paths.shape[1] == 5
    assert np.allclose(paths[:, 0], 100.0)


def test_fourier_and_rough_defaults_are_usable_without_compiled_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    _hide_cpp_kernels(monkeypatch)

    price = direct_price("bsm", {"sigma": 0.20}, 100.0, np.array([95.0, 105.0]), 0.03, 0.0, 0.25, n=128, engine="auto")
    rough = rough_heston_prices(
        [0.15, 0.04, 1.5, 0.04, 0.35, -0.45],
        np.array([95.0, 105.0]),
        np.array([0.25, 0.25]),
        100.0,
        0.03,
        0.0,
        n_terms=16,
        riccati_steps=32,
    )

    assert np.isfinite(price).all()
    assert np.isfinite(rough).all()
    assert (rough >= 0.0).all()
