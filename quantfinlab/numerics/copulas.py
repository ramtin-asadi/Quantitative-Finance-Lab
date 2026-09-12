"""Gaussian and Student-t copula draws with uniform marginal distributions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, t

from quantfinlab.portfolio.covariance import make_psd


def _correlated_normals(rho, n, rng):
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1] or not np.isfinite(rho).all():
        raise ValueError("rho must be a finite square correlation matrix.")
    if not np.allclose(rho, rho.T) or not np.allclose(np.diag(rho), 1):
        raise ValueError("rho must be symmetric with unit diagonal.")
    rho = make_psd(rho, eps=1e-8)
    sd = np.sqrt(np.diag(rho))
    rho = rho / np.outer(sd, sd)
    L = np.linalg.cholesky(rho + 1e-10 * np.eye(len(rho)))
    return rng.standard_normal((n, len(rho))) @ L.T


def gaussian_copula(rho, n: int, *, seed: int | None = None) -> np.ndarray:
    """Draw n correlated uniforms from a Gaussian copula."""
    return norm.cdf(_correlated_normals(rho, n, np.random.default_rng(seed)))


def student_t_copula(rho, n: int, *, nu: float = 5, seed: int | None = None) -> np.ndarray:
    """Draw n uniforms using one shared chi-square scale per multivariate draw."""
    if nu <= 0:
        raise ValueError("nu must be positive.")
    rng = np.random.default_rng(seed)
    z = _correlated_normals(rho, n, rng)
    scales = np.random.default_rng(seed + 1 if seed is not None else None)
    return t.cdf(z / np.sqrt(scales.chisquare(nu, size=(n, 1)) / nu), df=nu)
