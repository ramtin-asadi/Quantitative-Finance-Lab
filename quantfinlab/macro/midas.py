"""Normalized MIDAS lag polynomials and mixed-frequency regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .bridge import bridge_forecast, fit_bridge


def beta_weights(length: int, a: float, b: float) -> np.ndarray:
    """Beta lag weights; position zero denotes the most recent observation."""
    if length < 1 or a <= 0 or b <= 0:
        raise ValueError("Length and beta shape parameters must be positive.")
    x = (np.arange(length) + 0.5) / length
    log_w = (a - 1) * np.log(x) + (b - 1) * np.log1p(-x)
    weights = np.exp(log_w - log_w.max())
    return weights / weights.sum()


def almon_weights(length: int, theta_1: float, theta_2: float) -> np.ndarray:
    """Normalized exponential Almon polynomial on lag positions."""
    k = np.arange(length, dtype=float)
    log_w = theta_1 * k + theta_2 * k ** 2
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def release_lags(values: pd.Series, as_of, length: int) -> np.ndarray:
    """Last length available observations, ordered most recent first."""
    x = values.loc[:pd.Timestamp(as_of)].dropna().tail(length)
    return x.iloc[::-1].to_numpy(dtype=float) if len(x) == length else np.full(length, np.nan)


def fit_beta_shape(lags, target) -> tuple:
    """Profile intercept/slope least squares while fitting positive beta shapes."""
    complete = np.isfinite(lags).all(axis=1) & np.isfinite(target)
    X, y = np.asarray(lags)[complete], np.asarray(target)[complete]
    if len(y) < 3:
        raise ValueError("Insufficient complete MIDAS training rows.")
    def loss(log_shape):
        signal = X @ beta_weights(X.shape[1], *np.exp(log_shape))
        design = np.column_stack([np.ones(len(signal)), signal])
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        return np.mean((y - fitted) ** 2)
    result = minimize(loss, np.log([1.5, 2.5]), method="L-BFGS-B",
                       bounds=[(np.log(0.25), np.log(10))] * 2)
    if not result.success:
        raise RuntimeError(result.message)
    return np.exp(result.x), result.fun


def fit_midas(lags: dict, y, *, controls=None, weights=None, alpha=3.0,
                minimum=36, robust=True) -> dict:
    """Fit one MIDAS regression; supplied lag weights support frozen pre-sample shapes."""
    fitted_weights, signals = {}, {}
    for name, X in lags.items():
        w = beta_weights(X.shape[1], *fit_beta_shape(X, y)[0]) if weights is None else weights[name]
        fitted_weights[name] = w
        signals[name] = np.asarray(X) @ w
    design = pd.DataFrame(signals, index=controls.index if controls is not None else None)
    if controls is not None:
        design = pd.concat([design, controls], axis=1)
    valid = design.notna().all(axis=1) & np.isfinite(y)
    fit = fit_bridge(design.loc[valid], np.asarray(y)[valid], alpha=alpha,
                      robust=robust, minimum=minimum, residual_window=60)
    return {"weights": fitted_weights, "regression": fit}


def midas_forecast(fit: dict, lags: dict, *, controls=None) -> np.ndarray:
    """Apply the fitted lag weights and regression to new information."""
    design = pd.DataFrame({name: np.atleast_2d(lags[name]) @ w for name, w in fit["weights"].items()})
    if controls is not None:
        design = pd.concat([design, controls.reset_index(drop=True)], axis=1)
    return bridge_forecast(fit["regression"], design)
