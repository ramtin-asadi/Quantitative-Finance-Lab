"""Forecast weights and mixture moments, independent of forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_weights(history: pd.DataFrame, as_of, *, model="model", actual="actual",
                      prediction="mean", available="release_date", window=36,
                      minimum=8, scale=1.0, floor=0.02) -> pd.Series:
    """Exponential squared-error weights using only released outcomes.

    Supply one target and horizon at a time. Missing or immature model histories
    receive no weight; callers choose an explicit fallback if none is eligible.
    """
    known = history[history[available].lt(pd.Timestamp(as_of))].sort_values(available)
    losses = {}
    for name, group in known.groupby(model):
        x = group.dropna(subset=[actual, prediction]).tail(window)
        if len(x) >= minimum:
            losses[name] = ((x[actual] - x[prediction]) ** 2).mean() / max(scale ** 2, 1e-12)
    loss = pd.Series(losses, dtype=float)
    w = np.exp(-0.5 * (loss - loss.min())).clip(lower=floor)
    return w / w.sum()


def combine_forecasts(means, sigmas, weights) -> dict:
    """Mixture mean and standard deviation, including between-model dispersion."""
    mu, sigma, w = (np.asarray(x, dtype=float) for x in (means, sigmas, weights))
    if mu.shape != sigma.shape or mu.shape != w.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Means, standard deviations and nonnegative weights must align.")
    if not np.isfinite(mu + sigma + w).all() or np.any(sigma < 0):
        raise ValueError("Mixture inputs must be finite and standard deviations nonnegative.")
    w = w / w.sum()
    mean = np.dot(w, mu)
    variance = np.dot(w, sigma ** 2 + mu ** 2) - mean ** 2
    return {"mean": mean, "sigma": np.sqrt(max(variance, 0))}
