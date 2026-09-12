"""Inflation component forecasts and release-conditioned cross-index bridges."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .bridge import bridge_forecast, fit_bridge


def ar_forecast(history, *, window=60, alpha=2.0) -> float:
    """Notebook AR(1) ridge forecast, with its short-history trailing-mean fallback."""
    values = pd.Series(history, dtype=float).dropna().tail(window)
    if len(values) < 18:
        return values.tail(12).mean()
    design = pd.concat([values.rename("y"), values.shift(1).rename("lag")], axis=1).dropna()
    model = Ridge(alpha=alpha).fit(design[["lag"]], design["y"])
    return float(model.predict(pd.DataFrame({"lag": [values.iloc[-1]]}))[0])


def component_weights(X, y, *, prior=(0.80, 0.14, 0.06), penalty=250.0) -> np.ndarray:
    """Nonnegative, normalized ridge weights shrunk toward an economic prior.

    These are estimated approximation weights, not official relative-importance
    weights. The caller may instead supply official weights to aggregation.
    """
    X, y, prior = np.asarray(X), np.asarray(y), np.asarray(prior)
    w = np.linalg.solve(X.T @ X + penalty * np.eye(len(prior)), X.T @ y + penalty * prior)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        raise ValueError("Component weights have no positive mass.")
    return w / w.sum()


def inflation_components(history: pd.DataFrame, energy_training: pd.DataFrame,
                           energy_current: pd.Series, *, prior=(0.80, 0.14, 0.06),
                           energy="gasoline") -> dict:
    """Persistent core, trailing food, and oil/gas-driven energy, then aggregation.

    All growth series must share the target's units and all training rows must
    already be released. ``energy`` names the component being modeled.
    """
    core = ar_forecast(history["core"], window=72, alpha=3.0)
    food = history["food"].tail(12).mean()
    names = ["last_energy", "gas_signal", "oil_signal"]
    train = energy_training.dropna(subset=[energy, *names])
    if len(train) >= 36 and energy_current[names].notna().all():
        model = Ridge(alpha=5).fit(train[names], train[energy])
        value = float(model.predict(energy_current[names].to_frame().T)[0])
    else:
        value = ar_forecast(history[energy], window=48, alpha=4.0)
    sample = history.dropna(subset=["headline", "core", "food", energy]).tail(120)
    w = component_weights(sample[["core", "food", energy]], sample["headline"], prior=prior)
    forecasts = pd.Series([core, food, value], index=["core", "food", energy])
    return {"mean": float(np.dot(w, forecasts)), "components": forecasts,
            "weights": pd.Series(w, index=forecasts.index)}


def fit_inflation_bridge(history: pd.DataFrame, current: pd.Series, *, features,
                          current_features=(), available="current_available", target="actual",
                          minimum=30, alpha=4.0) -> dict:
    """Fit only past observations sharing the current release-information regime."""
    regime = bool(current[available])
    train = history[history[available].eq(regime)]
    names = [*features, *(current_features if regime else ())]
    train = train.dropna(subset=[target, *names])
    fit = None
    if len(train) >= minimum and current[names].notna().all():
        fit = fit_bridge(train[names], train[target], alpha=alpha, minimum=minimum, residual_window=60)
    return {"fit": fit, "regime": regime, "fallback": current["last_release"],
            "fallback_sigma": train[target].tail(36).std(ddof=1)}


def inflation_forecast(fit: dict, current: pd.Series) -> dict:
    """Forecast the destination index after its source-index release regime is selected."""
    regression = fit["fit"]
    return {"mean": fit["fallback"] if regression is None else float(bridge_forecast(regression, current.to_frame().T)[0]),
            "sigma": fit["fallback_sigma"] if regression is None else regression.sigma,
            "current_available": fit["regime"]}


def market_growth(series: pd.Series, as_of, observation_date, *, periods=1, scale=1200) -> float:
    """Growth in available within-month market averages, with explicit lag and scale."""
    known = series.loc[:pd.Timestamp(as_of)]
    month = pd.Timestamp(observation_date).to_period("M")
    current = known[known.index.to_period("M") == month]
    previous = known[known.index.to_period("M") == month - periods]
    if current.empty or previous.empty or current.mean() <= 0 or previous.mean() <= 0:
        return np.nan
    return float(scale * np.log(current.mean() / previous.mean()))
