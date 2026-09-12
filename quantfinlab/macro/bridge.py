"""Monthly path completion and transparent target bridge regressions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def complete_months(values: pd.Series, months, *, difference=False) -> pd.Series:
    """Complete missing source months using lagged changes, never a future observation."""
    completed = values.copy().sort_index()
    for month in pd.DatetimeIndex(months):
        if pd.notna(completed.get(month, np.nan)):
            continue
        prior = completed.loc[completed.index < month].dropna()
        if prior.empty:
            continue
        if difference or (prior.tail(24) <= 0).any():
            forecast = prior.iloc[-1]
        else:
            growth = np.log(prior).diff().dropna().tail(18)
            growth = growth.clip(growth.quantile(0.10), growth.quantile(0.90))
            forecast = prior.iloc[-1] * np.exp(growth.median())
        completed.loc[month] = forecast
    return completed.sort_index()


def quarterly_signal(values: pd.Series, quarter, *, difference=False) -> dict:
    """Compare completed current-quarter and previous-quarter means."""
    quarter = pd.Period(quarter, freq="Q")
    current = pd.period_range(quarter.start_time, quarter.end_time, freq="M").to_timestamp()
    previous = (pd.period_range(quarter.start_time, periods=3, freq="M") - 3).to_timestamp()
    count = int(values.reindex(current).notna().sum())
    completed = complete_months(values, current, difference=difference)
    a, b = completed.reindex(current).mean(), values.reindex(previous).mean()
    value = a - b if difference or a <= 0 or b <= 0 else 400 * np.log(a / b)
    return {"signal": value, "observed_months": count, "completed": completed.reindex(current)}


@dataclass
class BridgeFit:
    """A fitted bridge and its training transformation and residual uncertainty."""
    model: object
    columns: list
    scaler: object
    median: pd.Series
    sigma: float


def fit_bridge(X: pd.DataFrame, y, *, alpha=4.0, robust=False,
                 minimum=20, residual_window=None) -> BridgeFit:
    """Ridge bridge with training medians and optional notebook outlier weights."""
    y = pd.Series(y, index=X.index)
    valid = y.notna()
    X, y = X.loc[valid].astype(float), y.loc[valid]
    if len(X) < minimum:
        raise ValueError(f"At least {minimum} released target observations are required.")
    median = X.median().fillna(0)
    X = X.fillna(median)
    scaler = StandardScaler().fit(X)
    weights = None
    if robust:
        scale = max(1.4826 * (y - y.median()).abs().median(), 1e-6)
        weights = 1 / (1 + ((y - y.median()) / (4 * scale)) ** 2)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), y, sample_weight=weights)
    residual = y - model.predict(scaler.transform(X))
    if residual_window is not None:
        residual = residual.tail(residual_window)
    return BridgeFit(model, list(X), scaler, median, float(residual.std(ddof=1)))


def bridge_forecast(fit: BridgeFit, X: pd.DataFrame) -> np.ndarray:
    """Forecast using stored training scaling and imputation."""
    X = X[fit.columns].astype(float).fillna(fit.median)
    return fit.model.predict(fit.scaler.transform(X))


def component_contributions(forecasts, weights) -> pd.DataFrame:
    """Apply supplied component weights without claiming chain-weighted accounting."""
    return pd.DataFrame(forecasts).mul(weights, axis=1)


def bridge_nowcast(training: pd.DataFrame, current: pd.Series, features,
                     target: str, *, alpha=4.0, minimum=20) -> tuple:
    """Notebook GDP bridge: select 65%-covered current inputs, fit, and return slopes."""
    usable = [name for name in features if training[name].notna().mean() >= 0.65
              and pd.notna(current[name])]
    sample = training[[target, *usable]].dropna(subset=[target])
    if len(sample) < minimum or not usable:
        return sample[target].tail(12).mean(), sample[target].tail(24).std(ddof=1), pd.Series(dtype=float)
    fit = fit_bridge(sample[usable], sample[target], alpha=alpha, minimum=minimum)
    prediction = bridge_forecast(fit, current[usable].to_frame().T)[0]
    coefficients = pd.Series(fit.model.coef_ / fit.scaler.scale_, index=usable)
    return float(prediction), fit.sigma, coefficients
