"""Monthly Minnesota VAR and conditional Gaussian coefficient/forecast draws.

The innovation covariance is the notebook's shrunk residual estimate. Quarterly
targets use a separate bridge; this is not a latent-monthly-GDP MF-BVAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minnesota_prior(columns, *, lags=2, tightness=0.20, persistent=(), own_lag=0.5) -> tuple:
    """Coefficient prior mean and lag-decaying precision on standardized variables."""
    n = len(columns)
    mean = np.zeros((1 + n * lags, n))
    for name in persistent:
        i = list(columns).index(name)
        mean[1 + i, i] = own_lag
    precision = np.r_[1e-6, np.repeat((np.arange(1, lags + 1) / tightness) ** 2, n)]
    return mean, np.diag(precision)


def fit_bvar(monthly: pd.DataFrame, *, lags=2, tightness=0.20,
               persistent=("unemployment", "policy"), covariance_shrinkage=0.10) -> dict:
    """Estimate the same standardized Minnesota system used in the notebook."""
    sample = monthly.dropna().copy()
    if len(sample) - lags <= 1 + sample.shape[1] * lags:
        raise ValueError("Too few complete monthly observations for residual covariance estimation.")
    if len(sample.index) > 1 and not np.all(np.diff(sample.index.to_period("M").asi8) == 1):
        raise ValueError("Complete BVAR observations must form a contiguous monthly sequence.")
    center, scale = sample.mean(), sample.std(ddof=0).replace(0, 1)
    z = (sample - center) / scale
    y = z.iloc[lags:].to_numpy()
    x = np.column_stack([np.ones(len(y)), *[z.shift(lag).iloc[lags:].to_numpy() for lag in range(1, lags + 1)]])
    mean, precision = minnesota_prior(sample.columns, lags=lags, tightness=tightness,
                                      persistent=persistent)
    V = np.linalg.inv(x.T @ x + precision)
    B = V @ (x.T @ y + precision @ mean)
    residual = y - x @ B
    covariance = residual.T @ residual / (len(residual) - x.shape[1])
    covariance = ((1 - covariance_shrinkage) * covariance
                  + covariance_shrinkage * np.diag(np.diag(covariance)))
    return {"columns": list(sample.columns), "center": center, "scale": scale, "z": z,
            "coefficients": B, "posterior_v": V, "covariance": covariance, "lags": lags}


def bvar_paths(fit: dict, *, steps=12, draws=300, rng=None) -> np.ndarray:
    """Draw coefficients once per path, then innovations each month; shape draws/steps/variables."""
    rng = np.random.default_rng() if rng is None else rng
    B = fit["coefficients"]
    left = np.linalg.cholesky(fit["posterior_v"] + 1e-10 * np.eye(len(B)))
    right = np.linalg.cholesky(fit["covariance"] + 1e-10 * np.eye(len(fit["columns"])))
    Z = rng.normal(size=(draws, B.shape[0], B.shape[1]))
    coefficients = B + np.einsum("ij,djk,kl->dil", left, Z, right.T)
    history = np.repeat(fit["z"].iloc[-fit["lags"]:].to_numpy()[None, :, :], draws, axis=0)
    simulations = []
    for _ in range(steps):
        x = np.concatenate([np.ones((draws, 1)), *[history[:, -lag, :] for lag in range(1, fit["lags"] + 1)]], axis=1)
        mu = np.einsum("dk,dkm->dm", x, coefficients)
        shock = rng.multivariate_normal(np.zeros(len(fit["columns"])), fit["covariance"], size=draws)
        value = mu + shock
        simulations.append(value)
        history = np.concatenate([history[:, 1:, :], value[:, None, :]], axis=1)
    return (np.stack(simulations, axis=1) * fit["scale"].to_numpy()[None, None, :]
            + fit["center"].to_numpy()[None, None, :])


def quarterly_bridge_draws(fit: dict, paths, factors: pd.Series, truth: pd.DataFrame,
                            quarter, *, activity="activity", rng=None) -> np.ndarray:
    """Notebook BVAR activity paths fed into the separate robust quarterly GDP bridge."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng() if rng is None else rng
    quarter = pd.Period(quarter, freq="Q")
    months = pd.period_range(quarter.start_time, quarter.end_time, freq="M").to_timestamp()
    last = fit["z"].index.max().to_period("M")
    column = fit["columns"].index(activity)
    activity_draws = []
    for month in months:
        if month in factors.index:
            activity_draws.append(np.repeat(factors.loc[month], paths.shape[0]))
        else:
            step = max(1, (month.to_period("M") - last).n)
            activity_draws.append(paths[:, min(step, paths.shape[1]) - 1, column])
    history = truth.sort_values("observation_date").copy()
    quarterly = factors.groupby(factors.index.to_period("Q")).mean()
    history["activity"] = history["observation_date"].dt.to_period("Q").map(quarterly)
    history["lag"] = history["first"].shift(1)
    training = history.dropna(subset=["first", "activity", "lag"])
    scaler = StandardScaler().fit(training[["activity", "lag"]])
    mu = training["first"].median()
    sigma = max(1.4826 * (training["first"] - mu).abs().median(), 1e-6)
    weights = 1 / (1 + ((training["first"] - mu) / (4 * sigma)) ** 2)
    model = Ridge(alpha=2).fit(scaler.transform(training[["activity", "lag"]]), training["first"], sample_weight=weights)
    residual = training["first"] - model.predict(scaler.transform(training[["activity", "lag"]]))
    current = pd.DataFrame({"activity": np.mean(activity_draws, axis=0), "lag": history["first"].iloc[-1]})
    return model.predict(scaler.transform(current)) + rng.normal(scale=max(residual.std(ddof=1), .5), size=paths.shape[0])
