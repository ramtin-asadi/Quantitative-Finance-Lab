"""Merton equity-to-asset inversion and structural default probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def merton_assets(E, sigma_E, D, r, T=1.0, tol=1e-8, max_iter=100):
    """Solve equity-value and equity-volatility equations; return residual diagnostics."""
    E, sigma_E, D, r = np.broadcast_arrays(*[np.atleast_1d(x).astype(float) for x in (E, sigma_E, D, r)])
    if T <= 0:
        raise ValueError("T must be positive.")
    E = np.asarray(E, dtype=float)
    sigma_E = np.asarray(sigma_E, dtype=float)
    D = np.asarray(D, dtype=float)
    r = np.asarray(r, dtype=float)
    valid = np.isfinite(E + sigma_E + D + r) & (E > 0) & (sigma_E > 0) & (D > 0)
    result = {name: np.full(len(E), np.nan)
              for name in ["V", "sigma_V", "d1", "d2", "pd_merton", "residual"]}
    result["converged"] = np.zeros(len(E), dtype=bool)
    result["iterations"] = np.zeros(len(E), dtype=int)
    if not valid.any():
        return pd.DataFrame(result)
    Ev, sigma_Ev, Dv, rv = E[valid], sigma_E[valid], D[valid], r[valid]
    V = Ev + Dv * np.exp(-rv * T)
    sigma_V = sigma_Ev * Ev / V
    converged = np.zeros(len(Ev), dtype=bool)
    iterations = np.zeros(len(Ev), dtype=int)
    for iteration in range(1, max_iter + 1):
        d1 = (np.log(V / Dv) + (rv + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
        d2 = d1 - sigma_V * np.sqrt(T)
        Nd1 = np.clip(norm.cdf(d1), 1e-10, 1.0)
        V_new = (Ev + Dv * np.exp(-rv * T) * norm.cdf(d2)) / Nd1
        sigma_new = sigma_Ev * Ev / (Nd1 * V_new)
        change = np.maximum(np.abs(V_new / V - 1.0), np.abs(sigma_new / sigma_V - 1.0))
        newly_converged = ~converged & (change < tol)
        iterations[newly_converged] = iteration
        converged |= newly_converged
        V = 0.5 * V + 0.5 * V_new
        sigma_V = 0.5 * sigma_V + 0.5 * sigma_new
        if converged.all():
            break
    iterations[~converged] = max_iter
    d1 = (np.log(V / Dv) + (rv + 0.5 * sigma_V ** 2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    E_hat = V * norm.cdf(d1) - Dv * np.exp(-rv * T) * norm.cdf(d2)
    sigma_E_hat = norm.cdf(d1) * V * sigma_V / Ev
    residual = np.maximum(np.abs(E_hat / Ev - 1.0), np.abs(sigma_E_hat / sigma_Ev - 1.0))
    result["V"][valid] = V
    result["sigma_V"][valid] = sigma_V
    result["d1"][valid] = d1
    result["d2"][valid] = d2
    result["pd_merton"][valid] = norm.cdf(-d2)
    result["converged"][valid] = converged
    result["iterations"][valid] = iterations
    result["residual"][valid] = residual
    return pd.DataFrame(result)


def distance_to_default(V, sigma_V, D, mu, T=1.0):
    """Distance to the default barrier using an explicitly supplied asset drift."""
    V, sigma, D, mu = np.broadcast_arrays(V, sigma_V, D, mu)
    if T <= 0 or np.any(V <= 0) or np.any(D <= 0) or np.any(sigma <= 0):
        raise ValueError("Asset values, barrier, volatility and maturity must be positive.")
    return (np.log(V / D) + (mu - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def merton_pd(V, sigma_V, D, mu, T=1.0):
    """N(-DD); the drift supplied by the caller determines the probability measure."""
    return norm.cdf(-distance_to_default(V, sigma_V, D, mu, T))


def equity_volatility(returns: pd.DataFrame, *, window=252, minimum=126, annualization=252) -> pd.DataFrame:
    """Trailing daily equity volatility sampled on the final trading session each month."""
    sigma = returns.rolling(window, min_periods=minimum).std() * np.sqrt(annualization)
    dates = returns.index.to_series().groupby(returns.index.to_period("M")).max()
    result = sigma.loc[dates].rename_axis(index="date", columns="ticker").stack(
        future_stack=True).rename("sigma_E").reset_index().dropna(subset=["sigma_E"])
    result["decision_date"] = result["date"].dt.to_period("M").dt.to_timestamp("M")
    return result
