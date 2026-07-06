from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import optimize


def vg_cf(u, spot, rate, dividend_yield, tau, sigma, theta, nu):
    """Evaluate the Variance Gamma characteristic function of log spot at expiry.

    Parameters
    ----------
    u : array-like
        Complex Fourier argument.
    spot : float or array-like
        Spot price.
    rate : float or array-like
        Continuously compounded risk-free rate.
    dividend_yield : float or array-like
        Continuously compounded dividend yield.
    tau : float or array-like
        Time to expiry in years.
    sigma : float or array-like
        Volatility parameter.
    theta : float or array-like
        Variance Gamma drift/skew parameter.
    nu : float or array-like
        Variance rate parameter.

    Returns
    -------
    numpy.ndarray
        Martingale-corrected characteristic-function values.
    """

    u_arr = np.asarray(u, dtype=complex)
    sigma = np.asarray(sigma, dtype=float)
    theta = np.asarray(theta, dtype=float)
    nu = np.maximum(np.asarray(nu, dtype=float), 1e-8)
    tau = np.asarray(tau, dtype=float)
    mart = np.maximum(1.0 - theta * nu - 0.5 * sigma * sigma * nu, 1e-10)
    omega = np.log(mart) / nu
    drift = np.log(np.asarray(spot, dtype=float)) + (
        np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float) + omega
    ) * tau
    return np.exp(1j * u_arr * drift) * (1.0 - 1j * theta * nu * u_arr + 0.5 * sigma * sigma * nu * u_arr * u_arr) ** (-tau / nu)


def vg_price(option_type, spot, strike, tau, rate, dividend_yield, sigma, theta, nu, *, engine: str = "numba"):
    """Price vanilla options under the Variance Gamma model by Fourier integration.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    spot : array-like
        Spot prices.
    strike : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    rate : array-like
        Continuously compounded risk-free rates.
    dividend_yield : array-like
        Continuously compounded dividend yields.
    sigma, theta, nu : float
        Variance Gamma parameters.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='numba'
        Fourier-pricing backend.

    Returns
    -------
    numpy.ndarray
        Model option prices.
    """

    from quantfinlab.options.fourier import direct_price

    params = {"sigma": sigma, "theta": theta, "nu": nu}
    return direct_price("vg", params, spot, strike, rate, dividend_yield, tau, option_type=option_type, engine=engine)


def fit_variance_gamma(quotes: pd.DataFrame, weight_col: str = "obs_weight", max_nfev: int = 80, engine: str = "numba") -> dict:
    """Calibrate a Variance Gamma option-pricing model to a quote panel.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Calibration quotes containing spot, strike, rate, maturity, option type, mid
        price, and optional weights.
    weight_col : str, default='obs_weight'
        Observation-weight column.
    max_nfev : int, default=80
        Maximum least-squares evaluations per starting point.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='numba'
        Fourier-pricing backend.

    Returns
    -------
    dict
        Dictionary with model name, fitted parameters, quote-level fit, diagnostic
        summary, elapsed time, and engine metadata.
    """

    from quantfinlab.options.fourier import direct_price

    t0 = time.perf_counter()
    q = quotes.copy()
    if q.empty:
        return {"model": "vg", "params": pd.DataFrame(), "fit": pd.DataFrame(), "diag": pd.DataFrame(), "elapsed_sec": 0.0}
    target = q["mid"].to_numpy(float)
    scale = pd.to_numeric(q.get("calib_scale_px", q.get("half_spread", 1.0)), errors="coerce").fillna(1.0).clip(lower=1e-6).to_numpy(float)
    weight = pd.to_numeric(q.get(weight_col, 1.0), errors="coerce").fillna(1.0).clip(lower=0.05).to_numpy(float)
    atm = float(np.nanmedian(q.get("iv_mid", pd.Series(0.25, index=q.index))))
    starts = [np.array([atm, -0.05, 0.15]), np.array([0.8 * atm, -0.15, 0.35]), np.array([1.2 * atm, 0.05, 0.08])]
    bounds = ([0.03, -1.50, 0.01], [3.00, 1.50, 3.00])

    def residual(p):
        px = direct_price(
            "vg",
            {"sigma": p[0], "theta": p[1], "nu": p[2]},
            q["spot"].to_numpy(float),
            q["strike"].to_numpy(float),
            q["rate"].to_numpy(float),
            q.get("dividend_yield", pd.Series(0.0, index=q.index)).to_numpy(float),
            q["tau"].to_numpy(float),
            option_type=q["option_type"].to_numpy(),
            engine=engine,
        )
        return (np.asarray(px, dtype=float) - target) / scale * np.sqrt(weight)

    best = None
    for start in starts:
        res = optimize.least_squares(residual, np.clip(start, bounds[0], bounds[1]), bounds=bounds, max_nfev=int(max_nfev))
        loss = float(np.nanmean(res.fun**2))
        if best is None or loss < best[0]:
            best = (loss, res)
    p = best[1].x
    fit = q.copy()
    fit["model_price"] = direct_price(
        "vg",
        {"sigma": p[0], "theta": p[1], "nu": p[2]},
        fit["spot"].to_numpy(float),
        fit["strike"].to_numpy(float),
        fit["rate"].to_numpy(float),
        fit.get("dividend_yield", pd.Series(0.0, index=fit.index)).to_numpy(float),
        fit["tau"].to_numpy(float),
        option_type=fit["option_type"].to_numpy(),
        engine=engine,
    )
    fit["price_residual"] = fit["model_price"] - fit["mid"]
    params = pd.DataFrame([{"sigma": p[0], "theta": p[1], "nu": p[2], "loss": best[0], "success": bool(best[1].success), "nfev": int(best[1].nfev)}])
    diag = pd.DataFrame([{"model": "vg", "quotes": len(fit), "weighted_price_rmse": float(np.sqrt(np.nanmean(((fit["model_price"] - fit["mid"]) / scale) ** 2))), "median_abs_price_error": float(np.nanmedian(np.abs(fit["price_residual"])))}])
    return {"model": "vg", "params": params, "fit": fit, "diag": diag, "elapsed_sec": time.perf_counter() - t0, "engine": engine}


__all__ = ["fit_variance_gamma", "vg_cf", "vg_price"]
