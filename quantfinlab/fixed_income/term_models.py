from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def acf1(values) -> float:
    """Compute first-order autocorrelation of a numeric series.

    Parameters
    ----------
    values : array-like
        Input values.

    Returns
    -------
    float
        Lag-1 autocorrelation, or NaN when fewer than three observations are
        available or the series has zero standard deviation.

    Notes
    -----
    Missing values are dropped before calculation.
    """

    x = pd.Series(values).dropna().astype(float)
    if len(x) < 3 or x.std() == 0:
        return float("nan")
    return float(np.corrcoef(x.iloc[:-1], x.iloc[1:])[0, 1])


def hessian_condition(result) -> float:
    """Estimate the condition number of an optimizer inverse Hessian.

    Parameters
    ----------
    result : object
        Optimizer result object with a ``hess_inv`` attribute.

    Returns
    -------
    float
        Matrix condition number, or NaN if the inverse Hessian cannot be converted
        to a dense numeric array.

    Notes
    -----
    The helper is intended for fit diagnostics and handles missing or incompatible
    optimizer attributes defensively.
    """

    try:
        hess_inv = np.asarray(result.hess_inv.todense(), dtype=float)
        return float(np.linalg.cond(hess_inv))
    except Exception:
        return float("nan")


def vasicek_ab(kappa, theta, sigma, tau):
    """Compute Vasicek affine bond-pricing coefficients.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run short-rate mean.
    sigma : float
        Short-rate volatility.
    tau : array-like
        Maturities in years.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Affine ``A`` and ``B`` components for the Vasicek zero-coupon bond formula.

    Notes
    -----
    The implementation floors ``kappa`` away from zero for numerical stability.
    """

    tau = np.asarray(tau, dtype=float)
    kappa = max(float(kappa), 1e-8)
    b = (1 - np.exp(-kappa * tau)) / kappa
    a = (theta - sigma**2 / (2 * kappa**2)) * (b - tau) - sigma**2 * b**2 / (4 * kappa)
    return a, b


def vasicek_loadings(params, maturities):
    """Compute Vasicek yield intercept and short-rate loading.

    Parameters
    ----------
    params : Mapping[str, float]
        Model parameters containing ``kappa``, ``theta``, and ``sigma``.
    maturities : array-like
        Maturities in years.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Affine yield intercept and short-rate loading by maturity.

    Notes
    -----
    The returned arrays correspond to the yield representation derived from the
    Vasicek zero-coupon bond formula.
    """

    a, b = vasicek_ab(params["kappa"], params["theta"], params["sigma"], maturities)
    maturities = np.asarray(maturities, dtype=float)
    return -a / maturities, b / maturities


def vasicek_yield_loading(params, maturities):
    """Return the Vasicek short-rate loading for yields.

    Parameters
    ----------
    params : Mapping[str, float]
        Model parameters containing ``kappa``, ``theta``, and ``sigma``.
    maturities : array-like
        Maturities in years.

    Returns
    -------
    numpy.ndarray
        Short-rate loading by maturity.
    """

    return vasicek_loadings(params, maturities)[1]


def vasicek_expected_average(r0, params, years=10.0):
    """Compute the expected average short rate under a Vasicek model.

    Parameters
    ----------
    r0 : float
        Initial short rate.
    params : Mapping[str, float]
        Model parameters containing ``kappa`` and ``theta``.
    years : float, default 10.0
        Averaging horizon in years.

    Returns
    -------
    float
        Expected average short rate over the horizon.

    Notes
    -----
    The formula integrates the conditional mean of the Ornstein-Uhlenbeck short
    rate over the specified horizon.
    """

    kappa = max(float(params["kappa"]), 1e-8)
    theta = float(params["theta"])
    return theta + (float(r0) - theta) * (1 - np.exp(-kappa * years)) / (kappa * years)


def fit_vasicek_ar(short_history):
    """Estimate fallback Vasicek parameters from a short-rate history.

    Parameters
    ----------
    short_history : array-like
        Time series of short rates, typically monthly.

    Returns
    -------
    dict
        Parameter dictionary containing ``kappa``, ``theta``, ``sigma``,
        ``obs_sd``, method label, and optimizer-success flag.

    Notes
    -----
    When enough observations are available, an AR(1) approximation is used to map
    discrete dynamics to continuous-time Vasicek parameters. With very short
    history, conservative fallback parameters are returned.
    """

    x = pd.Series(short_history).dropna().astype(float)
    if len(x) < 12:
        return {
            "kappa": 0.35,
            "theta": float(x.mean()) if len(x) else 0.03,
            "sigma": 0.01,
            "obs_sd": 0.001,
            "method": "ar fallback",
            "optimizer success": False,
        }
    y = x.iloc[1:].to_numpy()
    lag = x.iloc[:-1].to_numpy()
    slope, intercept = np.polyfit(lag, y, 1)
    phi = float(np.clip(slope, 0.02, 0.995))
    dt = 1 / 12
    kappa = -np.log(phi) / dt
    theta = intercept / max(1 - phi, 1e-6)
    resid = y - (intercept + slope * lag)
    sigma = float(np.std(resid, ddof=1) * np.sqrt(2 * kappa / max(1 - phi**2, 1e-8)))
    return {
        "kappa": kappa,
        "theta": theta,
        "sigma": max(sigma, 1e-5),
        "obs_sd": 0.0015,
        "method": "ar fallback",
        "optimizer success": False,
    }


def vasicek_kalman(yields, maturities, params, *, state_hint=None):
    """Run a one-factor Vasicek Kalman filter on a yield panel.

    Parameters
    ----------
    yields : array-like
        Yield observations with rows as dates and columns as maturities.
    maturities : array-like
        Maturities corresponding to the yield columns.
    params : Mapping[str, float]
        Model parameters including ``kappa``, ``theta``, ``sigma``, and optionally
        observation parameters.
    state_hint : array-like or None, optional
        Optional short-rate state hint used for initialization or intercept
        adjustment.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Negative log-likelihood and filtered short-rate state path.

    Notes
    -----
    The filter supports panels with missing yield observations. Missing maturities
    are skipped observation by observation.
    """

    data = np.asarray(yields, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    affine_level, b = vasicek_loadings(params, maturities)
    if state_hint is not None:
        hint = np.asarray(state_hint, dtype=float)
        if len(hint) == len(data):
            a = np.nanmean(data - hint[:, None] * b[None, :], axis=0)
        else:
            a = affine_level
    else:
        a = np.asarray(params.get("obs_alpha", affine_level), dtype=float)
    kappa = max(float(params["kappa"]), 1e-8)
    theta = float(params["theta"])
    sigma = max(float(params["sigma"]), 1e-8)
    obs_sd = max(float(params.get("obs_sd", 0.001)), 1e-8)
    dt = 1 / 12
    phi = np.exp(-kappa * dt)
    q = sigma**2 * (1 - phi**2) / (2 * kappa)
    mean = float(state_hint[0]) if state_hint is not None and len(state_hint) else theta
    var = sigma**2 / (2 * kappa)
    filtered = []
    nll = 0.0

    if np.isfinite(data).all():
        eye = np.eye(data.shape[1])
        outer_b = np.outer(b, b)
        log_const = data.shape[1] * np.log(2 * np.pi)
        for row in data:
            mean_pred = theta + phi * (mean - theta)
            var_pred = phi**2 * var + q
            innov = row - (a + b * mean_pred)
            s = var_pred * outer_b + obs_sd**2 * eye
            sign, logdet = np.linalg.slogdet(s)
            inv_s = np.linalg.pinv(s)
            nll += 0.5 * (logdet + innov @ inv_s @ innov + log_const) if sign > 0 else 1e6
            gain = var_pred * b @ inv_s
            mean = mean_pred + gain @ innov
            var = max(float((1 - gain @ b) * var_pred), 1e-10)
            filtered.append(mean)
        return float(nll), np.asarray(filtered)

    for row in data:
        mean_pred = theta + phi * (mean - theta)
        var_pred = phi**2 * var + q
        mask = np.isfinite(row)
        if mask.any():
            obs = row[mask]
            aa = a[mask]
            bb = b[mask]
            innov = obs - (aa + bb * mean_pred)
            s = var_pred * np.outer(bb, bb) + obs_sd**2 * np.eye(mask.sum())
            sign, logdet = np.linalg.slogdet(s)
            inv_s = np.linalg.pinv(s)
            nll += 0.5 * (logdet + innov @ inv_s @ innov + mask.sum() * np.log(2 * np.pi)) if sign > 0 else 1e6
            gain = var_pred * bb @ inv_s
            mean = mean_pred + gain @ innov
            var = max(float((1 - gain @ bb) * var_pred), 1e-10)
        else:
            mean, var = mean_pred, var_pred
        filtered.append(mean)
    return float(nll), np.asarray(filtered)


def fit_vasicek_kalman(yields, maturities, *, maxiter: int = 80):
    """Fit Vasicek short-rate dynamics and filter the latent short rate.

    Parameters
    ----------
    yields : array-like or pandas.DataFrame
        Yield panel with observations in rows and maturities in columns.
    maturities : array-like
        Maturities corresponding to the yield columns.
    maxiter : int, default 80
        Maximum number of optimizer iterations for the OU transition fit.

    Returns
    -------
    tuple[dict, pandas.Series]
        Parameter dictionary and filtered short-rate series.

    Notes
    -----
    The routine estimates transition parameters from the shortest-maturity history
    using bounded maximum likelihood, then runs the Kalman filter across the yield
    panel. If optimization fails, AR-based fallback parameters are used.
    """

    y = pd.DataFrame(yields).dropna(how="all")
    y_values = y.to_numpy(float)
    maturities = np.asarray(maturities, dtype=float)
    short_history = y.iloc[:, 0]
    short_values = y_values[:, 0]
    base = fit_vasicek_ar(short_history)
    x = short_history.to_numpy(float)
    r0 = x[:-1]
    r1 = x[1:]
    dt = 1 / 12

    def objective(z):
        kappa = np.exp(z[0])
        theta = z[1]
        sigma = np.exp(z[2])
        phi = np.exp(-kappa * dt)
        mean = theta + phi * (r0 - theta)
        var = np.maximum(sigma**2 * (1 - phi**2) / (2 * kappa), 1e-12)
        return float(0.5 * np.sum(np.log(2 * np.pi * var) + (r1 - mean) ** 2 / var))

    x0 = np.array([np.log(np.clip(base["kappa"], 0.01, 3.0)), base["theta"], np.log(np.clip(base["sigma"], 0.001, 0.08))])
    bounds = [(np.log(0.01), np.log(5.0)), (-0.05, 0.18), (np.log(0.0005), np.log(0.12))]

    try:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("vasicek optimization failed")
        kappa = float(np.exp(result.x[0]))
        theta = float(result.x[1])
        sigma = float(np.exp(result.x[2]))
        phi = np.exp(-kappa * dt)
        transition_resid = r1 - (theta + phi * (r0 - theta))
        params = {
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
            "obs_sd": 0.00075,
            "method": "ou mle + kalman filter",
            "optimizer success": True,
            "hessian condition": hessian_condition(result),
            "residual acf1": acf1(transition_resid),
        }
    except Exception as exc:
        params = {**base, "fit message": str(exc), "hessian condition": np.nan}
    _, filtered = vasicek_kalman(y_values, maturities, params, state_hint=short_values)
    if "residual acf1" not in params:
        params["residual acf1"] = acf1(short_values - filtered)
    return params, pd.Series(filtered, index=y.index, name="filtered short rate")


def cir_expected_average(r0, params, years=10.0):
    """Compute the expected average short rate under a shifted CIR model.

    Parameters
    ----------
    r0 : float
        Initial observed short rate.
    params : Mapping[str, float]
        Model parameters containing ``kappa``, ``theta``, and optional ``shift``.
    years : float, default 10.0
        Averaging horizon in years.

    Returns
    -------
    float
        Expected average observed short rate over the horizon.

    Notes
    -----
    The expectation is computed in shifted-rate space and then transformed back by
    subtracting the shift.
    """

    kappa = max(float(params["kappa"]), 1e-8)
    theta = float(params["theta"])
    shift = float(params.get("shift", 0.0))
    shifted_r0 = float(r0) + shift
    avg_shifted = theta + (shifted_r0 - theta) * (1 - np.exp(-kappa * years)) / (kappa * years)
    return avg_shifted - shift


def cir_yield_loading(params, maturities):
    """Compute an approximate CIR yield loading.

    Parameters
    ----------
    params : Mapping[str, float]
        Model parameters containing ``kappa``.
    maturities : array-like
        Maturities in years.

    Returns
    -------
    numpy.ndarray
        Mean-reversion loading by maturity.

    Notes
    -----
    The loading is a simplified exponential mean-reversion loading and is not a
    full closed-form CIR bond-pricing coefficient.
    """

    kappa = max(float(params["kappa"]), 1e-6)
    t = np.asarray(maturities, dtype=float)
    return (1 - np.exp(-kappa * t)) / (kappa * np.maximum(t, 1e-8))


def fit_cir_fast(short_history, *, maxiter: int = 160, fit_log: list[dict] | None = None):
    """Fit a fast shifted-CIR approximation to short-rate history.

    Parameters
    ----------
    short_history : array-like
        Short-rate series, typically monthly.
    maxiter : int, default 160
        Maximum number of optimizer iterations.
    fit_log : list of dict or None, optional
        Optional mutable list that receives failure messages.

    Returns
    -------
    dict
        Parameter dictionary containing ``kappa``, ``theta``, ``sigma``, ``shift``,
        method label, optimizer-success flag, Hessian condition, residual
        autocorrelation, parameter bounds, and fit message.

    Notes
    -----
    The routine shifts rates upward when needed to keep the CIR state positive,
    uses a quasi-likelihood based on Euler dynamics, and penalizes Feller-condition
    violations. If optimization fails, bounded AR-based fallback parameters are
    returned.
    """

    raw = pd.Series(short_history).dropna().astype(float)
    shift = max(0.0, 0.002 - float(raw.min())) if len(raw) else 0.0
    x = raw + shift
    x_values = x.to_numpy(float)
    r0 = x_values[:-1]
    r1 = x_values[1:]
    r0_floor = np.maximum(r0, 1e-8)
    log_two_pi = np.log(2 * np.pi)
    dt = 1 / 12
    theta_low = max(float(x.mean()) * 0.35, 1e-5)
    theta_high = max(float(x.mean()) * 2.50, theta_low + 0.005)

    def objective(z):
        kappa, theta, sigma = np.exp(z[0]), np.exp(z[1]), np.exp(z[2])
        mean = r0 + kappa * (theta - r0) * dt
        var = np.maximum(sigma**2 * r0_floor * dt, 1e-10)
        nll = 0.5 * np.sum(log_two_pi + np.log(var) + (r1 - mean) ** 2 / var)
        feller_gap = max(0.0, sigma**2 - 2 * kappa * theta)
        return float(nll + 1e7 * feller_gap**2)

    ar = fit_vasicek_ar(raw)
    x0 = np.log([
        np.clip(ar["kappa"], 0.01, 2.0),
        np.clip(raw.mean() + shift, theta_low, theta_high),
        np.clip(ar["sigma"], 0.001, 0.08),
    ])
    bounds = [(np.log(0.005), np.log(4.0)), (np.log(theta_low), np.log(theta_high)), (np.log(0.0005), np.log(0.12))]
    try:
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": int(maxiter)})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(str(result.message))
        kappa, theta, sigma = np.exp(result.x)
        status = "quasi mle"
        success = True
        message = str(result.message)
        hcond = hessian_condition(result)
    except Exception as exc:
        kappa = ar["kappa"]
        theta = np.clip(raw.mean() + shift, theta_low, theta_high)
        sigma = max(min(ar["sigma"], 0.08), 0.001)
        status = "bounded ar fallback"
        success = False
        message = str(exc)
        hcond = np.nan
        if fit_log is not None:
            fit_log.append({"end date": raw.index[-1] if len(raw) else pd.NaT, "message": message})

    resid = r1 - (r0 + kappa * (theta - r0) * dt)
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "shift": float(shift),
        "method": status,
        "optimizer success": success,
        "hessian condition": hcond,
        "residual acf1": acf1(resid),
        "theta lower": theta_low,
        "theta upper": theta_high,
        "fit message": message,
    }


def fit_cir(short_history, **kwargs):
    """Fit a shifted-CIR short-rate model.

    Parameters
    ----------
    short_history : array-like
        Short-rate history.
    **kwargs
        Additional keyword arguments forwarded to the fast CIR fitting routine.

    Returns
    -------
    dict
        Fitted or fallback CIR parameter dictionary.

    Notes
    -----
    This function is a public convenience alias for the fast CIR fitting routine.
    """

    return fit_cir_fast(short_history, **kwargs)


def simulate_vasicek_paths(r0, params, years=5.0, steps_per_year=12, n_paths=200, seed=9):
    """Simulate short-rate paths under a Vasicek model.

    Parameters
    ----------
    r0 : float
        Initial short rate.
    params : Mapping[str, float]
        Model parameters containing ``kappa``, ``theta``, and ``sigma``.
    years : float, default 5.0
        Simulation horizon in years.
    steps_per_year : int, default 12
        Number of time steps per year.
    n_paths : int, default 200
        Number of simulated paths.
    seed : int, default 9
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Simulated paths indexed by time in years, with one column per path.

    Notes
    -----
    The exact Gaussian transition of the Ornstein-Uhlenbeck process is used.
    """

    local_rng = np.random.default_rng(seed)
    steps = int(years * steps_per_year)
    dt = 1 / steps_per_year
    paths = np.empty((steps + 1, n_paths))
    paths[0] = float(r0)
    kappa = max(float(params["kappa"]), 1e-8)
    theta = float(params["theta"])
    sigma = max(float(params["sigma"]), 1e-8)
    phi = np.exp(-kappa * dt)
    std = sigma * np.sqrt((1 - phi**2) / (2 * kappa))
    for step in range(1, steps + 1):
        paths[step] = theta + phi * (paths[step - 1] - theta) + std * local_rng.normal(size=n_paths)
    return pd.DataFrame(paths, index=np.arange(steps + 1) / steps_per_year)


def simulate_cir_paths(r0, params, years=5.0, steps_per_year=12, n_paths=200, seed=10):
    """Simulate shifted-CIR short-rate paths with Euler discretization.

    Parameters
    ----------
    r0 : float
        Initial observed short rate.
    params : Mapping[str, float]
        Model parameters containing ``kappa``, ``theta``, ``sigma``, and optional
        ``shift``.
    years : float, default 5.0
        Simulation horizon in years.
    steps_per_year : int, default 12
        Number of time steps per year.
    n_paths : int, default 200
        Number of simulated paths.
    seed : int, default 10
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Simulated observed-rate paths indexed by time in years.

    Notes
    -----
    Simulation is performed in shifted positive-rate space and transformed back by
    subtracting the shift.
    """

    local_rng = np.random.default_rng(seed)
    steps = int(years * steps_per_year)
    dt = 1 / steps_per_year
    shift = float(params.get("shift", 0.0))
    paths = np.empty((steps + 1, n_paths))
    paths[0] = max(float(r0) + shift, 1e-8)
    for step in range(1, steps + 1):
        prev = np.maximum(paths[step - 1], 1e-8)
        drift = params["kappa"] * (params["theta"] - prev) * dt
        shock = params["sigma"] * np.sqrt(prev * dt) * local_rng.normal(size=n_paths)
        paths[step] = np.maximum(prev + drift + shock, 1e-8)
    return pd.DataFrame(paths - shift, index=np.arange(steps + 1) / steps_per_year)


def ar1_expected_score(scores):
    """Estimate a one-step AR(1) forecast and persistence coefficient.

    Parameters
    ----------
    scores : array-like
        Time series of factor scores or model scores.

    Returns
    -------
    tuple[float, float]
        Expected next score and clipped AR(1) slope. Returns ``(0.0, 0.0)`` when
        there is insufficient variation or history.

    Notes
    -----
    The AR coefficient is clipped to a conservative range to reduce instability in
    small rolling samples.
    """

    x = pd.Series(scores).dropna().astype(float)
    if len(x) < 12 or x.std() == 0:
        return 0.0, 0.0
    y = x.iloc[1:].to_numpy()
    lag = x.iloc[:-1].to_numpy()
    rho, intercept = np.polyfit(lag, y, 1)
    rho = float(np.clip(rho, -0.35, 0.35))
    return float(intercept + rho * x.iloc[-1]), rho


def estimate_mean_reversion(series):
    """Estimate annualized mean reversion from a time series.

    Parameters
    ----------
    series : array-like
        Time series assumed to be sampled monthly.

    Returns
    -------
    float
        Annualized mean-reversion speed. Returns a default value when fewer than
        24 observations are available.

    Notes
    -----
    The estimate maps a monthly AR(1) slope to a continuous-time mean-reversion
    speed using ``-log(phi) * 12`` with slope clipping.
    """

    x = pd.Series(series).dropna().astype(float)
    if len(x) < 24:
        return 0.25
    y = x.iloc[1:].to_numpy()
    lag = x.iloc[:-1].to_numpy()
    slope, _ = np.polyfit(lag, y, 1)
    phi = np.clip(slope, 0.05, 0.98)
    return float(-np.log(phi) * 12)


def hw1f_loading(a, maturities):
    """Compute one-factor Hull-White-style yield loadings.

    Parameters
    ----------
    a : float
        Mean-reversion speed.
    maturities : array-like
        Maturities in years.

    Returns
    -------
    numpy.ndarray
        Yield loading by maturity.

    Notes
    -----
    The loading equals ``(1 - exp(-a T)) / (a T)`` with a small floor on ``a`` for
    numerical stability.
    """

    maturities = np.asarray(maturities, dtype=float)
    a = max(float(a), 1e-6)
    return (1 - np.exp(-a * maturities)) / (a * maturities)


def estimate_hw1f(history, maturities):
    """Estimate a one-factor curve-change model from zero-rate history.

    Parameters
    ----------
    history : pandas.DataFrame
        Zero-rate history with maturity columns.
    maturities : array-like
        Maturities to model.

    Returns
    -------
    dict
        Model dictionary containing mean-reversion speed, monthly factor volatility,
        loading vector, expected curve change, factor autocorrelation, and variance
        share.

    Notes
    -----
    The routine estimates a Hull-White-style loading, projects monthly curve
    changes onto that loading, and summarizes how much cross-sectional variance is
    explained by the one-factor approximation.
    """

    maturities = np.asarray(maturities, dtype=float)
    changes = history[maturities].diff().dropna()
    short_col = 0.25 if 0.25 in history else history.columns[0]
    a = estimate_mean_reversion(history[short_col])
    loading = hw1f_loading(a, maturities)
    factors = changes.to_numpy(float) @ loading / max(float(loading @ loading), 1e-10)
    sigma_month = max(float(np.std(factors, ddof=1)), 1e-6)
    expected_factor, rho = ar1_expected_score(factors)
    fitted = np.outer(factors, loading)
    variance_share = 1 - np.nanvar(changes.to_numpy(float) - fitted) / max(np.nanvar(changes.to_numpy(float)), 1e-12)
    return {
        "a": a,
        "sigma monthly": sigma_month,
        "loading": loading,
        "expected change": expected_factor * loading,
        "factor rho": rho,
        "variance share": float(np.clip(variance_share, 0, 1)),
    }


def simulate_hw1f_curves(base_curve, params, n_scenarios=2000, seed=21):
    """Simulate curve scenarios from a one-factor curve-change model.

    Parameters
    ----------
    base_curve : array-like
        Current zero-rate curve values.
    params : Mapping[str, Any]
        One-factor model parameters containing ``sigma monthly``, ``loading``, and
        optionally ``expected change``.
    n_scenarios : int, default 2000
        Number of simulated curve scenarios.
    seed : int, default 21
        Random seed.

    Returns
    -------
    numpy.ndarray
        Simulated curve scenarios with shape ``(n_scenarios, n_maturities)``.

    Notes
    -----
    Scenarios add expected curve change and one Gaussian factor shock multiplied by
    the estimated loading.
    """

    local_rng = np.random.default_rng(seed)
    base = np.asarray(base_curve, dtype=float)
    shocks = local_rng.normal(0, params["sigma monthly"], size=n_scenarios)
    return base + params.get("expected change", 0) + shocks[:, None] * params["loading"][None, :]


def g2_loadings(a, b, maturities):
    """Construct two orthogonalized curve loadings.

    Parameters
    ----------
    a : float
        Mean-reversion parameter for the first loading.
    b : float
        Mean-reversion parameter for the second loading.
    maturities : array-like
        Maturities in years.

    Returns
    -------
    numpy.ndarray
        Two-column loading matrix.

    Notes
    -----
    The second loading is orthogonalized against the first and rescaled to a
    comparable norm. The result is a stylized two-factor curve-shape basis.
    """

    first = hw1f_loading(a, maturities)
    second = hw1f_loading(b, maturities)
    second = second - first * float(first @ second) / max(float(first @ first), 1e-10)
    norm = np.sqrt(max(float(second @ second), 1e-10))
    second = second / norm * np.sqrt(max(float(first @ first), 1e-10))
    return np.column_stack([first, second])


def estimate_g2_style(history, maturities):
    """Estimate a two-factor curve-change model from zero-rate history.

    Parameters
    ----------
    history : pandas.DataFrame
        Zero-rate history with maturity columns.
    maturities : array-like
        Maturities to model.

    Returns
    -------
    dict
        Model dictionary containing loading parameters, loading matrix, factor
        covariance, expected factors, expected curve change, factor persistence,
        factor correlation, and degeneracy flag.

    Notes
    -----
    Curve changes are projected onto two stylized loadings. If the estimated factor
    covariance is nearly degenerate, a small diagonal regularization is added.
    """

    maturities = np.asarray(maturities, dtype=float)
    changes = history[maturities].diff().dropna().to_numpy(float)
    short_col = 0.25 if 0.25 in history else history.columns[0]
    a = estimate_mean_reversion(history[short_col])
    b = max(0.03, min(1.20, a / 4))
    loadings = g2_loadings(a, b, maturities)
    factors = np.linalg.lstsq(loadings, changes.T, rcond=None)[0].T
    cov = np.cov(factors, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    corr_raw = np.corrcoef(factors.T)[0, 1] if factors.shape[0] > 2 else 0.0
    degenerate = bool(np.min(eig) < 1e-10 or abs(corr_raw) > 0.97)
    if degenerate:
        cov = cov + np.eye(2) * max(float(np.trace(cov)) * 0.03, 1e-8)
    expected = np.array([ar1_expected_score(factors[:, j])[0] for j in range(2)])
    rho = np.array([ar1_expected_score(factors[:, j])[1] for j in range(2)])
    corr = cov[0, 1] / max(np.sqrt(cov[0, 0] * cov[1, 1]), 1e-12)
    return {
        "a": a,
        "b": b,
        "loadings": loadings,
        "factor covariance": cov,
        "expected factors": expected,
        "expected change": loadings @ expected,
        "factor rho": rho,
        "factor correlation": float(corr),
        "degenerate": degenerate,
    }


def simulate_g2_curves(base_curve, params, n_scenarios=2000, seed=31):
    """Simulate curve scenarios from a two-factor curve-change model.

    Parameters
    ----------
    base_curve : array-like
        Current zero-rate curve values.
    params : Mapping[str, Any]
        Two-factor model parameters containing ``factor covariance``, ``loadings``,
        and optionally ``expected change``.
    n_scenarios : int, default 2000
        Number of simulated scenarios.
    seed : int, default 31
        Random seed.

    Returns
    -------
    numpy.ndarray
        Simulated curve scenarios with shape ``(n_scenarios, n_maturities)``.

    Notes
    -----
    Factor shocks are drawn from the fitted two-factor covariance matrix and
    projected through the loading matrix.
    """

    local_rng = np.random.default_rng(seed)
    base = np.asarray(base_curve, dtype=float)
    factors = local_rng.multivariate_normal(np.zeros(2), params["factor covariance"], size=n_scenarios)
    return base + params.get("expected change", 0) + factors @ params["loadings"].T


def orient_pca_loadings(loadings, scores):
    """Orient PCA loadings and scores for consistent curve-factor interpretation.

    Parameters
    ----------
    loadings : numpy.ndarray
        PCA loading matrix with maturities in rows and components in columns.
    scores : numpy.ndarray
        PCA score matrix with observations in rows and components in columns.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Sign-adjusted loadings and scores.

    Notes
    -----
    The first component is oriented as a positive level factor, the second as a
    positive-slope factor, and the third as a negative-belly curvature factor when
    available.
    """

    loadings = loadings.copy()
    scores = scores.copy()
    if loadings[:, 0].mean() < 0:
        loadings[:, 0] *= -1
        scores[:, 0] *= -1
    if loadings[-1, 1] - loadings[0, 1] < 0:
        loadings[:, 1] *= -1
        scores[:, 1] *= -1
    middle = len(loadings) // 2
    if loadings[middle, 2] > 0:
        loadings[:, 2] *= -1
        scores[:, 2] *= -1
    return loadings, scores


def estimate_pca_curve(history, maturities, n_components=3):
    """Estimate PCA curve factors from zero-rate changes.

    Parameters
    ----------
    history : pandas.DataFrame
        Zero-rate history with maturity columns.
    maturities : array-like
        Maturities to include.
    n_components : int, default 3
        Number of principal components to estimate.

    Returns
    -------
    dict
        PCA output containing loadings, scores, explained variances, explained
        variance ratios, and AR(1) persistence estimates for component scores.

    Notes
    -----
    PCA is fit to demeaned first differences of the selected zero-rate history, and
    loadings are sign-oriented for stable interpretation.
    """

    maturities = np.asarray(maturities, dtype=float)
    changes = history[maturities].diff().dropna()
    demeaned = changes - changes.mean()
    fit = PCA(n_components=n_components).fit(demeaned)
    loadings = fit.components_.T
    scores = fit.transform(demeaned)
    loadings, scores = orient_pca_loadings(loadings, scores)
    score_rho = np.array([ar1_expected_score(scores[:, j])[1] for j in range(n_components)])
    return {
        "loadings": loadings,
        "scores": scores,
        "variances": fit.explained_variance_,
        "explained": fit.explained_variance_ratio_,
        "score rho": score_rho,
    }


def rolling_pca_diagnostics(zero_rates, maturities, dates=None, window=60, n_components=3):
    """Compute rolling PCA explained-variance diagnostics.

    Parameters
    ----------
    zero_rates : pandas.DataFrame
        Date-indexed zero-rate panel.
    maturities : array-like
        Maturities included in the PCA.
    dates : array-like or None, optional
        Dates at which to run the rolling diagnostic. If omitted, dates start after
        the initial window.
    window : int, default 60
        Rolling window length.
    n_components : int, default 3
        Number of PCA components.

    Returns
    -------
    pandas.DataFrame
        Diagnostic table indexed by date with total explained variance and
        component-level explained variance ratios.

    Notes
    -----
    Each row fits PCA using observations up to the diagnostic date and the last
    ``window`` rows.
    """

    dates = pd.DatetimeIndex(dates if dates is not None else zero_rates.index[int(window):])
    rows = []
    for date in dates:
        fit = estimate_pca_curve(zero_rates.loc[:date].tail(int(window)), maturities, n_components=n_components)
        row = {"date": date, "three pc total": float(fit["explained"].sum())}
        for j in range(n_components):
            row[f"pc{j + 1}"] = float(fit["explained"][j])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def _fit_view_on_full_curve(history, fit_maturities, out_maturities, model_name):
    if model_name == "hw1f":
        fit = estimate_hw1f(history, fit_maturities)
        full_cov = fit["sigma monthly"] ** 2 * np.outer(fit["loading"], fit["loading"])
    else:
        fit = estimate_g2_style(history, fit_maturities)
        full_cov = fit["loadings"] @ fit["factor covariance"] @ fit["loadings"].T
    expected = pd.Series(fit["expected change"], index=fit_maturities).reindex(out_maturities).to_numpy(float)
    cov = pd.DataFrame(full_cov, index=fit_maturities, columns=fit_maturities).loc[out_maturities, out_maturities].to_numpy(float)
    return expected, cov


def model_curve_view(
    model_name,
    date,
    maturities,
    *,
    zero_rates,
    short_rate,
    model_parameters,
    scenario_maturities,
    rolling_window=60,
):
    """Generate a model-implied expected curve change and covariance matrix.

    Parameters
    ----------
    model_name : str
        Model identifier. Supported names include Vasicek-style, CIR, one-factor
        curve, and two-factor curve specifications.
    date : date-like
        Decision date.
    maturities : array-like
        Scenario maturities for the returned view.
    zero_rates : pandas.DataFrame
        Historical zero-rate panel.
    short_rate : pandas.Series
        Short-rate history indexed by date.
    model_parameters : pandas.DataFrame
        Time-indexed fitted model-parameter table.
    scenario_maturities : array-like
        Full maturity grid used for fitting curve-shape models.
    rolling_window : int, default 60
        Number of trailing observations used for full-curve model views.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Expected curve change vector and covariance matrix for the requested
        maturities.

    Notes
    -----
    The function dispatches to different curve-view approximations based on model
    name. Short-rate models use parameter rows available up to ``date``; curve
    shape models are estimated from a rolling zero-rate history.
    """

    history = zero_rates.loc[:date].tail(int(rolling_window))
    maturities = np.asarray(maturities, dtype=float)
    dt = 1 / 12
    name = str(model_name).lower()
    if name == "cir":
        p = model_parameters.loc[:date].iloc[-1]
        params = {"kappa": p["cir kappa"], "theta": p["cir theta"], "sigma": p["cir sigma"], "shift": p["cir shift"]}
        loading = cir_yield_loading(params, maturities)
        shifted_short = float(short_rate.loc[date]) + params["shift"]
        drift = loading * params["kappa"] * (params["theta"] - shifted_short) * dt
        cov = params["sigma"] ** 2 * max(shifted_short, 1e-6) * dt * np.outer(loading, loading)
        return drift, cov
    if name in {"hw1f", "g2"}:
        return _fit_view_on_full_curve(history, np.asarray(scenario_maturities, dtype=float), maturities, name)
    p = model_parameters.loc[:date].iloc[-1]
    params = {"kappa": p["vasicek kappa"], "theta": p["vasicek theta"], "sigma": p["vasicek sigma"]}
    loading = vasicek_yield_loading(params, maturities)
    drift = loading * params["kappa"] * (params["theta"] - short_rate.loc[date]) * dt
    cov = params["sigma"] ** 2 * dt * np.outer(loading, loading)
    return drift, cov


def rolling_model_views(dates, model_names, maturities, **kwargs):
    """Compute model-implied curve views over dates and model names.

    Parameters
    ----------
    dates : array-like
        Decision dates.
    model_names : iterable of str
        Model names to evaluate at each date.
    maturities : array-like
        Maturities for expected curve changes.
    **kwargs
        Additional keyword arguments forwarded to the single-date curve-view
        routine.

    Returns
    -------
    pandas.DataFrame
        MultiIndex table indexed by ``date`` and ``model`` with expected changes
        by maturity and average covariance diagonal.

    Notes
    -----
    Rows are produced for every date/model combination. Errors from the underlying
    view routine are not caught here.
    """

    rows = []
    for date in dates:
        for model_name in model_names:
            expected, cov = model_curve_view(model_name, date, maturities, **kwargs)
            row = {"date": date, "model": model_name}
            for maturity, value in zip(maturities, expected, strict=False):
                row[f"expected {maturity:g}y"] = value
            row["average variance"] = float(np.nanmean(np.diag(cov)))
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "model"]).sort_index()


__all__ = [
    "acf1",
    "ar1_expected_score",
    "cir_expected_average",
    "cir_yield_loading",
    "estimate_g2_style",
    "estimate_hw1f",
    "estimate_mean_reversion",
    "estimate_pca_curve",
    "fit_cir",
    "fit_cir_fast",
    "fit_vasicek_ar",
    "fit_vasicek_kalman",
    "g2_loadings",
    "hessian_condition",
    "hw1f_loading",
    "model_curve_view",
    "orient_pca_loadings",
    "rolling_model_views",
    "rolling_pca_diagnostics",
    "simulate_cir_paths",
    "simulate_g2_curves",
    "simulate_hw1f_curves",
    "simulate_vasicek_paths",
    "vasicek_ab",
    "vasicek_expected_average",
    "vasicek_kalman",
    "vasicek_loadings",
    "vasicek_yield_loading",
]
