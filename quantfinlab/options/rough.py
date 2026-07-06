from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy import optimize

from quantfinlab._optional import prefer_auto_engine
from quantfinlab.calibration.fft_cos import calibration_weights, cos_group_arrays
from quantfinlab.options import bsm
from quantfinlab.options.fourier import cos_prices
from quantfinlab.options.iv import implied_vol
from quantfinlab.options.surface import surface_iv

try:  # optional fast path; all routines keep a NumPy fallback.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def _numeric(frame: pd.DataFrame, names, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def _params_array(params) -> np.ndarray:
    if isinstance(params, pd.Series):
        p = params.to_dict()
    elif isinstance(params, dict):
        p = params
    else:
        arr = np.asarray(params, dtype=float).reshape(-1)
        if arr.size >= 6:
            return arr[:6]
        if arr.size == 5:
            return np.r_[0.15, arr]
        return np.r_[0.15, np.resize(arr, 5)]
    h = float(p.get("h", p.get("H", p.get("alpha", 0.15))))
    v0 = float(p.get("v0", p.get("p0", 0.04)))
    kappa = float(p.get("kappa", p.get("p1", 2.0)))
    theta = float(p.get("theta", p.get("p2", 0.04)))
    sigma_v = float(p.get("sigma_v", p.get("xi", p.get("nu", p.get("p3", 0.60)))))
    rho = float(p.get("rho", p.get("p4", -0.50)))
    return np.asarray([h, v0, kappa, theta, sigma_v, rho], dtype=float)


def atm_skew_term_structure(
    quotes: pd.DataFrame,
    *,
    fit: dict | None = None,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    tau_values=None,
    dk: float = 0.01,
) -> pd.DataFrame:
    """Estimate the ATM implied-volatility skew term structure.

    Skew can be computed from a fitted surface by symmetric finite differences around
    ATM, or directly from quote slices by a local polynomial fit around zero
    log-moneyness.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Surface-ready quote table.
    fit : dict, optional
        Fitted volatility surface used to evaluate skew smoothly.
    k_col : str, default='k'
        Log-moneyness column.
    tau_col : str, default='tau'
        Maturity column in years.
    iv_col : str, default='iv_mid'
        Implied-volatility column.
    tau_values : array-like, optional
        Maturities at which to estimate ATM skew. If omitted, unique maturities from
        the quote table are used.
    dk : float, default=0.01
        Symmetric log-moneyness step used for finite-difference skew.

    Returns
    -------
    pandas.DataFrame
        Table with maturity, maturity in days, ATM IV, ATM skew, absolute skew, and
        supporting observation count.
    """

    q = quotes.copy()
    if tau_values is None:
        tau_values = np.sort(pd.to_numeric(q[tau_col], errors="coerce").dropna().unique())
    rows = []
    for tau in np.asarray(tau_values, dtype=float):
        if not np.isfinite(tau) or tau <= 0:
            continue
        if fit is not None:
            left = float(surface_iv(fit, np.asarray([-float(dk)]), np.asarray([tau]))[0])
            right = float(surface_iv(fit, np.asarray([float(dk)]), np.asarray([tau]))[0])
            atm = float(surface_iv(fit, np.asarray([0.0]), np.asarray([tau]))[0])
            skew = (right - left) / (2.0 * float(dk))
            n = int(len(q[np.isclose(pd.to_numeric(q[tau_col], errors="coerce"), tau, rtol=0.0, atol=max(2.0 / 365.25, 0.025 * tau))]))
        else:
            g = q[np.isclose(pd.to_numeric(q[tau_col], errors="coerce"), tau, rtol=0.0, atol=max(2.0 / 365.25, 0.025 * tau))].copy()
            g = g.dropna(subset=[k_col, iv_col])
            if len(g) < 5:
                continue
            x = pd.to_numeric(g[k_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(g[iv_col], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= max(0.08, 4.0 * float(dk)))
            if keep.sum() < 4:
                keep = np.isfinite(x) & np.isfinite(y)
            beta = np.polyfit(x[keep], y[keep], deg=min(2, keep.sum() - 1))
            deriv = np.polyder(beta)
            skew = float(np.polyval(deriv, 0.0))
            atm = float(np.polyval(beta, 0.0))
            n = int(keep.sum())
        rows.append({"tau": float(tau), "tau_days": float(tau * 365.25), "atm_iv": atm, "atm_skew": skew, "abs_atm_skew": abs(skew), "n": n})
    return pd.DataFrame(rows)


def skew_power_law(psi: pd.DataFrame, *, tau_col: str = "tau", skew_col: str = "atm_skew") -> pd.DataFrame:
    """Fit a power law to the absolute ATM skew term structure.

    The fitted relationship is ``|skew(tau)| = c * tau**(-alpha)`` and the roughness
    proxy is reported as ``h = 0.5 - alpha``.

    Parameters
    ----------
    psi : pandas.DataFrame
        ATM skew term-structure table.
    tau_col : str, default='tau'
        Maturity column in years.
    skew_col : str, default='atm_skew'
        ATM skew column.

    Returns
    -------
    pandas.DataFrame
        One-row table with ``c``, ``alpha``, ``h``, ``r2``, and number of fitted
        observations. Returns NaNs when too few valid points are available.
    """

    q = psi.copy()
    y = pd.to_numeric(q[skew_col], errors="coerce").abs()
    x = pd.to_numeric(q[tau_col], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 3:
        return pd.DataFrame([{"c": np.nan, "alpha": np.nan, "h": np.nan, "r2": np.nan, "n": int(mask.sum())}])
    lx = np.log(x[mask].to_numpy(dtype=float))
    ly = np.log(y[mask].to_numpy(dtype=float))
    design = np.column_stack([np.ones(len(lx)), lx])
    beta, *_ = np.linalg.lstsq(design, ly, rcond=None)
    fitted = design @ beta
    ss_res = float(np.sum((ly - fitted) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    alpha = -float(beta[1])
    return pd.DataFrame([{"c": float(np.exp(beta[0])), "alpha": alpha, "h": 0.5 - alpha, "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan, "n": int(mask.sum())}])


def forward_variance_curve(
    quotes: pd.DataFrame,
    *,
    fit: dict | None = None,
    tau_values=None,
    k: float = 0.0,
    iv_col: str = "iv_mid",
    tau_col: str = "tau",
    use_pchip: bool = True,
    floor: float = 1e-6,
) -> pd.DataFrame:
    """Estimate a forward variance curve from ATM total variance.

    The curve is computed as the derivative of total variance,
    ``xi_0(tau) = d/dtau [tau * sigma_atm(tau)^2]``. PCHIP interpolation is used by
    default to reduce negative forward-variance artifacts from uneven expiry spacing.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table containing maturity and implied-volatility information.
    fit : dict, optional
        Fitted volatility surface used to evaluate ATM IV.
    tau_values : array-like, optional
        Maturities at which to estimate the curve.
    k : float, default=0.0
        Log-moneyness used for ATM volatility evaluation.
    iv_col : str, default='iv_mid'
        Implied-volatility column for quote-based estimation.
    tau_col : str, default='tau'
        Maturity column in years.
    use_pchip : bool, default=True
        If True, use PCHIP derivative of total variance when enough points exist.
    floor : float, default=1e-6
        Positive lower bound for forward variance.

    Returns
    -------
    pandas.DataFrame
        Table with maturity, total variance, ATM variance, ATM IV, and forward
        variance.
    """

    q = quotes.copy()
    if tau_values is None:
        tau_values = np.sort(pd.to_numeric(q[tau_col], errors="coerce").dropna().unique())
    rows = []
    for tau in np.asarray(tau_values, dtype=float):
        if not np.isfinite(tau) or tau <= 0:
            continue
        if fit is not None:
            sigma = float(surface_iv(fit, np.asarray([float(k)]), np.asarray([tau]))[0])
        else:
            g = q[np.isclose(pd.to_numeric(q[tau_col], errors="coerce"), tau, rtol=0.0, atol=max(2.0 / 365.25, 0.025 * tau))].copy()
            if "k" in g.columns:
                g["dist"] = pd.to_numeric(g["k"], errors="coerce").abs()
                sigma = float(g.sort_values("dist")[iv_col].iloc[0]) if len(g) else np.nan
            else:
                sigma = float(pd.to_numeric(g[iv_col], errors="coerce").median()) if len(g) else np.nan
        rows.append(
            {
                "tau": float(tau),
                "total_variance": float(sigma * sigma * tau) if np.isfinite(sigma) else np.nan,
                "atm_variance": float(sigma * sigma) if np.isfinite(sigma) else np.nan,
                "atm_iv": sigma,
            }
        )
    out = pd.DataFrame(rows).dropna().sort_values("tau").reset_index(drop=True)
    if out.empty:
        return out
    tau_arr = out["tau"].to_numpy(float)
    tv_arr = out["total_variance"].to_numpy(float)
    if use_pchip and len(out) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator
            pchip = PchipInterpolator(tau_arr, tv_arr, extrapolate=True)
            xi_values = np.maximum(pchip.derivative()(tau_arr), float(floor))
            out["variance"] = xi_values
            return out
        except Exception:
            pass
    # Fallback: simple finite differences with positivity floor
    forward = []
    last_tau = 0.0
    last_total = 0.0
    for tau_i, total_i in zip(tau_arr, tv_arr, strict=False):
        forward.append(max((total_i - last_total) / max(tau_i - last_tau, 1e-8), float(floor)))
        last_tau = tau_i
        last_total = total_i
    out["variance"] = np.asarray(forward, dtype=float)
    return out


def _xi_arrays(xi) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(xi, pd.DataFrame):
        frame = xi.copy()
        if "tau" in frame.columns:
            tau = pd.to_numeric(frame["tau"], errors="coerce").to_numpy(dtype=float)
        else:
            tau = np.arange(1, len(frame) + 1, dtype=float) / 365.25
        if "variance" in frame.columns:
            var = pd.to_numeric(frame["variance"], errors="coerce").to_numpy(dtype=float)
        elif "atm_variance" in frame.columns:
            var = pd.to_numeric(frame["atm_variance"], errors="coerce").to_numpy(dtype=float)
        else:
            var = pd.to_numeric(frame.select_dtypes(include=[np.number]).iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    elif isinstance(xi, pd.Series):
        var = pd.to_numeric(xi, errors="coerce").to_numpy(dtype=float)
        tau = np.arange(1, len(var) + 1, dtype=float) / 365.25
    else:
        tau = np.asarray([1.0], dtype=float)
        var = np.asarray([float(xi)], dtype=float)
    mask = np.isfinite(tau) & np.isfinite(var) & (tau > 0.0) & (var > 0.0)
    if not np.any(mask):
        return np.asarray([1.0], dtype=float), np.asarray([0.04], dtype=float)
    order = np.argsort(tau[mask])
    return tau[mask][order].astype(float), np.clip(var[mask][order].astype(float), 1e-8, 25.0)


def _rbergomi_xi_interp_py(t: float, xi_tau: np.ndarray, xi_var: np.ndarray) -> float:
    if t <= xi_tau[0]:
        return float(xi_var[0])
    for i in range(1, len(xi_tau)):
        if t <= xi_tau[i]:
            w = (t - xi_tau[i - 1]) / max(xi_tau[i] - xi_tau[i - 1], 1e-12)
            return float(xi_var[i - 1] * (1.0 - w) + xi_var[i] * w)
    return float(xi_var[-1])


def _rbergomi_paths_python(spot, xi_tau, xi_var, h, nu, rho, rate, dividend_yield, tau, z1, z2):
    n, m = z1.shape
    dt = float(tau) / float(m)
    out_s = np.empty((n, m + 1), dtype=np.float64)
    out_v = np.empty((n, m + 1), dtype=np.float64)
    alpha = float(h) + 0.5
    scale = np.sqrt(max(2.0 * float(h), 1e-12)) * (dt**float(h)) / max(alpha, 1e-12)
    grid = np.arange(0, m + 1, dtype=np.float64)
    kernel = grid[1:] ** alpha - grid[:-1] ** alpha
    leverage_scale = np.sqrt(max(1.0 - float(rho) ** 2, 1e-12))
    for i in range(n):
        out_s[i, 0] = float(spot)
        out_v[i, 0] = _rbergomi_xi_interp_py(0.0, xi_tau, xi_var)
        for j in range(m):
            rough = 0.0
            for lag in range(j + 1):
                rough += kernel[lag] * z1[i, j - lag]
            rough *= scale
            tt = (j + 1.0) * dt
            xi_t = _rbergomi_xi_interp_py(tt, xi_tau, xi_var)
            v = xi_t * np.exp(float(nu) * rough - 0.5 * float(nu) ** 2 * max(tt, 1e-12) ** (2.0 * float(h)))
            if v < 1e-10:
                v = 1e-10
            if v > 25.0:
                v = 25.0
            out_v[i, j + 1] = v
            dw = float(rho) * z1[i, j] + leverage_scale * z2[i, j]
            out_s[i, j + 1] = out_s[i, j] * np.exp((float(rate) - float(dividend_yield) - 0.5 * v) * dt + np.sqrt(v * dt) * dw)
    return out_s, out_v


if njit is not None:
    _rbergomi_xi_interp_py = njit(_rbergomi_xi_interp_py)
    _rbergomi_paths_numba = njit(_rbergomi_paths_python)
else:  # pragma: no cover
    _rbergomi_paths_numba = None


def simulate_rbergomi(
    *,
    spot: float,
    xi: pd.DataFrame | pd.Series | float,
    h: float,
    nu: float,
    rho: float,
    tau: float,
    paths: int = 10000,
    steps: int = 120,
    seed: int = 7,
    engine: str = "auto",
    rate: float = 0.0,
    dividend_yield: float = 0.0,
    antithetic: bool = True,
    z1: np.ndarray | None = None,
    z2: np.ndarray | None = None,
) -> dict:
    """Simulate spot and variance paths under an rBergomi-style rough-volatility model.

    The function supports internally generated antithetic shocks or externally
    provided common-random-number arrays. Supplying ``z1`` and ``z2`` is useful for
    calibration because all candidate parameters are evaluated on the same noise.

    Parameters
    ----------
    spot : float
        Initial spot price.
    xi : pandas.DataFrame, pandas.Series, or float
        Initial forward variance curve or constant variance level.
    h : float
        Hurst exponent. Values below 0.5 correspond to rough volatility.
    nu : float
        Vol-of-vol parameter.
    rho : float
        Correlation between price and volatility shocks.
    tau : float
        Simulation horizon in years.
    paths : int, default=10000
        Number of Monte Carlo paths when shocks are generated internally.
    steps : int, default=120
        Number of time steps.
    seed : int, default=7
        Random seed for internally generated shocks.
    engine : {'auto', 'numpy', 'numba'}, default='auto'
        Simulation backend.
    rate : float, default=0.0
        Continuously compounded risk-free rate.
    dividend_yield : float, default=0.0
        Continuously compounded dividend yield.
    antithetic : bool, default=True
        Whether to use antithetic shocks when shocks are generated internally.
    z1, z2 : numpy.ndarray, optional
        Pre-generated common-random-number arrays with shape ``(paths, steps)``.

    Returns
    -------
    dict
        Dictionary containing simulated ``spot`` paths, ``variance`` paths, time grid,
        parameter metadata, and the forward-variance curve used for simulation.
    """

    xi_tau, xi_var = _xi_arrays(xi)
    if z1 is not None and z2 is not None:
        z1_use = np.asarray(z1, dtype=float)
        z2_use = np.asarray(z2, dtype=float)
        m = z1_use.shape[1]
    else:
        rng = np.random.default_rng(int(seed))
        m = int(steps)
        n = max(2, int(paths))
        half = n // 2 if bool(antithetic) else n
        z1_use = rng.standard_normal((half, m))
        z2_use = rng.standard_normal((half, m))
        if bool(antithetic):
            z1_use = np.vstack([z1_use, -z1_use])
            z2_use = np.vstack([z2_use, -z2_use])
    engine_key = str(engine).lower()
    if engine_key == "auto":
        engine_key = prefer_auto_engine(allow_cpp=False)
    if engine_key == "numba" and _rbergomi_paths_numba is not None:
        spot_paths, var_paths = _rbergomi_paths_numba(float(spot), xi_tau, xi_var, float(h), float(nu), float(rho), float(rate), float(dividend_yield), float(tau), z1_use, z2_use)
    else:
        spot_paths, var_paths = _rbergomi_paths_python(float(spot), xi_tau, xi_var, float(h), float(nu), float(rho), float(rate), float(dividend_yield), float(tau), z1_use, z2_use)
    return {
        "spot": spot_paths,
        "variance": var_paths,
        "time": np.linspace(0.0, float(tau), m + 1),
        "params": {"h": float(h), "nu": float(nu), "rho": float(rho), "xi0": float(xi_var[0]), "rate": float(rate), "dividend_yield": float(dividend_yield)},
        "xi": pd.DataFrame({"tau": xi_tau, "variance": xi_var}),
    }


def _mc_price_from_terminal(s_terminal, strikes, option_type="call", discount=1.0):
    s = np.asarray(s_terminal, dtype=float)
    k = np.asarray(strikes, dtype=float)
    flag = str(option_type).lower().startswith("c")
    payoff = np.maximum(s[:, None] - k[None, :], 0.0) if flag else np.maximum(k[None, :] - s[:, None], 0.0)
    return float(discount) * np.nanmean(payoff, axis=0)


def _mc_price_and_se_from_terminal(s_terminal, strikes, option_type="call", discount=1.0):
    s = np.asarray(s_terminal, dtype=float)
    k = np.asarray(strikes, dtype=float)
    flag = str(option_type).lower().startswith("c")
    payoff = np.maximum(s[:, None] - k[None, :], 0.0) if flag else np.maximum(k[None, :] - s[:, None], 0.0)
    disc = float(discount)
    return disc * np.nanmean(payoff, axis=0), disc * np.nanstd(payoff, axis=0, ddof=1) / np.sqrt(max(payoff.shape[0], 1))


def rbergomi_smile(
    quotes: pd.DataFrame,
    *,
    xi,
    params: dict | pd.Series,
    maturity_days=(7, 14, 30, 60, 90),
    paths: int = 20000,
    steps: int = 120,
    seed: int = 7,
    engine: str = "auto",
) -> pd.DataFrame:
    """Generate model-implied volatility smiles from rBergomi Monte Carlo simulations.

    For each requested maturity, the function simulates terminal spot values,
    computes call prices across a log-moneyness grid, and inverts the prices to
    Black-Scholes implied volatilities.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table used to infer spot, rate, and dividend/carry inputs.
    xi : pandas.DataFrame, pandas.Series, or float
        Initial forward variance curve.
    params : dict or pandas.Series
        rBergomi parameters containing roughness, vol-of-vol, and correlation values.
    maturity_days : iterable, default=(7, 14, 30, 60, 90)
        Maturities at which to simulate smiles.
    paths : int, default=20000
        Monte Carlo paths.
    steps : int, default=120
        Simulation time steps.
    seed : int, default=7
        Base random seed.
    engine : {'auto', 'numpy', 'numba'}, default='auto'
        Simulation backend.

    Returns
    -------
    pandas.DataFrame
        Smile table with maturity, log-moneyness, strike, price, price standard error,
        implied volatility, and model label.
    """

    q = quotes.copy()
    spot = float(pd.to_numeric(q["spot"], errors="coerce").median())
    rate = float(_numeric(q, ("rate",), 0.0).median())
    div = float(_numeric(q, ("implied_dividend_yield", "dividend_yield"), 0.0).median())
    rows = []
    p = _params_array(params)
    for d in maturity_days:
        tau = float(d) / 365.25
        sim = simulate_rbergomi(spot=spot, xi=xi, h=p[0], nu=p[4], rho=p[5], tau=tau, paths=paths, steps=steps, seed=int(seed) + int(d), engine=engine, rate=rate, dividend_yield=div, antithetic=True)
        k_values = np.linspace(-0.35, 0.25, 31)
        strikes = spot * np.exp(k_values)
        discount = np.exp(-rate * tau)
        price, price_se = _mc_price_and_se_from_terminal(sim["spot"][:, -1], strikes, "call", discount)
        forward = spot * np.exp((rate - div) * tau)
        iv = implied_vol("call", price, forward, strikes, tau, discount, engine="auto")
        for kk, strike, px, se, sigma in zip(k_values, strikes, price, price_se, iv, strict=False):
            rows.append({"tau": tau, "tau_days": float(d), "k": float(kk), "strike": float(strike), "price": float(px), "price_se": float(se), "iv": float(sigma), "model": "rbergomi"})
    return pd.DataFrame(rows)


def _generate_crn(paths: int, steps: int, seed: int, use_sobol: bool = True):
    """Generate common random numbers via Sobol QMC (antithetic pairs)."""
    half = max(1, paths // 2)
    if use_sobol:
        try:
            from scipy.special import ndtri
            from scipy.stats.qmc import Sobol
            n_sobol = int(2 ** int(np.ceil(np.log2(max(half, 2)))))
            sampler = Sobol(d=int(steps) * 2, scramble=True, seed=int(seed))
            q_samples = sampler.random(n_sobol)
            q_samples = np.clip(q_samples, 1e-9, 1.0 - 1e-9)
            z_all = ndtri(q_samples)
            z1_base = z_all[:half, :int(steps)].astype(float)
            z2_base = z_all[:half, int(steps):].astype(float)
            return np.vstack([z1_base, -z1_base]), np.vstack([z2_base, -z2_base])
        except Exception:
            pass
    rng = np.random.default_rng(int(seed))
    z1_base = rng.standard_normal((half, int(steps)))
    z2_base = rng.standard_normal((half, int(steps)))
    return np.vstack([z1_base, -z1_base]), np.vstack([z2_base, -z2_base])


def rbergomi_calibration(
    quotes: pd.DataFrame,
    *,
    xi,
    h_start: float = 0.12,
    nu_start: float = 2.0,
    rho_start: float = -0.70,
    paths: int = 15000,
    steps: int = 120,
    restarts: int = 4,
    seed: int = 7,
    engine: str = "auto",
    lambda_skew: float = 0.0,
    market_skew: pd.DataFrame | None = None,
    use_sobol: bool = True,
    vega_floor: float = 0.0,
) -> dict:
    """Calibrate rBergomi rough-volatility parameters to implied-volatility observations.

    The calibration focuses on short-to-medium maturities and a central moneyness
    range where rough-volatility skew is most informative. It uses common random
    numbers, a staged candidate search over Hurst exponent, vol-of-vol, and
    correlation, optional skew penalties, and a final full-path re-evaluation of the
    best candidates.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Surface-ready quote table with ``iv_mid`` and option inputs.
    xi : pandas.DataFrame, pandas.Series, or float
        Initial forward variance curve.
    h_start : float, default=0.12
        Starting Hurst exponent.
    nu_start : float, default=2.0
        Starting vol-of-vol.
    rho_start : float, default=-0.70
        Starting spot/volatility correlation.
    paths : int, default=15000
        Monte Carlo paths for calibration.
    steps : int, default=120
        Simulation time steps.
    restarts : int, default=4
        Number of additional random candidate starts.
    seed : int, default=7
        Random seed.
    engine : {'auto', 'numpy', 'numba'}, default='auto'
        Simulation backend.
    lambda_skew : float, default=0.0
        Weight of optional ATM-skew penalty.
    market_skew : pandas.DataFrame, optional
        Market ATM skew term-structure table for the skew penalty.
    use_sobol : bool, default=True
        Whether to use Sobol-style common random numbers when available.
    vega_floor : float, default=0.0
        Optional minimum vega filter.

    Returns
    -------
    dict
        Calibration result with best parameters, candidate fit table, and scalar loss.
    """

    q = quotes.copy()
    target = q.dropna(subset=["iv_mid"]).copy()
    if target.empty:
        return {"params": {"h": h_start, "nu": nu_start, "rho": rho_start}, "fit": pd.DataFrame(), "loss": np.nan}
    if "k" not in target.columns:
        forward = pd.to_numeric(target.get("forward", target["spot"]), errors="coerce")
        target["k"] = np.log(pd.to_numeric(target["strike"], errors="coerce") / forward)
    # Focus on short-to-medium maturities where rBergomi's roughness signature is clearest
    core = target[
        pd.to_numeric(target["tau"], errors="coerce").mul(365.25).between(7, 90)
        & pd.to_numeric(target["k"], errors="coerce").between(-0.25, 0.15)
    ].copy()
    if len(core) >= 24:
        target = core
    # Apply optional vega floor to filter low-information points
    if float(vega_floor) > 0.0 and "vega" in target.columns:
        target = target[pd.to_numeric(target["vega"], errors="coerce").ge(float(vega_floor))].copy()
    if target.empty:
        return {"params": {"h": h_start, "nu": nu_start, "rho": rho_start}, "fit": pd.DataFrame(), "loss": np.nan}

    spot = float(pd.to_numeric(target["spot"], errors="coerce").median())
    rate = float(_numeric(target, ("rate",), 0.0).median())
    div = float(_numeric(target, ("implied_dividend_yield", "dividend_yield"), 0.0).median())
    xi_tau, xi_var = _xi_arrays(xi)

    h_start = float(np.clip(h_start, 0.03, 0.49))
    nu_start = float(np.clip(nu_start, 0.2, 6.0))
    rho_start = float(np.clip(rho_start, -0.98, 0.40))

    unique = np.sort(np.unique(np.clip(
        np.round(pd.to_numeric(target["tau"], errors="coerce") * 365.25), 7, 180
    ).astype(int)))
    sample_days = unique[np.linspace(0, len(unique) - 1, min(5, len(unique))).round().astype(int)] if len(unique) else np.asarray([14, 30, 60, 90])

    # Pre-generate common random numbers once (shared across all candidate evaluations)
    z1_crn, z2_crn = _generate_crn(int(paths), int(steps), int(seed), use_sobol=bool(use_sobol))

    # Helper: evaluate IV loss for (h, nu, rho) with CRN
    def _iv_loss(h, nu, rho, z1_use, z2_use, eval_days):
        all_errors, all_weights = [], []
        for d in eval_days:
            tau_d = float(d) / 365.25
            discount = float(np.exp(-rate * tau_d))
            forward = float(spot * np.exp((rate - div) * tau_d))
            sim = simulate_rbergomi(
                spot=spot, xi=xi, h=h, nu=nu, rho=rho, tau=tau_d,
                rate=rate, dividend_yield=div,
                engine=engine, z1=z1_use, z2=z2_use,
            )
            terminal = sim["spot"][:, -1]
            tgt_d = target[np.isclose(
                pd.to_numeric(target["tau"], errors="coerce") * 365.25, float(d), atol=5.0
            )].copy()
            if tgt_d.empty:
                continue
            k_vals = pd.to_numeric(tgt_d["k"], errors="coerce").to_numpy(float)
            strikes = np.exp(k_vals) * forward
            payoff = np.maximum(terminal[:, None] - strikes[None, :], 0.0)
            price_mc = discount * np.nanmean(payoff, axis=0)
            sigma_mc = implied_vol("call", price_mc, forward, strikes, tau_d, discount, engine="auto")
            for i, (iv_market, w_raw) in enumerate(zip(
                pd.to_numeric(tgt_d["iv_mid"], errors="coerce").to_numpy(float),
                tgt_d.get("surface_weight", pd.Series(1.0, index=tgt_d.index)).to_numpy(float), strict=False,
            )):
                if not np.isfinite(sigma_mc[i]) or not np.isfinite(iv_market):
                    continue
                all_errors.append(sigma_mc[i] - iv_market)
                all_weights.append(max(float(w_raw), 0.05))
        if not all_errors:
            return np.inf, pd.DataFrame()
        errs = np.asarray(all_errors, float)
        ws = np.clip(np.asarray(all_weights, float), 0.05, 20.0)
        iv_loss = float(np.average(errs * errs, weights=ws))
        # Optional skew penalty
        if float(lambda_skew) > 0.0 and market_skew is not None and not market_skew.empty:
            sk_col = "atm_skew" if "atm_skew" in market_skew.columns else market_skew.columns[-1]
            skew_scale = max(float(market_skew[sk_col].abs().median()), 0.05)
            for d in eval_days:
                ms_row = market_skew[np.isclose(market_skew.get("tau_days", market_skew.get("tau", pd.Series())), float(d), atol=4.0)]
                if ms_row.empty:
                    continue
                psi_market = float(ms_row[sk_col].iloc[0])
                # model skew from MC: evaluate near-ATM
                tau_d = float(d) / 365.25
                forward_d = float(spot * np.exp((rate - div) * tau_d))
                dk_val = 0.015
                k_near = np.array([-dk_val, dk_val])
                st_near = forward_d * np.exp(k_near)
                payoff_near = np.maximum(terminal[:, None] - st_near[None, :], 0.0)
                price_near = float(np.exp(-rate * tau_d)) * np.nanmean(payoff_near, axis=0)
                iv_near = implied_vol("call", price_near, forward_d, st_near, tau_d, float(np.exp(-rate * tau_d)), engine="auto")
                if np.all(np.isfinite(iv_near)):
                    psi_model = float((iv_near[1] - iv_near[0]) / (2.0 * dk_val))
                    iv_loss += float(lambda_skew) * ((psi_model - psi_market) / skew_scale) ** 2
        return iv_loss, pd.DataFrame()

    # --- Stage 1: H fixed at h_start, grid search over nu x rho ---
    nu_grid = [nu_start * 0.55, nu_start * 0.80, nu_start, nu_start * 1.35, nu_start * 1.80, 0.8, 1.4, 2.2, 3.5]
    rho_grid = [rho_start - 0.20, rho_start, rho_start + 0.15, -0.90, -0.70, -0.50, -0.30]
    stage1_candidates = [
        (h_start, float(np.clip(nu, 0.2, 6.0)), float(np.clip(rho, -0.98, 0.40)))
        for nu in nu_grid for rho in rho_grid
    ]
    # Also vary H
    h_candidates_full = [h_start, max(0.03, h_start - 0.08), min(0.49, h_start + 0.08), 0.06, 0.12, 0.20, 0.30]
    for hh in h_candidates_full:
        stage1_candidates.extend([(float(np.clip(hh, 0.03, 0.49)), nu_start, rho_start)])
    rng = np.random.default_rng(int(seed))
    for _ in range(max(0, int(restarts) - 1)):
        stage1_candidates.append((
            float(np.clip(h_start + rng.normal(0, 0.05), 0.03, 0.49)),
            float(np.clip(nu_start * np.exp(rng.normal(0, 0.40)), 0.2, 6.0)),
            float(np.clip(rho_start + rng.normal(0, 0.15), -0.98, 0.40)),
        ))
    stage1_candidates = list(dict.fromkeys(
        (round(h, 5), round(nu, 5), round(rho, 5)) for h, nu, rho in stage1_candidates
    ))

    best = {"params": {"h": h_start, "nu": nu_start, "rho": rho_start}, "loss": np.inf}
    loss_rows = []
    # Use a smaller sub-grid of z for fast screening (half paths)
    half = z1_crn.shape[0] // 2
    z1_screen = z1_crn[:half]
    z2_screen = z2_crn[:half]
    for restart, (h, nu, rho) in enumerate(stage1_candidates):
        loss_val, _ = _iv_loss(h, nu, rho, z1_screen, z2_screen, sample_days)
        loss_rows.append({"restart": int(restart), "h": float(h), "nu": float(nu), "rho": float(rho), "iv_mse": loss_val, "iv_rmse": float(np.sqrt(loss_val)) if np.isfinite(loss_val) else np.inf})
        if loss_val < best["loss"]:
            best = {"params": {"h": float(h), "nu": float(nu), "rho": float(rho)}, "loss": loss_val}

    # --- Stage 2: Re-evaluate top-3 candidates with full CRN paths ---
    loss_df_s1 = pd.DataFrame(loss_rows).sort_values("iv_mse").head(3)
    for _, row_s1 in loss_df_s1.iterrows():
        h, nu, rho = float(row_s1["h"]), float(row_s1["nu"]), float(row_s1["rho"])
        loss_val, _ = _iv_loss(h, nu, rho, z1_crn, z2_crn, sample_days)
        loss_rows.append({"restart": -1, "h": h, "nu": nu, "rho": rho, "iv_mse": loss_val, "iv_rmse": float(np.sqrt(loss_val)) if np.isfinite(loss_val) else np.inf})
        if loss_val < best["loss"]:
            best = {"params": {"h": h, "nu": nu, "rho": rho}, "loss": loss_val}

    loss_table = pd.DataFrame(loss_rows).sort_values("iv_mse").reset_index(drop=True)
    params = dict(best["params"])
    params["boundary_hit"] = bool(
        np.isclose(params["h"], 0.03, atol=1e-4)
        or np.isclose(params["h"], 0.49, atol=1e-4)
        or np.isclose(params["rho"], -0.98, atol=1e-4)
        or np.isclose(params["rho"], 0.40, atol=1e-4)
    )
    params["target_rows"] = int(len(target))
    params["iv_rmse"] = float(np.sqrt(best["loss"])) if np.isfinite(best["loss"]) else np.nan
    return {"params": params, "fit": loss_table, "loss": best["loss"]}


def _rough_riccati_rhs(u: np.ndarray, psi: np.ndarray, kappa: float, sigma_v: float, rho: float) -> np.ndarray:
    return -0.5 * (u * u + 1j * u) + (1j * u * rho * sigma_v - kappa) * psi + 0.5 * sigma_v * sigma_v * psi * psi


def _fractional_riccati_terms(u, params, tau: float, *, n_steps: int = 512, scheme: str = "adams") -> dict:
    p = _params_array(params)
    h, _, kappa, _, sigma_v, rho = p
    alpha = float(np.clip(h, 1e-6, 0.5)) + 0.5
    m = max(1, int(n_steps))
    tau = max(float(tau), 1e-12)
    t = np.linspace(0.0, tau, m + 1)
    dt = tau / float(m)
    u_arr = np.asarray(u, dtype=complex).reshape(-1)
    psi = np.zeros((m + 1, u_arr.size), dtype=complex)
    f = np.zeros_like(psi)
    f[0] = _rough_riccati_rhs(u_arr, psi[0], float(kappa), float(sigma_v), float(rho))
    j = np.arange(1, m + 1, dtype=float)
    predictor_weights = (j**alpha - (j - 1.0) ** alpha) * (dt**alpha) / math.gamma(alpha + 1.0)
    key = str(scheme).lower()
    for i in range(1, m + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            if key in {"adams", "predictor_corrector", "pece"}:
                pred = predictor_weights[:i][::-1] @ f[:i]
                f_pred = _rough_riccati_rhs(u_arr, pred, float(kappa), float(sigma_v), float(rho))
                n = i - 1
                a = np.empty(i, dtype=float)
                a[0] = n ** (alpha + 1.0) - (n - alpha) * (n + 1.0) ** alpha
                if i > 1:
                    jj = np.arange(1, i, dtype=float)
                    a[1:] = (n - jj + 2.0) ** (alpha + 1.0) + (n - jj) ** (alpha + 1.0) - 2.0 * (n - jj + 1.0) ** (alpha + 1.0)
                psi[i] = (dt**alpha) * (a @ f[:i] + f_pred) / math.gamma(alpha + 2.0)
            else:
                psi[i] = predictor_weights[:i][::-1] @ f[:i]
            f[i] = _rough_riccati_rhs(u_arr, psi[i], float(kappa), float(sigma_v), float(rho))
    integral = np.trapezoid(psi, t, axis=0)
    beta = max(1.0 - alpha, 0.0)
    if beta <= 1e-10:
        fractional_terminal = psi[-1]
    else:
        j = np.arange(1, m + 1, dtype=float)
        frac_weights = ((m - j + 1.0) ** beta - (m - j) ** beta) * (dt**beta) / math.gamma(beta + 1.0)
        fractional_terminal = frac_weights @ psi[1:]
    return {
        "t": t,
        "psi": psi,
        "integral": integral,
        "fractional_terminal": fractional_terminal,
        "alpha": alpha,
        "scheme": key,
    }


def fractional_riccati(u, params, tau: float, *, n_steps: int = 512, scheme: str = "adams") -> pd.DataFrame:
    """Solve and return the fractional Riccati path used by rough-Heston pricing.

    Parameters
    ----------
    u : complex or array-like
        Fourier argument.
    params : mapping or array-like
        Rough-Heston parameters.
    tau : float
        Time horizon in years.
    n_steps : int, default=512
        Number of time-discretization steps.
    scheme : str, default='adams'
        Numerical scheme label.

    Returns
    -------
    pandas.DataFrame
        Time-indexed table containing real, imaginary, absolute Riccati values,
        fractional exponent metadata, and scheme label.
    """

    solved = _fractional_riccati_terms(u, params, tau, n_steps=n_steps, scheme=scheme)
    psi = solved["psi"][:, 0]
    return pd.DataFrame(
        {
            "t": solved["t"],
            "real": np.real(psi),
            "imag": np.imag(psi),
            "abs": np.abs(psi),
            "alpha": float(solved["alpha"]),
            "scheme": str(solved["scheme"]),
        }
    )


def _rough_heston_cf_values(
    u,
    params,
    spot,
    rate,
    dividend_yield,
    tau,
    *,
    riccati_steps: int = 512,
    scheme: str = "adams",
    allow_clip: bool = False,
):
    p = _params_array(params)
    _, v0, kappa, theta, _, _ = p
    u_arr = np.asarray(u, dtype=complex)
    original_shape = u_arr.shape
    flat_u = u_arr.reshape(-1)
    tau = float(tau)
    if tau <= 0.0:
        out = np.exp(1j * u_arr * np.log(float(spot)))
        return out, {"cf_clip_count": 0, "cf_nonfinite_count": 0, "max_real_exponent": 0.0, "min_real_exponent": 0.0}
    solved = _fractional_riccati_terms(flat_u, p, tau, n_steps=int(riccati_steps), scheme=scheme)
    exponent = (
        1j * flat_u * (np.log(float(spot)) + (float(rate) - float(dividend_yield)) * tau)
        + float(kappa) * float(theta) * solved["integral"]
        + float(v0) * solved["fractional_terminal"]
    )
    real = np.real(exponent)
    imag = np.imag(exponent)
    finite = np.isfinite(real) & np.isfinite(imag)
    clip_mask = finite & ((real > 700.0) | (real < -700.0))
    diag = {
        "cf_clip_count": int(np.count_nonzero(clip_mask)),
        "cf_nonfinite_count": int(np.count_nonzero(~finite)),
        "max_real_exponent": float(np.nanmax(real)) if real.size else 0.0,
        "min_real_exponent": float(np.nanmin(real)) if real.size else 0.0,
    }
    if (diag["cf_clip_count"] > 0 or diag["cf_nonfinite_count"] > 0) and not bool(allow_clip):
        raise FloatingPointError(f"rough-Heston CF exponent unstable: {diag}")
    safe_real = np.clip(np.where(finite, real, -700.0), -700.0, 700.0)
    safe_imag = np.where(finite, imag, 0.0)
    out = np.exp(safe_real + 1j * safe_imag)
    return out.reshape(original_shape), diag


def rough_heston_cf(
    u,
    params,
    spot,
    rate,
    dividend_yield,
    tau,
    *,
    riccati_steps: int = 512,
    scheme: str = "adams",
    allow_clip: bool = False,
    return_diagnostics: bool = False,
):
    """Evaluate the rough-Heston characteristic function.

    Parameters
    ----------
    u : array-like
        Complex Fourier argument.
    params : mapping or array-like
        Rough-Heston parameters.
    spot : float
        Spot price.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuously compounded dividend yield.
    tau : float
        Time to expiry in years.
    riccati_steps : int, default=512
        Number of steps used in the fractional Riccati solver.
    scheme : str, default='adams'
        Riccati discretization scheme.
    allow_clip : bool, default=False
        If True, allow numerical clipping in the characteristic-function evaluation.
    return_diagnostics : bool, default=False
        If True, return both characteristic-function values and diagnostics.

    Returns
    -------
    numpy.ndarray or tuple[numpy.ndarray, dict]
        Characteristic-function values, optionally with diagnostic metadata.
    """

    out, diag = _rough_heston_cf_values(
        u,
        params,
        spot,
        rate,
        dividend_yield,
        tau,
        riccati_steps=riccati_steps,
        scheme=scheme,
        allow_clip=allow_clip,
    )
    if return_diagnostics:
        return out, diag
    return out


def rough_heston_cf_diagnostics(
    params,
    spot,
    rate,
    dividend_yield,
    tau,
    *,
    u_values=None,
    riccati_steps: int = 512,
    scheme: str = "adams",
) -> pd.DataFrame:
    """Run diagnostic checks on the rough-Heston characteristic function.

    The function evaluates characteristic-function normalization, martingale behavior,
    maximum magnitude, and Riccati metadata at selected Fourier arguments.

    Parameters
    ----------
    params : mapping or array-like
        Rough-Heston parameters.
    spot : float
        Spot price.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuously compounded dividend yield.
    tau : float
        Time to expiry in years.
    u_values : array-like, optional
        Fourier arguments used for diagnostics.
    riccati_steps : int, default=512
        Number of Riccati steps.
    scheme : str, default='adams'
        Riccati scheme.

    Returns
    -------
    pandas.DataFrame
        One-row diagnostic table including normalization and martingale errors.
    """

    if u_values is None:
        u_values = np.r_[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, -1j]
    u_values = np.asarray(u_values, dtype=complex)
    phi, diag = rough_heston_cf(
        u_values,
        params,
        spot,
        rate,
        dividend_yield,
        tau,
        riccati_steps=riccati_steps,
        scheme=scheme,
        allow_clip=True,
        return_diagnostics=True,
    )
    phi_zero = rough_heston_cf(
        np.asarray([0.0]),
        params,
        spot,
        rate,
        dividend_yield,
        tau,
        riccati_steps=riccati_steps,
        scheme=scheme,
        allow_clip=True,
    )[0]
    phi_mi = rough_heston_cf(
        np.asarray([-1j]),
        params,
        spot,
        rate,
        dividend_yield,
        tau,
        riccati_steps=riccati_steps,
        scheme=scheme,
        allow_clip=True,
    )[0]
    expected_forward_spot = float(spot) * np.exp((float(rate) - float(dividend_yield)) * float(tau))
    row = dict(diag)
    row.update(
        {
            "tau": float(tau),
            "riccati_steps": int(riccati_steps),
            "phi_zero_error": float(abs(phi_zero - 1.0)),
            "martingale_error": float(abs(phi_mi - expected_forward_spot) / max(expected_forward_spot, 1e-12)),
            "max_abs_phi": float(np.nanmax(np.abs(phi))) if phi.size else np.nan,
        }
    )
    return pd.DataFrame([row])


def rough_heston_prices(
    params,
    strikes,
    tau,
    spot,
    rate=0.0,
    dividend_yield=0.0,
    *,
    option_type="call",
    engine: str = "auto",
    n_terms: int = 160,
    truncation_width: float = 16.0,
    riccati_steps: int = 512,
    scheme: str = "adams",
    allow_cf_clip: bool = False,
) -> np.ndarray:
    """Price vanilla options under the rough-Heston model with a COS/Fourier method.

    Parameters
    ----------
    params : mapping or array-like
        Rough-Heston parameters.
    strikes : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    spot : float or array-like
        Spot prices.
    rate : float or array-like, default=0.0
        Continuously compounded risk-free rates.
    dividend_yield : float or array-like, default=0.0
        Continuously compounded dividend yields.
    option_type : array-like or scalar, default='call'
        Option type labels.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend.
    n_terms : int, default=160
        Number of COS expansion terms.
    truncation_width : float, default=16.0
        COS truncation width.
    riccati_steps : int, default=512
        Number of fractional Riccati steps.
    scheme : str, default='adams'
        Riccati scheme.
    allow_cf_clip : bool, default=False
        Whether to allow clipping inside the characteristic-function evaluator.

    Returns
    -------
    numpy.ndarray
        Rough-Heston option prices.
    """

    p = _params_array(params)
    variance_hint = max(float(p[1]), float(p[3]), 1e-6)

    def cf(u, tau_i, spot_i, rate_i, dividend_i):
        return rough_heston_cf(
            u,
            p,
            spot_i,
            rate_i,
            dividend_i,
            tau_i,
            riccati_steps=int(riccati_steps),
            scheme=scheme,
            allow_clip=allow_cf_clip,
        )

    return cos_prices(
        "custom",
        None,
        strikes,
        tau,
        spot,
        rate,
        dividend_yield,
        option_type=option_type,
        engine=engine,
        n_terms=n_terms,
        truncation_width=truncation_width,
        cf=cf,
        variance_hint=variance_hint,
    )


def rough_heston_iv(params, strikes, tau, spot, rate=0.0, dividend_yield=0.0, *, option_type="call", **kwargs):
    """Compute rough-Heston implied volatilities by pricing and Black-Scholes inversion.

    Parameters
    ----------
    params : mapping or array-like
        Rough-Heston parameters.
    strikes : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    spot : float or array-like
        Spot prices.
    rate : float or array-like, default=0.0
        Continuously compounded risk-free rates.
    dividend_yield : float or array-like, default=0.0
        Continuously compounded dividend yields.
    option_type : array-like or scalar, default='call'
        Option type labels.
    **kwargs
        Additional keyword arguments forwarded to the rough-Heston pricer.

    Returns
    -------
    numpy.ndarray
        Black-Scholes implied volatilities corresponding to rough-Heston prices.
    """

    price = rough_heston_prices(params, strikes, tau, spot, rate, dividend_yield, option_type=option_type, **kwargs)
    forward = np.asarray(spot, dtype=float) * np.exp((np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float)) * np.asarray(tau, dtype=float))
    df = np.exp(-np.asarray(rate, dtype=float) * np.asarray(tau, dtype=float))
    return implied_vol(option_type, price, forward, strikes, tau, df, engine="auto")


def fit_rough_heston_dates(
    quotes: pd.DataFrame,
    *,
    calibration_dates=None,
    h_start: float = 0.12,
    h_by_date=None,
    h_mode: str = "fixed",
    h_penalty: float = 0.0,
    min_quotes: int = 80,
    max_nfev: int = 55,
    engine: str = "auto",
    n_terms: int = 160,
    truncation_width: float = 16.0,
    riccati_steps: int = 512,
    scheme: str = "adams",
    lambda_skew: float = 0.0,
    skew_tau_days: tuple = (7, 14, 21, 30, 45, 60),
    surface_fit_map: dict | None = None,
    n_jobs: int = 1,
) -> dict:
    """Calibrate rough-Heston parameters across multiple quote dates.

    The function prepares weighted calibration quotes, optionally anchors or penalizes
    Hurst exponents by date, calibrates fixed-H and optionally free-H parameter sets,
    uses reduced Riccati resolution during inner optimization, reprices with full
    accuracy for reporting, and returns daily parameter and quote-level fit tables.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Surface-ready option quote table.
    calibration_dates : iterable, optional
        Dates to calibrate. If omitted, dates with enough quotes are selected.
    h_start : float, default=0.12
        Default Hurst exponent anchor.
    h_by_date : Series, DataFrame, dict, optional
        Date-specific Hurst anchors.
    h_mode : {'fixed', 'penalized', 'free'}, default='fixed'
        Hurst treatment during calibration.
    h_penalty : float, default=0.0
        Penalty weight for deviations from H anchor when ``h_mode='penalized'``.
    min_quotes : int, default=80
        Minimum quotes per calibration date.
    max_nfev : int, default=55
        Maximum optimizer evaluations.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend.
    n_terms : int, default=160
        COS expansion terms.
    truncation_width : float, default=16.0
        COS truncation width.
    riccati_steps : int, default=512
        Full Riccati step count for final reporting.
    scheme : str, default='adams'
        Riccati solver scheme.
    lambda_skew : float, default=0.0
        Optional skew-penalty weight.
    skew_tau_days : tuple, default=(7, 14, 21, 30, 45, 60)
        Maturities used in the skew penalty.
    surface_fit_map : dict, optional
        Date-to-surface mapping used for market-skew penalty evaluation.
    n_jobs : int, default=1
        Reserved parallelism parameter.

    Returns
    -------
    dict
        Dictionary containing daily ``params`` and quote-level ``fit`` DataFrames.
    """

    q = calibration_weights(quotes)
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    if calibration_dates is None:
        counts = q.groupby("date").size()
        dates = list(counts[counts >= int(min_quotes)].index)
    else:
        dates = [pd.Timestamp(d).normalize() for d in calibration_dates]
    rows = []
    fits = []
    last = None
    h_lookup = {}
    if h_by_date is not None:
        if isinstance(h_by_date, pd.Series):
            h_lookup = {pd.Timestamp(k).normalize(): float(v) for k, v in h_by_date.dropna().items()}
        elif isinstance(h_by_date, pd.DataFrame):
            h_col = "h_from_skew" if "h_from_skew" in h_by_date.columns else "h"
            h_lookup = {
                pd.Timestamp(r["date"]).normalize(): float(r[h_col])
                for _, r in h_by_date.dropna(subset=["date", h_col]).iterrows()
            }
        elif isinstance(h_by_date, dict):
            h_lookup = {pd.Timestamp(k).normalize(): float(v) for k, v in h_by_date.items() if np.isfinite(v)}
        else:
            raise TypeError("h_by_date must be a Series, DataFrame, dict, or None.")
    h_mode_key = str(h_mode).lower()
    if h_mode_key not in {"fixed", "penalized", "free"}:
        raise ValueError("h_mode must be one of {'fixed', 'penalized', 'free'}.")
    for d in dates:
        day = q[q["date"].eq(d)].copy()
        if len(day) < int(min_quotes):
            continue
        t0 = time.perf_counter()
        atm = float(np.nanmedian(day.get("iv_mid", pd.Series(0.25, index=day.index))))
        h_anchor = float(h_lookup.get(pd.Timestamp(d).normalize(), h_start))
        start = np.asarray(last if last is not None else [h_anchor, atm * atm, 2.0, atm * atm, 0.60, -0.60], dtype=float)
        lo = np.asarray([0.04, 1e-5, 0.05, 1e-5, 0.02, -0.999], dtype=float)
        hi = np.asarray([0.45, 4.0, 12.0, 4.0, 4.0, 0.999], dtype=float)
        start[0] = h_anchor
        target = day["mid"].to_numpy(float)
        scale = day["calib_scale_px"].to_numpy(float) / np.sqrt(day["obs_weight"].to_numpy(float))
        groups = cos_group_arrays(day)
        # Auxiliary scalars for skew penalty
        spot_d = float(_numeric(day, ("spot",), 1.0).median())
        rate_d = float(_numeric(day, ("rate",), 0.0).median())
        div_d = float(_numeric(day, ("implied_dividend_yield", "dividend_yield"), 0.0).median())
        sfit = (surface_fit_map or {}).get(d) if float(lambda_skew) > 0.0 else None
        # Use fewer Riccati steps for inner optimization evaluations - roughly 3x
        # faster with negligible effect on convergence direction.  Final prices
        # are always computed with the full riccati_steps.
        riccati_inner = max(32, int(riccati_steps) // 3)
        _dk_skew = 0.01

        def prices_for(p, n_ric=riccati_inner):
            out = np.empty(len(day), dtype=float)
            for pos, strike, tau_g, spot_g, rate_g, div_g, option_type in groups:
                try:
                    out[pos] = rough_heston_prices(
                        p, strike, tau_g, spot_g, rate_g, div_g,
                        option_type=option_type, engine=engine, n_terms=n_terms,
                        truncation_width=truncation_width, riccati_steps=n_ric, scheme=scheme,
                    )
                except FloatingPointError:
                    out[pos] = np.nan
            return out

        def skew_penalty_vec(p):
            """Append sqrt(lambda_skew) * (psi_model - psi_market) / scale residuals."""
            if sfit is None:
                return np.empty(0, dtype=float)
            tau_min_d = float(pd.to_numeric(day["tau"], errors="coerce").min())
            tau_max_d = float(pd.to_numeric(day["tau"], errors="coerce").max())
            sk_scale = 0.25
            sk_res = []
            for tau_sk_d in np.asarray(skew_tau_days, dtype=float):
                tau_sk = tau_sk_d / 365.25
                if tau_sk < tau_min_d - 3.0 / 365.25 or tau_sk > tau_max_d + 3.0 / 365.25:
                    continue
                strikes_sk = np.asarray([spot_d * np.exp(-_dk_skew), spot_d * np.exp(_dk_skew)], dtype=float)
                tau_arr_sk = np.full(2, tau_sk, dtype=float)
                try:
                    px_sk = rough_heston_prices(
                        p, strikes_sk, tau_arr_sk, spot_d, rate_d, div_d,
                        option_type="call", engine=engine, n_terms=max(32, n_terms // 2),
                        truncation_width=truncation_width, riccati_steps=riccati_inner, scheme=scheme,
                    )
                except FloatingPointError:
                    continue
                if not np.all(np.isfinite(px_sk)):
                    continue
                disc_sk = float(np.exp(-rate_d * tau_sk))
                fwd_sk = float(spot_d * np.exp((rate_d - div_d) * tau_sk))
                iv_sk = implied_vol("call", px_sk, fwd_sk, strikes_sk, tau_sk, disc_sk, engine="auto")
                if not np.all(np.isfinite(iv_sk)):
                    continue
                psi_model = float((iv_sk[1] - iv_sk[0]) / (2.0 * _dk_skew))
                # market skew from the fitted surface
                try:
                    psi_market = float((
                        surface_iv(sfit, np.asarray([_dk_skew]), np.asarray([tau_sk]))[0]
                        - surface_iv(sfit, np.asarray([-_dk_skew]), np.asarray([tau_sk]))[0]
                    ) / (2.0 * _dk_skew))
                except Exception:
                    continue
                if np.isfinite(psi_market):
                    sk_res.append(float(np.sqrt(float(lambda_skew))) * (psi_model - psi_market) / sk_scale)
            return np.asarray(sk_res, dtype=float)

        def residual(p):
            if not np.all(np.isfinite(p)):
                return np.full_like(target, 1e6)
            out = prices_for(p)
            if not np.all(np.isfinite(out)):
                return np.full_like(target, 1e6)
            price_res = (out - target) / scale
            if float(lambda_skew) > 0.0:
                sk_vec = skew_penalty_vec(p)
                if sk_vec.size > 0:
                    return np.r_[price_res, sk_vec]
            return price_res

        h_fixed = float(np.clip(h_anchor, lo[0], hi[0]))

        def residual_fixed(x):
            return residual(np.r_[h_fixed, x])

        fixed_nfev = max(6, int(max_nfev) // 2)
        base_guess = np.asarray([atm * atm, 2.0, atm * atm, 0.80, -0.60], dtype=float)
        calm_last = np.clip(start[1:].copy(), lo[1:], hi[1:])
        calm_last[3] = min(calm_last[3], 2.50)
        fixed_guesses = [np.clip(start[1:], lo[1:], hi[1:]), np.clip(base_guess, lo[1:], hi[1:]), calm_last]
        fixed_best = None
        for guess in fixed_guesses:
            fixed_i = optimize.least_squares(
                residual_fixed,
                guess,
                bounds=(lo[1:], hi[1:]),
                max_nfev=fixed_nfev,
                xtol=1e-7,
                ftol=1e-7,
                gtol=1e-7,
            )
            loss_i = float(np.nanmean(fixed_i.fun * fixed_i.fun))
            if fixed_best is None or loss_i < fixed_best[0]:
                fixed_best = (loss_i, fixed_i)
        fixed = fixed_best[1]
        fixed_params = np.r_[h_fixed, fixed.x]
        if h_mode_key == "fixed":
            class FixedResult:
                pass

            res = FixedResult()
            res.x = fixed_params
            res.fun = residual(fixed_params)
            res.nfev = fixed.nfev
            res.success = fixed.success
            best = (float(np.nanmean(res.fun * res.fun)), res)
        else:
            def residual_anchor(p):
                base = residual(p)
                if h_mode_key == "penalized" and float(h_penalty) > 0.0:
                    base = np.r_[base, np.sqrt(float(h_penalty)) * (float(p[0]) - h_fixed)]
                return base

            free_nfev = max(8, int(max_nfev))
            candidates = [fixed_params, np.clip(start, lo, hi)]
            if last is not None:
                last_guess = np.clip(last, lo, hi)
                last_guess[0] = h_fixed
                candidates.append(last_guess)
            best = None
            for guess in candidates:
                res_i = optimize.least_squares(
                    residual_anchor,
                    np.clip(guess, lo, hi),
                    bounds=(lo, hi),
                    max_nfev=free_nfev,
                    xtol=1e-7,
                    ftol=1e-7,
                    gtol=1e-7,
                )
                loss = float(np.nanmean(residual(res_i.x) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, res_i)
            res = best[1]
        px = prices_for(res.x, riccati_steps)  # full accuracy for reporting
        fit = day.copy()
        fit["model"] = "rough_heston"
        fit["model_price"] = px
        fit["price_residual"] = px - target
        finite_px = np.isfinite(px)
        err = np.full_like(target, np.nan, dtype=float)
        err[finite_px] = (px[finite_px] - target[finite_px]) / scale[finite_px]
        finite_share = float(np.mean(finite_px)) if len(finite_px) else 0.0
        vega = _numeric(fit, ("vega",), 1.0).abs().replace(0.0, np.nan).to_numpy(float)
        inside = ((px >= pd.to_numeric(fit.get("bid", fit["mid"]), errors="coerce").to_numpy(float)) & (px <= pd.to_numeric(fit.get("ask", fit["mid"]), errors="coerce").to_numpy(float)))
        opt = fit["option_type"].astype(str).str.lower()
        dte = pd.to_numeric(fit.get("dte_days", fit["tau"] * 365.25), errors="coerce").to_numpy(float)
        moneyness = pd.to_numeric(fit.get("moneyness", fit["strike"] / fit["spot"]), errors="coerce").to_numpy(float)
        tail = opt.str.startswith("p").to_numpy() & (moneyness <= 0.90)
        short = dte <= 30.0
        rmse_scaled = float(np.sqrt(np.nanmean(err * err))) if np.isfinite(err).any() else np.nan
        iv_err = (px - target) / np.maximum(vega, 1e-6)
        nfev_total = int(res.nfev if h_mode_key == "fixed" else res.nfev + fixed.nfev)
        row_success = bool(finite_share >= 0.999 and np.isfinite(rmse_scaled) and (res.success or rmse_scaled < 5.0))
        row = {
            "model": "rough_heston",
            "date": d,
            "success": row_success,
            "nfev": nfev_total,
            "runtime_sec": float(time.perf_counter() - t0),
            "quotes": int(len(day)),
            "finite_price_share": finite_share,
            "weighted_price_rmse": rmse_scaled,
            "median_abs_price_error": float(np.nanmedian(np.abs(px[finite_px] - target[finite_px]))) if finite_px.any() else np.nan,
            "weighted_iv_rmse": float(np.sqrt(np.nanmean(iv_err[np.isfinite(iv_err)] ** 2))) if np.isfinite(iv_err).any() else np.nan,
            "bid_ask_hit_rate": float(np.nanmean(inside)),
            "otm_put_rmse": float(np.sqrt(np.nanmean((err[tail]) ** 2))) if np.any(tail) else np.nan,
            "short_maturity_rmse": float(np.sqrt(np.nanmean((err[short]) ** 2))) if np.any(short) else np.nan,
            "stage_fixed_rmse": float(np.sqrt(np.nanmean(fixed.fun * fixed.fun))),
            "stage_free_rmse": float(np.sqrt(max(best[0], 0.0))),
            "h_mode": h_mode_key,
            "h_anchor": h_fixed,
            "h_fixed": h_fixed,
            "h": float(res.x[0]),
            "p0": float(res.x[1]),
            "p1": float(res.x[2]),
            "p2": float(res.x[3]),
            "p3": float(res.x[4]),
            "p4": float(res.x[5]),
            "v0": float(res.x[1]),
            "kappa": float(res.x[2]),
            "theta": float(res.x[3]),
            "sigma_v": float(res.x[4]),
            "rho": float(res.x[5]),
        }
        rows.append(row)
        fits.append(
            fit[
                [
                    c
                    for c in [
                        "date",
                        "expiry",
                        "strike",
                        "option_type",
                        "mid",
                        "bid",
                        "ask",
                        "spot",
                        "tau",
                        "dte_days",
                        "moneyness",
                        "log_moneyness",
                        "k",
                        "iv_mid",
                        "vega",
                        "model",
                        "model_price",
                        "price_residual",
                        "calib_scale_px",
                        "obs_weight",
                    ]
                    if c in fit.columns
                ]
            ].copy()
        )
        if row["success"]:
            last = res.x
    return {"params": pd.DataFrame(rows), "fit": pd.concat(fits, ignore_index=True) if fits else pd.DataFrame()}


def rough_heston_residuals(fit: pd.DataFrame, *, scale_col: str = "calib_scale_px") -> pd.DataFrame:
    """Aggregate rough-Heston scaled residuals by moneyness and maturity buckets.

    Parameters
    ----------
    fit : pandas.DataFrame
        Quote-level rough-Heston fit table containing price residuals.
    scale_col : str, default='calib_scale_px'
        Residual scale column.

    Returns
    -------
    pandas.DataFrame
        Grouped residual table by model, moneyness bucket, and DTE bucket. Returns an
        empty DataFrame for empty input.
    """

    if fit.empty:
        return pd.DataFrame()
    q = fit.copy()
    if "log_moneyness" not in q.columns:
        q["log_moneyness"] = np.log(pd.to_numeric(q["strike"], errors="coerce") / pd.to_numeric(q["spot"], errors="coerce"))
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q["tau"], errors="coerce") * 365.25
    q["moneyness_bucket"] = pd.cut(q["log_moneyness"], [-0.60, -0.30, -0.15, -0.05, 0.05, 0.15, 0.30, 0.60])
    q["dte_bucket"] = pd.cut(q["dte_days"], [0, 14, 30, 60, 90, 120, 180, 365])
    scale = pd.to_numeric(q.get(scale_col, 1.0), errors="coerce").fillna(1.0).clip(lower=1e-6)
    q["scaled_residual"] = pd.to_numeric(q["price_residual"], errors="coerce") / scale
    return q.groupby(["model", "moneyness_bucket", "dte_bucket"], observed=True).agg(median_scaled_residual=("scaled_residual", "median"), rows=("scaled_residual", "size")).reset_index()


def compare_heston_rough_heston(*, heston_daily: pd.DataFrame, rough_daily: pd.DataFrame) -> pd.DataFrame:
    """Compare daily Heston and rough-Heston calibration diagnostics.

    Parameters
    ----------
    heston_daily : pandas.DataFrame
        Daily Heston diagnostic table.
    rough_daily : pandas.DataFrame
        Daily rough-Heston diagnostic table.

    Returns
    -------
    pandas.DataFrame
        Model-level comparison table including dates, quote counts, success rate,
        average price/IV RMSE, tail and short-maturity errors, bid-ask hit rate, and
        total runtime.
    """

    rows = []
    for name, frame in [("heston", heston_daily), ("rough_heston", rough_daily)]:
        if frame is None or frame.empty:
            rows.append({"model": name, "dates": 0})
            continue
        rows.append(
            {
                "model": name,
                "dates": int(frame["date"].nunique()) if "date" in frame.columns else int(len(frame)),
                "quotes": int(pd.to_numeric(frame.get("quotes", 0), errors="coerce").sum()),
                "success_rate": float(pd.to_numeric(frame.get("success", True), errors="coerce").mean()),
                "weighted_price_rmse": float(pd.to_numeric(frame.get("weighted_price_rmse"), errors="coerce").mean()),
                "weighted_iv_rmse": float(pd.to_numeric(frame.get("weighted_iv_rmse"), errors="coerce").mean()) if "weighted_iv_rmse" in frame.columns else np.nan,
                "otm_put_rmse": float(pd.to_numeric(frame.get("otm_put_rmse"), errors="coerce").mean()) if "otm_put_rmse" in frame.columns else np.nan,
                "short_maturity_rmse": float(pd.to_numeric(frame.get("short_maturity_rmse"), errors="coerce").mean()) if "short_maturity_rmse" in frame.columns else np.nan,
                "median_abs_price_error": float(pd.to_numeric(frame.get("median_abs_price_error"), errors="coerce").median()),
                "bid_ask_hit_rate": float(pd.to_numeric(frame.get("bid_ask_hit_rate"), errors="coerce").mean()) if "bid_ask_hit_rate" in frame.columns else np.nan,
                "runtime_sec": float(pd.to_numeric(frame.get("runtime_sec", 0), errors="coerce").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_price_rmse").reset_index(drop=True)


def riccati_convergence(quotes: pd.DataFrame, *, params, n_grid_values=(128, 256, 512, 1024), n_terms: int = 160, engine: str = "auto") -> pd.DataFrame:
    """Check rough-Heston price convergence as the Riccati grid is refined.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table; the first row is used as the test contract.
    params : mapping or array-like
        Rough-Heston parameters.
    n_grid_values : iterable, default=(128, 256, 512, 1024)
        Riccati step counts to evaluate.
    n_terms : int, default=160
        COS expansion terms.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend.

    Returns
    -------
    pandas.DataFrame
        Convergence table with Riccati step count, terminal Riccati magnitude, price,
        and absolute price change from the previous grid size.
    """

    q = quotes.head(1).copy()
    if q.empty:
        return pd.DataFrame()
    row = q.iloc[0]
    rows = []
    last = np.nan
    for n in n_grid_values:
        ric = fractional_riccati(1.0 - 0.5j, params, float(row["tau"]), n_steps=int(n))
        px = rough_heston_prices(params, np.asarray([row["strike"]]), np.asarray([row["tau"]]), float(row["spot"]), float(row.get("rate", 0.0)), float(row.get("implied_dividend_yield", 0.0)), option_type=str(row["option_type"]), engine=engine, n_terms=n_terms, riccati_steps=int(n))
        price = float(px[0])
        rows.append({"riccati_steps": int(n), "terminal_abs": float(ric["abs"].iloc[-1]), "price": price, "abs_change": abs(price - last) if np.isfinite(last) else np.nan})
        last = price
    return pd.DataFrame(rows)


def rough_delta_grid(
    quotes: pd.DataFrame,
    *,
    heston_params,
    rough_params,
    rbergomi_params=None,
    xi=None,
    surface_fit=None,
    k_values=None,
    tau_days=None,
    n_terms: int = 160,
    riccati_steps: int = 512,
    bump: float = 0.005,
    paths: int = 10000,
    steps: int = 120,
    seed: int = 7,
    engine: str = "auto",
    mc_engine: str = "auto",
) -> pd.DataFrame:
    """Compare rough-Heston, Heston, flat-BSM, and surface-BSM deltas on a strike-maturity grid.

    The function computes finite-difference deltas for Heston and rough-Heston prices,
    analytic flat-BSM deltas, and optionally surface-implied BSM deltas.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table used to infer spot, rate, dividend/carry, and default IVs.
    heston_params : mapping or array-like
        Heston parameters.
    rough_params : mapping or array-like
        Rough-Heston parameters.
    rbergomi_params : optional
        Reserved parameter for compatibility.
    xi : optional
        Reserved parameter for compatibility with rough-volatility workflows.
    surface_fit : dict, optional
        Fitted volatility surface used for surface-BSM delta.
    k_values : array-like, optional
        Log-moneyness grid.
    tau_days : array-like, optional
        Maturities in days.
    n_terms : int, default=160
        COS expansion terms.
    riccati_steps : int, default=512
        Rough-Heston Riccati steps.
    bump : float, default=0.005
        Relative spot bump for finite-difference deltas.
    paths, steps, seed : int
        Reserved Monte Carlo parameters for compatibility.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Fourier-pricing backend.
    mc_engine : str, default='auto'
        Reserved Monte Carlo backend label.

    Returns
    -------
    pandas.DataFrame
        Delta comparison grid with model deltas and rough-minus-benchmark columns.
    """

    q = quotes.copy()
    spot = float(pd.to_numeric(q["spot"], errors="coerce").median())
    rate = float(_numeric(q, ("rate",), 0.0).median())
    div = float(_numeric(q, ("implied_dividend_yield", "dividend_yield"), 0.0).median())
    if k_values is None:
        k_values = np.linspace(-0.35, 0.25, 49)
    if tau_days is None:
        tau_days = np.array([7, 14, 30, 60, 90], dtype=float)
    rows = []
    hp = _params_array(heston_params)[1:]
    rp = _params_array(rough_params)
    ds = spot * float(bump)
    flat_sigma = float(pd.to_numeric(q["iv_mid"], errors="coerce").median()) if "iv_mid" in q.columns else 0.25
    for d in np.asarray(tau_days, dtype=float):
        tau = d / 365.25
        for k in np.asarray(k_values, dtype=float):
            strike = spot * np.exp(k)
            df = np.exp(-rate * tau)
            fwd = spot * np.exp((rate - div) * tau)
            flat_bsm_delta = float(bsm.forward_bsm_delta("call", fwd, strike, tau, flat_sigma, df) * np.exp((rate - div) * tau))
            if surface_fit is not None:
                surface_sigma = float(surface_iv(surface_fit, np.asarray([k]), np.asarray([tau]))[0])
            elif "iv_mid" in q.columns and "k" in q.columns:
                score = (pd.to_numeric(q["tau"], errors="coerce").sub(tau).abs() + 0.04 * pd.to_numeric(q["k"], errors="coerce").sub(k).abs()).to_numpy(dtype=float)
                j = int(np.nanargmin(score)) if np.isfinite(score).any() else 0
                surface_sigma = float(pd.to_numeric(q["iv_mid"], errors="coerce").iloc[j])
            else:
                surface_sigma = flat_sigma
            surface_sigma = float(np.clip(surface_sigma, 0.03, 5.0))
            surface_bsm_delta = float(bsm.forward_bsm_delta("call", fwd, strike, tau, surface_sigma, df) * np.exp((rate - div) * tau))
            hp_up = cos_prices("heston", hp, np.asarray([strike]), np.asarray([tau]), spot + ds, rate, div, option_type="call", engine=engine, n_terms=n_terms)[0]
            hp_dn = cos_prices("heston", hp, np.asarray([strike]), np.asarray([tau]), spot - ds, rate, div, option_type="call", engine=engine, n_terms=n_terms)[0]
            rp_up = rough_heston_prices(rp, np.asarray([strike]), np.asarray([tau]), spot + ds, rate, div, option_type="call", engine=engine, n_terms=n_terms, riccati_steps=riccati_steps)[0]
            rp_dn = rough_heston_prices(rp, np.asarray([strike]), np.asarray([tau]), spot - ds, rate, div, option_type="call", engine=engine, n_terms=n_terms, riccati_steps=riccati_steps)[0]
            rows.append(
                {
                    "tau_days": float(d),
                    "tau": float(tau),
                    "k": float(k),
                    "strike": float(strike),
                    "bsm_delta": flat_bsm_delta,
                    "flat_bsm_delta": flat_bsm_delta,
                    "surface_bsm_delta": surface_bsm_delta,
                    "heston_delta": float((hp_up - hp_dn) / (2.0 * ds)),
                    "rough_heston_delta": float((rp_up - rp_dn) / (2.0 * ds)),
                }
            )
    out = pd.DataFrame(rows)
    out["rough_minus_bsm"] = out["rough_heston_delta"] - out["bsm_delta"]
    out["rough_minus_flat_bsm"] = out["rough_heston_delta"] - out["flat_bsm_delta"]
    out["rough_minus_surface_bsm"] = out["rough_heston_delta"] - out["surface_bsm_delta"]
    out["rough_minus_heston"] = out["rough_heston_delta"] - out["heston_delta"]
    return out


__all__ = [
    "atm_skew_term_structure",
    "compare_heston_rough_heston",
    "fit_rough_heston_dates",
    "forward_variance_curve",
    "fractional_riccati",
    "rbergomi_calibration",
    "rbergomi_smile",
    "riccati_convergence",
    "rough_delta_grid",
    "rough_heston_cf_diagnostics",
    "rough_heston_cf",
    "rough_heston_iv",
    "rough_heston_prices",
    "rough_heston_residuals",
    "simulate_rbergomi",
    "skew_power_law",
]
