from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab._optional import get_cpp_kernels, prefer_auto_engine
from quantfinlab.options.bates import bates_cf
from quantfinlab.options.bsm import bsm_cf
from quantfinlab.options.heston import heston_cf
from quantfinlab.options.merton import merton_cf
from quantfinlab.options.variance_gamma import vg_cf

MODEL_IDS = {"bsm": 0, "merton": 1, "vg": 2, "variance_gamma": 2, "heston": 3, "bates": 4}


def _resolve_engine(engine: str) -> str:
    key = str(engine).lower()
    if key == "auto":
        return prefer_auto_engine()
    if key in {"numpy", "python"}:
        return "numpy"
    if key == "numba":
        return "numba"
    if key in {"cpp", "c++"}:
        return "cpp"
    raise ValueError("engine must be one of {'auto', 'numpy', 'numba', 'cpp'}.")


def _model_id(model) -> int:
    if isinstance(model, (int, np.integer)):
        return int(model)
    key = str(model).lower()
    if key not in MODEL_IDS:
        raise ValueError(f"unknown Fourier model {model!r}")
    return MODEL_IDS[key]


def _flag(option_type) -> np.ndarray:
    arr = np.asarray(option_type)
    if np.issubdtype(arr.dtype, np.number):
        return np.where(arr.astype(float) > 0, 1, -1).astype(np.int32)
    text = np.char.lower(arr.astype(str))
    return np.where(np.char.startswith(text, "c"), 1, -1).astype(np.int32)


def _params(model, params) -> np.ndarray:
    if isinstance(params, dict):
        mid = _model_id(model)
        if mid == 0:
            vals = [params.get("sigma", 0.20)]
        elif mid == 1:
            vals = [params.get("sigma", 0.20), params.get("lambda_jump", params.get("lam", 0.30)), params.get("mu_jump", -0.05), params.get("sigma_jump", 0.20)]
        elif mid == 2:
            vals = [params.get("sigma", 0.20), params.get("theta", -0.05), params.get("nu", 0.20)]
        elif mid == 3:
            vals = [params.get("v0", 0.04), params.get("kappa", 2.0), params.get("theta", 0.04), params.get("sigma_v", params.get("xi", 0.60)), params.get("rho", -0.50)]
        else:
            vals = [
                params.get("v0", 0.04),
                params.get("kappa", 2.0),
                params.get("theta", 0.04),
                params.get("sigma_v", params.get("xi", 0.60)),
                params.get("rho", -0.50),
                params.get("lambda_jump", params.get("lam", 0.30)),
                params.get("mu_jump", -0.05),
                params.get("sigma_jump", 0.20),
            ]
        return np.asarray(vals, dtype=float)
    return np.asarray(params, dtype=float)


def model_cf(model, u, params, spot, rate, dividend_yield, tau):
    """Evaluate a supported model characteristic function of log spot at expiry.

    The function dispatches to Black-Scholes, Merton jump-diffusion, Variance Gamma,
    Heston, or Bates characteristic functions according to the model identifier and
    parameter set.

    Parameters
    ----------
    model : str or identifier
        Model name or supported model identifier.
    u : array-like
        Complex Fourier argument.
    params : mapping or array-like
        Model parameters in the convention expected by the selected model.
    spot : float or array-like
        Current spot price.
    rate : float or array-like
        Continuously compounded risk-free rate.
    dividend_yield : float or array-like
        Continuously compounded dividend yield.
    tau : float or array-like
        Time to expiry in years.

    Returns
    -------
    numpy.ndarray
        Characteristic-function values.
    """

    mid = _model_id(model)
    p = _params(model, params)
    if mid == 0:
        return bsm_cf(u, spot, rate, dividend_yield, tau, p[0])
    if mid == 1:
        return merton_cf(u, spot, rate, dividend_yield, tau, p[0], p[1], p[2], p[3])
    if mid == 2:
        return vg_cf(u, spot, rate, dividend_yield, tau, p[0], p[1], p[2])
    if mid == 3:
        return heston_cf(u, spot, rate, dividend_yield, tau, p[0], p[1], p[2], p[3], p[4])
    return bates_cf(u, spot, rate, dividend_yield, tau, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7])


def _direct_call_numpy(model, params, spot, strike, rate, dividend_yield, tau, *, n: int = 512, u_max: float = 120.0):
    spot = float(spot)
    strike = float(strike)
    rate = float(rate)
    dividend_yield = float(dividend_yield)
    tau = float(tau)
    u = (np.arange(1, int(n) + 1) - 0.5) * float(u_max) / int(n)
    du = float(u_max) / int(n)
    phi_mi = model_cf(model, -1j, params, spot, rate, dividend_yield, tau)
    log_k = np.log(strike)
    p1 = 0.5 + du / np.pi * np.sum(np.real(np.exp(-1j * u * log_k) * model_cf(model, u - 1j, params, spot, rate, dividend_yield, tau) / (1j * u * phi_mi)))
    p2 = 0.5 + du / np.pi * np.sum(np.real(np.exp(-1j * u * log_k) * model_cf(model, u, params, spot, rate, dividend_yield, tau) / (1j * u)))
    return max(spot * np.exp(-dividend_yield * tau) * p1 - strike * np.exp(-rate * tau) * p2, 0.0)


def direct_price(
    model,
    params,
    spot,
    strike,
    rate,
    dividend_yield,
    tau,
    *,
    option_type="call",
    n: int = 512,
    u_max: float = 120.0,
    engine: str = "auto",
):
    """Price vanilla options by direct Fourier integration.

    The function evaluates model prices contract by contract or through accelerated
    backends when available. Put prices are obtained from call prices through forward
    put-call parity.

    Parameters
    ----------
    model : str or identifier
        Supported model name.
    params : mapping or array-like
        Model parameters.
    spot : array-like
        Spot prices.
    strike : array-like
        Strike prices.
    rate : array-like
        Continuously compounded risk-free rates.
    dividend_yield : array-like
        Continuously compounded dividend yields.
    tau : array-like
        Times to expiry in years.
    option_type : array-like or scalar, default='call'
        Option type labels.
    n : int, default=512
        Number of integration grid points.
    u_max : float, default=120.0
        Upper integration frequency.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.

    Returns
    -------
    numpy.ndarray
        Option prices with the broadcast input shape.

    Raises
    ------
    ValueError
        If an explicitly requested C++ batch path cannot support the supplied batch
        shape.
    """

    s, k, r, q, t = np.broadcast_arrays(np.asarray(spot, dtype=float), np.asarray(strike, dtype=float), np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float), np.asarray(tau, dtype=float))
    flags = _flag(option_type)
    if flags.size == 1 and k.size > 1:
        flags = np.full(k.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    resolved = _resolve_engine(engine)
    p = _params(model, params)
    if resolved == "cpp":
        kernels = get_cpp_kernels("Fourier direct option pricing")
        flat_s = s.reshape(-1)
        flat_r = r.reshape(-1)
        flat_q = q.reshape(-1)
        flat_flags = flags.reshape(-1)
        if (
            np.unique(np.round(flat_s, 12)).size == 1
            and np.unique(np.round(flat_r, 12)).size == 1
            and np.unique(np.round(flat_q, 12)).size == 1
            and np.unique(flat_flags).size == 1
            and hasattr(kernels, "direct_prices")
        ):
            return np.asarray(
                kernels.direct_prices(
                    _model_id(model),
                    p,
                    k.reshape(-1),
                    t.reshape(-1),
                    float(flat_s[0]),
                    float(flat_r[0]),
                    float(flat_q[0]),
                    int(n),
                    float(u_max),
                    int(flat_flags[0]),
                ),
                dtype=float,
            ).reshape(k.shape)
        if str(engine).lower() in {"cpp", "c++"}:
            raise ValueError("The C++ Fourier direct engine requires common spot, rate, dividend yield, and option type across the batch.")
        resolved = "numba"
    if resolved == "numba":
        try:
            from quantfinlab.numerics.fourier import direct_price_numba

            return direct_price_numba(_model_id(model), p, s.reshape(-1), k.reshape(-1), r.reshape(-1), q.reshape(-1), t.reshape(-1), flags.reshape(-1), n=n, u_max=u_max).reshape(k.shape)
        except Exception:
            if str(engine).lower() == "numba":
                raise
    out = np.empty(k.size, dtype=float)
    for i, vals in enumerate(zip(s.reshape(-1), k.reshape(-1), r.reshape(-1), q.reshape(-1), t.reshape(-1), flags.reshape(-1), strict=False)):
        call = _direct_call_numpy(model, p, vals[0], vals[1], vals[2], vals[3], vals[4], n=n, u_max=u_max)
        out[i] = call if vals[5] > 0 else call - vals[0] * np.exp(-vals[3] * vals[4]) + vals[1] * np.exp(-vals[2] * vals[4])
    return out.reshape(k.shape)


def fft_grid(model, params, spot, rate, dividend_yield, tau, *, alpha: float = 1.5, n: int = 256, eta: float = 0.25, option_type="call", engine: str = "auto") -> pd.DataFrame:
    """Generate Carr-Madan FFT option prices over a strike grid.

    Parameters
    ----------
    model : str or identifier
        Supported model name.
    params : mapping or array-like
        Model parameters.
    spot : float
        Spot price.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuously compounded dividend yield.
    tau : float
        Time to expiry in years.
    alpha : float, default=1.5
        Dampening parameter in the Carr-Madan transform.
    n : int, default=256
        Number of FFT grid points.
    eta : float, default=0.25
        Frequency-grid spacing.
    option_type : {'call', 'put'}, default='call'
        Option type to return. Put prices are obtained through parity from the call
        grid.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.

    Returns
    -------
    pandas.DataFrame
        Strike-price grid with columns ``strike`` and ``price``.
    """

    resolved = _resolve_engine(engine)
    p = _params(model, params)
    flag = int(_flag(option_type).reshape(-1)[0])
    if resolved == "cpp":
        kernels = get_cpp_kernels("Carr-Madan FFT option pricing")
        out = kernels.fft_prices(_model_id(model), p, float(spot), float(rate), float(dividend_yield), float(tau), float(alpha), int(n), float(eta), flag)
        return pd.DataFrame({"strike": np.asarray(out["strikes"], dtype=float), "price": np.asarray(out["prices"], dtype=float)})
    if resolved == "numba":
        from quantfinlab.numerics.fourier import carr_madan_fft_numba

        strikes, call_prices = carr_madan_fft_numba(_model_id(model), p, float(spot), float(rate), float(dividend_yield), float(tau), alpha=float(alpha), n=int(n), eta=float(eta), center_log=float(np.log(spot)))
        if flag > 0:
            prices = call_prices
        else:
            prices = call_prices - float(spot) * np.exp(-float(dividend_yield) * float(tau)) + strikes * np.exp(-float(rate) * float(tau))
        return pd.DataFrame({"strike": np.asarray(strikes, dtype=float), "price": np.asarray(prices, dtype=float)})
    nn = int(n)
    eta = float(eta)
    alpha = float(alpha)
    dk = 2.0 * np.pi / (nn * eta)
    start = float(np.log(spot)) - 0.5 * nn * dk
    u = np.arange(nn, dtype=float) * eta
    shifted = u - 1j * (alpha + 1.0)
    denom = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    weight = np.ones(nn, dtype=float)
    weight[0] = 0.5
    psi = np.exp(-float(rate) * float(tau)) * model_cf(model, shifted, p, float(spot), float(rate), float(dividend_yield), float(tau)) / denom
    x = psi * np.exp(-1j * u * start) * eta * weight
    y = np.fft.fft(x)
    log_k = start + dk * np.arange(nn, dtype=float)
    strikes = np.exp(log_k)
    call_prices = np.maximum(np.exp(-alpha * log_k) * np.real(y) / np.pi, 0.0)
    if flag > 0:
        prices = call_prices
    else:
        prices = call_prices - float(spot) * np.exp(-float(dividend_yield) * float(tau)) + strikes * np.exp(-float(rate) * float(tau))
    return pd.DataFrame({"strike": strikes, "price": np.asarray(prices, dtype=float)})


def fft_prices(*args, **kwargs) -> pd.DataFrame:
    """Compatibility alias for Carr-Madan FFT strike-grid pricing.

    Parameters
    ----------
    *args, **kwargs
        Arguments forwarded to the FFT grid pricer.

    Returns
    -------
    pandas.DataFrame
        Strike-price grid with model prices.
    """

    return fft_grid(*args, **kwargs)


def _cos_chi_psi(u: np.ndarray, a: float, c: float, d: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    chi = (
        np.cos(u * (d - a)) * np.exp(d)
        - np.cos(u * (c - a)) * np.exp(c)
        + u * (np.sin(u * (d - a)) * np.exp(d) - np.sin(u * (c - a)) * np.exp(c))
    ) / (1.0 + u * u)
    psi = np.empty_like(u, dtype=float)
    psi[0] = d - c
    if len(u) > 1:
        uu = u[1:]
        psi[1:] = (np.sin(uu * (d - a)) - np.sin(uu * (c - a))) / uu
    return chi, psi


def _call_custom_cf(cf, u: np.ndarray, tau: float, spot: float, rate: float, dividend_yield: float):
    try:
        return cf(u, tau, spot, rate, dividend_yield)
    except TypeError:
        return cf(u, spot=spot, rate=rate, dividend_yield=dividend_yield, tau=tau)


def _value_hint(value, idx: int, default: float) -> float:
    if value is None:
        return float(default)
    if callable(value):
        return float(value(idx))
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[idx])


def _cos_prices_custom_cf(
    cf,
    strikes,
    tau,
    spot,
    rate,
    dividend_yield,
    *,
    n_terms: int,
    truncation_width: float,
    option_type="call",
    variance_hint=None,
    x_center=None,
    x_width=None,
) -> np.ndarray:
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(tau, dtype=float)
    s, k, r, q, t = np.broadcast_arrays(np.asarray(spot, dtype=float), k, np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float), t)
    flags = _flag(option_type)
    if flags.size == 1 and k.size > 1:
        flags = np.full(k.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    elif flags.size != k.size:
        flags = np.broadcast_to(flags, k.shape).reshape(-1).astype(np.int32)
    nn = int(n_terms)
    out = np.full(k.size, np.nan, dtype=float)
    flat_s = s.reshape(-1)
    flat_k = k.reshape(-1)
    flat_r = r.reshape(-1)
    flat_q = q.reshape(-1)
    flat_t = t.reshape(-1)
    flat_flags = flags.reshape(-1)
    valid = np.isfinite(flat_s + flat_k + flat_r + flat_q + flat_t) & (flat_s > 0.0) & (flat_k > 0.0) & (flat_t > 0.0)
    if not np.any(valid):
        return out.reshape(k.shape)
    valid_idx = np.flatnonzero(valid)
    keys = np.column_stack(
        [
            np.round(flat_s[valid_idx], 12),
            np.round(flat_r[valid_idx], 12),
            np.round(flat_q[valid_idx], 12),
            np.round(flat_t[valid_idx], 12),
            flat_flags[valid_idx].astype(float),
        ]
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    for group in range(int(inverse.max()) + 1):
        idx = valid_idx[inverse == group]
        if idx.size == 0:
            continue
        first = int(idx[0])
        si = float(flat_s[first])
        ri = float(flat_r[first])
        qi = float(flat_q[first])
        ti = float(flat_t[first])
        flag = int(flat_flags[first])
        var_level = max(_value_hint(variance_hint, first, 0.04), 1e-8)
        center_default = np.log(si) + (ri - qi - 0.5 * var_level) * ti
        center = _value_hint(x_center, first, center_default)
        width_default = float(truncation_width) * np.sqrt(max(var_level * ti, 1e-8))
        width = max(_value_hint(x_width, first, width_default), 1e-4)
        log_strikes = np.log(flat_k[idx])
        a = min(center - width, float(np.nanmin(log_strikes)) - 0.15 * width)
        b = max(center + width, float(np.nanmax(log_strikes)) + 0.15 * width)
        if b <= a:
            continue
        u = np.arange(nn, dtype=float) * np.pi / (b - a)
        phi = np.asarray(_call_custom_cf(cf, u, float(ti), float(si), float(ri), float(qi)), dtype=complex)
        if phi.size != nn:
            phi = np.broadcast_to(phi, (nn,)).astype(complex)
        base = phi * np.exp(-1j * u * a)
        d = float(b)
        for i in idx:
            ki = float(flat_k[i])
            c = max(float(np.log(ki)), a)
            if c >= d:
                call = 0.0
            else:
                chi, psi = _cos_chi_psi(u, a, c, d)
                coeff = 2.0 / (b - a) * (chi - ki * psi)
                coeff[0] *= 0.5
                call = float(np.exp(-ri * ti) * np.real(np.sum(base * coeff)))
                if not np.isfinite(call):
                    call = np.nan
                else:
                    intrinsic = max(si * np.exp(-qi * ti) - ki * np.exp(-ri * ti), 0.0)
                    call = max(call, intrinsic * 0.999, 0.0)
            if flag > 0:
                out[i] = call
            else:
                out[i] = call - si * np.exp(-qi * ti) + ki * np.exp(-ri * ti)
    return out.reshape(k.shape)


def cos_prices(
    model,
    params,
    strikes,
    tau,
    spot,
    rate,
    dividend_yield,
    *,
    n_terms: int = 256,
    truncation_width: float = 12.0,
    option_type="call",
    engine: str = "auto",
    cf=None,
    variance_hint=None,
    x_center=None,
    x_width=None,
) -> np.ndarray:
    """Price vanilla options with the COS method or a compatible Fourier fallback.

    The function supports built-in models and custom characteristic functions. When a
    custom characteristic function is supplied, it is called directly; otherwise the
    selected model parameters are dispatched to accelerated COS implementations when
    available.

    Parameters
    ----------
    model : str or identifier
        Supported model name, or ``'custom'`` when ``cf`` is provided.
    params : mapping or array-like
        Model parameters. May be ``None`` for custom characteristic functions.
    strikes : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    spot : array-like
        Spot prices.
    rate : array-like
        Continuously compounded risk-free rates.
    dividend_yield : array-like
        Continuously compounded dividend yields.
    n_terms : int, default=256
        Number of COS expansion terms.
    truncation_width : float, default=12.0
        Width parameter for the log-price integration interval.
    option_type : array-like or scalar, default='call'
        Option type labels.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.
    cf : callable, optional
        Custom characteristic-function callback.
    variance_hint : float, optional
        Variance scale used by custom-CF truncation logic.
    x_center, x_width : float, optional
        Optional custom log-price truncation center and width.

    Returns
    -------
    numpy.ndarray
        Option prices with the broadcast strike/maturity shape.

    Notes
    -----
    When the requested accelerated backend cannot support heterogeneous batch inputs,
    the function falls back to a supported numerical path unless the backend was
    explicitly requested in a way that requires failure.
    """

    if cf is not None:
        return _cos_prices_custom_cf(
            cf,
            strikes,
            tau,
            spot,
            rate,
            dividend_yield,
            n_terms=int(n_terms),
            truncation_width=float(truncation_width),
            option_type=option_type,
            variance_hint=variance_hint,
            x_center=x_center,
            x_width=x_width,
        )
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(tau, dtype=float)
    s, k, r, q, t = np.broadcast_arrays(np.asarray(spot, dtype=float), k, np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float), t)
    flags = _flag(option_type)
    if flags.size == 1 and k.size > 1:
        flags = np.full(k.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    resolved = _resolve_engine(engine)
    p = _params(model, params)
    if resolved == "cpp":
        kernels = get_cpp_kernels("COS option pricing")
        if np.unique(np.round(s.reshape(-1), 12)).size == 1 and np.unique(np.round(r.reshape(-1), 12)).size == 1 and np.unique(np.round(q.reshape(-1), 12)).size == 1 and np.unique(flags.reshape(-1)).size == 1:
            return np.asarray(
                kernels.cos_prices(
                    _model_id(model),
                    p,
                    k.reshape(-1),
                    t.reshape(-1),
                    float(s.reshape(-1)[0]),
                    float(r.reshape(-1)[0]),
                    float(q.reshape(-1)[0]),
                    int(n_terms),
                    float(truncation_width),
                    int(flags.reshape(-1)[0]),
                ),
                dtype=float,
            ).reshape(k.shape)
        if str(engine).lower() in {"cpp", "c++"}:
            raise ValueError("The C++ COS engine requires common spot, rate, dividend yield, and option type across the batch.")
        resolved = "numba"
    if resolved == "numba":
        from quantfinlab.numerics.fourier import cos_price_numba

        return cos_price_numba(
            _model_id(model),
            p,
            s.reshape(-1),
            k.reshape(-1),
            r.reshape(-1),
            q.reshape(-1),
            t.reshape(-1),
            flags.reshape(-1),
            n_terms=int(n_terms),
            truncation_width=float(truncation_width),
        ).reshape(k.shape)
    return np.asarray(direct_price(model, p, s, k, r, q, t, option_type=option_type, n=max(1024, int(n_terms)), u_max=max(120.0, float(truncation_width) * 12.0), engine="numpy"), dtype=float)


def cos_density(model, params, x_grid, spot, rate, dividend_yield, tau, *, n_terms: int = 512):
    """Approximate the risk-neutral density on a log-price grid from a model characteristic function.

    Parameters
    ----------
    model : str or identifier
        Supported model name.
    params : mapping or array-like
        Model parameters.
    x_grid : array-like
        Log-price grid on which to evaluate the density.
    spot : float
        Spot price.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuously compounded dividend yield.
    tau : float
        Time to expiry in years.
    n_terms : int, default=512
        Number of integration frequencies.

    Returns
    -------
    numpy.ndarray
        Non-negative density values normalized to integrate to one when possible.
    """

    x = np.asarray(x_grid, dtype=float)
    u = np.linspace(1e-6, 160.0, int(n_terms))
    phi = model_cf(model, u, params, spot, rate, dividend_yield, tau)
    dens = []
    for xv in x:
        dens.append(float(np.trapezoid(np.real(np.exp(-1j * u * xv) * phi), u) / np.pi))
    d = np.maximum(np.asarray(dens), 0.0)
    area = np.trapezoid(d, x)
    return d / area if area > 0 else d


def risk_neutral_density(model, params, x_grid, spot, rate, dividend_yield, tau, *, n_terms: int = 512):
    """Alias for risk-neutral density evaluation on a log-price grid.

    Parameters
    ----------
    model, params, x_grid, spot, rate, dividend_yield, tau
        Inputs forwarded to the density evaluator.
    n_terms : int, default=512
        Number of integration frequencies.

    Returns
    -------
    numpy.ndarray
        Risk-neutral density values.
    """

    return cos_density(model, params, x_grid, spot, rate, dividend_yield, tau, n_terms=n_terms)


def tail_probability(x_grid, density, threshold):
    """Integrate left-tail probability up to a threshold on a density grid.

    Parameters
    ----------
    x_grid : array-like
        Grid values.
    density : array-like
        Density values on the same grid.
    threshold : float
        Upper threshold for the left-tail event.

    Returns
    -------
    float
        Numerical integral of the density over ``x_grid <= threshold``.
    """

    x = np.asarray(x_grid, dtype=float)
    d = np.asarray(density, dtype=float)
    mask = x <= float(threshold)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(d[mask], x[mask]))


__all__ = [
    "MODEL_IDS",
    "cos_density",
    "cos_prices",
    "direct_price",
    "fft_grid",
    "fft_prices",
    "model_cf",
    "risk_neutral_density",
    "tail_probability",
]
