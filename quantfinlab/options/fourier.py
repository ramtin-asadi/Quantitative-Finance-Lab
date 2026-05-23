from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.bsm import bsm_cf
from quantfinlab.options.merton import merton_cf
from quantfinlab.options.heston import heston_cf
from quantfinlab.options.bates import bates_cf
from quantfinlab.options.variance_gamma import vg_cf


MODEL_IDS = {"bsm": 0, "merton": 1, "vg": 2, "variance_gamma": 2, "heston": 3, "bates": 4}


def _resolve_engine(engine: str) -> str:
    key = str(engine).lower()
    if key == "auto":
        try:
            import numba  # noqa: F401

            return "numba"
        except Exception:
            return "numpy"
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
    engine: str = "numba",
):
    s, k, r, q, t = np.broadcast_arrays(np.asarray(spot, dtype=float), np.asarray(strike, dtype=float), np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float), np.asarray(tau, dtype=float))
    flags = _flag(option_type)
    if flags.size == 1 and k.size > 1:
        flags = np.full(k.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    resolved = _resolve_engine(engine)
    p = _params(model, params)
    if resolved == "cpp":
        from quantfinlab import _kernels

        flat_s = s.reshape(-1)
        flat_r = r.reshape(-1)
        flat_q = q.reshape(-1)
        flat_flags = flags.reshape(-1)
        if (
            np.unique(np.round(flat_s, 12)).size == 1
            and np.unique(np.round(flat_r, 12)).size == 1
            and np.unique(np.round(flat_q, 12)).size == 1
            and np.unique(flat_flags).size == 1
            and hasattr(_kernels, "direct_prices")
        ):
            return np.asarray(
                _kernels.direct_prices(
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
        resolved = "numba"
    if resolved == "numba":
        try:
            from quantfinlab.numerics.fourier import direct_price_numba

            return direct_price_numba(_model_id(model), p, s.reshape(-1), k.reshape(-1), r.reshape(-1), q.reshape(-1), t.reshape(-1), flags.reshape(-1), n=n, u_max=u_max).reshape(k.shape)
        except Exception:
            if str(engine).lower() == "numba":
                raise
    out = np.empty(k.size, dtype=float)
    for i, vals in enumerate(zip(s.reshape(-1), k.reshape(-1), r.reshape(-1), q.reshape(-1), t.reshape(-1), flags.reshape(-1))):
        call = _direct_call_numpy(model, p, vals[0], vals[1], vals[2], vals[3], vals[4], n=n, u_max=u_max)
        out[i] = call if vals[5] > 0 else call - vals[0] * np.exp(-vals[3] * vals[4]) + vals[1] * np.exp(-vals[2] * vals[4])
    return out.reshape(k.shape)


def fft_grid(model, params, spot, rate, dividend_yield, tau, *, alpha: float = 1.5, n: int = 256, eta: float = 0.25, option_type="call", engine: str = "numba") -> pd.DataFrame:
    resolved = _resolve_engine(engine)
    p = _params(model, params)
    flag = int(_flag(option_type).reshape(-1)[0])
    if resolved == "cpp":
        from quantfinlab import _kernels

        out = _kernels.fft_prices(_model_id(model), p, float(spot), float(rate), float(dividend_yield), float(tau), float(alpha), int(n), float(eta), flag)
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
    return fft_grid(*args, **kwargs)


def cos_prices(model, params, strikes, tau, spot, rate, dividend_yield, *, n_terms: int = 256, truncation_width: float = 12.0, option_type="call", engine: str = "numba") -> np.ndarray:
    k = np.asarray(strikes, dtype=float)
    t = np.asarray(tau, dtype=float)
    s, k, r, q, t = np.broadcast_arrays(np.asarray(spot, dtype=float), k, np.asarray(rate, dtype=float), np.asarray(dividend_yield, dtype=float), t)
    flags = _flag(option_type)
    if flags.size == 1 and k.size > 1:
        flags = np.full(k.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    resolved = _resolve_engine(engine)
    p = _params(model, params)
    if resolved == "cpp":
        from quantfinlab import _kernels

        if np.unique(np.round(s.reshape(-1), 12)).size == 1 and np.unique(np.round(r.reshape(-1), 12)).size == 1 and np.unique(np.round(q.reshape(-1), 12)).size == 1 and np.unique(flags.reshape(-1)).size == 1:
            return np.asarray(
                _kernels.cos_prices(
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
    return cos_density(model, params, x_grid, spot, rate, dividend_yield, tau, n_terms=n_terms)


def tail_probability(x_grid, density, threshold):
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
