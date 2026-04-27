from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from quantfinlab.fixed_income import discounting

from . import bsm


def _is_call(option_type) -> np.ndarray:
    arr = np.asarray(option_type)
    return np.isin(np.char.upper(arr.astype(str)), ["C", "CE", "CALL"])


def _inputs(frame: pd.DataFrame, iv_col: str):
    option_col = "option_type" if "option_type" in frame.columns else "cp"
    option_type = frame[option_col]
    forward = frame["forward"] if "forward" in frame.columns else frame.get("f_hat")
    if forward is None:
        rate = pd.to_numeric(frame.get("rate", 0.0), errors="coerce")
        q = pd.to_numeric(frame.get("dividend_yield", 0.0), errors="coerce")
        forward = frame["spot"] * np.exp((rate - q) * frame["tau"])
    df = frame["discount_factor"] if "discount_factor" in frame.columns else frame.get("df")
    if df is None:
        df = discounting.discount_factor_from_rate(frame.get("rate", 0.0), frame["tau"])
    sigma = pd.to_numeric(frame[iv_col], errors="coerce")
    return option_type, pd.Series(forward, index=frame.index), pd.Series(df, index=frame.index), sigma


def forward_bsm_greeks_numpy(
    option_type,
    forward,
    strike,
    tau,
    sigma,
    discount_factor=1.0,
    *,
    rate=0.0,
    spot=None,
) -> pd.DataFrame:
    """Compute forward-BSM Greeks using NumPy/SciPy formulas."""
    index = None
    for value in [forward, strike, tau, sigma, discount_factor, rate, spot]:
        if isinstance(value, pd.Series):
            index = value.index
            break

    fwd, strike_arr, tau_arr, sigma_arr, df_arr, rate_arr = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(discount_factor, dtype=float),
        np.asarray(rate, dtype=float),
    )
    opt = np.asarray(option_type)
    if opt.ndim == 0:
        opt = np.full(fwd.size, opt.item(), dtype=object)
    is_call = _is_call(opt).reshape(-1)
    if len(is_call) == 1 and fwd.size > 1:
        is_call = np.full(fwd.size, bool(is_call[0]), dtype=bool)
    opt_labels = np.where(is_call, "call", "put")

    fwd = fwd.reshape(-1)
    strike_arr = strike_arr.reshape(-1)
    tau_arr = tau_arr.reshape(-1)
    sigma_arr = sigma_arr.reshape(-1)
    df_arr = df_arr.reshape(-1)
    rate_arr = rate_arr.reshape(-1)

    forward_delta = np.asarray(
        bsm.forward_bsm_delta(opt_labels, fwd, strike_arr, tau_arr, sigma_arr, df_arr),
        dtype=float,
    )
    forward_gamma = np.asarray(bsm.forward_bsm_gamma(fwd, strike_arr, tau_arr, sigma_arr, df_arr), dtype=float)
    vega = np.asarray(bsm.forward_bsm_vega(fwd, strike_arr, tau_arr, sigma_arr, df_arr), dtype=float)
    d1, d2 = bsm.d1_d2_forward(fwd, strike_arr, tau_arr, sigma_arr)
    pdf = bsm.norm_pdf(d1)
    volga = vega * d1 * d2 / np.clip(sigma_arr, 1e-12, None)
    vanna = -df_arr * pdf * d2 / np.clip(sigma_arr, 1e-12, None)
    theta = np.asarray(
        bsm.forward_bsm_theta(opt_labels, fwd, strike_arr, tau_arr, sigma_arr, df_arr, rate=rate_arr),
        dtype=float,
    )
    rho = np.asarray(bsm.forward_bsm_rho(opt_labels, fwd, strike_arr, tau_arr, sigma_arr, df_arr), dtype=float)

    delta = forward_delta.copy()
    gamma = forward_gamma.copy()
    if spot is not None:
        spot_arr = np.asarray(spot, dtype=float).reshape(-1)
        if len(spot_arr) == 1 and len(fwd) > 1:
            spot_arr = np.full(len(fwd), float(spot_arr[0]), dtype=float)
        carry_disc = df_arr * fwd / np.clip(spot_arr, 1e-12, None)
        spot_delta = np.where(is_call, carry_disc * bsm.norm_cdf(d1), carry_disc * (bsm.norm_cdf(d1) - 1.0))
        spot_gamma = carry_disc * bsm.norm_pdf(d1) / (
            np.clip(spot_arr, 1e-12, None) * sigma_arr * np.sqrt(np.clip(tau_arr, 1e-12, None))
        )
        spot_vanna = -carry_disc * pdf * d2 / np.clip(sigma_arr, 1e-12, None)
        theta_call = (
            -(spot_arr * carry_disc * pdf * sigma_arr) / (2.0 * np.sqrt(np.clip(tau_arr, 1e-12, None)))
            - rate_arr * strike_arr * df_arr * bsm.norm_cdf(d2)
            + (rate_arr - np.log(np.clip(fwd, 1e-12, None) / np.clip(spot_arr, 1e-12, None)) / np.clip(tau_arr, 1e-12, None))
            * spot_arr
            * carry_disc
            * bsm.norm_cdf(d1)
        )
        theta_put = (
            -(spot_arr * carry_disc * pdf * sigma_arr) / (2.0 * np.sqrt(np.clip(tau_arr, 1e-12, None)))
            + rate_arr * strike_arr * df_arr * bsm.norm_cdf(-d2)
            - (rate_arr - np.log(np.clip(fwd, 1e-12, None) / np.clip(spot_arr, 1e-12, None)) / np.clip(tau_arr, 1e-12, None))
            * spot_arr
            * carry_disc
            * bsm.norm_cdf(-d1)
        )
        spot_theta = np.where(is_call, theta_call, theta_put)
        spot_rho = np.where(
            is_call,
            strike_arr * tau_arr * df_arr * bsm.norm_cdf(d2),
            -strike_arr * tau_arr * df_arr * bsm.norm_cdf(-d2),
        )
        delta = np.where(np.isfinite(spot_delta), spot_delta, delta)
        gamma = np.where(np.isfinite(spot_gamma), spot_gamma, gamma)
        vanna = np.where(np.isfinite(spot_vanna), spot_vanna, vanna)
        theta = np.where(np.isfinite(spot_theta), spot_theta, theta)
        rho = np.where(np.isfinite(spot_rho), spot_rho, rho)

    out = pd.DataFrame(
        {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "volga": volga,
            "vanna": vanna,
            "theta": theta,
            "rho": rho,
            "forward_delta": forward_delta,
            "forward_gamma": forward_gamma,
        },
        index=index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def compute_greeks_numpy(
    quotes: pd.DataFrame,
    iv_col: str = "iv_mid",
    price_model: str = "forward_bsm",
) -> pd.DataFrame:
    """Compute forward-BSM Greeks with the NumPy engine."""
    if price_model != "forward_bsm":
        raise ValueError("Only price_model='forward_bsm' is supported.")
    if iv_col not in quotes.columns:
        raise ValueError(f"{iv_col!r} is required.")

    out = quotes.copy()
    if "source_index" not in out.columns:
        out["source_index"] = out.index
    option_type, forward, df, sigma = _inputs(out, iv_col)
    strike = pd.to_numeric(out["strike"], errors="coerce")
    tau = pd.to_numeric(out["tau"], errors="coerce")
    rate = pd.to_numeric(out["rate"], errors="coerce") if "rate" in out.columns else 0.0
    spot = pd.to_numeric(out["spot"], errors="coerce") if "spot" in out.columns else None

    greeks = forward_bsm_greeks_numpy(
        option_type,
        forward,
        strike,
        tau,
        sigma,
        df,
        rate=rate,
        spot=spot,
    )
    suffix = iv_col.replace("iv_", "") if iv_col.startswith("iv_") else iv_col
    for col in greeks.columns:
        out[col] = greeks[col]
        out[f"{col}_{suffix}"] = greeks[col]
        out[f"{col}_numpy"] = greeks[col]
    out["delta_spot_mid"] = out.get("delta_mid", out["delta"])
    out["gamma_spot_mid"] = out.get("gamma_mid", out["gamma"])
    out.attrs["greek_engine"] = "numpy"
    return out


def _jax_modules(strict: bool = False):
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        if strict:
            raise ImportError("JAX Greeks requested but JAX is not available.") from exc
        return None, None, exc
    return jax, jnp, None


def forward_bsm_price_jax(option_type, forward, strike, tau, sigma, rate=0.0, *, strict: bool = True):
    """JAX implementation of the forward-BSM price."""
    jax, jnp, exc = _jax_modules(strict=strict)
    if exc is not None:
        return None

    is_call = jnp.asarray(_is_call(option_type), dtype=bool)
    fwd = jnp.asarray(forward, dtype=float)
    strike_arr = jnp.asarray(strike, dtype=float)
    tau_arr = jnp.maximum(jnp.asarray(tau, dtype=float), 1e-12)
    sigma_arr = jnp.maximum(jnp.asarray(sigma, dtype=float), 1e-12)
    rate_arr = jnp.asarray(rate, dtype=float)
    df = jnp.exp(-rate_arr * tau_arr)
    sqrt_tau = jnp.sqrt(tau_arr)
    d1 = (jnp.log(jnp.maximum(fwd, 1e-300) / jnp.maximum(strike_arr, 1e-300)) + 0.5 * sigma_arr * sigma_arr * tau_arr) / (
        sigma_arr * sqrt_tau
    )
    d2 = d1 - sigma_arr * sqrt_tau
    def ncdf(x):
        return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))

    call = df * (fwd * ncdf(d1) - strike_arr * ncdf(d2))
    put = df * (strike_arr * ncdf(-d2) - fwd * ncdf(-d1))
    return jnp.where(is_call, call, put)


def forward_bsm_greeks_jax(
    option_type,
    forward,
    strike,
    tau,
    sigma,
    *,
    rate=0.0,
    strict: bool = False,
) -> pd.DataFrame:
    """Compute forward-BSM Greeks with JAX autodiff."""
    jax, jnp, exc = _jax_modules(strict=strict)
    if exc is not None:
        out = pd.DataFrame(index=getattr(forward, "index", None))
        out.attrs["diagnostic"] = "jax_unavailable"
        out.attrs["message"] = str(exc)
        return out

    index = getattr(forward, "index", None)
    opt = np.asarray(option_type)
    fwd = np.asarray(forward, dtype=float).reshape(-1)
    strike_arr = np.asarray(strike, dtype=float).reshape(-1)
    tau_arr = np.asarray(tau, dtype=float).reshape(-1)
    sigma_arr = np.asarray(sigma, dtype=float).reshape(-1)
    rate_arr = np.asarray(rate, dtype=float).reshape(-1)
    arrays = np.broadcast_arrays(fwd, strike_arr, tau_arr, sigma_arr, rate_arr)
    fwd, strike_arr, tau_arr, sigma_arr, rate_arr = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    if opt.ndim == 0:
        opt = np.full(len(fwd), opt.item(), dtype=object)
    opt_bool = _is_call(opt).reshape(-1)
    if len(opt_bool) == 1 and len(fwd) > 1:
        opt_bool = np.full(len(fwd), bool(opt_bool[0]), dtype=bool)

    def scalar_price(f, k, t, sig, r, is_call):
        df = jnp.exp(-r * t)
        sqrt_t = jnp.sqrt(jnp.maximum(t, 1e-12))
        sig = jnp.maximum(sig, 1e-12)
        d1 = (jnp.log(jnp.maximum(f, 1e-300) / jnp.maximum(k, 1e-300)) + 0.5 * sig * sig * t) / (
            sig * sqrt_t
        )
        d2 = d1 - sig * sqrt_t
        def ncdf(x):
            return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))

        call = df * (f * ncdf(d1) - k * ncdf(d2))
        put = df * (k * ncdf(-d2) - f * ncdf(-d1))
        return jnp.where(is_call, call, put)

    delta_fn = jax.grad(scalar_price, argnums=0)
    gamma_fn = jax.grad(delta_fn, argnums=0)
    vega_fn = jax.grad(scalar_price, argnums=3)
    volga_fn = jax.grad(vega_fn, argnums=3)
    vanna_fn = jax.grad(delta_fn, argnums=3)
    tau_fn = jax.grad(scalar_price, argnums=2)
    rho_fn = jax.grad(scalar_price, argnums=4)
    vmapped = jax.vmap(
        lambda f, k, t, sig, r, c: (
            delta_fn(f, k, t, sig, r, c),
            gamma_fn(f, k, t, sig, r, c),
            vega_fn(f, k, t, sig, r, c),
            volga_fn(f, k, t, sig, r, c),
            vanna_fn(f, k, t, sig, r, c),
            -tau_fn(f, k, t, sig, r, c),
            rho_fn(f, k, t, sig, r, c),
        )
    )
    vals = vmapped(
        jnp.asarray(fwd),
        jnp.asarray(strike_arr),
        jnp.asarray(tau_arr),
        jnp.asarray(sigma_arr),
        jnp.asarray(rate_arr),
        jnp.asarray(opt_bool),
    )
    out = pd.DataFrame(
        {
            "delta": np.asarray(vals[0], dtype=float),
            "gamma": np.asarray(vals[1], dtype=float),
            "vega": np.asarray(vals[2], dtype=float),
            "volga": np.asarray(vals[3], dtype=float),
            "vanna": np.asarray(vals[4], dtype=float),
            "theta": np.asarray(vals[5], dtype=float),
            "rho": np.asarray(vals[6], dtype=float),
        },
        index=index,
    )
    out["forward_delta"] = out["delta"]
    out["forward_gamma"] = out["gamma"]
    out.attrs["greek_engine"] = "jax"
    return out.replace([np.inf, -np.inf], np.nan)


def _spot_bsm_greeks_jax(
    option_type,
    spot,
    strike,
    tau,
    rate,
    div,
    sigma,
    *,
    forward=None,
    index=None,
    strict: bool = False,
) -> pd.DataFrame:
    jax, jnp, exc = _jax_modules(strict=strict)
    if exc is not None:
        out = pd.DataFrame(index=index)
        out.attrs["diagnostic"] = "jax_unavailable"
        out.attrs["message"] = str(exc)
        return out

    opt = np.asarray(option_type)
    spot_arr, strike_arr, tau_arr, rate_arr, div_arr, sigma_arr = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(div, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    spot_arr, strike_arr, tau_arr, rate_arr, div_arr, sigma_arr = [
        np.asarray(a, dtype=float).reshape(-1) for a in [spot_arr, strike_arr, tau_arr, rate_arr, div_arr, sigma_arr]
    ]
    if opt.ndim == 0:
        opt = np.full(len(spot_arr), opt.item(), dtype=object)
    opt_bool = _is_call(opt).reshape(-1)
    if len(opt_bool) == 1 and len(spot_arr) > 1:
        opt_bool = np.full(len(spot_arr), bool(opt_bool[0]), dtype=bool)

    def scalar_price(s, k, t, r, q, sig, is_call):
        t = jnp.maximum(t, 1e-12)
        sig = jnp.maximum(sig, 1e-12)
        sqrt_t = jnp.sqrt(t)
        disc_r = jnp.exp(-r * t)
        disc_q = jnp.exp(-q * t)
        d1 = (jnp.log(jnp.maximum(s, 1e-300) / jnp.maximum(k, 1e-300)) + (r - q + 0.5 * sig * sig) * t) / (
            sig * sqrt_t
        )
        d2 = d1 - sig * sqrt_t

        def ncdf(x):
            return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0)))

        call = disc_q * s * ncdf(d1) - disc_r * k * ncdf(d2)
        put = disc_r * k * ncdf(-d2) - disc_q * s * ncdf(-d1)
        return jnp.where(is_call, call, put)

    delta_fn = jax.grad(scalar_price, argnums=0)
    gamma_fn = jax.grad(delta_fn, argnums=0)
    vega_fn = jax.grad(scalar_price, argnums=5)
    volga_fn = jax.grad(vega_fn, argnums=5)
    vanna_fn = jax.grad(delta_fn, argnums=5)

    def theta_fn(s, k, t, r, q, sig, c):
        return -jax.grad(scalar_price, argnums=2)(s, k, t, r, q, sig, c)

    rho_fn = jax.grad(scalar_price, argnums=3)
    vmapped = jax.vmap(
        lambda s, k, t, r, q, sig, c: (
            delta_fn(s, k, t, r, q, sig, c),
            gamma_fn(s, k, t, r, q, sig, c),
            vega_fn(s, k, t, r, q, sig, c),
            volga_fn(s, k, t, r, q, sig, c),
            vanna_fn(s, k, t, r, q, sig, c),
            theta_fn(s, k, t, r, q, sig, c),
            rho_fn(s, k, t, r, q, sig, c),
        )
    )
    vals = vmapped(
        jnp.asarray(spot_arr),
        jnp.asarray(strike_arr),
        jnp.asarray(tau_arr),
        jnp.asarray(rate_arr),
        jnp.asarray(div_arr),
        jnp.asarray(sigma_arr),
        jnp.asarray(opt_bool),
    )
    out = pd.DataFrame(
        {
            "delta": np.asarray(vals[0], dtype=float),
            "gamma": np.asarray(vals[1], dtype=float),
            "vega": np.asarray(vals[2], dtype=float),
            "volga": np.asarray(vals[3], dtype=float),
            "vanna": np.asarray(vals[4], dtype=float),
            "theta": np.asarray(vals[5], dtype=float),
            "rho": np.asarray(vals[6], dtype=float),
        },
        index=index,
    )
    if forward is not None:
        fwd = np.asarray(forward, dtype=float).reshape(-1)
        if len(fwd) == 1 and len(out) > 1:
            fwd = np.full(len(out), float(fwd[0]), dtype=float)
        scale = spot_arr / np.clip(fwd, 1e-12, None)
        out["forward_delta"] = out["delta"] * scale
        out["forward_gamma"] = out["gamma"] * scale * scale
    else:
        out["forward_delta"] = np.nan
        out["forward_gamma"] = np.nan
    out.attrs["greek_engine"] = "jax"
    return out.replace([np.inf, -np.inf], np.nan)


def compute_greeks_jax(
    quotes: pd.DataFrame,
    iv_col: str = "iv_mid",
    price_model: str = "forward_bsm",
    on_missing: str = "warn",
    strict: bool = False,
) -> pd.DataFrame:
    """Compute Greeks with JAX when available, otherwise return diagnostics."""
    if price_model != "forward_bsm":
        raise ValueError("Only price_model='forward_bsm' is supported.")
    out = quotes.copy()
    if "source_index" not in out.columns:
        out["source_index"] = out.index
    if iv_col not in out.columns:
        raise ValueError(f"{iv_col!r} is required.")

    option_type, forward, df, sigma = _inputs(out, iv_col)
    tau = pd.to_numeric(out["tau"], errors="coerce")
    strike = pd.to_numeric(out["strike"], errors="coerce")
    if "rate" in out.columns:
        rate = pd.to_numeric(out["rate"], errors="coerce")
    else:
        rate = discounting.continuous_rate_from_discount_factor(df, tau)

    if "spot" in out.columns:
        spot = pd.to_numeric(out["spot"], errors="coerce")
        div = rate - np.log(pd.to_numeric(forward, errors="coerce") / spot.clip(lower=1e-12)) / tau.clip(lower=1e-12)
        greeks = _spot_bsm_greeks_jax(
            option_type,
            spot,
            strike,
            tau,
            rate,
            div,
            sigma,
            forward=forward,
            index=out.index,
            strict=strict,
        )
    else:
        greeks = forward_bsm_greeks_jax(option_type, forward, strike, tau, sigma, rate=rate, strict=strict)
    if greeks.empty and greeks.attrs.get("diagnostic") == "jax_unavailable":
        message = greeks.attrs.get("message", "JAX unavailable")
        if on_missing == "warn":
            warnings.warn(f"Skipping JAX Greeks: {message}", RuntimeWarning, stacklevel=2)
        for greek in ["delta", "gamma", "vega", "volga", "vanna", "theta", "rho", "forward_delta", "forward_gamma"]:
            out[greek] = np.nan
            out[f"{greek}_jax"] = np.nan
        out.attrs["greek_engine"] = "jax_unavailable"
        out.attrs["diagnostic"] = "jax_unavailable"
        out.attrs["message"] = message
        return out

    for col in greeks.columns:
        out[col] = greeks[col]
        out[f"{col}_jax"] = greeks[col]
    out.attrs["greek_engine"] = "jax"
    return out


def compare_numpy_jax_greeks(
    greeks_numpy: pd.DataFrame,
    greeks_jax: pd.DataFrame,
    greek_cols: tuple[str, ...] = ("delta", "gamma", "vega", "volga", "vanna", "theta", "rho"),
    greek_bands: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Compare NumPy and JAX Greek tables with absolute and relative errors."""
    if greeks_jax.attrs.get("diagnostic") == "jax_unavailable":
        summary = pd.DataFrame(
            [
                {
                    "diagnostic": "jax_unavailable",
                    "message": greeks_jax.attrs.get("message", ""),
                    "n": 0,
                }
            ]
        )
        return {"comparison": pd.DataFrame(), "summary": summary}

    common = greeks_numpy.index.intersection(greeks_jax.index)
    rows = []
    comp = pd.DataFrame(index=common)
    meta_cols = [
        "date",
        "trade_date",
        "expiry",
        "expiry_datetime",
        "strike",
        "option_type",
        "cp",
        "dte",
        "dte_days",
        "moneyness",
        "log_moneyness",
        "lm_f",
        "spot",
        "forward",
        "f_hat",
    ]
    for col in meta_cols:
        if col in greeks_numpy.columns:
            comp[col] = greeks_numpy.loc[common, col]
    for greek in greek_cols:
        np_col = f"{greek}_numpy" if f"{greek}_numpy" in greeks_numpy.columns else greek
        jax_col = f"{greek}_jax" if f"{greek}_jax" in greeks_jax.columns else greek
        if np_col not in greeks_numpy.columns or jax_col not in greeks_jax.columns:
            continue
        np_vals = pd.to_numeric(greeks_numpy.loc[common, np_col], errors="coerce")
        jax_vals = pd.to_numeric(greeks_jax.loc[common, jax_col], errors="coerce")
        abs_err = (np_vals - jax_vals).abs()
        rel_err = abs_err / np_vals.abs().replace(0, np.nan)
        comp[f"{greek}_numpy"] = np_vals
        comp[f"{greek}_jax"] = jax_vals
        comp[f"{greek}_abs_error"] = abs_err
        comp[f"{greek}_rel_error"] = rel_err
        rows.append(
            {
                "greek": greek,
                "n": int(abs_err.notna().sum()),
                "mae": float(np.nanmean(abs_err)) if abs_err.notna().any() else np.nan,
                "median_abs_error": float(np.nanmedian(abs_err)) if abs_err.notna().any() else np.nan,
                "max_abs_error": float(np.nanmax(abs_err)) if abs_err.notna().any() else np.nan,
                "median_rel_error": float(np.nanmedian(rel_err)) if rel_err.notna().any() else np.nan,
            }
        )
    out = {"comparison": comp, "summary": pd.DataFrame(rows)}
    if greek_bands is not None:
        out["bands"] = greek_bands
    return out


def compute_greek_bands_from_iv_band(
    iv_table: pd.DataFrame,
    iv_low_col: str = "iv_bid",
    iv_mid_col: str = "iv_mid",
    iv_high_col: str = "iv_ask",
    price_model: str = "forward_bsm",
) -> pd.DataFrame:
    """Recompute Greeks across bid/mid/ask IV to show quote-driven uncertainty."""
    if price_model != "forward_bsm":
        raise ValueError("Only price_model='forward_bsm' is supported.")
    out = iv_table.copy()
    if iv_mid_col not in out.columns:
        raise ValueError(f"{iv_mid_col!r} is required.")
    for col in [iv_low_col, iv_high_col]:
        if col not in out.columns:
            out[col] = out[iv_mid_col]
    out[iv_low_col] = pd.to_numeric(out[iv_low_col], errors="coerce").combine_first(out[iv_mid_col])
    out[iv_high_col] = pd.to_numeric(out[iv_high_col], errors="coerce").combine_first(out[iv_mid_col])
    out[iv_low_col] = np.minimum(out[iv_low_col], out[iv_mid_col])
    out[iv_high_col] = np.maximum(out[iv_high_col], out[iv_mid_col])

    low = compute_greeks_numpy(out, iv_col=iv_low_col, price_model=price_model)
    mid = compute_greeks_numpy(out, iv_col=iv_mid_col, price_model=price_model)
    high = compute_greeks_numpy(out, iv_col=iv_high_col, price_model=price_model)
    for greek in ["delta", "gamma", "vega", "volga", "vanna", "theta", "rho"]:
        vals = pd.concat([low[greek], mid[greek], high[greek]], axis=1)
        out[f"{greek}_low"] = vals.min(axis=1)
        out[f"{greek}_mid"] = mid[greek]
        out[f"{greek}_high"] = vals.max(axis=1)
        out[f"{greek}_band"] = out[f"{greek}_high"] - out[f"{greek}_low"]
    return out


def compute_greeks(
    quotes: pd.DataFrame,
    iv_col: str = "iv_mid",
    method: str = "analytic",
    price_model: str = "forward_bsm",
) -> pd.DataFrame:
    """Compatibility wrapper for the old analytic API."""
    if method not in {"analytic", "numpy"}:
        raise ValueError("method must be 'analytic' or 'numpy'.")
    return compute_greeks_numpy(quotes, iv_col=iv_col, price_model=price_model)


def compare_analytic_vs_jax_greeks(
    quotes: pd.DataFrame,
    iv_col: str = "iv_mid",
) -> pd.DataFrame:
    """Compatibility wrapper returning the summary table only."""
    sample = quotes.dropna(subset=[iv_col, "strike", "tau"]).head(250).copy()
    if sample.empty:
        return pd.DataFrame([{"diagnostic": "no_finite_rows", "n": 0}])
    np_greeks = compute_greeks_numpy(sample, iv_col=iv_col)
    jax_greeks = compute_greeks_jax(sample, iv_col=iv_col, on_missing="ignore")
    return compare_numpy_jax_greeks(np_greeks, jax_greeks)["summary"]


def greek_summary_table(greek_table: pd.DataFrame, greek_bands: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    source = greek_bands if greek_bands is not None else greek_table
    for greek in ["delta", "gamma", "vega", "volga", "vanna", "theta", "rho"]:
        col = greek if greek in greek_table.columns else f"{greek}_mid"
        vals = pd.to_numeric(greek_table.get(col), errors="coerce")
        rows.append(
            {
                "greek": greek,
                "n_finite": int(np.isfinite(vals).sum()),
                "median": float(np.nanmedian(vals)) if vals.notna().any() else np.nan,
                "median_band": float(np.nanmedian(pd.to_numeric(source.get(f"{greek}_band"), errors="coerce")))
                if f"{greek}_band" in source.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


black76_delta = bsm.forward_bsm_delta
black76_gamma = bsm.forward_bsm_gamma
black76_vega = bsm.forward_bsm_vega
black76_theta = bsm.forward_bsm_theta
black76_rho = bsm.forward_bsm_rho

__all__ = [
    "black76_delta",
    "black76_gamma",
    "black76_rho",
    "black76_theta",
    "black76_vega",
    "compare_analytic_vs_jax_greeks",
    "compare_numpy_jax_greeks",
    "compute_greek_bands_from_iv_band",
    "compute_greeks",
    "compute_greeks_jax",
    "compute_greeks_numpy",
    "forward_bsm_greeks_jax",
    "forward_bsm_greeks_numpy",
    "forward_bsm_price_jax",
    "greek_summary_table",
]
