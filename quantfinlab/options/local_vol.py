from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .bsm import black76_price
from .surface import surface_iv, surface_iv_jax

_dupire_eval_grid_jax_cached = None


def _jax_available() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401

        return True
    except Exception:
        return False


def curve_by_tau(q, value_col, tau_values, fallback=np.nan):
    """Interpolate a median-by-maturity curve onto requested maturity values.

    Parameters
    ----------
    q : pandas.DataFrame
        Table containing a ``tau`` column and the requested value column.
    value_col : str
        Column to aggregate by maturity and interpolate.
    tau_values : array-like
        Target maturities in years.
    fallback : float, default=nan
        Fill value returned when no usable observations are available.

    Returns
    -------
    numpy.ndarray
        Interpolated values with the same shape as ``tau_values``.
    """

    tau_values = np.asarray(tau_values, dtype=float)
    out = np.full(tau_values.shape, float(fallback), dtype=float)
    if q is None or len(q) == 0 or value_col not in q.columns or "tau" not in q.columns:
        return out
    z = q[["tau", value_col]].copy()
    z["tau"] = pd.to_numeric(z["tau"], errors="coerce")
    z[value_col] = pd.to_numeric(z[value_col], errors="coerce")
    z = z[np.isfinite(z["tau"]) & np.isfinite(z[value_col]) & (z["tau"] > 0)].copy()
    if z.empty:
        return out
    g = z.groupby("tau", as_index=False)[value_col].median().sort_values("tau")
    x = g["tau"].to_numpy(dtype=float)
    y = g[value_col].to_numpy(dtype=float)
    if len(x) == 1:
        return np.full(tau_values.shape, float(y[0]), dtype=float)
    return np.interp(tau_values, x, y, left=y[0], right=y[-1])


def rate_by_tau(q, tau_values, rate_col: str = "rate"):
    """Interpolate median continuous rates by maturity and fill missing values robustly.

    Parameters
    ----------
    q : pandas.DataFrame
        Quote table containing maturity and rate columns.
    tau_values : array-like
        Target maturities in years.
    rate_col : str, default='rate'
        Rate column to interpolate.

    Returns
    -------
    numpy.ndarray
        Rate values aligned to ``tau_values``. Missing curves are filled with zeros.
    """

    r = curve_by_tau(q, rate_col, tau_values, fallback=np.nan)
    if np.isfinite(r).any():
        fill = float(np.nanmedian(r[np.isfinite(r)]))
        return np.where(np.isfinite(r), r, fill)
    return np.zeros_like(np.asarray(tau_values, dtype=float))


def carry_by_tau(q, tau_values, spot_value=None, carry_col: str = "implied_carry", forward_col: str = "forward"):
    """Interpolate implied carry by maturity or infer it from forwards and spot.

    Parameters
    ----------
    q : pandas.DataFrame
        Quote table containing maturity and carry or forward information.
    tau_values : array-like
        Target maturities in years.
    spot_value : float, optional
        Spot price used to infer carry from forwards when carry is unavailable.
    carry_col : str, default='implied_carry'
        Existing carry column.
    forward_col : str, default='forward'
        Forward column used for fallback carry inference.

    Returns
    -------
    numpy.ndarray
        Continuous carry values aligned to ``tau_values``. Missing curves are filled
        with zeros.
    """

    tau_values = np.asarray(tau_values, dtype=float)
    b = curve_by_tau(q, carry_col, tau_values, fallback=np.nan)
    if np.isfinite(b).any():
        fill = float(np.nanmedian(b[np.isfinite(b)]))
        return np.where(np.isfinite(b), b, fill)
    if q is not None and len(q) and forward_col in q.columns:
        if spot_value is None:
            spot_value = float(np.nanmedian(pd.to_numeric(q.get("spot"), errors="coerce")))
        z = q[["tau", forward_col]].copy()
        z["tau"] = pd.to_numeric(z["tau"], errors="coerce")
        z[forward_col] = pd.to_numeric(z[forward_col], errors="coerce")
        z = z[np.isfinite(z["tau"]) & np.isfinite(z[forward_col]) & (z["tau"] > 0) & (z[forward_col] > 0)].copy()
        if not z.empty and np.isfinite(spot_value) and spot_value > 0:
            z["carry"] = np.log(z[forward_col] / float(spot_value)) / z["tau"]
            b = curve_by_tau(z.rename(columns={"carry": carry_col}), carry_col, tau_values, fallback=np.nan)
            if np.isfinite(b).any():
                fill = float(np.nanmedian(b[np.isfinite(b)]))
                return np.where(np.isfinite(b), b, fill)
    return np.zeros_like(tau_values)


def dividend_yield_by_tau(q, tau_values, spot_value=None, rate_col: str = "rate", carry_col: str = "implied_carry"):
    """Infer a maturity-aligned dividend-yield curve as rate minus carry.

    Parameters
    ----------
    q : pandas.DataFrame
        Quote table containing rate and carry or forward information.
    tau_values : array-like
        Target maturities in years.
    spot_value : float, optional
        Spot price used when carry must be inferred from forwards.
    rate_col : str, default='rate'
        Rate column.
    carry_col : str, default='implied_carry'
        Carry column.

    Returns
    -------
    numpy.ndarray
        Continuous dividend-yield values aligned to ``tau_values``.
    """

    return rate_by_tau(q, tau_values, rate_col=rate_col) - carry_by_tau(q, tau_values, spot_value=spot_value, carry_col=carry_col)


def _support_mask(q, model_k, tau_grid, k_col="k", tau_col="tau", k_quantiles=(0.01, 0.99)):
    if q is None or len(q) == 0:
        return np.isfinite(model_k)
    k = pd.to_numeric(q[k_col], errors="coerce")
    tau = pd.to_numeric(q[tau_col], errors="coerce")
    ok = np.isfinite(k) & np.isfinite(tau)
    if not ok.any():
        return np.isfinite(model_k)
    k_lo, k_hi = np.nanquantile(k[ok], k_quantiles)
    tau_lo, tau_hi = np.nanmin(tau[ok]), np.nanmax(tau[ok])
    return (model_k >= k_lo) & (model_k <= k_hi) & (tau_grid[:, None] >= tau_lo) & (tau_grid[:, None] <= tau_hi)


def _base_grid(q, k_min, k_max, tau_min, tau_max, n_k, n_tau, annualization_days):
    q = q.copy()
    tau = np.linspace(float(tau_min), float(tau_max), int(n_tau))
    k_spot = np.linspace(float(k_min), float(k_max), int(n_k))
    spot = float(np.nanmedian(pd.to_numeric(q["spot"], errors="coerce")))
    strike = spot * np.exp(k_spot)
    rate = rate_by_tau(q, tau)
    carry = carry_by_tau(q, tau, spot_value=spot)
    div = rate - carry
    forward = spot * np.exp(carry * tau)
    return {
        "tau": tau,
        "tau_days": tau * float(annualization_days),
        "k_spot": k_spot,
        "strike": strike,
        "spot": spot,
        "rate": rate,
        "carry": carry,
        "dividend_yield": div,
        "forward": forward,
    }


def _jax_fit_arrays(fit: dict) -> dict:
    import jax.numpy as jnp

    return {
        "coef": jnp.asarray(fit["coef"]),
        "knots_x": jnp.asarray(fit.get("knots_x", fit.get("knots_k"))),
        "knots_y": jnp.asarray(fit.get("knots_y", fit.get("knots_tau"))),
        "center_x": jnp.asarray(fit.get("center_x", fit.get("center_k", fit.get("k_center")))),
        "scale_x": jnp.asarray(fit.get("scale_x", fit.get("scale_k", fit.get("k_scale")))),
        "center_y": jnp.asarray(fit.get("center_y", fit.get("center_tau", fit.get("tau_center")))),
        "scale_y": jnp.asarray(fit.get("scale_y", fit.get("scale_tau", fit.get("tau_scale")))),
    }


def _dupire_eval_grid_jax_fn():
    global _dupire_eval_grid_jax_cached
    if _dupire_eval_grid_jax_cached is not None:
        return _dupire_eval_grid_jax_cached

    import jax
    import jax.numpy as jnp

    def eval_grid(fit_j, tau_j, strike_j, rate_j, carry_j, spot_j):
        def r_of_t(t):
            return jnp.interp(t, tau_j, rate_j)

        def b_of_t(t):
            return jnp.interp(t, tau_j, carry_j)

        def call_price(k_value, t_value):
            t_value = jnp.maximum(t_value, 1e-10)
            r_value = r_of_t(t_value)
            b_value = b_of_t(t_value)
            forward_value = spot_j * jnp.exp(b_value * t_value)
            sigma_value = surface_iv_jax(fit_j, jnp.log(k_value / jnp.maximum(forward_value, 1e-300)), t_value)
            sqrt_t = jnp.sqrt(t_value)
            d1 = (
                jnp.log(jnp.maximum(forward_value, 1e-300) / jnp.maximum(k_value, 1e-300))
                + 0.5 * sigma_value**2 * t_value
            ) / (jnp.maximum(sigma_value, 1e-8) * sqrt_t)
            d2 = d1 - sigma_value * sqrt_t
            cdf1 = 0.5 * (1.0 + jax.lax.erf(d1 / jnp.sqrt(2.0)))
            cdf2 = 0.5 * (1.0 + jax.lax.erf(d2 / jnp.sqrt(2.0)))
            return jnp.exp(-r_value * t_value) * (forward_value * cdf1 - k_value * cdf2)

        c_t = jax.grad(call_price, argnums=1)
        c_k = jax.grad(call_price, argnums=0)
        c_kk = jax.grad(c_k, argnums=0)
        tt, kk = jnp.meshgrid(tau_j, strike_j, indexing="ij")
        flat_k = kk.reshape(-1)
        flat_t = tt.reshape(-1)

        def one(k_value, t_value):
            r_value = r_of_t(t_value)
            b_value = b_of_t(t_value)
            q_value = r_value - b_value
            c_value = call_price(k_value, t_value)
            c_t_value = c_t(k_value, t_value)
            c_k_value = c_k(k_value, t_value)
            c_kk_value = c_kk(k_value, t_value)
            numerator_value = c_t_value + q_value * c_value + (r_value - q_value) * k_value * c_k_value
            denominator_value = 0.5 * k_value**2 * c_kk_value
            local_var_value = numerator_value / denominator_value
            forward_value = spot_j * jnp.exp(b_value * t_value)
            k_model = jnp.log(k_value / jnp.maximum(forward_value, 1e-300))
            sigma_value = surface_iv_jax(fit_j, k_model, t_value)
            return c_value, sigma_value, k_model, local_var_value, numerator_value, denominator_value

        vals = jax.vmap(one)(flat_k, flat_t)
        return [x.reshape(tt.shape) for x in vals]

    _dupire_eval_grid_jax_cached = jax.jit(eval_grid)
    return _dupire_eval_grid_jax_cached


def _finish_grid(
    *,
    base,
    q,
    model_k,
    iv,
    price,
    local_var,
    numerator,
    denominator,
    max_local_vol,
    ratio_bounds,
    engine_requested,
    engine_used,
    fallback_used,
):
    tau = base["tau"]
    local_vol_raw = np.sqrt(np.where(local_var > 0, local_var, np.nan))
    ratio_raw = local_vol_raw / iv
    boundary = np.zeros_like(local_var, dtype=bool)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    support = _support_mask(q, model_k, tau)
    denom_floor = 1e-10
    nonfinite_var = ~np.isfinite(local_var)
    negative_var = np.isfinite(local_var) & (local_var <= 0)
    negative_density = np.isfinite(denominator) & (denominator <= denom_floor)
    near_zero_denominator = ~np.isfinite(denominator) | (np.abs(denominator) <= denom_floor)
    hard_invalid = nonfinite_var | negative_var | negative_density | near_zero_denominator
    valid_ratio = support & (~boundary) & (~hard_invalid) & np.isfinite(ratio_raw) & np.isfinite(local_vol_raw)
    lo, hi = float(ratio_bounds[0]), float(ratio_bounds[1])
    extreme_ratio = valid_ratio & ((ratio_raw < lo) | (ratio_raw > hi) | (local_vol_raw > float(max_local_vol)))
    flagged = hard_invalid | extreme_ratio
    return {
        "engine_requested": engine_requested,
        "engine_used": engine_used,
        "fallback_used": bool(fallback_used),
        "k": model_k,
        "k_forward": model_k,
        "k_spot": base["k_spot"],
        "tau": base["tau"],
        "tau_days": base["tau_days"],
        "strike": base["strike"],
        "iv": np.where(support, iv, np.nan),
        "price": np.where(support, price, np.nan),
        "local_var": local_var,
        "local_vol": np.where(flagged | boundary | (~support), np.nan, local_vol_raw),
        "local_vol_raw": local_vol_raw,
        "local_vol_to_iv": np.where(flagged | boundary | (~support), np.nan, ratio_raw),
        "local_vol_to_iv_raw": ratio_raw,
        "numerator": numerator,
        "denominator": denominator,
        "hard_invalid": hard_invalid,
        "negative_var": negative_var,
        "negative_density": negative_density,
        "near_zero_denominator": near_zero_denominator,
        "extreme_ratio": extreme_ratio,
        "boundary": boundary,
        "support": support,
        "rate": base["rate"],
        "carry": base["carry"],
        "dividend_yield": base["dividend_yield"],
        "forward": base["forward"],
        "spot": base["spot"],
    }


def dupire_grid_numpy(
    fit,
    q,
    *,
    k_min=-0.22,
    k_max=0.08,
    tau_min=21 / 365.25,
    tau_max=150 / 365.25,
    n_k=51,
    n_tau=31,
    max_local_vol=2.50,
    ratio_bounds=(0.50, 1.80),
    annualization_days=365.25,
    engine_requested="numpy",
    fallback_used=False,
) -> dict:
    """Evaluate a Dupire local-volatility grid with finite differences.

    The function builds a strike-maturity grid, evaluates the fitted implied-volatility
    surface, prices calls with the forward Black-Scholes model, estimates the required
    Dupire derivatives by finite differences, and finishes the result with validity,
    support, and stress masks.

    Parameters
    ----------
    fit : dict
        Fitted implied-volatility surface.
    q : pandas.DataFrame
        Quote table used to infer spot, rate, carry, and grid support.
    k_min, k_max : float
        Spot log-moneyness grid bounds.
    tau_min, tau_max : float
        Maturity grid bounds in years.
    n_k, n_tau : int
        Number of moneyness and maturity grid nodes.
    max_local_vol : float, default=2.50
        Upper cap used for stress flagging and clipping.
    ratio_bounds : tuple[float, float], default=(0.50, 1.80)
        Acceptable local-vol-to-implied-vol ratio range.
    annualization_days : float, default=365.25
        Days per year used for grid metadata.
    engine_requested : str, default='numpy'
        Requested engine label stored in output metadata.
    fallback_used : bool, default=False
        Whether this evaluation is a fallback from another backend.

    Returns
    -------
    dict
        Local-volatility diagnostic grid containing IV, price, local variance,
        numerator/denominator terms, support masks, invalid flags, local-vol ratios,
        and backend metadata.
    """

    base = _base_grid(q, k_min, k_max, tau_min, tau_max, n_k, n_tau, annualization_days)
    tau = base["tau"]
    strike = base["strike"]
    iv = np.full((len(tau), len(strike)), np.nan)
    price = np.full_like(iv, np.nan)
    model_k = np.full_like(iv, np.nan)
    for i, t in enumerate(tau):
        row_k = np.log(strike / base["forward"][i])
        model_k[i] = row_k
        iv_row = surface_iv(fit, row_k, np.full_like(row_k, t))
        iv[i] = iv_row
        price[i] = black76_price("call", base["forward"][i], strike, t, iv_row, np.exp(-base["rate"][i] * t))
    c_tau = np.gradient(price, tau, axis=0, edge_order=2)
    c_k = np.gradient(price, strike, axis=1, edge_order=2)
    c_kk = np.gradient(c_k, strike, axis=1, edge_order=2)
    numerator = c_tau + base["dividend_yield"][:, None] * price + base["carry"][:, None] * strike[None, :] * c_k
    denominator = 0.5 * strike[None, :] ** 2 * c_kk
    local_var = numerator / denominator
    return _finish_grid(
        base=base,
        q=q,
        model_k=model_k,
        iv=iv,
        price=price,
        local_var=local_var,
        numerator=numerator,
        denominator=denominator,
        max_local_vol=max_local_vol,
        ratio_bounds=ratio_bounds,
        engine_requested=engine_requested,
        engine_used="numpy",
        fallback_used=fallback_used,
    )


def dupire_grid_jax(
    fit,
    q,
    *,
    k_min=-0.22,
    k_max=0.08,
    tau_min=21 / 365.25,
    tau_max=150 / 365.25,
    n_k=51,
    n_tau=31,
    max_local_vol=2.50,
    ratio_bounds=(0.50, 1.80),
    annualization_days=365.25,
    engine_requested="jax",
) -> dict:
    """Evaluate a Dupire local-volatility grid with JAX automatic differentiation.

    The function uses autodiff to compute the derivatives needed in the Dupire formula
    from the fitted implied-volatility surface and then applies the same finishing and
    stress-flagging logic as the finite-difference implementation.

    Parameters
    ----------
    fit : dict
        Fitted implied-volatility surface.
    q : pandas.DataFrame
        Quote table used to infer spot, rate, carry, and grid support.
    k_min, k_max : float
        Spot log-moneyness grid bounds.
    tau_min, tau_max : float
        Maturity grid bounds in years.
    n_k, n_tau : int
        Number of grid nodes.
    max_local_vol : float, default=2.50
        Upper cap used for stress flagging and clipping.
    ratio_bounds : tuple[float, float], default=(0.50, 1.80)
        Acceptable local-vol-to-IV ratio range.
    annualization_days : float, default=365.25
        Days per year used for metadata.
    engine_requested : str, default='jax'
        Requested engine label stored in output metadata.

    Returns
    -------
    dict
        Local-volatility diagnostic grid with ``engine_used='jax'`` metadata.
    """

    import jax.numpy as jnp

    base = _base_grid(q, k_min, k_max, tau_min, tau_max, n_k, n_tau, annualization_days)
    tau_np = base["tau"]
    strike_np = base["strike"]
    tau_j = jnp.asarray(tau_np)
    rate_j = jnp.asarray(base["rate"])
    carry_j = jnp.asarray(base["carry"])
    strike_j = jnp.asarray(strike_np)
    spot_j = jnp.asarray(base["spot"])
    fit_j = _jax_fit_arrays(fit)
    price, iv, model_k, local_var, numerator, denominator = _dupire_eval_grid_jax_fn()(
        fit_j,
        tau_j,
        strike_j,
        rate_j,
        carry_j,
        spot_j,
    )
    price.block_until_ready()
    return _finish_grid(
        base=base,
        q=q,
        model_k=np.asarray(model_k, dtype=float),
        iv=np.asarray(iv, dtype=float),
        price=np.asarray(price, dtype=float),
        local_var=np.asarray(local_var, dtype=float),
        numerator=np.asarray(numerator, dtype=float),
        denominator=np.asarray(denominator, dtype=float),
        max_local_vol=max_local_vol,
        ratio_bounds=ratio_bounds,
        engine_requested=engine_requested,
        engine_used="jax",
        fallback_used=False,
    )


def dupire_grid(
    fit,
    q,
    *,
    k_min: float = -0.22,
    k_max: float = 0.08,
    tau_min: float = 21 / 365.25,
    tau_max: float = 150 / 365.25,
    n_k: int = 51,
    n_tau: int = 31,
    max_local_vol: float = 2.50,
    ratio_bounds: tuple[float, float] = (0.50, 1.80),
    annualization_days: float = 365.25,
    engine: str = "jax",
    fallback: bool = True,
) -> dict:
    """Compute Dupire local-volatility diagnostics with JAX or finite-difference fallback.

    Parameters
    ----------
    fit : dict
        Fitted implied-volatility surface.
    q : pandas.DataFrame
        Quote table used for market inputs and grid support.
    k_min, k_max : float
        Spot log-moneyness grid bounds.
    tau_min, tau_max : float
        Maturity grid bounds in years.
    n_k, n_tau : int
        Number of grid nodes.
    max_local_vol : float, default=2.50
        Upper cap used in local-vol diagnostics.
    ratio_bounds : tuple[float, float], default=(0.50, 1.80)
        Acceptable local-vol-to-IV ratio range.
    annualization_days : float, default=365.25
        Days per year used for maturity metadata.
    engine : {'jax', 'auto', 'numpy', 'finite_difference'}, default='jax'
        Evaluation backend.
    fallback : bool, default=True
        If True, use finite differences when JAX is unavailable or fails.

    Returns
    -------
    dict
        Local-volatility grid and diagnostic masks.

    Raises
    ------
    ValueError
        If an unsupported engine is requested.
    ImportError
        If JAX is requested without fallback and JAX is unavailable.
    """

    engine_requested = str(engine).lower()
    common = dict(
        k_min=k_min,
        k_max=k_max,
        tau_min=tau_min,
        tau_max=tau_max,
        n_k=n_k,
        n_tau=n_tau,
        max_local_vol=max_local_vol,
        ratio_bounds=ratio_bounds,
        annualization_days=annualization_days,
    )
    if engine_requested in {"numpy", "finite_difference"}:
        return dupire_grid_numpy(fit, q, engine_requested=engine_requested, **common)
    if engine_requested not in {"jax", "auto"}:
        raise ValueError("engine must be one of {'jax', 'auto', 'numpy', 'finite_difference'}.")
    if _jax_available():
        try:
            return dupire_grid_jax(fit, q, engine_requested=engine_requested, **common)
        except Exception as exc:
            if not fallback:
                raise
            warnings.warn(
                "JAX Dupire evaluation failed; falling back to NumPy finite-difference engine. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return dupire_grid_numpy(fit, q, engine_requested=engine_requested, fallback_used=True, **common)
    if not fallback:
        raise ImportError("JAX is not installed; install jax for autodiff Dupire local volatility.")
    warnings.warn(
        "JAX is not installed; falling back to NumPy finite-difference engine. "
        "Install jax for autodiff Dupire and surface Greeks.",
        RuntimeWarning,
        stacklevel=2,
    )
    return dupire_grid_numpy(fit, q, engine_requested=engine_requested, fallback_used=True, **common)


def dupire_stress_summary(lv: dict) -> dict:
    """Summarize Dupire local-volatility stress flags inside supported grid regions.

    Parameters
    ----------
    lv : dict
        Local-volatility diagnostic grid returned by the Dupire evaluator.

    Returns
    -------
    dict
        Stress metrics including invalid-node shares, negative-density share,
        local-vol-to-IV ratio summaries, ATM/downside/upside local-vol levels, and a
        composite ``dupire_stress`` score.
    """

    region = np.asarray(lv["support"], dtype=bool) & (~np.asarray(lv["boundary"], dtype=bool))
    if not region.any():
        region = np.isfinite(lv["iv"])
    hard = np.asarray(lv["hard_invalid"], dtype=bool)
    extreme = np.asarray(lv["extreme_ratio"], dtype=bool)
    neg_density = np.asarray(lv["negative_density"], dtype=bool)
    hard_other = hard & (~neg_density)
    ratio = np.asarray(lv["local_vol_to_iv_raw"], dtype=float)
    valid_ratio = ratio[region & (~hard) & (~extreme) & np.isfinite(ratio)]
    hard_share = float(np.nanmean(hard[region])) if region.any() else np.nan
    hard_other_share = float(np.nanmean(hard_other[region])) if region.any() else np.nan
    extreme_share = float(np.nanmean(extreme[region & (~hard)])) if (region & (~hard)).any() else np.nan
    total_flagged = float(np.nanmean((hard | extreme)[region])) if region.any() else np.nan
    neg_density_share = float(np.nanmean(neg_density[region])) if region.any() else np.nan
    med_abs = float(np.nanmedian(np.abs(valid_ratio - 1.0))) if len(valid_ratio) else np.nan
    k_spot = np.asarray(lv["k_spot"], dtype=float)
    local_vol = np.asarray(lv["local_vol"], dtype=float)
    atm = local_vol[:, int(np.nanargmin(np.abs(k_spot)))]
    down = local_vol[:, int(np.nanargmin(np.abs(k_spot + 0.125)))]
    up = local_vol[:, int(np.nanargmin(np.abs(k_spot - 0.08)))]
    stress = 1.00 * hard_other_share + 0.75 * extreme_share + 0.50 * med_abs + 0.25 * neg_density_share

    def med(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.nanmedian(values)) if len(values) else np.nan

    return {
        "engine_used": lv.get("engine_used", "unknown"),
        "fallback_used": bool(lv.get("fallback_used", False)),
        "dupire_hard_invalid_share": hard_share,
        "dupire_hard_invalid_other_share": hard_other_share,
        "dupire_extreme_ratio_share": extreme_share,
        "dupire_total_flagged_share": total_flagged,
        "dupire_negative_density_share": neg_density_share,
        "dupire_median_lv_to_iv": float(np.nanmedian(valid_ratio)) if len(valid_ratio) else np.nan,
        "dupire_median_abs_lv_to_iv_minus_1": med_abs,
        "dupire_atm_local_vol_level": med(atm),
        "dupire_downside_local_vol_premium": med(down - atm),
        "dupire_upside_local_vol_premium": med(up - atm),
        "dupire_stress": float(stress),
    }


def dupire_stress_panel(
    quotes: pd.DataFrame,
    *,
    fits: dict,
    date_col: str = "date",
    **kwargs,
) -> pd.DataFrame:
    """Compute Dupire stress summaries for multiple fitted surfaces by date.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table with a date column.
    fits : dict
        Mapping from date to fitted volatility-surface object.
    date_col : str, default='date'
        Quote date column.
    **kwargs
        Additional arguments forwarded to the Dupire grid evaluator.

    Returns
    -------
    pandas.DataFrame
        Date-level Dupire stress panel. Failed dates are reported with an ``error``
        message.
    """

    rows = []
    data = quotes.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    for date, fit in fits.items():
        date = pd.Timestamp(date).normalize()
        q = data[data[date_col].eq(date)].copy()
        if q.empty:
            continue
        try:
            lv = dupire_grid(fit, q, **kwargs)
            rows.append({"date": date, **dupire_stress_summary(lv)})
        except Exception as exc:
            rows.append({"date": date, "error": str(exc)[:160]})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()


__all__ = [
    "carry_by_tau",
    "curve_by_tau",
    "dividend_yield_by_tau",
    "dupire_grid",
    "dupire_grid_jax",
    "dupire_grid_numpy",
    "dupire_stress_panel",
    "dupire_stress_summary",
    "rate_by_tau",
]
