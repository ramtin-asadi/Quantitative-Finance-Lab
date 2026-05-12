from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from quantfinlab.numerics.interpolation import (
    slice_pchip_grid,
    tensor_spline_fit,
    tensor_spline_values,
    tensor_spline_values_jax,
)

from .quote_cleaning import surface_common_support, surface_support_by_date


def _as_date(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def _grid_tau_days(grid: dict, annualization_days: float = 365.25) -> np.ndarray:
    if "tau_days" in grid:
        return np.asarray(grid["tau_days"], dtype=float)
    return np.asarray(grid["tau"], dtype=float) * float(grid.get("annualization_days", annualization_days))


def _weighted_rmse(err, weight) -> float:
    err = np.asarray(err, dtype=float)
    weight = np.asarray(weight, dtype=float)
    ok = np.isfinite(err) & np.isfinite(weight) & (weight > 0)
    if not ok.any():
        return np.nan
    return float(np.sqrt(np.average(err[ok] ** 2, weights=weight[ok])))


def surface_grid(
    quotes: pd.DataFrame,
    *,
    k_col: str = "k",
    k_spot_col: str = "k_spot",
    tau_col: str = "tau",
    k_quantiles: tuple[float, float] = (0.02, 0.98),
    tau_quantiles: tuple[float, float] = (0.02, 0.98),
    n_k: int = 65,
    n_tau: int = 35,
    annualization_days: float = 365.25,
) -> dict:
    """Date-specific support-aware log-moneyness/maturity grid."""
    q = quotes.copy()
    k = pd.to_numeric(q[k_col], errors="coerce")
    tau = pd.to_numeric(q[tau_col], errors="coerce")
    ok = np.isfinite(k) & np.isfinite(tau)
    if not ok.any():
        raise ValueError("No finite k/tau observations for surface grid.")
    k_lo, k_hi = np.nanquantile(k[ok], k_quantiles)
    t_lo, t_hi = np.nanquantile(tau[ok], tau_quantiles)
    grid = {
        "k": np.linspace(float(k_lo), float(k_hi), int(n_k)),
        "tau": np.linspace(float(t_lo), float(t_hi), int(n_tau)),
        "tau_days": np.linspace(float(t_lo), float(t_hi), int(n_tau)) * float(annualization_days),
        "annualization_days": float(annualization_days),
    }
    if k_spot_col in q.columns:
        ks = pd.to_numeric(q[k_spot_col], errors="coerce")
        ok_spot = np.isfinite(ks)
        if ok_spot.any():
            s_lo, s_hi = np.nanquantile(ks[ok_spot], k_quantiles)
            grid["k_spot"] = np.linspace(float(s_lo), float(s_hi), int(n_k))
    return grid


def common_surface_grid(
    quotes: pd.DataFrame,
    *,
    date_col: str = "date",
    k_col: str = "k",
    tau_col: str = "tau",
    k_min: float = -0.25,
    k_max: float = 0.10,
    tau_min: float = 21 / 365.25,
    tau_max: float = 150 / 365.25,
    n_k: int = 61,
    n_tau: int = 31,
    min_support_share: float = 0.85,
    annualization_days: float = 365.25,
) -> dict:
    """Conservative common grid and support mask for historical work."""
    k_grid = np.linspace(float(k_min), float(k_max), int(n_k))
    tau_grid = np.linspace(float(tau_min), float(tau_max), int(n_tau))
    support = surface_common_support(
        quotes,
        date_col=date_col,
        k_col=k_col,
        tau_col=tau_col,
        k_grid=k_grid,
        tau_grid=tau_grid,
        min_support_share=min_support_share,
    )
    support["tau_days"] = tau_grid * float(annualization_days)
    support["annualization_days"] = float(annualization_days)
    support["min_support_share"] = float(min_support_share)
    return support


def fit_log_total_variance_surface(
    quotes: pd.DataFrame,
    *,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    weight_col: str = "surface_weight",
    n_k_basis: int = 12,
    n_tau_basis: int = 8,
    degree: int = 3,
    lambda_k: float = 10.0,
    lambda_tau: float = 10.0,
    label: str | None = None,
) -> dict:
    """Fit ``log(iv**2 * tau)`` with a penalized tensor B-spline."""
    q = quotes.copy()
    total_var = (pd.to_numeric(q[iv_col], errors="coerce") ** 2) * pd.to_numeric(q[tau_col], errors="coerce")
    y = np.log(total_var.clip(lower=1e-12))
    weight = q[weight_col] if weight_col in q.columns else np.ones(len(q), dtype=float)
    fit = tensor_spline_fit(
        q[k_col],
        q[tau_col],
        y,
        weights=weight,
        n_x_basis=n_k_basis,
        n_y_basis=n_tau_basis,
        degree=degree,
        lambda_x=lambda_k,
        lambda_y=lambda_tau,
    )
    fit["target"] = "log_total_variance"
    fit["label"] = label or "surface"
    fit["columns"] = {"k": k_col, "tau": tau_col, "iv": iv_col, "weight": weight_col}
    fit["knots_k"] = fit["knots_x"]
    fit["knots_tau"] = fit["knots_y"]
    fit["center_k"] = fit["center_x"]
    fit["scale_k"] = fit["scale_x"]
    fit["center_tau"] = fit["center_y"]
    fit["scale_tau"] = fit["scale_y"]
    fit["diagnostics"] = surface_fit_summary(q, {fit["label"]: fit}, k_col=k_col, tau_col=tau_col, iv_col=iv_col, weight_col=weight_col).iloc[0].to_dict()
    return fit


def surface_log_total_variance(fit: dict, k, tau, *, grid: bool = False, der_k: int = 0, der_tau: int = 0) -> np.ndarray:
    return tensor_spline_values(fit, k, tau, grid=grid, der_x=der_k, der_y=der_tau)


def surface_total_variance(fit: dict, k, tau, *, grid: bool = False) -> np.ndarray:
    log_w = surface_log_total_variance(fit, k, tau, grid=grid)
    return np.exp(np.clip(log_w, -50.0, 20.0))


def surface_total_variance_jax(fit: dict, k, tau):
    import jax.numpy as jnp

    log_w = tensor_spline_values_jax(fit, k, tau)
    return jnp.exp(jnp.clip(log_w, -50.0, 20.0))


def surface_iv(fit: dict, k, tau, *, grid: bool = False) -> np.ndarray:
    tau_arr = np.asarray(tau, dtype=float)
    w = surface_total_variance(fit, k, tau, grid=grid)
    if grid:
        tau_safe = np.maximum(tau_arr.reshape(-1, 1), 1e-10)
    else:
        tau_safe = np.maximum(tau_arr, 1e-10)
    return np.sqrt(w / tau_safe)


def surface_iv_jax(fit: dict, k, tau, tau_floor: float = 1e-10):
    import jax.numpy as jnp

    tau_arr = jnp.asarray(tau)
    w = surface_total_variance_jax(fit, k, tau_arr)
    return jnp.sqrt(w / jnp.maximum(tau_arr, float(tau_floor)))


def surface_iv_grid(fit: dict, grid: dict) -> np.ndarray:
    return surface_iv(fit, grid["k"], grid["tau"], grid=True)


def surface_fit_summary(
    quotes: pd.DataFrame,
    fits: dict[str, dict],
    *,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    weight_col: str = "surface_weight",
) -> pd.DataFrame:
    rows = []
    q = quotes.copy()
    weight = pd.to_numeric(q[weight_col], errors="coerce") if weight_col in q.columns else pd.Series(1.0, index=q.index)
    for name, fit in fits.items():
        fitted = surface_iv(fit, q[k_col].to_numpy(dtype=float), q[tau_col].to_numpy(dtype=float))
        observed = pd.to_numeric(q[iv_col], errors="coerce").to_numpy(dtype=float)
        err = observed - np.asarray(fitted, dtype=float)
        abs_err = np.abs(err)
        rows.append(
            {
                "fit": name,
                "target": fit.get("target", "log_total_variance"),
                "quote_count": int(np.isfinite(err).sum()),
                "number_of_maturities": int(q["expiry"].nunique()) if "expiry" in q.columns else int(q[tau_col].nunique()),
                "rmse": float(np.sqrt(np.nanmean(err**2))),
                "weighted_rmse": _weighted_rmse(err, weight),
                "mae": float(np.nanmean(abs_err)),
                "median_abs_error": float(np.nanmedian(abs_err)),
                "p95_abs_error": float(np.nanquantile(abs_err[np.isfinite(abs_err)], 0.95)) if np.isfinite(abs_err).any() else np.nan,
                "mean_residual": float(np.nanmean(err)),
                "residual_std": float(np.nanstd(err)),
                "k_min": float(np.nanmin(q[k_col])),
                "k_max": float(np.nanmax(q[k_col])),
                "tau_min": float(np.nanmin(q[tau_col])),
                "tau_max": float(np.nanmax(q[tau_col])),
            },
        )
    return pd.DataFrame(rows)


def surface_residuals(
    quotes: pd.DataFrame,
    fits: dict[str, dict],
    *,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
) -> pd.DataFrame:
    out = quotes.copy()
    for name, fit in fits.items():
        fitted = surface_iv(fit, out[k_col].to_numpy(dtype=float), out[tau_col].to_numpy(dtype=float))
        out[f"fit_iv_{name}"] = fitted
        out[f"residual_{name}"] = pd.to_numeric(out[iv_col], errors="coerce") - fitted
        out[f"abs_residual_{name}"] = out[f"residual_{name}"].abs()
    return out


def pchip_surface(
    quotes: pd.DataFrame,
    *,
    grid: dict,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    weight_col: str = "surface_weight",
    slice_col: str = "expiry",
    min_k: int = 6,
) -> np.ndarray:
    return slice_pchip_grid(
        quotes,
        x_col=k_col,
        y_col=tau_col,
        z_col=iv_col,
        x_grid=grid["k"],
        y_grid=grid["tau"],
        weight_col=weight_col if weight_col in quotes.columns else None,
        slice_col=slice_col if slice_col in quotes.columns else None,
        min_x=min_k,
    )[0]


def pchip_spline_comparison(
    quotes: pd.DataFrame,
    *,
    fit: dict,
    grid: dict,
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    weight_col: str = "surface_weight",
) -> pd.DataFrame:
    raw = pchip_surface(quotes, grid=grid, k_col=k_col, tau_col=tau_col, iv_col=iv_col, weight_col=weight_col)
    smooth = surface_iv_grid(fit, grid)
    diff = raw - smooth
    rows = [
        {
            "comparison": "pchip_minus_spline_grid",
            "finite_nodes": int(np.isfinite(diff).sum()),
            "rmse": float(np.sqrt(np.nanmean(diff**2))),
            "mae": float(np.nanmean(np.abs(diff))),
            "median_abs_error": float(np.nanmedian(np.abs(diff))),
        },
    ]
    return pd.DataFrame(rows)


def fit_surface_panel(
    quotes: pd.DataFrame,
    *,
    date_col: str = "date",
    k_col: str = "k",
    tau_col: str = "tau",
    iv_col: str = "iv_mid",
    weight_col: str = "surface_weight",
    visual_params: dict | None = None,
    dupire_params: dict | None = None,
    min_quotes: int = 160,
    min_expiries: int = 5,
) -> dict:
    visual_params = dict(visual_params or {})
    dupire_params = dict(dupire_params or visual_params)
    visual_fits: dict[pd.Timestamp, dict] = {}
    dupire_fits: dict[pd.Timestamp, dict] = {}
    rows: list[dict] = []
    skipped: list[dict] = []
    data = quotes.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    tic = time.perf_counter()
    for date, q in data.groupby(date_col):
        date = _as_date(date)
        n_exp = int(q["expiry"].nunique()) if "expiry" in q.columns else int(q[tau_col].nunique())
        if len(q) < int(min_quotes) or n_exp < int(min_expiries):
            skipped.append({"date": date, "error": "too few quotes or maturities", "quotes": len(q), "expiries": n_exp})
            continue
        try:
            fit_v = fit_log_total_variance_surface(q, k_col=k_col, tau_col=tau_col, iv_col=iv_col, weight_col=weight_col, label="visual", **visual_params)
            fit_d = fit_log_total_variance_surface(q, k_col=k_col, tau_col=tau_col, iv_col=iv_col, weight_col=weight_col, label="dupire", **dupire_params)
            visual_fits[date] = fit_v
            dupire_fits[date] = fit_d
            rows.extend(surface_fit_summary(q, {"visual": fit_v, "dupire": fit_d}, k_col=k_col, tau_col=tau_col, iv_col=iv_col, weight_col=weight_col).assign(date=date).to_dict("records"))
        except Exception as exc:
            skipped.append({"date": date, "error": str(exc)[:160], "quotes": len(q), "expiries": n_exp})
    return {
        "visual_fits": visual_fits,
        "dupire_fits": dupire_fits,
        "fit_summary": pd.DataFrame(rows).sort_values(["date", "fit"]).reset_index(drop=True) if rows else pd.DataFrame(),
        "skipped": pd.DataFrame(skipped),
        "elapsed_sec": float(time.perf_counter() - tic),
    }


def surface_cube(
    quotes: pd.DataFrame,
    *,
    fits: dict,
    grid: dict,
    date_col: str = "date",
    output: str = "iv",
) -> dict:
    dates = sorted(pd.to_datetime(list(fits.keys())))
    values = []
    for date in dates:
        fit = fits[_as_date(date)]
        arr = surface_iv_grid(fit, grid) if output == "iv" else surface_total_variance(fit, grid["k"], grid["tau"], grid=True)
        mask = grid.get("support_mask")
        if mask is not None:
            arr = np.where(mask, arr, np.nan)
        values.append(arr)
    return {"values": np.asarray(values, dtype=float), "dates": pd.to_datetime(pd.Index(dates)), "grid": grid, "output": output}


def _pca_one(x: np.ndarray, n_components: int, standardize: bool, random_state: int | None) -> dict:
    if x.shape[0] < 2 or x.shape[1] < 2:
        return {"scores": pd.DataFrame(), "explained_variance_table": pd.DataFrame(), "components": np.empty((0, x.shape[1])), "scaler": None}
    scaler = StandardScaler(with_mean=True, with_std=True) if standardize else None
    x_fit = scaler.fit_transform(x) if scaler is not None else x - np.nanmean(x, axis=0)
    n = min(int(n_components), x_fit.shape[0], x_fit.shape[1])
    model = PCA(n_components=n, random_state=random_state)
    scores = model.fit_transform(x_fit)
    var = pd.DataFrame(
        {
            "component": [f"pc{i}" for i in range(1, n + 1)],
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative": np.cumsum(model.explained_variance_ratio_),
            "score_std": np.sqrt(model.explained_variance_),
        },
    )
    return {"model": model, "scores_array": scores, "explained_variance_table": var, "components": model.components_, "scaler": scaler}


def surface_pca(
    cube: dict,
    *,
    n_components: int = 5,
    mode: str = "level",
    standardize: bool = False,
    random_state: int | None = None,
) -> dict:
    mode_key = str(mode).lower()
    if mode_key == "level_and_shape":
        level = surface_pca(
            cube,
            n_components=n_components,
            mode="level",
            standardize=standardize,
            random_state=random_state,
        )
        level["mode"] = "level_and_shape"
        level["shape"] = surface_pca(
            cube,
            n_components=n_components,
            mode="shape",
            standardize=standardize,
            random_state=random_state,
        )
        return level
    values = np.asarray(cube["values"], dtype=float)
    dates = pd.to_datetime(cube["dates"])
    if values.shape[0] < 2:
        return {
            "mode": mode,
            "grid": cube["grid"],
            "dates": dates,
            "node_positions": np.array([], dtype=int),
            "explained_variance_table": pd.DataFrame(columns=["component", "explained_variance_ratio", "cumulative", "score_std"]),
            "scores": pd.DataFrame(columns=["date"]),
            "components": np.empty((0, 0)),
            "diagnostic": "not_enough_dates",
        }
    changes = np.diff(values, axis=0)
    flat = changes.reshape(changes.shape[0], -1)
    finite_share = np.isfinite(flat).mean(axis=0)
    keep = finite_share >= 0.75
    x = flat[:, keep]
    col_mean = np.nanmean(x, axis=0)
    col_std = np.nanstd(x, axis=0)
    good = np.isfinite(col_mean) & np.isfinite(col_std) & (col_std > 1e-10)
    x = x[:, good]
    node_positions = np.where(keep)[0][good]
    missing = ~np.isfinite(x)
    x[missing] = np.take(np.nanmean(x, axis=0), np.where(missing)[1])
    if mode_key in {"shape", "shape_only"}:
        x = x - np.nanmean(x, axis=1, keepdims=True)
    fit = _pca_one(x, n_components, standardize, random_state)
    scores = pd.DataFrame(fit.get("scores_array", np.empty((len(dates) - 1, 0))), columns=[f"pc{i}" for i in range(1, fit["components"].shape[0] + 1)])
    scores["date"] = dates[1:]
    fit.update({"mode": mode, "grid": cube["grid"], "dates": dates[1:], "scores": scores, "node_positions": node_positions, "x_std": np.nanstd(x, axis=0), "cube_shape": values.shape[1:]})
    return fit


def surface_pca_shocks(pca: dict, *, grid: dict | None = None, components=(1, 2, 3), output_units: str = "iv_points") -> dict:
    grid = grid or pca.get("grid", {})
    shape = pca.get("cube_shape")
    if shape is None and "tau" in grid and "k" in grid:
        shape = (len(grid["tau"]), len(grid["k"]))
    out = {"grid": grid, "output_units": output_units}
    comps = np.asarray(pca.get("components", np.empty((0, 0))), dtype=float)
    positions = np.asarray(pca.get("node_positions", []), dtype=int)
    for comp in components:
        idx = int(comp) - 1
        arr = np.full(int(np.prod(shape)), np.nan, dtype=float)
        if idx < comps.shape[0] and len(positions):
            scale = float(pca["explained_variance_table"].iloc[idx]["score_std"]) if not pca["explained_variance_table"].empty else 1.0
            shock = comps[idx] * scale
            scaler = pca.get("scaler")
            if scaler is not None and hasattr(scaler, "scale_"):
                shock = shock * scaler.scale_
            arr[positions] = shock
        out[f"pc{comp}"] = arr.reshape(shape)
    return out


def surface_features(cube: dict, pca: dict | None = None, *, down_k: float = -0.125, up_k: float = 0.08) -> pd.DataFrame:
    values = np.asarray(cube["values"], dtype=float)
    grid = cube["grid"]
    dates = pd.to_datetime(cube["dates"])
    k = np.asarray(grid["k"], dtype=float)
    tau_days = _grid_tau_days(grid)
    rows = []
    short_i = int(np.nanargmin(np.abs(tau_days - 30)))
    mid_i = int(np.nanargmin(np.abs(tau_days - 75)))
    long_i = int(np.nanargmin(np.abs(tau_days - 150)))
    atm_j = int(np.nanargmin(np.abs(k)))
    down_j = int(np.nanargmin(np.abs(k - float(down_k))))
    up_j = int(np.nanargmin(np.abs(k - float(up_k))))
    for i, date in enumerate(dates):
        arr = values[i]
        atm_short = arr[short_i, atm_j]
        atm_mid = arr[mid_i, atm_j]
        atm_long = arr[long_i, atm_j]
        down = arr[mid_i, down_j]
        up = arr[mid_i, up_j]
        rows.append(
            {
                "date": _as_date(date),
                "short_atm_iv": float(atm_short),
                "medium_atm_iv": float(atm_mid),
                "long_atm_iv": float(atm_long),
                "term_slope": float(atm_long - atm_short),
                "downside_skew": float(down - atm_mid),
                "upside_skew": float(up - atm_mid),
                "curvature": float(down + up - 2.0 * atm_mid),
                "smile_asymmetry": float((down - atm_mid) - (up - atm_mid)),
            },
        )
    out = pd.DataFrame(rows)
    if pca is not None and isinstance(pca.get("scores"), pd.DataFrame) and not pca["scores"].empty:
        out = out.merge(pca["scores"], on="date", how="left")
    return out


def surface_project_tables(
    *,
    fit_summary: pd.DataFrame | None = None,
    pchip_comparison: pd.DataFrame | None = None,
    residuals: pd.DataFrame | None = None,
    pca: dict | None = None,
    features: pd.DataFrame | None = None,
) -> dict:
    tables = {
        "fit_summary": fit_summary if fit_summary is not None else pd.DataFrame(),
        "pchip_comparison": pchip_comparison if pchip_comparison is not None else pd.DataFrame(),
        "largest_residuals": pd.DataFrame(),
        "pca_explained_variance": pd.DataFrame(),
        "feature_tail": pd.DataFrame(),
        "top_regimes": pd.DataFrame(),
    }
    if residuals is not None and not residuals.empty:
        col = "abs_residual_visual" if "abs_residual_visual" in residuals.columns else next((c for c in residuals.columns if c.startswith("abs_residual_")), None)
        if col:
            tables["largest_residuals"] = residuals.sort_values(col, ascending=False).head(20)
    if pca is not None:
        tables["pca_explained_variance"] = pca.get("explained_variance_table", pd.DataFrame())
    if features is not None and not features.empty:
        tables["feature_tail"] = features.tail(10)
        sort_col = "dupire_stress" if "dupire_stress" in features.columns else "short_atm_iv"
        tables["top_regimes"] = features.sort_values(sort_col, ascending=False).head(10)
    return tables


__all__ = [
    "common_surface_grid",
    "fit_log_total_variance_surface",
    "fit_surface_panel",
    "pchip_spline_comparison",
    "pchip_surface",
    "surface_cube",
    "surface_features",
    "surface_fit_summary",
    "surface_grid",
    "surface_iv",
    "surface_iv_grid",
    "surface_iv_jax",
    "surface_log_total_variance",
    "surface_pca",
    "surface_pca_shocks",
    "surface_project_tables",
    "surface_residuals",
    "surface_total_variance",
    "surface_total_variance_jax",
    "surface_support_by_date",
]
