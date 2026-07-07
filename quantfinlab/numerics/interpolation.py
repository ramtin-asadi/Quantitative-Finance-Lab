from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.linalg import solve


def bspline_knots(n_basis: int, degree: int = 3, x_min: float = -1.0, x_max: float = 1.0) -> np.ndarray:
    """Open uniform B-spline knot vector."""
    n_basis = int(n_basis)
    degree = int(degree)
    if n_basis <= degree:
        raise ValueError("n_basis must be larger than degree.")
    x_min = float(x_min)
    x_max = float(x_max)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("x_min and x_max must be finite with x_max > x_min.")
    n_inner = n_basis - degree - 1
    inner = np.linspace(x_min, x_max, n_inner + 2)[1:-1] if n_inner > 0 else np.array([], dtype=float)
    return np.r_[np.repeat(x_min, degree + 1), inner, np.repeat(x_max, degree + 1)].astype(float)


def bspline_basis(x, knots, degree: int = 3, derivative: int = 0) -> np.ndarray:
    """Dense B-spline basis matrix for one coordinate."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float)
    degree = int(degree)
    derivative = int(derivative)
    n_basis = len(knots) - degree - 1
    if n_basis <= 0:
        raise ValueError("Invalid knot vector for requested degree.")
    x_eval = np.clip(x_arr, knots[0] + 1e-12, knots[-1] - 1e-12)
    eye = np.eye(n_basis)
    out = np.empty((len(x_eval), n_basis), dtype=float)
    for i in range(n_basis):
        fn = BSpline(knots, eye[i], degree, extrapolate=False)
        if derivative:
            fn = fn.derivative(derivative)
        out[:, i] = fn(x_eval)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def second_difference_matrix(n: int) -> np.ndarray:
    """Second-difference penalty matrix with shape ``(n - 2, n)``."""
    n = int(n)
    if n < 3:
        return np.zeros((0, n), dtype=float)
    out = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        out[i, i : i + 3] = [1.0, -2.0, 1.0]
    return out


def _center_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0
    lo, hi = np.nanquantile(finite, [0.01, 0.99])
    center = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = np.nanstd(finite)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return float(center), float(scale)


def tensor_spline_fit(
    x,
    y,
    z,
    *,
    weights=None,
    n_x_basis: int = 12,
    n_y_basis: int = 8,
    degree: int = 3,
    lambda_x: float = 10.0,
    lambda_y: float = 10.0,
    ridge: float = 1e-8,
) -> dict:
    """Fit a weighted penalized tensor-product B-spline to scattered data.

    The function constructs standardized x/y coordinates, builds B-spline bases in
    each dimension, solves a weighted least-squares system with second-difference
    smoothness penalties, and returns all objects needed for later evaluation.

    Parameters
    ----------
    x, y, z : array-like
        Scattered coordinate/value observations.
    weights : array-like, optional
        Positive observation weights.
    n_x_basis : int, default=12
        Number of basis functions in the x dimension.
    n_y_basis : int, default=8
        Number of basis functions in the y dimension.
    degree : int, default=3
        Spline degree.
    lambda_x : float, default=10.0
        Smoothness penalty in the x dimension.
    lambda_y : float, default=10.0
        Smoothness penalty in the y dimension.
    ridge : float, default=1e-8
        Numerical ridge added to the normal equations.

    Returns
    -------
    dict
        Spline fit dictionary containing coefficients, knots, scaling parameters,
        degree, basis sizes, penalties, ridge, and observation count.

    Raises
    ------
    ValueError
        If too few finite positive-weight observations are available.

    Notes
    -----
    Inputs are centered and scaled before basis construction. Evaluation functions
    apply the same scaling automatically.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if weights is None:
        w_arr = np.ones_like(z_arr, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).reshape(-1)
    ok = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr) & np.isfinite(w_arr) & (w_arr > 0)
    if ok.sum() < max(int(n_x_basis) + int(n_y_basis), 12):
        raise ValueError("Not enough finite points for tensor spline fit.")
    x_arr, y_arr, z_arr, w_arr = x_arr[ok], y_arr[ok], z_arr[ok], w_arr[ok]

    center_x, scale_x = _center_scale(x_arr)
    center_y, scale_y = _center_scale(y_arr)
    sx = (x_arr - center_x) / scale_x
    sy = (y_arr - center_y) / scale_y
    knots_x = bspline_knots(n_x_basis, degree=degree)
    knots_y = bspline_knots(n_y_basis, degree=degree)
    bx = bspline_basis(sx, knots_x, degree=degree)
    by = bspline_basis(sy, knots_y, degree=degree)
    design = (bx[:, :, None] * by[:, None, :]).reshape(len(z_arr), int(n_x_basis) * int(n_y_basis))

    sw = np.sqrt(w_arr / max(float(np.nanmedian(w_arr)), 1e-12))
    aw = design * sw[:, None]
    zw = z_arr * sw
    lhs = aw.T @ aw
    rhs = aw.T @ zw

    dx = second_difference_matrix(n_x_basis)
    dy = second_difference_matrix(n_y_basis)
    if len(dx):
        px = np.kron(dx, np.eye(int(n_y_basis)))
        lhs = lhs + float(lambda_x) * (px.T @ px)
    if len(dy):
        py = np.kron(np.eye(int(n_x_basis)), dy)
        lhs = lhs + float(lambda_y) * (py.T @ py)
    lhs = lhs + float(ridge) * np.eye(lhs.shape[0])

    coef_vec = solve(lhs, rhs, assume_a="pos", check_finite=False)
    coef = coef_vec.reshape(int(n_x_basis), int(n_y_basis))
    return {
        "coef": coef,
        "knots_x": knots_x,
        "knots_y": knots_y,
        "center_x": center_x,
        "scale_x": scale_x,
        "center_y": center_y,
        "scale_y": scale_y,
        "degree": int(degree),
        "n_x_basis": int(n_x_basis),
        "n_y_basis": int(n_y_basis),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "ridge": float(ridge),
        "n_obs": int(ok.sum()),
    }


def tensor_spline_values(fit: dict, x, y, *, grid: bool = False, der_x: int = 0, der_y: int = 0) -> np.ndarray:
    """Evaluate a tensor-product B-spline fit.

    Parameters
    ----------
    fit : dict
        Fit dictionary returned by ``tensor_spline_fit``.
    x, y : array-like
        Evaluation coordinates.
    grid : bool, default=False
        If true, evaluate on the full ``y`` by ``x`` grid. If false, evaluate at
        paired broadcast coordinates.
    der_x : int, default=0
        Derivative order in x.
    der_y : int, default=0
        Derivative order in y.

    Returns
    -------
    numpy.ndarray
        Evaluated spline values. In grid mode, the shape is
        ``(len(y), len(x))``; otherwise the output follows the broadcast input
        shape.
    """
    coef = np.asarray(fit["coef"], dtype=float)
    degree = int(fit.get("degree", 3))
    scale_x = float(fit.get("scale_x", 1.0))
    scale_y = float(fit.get("scale_y", 1.0))
    if grid:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        sx = (x_arr.reshape(-1) - float(fit["center_x"])) / scale_x
        sy = (y_arr.reshape(-1) - float(fit["center_y"])) / scale_y
        bx = bspline_basis(sx, fit["knots_x"], degree=degree, derivative=der_x) / (scale_x ** int(der_x))
        by = bspline_basis(sy, fit["knots_y"], degree=degree, derivative=der_y) / (scale_y ** int(der_y))
        return (bx @ coef @ by.T).T
    x_arr, y_arr = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    shape = x_arr.shape
    sx = (x_arr.reshape(-1) - float(fit["center_x"])) / scale_x
    sy = (y_arr.reshape(-1) - float(fit["center_y"])) / scale_y
    bx = bspline_basis(sx, fit["knots_x"], degree=degree, derivative=der_x) / (scale_x ** int(der_x))
    by = bspline_basis(sy, fit["knots_y"], degree=degree, derivative=der_y) / (scale_y ** int(der_y))
    return np.einsum("ij,jk,ik->i", bx, coef, by).reshape(shape)


def tensor_spline_values_jax(fit: dict, x, y):
    """JAX-compatible paired/scalar tensor B-spline evaluation."""
    import jax
    import jax.numpy as jnp

    coef = jnp.asarray(fit["coef"])
    knots_x = jnp.asarray(fit["knots_x"])
    knots_y = jnp.asarray(fit["knots_y"])
    degree = int(fit.get("degree", 3))
    center_x = jnp.asarray(fit["center_x"])
    scale_x = jnp.asarray(fit["scale_x"])
    center_y = jnp.asarray(fit["center_y"])
    scale_y = jnp.asarray(fit["scale_y"])

    def basis_one(value, knots):
        value = jnp.clip(value, knots[0] + 1e-10, knots[-1] - 1e-10)
        b = jnp.where((value >= knots[:-1]) & (value < knots[1:]), 1.0, 0.0)
        for d in range(1, degree + 1):
            n = knots.shape[0] - d - 1
            left_den = knots[d : n + d] - knots[:n]
            right_den = knots[d + 1 : n + d + 1] - knots[1 : n + 1]
            left_den_safe = jnp.where(left_den > 0, left_den, 1.0)
            right_den_safe = jnp.where(right_den > 0, right_den, 1.0)
            left = jnp.where(left_den > 0, (value - knots[:n]) / left_den_safe, 0.0) * b[:n]
            right = jnp.where(right_den > 0, (knots[d + 1 : n + d + 1] - value) / right_den_safe, 0.0) * b[1 : n + 1]
            b = left + right
        return b

    def scalar_eval(x_value, y_value):
        bx = basis_one((x_value - center_x) / scale_x, knots_x)
        by = basis_one((y_value - center_y) / scale_y, knots_y)
        return bx @ coef @ by

    x_arr, y_arr = jnp.broadcast_arrays(jnp.asarray(x), jnp.asarray(y))
    vals = jax.vmap(scalar_eval)(x_arr.reshape(-1), y_arr.reshape(-1))
    return vals.reshape(x_arr.shape)


def _weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if weights is None:
        return float(np.nanmean(x))
    w = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float(np.nanmean(x))
    return float(np.average(x[ok], weights=w[ok]))


def slice_pchip_grid(
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    z_col: str,
    x_grid,
    y_grid,
    weight_col: str | None = None,
    slice_col: str | None = None,
    min_x: int = 6,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Interpolate scattered slice data onto a support-limited two-dimensional grid.

    The function first applies monotone PCHIP interpolation along ``x`` within each
    observed ``y`` slice, then interpolates across observed ``y`` slices for each
    x-grid point. Extrapolated regions outside observed support are left missing.

    Parameters
    ----------
    data : pandas.DataFrame
        Scattered data table.
    x_col : str
        X-coordinate column.
    y_col : str
        Y-coordinate column.
    z_col : str
        Value column.
    x_grid, y_grid : array-like
        Target x and y grids.
    weight_col : str, optional
        Observation weight column used when collapsing duplicate x values.
    slice_col : str, optional
        Slice key. Defaults to ``y_col``.
    min_x : int, default=6
        Minimum unique x values required in a slice.

    Returns
    -------
    tuple
        ``(grid, meta)`` where ``grid`` has shape ``(len(y_grid), len(x_grid))``
        and ``meta`` records slice support.

    Notes
    -----
    This helper is useful for surface construction when data are naturally grouped
    by maturity or another slice dimension.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    if data.empty:
        return np.full((len(y_grid), len(x_grid)), np.nan), pd.DataFrame()

    key = slice_col or y_col
    rows: list[np.ndarray] = []
    meta: list[dict] = []
    for name, grp in data.groupby(key, dropna=False):
        y_val = float(np.nanmedian(pd.to_numeric(grp[y_col], errors="coerce")))
        cols = [x_col, z_col] + ([weight_col] if weight_col and weight_col in grp.columns else [])
        g = grp[cols].copy()
        g[x_col] = pd.to_numeric(g[x_col], errors="coerce")
        g[z_col] = pd.to_numeric(g[z_col], errors="coerce")
        g = g[np.isfinite(g[x_col]) & np.isfinite(g[z_col])].copy()
        if len(g) < int(min_x) or not np.isfinite(y_val):
            continue
        g["_x_round"] = g[x_col].round(8)
        if weight_col and weight_col in g.columns:
            collapsed = (
                g.groupby("_x_round")
                .apply(lambda q: _weighted_mean(q[z_col], q[weight_col]), include_groups=False)
                .rename(z_col)
                .reset_index()
                .rename(columns={"_x_round": x_col})
            )
        else:
            collapsed = g.groupby("_x_round", as_index=False)[z_col].mean().rename(columns={"_x_round": x_col})
        collapsed = collapsed.sort_values(x_col)
        if collapsed[x_col].nunique() < int(min_x):
            continue
        xx = collapsed[x_col].to_numpy(dtype=float)
        zz = collapsed[z_col].to_numpy(dtype=float)
        fn = PchipInterpolator(xx, zz, extrapolate=False)
        row = fn(x_grid)
        row[(x_grid < xx.min()) | (x_grid > xx.max())] = np.nan
        rows.append(row)
        meta.append({"slice": name, "y": y_val, "x_min": float(xx.min()), "x_max": float(xx.max()), "n": int(len(xx))})

    if not rows:
        return np.full((len(y_grid), len(x_grid)), np.nan), pd.DataFrame()
    meta_df = pd.DataFrame(meta).sort_values("y").reset_index(drop=True)
    slice_mat = np.asarray(rows, dtype=float)[meta_df.index]
    obs_y = meta_df["y"].to_numpy(dtype=float)
    out = np.full((len(y_grid), len(x_grid)), np.nan)
    for j in range(len(x_grid)):
        vals = slice_mat[:, j]
        ok = np.isfinite(vals)
        if ok.sum() >= 3:
            fn = PchipInterpolator(obs_y[ok], vals[ok], extrapolate=False)
            z = fn(y_grid)
            z[(y_grid < obs_y[ok].min()) | (y_grid > obs_y[ok].max())] = np.nan
            out[:, j] = z
        elif ok.sum() == 2:
            z = np.interp(y_grid, obs_y[ok], vals[ok], left=np.nan, right=np.nan)
            z[(y_grid < obs_y[ok].min()) | (y_grid > obs_y[ok].max())] = np.nan
            out[:, j] = z
    return out, meta_df


__all__ = [
    "bspline_basis",
    "bspline_knots",
    "second_difference_matrix",
    "slice_pchip_grid",
    "tensor_spline_fit",
    "tensor_spline_values",
    "tensor_spline_values_jax",
]
