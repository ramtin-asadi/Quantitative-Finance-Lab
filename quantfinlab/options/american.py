from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab._optional import get_cpp_kernels, prefer_auto_engine
from quantfinlab.options.bsm import bsm_price


def _option_flag(option_type) -> np.ndarray:
    arr = np.asarray(option_type)
    if np.issubdtype(arr.dtype, np.number):
        return np.where(arr.astype(float) > 0, 1, -1).astype(np.int32)
    text = np.char.lower(arr.astype(str))
    return np.where(np.char.startswith(text, "c"), 1, -1).astype(np.int32)


def _payoff(s, k, option_flag):
    s_arr, k_arr = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(k, dtype=float))
    return np.where(option_flag > 0, np.maximum(s_arr - k_arr, 0.0), np.maximum(k_arr - s_arr, 0.0))


def _tree_ud_p(r: float, q: float, sigma: float, dt: float, tree_type: str) -> tuple[float, float, float]:
    drift = math.exp((r - q) * dt)
    if str(tree_type).lower().startswith("tian"):
        v = math.exp(sigma * sigma * dt)
        root = math.sqrt(max(v * v + 2.0 * v - 3.0, 0.0))
        u = 0.5 * drift * v * (v + 1.0 + root)
        d = 0.5 * drift * v * (v + 1.0 - root)
        if not (u > d and math.isfinite(u) and math.isfinite(d)):
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
    else:
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
    p = (drift - d) / (u - d)
    return u, d, min(max(p, 1e-10), 1.0 - 1e-10)


_TREE_NUMBA = None
_PDE_NUMBA = None


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


def _get_tree_numba():
    global _TREE_NUMBA
    if _TREE_NUMBA is not None:
        return _TREE_NUMBA
    try:
        from numba import njit, prange
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba tree engine requested but Numba is not available.") from exc

    @njit(cache=True)
    def payoff_nb(s, k, flag):
        if flag > 0:
            return max(s - k, 0.0)
        return max(k - s, 0.0)

    @njit(cache=True)
    def ud_p_nb(r, q, sigma, dt, tree_code):
        drift = math.exp((r - q) * dt)
        if tree_code == 1:
            v = math.exp(sigma * sigma * dt)
            root = math.sqrt(max(v * v + 2.0 * v - 3.0, 0.0))
            u = 0.5 * drift * v * (v + 1.0 + root)
            d = 0.5 * drift * v * (v + 1.0 - root)
            if (not math.isfinite(u)) or (not math.isfinite(d)) or u <= d:
                u = math.exp(sigma * math.sqrt(dt))
                d = 1.0 / u
        else:
            u = math.exp(sigma * math.sqrt(dt))
            d = 1.0 / u
        p = (drift - d) / (u - d)
        p = min(max(p, 1e-10), 1.0 - 1e-10)
        return u, d, p

    @njit(cache=True)
    def tree_one_nb(s, k, r, q, sigma, tau, flag, steps, tree_code, american):
        if s <= 0.0 or k <= 0.0:
            return np.nan
        if tau <= 0.0 or sigma <= 0.0 or steps <= 1:
            return payoff_nb(s, k, flag)
        n = max(2, steps)
        dt = tau / n
        u, d, p = ud_p_nb(r, q, sigma, dt, tree_code)
        disc = math.exp(-r * dt)
        values = np.empty(n + 1, dtype=np.float64)
        for j in range(n + 1):
            st = s * (u**j) * (d ** (n - j))
            values[j] = payoff_nb(st, k, flag)
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
                if american:
                    st = s * (u**j) * (d ** (i - j))
                    ex = payoff_nb(st, k, flag)
                    values[j] = max(ex, cont)
                else:
                    values[j] = cont
        return values[0]

    @njit(cache=True, parallel=True)
    def tree_batch_nb(s, k, r, q, sigma, tau, flag, steps, tree_code, american):
        out = np.empty(s.size, dtype=np.float64)
        for i in prange(s.size):
            out[i] = tree_one_nb(s[i], k[i], r[i], q[i], sigma[i], tau[i], flag[i], steps, tree_code, american)
        return out

    _TREE_NUMBA = tree_batch_nb
    return _TREE_NUMBA


def _get_pde_numba():
    global _PDE_NUMBA
    if _PDE_NUMBA is not None:
        return _PDE_NUMBA
    try:
        from numba import njit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba PDE engine requested but Numba is not available.") from exc

    @njit(cache=True)
    def payoff_nb(x, k, flag):
        if flag > 0:
            return max(x - k, 0.0)
        return max(k - x, 0.0)

    @njit(cache=True)
    def pde_one_nb(s, k, r, q, sigma, tau, flag, s_steps, t_steps, s_max_mult, omega, tol, max_iter, american):
        m = max(20, s_steps)
        n = max(4, t_steps)
        s_max = max(s_max_mult * max(s, k), 1.5 * max(s, k))
        ds = s_max / m
        dt = max(tau, 1e-10) / n
        grid = np.empty(m + 1, dtype=np.float64)
        pay = np.empty(m + 1, dtype=np.float64)
        oldv = np.empty(m + 1, dtype=np.float64)
        newv = np.empty(m + 1, dtype=np.float64)
        rhs = np.empty(m + 1, dtype=np.float64)
        values = np.empty((n + 1, m + 1), dtype=np.float64)
        boundary = np.empty(n + 1, dtype=np.float64)
        residuals = np.empty(n, dtype=np.float64)
        for i in range(m + 1):
            grid[i] = i * ds
            pay[i] = payoff_nb(grid[i], k, flag)
            oldv[i] = pay[i]
            newv[i] = pay[i]
            values[n, i] = oldv[i]
        for j in range(n + 1):
            boundary[j] = np.nan
        boundary[n] = k
        sig2 = sigma * sigma
        for step in range(n - 1, -1, -1):
            t = step * dt
            if flag > 0:
                rhs[0] = 0.0
                rhs[m] = s_max * math.exp(-q * (tau - t)) - k * math.exp(-r * (tau - t))
            else:
                rhs[0] = k * math.exp(-r * (tau - t))
                rhs[m] = 0.0
            rhs[0] = max(rhs[0], pay[0])
            rhs[m] = max(rhs[m], pay[m])
            newv[0] = rhs[0]
            newv[m] = rhs[m]
            for i in range(1, m):
                rhs[i] = oldv[i]
                newv[i] = oldv[i]
            max_update = 0.0
            for _ in range(max_iter):
                max_update = 0.0
                for i in range(1, m):
                    ii = float(i)
                    a = -0.5 * dt * (sig2 * ii * ii - (r - q) * ii)
                    c = -0.5 * dt * (sig2 * ii * ii + (r - q) * ii)
                    diag = 1.0 + dt * (sig2 * ii * ii + r)
                    y = (rhs[i] - a * newv[i - 1] - c * newv[i + 1]) / diag
                    cand = newv[i] + omega * (y - newv[i])
                    if american:
                        cand = max(cand, pay[i])
                    max_update = max(max_update, abs(cand - newv[i]))
                    newv[i] = cand
                if max_update < tol:
                    break
            residuals[step] = max_update
            level = np.nan
            for i in range(1, m):
                if american and abs(newv[i] - pay[i]) < 5e-5 and pay[i] > 0.0:
                    if flag > 0:
                        if math.isnan(level) or grid[i] < level:
                            level = grid[i]
                    else:
                        if math.isnan(level) or grid[i] > level:
                            level = grid[i]
            boundary[step] = level
            for i in range(m + 1):
                oldv[i] = newv[i]
                values[step, i] = oldv[i]
        j = int(math.floor(s / ds))
        j = min(max(j, 0), m - 1)
        w = (s - grid[j]) / ds
        price = oldv[j] * (1.0 - w) + oldv[j + 1] * w
        return price, grid, values, boundary, residuals

    _PDE_NUMBA = pde_one_nb
    return _PDE_NUMBA


def _pde_price_numpy(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    flag: int,
    s_steps: int,
    t_steps: int,
    s_max_mult: float,
    omega: float,
    tol: float,
    max_iter: int,
    american: bool,
) -> dict[str, Any]:
    m = int(max(20, s_steps))
    n = int(max(4, t_steps))
    s_max = max(float(s_max_mult) * max(float(s), float(k)), 1.5 * max(float(s), float(k)))
    grid = np.linspace(0.0, s_max, m + 1)
    ds = grid[1] - grid[0]
    dt = max(float(tau), 1e-10) / n
    pay = _payoff(grid, k, flag)
    oldv = pay.copy()
    newv = pay.copy()
    values = np.empty((n + 1, m + 1), dtype=float)
    values[-1] = oldv
    boundary = np.full(n + 1, np.nan)
    boundary[-1] = k
    residuals = np.empty(n, dtype=float)
    sig2 = float(sigma) ** 2
    for step in range(n - 1, -1, -1):
        t = step * dt
        rhs = oldv.copy()
        if flag > 0:
            rhs[0] = 0.0
            rhs[-1] = s_max * np.exp(-float(q) * (float(tau) - t)) - float(k) * np.exp(-float(r) * (float(tau) - t))
        else:
            rhs[0] = float(k) * np.exp(-float(r) * (float(tau) - t))
            rhs[-1] = 0.0
        rhs[0] = max(rhs[0], pay[0])
        rhs[-1] = max(rhs[-1], pay[-1])
        newv[:] = oldv
        newv[0] = rhs[0]
        newv[-1] = rhs[-1]
        max_update = np.inf
        for _ in range(int(max_iter)):
            old_iter = newv.copy()
            for i in range(1, m):
                ii = float(i)
                a = -0.5 * dt * (sig2 * ii * ii - (float(r) - float(q)) * ii)
                c = -0.5 * dt * (sig2 * ii * ii + (float(r) - float(q)) * ii)
                diag = 1.0 + dt * (sig2 * ii * ii + float(r))
                y = (rhs[i] - a * newv[i - 1] - c * newv[i + 1]) / diag
                cand = newv[i] + float(omega) * (y - newv[i])
                if american:
                    cand = max(cand, pay[i])
                newv[i] = cand
            max_update = float(np.max(np.abs(newv - old_iter)))
            if max_update < float(tol):
                break
        residuals[step] = max_update
        bind = np.isclose(newv, pay, atol=5e-5) & (pay > 0)
        if bind.any():
            boundary[step] = float(grid[bind][0] if flag > 0 else grid[bind][-1])
        oldv[:] = newv
        values[step] = oldv
    j = int(np.clip(np.floor(float(s) / ds), 0, m - 1))
    w = (float(s) - grid[j]) / ds
    price = oldv[j] * (1.0 - w) + oldv[j + 1] * w
    return {
        "price": float(price),
        "s_grid": grid,
        "values": values,
        "boundary": boundary,
        "residuals": residuals,
        "engine_used": "numpy",
    }


def tree_price(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: str | int = "put",
    *,
    steps: int = 200,
    tree_type: str = "crr",
    american: bool = True,
    engine: str = "auto",
) -> float:
    """Price a vanilla European or American option with a recombining binomial tree.

    The function supports call and put options, continuous risk-free and dividend-yield
    inputs, CRR- or Tian-style tree parameters, and optional acceleration through the
    configured numerical backend. For American contracts, early exercise is checked at
    each backward-induction node.

    Parameters
    ----------
    s : float
        Current underlying price. Must be positive for a finite model price.
    k : float
        Strike price. Must be positive for a finite model price.
    r : float
        Continuously compounded annual risk-free rate.
    q : float
        Continuously compounded annual dividend yield or carry adjustment.
    sigma : float
        Annualized volatility in decimal units.
    tau : float
        Time to expiry in years.
    option_type : {'call', 'put', 1, -1}, default='put'
        Option direction. String labels starting with ``'c'`` are treated as calls;
        labels starting with ``'p'`` are treated as puts. Integer flags follow the
        same internal convention used by the vectorized tree routines.
    steps : int, default=200
        Number of time steps in the tree. Values below two are treated as degenerate
        cases and return intrinsic value.
    tree_type : {'crr', 'tian'}, default='crr'
        Binomial tree parameterization. Labels beginning with ``'tian'`` use the Tian
        moment-matching tree; all other values use the CRR-style tree.
    american : bool, default=True
        If True, allow early exercise during backward induction. If False, compute the
        European tree value.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend. ``'auto'`` selects the fastest available backend and falls
        back to the pure Python/NumPy implementation when optional accelerators are
        unavailable.

    Returns
    -------
    float
        Option value. Returns ``nan`` for invalid positive-price inputs and intrinsic
        value for expired or zero-volatility cases.

    Notes
    -----
    Rates and dividend yields are annualized continuous rates. The output is in the
    same currency units as the underlying and strike inputs. The tree is intended for
    plain-vanilla contracts; it does not model discrete dividends, barriers, or path
    dependence.
    """

    resolved = _resolve_engine(engine)
    if resolved in {"numba", "cpp"}:
        return float(
            tree_batch(
                np.asarray([s], dtype=float),
                np.asarray([k], dtype=float),
                np.asarray([r], dtype=float),
                np.asarray([q], dtype=float),
                np.asarray([sigma], dtype=float),
                np.asarray([tau], dtype=float),
                np.asarray([option_type]),
                steps=steps,
                tree_type=tree_type,
                american=american,
                engine=resolved,
            )[0]
        )
    flag = int(_option_flag(option_type).reshape(-1)[0]) if not isinstance(option_type, int) else int(option_type)
    if s <= 0 or k <= 0:
        return np.nan
    if tau <= 0 or sigma <= 0 or steps <= 1:
        return float(_payoff(s, k, flag))
    n = int(max(2, steps))
    dt = float(tau) / n
    u, d, p = _tree_ud_p(float(r), float(q), float(sigma), dt, tree_type)
    disc = math.exp(-float(r) * dt)
    values = np.empty(n + 1, dtype=float)
    for j in range(n + 1):
        st = float(s) * (u**j) * (d ** (n - j))
        values[j] = float(_payoff(st, k, flag))
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american:
                st = float(s) * (u**j) * (d ** (i - j))
                values[j] = max(float(_payoff(st, k, flag)), cont)
            else:
                values[j] = cont
    return float(values[0])


def tree_batch(
    s,
    k,
    r,
    q,
    sigma,
    tau,
    option_type,
    *,
    steps: int = 200,
    tree_type: str = "crr",
    american: bool = True,
    engine: str = "auto",
) -> np.ndarray:
    """Vectorize binomial-tree pricing over arrays of option inputs.

    All numeric inputs are broadcast to a common shape, priced contract by contract,
    and returned as an array with that broadcast shape. The function is suitable for
    large quote panels where strikes, expiries, volatilities, rates, and option types
    vary by row.

    Parameters
    ----------
    s, k, r, q, sigma, tau : array-like
        Broadcastable arrays of underlying price, strike, continuous risk-free rate,
        continuous dividend yield, annualized volatility, and time to expiry in years.
    option_type : array-like or scalar
        Option type labels or integer flags. A scalar option type is broadcast to all
        contracts.
    steps : int, default=200
        Number of time steps used for each tree.
    tree_type : {'crr', 'tian'}, default='crr'
        Tree parameterization.
    american : bool, default=True
        Whether to apply early-exercise checks.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Pricing backend. ``'auto'`` attempts accelerated engines when available and
        falls back to NumPy-compatible pricing.

    Returns
    -------
    numpy.ndarray
        Array of option values with the broadcast input shape.

    Notes
    -----
    The vectorized interface is the preferred entry point for quote-table pricing.
    Invalid contracts produce ``nan`` values rather than stopping the full batch,
    unless a requested explicit backend raises during execution.
    """

    s_arr, k_arr, r_arr, q_arr, sig_arr, tau_arr = np.broadcast_arrays(
        np.asarray(s, dtype=float),
        np.asarray(k, dtype=float),
        np.asarray(r, dtype=float),
        np.asarray(q, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(tau, dtype=float),
    )
    flags = _option_flag(option_type)
    if flags.size == 1 and s_arr.size > 1:
        flags = np.full(s_arr.size, int(flags.reshape(-1)[0]), dtype=np.int32)
    flags = flags.reshape(-1).astype(np.int32)
    tree_code = 1 if str(tree_type).lower().startswith("tian") else 0
    resolved = _resolve_engine(engine)
    if resolved == "numba":
        try:
            tree_nb = _get_tree_numba()
            return tree_nb(
                np.ascontiguousarray(s_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(k_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(r_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(q_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(sig_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(tau_arr.reshape(-1), dtype=np.float64),
                np.ascontiguousarray(flags, dtype=np.int32),
                int(steps),
                int(tree_code),
                bool(american),
            ).reshape(s_arr.shape)
        except Exception:
            if str(engine).lower() == "numba":
                raise
            resolved = "numpy"
    if resolved == "cpp":
        kernels = get_cpp_kernels("American tree pricing")
        return np.asarray(
            kernels.american_tree_batch(
                s_arr.reshape(-1),
                k_arr.reshape(-1),
                r_arr.reshape(-1),
                q_arr.reshape(-1),
                sig_arr.reshape(-1),
                tau_arr.reshape(-1),
                flags,
                int(steps),
                int(tree_code),
                bool(american),
            ),
            dtype=float,
        ).reshape(s_arr.shape)
    out = np.empty(s_arr.size, dtype=float)
    for i, vals in enumerate(zip(s_arr.reshape(-1), k_arr.reshape(-1), r_arr.reshape(-1), q_arr.reshape(-1), sig_arr.reshape(-1), tau_arr.reshape(-1), flags, strict=False)):
        out[i] = tree_price(*vals[:6], int(vals[6]), steps=steps, tree_type=tree_type, american=american, engine=resolved)
    return out.reshape(s_arr.shape)


def european_tree_batch(
    s,
    k,
    r,
    q,
    sigma,
    tau,
    option_type,
    *,
    steps: int = 200,
    tree_type: str = "crr",
    engine: str = "auto",
) -> np.ndarray:
    """Price a batch of European vanilla options with a recombining binomial tree.

    This convenience wrapper calls the vectorized tree pricer with early exercise
    disabled. It is useful for comparing tree convergence against closed-form
    Black-Scholes/Black-76 prices or for measuring the early-exercise premium of
    American contracts.

    Parameters
    ----------
    s, k, r, q, sigma, tau : array-like
        Broadcastable option-pricing inputs: underlying price, strike, continuous
        risk-free rate, continuous dividend yield, annualized volatility, and time to
        expiry in years.
    option_type : array-like or scalar
        Option type labels or integer flags.
    steps : int, default=200
        Number of tree steps.
    tree_type : {'crr', 'tian'}, default='crr'
        Tree parameterization.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.

    Returns
    -------
    numpy.ndarray
        European option values with the broadcast input shape.
    """

    return tree_batch(
        s,
        k,
        r,
        q,
        sigma,
        tau,
        option_type,
        steps=steps,
        tree_type=tree_type,
        american=False,
        engine=engine,
    )


def tree_boundary(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: str | int = "put",
    *,
    steps: int = 200,
    tree_type: str = "crr",
    american: bool = True,
    engine: str = "auto",
) -> pd.DataFrame:
    """Extract the early-exercise boundary from a recombining option tree.

    The function prices the option backward through the tree and records the boundary
    level at each time step where immediate exercise dominates continuation value. For
    puts the boundary is the highest exercise stock level; for calls it is the lowest
    exercise stock level under the internal option-flag convention.

    Parameters
    ----------
    s : float
        Current underlying price.
    k : float
        Strike price.
    r : float
        Continuously compounded annual risk-free rate.
    q : float
        Continuously compounded annual dividend yield.
    sigma : float
        Annualized volatility.
    tau : float
        Time to expiry in years.
    option_type : {'call', 'put', 1, -1}, default='put'
        Option type.
    steps : int, default=200
        Number of tree steps.
    tree_type : {'crr', 'tian'}, default='crr'
        Tree parameterization.
    american : bool, default=True
        Whether to perform early-exercise checks. If False, the boundary is generally
        undefined except at expiry.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``time`` and ``boundary``. Missing boundary values are
        represented by ``nan`` when no early-exercise region is detected at a time
        level.

    Notes
    -----
    The boundary is a numerical diagnostic rather than an analytical exercise rule.
    It becomes more stable as the number of tree steps increases.
    """

    flag = int(_option_flag(option_type).reshape(-1)[0]) if not isinstance(option_type, int) else int(option_type)
    tree_code = 1 if str(tree_type).lower().startswith("tian") else 0
    resolved = _resolve_engine(engine)
    if resolved == "cpp":
        kernels = get_cpp_kernels("American tree exercise-boundary extraction")
        out = kernels.american_tree_boundary(s, k, r, q, sigma, tau, flag, int(steps), int(tree_code), bool(american))
        return pd.DataFrame({"time": np.asarray(out["times"], dtype=float), "boundary": np.asarray(out["boundary"], dtype=float)})
    n = int(max(2, steps))
    dt = float(tau) / n
    u, d, p = _tree_ud_p(float(r), float(q), float(sigma), dt, tree_type)
    disc = math.exp(-float(r) * dt)
    values = np.array([float(_payoff(float(s) * (u**j) * (d ** (n - j)), k, flag)) for j in range(n + 1)])
    boundary = np.full(n + 1, np.nan)
    boundary[-1] = k
    for i in range(n - 1, -1, -1):
        level = np.nan
        for j in range(i + 1):
            st = float(s) * (u**j) * (d ** (i - j))
            ex = float(_payoff(st, k, flag))
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            if american and ex > cont + 1e-10:
                level = st if np.isnan(level) else (min(level, st) if flag > 0 else max(level, st))
            values[j] = max(ex, cont) if american else cont
        boundary[i] = level
    return pd.DataFrame({"time": np.linspace(0.0, tau, n + 1), "boundary": boundary})


def pde_price(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: str | int = "put",
    *,
    s_steps: int = 160,
    t_steps: int = 120,
    s_max_mult: float = 3.0,
    omega: float = 1.35,
    tol: float = 1e-7,
    max_iter: int = 5000,
    american: bool = True,
    engine: str = "auto",
) -> dict[str, Any]:
    """Price a vanilla European or American option with a finite-difference PDE solver.

    The solver builds an asset-price grid, steps backward through time, and applies a
    projected successive-over-relaxation style constraint when American exercise is
    enabled. The returned dictionary contains both the interpolated price and the
    terminal numerical objects needed for diagnostics.

    Parameters
    ----------
    s : float
        Current underlying price.
    k : float
        Strike price.
    r : float
        Continuously compounded annual risk-free rate.
    q : float
        Continuously compounded annual dividend yield.
    sigma : float
        Annualized volatility.
    tau : float
        Time to expiry in years.
    option_type : {'call', 'put', 1, -1}, default='put'
        Option type.
    s_steps : int, default=160
        Number of asset-price grid intervals.
    t_steps : int, default=120
        Number of time steps.
    s_max_mult : float, default=3.0
        Upper asset-grid bound as a multiple of the larger of spot and strike.
    omega : float, default=1.35
        Relaxation parameter used by the iterative obstacle solver.
    tol : float, default=1e-7
        Convergence tolerance for the iterative linear solve.
    max_iter : int, default=5000
        Maximum number of iterations per time step.
    american : bool, default=True
        If True, enforce the early-exercise obstacle.
    engine : {'auto', 'numpy', 'numba', 'cpp'}, default='auto'
        Numerical backend.

    Returns
    -------
    dict
        Dictionary containing at least ``price``, ``s_grid``, ``values``, ``boundary``,
        ``residuals``, and ``engine_used`` when the backend reports it.

    Notes
    -----
    PDE prices are sensitive to grid size, domain truncation, and relaxation settings.
    For production-style analysis, compare this output with tree prices and check the
    residual path and boundary stability.
    """

    flag = int(_option_flag(option_type).reshape(-1)[0]) if not isinstance(option_type, int) else int(option_type)
    resolved = _resolve_engine(engine)
    if resolved == "numba":
        try:
            pde_nb = _get_pde_numba()
            price, s_grid, values, boundary, residuals = pde_nb(
                float(s),
                float(k),
                float(r),
                float(q),
                float(sigma),
                float(tau),
                int(flag),
                int(s_steps),
                int(t_steps),
                float(s_max_mult),
                float(omega),
                float(tol),
                int(max_iter),
                bool(american),
            )
            return {
                "price": float(price),
                "s_grid": np.asarray(s_grid, dtype=float),
                "values": np.asarray(values, dtype=float),
                "boundary": np.asarray(boundary, dtype=float),
                "residuals": np.asarray(residuals, dtype=float),
                "engine_used": "numba",
            }
        except Exception:
            if str(engine).lower() == "numba":
                raise
            resolved = "numpy"
    if resolved == "cpp":
        kernels = get_cpp_kernels("American finite-difference PDE pricing")
        out = kernels.american_pde_psor(
            float(s),
            float(k),
            float(r),
            float(q),
            float(sigma),
            float(tau),
            int(flag),
            int(s_steps),
            int(t_steps),
            float(s_max_mult),
            float(omega),
            float(tol),
            int(max_iter),
            bool(american),
        )
        out["engine_used"] = "cpp"
        return out
    return _pde_price_numpy(
        float(s),
        float(k),
        float(r),
        float(q),
        float(sigma),
        float(tau),
        int(flag),
        int(s_steps),
        int(t_steps),
        float(s_max_mult),
        float(omega),
        float(tol),
        int(max_iter),
        bool(american),
    )


def pde_boundary(result: dict[str, Any], tau: float | None = None) -> pd.DataFrame:
    """Convert a PDE pricing result into a time-indexed exercise-boundary table.

    Parameters
    ----------
    result : dict
        Result dictionary returned by the PDE pricer. The dictionary should contain a
        ``boundary`` array.
    tau : float, optional
        Expiry horizon in years. If omitted, the time grid is normalized to ``[0, 1]``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``time`` and ``boundary``.
    """

    boundary = np.asarray(result.get("boundary", []), dtype=float)
    if tau is None:
        time_grid = np.linspace(0.0, 1.0, len(boundary))
    else:
        time_grid = np.linspace(0.0, float(tau), len(boundary))
    return pd.DataFrame({"time": time_grid, "boundary": boundary})


def american_premium(
    american_price,
    option_type,
    spot,
    strike,
    tau,
    sigma,
    rate=0.0,
    dividend_yield=0.0,
) -> np.ndarray:
    """Compute the American early-exercise premium over the corresponding European price.

    The European benchmark is evaluated with the Black-Scholes model using the same
    spot, strike, time to expiry, volatility, rate, dividend yield, and option type.
    The result is useful for identifying contracts where early exercise materially
    changes valuation.

    Parameters
    ----------
    american_price : array-like
        American model prices.
    option_type : array-like or scalar
        Option type labels.
    spot : array-like
        Current underlying prices.
    strike : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    sigma : array-like
        Annualized volatilities.
    rate : array-like or scalar, default=0.0
        Continuously compounded risk-free rates.
    dividend_yield : array-like or scalar, default=0.0
        Continuously compounded dividend yields.

    Returns
    -------
    numpy.ndarray
        ``american_price - european_price`` with broadcast-compatible shape.
    """

    euro = bsm_price(option_type, spot, strike, tau, sigma, rate=rate, dividend_yield=dividend_yield)
    return np.asarray(american_price, dtype=float) - np.asarray(euro, dtype=float)


def pricing_error(model_price, market_mid) -> np.ndarray:
    """Compute signed pricing errors against observed market mid prices.

    Parameters
    ----------
    model_price : array-like
        Model-implied prices.
    market_mid : array-like
        Observed mid prices or other reference prices.

    Returns
    -------
    numpy.ndarray
        Signed errors ``model_price - market_mid``.

    Notes
    -----
    Positive values indicate model prices above the market reference; negative values
    indicate model prices below the market reference.
    """

    return np.asarray(model_price, dtype=float) - np.asarray(market_mid, dtype=float)


def model_disagreement(*prices) -> np.ndarray:
    """Measure cross-model price dispersion for one or more model-price arrays.

    Parameters
    ----------
    *prices : array-like
        One or more arrays of model prices. ``None`` inputs are ignored.

    Returns
    -------
    numpy.ndarray
        Pointwise range across supplied model prices, computed as
        ``nanmax(prices) - nanmin(prices)`` after broadcasting.

    Notes
    -----
    The output is a simple model-uncertainty proxy. It measures disagreement in price
    units and does not adjust for bid-ask spreads, vega, or quote quality.
    """

    arrays = [np.asarray(x, dtype=float) for x in prices if x is not None]
    if not arrays:
        return np.asarray([], dtype=float)
    stacked = np.vstack([np.ravel(x) for x in np.broadcast_arrays(*arrays)])
    return (np.nanmax(stacked, axis=0) - np.nanmin(stacked, axis=0)).reshape(np.broadcast_arrays(*arrays)[0].shape)


def boundary_distance(spot, boundary, option_type="put") -> np.ndarray:
    """Compute normalized distance between spot and an early-exercise boundary.

    For puts, positive values mean the boundary is above spot. For calls, positive
    values mean spot is above the boundary. The output is scaled by spot so that the
    measure is comparable across underlying price levels.

    Parameters
    ----------
    spot : array-like
        Current underlying prices.
    boundary : array-like
        Exercise-boundary levels.
    option_type : {'call', 'put'} or array-like, default='put'
        Option type used to orient the distance.

    Returns
    -------
    numpy.ndarray
        Normalized boundary distance with the broadcast input shape. Invalid or
        non-positive spot values produce ``nan``.
    """

    spot_arr, bound_arr = np.broadcast_arrays(np.asarray(spot, dtype=float), np.asarray(boundary, dtype=float))
    flag = _option_flag(option_type)
    if flag.size == 1 and spot_arr.size > 1:
        flag = np.full(spot_arr.size, int(flag.reshape(-1)[0]), dtype=np.int32)
    flat_spot = spot_arr.reshape(-1)
    flat_bound = bound_arr.reshape(-1)
    flat_flag = flag.reshape(-1)
    dist = np.full(flat_spot.size, np.nan, dtype=float)
    ok = np.isfinite(flat_spot) & np.isfinite(flat_bound) & (flat_spot > 0.0)
    dist[ok] = np.where(
        flat_flag[ok] > 0,
        (flat_bound[ok] - flat_spot[ok]) / flat_spot[ok],
        (flat_spot[ok] - flat_bound[ok]) / flat_spot[ok],
    )
    return dist.reshape(spot_arr.shape)


def assignment_risk(
    quotes: pd.DataFrame,
    *,
    dividend_col: str = "next_dividend",
    time_value_col: str = "time_value",
    boundary_distance_col: str = "boundary_distance",
    spread_col: str = "rel_spread",
    disagreement_col: str = "model_disagreement",
) -> pd.DataFrame:
    """Score early-assignment risk for option quotes using liquidity, moneyness, and exercise diagnostics.

    The function builds a heuristic risk score from intrinsic value, proximity to the
    exercise boundary, dividend-versus-time-value pressure, low remaining time value,
    ex-dividend proximity when available, spread quality, and model disagreement.
    Scores are clipped to the interval ``[0, 1]``.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table containing at least option type, spot, strike, and mid
        price columns. Optional columns improve the score when present.
    dividend_col : str, default='next_dividend'
        Column containing the next expected cash dividend amount.
    time_value_col : str, default='time_value'
        Column to read or create for option time value.
    boundary_distance_col : str, default='boundary_distance'
        Column containing normalized distance to the exercise boundary.
    spread_col : str, default='rel_spread'
        Relative bid-ask spread column used as a liquidity penalty.
    disagreement_col : str, default='model_disagreement'
        Model price dispersion column used as a model-uncertainty input.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with intermediate score columns and ``assignment_risk``.

    Notes
    -----
    This is a screening heuristic, not an exchange assignment model. It is most useful
    for ranking contracts and flagging positions that require manual review or rolling.
    """

    out = quotes.copy()
    option_type = out.get("option_type", "call").astype(str).str.lower()
    spot = pd.to_numeric(out.get("spot"), errors="coerce")
    strike = pd.to_numeric(out.get("strike"), errors="coerce")
    mid = pd.to_numeric(out.get("mid"), errors="coerce")
    intrinsic = np.where(option_type.str.startswith("c"), np.maximum(spot - strike, 0.0), np.maximum(strike - spot, 0.0))
    if time_value_col not in out.columns:
        out[time_value_col] = mid - intrinsic
    itm_score = (intrinsic / spot.replace(0, np.nan)).clip(lower=0.0, upper=1.0)
    distance = pd.to_numeric(
        out[boundary_distance_col]
        if boundary_distance_col in out.columns
        else pd.Series(np.nan, index=out.index),
        errors="coerce",
    )
    boundary_proximity = (1.0 - (distance.abs() / 0.08)).clip(lower=0.0, upper=1.0).fillna(0.0)
    dividend_gap = (pd.to_numeric(out.get(dividend_col, 0.0), errors="coerce") - pd.to_numeric(out[time_value_col], errors="coerce")).clip(lower=0.0)
    dividend_gap_score = (dividend_gap / spot.replace(0, np.nan) / 0.01).clip(lower=0.0, upper=1.0).fillna(0.0)
    low_time_value_score = (1.0 - (pd.to_numeric(out[time_value_col], errors="coerce") / spot.replace(0, np.nan) / 0.015)).clip(lower=0.0, upper=1.0).fillna(0.0)
    if "days_to_next_dividend" in out.columns:
        ex_days = pd.to_numeric(out["days_to_next_dividend"], errors="coerce")
        ex_div_proximity = (1.0 - (ex_days / 14.0)).clip(lower=0.0, upper=1.0).fillna(0.0)
    else:
        ex_div_proximity = pd.Series(0.0, index=out.index)
    spread_penalty = pd.to_numeric(out.get(spread_col, 0.0), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    disagreement = pd.to_numeric(out.get(disagreement_col, 0.0), errors="coerce").fillna(0.0)
    model_uncertainty = (disagreement / spot.replace(0, np.nan) / 0.02).clip(lower=0.0, upper=1.0).fillna(0.0)
    out["itm_score"] = np.asarray(itm_score.fillna(0.0), dtype=float)
    out["boundary_proximity"] = np.asarray(boundary_proximity, dtype=float)
    out["dividend_gap"] = np.asarray(dividend_gap.fillna(0.0), dtype=float)
    out["dividend_gap_score"] = np.asarray(dividend_gap_score, dtype=float)
    out["low_time_value_score"] = np.asarray(low_time_value_score, dtype=float)
    out["ex_div_proximity"] = np.asarray(ex_div_proximity, dtype=float)
    out["spread_penalty"] = np.asarray(spread_penalty, dtype=float)
    out["model_uncertainty_score"] = np.asarray(model_uncertainty, dtype=float)
    risk = (
        0.20 * out["itm_score"]
        + 0.25 * out["boundary_proximity"]
        + 0.25 * out["dividend_gap_score"]
        + 0.15 * out["low_time_value_score"]
        + 0.10 * out["ex_div_proximity"]
        + 0.05 * out["model_uncertainty_score"]
    )
    out["assignment_risk"] = np.asarray(risk.clip(lower=0.0, upper=1.0), dtype=float)
    return out


def roll_signal(
    quotes: pd.DataFrame,
    *,
    risk_col: str = "assignment_risk",
    dte_col: str = "dte_days",
    spread_col: str = "rel_spread",
    threshold: float = 1.0,
) -> pd.DataFrame:
    """Create a roll-urgency score and boolean roll signal from assignment-risk inputs.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table containing assignment-risk, DTE, and spread columns.
    risk_col : str, default='assignment_risk'
        Column containing assignment-risk scores.
    dte_col : str, default='dte_days'
        Column containing days to expiry. If missing, ``'dte'`` is used when present.
    spread_col : str, default='rel_spread'
        Relative-spread column.
    threshold : float, default=1.0
        Roll-urgency cutoff for setting ``roll_signal`` to True.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with ``roll_urgency`` and ``roll_signal`` columns.
    """

    out = quotes.copy()
    risk = pd.to_numeric(out.get(risk_col, 0.0), errors="coerce").fillna(0.0)
    dte = pd.to_numeric(out.get(dte_col, out.get("dte", 30.0)), errors="coerce").fillna(30.0)
    spread = pd.to_numeric(out.get(spread_col, 0.0), errors="coerce").fillna(0.0)
    out["roll_urgency"] = risk + 0.04 * np.maximum(21.0 - dte, 0.0) + 0.5 * spread
    out["roll_signal"] = out["roll_urgency"] >= float(threshold)
    return out


def speed_table(fn, sizes: list[int], repeats: int = 3) -> pd.DataFrame:
    """Benchmark a callable across a sequence of problem sizes.

    Parameters
    ----------
    fn : callable
        Function that accepts a single integer size argument.
    sizes : list[int]
        Problem sizes to evaluate.
    repeats : int, default=3
        Number of repetitions per size. The reported runtime is the median.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``n``, ``seconds``, and ``runs_per_sec``.

    Notes
    -----
    This helper is intended for lightweight backend and vectorization diagnostics. It
    does not isolate warm-up, compilation, or garbage-collection effects.
    """

    rows = []
    for n in sizes:
        times = []
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            fn(int(n))
            times.append(time.perf_counter() - t0)
        rows.append({"n": int(n), "seconds": float(np.median(times)), "runs_per_sec": float(1.0 / max(np.median(times), 1e-12))})
    return pd.DataFrame(rows)


__all__ = [
    "american_premium",
    "assignment_risk",
    "boundary_distance",
    "european_tree_batch",
    "model_disagreement",
    "pde_boundary",
    "pde_price",
    "pricing_error",
    "roll_signal",
    "speed_table",
    "tree_batch",
    "tree_boundary",
    "tree_price",
]
