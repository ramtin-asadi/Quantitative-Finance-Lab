from __future__ import annotations

import math
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.fixed_income import discounting

from . import bsm

IV_STATUS = {0: "ok", 1: "nan_input", 2: "bounds_violation", 3: "no_convergence"}
_NUMBA_SOLVER = None


def _is_call(option_type) -> np.ndarray:
    arr = np.asarray(option_type)
    return np.isin(np.char.upper(arr.astype(str)), ["C", "CE", "CALL"])


def _as_array(x, n: int | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0 and n is not None:
        arr = np.full(n, float(arr), dtype=float)
    return arr.astype(float, copy=False)


def _broadcast_inputs(option_type, price, forward, strike, tau, discount_factor):
    price_arr = np.asarray(price, dtype=float)
    n = int(price_arr.size) if price_arr.ndim else 1
    opt = np.asarray(option_type)
    if opt.ndim == 0:
        opt = np.full(n, opt.item(), dtype=object)
    is_call = _is_call(opt).reshape(-1)
    arrays = np.broadcast_arrays(
        _as_array(price).reshape(-1),
        _as_array(forward, n).reshape(-1),
        _as_array(strike, n).reshape(-1),
        _as_array(tau, n).reshape(-1),
        _as_array(discount_factor, n).reshape(-1),
    )
    if len(is_call) == 1 and len(arrays[0]) > 1:
        is_call = np.full(len(arrays[0]), bool(is_call[0]), dtype=bool)
    return is_call, arrays


def _initial_guess(price, low, discount_factor, forward, strike, tau, solver: str) -> np.ndarray:
    time_value = np.maximum(np.asarray(price, dtype=float) - np.asarray(low, dtype=float), 1e-12)
    scale = np.maximum(
        np.asarray(discount_factor, dtype=float)
        * np.maximum(np.asarray(forward, dtype=float), np.asarray(strike, dtype=float)),
        1e-12,
    )
    guess = np.sqrt(2.0 * np.pi / np.maximum(np.asarray(tau, dtype=float), 1e-12)) * (time_value / scale)
    if solver == "lbr_lite":
        x = np.abs(np.log(np.clip(forward, 1e-12, None) / np.clip(strike, 1e-12, None)))
        guess = guess * (1.0 + 0.35 * x + 0.08 * x * x)
    return np.clip(guess, 0.01, 4.0)


def _solve_one_newton_bisection(
    is_call: bool,
    price: float,
    forward: float,
    strike: float,
    tau: float,
    discount_factor: float,
    solver: str,
    vol_lower: float,
    vol_upper: float,
    tol: float,
    max_iter: int,
) -> tuple[float, int, int]:
    if not all(np.isfinite(x) for x in [price, forward, strike, tau, discount_factor]):
        return np.nan, 1, 0
    if price <= 0 or forward <= 0 or strike <= 0 or tau <= 0 or discount_factor <= 0:
        return np.nan, 1, 0

    low, high = bsm.no_arbitrage_bounds("call" if is_call else "put", forward, strike, discount_factor)
    low = float(np.asarray(low).reshape(-1)[0])
    high = float(np.asarray(high).reshape(-1)[0])
    if price < low - 1e-10 or price > high + 1e-10:
        return np.nan, 2, 0
    if abs(price - low) < tol:
        return float(vol_lower), 0, 0

    lo = float(vol_lower)
    hi = float(vol_upper)
    guess = float(_initial_guess(price, low, discount_factor, forward, strike, tau, solver))
    sigma = float(np.clip(guess, lo, hi))
    option_type = "call" if is_call else "put"
    last_diff = np.nan

    for iteration in range(1, max_iter + 1):
        model = float(bsm.black76_price(option_type, forward, strike, tau, sigma, discount_factor))
        diff = model - price
        last_diff = diff
        if abs(diff) <= tol:
            return sigma, 0, iteration
        if diff > 0:
            hi = min(hi, sigma)
        else:
            lo = max(lo, sigma)

        d1, _ = bsm.d1_d2_forward(forward, strike, tau, sigma)
        vega = float(discount_factor * forward * bsm.norm_pdf(d1) * np.sqrt(tau))
        if solver == "bisection" or (not np.isfinite(vega)) or vega < 1e-10:
            sigma = 0.5 * (lo + hi)
            continue

        nxt = sigma - diff / vega
        if solver == "lbr_lite":
            _, d2 = bsm.d1_d2_forward(forward, strike, tau, sigma)
            volga = vega * float(d1) * float(d2) / max(sigma, 1e-12)
            denom = 1.0 - 0.5 * (diff / max(vega, 1e-12)) * volga / max(vega, 1e-12)
            if np.isfinite(denom) and abs(denom) > 0.35:
                nxt = sigma - (diff / vega) / denom
        sigma = float(nxt) if np.isfinite(nxt) and lo < nxt < hi else 0.5 * (lo + hi)

    if np.isfinite(last_diff) and abs(last_diff) <= max(tol, 1e-6):
        return sigma, 0, max_iter
    return sigma, 3, max_iter


def implied_vol_bisection_python(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor=1.0,
    *,
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
    return_status: bool = False,
):
    """Invert Black-76 implied volatility with a pure-Python bisection solver.

    This wrapper calls the general implied-volatility inverter with the bisection-only
    solver path. It is deterministic and robust for prices inside no-arbitrage bounds,
    but typically slower than Newton or accelerated backends.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    price : array-like
        Observed option prices.
    forward : array-like
        Forward prices for the corresponding expiries.
    strike : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factors.
    vol_lower : float, default=1e-8
        Lower volatility bracket.
    vol_upper : float, default=5.0
        Upper volatility bracket.
    tol : float, default=1e-8
        Price or volatility convergence tolerance used by the solver.
    max_iter : int, default=100
        Maximum number of iterations per contract.
    return_status : bool, default=False
        If True, return volatility, integer status code, and iteration count.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatility values. If ``return_status=True``, returns
        ``(sigma, status, iterations)``. Status code ``0`` indicates success.

    Notes
    -----
    Failed inversions are returned as ``nan`` volatility values. The function accepts
    scalar and vector inputs and preserves scalar output for scalar inputs.
    """

    return implied_vol_newton_bisection_python(
        option_type,
        price,
        forward,
        strike,
        tau,
        discount_factor,
        solver="bisection",
        vol_lower=vol_lower,
        vol_upper=vol_upper,
        tol=tol,
        max_iter=max_iter,
        return_status=return_status,
    )


def implied_vol_newton_bisection_python(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor=1.0,
    *,
    solver: str = "newton_bisection",
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
    tol: float = 1e-8,
    max_iter: int = 100,
    return_status: bool = False,
):
    """Invert Black-76 implied volatility with pure-Python scalar solvers.

    The function supports Newton-bisection, bisection-only, and a lightweight
    Li-like Halley/Newton-bisection hybrid. Inputs are broadcast to a common shape,
    and each contract is solved independently against the forward Black-Scholes price.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    price : array-like
        Observed option prices.
    forward : array-like
        Forward prices.
    strike : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factors.
    solver : {'newton_bisection', 'bisection', 'lbr_lite'}, default='newton_bisection'
        Root-finding method.
    vol_lower : float, default=1e-8
        Lower volatility bracket.
    vol_upper : float, default=5.0
        Upper volatility bracket.
    tol : float, default=1e-8
        Solver tolerance.
    max_iter : int, default=100
        Maximum iterations per contract.
    return_status : bool, default=False
        If True, include integer status codes and iteration counts.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatilities, optionally with status and iteration arrays.

    Raises
    ------
    ValueError
        If ``solver`` is not one of the supported solver names.

    Notes
    -----
    Status code ``0`` indicates a successful inversion. Nonzero statuses are mapped to
    ``nan`` volatility values in the returned volatility array.
    """

    solver = str(solver).lower()
    if solver not in {"newton_bisection", "bisection", "lbr_lite"}:
        raise ValueError("solver must be one of {'newton_bisection', 'bisection', 'lbr_lite'}.")

    is_call, arrays = _broadcast_inputs(option_type, price, forward, strike, tau, discount_factor)
    price_arr, fwd_arr, strike_arr, tau_arr, df_arr = arrays
    n = len(price_arr)
    sigma = np.full(n, np.nan, dtype=float)
    status = np.full(n, 1, dtype=int)
    iterations = np.zeros(n, dtype=int)

    for i in range(n):
        sigma[i], status[i], iterations[i] = _solve_one_newton_bisection(
            bool(is_call[i]),
            float(price_arr[i]),
            float(fwd_arr[i]),
            float(strike_arr[i]),
            float(tau_arr[i]),
            float(df_arr[i]),
            solver,
            vol_lower,
            vol_upper,
            tol,
            max_iter,
        )
        if status[i] != 0:
            sigma[i] = np.nan

    if np.asarray(price).ndim == 0 and np.asarray(option_type).ndim == 0:
        if return_status:
            return float(sigma[0]), int(status[0]), int(iterations[0])
        return float(sigma[0])
    if return_status:
        return sigma, status, iterations
    return sigma


def _get_numba_solver():
    global _NUMBA_SOLVER
    if _NUMBA_SOLVER is not None:
        return _NUMBA_SOLVER
    try:
        from numba import njit, prange
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Numba IV engine requested but Numba is not available. "
            "Install with `pip install -e .[speed]` or use engine='numpy'/'auto'."
        ) from exc

    @njit
    def norm_cdf_nb(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @njit
    def norm_pdf_nb(x):
        return 0.3989422804014327 * math.exp(-0.5 * x * x)

    @njit
    def price_nb(is_call, df, fwd, strike, tau, sigma):
        sigma = max(sigma, 1e-12)
        tau = max(tau, 1e-12)
        sqrt_tau = math.sqrt(tau)
        log_fk = math.log(max(fwd, 1e-300) / max(strike, 1e-300))
        d1 = (log_fk + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        if is_call:
            return df * (fwd * norm_cdf_nb(d1) - strike * norm_cdf_nb(d2))
        return df * (strike * norm_cdf_nb(-d2) - fwd * norm_cdf_nb(-d1))

    @njit
    def vega_nb(df, fwd, strike, tau, sigma):
        sigma = max(sigma, 1e-12)
        tau = max(tau, 1e-12)
        sqrt_tau = math.sqrt(tau)
        log_fk = math.log(max(fwd, 1e-300) / max(strike, 1e-300))
        d1 = (log_fk + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
        return df * fwd * norm_pdf_nb(d1) * sqrt_tau

    @njit
    def volga_nb(df, fwd, strike, tau, sigma):
        sigma = max(sigma, 1e-12)
        tau = max(tau, 1e-12)
        sqrt_tau = math.sqrt(tau)
        log_fk = math.log(max(fwd, 1e-300) / max(strike, 1e-300))
        d1 = (log_fk + 0.5 * sigma * sigma * tau) / (sigma * sqrt_tau)
        d2 = d1 - sigma * sqrt_tau
        return vega_nb(df, fwd, strike, tau, sigma) * d1 * d2 / sigma

    @njit
    def bounds_nb(is_call, df, fwd, strike):
        if is_call:
            return df * max(fwd - strike, 0.0), df * fwd
        return df * max(strike - fwd, 0.0), df * strike

    @njit
    def guess_nb(price, low, df, fwd, strike, tau, solver_code):
        time_value = max(price - low, 1e-12)
        scale = max(df * max(fwd, strike), 1e-12)
        sigma = math.sqrt(2.0 * math.pi / max(tau, 1e-12)) * (time_value / scale)
        if solver_code == 2:
            x = abs(math.log(max(fwd, 1e-12) / max(strike, 1e-12)))
            sigma *= 1.0 + 0.35 * x + 0.08 * x * x
        return min(max(sigma, 0.01), 4.0)

    @njit
    def solve_one_nb(is_call, price, fwd, strike, tau, df, solver_code):
        if not (
            math.isfinite(price)
            and math.isfinite(fwd)
            and math.isfinite(strike)
            and math.isfinite(tau)
            and math.isfinite(df)
        ):
            return math.nan, 1, 0
        if price <= 0.0 or fwd <= 0.0 or strike <= 0.0 or tau <= 0.0 or df <= 0.0:
            return math.nan, 1, 0
        low, high = bounds_nb(is_call, df, fwd, strike)
        if price < low - 1e-10 or price > high + 1e-10:
            return math.nan, 2, 0
        if abs(price - low) < 1e-8:
            return 1e-8, 0, 0

        lo = 1e-8
        hi = 5.0
        sigma = guess_nb(price, low, df, fwd, strike, tau, solver_code)
        diff = 0.0
        for iteration in range(1, 101):
            model = price_nb(is_call, df, fwd, strike, tau, sigma)
            diff = model - price
            if abs(diff) <= 1e-8:
                return sigma, 0, iteration
            if diff > 0.0:
                hi = min(hi, sigma)
            else:
                lo = max(lo, sigma)
            v = vega_nb(df, fwd, strike, tau, sigma)
            if solver_code == 0 or (not math.isfinite(v)) or v < 1e-10:
                sigma = 0.5 * (lo + hi)
                continue
            step = diff / v
            nxt = sigma - step
            if solver_code == 2:
                volga = volga_nb(df, fwd, strike, tau, sigma)
                denom = 1.0 - 0.5 * step * volga / max(v, 1e-12)
                if math.isfinite(denom) and abs(denom) > 0.35:
                    nxt = sigma - step / denom
            sigma = nxt if math.isfinite(nxt) and lo < nxt < hi else 0.5 * (lo + hi)
        if abs(diff) <= 1e-6 or (hi - lo) < 1e-7:
            return sigma, 0, 100
        return sigma, 3, 100

    @njit(parallel=True)
    def solve_array_nb(is_call, price, forward, strike, tau, discount_factor, solver_code):
        n = len(price)
        sigma = np.empty(n, dtype=np.float64)
        status = np.empty(n, dtype=np.int64)
        iterations = np.empty(n, dtype=np.int64)
        for i in prange(n):
            sigma_i, status_i, iter_i = solve_one_nb(
                is_call[i],
                price[i],
                forward[i],
                strike[i],
                tau[i],
                discount_factor[i],
                solver_code,
            )
            sigma[i] = sigma_i if status_i == 0 else math.nan
            status[i] = status_i
            iterations[i] = iter_i
        return sigma, status, iterations

    _NUMBA_SOLVER = solve_array_nb
    return _NUMBA_SOLVER


def _implied_vol_numba(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor,
    solver: str,
    return_status: bool,
):
    solve_array_nb = _get_numba_solver()
    price_arr = np.asarray(price, dtype=float).reshape(-1)
    arrays = np.broadcast_arrays(
        price_arr,
        np.asarray(forward, dtype=float).reshape(-1),
        np.asarray(strike, dtype=float).reshape(-1),
        np.asarray(tau, dtype=float).reshape(-1),
        np.asarray(discount_factor, dtype=float).reshape(-1),
    )
    price_arr, forward_arr, strike_arr, tau_arr, df_arr = (np.asarray(a, dtype=float) for a in arrays)
    opt = np.asarray(option_type)
    if opt.ndim == 0:
        opt = np.full(len(price_arr), opt.item(), dtype=object)
    is_call = _is_call(opt).reshape(-1)
    if len(is_call) == 1 and len(price_arr) > 1:
        is_call = np.full(len(price_arr), bool(is_call[0]), dtype=bool)
    solver_key = str(solver).lower()
    solver_code = 0 if solver_key == "bisection" else 2 if solver_key == "lbr_lite" else 1
    sigma, status, iterations = solve_array_nb(
        is_call.astype(np.bool_),
        price_arr,
        forward_arr,
        strike_arr,
        tau_arr,
        df_arr,
        solver_code,
    )
    scalar = np.asarray(price).ndim == 0 and np.asarray(option_type).ndim == 0
    if scalar:
        if return_status:
            return float(sigma[0]), int(status[0]), int(iterations[0])
        return float(sigma[0])
    if return_status:
        return sigma, status, iterations
    return sigma


def implied_vol(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor=1.0,
    *,
    engine: str = "auto",
    solver: str = "lbr_lite",
    return_status: bool = False,
):
    """Invert forward Black-Scholes implied volatility with optional acceleration.

    The public interface selects between accelerated and pure-Python solvers while
    preserving the same pricing convention: option prices are matched against the
    forward Black-Scholes/Black-76 formula using forward prices and discount factors.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    price : array-like
        Observed option prices.
    forward : array-like
        Forward prices.
    strike : array-like
        Strike prices.
    tau : array-like
        Times to expiry in years.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factors.
    engine : {'auto', 'python', 'numpy', 'numba'}, default='auto'
        Solver backend. ``'auto'`` attempts the accelerated backend and falls back to
        the Python/NumPy implementation.
    solver : {'lbr_lite', 'newton_bisection', 'bisection'}, default='lbr_lite'
        Root-finding algorithm.
    return_status : bool, default=False
        If True, return volatility, status, and iteration arrays.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatilities, optionally with solver diagnostics.

    Raises
    ------
    ValueError
        If an unsupported engine is requested.
    """

    engine = str(engine).lower()
    if engine == "numba":
        return _implied_vol_numba(
            option_type,
            price,
            forward,
            strike,
            tau,
            discount_factor,
            solver,
            return_status,
        )
    if engine == "auto":
        try:
            return _implied_vol_numba(
                option_type,
                price,
                forward,
                strike,
                tau,
                discount_factor,
                solver,
                return_status,
            )
        except Exception:
            engine = "numpy"
    if engine in {"numpy", "python"}:
        return implied_vol_newton_bisection_python(
            option_type,
            price,
            forward,
            strike,
            tau,
            discount_factor,
            solver=solver,
            return_status=return_status,
        )
    raise ValueError("engine must be one of {'auto', 'python', 'numpy', 'numba'}.")


def _resolve_engine(engine: str) -> str:
    engine = str(engine).lower()
    if engine == "auto":
        try:
            _get_numba_solver()
            return "numba"
        except Exception:
            return "python"
    if engine in {"python", "numpy"}:
        return "python"
    if engine == "numba":
        _get_numba_solver()
        return "numba"
    raise ValueError("engine must be one of {'auto', 'python', 'numpy', 'numba'}.")


def iv_newton_bisection(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor=1.0,
    *,
    engine: str = "auto",
    return_status: bool = False,
):
    """Invert implied volatility using the Newton-bisection solver.

    Parameters
    ----------
    option_type, price, forward, strike, tau, discount_factor
        Standard forward Black-Scholes implied-volatility inputs.
    engine : {'auto', 'python', 'numpy', 'numba'}, default='auto'
        Numerical backend.
    return_status : bool, default=False
        If True, include solver status and iteration count.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatility values, optionally with diagnostics.
    """

    return implied_vol(
        option_type,
        price,
        forward,
        strike,
        tau,
        discount_factor,
        engine=engine,
        solver="newton_bisection",
        return_status=return_status,
    )


def iv_lbr_lite(
    option_type,
    price,
    forward,
    strike,
    tau,
    discount_factor=1.0,
    *,
    engine: str = "auto",
    return_status: bool = False,
):
    """Invert implied volatility using the LBR-lite solver.

    The LBR-lite path combines a fast higher-order update with bracketing safeguards,
    providing a practical default for large quote panels.

    Parameters
    ----------
    option_type, price, forward, strike, tau, discount_factor
        Standard forward Black-Scholes implied-volatility inputs.
    engine : {'auto', 'python', 'numpy', 'numba'}, default='auto'
        Numerical backend.
    return_status : bool, default=False
        If True, include solver status and iteration count.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatility values, optionally with diagnostics.
    """

    return implied_vol(
        option_type,
        price,
        forward,
        strike,
        tau,
        discount_factor,
        engine=engine,
        solver="lbr_lite",
        return_status=return_status,
    )


def iv_newton_bisection_vectorized(*args, **kwargs):
    """Vectorized compatibility alias for Newton-bisection implied-volatility inversion.

    Parameters
    ----------
    *args, **kwargs
        Arguments forwarded to the Newton-bisection implied-volatility solver.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatility output from the underlying solver.
    """

    return iv_newton_bisection(*args, **kwargs)


def iv_lbr_lite_vectorized(*args, **kwargs):
    """Vectorized compatibility alias for LBR-lite implied-volatility inversion.

    Parameters
    ----------
    *args, **kwargs
        Arguments forwarded to the LBR-lite implied-volatility solver.

    Returns
    -------
    float, numpy.ndarray, or tuple
        Implied volatility output from the underlying solver.
    """

    return iv_lbr_lite(*args, **kwargs)


def _solver_label(solver: str) -> str:
    solver_key = str(solver).lower()
    if solver_key not in {"lbr_lite", "newton_bisection"}:
        raise ValueError("solver must be one of {'lbr_lite', 'newton_bisection'}.")
    return solver_key


def _prepare_quote_inputs(quotes: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    option_col = "option_type" if "option_type" in quotes.columns else "cp"
    if "forward" in quotes.columns:
        forward = quotes["forward"]
    elif "f_hat" in quotes.columns:
        forward = quotes["f_hat"]
    elif {"spot", "rate", "tau"}.issubset(quotes.columns):
        q = quotes["dividend_yield"] if "dividend_yield" in quotes.columns else 0.0
        forward = pd.to_numeric(quotes["spot"], errors="coerce") * np.exp(
            (pd.to_numeric(quotes["rate"], errors="coerce") - q) * pd.to_numeric(quotes["tau"], errors="coerce")
        )
    else:
        raise ValueError("quotes must contain forward/f_hat or spot/rate/tau columns.")

    if "discount_factor" in quotes.columns:
        df = quotes["discount_factor"]
    elif "df" in quotes.columns:
        df = quotes["df"]
    elif "rate" in quotes.columns:
        df = discounting.discount_factor_from_rate(quotes["rate"], quotes["tau"])
    else:
        df = pd.Series(1.0, index=quotes.index)
    return quotes[option_col], pd.Series(forward, index=quotes.index), quotes["strike"], pd.Series(df, index=quotes.index)


def compute_iv_table(
    quotes: pd.DataFrame,
    price_cols: tuple[str, ...] = ("bid", "mid", "ask"),
    *,
    solver: str = "lbr_lite",
    engine: str = "auto",
) -> pd.DataFrame:
    """Compute bid, mid, and ask implied-volatility columns for an option quote table.

    The function prepares option type, forward, strike, expiry horizon, and discount
    factor inputs, solves implied volatility for each requested price column, and
    adds solver-status diagnostics, iteration counts, absolute repricing errors, and
    engine metadata.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Normalized option quote table. Required columns include option type, strike,
        ``tau``, and enough information to identify forward and discount-factor inputs.
    price_cols : tuple[str, ...], default=('bid', 'mid', 'ask')
        Price columns for which implied volatilities should be computed.
    solver : {'lbr_lite', 'newton_bisection', 'bisection'}, default='lbr_lite'
        Implied-volatility solver.
    engine : {'auto', 'numpy', 'python', 'numba'}, default='auto'
        Numerical backend.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with ``iv_<price_col>``, status, success, iteration, and
        absolute price-error columns for each requested price column. If ``mid`` is
        included, convenience columns such as ``iv_success``, ``iv_status``,
        ``iv_iterations``, ``iv_solver``, and ``engine_used`` are also attached.

    Notes
    -----
    Missing price columns are reported explicitly with ``missing_price`` statuses
    rather than causing the whole table computation to fail.
    """

    solver_key = _solver_label(solver)
    out = quotes.copy()
    if "source_index" not in out.columns:
        out["source_index"] = out.index
    option_type, forward, strike, df = _prepare_quote_inputs(out)
    tau = out["tau"]
    engine_used = _resolve_engine(engine)

    for price_col in price_cols:
        iv_col = f"iv_{price_col}"
        status_col = f"{iv_col}_status"
        success_col = f"{iv_col}_success"
        iter_col = f"{iv_col}_iters"
        err_col = f"{iv_col}_abs_price_error"

        if price_col not in out.columns:
            out[iv_col] = np.nan
            out[status_col] = "missing_price"
            out[success_col] = False
            out[iter_col] = 0
            out[err_col] = np.nan
            continue

        if engine_used == "numba":
            sigma, status, iterations = _implied_vol_numba(
                option_type.to_numpy(),
                out[price_col].to_numpy(dtype=float),
                forward.to_numpy(dtype=float),
                pd.to_numeric(strike, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(tau, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df, errors="coerce").to_numpy(dtype=float),
                solver_key,
                True,
            )
        else:
            sigma, status, iterations = implied_vol_newton_bisection_python(
                option_type.to_numpy(),
                out[price_col].to_numpy(dtype=float),
                forward.to_numpy(dtype=float),
                pd.to_numeric(strike, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(tau, errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(df, errors="coerce").to_numpy(dtype=float),
                solver=solver_key,
                return_status=True,
            )

        out[iv_col] = sigma
        out[status_col] = pd.Series(status, index=out.index).map(IV_STATUS).fillna("unknown")
        out[success_col] = np.asarray(status) == 0
        out[iter_col] = iterations
        model = bsm.black76_price(option_type.to_numpy(), forward, strike, tau, out[iv_col], df)
        out[err_col] = np.abs(np.asarray(model, dtype=float) - pd.to_numeric(out[price_col], errors="coerce"))

    if "iv_mid" in out.columns:
        out["iv_success"] = out.get("iv_mid_success", False)
        out["iv_status"] = out.get("iv_mid_status", "unknown")
        out["iv_failure_reason"] = out["iv_status"].where(~out["iv_success"], "")
        out["iv_iterations"] = out.get("iv_mid_iters", 0)
        out["iv_solver"] = solver_key
        out["solver"] = solver_key
        out["engine_used"] = engine_used
    if "price_unit_detected" not in out.columns and "price_unit" in out.columns:
        out["price_unit_detected"] = out["price_unit"]
    if "valuation_currency" not in out.columns and "currency" in out.columns:
        out["valuation_currency"] = out["currency"]
    out.attrs["engine_used"] = engine_used
    out.attrs["solver"] = solver_key
    return out


def compute_implied_vols(quotes: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Compatibility alias for table-level implied-volatility computation.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    **kwargs
        Keyword arguments forwarded to the implied-volatility table builder.

    Returns
    -------
    pandas.DataFrame
        Quote table augmented with implied-volatility columns and solver diagnostics.
    """

    return compute_iv_table(quotes, **kwargs)


def implied_vol_table(
    quotes: pd.DataFrame,
    price_cols: tuple[str, ...] = ("bid", "mid", "ask"),
    *,
    solver: str = "lbr_lite",
    engine: str = "auto",
    **_: Any,
) -> pd.DataFrame:
    """Notebook-friendly wrapper for bid, mid, and ask implied-volatility tables.

    Extra keyword arguments that describe already-standardized columns are ignored,
    which makes the function convenient in pipelines that pass broader configuration
    dictionaries.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    price_cols : tuple[str, ...], default=('bid', 'mid', 'ask')
        Price columns to invert.
    solver : {'lbr_lite', 'newton_bisection', 'bisection'}, default='lbr_lite'
        Implied-volatility solver.
    engine : {'auto', 'numpy', 'python', 'numba'}, default='auto'
        Numerical backend.
    **_ : Any
        Ignored compatibility keywords.

    Returns
    -------
    pandas.DataFrame
        Quote table augmented with implied-volatility and solver-diagnostic columns.
    """

    return compute_iv_table(quotes, price_cols=price_cols, solver=solver, engine=engine)


def implied_vol_bid_mid_ask(quotes: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """Compute bid, mid, and ask implied volatilities for a quote table.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    **kwargs
        Additional keyword arguments forwarded to the implied-volatility table builder.

    Returns
    -------
    pandas.DataFrame
        Quote table with ``iv_bid``, ``iv_mid``, ``iv_ask``, and associated solver
        diagnostics.
    """

    return compute_iv_table(quotes, price_cols=("bid", "mid", "ask"), **kwargs)


def iv_uncertainty_band(
    iv_table: pd.DataFrame,
    low_col: str = "iv_bid",
    mid_col: str = "iv_mid",
    high_col: str = "iv_ask",
) -> pd.DataFrame:
    """Construct an implied-volatility uncertainty band from bid, mid, and ask IV columns.

    Parameters
    ----------
    iv_table : pandas.DataFrame
        Table containing implied-volatility columns.
    low_col : str, default='iv_bid'
        Lower implied-volatility column.
    mid_col : str, default='iv_mid'
        Mid implied-volatility column.
    high_col : str, default='iv_ask'
        Upper implied-volatility column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``iv_table`` with ``iv_low``, ``iv_high``, and ``iv_band`` columns.

    Notes
    -----
    If bid or ask implied volatility is missing, the mid IV is used as a fallback.
    The final low/high values are ordered around the mid value.
    """

    out = iv_table.copy()
    low = out[low_col] if low_col in out.columns else out[mid_col]
    high = out[high_col] if high_col in out.columns else out[mid_col]
    out["iv_low"] = pd.to_numeric(low, errors="coerce").combine_first(out[mid_col])
    out["iv_high"] = pd.to_numeric(high, errors="coerce").combine_first(out[mid_col])
    out["iv_low"] = np.minimum(out["iv_low"], out[mid_col])
    out["iv_high"] = np.maximum(out["iv_high"], out[mid_col])
    out["iv_band"] = out["iv_high"] - out["iv_low"]
    return out


def compare_iv_solvers(
    quotes: pd.DataFrame,
    solvers: Iterable[str] = ("newton_bisection", "lbr_lite"),
    price_col: str = "mid",
    engine: str = "auto",
) -> pd.DataFrame:
    """Benchmark implied-volatility solvers on the same quote table.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    solvers : Iterable[str], default=('newton_bisection', 'lbr_lite')
        Solver names to compare.
    price_col : str, default='mid'
        Price column to invert.
    engine : {'auto', 'numpy', 'python', 'numba'}, default='auto'
        Numerical backend.

    Returns
    -------
    pandas.DataFrame
        Summary table with solver name, backend used, success rate, elapsed time,
        quotes per second, median iterations, and median absolute repricing error.
    """

    rows: list[dict[str, Any]] = []
    for solver in solvers:
        t0 = time.perf_counter()
        table = compute_iv_table(quotes, price_cols=(price_col,), solver=solver, engine=engine)
        elapsed = time.perf_counter() - t0
        iv_col = f"iv_{price_col}"
        success_col = f"{iv_col}_success"
        iter_col = f"{iv_col}_iters"
        err_col = f"{iv_col}_abs_price_error"
        success = table[success_col].to_numpy(dtype=bool)
        rows.append(
            {
                "solver": solver,
                "engine_used": table.attrs.get("engine_used", "unknown"),
                "success_rate": float(np.nanmean(success)) if len(success) else np.nan,
                "elapsed_sec": float(elapsed),
                "quotes_per_sec": float(len(table) / max(elapsed, 1e-12)),
                "median_iterations": float(np.nanmedian(table.loc[success, iter_col])) if success.any() else np.nan,
                "median_abs_price_error": float(np.nanmedian(table.loc[success, err_col])) if success.any() else np.nan,
            },
        )
    return pd.DataFrame(rows)


def weighted_median(values, weights):
    """Compute a finite positive-weight weighted median.

    Parameters
    ----------
    values : array-like
        Values to summarize.
    weights : array-like
        Non-negative weights. Only strictly positive finite weights are used.

    Returns
    -------
    float
        Weighted median, or ``nan`` when no finite positively weighted observations
        are available.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    idx = int(np.searchsorted(np.cumsum(weights), 0.5 * weights.sum()))
    return float(values[np.clip(idx, 0, len(values) - 1)])


def iv_pricing_error_summary(
    iv_table: pd.DataFrame,
    price_col: str = "mid",
    iv_col: str = "iv_mid",
) -> pd.DataFrame:
    """Summarize repricing error from an implied-volatility table.

    The function reprices each option with the specified implied-volatility column and
    compares the model price with the selected market price column.

    Parameters
    ----------
    iv_table : pandas.DataFrame
        Quote table containing option inputs and implied-volatility columns.
    price_col : str, default='mid'
        Market price column used as the reference.
    iv_col : str, default='iv_mid'
        Implied-volatility column used for repricing.

    Returns
    -------
    pandas.DataFrame
        One-row summary with count, median signed error, median absolute error,
        90th-percentile absolute error, and maximum absolute error. Returns an empty
        DataFrame if required columns are missing.
    """

    if iv_col not in iv_table.columns or price_col not in iv_table.columns:
        return pd.DataFrame()
    option_type, forward, strike, df = _prepare_quote_inputs(iv_table)
    model = bsm.black76_price(option_type.to_numpy(), forward, strike, iv_table["tau"], iv_table[iv_col], df)
    err = np.asarray(model, dtype=float) - pd.to_numeric(iv_table[price_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(err)
    return pd.DataFrame(
        [
            {
                "n": int(finite.sum()),
                "median_error": float(np.nanmedian(err)) if finite.any() else np.nan,
                "median_abs_error": float(np.nanmedian(np.abs(err))) if finite.any() else np.nan,
                "p90_abs_error": float(np.nanquantile(np.abs(err[finite]), 0.90)) if finite.any() else np.nan,
                "max_abs_error": float(np.nanmax(np.abs(err))) if finite.any() else np.nan,
            },
        ],
    )


__all__ = [
    "IV_STATUS",
    "compare_iv_solvers",
    "compute_implied_vols",
    "compute_iv_table",
    "implied_vol_table",
    "implied_vol",
    "implied_vol_bid_mid_ask",
    "implied_vol_bisection_python",
    "implied_vol_newton_bisection_python",
    "iv_lbr_lite",
    "iv_lbr_lite_vectorized",
    "iv_newton_bisection",
    "iv_newton_bisection_vectorized",
    "iv_pricing_error_summary",
    "iv_uncertainty_band",
    "weighted_median",
]
