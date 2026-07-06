from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from quantfinlab.fixed_income import discounting


def _is_call(option_type) -> np.ndarray:
    arr = np.asarray(option_type)
    text = np.char.upper(arr.astype(str))
    return np.isin(text, ["C", "CE", "CALL"])


def _first_index(*values) -> pd.Index | None:
    for value in values:
        if isinstance(value, pd.Series):
            return value.index
    return None


def _wrap(out, *templates):
    index = _first_index(*templates)
    if index is not None:
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 0:
            arr = np.full(len(index), float(arr), dtype=float)
        return pd.Series(arr, index=index)
    if all(np.isscalar(t) for t in templates if t is not None):
        return float(np.asarray(out, dtype=float).reshape(-1)[0])
    return np.asarray(out, dtype=float)


def norm_cdf(x):
    """Evaluate the standard normal cumulative distribution function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    array-like
        Standard normal CDF evaluated elementwise.
    """

    return ndtr(x)


def norm_pdf(x):
    """Evaluate the standard normal probability density function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    numpy.ndarray
        Standard normal density evaluated elementwise.
    """

    arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * arr * arr) / np.sqrt(2.0 * np.pi)


def d1_d2_forward(forward, strike, tau, sigma) -> tuple[np.ndarray, np.ndarray]:
    """Compute Black-76 ``d1`` and ``d2`` terms from forward-price inputs.

    Parameters
    ----------
    forward : array-like
        Forward price of the underlying for the option expiry.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility in decimal units.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Broadcast arrays ``(d1, d2)``.

    Notes
    -----
    Inputs are clipped internally to avoid division by zero and logarithms of
    non-positive values. Downstream pricing functions handle final validity checks.
    """

    fwd, strike_arr, tau_arr, sigma_arr = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(sigma, dtype=float),
    )
    sqrt_tau = np.sqrt(np.clip(tau_arr, 1e-12, None))
    sigma_safe = np.clip(sigma_arr, 1e-12, None)
    log_fk = np.log(np.clip(fwd, 1e-300, None) / np.clip(strike_arr, 1e-300, None))
    d1 = (log_fk + 0.5 * sigma_safe * sigma_safe * np.clip(tau_arr, 1e-12, None)) / (
        sigma_safe * sqrt_tau
    )
    d2 = d1 - sigma_safe * sqrt_tau
    return d1, d2


def intrinsic_value(option_type, underlying, strike):
    """Compute intrinsic value for call or put options.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels. Calls use ``max(underlying - strike, 0)`` and puts use
        ``max(strike - underlying, 0)``.
    underlying : array-like
        Underlying or forward price, depending on the intended convention.
    strike : array-like
        Strike price.

    Returns
    -------
    array-like
        Intrinsic values with broadcasting applied and pandas metadata preserved when
        possible.
    """

    is_call = _is_call(option_type)
    underlying_arr, strike_arr = np.broadcast_arrays(
        np.asarray(underlying, dtype=float),
        np.asarray(strike, dtype=float),
    )
    call = np.maximum(underlying_arr - strike_arr, 0.0)
    put = np.maximum(strike_arr - underlying_arr, 0.0)
    return _wrap(np.where(is_call, call, put), underlying, strike)


def forward_bsm_price(option_type, forward, strike, tau, sigma, discount_factor=1.0):
    """Price vanilla European options with the forward Black-Scholes/Black-76 formula.

    The model prices calls and puts from a forward price and discount factor rather
    than separately modelling spot carry. This is the preferred convention when the
    forward has been inferred directly from put-call parity or an external forward
    curve.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    forward : array-like
        Forward price for the option expiry.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility in decimal units.
    discount_factor : array-like or scalar, default=1.0
        Discount factor from valuation date to expiry.

    Returns
    -------
    array-like
        Option prices in discounted currency units, preserving pandas metadata when
        possible.

    Notes
    -----
    Expired options return expiry intrinsic value. Positive time-to-expiry contracts
    with non-positive volatility return discounted intrinsic value. Invalid forwards,
    strikes, or discount factors return ``nan``.
    """

    is_call = _is_call(option_type)
    fwd, strike_arr, tau_arr, sigma_arr, df_arr = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(sigma, dtype=float),
        np.asarray(discount_factor, dtype=float),
    )
    d1, d2 = d1_d2_forward(fwd, strike_arr, tau_arr, sigma_arr)
    call = df_arr * (fwd * norm_cdf(d1) - strike_arr * norm_cdf(d2))
    put = df_arr * (strike_arr * norm_cdf(-d2) - fwd * norm_cdf(-d1))
    discounted_intrinsic = df_arr * np.where(is_call, np.maximum(fwd - strike_arr, 0.0), np.maximum(strike_arr - fwd, 0.0))
    expiry_intrinsic = np.where(is_call, np.maximum(fwd - strike_arr, 0.0), np.maximum(strike_arr - fwd, 0.0))
    price = np.where(is_call, call, put)
    price = np.where((tau_arr <= 0) | ~np.isfinite(tau_arr), expiry_intrinsic, price)
    price = np.where((sigma_arr <= 0) & (tau_arr > 0), discounted_intrinsic, price)
    invalid = (fwd <= 0) | (strike_arr <= 0) | (df_arr <= 0)
    price = np.where(invalid, np.nan, price)
    return _wrap(price, forward, strike, tau, sigma, discount_factor)


def forward_bsm_call(forward, strike, tau, sigma, discount_factor=1.0):
    """Price European calls with the forward Black-Scholes/Black-76 formula.

    Parameters
    ----------
    forward : array-like
        Forward price for expiry.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Call prices.
    """

    return forward_bsm_price("call", forward, strike, tau, sigma, discount_factor)


def forward_bsm_put(forward, strike, tau, sigma, discount_factor=1.0):
    """Price European puts with the forward Black-Scholes/Black-76 formula.

    Parameters
    ----------
    forward : array-like
        Forward price for expiry.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Put prices.
    """

    return forward_bsm_price("put", forward, strike, tau, sigma, discount_factor)


def forward_bsm_delta(option_type, forward, strike, tau, sigma, discount_factor=1.0):
    """Compute Black-76 delta with respect to the forward price.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Forward delta. Calls return ``DF * N(d1)`` and puts return
        ``DF * (N(d1) - 1)``.
    """

    d1, _ = d1_d2_forward(forward, strike, tau, sigma)
    is_call = _is_call(option_type)
    out = np.asarray(discount_factor, dtype=float) * np.where(is_call, norm_cdf(d1), norm_cdf(d1) - 1.0)
    return _wrap(out, forward, strike, tau, sigma, discount_factor)


def forward_bsm_gamma(forward, strike, tau, sigma, discount_factor=1.0):
    """Compute Black-76 gamma with respect to the forward price.

    Parameters
    ----------
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Forward gamma.
    """

    d1, _ = d1_d2_forward(forward, strike, tau, sigma)
    denom = np.asarray(forward, dtype=float) * np.asarray(sigma, dtype=float) * np.sqrt(np.asarray(tau, dtype=float))
    out = np.asarray(discount_factor, dtype=float) * norm_pdf(d1) / np.clip(denom, 1e-12, None)
    return _wrap(out, forward, strike, tau, sigma, discount_factor)


def forward_bsm_vega(forward, strike, tau, sigma, discount_factor=1.0):
    """Compute Black-76 vega with respect to volatility in decimal units.

    Parameters
    ----------
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Price sensitivity to a one-unit change in volatility, not a one-vol-point
        change. Divide by 100 for sensitivity per one volatility percentage point.
    """

    d1, _ = d1_d2_forward(forward, strike, tau, sigma)
    out = np.asarray(discount_factor, dtype=float) * np.asarray(forward, dtype=float) * norm_pdf(d1) * np.sqrt(
        np.asarray(tau, dtype=float),
    )
    return _wrap(out, forward, strike, tau, sigma, discount_factor)


def forward_bsm_theta(option_type, forward, strike, tau, sigma, discount_factor=1.0, rate=0.0):
    """Compute Black-76 theta under the forward-pricing convention.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.
    rate : array-like or scalar, default=0.0
        Continuously compounded risk-free rate used for the discounting contribution.

    Returns
    -------
    array-like
        Annualized theta under the function's convention.

    Notes
    -----
    The result follows the internal forward-price convention. If spot-carry Greeks are
    required, use the higher-level Greek routines that accept a spot input.
    """

    price = forward_bsm_price(option_type, forward, strike, tau, sigma, discount_factor)
    d1, _ = d1_d2_forward(forward, strike, tau, sigma)
    decay = -0.5 * np.asarray(discount_factor, dtype=float) * np.asarray(forward, dtype=float) * norm_pdf(d1) * np.asarray(
        sigma,
        dtype=float,
    ) / np.sqrt(np.asarray(tau, dtype=float))
    out = decay + np.asarray(rate, dtype=float) * np.asarray(price, dtype=float)
    return _wrap(out, forward, strike, tau, sigma, discount_factor)


def forward_bsm_rho(option_type, forward, strike, tau, sigma, discount_factor=1.0):
    """Compute the discount-factor rho proxy for a forward Black-Scholes price.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    array-like
        Approximate rate sensitivity induced by discounting, equal to
        ``-tau * price`` under the forward-input convention.
    """

    price = forward_bsm_price(option_type, forward, strike, tau, sigma, discount_factor)
    out = -np.asarray(tau, dtype=float) * np.asarray(price, dtype=float)
    return _wrap(out, forward, strike, tau, sigma, discount_factor)


black76_price = forward_bsm_price
black76_call = forward_bsm_call
black76_put = forward_bsm_put
black76_delta = forward_bsm_delta
black76_gamma = forward_bsm_gamma
black76_vega = forward_bsm_vega
black76_theta = forward_bsm_theta
black76_rho = forward_bsm_rho


def bsm_price(
    option_type,
    spot,
    strike,
    tau,
    sigma,
    rate=0.0,
    dividend_yield=0.0,
    forward=None,
    discount_factor=None,
):
    """Price European options with Black-Scholes or forward Black-Scholes inputs.

    If a forward price is supplied, the function prices with the forward Black-76
    formula. Otherwise it builds the forward from spot, continuous rate, dividend
    yield, and time to expiry.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    spot : array-like
        Current underlying spot price. Used only when ``forward`` is not supplied.
    strike : array-like
        Strike price.
    tau : array-like
        Time to expiry in years.
    sigma : array-like
        Annualized volatility.
    rate : array-like or scalar, default=0.0
        Continuously compounded risk-free rate.
    dividend_yield : array-like or scalar, default=0.0
        Continuously compounded dividend yield.
    forward : array-like, optional
        Forward price for expiry. When supplied, it overrides spot-based carry
        construction.
    discount_factor : array-like, optional
        Expiry discount factor. If omitted, it is computed from ``rate`` and ``tau``.

    Returns
    -------
    array-like
        European option prices.
    """

    if discount_factor is None:
        discount_factor = discounting.discount_factor_from_rate(rate, tau)
    if forward is None:
        spot_arr, rate_arr, q_arr, tau_arr = np.broadcast_arrays(
            np.asarray(spot, dtype=float),
            np.asarray(rate, dtype=float),
            np.asarray(dividend_yield, dtype=float),
            np.asarray(tau, dtype=float),
        )
        forward = spot_arr * np.exp((rate_arr - q_arr) * np.clip(tau_arr, 0.0, None))
    return forward_bsm_price(option_type, forward, strike, tau, sigma, discount_factor)


def time_value(option_type, price, forward, strike, discount_factor=1.0):
    """Compute option time value relative to discounted forward intrinsic value.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    price : array-like
        Observed or model option prices.
    forward : array-like
        Forward price used for intrinsic-value calculation.
    strike : array-like
        Strike price.
    discount_factor : array-like or scalar, default=1.0
        Discount factor applied to forward intrinsic value.

    Returns
    -------
    array-like
        ``price - discounted_intrinsic``.
    """

    intrinsic = np.asarray(intrinsic_value(option_type, forward, strike), dtype=float) * np.asarray(
        discount_factor,
        dtype=float,
    )
    return _wrap(np.asarray(price, dtype=float) - intrinsic, price, forward, strike)


def no_arbitrage_bounds(option_type, forward, strike, discount_factor=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward-model no-arbitrage lower and upper price bounds.

    Parameters
    ----------
    option_type : array-like or scalar
        Option type labels.
    forward : array-like
        Forward price.
    strike : array-like
        Strike price.
    discount_factor : array-like or scalar, default=1.0
        Expiry discount factor.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper bounds. The lower bound is discounted intrinsic value. The
        upper bound is ``DF * forward`` for calls and ``DF * strike`` for puts.
    """

    is_call = _is_call(option_type)
    fwd, strike_arr, df_arr = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(discount_factor, dtype=float),
    )
    lower = df_arr * np.where(is_call, np.maximum(fwd - strike_arr, 0.0), np.maximum(strike_arr - fwd, 0.0))
    upper = df_arr * np.where(is_call, fwd, strike_arr)
    return lower, upper


def bsm_cf(u, spot, rate, dividend_yield, tau, sigma):
    """Evaluate the Black-Scholes characteristic function of log spot at expiry.

    Parameters
    ----------
    u : array-like
        Complex Fourier argument.
    spot : array-like or scalar
        Current spot price.
    rate : array-like or scalar
        Continuously compounded risk-free rate.
    dividend_yield : array-like or scalar
        Continuously compounded dividend yield.
    tau : array-like or scalar
        Time to expiry in years.
    sigma : array-like or scalar
        Annualized volatility.

    Returns
    -------
    numpy.ndarray
        Characteristic-function values ``E[exp(i u log(S_T))]``.
    """

    u_arr = np.asarray(u, dtype=complex)
    sigma_arr = np.asarray(sigma, dtype=float)
    mu = np.log(np.asarray(spot, dtype=float)) + (
        np.asarray(rate, dtype=float) - np.asarray(dividend_yield, dtype=float) - 0.5 * sigma_arr * sigma_arr
    ) * np.asarray(tau, dtype=float)
    return np.exp(1j * u_arr * mu - 0.5 * sigma_arr * sigma_arr * u_arr * u_arr * np.asarray(tau, dtype=float))


__all__ = [
    "black76_call",
    "black76_delta",
    "black76_gamma",
    "black76_price",
    "black76_put",
    "black76_rho",
    "black76_theta",
    "black76_vega",
    "bsm_cf",
    "bsm_price",
    "d1_d2_forward",
    "forward_bsm_call",
    "forward_bsm_delta",
    "forward_bsm_gamma",
    "forward_bsm_price",
    "forward_bsm_put",
    "forward_bsm_rho",
    "forward_bsm_theta",
    "forward_bsm_vega",
    "intrinsic_value",
    "no_arbitrage_bounds",
    "norm_cdf",
    "norm_pdf",
    "time_value",
]
