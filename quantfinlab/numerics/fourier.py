from __future__ import annotations

import numpy as np

_DIRECT_NUMBA = None
_DIRECT_GRID_NUMBA = None
_COS_NUMBA = None
_FFT_NUMBA = None


def _get_kernels():
    global _DIRECT_NUMBA, _DIRECT_GRID_NUMBA, _COS_NUMBA, _FFT_NUMBA
    if _DIRECT_NUMBA is not None and _DIRECT_GRID_NUMBA is not None and _COS_NUMBA is not None and _FFT_NUMBA is not None:
        return _DIRECT_NUMBA, _DIRECT_GRID_NUMBA, _COS_NUMBA, _FFT_NUMBA
    try:
        from numba import njit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba Fourier engine requested but Numba is not available.") from exc

    @njit(cache=True)
    def cf_model(model_id, u, params, s, r, q, tau):
        i = 1j
        log_s = np.log(s)
        if model_id == 1:
            sigma = params[0]
            lam = params[1]
            muj = params[2]
            sj = params[3]
            omega = -lam * (np.exp(muj + 0.5 * sj * sj) - 1.0)
            return np.exp(
                i * u * (log_s + (r - q + omega - 0.5 * sigma * sigma) * tau)
                - 0.5 * sigma * sigma * u * u * tau
                + lam * tau * (np.exp(i * u * muj - 0.5 * sj * sj * u * u) - 1.0)
            )
        if model_id == 2:
            sigma = params[0]
            theta = params[1]
            nu = max(params[2], 1e-8)
            mart = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
            if mart <= 0.0:
                return np.nan + 1j * np.nan
            omega = np.log(mart) / nu
            return np.exp(i * u * (log_s + (r - q + omega) * tau)) * (
                1.0 - i * theta * nu * u + 0.5 * sigma * sigma * nu * u * u
            ) ** (-tau / nu)
        if model_id == 3 or model_id == 4:
            v0 = max(params[0], 1e-10)
            kappa = max(params[1], 1e-10)
            theta = max(params[2], 1e-10)
            sigv = max(params[3], 1e-10)
            rho = min(max(params[4], -0.999), 0.999)
            drift_adj = 0.0
            jump = 1.0 + 0.0j
            if model_id == 4:
                lam = params[5]
                muj = params[6]
                sj = params[7]
                drift_adj = -lam * (np.exp(muj + 0.5 * sj * sj) - 1.0)
                jump = np.exp(lam * tau * (np.exp(i * u * muj - 0.5 * sj * sj * u * u) - 1.0))
            d = np.sqrt((rho * sigv * i * u - kappa) ** 2 + sigv * sigv * (i * u + u * u))
            g = (kappa - rho * sigv * i * u - d) / (kappa - rho * sigv * i * u + d)
            exp_dt = np.exp(-d * tau)
            c = (
                i * u * (log_s + (r - q + drift_adj) * tau)
                + (kappa * theta / (sigv * sigv))
                * ((kappa - rho * sigv * i * u - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
            )
            dcoef = ((kappa - rho * sigv * i * u - d) / (sigv * sigv)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
            return np.exp(c + dcoef * v0) * jump
        sigma = params[0]
        return np.exp(i * u * (log_s + (r - q - 0.5 * sigma * sigma) * tau) - 0.5 * sigma * sigma * u * u * tau)

    @njit(cache=True)
    def variance_rate_nb(model_id, params, tau):
        if model_id == 1:
            sigma = params[0]
            lam = max(params[1], 0.0)
            muj = params[2]
            sj = max(params[3], 1e-10)
            return sigma * sigma + lam * (muj * muj + sj * sj)
        if model_id == 2:
            sigma = params[0]
            theta = params[1]
            nu = max(params[2], 1e-10)
            return sigma * sigma + theta * theta * nu
        if model_id == 3 or model_id == 4:
            v0 = max(params[0], 1e-10)
            kappa = max(params[1], 1e-10)
            theta = max(params[2], 1e-10)
            if tau <= 1e-10:
                base = v0
            else:
                base = theta + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / (kappa * tau)
            if model_id == 4:
                lam = max(params[5], 0.0)
                muj = params[6]
                sj = max(params[7], 1e-10)
                base += lam * (muj * muj + sj * sj)
            return base
        sigma = params[0]
        return sigma * sigma

    @njit(cache=True)
    def call_price_nb(model_id, params, s, k, r, q, tau, n, u_max):
        pi = np.pi
        i = 1j
        log_k = np.log(k)
        du = u_max / n
        phi_mi = cf_model(model_id, -1j, params, s, r, q, tau)
        p1 = 0.5
        p2 = 0.5
        for j in range(1, n + 1):
            u = (j - 0.5) * du
            uc = u + 0.0j
            e = np.exp(-i * uc * log_k)
            den = i * uc
            p1 += du / pi * np.real(e * cf_model(model_id, uc - i, params, s, r, q, tau) / (den * phi_mi))
            p2 += du / pi * np.real(e * cf_model(model_id, uc, params, s, r, q, tau) / den)
        return max(s * np.exp(-q * tau) * p1 - k * np.exp(-r * tau) * p2, 0.0)

    @njit(cache=True)
    def direct_batch_nb(model_id, params, s, k, r, q, tau, flag, n, u_max):
        out = np.empty(k.size, dtype=np.float64)
        for idx in range(k.size):
            call = call_price_nb(model_id, params, s[idx], k[idx], r[idx], q[idx], tau[idx], n, u_max)
            if flag[idx] > 0:
                out[idx] = call
            else:
                out[idx] = call - s[idx] * np.exp(-q[idx] * tau[idx]) + k[idx] * np.exp(-r[idx] * tau[idx])
        return out

    @njit(cache=True)
    def direct_grid_nb(model_id, params, s, k, r, q, tau, flag, n, u_max):
        out = np.empty(k.size, dtype=np.float64)
        p1 = np.empty(k.size, dtype=np.float64)
        p2 = np.empty(k.size, dtype=np.float64)
        pi = np.pi
        i = 1j
        du = u_max / n
        phi_mi = cf_model(model_id, -1j, params, s, r, q, tau)
        for idx in range(k.size):
            p1[idx] = 0.5
            p2[idx] = 0.5
        for j in range(1, n + 1):
            u = (j - 0.5) * du
            uc = u + 0.0j
            den = i * uc
            a1 = cf_model(model_id, uc - i, params, s, r, q, tau) / (den * phi_mi)
            a2 = cf_model(model_id, uc, params, s, r, q, tau) / den
            weight = du / pi
            for idx in range(k.size):
                e = np.exp(-i * uc * np.log(k[idx]))
                p1[idx] += weight * np.real(e * a1)
                p2[idx] += weight * np.real(e * a2)
        stock_disc = s * np.exp(-q * tau)
        bond_disc = np.exp(-r * tau)
        for idx in range(k.size):
            call = stock_disc * p1[idx] - k[idx] * bond_disc * p2[idx]
            if call < 0.0:
                call = 0.0
            if flag[idx] > 0:
                out[idx] = call
            else:
                out[idx] = call - stock_disc + k[idx] * bond_disc
        return out

    @njit(cache=True)
    def chi_psi_nb(n, a, b, c, d):
        if d <= c:
            return 0.0, 0.0
        width = b - a
        if n == 0:
            psi = d - c
            chi = np.exp(d) - np.exp(c)
            return chi, psi
        u = n * np.pi / width
        xd = u * (d - a)
        xc = u * (c - a)
        chi = (
            np.cos(xd) * np.exp(d)
            - np.cos(xc) * np.exp(c)
            + u * (np.sin(xd) * np.exp(d) - np.sin(xc) * np.exp(c))
        ) / (1.0 + u * u)
        psi = (np.sin(xd) - np.sin(xc)) / u
        return chi, psi

    @njit(cache=True)
    def cos_one_nb(model_id, params, s, k, r, q, tau, flag, n_terms, width_mult):
        var_rate = max(variance_rate_nb(model_id, params, tau), 1e-10)
        drift = np.log(s / k) + (r - q) * tau
        stdev = np.sqrt(max(var_rate * tau, 1e-10))
        half = max(width_mult * stdev + abs(drift) * 0.10, 0.35)
        a = drift - half
        b = drift + half
        width = b - a
        total = 0.0
        for n in range(n_terms):
            u = n * np.pi / width
            phi = cf_model(model_id, u + 0.0j, params, s, r, q, tau) * np.exp(-1j * u * np.log(k))
            phase = np.exp(-1j * u * a)
            if flag > 0:
                c = max(0.0, a)
                d = b
                chi, psi = chi_psi_nb(n, a, b, c, d)
                coeff = 2.0 * k * (chi - psi) / width
            else:
                c = a
                d = min(0.0, b)
                chi, psi = chi_psi_nb(n, a, b, c, d)
                coeff = 2.0 * k * (psi - chi) / width
            term = np.real(phi * phase) * coeff
            if n == 0:
                term *= 0.5
            total += term
        return max(np.exp(-r * tau) * total, 0.0)

    @njit(cache=True)
    def cos_batch_nb(model_id, params, s, k, r, q, tau, flag, n_terms, width_mult):
        out = np.empty(k.size, dtype=np.float64)
        for idx in range(k.size):
            out[idx] = cos_one_nb(model_id, params, s[idx], k[idx], r[idx], q[idx], tau[idx], flag[idx], n_terms, width_mult)
        return out

    @njit(cache=True)
    def carr_madan_grid_nb(model_id, params, s, r, q, tau, alpha, n, eta, center_log):
        strikes = np.empty(n, dtype=np.float64)
        prices = np.empty(n, dtype=np.float64)
        log_spacing = 2.0 * np.pi / (n * eta)
        half_width = 0.5 * n * log_spacing
        start_log = center_log - half_width
        x = np.empty(n, dtype=np.complex128)
        for j in range(n):
            u = j * eta
            uc = u + 0.0j
            shifted = uc - 1j * (alpha + 1.0)
            denom = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
            weight = 0.5 if j == 0 else 1.0
            psi = np.exp(-r * tau) * cf_model(model_id, shifted, params, s, r, q, tau) / denom
            x[j] = psi * np.exp(-1j * uc * start_log) * eta * weight
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                tmp = x[i]
                x[i] = x[j]
                x[j] = tmp
        length = 2
        while length <= n:
            angle = -2.0 * np.pi / length
            wlen = np.cos(angle) + 1j * np.sin(angle)
            half_len = length >> 1
            for i0 in range(0, n, length):
                w = 1.0 + 0.0j
                for j0 in range(half_len):
                    u0 = x[i0 + j0]
                    v0 = x[i0 + j0 + half_len] * w
                    x[i0 + j0] = u0 + v0
                    x[i0 + j0 + half_len] = u0 - v0
                    w *= wlen
            length <<= 1
        for m in range(n):
            log_k = start_log + m * log_spacing
            strikes[m] = np.exp(log_k)
            prices[m] = max(np.exp(-alpha * log_k) * np.real(x[m]) / np.pi, 0.0)
        return strikes, prices

    _DIRECT_NUMBA = direct_batch_nb
    _DIRECT_GRID_NUMBA = direct_grid_nb
    _COS_NUMBA = cos_batch_nb
    _FFT_NUMBA = carr_madan_grid_nb
    return _DIRECT_NUMBA, _DIRECT_GRID_NUMBA, _COS_NUMBA, _FFT_NUMBA


def direct_price_numba(model_id, params, s, k, r, q, tau, flag, *, n: int = 512, u_max: float = 120.0):
    """Price options by direct Fourier integration using Numba kernels.

    The function dispatches either to a grid-specialized kernel when spot, rate,
    dividend yield, and maturity are common across a strike vector, or to a fully
    vectorized quote kernel otherwise.

    Parameters
    ----------
    model_id : int
        Numeric model identifier.
    params : array-like
        Model parameter vector.
    s, k, r, q, tau : array-like
        Spot, strike, rate, dividend yield, and maturity arrays.
    flag : array-like
        Option-type flags.
    n : int, default=512
        Number of integration grid points.
    u_max : float, default=120.0
        Upper integration limit.

    Returns
    -------
    numpy.ndarray
        Option prices aligned to the broadcast input arrays.
    """
    fn, grid_fn, _, _ = _get_kernels()
    s_arr = np.ascontiguousarray(np.asarray(s, dtype=float).reshape(-1))
    k_arr = np.ascontiguousarray(np.asarray(k, dtype=float).reshape(-1))
    r_arr = np.ascontiguousarray(np.asarray(r, dtype=float).reshape(-1))
    q_arr = np.ascontiguousarray(np.asarray(q, dtype=float).reshape(-1))
    tau_arr = np.ascontiguousarray(np.asarray(tau, dtype=float).reshape(-1))
    flag_arr = np.ascontiguousarray(np.asarray(flag, dtype=np.int32).reshape(-1))
    if (
        k_arr.size > 1
        and np.nanmax(np.abs(s_arr - s_arr[0])) <= 1e-13
        and np.nanmax(np.abs(r_arr - r_arr[0])) <= 1e-13
        and np.nanmax(np.abs(q_arr - q_arr[0])) <= 1e-13
        and np.nanmax(np.abs(tau_arr - tau_arr[0])) <= 1e-13
    ):
        return grid_fn(
            int(model_id),
            np.asarray(params, dtype=np.float64),
            float(s_arr[0]),
            k_arr,
            float(r_arr[0]),
            float(q_arr[0]),
            float(tau_arr[0]),
            flag_arr,
            int(n),
            float(u_max),
        )
    return fn(
        int(model_id),
        np.asarray(params, dtype=np.float64),
        s_arr,
        k_arr,
        r_arr,
        q_arr,
        tau_arr,
        flag_arr,
        int(n),
        float(u_max),
    )


def cos_price_numba(model_id, params, s, k, r, q, tau, flag, *, n_terms: int = 256, truncation_width: float = 10.0):
    """Price options with a COS expansion using Numba kernels.

    Parameters
    ----------
    model_id : int
        Numeric model identifier.
    params : array-like
        Model parameter vector.
    s, k, r, q, tau : array-like
        Spot, strike, rate, dividend yield, and maturity arrays.
    flag : array-like
        Option-type flags.
    n_terms : int, default=256
        Number of COS terms.
    truncation_width : float, default=10.0
        Width of the COS truncation interval.

    Returns
    -------
    numpy.ndarray
        COS model prices.
    """
    _, _, fn, _ = _get_kernels()
    return fn(
        int(model_id),
        np.asarray(params, dtype=np.float64),
        np.ascontiguousarray(np.asarray(s, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(k, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(r, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(q, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(tau, dtype=float).reshape(-1)),
        np.ascontiguousarray(np.asarray(flag, dtype=np.int32).reshape(-1)),
        int(n_terms),
        float(truncation_width),
    )


def carr_madan_fft_numba(model_id, params, spot, rate, dividend_yield, tau, *, alpha: float = 1.5, n: int = 4096, eta: float = 0.35, center_log: float | None = None):
    """Compute a Carr-Madan FFT call-price grid with Numba kernels.

    Parameters
    ----------
    model_id : int
        Numeric model identifier.
    params : array-like
        Model parameter vector.
    spot : float
        Current spot price.
    rate : float
        Continuously compounded risk-free rate.
    dividend_yield : float
        Continuous dividend yield.
    tau : float
        Time to maturity in years.
    alpha : float, default=1.5
        Damping parameter.
    n : int, default=4096
        FFT grid size.
    eta : float, default=0.35
        Fourier-grid spacing.
    center_log : float, optional
        Log-strike center. Defaults to ``log(spot)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(strikes, prices)`` arrays from the FFT grid.
    """
    _, _, _, fn = _get_kernels()
    center = float(np.log(spot) if center_log is None else center_log)
    strikes, prices = fn(
        int(model_id),
        np.asarray(params, dtype=np.float64),
        float(spot),
        float(rate),
        float(dividend_yield),
        float(tau),
        float(alpha),
        int(n),
        float(eta),
        center,
    )
    return strikes, prices


__all__ = ["carr_madan_fft_numba", "cos_price_numba", "direct_price_numba"]
