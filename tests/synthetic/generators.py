from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import Curve

ASSETS = ("AAA", "BBB", "CCC", "CASH")


def business_dates(n: int = 80, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=int(n))


def return_panel(n: int = 80, assets: tuple[str, ...] = ASSETS) -> pd.DataFrame:
    idx = business_dates(n)
    base = np.linspace(0.0, 2.0 * np.pi, len(idx))
    data = {}
    for i, asset in enumerate(assets):
        if asset.upper() == "CASH":
            data[asset] = np.full(len(idx), 0.00005)
        else:
            drift = 0.00015 * (i + 1)
            wave = 0.006 * np.sin(base + 0.55 * i)
            shock = 0.002 * np.cos(2.0 * base - 0.25 * i)
            data[asset] = drift + wave + shock
    return pd.DataFrame(data, index=idx)


def price_panel(n: int = 80, assets: tuple[str, ...] = ASSETS) -> pd.DataFrame:
    rets = return_panel(n=n, assets=assets)
    start = pd.Series({asset: 100.0 + 7.0 * i for i, asset in enumerate(assets)})
    return start * (1.0 + rets).cumprod()


def volume_panel(n: int = 80, assets: tuple[str, ...] = ASSETS) -> pd.DataFrame:
    idx = business_dates(n)
    ramp = np.arange(len(idx), dtype=float)
    data = {
        asset: 1_000_000.0 + 50_000.0 * i + 2_500.0 * ramp
        for i, asset in enumerate(assets)
    }
    return pd.DataFrame(data, index=idx)


def option_quotes(
    date: str = "2024-01-02",
    spot: float = 100.0,
    strikes: tuple[float, ...] = (90.0, 100.0, 110.0),
) -> pd.DataFrame:
    quote_date = pd.Timestamp(date)
    expiry = quote_date + pd.Timedelta(days=45)
    rows: list[dict[str, object]] = []
    for strike in strikes:
        intrinsic_call = max(spot - strike, 0.0)
        intrinsic_put = max(strike - spot, 0.0)
        time_value = 2.4 + 0.02 * abs(strike - spot)
        for option_type, intrinsic, delta in (
            ("call", intrinsic_call, 0.58 if strike <= spot else 0.42),
            ("put", intrinsic_put, -0.42 if strike >= spot else -0.28),
        ):
            mid = intrinsic + time_value
            rows.append(
                {
                    "date": quote_date,
                    "timestamp": quote_date + pd.Timedelta(hours=16),
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": option_type,
                    "underlying": "SYN",
                    "spot": spot,
                    "bid": mid - 0.05,
                    "ask": mid + 0.05,
                    "mid": mid,
                    "volume": 100 + int(strike),
                    "open_interest": 500 + int(strike),
                    "iv": 0.22 + 0.001 * abs(strike - spot),
                    "delta": delta,
                    "gamma": 0.025,
                    "vega": 10.0,
                    "theta": -0.03,
                    "rho": 0.01,
                    "tau": 45.0 / 365.25,
                    "dte_calendar": 45.0,
                    "k_over_s": strike / spot,
                }
            )
    return pd.DataFrame(rows)


def option_surface_quotes(
    dates: tuple[str, ...] = ("2024-01-02",),
    *,
    spot: float = 100.0,
    rate: float = 0.035,
    dividend_yield: float = 0.010,
    tau_days: tuple[int, ...] = (21, 45, 75, 105),
    k_values: tuple[float, ...] = (-0.24, -0.17, -0.10, -0.04, 0.0, 0.05, 0.12, 0.20),
) -> pd.DataFrame:
    from quantfinlab.options.bsm import (
        forward_bsm_delta,
        forward_bsm_gamma,
        forward_bsm_price,
        forward_bsm_rho,
        forward_bsm_theta,
        forward_bsm_vega,
    )

    rows: list[dict[str, object]] = []
    quote_id = 0
    for date_i, date in enumerate(dates):
        quote_date = pd.Timestamp(date)
        spot_i = float(spot) * (1.0 + 0.012 * date_i)
        rate_i = float(rate) + 0.001 * date_i
        div_i = float(dividend_yield)
        for dte in tau_days:
            tau = float(dte) / 365.25
            expiry = quote_date + pd.Timedelta(days=int(dte))
            forward = spot_i * np.exp((rate_i - div_i) * tau)
            discount_factor = np.exp(-rate_i * tau)
            for k in k_values:
                strike = forward * np.exp(float(k))
                iv_mid = 0.205 + 0.030 * np.sqrt(tau) + 0.080 * max(-float(k), 0.0) + 0.030 * float(k) ** 2 + 0.006 * date_i
                iv_bid = max(iv_mid - 0.010, 0.02)
                iv_ask = iv_mid + 0.010
                for option_type in ("call", "put"):
                    mid = float(forward_bsm_price(option_type, forward, strike, tau, iv_mid, discount_factor))
                    half_spread = min(0.45 * mid, max(0.001, 0.018 * mid))
                    bid = mid - half_spread
                    ask = mid + half_spread
                    delta = float(forward_bsm_delta(option_type, forward, strike, tau, iv_mid, discount_factor))
                    gamma = float(forward_bsm_gamma(forward, strike, tau, iv_mid, discount_factor))
                    vega = float(forward_bsm_vega(forward, strike, tau, iv_mid, discount_factor))
                    rows.append(
                        {
                            "quote_id": quote_id,
                            "source_index": quote_id,
                            "date": quote_date,
                            "timestamp": quote_date + pd.Timedelta(hours=16),
                            "expiry": expiry,
                            "strike": strike,
                            "option_type": option_type,
                            "underlying": "SYN",
                            "spot": spot_i,
                            "forward": forward,
                            "rate": rate_i,
                            "dividend_yield": div_i,
                            "implied_carry": rate_i - div_i,
                            "discount_factor": discount_factor,
                            "tau": tau,
                            "dte": float(dte),
                            "dte_days": float(dte),
                            "dte_calendar": float(dte),
                            "k": float(k),
                            "k_spot": float(np.log(strike / spot_i)),
                            "moneyness": strike / spot_i,
                            "log_moneyness": float(np.log(strike / spot_i)),
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "spread": 2.0 * half_spread,
                            "half_spread": half_spread,
                            "rel_spread": (2.0 * half_spread) / max(mid, 1e-12),
                            "relative_spread": (2.0 * half_spread) / max(mid, 1e-12),
                            "iv": iv_mid,
                            "iv_bid": iv_bid,
                            "iv_mid": iv_mid,
                            "iv_ask": iv_ask,
                            "delta": delta,
                            "gamma": gamma,
                            "vega": vega,
                            "theta": float(forward_bsm_theta(option_type, forward, strike, tau, iv_mid, discount_factor, rate=rate_i)),
                            "rho": float(forward_bsm_rho(option_type, forward, strike, tau, iv_mid, discount_factor)),
                            "volume": 100 + int(10_000 * abs(float(k))) + 5 * date_i,
                            "open_interest": 500 + int(5_000 * (1.0 - min(abs(float(k)), 0.5))),
                            "surface_weight": 1.0,
                            "obs_weight": 1.0,
                            "calib_scale_px": max(half_spread, 0.10 * vega * 0.01, 1e-4),
                            "contract_key": f"SYN_{expiry:%Y%m%d}_{strike:.4f}_{option_type}",
                        }
                    )
                    quote_id += 1
    return pd.DataFrame(rows)


def yield_curve_panel() -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    return pd.DataFrame(
        {
            "3M": [0.0400, 0.0405, 0.0410],
            "6M": [0.0410, 0.0415, 0.0420],
            "1Y": [0.0420, 0.0425, 0.0430],
            "2Y": [0.0430, 0.0435, 0.0440],
            "5Y": [0.0440, 0.0445, 0.0450],
            "10Y": [0.0450, 0.0455, 0.0460],
        },
        index=idx,
    )


def macro_panel(n: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2023-01-31", periods=int(n), freq="ME")
    return pd.DataFrame(
        {
            "growth": np.linspace(-0.6, 0.8, len(idx)),
            "inflation": np.linspace(0.5, -0.2, len(idx)),
            "financial_conditions": np.sin(np.linspace(0.0, np.pi, len(idx))),
        },
        index=idx,
    )


def flat_curve(rate: float = 0.04, *, method: str = "flat") -> Curve:
    grid = np.linspace(1.0 / 12.0, 30.0, 400)

    def df_func(t) -> np.ndarray:
        tau = np.asarray(t, dtype=float)
        return np.exp(-float(rate) * tau)

    df_grid = df_func(grid)
    return Curve(
        method=method,
        name=f"Flat {float(rate):.2%}",
        grid=grid,
        df_grid=df_grid,
        z_grid=np.full_like(grid, float(rate)),
        fwd_grid=np.full_like(grid, float(rate)),
        df=df_func,
    )


def zero_rate_panel(rate: float = 0.04) -> pd.DataFrame:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    cols = np.asarray([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    rows = [float(rate) + i * 0.0005 + 0.0002 * cols for i in range(len(idx))]
    return pd.DataFrame(rows, index=idx, columns=cols)
