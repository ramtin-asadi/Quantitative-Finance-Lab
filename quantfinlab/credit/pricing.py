"""CDS cash flows and public credit-risk pricing approximations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .curves import survival_probability


def cds_legs(lambdas, knots, discount, T: float, *, R: float = 0.4,
             frequency: int = 4) -> tuple:
    """Risky annuity and protection PV per unit notional.

    Payments include a final stub; default accrual uses the half-period
    approximation. Protection is paid at the interval end.
    """
    if T <= 0 or frequency < 1 or not 0 <= R < 1:
        raise ValueError("Require T > 0, frequency >= 1, and recovery in [0, 1).")
    t = np.unique(np.r_[np.arange(1 / frequency, T, 1 / frequency), T])
    dt = np.diff(t, prepend=0)
    df = np.asarray(discount(t), dtype=float)
    if not np.isfinite(df).all() or np.any(df <= 0):
        raise ValueError("Discount factors must be finite and positive.")
    S = survival_probability(t, knots, lambdas)
    dQ = np.concatenate([np.ones_like(S[..., :1]), S[..., :-1]], axis=-1) - S
    annuity = np.sum(df * dt * (S + 0.5 * dQ), axis=-1)
    protection = (1 - R) * np.sum(df * dQ, axis=-1)
    return annuity, protection


def cds_spread(lambdas, knots, discount, T: float, *, R: float = 0.4,
               frequency: int = 4):
    """Par CDS spread in decimal annual units."""
    A, P = cds_legs(lambdas, knots, discount, T, R=R, frequency=frequency)
    return P / A


def cds_value(coupon, lambdas, knots, discount, T: float, *, R: float = 0.4,
              notional: float = 1.0, frequency: int = 4):
    """Protection-buyer PV at a fixed contractual coupon."""
    A, P = cds_legs(lambdas, knots, discount, T, R=R, frequency=frequency)
    return notional * (P - coupon * A)


def risky_pv01(lambdas, knots, discount, T: float, *, R: float = 0.4,
               notional: float = 1.0):
    """Risky annuity times one basis point and notional (positive magnitude)."""
    A, _ = cds_legs(lambdas, knots, discount, T, R=R)
    return notional * A * 1e-4


def cds_cs01(spreads, knots, discount, T: float, *, coupon: float,
             R: float = 0.4, notional: float = 1.0, bump: float = 1e-4):
    """Buyer PV change per 1bp parallel quote increase, with hazard rebootstrap."""
    from .curves import bootstrap_hazards

    base = bootstrap_hazards(spreads, knots, discount, R=R)["hazards"]
    bumped = bootstrap_hazards(np.asarray(spreads) + bump, knots, discount, R=R)["hazards"]
    return (cds_value(coupon, bumped, knots, discount, T, R=R, notional=notional)
            - cds_value(coupon, base, knots, discount, T, R=R, notional=notional)) * 1e-4 / bump


def fit_hazard_multiplier(lambda_p, knots, discount, target_spread: float, weights,
                          *, T: float = 5.0, R: float = 0.4) -> dict:
    """Fit kappa so exposure-weighted CDS spreads match a supplied default component.

    The caller supplies the default-related component, not the total credit
    spread. The multiplier is a public approximation, not a bond OAS calibration.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0 or target_spread <= 0:
        raise ValueError("Require nonnegative finite weights and a positive target spread.")
    w = w / w.sum()
    def error(kappa):
        return np.dot(w, cds_spread(kappa * np.asarray(lambda_p), knots, discount, T, R=R)) - target_spread
    high = 1.0
    while error(high) < 0 and high < 1e6:
        high *= 2
    kappa = brentq(error, 0, high)
    return {"kappa": kappa, "lambda_q": kappa * np.asarray(lambda_p),
            "target_spread": target_spread, "error_bp": error(kappa) * 1e4}
