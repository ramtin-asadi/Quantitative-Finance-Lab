"""Piecewise-constant default intensities; rates and probabilities are decimals."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def hazards_from_pd(pd, T, *, monotone: bool = False) -> np.ndarray:
    """Convert cumulative PDs at year maturities into interval intensities.

    The final axis contains maturities. ``monotone=True`` applies the notebook's
    cumulative-maximum correction; otherwise crossing cumulative PDs raise.
    No tail extrapolation is performed here.
    """
    p, T = np.asarray(pd, dtype=float), np.asarray(T, dtype=float)
    if T.ndim != 1 or np.any(T <= 0) or np.any(np.diff(T) <= 0):
        raise ValueError("T must contain strictly increasing positive maturities.")
    if p.shape[-1] != len(T) or not np.isfinite(p).all() or np.any((p < 0) | (p >= 1)):
        raise ValueError("PDs must be finite, in [0, 1), and aligned with T.")
    if monotone:
        p = np.maximum.accumulate(p, axis=-1)
    if np.any(np.diff(p, axis=-1) < -1e-12):
        raise ValueError("Cumulative PDs decrease with maturity.")
    H = -np.log1p(-p)
    return np.diff(H, axis=-1, prepend=np.zeros_like(H[..., :1])) / np.diff(T, prepend=0)


def survival_probability(t, knots, lambdas) -> np.ndarray:
    """S(t)=exp(-integral lambda); extend the last interval intensity past its knot."""
    t, knots, lam = (np.asarray(x, dtype=float) for x in (t, knots, lambdas))
    if knots.ndim != 1 or len(knots) == 0 or lam.shape[-1] != len(knots):
        raise ValueError("The last intensity axis must match the maturity knots.")
    if np.any(knots <= 0) or np.any(np.diff(knots) <= 0) or np.any(t < 0):
        raise ValueError("Times must be nonnegative and knots positive and increasing.")
    if not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError("Intensities must be finite and nonnegative.")
    left = np.r_[0.0, knots[:-1]]
    dt = np.clip(np.atleast_1d(t)[..., None] - left, 0, knots - left)
    dt[..., -1] += np.maximum(np.atleast_1d(t) - knots[-1], 0)
    return np.exp(-np.einsum("...k,tk->...t", lam, dt))


def default_probability(t, knots, lambdas) -> np.ndarray:
    """Cumulative default probability for the supplied intensity curve."""
    return 1 - survival_probability(t, knots, lambdas)


def bootstrap_hazards(spreads, knots, discount, *, R: float = 0.4,
                      frequency: int = 4, quote_kind: str = "cds", on_failure="raise") -> dict:
    """Bootstrap CDS-equivalent hazards and report repricing errors.

    ``discount`` is a callable accepting year maturities. ``quote_kind`` records
    whether the input is a CDS par spread or a yield-spread proxy. Incompatible
    quotes raise instead of silently accepting an optimization boundary.
    """
    from .pricing import cds_spread

    T, s = np.asarray(knots, dtype=float), np.asarray(spreads, dtype=float)
    if s.shape != T.shape or np.any(s < 0) or not np.isfinite(s).all():
        raise ValueError("Spreads must be finite, nonnegative and match maturities.")
    if quote_kind not in {"cds", "yield_spread_proxy"}:
        raise ValueError("quote_kind must be 'cds' or 'yield_spread_proxy'.")
    hazards, failed = [], []
    for j, spread in enumerate(s):
        def error(lam):
            return cds_spread(np.r_[hazards, lam], T[:j + 1], discount,
                              T[j], R=R, frequency=frequency) - spread
        high = 1.0
        while error(high) < 0 and high < 1024:
            high *= 2
        if error(0) > 1e-10 or error(high) < 0:
            if on_failure != "boundary":
                raise ValueError(f"No nonnegative intensity reprices the {T[j]:g}Y quote.")
            low, high = 1e-8, 5.0
            hazards.append(low if abs(error(low)) < abs(error(high)) else high)
            failed.append(j)
        else:
            hazards.append(0.0 if abs(error(0)) < 1e-12 else brentq(error, 0, high))
    lam = np.asarray(hazards)
    repriced = np.array([cds_spread(lam, T, discount, t, R=R, frequency=frequency) for t in T])
    return {"knots": T, "hazards": lam, "repriced": repriced,
            "error_bp": (repriced - s) * 1e4, "quote_kind": quote_kind,
            "boundary_intervals": failed}
