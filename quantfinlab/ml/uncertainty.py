from __future__ import annotations

import numpy as np
import pandas as pd


def model_disagreement(predictions: pd.DataFrame | np.ndarray) -> pd.Series:
    """Cross-model standard deviation for aligned forecasts."""
    p = pd.DataFrame(predictions).apply(pd.to_numeric, errors="coerce")
    return p.std(axis=1, ddof=0).rename("model_disagreement")


def forecast_confidence(
    mu,
    width,
    *,
    min_width: float = 1e-6,
    power: float = 1.0,
) -> pd.Series:
    """Convert forecast signal-to-interval-width into a 0-1 confidence score."""
    m = pd.Series(mu, dtype=float)
    w = pd.Series(width, dtype=float).reindex(m.index).abs().clip(lower=float(min_width))
    raw = m.abs() / (m.abs() + w)
    return raw.clip(0.0, 1.0).pow(float(power)).rename("forecast_confidence")


def disagreement_confidence(disagreement, *, floor: float = 0.10) -> pd.Series:
    d = pd.Series(disagreement, dtype=float).replace([np.inf, -np.inf], np.nan)
    scale = d.expanding(min_periods=20).median().replace(0.0, np.nan)
    score = 1.0 / (1.0 + d.div(scale).replace([np.inf, -np.inf], np.nan))
    return score.fillna(score.median()).clip(float(floor), 1.0).rename("disagreement_confidence")


def confidence_adjusted_mu(
    mu,
    c_width,
    c_model,
    *extra_confidences,
    floor: float = 0.50,
    cap: float = 1.00,
) -> pd.Series:
    """Softly shrink forecasts by uncertainty without crushing the signal.

    Project 19 uses uncertainty as a sizing input, so this helper intentionally
    blends confidence terms instead of multiplying them.  Multiplication made
    reasonable forecasts vanish whenever one diagnostic was pessimistic.
    """
    m = pd.Series(mu, dtype=float)
    cw = pd.Series(c_width, dtype=float).reindex(m.index).fillna(0.0).clip(0.0, 1.0)
    cm = pd.Series(c_model, dtype=float).reindex(m.index).fillna(1.0).clip(0.0, 1.0)
    if extra_confidences:
        extras = [
            pd.Series(conf, dtype=float).reindex(m.index).clip(0.0, 1.0)
            for conf in extra_confidences
        ]
        cn = pd.concat(extras, axis=1).mean(axis=1).fillna(1.0)
    else:
        cn = pd.Series(1.0, index=m.index, dtype=float)
    blended = 0.50 + 0.50 * (0.50 * cw + 0.30 * cm + 0.20 * cn)
    return (m * blended.clip(float(floor), float(cap))).rename("mu_adj")


def interval_width(q_low, q_high) -> pd.Series:
    lo = pd.Series(q_low, dtype=float)
    hi = pd.Series(q_high, dtype=float).reindex(lo.index)
    return (hi - lo).rename("width")


__all__ = [
    "confidence_adjusted_mu",
    "disagreement_confidence",
    "forecast_confidence",
    "interval_width",
    "model_disagreement",
]
