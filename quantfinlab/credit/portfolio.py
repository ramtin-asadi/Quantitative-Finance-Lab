"""Issuer exposure aggregation and simulated credit losses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.risk import hist_var_es


def expected_loss(pd, exposure, recovery=0.4):
    """Issuer expected loss, EAD times PD times LGD, in exposure currency units."""
    p, E, R = np.broadcast_arrays(pd, exposure, recovery)
    if not np.isfinite(p + E + R).all() or np.any((p < 0) | (p > 1) | (E < 0) | (R < 0) | (R > 1)):
        raise ValueError("Invalid PD, exposure or recovery.")
    return p * E * (1 - R)


def portfolio_losses(U, pd, exposure, *, recovery=0.4) -> np.ndarray:
    """Apply issuer default thresholds to copula uniforms and aggregate losses."""
    p = np.asarray(pd, dtype=float)
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != p.size or np.any((U < 0) | (U > 1)):
        raise ValueError("U must contain one uniform column per issuer.")
    loss = expected_loss(np.ones_like(p), exposure, recovery)
    expected_loss(p, exposure, recovery)
    return (U < p) @ loss


def loss_summary(losses, *, notional: float = 1, levels=(0.95, 0.99)) -> pd.Series:
    """EL, loss standard deviation, VaR and ES as fractions of total notional."""
    rate = np.asarray(losses, dtype=float) / notional
    result = {"expected_loss": rate.mean(), "unexpected_loss": rate.std(ddof=0)}
    for level in levels:
        var, es = hist_var_es(-rate, alpha=1 - level)
        result.update({f"var_{level:.0%}": var, f"es_{level:.0%}": es})
    return pd.Series(result)


def credit_concentration(exposure, probabilities, groups, *, recovery=0.4) -> pd.DataFrame:
    """Exposure and expected-loss weights by issuer group."""
    data = pd.DataFrame({"group": groups, "exposure": exposure,
                         "expected_loss": expected_loss(probabilities, exposure, recovery)})
    totals = data.groupby("group", dropna=False)[["exposure", "expected_loss"]].sum()
    totals["exposure_weight"] = totals["exposure"] / totals["exposure"].sum()
    totals["loss_weight"] = totals["expected_loss"] / totals["expected_loss"].sum()
    return totals
