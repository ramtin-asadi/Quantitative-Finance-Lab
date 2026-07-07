from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantfinlab.risk.var import cf_var_es, fhs_var_es, hist_var_es


def historical_es(returns: pd.Series | Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> float:
    """Compute historical expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.

    Returns
    -------
    float
        Positive loss expected shortfall from the historical method.
    """

    return hist_var_es(returns, alpha=alpha)[1]


def cornish_fisher_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_sim: int = 70_000,
    seed: int = 7,
) -> float:
    """Compute Cornish-Fisher expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.
    n_sim : int, default=70000
        Number of simulated transformed-normal draws used to approximate ES.
    seed : int, default=7
        Random seed.

    Returns
    -------
    float
        Positive loss expected shortfall.
    """

    return cf_var_es(returns, alpha=alpha, n_sim=n_sim, seed=seed)[1]


def filtered_historical_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.94,
) -> float:
    """Compute filtered-historical-simulation expected shortfall.

    Parameters
    ----------
    returns : array-like
        Return series.
    alpha : float, default=0.05
        Lower-tail probability.
    lam : float, default=0.94
        EWMA volatility decay parameter.

    Returns
    -------
    float
        Positive loss expected shortfall.
    """

    return fhs_var_es(returns, alpha=alpha, lam=lam)[1]


__all__ = ["cornish_fisher_es", "filtered_historical_es", "historical_es"]
