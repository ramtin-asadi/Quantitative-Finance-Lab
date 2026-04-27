from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantfinlab.risk.var import cf_var_es, fhs_var_es, hist_var_es


def historical_es(returns: pd.Series | Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> float:
    return hist_var_es(returns, alpha=alpha)[1]


def cornish_fisher_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_sim: int = 70_000,
    seed: int = 7,
) -> float:
    return cf_var_es(returns, alpha=alpha, n_sim=n_sim, seed=seed)[1]


def filtered_historical_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.94,
) -> float:
    return fhs_var_es(returns, alpha=alpha, lam=lam)[1]


__all__ = ["cornish_fisher_es", "filtered_historical_es", "historical_es"]
