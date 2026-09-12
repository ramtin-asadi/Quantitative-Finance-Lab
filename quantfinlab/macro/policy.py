"""Policy-rule forecasts and matched overnight-rate distribution comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from quantfinlab.fixed_income.overnight import compound_overnight

from .bridge import bridge_forecast, fit_bridge


def fit_policy_rule(monthly: pd.DataFrame, *, features, policy="policy", horizon=3, alpha=4.0):
    """Horizon-specific ridge policy rule fitted to matured monthly policy outcomes."""
    target = monthly[policy].shift(-horizon)
    train = monthly[features].assign(target=target).dropna()
    return fit_bridge(train[features], train["target"], alpha=alpha, minimum=20)


def policy_forecast(fit, current: pd.DataFrame) -> dict:
    """Policy-rule mean and residual dispersion in the supplied rate units."""
    return {"mean": bridge_forecast(fit, current), "sigma": fit.sigma}


def policy_window_draws(paths, months, fixings: pd.Series, *, as_of, start, end,
                         calendar, basis=0.0, day_count=360) -> np.ndarray:
    """Combine known overnight fixings with monthly model draws and compound the window.

    Paths, fixings and basis are decimal rates. ``calendar`` is the explicit
    fixing calendar; use the applicable market holidays. Months index path steps.
    """
    paths = np.asarray(paths)
    periods = pd.DatetimeIndex(months).to_period("M")
    dates = pd.DatetimeIndex(calendar)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end))]
    daily = []
    for date in dates:
        if date <= pd.Timestamp(as_of) and date in fixings.index:
            daily.append(np.repeat(fixings.loc[date], len(paths)))
        else:
            step = periods.get_indexer([date.to_period("M")])[0]
            if step < 0:
                raise ValueError(f"No policy path covers {date:%Y-%m}.")
            daily.append(paths[:, step] + basis)
    return compound_overnight(np.column_stack(daily), dates, end, day_count=day_count)


def policy_distribution_gap(draws, rates, probabilities) -> dict:
    """Empirical Wasserstein distance and moments against actual market probability bins."""
    p, r = np.asarray(probabilities, dtype=float), np.asarray(rates, dtype=float)
    if np.any(p < 0) or p.sum() <= 0 or not np.isfinite(p + r).all():
        raise ValueError("Market bins require finite rates and nonnegative probability mass.")
    p = p / p.sum()
    mean = np.dot(p, r)
    return {"macro_mean": np.mean(draws), "market_mean": mean,
            "mean_gap": np.mean(draws) - mean, "macro_sigma": np.std(draws, ddof=1),
            "market_sigma": np.sqrt(np.dot(p, (r - mean) ** 2)),
            "wasserstein": wasserstein_distance(draws, r, v_weights=p)}
