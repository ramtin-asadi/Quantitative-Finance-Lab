"""Grouped mixed-frequency states, factor bridges, and Kalman news."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .transforms import robust_standardize


@dataclass
class DFMFit:
    """Parameters and scaling fixed at a historical refit date."""
    result: object
    columns: list
    factors: dict
    orders: dict
    location: pd.Series
    scale: pd.Series
    quarterly_location: pd.Series | None
    quarterly_scale: pd.Series | None
    signs: pd.Series
    cap: float


def _model(monthly, quarterly, factors, orders):
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

    return DynamicFactorMQ(monthly, endog_quarterly=quarterly, factors=factors,
                            factor_orders=orders, idiosyncratic_ar1=False,
                            standardize=False, obs_cov_diag=True)


def fit_dfm(monthly: pd.DataFrame, quarterly=None, *, factors: dict, orders: dict,
              anchors: dict, cap=8.0, max_iter=100, tolerance=1e-4) -> DFMFit:
    """Fit grouped DynamicFactorMQ, with quarterly targets in the observation system."""
    z, location, scale, _ = robust_standardize(monthly, monthly.index.max(), cap=cap)
    z = z.asfreq("MS")
    q_location = q_scale = qz = None
    if quarterly is not None:
        quarterly = pd.DataFrame(quarterly)
        q_location = quarterly.median()
        q_scale = 1.4826 * (quarterly - q_location).abs().median()
        if q_scale.le(0).any():
            raise ValueError("Quarterly target scale must be positive.")
        qz = (quarterly - q_location) / q_scale
        qz.index = qz.index.to_period("Q") if isinstance(qz.index, pd.DatetimeIndex) else qz.index
    model = _model(z, qz, factors, orders)
    initial = _model(z.interpolate(limit_direction="both").fillna(0), qz, factors, orders)
    result = model.fit(start_params=initial.start_params, maxiter=max_iter,
                       tolerance=tolerance, disp=False)
    scores = result.factors.filtered[list(orders)].copy()
    scores.index = scores.index.to_timestamp()
    signs = pd.Series({name: np.sign(scores[name].corr(monthly[anchors[name]].reindex(scores.index))) or 1
                       for name in orders}).fillna(1)
    return DFMFit(result, list(monthly), factors, orders, location, scale, q_location, q_scale, signs, cap)


def filter_dfm(fit: DFMFit, monthly: pd.DataFrame, quarterly=None, *, end=None) -> dict:
    """Update one information set using frozen parameters; return filtered factors and result."""
    z = ((monthly[fit.columns] - fit.location) / fit.scale).clip(-fit.cap, fit.cap).asfreq("MS")
    if end is not None:
        z = z.reindex(pd.date_range(z.index.min(), max(z.index.max(), pd.Timestamp(end)), freq="MS"))
    qz = None
    if quarterly is not None:
        qz = (pd.DataFrame(quarterly) - fit.quarterly_location) / fit.quarterly_scale
        qz.index = qz.index.to_period("Q") if isinstance(qz.index, pd.DatetimeIndex) else qz.index
    result = _model(z, qz, fit.factors, fit.orders).smooth(fit.result.params)
    factors = result.factors.filtered[list(fit.orders)].mul(fit.signs, axis=1)
    factors.index = factors.index.to_timestamp()
    return {"factors": factors, "result": result}


def factor_forecast(training: pd.DataFrame, current: pd.Series, features, target,
                      *, alpha=5.0, feature_cap=5.0, target_cap=8.0) -> dict:
    """Robust factor-augmented ridge forecast, separate from the state-space target."""
    train = training.dropna(subset=[target, *features])
    scaler = StandardScaler().fit(train[features])
    mu = train[target].median()
    sigma = max(1.4826 * (train[target] - mu).abs().median(), 1e-6)
    weights = 1 / (1 + ((train[target] - mu) / (4 * sigma)) ** 2)
    model = Ridge(alpha=alpha).fit(scaler.transform(train[features]), train[target], sample_weight=weights)
    point = np.clip(scaler.transform(current[features].to_frame().T), -feature_cap, feature_cap)
    prediction = float(np.clip(model.predict(point)[0], mu - target_cap * sigma, mu + target_cap * sigma))
    residual = train[target] - model.predict(scaler.transform(train[features]))
    return {"mean": prediction, "sigma": residual.tail(60).std(ddof=1),
            "model": model, "scaler": scaler}


def dfm_news(before, after, *, variable, impact_date, location=0.0, scale=1.0) -> dict:
    """Native model-based news and revision impacts in the target's original units."""
    if not np.allclose(before.params, after.params):
        raise ValueError("News attribution requires identical model parameters.")
    news = after.news(before, impact_date=pd.Period(impact_date, freq="M"),
                       impacted_variable=variable, comparison_type="previous", original_scale=False)
    details = news.details_by_impact.reset_index()
    details["impact"] *= scale
    details["weight"] *= scale
    impacts = news.impacts.reset_index()
    for name in ["estimate (prev)", "impact of revisions", "impact of news", "total impact", "estimate (new)"]:
        impacts[name] *= scale
        if name.startswith("estimate"):
            impacts[name] += location
    return {"impacts": impacts, "details": details, "result": news}


def factor_history(factors: pd.DataFrame, truth: pd.DataFrame, *, frequency="M") -> pd.DataFrame:
    """Align factors to target observation periods and attach the previous release."""
    result = truth.sort_values("observation_date").copy()
    if frequency == "Q":
        quarterly = factors.groupby(factors.index.to_period("Q")).mean()
        periods = result["observation_date"].dt.to_period("Q")
        for name in factors:
            result[name] = periods.map(quarterly[name])
    else:
        result[list(factors)] = factors.reindex(pd.DatetimeIndex(result["observation_date"]), method="ffill").to_numpy()
    result["last_release"] = result["first"].shift(1)
    return result


def factor_point(factors: pd.DataFrame, observation_date, *, frequency="M") -> pd.Series:
    """Known factor values at the target period, matching the notebook aggregation."""
    date = pd.Timestamp(observation_date)
    if frequency == "Q":
        current = factors[factors.index.to_period("Q") == date.to_period("Q")]
        return current.mean() if len(current) else factors.iloc[-1].copy()
    eligible = factors.loc[:date]
    return eligible.iloc[-1].copy() if len(eligible) else factors.iloc[-1].copy()


def factor_loadings(fit: DFMFit, *, orient=True) -> pd.DataFrame:
    """Monthly observation loadings, optionally oriented like the returned factors."""
    result = pd.DataFrame(0.0, index=fit.columns, columns=list(fit.orders))
    for name in fit.columns:
        for factor in fit.factors[name]:
            result.loc[name, factor] = fit.result.params[f"loading.{factor}->{name}"]
    return result.mul(fit.signs, axis=1) if orient else result
