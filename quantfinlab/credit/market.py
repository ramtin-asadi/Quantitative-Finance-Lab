"""Observed credit-market activity and default/non-default spread decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from quantfinlab.fundamentals import safe_ratio


def credit_breadth(data: pd.DataFrame) -> pd.DataFrame:
    """FINRA advance/decline and new-high/new-low breadth by published credit category."""
    source = data[data["market"].eq("corporate") & data["dataset"].eq("breadth")]
    result = source.pivot_table(index=["date", "security_category"], columns="source_field",
                                values="value", aggfunc="last").reset_index()
    result["advance_decline"] = safe_ratio(result["advances"] - result["declines"],
                                            result["advances"] + result["declines"])
    result["high_low"] = safe_ratio(result["fifty_two_week_high"] - result["fifty_two_week_low"],
                                     result["fifty_two_week_high"] + result["fifty_two_week_low"])
    return result


def customer_imbalance(data: pd.DataFrame) -> pd.DataFrame:
    """Customer buy-minus-sell volume over total customer volume."""
    source = data[data["market"].eq("corporate") & data["dataset"].eq("sentiment")
                  & data["source_field"].eq("total_volume")
                  & data["trade_side"].isin(["customer buy", "customer sell"])]
    result = source.pivot_table(index=["date", "security_category"], columns="trade_side",
                                values="value", aggfunc="last").reset_index()
    result["customer_imbalance"] = safe_ratio(result["customer buy"] - result["customer sell"],
                                              result["customer buy"] + result["customer sell"])
    return result


def aggregate_credit_loss(data: pd.DataFrame, value: str, *, exposure="debt",
                           date="decision_date", issuer="cik") -> pd.DataFrame:
    """Exposure-weighted issuer loss rates and contributing issuer counts."""
    known = data[data[value].notna() & data[exposure].gt(0)].copy()
    known["weighted"] = known[value] * known[exposure]
    result = known.groupby(date).agg(weighted=("weighted", "sum"),
                                      exposure=(exposure, "sum"), issuers=(issuer, "nunique"))
    result[value] = result["weighted"] / result["exposure"]
    return result[[value, "issuers"]]


def fit_loss_spread(X: pd.DataFrame, y, *, alpha=1.0) -> dict:
    """Accounting-loss to market default-component ridge mapping."""
    mu, sigma = X.mean(), X.std().replace(0, 1)
    model = Ridge(alpha=alpha).fit((X - mu) / sigma, y)
    return {"model": model, "mean": mu, "scale": sigma, "columns": list(X)}


def loss_spread_forecast(fit: dict, X: pd.DataFrame) -> np.ndarray:
    """Mapped nonnegative default-related spread in the training target's units."""
    return fit["model"].predict((X[fit["columns"]] - fit["mean"]) / fit["scale"]).clip(min=0)


def loss_spread_history(data: pd.DataFrame, features, target, *, minimum=12) -> pd.Series:
    """Expanding loss/spread mapping on a chronologically ordered monthly table."""
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    for i in range(minimum, len(data)):
        train = data.iloc[:i].dropna(subset=[*features, target])
        test = data.iloc[[i]]
        if len(train) >= minimum and test[features].notna().all(axis=None):
            fit = fit_loss_spread(train[features], train[target])
            prediction.iloc[i] = loss_spread_forecast(fit, test)[0]
    return prediction


def excess_credit_premium(spread, default_component):
    """Observed spread less a separately estimated default-related component."""
    return spread - default_component


def premium_validation(data: pd.DataFrame, estimate: str, *, benchmark="ebp") -> pd.Series:
    """Matched-month correlation, benchmark-on-estimate regression and level errors."""
    known = data[[estimate, benchmark]].dropna()
    fit = LinearRegression().fit(known[[estimate]], known[benchmark])
    error = known[estimate] - known[benchmark]
    return pd.Series({"months": len(known), "correlation": known.corr().iloc[0, 1],
                       "intercept": fit.intercept_, "slope": fit.coef_[0],
                       "RMSE": np.sqrt(np.mean(error ** 2)), "mean_error": error.mean()})
