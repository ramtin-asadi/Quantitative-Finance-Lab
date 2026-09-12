"""Release-relative forecasts, revisions, benchmark alignment and event reactions."""

from __future__ import annotations

import pandas as pd

from quantfinlab.ml.evaluation import forecast_metrics


def forecast_origins(truth: pd.DataFrame, horizons, *, start=None) -> pd.DataFrame:
    """Evaluation dates a specified number of business days before each actual release."""
    source = truth if start is None else truth[truth["observation_date"].ge(pd.Timestamp(start))]
    rows = []
    for row in source.itertuples():
        days = horizons[row.target] if isinstance(horizons, dict) else horizons
        for h in days:
            rows.append({"target": row.target, "observation_date": row.observation_date,
                         "release_date": row.release_date, "evaluation_date": row.release_date - pd.offsets.BDay(h),
                         "horizon": h, "actual": row.first})
    return pd.DataFrame(rows).sort_values(["evaluation_date", "target", "observation_date", "horizon"]).reset_index(drop=True)


def benchmark_alignment(forecasts: pd.DataFrame, benchmarks: pd.DataFrame,
                          *, cutoff="release_date", on=("target", "observation_date")) -> pd.DataFrame:
    """Select the last model forecast available by each benchmark deadline.

    Exact target and reference period are mandatory join keys. The caller must
    first normalize annual, quarterly, and endpoint targets to the same object.
    """
    rows = []
    for _, row in benchmarks.iterrows():
        eligible = forecasts[forecasts["evaluation_date"].le(row[cutoff])]
        for name in on:
            eligible = eligible[eligible[name].eq(row[name])]
        if not eligible.empty:
            for _, point in eligible.sort_values("evaluation_date").groupby("model").tail(1).iterrows():
                rows.append({**row.to_dict(), "model": point["model"],
                             "evaluation_date": point["evaluation_date"], "model_forecast": point["mean"]})
    return pd.DataFrame(rows)


def revision_summary(truth: pd.DataFrame, *, latest="latest") -> pd.DataFrame:
    """Revision bias and magnitude by target."""
    data = truth.assign(revision=truth[latest] - truth["first"])
    return data.groupby("target").agg(observations=("revision", "count"),
                                       bias=("revision", "mean"), mae=("revision", lambda x: x.abs().mean()),
                                       std=("revision", "std"))


def release_surprises(actual, expected, *, minimum=24, cap=None) -> pd.Series:
    """Forecast innovations standardized by errors known before the current release."""
    errors = pd.Series(actual) - pd.Series(expected)
    scale = errors.expanding(min_periods=minimum).std(ddof=1).shift(1)
    news = errors / scale
    return news.clip(-cap, cap) if cap is not None else news


def release_response(news: pd.DataFrame, market: pd.DataFrame, *, drivers,
                       date="release_date", scale=1e4, covariance="HC1") -> pd.DataFrame:
    """Close-to-close market response to a joint release bundle, with robust errors."""
    from statsmodels.api import OLS, add_constant

    dates = pd.DatetimeIndex(news[date])
    positions = market.index.searchsorted(dates)
    valid = (positions > 0) & (positions < len(market))
    sample = news.loc[valid].reset_index(drop=True).copy()
    pos = positions[valid]
    reactions = scale * (market.iloc[pos].to_numpy() - market.iloc[pos - 1].to_numpy())
    rows = []
    for j, name in enumerate(market.columns):
        data = sample[drivers].assign(response=reactions[:, j]).dropna()
        fitted = OLS(data["response"], add_constant(data[drivers], has_constant="add")).fit(cov_type=covariance)
        for driver in drivers:
            rows.append({"market": name, "driver": driver, "events": len(data),
                         "beta": fitted.params[driver], "se": fitted.bse[driver],
                         "p_value": fitted.pvalues[driver], "adjusted_r2": fitted.rsquared_adj,
                         "window": "close-to-close; simultaneous releases and other news remain"})
    return pd.DataFrame(rows)


def nowcast_scores(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Existing forecast metrics grouped by target, horizon and model."""
    rows = []
    for (target, horizon, model), sample in forecasts.groupby(["target", "horizon", "model"]):
        score = forecast_metrics(sample, y_col="actual", prediction_cols=["mean"]).iloc[0].to_dict()
        rows.append({"target": target, "horizon": horizon, "model": model, **score})
    return pd.DataFrame(rows)


def baseline_forecast(history, *, quarterly=False) -> dict:
    """Notebook last-release, rolling-mean and OLS AR(1) benchmarks."""
    from sklearn.linear_model import LinearRegression

    x = pd.Series(history, dtype=float).dropna()
    if x.empty:
        raise ValueError("A released target observation is required for baselines.")
    last, mean = x.iloc[-1], x.tail(12 if quarterly else 24).mean()
    lagged = pd.concat([x.rename("y"), x.shift(1).rename("lag")], axis=1).dropna()
    ar = mean
    if len(lagged) >= 20:
        fit = LinearRegression().fit(lagged[["lag"]], lagged["y"])
        ar = float(fit.predict(pd.DataFrame({"lag": [last]}))[0])
    return {"Last release": last, "Rolling mean": mean, "AR(1)": ar}
