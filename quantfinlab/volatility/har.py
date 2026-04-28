from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HARRVFit:
    """Small immutable result object for a direct-horizon HAR-RV regression."""

    params: pd.Series
    feature_columns: tuple[str, ...]
    use_log: bool
    n_obs: int
    residual_std: float

    def predict(self, features: pd.DataFrame | pd.Series, eps: float = 1e-12) -> pd.Series:
        if isinstance(features, pd.Series):
            x = features.to_frame().T
        else:
            x = features.copy()
        x = x.loc[:, list(self.feature_columns)]
        design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
        pred = design @ self.params.to_numpy(dtype=float)
        if self.use_log:
            pred = np.exp(pred)
        pred = np.clip(pred, float(eps), None)
        return pd.Series(pred, index=x.index, name="forecast_var_daily")


def _as_datetime_series(values: pd.Series) -> pd.Series:
    out = pd.Series(values, copy=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).astype("datetime64[ns]")
    out = pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def make_har_features(
    rv_daily: pd.Series,
    weekly_window: int = 5,
    monthly_window: int = 22,
    use_log: bool = True,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Build HAR-RV daily, weekly, and monthly realized-variance features.

    Features at date ``t`` use only realized variance observed through ``t``.
    """
    rv = _as_datetime_series(rv_daily)
    features = pd.DataFrame(index=rv.index)
    features["rv_daily"] = rv
    features["rv_weekly"] = rv.rolling(int(weekly_window), min_periods=int(weekly_window)).mean()
    features["rv_monthly"] = rv.rolling(int(monthly_window), min_periods=int(monthly_window)).mean()
    if use_log:
        features = np.log(features.clip(lower=float(eps)))
        features = features.rename(columns=lambda c: f"log_{c}")
    return features


def _fit_ols(features: pd.DataFrame, target: pd.Series, *, use_log: bool, eps: float) -> HARRVFit:
    data = features.join(target.rename("target"), how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < features.shape[1] + 10:
        raise ValueError("Not enough observations to fit HAR-RV.")

    y = pd.to_numeric(data["target"], errors="coerce")
    if use_log:
        y = np.log(y.clip(lower=float(eps)))
    x = data.loc[:, features.columns].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), x])
    beta, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float), rcond=None)
    fitted = design @ beta
    resid = np.asarray(y, dtype=float) - fitted
    params = pd.Series(beta, index=["const", *features.columns], dtype=float)
    return HARRVFit(
        params=params,
        feature_columns=tuple(features.columns),
        use_log=use_log,
        n_obs=int(len(data)),
        residual_std=float(np.nanstd(resid, ddof=min(len(beta), len(resid) - 1))) if len(resid) > 1 else np.nan,
    )


def fit_har_rv(
    rv_daily: pd.Series,
    target: pd.Series,
    weekly_window: int = 5,
    monthly_window: int = 22,
    use_log: bool = True,
    eps: float = 1e-12,
) -> Any:
    """Fit a direct-horizon HAR-RV model."""
    features = make_har_features(
        rv_daily,
        weekly_window=weekly_window,
        monthly_window=monthly_window,
        use_log=use_log,
        eps=eps,
    )
    target_series = _as_datetime_series(target)
    return _fit_ols(features, target_series, use_log=use_log, eps=eps)


def _forward_realized_variance(returns: pd.Series, horizon: int) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    h = int(horizon)
    for i in range(0, len(values) - h):
        window = values[i + 1 : i + 1 + h]
        if np.isfinite(window).all():
            out[i] = float(np.sum(window * window))
    return pd.Series(out, index=returns.index, name=f"realized_var_sum_{h}")


def _signal_positions(
    index: pd.DatetimeIndex,
    *,
    train_window: int,
    max_horizon: int,
    refit_every: int,
) -> list[int]:
    last = len(index) - int(max_horizon) - 1
    if last <= train_window:
        return []
    return list(range(int(train_window), last + 1, max(1, int(refit_every))))


def rolling_har_forecasts(
    returns: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    train_window: int = 756,
    refit_every: int = 5,
    annualization: int = 252,
    use_log: bool = True,
    eps: float = 1e-12,
    forecast_start: str | pd.Timestamp | None = None,
    forecast_end: str | pd.Timestamp | None = None,
    signal_dates: pd.Index | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Generate direct-horizon HAR-RV forecasts on weekly-style signal dates.

    The target for horizon ``h`` is the average daily variance over the next
    ``h`` trading days. Training rows at signal date ``t`` are restricted to
    rows whose future horizon has already been realized by ``t``.
    """
    ret = _as_datetime_series(returns).dropna()
    if ret.empty:
        return pd.DataFrame()

    rv_daily = ret * ret
    features = make_har_features(rv_daily, use_log=use_log, eps=eps)
    max_h = int(max(horizons))
    if signal_dates is not None:
        wanted = pd.DatetimeIndex(pd.to_datetime(pd.Series(signal_dates).dropna(), errors="coerce")).normalize()
        locs = pd.DatetimeIndex(ret.index).get_indexer(wanted)
        positions = [int(pos) for pos in locs if int(pos) >= int(train_window) and int(pos) <= len(ret) - max_h - 1]
    else:
        positions = _signal_positions(
            pd.DatetimeIndex(ret.index),
            train_window=train_window,
            max_horizon=max_h,
            refit_every=refit_every,
        )
        if forecast_start is not None:
            start = pd.Timestamp(forecast_start).normalize()
            positions = [pos for pos in positions if pd.Timestamp(ret.index[pos]).normalize() >= start]
        if forecast_end is not None:
            end = pd.Timestamp(forecast_end).normalize()
            positions = [pos for pos in positions if pd.Timestamp(ret.index[pos]).normalize() <= end]
    targets_sum = {int(h): _forward_realized_variance(ret, int(h)) for h in horizons}
    records: list[dict[str, Any]] = []

    for pos in positions:
        signal_date = pd.Timestamp(ret.index[pos])
        if signal_date not in features.index or features.loc[[signal_date]].isna().any(axis=None):
            continue

        for horizon in horizons:
            h = int(horizon)
            target_daily = targets_sum[h] / float(h)
            known_end_pos = pos - h
            if known_end_pos <= 0:
                continue
            train_start = max(0, known_end_pos - int(train_window) + 1)
            train_dates = ret.index[train_start : known_end_pos + 1]
            x_train = features.loc[train_dates]
            y_train = target_daily.loc[train_dates]
            try:
                fit = _fit_ols(x_train, y_train, use_log=use_log, eps=eps)
                forecast_daily = float(fit.predict(features.loc[signal_date], eps=eps).iloc[0])
            except Exception:
                continue

            forecast_sum = forecast_daily * float(h)
            realized_sum = float(targets_sum[h].loc[signal_date])
            records.append(
                {
                    "date": signal_date,
                    "model": "har_rv",
                    "horizon": h,
                    "forecast_var_daily": forecast_daily,
                    "forecast_var_sum": forecast_sum,
                    "forecast_var_ann": float(annualization) * forecast_daily,
                    "forecast_vol_ann": float(np.sqrt(float(annualization) * forecast_daily)),
                    "realized_var_sum": realized_sum,
                    "realized_var_ann": float(annualization) / float(h) * realized_sum
                    if np.isfinite(realized_sum)
                    else np.nan,
                    "realized_vol_ann": float(np.sqrt(float(annualization) / float(h) * realized_sum))
                    if np.isfinite(realized_sum) and realized_sum >= 0
                    else np.nan,
                    "n_train": fit.n_obs,
                }
            )

    return pd.DataFrame(records).sort_values(["date", "horizon"]).reset_index(drop=True)


__all__ = [
    "HARRVFit",
    "fit_har_rv",
    "make_har_features",
    "rolling_har_forecasts",
]
