from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.options.diagnostics import build_atm_iv_panel_from_option_quotes


def _target_frame_from_dte(target_dte: pd.Series | np.ndarray | pd.DataFrame) -> pd.DataFrame:
    if isinstance(target_dte, pd.DataFrame):
        data = target_dte.copy()
        if "target_dte" not in data.columns:
            dte_col = "dte_trading" if "dte_trading" in data.columns else "dte" if "dte" in data.columns else data.columns[-1]
            data = data.rename(columns={dte_col: "target_dte"})
        if "date" not in data.columns:
            data = data.reset_index().rename(columns={"index": "date"})
        keep = ["date", "target_dte"]
        if "_row_id" in data.columns:
            keep.append("_row_id")
        return data[keep].copy()
    if isinstance(target_dte, pd.Series):
        data = target_dte.rename("target_dte").reset_index()
        data = data.rename(columns={data.columns[0]: "date"})
        return data
    return pd.DataFrame({"target_dte": np.asarray(target_dte, dtype=float)})


def _cumulative_forecast_variance(group: pd.DataFrame, annualization: int) -> pd.Series:
    horizon = pd.to_numeric(group["horizon"], errors="coerce")
    if "forecast_var_sum" in group.columns:
        return pd.to_numeric(group["forecast_var_sum"], errors="coerce")
    if "forecast_var_daily" in group.columns:
        return pd.to_numeric(group["forecast_var_daily"], errors="coerce") * horizon
    if "forecast_var_ann" in group.columns:
        return pd.to_numeric(group["forecast_var_ann"], errors="coerce") / float(annualization) * horizon
    raise ValueError("forecast_panel must contain forecast_var_sum, forecast_var_daily, or forecast_var_ann.")


def _rolling_past_rank(values: pd.Series, window: int, min_periods: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i, value in enumerate(arr):
        hist = arr[max(0, i - int(window)) : i]
        hist = hist[np.isfinite(hist)]
        if np.isfinite(value) and len(hist) >= int(min_periods):
            out[i] = float(np.mean(hist <= value))
    return pd.Series(out, index=values.index, name="vrp_rank")


def interpolate_forecast_variance_to_dte(
    forecast_panel: pd.DataFrame,
    target_dte: pd.Series | np.ndarray | pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    annualization: int = 252,
) -> pd.DataFrame:
    """Interpolate forecast cumulative variance to target trading-day DTE."""
    fc = forecast_panel.copy()
    if "date" not in fc.columns and "trade_date" in fc.columns:
        fc = fc.rename(columns={"trade_date": "date"})
    fc["date"] = pd.to_datetime(fc["date"], errors="coerce").dt.normalize()
    fc["horizon"] = pd.to_numeric(fc["horizon"], errors="coerce")
    fc = fc[fc["horizon"].isin([float(h) for h in horizons])].copy()
    if fc.empty:
        return pd.DataFrame()

    targets = _target_frame_from_dte(target_dte)
    if "date" in targets.columns:
        targets["date"] = pd.to_datetime(targets["date"], errors="coerce").dt.normalize()
    else:
        unique_dates = pd.Index(sorted(fc["date"].dropna().unique()))
        if len(unique_dates) != len(targets):
            raise ValueError("target_dte without dates must have one row per forecast date.")
        targets["date"] = unique_dates
    targets["target_dte"] = pd.to_numeric(targets["target_dte"], errors="coerce")

    records: list[dict[str, Any]] = []
    by_date = {date: group for date, group in fc.groupby("date", sort=False)}
    for _, row in targets.iterrows():
        date = pd.Timestamp(row["date"])
        target = float(row["target_dte"])
        group = by_date.get(date)
        if group is None or not np.isfinite(target) or target <= 0:
            continue
        g = group.copy()
        g["_cum_var"] = _cumulative_forecast_variance(g, annualization=annualization)
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["horizon", "_cum_var"]).sort_values("horizon")
        if g.empty:
            continue
        x = g["horizon"].to_numpy(dtype=float)
        y = g["_cum_var"].to_numpy(dtype=float)
        clipped_target = float(np.clip(target, np.nanmin(x), np.nanmax(x)))
        cum_var = float(np.interp(clipped_target, x, y))
        var_ann = float(annualization) / clipped_target * cum_var
        nearest = g.iloc[int(np.argmin(np.abs(x - clipped_target)))]
        rec = {
            "date": date,
            "target_dte": target,
            "target_horizon": clipped_target,
            "forecast_cum_var": cum_var,
            "forecast_var_ann": var_ann,
            "forecast_vol_ann": float(np.sqrt(var_ann)) if var_ann >= 0 else np.nan,
            "selected_model": nearest.get("selected_model", nearest.get("model", np.nan)),
            "nearest_horizon": int(nearest["horizon"]),
        }
        if "_row_id" in row.index:
            rec["_row_id"] = row["_row_id"]
        records.append(rec)
    return pd.DataFrame(records)


def compute_vrp_panel(
    iv_panel: pd.DataFrame,
    selected_forecasts: pd.DataFrame,
    dte_col: str = "dte",
    iv_col: str = "atm_iv_mid",
    annualization: int = 252,
    z_window: int = 126,
    rank_window: int | None = None,
    min_periods: int | None = None,
    calendar_days_per_year: float = 365.25,
) -> pd.DataFrame:
    """Compute a variance-risk-premium panel from ATM IV and selected variance forecasts.

    The variance risk premium is computed as annualized implied variance minus
    annualized forecast variance. The function interpolates selected forecast
    variance to each option row's target maturity and then adds VRP level, volatility
    spread, rolling z-score, and rolling historical rank.

    Parameters
    ----------
    iv_panel : pandas.DataFrame
        ATM or near-ATM implied-volatility panel containing quote dates, DTE, and
        implied volatility.
    selected_forecasts : pandas.DataFrame
        Forecast panel selected or combined by date and horizon.
    dte_col : str, default="dte"
        DTE column in ``iv_panel``. Calendar DTE is converted to trading-day DTE
        unless ``dte_col`` is ``"dte_trading"``.
    iv_col : str, default="atm_iv_mid"
        Annualized implied-volatility column.
    annualization : int, default=252
        Trading-day annualization factor for forecast variance.
    z_window : int, default=126
        Rolling window used for past-only VRP z-scores.
    rank_window : int, optional
        Rolling window for past-only VRP ranks. Defaults to ``z_window``.
    min_periods : int, optional
        Minimum observations for rolling mean, standard deviation, and rank.
    calendar_days_per_year : float, default=365.25
        Calendar-day denominator used to convert calendar DTE to trading DTE.

    Returns
    -------
    pandas.DataFrame
        IV panel with interpolated forecast fields plus ``vrp_var``,
        ``vol_spread``, ``vrp_mean``, ``vrp_std``, ``vrp_z``, and ``vrp_rank``.

    Notes
    -----
    Rolling z-scores and ranks use lagged VRP values, so the current observation is
    not included in its own historical normalization.
    """

    if iv_panel.empty or selected_forecasts.empty:
        return pd.DataFrame()
    data = iv_panel.copy()
    if "date" not in data.columns and "trade_date" in data.columns:
        data = data.rename(columns={"trade_date": "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date", dte_col, iv_col]).copy()
    data["_row_id"] = np.arange(len(data))
    if dte_col == "dte_trading":
        data["_target_trading_dte"] = pd.to_numeric(data[dte_col], errors="coerce")
    else:
        data["_target_trading_dte"] = (
            pd.to_numeric(data[dte_col], errors="coerce") * float(annualization) / float(calendar_days_per_year)
        )
    target = data[["date", "_target_trading_dte", "_row_id"]].rename(columns={"_target_trading_dte": "target_dte"})
    interp = interpolate_forecast_variance_to_dte(selected_forecasts, target, annualization=annualization)
    if interp.empty:
        return pd.DataFrame()
    out = data.merge(interp, on=["date", "_row_id"], how="inner")
    out["iv_ann"] = pd.to_numeric(out[iv_col], errors="coerce")
    finite_iv = out["iv_ann"].replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_iv.empty and float(finite_iv.median()) > 2.0:
        out["iv_ann"] = out["iv_ann"] / 100.0
    out["vrp_var"] = out["iv_ann"] ** 2 - pd.to_numeric(out["forecast_var_ann"], errors="coerce")
    out["vol_spread"] = out["iv_ann"] - pd.to_numeric(out["forecast_vol_ann"], errors="coerce")
    out = out.sort_values("date").reset_index(drop=True)

    window = max(1, int(z_window))
    rank_window = window if rank_window is None else max(1, int(rank_window))
    min_obs = int(min_periods) if min_periods is not None else min(window, max(10, window // 3 if window >= 30 else window))
    past = out["vrp_var"].shift(1)
    out["vrp_mean"] = past.rolling(window, min_periods=min_obs).mean()
    out["vrp_std"] = past.rolling(window, min_periods=min_obs).std(ddof=0)
    out["vrp_z"] = (out["vrp_var"] - out["vrp_mean"]) / out["vrp_std"].replace(0, np.nan)
    out["vrp_rank"] = _rolling_past_rank(out["vrp_var"], window=rank_window, min_periods=min_obs)
    return out.drop(columns=["_row_id"], errors="ignore")


__all__ = [
    "build_atm_iv_panel_from_option_quotes",
    "compute_vrp_panel",
    "interpolate_forecast_variance_to_dte",
]
