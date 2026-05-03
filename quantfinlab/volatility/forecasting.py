from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ARCH_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "garch11_normal", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    {"name": "garch11_student", "vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    {"name": "garch22_normal", "vol": "GARCH", "p": 2, "o": 0, "q": 2, "dist": "normal"},
    {"name": "garch22_student", "vol": "GARCH", "p": 2, "o": 0, "q": 2, "dist": "t"},
    {"name": "gjr11_normal", "vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "normal"},
    {"name": "gjr11_student", "vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    {"name": "egarch11_normal", "vol": "EGARCH", "p": 1, "o": 0, "q": 1, "dist": "normal"},
    {"name": "egarch11_student", "vol": "EGARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
)


def _returns_series(returns: pd.Series | pd.DataFrame, return_col: str = "return") -> pd.Series:
    if isinstance(returns, pd.DataFrame):
        if return_col in returns.columns:
            series = returns[return_col]
        elif "spx_ret" in returns.columns:
            series = returns["spx_ret"]
        else:
            numeric = returns.select_dtypes(include=[np.number]).columns
            if len(numeric) == 0:
                raise ValueError("returns DataFrame must contain a numeric return column.")
            series = returns[numeric[0]]
        if not isinstance(returns.index, pd.DatetimeIndex):
            date_col = "date" if "date" in returns.columns else "trade_date" if "trade_date" in returns.columns else None
            if date_col is not None:
                series = pd.Series(series.to_numpy(dtype=float), index=pd.to_datetime(returns[date_col], errors="coerce"))
    else:
        series = returns
    out = pd.Series(series, copy=True)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).astype("datetime64[ns]")
    out = pd.to_numeric(out, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out


def future_realized_variance(
    returns: pd.Series | pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    *,
    return_col: str = "return",
    annualization: int = 252,
) -> pd.DataFrame:
    """Forward realized variance targets aligned to forecast dates."""
    ret = _returns_series(returns, return_col=return_col)
    values = ret.to_numpy(dtype=float)
    out = pd.DataFrame(index=ret.index)
    for horizon in horizons:
        h = int(horizon)
        rv_sum = np.full(len(values), np.nan, dtype=float)
        for i in range(len(values) - h):
            window = values[i + 1 : i + 1 + h]
            if np.isfinite(window).all():
                rv_sum[i] = float(np.sum(window * window))
        out[f"realized_var_sum_{h}"] = rv_sum
        out[f"realized_var_ann_{h}"] = float(annualization) / float(h) * rv_sum
        out[f"realized_vol_ann_{h}"] = np.sqrt(out[f"realized_var_ann_{h}"].clip(lower=0.0))
    out.index.name = "date"
    return out


def make_weekly_signal_dates(
    dates: pd.Index | pd.Series,
    *,
    step: int = 5,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Use every ``step``-th trading date as the weekly signal calendar."""
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates).dropna(), errors="coerce")).sort_values().unique()
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    return pd.DatetimeIndex(idx[:: max(1, int(step))])


def fit_arch_model(
    returns_pct: pd.Series | np.ndarray,
    *,
    vol: str,
    p: int,
    q: int,
    dist: str,
    o: int = 0,
    mean: str = "Zero",
    maxiter: int = 500,
):
    """Fit one ARCH-family model using the optional ``arch`` dependency."""
    try:
        from arch import arch_model
    except Exception as exc:
        raise ImportError("rolling_arch_forecasts_weekly requires the 'arch' package.") from exc

    model = arch_model(
        returns_pct,
        mean=mean,
        vol=vol,
        p=int(p),
        o=int(o),
        q=int(q),
        dist=dist,
        rescale=False,
    )
    res = model.fit(disp="off", show_warning=False, options={"maxiter": int(maxiter)})
    return model, res


def _arch_forecast_variance(res: Any, max_horizon: int, *, simulations: int, seed: int | None) -> np.ndarray:
    try:
        fc = res.forecast(horizon=int(max_horizon), reindex=False, method="analytic")
    except Exception:
        if seed is not None:
            np.random.seed(int(seed))
        fc = res.forecast(
            horizon=int(max_horizon),
            reindex=False,
            method="simulation",
            simulations=int(simulations),
        )
    return np.asarray(fc.variance.values[-1], dtype=float).reshape(-1)


def rolling_arch_forecasts_weekly(
    returns: pd.Series | pd.DataFrame,
    specs: tuple[dict[str, Any], ...] | list[dict[str, Any]] = DEFAULT_ARCH_MODEL_SPECS,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    *,
    train_window: int = 756,
    signal_step: int = 5,
    forecast_start: str | pd.Timestamp | None = None,
    forecast_end: str | pd.Timestamp | None = None,
    annualization: int = 252,
    return_col: str = "return",
    maxiter: int = 500,
    simulations: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Fit ARCH-family models only on weekly signal dates and forecast horizons.

    Each row is a fresh signal-date forecast. The function does not fill daily
    dates between refits, which prevents stale repeated forecasts from looking
    like new information.
    """
    ret = _returns_series(returns, return_col=return_col)
    max_h = int(max(horizons))
    all_signal_dates = make_weekly_signal_dates(ret.index, step=signal_step, start=forecast_start, end=forecast_end)
    signal_lookup = set(pd.Timestamp(d) for d in all_signal_dates)
    records: list[dict[str, Any]] = []
    values = ret.to_numpy(dtype=float)
    ret_pct = 100.0 * values
    dates = pd.DatetimeIndex(ret.index)

    for pos in range(int(train_window), len(ret) - max_h):
        signal_date = pd.Timestamp(dates[pos])
        if signal_date not in signal_lookup:
            continue
        train = ret_pct[: pos + 1]
        if not np.isfinite(train).all():
            train = train[np.isfinite(train)]
        if len(train) < int(train_window):
            continue

        for spec_idx, spec in enumerate(specs):
            name = str(spec["name"])
            try:
                _, res = fit_arch_model(
                    train,
                    vol=str(spec["vol"]),
                    p=int(spec["p"]),
                    o=int(spec.get("o", 0)),
                    q=int(spec["q"]),
                    dist=str(spec["dist"]),
                    maxiter=maxiter,
                )
                fc_var_pct2 = _arch_forecast_variance(
                    res,
                    max_h,
                    simulations=simulations,
                    seed=int(seed + 1009 * spec_idx + pos),
                )
            except Exception as exc:
                records.append(
                    {
                        "date": signal_date,
                        "model": name,
                        "horizon": np.nan,
                        "error": str(exc),
                    }
                )
                continue

            for horizon in horizons:
                h = int(horizon)
                forecast_sum = float(np.sum(fc_var_pct2[:h]) / (100.0**2))
                realized_window = values[pos + 1 : pos + 1 + h]
                realized_sum = (
                    float(np.sum(realized_window * realized_window))
                    if len(realized_window) == h and np.isfinite(realized_window).all()
                    else np.nan
                )
                forecast_daily = forecast_sum / float(h)
                records.append(
                    {
                        "date": signal_date,
                        "model": name,
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
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.dropna(subset=["horizon"]).copy()
    out["horizon"] = out["horizon"].astype(int)
    return out.sort_values(["date", "model", "horizon"]).reset_index(drop=True)


def qlike_loss(realized_var: np.ndarray | pd.Series, forecast_var: np.ndarray | pd.Series, eps: float = 1e-12) -> float:
    """Mean QLIKE loss for positive realized and forecast variance."""
    rv = np.asarray(realized_var, dtype=float)
    fv = np.asarray(forecast_var, dtype=float)
    mask = np.isfinite(rv) & np.isfinite(fv) & (rv > 0) & (fv > 0)
    if not mask.any():
        return np.nan
    rv = np.clip(rv[mask], float(eps), None)
    fv = np.clip(fv[mask], float(eps), None)
    return float(np.mean(np.log(fv) + rv / fv))


def score_forecasts_by_model(forecast_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute compact forecast tournament scores by model and horizon."""
    data = forecast_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["model", "horizon", "forecast_var_sum", "realized_var_sum"]
    )
    rows: list[dict[str, Any]] = []
    for (model, horizon), group in data.groupby(["model", "horizon"], sort=True):
        err_var = group["forecast_var_sum"] - group["realized_var_sum"]
        err_vol = group["forecast_vol_ann"] - group["realized_vol_ann"]
        corr = group[["forecast_vol_ann", "realized_vol_ann"]].corr().iloc[0, 1] if len(group) >= 3 else np.nan
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "qlike_var": qlike_loss(group["realized_var_sum"], group["forecast_var_sum"]),
                "rmse_var": float(np.sqrt(np.nanmean(err_var * err_var))),
                "mae_var": float(np.nanmean(np.abs(err_var))),
                "rmse_vol": float(np.sqrt(np.nanmean(err_vol * err_vol))),
                "mae_vol": float(np.nanmean(np.abs(err_vol))),
                "corr_vol": float(corr) if np.isfinite(corr) else np.nan,
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "qlike_var", "rmse_vol"]).reset_index(drop=True)


def mincer_zarnowitz_table(forecast_panel: pd.DataFrame) -> pd.DataFrame:
    """OLS calibration table: realized annualized variance on forecast variance."""
    rows: list[dict[str, Any]] = []
    data = forecast_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["model", "horizon", "forecast_var_ann", "realized_var_ann"]
    )
    for (model, horizon), group in data.groupby(["model", "horizon"], sort=True):
        if len(group) < 25:
            continue
        x = group["forecast_var_ann"].to_numpy(dtype=float)
        y = group["realized_var_ann"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        rows.append(
            {
                "model": model,
                "horizon": int(horizon),
                "alpha": float(beta[0]),
                "beta": float(beta[1]),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "n_obs": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["horizon", "model"]).reset_index(drop=True)


def dm_test(loss_a: np.ndarray | pd.Series, loss_b: np.ndarray | pd.Series, h: int = 1) -> dict[str, float | int]:
    """Diebold-Mariano test with a simple Newey-West long-run variance."""
    diff = pd.Series(np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    d = diff.to_numpy(dtype=float)
    n = len(d)
    if n < max(20, int(h) + 5):
        return {"dm_stat": np.nan, "dm_pvalue": np.nan, "n": int(n)}
    mean_d = float(np.mean(d))
    long_run = float(np.var(d, ddof=1))
    max_lag = int(max(1, h - 1))
    for lag in range(1, max_lag + 1):
        cov = float(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
        long_run += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    if long_run <= 0 or not np.isfinite(long_run):
        return {"dm_stat": np.nan, "dm_pvalue": np.nan, "n": int(n)}
    stat = mean_d / math.sqrt(long_run / n)
    pvalue = math.erfc(abs(stat) / math.sqrt(2.0))
    return {"dm_stat": float(stat), "dm_pvalue": float(pvalue), "n": int(n)}


def diebold_mariano_table(forecast_panel: pd.DataFrame, benchmark_model: str | None = None) -> pd.DataFrame:
    """Compare each model against the best average-QLIKE model by horizon."""
    data = forecast_panel.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "model", "horizon", "forecast_var_sum", "realized_var_sum"]
    )
    if data.empty:
        return pd.DataFrame()
    scores = score_forecasts_by_model(data)
    rows: list[dict[str, Any]] = []
    for horizon, group in data.groupby("horizon", sort=True):
        bench = benchmark_model
        if bench is None:
            h_scores = scores[scores["horizon"] == int(horizon)]
            if h_scores.empty:
                continue
            bench = str(h_scores.iloc[0]["model"])
        base = group[group["model"] == bench].sort_values("date")
        for competitor in sorted(set(group["model"]) - {bench}):
            other = group[group["model"] == competitor].sort_values("date")
            joined = base[["date", "forecast_var_sum", "realized_var_sum"]].merge(
                other[["date", "forecast_var_sum"]],
                on="date",
                how="inner",
                suffixes=("_benchmark", "_other"),
            )
            if joined.empty:
                continue
            loss_b = np.log(joined["forecast_var_sum_benchmark"]) + joined["realized_var_sum"] / joined[
                "forecast_var_sum_benchmark"
            ]
            loss_o = np.log(joined["forecast_var_sum_other"]) + joined["realized_var_sum"] / joined[
                "forecast_var_sum_other"
            ]
            rows.append(
                {
                    "horizon": int(horizon),
                    "benchmark_model": bench,
                    "competitor": competitor,
                    **dm_test(loss_b, loss_o, h=int(horizon)),
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon", "dm_pvalue"]).reset_index(drop=True)


def _loss_by_row(frame: pd.DataFrame, loss: str, eps: float) -> pd.Series:
    if loss != "qlike":
        raise ValueError("Only loss='qlike' is currently supported.")
    rv = pd.to_numeric(frame["realized_var_sum"], errors="coerce").clip(lower=eps)
    fv = pd.to_numeric(frame["forecast_var_sum"], errors="coerce").clip(lower=eps)
    return np.log(fv) + rv / fv


def select_forecast_by_rolling_loss(
    forecast_panel: pd.DataFrame,
    realized_var_targets: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 42, 63),
    lookback: int = 126,
    min_obs: int = 40,
    loss: str = "qlike",
    mode: str = "best",
    top_k: int = 3,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Select or combine forecasts using only past forecast errors.

    Selection is date-specific and horizon-specific. At date ``t`` the ranking
    uses only rows with forecast dates strictly before ``t``.
    """
    if mode not in {"best", "topk_inverse_loss"}:
        raise ValueError("mode must be 'best' or 'topk_inverse_loss'.")

    data = forecast_panel.copy()
    if "date" not in data.columns and "trade_date" in data.columns:
        data = data.rename(columns={"trade_date": "date"})
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["horizon"] = pd.to_numeric(data["horizon"], errors="coerce").astype("Int64")

    if realized_var_targets is not None and "realized_var_sum" not in data.columns:
        targets = realized_var_targets.copy()
        if "date" not in targets.columns:
            targets = targets.reset_index().rename(columns={"index": "date"})
        targets["date"] = pd.to_datetime(targets["date"], errors="coerce").dt.normalize()
        if {"horizon", "realized_var_sum"}.issubset(targets.columns):
            data = data.merge(targets[["date", "horizon", "realized_var_sum"]], on=["date", "horizon"], how="left")

    data = data.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["date", "model", "horizon", "forecast_var_sum"]
    )
    data = data[data["horizon"].isin([int(h) for h in horizons])].copy()
    if data.empty:
        return pd.DataFrame()
    data["loss_value"] = _loss_by_row(data, loss=loss, eps=eps)

    rows: list[dict[str, Any]] = []
    fallback_order = ["har_rv", "garch11_student", "garch11_normal"]

    for (date, horizon), current in data.groupby(["date", "horizon"], sort=True):
        current = current.dropna(subset=["forecast_var_sum"]).copy()
        if current.empty:
            continue

        past = data[(data["horizon"] == horizon) & (data["date"] < date)].dropna(subset=["loss_value"])
        if lookback and lookback > 0:
            dates = pd.Index(sorted(past["date"].dropna().unique()))
            keep_dates = set(dates[-int(lookback) :])
            past = past[past["date"].isin(keep_dates)]

        recent = (
            past.groupby("model")
            .agg(recent_loss=("loss_value", "mean"), n_loss=("loss_value", "count"))
            .query("n_loss >= @min_obs")
            .sort_values(["recent_loss", "n_loss"], ascending=[True, False])
        )

        valid_current = current.set_index("model", drop=False)
        selected_model = None
        selection_reason = "rolling_loss"
        forecast_sum = np.nan
        forecast_daily = np.nan
        forecast_ann = np.nan

        if not recent.empty:
            ranked_models = [m for m in recent.index if m in valid_current.index]
            if mode == "best" and ranked_models:
                selected_model = str(ranked_models[0])
                row = valid_current.loc[selected_model]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                forecast_sum = float(row["forecast_var_sum"])
                forecast_daily = float(row.get("forecast_var_daily", forecast_sum / float(horizon)))
                forecast_ann = float(row.get("forecast_var_ann", 252.0 * forecast_daily))
            elif mode == "topk_inverse_loss" and ranked_models:
                top_models = ranked_models[: max(1, int(top_k))]
                losses = recent.loc[top_models, "recent_loss"].astype(float)
                shifted = losses - float(losses.min()) + float(eps)
                weights = (1.0 / shifted).replace([np.inf, -np.inf], np.nan)
                weights = weights / weights.sum()
                chosen = valid_current.loc[top_models].copy()
                if isinstance(chosen, pd.Series):
                    chosen = chosen.to_frame().T
                forecast_sum = float(np.sum(chosen["forecast_var_sum"].to_numpy(dtype=float) * weights.to_numpy(dtype=float)))
                forecast_daily = forecast_sum / float(horizon)
                forecast_ann = 252.0 * forecast_daily
                selected_model = "top{}_inverse_{}".format(len(top_models), loss)

        if selected_model is None:
            for fallback in fallback_order:
                if fallback in valid_current.index:
                    row = valid_current.loc[fallback]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    selected_model = str(fallback)
                    forecast_sum = float(row["forecast_var_sum"])
                    forecast_daily = float(row.get("forecast_var_daily", forecast_sum / float(horizon)))
                    forecast_ann = float(row.get("forecast_var_ann", 252.0 * forecast_daily))
                    selection_reason = "fallback_model"
                    break
        if selected_model is None:
            forecast_sum = float(current["forecast_var_sum"].mean())
            forecast_daily = forecast_sum / float(horizon)
            forecast_ann = 252.0 * forecast_daily
            selected_model = "equal_weight_valid_models"
            selection_reason = "fallback_average"

        realized_sum = current["realized_var_sum"].dropna().iloc[0] if current["realized_var_sum"].notna().any() else np.nan
        rows.append(
            {
                "date": pd.Timestamp(date),
                "horizon": int(horizon),
                "selected_model": selected_model,
                "selection_reason": selection_reason,
                "forecast_var": forecast_sum,
                "forecast_var_sum": forecast_sum,
                "forecast_var_daily": forecast_daily,
                "forecast_var_ann": forecast_ann,
                "forecast_vol_ann": float(np.sqrt(forecast_ann)) if forecast_ann >= 0 else np.nan,
                "realized_var_sum": float(realized_sum) if pd.notna(realized_sum) else np.nan,
                "recent_loss": float(recent.iloc[0]["recent_loss"]) if not recent.empty else np.nan,
                "n_loss": int(recent.iloc[0]["n_loss"]) if not recent.empty else 0,
            }
        )

    return pd.DataFrame(rows).sort_values(["date", "horizon"]).reset_index(drop=True)


__all__ = [
    "DEFAULT_ARCH_MODEL_SPECS",
    "diebold_mariano_table",
    "dm_test",
    "fit_arch_model",
    "future_realized_variance",
    "make_weekly_signal_dates",
    "mincer_zarnowitz_table",
    "qlike_loss",
    "rolling_arch_forecasts_weekly",
    "score_forecasts_by_model",
    "select_forecast_by_rolling_loss",
]
