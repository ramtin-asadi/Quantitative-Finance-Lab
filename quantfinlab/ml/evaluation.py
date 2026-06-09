from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quantfinlab.ml.probabilistic import interval_coverage, interval_width, pinball_loss


def _clean_pair(data: pd.DataFrame, y_col: str, pred_col: str) -> pd.DataFrame:
    cols = [y_col, pred_col]
    out = (
        data[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return out


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def forecast_metrics(
    data: pd.DataFrame,
    *,
    y_col: str,
    prediction_cols: Sequence[str],
) -> pd.DataFrame:
    """Point forecast metrics for Project 19 model comparison."""
    rows = []
    for col in prediction_cols:
        if col not in data.columns:
            continue
        pair = _clean_pair(data, y_col, col)
        if pair.empty:
            continue
        err = pair[col] - pair[y_col]
        rows.append(
            {
                "model": col,
                "n": int(len(pair)),
                "MAE": float(err.abs().mean()),
                "RMSE": float(np.sqrt(np.mean(np.square(err)))),
                "Spearman IC": _spearman(pair[col], pair[y_col]),
                "Directional Accuracy": float((np.sign(pair[col]) == np.sign(pair[y_col])).mean()),
                "Bias": float(err.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def rank_metrics(
    data: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
    y_col: str,
    prediction_cols: Sequence[str],
    top_frac: float = 0.25,
) -> pd.DataFrame:
    """Cross-sectional rank IC and long-short bucket spread by date."""
    rows = []
    for col in prediction_cols:
        if col not in data.columns:
            continue
        daily_ic = []
        daily_spread = []
        daily_hit = []
        daily_mono = []
        for _, group in data[[date_col, asset_col, y_col, col]].dropna().groupby(date_col):
            if len(group) < 4:
                continue
            daily_ic.append(_spearman(group[col], group[y_col]))
            n = max(1, int(np.ceil(len(group) * float(top_frac))))
            ordered = group.sort_values(col)
            low = ordered.head(n)[y_col].mean()
            high = ordered.tail(n)[y_col].mean()
            daily_spread.append(float(high - low))
            pred_top = set(ordered.tail(n)[asset_col].astype(str))
            actual_top = set(group.sort_values(y_col).tail(n)[asset_col].astype(str))
            daily_hit.append(float(len(pred_top & actual_top) / max(1, n)))
            try:
                ranks = group[col].rank(method="first")
                bucket = pd.qcut(ranks, min(5, len(group)), labels=False) + 1
                bucket_mean = group.assign(_bucket=bucket.astype(int)).groupby("_bucket")[y_col].mean()
                mono = _spearman(pd.Series(bucket_mean.index, dtype=float), bucket_mean.astype(float))
                daily_mono.append(mono)
            except ValueError:
                pass
        rows.append(
            {
                "model": col,
                "mean_rank_ic": float(pd.Series(daily_ic).mean()) if daily_ic else np.nan,
                "rank_ic_t": (
                    float(pd.Series(daily_ic).mean() / pd.Series(daily_ic).std(ddof=1) * np.sqrt(len(daily_ic)))
                    if len(daily_ic) > 2 and pd.Series(daily_ic).std(ddof=1) > 0
                    else np.nan
                ),
                "bucket_spread": float(pd.Series(daily_spread).mean()) if daily_spread else np.nan,
                "top_k_hit_rate": float(pd.Series(daily_hit).mean()) if daily_hit else np.nan,
                "bucket_monotonicity": float(pd.Series(daily_mono).mean()) if daily_mono else np.nan,
                "positive_ic_share": float((pd.Series(daily_ic) > 0).mean()) if daily_ic else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def forecast_buckets(
    data: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    score_col: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Average realized outcome by within-date forecast bucket."""
    rows = []
    needed = [date_col, y_col, score_col]
    for dt, group in data[needed].dropna().groupby(date_col):
        if len(group) < int(n_buckets):
            continue
        ranks = group[score_col].rank(method="first")
        try:
            bucket = pd.qcut(ranks, int(n_buckets), labels=False) + 1
        except ValueError:
            continue
        tmp = group.assign(bucket=bucket.astype(int), date=pd.Timestamp(dt))
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["bucket", "mean", "median", "count"])
    stacked = pd.concat(rows, ignore_index=True)
    out = stacked.groupby("bucket")[y_col].agg(["mean", "median", "count"])
    out.index.name = "bucket"
    return out


def quantile_metrics(
    data: pd.DataFrame,
    *,
    y_col: str,
    quantile_sets: Mapping[str, tuple[str, str, str]],
) -> pd.DataFrame:
    """Coverage, width, and pinball diagnostics for q10/q50/q90 forecasts."""
    rows = []
    for name, (low_col, mid_col, high_col) in quantile_sets.items():
        cols = [y_col, low_col, mid_col, high_col]
        if not set(cols).issubset(data.columns):
            continue
        d = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if d.empty:
            continue
        rows.append(
            {
                "model": name,
                "n": int(len(d)),
                "coverage_80": interval_coverage(d[y_col], d[low_col], d[high_col]),
                "avg_width": interval_width(d[low_col], d[high_col]),
                "pinball_q10": pinball_loss(d[y_col], d[low_col], 0.10),
                "pinball_q50": pinball_loss(d[y_col], d[mid_col], 0.50),
                "pinball_q90": pinball_loss(d[y_col], d[high_col], 0.90),
            }
        )
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def rolling_rank_ic(
    data: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
    y_col: str,
    pred_col: str,
    window: int = 12,
) -> pd.Series:
    vals = []
    for dt, group in data[[date_col, asset_col, y_col, pred_col]].dropna().groupby(date_col):
        vals.append((pd.Timestamp(dt), _spearman(group[pred_col], group[y_col])))
    if not vals:
        return pd.Series(dtype=float, name="rolling_rank_ic")
    s = pd.Series(dict(vals)).sort_index()
    return s.rolling(int(window), min_periods=max(3, int(window) // 3)).mean().rename("rolling_rank_ic")


def walkforward_tabular_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    *,
    refit_dates: Sequence[pd.Timestamp | str],
    prediction_dates: Sequence[pd.Timestamp | str],
    estimators: Mapping[str, object],
    train_window: int,
    horizon: int = 21,
    n_jobs: int = 1,
    inner_threads: int | None = None,
    min_train: int = 500,
) -> pd.DataFrame:
    """Fit sklearn-style estimators on rolling windows and predict by date.

    The training cutoff subtracts the forecast horizon from each refit date so
    labels cannot leak from the prediction month into the fitted model.
    """
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from threadpoolctl import threadpool_limits

    x_df = pd.DataFrame(x)
    y_s = pd.Series(y, index=x_df.index)
    date_s = pd.Series(pd.to_datetime(dates), index=x_df.index)
    asset_s = pd.Series(assets, index=x_df.index).astype(str)
    refit_idx = pd.DatetimeIndex(pd.to_datetime(list(refit_dates))).sort_values().unique()
    forecast_idx = pd.DatetimeIndex(pd.to_datetime(list(prediction_dates))).sort_values().unique()
    if len(refit_idx) == 0 or len(forecast_idx) == 0:
        return pd.DataFrame(columns=["date", "asset", *estimators.keys()])

    def _prediction_dates_for_refit(i: int) -> pd.DatetimeIndex:
        start = pd.Timestamp(refit_idx[i])
        end = (
            pd.Timestamp(refit_idx[i + 1])
            if i + 1 < len(refit_idx)
            else pd.Timestamp(forecast_idx.max()) + pd.tseries.offsets.BDay(1)
        )
        return pd.DatetimeIndex([d for d in forecast_idx if start <= pd.Timestamp(d) < end])

    def _fit_one(i: int, dt: pd.Timestamp) -> pd.DataFrame | None:
        label_end = pd.Timestamp(dt) - pd.tseries.offsets.BDay(int(horizon))
        train_start = label_end - pd.tseries.offsets.BDay(int(train_window))
        train_mask = date_s.between(train_start, label_end)
        pred_dates = _prediction_dates_for_refit(i)
        if int(train_mask.sum()) < int(min_train) or len(pred_dates) == 0:
            return None

        fitted = {}
        for col, estimator in estimators.items():
            model = clone(estimator)
            with threadpool_limits(limits=inner_threads):
                model.fit(x_df.loc[train_mask], y_s.loc[train_mask])
            fitted[str(col)] = model

        frames = []
        for pred_dt in pred_dates:
            test_mask = date_s.eq(pd.Timestamp(pred_dt))
            if int(test_mask.sum()) == 0:
                continue
            out = pd.DataFrame(
                {
                    "date": date_s.loc[test_mask].to_numpy(),
                    "asset": asset_s.loc[test_mask].to_numpy(),
                }
            )
            x_test = x_df.loc[test_mask]
            for col, model in fitted.items():
                out[col] = model.predict(x_test)
            frames.append(out)
        return pd.concat(frames, ignore_index=True) if frames else None

    pieces = Parallel(n_jobs=int(n_jobs), prefer="threads", batch_size=1)(
        delayed(_fit_one)(i, dt) for i, dt in enumerate(refit_idx)
    )
    pieces = [p for p in pieces if p is not None and len(p)]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["date", "asset", *estimators.keys()])


def feature_screen_table(
    importance: pd.Series,
    coefficients: pd.Series | None = None,
    *,
    top_n: int = 40,
) -> pd.DataFrame:
    """Combine tree importance and linear absolute coefficient screens."""
    imp = pd.Series(importance, dtype=float).rename("rf_importance")
    pieces = [imp]
    if coefficients is not None:
        pieces.append(pd.Series(coefficients, dtype=float).abs().rename("elastic_net_abs_coef"))
    out = pd.concat(pieces, axis=1).fillna(0.0)
    ranks = pd.DataFrame({"rf_rank": out["rf_importance"].rank(ascending=False, method="average")})
    if "elastic_net_abs_coef" in out:
        ranks["enet_rank"] = out["elastic_net_abs_coef"].rank(ascending=False, method="average")
    out["mean_rank"] = ranks.mean(axis=1)
    return out.sort_values("mean_rank").head(int(top_n))


def active_performance_table(
    *,
    strategy_returns: pd.DataFrame,
    benchmark: str,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Benchmark-relative performance summary for policy comparisons."""
    R = pd.DataFrame(strategy_returns).copy().apply(pd.to_numeric, errors="coerce")
    if benchmark not in R.columns:
        raise ValueError(f"{benchmark!r} is not in strategy_returns.")
    b = R[benchmark].dropna()
    rows = []
    for name in R.columns:
        if name == benchmark:
            continue
        pair = pd.concat([R[name], b], axis=1, keys=["strategy", "benchmark"]).dropna()
        if pair.empty:
            continue
        active = pair["strategy"] - pair["benchmark"]
        te = float(active.std(ddof=1) * np.sqrt(float(annualization))) if len(active) > 1 else np.nan
        active_ann = float(active.mean() * float(annualization)) if len(active) else np.nan
        ir = float(active_ann / te) if np.isfinite(te) and te > 0 else np.nan
        active_nav = (1.0 + active).cumprod()
        active_dd = active_nav / active_nav.cummax() - 1.0
        monthly = active.resample("ME").sum()
        rows.append(
            {
                "Strategy": str(name),
                "Active CAGR": active_ann,
                "Tracking Error": te,
                "Information Ratio": ir,
                "Active Max Drawdown": float(active_dd.min()) if not active_dd.empty else np.nan,
                "Monthly Active Hit Rate": float((monthly > 0.0).mean()) if len(monthly) else np.nan,
                "Mean Daily Active": float(active.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


def policy_diagnostics_table(
    *,
    weights_by_strategy: Mapping[str, pd.DataFrame],
    returns: pd.DataFrame | None = None,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Turnover, exposure, concentration, and cost diagnostics for policies."""
    rows = []
    for name, weights in weights_by_strategy.items():
        W = pd.DataFrame(weights).copy()
        if W.empty:
            continue
        W.index = pd.to_datetime(W.index)
        W = W.sort_index().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cash_cols = [c for c in W.columns if str(c).upper() in {"SHY", "CASH"}]
        risky_cols = [c for c in W.columns if c not in cash_cols]
        turnover = 0.5 * W.diff().abs().sum(axis=1).fillna(0.0)
        risky_exposure = W[risky_cols].sum(axis=1) if risky_cols else W.sum(axis=1)
        hhi = (W[risky_cols].clip(lower=0.0) ** 2).sum(axis=1) if risky_cols else (W.clip(lower=0.0) ** 2).sum(axis=1)
        rows.append(
            {
                "Strategy": str(name),
                "Avg Risky Exposure": float(risky_exposure.mean()),
                "Avg Cash Weight": float(W[cash_cols].sum(axis=1).mean()) if cash_cols else 0.0,
                "Avg Turnover": float(turnover.mean()),
                "Annualized Turnover": float(turnover.mean() * 52.0),
                "Cost Drag Estimate": float(turnover.mean() * 52.0 * float(cost_bps) / 10000.0),
                "Avg HHI": float(hhi.mean()),
                "Effective N": float(1.0 / hhi.replace(0.0, np.nan).mean()) if hhi.replace(0.0, np.nan).notna().any() else np.nan,
                "Avg Max Weight": float(W[risky_cols].max(axis=1).mean()) if risky_cols else float(W.max(axis=1).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


def _ablate_state(state, ablation: str, *, seed: int = 42):
    arr = np.array(state.asset_state, copy=True)
    prior = np.array(state.prior_weights, copy=True)
    key = str(ablation).lower()
    names = list(getattr(state, "asset_feature_names", []))
    if key in {"no_forecasts", "no forecast features", "no_forecast"}:
        for i, name in enumerate(names):
            lname = str(name).lower()
            if "tcn" in lname or "forecast" in lname or "alpha" in lname or "hgb" in lname or "rf" in lname or lname.startswith("z_"):
                arr[:, :, i] = 0.0
    elif key in {"no_priors", "no prior weights", "no_prior"}:
        for i, name in enumerate(names):
            if str(name).lower().startswith("prior_"):
                arr[:, :, i] = 0.0
        prior[:] = 0.0
    elif key in {"shuffled_forecasts", "shuffled forecast features", "shuffle_forecasts"}:
        rng = np.random.default_rng(int(seed))
        for i, name in enumerate(names):
            lname = str(name).lower()
            if "tcn" in lname or "forecast" in lname or "alpha" in lname or "hgb" in lname or "rf" in lname or lname.startswith("z_"):
                perm = rng.permutation(arr.shape[0])
                arr[:, :, i] = arr[perm, :, i]
    return state.copy_with(asset_state=arr, prior_weights=prior)


def ablation_table(
    *,
    state,
    policies: Mapping[str, object],
    returns: pd.DataFrame | None = None,
    benchmark_weights: pd.DataFrame | None = None,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp],
    reward_settings: Mapping[str, object],
    ablations: Sequence[str] = ("no_forecasts", "no_priors", "shuffled_forecasts"),
    device=None,
) -> pd.DataFrame:
    """Evaluate trained policies after removing or disturbing key state blocks."""
    from quantfinlab.ml.rl import evaluate_policy

    rows = []
    for policy_name, policy in policies.items():
        base = evaluate_policy(
            policy=policy,
            state=state,
            period=period,
            reward_settings=reward_settings,
            device=device,
        ).validation
        rows.append({"Policy": str(policy_name), "Ablation": "base", **base})
        for ablation in ablations:
            ablated = _ablate_state(state, str(ablation))
            stats = evaluate_policy(
                policy=policy,
                state=ablated,
                period=period,
                reward_settings=reward_settings,
                device=device,
            ).validation
            rows.append({"Policy": str(policy_name), "Ablation": str(ablation), **stats})
    out = pd.DataFrame(rows)
    return out.set_index(["Policy", "Ablation"]) if {"Policy", "Ablation"}.issubset(out.columns) else out


def stress_active_table(
    *,
    strategy_returns: pd.DataFrame,
    benchmark: str,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
) -> pd.DataFrame:
    """Active returns by named stress windows."""
    R = pd.DataFrame(strategy_returns).copy().apply(pd.to_numeric, errors="coerce")
    if benchmark not in R.columns:
        raise ValueError(f"{benchmark!r} is not in strategy_returns.")
    rows = []
    for window_name, (start, end) in windows.items():
        part = R.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        if part.empty:
            continue
        bench_total = float((1.0 + part[benchmark].dropna()).prod() - 1.0)
        for name in part.columns:
            total = float((1.0 + part[name].dropna()).prod() - 1.0)
            rows.append(
                {
                    "Window": str(window_name),
                    "Strategy": str(name),
                    "Return": total,
                    "Benchmark Return": bench_total,
                    "Active Return": total - bench_total,
                }
            )
    out = pd.DataFrame(rows)
    return out.set_index(["Window", "Strategy"]) if {"Window", "Strategy"}.issubset(out.columns) else out


__all__ = [
    "ablation_table",
    "active_performance_table",
    "feature_screen_table",
    "forecast_buckets",
    "forecast_metrics",
    "policy_diagnostics_table",
    "quantile_metrics",
    "rank_metrics",
    "rolling_rank_ic",
    "stress_active_table",
    "walkforward_tabular_predictions",
]
