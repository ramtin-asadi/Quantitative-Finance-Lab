from __future__ import annotations

import numpy as np
import pandas as pd


def _series(x: pd.Series | pd.DataFrame | np.ndarray, name: str = "x") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        numeric = x.select_dtypes(include=[np.number]).columns
        if len(numeric) == 0:
            return pd.Series(dtype=float, name=name)
        s = x[numeric[0]]
    else:
        s = pd.Series(x) if not isinstance(x, pd.Series) else x
    out = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
    out.name = getattr(s, "name", name)
    return out


def daily_variance(returns: pd.Series | pd.DataFrame, annualization: float | None = None) -> pd.Series:
    ret = _series(returns, "return")
    out = (ret * ret).rename("daily_variance")
    if annualization is not None:
        out.attrs["annualization"] = float(annualization)
    return out


def log_variance(rv: pd.Series | pd.DataFrame, eps: float = 1e-12) -> pd.Series:
    v = _series(rv, "variance").clip(lower=float(eps))
    return np.log(v).rename("log_variance")


def fgn_covariance(h: float, n: int) -> np.ndarray:
    h = float(h)
    n = int(n)
    k = np.arange(n, dtype=float)
    cov = 0.5 * (np.abs(k + 1.0) ** (2.0 * h) - 2.0 * np.abs(k) ** (2.0 * h) + np.abs(k - 1.0) ** (2.0 * h))
    return cov


def fbm_cholesky_paths(
    h_values=(0.05, 0.10, 0.20, 0.50, 0.80),
    n_steps: int = 512,
    n_paths: int = 4,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    n = int(n_steps)
    for h in h_values:
        cov = fgn_covariance(float(h), n)
        toeplitz = cov[np.abs(np.subtract.outer(np.arange(n), np.arange(n)))]
        chol = np.linalg.cholesky(toeplitz + 1e-12 * np.eye(n))
        z = rng.standard_normal((n, int(n_paths)))
        increments = chol @ z / (n ** float(h))
        paths = np.vstack([np.zeros((1, int(n_paths))), np.cumsum(increments, axis=0)])
        t = np.linspace(0.0, 1.0, n + 1)
        for j in range(int(n_paths)):
            rows.append(pd.DataFrame({"h": float(h), "path": j, "t": t, "x": paths[:, j]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["h", "path", "t", "x"])


def moment_scaling(
    x: pd.Series | pd.DataFrame,
    q_values=(0.5, 1.0, 1.5, 2.0, 3.0),
    lags=(1, 2, 4, 8, 16, 32),
) -> pd.DataFrame:
    s = _series(x, "x")
    values = s.to_numpy(dtype=float)
    rows = []
    for q in q_values:
        qf = float(q)
        for lag in lags:
            lag_i = int(lag)
            if lag_i <= 0 or lag_i >= len(values):
                continue
            inc = values[lag_i:] - values[:-lag_i]
            inc = inc[np.isfinite(inc)]
            moment = float(np.mean(np.abs(inc) ** qf)) if inc.size else np.nan
            rows.append(
                {
                    "q": qf,
                    "lag": lag_i,
                    "moment": moment,
                    "log_lag": np.log(float(lag_i)),
                    "log_moment": np.log(moment) if np.isfinite(moment) and moment > 0 else np.nan,
                    "n": int(inc.size),
                }
            )
    return pd.DataFrame(rows)


def hurst_from_moments(
    scaling: pd.DataFrame,
    *,
    q_col: str = "q",
    lag_col: str = "lag",
    moment_col: str = "moment",
) -> pd.DataFrame:
    rows = []
    if scaling.empty:
        return pd.DataFrame(columns=["q", "slope", "h", "standard_error", "r2", "n"])
    data = scaling.copy()
    data["log_lag_fit"] = np.log(pd.to_numeric(data[lag_col], errors="coerce"))
    data["log_moment_fit"] = np.log(pd.to_numeric(data[moment_col], errors="coerce"))
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[q_col, "log_lag_fit", "log_moment_fit"])
    for q, g in data.groupby(q_col, sort=True):
        if len(g) < 3:
            continue
        x = g["log_lag_fit"].to_numpy(dtype=float)
        y = g["log_moment_fit"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        resid = y - fitted
        ss_res = float(np.sum(resid * resid))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        dof = max(len(x) - 2, 1)
        sigma2 = ss_res / dof
        cov = sigma2 * np.linalg.pinv(design.T @ design)
        slope = float(beta[1])
        rows.append(
            {
                "q": float(q),
                "slope": slope,
                "h": slope / float(q) if float(q) != 0 else np.nan,
                "standard_error": float(np.sqrt(max(cov[1, 1], 0.0))),
                "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def rough_kernel_weights(h: float, lookback: int, horizon: int = 1) -> np.ndarray:
    lookback = int(lookback)
    j = np.arange(1, lookback + 1, dtype=float)
    horizon = max(1, int(horizon))
    w = (j + float(horizon)) ** (float(h) + 0.5) - j ** (float(h) + 0.5)
    w = np.maximum(w, 0.0)
    w = w / np.sum(w)
    return w[::-1]


def rough_kernel_forecasts(
    rv: pd.Series | pd.DataFrame,
    *,
    h: float,
    horizons=(1, 5, 10, 21, 42, 63),
    train_window: int = 756,
    signal_step: int = 5,
    annualization: float = 252.0,
    eps: float = 1e-12,
) -> pd.DataFrame:
    v = _series(rv, "variance").clip(lower=float(eps))
    max_h = int(max(horizons))
    records = []
    dates = pd.Index(v.index)
    values = v.to_numpy(dtype=float)
    for pos in range(int(train_window), len(v) - max_h, max(1, int(signal_step))):
        hist = values[pos - int(train_window) + 1 : pos + 1]
        if len(hist) != int(train_window) or not np.isfinite(hist).all():
            continue
        for horizon in horizons:
            hzn = int(horizon)
            weights = rough_kernel_weights(float(h), int(train_window), hzn)
            base = float(np.sum(hist * weights))
            records.append(
                {
                    "date": pd.Timestamp(dates[pos]),
                    "model": "rough_kernel",
                    "horizon": hzn,
                    "forecast_var_daily": base,
                    "forecast_var_sum": base * hzn,
                    "forecast_var_ann": float(annualization) * base,
                    "forecast_vol_ann": float(np.sqrt(max(float(annualization) * base, 0.0))),
                }
            )
    return pd.DataFrame(records)


def rough_forecast_frame(
    *,
    rough_fc: pd.DataFrame | None = None,
    har_fc: pd.DataFrame | None = None,
    arch_fc: pd.DataFrame | None = None,
    rv_targets: pd.DataFrame | None = None,
    horizons=(1, 5, 10, 21, 42, 63),
) -> pd.DataFrame:
    frames = [x for x in (rough_fc, har_fc, arch_fc) if isinstance(x, pd.DataFrame) and not x.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if rv_targets is not None and not rv_targets.empty:
        target = rv_targets.copy()
        if "date" not in target.columns:
            target = target.reset_index().rename(columns={"index": "date"})
        target["date"] = pd.to_datetime(target["date"], errors="coerce").dt.normalize()
        pieces = []
        for horizon in horizons:
            h = int(horizon)
            cols = ["date"]
            rename = {}
            for src, dst in [
                (f"realized_var_sum_{h}", "realized_var_sum"),
                (f"realized_var_ann_{h}", "realized_var_ann"),
                (f"realized_vol_ann_{h}", "realized_vol_ann"),
            ]:
                if src in target.columns:
                    cols.append(src)
                    rename[src] = dst
            if len(cols) > 1:
                tmp = target[cols].rename(columns=rename)
                tmp["horizon"] = h
                pieces.append(tmp)
        if pieces:
            targets = pd.concat(pieces, ignore_index=True)
            out = out.drop(columns=[c for c in ["realized_var_sum", "realized_var_ann", "realized_vol_ann"] if c in out.columns], errors="ignore")
            out = out.merge(targets, on=["date", "horizon"], how="left")
    return out.sort_values(["date", "model", "horizon"]).reset_index(drop=True)


def hurst_from_moments_pooled(
    scaling: pd.DataFrame,
    *,
    q_col: str = "q",
    lag_col: str = "lag",
    moment_col: str = "moment",
    q_exclude: tuple = (3.0,),
    use_huber: bool = True,
) -> pd.DataFrame:
    """Pooled regression: log m(q, lag) = alpha_q + H * q * log(lag).

    Estimates a single H using all (q, lag) pairs simultaneously. This is much
    more stable than dividing per-q slopes by q, because estimation error in
    individual slopes cancels when pooled.  Uses Huber regression by default to
    down-weight the influence of crisis-period jumps.
    """
    data = scaling.copy()
    data["log_lag_fit"] = np.log(pd.to_numeric(data[lag_col], errors="coerce").replace(0, np.nan))
    data["log_moment_fit"] = np.log(pd.to_numeric(data[moment_col], errors="coerce").replace([0, np.inf, -np.inf], np.nan))
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[q_col, "log_lag_fit", "log_moment_fit"])
    if q_exclude:
        data = data[~data[q_col].isin(q_exclude)].copy()
    if len(data) < 4:
        return pd.DataFrame([{"H": np.nan, "standard_error": np.nan, "r2": np.nan, "n": int(len(data)), "method": "pooled"}])
    q_unique = sorted(data[q_col].unique())
    n = len(data)
    n_q = len(q_unique)
    q_idx = {q: i for i, q in enumerate(q_unique)}
    # Feature matrix: slope column (q * log_lag) + per-q intercept dummies
    X = np.zeros((n, 1 + n_q), dtype=float)
    q_arr = data[q_col].to_numpy(float)
    ll_arr = data["log_lag_fit"].to_numpy(float)
    X[:, 0] = q_arr * ll_arr
    for i, row_q in enumerate(q_arr):
        X[i, 1 + q_idx[row_q]] = 1.0
    y = data["log_moment_fit"].to_numpy(float)
    h_est = np.nan
    se = np.nan
    method_used = "ols"
    if use_huber:
        try:
            from sklearn.linear_model import HuberRegressor
            mdl = HuberRegressor(epsilon=1.35, max_iter=500, fit_intercept=False)
            mdl.fit(X, y)
            h_est = float(mdl.coef_[0])
            y_hat = mdl.predict(X)
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            method_used = "huber"
        except Exception:
            use_huber = False
    if not use_huber:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        h_est = float(beta[0])
        y_hat = X @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        dof = max(n - 1 - n_q, 1)
        sigma2 = ss_res / dof
        try:
            cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = float(np.sqrt(max(cov[0, 0], 0.0)))
        except Exception:
            se = np.nan
        method_used = "ols"
    return pd.DataFrame([{"H": h_est, "standard_error": se, "r2": r2, "n": int(n), "method": method_used}])


def hurst_multi_window(
    returns: pd.Series | pd.DataFrame,
    *,
    windows: tuple = (1, 2, 3, 5, 10, 21),
    q_values: tuple = (0.5, 1.0, 1.5, 2.0),
    lags: tuple = (1, 2, 4, 8, 16, 32),
    eps: float = 1e-12,
    use_huber: bool = True,
) -> pd.DataFrame:
    """Estimate Hurst exponent from log-RV at multiple smoothing windows.

    For each window, builds a rolling sum proxy, log-transforms, and runs the
    pooled moment-scaling regression.  The reported *main_H* is the median over
    the short windows (1–10d) to avoid the upward bias from long smoothing.
    """
    ret = _series(returns, "return")
    rv_1d = (ret * ret).clip(lower=float(eps))
    rows = []
    for w in windows:
        w = int(w)
        if w == 1:
            log_rv = np.log(rv_1d.clip(lower=float(eps))).rename(f"log_rv_{w}d")
        else:
            rv_w = rv_1d.rolling(w).sum() / float(w)
            log_rv = np.log(rv_w.dropna().clip(lower=float(eps))).rename(f"log_rv_{w}d")
        sc = moment_scaling(log_rv, q_values=q_values, lags=lags)
        h_per_q = hurst_from_moments(sc)
        h_pooled = hurst_from_moments_pooled(sc, use_huber=use_huber)
        h_simple = float(h_per_q[h_per_q["q"].between(0.9, 1.1)]["h"].mean()) if not h_per_q.empty else np.nan
        rows.append({
            "window_days": w,
            "n_obs": int(log_rv.shape[0]),
            "H_pooled": float(h_pooled["H"].iloc[0]),
            "H_q1": h_simple,
            "H_median_per_q": float(h_per_q["h"].median()) if not h_per_q.empty else np.nan,
            "r2_pooled": float(h_pooled["r2"].iloc[0]),
            "method": str(h_pooled["method"].iloc[0]),
        })
    df = pd.DataFrame(rows)
    short_mask = df["window_days"].le(10)
    df["main_H"] = float(df.loc[short_mask, "H_pooled"].median()) if short_mask.any() else float(df["H_pooled"].median())
    return df


__all__ = [
    "daily_variance",
    "fgn_covariance",
    "fbm_cholesky_paths",
    "hurst_from_moments",
    "hurst_from_moments_pooled",
    "hurst_multi_window",
    "log_variance",
    "moment_scaling",
    "rough_forecast_frame",
    "rough_kernel_forecasts",
    "rough_kernel_weights",
]
