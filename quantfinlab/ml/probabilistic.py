from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """Compute quantile pinball loss.

    Parameters
    ----------
    y_true : array-like
        Realized target values.
    y_pred : array-like
        Predicted quantile values.
    tau : float
        Quantile level in ``(0, 1)``.

    Returns
    -------
    float
        Mean pinball loss over valid aligned observations.
    """
    y = pd.Series(y_true, dtype=float)
    q = pd.Series(y_pred, dtype=float).reindex(y.index)
    err = y - q
    loss = np.maximum(float(tau) * err, (float(tau) - 1.0) * err)
    return float(pd.Series(loss).replace([np.inf, -np.inf], np.nan).dropna().mean())


def gaussian_nll(y_true, mean, variance, *, eps: float = 1e-8) -> float:
    """Compute Gaussian negative log likelihood for probabilistic forecasts.

    Parameters
    ----------
    y_true : array-like
        Realized target values.
    mean : array-like
        Forecast mean.
    variance : array-like
        Forecast variance.
    eps : float, default=1e-8
        Lower variance floor for numerical stability.

    Returns
    -------
    float
        Mean Gaussian negative log likelihood.
    """
    y = pd.Series(y_true, dtype=float)
    mu = pd.Series(mean, dtype=float).reindex(y.index)
    var = pd.Series(variance, dtype=float).reindex(y.index).clip(lower=float(eps))
    nll = 0.5 * (np.log(2.0 * np.pi * var) + np.square(y - mu) / var)
    return float(pd.Series(nll).replace([np.inf, -np.inf], np.nan).dropna().mean())


def enforce_quantile_order(q_low, q_mid, q_high):
    """Sort low, middle, and high quantile forecasts row by row.

    Parameters
    ----------
    q_low, q_mid, q_high : array-like
        Lower, central, and upper quantile forecasts.

    Returns
    -------
    tuple of pandas.Series
        Ordered ``(q_low, q_mid, q_high)`` series with the same index.

    Notes
    -----
    This is a simple post-processing guard against quantile crossing.
    """
    q = pd.concat(
        [
            pd.Series(q_low, dtype=float).rename("q_low"),
            pd.Series(q_mid, dtype=float).rename("q_mid"),
            pd.Series(q_high, dtype=float).rename("q_high"),
        ],
        axis=1,
    )
    vals = np.sort(q.to_numpy(dtype=float), axis=1)
    return (
        pd.Series(vals[:, 0], index=q.index, name="q_low"),
        pd.Series(vals[:, 1], index=q.index, name="q_mid"),
        pd.Series(vals[:, 2], index=q.index, name="q_high"),
    )


ordered_quantiles = enforce_quantile_order


def quantile_metrics(
    data: pd.DataFrame,
    *,
    y_col: str,
    quantile_sets: dict[str, tuple[str, str, str]],
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
                "model": str(name),
                "n": int(len(d)),
                "coverage_80": interval_coverage(d[y_col], d[low_col], d[high_col]),
                "avg_width": interval_width(d[low_col], d[high_col]),
                "pinball_q10": pinball_loss(d[y_col], d[low_col], 0.10),
                "pinball_q50": pinball_loss(d[y_col], d[mid_col], 0.50),
                "pinball_q90": pinball_loss(d[y_col], d[high_col], 0.90),
            }
        )
    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def nll_metrics(
    data: pd.DataFrame,
    *,
    y_col: str,
    mean_col: str,
    sigma_col: str,
) -> pd.DataFrame:
    """Gaussian NLL diagnostics for a mean/sigma forecast."""
    cols = [y_col, mean_col, sigma_col]
    if not set(cols).issubset(data.columns):
        return pd.DataFrame()
    d = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty:
        return pd.DataFrame()
    variance = np.square(d[sigma_col].clip(lower=1e-8))
    return pd.DataFrame(
        {
            "n": [int(len(d))],
            "gaussian_nll": [gaussian_nll(d[y_col], d[mean_col], variance)],
            "avg_sigma": [float(d[sigma_col].mean())],
            "mean_abs_error": [float((d[mean_col] - d[y_col]).abs().mean())],
        },
        index=[mean_col],
    )


def conformal_offsets(
    y,
    q_low,
    q_high,
    *,
    alpha: float = 0.20,
) -> tuple[float, float]:
    """Compute split-conformal expansion offsets for a central prediction interval.

    Parameters
    ----------
    y : array-like
        Realized outcomes in the calibration sample.
    q_low : array-like
        Lower interval forecast.
    q_high : array-like
        Upper interval forecast.
    alpha : float, default=0.20
        Miscoverage level for the two-sided interval.

    Returns
    -------
    tuple of float
        ``(offset_low, offset_high)`` to subtract from the lower bound and add to
        the upper bound.

    Notes
    -----
    The lower and upper offsets are calibrated separately from one-sided residual
    scores.
    """
    yy = pd.Series(y, dtype=float)
    lo = pd.Series(q_low, dtype=float).reindex(yy.index)
    hi = pd.Series(q_high, dtype=float).reindex(yy.index)
    data = pd.concat([yy.rename("y"), lo.rename("lo"), hi.rename("hi")], axis=1).dropna()
    if data.empty:
        return 0.0, 0.0
    low_score = (data["lo"] - data["y"]).clip(lower=0.0)
    high_score = (data["y"] - data["hi"]).clip(lower=0.0)
    q = 1.0 - float(alpha) / 2.0
    return float(low_score.quantile(q)), float(high_score.quantile(q))


def conformalize_interval(
    q_low,
    q_high,
    *,
    offset_low: float,
    offset_high: float,
) -> tuple[pd.Series, pd.Series]:
    """Apply conformal offsets to interval forecasts.

    Parameters
    ----------
    q_low : array-like
        Lower interval forecast.
    q_high : array-like
        Upper interval forecast.
    offset_low : float
        Amount subtracted from the lower bound.
    offset_high : float
        Amount added to the upper bound.

    Returns
    -------
    tuple of pandas.Series
        Conformalized lower and upper interval series named ``q_low_c`` and
        ``q_high_c``.
    """
    low = pd.Series(q_low, dtype=float) - float(offset_low)
    high = pd.Series(q_high, dtype=float).reindex(low.index) + float(offset_high)
    return low.rename("q_low_c"), high.rename("q_high_c")


def conformal_quantiles(
    q_low,
    q_high,
    *,
    y=None,
    alpha: float = 0.20,
    offset_low: float | None = None,
    offset_high: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Apply split-conformal offsets, computing them from ``y`` when needed."""
    if offset_low is None or offset_high is None:
        if y is None:
            raise ValueError("y is required when conformal offsets are not supplied.")
        offset_low, offset_high = conformal_offsets(y, q_low, q_high, alpha=alpha)
    return conformalize_interval(q_low, q_high, offset_low=float(offset_low), offset_high=float(offset_high))


def rolling_conformal_offsets(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str,
    low_col: str,
    high_col: str,
    alpha: float = 0.20,
    lookback_days: int = 504,
    gap_days: int = 21,
    min_obs: int = 126,
) -> pd.DataFrame:
    """Compute date-wise rolling conformal interval offsets.

    For each forecast date, the calibration set contains only historical rows in a
    lookback window ending ``gap_days`` business days before the forecast date. This
    gap is intended to avoid using still-overlapping forward labels.

    Parameters
    ----------
    frame : pandas.DataFrame
        Forecast table containing date, realized outcome, and interval columns.
    date_col : str, default="date"
        Date column.
    y_col : str
        Realized outcome column.
    low_col : str
        Lower interval forecast column.
    high_col : str
        Upper interval forecast column.
    alpha : float, default=0.20
        Miscoverage level.
    lookback_days : int, default=504
        Historical calibration lookback in business days.
    gap_days : int, default=21
        Business-day embargo between calibration rows and the forecast date.
    min_obs : int, default=126
        Minimum calibration observations required before nonzero offsets are used.

    Returns
    -------
    pandas.DataFrame
        Offset table with date, ``offset_low``, ``offset_high``, and
        ``calibration_n``.

    Notes
    -----
    When the calibration sample is too small, offsets are set to zero and the
    calibration count is still reported.
    """
    data = pd.DataFrame(frame).copy()
    if data.empty:
        return pd.DataFrame(columns=[date_col, "offset_low", "offset_high", "calibration_n"])
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[date_col, y_col, low_col, high_col])
    rows: list[dict[str, float | pd.Timestamp]] = []
    q = 1.0 - float(alpha) / 2.0
    for dt in pd.DatetimeIndex(data[date_col].drop_duplicates()).sort_values():
        start = pd.Timestamp(dt) - pd.tseries.offsets.BDay(int(lookback_days))
        end = pd.Timestamp(dt) - pd.tseries.offsets.BDay(int(gap_days))
        hist = data[data[date_col].between(start, end)]
        if len(hist) < int(min_obs):
            rows.append({"date": pd.Timestamp(dt), "offset_low": 0.0, "offset_high": 0.0, "calibration_n": len(hist)})
            continue
        low_score = (hist[low_col].astype(float) - hist[y_col].astype(float)).clip(lower=0.0)
        high_score = (hist[y_col].astype(float) - hist[high_col].astype(float)).clip(lower=0.0)
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "offset_low": float(low_score.quantile(q)),
                "offset_high": float(high_score.quantile(q)),
                "calibration_n": len(hist),
            }
        )
    out = pd.DataFrame(rows)
    return out.rename(columns={"date": date_col})


def apply_rolling_conformal(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    y_col: str,
    low_col: str,
    high_col: str,
    alpha: float = 0.20,
    lookback_days: int = 504,
    gap_days: int = 21,
    min_obs: int = 126,
    output_low: str = "q_low_c",
    output_high: str = "q_high_c",
) -> pd.DataFrame:
    """Attach rolling conformal offsets and adjusted interval columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Forecast table.
    date_col : str, default="date"
        Date column.
    y_col : str
        Realized outcome column.
    low_col : str
        Lower interval forecast column.
    high_col : str
        Upper interval forecast column.
    alpha : float, default=0.20
        Miscoverage level.
    lookback_days : int, default=504
        Calibration lookback in business days.
    gap_days : int, default=21
        Embargo gap in business days.
    min_obs : int, default=126
        Minimum calibration observations.
    output_low : str, default="q_low_c"
        Output lower interval column.
    output_high : str, default="q_high_c"
        Output upper interval column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with conformal offsets and adjusted interval bounds.
    """
    data = pd.DataFrame(frame).copy()
    offsets = rolling_conformal_offsets(
        data,
        date_col=date_col,
        y_col=y_col,
        low_col=low_col,
        high_col=high_col,
        alpha=alpha,
        lookback_days=lookback_days,
        gap_days=gap_days,
        min_obs=min_obs,
    )
    out = data.merge(offsets, on=date_col, how="left")
    out[["offset_low", "offset_high", "calibration_n"]] = out[
        ["offset_low", "offset_high", "calibration_n"]
    ].fillna(0.0)
    out[output_low] = out[low_col].astype(float) - out["offset_low"].astype(float)
    out[output_high] = out[high_col].astype(float) + out["offset_high"].astype(float)
    return out


def interval_coverage(y, q_low, q_high) -> float:
    yy = pd.Series(y, dtype=float)
    lo = pd.Series(q_low, dtype=float).reindex(yy.index)
    hi = pd.Series(q_high, dtype=float).reindex(yy.index)
    data = pd.concat([yy.rename("y"), lo.rename("lo"), hi.rename("hi")], axis=1).dropna()
    if data.empty:
        return float("nan")
    return float(data["y"].between(data["lo"], data["hi"]).mean())


def interval_width(q_low, q_high) -> float:
    lo = pd.Series(q_low, dtype=float)
    hi = pd.Series(q_high, dtype=float).reindex(lo.index)
    width = (hi - lo).replace([np.inf, -np.inf], np.nan).dropna()
    return float(width.mean()) if len(width) else float("nan")


def calibration_table(y, q_low, q_high, *, n_bins: int = 5) -> pd.DataFrame:
    yy = pd.Series(y, dtype=float)
    lo = pd.Series(q_low, dtype=float).reindex(yy.index)
    hi = pd.Series(q_high, dtype=float).reindex(yy.index)
    data = pd.concat([yy.rename("y"), lo.rename("lo"), hi.rename("hi")], axis=1).dropna()
    if data.empty:
        return pd.DataFrame(columns=["bin", "coverage", "width", "count"])
    data["width"] = data["hi"] - data["lo"]
    ranks = data["width"].rank(method="first")
    data["bin"] = pd.qcut(ranks, int(n_bins), labels=False) + 1
    return (
        data.assign(hit=data["y"].between(data["lo"], data["hi"]))
        .groupby("bin")
        .agg(coverage=("hit", "mean"), width=("width", "mean"), count=("hit", "size"))
    )


__all__ = [
    "calibration_table",
    "apply_rolling_conformal",
    "conformal_quantiles",
    "conformal_offsets",
    "conformalize_interval",
    "enforce_quantile_order",
    "gaussian_nll",
    "interval_coverage",
    "interval_width",
    "nll_metrics",
    "ordered_quantiles",
    "pinball_loss",
    "quantile_metrics",
    "rolling_conformal_offsets",
]
