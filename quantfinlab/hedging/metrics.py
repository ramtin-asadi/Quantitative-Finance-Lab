from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.hedging.relations import hedge_proxy_ret, rel
from quantfinlab.risk import capm_ols, historical_es, max_drawdown, rolling_beta, total_return


def _frame(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise InputError("input must be a pandas DataFrame.")
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _align(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    z = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    z.columns = ["x", "y"]
    return z


def _ann_vol(x: pd.Series, ann: float) -> float:
    s = pd.to_numeric(pd.Series(x), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(s.std(ddof=1) * math.sqrt(float(ann))) if len(s) > 1 else float("nan")


def _average_holding_days(signal: pd.Series) -> float:
    active = pd.Series(signal).fillna(0.0).ne(0.0)
    runs = []
    n = 0
    for on in active:
        if on:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return float(np.mean(runs)) if runs else 0.0


def coverage_table(frame: pd.DataFrame, rels: Sequence[rel]) -> pd.DataFrame:
    """Report data availability for target/hedge relationships.

    Parameters
    ----------
    frame : pandas.DataFrame
        Price or return panel.
    rels : sequence
        Relationship objects containing target and hedge asset names.

    Returns
    -------
    pandas.DataFrame
        One row per relationship with target, hedge labels, first/last valid date,
        complete-row observation count, missing percentage, and inclusion flag.

    Notes
    -----
    A relationship is included only when all required assets are present and at
    least one complete observation exists.
    """

    x = _frame(frame)
    rows = []
    for r in rels:
        present = [a for a in r.assets if a in x.columns]
        missing = [a for a in r.assets if a not in x.columns]
        sub = x[present] if present else pd.DataFrame(index=x.index)
        obs = int(sub.dropna(how="any").shape[0]) if len(present) == len(r.assets) else 0
        miss_pct = float(sub.isna().mean().mean()) if len(present) else 1.0
        valid = sub.dropna(how="any")
        rows.append(
            {
                "relationship": r.name,
                "target": r.target,
                "hedges": ", ".join(r.hedges),
                "start": valid.index.min() if len(valid) else pd.NaT,
                "end": valid.index.max() if len(valid) else pd.NaT,
                "obs": obs,
                "missing_pct": miss_pct,
                "included": len(missing) == 0 and obs > 0,
            }
        )
    return pd.DataFrame(rows)


def diag_table(
    ret: pd.DataFrame,
    rels: Sequence[rel],
    *,
    beta_log: Mapping[str, Mapping[str, object]] | None = None,
    ann: float = 252.0,
    win: int = 252,
) -> pd.DataFrame:
    """Build relationship-level diagnostics before hedge-model scoring.

    The table summarizes target volatility, hedge-proxy volatility, correlation,
    OLS R-squared, beta instability, and observation count for each relationship.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel.
    rels : sequence
        Hedge relationships to evaluate.
    beta_log : mapping, optional
        Optional beta schedules or logs used to estimate beta dispersion.
    ann : float, default=252.0
        Annualization factor.
    win : int, default=252
        Rolling window used for beta-dispersion fallback.

    Returns
    -------
    pandas.DataFrame
        Relationship diagnostic table.

    Notes
    -----
    The hedge proxy is an equal-weight average of the hedge assets. The beta IQR
    uses supplied beta logs when available; otherwise it is estimated from rolling
    single-factor beta.
    """

    rret = _frame(ret)
    rows = []
    for r in rels:
        if any(a not in rret.columns for a in r.assets):
            continue
        y = rret[r.target]
        proxy = hedge_proxy_ret(rret, r)
        z = _align(y, proxy)
        if len(z) < 5:
            continue
        _, _, r2 = capm_ols(z["x"], z["y"])
        beta_iqr = float("nan")
        if beta_log:
            frames = [
                v.get("traded")
                for v in beta_log.values()
                if isinstance(v, Mapping) and v.get("rel") == r.name and isinstance(v.get("traded"), pd.DataFrame)
            ]
            if frames:
                vals = [float(f.quantile(0.75).sub(f.quantile(0.25)).mean()) for f in frames if not f.empty]
                beta_iqr = float(np.nanmean(vals)) if vals else float("nan")
        if not np.isfinite(beta_iqr):
            b = rolling_beta(z["x"], z["y"], window=min(int(win), max(len(z) // 2, 20)))
            beta_iqr = float(b.quantile(0.75) - b.quantile(0.25)) if b.notna().sum() else float("nan")
        rows.append(
            {
                "relationship": r.name,
                "target_vol": _ann_vol(z["x"], ann),
                "hedge_vol": _ann_vol(z["y"], ann),
                "corr": float(z["x"].corr(z["y"])),
                "r2": r2,
                "beta_iqr": beta_iqr,
                "obs": int(len(z)),
            }
        )
    return pd.DataFrame(rows)


def _strategy_name(rel_name: str, model: str) -> str:
    return f"{rel_name} | {model}"


def model_table(
    bt: Mapping[str, object],
    rels: Sequence[rel],
    ret: pd.DataFrame,
    *,
    ann: float = 252.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute hedge-model risk-reduction metrics.

    For each relationship and hedge model, the function compares the hedged return
    stream with the unhedged target book. It reports volatility reduction, ES
    reduction, drawdown improvement, beta reduction, turnover, cost drag, hedge
    error volatility, beta stability, and rebalance count.

    Parameters
    ----------
    bt : mapping
        Mapping of strategy name to backtest result. Names are expected to follow
        the relationship/model naming convention used by the hedging workflows.
    rels : sequence
        Hedge relationships.
    ret : pandas.DataFrame
        Original return panel used to build hedge proxies.
    ann : float, default=252.0
        Annualization factor.
    alpha : float, default=0.05
        Tail probability used for expected shortfall.

    Returns
    -------
    pandas.DataFrame
        One row per relationship/model combination.

    Notes
    -----
    Positive reduction metrics indicate improvement relative to the unhedged target.
    Turnover and cost metrics are retained so the scoring layer can penalize models
    that reduce volatility only by trading excessively.
    """

    rret = _frame(ret)
    rows = []
    for r in rels:
        base_name = _strategy_name(r.name, "target")
        if base_name not in bt or any(a not in rret.columns for a in r.assets):
            continue
        base = bt[base_name]
        base_ret = pd.Series(base.net_returns, dtype=float)
        proxy = hedge_proxy_ret(rret, r)
        zbase = _align(base_ret, proxy)
        _, beta_base, _ = capm_ols(zbase["x"], zbase["y"]) if len(zbase) >= 3 else (np.nan, np.nan, np.nan)
        for name, res in bt.items():
            if not str(name).startswith(f"{r.name} | ") or str(name) == base_name:
                continue
            model = str(name).split(" | ", 1)[1]
            mret = pd.Series(res.net_returns, dtype=float)
            z = pd.concat([base_ret.rename("base"), mret.rename("model")], axis=1).dropna()
            if len(z) < 5:
                continue
            vol_base = float(z["base"].std(ddof=1) * math.sqrt(float(ann)))
            vol_model = float(z["model"].std(ddof=1) * math.sqrt(float(ann)))
            es_base = historical_es(z["base"], alpha=alpha)
            es_model = historical_es(z["model"], alpha=alpha)
            dd_base = max_drawdown(z["base"], input_kind="returns")
            dd_model = max_drawdown(z["model"], input_kind="returns")
            zbeta = _align(mret, proxy)
            _, beta_model, _ = capm_ols(zbeta["x"], zbeta["y"]) if len(zbeta) >= 3 else (np.nan, np.nan, np.nan)
            weights = getattr(res, "beta", None)
            if not isinstance(weights, pd.DataFrame):
                weights = getattr(res, "weights", pd.DataFrame())
            beta_iqr = float("nan")
            beta_jump = float("nan")
            if isinstance(weights, pd.DataFrame) and not weights.empty:
                cols = [h for h in r.hedges if h in weights.columns]
                if cols:
                    bpath = weights[cols].replace([np.inf, -np.inf], np.nan)
                    if not hasattr(res, "beta"):
                        bpath = -bpath
                    beta_iqr = float(bpath.quantile(0.75).sub(bpath.quantile(0.25)).mean())
                    beta_jump = float(bpath.diff().abs().mean().mean())
            turnover_s = pd.Series(getattr(res, "turnover", pd.Series(dtype=float)), dtype=float)
            cost_s = pd.Series(getattr(res, "cost", pd.Series(dtype=float)), dtype=float)
            rows.append(
                {
                    "relationship": r.name,
                    "model": model,
                    "vol_red": 1.0 - vol_model / vol_base if vol_base > 1e-12 else np.nan,
                    "es_red": 1.0 - abs(es_model) / abs(es_base) if abs(es_base) > 1e-12 else np.nan,
                    "maxdd_diff": abs(dd_base) - abs(dd_model) if np.isfinite(dd_base) and np.isfinite(dd_model) else np.nan,
                    "beta_red": 1.0 - abs(beta_model) / abs(beta_base) if abs(beta_base) > 1e-12 else np.nan,
                    "turnover": float(turnover_s.mean()) if len(turnover_s) else 0.0,
                    "turnover_ann": float(turnover_s.mean() * float(ann)) if len(turnover_s) else 0.0,
                    "cost_drag": max(total_return(res.gross_values) - total_return(res.net_values), 0.0),
                    "cost_drag_ann": float(cost_s.mean() * float(ann)) if len(cost_s) else 0.0,
                    "hedge_err_vol": vol_model,
                    "beta_iqr": beta_iqr,
                    "beta_jump": beta_jump,
                    "beta_stability": 1.0 / (1.0 + beta_iqr + 5.0 * beta_jump)
                    if np.isfinite(beta_iqr) and np.isfinite(beta_jump)
                    else np.nan,
                    "rebalance_count": int((turnover_s.abs() > 1e-12).sum()) if len(turnover_s) else 0,
                }
            )
    return pd.DataFrame(rows)


def quality_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Raw relationship quality summary across best available models."""
    if tab.empty:
        return pd.DataFrame()
    rows = []
    for rel_name, g in tab.groupby("relationship"):
        rows.append(
            {
                "relationship": rel_name,
                "best_vol_red": float(g["vol_red"].max()),
                "median_vol_red": float(g["vol_red"].median()),
                "best_es_red": float(g["es_red"].max()),
                "median_es_red": float(g["es_red"].median()),
                "best_maxdd_diff": float(g["maxdd_diff"].max()),
                "median_cost_drag_ann": float(g["cost_drag_ann"].median()) if "cost_drag_ann" in g else float("nan"),
                "median_turnover_ann": float(g["turnover_ann"].median()) if "turnover_ann" in g else float("nan"),
                "model_count": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("best_vol_red", ascending=False).reset_index(drop=True)


def _norm_within_group(s: pd.Series, *, lower_is_better: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() == 0:
        return pd.Series(0.0 if not lower_is_better else 1.0, index=s.index, dtype=float)
    lo = float(x.quantile(0.05))
    hi = float(x.quantile(0.95))
    xc = x.clip(lower=lo, upper=hi)
    span = float(xc.max() - xc.min())
    if span <= 1e-12 or not np.isfinite(span):
        out = pd.Series(0.5, index=s.index, dtype=float)
    else:
        out = (xc - float(xc.min())) / span
    if lower_is_better:
        out = 1.0 - out
    return out.fillna(0.0 if not lower_is_better else 1.0)


def score_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Score hedge models within each relationship.

    The score combines normalized volatility reduction, ES reduction, drawdown
    improvement, beta reduction, and beta stability, while penalizing annualized
    turnover, cost drag, and materially worse drawdowns.

    Parameters
    ----------
    tab : pandas.DataFrame
        Output of ``model_table`` or a compatible table with hedge metrics.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with component score columns and a final ``score`` column.

    Notes
    -----
    Scores are normalized within each relationship, not globally. This prevents
    relationships with naturally larger or smaller hedge benefits from dominating
    the ranking across unrelated hedge books.
    """

    if tab.empty:
        return tab.copy()
    out = tab.copy()
    positive = ["vol_red", "es_red", "maxdd_diff", "beta_red", "beta_stability"]
    lower = ["turnover_ann", "cost_drag_ann"]
    for col in positive + lower:
        if col not in out.columns:
            out[col] = np.nan
    parts = []
    for _, g in out.groupby("relationship", sort=False):
        h = g.copy()
        h["vol_score"] = _norm_within_group(h["vol_red"])
        h["es_score"] = _norm_within_group(h["es_red"])
        h["maxdd_score"] = _norm_within_group(h["maxdd_diff"])
        h["beta_score"] = _norm_within_group(h["beta_red"])
        h["stability_score"] = _norm_within_group(h["beta_stability"])
        h["turnover_penalty"] = 1.0 - _norm_within_group(h["turnover_ann"], lower_is_better=True)
        h["cost_penalty"] = 1.0 - _norm_within_group(h["cost_drag_ann"], lower_is_better=True)
        h["drawdown_worse_penalty"] = (-pd.to_numeric(h["maxdd_diff"], errors="coerce")).clip(lower=0.0) / 0.10
        h["drawdown_worse_penalty"] = h["drawdown_worse_penalty"].clip(upper=1.0).fillna(0.0)
        h.loc[h["maxdd_diff"] < -0.05, "vol_score"] = np.minimum(
            h.loc[h["maxdd_diff"] < -0.05, "vol_score"], 0.50
        )
        h["score"] = (
            0.30 * h["vol_score"]
            + 0.20 * h["es_score"]
            + 0.15 * h["maxdd_score"]
            + 0.15 * h["beta_score"]
            + 0.10 * h["stability_score"]
            - 0.05 * h["turnover_penalty"]
            - 0.05 * h["cost_penalty"]
            - 0.10 * h["drawdown_worse_penalty"]
        )
        parts.append(h)
    return pd.concat(parts, axis=0).reset_index(drop=True)


def best_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Select the best hedge model for each relationship.

    Parameters
    ----------
    tab : pandas.DataFrame
        Scored hedge table containing ``relationship``, ``model``, and ``score``.

    Returns
    -------
    pandas.DataFrame
        Compact table with the best model and key metrics for each relationship.

    Notes
    -----
    If the input is empty, the function returns an empty table with the expected
    columns.
    """

    if tab.empty:
        return pd.DataFrame(columns=["relationship", "best_model", "score", "vol_red", "es_red", "maxdd_diff", "cost_drag_ann"])
    idx = tab.groupby("relationship")["score"].idxmax()
    cols = ["relationship", "model", "score", "vol_red", "es_red", "maxdd_diff", "cost_drag_ann"]
    out = tab.loc[idx, cols].copy().sort_values("relationship")
    return out.rename(columns={"model": "best_model"}).reset_index(drop=True)


def robust_table(tab: pd.DataFrame) -> pd.DataFrame:
    """Model robustness across relationships."""
    if tab.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "median_score",
                "median_vol_red",
                "median_es_red",
                "median_cost_drag_ann",
                "win_count",
                "failure_count",
            ]
        )
    wins = best_table(tab)["best_model"].value_counts()
    rows = []
    for model, g in tab.groupby("model"):
        rows.append(
            {
                "model": model,
                "median_score": float(g["score"].median()),
                "median_vol_red": float(g["vol_red"].median()),
                "median_es_red": float(g["es_red"].median()),
                "median_cost_drag_ann": float(g["cost_drag_ann"].median()) if "cost_drag_ann" in g else float("nan"),
                "win_count": int(wins.get(model, 0)),
                "failure_count": int(((g["vol_red"] <= 0) | (g["score"] <= 0)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["median_score", "win_count"], ascending=False).reset_index(drop=True)


def residual_trade_table(
    backtests: Mapping[str, object],
    signals: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, object]],
    *,
    ann: float = 252.0,
) -> pd.DataFrame:
    """Summarize residual spread-trading backtests.

    Parameters
    ----------
    backtests : mapping
        Mapping from strategy key to residual-trade backtest result.
    signals : mapping
        Mapping from strategy key to signal DataFrame containing a ``signal``
        column.
    metadata : mapping
        Mapping from strategy key to metadata such as pair name and beta source.
    ann : float, default=252.0
        Annualization factor.

    Returns
    -------
    pandas.DataFrame
        Performance table with trade count, average holding period, net return,
        annualized volatility, Sharpe ratio, maximum drawdown, cost drag, and
        identifying metadata.

    Notes
    -----
    Trade count is inferred from transitions from flat to nonzero signal.
    """

    rows = []
    for key, res in backtests.items():
        ret = pd.Series(getattr(res, "net_returns", pd.Series(dtype=float)), dtype=float).dropna()
        sig = signals.get(key, pd.DataFrame()).get("signal", pd.Series(dtype=float))
        meta = metadata.get(key, {})
        vol = float(ret.std(ddof=1) * math.sqrt(float(ann))) if len(ret) > 1 else float("nan")
        sharpe = float(ret.mean() / ret.std(ddof=1) * math.sqrt(float(ann))) if ret.std(ddof=1) > 1e-12 else float("nan")
        rows.append({
            "pair": meta.get("pair", ""), "beta_source": meta.get("beta_source", ""),
            "trades": int((sig.ne(0) & sig.shift(1).fillna(0).eq(0)).sum()) if len(sig) else 0,
            "avg_hold": _average_holding_days(sig), "net_return": total_return(res.net_values),
            "ann_vol": vol, "sharpe": sharpe, "maxdd": max_drawdown(ret, input_kind="returns"),
            "cost_drag": max(total_return(res.gross_values) - total_return(res.net_values), 0.0),
            "cost_drag_ann": res.cost.mean() * float(ann) if len(res.cost) else 0.0, "key": key})
    return pd.DataFrame(rows)


__all__ = [
    "best_table",
    "coverage_table",
    "diag_table",
    "model_table",
    "quality_table",
    "residual_trade_table",
    "robust_table",
    "score_table",
]
