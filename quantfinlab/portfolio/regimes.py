from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantfinlab.ml.features import drawdown_level


def regime_asset_scores(
    returns: pd.DataFrame,
    proba: pd.DataFrame | None = None,
    *,
    probabilities: pd.DataFrame | None = None,
    assets: Sequence[str] | None = None,
    trend: pd.Series | None = None,
    trend_window: int = 126,
    vol_window: int = 126,
    drawdown_window: int = 252,
    min_weight: float = 1e-4,
    orient: str | None = None,
) -> pd.DataFrame:
    """Score assets conditionally on regime probabilities.

    For each regime state, the function combines probability-weighted return,
    hit rate, trend, volatility penalty, and drawdown quality into a standardized
    asset score. The output can be state-by-asset or asset-by-state.

    Parameters
    ----------
    returns : pandas.DataFrame
        Asset return panel.
    proba : pandas.DataFrame, optional
        Regime probability panel indexed by date.
    probabilities : pandas.DataFrame, optional
        Alias for ``proba``.
    assets : sequence of str, optional
        Asset subset to score.
    trend : pandas.Series, optional
        External trend score. If omitted, trend is computed from trailing returns.
    trend_window : int, default=126
        Window for internal trend calculation.
    vol_window : int, default=126
        Window for trailing volatility.
    drawdown_window : int, default=252
        Window for drawdown quality.
    min_weight : float, default=1e-4
        Minimum probability weight used to judge effective state sample size.
    orient : str, optional
        Output orientation. Values beginning with ``"asset"`` return assets in
        rows; otherwise states are returned in rows.

    Returns
    -------
    pandas.DataFrame
        Regime-conditioned asset scores.

    Raises
    ------
    ValueError
        If neither ``proba`` nor ``probabilities`` is supplied.

    Notes
    -----
    States with very low effective sample size receive a score haircut.
    """

    if proba is None:
        if probabilities is None:
            raise ValueError("proba or probabilities is required.")
        proba = probabilities
        if orient is None:
            orient = "assets"
    orient = "states" if orient is None else str(orient).lower()
    cols = list(assets) if assets is not None else list(returns.columns)
    r = returns[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p = proba.reindex(r.index).astype(float).fillna(0.0)
    if p.empty:
        return pd.DataFrame(columns=cols)
    if r.empty:
        out = pd.DataFrame(0.0, index=p.columns, columns=cols)
        return out.T if orient.startswith("asset") else out
    if trend is None:
        trend_use = (1.0 + r).tail(min(len(r), int(trend_window))).prod(axis=0) - 1.0
    else:
        trend_use = pd.Series(trend, dtype=float).reindex(cols).fillna(0.0)
    vol = r.tail(min(len(r), int(vol_window))).std(ddof=1).replace(0.0, np.nan) * np.sqrt(252.0)
    wealth = (1.0 + r).cumprod()
    dd_window = min(int(drawdown_window), max(20, len(r) // 2))
    dd = drawdown_level(wealth, window=dd_window).iloc[-1].reindex(cols).fillna(0.0)
    rows = {}
    for state in p.columns:
        w = p[state].clip(lower=0.0)
        if float(w.sum()) <= 1e-12:
            w = pd.Series(1.0, index=r.index)
        w = w / float(w.sum())
        mu = r.mul(w, axis=0).sum(axis=0) * 252.0
        hit = r.gt(0.0).mul(w, axis=0).sum(axis=0)
        parts = []
        for s in (mu, hit, trend_use, -vol.fillna(vol.median()), dd):
            s = pd.Series(s, index=cols, dtype=float)
            scale = s.std(ddof=0)
            parts.append((s - s.mean()) / (scale if scale > 1e-12 else 1.0))
        score = 0.35 * parts[0] + 0.20 * parts[1] + 0.25 * parts[2] + 0.10 * parts[3] + 0.10 * parts[4]
        eff_n = float((w > float(min_weight)).sum())
        if eff_n < max(20, 0.05 * len(w)):
            score = score * 0.75
        rows[state] = score
    out = pd.DataFrame(rows).T.reindex(columns=cols)
    return out.T if orient.startswith("asset") else out


def sleeve_weights(
    scores: pd.Series,
    *,
    assets: Sequence[str] | None = None,
    cash_ticker: str = "SHY",
    top_n: int = 5,
    max_weight: float = 0.30,
) -> pd.Series:
    """Convert regime or sleeve scores into a capped risky-sleeve allocation.

    The function selects the top positive scores, normalizes them, applies a
    risk-budget rule, caps individual weights, and assigns any residual capital
    to a cash ticker.

    Parameters
    ----------
    scores : pandas.Series
        Asset score vector.
    assets : sequence of str, optional
        Eligible risky assets. If omitted, all score names except the cash ticker
        are eligible.
    cash_ticker : str, default="SHY"
        Cash or defensive asset receiving residual capital.
    top_n : int, default=5
        Maximum number of risky assets selected.
    max_weight : float, default=0.30
        Maximum per-risky-asset weight.

    Returns
    -------
    pandas.Series
        Fully normalized sleeve weights including the cash ticker.

    Notes
    -----
    If no positive risky score is available, the function allocates fully to the
    cash ticker.
    """

    if assets is None:
        assets = [str(x) for x in pd.Series(scores).index if str(x) != str(cash_ticker)]
    cols = list(dict.fromkeys([*assets, cash_ticker]))
    s = pd.Series(scores, dtype=float).reindex(cols).drop(labels=[cash_ticker], errors="ignore")
    s = s.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    w = pd.Series(0.0, index=cols, dtype=float)
    if s.empty or float(s.iloc[0]) <= 0.0:
        w[cash_ticker] = 1.0
        return w
    selected = s.head(int(top_n))
    positive = selected.clip(lower=0.0)
    if float(positive.sum()) <= 1e-12:
        positive = pd.Series(1.0, index=selected.index, dtype=float)
    raw = positive / float(positive.sum())
    risky_budget = 1.0 if float(selected.mean()) > 0.15 else 0.75
    alloc = raw * risky_budget
    for _ in range(10):
        over = alloc > float(max_weight)
        if not bool(over.any()):
            break
        excess = float((alloc[over] - float(max_weight)).sum())
        alloc[over] = float(max_weight)
        room = (float(max_weight) - alloc[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        alloc.loc[room.index] += excess * room / float(room.sum())
    w.loc[alloc.index] = alloc.clip(upper=float(max_weight))
    w[cash_ticker] = max(1.0 - float(w.drop(labels=[cash_ticker], errors="ignore").sum()), 0.0)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else w


def blend_sleeves(
    proba: pd.Series | None = None,
    sleeves: pd.DataFrame | Mapping[str, pd.Series] | None = None,
    *,
    probabilities: pd.Series | None = None,
    cash_ticker: str | None = None,
) -> pd.Series:
    """Blend state-specific sleeve allocations using regime probabilities.

    Parameters
    ----------
    proba : pandas.Series, optional
        Regime probabilities indexed by sleeve/state name.
    sleeves : pandas.DataFrame or mapping
        State-specific sleeve weights. Rows or mapping keys should correspond to
        probability labels.
    probabilities : pandas.Series, optional
        Alias for ``proba``.
    cash_ticker : str, optional
        Accepted for API compatibility and ignored.

    Returns
    -------
    pandas.Series
        Probability-weighted blended allocation, normalized to sum to one when
        possible.

    Raises
    ------
    ValueError
        If probabilities or sleeves are missing.

    Notes
    -----
    If probabilities sum to zero, they are replaced with equal probabilities.
    """

    del cash_ticker
    if proba is None:
        if probabilities is None:
            raise ValueError("proba or probabilities is required.")
        proba = probabilities
    if sleeves is None:
        raise ValueError("sleeves is required.")
    P = pd.Series(proba, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if float(P.sum()) <= 1e-12:
        P[:] = 1.0 / max(len(P), 1)
    else:
        P = P / float(P.sum())
    S = pd.DataFrame(sleeves).T if isinstance(sleeves, Mapping) else sleeves.copy()
    common = [c for c in P.index if c in S.index]
    if not common:
        return pd.Series(0.0, index=S.columns, dtype=float)
    w = S.loc[common].mul(P.reindex(common), axis=0).sum(axis=0)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else w


def hybrid_weights(
    weights_a: pd.DataFrame | pd.Series,
    weights_b: pd.DataFrame | pd.Series,
    alpha: float = 0.50,
    *,
    weight_a: float | None = None,
    weight_b: float | None = None,
) -> pd.DataFrame | pd.Series:
    if weight_a is not None or weight_b is not None:
        wa = 0.50 if weight_a is None else float(weight_a)
        wb = 0.50 if weight_b is None else float(weight_b)
        denom = wa + wb
        a = 0.50 if denom <= 1e-12 else wa / denom
    else:
        a = float(np.clip(alpha, 0.0, 1.0))
    if isinstance(weights_a, pd.Series) and isinstance(weights_b, pd.Series):
        cols = weights_a.index.union(weights_b.index)
        out = a * weights_a.reindex(cols).fillna(0.0) + (1.0 - a) * weights_b.reindex(cols).fillna(0.0)
        return out / float(out.sum()) if float(out.sum()) > 1e-12 else out
    A = pd.DataFrame(weights_a).copy()
    B = pd.DataFrame(weights_b).copy()
    idx = A.index.union(B.index)
    cols = A.columns.union(B.columns)
    out = a * A.reindex(index=idx, columns=cols).fillna(0.0) + (1.0 - a) * B.reindex(index=idx, columns=cols).fillna(0.0)
    row_sum = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(row_sum, axis=0).fillna(0.0)


def risky_allocation(
    weights: pd.DataFrame | pd.Series,
    risky_assets: Sequence[str] | None = None,
    cash_ticker: str = "SHY",
) -> pd.Series | float:
    if isinstance(weights, pd.Series):
        if risky_assets is not None:
            return float(weights.reindex(list(risky_assets)).fillna(0.0).sum())
        return float(1.0 - weights.get(cash_ticker, 0.0))
    W = weights.astype(float)
    if risky_assets is not None:
        return W.reindex(columns=list(risky_assets)).fillna(0.0).sum(axis=1)
    return 1.0 - W.get(cash_ticker, pd.Series(0.0, index=W.index)).astype(float)


__all__ = [
    "blend_sleeves",
    "hybrid_weights",
    "regime_asset_scores",
    "risky_allocation",
    "sleeve_weights",
]
