from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


def state_profile(
    features: pd.DataFrame,
    labels: pd.Series | Sequence[int],
    outcomes: pd.DataFrame | pd.Series | None = None,
) -> pd.DataFrame:
    if isinstance(labels, pd.Series):
        lab = labels.copy()
    else:
        lab = pd.Series(labels, index=features.index if len(labels) == len(features) else None)
    lab.name = "state"
    parts = [features.copy()]
    if outcomes is not None:
        out = outcomes.to_frame() if isinstance(outcomes, pd.Series) else outcomes.copy()
        parts.append(out.add_prefix("outcome_"))
    z = pd.concat(parts + [lab], axis=1).replace([np.inf, -np.inf], np.nan).dropna(subset=["state"])
    numeric_cols = [c for c in z.columns if c != "state"]
    grouped = z.groupby("state")[numeric_cols].mean()
    grouped.insert(0, "share", z.groupby("state").size() / len(z))
    grouped.insert(0, "observations", z.groupby("state").size())
    grouped.index = grouped.index.astype(int)
    return grouped.sort_index()


def sort_states_by_profile(profile: pd.DataFrame) -> list[int]:
    if profile.empty:
        return []
    cols = [c for c in profile.columns if c not in {"observations", "share"}]
    if not cols:
        return list(profile.index)
    z = profile[cols].astype(float)
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=0).replace(0.0, np.nan)
    z = z.fillna(0.0)
    risk_terms = (
        "spy",
        "qqq",
        "iwm",
        "efa",
        "eem",
        "hyg",
        "risk_on",
        "breadth",
        "growth",
        "credit",
        "cyclical",
        "rotation",
        "sector_ew",
        "hit",
        "return",
    )
    stress_terms = ("vol", "corr", "dispersion")
    score = pd.Series(0.0, index=profile.index)
    for col in cols:
        name = str(col).lower()
        if "risk_defensive_spread" in name:
            score = score + 3.0 * z[col]
            continue
        if any(term in name for term in risk_terms):
            score = score + z[col]
        if any(term in name for term in stress_terms):
            score = score - z[col]
        if "dd" in name or "drawdown" in name:
            score = score + z[col]
        if "defensive_sleeve" in name:
            score = score - 0.5 * z[col]
    return score.sort_values(ascending=False).index.astype(int).tolist()


def remap_labels(labels: pd.Series | Sequence[int], order: Sequence[int]) -> pd.Series | np.ndarray:
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    if isinstance(labels, pd.Series):
        return labels.map(mapping).astype("Int64").rename(labels.name)
    arr = np.asarray(labels)
    return np.asarray([mapping.get(int(v), int(v)) for v in arr], dtype=int)


def proba_frame(
    proba: np.ndarray | pd.DataFrame,
    index: Sequence[pd.Timestamp] | pd.Index,
    prefix: str = "state",
) -> pd.DataFrame:
    if isinstance(proba, pd.DataFrame):
        out = proba.copy()
        out.index = pd.to_datetime(index)
        return out
    arr = np.asarray(proba, dtype=float)
    cols = [f"{prefix}_{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.to_datetime(index), columns=cols)


def transition_table(labels: pd.Series | Sequence[int], normalize: bool = True) -> pd.DataFrame:
    lab = pd.Series(labels).dropna().astype(int)
    states = sorted(lab.unique())
    out = pd.DataFrame(0.0, index=states, columns=states)
    for current, nxt in zip(lab.iloc[:-1], lab.iloc[1:], strict=False):
        out.loc[int(current), int(nxt)] += 1.0
    if normalize:
        out = out.div(out.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return out


def duration_table(labels: pd.Series | Sequence[int]) -> pd.DataFrame:
    lab = pd.Series(labels).dropna().astype(int)
    if lab.empty:
        return pd.DataFrame(columns=["state", "episodes", "avg_duration", "median_duration", "max_duration"])
    groups = (lab != lab.shift()).cumsum()
    groups.name = None
    runs = pd.DataFrame({"state": lab.groupby(groups).first(), "duration": lab.groupby(groups).size()})
    out = runs.groupby("state")["duration"].agg(["count", "mean", "median", "max"])
    out.columns = ["episodes", "avg_duration", "median_duration", "max_duration"]
    return out.reset_index()


def posterior_confidence(proba: pd.DataFrame | np.ndarray) -> pd.Series:
    p = pd.DataFrame(proba).astype(float)
    return p.max(axis=1).rename("posterior_confidence")


def posterior_entropy(proba: pd.DataFrame | np.ndarray) -> pd.Series:
    p = pd.DataFrame(proba).astype(float).clip(lower=1e-12)
    p = p.div(p.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    ent = -(p * np.log(p.clip(lower=1e-12))).sum(axis=1)
    if p.shape[1] > 1:
        ent = ent / np.log(p.shape[1])
    return ent.rename("posterior_entropy")


def regime_separation_score(
    outcomes: pd.DataFrame | pd.Series,
    labels: pd.Series | Sequence[int],
) -> float:
    out = outcomes.to_frame() if isinstance(outcomes, pd.Series) else outcomes.copy()
    lab = labels if isinstance(labels, pd.Series) else pd.Series(labels, index=out.index if len(labels) == len(out) else None)
    joined = pd.concat([out, pd.Series(lab, name="state")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty or joined["state"].nunique() < 2:
        return float("nan")
    by_state = joined.groupby("state")[out.columns].mean()
    denom = joined[out.columns].std(ddof=1).replace(0.0, np.nan)
    return float((by_state.max() - by_state.min()).div(denom).replace([np.inf, -np.inf], np.nan).mean())


def _coerce_labels(labels, x=None) -> pd.Series:
    if isinstance(labels, pd.Series):
        return labels.copy()
    if isinstance(x, pd.DataFrame) and len(labels) == len(x):
        return pd.Series(labels, index=x.index)
    return pd.Series(labels)


def model_quality_row(
    name: str,
    labels_or_x,
    labels: pd.Series | Sequence[int] | None = None,
    *,
    x: pd.DataFrame | np.ndarray | None = None,
    proba: pd.DataFrame | np.ndarray | None = None,
    outcomes: pd.DataFrame | pd.Series | None = None,
    loglike: float | None = None,
    likelihood: float | None = None,
    aic: float | None = None,
    bic: float | None = None,
) -> dict[str, float | str]:
    if labels is not None:
        if x is None:
            x = labels_or_x
        labels_use = labels
    else:
        labels_use = labels_or_x
    if loglike is None and likelihood is not None:
        loglike = likelihood
    lab = _coerce_labels(labels_use, x=x).dropna().astype(int)
    n_states = int(lab.nunique()) if not lab.empty else 0
    shares = lab.value_counts(normalize=True)

    silhouette = np.nan
    if x is not None and n_states > 1 and len(lab) > n_states:
        X = x.loc[lab.index] if isinstance(x, pd.DataFrame) else np.asarray(x)
        if len(X) != len(lab):
            X = np.asarray(X)[: len(lab)]
        try:
            silhouette = float(silhouette_score(np.asarray(X, dtype=float), lab.to_numpy(dtype=int)))
        except Exception:
            silhouette = np.nan

    confidence = np.nan
    entropy = np.nan
    if proba is not None:
        p = proba.loc[lab.index] if isinstance(proba, pd.DataFrame) else pd.DataFrame(proba).iloc[: len(lab)]
        if not p.empty:
            confidence = float(posterior_confidence(p).mean())
            entropy = float(posterior_entropy(p).mean())

    durations = duration_table(lab)
    transitions = int((lab != lab.shift()).sum() - 1) if len(lab) else 0
    years = np.nan
    if isinstance(lab.index, pd.DatetimeIndex) and len(lab) > 1:
        years = max((lab.index[-1] - lab.index[0]).days / 365.25, 1.0 / 252.0)
    elif len(lab) > 1:
        years = len(lab) / 252.0

    sep = regime_separation_score(outcomes, lab) if outcomes is not None and n_states > 1 else np.nan
    avg_duration = float(durations["avg_duration"].mean()) if not durations.empty else np.nan

    return {
        "model": str(name),
        "states": n_states,
        "loglike": np.nan if loglike is None else float(loglike),
        "aic": np.nan if aic is None else float(aic),
        "bic": np.nan if bic is None else float(bic),
        "silhouette": silhouette,
        "min_state_share": float(shares.min()) if not shares.empty else np.nan,
        "avg_state_duration": avg_duration,
        "transitions_per_year": float(transitions / years) if years and np.isfinite(years) else np.nan,
        "posterior_confidence": confidence,
        "posterior_entropy": entropy,
        "economic_separation": sep,
    }


__all__ = [
    "duration_table",
    "model_quality_row",
    "posterior_confidence",
    "posterior_entropy",
    "proba_frame",
    "regime_separation_score",
    "remap_labels",
    "sort_states_by_profile",
    "state_profile",
    "transition_table",
]
