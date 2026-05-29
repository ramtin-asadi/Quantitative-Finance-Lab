from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .regimes import model_quality_row, proba_frame


def hmm_proba_frame(model, x, index: Sequence[pd.Timestamp] | pd.Index | None = None, prefix: str = "state") -> pd.DataFrame:
    p = model.predict_proba(x)
    if index is None:
        index = getattr(x, "index", pd.RangeIndex(len(p)))
    return proba_frame(p, index=index, prefix=prefix)


def _gaussian_hmm_n_params(model, n_features: int) -> int:
    n_states = int(getattr(model, "n_components", 1))
    cov_type = str(getattr(model, "covariance_type", "diag"))
    start_params = n_states - 1
    transition_params = n_states * (n_states - 1)
    mean_params = n_states * n_features
    if cov_type == "full":
        cov_params = int(n_states * n_features * (n_features + 1) / 2)
    elif cov_type == "tied":
        cov_params = int(n_features * (n_features + 1) / 2)
    elif cov_type == "spherical":
        cov_params = n_states
    else:
        cov_params = n_states * n_features
    return start_params + transition_params + mean_params + cov_params


def hmm_quality_row(
    name: str,
    model,
    x,
    *,
    outcomes: pd.DataFrame | pd.Series | None = None,
    labels: pd.Series | Sequence[int] | None = None,
    proba: pd.DataFrame | np.ndarray | None = None,
    n_params: int | None = None,
) -> dict[str, float | str]:
    X = np.asarray(x, dtype=float)
    n = len(X)
    if labels is None:
        labels = model.predict(X)
    if proba is None:
        proba = model.predict_proba(X)
    loglike = float(model.score(X))
    if n_params is None:
        n_params = _gaussian_hmm_n_params(model, X.shape[1] if X.ndim == 2 else 1)
    aic = 2.0 * n_params - 2.0 * loglike
    bic = np.log(max(n, 1)) * n_params - 2.0 * loglike
    return model_quality_row(
        name,
        x,
        labels,
        proba=proba,
        outcomes=outcomes,
        loglike=loglike,
        aic=aic,
        bic=bic,
    )


def pca_hmm_inputs(
    x: pd.DataFrame | np.ndarray,
    n_components: int = 5,
    *,
    random_state: int = 42,
    scaler: StandardScaler | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame | np.ndarray, StandardScaler, PCA]:
    X = x.replace([np.inf, -np.inf], np.nan).dropna() if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=float)
    n_comp = int(max(1, min(int(n_components), min(np.asarray(X).shape))))
    scaler = StandardScaler() if scaler is None else scaler
    pca = PCA(n_components=n_comp, random_state=int(random_state)) if pca is None else pca
    if not hasattr(scaler, "mean_"):
        z = scaler.fit_transform(X)
    else:
        z = scaler.transform(X)
    if not hasattr(pca, "components_"):
        arr = pca.fit_transform(z)
    else:
        arr = pca.transform(z)
    if isinstance(X, pd.DataFrame):
        cols = [f"PC{i + 1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=X.index, columns=cols), scaler, pca
    return arr, scaler, pca


def align_state_probabilities(
    proba: pd.DataFrame,
    order: Sequence[int],
    prefix: str = "state",
) -> pd.DataFrame:
    cols = list(proba.columns)
    ordered_cols = [cols[int(i)] for i in order if int(i) < len(cols)]
    out = proba.reindex(columns=ordered_cols).copy()
    out.columns = [f"{prefix}_{i}" for i in range(out.shape[1])]
    row_sum = out.sum(axis=1).replace(0.0, np.nan)
    return out.div(row_sum, axis=0).fillna(0.0)


def walkforward_hmm_probabilities(
    model_factory,
    x: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp],
    *,
    train_days: int = 1260,
    pca_components: int | None = None,
) -> pd.DataFrame:
    rows = {}
    for dt in pd.to_datetime(rebalance_dates):
        hist = x.loc[:dt].tail(int(train_days))
        if len(hist) < max(250, int(train_days) // 2):
            continue
        if pca_components is None:
            scaler = StandardScaler().fit(hist)
            z_train = scaler.transform(hist)
            z_last = scaler.transform(hist.iloc[[-1]])
        else:
            z_train, scaler, pca = pca_hmm_inputs(hist, n_components=int(pca_components))
            z_last, _, _ = pca_hmm_inputs(hist.iloc[[-1]], n_components=int(pca_components), scaler=scaler, pca=pca)
        model = model_factory()
        model.fit(z_train)
        rows[pd.Timestamp(dt)] = model.predict_proba(z_last)[0]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index().rename(columns=lambda i: f"state_{i}")


__all__ = [
    "align_state_probabilities",
    "hmm_proba_frame",
    "hmm_quality_row",
    "pca_hmm_inputs",
    "walkforward_hmm_probabilities",
]
