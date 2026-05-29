from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def classifier_scores(
    name,
    model=None,
    x_test: pd.DataFrame | np.ndarray | None = None,
    y_test: pd.Series | np.ndarray | None = None,
    labels: Sequence[int] | None = None,
) -> dict[str, float | str] | pd.DataFrame:
    if isinstance(name, Mapping):
        models = name
        X = model
        y = x_test
        rows = [
            classifier_scores(str(model_name), fitted, X, y, labels=labels)  # type: ignore[arg-type]
            for model_name, fitted in models.items()
        ]
        return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()
    if model is None or x_test is None or y_test is None:
        raise ValueError("model, x_test, and y_test are required for one classifier.")
    y = pd.Series(y_test).astype(int)
    pred = np.asarray(model.predict(x_test), dtype=int)
    row: dict[str, float | str] = {
        "model": str(name),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": np.nan,
    }
    if hasattr(model, "predict_proba"):
        try:
            p_raw = np.asarray(model.predict_proba(x_test), dtype=float)
            model_classes = list(getattr(model, "classes_", np.arange(p_raw.shape[1])))
            all_labels = list(labels) if labels is not None else sorted(set(y).union(model_classes))
            p = np.full((len(y), len(all_labels)), 1e-12, dtype=float)
            loc = {int(label): i for i, label in enumerate(all_labels)}
            for j, cls in enumerate(model_classes):
                key = int(cls)
                if key in loc:
                    p[:, loc[key]] = p_raw[:, j]
            p = p / p.sum(axis=1, keepdims=True)
            row["log_loss"] = float(log_loss(y, p, labels=all_labels))
        except Exception:
            row["log_loss"] = np.nan
    return row


def rf_importance(
    model,
    x: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise ValueError("model must expose feature_importances_.")
    out = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
    out["permutation_importance"] = np.nan
    out["permutation_std"] = np.nan
    if y is not None:
        try:
            perm = permutation_importance(
                model,
                x,
                y,
                n_repeats=int(n_repeats),
                random_state=int(random_state),
                scoring="balanced_accuracy",
            )
            out["permutation_importance"] = perm.importances_mean
            out["permutation_std"] = perm.importances_std
        except Exception:
            pass
    score_col = "permutation_importance" if out["permutation_importance"].notna().any() else "importance"
    out = out.sort_values(score_col, ascending=False)
    return out.set_index("feature", drop=False)


__all__ = ["classifier_scores", "rf_importance"]
