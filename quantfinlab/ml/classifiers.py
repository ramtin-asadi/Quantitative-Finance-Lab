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
    """Evaluate one classifier or a collection of classifiers on a test set.

    The function supports two calling patterns. When ``name`` is a mapping, the
    mapping is interpreted as ``{model_name: fitted_model}`` and a comparison table
    is returned. Otherwise, the function evaluates one fitted classifier and returns
    a dictionary of scalar metrics.

    Parameters
    ----------
    name : str or mapping
        Model name for a single classifier, or a mapping from model names to fitted
        classifier objects.
    model : object, optional
        Fitted classifier for the single-model call. When ``name`` is a mapping,
        this argument is interpreted as the test feature matrix.
    x_test : pandas.DataFrame or numpy.ndarray, optional
        Test feature matrix for the single-model call. When ``name`` is a mapping,
        this argument is interpreted as the test target vector.
    y_test : pandas.Series or numpy.ndarray, optional
        Test target vector for the single-model call.
    labels : sequence of int, optional
        Complete class label set used when computing multiclass log loss. Supplying
        this is useful when a fitted model did not observe every class during
        training but the comparison table should use a consistent label universe.

    Returns
    -------
    dict or pandas.DataFrame
        For a single classifier, returns a dictionary with ``model``, ``accuracy``,
        ``balanced_accuracy``, ``macro_f1``, and ``log_loss``. For a mapping of
        classifiers, returns a DataFrame indexed by model name.

    Raises
    ------
    ValueError
        If the single-model call is missing ``model``, ``x_test``, or ``y_test``.

    Notes
    -----
    ``log_loss`` is reported only when the model exposes ``predict_proba`` and the
    probability matrix can be aligned to the requested labels. If probability
    scoring fails, the log-loss value is set to ``NaN`` while classification metrics
    based on hard labels are still returned.
    """
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
    """Build a feature-importance table for a fitted tree-based model.

    The function reports the model's built-in impurity-based feature importance and,
    when a target vector is provided, also attempts permutation importance using
    balanced accuracy as the scoring metric.

    Parameters
    ----------
    model : object
        Fitted estimator exposing ``feature_importances_``.
    x : pandas.DataFrame
        Feature matrix used to name the features and optionally compute permutation
        importance.
    y : pandas.Series or numpy.ndarray, optional
        Target vector for permutation importance. If omitted, only built-in
        importance is reported.
    n_repeats : int, default=10
        Number of permutation repeats.
    random_state : int, default=42
        Random seed for permutation importance.

    Returns
    -------
    pandas.DataFrame
        Feature-importance table indexed by feature name. Columns include
        ``feature``, ``importance``, ``permutation_importance``, and
        ``permutation_std``. Rows are sorted by permutation importance when
        available, otherwise by built-in importance.

    Raises
    ------
    ValueError
        If the model does not expose ``feature_importances_``.

    Notes
    -----
    Permutation importance is attempted in a best-effort manner. If it fails, the
    function still returns built-in feature importances.
    """
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
