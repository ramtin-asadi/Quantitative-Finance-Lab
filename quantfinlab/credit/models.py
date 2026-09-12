"""Interpretable and boosted default models with chronological horizon validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import SplineTransformer, StandardScaler

from quantfinlab.ml.validation import available_labels

spline_features = ["log_assets", "liabilities_assets", "debt_assets", "debt_ebitda",
                   "interest_coverage", "cfo_debt", "cash_assets", "current_ratio",
                   "wc_assets", "fcf_assets", "roa", "operating_margin", "accruals"]
boosting_specs = [
    {"num_leaves": 7, "max_depth": 3, "min_child_samples": 150,
     "reg_alpha": 2.0, "reg_lambda": 12.0, "weight": "none"},
    {"num_leaves": 15, "max_depth": 4, "min_child_samples": 200,
     "reg_alpha": 3.0, "reg_lambda": 20.0, "weight": "none"},
    {"num_leaves": 7, "max_depth": 3, "min_child_samples": 200,
     "reg_alpha": 3.0, "reg_lambda": 20.0, "weight": "sqrt"}]


@dataclass
class DefaultFit:
    """Estimator and training-only transformations needed to reproduce predictions."""
    model: object
    columns: list
    kind: str
    lo: pd.Series
    hi: pd.Series
    median: pd.Series | None = None
    mean: pd.Series | None = None
    scale: object = None
    spline: object = None
    continuous: list | None = None
    calibration: object = None


def matured(data: pd.DataFrame, cutoff, horizon: int, target: str) -> pd.DataFrame:
    """Keep fully matured horizon labels before the refit cutoff."""
    end = data["decision_date"] + pd.offsets.MonthEnd(horizon)
    mask = available_labels(data["decision_date"], end, cutoff,
                            observed=data[f"observed_{target}"])
    return data.loc[mask]


def fit_logit(X: pd.DataFrame, y, *, C=0.25, max_iter=500) -> DefaultFit:
    """L2 logit with 1/99% winsorization, median imputation and standardization."""
    X = X.astype(float)
    q = X.quantile([0.01, 0.99])
    lo, hi = q.loc[0.01], q.loc[0.99]
    X = X.clip(lo, hi, axis=1)
    median = X.median().fillna(0)
    X = X.fillna(median)
    mu, sigma = X.mean(), X.std().replace(0, 1).fillna(1)
    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit((X - mu) / sigma, np.asarray(y, dtype=int))
    return DefaultFit(model, list(X), "logit", lo, hi, median, mu, sigma)


def fit_spline(X: pd.DataFrame, y, *, continuous=None, C=0.10,
                knots=4, degree=2, max_iter=500) -> DefaultFit:
    """Additive spline logit with linear controls and the notebook's preprocessing."""
    continuous = list(continuous if continuous is not None else [n for n in spline_features if n in X])
    if not continuous:
        raise ValueError("At least one continuous spline feature is required.")
    linear = [name for name in X if name not in continuous]
    X = X.astype(float)
    q = X.quantile([0.01, 0.99])
    lo, hi = q.loc[0.01], q.loc[0.99]
    X = X.clip(lo, hi, axis=1)
    median = X.median().fillna(0)
    X = X.fillna(median)
    spline = SplineTransformer(n_knots=knots, degree=degree, include_bias=False, extrapolation="linear")
    curved = spline.fit_transform(X[continuous]).astype("float32")
    Z = X[linear].to_numpy(dtype="float32")
    scale = StandardScaler()
    design = scale.fit_transform(np.column_stack([curved, Z])).astype("float32")
    model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
    model.fit(design, np.asarray(y, dtype=int))
    return DefaultFit(model, list(X), "spline", lo, hi, median, scale=scale,
                      spline=spline, continuous=continuous)


def _lightgbm(spec, positives, rows, trees, seed, n_jobs):
    import lightgbm as lgb

    params = dict(spec)
    weight = params.pop("weight", "none")
    weight = np.sqrt((rows - positives) / max(positives, 1)) if weight == "sqrt" else 1
    return lgb.LGBMClassifier(objective="binary", learning_rate=0.035, n_estimators=trees,
                              subsample=0.85, colsample_bytree=0.80, scale_pos_weight=weight,
                              random_state=seed, n_jobs=n_jobs, verbosity=-1, **params)


def fit_lightgbm(X: pd.DataFrame, y, *, spec=None, trees=600, seed=22,
                  n_jobs=-1, bounds=None, validation=None, early_stopping=40) -> DefaultFit:
    """Shallow regularized boosting with fitted 0.5/99.5% clipping bounds."""
    import lightgbm as lgb

    X = X.astype(float)
    q = X.quantile([0.005, 0.995]) if bounds is None else None
    lo, hi = (q.loc[0.005], q.loc[0.995]) if bounds is None else bounds
    model = _lightgbm(boosting_specs[0] if spec is None else spec, np.sum(y), len(y), trees, seed, n_jobs)
    kwargs = {"callbacks": [lgb.log_evaluation(0)]}
    if validation is not None:
        xv, yv = validation
        kwargs["eval_set"] = [(xv.astype(float).clip(lo, hi, axis=1), np.asarray(yv, dtype=int))]
        kwargs["callbacks"].append(lgb.early_stopping(early_stopping, verbose=False))
    model.fit(X.clip(lo, hi, axis=1), np.asarray(y, dtype=int), **kwargs)
    return DefaultFit(model, list(X), "lightgbm", lo, hi)


def default_design(fit: DefaultFit, X: pd.DataFrame):
    """Apply stored training transformations without estimating anything on X."""
    X = X[fit.columns].astype(float).clip(fit.lo, fit.hi, axis=1)
    if fit.kind == "lightgbm":
        return X
    X = X.fillna(fit.median)
    if fit.kind == "logit":
        return (X - fit.mean) / fit.scale
    linear = [name for name in fit.columns if name not in fit.continuous]
    curved = fit.spline.transform(X[fit.continuous]).astype("float32")
    return fit.scale.transform(np.column_stack([curved, X[linear].to_numpy(dtype="float32")]))


def predict_default(fit: DefaultFit, X: pd.DataFrame, *, calibrated=True) -> np.ndarray:
    """Return default probabilities using the stored transform and optional calibration."""
    p = fit.model.predict_proba(default_design(fit, X))[:, 1]
    if calibrated and fit.calibration is not None:
        return fit.calibration.predict_proba(logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1))[:, 1]
    return p


def default_contributions(fit: DefaultFit, X: pd.DataFrame) -> pd.DataFrame:
    """Native tree SHAP values in raw log-odds, including the base value.

    Contributions explain the fitted tree before probability calibration; they
    are not an additive explanation of a multi-model probability ensemble.
    """
    if fit.kind != "lightgbm":
        raise ValueError("Native tree contributions require a LightGBM fit.")
    values = fit.model.booster_.predict(default_design(fit, X), pred_contrib=True)
    return pd.DataFrame(values, index=X.index, columns=[*fit.columns, "base_value"])


def probability_calibration(p, y, *, C=10.0):
    """Platt calibration on held-out predictions; preserve the fitted intercept."""
    model = LogisticRegression(C=C, max_iter=200)
    return model.fit(logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1), np.asarray(y, dtype=int))


def select_default_model(train, target, features, cutoff, horizon, *, specs=None,
                          fixed=None, seed=22, n_jobs=-1) -> tuple[DefaultFit, dict]:
    """Nested chronological boosting selection, held-out calibration, and final refit.

    The historical notebook's small-event fallback is an origin-date 80/20 split;
    ``validation_purged`` reports whether the horizon embargo survived that fallback.
    ``fixed`` reuses a chosen specification/tree count while refitting calibration.
    """
    specs = boosting_specs if specs is None else specs
    last = (pd.Timestamp(cutoff) - pd.DateOffset(months=horizon)).to_period("M").to_timestamp("M")
    first = last - pd.DateOffset(months=12) + pd.offsets.MonthEnd(0)
    inner = train[(train["decision_date"] + pd.offsets.MonthEnd(horizon)).lt(first)]
    valid = train[train["decision_date"].between(first, last)]
    purged = True
    if inner[target].sum() < 10 or valid[target].sum() < 5:
        split = train["decision_date"].quantile(0.8)
        inner, valid = train[train["decision_date"].lt(split)], train[train["decision_date"].ge(split)]
        purged = False
    if inner[target].nunique() < 2 or valid.empty:
        raise ValueError("Insufficient chronological events for boosting validation.")
    candidates = range(len(specs)) if fixed is None else [int(fixed["spec"])]
    trials = []
    for number in candidates:
        trees = 600 if fixed is None else int(fixed["trees"])
        fit = fit_lightgbm(inner[features], inner[target], spec=specs[number], trees=trees,
                           seed=seed, n_jobs=n_jobs,
                           validation=(valid[features], valid[target]) if fixed is None else None)
        p = predict_default(fit, valid)
        count = fit.model.best_iteration_ if fixed is None else trees
        trials.append((average_precision_score(valid[target], p),
                       -log_loss(valid[target], p, labels=[0, 1]), number, count, p, fit))
    ap, nll, number, trees, p, chosen = max(trials, key=lambda row: (row[0], row[1]))
    calibration = probability_calibration(p, valid[target]) if valid[target].nunique() == 2 else None
    fit = fit_lightgbm(train[features], train[target], spec=specs[number], trees=trees,
                       bounds=(chosen.lo, chosen.hi), seed=seed, n_jobs=n_jobs)
    fit.calibration = calibration
    return fit, {"spec": number, "trees": trees, "validation_ap": ap,
                 "validation_log_loss": -nll, "validation_purged": purged}


def default_forecasts(data, target, features, *, horizon=1, spacing=1, method="logit",
                       refit_dates, choices=None, specs=None, seed=22, n_jobs=-1) -> tuple:
    """Expanding forecasts for one specified architecture and horizon.

    Returns predictions and fit diagnostics. This function never chooses targets,
    combines models, constructs accounting features, or creates figures.
    """
    parts, decisions = [], []
    cuts = pd.DatetimeIndex(refit_dates)
    for j, cutoff in enumerate(cuts):
        train = matured(data, cutoff, horizon, target)
        train = train[train["origin"].mod(spacing).eq(0)]
        stop = cuts[j + 1] if j + 1 < len(cuts) else pd.Timestamp.max
        test = data[data["decision_date"].ge(cutoff) & data["decision_date"].lt(stop)]
        if test.empty or train[target].sum() < 10:
            continue
        info = {}
        if method == "logit":
            fit = fit_logit(train[features], train[target])
        elif method == "spline":
            fit = fit_spline(train[features], train[target])
        elif method == "lightgbm":
            fixed = None if choices is None else choices.loc[choices["cutoff"].eq(cutoff)].iloc[0]
            fit, info = select_default_model(train, target, features, cutoff, horizon,
                                             fixed=fixed, specs=specs, seed=seed, n_jobs=n_jobs)
        else:
            raise ValueError("method must be logit, spline or lightgbm.")
        part = test[["decision_date", "cik", target, f"observed_{target}"]].copy()
        part["p"] = predict_default(fit, test)
        parts.append(part)
        decisions.append({"cutoff": cutoff, "training_rows": len(train),
                          "positives": int(train[target].sum()), **info})
    if not parts:
        raise ValueError("No refit interval has enough events for the requested model.")
    return pd.concat(parts, ignore_index=True), pd.DataFrame(decisions)


def fit_cox(X, duration, status, *, entry=None):
    """Landmark Cox PH with Efron ties; caller supplies the landmark sampling design."""
    from statsmodels.duration.hazard_regression import PHReg

    q = X.quantile([0.01, 0.99])
    Z = X.clip(q.loc[0.01], q.loc[0.99], axis=1)
    median = Z.median()
    Z = Z.fillna(median)
    mean, scale = Z.mean(), Z.std().replace(0, 1)
    fit = PHReg(duration, (Z - mean) / scale, status=status, entry=entry, ties="efron").fit(disp=0)
    return {"model": fit, "lo": q.loc[0.01], "hi": q.loc[0.99], "median": median,
            "mean": mean, "scale": scale, "risk": ((Z - mean) / scale).to_numpy() @ fit.params}


def harrell_c(duration, status, score) -> float:
    """Concordance over comparable right-censored pairs, with half credit for ties."""
    duration, status, score = np.asarray(duration), np.asarray(status, dtype=bool), np.asarray(score)
    concordant, comparable = 0.0, 0
    for i in np.flatnonzero(status):
        at_risk = duration > duration[i]
        comparable += at_risk.sum()
        concordant += np.sum(score[i] > score[at_risk]) + 0.5 * np.sum(score[i] == score[at_risk])
    return concordant / comparable if comparable else np.nan
