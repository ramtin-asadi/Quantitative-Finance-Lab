from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quantfinlab.macro.indicators import BLOCK_COLUMNS, expanding_zscore

ECON_WEIGHTS = {
    "inflation_pressure_block": 0.18,
    "policy_rate_pressure_block": 0.18,
    "growth_recession_block": 0.18,
    "labor_cooling_block": 0.12,
    "housing_domestic_block": 0.10,
    "external_trade_block": 0.10,
    "macro_breadth_conflict_block": 0.14,
}


def _block_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in BLOCK_COLUMNS if c in data.columns]


def _complete_features(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def economic_fci(blocks: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = [c for c in ECON_WEIGHTS if c in blocks.columns]
    if not cols:
        return pd.Series(np.nan, index=blocks.index, name="FCI_ECON")
    weights = pd.Series({c: ECON_WEIGHTS[c] for c in cols}, dtype=float)
    data = blocks[cols].astype(float)
    weighted_sum = data.mul(weights, axis=1).sum(axis=1, skipna=True)
    available_weight = data.notna().mul(weights, axis=1).sum(axis=1).replace(0.0, np.nan)
    raw = weighted_sum / available_weight
    out = expanding_zscore(raw, min_history=max(12, int(min_history // 4)))
    out.name = "FCI_ECON"
    return out


def blended_fci(
    fci_pls: pd.Series,
    fci_pca: pd.Series,
    fci_econ: pd.Series,
    *,
    min_history: int = 12,
) -> pd.Series:
    raw = (
        0.50 * pd.Series(fci_pls, dtype=float)
        + 0.30 * pd.Series(fci_pca, dtype=float)
        + 0.20 * pd.Series(fci_econ, dtype=float)
    )
    out = expanding_zscore(raw, min_history=max(12, int(min_history)))
    out.name = "FCI_BLEND"
    return out


def pca_fci(
    blocks: pd.DataFrame,
    *,
    min_history: int = 60,
    min_blocks: int = 5,
) -> pd.Series:
    cols = _block_columns(blocks)
    x = _complete_features(blocks, cols)
    out = pd.Series(np.nan, index=x.index, name="FCI_PCA")
    average_stress = x.mean(axis=1, skipna=True)

    for date in x.index:
        hist = x.loc[:date]
        good_cols = [c for c in cols if hist[c].notna().sum() >= int(min_history)]
        row = x.loc[date, good_cols]
        good_cols = [c for c in good_cols if np.isfinite(row.get(c, np.nan))]
        if len(good_cols) < int(min_blocks):
            continue
        sample = hist[good_cols].dropna(how="any")
        if len(sample) < int(min_history):
            continue
        scaler = StandardScaler()
        train = scaler.fit_transform(sample)
        pca = PCA(n_components=1).fit(train)
        current = scaler.transform(x.loc[[date], good_cols])
        score = float(pca.transform(current)[0, 0])
        sample_scores = pd.Series(pca.transform(train)[:, 0], index=sample.index)
        aligned = pd.concat([sample_scores, average_stress.reindex(sample.index)], axis=1).dropna()
        if len(aligned) > 12 and aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) < 0:
            score *= -1.0
        out.loc[date] = score

    out = expanding_zscore(out, min_history=max(12, int(min_history // 4)))
    out.name = "FCI_PCA"
    return out


def future_stress_target(
    returns: pd.DataFrame,
    *,
    asset: str = "SPY",
    min_history: int = 60,
) -> pd.DataFrame:
    r = returns[str(asset)].astype(float).replace([np.inf, -np.inf], np.nan)
    next_1m = r.shift(-1)
    next_3m = (1.0 + r.shift(-1)).rolling(3).apply(np.prod, raw=True).shift(-2) - 1.0
    fut_vol = r.shift(-1).rolling(3).std(ddof=1).shift(-2) * np.sqrt(12.0)

    drawdowns = []
    for i in range(len(r)):
        sample = r.iloc[i + 1 : i + 4].dropna()
        if len(sample) < 3:
            drawdowns.append(np.nan)
            continue
        nav = (1.0 + sample).cumprod()
        drawdowns.append(float((nav / nav.cummax() - 1.0).min()))
    fut_dd = pd.Series(drawdowns, index=r.index, name="future_3m_max_drawdown")

    stress = (
        expanding_zscore(-next_3m, min_history=min_history)
        + expanding_zscore(fut_vol, min_history=min_history)
        + expanding_zscore(-fut_dd, min_history=min_history)
    )
    out = pd.DataFrame(
        {
            "future_1m_return": next_1m,
            "future_3m_return": next_3m,
            "future_3m_volatility": fut_vol,
            "future_3m_max_drawdown": fut_dd,
            "future_stress": stress,
        }
    )
    return out


def targeted_pls_fci(
    blocks: pd.DataFrame,
    future_stress: pd.Series,
    *,
    min_history: int = 60,
    n_components: int = 1,
    min_blocks: int = 5,
    embargo_months: int = 3,
) -> pd.Series:
    cols = _block_columns(blocks)
    x = _complete_features(blocks, cols)
    y = pd.Series(future_stress, index=blocks.index, dtype=float)
    out = pd.Series(np.nan, index=x.index, name="FCI_PLS")

    for date in x.index:
        cutoff = pd.Timestamp(date) - pd.offsets.MonthEnd(int(embargo_months))
        hist_x = x.loc[x.index < cutoff]
        hist_y = y.reindex(hist_x.index)
        good_cols = [c for c in cols if hist_x[c].notna().sum() >= int(min_history)]
        row = x.loc[date, good_cols]
        good_cols = [c for c in good_cols if np.isfinite(row.get(c, np.nan))]
        if len(good_cols) < int(min_blocks):
            continue
        train = pd.concat([hist_x[good_cols], hist_y.rename("target")], axis=1).dropna()
        if len(train) < int(min_history):
            continue
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train[good_cols])
        comp = max(1, min(int(n_components), len(good_cols), len(train) - 1))
        model = PLSRegression(n_components=comp)
        model.fit(train_x, train["target"].to_numpy(dtype=float))
        current = scaler.transform(x.loc[[date], good_cols])
        prediction = np.asarray(model.predict(current), dtype=float).reshape(-1)
        out.loc[date] = float(prediction[0])

    out = expanding_zscore(out, min_history=max(12, int(min_history // 2)))
    out.name = "FCI_PLS"
    return out


def stress_probability_fci(
    features: pd.DataFrame,
    future_stress: pd.Series,
    *,
    min_history: int = 60,
    embargo_months: int = 3,
) -> pd.DataFrame:
    x = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = pd.Series(future_stress, index=x.index, dtype=float)
    threshold = y.expanding(min_periods=int(min_history)).quantile(0.75).shift(1)
    label = (y > threshold).where(threshold.notna())
    prob = pd.Series(np.nan, index=x.index, name="FCI_PROB")

    for date in x.index:
        current = x.loc[date].dropna()
        cutoff = pd.Timestamp(date) - pd.offsets.MonthEnd(int(embargo_months))
        history = x.loc[x.index < cutoff]
        usable_cols = [c for c in current.index if c in x.columns and history[c].notna().sum() >= int(min_history)]
        if len(usable_cols) < 2:
            continue
        train = pd.concat(
            [x.loc[x.index < date, usable_cols], label.loc[x.index < date].rename("stress_label")],
            axis=1,
        ).dropna()
        if len(usable_cols) < 2 or len(train) < int(min_history):
            continue
        train = train[[*usable_cols, "stress_label"]].dropna()
        if len(train) < int(min_history) or train["stress_label"].nunique() < 2:
            continue
        scaler = StandardScaler()
        train_x = scaler.fit_transform(train[usable_cols])
        model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        model.fit(train_x, train["stress_label"].astype(int))
        current_x = scaler.transform(x.loc[[date], usable_cols])
        prob.loc[date] = float(model.predict_proba(current_x)[0, 1])

    z = expanding_zscore(prob, min_history=max(12, int(min_history // 2))).rename("FCI_PROB_Z")
    return pd.concat([prob, z], axis=1)


def fci_percentile(series: pd.Series, *, min_history: int = 60) -> pd.Series:
    s = pd.Series(series, dtype=float)

    def rank_last(x: np.ndarray) -> float:
        last = x[-1]
        valid = x[np.isfinite(x)]
        if len(valid) == 0 or not np.isfinite(last):
            return np.nan
        return float((valid <= last).mean())

    out = s.expanding(min_periods=int(min_history)).apply(rank_last, raw=True)
    out.name = f"{s.name or 'fci'}_percentile"
    return out


def fci_change(series: pd.Series, *, periods: int = 3) -> pd.Series:
    out = pd.Series(series, dtype=float).diff(int(periods))
    out.name = f"{series.name or 'fci'}_{periods}m_change"
    return out


def fci_quintile_report(fci: pd.Series, target_table: pd.DataFrame) -> pd.DataFrame:
    data = pd.concat([pd.Series(fci, name="fci"), target_table], axis=1).dropna(subset=["fci"])
    data = data.dropna(subset=["future_stress"])
    if len(data) < 20:
        return pd.DataFrame()
    ranks = data["fci"].rank(method="first")
    data["fci_quintile"] = pd.qcut(ranks, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    report = data.groupby("fci_quintile", observed=False)[
        [
            "future_1m_return",
            "future_3m_return",
            "future_3m_volatility",
            "future_3m_max_drawdown",
            "future_stress",
        ]
    ].mean()
    report["observations"] = data.groupby("fci_quintile", observed=False).size()
    return report


def _monotonic_score(values: pd.Series) -> float:
    y = pd.Series(values, dtype=float).dropna()
    if len(y) < 3:
        return np.nan
    x = np.arange(1, len(y) + 1)
    return float(spearmanr(x, y).correlation)


def _rank01(values: pd.Series) -> pd.Series:
    x = pd.Series(values, dtype=float)
    if x.notna().sum() <= 1:
        return pd.Series(0.5, index=x.index)
    return (x.rank(pct=True) - 1.0 / x.notna().sum()) / (1.0 - 1.0 / x.notna().sum())


def fci_model_scores(fci_models: pd.DataFrame, target_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in fci_models.columns:
        report = fci_quintile_report(fci_models[name], target_table)
        joined = pd.concat(
            [fci_models[name].rename("fci"), target_table["future_stress"]],
            axis=1,
        ).dropna()
        rank_ic = (
            float(spearmanr(joined["fci"], joined["future_stress"]).correlation)
            if len(joined) >= 20
            else np.nan
        )
        if report.empty:
            rows.append(
                {
                    "model": name,
                    "observations": int(fci_models[name].notna().sum()),
                    "first_valid_date": fci_models[name].first_valid_index(),
                    "last_valid_date": fci_models[name].last_valid_index(),
                    "future_stress_rank_ic": rank_ic,
                }
            )
            continue
        drawdown_spread = float(
            -report.loc["Q5", "future_3m_max_drawdown"]
            + report.loc["Q1", "future_3m_max_drawdown"]
        )
        volatility_spread = float(
            report.loc["Q5", "future_3m_volatility"] - report.loc["Q1", "future_3m_volatility"]
        )
        monotonicity = _monotonic_score(report["future_stress"])
        stability = 1.0 / (1.0 + float(fci_models[name].diff().std(skipna=True)))
        rows.append(
            {
                "model": name,
                "observations": int(fci_models[name].notna().sum()),
                "first_valid_date": fci_models[name].first_valid_index(),
                "last_valid_date": fci_models[name].last_valid_index(),
                "future_stress_rank_ic": rank_ic,
                "drawdown_spread": drawdown_spread,
                "monotonicity_score": monotonicity,
                "volatility_spread": volatility_spread,
                "stability_score": stability,
            }
        )
    scores = pd.DataFrame(rows).set_index("model")
    pieces = {
        "future_stress_rank_ic": _rank01(scores.get("future_stress_rank_ic")),
        "drawdown_spread": _rank01(scores.get("drawdown_spread")),
        "monotonicity_score": _rank01(scores.get("monotonicity_score")),
        "volatility_spread": _rank01(scores.get("volatility_spread")),
        "stability_score": _rank01(scores.get("stability_score")),
    }
    scores["final_score"] = (
        0.30 * pieces["future_stress_rank_ic"]
        + 0.25 * pieces["drawdown_spread"]
        + 0.20 * pieces["monotonicity_score"]
        + 0.15 * pieces["volatility_spread"]
        + 0.10 * pieces["stability_score"]
    )
    return scores.sort_values("final_score", ascending=False)


def select_fci_model(
    fci_models: pd.DataFrame,
    scoreboard: pd.DataFrame,
    *,
    min_observations: int = 96,
) -> tuple[str, pd.Series]:
    valid = scoreboard.dropna(subset=["final_score"])
    if "observations" in valid.columns:
        enough_history = valid["observations"] >= int(min_observations)
        if bool(enough_history.any()):
            valid = valid.loc[enough_history]
    if valid.empty:
        name = str(fci_models.notna().sum().sort_values(ascending=False).index[0])
    else:
        name = str(valid.index[0])
    return name, fci_models[name].rename(name)


__all__ = [
    "ECON_WEIGHTS",
    "blended_fci",
    "economic_fci",
    "fci_change",
    "fci_model_scores",
    "fci_percentile",
    "fci_quintile_report",
    "future_stress_target",
    "pca_fci",
    "select_fci_model",
    "stress_probability_fci",
    "targeted_pls_fci",
]
