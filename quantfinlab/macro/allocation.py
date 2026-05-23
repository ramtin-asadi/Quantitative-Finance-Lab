from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def _cap_weights(weights: pd.Series, *, cap: float = 0.40) -> pd.Series:
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    if float(w.sum()) <= 1e-12:
        return w
    w = w / float(w.sum())
    cap = float(cap)
    for _ in range(20):
        over = w > cap
        if not bool(over.any()):
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        room = (cap - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += excess * room / float(room.sum())
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else w


def _groups(assets: list[str]) -> dict[str, set[str]]:
    asset_set = set(assets)
    return {
        "defensive": asset_set
        & {"XLP", "XLU", "XLV", "XST.TO", "XUT.TO", "XRE.TO"},
        "cyclical": asset_set
        & {"XLF", "XLI", "XLY", "XLB", "XFN.TO", "XEG.TO", "XMA.TO", "XRE.TO", "XIT.TO"},
        "inflation": asset_set
        & {"XLE", "XLB", "DBC", "GLD", "XEG.TO", "XMA.TO", "XGD.TO"},
        "rate": asset_set
        & {"XLK", "XLY", "VNQ", "IYR", "TLT", "XRE.TO", "XUT.TO", "XIT.TO"},
    }


def _feature(data: pd.DataFrame | pd.Series, name: str, default: float = 0.0) -> pd.Series:
    if isinstance(data, pd.Series):
        return pd.Series(default, index=[data.name or pd.Timestamp.today()])
    return data[name].astype(float) if name in data.columns else pd.Series(default, index=data.index)


def etf_momentum_score(
    returns: pd.DataFrame,
    assets: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    assets_use = list(assets or returns.columns)
    r = returns.reindex(columns=assets_use).astype(float)
    mom_12_1 = (1.0 + r.shift(1)).rolling(11).apply(np.prod, raw=True) - 1.0
    ret_6 = (1.0 + r.shift(1)).rolling(6).apply(np.prod, raw=True) - 1.0
    vol_6 = r.shift(1).rolling(6).std(ddof=1) * np.sqrt(12.0)
    risk_adj_6 = ret_6 / vol_6.replace(0.0, np.nan)
    score = 0.60 * _cross_z(mom_12_1) + 0.40 * _cross_z(risk_adj_6)
    return score.reindex(columns=assets_use)


def fci_risky_weight(
    fci_percentile: pd.Series,
    fci_3m_change: pd.Series,
    *,
    lower: float = 0.50,
    upper: float = 1.00,
) -> pd.Series:
    stress = pd.Series(fci_percentile, dtype=float).fillna(0.0)
    slope = pd.Series(fci_3m_change, dtype=float).reindex(stress.index).fillna(0.0).clip(lower=0.0)
    risky = pd.Series(1.0, index=stress.index, dtype=float)
    risky -= ((stress - 0.60).clip(0.0, 0.25) / 0.25) * 0.35
    risky -= (slope.clip(0.0, 0.75) / 0.75) * 0.10
    return risky.clip(float(lower), float(upper)).rename("risky_weight")


def defensive_weights(
    features: pd.Series,
    defensive_assets: list[str] | tuple[str, ...],
) -> pd.Series:
    assets = list(defensive_assets)
    inflation_policy = float(features.get("inflation_pressure_block", 0.0)) + float(
        features.get("policy_rate_pressure_block", 0.0)
    )
    growth = float(features.get("growth_recession_block", 0.0))
    inflation = float(features.get("inflation_pressure_block", 0.0))
    if inflation_policy > 1.0:
        preferred = ["XSB.TO", "CGL-C.TO", "SHY", "GLD", "DBC"]
    elif growth > 0.75 and inflation <= 0.75:
        preferred = ["XBB.TO", "XLB.TO", "XSB.TO", "CGL-C.TO", "IEF", "SHY", "GLD"]
    else:
        preferred = ["XSB.TO", "XBB.TO", "SHY", "IEF"]
    keep = [a for a in preferred if a in assets]
    if not keep:
        keep = assets[:]
    out = pd.Series(0.0, index=assets, dtype=float)
    if keep:
        out.loc[keep] = 1.0 / len(keep)
    return out


def equal_sector_weights(
    dates: pd.Index,
    sectors: list[str] | tuple[str, ...],
    all_assets: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    cols = list(all_assets or sectors)
    rows = []
    for date in pd.DatetimeIndex(dates):
        row = pd.Series(0.0, index=cols, dtype=float)
        present = [s for s in sectors if s in row.index]
        if present:
            row.loc[present] = 1.0 / len(present)
        rows.append(row.rename(date))
    return pd.DataFrame(rows)


def momentum_sector_weights(
    returns: pd.DataFrame,
    sectors: list[str] | tuple[str, ...],
    *,
    top_n: int = 3,
    cap: float = 0.40,
) -> pd.DataFrame:
    sectors_use = [s for s in sectors if s in returns.columns]
    score = etf_momentum_score(returns, sectors_use)
    vol = returns[sectors_use].rolling(6).std(ddof=1) * np.sqrt(12.0)
    rows = []
    for date, row in score.iterrows():
        picked = row.dropna().sort_values(ascending=False).head(int(top_n)).index.tolist()
        w = pd.Series(0.0, index=sectors_use, dtype=float)
        if picked:
            inv = 1.0 / vol.loc[date, picked].replace(0.0, np.nan)
            inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
            if inv.empty:
                inv = pd.Series(1.0, index=picked)
            w.loc[inv.index] = _cap_weights(inv, cap=cap)
        rows.append(w.rename(date))
    return pd.DataFrame(rows)


def sector_macro_fit(
    features: pd.DataFrame,
    sectors: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    sectors_use = list(sectors)
    groups = _groups(sectors_use)
    out = pd.DataFrame(0.0, index=features.index, columns=sectors_use)
    fci_pct = _feature(features, "best_fci_percentile")
    fci_change = _feature(features, "best_fci_3m_change")
    inflation = _feature(features, "inflation_pressure_block")
    policy = _feature(features, "policy_rate_pressure_block")
    growth = _feature(features, "growth_recession_block")
    breadth = _feature(features, "macro_breadth_conflict_block")

    for asset in sectors_use:
        score = pd.Series(0.0, index=features.index)
        if asset in groups["defensive"]:
            score += 0.70 * fci_pct + 0.45 * growth + 0.20 * fci_change.clip(lower=0.0)
        if asset in groups["cyclical"]:
            score += 0.80 * (1.0 - fci_pct) - 0.40 * growth - 0.30 * breadth
        if asset in groups["inflation"]:
            score += 0.55 * inflation - 0.25 * growth.clip(lower=0.0)
        if asset in groups["rate"]:
            score -= 0.45 * policy
        if asset in {"XLF", "XFN.TO", "ZEB.TO"}:
            score -= 0.35 * (policy.clip(lower=0.0) * growth.clip(lower=0.0))
        out[asset] = score
    return _cross_z(out)


def _macro_support_score(features: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    groups = _groups(sectors)
    out = pd.DataFrame(0.0, index=features.index, columns=sectors)
    goldilocks = _feature(features, "goldilocks_support")
    stress = _feature(features, "macro_breadth_conflict_block")
    inflation = _feature(features, "inflation_pressure_block")
    for asset in sectors:
        score = 0.20 * goldilocks - 0.20 * stress
        if asset in groups["inflation"]:
            score = score + 0.35 * inflation
        if asset in groups["defensive"]:
            score = score + 0.25 * stress.clip(lower=0.0)
        out[asset] = score
    return _cross_z(out)


def _recent_risk_penalty(returns: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    r = returns.reindex(columns=sectors).astype(float)
    vol = r.rolling(3).std(ddof=1) * np.sqrt(12.0)
    dd = (1.0 + r).rolling(3).apply(np.prod, raw=True) - 1.0
    penalty = _cross_z(vol) + _cross_z((-dd).clip(lower=0.0))
    return penalty.fillna(0.0)


def fci_gated_weights(
    features: pd.DataFrame,
    sectors: list[str] | tuple[str, ...],
    defensive_assets: list[str] | tuple[str, ...],
    *,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series]]:
    sectors_use = list(sectors)
    defensive_use = list(defensive_assets)
    cols = list(dict.fromkeys(sectors_use + defensive_use))
    risky = fci_risky_weight(features["best_fci_percentile"], features["best_fci_3m_change"])
    rows = []
    for date, feature_row in features.iterrows():
        row = pd.Series(0.0, index=cols, dtype=float)
        sector_present = [s for s in sectors_use if s in row.index]
        if sector_present:
            row.loc[sector_present] = float(risky.loc[date]) / len(sector_present)
        d = defensive_weights(feature_row, defensive_use)
        row.loc[d.index] += (1.0 - float(risky.loc[date])) * d
        rows.append(row.rename(date))
    weights = pd.DataFrame(rows).fillna(0.0)
    if return_details:
        return weights, {"risky_weight": risky, "defensive_weight": 1.0 - risky}
    return weights


def fci_momentum_weights(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    sectors: list[str] | tuple[str, ...],
    defensive_assets: list[str] | tuple[str, ...],
    *,
    top_n: int = 3,
    cap: float = 0.40,
    return_details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, pd.DataFrame | pd.Series]]:
    sectors_use = [s for s in sectors if s in returns.columns]
    defensive_use = [a for a in defensive_assets if a in returns.columns]
    cols = list(dict.fromkeys(sectors_use + defensive_use))
    momentum = etf_momentum_score(returns, sectors_use).reindex(features.index)
    macro_fit = sector_macro_fit(features, sectors_use)
    macro_support = _macro_support_score(features, sectors_use)
    risk_penalty = _recent_risk_penalty(returns, sectors_use).reindex(features.index).fillna(0.0)
    final_score = 0.65 * momentum + 0.20 * macro_fit + 0.10 * macro_support - 0.05 * risk_penalty
    vol = returns[sectors_use].rolling(6).std(ddof=1).reindex(features.index) * np.sqrt(12.0)
    risky = fci_risky_weight(features["best_fci_percentile"], features["best_fci_3m_change"])

    rows = []
    for date, row_score in final_score.iterrows():
        picked = row_score.dropna().sort_values(ascending=False).head(int(top_n)).index.tolist()
        row = pd.Series(0.0, index=cols, dtype=float)
        if picked:
            inv = 1.0 / vol.loc[date, picked].replace(0.0, np.nan)
            inv = inv.replace([np.inf, -np.inf], np.nan).dropna()
            if inv.empty:
                inv = pd.Series(1.0, index=picked)
            sector_w = _cap_weights(inv, cap=cap) * float(risky.loc[date])
            row.loc[sector_w.index] = sector_w
        d = defensive_weights(features.loc[date], defensive_use)
        row.loc[d.index] += (1.0 - float(risky.loc[date])) * d
        rows.append(row.rename(date))

    weights = pd.DataFrame(rows).fillna(0.0)
    if return_details:
        return weights, {
            "momentum_score": momentum,
            "macro_fit_score": macro_fit,
            "macro_support_score": macro_support,
            "recent_risk_penalty": risk_penalty,
            "final_score": final_score,
            "risky_weight": risky,
            "defensive_weight": 1.0 - risky,
        }
    return weights


def latest_decision_table(
    features: pd.DataFrame,
    weights: pd.DataFrame,
    details: dict[str, pd.DataFrame | pd.Series],
    *,
    selected_fci_model: str,
) -> pd.DataFrame:
    date = pd.Timestamp(weights.index.max())
    assets = list(weights.columns)
    groups = _groups(assets)
    rows = []
    for asset in assets:
        momentum = float(details["momentum_score"].reindex(columns=assets).loc[date, asset]) if asset in details["momentum_score"].columns else 0.0
        fit = float(details["macro_fit_score"].reindex(columns=assets).loc[date, asset]) if asset in details["macro_fit_score"].columns else 0.0
        support = float(details["macro_support_score"].reindex(columns=assets).loc[date, asset]) if asset in details["macro_support_score"].columns else 0.0
        penalty = float(details["recent_risk_penalty"].reindex(columns=assets).loc[date, asset]) if asset in details["recent_risk_penalty"].columns else 0.0
        final = float(details["final_score"].reindex(columns=assets).loc[date, asset]) if asset in details["final_score"].columns else 0.0
        weight = float(weights.loc[date, asset])
        f = features.loc[date]
        if asset in groups["inflation"] and f.get("inflation_pressure_block", 0.0) > 0.75 and weight > 0:
            reason = "inflation hedge"
        elif asset in groups["defensive"] and f.get("best_fci_percentile", 0.0) > 0.65 and weight > 0:
            reason = "defensive stress protection"
        elif asset in groups["rate"] and f.get("policy_rate_pressure_block", 0.0) > 0.75 and fit < 0:
            reason = "rate-pressure penalty"
        elif asset in groups["cyclical"] and f.get("best_fci_percentile", 1.0) < 0.45 and weight > 0:
            reason = "risk-on cyclical"
        elif momentum < 0 and weight <= 1e-8:
            reason = "weak momentum"
        else:
            reason = "macro stress exclusion" if weight <= 1e-8 else "risk-on cyclical"
        rows.append(
            {
                "date": date,
                "selected fci model": selected_fci_model,
                "selected fci value": f.get("best_fci_value", np.nan),
                "selected fci percentile": f.get("best_fci_percentile", np.nan),
                "selected fci 3m change": f.get("best_fci_3m_change", np.nan),
                "dominant macro block": f.get("dominant_macro_block", ""),
                "sector": asset,
                "momentum_score": momentum,
                "macro_fit_score": fit,
                "macro_support_score": support,
                "recent_risk_penalty": penalty,
                "final_score": final,
                "portfolio_weight": weight,
                "short reason category": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("portfolio_weight", ascending=False)


__all__ = [
    "defensive_weights",
    "equal_sector_weights",
    "etf_momentum_score",
    "fci_gated_weights",
    "fci_momentum_weights",
    "fci_risky_weight",
    "latest_decision_table",
    "momentum_sector_weights",
    "sector_macro_fit",
]
