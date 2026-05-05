from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.portfolio import views as view_lib


def p_row_from_view(row: Mapping[str, Any], assets: Sequence[str]) -> pd.Series | None:
    if isinstance(row.get("p_vector"), Mapping):
        return pd.Series(row["p_vector"], dtype=float).reindex([str(x) for x in assets]).fillna(0.0)
    return view_lib.p_series_from_assets(row.get("long_assets", []), row.get("short_assets", []), [str(x) for x in assets])


def stress_state_from_values(values: Mapping[str, Any]) -> bool:
    spy_dd = view_lib.state_value(values, "spy_drawdown_252", 0.0)
    spy_vol_z = view_lib.state_value(values, "spy_vol_z", 0.0)
    breadth = view_lib.state_value(values, "risky_trend_breadth")
    credit = view_lib.state_value(values, "hyg_lqd_63", 0.0)
    return bool(
        (np.isfinite(spy_dd) and spy_dd < -0.08)
        or spy_vol_z > 0.80
        or (np.isfinite(breadth) and breadth < 0.45)
        or credit < -0.020
    )


def payoff_history(
    view_log: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    horizon: int = 21,
    state_log: pd.DataFrame | None = None,
    annualization: float = 252.0,
) -> pd.DataFrame:
    if view_log is None or view_log.empty:
        return pd.DataFrame()
    R = returns.copy()
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()
    state_by_date: dict[pd.Timestamp, Mapping[str, Any]] = {}
    if state_log is not None and not state_log.empty:
        for _, state_row in state_log.iterrows():
            state_by_date[pd.Timestamp(state_row["date"])] = dict(state_row)
    rows: list[dict[str, Any]] = []
    for _, row in view_log.iterrows():
        dt = pd.Timestamp(row["date"])
        pos = R.index.searchsorted(dt, side="right") - 1
        if pos < 0 or pos + int(horizon) >= len(R.index):
            continue
        forward_asset_return = (1.0 + R.iloc[pos + 1 : pos + int(horizon) + 1]).prod() - 1.0
        p = p_row_from_view(row, R.columns)
        if p is None:
            continue
        payoff_horizon = float(p @ forward_asset_return.reindex(p.index).fillna(0.0))
        payoff_ann = float(np.log1p(np.clip(payoff_horizon, -0.95, None)) * float(annualization) / int(horizon))
        state_values = state_by_date.get(dt, {})
        rows.append(
            {
                "date": dt,
                "payoff_end_date": R.index[pos + int(horizon)],
                "view_family": row["view_family"],
                "family_name": row.get("family_name", row.get("family_display_name", row["view_family"])),
                "family_display_name": row.get("family_display_name", row.get("family_name", row["view_family"])),
                "view_name": row["view_name"],
                "view_state": row.get("view_state", row.get("risk_orientation", "neutral")),
                "economic_theme": row.get("economic_theme", row.get("family_display_name", row["view_family"])),
                "signal_value": row.get("signal_value", np.nan),
                "raw_strength": row.get("raw_strength", abs(row.get("signal_value", np.nan))),
                "signal_strength": abs(row.get("signal_value", np.nan)),
                "q_tilt": row.get("q_tilt", np.nan),
                "q": row.get("q", np.nan),
                "payoff_horizon": payoff_horizon,
                "payoff_ann": payoff_ann,
                "payoff": payoff_ann,
                "hit": payoff_horizon > 0,
                "risk_orientation": row.get("risk_orientation", "neutral"),
                "economic_label": row.get("economic_label", ""),
                "spy_drawdown_252": state_values.get("spy_drawdown_252", np.nan),
                "spy_vol_z": state_values.get("spy_vol_z", np.nan),
                "risky_trend_breadth": state_values.get("risky_trend_breadth", np.nan),
                "hyg_lqd_63": state_values.get("hyg_lqd_63", np.nan),
                "stress_state": stress_state_from_values(state_values) if state_values else False,
                "diagnostics": row.get("diagnostics", {}),
                "p_vector": p.round(6).to_dict(),
            }
        )
    return pd.DataFrame(rows)


def family_reliability(
    view_family: str,
    history: pd.DataFrame | None,
    current_date: pd.Timestamp | str,
    *,
    protective_families: Sequence[str] = ("correlation_stress",),
) -> dict[str, float]:
    current_date = pd.Timestamp(current_date)
    if history is None or len(history) == 0:
        sample = pd.DataFrame()
    else:
        sample = history[
            (history["view_family"] == view_family)
            & (pd.to_datetime(history["date"]) < current_date)
            & (pd.to_datetime(history["payoff_end_date"]) < current_date)
        ].sort_values("date")
    empty = {
        "n_obs": 0,
        "hit_rate": np.nan,
        "recent_hit_rate": np.nan,
        "avg_payoff": 0.0,
        "payoff_vol": np.nan,
        "payoff_ir": 0.0,
        "t_stat": 0.0,
        "info_coefficient": 0.0,
        "sign_stability": np.nan,
        "recent_decay": 0.0,
        "sample_score": 0.0,
        "stress_n_obs": 0,
        "stress_hit_rate": np.nan,
        "stress_avg_payoff": 0.0,
        "stress_payoff_ir": 0.0,
    }
    if sample.empty:
        return empty
    payoff_col = "payoff_ann" if "payoff_ann" in sample.columns else "payoff"
    payoff = pd.to_numeric(sample[payoff_col], errors="coerce").dropna()
    n_obs = int(len(payoff))
    if n_obs == 0:
        return empty
    aligned = sample.loc[payoff.index]
    avg_payoff = float(payoff.mean())
    payoff_vol = float(payoff.std(ddof=1)) if n_obs > 1 else np.nan
    payoff_ir = avg_payoff / payoff_vol if np.isfinite(payoff_vol) and payoff_vol > 1e-10 else 0.0
    t_stat = payoff_ir * math.sqrt(n_obs) if n_obs > 1 else 0.0
    hit_rate = float((payoff > 0).mean())
    recent = payoff.tail(6)
    recent_hit_rate = float((recent > 0).mean()) if len(recent) else np.nan
    signal_strength = pd.to_numeric(aligned.get("signal_strength", aligned.get("signal_value", 0.0)), errors="coerce").reindex(payoff.index).fillna(0.0)
    info_coefficient = (
        float(signal_strength.corr(payoff))
        if n_obs >= 8 and signal_strength.std(ddof=1) > 1e-10 and payoff.std(ddof=1) > 1e-10
        else 0.0
    )
    if not np.isfinite(info_coefficient):
        info_coefficient = 0.0
    stress_n_obs, stress_hit_rate, stress_avg_payoff, stress_payoff_ir = 0, np.nan, 0.0, 0.0
    if view_family in set(protective_families) and "stress_state" in aligned.columns:
        stress_mask = aligned["stress_state"].astype(bool).reindex(payoff.index).fillna(False)
        stress_payoff = payoff[stress_mask]
        stress_n_obs = int(len(stress_payoff))
        if stress_n_obs:
            stress_avg_payoff = float(stress_payoff.mean())
            stress_hit_rate = float((stress_payoff > 0).mean())
            stress_vol = float(stress_payoff.std(ddof=1)) if stress_n_obs > 1 else np.nan
            stress_payoff_ir = stress_avg_payoff / stress_vol if np.isfinite(stress_vol) and stress_vol > 1e-10 else 0.0
    return {
        "n_obs": n_obs,
        "hit_rate": hit_rate,
        "recent_hit_rate": recent_hit_rate,
        "avg_payoff": avg_payoff,
        "payoff_vol": payoff_vol,
        "payoff_ir": payoff_ir,
        "t_stat": t_stat,
        "info_coefficient": info_coefficient,
        "sign_stability": hit_rate,
        "recent_decay": float(recent.mean() - avg_payoff) if len(recent) else 0.0,
        "sample_score": float(np.clip((n_obs - 4) / 20.0, 0.0, 1.0)),
        "stress_n_obs": stress_n_obs,
        "stress_hit_rate": stress_hit_rate,
        "stress_avg_payoff": stress_avg_payoff,
        "stress_payoff_ir": stress_payoff_ir,
    }


def reliability_note(stats: Mapping[str, Any], row: Mapping[str, Any], *, protective_families: Sequence[str] = ("correlation_stress",)) -> str | None:
    n_obs = stats.get("n_obs", 0)
    hit_rate = stats.get("hit_rate", np.nan)
    recent_hit_rate = stats.get("recent_hit_rate", np.nan)
    payoff_ir = stats.get("payoff_ir", np.nan)
    if n_obs >= 12 and np.isfinite(hit_rate) and hit_rate < 0.52:
        return "soft reliability note: hit rate below 52 percent"
    if n_obs >= 12 and np.isfinite(payoff_ir) and payoff_ir <= 0.03:
        return "soft reliability note: payoff IR too low"
    if n_obs >= 8 and np.isfinite(recent_hit_rate) and recent_hit_rate < 0.35:
        return "soft reliability note: recent hit rate below 35 percent"
    if row.get("view_family") in set(protective_families) and stats.get("stress_n_obs", 0) >= 6:
        stress_hit = stats.get("stress_hit_rate", np.nan)
        stress_ir = stats.get("stress_payoff_ir", np.nan)
        if (np.isfinite(stress_hit) and stress_hit < 0.50) or (np.isfinite(stress_ir) and stress_ir <= 0.0):
            return "soft reliability note: protective stress payoff weak"
    return None


def confidence_score(
    stats: Mapping[str, Any],
    row: Mapping[str, Any],
    market_state: Mapping[str, Any],
    *,
    confidence_mode: str = "learned",
    conf_floor: float = 0.30,
    conf_cap: float = 0.90,
    haircut_conf_floor: float = 0.20,
    haircut_conf_cap: float = 0.85,
    protective_families: Sequence[str] = ("correlation_stress",),
) -> dict[str, Any]:
    n_obs = stats.get("n_obs", 0)
    hit_rate = stats.get("hit_rate", np.nan) if np.isfinite(stats.get("hit_rate", np.nan)) else 0.50
    recent_hit_rate = stats.get("recent_hit_rate", np.nan) if np.isfinite(stats.get("recent_hit_rate", np.nan)) else hit_rate
    payoff_ir = stats.get("payoff_ir", 0.0) if np.isfinite(stats.get("payoff_ir", 0.0)) else 0.0
    t_stat = stats.get("t_stat", 0.0) if np.isfinite(stats.get("t_stat", 0.0)) else 0.0
    info_coefficient = stats.get("info_coefficient", 0.0) if np.isfinite(stats.get("info_coefficient", 0.0)) else 0.0
    score = 0.40 + 0.14 * math.tanh(t_stat / 2.0) + 0.12 * math.tanh(2.0 * payoff_ir) + 0.08 * math.tanh(2.0 * info_coefficient)
    score += 0.18 * (hit_rate - 0.50) * 2.0 + 0.05 * stats.get("sample_score", 0.0) + 0.05 * float(row.get("confluence_score", 0.0))
    if n_obs < 8:
        score = min(score, 0.52)
    if payoff_ir <= 0.03 and hit_rate <= 0.52 and n_obs >= 8:
        score = min(score, 0.46)
    if recent_hit_rate < 0.35 and n_obs >= 8:
        score = min(score - 0.08, 0.42)
    if row.get("view_family") in set(protective_families) and stats.get("stress_n_obs", 0) >= 6:
        if stats.get("stress_hit_rate", 1.0) < 0.50 or stats.get("stress_payoff_ir", 1.0) <= 0.0:
            score = min(score, 0.40)
    if market_state.get("equity_stress", False) and row.get("risk_orientation") == "risk_on":
        score -= 0.03
    haircut_multiplier = 1.0
    if confidence_mode == "fixed":
        score = 0.50
    elif confidence_mode == "haircut":
        if market_state.get("equity_stress", False) and row.get("risk_orientation") == "risk_on":
            haircut_multiplier *= 0.85
        if recent_hit_rate < 0.35 and n_obs >= 8:
            haircut_multiplier *= 0.75
        score = float(np.clip(score * haircut_multiplier, haircut_conf_floor, haircut_conf_cap))
    else:
        score = float(np.clip(score, conf_floor, conf_cap))
    out = dict(stats)
    out.update(
        {
            "confidence": score,
            "confidence_mode": confidence_mode,
            "haircut_multiplier": haircut_multiplier,
            "risk_orientation": row.get("risk_orientation", "neutral"),
            "reliability_note": reliability_note(stats, row, protective_families=protective_families),
        }
    )
    return out


def q_from_strength(view_family: str, view_strength: float, family_q_caps: Mapping[str, float], *, q_strength_scale: float = 1.25) -> float:
    cap = float(family_q_caps.get(view_family, 0.020))
    if not np.isfinite(view_strength) or cap <= 0:
        return 0.0
    return float(cap * math.tanh(abs(float(view_strength)) / float(q_strength_scale)))


def soft_multiplier(stats: Mapping[str, Any]) -> float:
    n_obs = stats.get("n_obs", 0)
    hit_rate = stats.get("hit_rate", np.nan)
    recent_hit_rate = stats.get("recent_hit_rate", np.nan)
    payoff_ir = stats.get("payoff_ir", np.nan)
    if n_obs < 8:
        return 0.60
    mult = 1.00
    if np.isfinite(hit_rate):
        if hit_rate < 0.48:
            mult *= 0.60
        elif hit_rate < 0.52:
            mult *= 0.80
        elif hit_rate > 0.58:
            mult *= 1.10
    if np.isfinite(payoff_ir):
        if payoff_ir < -0.10:
            mult *= 0.50
        elif payoff_ir <= 0.03:
            mult *= 0.75
        elif payoff_ir > 0.20:
            mult *= 1.15
    if np.isfinite(recent_hit_rate):
        if recent_hit_rate < 0.35:
            mult *= 0.75
        elif recent_hit_rate > 0.65:
            mult *= 1.10
    return float(np.clip(mult, 0.35, 1.25))


def learned_q_for_view(
    row: Mapping[str, Any],
    raw_q: float,
    history: pd.DataFrame | None,
    current_date: pd.Timestamp | str,
    *,
    family_q_caps: Mapping[str, float],
    use_learned_q: bool = True,
    protective_families: Sequence[str] = ("correlation_stress",),
) -> tuple[float, float, float, str, dict[str, Any]]:
    cap = float(family_q_caps.get(row.get("view_family"), 0.020))
    stats = family_reliability(str(row.get("view_family")), history, current_date, protective_families=protective_families)
    raw_q = float(np.clip(abs(raw_q), 0.0, cap))
    learned_component = 0.0
    q_final = raw_q
    q_model = "rule q only"
    if use_learned_q:
        mult = soft_multiplier(stats)
        q_final = float(np.clip(raw_q * mult, 0.0, cap))
        learned_component = q_final - raw_q
        q_model = "soft reliability scaled q"
    q_shrinkage_or_boost = (q_final / raw_q - 1.0) if raw_q > 1e-12 else np.nan
    return q_final, learned_component, q_shrinkage_or_boost, q_model, stats


def payoff_quality(stats: Mapping[str, Any]) -> float:
    n_obs = stats.get("n_obs", 0)
    if n_obs < 8:
        return 0.45
    hit_rate = stats.get("hit_rate", np.nan)
    recent_hit_rate = stats.get("recent_hit_rate", np.nan)
    payoff_ir = stats.get("payoff_ir", np.nan)
    sample_score = stats.get("sample_score", 0.0)
    hit_component = np.clip(((hit_rate if np.isfinite(hit_rate) else 0.50) - 0.50) / 0.15, 0.0, 1.0)
    recent_component = np.clip(((recent_hit_rate if np.isfinite(recent_hit_rate) else hit_rate) - 0.50) / 0.20, 0.0, 1.0)
    ir_component = np.clip((payoff_ir if np.isfinite(payoff_ir) else 0.0) / 0.30, 0.0, 1.0)
    return float(np.clip(0.35 * hit_component + 0.25 * ir_component + 0.20 * recent_component + 0.20 * sample_score, 0.0, 1.0))


def novelty_score(row: Mapping[str, Any], history: pd.DataFrame | None, current_date: pd.Timestamp | str) -> float:
    if history is None or len(history) == 0:
        return 1.0
    current = pd.Timestamp(current_date)
    recent = history[(pd.to_datetime(history["date"]) < current) & (pd.to_datetime(history["date"]) >= current - pd.DateOffset(months=12))]
    family_count = int((recent["view_family"] == row["view_family"]).sum()) if not recent.empty else 0
    return float(1.0 / (1.0 + family_count / 6.0))


def view_score(
    row: Mapping[str, Any],
    stats: Mapping[str, Any],
    history: pd.DataFrame | None,
    current_date: pd.Timestamp | str,
    *,
    family_q_caps: Mapping[str, float],
) -> dict[str, float]:
    cap = float(family_q_caps.get(row.get("view_family"), 0.020))
    q_score = float(np.clip(abs(float(row.get("q_tilt", 0.0))) / cap, 0.0, 1.0)) if cap > 0 else 0.0
    confluence_score = float(np.clip(row.get("confluence_score", 0.0), 0.0, 1.0))
    historical_confidence = float(stats["confidence"] if np.isfinite(stats.get("confidence", np.nan)) else 0.50)
    payoff_quality_val = payoff_quality(stats)
    novelty = novelty_score(row, history, current_date)
    basket_breadth = float(np.clip((len(row.get("long_assets", [])) + len(row.get("short_assets", []))) / 8.0, 0.0, 1.0))
    diversification_bonus = float(np.clip(0.70 * basket_breadth + 0.30 * novelty, 0.0, 1.0))
    economic_priority = float(np.clip(row.get("economic_priority", row.get("priority", 0.0)), 0.0, 1.0))
    selected_score = 0.25 * historical_confidence + 0.25 * payoff_quality_val + 0.20 * confluence_score + 0.15 * q_score + 0.10 * economic_priority + 0.05 * diversification_bonus
    return {
        "historical_confidence": historical_confidence,
        "payoff_quality": payoff_quality_val,
        "novelty_score": novelty,
        "diversification_bonus": diversification_bonus,
        "economic_priority": economic_priority,
        "q_score": q_score,
        "selected_score": float(selected_score),
    }


def exposure_vector(row: Mapping[str, Any], assets: Sequence[str]) -> pd.Series:
    p = p_row_from_view(row, assets)
    return p if p is not None else pd.Series(0.0, index=[str(x) for x in assets], dtype=float)


def exposure_similarity(row_a: Mapping[str, Any], row_b: Mapping[str, Any], assets: Sequence[str]) -> float:
    a = exposure_vector(row_a, assets).values
    b = exposure_vector(row_b, assets).values
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return np.nan
    return float(a @ b / denom)


def select_views(
    active_views: Sequence[Mapping[str, Any]],
    history: pd.DataFrame | None,
    current_date: pd.Timestamp | str,
    market_state: Mapping[str, Any],
    *,
    assets: Sequence[str],
    family_q_caps: Mapping[str, float],
    confidence_mode: str = "learned",
    max_selected_views: int = 5,
    redundancy_similarity: float = 0.85,
    max_same_direction: int = 3,
    protective_families: Sequence[str] = ("correlation_stress",),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluated: list[dict[str, Any]] = []
    for raw_row in active_views:
        row = dict(raw_row)
        stats_raw = family_reliability(str(row["view_family"]), history, current_date, protective_families=protective_families)
        stats = confidence_score(stats_raw, row, market_state, confidence_mode=confidence_mode, protective_families=protective_families)
        cap = float(family_q_caps.get(row.get("view_family"), 0.020))
        if stats["n_obs"] < 8:
            row["q_tilt"] = min(float(row.get("q_tilt", 0.0)), min(cap, 0.012))
        score_parts = view_score(row, stats, history, current_date, family_q_caps=family_q_caps)
        row.update(
            {
                "confidence": score_parts["historical_confidence"],
                "hit_rate": stats["hit_rate"],
                "recent_hit_rate": stats["recent_hit_rate"],
                "avg_payoff": stats["avg_payoff"],
                "payoff_vol": stats["payoff_vol"],
                "payoff_ir": stats["payoff_ir"],
                "t_stat": stats["t_stat"],
                "info_coefficient": stats["info_coefficient"],
                "n_obs": stats["n_obs"],
                "stress_n_obs": stats.get("stress_n_obs", 0),
                "stress_hit_rate": stats.get("stress_hit_rate", np.nan),
                "stress_payoff_ir": stats.get("stress_payoff_ir", np.nan),
                **score_parts,
                "reliability_note": stats.get("reliability_note"),
            }
        )
        evaluated.append({"row": row, "skip": False, "reason": "eligible", "redundancy_similarity": np.nan})

    selected: list[dict[str, Any]] = []
    for item in sorted(evaluated, key=lambda x: x["row"]["selected_score"], reverse=True):
        row = item["row"]
        max_similarity, redundant = np.nan, False
        for kept in selected:
            similarity = exposure_similarity(row, kept, assets)
            if np.isfinite(similarity):
                max_similarity = max(abs(similarity), max_similarity if np.isfinite(max_similarity) else 0.0)
                if abs(similarity) > float(redundancy_similarity):
                    redundant = True
        item["redundancy_similarity"] = max_similarity
        if redundant:
            item["skip"], item["reason"] = True, "redundancy gate: exposure cosine similarity"
            continue
        same_direction = sum(
            kept.get("risk_orientation") == row.get("risk_orientation") and row.get("risk_orientation") in ["risk_on", "risk_off"]
            for kept in selected
        )
        if same_direction >= int(max_same_direction) and row["selected_score"] < 0.78:
            item["skip"], item["reason"] = True, "direction crowding gate"
            continue
        if len(selected) >= int(max_selected_views):
            item["skip"], item["reason"] = True, "below top selected views"
            continue
        selected.append(row)

    selected_ids = {id(row) for row in selected}
    log_rows: list[dict[str, Any]] = []
    for item in evaluated:
        row = item["row"]
        kept = id(row) in selected_ids
        log_rows.append(
            {
                "date": pd.Timestamp(current_date),
                "view_family": row["view_family"],
                "family_display_name": row.get("family_display_name", row.get("family_name", row["view_family"])),
                "view_name": row["view_name"],
                "view_state": row.get("view_state", row.get("risk_orientation", "neutral")),
                "economic_theme": row.get("economic_theme", row.get("family_display_name", row["view_family"])),
                "kept": kept,
                "scale_reason": "kept" if kept else item["reason"],
                "confidence_mode": confidence_mode,
                "confidence": row.get("confidence", np.nan),
                "historical_confidence": row.get("historical_confidence", np.nan),
                "payoff_quality": row.get("payoff_quality", np.nan),
                "confluence_score": row.get("confluence_score", np.nan),
                "novelty_score": row.get("novelty_score", np.nan),
                "diversification_bonus": row.get("diversification_bonus", np.nan),
                "economic_priority": row.get("economic_priority", np.nan),
                "q_tilt": row.get("q_tilt", np.nan),
                "q_score": row.get("q_score", np.nan),
                "selected_score": row.get("selected_score", np.nan),
                "n_obs": row.get("n_obs", np.nan),
                "hit_rate": row.get("hit_rate", np.nan),
                "recent_hit_rate": row.get("recent_hit_rate", np.nan),
                "payoff_ir": row.get("payoff_ir", np.nan),
                "avg_payoff": row.get("avg_payoff", np.nan),
                "t_stat": row.get("t_stat", np.nan),
                "info_coefficient": row.get("info_coefficient", np.nan),
                "stress_n_obs": row.get("stress_n_obs", np.nan),
                "stress_hit_rate": row.get("stress_hit_rate", np.nan),
                "stress_payoff_ir": row.get("stress_payoff_ir", np.nan),
                "signal_value": row.get("signal_value", np.nan),
                "raw_strength": row.get("raw_strength", row.get("view_strength", np.nan)),
                "view_strength": row.get("view_strength", np.nan),
                "redundancy_similarity": item.get("redundancy_similarity", np.nan),
                "source": row.get("source", ""),
                "risk_orientation": row.get("risk_orientation", "neutral"),
                "reliability_note": row.get("reliability_note"),
            }
        )
    return selected, log_rows


def view_matrix(
    active_views: Sequence[Mapping[str, Any]],
    assets: Sequence[str],
    *,
    prior_mu: pd.Series | Sequence[float] | None = None,
    history: pd.DataFrame | None = None,
    current_date: pd.Timestamp | str | None = None,
    family_q_caps: Mapping[str, float],
    q_strength_scale: float = 1.25,
    use_learned_q: bool = True,
    protective_families: Sequence[str] = ("correlation_stress",),
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    clean_rows: list[dict[str, Any]] = []
    view_p_rows: list[np.ndarray] = []
    view_q_vals: list[float] = []
    asset_list = [str(x) for x in assets]
    prior = pd.Series(prior_mu, index=asset_list, dtype=float).reindex(asset_list).fillna(0.0) if prior_mu is not None else pd.Series(0.0, index=asset_list)
    for row_raw in active_views:
        row = dict(row_raw)
        long_assets = [str(x) for x in row.get("long_assets", []) if str(x) in asset_list]
        short_assets = [str(x) for x in row.get("short_assets", []) if str(x) in asset_list]
        if not long_assets or not short_assets:
            continue
        p = p_row_from_view(row, asset_list)
        if p is None:
            continue
        p_row = p.reindex(asset_list).fillna(0.0).to_numpy(dtype=float)
        cap = float(family_q_caps.get(row.get("view_family"), 0.020))
        strength = float(abs(row.get("view_strength", row.get("signal_value", 0.0))))
        raw_q_tilt = float(np.clip(abs(row.get("q_tilt", q_from_strength(row.get("view_family"), strength, family_q_caps, q_strength_scale=q_strength_scale))), 0.0, cap))
        q_final, learned_component, q_shrinkage_or_boost, q_model, stats = learned_q_for_view(
            row,
            raw_q_tilt,
            history,
            current_date or pd.Timestamp("1900-01-01"),
            family_q_caps=family_q_caps,
            use_learned_q=use_learned_q,
            protective_families=protective_families,
        )
        base_spread = float(p_row @ prior.values)
        q_val = base_spread + q_final
        expression_parts = [f"{1.0 / len(long_assets):.2f}*{asset}" for asset in long_assets] + [f"-{1.0 / len(short_assets):.2f}*{asset}" for asset in short_assets]
        rec = dict(row)
        rec["base_view_spread"] = base_spread
        rec["raw_q"] = raw_q_tilt
        rec["raw_q_tilt"] = raw_q_tilt
        rec["learned_q_component"] = learned_component
        rec["final_q"] = q_final
        rec["q_tilt"] = q_final
        rec["q_tilt_final"] = q_final
        rec["q"] = q_val
        rec["q_model"] = q_model
        rec["q_shrinkage_or_boost"] = q_shrinkage_or_boost
        rec["q_shrink"] = raw_q_tilt - q_final
        rec["q_n_obs"] = stats.get("n_obs", np.nan)
        rec["q_hit_rate"] = stats.get("hit_rate", np.nan)
        rec["q_recent_hit_rate"] = stats.get("recent_hit_rate", np.nan)
        rec["q_payoff_ir"] = stats.get("payoff_ir", np.nan)
        rec["q_avg_payoff"] = stats.get("avg_payoff", np.nan)
        rec["q_t_stat"] = stats.get("t_stat", np.nan)
        rec["q_info_coefficient"] = stats.get("info_coefficient", np.nan)
        rec["p_expression"] = " + ".join(expression_parts).replace("+ -", "- ")
        rec["p_vector"] = pd.Series(p_row, index=asset_list).round(6).to_dict()
        clean_rows.append(rec)
        view_p_rows.append(p_row)
        view_q_vals.append(q_val)
    view_p = np.vstack(view_p_rows) if view_p_rows else np.empty((0, len(asset_list)))
    view_q = np.asarray(view_q_vals, dtype=float)
    return view_p, view_q, pd.DataFrame(clean_rows)


__all__ = [
    "confidence_score",
    "exposure_similarity",
    "exposure_vector",
    "family_reliability",
    "learned_q_for_view",
    "novelty_score",
    "payoff_history",
    "payoff_quality",
    "p_row_from_view",
    "q_from_strength",
    "reliability_note",
    "select_views",
    "soft_multiplier",
    "stress_state_from_values",
    "view_matrix",
    "view_score",
]
