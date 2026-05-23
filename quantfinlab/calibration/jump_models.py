from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.calibration.fft_cos import compare_fourier_models, family_winner
from quantfinlab.options.fourier import cos_density, tail_probability


def _numeric_column(frame: pd.DataFrame, names, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def jump_family_table(merton_fit: dict, vg_fit: dict) -> pd.DataFrame:
    return compare_fourier_models(pd.DataFrame(), {"merton": merton_fit, "vg": vg_fit})


def sv_family_table(heston_fit: dict, bates_fit: dict) -> pd.DataFrame:
    return compare_fourier_models(pd.DataFrame(), {"heston": heston_fit, "bates": bates_fit})


def density_summary(models: dict[str, tuple[str, dict]], spot: float, rate: float, dividend_yield: float, tau: float, *, crash_level: float = -0.15) -> pd.DataFrame:
    x = np.linspace(np.log(float(spot)) - 0.8, np.log(float(spot)) + 0.6, 501)
    rows = []
    for label, (model, params) in models.items():
        density = cos_density(model, params, x, spot, rate, dividend_yield, tau)
        rows.append(
            {
                "model": label,
                "tau": float(tau),
                "left_tail_probability": tail_probability(x - np.log(float(spot)), density, crash_level),
                "density_peak": float(np.nanmax(density)),
                "mean_log_level": float(np.trapezoid(x * density, x)),
            }
        )
    return pd.DataFrame(rows)


def hedge_score_components(candidates: pd.DataFrame, *, crash_level: float = 0.85) -> pd.DataFrame:
    q = candidates.copy()
    spot = _numeric_column(q, ("spot",), 1.0).clip(lower=1e-8)
    strike = _numeric_column(q, ("strike",), 0.0)
    mid = _numeric_column(q, ("mid", "mark"), 0.0).clip(lower=1e-8)
    dte = _numeric_column(q, ("dte_days",), 45.0).clip(lower=1.0)
    payoff_90 = _numeric_column(q, ("payoff_at_90",), np.nan)
    payoff_85 = _numeric_column(q, ("payoff_at_85",), np.nan)
    payoff_80 = _numeric_column(q, ("payoff_at_80",), np.nan)
    payoff_90 = payoff_90.where(np.isfinite(payoff_90), np.maximum(strike - 0.90 * spot, 0.0))
    payoff_85 = payoff_85.where(np.isfinite(payoff_85), np.maximum(strike - 0.85 * spot, 0.0))
    payoff_80 = payoff_80.where(np.isfinite(payoff_80), np.maximum(strike - 0.80 * spot, 0.0))
    p90 = _numeric_column(q, ("p90", "tail_prob_90"), 0.12).clip(lower=0.0, upper=1.0)
    p85 = _numeric_column(q, ("p85", "tail_prob_85"), 0.07).clip(lower=0.0, upper=1.0)
    p80 = _numeric_column(q, ("p80", "tail_prob_80", "crash_probability"), 0.04).clip(lower=0.0, upper=1.0)
    crash_efficiency = p90 * payoff_90 / mid + p85 * payoff_85 / mid + p80 * payoff_80 / mid
    model_edge = _numeric_column(q, ("model_edge",), 0.0)
    if "model_edge" not in q.columns and "price_residual" in q.columns:
        model_edge = -_numeric_column(q, ("price_residual",), 0.0)
    edge_ratio = model_edge / mid
    convexity = _numeric_column(q, ("convexity", "gamma", "vega"), 0.0).abs()
    downside_slope = (payoff_80 - payoff_90).clip(lower=0.0) / (0.10 * spot)
    convexity_per_premium = (convexity * spot + downside_slope) / mid
    spread = _numeric_column(q, ("relative_spread", "rel_spread"), 0.0).clip(lower=0.0)
    if "model_uncertainty" in q.columns:
        uncertainty = _numeric_column(q, ("model_uncertainty",), 0.0)
    else:
        uncertainty = _numeric_column(q, ("model_disagreement",), 0.0) / mid
    bleed = mid / spot / np.sqrt(dte / 365.25)
    penalty = _numeric_column(q, ("calibration_penalty", "fit_penalty"), 0.0).clip(lower=0.0)
    return pd.DataFrame(
        {
            "expected_crash_efficiency": crash_efficiency.clip(lower=0.0),
            "fair_value_edge": edge_ratio,
            "convexity_per_premium": convexity_per_premium.clip(lower=0.0),
            "relative_spread": spread,
            "premium_bleed": bleed,
            "calibration_penalty": penalty,
            "model_uncertainty": uncertainty,
        },
        index=q.index,
    )


def tail_hedge_score(candidates: pd.DataFrame, *, crash_level: float = 0.85) -> pd.Series:
    c = hedge_score_components(candidates, crash_level=crash_level)
    return (
        0.45 * c["expected_crash_efficiency"]
        + 0.25 * c["fair_value_edge"]
        + 0.15 * c["convexity_per_premium"]
        - 0.10 * c["relative_spread"]
        - 0.10 * c["premium_bleed"]
        - 0.10 * c["calibration_penalty"]
        - 0.10 * c["model_uncertainty"]
    )


def tail_hedge_candidates(quotes: pd.DataFrame, *, top_n: int | None = None, top_n_per_date: int = 5, crash_level: float = 0.85) -> pd.DataFrame:
    out = quotes.copy()
    out = out[out["option_type"].astype(str).str.lower().str.startswith("p")].copy()
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if "dte_days" not in out.columns:
        out["dte_days"] = pd.to_numeric(out["tau"], errors="coerce") * 365.25
    out = out[out["dte_days"].between(14, 120)].copy()
    spot = pd.to_numeric(out["spot"], errors="coerce").clip(lower=1e-8)
    strike = pd.to_numeric(out["strike"], errors="coerce")
    mid = pd.to_numeric(out.get("mid"), errors="coerce").clip(lower=1e-8)
    out["moneyness"] = strike / spot
    out["tail_payoff"] = np.maximum(strike - float(crash_level) * spot, 0.0)
    out["payoff_at_90"] = np.maximum(strike - 0.90 * spot, 0.0)
    out["payoff_at_85"] = np.maximum(strike - 0.85 * spot, 0.0)
    out["payoff_at_80"] = np.maximum(strike - 0.80 * spot, 0.0)
    out["payoff_per_premium"] = out["tail_payoff"] / mid
    out["premium_bleed"] = mid / spot / np.sqrt(out["dte_days"].clip(lower=1.0) / 365.25)
    out["edge_ratio"] = _numeric_column(out, ("model_edge",), 0.0) / mid
    out["hedge_score"] = tail_hedge_score(out, crash_level=crash_level)
    out = out.sort_values(["date", "hedge_score"], ascending=[True, False])
    if top_n_per_date is not None and int(top_n_per_date) > 0:
        out = out.groupby("date", group_keys=False).head(int(top_n_per_date))
    if top_n is not None:
        out = out.sort_values("hedge_score", ascending=False).head(int(top_n))
    return out.reset_index(drop=True)


def fixed_delta_hedge_candidates(
    quotes: pd.DataFrame,
    *,
    target_delta: float = 0.25,
    target_moneyness: float = 0.85,
    top_n: int | None = None,
    top_n_per_date: int = 1,
) -> pd.DataFrame:
    out = quotes.copy()
    out = out[out["option_type"].astype(str).str.lower().str.startswith("p")].copy()
    if out.empty:
        return out
    if "dte_days" not in out.columns:
        out["dte_days"] = pd.to_numeric(out["tau"], errors="coerce") * 365.25
    out = out[out["dte_days"].between(14, 120)].copy()
    if out.empty:
        return out
    m = pd.to_numeric(out["strike"], errors="coerce") / pd.to_numeric(out["spot"], errors="coerce")
    if "delta" in out.columns:
        delta_score = (_numeric_column(out, ("delta",), -target_delta).abs() - float(target_delta)).abs()
    else:
        delta_score = (m - float(target_moneyness)).abs()
    spread = _numeric_column(out, ("rel_spread", "relative_spread"), 0.0)
    out["hedge_score"] = -(delta_score + 0.25 * spread)
    spot = pd.to_numeric(out["spot"], errors="coerce").clip(lower=1e-8)
    strike = pd.to_numeric(out["strike"], errors="coerce")
    mid = pd.to_numeric(out.get("mid"), errors="coerce").clip(lower=1e-8)
    out["moneyness"] = strike / spot
    out["tail_payoff"] = np.maximum(strike - 0.85 * spot, 0.0)
    out["payoff_per_premium"] = out["tail_payoff"] / mid
    out["premium_bleed"] = mid / spot / np.sqrt(out["dte_days"].clip(lower=1.0) / 365.25)
    out["edge_ratio"] = 0.0
    out = out.sort_values(["date", "hedge_score"], ascending=[True, False])
    if top_n_per_date is not None and int(top_n_per_date) > 0:
        out = out.groupby("date", group_keys=False).head(int(top_n_per_date))
    if top_n is not None:
        out = out.head(int(top_n))
    return out.reset_index(drop=True)


def tail_hedge_schedule(
    candidates: pd.DataFrame,
    *,
    max_entries: int | None = None,
    spacing_days: int = 21,
    budget_notional: float = 1_000_000.0,
    premium_budget_bps: float = 100.0,
    budget_col: str | None = None,
    contract_multiplier: float = 100.0,
    label: str = "tail_put",
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=["entry_date", "contract_key", "quantity", "label"])
    q = candidates.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    if "contract_key" not in q.columns:
        q["contract_key"] = q["option_type"].astype(str) + "_" + pd.to_datetime(q["expiry"], errors="coerce").dt.strftime("%Y-%m-%d") + "_" + q["strike"].round(6).astype(str)
    rows = []
    last = None
    for d, group in q.sort_values(["date", "hedge_score"], ascending=[True, False]).groupby("date", sort=True):
        d = pd.Timestamp(d).normalize()
        if last is not None and (d - last).days < int(spacing_days):
            continue
        row = group.iloc[0]
        price = float(pd.to_numeric(row.get("ask", row.get("mid", np.nan)), errors="coerce"))
        if not np.isfinite(price) or price <= 0.0:
            continue
        budget_bps = float(row[budget_col]) if budget_col is not None and budget_col in row and pd.notna(row[budget_col]) else float(premium_budget_bps)
        budget = float(budget_notional) * budget_bps / 10000.0
        qty = budget / (price * float(contract_multiplier))
        rows.append(
            {
                "entry_date": d,
                "contract_key": row["contract_key"],
                "expiry": pd.Timestamp(row["expiry"]).normalize() if "expiry" in row and pd.notna(row["expiry"]) else pd.NaT,
                "strike": float(row["strike"]) if "strike" in row and pd.notna(row["strike"]) else np.nan,
                "option_type": str(row["option_type"]) if "option_type" in row else "",
                "quantity": float(qty),
                "label": label,
                "entry_score": float(row["hedge_score"]),
                "premium_budget": float(budget),
                "premium_budget_bps": float(budget_bps),
            }
        )
        last = d
        if max_entries is not None and len(rows) >= int(max_entries):
            break
    return pd.DataFrame(rows)


__all__ = [
    "density_summary",
    "family_winner",
    "fixed_delta_hedge_candidates",
    "hedge_score_components",
    "jump_family_table",
    "sv_family_table",
    "tail_hedge_candidates",
    "tail_hedge_schedule",
    "tail_hedge_score",
]
