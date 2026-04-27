from __future__ import annotations

import numpy as np
import pandas as pd


def option_position_greeks(
    row: pd.Series | dict,
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> pd.Series:
    data = pd.Series(row)
    scale = float(quantity) * float(multiplier)
    out = {}
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        value = data.get(greek, data.get(f"{greek}_mid", np.nan))
        out[f"{greek}_exposure"] = scale * float(value) if np.isfinite(value) else np.nan
    return pd.Series(out)


def portfolio_greek_exposure(
    positions: pd.DataFrame,
    quantity_col: str = "quantity",
    multiplier_col: str = "multiplier",
) -> pd.Series:
    if positions.empty:
        return pd.Series({f"{g}_exposure": 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]})
    frame = positions.copy()
    if multiplier_col not in frame.columns:
        frame[multiplier_col] = 1.0
    rows = [
        option_position_greeks(row, row.get(quantity_col, 1.0), row.get(multiplier_col, 1.0))
        for _, row in frame.iterrows()
    ]
    return pd.DataFrame(rows).sum(axis=0)


def target_delta_hedge(delta_exposure: float, hedge_delta: float = 1.0) -> float:
    hedge_delta = float(hedge_delta)
    if abs(hedge_delta) < 1e-12:
        return np.nan
    return -float(delta_exposure) / hedge_delta


def target_vega_hedge(vega_exposure: float, hedge_vega: float) -> float:
    hedge_vega = float(hedge_vega)
    if abs(hedge_vega) < 1e-12:
        return np.nan
    return -float(vega_exposure) / hedge_vega


def build_delta_hedge_targets(
    greek_table: pd.DataFrame,
    option_quantity: float = 1.0,
    multiplier: float = 1.0,
    delta_col: str = "delta",
) -> pd.DataFrame:
    out = greek_table.copy()
    out["option_delta_exposure"] = pd.to_numeric(out[delta_col], errors="coerce") * option_quantity * multiplier
    out["target_underlying_units"] = -out["option_delta_exposure"]
    return out


def build_delta_vega_hedge_targets(
    greek_table: pd.DataFrame,
    vega_hedge_table: pd.DataFrame,
    option_quantity: float = 1.0,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    out = build_delta_hedge_targets(greek_table, option_quantity=option_quantity, multiplier=multiplier)
    hedge = vega_hedge_table.copy()
    hedge_cols = ["date", "vega", "delta"]
    hedge = hedge[[c for c in hedge_cols if c in hedge.columns]].rename(
        columns={"vega": "hedge_vega", "delta": "hedge_delta"},
    )
    out = out.merge(hedge, on="date", how="left")
    out["target_vega_contracts"] = -out["vega"] * option_quantity * multiplier / out["hedge_vega"].replace(0, np.nan)
    residual_delta = out["delta"] * option_quantity * multiplier + out["target_vega_contracts"] * out["hedge_delta"] * multiplier
    out["target_underlying_units"] = -residual_delta
    return out


def hedge_trade_from_band(
    current_units: float,
    target_units: float,
    exposure: float,
    band: float,
) -> float:
    if not np.isfinite(target_units):
        return 0.0
    if abs(float(exposure)) <= float(band):
        return 0.0
    return float(target_units) - float(current_units)


def hedge_exposure_table(
    greek_table: pd.DataFrame,
    underlying_units_col: str = "underlying_units",
) -> pd.DataFrame:
    out = greek_table.copy()
    under = out[underlying_units_col] if underlying_units_col in out.columns else 0.0
    out["net_delta_exposure"] = out.get("delta", 0.0) + under
    out["net_vega_exposure"] = out.get("vega", 0.0)
    return out


def hedging_summary_table(results: dict) -> pd.DataFrame:
    if "summary" in results:
        return results["summary"]
    return pd.DataFrame()


__all__ = [
    "build_delta_hedge_targets",
    "build_delta_vega_hedge_targets",
    "hedge_exposure_table",
    "hedge_trade_from_band",
    "hedging_summary_table",
    "option_position_greeks",
    "portfolio_greek_exposure",
    "target_delta_hedge",
    "target_vega_hedge",
]
