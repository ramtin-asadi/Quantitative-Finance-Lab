from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = [
    "inflation_level_pressure",
    "inflation_impulse",
    "inflation_acceleration",
    "inflation_diffusion",
    "policy_tightness",
    "policy_shock",
    "real_rate_squeeze",
    "policy_catchup_risk",
    "growth_momentum_stress",
    "growth_acceleration_stress",
    "growth_breadth_stress",
    "survey_warning",
    "labor_cooling",
    "sahm_pressure",
    "housing_impulse_stress",
    "housing_rate_squeeze",
    "external_demand_stress",
    "external_vulnerability",
    "stress_breadth",
    "severe_stress_breadth",
    "stagflation_pressure",
    "goldilocks_support",
]

BLOCK_COLUMNS = [
    "inflation_pressure_block",
    "policy_rate_pressure_block",
    "growth_recession_block",
    "labor_cooling_block",
    "housing_domestic_block",
    "external_trade_block",
    "macro_breadth_conflict_block",
]

INFLATION_PREFIXES = ("cpi", "cpc", "ppi", "ipp", "rmp", "cpu")
POLICY_PREFIXES = ("inrt", "fdrh", "fdph", "gvbg")
GROWTH_PREFIXES = ("gdp", "inp", "rsa", "rsl", "pmmn", "pmsr", "pmcp", "ism", "isnf", "clin", "ldii", "ivp")
LABOR_UNEMPLOYMENT_PREFIXES = ("unrt",)
LABOR_CLAIMS_PREFIXES = ("injc", "ctcl")
LABOR_JOBS_PREFIXES = ("nfp", "adpe", "emch", "emci", "whs", "avwh")
HOUSING_PREFIXES = ("hbp", "hsp", "hos", "hoe", "hon", "hop", "nhp", "nahb", "psi", "s20")
EXTERNAL_DEMAND_PREFIXES = ("exp", "imp")
EXTERNAL_BALANCE_PREFIXES = ("trbn", "crab")
SURVEY_PREFIXES = ("cnci", "cncr", "cnrm", "cnry", "umcc", "bsi", "zwi", "zwe", "sntx", "cct", "cit")


def expanding_zscore(
    series: pd.Series,
    *,
    min_history: int = 60,
    clip: float | None = 5.0,
) -> pd.Series:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan)
    mean = s.expanding(min_periods=int(min_history)).mean()
    std = s.expanding(min_periods=int(min_history)).std(ddof=1).replace(0.0, np.nan)
    z = (s - mean) / std
    if clip is not None:
        z = z.clip(-float(clip), float(clip))
    return z


def prefix_columns(factors: pd.DataFrame, prefixes: tuple[str, ...] | list[str]) -> list[str]:
    lookup = tuple(str(prefix).lower() for prefix in prefixes)
    return [col for col in factors.columns if str(col).lower().startswith(lookup)]


def mean_z(factors: pd.DataFrame, columns: list[str], *, min_history: int, sign: float = 1.0) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=factors.index, dtype=float)
    z = factors[columns].apply(lambda s: expanding_zscore(s, min_history=min_history))
    return float(sign) * z.mean(axis=1, skipna=True)


def change_z(
    factors: pd.DataFrame,
    columns: list[str],
    *,
    periods: int,
    min_history: int,
    sign: float = 1.0,
) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=factors.index, dtype=float)
    z = factors[columns].diff(int(periods)).apply(lambda s: expanding_zscore(s, min_history=min_history))
    return float(sign) * z.mean(axis=1, skipna=True)


def stress_share(
    factors: pd.DataFrame,
    columns: list[str],
    *,
    min_history: int,
    threshold: float,
    sign: float = 1.0,
) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=factors.index, dtype=float)
    z = factors[columns].apply(lambda s: expanding_zscore(s, min_history=min_history))
    events = float(sign) * z > float(threshold)
    share = events.where(z.notna()).mean(axis=1, skipna=True)
    return expanding_zscore(share, min_history=min_history)


def inflation_level_pressure(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, INFLATION_PREFIXES)
    return mean_z(factors, cols, min_history=min_history).rename("inflation_level_pressure")


def inflation_impulse(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, INFLATION_PREFIXES)
    return change_z(factors, cols, periods=3, min_history=min_history).rename("inflation_impulse")


def inflation_acceleration(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, INFLATION_PREFIXES)
    if not cols:
        return pd.Series(np.nan, index=factors.index, name="inflation_acceleration", dtype=float)
    accel = factors[cols].diff(3).diff(3)
    z = accel.apply(lambda s: expanding_zscore(s, min_history=min_history))
    return z.mean(axis=1, skipna=True).rename("inflation_acceleration")


def inflation_diffusion(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, INFLATION_PREFIXES)
    return stress_share(factors, cols, min_history=min_history, threshold=0.5).rename("inflation_diffusion")


def policy_tightness(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, POLICY_PREFIXES)
    return mean_z(factors, cols, min_history=min_history).rename("policy_tightness")


def policy_shock(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, POLICY_PREFIXES)
    return change_z(factors, cols, periods=3, min_history=min_history).rename("policy_shock")


def real_rate_squeeze(
    policy_level: pd.Series | pd.DataFrame,
    inflation_impulse_series: pd.Series | None = None,
    *,
    min_history: int = 60,
) -> pd.Series:
    if inflation_impulse_series is None:
        factors = pd.DataFrame(policy_level)
        tightness = policy_tightness(factors, min_history=min_history)
        impulse = inflation_impulse(factors, min_history=min_history)
    else:
        tightness = pd.Series(policy_level, dtype=float)
        impulse = pd.Series(inflation_impulse_series, dtype=float).reindex(tightness.index)
    return (tightness - impulse).rename("real_rate_squeeze")


def policy_catchup_risk(inflation_level: pd.Series, policy_level: pd.Series, *, min_history: int = 60) -> pd.Series:
    risk = pd.Series(inflation_level, dtype=float) - pd.Series(policy_level, dtype=float)
    risk = risk.clip(lower=0.0)
    return expanding_zscore(risk, min_history=min_history).rename("policy_catchup_risk")


def growth_momentum_stress(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, GROWTH_PREFIXES)
    return mean_z(factors, cols, min_history=min_history, sign=-1.0).rename("growth_momentum_stress")


def growth_acceleration_stress(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, GROWTH_PREFIXES)
    return change_z(factors, cols, periods=3, min_history=min_history, sign=-1.0).rename("growth_acceleration_stress")


def growth_breadth_stress(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, GROWTH_PREFIXES)
    return stress_share(factors, cols, min_history=min_history, threshold=0.5, sign=-1.0).rename("growth_breadth_stress")


def survey_warning(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, SURVEY_PREFIXES)
    return mean_z(factors, cols, min_history=min_history, sign=-1.0).rename("survey_warning")


def labor_cooling(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    unemp = mean_z(factors, prefix_columns(factors, LABOR_UNEMPLOYMENT_PREFIXES), min_history=min_history)
    claims = mean_z(factors, prefix_columns(factors, LABOR_CLAIMS_PREFIXES), min_history=min_history)
    jobs = mean_z(factors, prefix_columns(factors, LABOR_JOBS_PREFIXES), min_history=min_history, sign=-1.0)
    return pd.concat([unemp, claims, jobs], axis=1).mean(axis=1, skipna=True).rename("labor_cooling")


def sahm_pressure(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, LABOR_UNEMPLOYMENT_PREFIXES)
    if not cols:
        return pd.Series(np.nan, index=factors.index, name="sahm_pressure", dtype=float)
    unemployment = factors[cols].mean(axis=1, skipna=True)
    gap = unemployment.rolling(3, min_periods=3).mean() - unemployment.rolling(12, min_periods=6).min()
    return expanding_zscore(gap, min_history=min_history).rename("sahm_pressure")


def housing_impulse_stress(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, HOUSING_PREFIXES)
    level = mean_z(factors, cols, min_history=min_history, sign=-1.0)
    impulse = change_z(factors, cols, periods=3, min_history=min_history, sign=-1.0)
    return pd.concat([level, impulse], axis=1).mean(axis=1, skipna=True).rename("housing_impulse_stress")


def housing_rate_squeeze(housing_impulse: pd.Series, policy_level: pd.Series) -> pd.Series:
    out = pd.concat([housing_impulse, policy_level], axis=1).mean(axis=1, skipna=True)
    return out.rename("housing_rate_squeeze")


def external_demand_stress(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, EXTERNAL_DEMAND_PREFIXES)
    return mean_z(factors, cols, min_history=min_history, sign=-1.0).rename("external_demand_stress")


def external_vulnerability(factors: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    cols = prefix_columns(factors, EXTERNAL_BALANCE_PREFIXES)
    return mean_z(factors, cols, min_history=min_history, sign=-1.0).rename("external_vulnerability")


def stress_breadth(signals: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    base = signals[[col for col in SIGNAL_COLUMNS[:18] if col in signals.columns]]
    share = (base > 0.5).where(base.notna()).mean(axis=1, skipna=True)
    return expanding_zscore(share, min_history=min_history).rename("stress_breadth")


def severe_stress_breadth(signals: pd.DataFrame, *, min_history: int = 60) -> pd.Series:
    base = signals[[col for col in SIGNAL_COLUMNS[:18] if col in signals.columns]]
    share = (base > 1.0).where(base.notna()).mean(axis=1, skipna=True)
    return expanding_zscore(share, min_history=min_history).rename("severe_stress_breadth")


def stagflation_pressure(signals: pd.DataFrame) -> pd.Series:
    cols = ["inflation_level_pressure", "policy_tightness", "growth_momentum_stress"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("stagflation_pressure")


def goldilocks_support(signals: pd.DataFrame) -> pd.Series:
    cols = ["inflation_level_pressure", "policy_tightness", "growth_momentum_stress", "labor_cooling"]
    return -signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("goldilocks_support")


def inflation_pressure_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["inflation_level_pressure", "inflation_impulse", "inflation_acceleration", "inflation_diffusion"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("inflation_pressure_block")


def policy_rate_pressure_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["policy_tightness", "policy_shock", "real_rate_squeeze", "policy_catchup_risk"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("policy_rate_pressure_block")


def growth_recession_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["growth_momentum_stress", "growth_acceleration_stress", "growth_breadth_stress", "survey_warning"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("growth_recession_block")


def labor_cooling_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["labor_cooling", "sahm_pressure"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("labor_cooling_block")


def housing_domestic_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["housing_impulse_stress", "housing_rate_squeeze"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("housing_domestic_block")


def external_trade_block(signals: pd.DataFrame) -> pd.Series:
    cols = ["external_demand_stress", "external_vulnerability"]
    return signals[[col for col in cols if col in signals.columns]].mean(axis=1, skipna=True).rename("external_trade_block")


def macro_breadth_conflict_block(signals: pd.DataFrame) -> pd.Series:
    parts = [
        signals.get("stress_breadth"),
        signals.get("severe_stress_breadth"),
        signals.get("stagflation_pressure"),
        -signals.get("goldilocks_support") if "goldilocks_support" in signals else None,
    ]
    parts = [part for part in parts if part is not None]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).rename("macro_breadth_conflict_block")


def condition_blocks(signals: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            inflation_pressure_block(signals),
            policy_rate_pressure_block(signals),
            growth_recession_block(signals),
            labor_cooling_block(signals),
            housing_domestic_block(signals),
            external_trade_block(signals),
            macro_breadth_conflict_block(signals),
        ],
        axis=1,
    )


def macro_block_snapshot(data: pd.DataFrame, dates: list[str | pd.Timestamp] | None = None) -> pd.DataFrame:
    block_cols = [col for col in BLOCK_COLUMNS if col in data.columns]
    if not block_cols:
        return pd.DataFrame()
    if dates is None:
        dates_use = [data.index.max()]
    else:
        dates_use = []
        idx = pd.DatetimeIndex(data.index)
        for date in dates:
            month = pd.Timestamp(date).to_period("M").to_timestamp("M")
            pos = idx.searchsorted(month, side="right") - 1
            if pos >= 0:
                dates_use.append(idx[pos])
    return data.reindex(pd.DatetimeIndex(dates_use))[block_cols]


__all__ = [
    "BLOCK_COLUMNS",
    "EXTERNAL_BALANCE_PREFIXES",
    "EXTERNAL_DEMAND_PREFIXES",
    "GROWTH_PREFIXES",
    "HOUSING_PREFIXES",
    "INFLATION_PREFIXES",
    "LABOR_CLAIMS_PREFIXES",
    "LABOR_JOBS_PREFIXES",
    "LABOR_UNEMPLOYMENT_PREFIXES",
    "POLICY_PREFIXES",
    "SIGNAL_COLUMNS",
    "SURVEY_PREFIXES",
    "change_z",
    "condition_blocks",
    "expanding_zscore",
    "external_demand_stress",
    "external_trade_block",
    "external_vulnerability",
    "goldilocks_support",
    "growth_acceleration_stress",
    "growth_breadth_stress",
    "growth_momentum_stress",
    "growth_recession_block",
    "housing_domestic_block",
    "housing_impulse_stress",
    "housing_rate_squeeze",
    "inflation_acceleration",
    "inflation_diffusion",
    "inflation_impulse",
    "inflation_level_pressure",
    "inflation_pressure_block",
    "labor_cooling",
    "labor_cooling_block",
    "macro_block_snapshot",
    "macro_breadth_conflict_block",
    "mean_z",
    "policy_catchup_risk",
    "policy_rate_pressure_block",
    "policy_shock",
    "policy_tightness",
    "prefix_columns",
    "real_rate_squeeze",
    "sahm_pressure",
    "severe_stress_breadth",
    "stagflation_pressure",
    "stress_breadth",
    "stress_share",
    "survey_warning",
]
