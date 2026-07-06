from __future__ import annotations

import numpy as np
import pandas as pd


def option_position_greeks(
    row: pd.Series | dict,
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> pd.Series:
    """Scale a single option row's Greeks by position quantity and contract multiplier.

    Parameters
    ----------
    row : pandas.Series or dict
        Option row containing Greek columns such as ``delta``, ``gamma``, ``vega``,
        ``theta``, and ``rho``. ``<greek>_mid`` columns are used as fallbacks.
    quantity : float, default=1.0
        Number of contracts or option units.
    multiplier : float, default=1.0
        Contract multiplier.

    Returns
    -------
    pandas.Series
        Series with ``delta_exposure``, ``gamma_exposure``, ``vega_exposure``,
        ``theta_exposure``, and ``rho_exposure``.
    """

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
    """Aggregate Greek exposures across an option-position table.

    Parameters
    ----------
    positions : pandas.DataFrame
        Position table containing Greek values and quantity/multiplier columns.
    quantity_col : str, default='quantity'
        Quantity column.
    multiplier_col : str, default='multiplier'
        Contract multiplier column. If missing, a multiplier of one is used.

    Returns
    -------
    pandas.Series
        Portfolio-level Greek exposure totals.
    """

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
    """Compute hedge units required to neutralize delta exposure.

    Parameters
    ----------
    delta_exposure : float
        Current portfolio delta exposure.
    hedge_delta : float, default=1.0
        Delta per hedge unit.

    Returns
    -------
    float
        Hedge quantity ``-delta_exposure / hedge_delta``. Returns ``nan`` when hedge
        delta is effectively zero.
    """

    hedge_delta = float(hedge_delta)
    if abs(hedge_delta) < 1e-12:
        return np.nan
    return -float(delta_exposure) / hedge_delta


def target_vega_hedge(vega_exposure: float, hedge_vega: float) -> float:
    """Compute hedge units required to neutralize vega exposure.

    Parameters
    ----------
    vega_exposure : float
        Current portfolio vega exposure.
    hedge_vega : float
        Vega per hedge instrument.

    Returns
    -------
    float
        Hedge quantity ``-vega_exposure / hedge_vega``. Returns ``nan`` when hedge
        vega is effectively zero.
    """

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
    """Add target underlying hedge units for delta-neutral option hedging.

    Parameters
    ----------
    greek_table : pandas.DataFrame
        Table containing option deltas.
    option_quantity : float, default=1.0
        Option position quantity.
    multiplier : float, default=1.0
        Contract multiplier.
    delta_col : str, default='delta'
        Delta column used for exposure calculation.

    Returns
    -------
    pandas.DataFrame
        Copy of ``greek_table`` with ``option_delta_exposure`` and
        ``target_underlying_units`` columns.
    """

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
    """Build combined delta and vega hedge targets from option and hedge-instrument Greeks.

    Parameters
    ----------
    greek_table : pandas.DataFrame
        Option Greek table containing ``delta`` and ``vega``.
    vega_hedge_table : pandas.DataFrame
        Hedge-instrument table with date, vega, and delta columns.
    option_quantity : float, default=1.0
        Option position quantity.
    multiplier : float, default=1.0
        Contract multiplier applied to both option and hedge exposures.

    Returns
    -------
    pandas.DataFrame
        Table with target vega-hedge contracts and residual target underlying units.
    """

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
    """Compute a hedge trade only when exposure breaches a tolerance band.

    Parameters
    ----------
    current_units : float
        Current hedge units.
    target_units : float
        Desired hedge units.
    exposure : float
        Current net exposure used for the band test.
    band : float
        Absolute exposure tolerance.

    Returns
    -------
    float
        Trade size ``target_units - current_units`` when the exposure exceeds the
        band; otherwise zero. Returns zero for non-finite targets.
    """

    if not np.isfinite(target_units):
        return 0.0
    if abs(float(exposure)) <= float(band):
        return 0.0
    return float(target_units) - float(current_units)


def hedge_exposure_table(
    greek_table: pd.DataFrame,
    underlying_units_col: str = "underlying_units",
) -> pd.DataFrame:
    """Attach net delta and vega exposure columns to a Greek table.

    Parameters
    ----------
    greek_table : pandas.DataFrame
        Table containing option Greeks and optional underlying hedge units.
    underlying_units_col : str, default='underlying_units'
        Column containing underlying hedge units.

    Returns
    -------
    pandas.DataFrame
        Copy of ``greek_table`` with ``net_delta_exposure`` and
        ``net_vega_exposure`` columns.
    """

    out = greek_table.copy()
    under = out[underlying_units_col] if underlying_units_col in out.columns else 0.0
    out["net_delta_exposure"] = out.get("delta", 0.0) + under
    out["net_vega_exposure"] = out.get("vega", 0.0)
    return out


def hedging_summary_table(results: dict) -> pd.DataFrame:
    """Extract a hedging summary table from a results dictionary.

    Parameters
    ----------
    results : dict
        Results dictionary that may contain a ``summary`` DataFrame.

    Returns
    -------
    pandas.DataFrame
        The stored summary table, or an empty DataFrame when no summary is present.
    """

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
