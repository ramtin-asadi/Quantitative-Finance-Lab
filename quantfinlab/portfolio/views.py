from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PRIMARY_FAMILY_Q_CAPS = {
    "liquid_leadership": 0.042,
    "dual_momentum": 0.022,
    "inflation_rotation": 0.024,
    "risk_adj_momentum": 0.014,
    "credit_switch": 0.010,
    "reflation_breadth": 0.042,
    "growth_duration": 0.032,
    "correlation_stress": 0.024,
    "international_rotation": 0.018,
    "duration_quality": 0.010,
}

DEFAULT_PRIMARY_DISPLAY_NAMES = {
    "liquid_leadership": "Liquidity leadership",
    "dual_momentum": "Dual momentum",
    "inflation_rotation": "Inflation rotation",
    "risk_adj_momentum": "Risk-adjusted momentum",
    "credit_switch": "Credit switch",
    "reflation_breadth": "Reflation breadth",
    "growth_duration": "Growth-duration barbell",
    "correlation_stress": "Correlation stress",
    "international_rotation": "International rotation",
    "duration_quality": "Duration quality",
}

DEFAULT_SECTOR_FAMILY_Q_CAPS = {
    "sector_momentum": 0.012,
    "growth_leadership": 0.030,
    "defensive_rotation": 0.006,
    "cyclical_breadth": 0.020,
    "credit_beta": 0.012,
    "inflation_beneficiaries": 0.008,
    "duration_sensitive": 0.014,
    "small_cap_risk_on": 0.010,
    "quality_defensive": 0.004,
    "sector_reversal": 0.004,
}

DEFAULT_SECTOR_DISPLAY_NAMES = {
    "sector_momentum": "Sector momentum",
    "growth_leadership": "Growth leadership",
    "defensive_rotation": "Defensive rotation",
    "cyclical_breadth": "Cyclical breadth",
    "credit_beta": "Credit beta",
    "inflation_beneficiaries": "Inflation beneficiaries",
    "duration_sensitive": "Duration sensitive",
    "small_cap_risk_on": "Small-cap risk-on",
    "quality_defensive": "Quality defensive",
    "sector_reversal": "Sector reversal",
}


@dataclass
class ViewSettings:
    """Configuration container for signal construction and view generation.

    Attributes
    ----------
    family_q_caps : mapping
        Maximum absolute q tilt by view family.
    family_display_names : mapping
        Human-readable display names by view family.
    assets : sequence of str, optional
        Explicit asset universe. If omitted, assets are inferred from roles and
        signal data.
    annualization : float
        Annualization factor used in signal calculations.
    entry_z : float
        Generic signal-entry threshold used by some view rules.
    q_strength_scale : float
        Scale used when converting signal strength to q tilt.
    min_signal_obs : int
        Minimum observations required before an asset can enter the signal table.
    trend_window : int
        Long trend window, typically 200 trading days.
    short_trend_window : int
        Short trend window, typically 50 trading days.
    medium_window : int
        Medium window used by signal calculations.
    long_window : int
        Long window used by signal calculations.
    view_horizon_days : int
        Horizon used for view payoff evaluation.

    Notes
    -----
    The settings object allows the same view rules to be reused across universes
    while preserving q caps and signal conventions.
    """

    family_q_caps: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_PRIMARY_FAMILY_Q_CAPS))
    family_display_names: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_PRIMARY_DISPLAY_NAMES))
    assets: Sequence[str] | None = None
    annualization: float = 252.0
    entry_z: float = 0.50
    q_strength_scale: float = 1.25
    min_signal_obs: int = 63
    trend_window: int = 200
    short_trend_window: int = 50
    medium_window: int = 63
    long_window: int = 126
    view_horizon_days: int = 21


@dataclass
class View:
    """Structured representation of one relative Black-Litterman view.

    A view states that a basket of long assets should outperform a basket of
    short assets by a q tilt derived from signal strength, subject to confidence
    and selection rules.

    Attributes
    ----------
    view_family : str
        Family identifier used for q caps and reliability learning.
    view_name : str
        Human-readable view name.
    economic_label : str
        Economic description of the view.
    long_assets : list of str
        Assets on the long side of the relative view.
    short_assets : list of str
        Assets on the short side of the relative view.
    signal_value : float
        Raw signed or absolute signal value.
    q_tilt : float
        Active return-spread tilt associated with the view.
    confidence : float
        Optional confidence value.
    raw_strength : float, optional
        Raw strength used for display and scoring.
    economic_priority : float
        Priority score used during view selection.
    confluence_score : float
        Cross-signal confirmation score.
    risk_orientation : str
        Broad risk orientation, such as risk_on, risk_off, or neutral.
    view_state : str
        More specific state label.
    source : str
        Source label for the rule that generated the view.
    diagnostics : mapping
        Additional signal diagnostics.
    p_vector : mapping
        View exposure vector by asset.
    family_display_name : str, optional
        Human-readable family name.

    Methods
    -------
    as_dict()
        Convert the view to the standardized dictionary format used by selection
        and Black-Litterman matrix construction.
    """

    view_family: str
    view_name: str
    economic_label: str
    long_assets: list[str]
    short_assets: list[str]
    signal_value: float
    q_tilt: float
    confidence: float = np.nan
    raw_strength: float | None = None
    economic_priority: float = 0.50
    confluence_score: float = 0.0
    risk_orientation: str = "neutral"
    view_state: str = "neutral"
    source: str = "family_rule"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    p_vector: Mapping[str, float] = field(default_factory=dict)
    family_display_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        display = self.family_display_name or self.view_family.replace("_", " ").title()
        raw = float(abs(self.signal_value if self.raw_strength is None else self.raw_strength))
        return {
            "view_family": self.view_family,
            "family_name": display,
            "family_display_name": display,
            "view_name": self.view_name,
            "view_side": self.view_state,
            "view_state": self.view_state,
            "economic_label": self.economic_label,
            "long_assets": list(self.long_assets),
            "short_assets": list(self.short_assets),
            "signal_value": float(self.signal_value),
            "raw_strength": raw,
            "view_strength": raw,
            "confluence_score": float(self.confluence_score),
            "q_tilt": float(self.q_tilt),
            "q": float(self.q_tilt),
            "risk_orientation": self.risk_orientation,
            "source": self.source,
            "priority": float(self.economic_priority),
            "economic_priority": float(self.economic_priority),
            "diagnostics": clean_diag_value(dict(self.diagnostics)),
            "p_vector": dict(self.p_vector),
        }


def view_rows(views: Sequence[View | Mapping[str, Any] | None]) -> list[dict[str, Any]]:
    """Convert a sequence of view objects or mappings to standardized dictionaries.

    Parameters
    ----------
    views : sequence of View, mapping, or None
        View objects, dictionaries, or missing entries.

    Returns
    -------
    list of dict
        List of view dictionaries. ``None`` entries are skipped.

    Notes
    -----
    This helper makes view-generation functions composable: each rule may return
    a ``View``, a mapping, or ``None``.
    """

    rows: list[dict[str, Any]] = []
    for view in views:
        if view is None:
            continue
        rows.append(view.as_dict() if isinstance(view, View) else dict(view))
    return rows


def clean_diag_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (list, tuple)):
        return [clean_diag_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): clean_diag_value(v) for k, v in value.items()}
    return value


def assets_from_roles(roles: Mapping[str, Any], settings: ViewSettings) -> list[str]:
    if settings.assets is not None:
        return [str(x) for x in settings.assets]
    raw = roles.get("assets", [])
    if raw:
        return [str(x) for x in raw]
    names: list[str] = []
    for value in roles.values():
        if isinstance(value, (list, tuple, set, pd.Index)):
            names.extend(str(x) for x in value)
    return list(dict.fromkeys(names))


def role_assets(roles: Mapping[str, Any], key: str, assets: Sequence[str] | None = None) -> list[str]:
    vals = [str(x) for x in roles.get(key, [])]
    if assets is not None:
        universe = set(str(x) for x in assets)
        vals = [x for x in vals if x in universe]
    return vals


def p_series_from_assets(long_assets: Sequence[str], short_assets: Sequence[str], asset_list: Sequence[str]) -> pd.Series | None:
    p = pd.Series(0.0, index=[str(x) for x in asset_list], dtype=float)
    longs = [str(x) for x in long_assets if str(x) in p.index]
    shorts = [str(x) for x in short_assets if str(x) in p.index and str(x) not in longs]
    if not longs or not shorts:
        return None
    p.loc[longs] = 1.0 / len(longs)
    p.loc[shorts] = -1.0 / len(shorts)
    return p


def q_from_strength(view_family: str, view_strength: float, settings: ViewSettings) -> float:
    cap = float(settings.family_q_caps.get(view_family, 0.020))
    if not np.isfinite(view_strength) or cap <= 0:
        return 0.0
    return float(cap * math.tanh(abs(float(view_strength)) / float(settings.q_strength_scale)))


def make_view(
    view_family: str,
    view_name: str,
    economic_label: str,
    long_assets: Sequence[str],
    short_assets: Sequence[str],
    view_strength: float,
    risk_orientation: str,
    *,
    roles: Mapping[str, Any],
    settings: ViewSettings,
    source: str = "family_rule",
    priority: float = 0.50,
    diagnostics: Mapping[str, Any] | None = None,
    confluence_score: float | None = None,
    view_state: str | None = None,
) -> View | None:
    """Create a validated relative view from long/short asset lists and signal strength.

    The function filters long and short assets to the configured universe,
    checks that the view family has a q cap, converts signal strength into q
    tilt, builds a P-vector, and returns a structured ``View`` object.

    Parameters
    ----------
    view_family : str
        View-family identifier.
    view_name : str
        Human-readable view name.
    economic_label : str
        Economic interpretation of the view.
    long_assets : sequence of str
        Candidate long-side assets.
    short_assets : sequence of str
        Candidate short-side assets.
    view_strength : float
        Raw view strength. Only the absolute value is used.
    risk_orientation : str
        Broad risk orientation.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.
    source : str, default="family_rule"
        Source label.
    priority : float, default=0.50
        Economic priority score in ``[0, 1]``.
    diagnostics : mapping, optional
        Additional signal diagnostics.
    confluence_score : float, optional
        Confirmation score. If omitted, it is inferred from strength.
    view_state : str, optional
        More specific state label.

    Returns
    -------
    View or None
        Validated view object, or ``None`` if the view is invalid, has no valid
        long/short assets, lacks a q cap, or has zero/non-finite strength.

    Notes
    -----
    Short assets that also appear on the long side are removed to avoid
    self-offsetting views.
    """

    assets = assets_from_roles(roles, settings)
    long_clean = list(dict.fromkeys([str(x) for x in long_assets if str(x) in assets]))
    short_clean = list(dict.fromkeys([str(x) for x in short_assets if str(x) in assets and str(x) not in long_clean]))
    if view_family not in settings.family_q_caps or not long_clean or not short_clean or not np.isfinite(view_strength):
        return None
    strength = float(abs(view_strength))
    if strength <= 1e-8:
        return None
    p = p_series_from_assets(long_clean, short_clean, assets)
    if p is None:
        return None
    confluence = float(np.clip(confluence_score if confluence_score is not None else strength / 3.0, 0.0, 1.0))
    display = settings.family_display_names.get(view_family, view_family.replace("_", " ").title())
    state = view_state or risk_orientation
    return View(
        view_family=view_family,
        family_display_name=display,
        view_name=view_name,
        economic_label=economic_label,
        long_assets=long_clean,
        short_assets=short_clean,
        signal_value=strength,
        raw_strength=strength,
        q_tilt=q_from_strength(view_family, strength, settings),
        confluence_score=confluence,
        economic_priority=float(np.clip(priority, 0.0, 1.0)),
        risk_orientation=risk_orientation,
        view_state=state,
        source=source,
        diagnostics=clean_diag_value(diagnostics or {}),
        p_vector=p.round(6).to_dict(),
    )


def winsorized_zscore(series: pd.Series | Sequence[float]) -> pd.Series:
    s = pd.Series(series, dtype=float).replace([np.inf, -np.inf], np.nan)
    good = s.dropna()
    if good.empty:
        return pd.Series(0.0, index=s.index)
    clipped = s.clip(good.quantile(0.05), good.quantile(0.95))
    std = clipped.std(ddof=0)
    if not np.isfinite(std) or std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((clipped - clipped.mean()) / std).fillna(0.0)


def cumulative_return(window: pd.Series | pd.DataFrame | Sequence[float] | None) -> float:
    if window is None or len(window) == 0:
        return np.nan
    s = pd.Series(window, dtype=float).dropna()
    return float((1.0 + s).prod() - 1.0) if len(s) else np.nan


def basket_return(ret_hist: pd.DataFrame, names: Sequence[str], lookback: int = 63) -> float:
    names = [str(x) for x in names if str(x) in ret_hist.columns]
    if not names or len(ret_hist) < lookback:
        return np.nan
    vals = [cumulative_return(ret_hist[x].tail(lookback)) for x in names]
    vals = [x for x in vals if np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def relative_cumulative_return(ret_hist: pd.DataFrame, long_asset: str, short_asset: str, lookback: int = 63) -> float:
    if long_asset not in ret_hist.columns or short_asset not in ret_hist.columns or len(ret_hist) < lookback:
        return np.nan
    long_ret = cumulative_return(ret_hist[long_asset].tail(lookback))
    short_ret = cumulative_return(ret_hist[short_asset].tail(lookback))
    return float(long_ret - short_ret) if np.isfinite(long_ret) and np.isfinite(short_ret) else np.nan


def relative_basket_return(ret_hist: pd.DataFrame, long_assets: Sequence[str], short_assets: Sequence[str], lookback: int = 63) -> float:
    long_ret = basket_return(ret_hist, long_assets, lookback)
    short_ret = basket_return(ret_hist, short_assets, lookback)
    return float(long_ret - short_ret) if np.isfinite(long_ret) and np.isfinite(short_ret) else np.nan


def trailing_volatility(ret_hist: pd.DataFrame, asset: str, lookback: int = 63, annualization: float = 252.0) -> float:
    if asset not in ret_hist.columns or len(ret_hist) < lookback:
        return np.nan
    s = ret_hist[asset].dropna().tail(lookback)
    return float(s.std(ddof=1) * math.sqrt(float(annualization))) if len(s) > 2 else np.nan


def trailing_pair_correlation(ret_hist: pd.DataFrame, asset_a: str, asset_b: str, lookback: int = 126) -> float:
    if asset_a not in ret_hist.columns or asset_b not in ret_hist.columns or len(ret_hist) < lookback:
        return np.nan
    window = ret_hist[[asset_a, asset_b]].tail(lookback).dropna()
    if len(window) < max(30, lookback // 3):
        return np.nan
    if window[asset_a].std(ddof=1) <= 1e-12 or window[asset_b].std(ddof=1) <= 1e-12:
        return np.nan
    corr = float(window[asset_a].corr(window[asset_b]))
    return corr if np.isfinite(corr) else np.nan


def trailing_average_correlation(ret_hist: pd.DataFrame, names: Sequence[str], lookback: int = 126) -> float:
    names = [str(x) for x in names if str(x) in ret_hist.columns]
    if len(names) < 2 or len(ret_hist) < lookback:
        return np.nan
    window = ret_hist[names].tail(lookback).dropna(how="any")
    if len(window) < max(30, lookback // 3):
        return np.nan
    corr = window.corr().replace([np.inf, -np.inf], np.nan)
    vals = corr.where(~np.eye(len(corr), dtype=bool)).stack().dropna()
    return float(vals.mean()) if len(vals) else np.nan


def trend_breadth_from_prices(price_hist: pd.DataFrame, names: Sequence[str], ma_window: int = 200, offset: int = 0) -> float:
    names = [str(x) for x in names if str(x) in price_hist.columns]
    if not names:
        return np.nan
    hist = price_hist.iloc[: len(price_hist) - offset] if offset and len(price_hist) > offset else price_hist
    vals: list[float] = []
    for ticker in names:
        px = hist[ticker].dropna()
        if len(px) < ma_window:
            continue
        ma = px.tail(ma_window).mean()
        if ma > 0:
            vals.append(float(px.iloc[-1] / ma - 1.0))
    return float((pd.Series(vals) > 0).mean()) if vals else np.nan


def signal_table_from_returns(
    signal_returns: pd.DataFrame,
    date: pd.Timestamp | str,
    *,
    roles: Mapping[str, Any] | None = None,
    settings: ViewSettings | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build cross-asset signal table and market-state values from return history.

    The function computes cumulative momentum, trend, volatility, drawdown
    quality, and a composite score for each eligible signal asset using only
    observations available through the decision date.

    Parameters
    ----------
    signal_returns : pandas.DataFrame
        Return panel used to compute signals.
    date : pandas.Timestamp or str
        Decision date.
    roles : mapping, optional
        Asset-role configuration used to assign sleeves and state values.
    settings : ViewSettings, optional
        Signal construction settings.

    Returns
    -------
    signal_table : pandas.DataFrame
        Asset-indexed signal table sorted by composite score.
    values : dict
        Derived market-state values used by view rules.

    Notes
    -----
    Assets with insufficient signal history are skipped. If the date precedes
    the available history, an empty table and empty dictionary are returned.
    """

    settings = settings or ViewSettings()
    roles = roles or {}
    date = pd.Timestamp(date)
    ret = signal_returns.copy()
    ret.index = pd.to_datetime(ret.index)
    ret = ret.sort_index().apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    pos = ret.index.searchsorted(date, side="right") - 1
    if pos < 0:
        return pd.DataFrame(), {}
    ret_hist = ret.iloc[: pos + 1].fillna(0.0)
    price_hist = (1.0 + ret_hist).cumprod()
    sleeve_map = dict(roles.get("sleeve_map", {}))
    rows: list[dict[str, Any]] = []
    for ticker in ret_hist.columns:
        rt = ret_hist[ticker].dropna()
        px = price_hist[ticker].dropna()
        if len(rt) < int(settings.min_signal_obs) or len(px) < int(settings.medium_window):
            continue
        ma50 = px.tail(settings.short_trend_window).mean() if len(px) >= settings.short_trend_window else np.nan
        ma200 = px.tail(settings.trend_window).mean() if len(px) >= settings.trend_window else np.nan
        high_252 = px.tail(252).max()
        rows.append(
            {
                "ticker": ticker,
                "sleeve": sleeve_map.get(ticker, "Other"),
                "mom_1_0": cumulative_return(rt.tail(21)),
                "mom_3_0": cumulative_return(rt.tail(63)),
                "mom_6_1": cumulative_return(rt.iloc[-147:-21]) if len(rt) >= 147 else np.nan,
                "mom_12_1": cumulative_return(rt.iloc[-273:-21]) if len(rt) >= 273 else np.nan,
                "trend_50": float(px.iloc[-1] / ma50 - 1.0) if np.isfinite(ma50) and ma50 > 0 else np.nan,
                "trend_200": float(px.iloc[-1] / ma200 - 1.0) if np.isfinite(ma200) and ma200 > 0 else np.nan,
                "vol_63": trailing_volatility(ret_hist, ticker, 63, settings.annualization),
                "dd_252": float(px.iloc[-1] / high_252 - 1.0) if high_252 > 0 else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
    if table.empty:
        return table, {}
    table["drawdown_quality"] = table["dd_252"]
    table["score"] = (
        0.25 * winsorized_zscore(table["mom_3_0"])
        + 0.25 * winsorized_zscore(table["mom_6_1"])
        + 0.15 * winsorized_zscore(table["mom_12_1"])
        + 0.20 * winsorized_zscore(table["trend_200"])
        - 0.10 * winsorized_zscore(table["vol_63"])
        + 0.05 * winsorized_zscore(table["drawdown_quality"])
    )
    values = state_values(table, ret_hist, price_hist, roles=roles, settings=settings)
    return table.sort_values("score", ascending=False), values


def state_values(
    signal_table: pd.DataFrame,
    ret_hist: pd.DataFrame,
    price_hist: pd.DataFrame | None = None,
    *,
    roles: Mapping[str, Any],
    settings: ViewSettings,
) -> dict[str, Any]:
    scores = signal_table["score"] if "score" in signal_table else pd.Series(dtype=float)
    assets = assets_from_roles(roles, settings)
    risky = role_assets(roles, "risky", signal_table.index) or role_assets(roles, "assets", signal_table.index)
    defensive = role_assets(roles, "defensive", signal_table.index)
    cyclical = role_assets(roles, "cyclical", signal_table.index)
    risky_breadth = float((signal_table.reindex(risky)["trend_200"] > 0).mean()) if risky else np.nan
    cyclical_breadth = float((signal_table.reindex(cyclical)["trend_200"] > 0).mean()) if cyclical else np.nan
    if price_hist is None:
        price_hist = (1.0 + ret_hist.fillna(0.0)).cumprod()
    risky_breadth_ago = trend_breadth_from_prices(price_hist, risky, offset=63)
    cyclical_breadth_ago = trend_breadth_from_prices(price_hist, cyclical, offset=63)
    spy_drawdown = metric(signal_table, "SPY", "dd_252")
    spy_vol = metric(signal_table, "SPY", "vol_63")
    spy_vol_line = ret_hist["SPY"].rolling(63).std() * math.sqrt(float(settings.annualization)) if "SPY" in ret_hist.columns else pd.Series(dtype=float)
    vol_sample = spy_vol_line.dropna().tail(252)
    spy_vol_threshold = vol_sample.quantile(0.80) if not vol_sample.empty else np.nan
    spy_vol_z = float((spy_vol - vol_sample.mean()) / vol_sample.std(ddof=1)) if len(vol_sample) > 20 and vol_sample.std(ddof=1) > 1e-12 and np.isfinite(spy_vol) else np.nan
    spy_tlt_corr = trailing_pair_correlation(ret_hist, "SPY", "TLT", 126)
    spy_ief_corr = trailing_pair_correlation(ret_hist, "SPY", "IEF", 126)
    stock_bond_corr = spy_tlt_corr if np.isfinite(spy_tlt_corr) else spy_ief_corr
    values: dict[str, Any] = {
        "risky_median_score": float(scores.reindex(risky).median()) if risky else np.nan,
        "defensive_median_score": float(scores.reindex(defensive).median()) if defensive else np.nan,
        "risk_on_score": (float(scores.reindex(risky).median()) - float(scores.reindex(defensive).median())) if risky and defensive else np.nan,
        "risky_trend_breadth": risky_breadth,
        "risky_trend_breadth_3m_ago": risky_breadth_ago,
        "risky_trend_breadth_change_63": risky_breadth - risky_breadth_ago if np.isfinite(risky_breadth) and np.isfinite(risky_breadth_ago) else np.nan,
        "cyclical_trend_breadth": cyclical_breadth,
        "cyclical_trend_breadth_3m_ago": cyclical_breadth_ago,
        "cyclical_trend_breadth_change_63": cyclical_breadth - cyclical_breadth_ago if np.isfinite(cyclical_breadth) and np.isfinite(cyclical_breadth_ago) else np.nan,
        "spy_trend_200": metric(signal_table, "SPY", "trend_200"),
        "spy_drawdown_252": spy_drawdown,
        "spy_vol_63": spy_vol,
        "spy_vol_z": spy_vol_z,
        "equity_stress": bool((np.isfinite(spy_drawdown) and spy_drawdown < -0.10) or (np.isfinite(spy_vol) and np.isfinite(spy_vol_threshold) and spy_vol > spy_vol_threshold)),
        "qqq_spy_63": relative_cumulative_return(ret_hist, "QQQ", "SPY", 63),
        "qqq_spy_126": relative_cumulative_return(ret_hist, "QQQ", "SPY", 126),
        "hyg_lqd_63": relative_cumulative_return(ret_hist, "HYG", "LQD", 63),
        "hyg_lqd_126": relative_cumulative_return(ret_hist, "HYG", "LQD", 126),
        "hyg_shy_63": relative_cumulative_return(ret_hist, "HYG", "SHY", 63),
        "iwm_spy_63": relative_cumulative_return(ret_hist, "IWM", "SPY", 63),
        "iwm_spy_126": relative_cumulative_return(ret_hist, "IWM", "SPY", 126),
        "eem_efa_63": relative_cumulative_return(ret_hist, "EEM", "EFA", 63),
        "dbc_ief_63": relative_cumulative_return(ret_hist, "DBC", "IEF", 63),
        "dbc_agg_63": relative_cumulative_return(ret_hist, "DBC", "AGG", 63),
        "dollar_trend": metric(signal_table, "UUP", "trend_200"),
        "dollar_mom_3_0": metric(signal_table, "UUP", "mom_3_0"),
        "spy_tlt_corr_126": spy_tlt_corr,
        "spy_ief_corr_126": spy_ief_corr,
        "stock_bond_corr_126": stock_bond_corr,
        "avg_risky_corr_126": trailing_average_correlation(ret_hist, risky, 126),
        "avg_risky_corr_63": trailing_average_correlation(ret_hist, risky, 63),
        "asset_corr_to_spy_126": {asset: trailing_pair_correlation(ret_hist, asset, "SPY", 126) for asset in signal_table.index if asset != "SPY"},
        "real_asset_score": float(scores.reindex([x for x in ["GLD", "DBC"] if x in scores.index]).median()) if any(x in scores.index for x in ["GLD", "DBC"]) else np.nan,
        "bond_score": float(scores.reindex([x for x in ["TLT", "IEF", "AGG"] if x in scores.index]).median()) if any(x in scores.index for x in ["TLT", "IEF", "AGG"]) else np.nan,
        "sector_avg_corr_126": trailing_average_correlation(ret_hist, assets, 126),
        "sector_avg_corr_63": trailing_average_correlation(ret_hist, assets, 63),
        "sector_21d_dispersion": float(ret_hist[assets].tail(21).apply(cumulative_return).std(ddof=0)) if len([x for x in assets if x in ret_hist.columns]) >= 3 and len(ret_hist) >= 21 else np.nan,
    }
    return values


def _signals(state: Any) -> pd.DataFrame:
    return getattr(state, "signal_table", getattr(state, "signals", pd.DataFrame()))


def _values(state: Any) -> Mapping[str, Any]:
    return getattr(state, "values", getattr(state, "market_state", {}))


def _ret_hist(state: Any) -> pd.DataFrame:
    return getattr(state, "signal_returns", getattr(state, "returns", pd.DataFrame()))


def available(signal_table: pd.DataFrame, names: Sequence[str]) -> list[str]:
    return [str(x) for x in names if str(x) in signal_table.index]


def metric(signal_table: pd.DataFrame, asset: str, col: str, default: float = np.nan) -> float:
    if asset not in signal_table.index or col not in signal_table.columns:
        return default
    val = signal_table.loc[asset, col]
    return float(val) if np.isfinite(val) else default


def state_value(values: Mapping[str, Any], key: str, default: float = np.nan) -> float:
    val = values.get(key, default)
    try:
        return float(val) if np.isfinite(val) else default
    except Exception:
        return default


def score_value(signal_table: pd.DataFrame, asset: str) -> float:
    return metric(signal_table, asset, "score")


def median_score(signal_table: pd.DataFrame, names: Sequence[str]) -> float:
    names = available(signal_table, names)
    return float(signal_table.reindex(names)["score"].median()) if names else np.nan


def robust_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if np.isfinite(x)]
    return float(np.mean(vals)) if vals else np.nan


def clipped(value: float, scale: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> float:
    if not np.isfinite(value) or scale == 0:
        return np.nan
    return float(np.clip(value / scale, lower, upper))


def best_assets(signal_table: pd.DataFrame, candidates: Sequence[str], n: int = 1, require_positive: bool = False) -> list[str]:
    names = available(signal_table, candidates)
    if not names:
        return []
    frame = signal_table.reindex(names).sort_values("score", ascending=False)
    if require_positive:
        frame = frame[(frame["score"] > 0) | (frame["trend_200"] > 0) | (frame["mom_3_0"] > 0)]
    return list(frame.head(n).index)


def worst_assets(signal_table: pd.DataFrame, candidates: Sequence[str], n: int = 1, require_weak: bool = False) -> list[str]:
    names = available(signal_table, candidates)
    if not names:
        return []
    frame = signal_table.reindex(names).sort_values("score", ascending=True)
    if require_weak:
        frame = frame[(frame["score"] < 0) | (frame["trend_200"] < 0) | (frame["mom_3_0"] < 0)]
    return list(frame.head(n).index)


def sleeve_limited_assets(signal_table: pd.DataFrame, candidates: Sequence[str], roles: Mapping[str, Any], n: int = 3, side: str = "long", per_sleeve: int = 1) -> list[str]:
    names = available(signal_table, candidates)
    if not names:
        return []
    sleeve_map = dict(roles.get("sleeve_map", {}))
    frame = signal_table.reindex(names).sort_values("score", ascending=(side == "short"))
    picks: list[str] = []
    sleeve_counts: dict[str, int] = {}
    for asset, row in frame.iterrows():
        sleeve = row.get("sleeve", sleeve_map.get(asset, "Other"))
        if sleeve_counts.get(sleeve, 0) >= per_sleeve:
            continue
        if side == "long" and row.get("trend_200", 0.0) <= 0 and row.get("mom_3_0", 0.0) <= 0:
            continue
        if side == "short" and row.get("trend_200", 0.0) >= 0 and row.get("mom_3_0", 0.0) >= 0:
            continue
        picks.append(str(asset))
        sleeve_counts[sleeve] = sleeve_counts.get(sleeve, 0) + 1
        if len(picks) >= n:
            break
    return picks


def liquid_leadership(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a liquidity-leadership view when large-cap/growth leadership is narrow.

    The rule looks for positive SPY/QQQ trend, QQQ leadership over SPY, weak
    cyclicals, and lack of broad reflation confirmation. When active, it creates
    a risk-on relative view favoring liquid US large-cap/growth leaders versus
    weaker cyclical assets.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Liquidity-leadership view when conditions are met; otherwise ``None``.

    Notes
    -----
    The rule includes a reflation-confirmation filter to avoid expressing narrow
    growth leadership when cyclicals, credit, commodities, and real assets are
    already broadening.
    """

    signal_table, values = _signals(state), _values(state)
    if signal_table.empty:
        return None
    spy_trend = metric(signal_table, "SPY", "trend_200")
    qqq_trend = metric(signal_table, "QQQ", "trend_200")
    cyc_breadth = state_value(values, "cyclical_trend_breadth")
    cyc_breadth_change = state_value(values, "cyclical_trend_breadth_change_63", 0.0)
    qqq_spy = state_value(values, "qqq_spy_63", 0.0)
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    iwm_spy = state_value(values, "iwm_spy_63", 0.0)
    leaders = best_assets(signal_table, ["SPY", "QQQ"], n=2, require_positive=True)
    weak_cyclicals = worst_assets(signal_table, ["IWM", "EEM", "VNQ", "HYG"], n=3, require_weak=True)
    if len(weak_cyclicals) < 2:
        weak_cyclicals = worst_assets(signal_table, ["IWM", "EEM", "VNQ", "HYG"], n=3)
    reflation_confirmed = (
        iwm_spy > 0.015
        and hyg_lqd > 0.010
        and np.isfinite(cyc_breadth)
        and cyc_breadth >= 0.75
        and (metric(signal_table, "DBC", "trend_200") > 0 or metric(signal_table, "VNQ", "trend_200") > 0)
    )
    leadership_spread = median_score(signal_table, leaders) - median_score(signal_table, weak_cyclicals) if leaders and weak_cyclicals else np.nan
    strength = robust_mean([clipped(max(spy_trend, qqq_trend), 0.06), clipped(qqq_spy, 0.035), clipped(leadership_spread, 1.25), clipped(0.55 - cyc_breadth, 0.25), clipped(-cyc_breadth_change, 0.25)])
    if leaders and weak_cyclicals and max(spy_trend, qqq_trend) > 0 and strength > 0.25 and not reflation_confirmed:
        return make_view("liquid_leadership", "Liquidity leadership", "liquid US large-cap and growth leadership versus weak cyclicals", leaders, weak_cyclicals, strength, "risk_on", roles=roles, settings=settings, priority=0.90, diagnostics={"leadership_spread": leadership_spread, "cyclical_breadth": cyc_breadth, "cyclical_breadth_change_63": cyc_breadth_change, "qqq_spy_63": qqq_spy, "reflation_confirmed_filter": reflation_confirmed}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="narrow_leadership")
    return None


def dual_momentum(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a diversified dual-momentum relative view.

    The rule ranks eligible assets using a composite of intermediate and long
    momentum plus trend, then forms long and short baskets across sleeves when
    cross-sectional dispersion is sufficiently strong.

    Parameters
    ----------
    state : object
        Market state with signal table.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Dual-momentum view when dispersion and spread thresholds are met;
        otherwise ``None``.

    Notes
    -----
    The rule prefers sleeve-limited long and short baskets to avoid expressing a
    single crowded asset class as the entire view.
    """

    signal_table = _signals(state)
    candidates = [asset for asset in assets_from_roles(roles, settings) if asset in signal_table.index]
    if len(candidates) < 4:
        return None
    table = signal_table.reindex(candidates).copy()
    table["score"] = 0.40 * winsorized_zscore(table["mom_6_1"]) + 0.25 * winsorized_zscore(table["mom_12_1"]) + 0.20 * winsorized_zscore(table["mom_3_0"]) + 0.15 * winsorized_zscore(table["trend_200"])
    dispersion = float(table["score"].quantile(0.80) - table["score"].quantile(0.20)) if len(table) >= 6 else np.nan
    longs = sleeve_limited_assets(table, candidates, roles, n=4, side="long", per_sleeve=1)
    if len(longs) < 3:
        longs = list(table[(table["trend_200"] > 0) | (table["mom_3_0"] > 0)].sort_values("score", ascending=False).head(4).index)
    shorts = sleeve_limited_assets(table, candidates, roles, n=4, side="short", per_sleeve=1)
    if len(shorts) < 3:
        shorts = list(table[(table["trend_200"] < 0) | (table["mom_3_0"] < 0)].sort_values("score").head(4).index)
    spread = median_score(table, longs) - median_score(table, shorts) if longs and shorts else np.nan
    if longs and shorts and np.isfinite(dispersion) and dispersion > 0.65 and spread > 0.50:
        return make_view("dual_momentum", "Dual momentum", "diversified cross-asset momentum with positive absolute trend preference", longs, shorts, max(spread, dispersion), "neutral", roles=roles, settings=settings, priority=0.85, diagnostics={"dual_dispersion": dispersion, "long_scores": table.reindex(longs)["score"].to_dict(), "short_scores": table.reindex(shorts)["score"].to_dict()}, confluence_score=min(1.0, max(spread, 0.0) / 2.5), view_state="trend_following")
    return None


def inflation_rotation(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate an inflation-rotation view.

    The rule activates when commodities/gold lead duration assets while duration
    is weak and credit confirmation is not strongly risk-on. It expresses a
    relative view favoring inflation-sensitive assets versus duration and credit
    defensives.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Inflation-rotation view when conditions are met; otherwise ``None``.

    Notes
    -----
    The rule uses commodity confirmation, duration weakness, and relative
    commodity-versus-bond momentum as core inputs.
    """

    signal_table, values = _signals(state), _values(state)
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    dbc_ief = state_value(values, "dbc_ief_63", 0.0)
    dbc_agg = state_value(values, "dbc_agg_63", 0.0)
    commodity_score = median_score(signal_table, ["DBC", "GLD"])
    duration_score = median_score(signal_table, ["TLT", "IEF", "AGG"])
    commodity_confirmed = any(metric(signal_table, asset, "trend_200") > 0 and metric(signal_table, asset, "mom_3_0") > 0 for asset in ["DBC", "GLD"] if asset in signal_table.index)
    duration_weak = any(metric(signal_table, asset, "trend_200") < 0 for asset in ["TLT", "IEF", "AGG"] if asset in signal_table.index)
    strength = robust_mean([clipped(max(dbc_ief, dbc_agg), 0.035), clipped(commodity_score - duration_score, 1.25), clipped(-hyg_lqd, 0.04)])
    if commodity_confirmed and duration_weak and hyg_lqd < 0.020 and strength > 0.20:
        longs = best_assets(signal_table, ["DBC", "GLD"], n=2, require_positive=True)
        shorts = worst_assets(signal_table, ["TLT", "IEF", "AGG", "LQD", "HYG", "VNQ"], n=4, require_weak=True)
        if len(shorts) < 3:
            shorts = worst_assets(signal_table, ["TLT", "IEF", "AGG", "LQD"], n=3)
        return make_view("inflation_rotation", "Inflation rotation", "commodities and gold versus duration and credit when inflation pressure dominates", longs, shorts, max(strength, commodity_score - duration_score), "neutral", roles=roles, settings=settings, priority=0.80, diagnostics={"commodity_score": commodity_score, "duration_score": duration_score, "dbc_ief_63": dbc_ief, "dbc_agg_63": dbc_agg, "hyg_lqd_63": hyg_lqd}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="inflation_shock")
    return None


def risk_adjusted_momentum(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a volatility- and drawdown-adjusted momentum view.

    The rule ranks assets using momentum, trend, volatility penalty, and
    drawdown quality. It forms long and short baskets when cross-sectional
    dispersion and long-short score spread are strong enough.

    Parameters
    ----------
    state : object
        Market state with signal table.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Risk-adjusted momentum view when active; otherwise ``None``.

    Notes
    -----
    This rule is designed to prefer momentum that is not purely driven by high
    volatility or poor drawdown quality.
    """

    signal_table = _signals(state)
    candidates = [asset for asset in assets_from_roles(roles, settings) if asset in signal_table.index]
    if len(candidates) < 4:
        return None
    table = signal_table.reindex(candidates).copy()
    table["score"] = 0.30 * winsorized_zscore(table["mom_6_1"]) + 0.25 * winsorized_zscore(table["mom_12_1"]) + 0.15 * winsorized_zscore(table["mom_3_0"]) + 0.15 * winsorized_zscore(table["trend_200"]) - 0.20 * winsorized_zscore(table["vol_63"]) + 0.15 * winsorized_zscore(table["drawdown_quality"])
    dispersion = float(table["score"].quantile(0.80) - table["score"].quantile(0.20)) if len(table) >= 6 else np.nan
    longs = list(table[(table["trend_200"] > 0) | (table["mom_3_0"] > 0)].sort_values("score", ascending=False).head(4).index)
    if len(longs) < 3:
        longs = list(table.sort_values("score", ascending=False).head(4).index)
    weak_mask = (table["trend_200"] < 0) | (table["vol_63"] > table["vol_63"].median()) | (table["dd_252"] < table["dd_252"].median())
    shorts = list(table[weak_mask].sort_values("score").head(4).index)
    if len(shorts) < 3:
        shorts = list(table.sort_values("score").head(4).index)
    spread = median_score(table, longs) - median_score(table, shorts) if longs and shorts else np.nan
    if longs and shorts and np.isfinite(dispersion) and dispersion > 0.55 and spread > 0.50:
        return make_view("risk_adj_momentum", "Risk-adjusted momentum", "momentum adjusted for volatility and drawdown quality", longs, shorts, max(spread, dispersion), "neutral", roles=roles, settings=settings, priority=1.00, diagnostics={"risk_adj_dispersion": dispersion, "long_scores": table.reindex(longs)["score"].to_dict(), "short_scores": table.reindex(shorts)["score"].to_dict()}, confluence_score=min(1.0, max(spread, 0.0) / 2.5), view_state="quality_momentum")
    return None


def credit_switch(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate credit-confirmed risk-on or risk-off views.

    The rule uses equity trend, equity drawdown, volatility state, risky breadth,
    and high-yield-versus-investment-grade credit relative performance. It
    creates either a risk-on view favoring equities/high yield or a risk-off
    view favoring defensive assets.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Credit-switch view when either risk-on or risk-off conditions are met;
        otherwise ``None``.

    Notes
    -----
    The rule treats credit confirmation as a broad risk-appetite signal rather
    than a standalone asset-ranking signal.
    """

    signal_table, values = _signals(state), _values(state)
    spy_trend = metric(signal_table, "SPY", "trend_200")
    spy_drawdown = state_value(values, "spy_drawdown_252", 0.0)
    spy_vol_z = state_value(values, "spy_vol_z", 0.0)
    risky_breadth = state_value(values, "risky_trend_breadth")
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    iwm_spy = state_value(values, "iwm_spy_63", 0.0)
    risk_on = spy_trend > 0 and hyg_lqd > 0.010 and np.isfinite(risky_breadth) and risky_breadth >= 0.55
    risk_off = (spy_trend < 0 or spy_drawdown < -0.08 or spy_vol_z > 0.80) and hyg_lqd < -0.005 and (not np.isfinite(risky_breadth) or risky_breadth < 0.55)
    if risk_on:
        longs = [x for x in ["SPY", "QQQ", "HYG"] if x in signal_table.index]
        if iwm_spy > 0 and metric(signal_table, "IWM", "trend_200") > 0:
            longs.append("IWM")
        shorts = available(signal_table, ["SHY", "IEF", "AGG"])
        strength = robust_mean([clipped(spy_trend, 0.06), clipped(hyg_lqd, 0.025), clipped(risky_breadth - 0.50, 0.30)])
        return make_view("credit_switch", "Credit switch: risk-on", "credit confirmation supports equity and high-yield risk", longs, shorts, strength, "risk_on", roles=roles, settings=settings, priority=0.88, diagnostics={"spy_trend_200": spy_trend, "hyg_lqd_63": hyg_lqd, "risky_breadth": risky_breadth}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="risk_on")
    if risk_off:
        longs = best_assets(signal_table, ["SHY", "IEF", "AGG", "GLD"], n=4)
        shorts = worst_assets(signal_table, ["HYG", "IWM", "EEM", "VNQ"], n=4, require_weak=True)
        if len(shorts) < 3:
            shorts = worst_assets(signal_table, ["HYG", "IWM", "EEM", "VNQ"], n=4)
        strength = robust_mean([clipped(-spy_trend, 0.06), clipped(-hyg_lqd, 0.025), clipped(0.55 - risky_breadth, 0.30), clipped(-spy_drawdown, 0.15)])
        return make_view("credit_switch", "Credit switch: risk-off", "credit rejection shifts toward short-duration, core bonds, and gold", longs, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.88, diagnostics={"spy_trend_200": spy_trend, "spy_drawdown_252": spy_drawdown, "hyg_lqd_63": hyg_lqd, "risky_breadth": risky_breadth}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="risk_off")
    return None


def international_rotation(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate an emerging-market versus developed ex-US rotation view.

    The rule compares EEM and EFA relative strength while accounting for dollar
    trend and momentum. It can express either EM leadership or developed ex-US
    leadership depending on the state.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        International-rotation view when conditions are met; otherwise ``None``.

    Notes
    -----
    Dollar pressure is used as a filter because EM leadership can be fragile
    when the dollar is strengthening.
    """

    signal_table, values = _signals(state), _values(state)
    eem_efa = state_value(values, "eem_efa_63", 0.0)
    dollar_trend = state_value(values, "dollar_trend", 0.0)
    dollar_mom = state_value(values, "dollar_mom_3_0", 0.0)
    eem_trend = metric(signal_table, "EEM", "trend_200")
    efa_trend = metric(signal_table, "EFA", "trend_200")
    score_spread = score_value(signal_table, "EEM") - score_value(signal_table, "EFA")
    if eem_efa > 0.040 and dollar_trend < 0.040 and dollar_mom < 0.030 and eem_trend > -0.02:
        strength = robust_mean([clipped(eem_efa, 0.050), clipped(0.040 - dollar_trend, 0.040), clipped(score_spread, 1.00), clipped(eem_trend, 0.08)])
        return make_view("international_rotation", "International rotation", "EM leadership over developed ex-US when EEM relative strength is strong and the dollar is not a headwind", ["EEM"], ["EFA"], strength, "risk_on", roles=roles, settings=settings, priority=0.72, diagnostics={"eem_efa_63": eem_efa, "dollar_trend": dollar_trend, "dollar_mom_3_0": dollar_mom, "eem_trend_200": eem_trend, "efa_trend_200": efa_trend, "score_spread": score_spread}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="em_leadership")
    if eem_efa < -0.020 and dollar_trend > 0.020 and dollar_mom > 0.0:
        strength = robust_mean([clipped(-eem_efa, 0.040), clipped(dollar_trend, 0.050), clipped(dollar_mom, 0.030), clipped(-score_spread, 1.00)])
        return make_view("international_rotation", "International rotation", "developed ex-US leadership over EM when EEM is weak and dollar pressure is strong", ["EFA"], ["EEM"], strength, "risk_off", roles=roles, settings=settings, priority=0.72, diagnostics={"eem_efa_63": eem_efa, "dollar_trend": dollar_trend, "dollar_mom_3_0": dollar_mom, "eem_trend_200": eem_trend, "efa_trend_200": efa_trend, "score_spread": score_spread}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="developed_ex_us_leadership")
    return None


def reflation_breadth(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a broad reflation view.

    The rule activates when small caps, high yield, risky breadth, and real
    assets confirm a broadening cyclical/reflation environment. It favors
    cyclical, credit, real-asset, and EM exposures versus defensive ballast.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Reflation-breadth view when conditions are met; otherwise ``None``.

    Notes
    -----
    The rule requires multiple confirmations to avoid treating a single cyclical
    asset rally as broad reflation.
    """

    signal_table, values = _signals(state), _values(state)
    iwm_spy = state_value(values, "iwm_spy_63", 0.0)
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    risky_breadth = state_value(values, "risky_trend_breadth")
    spy_drawdown = state_value(values, "spy_drawdown_252", 0.0)
    active = iwm_spy > 0.010 and hyg_lqd > 0.008 and np.isfinite(risky_breadth) and risky_breadth >= 0.55 and (metric(signal_table, "DBC", "trend_200") > 0 or metric(signal_table, "VNQ", "trend_200") > 0) and spy_drawdown > -0.12
    if active:
        longs = [x for x in ["IWM", "HYG", "VNQ", "DBC", "EEM"] if x in signal_table.index and (metric(signal_table, x, "trend_200") > 0 or metric(signal_table, x, "mom_3_0") > 0)]
        if len(longs) < 3:
            longs = best_assets(signal_table, ["IWM", "HYG", "VNQ", "DBC", "EEM"], n=4, require_positive=True)
        shorts = available(signal_table, ["SHY", "IEF", "AGG", "GLD"])
        strength = robust_mean([clipped(iwm_spy, 0.030), clipped(hyg_lqd, 0.020), clipped(risky_breadth - 0.50, 0.25), clipped(max(metric(signal_table, "DBC", "trend_200"), metric(signal_table, "VNQ", "trend_200")), 0.08)])
        return make_view("reflation_breadth", "Reflation breadth", "cyclicals, credit, real assets, and EM broaden versus defensive ballast", longs, shorts, strength, "risk_on", roles=roles, settings=settings, priority=0.78, diagnostics={"iwm_spy_63": iwm_spy, "hyg_lqd_63": hyg_lqd, "risky_breadth": risky_breadth, "spy_drawdown_252": spy_drawdown}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="broad_reflation")
    return None


def growth_duration(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a growth-duration barbell view.

    The rule looks for growth equity leadership, supportive duration trend,
    weak commodity pressure, and hedge-like stock-bond correlation. It favors
    growth equities and duration assets versus weaker cyclical or commodity
    exposures.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Growth-duration view when conditions are met; otherwise ``None``.

    Notes
    -----
    The rule is intended to capture soft-landing-style states where growth and
    duration can work together.
    """

    signal_table, values = _signals(state), _values(state)
    qqq_spy = state_value(values, "qqq_spy_63", 0.0)
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    stock_bond_corr = state_value(values, "stock_bond_corr_126")
    duration_candidates = [x for x in ["IEF", "AGG", "TLT"] if x in signal_table.index and (metric(signal_table, x, "trend_200") > 0 or metric(signal_table, x, "trend_50") > 0)]
    dbc_weak = metric(signal_table, "DBC", "trend_200") < 0 or metric(signal_table, "DBC", "mom_3_0") < 0
    active = qqq_spy > 0.005 and duration_candidates and dbc_weak and ((not np.isfinite(stock_bond_corr)) or stock_bond_corr < 0.25) and hyg_lqd > -0.040
    if active:
        longs = [x for x in ["QQQ", "SPY"] if x in signal_table.index and (metric(signal_table, x, "trend_200") > 0 or metric(signal_table, x, "mom_3_0") > 0)] + duration_candidates[:3]
        shorts = worst_assets(signal_table, ["DBC", "HYG", "IWM", "VNQ"], n=4, require_weak=True)
        if len(shorts) < 2:
            shorts = worst_assets(signal_table, ["DBC", "HYG", "IWM", "VNQ"], n=3)
        strength = robust_mean([clipped(qqq_spy, 0.030), clipped(median_score(signal_table, duration_candidates), 1.00), clipped(-metric(signal_table, "DBC", "trend_200"), 0.08), clipped(0.25 - stock_bond_corr, 0.35) if np.isfinite(stock_bond_corr) else np.nan])
        return make_view("growth_duration", "Growth-duration barbell", "growth equities and duration benefit when commodity pressure fades and bonds hedge", longs, shorts, strength, "risk_on", roles=roles, settings=settings, priority=0.75, diagnostics={"qqq_spy_63": qqq_spy, "duration_candidates": duration_candidates, "dbc_trend_200": metric(signal_table, "DBC", "trend_200"), "stock_bond_corr_126": stock_bond_corr, "hyg_lqd_63": hyg_lqd}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="soft_landing")
    return None


def correlation_stress(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a protective view under risky-asset correlation stress.

    The rule activates when average risky correlations and equity volatility are
    elevated, breadth is weak, and duration is not behaving as a reliable hedge.
    It favors short-duration/gold-style ballast against weak risky assets.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Correlation-stress view when conditions are met; otherwise ``None``.

    Notes
    -----
    This is a protective family. Its learned confidence can be evaluated
    separately in stress states.
    """

    signal_table, values = _signals(state), _values(state)
    avg_risky_corr = state_value(values, "avg_risky_corr_126")
    spy_vol_z = state_value(values, "spy_vol_z", 0.0)
    risky_breadth = state_value(values, "risky_trend_breadth")
    stock_bond_corr = state_value(values, "stock_bond_corr_126")
    duration_not_protective = (np.isfinite(stock_bond_corr) and stock_bond_corr > 0.15) or median_score(signal_table, ["TLT", "IEF", "AGG"]) < 0
    if np.isfinite(avg_risky_corr) and avg_risky_corr > 0.55 and spy_vol_z > 0.50 and (not np.isfinite(risky_breadth) or risky_breadth < 0.55) and duration_not_protective:
        longs = [x for x in ["SHY", "GLD"] if x in signal_table.index]
        if metric(signal_table, "IEF", "trend_200") > 0:
            longs.append("IEF")
        shorts = worst_assets(signal_table, ["IWM", "EEM", "HYG", "VNQ", "DBC", "SPY", "QQQ"], n=5, require_weak=True)
        if len(shorts) < 3:
            shorts = worst_assets(signal_table, ["IWM", "EEM", "HYG", "VNQ", "DBC"], n=5)
        strength = robust_mean([clipped(avg_risky_corr - 0.45, 0.25), clipped(spy_vol_z, 1.5), clipped(0.55 - risky_breadth, 0.30), clipped(stock_bond_corr, 0.35) if np.isfinite(stock_bond_corr) else np.nan])
        return make_view("correlation_stress", "Correlation stress", "protect against rising risky correlations and weak diversification", longs, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.72, diagnostics={"avg_risky_corr_126": avg_risky_corr, "spy_vol_z": spy_vol_z, "risky_breadth": risky_breadth, "stock_bond_corr_126": stock_bond_corr, "duration_not_protective": duration_not_protective}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="correlation_breakdown")
    return None


def duration_quality(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    """Generate a duration-quality view.

    The rule distinguishes between two bond regimes: weak long-duration exposure
    when stock-bond correlation is positive, and helpful duration exposure when
    duration trend is positive and correlation is hedge-like.

    Parameters
    ----------
    state : object
        Market state with signal table and derived values.
    roles : mapping
        Asset-role configuration.
    settings : ViewSettings
        View-generation settings.

    Returns
    -------
    View or None
        Duration-quality view when either duration-weak or duration-helpful
        conditions are met; otherwise ``None``.

    Notes
    -----
    The rule can express either shorter/higher-quality bonds over long duration,
    or long-duration bonds over cash-like bonds and credit, depending on the
    market state.
    """

    signal_table, values = _signals(state), _values(state)
    stock_bond_corr = state_value(values, "stock_bond_corr_126")
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    tlt_trend = metric(signal_table, "TLT", "trend_200")
    ief_trend = metric(signal_table, "IEF", "trend_200")
    agg_trend = metric(signal_table, "AGG", "trend_200")
    duration_pair_score = median_score(signal_table, ["TLT", "IEF"])
    quality_score = median_score(signal_table, ["SHY", "AGG", "IEF"])
    duration_weak = (tlt_trend < -0.015 or duration_pair_score < -0.20) and np.isfinite(stock_bond_corr) and stock_bond_corr > 0.10
    duration_helpful = (tlt_trend > 0.015 or ief_trend > 0.010 or duration_pair_score > 0.20) and ((not np.isfinite(stock_bond_corr)) or stock_bond_corr < 0.05)
    if duration_weak:
        longs = [x for x in ["SHY", "AGG"] if x in signal_table.index]
        if "IEF" in signal_table.index and ief_trend > -0.010:
            longs.append("IEF")
        shorts = ["TLT"] if "TLT" in signal_table.index else []
        if "HYG" in signal_table.index and (hyg_lqd < 0.0 or metric(signal_table, "HYG", "trend_200") < 0):
            shorts.append("HYG")
        strength = robust_mean([clipped(-tlt_trend, 0.060), clipped(stock_bond_corr, 0.30), clipped(quality_score - duration_pair_score, 1.00), clipped(-hyg_lqd, 0.030)])
        return make_view("duration_quality", "Duration quality", "shorter and higher-quality bonds over weak long duration when stock-bond correlation is positive", longs, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.70, diagnostics={"tlt_trend_200": tlt_trend, "ief_trend_200": ief_trend, "agg_trend_200": agg_trend, "stock_bond_corr_126": stock_bond_corr, "hyg_lqd_63": hyg_lqd, "duration_pair_score": duration_pair_score, "quality_score": quality_score}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="short_duration_quality")
    if duration_helpful:
        longs = [x for x in ["TLT", "IEF"] if x in signal_table.index and (metric(signal_table, x, "trend_200") > 0 or metric(signal_table, x, "mom_3_0") > 0)]
        shorts = [x for x in ["SHY", "HYG"] if x in signal_table.index]
        strength = robust_mean([clipped(max(tlt_trend, ief_trend), 0.060), clipped(0.05 - stock_bond_corr, 0.25) if np.isfinite(stock_bond_corr) else np.nan, clipped(duration_pair_score - score_value(signal_table, "SHY"), 1.00), clipped(-hyg_lqd, 0.050)])
        return make_view("duration_quality", "Duration quality", "long-duration bonds over cash-like bonds and credit when duration trend is positive and correlation is hedge-like", longs, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.70, diagnostics={"tlt_trend_200": tlt_trend, "ief_trend_200": ief_trend, "stock_bond_corr_126": stock_bond_corr, "hyg_lqd_63": hyg_lqd, "duration_pair_score": duration_pair_score}, confluence_score=min(1.0, max(strength, 0.0) / 2.0), view_state="duration_hedge")
    return None


PRIMARY_VIEW_FUNCTIONS = (
    liquid_leadership,
    dual_momentum,
    inflation_rotation,
    risk_adjusted_momentum,
    credit_switch,
    reflation_breadth,
    growth_duration,
    correlation_stress,
    international_rotation,
    duration_quality,
)


def sector_momentum(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals = _signals(state)
    sectors = role_assets(roles, "assets", signals.index)
    if len(sectors) < 6:
        return None
    table = signals.reindex(sectors).copy()
    table["sector_score"] = 0.50 * winsorized_zscore(table["mom_3_0"]) + 0.35 * winsorized_zscore(table["mom_6_1"]) - 0.15 * winsorized_zscore(table["vol_63"])
    longs = list(table.sort_values("sector_score", ascending=False).head(3).index)
    shorts = list(table.sort_values("sector_score").head(3).index)
    strength = float(table.reindex(longs)["sector_score"].mean() - table.reindex(shorts)["sector_score"].mean())
    score_dispersion = float(table["sector_score"].max() - table["sector_score"].min())
    min_score_spread = max(settings.entry_z * 4.0, 2.00)
    min_score_dispersion = max(settings.entry_z * 5.6, 2.80)
    if strength > min_score_spread or score_dispersion > min_score_dispersion:
        return make_view("sector_momentum", "Sector momentum", "cross-sectional sector relative strength", longs, shorts, strength, "neutral", roles=roles, settings=settings, priority=0.90, diagnostics={"score_spread": strength, "score_dispersion": score_dispersion, "min_score_spread": min_score_spread, "min_score_dispersion": min_score_dispersion, "long_scores": table.reindex(longs)["sector_score"].to_dict(), "short_scores": table.reindex(shorts)["sector_score"].to_dict()}, confluence_score=min(1.0, max(strength / 2.5, score_dispersion / 3.5)), view_state="relative_strength")
    return None


def growth_leadership(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    growth = role_assets(roles, "growth", signals.index)
    defensive = role_assets(roles, "defensive", signals.index)
    qqq_spy = state_value(values, "qqq_spy_63", 0.0)
    growth_def = relative_basket_return(returns, growth, defensive, 63)
    xlk_xlp = relative_cumulative_return(returns, "XLK", "XLP", 63)
    strength = robust_mean([clipped(qqq_spy, 0.035), clipped(growth_def, 0.045), clipped(xlk_xlp, 0.050)])
    if growth and defensive and strength > settings.entry_z:
        return make_view("growth_leadership", "Growth leadership", "QQQ and growth sectors lead broad market and defensives", growth, defensive, strength, "risk_on", roles=roles, settings=settings, priority=0.82, diagnostics={"qqq_spy_63": qqq_spy, "growth_vs_defensive_63": growth_def, "xlk_xlp_63": xlk_xlp}, confluence_score=min(1.0, strength / 2.0), view_state="growth_leadership")
    return None


def defensive_rotation(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    defensive = role_assets(roles, "defensive", signals.index)
    growth = role_assets(roles, "growth", signals.index)
    cyclical = role_assets(roles, "cyclical", signals.index)
    spy_63 = metric(signals, "SPY", "mom_3_0")
    spy_dd = state_value(values, "spy_drawdown_252", 0.0)
    def_vs_growth = relative_basket_return(returns, defensive, growth, 63)
    def_vs_cyc = relative_basket_return(returns, defensive, cyclical, 63)
    stress = spy_63 < -0.02 or spy_dd < -0.08 or def_vs_growth > 0.015 or def_vs_cyc > 0.015
    strength = robust_mean([clipped(-spy_63, 0.06), clipped(-spy_dd, 0.12), clipped(def_vs_growth, 0.04), clipped(def_vs_cyc, 0.04)])
    weak_cyclicals = worst_assets(signals, growth + cyclical, n=3)
    shorts = list(dict.fromkeys(growth + weak_cyclicals))[:4]
    if defensive and shorts and stress and strength > 0.35:
        return make_view("defensive_rotation", "Defensive rotation", "defensive sectors lead during equity trend and drawdown stress", defensive, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.80, diagnostics={"spy_63": spy_63, "spy_drawdown_252": spy_dd, "defensive_vs_growth_63": def_vs_growth, "defensive_vs_cyclical_63": def_vs_cyc}, confluence_score=min(1.0, strength / 2.0), view_state="risk_off_defensive")
    return None


def cyclical_breadth(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns = _signals(state), _ret_hist(state)
    cyclical = role_assets(roles, "cyclical", signals.index)
    defensive = role_assets(roles, "defensive", signals.index)
    if not cyclical or not defensive:
        return None
    breadth = float((signals.reindex(cyclical)["mom_3_0"] > 0).mean())
    cyc_def_63 = relative_basket_return(returns, cyclical, defensive, 63)
    cyc_def_126 = relative_basket_return(returns, cyclical, defensive, 126)
    spy_trend = metric(signals, "SPY", "mom_3_0")
    strength = robust_mean([clipped(breadth - 0.50, 0.30), clipped(cyc_def_63, 0.045), clipped(cyc_def_126, 0.065), clipped(spy_trend, 0.06)])
    if breadth >= 0.60 and spy_trend > 0 and strength > 0.35:
        return make_view("cyclical_breadth", "Cyclical breadth", "cyclical sectors lead as risk appetite and participation broaden", cyclical, defensive, strength, "risk_on", roles=roles, settings=settings, priority=0.78, diagnostics={"cyclical_positive_63_share": breadth, "cyclical_vs_defensive_63": cyc_def_63, "cyclical_vs_defensive_126": cyc_def_126, "spy_mom_3_0": spy_trend}, confluence_score=min(1.0, strength / 2.0), view_state="cyclical_risk_on")
    if breadth <= 0.20 and cyc_def_63 < -0.050 and strength < -settings.entry_z:
        return make_view("cyclical_breadth", "Cyclical breadth: negative", "cyclical participation breaks down versus defensive sectors", defensive, cyclical, abs(strength), "risk_off", roles=roles, settings=settings, priority=0.65, diagnostics={"cyclical_positive_63_share": breadth, "cyclical_vs_defensive_63": cyc_def_63, "cyclical_vs_defensive_126": cyc_def_126}, confluence_score=min(1.0, abs(strength) / 2.0), view_state="cyclical_breakdown")
    return None


def credit_beta(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, values = _signals(state), _values(state)
    credit_sensitive = role_assets(roles, "credit_sensitive", signals.index)
    defensive = role_assets(roles, "defensive", signals.index)
    hyg_lqd = state_value(values, "hyg_lqd_63", 0.0)
    hyg_shy = state_value(values, "hyg_shy_63", 0.0)
    hyg_abs = metric(signals, "HYG", "mom_3_0")
    strength = robust_mean([clipped(hyg_lqd, 0.025), clipped(hyg_shy, 0.035), clipped(hyg_abs, 0.040)])
    if credit_sensitive and defensive and strength > 0.35:
        return make_view("credit_beta", "Credit beta", "credit-sensitive sectors benefit when high yield confirms risk appetite", credit_sensitive, defensive, strength, "risk_on", roles=roles, settings=settings, priority=0.70, diagnostics={"hyg_lqd_63": hyg_lqd, "hyg_shy_63": hyg_shy, "hyg_mom_3_0": hyg_abs}, confluence_score=min(1.0, strength / 2.0), view_state="credit_risk_on")
    if credit_sensitive and defensive and strength < -0.65:
        return make_view("credit_beta", "Credit beta: risk-off", "credit-sensitive sectors lag when high yield breaks down", defensive, credit_sensitive, abs(strength), "risk_off", roles=roles, settings=settings, priority=0.62, diagnostics={"hyg_lqd_63": hyg_lqd, "hyg_shy_63": hyg_shy, "hyg_mom_3_0": hyg_abs}, confluence_score=min(1.0, abs(strength) / 2.0), view_state="credit_risk_off")
    return None


def inflation_beneficiaries(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    beneficiaries = role_assets(roles, "inflation_beneficiaries", signals.index)
    growth = [x for x in ["XLK", "XLY"] if x in signals.index] or role_assets(roles, "growth", signals.index)
    dbc_63 = metric(signals, "DBC", "mom_3_0")
    dbc_126 = metric(signals, "DBC", "mom_6_1")
    gld_63 = metric(signals, "GLD", "mom_3_0")
    beneficiary_spread = relative_basket_return(returns, beneficiaries, growth, 63)
    dollar_mom = state_value(values, "dollar_mom_3_0", 0.0)
    strength = robust_mean([clipped(dbc_63, 0.070), clipped(dbc_126, 0.100), clipped(gld_63, 0.070), clipped(beneficiary_spread, 0.055), clipped(-max(dollar_mom - 0.03, 0.0), 0.05)])
    if beneficiaries and growth and strength > 0.35:
        return make_view("inflation_beneficiaries", "Inflation beneficiaries", "energy and materials lead growth sectors when commodity pressure rises", beneficiaries, growth, strength, "neutral", roles=roles, settings=settings, priority=0.76, diagnostics={"dbc_mom_3_0": dbc_63, "dbc_mom_6_1": dbc_126, "gld_mom_3_0": gld_63, "beneficiary_vs_growth_63": beneficiary_spread, "dollar_mom_3_0": dollar_mom}, confluence_score=min(1.0, strength / 2.0), view_state="inflation_pressure")
    if beneficiaries and growth and strength < -0.85:
        return make_view("inflation_beneficiaries", "Inflation beneficiaries: disinflation", "growth sectors recover versus energy and materials when commodity pressure fades", growth, beneficiaries, abs(strength), "risk_on", roles=roles, settings=settings, priority=0.58, diagnostics={"dbc_mom_3_0": dbc_63, "dbc_mom_6_1": dbc_126, "gld_mom_3_0": gld_63, "beneficiary_vs_growth_63": beneficiary_spread}, confluence_score=min(1.0, abs(strength) / 2.0), view_state="disinflation")
    return None


def duration_sensitive(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns = _signals(state), _ret_hist(state)
    rate_sensitive = role_assets(roles, "rate_sensitive", signals.index)
    shorts = [x for x in ["XLF", "XLE"] if x in signals.index]
    tlt_shy = relative_cumulative_return(returns, "TLT", "SHY", 63)
    ief_shy = relative_cumulative_return(returns, "IEF", "SHY", 63)
    rs_vs_short = relative_basket_return(returns, rate_sensitive, shorts, 63)
    strength = robust_mean([clipped(tlt_shy, 0.040), clipped(ief_shy, 0.025), clipped(rs_vs_short, 0.045)])
    if rate_sensitive and shorts and strength > 0.40:
        return make_view("duration_sensitive", "Duration sensitive", "rate-sensitive sectors benefit when duration rallies", rate_sensitive, shorts, strength, "risk_off", roles=roles, settings=settings, priority=0.66, diagnostics={"tlt_shy_63": tlt_shy, "ief_shy_63": ief_shy, "rate_sensitive_vs_xlf_xle_63": rs_vs_short}, confluence_score=min(1.0, strength / 2.0), view_state="duration_rally")
    if rate_sensitive and shorts and strength < -0.75:
        return make_view("duration_sensitive", "Duration sensitive: selloff", "financials and energy lead rate-sensitive sectors when duration sells off", shorts, rate_sensitive, abs(strength), "risk_on", roles=roles, settings=settings, priority=0.58, diagnostics={"tlt_shy_63": tlt_shy, "ief_shy_63": ief_shy, "rate_sensitive_vs_xlf_xle_63": rs_vs_short}, confluence_score=min(1.0, abs(strength) / 2.0), view_state="duration_selloff")
    return None


def small_cap_risk_on(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    small_cap_sensitive = role_assets(roles, "small_cap_sensitive", signals.index)
    defensive = role_assets(roles, "defensive", signals.index)
    iwm_spy = state_value(values, "iwm_spy_63", 0.0)
    iwm_abs = metric(signals, "IWM", "mom_3_0")
    small_vs_def = relative_basket_return(returns, small_cap_sensitive, defensive, 63)
    strength = robust_mean([clipped(iwm_spy, 0.035), clipped(iwm_abs, 0.060), clipped(small_vs_def, 0.050)])
    if small_cap_sensitive and defensive and strength > 0.40:
        return make_view("small_cap_risk_on", "Small-cap risk-on", "IWM leadership supports economically sensitive domestic sectors", small_cap_sensitive, defensive, strength, "risk_on", roles=roles, settings=settings, priority=0.70, diagnostics={"iwm_spy_63": iwm_spy, "iwm_mom_3_0": iwm_abs, "small_cap_sensitive_vs_defensive_63": small_vs_def}, confluence_score=min(1.0, strength / 2.0), view_state="domestic_risk_on")
    if small_cap_sensitive and defensive and strength < -0.75:
        return make_view("small_cap_risk_on", "Small-cap risk-off", "small-cap weakness favors defensive sectors", defensive, small_cap_sensitive, abs(strength), "risk_off", roles=roles, settings=settings, priority=0.58, diagnostics={"iwm_spy_63": iwm_spy, "iwm_mom_3_0": iwm_abs, "small_cap_sensitive_vs_defensive_63": small_vs_def}, confluence_score=min(1.0, abs(strength) / 2.0), view_state="domestic_risk_off")
    return None


def quality_defensive(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    quality = role_assets(roles, "quality_defensive", signals.index)
    high_beta = [x for x in ["XLE", "XLF", "XLY"] if x in signals.index]
    spy_vol_z = state_value(values, "spy_vol_z", 0.0)
    avg_corr = state_value(values, "sector_avg_corr_126", np.nan)
    q_vs_beta = relative_basket_return(returns, quality, high_beta, 63)
    strength = robust_mean([clipped(spy_vol_z, 1.2), clipped((avg_corr - 0.45) if np.isfinite(avg_corr) else np.nan, 0.25), clipped(q_vs_beta, 0.045)])
    if quality and high_beta and strength > 0.35:
        weak_high_beta = worst_assets(signals, high_beta, n=min(3, len(high_beta)))
        return make_view("quality_defensive", "Quality defensive", "quality defensive sectors lead under volatility and correlation stress", quality, weak_high_beta or high_beta, strength, "risk_off", roles=roles, settings=settings, priority=0.68, diagnostics={"spy_vol_z": spy_vol_z, "sector_avg_corr_126": avg_corr, "quality_vs_high_beta_63": q_vs_beta}, confluence_score=min(1.0, strength / 2.0), view_state="quality_stress")
    return None


def sector_reversal(state: Any, roles: Mapping[str, Any], settings: ViewSettings) -> View | None:
    signals, returns, values = _signals(state), _ret_hist(state), _values(state)
    sectors = role_assets(roles, "assets", signals.index)
    sectors = [x for x in sectors if x in returns.columns]
    if len(sectors) < 6 or len(returns) < 126:
        return None
    r21 = returns[sectors].tail(21).apply(cumulative_return)
    r126 = returns[sectors].tail(126).apply(cumulative_return)
    dispersion = float(r21.std(ddof=0))
    hist_disp = values.get("sector_21d_dispersion", dispersion)
    if not np.isfinite(dispersion) or dispersion < 0.045:
        return None
    laggards = list(r21[(r126 > -0.10)].sort_values().head(2).index)
    winners = list(r21.sort_values(ascending=False).head(2).index)
    strength = robust_mean([clipped(dispersion, 0.060), clipped(float(r21.reindex(winners).mean() - r21.reindex(laggards).mean()), 0.080)])
    if len(laggards) >= 2 and len(winners) >= 2 and strength > 0.45:
        return make_view("sector_reversal", "Sector reversal", "short-term sector extremes mean-revert after high dispersion", laggards, winners, strength, "neutral", roles=roles, settings=settings, priority=0.45, diagnostics={"sector_21d_dispersion": dispersion, "state_sector_21d_dispersion": hist_disp, "laggard_21d": r21.reindex(laggards).to_dict(), "winner_21d": r21.reindex(winners).to_dict(), "laggard_126d": r126.reindex(laggards).to_dict()}, confluence_score=min(1.0, strength / 2.0), view_state="short_horizon_reversal")
    return None


SECTOR_VIEW_FUNCTIONS = (
    sector_momentum,
    growth_leadership,
    defensive_rotation,
    cyclical_breadth,
    credit_beta,
    inflation_beneficiaries,
    duration_sensitive,
    small_cap_risk_on,
    quality_defensive,
    sector_reversal,
)


SECTOR_VIEW_SPECS = [
    {"family": "sector_momentum", "function": "sector_momentum", "economic_idea": "Cross-sectional relative sector strength.", "typical_long": "Top 3 sectors by momentum/vol score", "typical_short": "Bottom 3 sectors by momentum/vol score", "main_signal": "63d/126d sector momentum minus volatility"},
    {"family": "growth_leadership", "function": "growth_leadership", "economic_idea": "Growth sectors lead when QQQ and growth baskets outperform.", "typical_long": "XLK, XLC, XLY", "typical_short": "XLP, XLU, XLV", "main_signal": "QQQ/SPY and growth/defensive relative momentum"},
    {"family": "defensive_rotation", "function": "defensive_rotation", "economic_idea": "Defensive sectors outperform in equity stress.", "typical_long": "XLP, XLU, XLV", "typical_short": "Growth sectors and weak cyclicals", "main_signal": "SPY drawdown, SPY momentum, defensive relative momentum"},
    {"family": "cyclical_breadth", "function": "cyclical_breadth", "economic_idea": "Cyclicals lead when participation broadens.", "typical_long": "XLI, XLF, XLB, XLE", "typical_short": "XLP, XLU, XLV", "main_signal": "Cyclical breadth and cyclical/defensive relative strength"},
    {"family": "credit_beta", "function": "credit_beta", "economic_idea": "Credit-sensitive sectors benefit when high-yield credit confirms risk appetite.", "typical_long": "XLF, XLY, XLI", "typical_short": "XLV, XLP, XLU", "main_signal": "HYG/LQD, HYG/SHY, HYG momentum"},
    {"family": "inflation_beneficiaries", "function": "inflation_beneficiaries", "economic_idea": "Energy and materials benefit from commodity/inflation pressure.", "typical_long": "XLE, XLB", "typical_short": "XLK, XLY", "main_signal": "DBC/GLD momentum and inflation basket leadership"},
    {"family": "duration_sensitive", "function": "duration_sensitive", "economic_idea": "Rate-sensitive sectors respond to duration regimes.", "typical_long": "XLU, XLV, XLRE", "typical_short": "XLF, XLE", "main_signal": "TLT/SHY, IEF/SHY, rate-sensitive relative strength"},
    {"family": "small_cap_risk_on", "function": "small_cap_risk_on", "economic_idea": "IWM/SPY leadership supports domestic cyclicals.", "typical_long": "XLI, XLF, XLY", "typical_short": "XLP, XLU, XLV", "main_signal": "IWM/SPY and small-cap-sensitive leadership"},
    {"family": "quality_defensive", "function": "quality_defensive", "economic_idea": "Quality defensive sectors lead under volatility and correlation stress.", "typical_long": "XLV, XLP", "typical_short": "XLE, XLF, XLY", "main_signal": "SPY volatility z-score and sector correlation"},
    {"family": "sector_reversal", "function": "sector_reversal", "economic_idea": "Short-term sector extremes mean-revert after high dispersion.", "typical_long": "Short-term laggards with acceptable 126d trend", "typical_short": "Overextended 21d winners", "main_signal": "21d sector dispersion and 126d trend filter"},
]


__all__ = [
    "DEFAULT_PRIMARY_DISPLAY_NAMES",
    "DEFAULT_PRIMARY_FAMILY_Q_CAPS",
    "DEFAULT_SECTOR_DISPLAY_NAMES",
    "DEFAULT_SECTOR_FAMILY_Q_CAPS",
    "PRIMARY_VIEW_FUNCTIONS",
    "SECTOR_VIEW_FUNCTIONS",
    "SECTOR_VIEW_SPECS",
    "View",
    "ViewSettings",
    "assets_from_roles",
    "available",
    "basket_return",
    "best_assets",
    "clean_diag_value",
    "clipped",
    "correlation_stress",
    "credit_beta",
    "credit_switch",
    "cumulative_return",
    "cyclical_breadth",
    "defensive_rotation",
    "dual_momentum",
    "duration_quality",
    "duration_sensitive",
    "growth_duration",
    "growth_leadership",
    "inflation_beneficiaries",
    "inflation_rotation",
    "international_rotation",
    "liquid_leadership",
    "make_view",
    "median_score",
    "metric",
    "p_series_from_assets",
    "q_from_strength",
    "quality_defensive",
    "relative_basket_return",
    "relative_cumulative_return",
    "reflation_breadth",
    "risk_adjusted_momentum",
    "role_assets",
    "sector_momentum",
    "sector_reversal",
    "signal_table_from_returns",
    "small_cap_risk_on",
    "state_value",
    "state_values",
    "trailing_average_correlation",
    "trailing_pair_correlation",
    "trailing_volatility",
    "trend_breadth_from_prices",
    "view_rows",
    "winsorized_zscore",
    "worst_assets",
]
