from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from quantfinlab.portfolio import views

PRIMARY_ASSETS = [
    "SPY",
    "QQQ",
    "IWM",
    "EEM",
    "EFA",
    "VNQ",
    "HYG",
    "LQD",
    "SHY",
    "IEF",
    "AGG",
    "TLT",
    "DBC",
    "GLD",
    "UUP",
]

SECTOR_ASSETS = [
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLC",
    "XLY",
    "XLP",
    "XLU",
    "XLV",
    "XLRE",
]

SECTOR_CONTEXT = ["SPY", "QQQ", "IWM", "HYG", "LQD", "SHY", "TLT", "IEF", "DBC", "GLD", "UUP"]


def _signal_table(asset_scores: dict[str, float]) -> pd.DataFrame:
    rows = []
    sleeve_map = {
        "SPY": "Equity",
        "QQQ": "Growth",
        "IWM": "Cyclical",
        "EEM": "International",
        "EFA": "International",
        "VNQ": "Real Assets",
        "HYG": "Credit",
        "LQD": "Credit",
        "SHY": "Rates",
        "IEF": "Rates",
        "AGG": "Rates",
        "TLT": "Rates",
        "DBC": "Commodities",
        "GLD": "Gold",
        "UUP": "Dollar",
        "XLB": "Cyclical",
        "XLE": "Cyclical",
        "XLF": "Cyclical",
        "XLI": "Cyclical",
        "XLK": "Growth",
        "XLC": "Growth",
        "XLY": "Growth",
        "XLP": "Defensive",
        "XLU": "Defensive",
        "XLV": "Defensive",
        "XLRE": "Rate Sensitive",
    }
    for asset, score in asset_scores.items():
        rows.append(
            {
                "ticker": asset,
                "sleeve": sleeve_map.get(asset, "Other"),
                "score": float(score),
                "mom_1_0": 0.012 * score,
                "mom_3_0": 0.045 * score,
                "mom_6_1": 0.075 * score,
                "mom_12_1": 0.105 * score,
                "trend_50": 0.035 * score,
                "trend_200": 0.060 * score,
                "vol_63": 0.16 - 0.010 * score,
                "dd_252": -0.035 + 0.020 * score,
                "drawdown_quality": -0.035 + 0.020 * score,
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def _constant_returns(asset_means: dict[str, float], n: int = 160) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    data = {}
    wave = 0.00015 * np.sin(np.linspace(0.0, 6.0, n))
    for i, asset in enumerate(asset_means):
        data[asset] = float(asset_means[asset]) + wave * (1.0 + 0.05 * i)
    return pd.DataFrame(data, index=idx)


def _primary_state(
    scores: dict[str, float],
    values: dict[str, float],
    *,
    returns: pd.DataFrame | None = None,
) -> SimpleNamespace:
    signal_table = _signal_table(scores)
    return SimpleNamespace(
        signal_table=signal_table,
        signals=signal_table,
        values=values,
        market_state=values,
        signal_returns=returns if returns is not None else _constant_returns({asset: 0.0001 for asset in signal_table.index}),
        returns=returns if returns is not None else _constant_returns({asset: 0.0001 for asset in signal_table.index}),
    )


def _base_primary_scores() -> dict[str, float]:
    return {
        "SPY": 1.0,
        "QQQ": 1.5,
        "IWM": -0.9,
        "EEM": -0.7,
        "EFA": -0.3,
        "VNQ": -0.8,
        "HYG": -0.5,
        "LQD": -0.2,
        "SHY": 0.1,
        "IEF": 0.2,
        "AGG": 0.0,
        "TLT": -0.4,
        "DBC": 1.1,
        "GLD": 0.9,
        "UUP": -0.6,
    }


def _primary_roles() -> dict[str, object]:
    return {
        "assets": PRIMARY_ASSETS,
        "risky": ["SPY", "QQQ", "IWM", "EEM", "VNQ", "HYG"],
        "defensive": ["SHY", "IEF", "AGG", "TLT", "GLD"],
        "cyclical": ["IWM", "EEM", "VNQ", "HYG", "DBC"],
        "sleeve_map": {asset: asset for asset in PRIMARY_ASSETS},
    }


def _assert_view(view: views.View | None, expected_family: str) -> None:
    assert view is not None
    row = view.as_dict()
    assert row["view_family"] == expected_family
    assert row["q_tilt"] > 0.0
    assert row["long_assets"]
    assert row["short_assets"]
    assert abs(sum(row["p_vector"].values())) < 2e-6


def test_primary_view_rules_emit_views_for_distinct_market_regimes() -> None:
    settings = views.ViewSettings(assets=PRIMARY_ASSETS)
    roles = _primary_roles()
    base = _base_primary_scores()

    scenarios = [
        (
            views.liquid_leadership,
            _primary_state(
                base,
                {
                    "cyclical_trend_breadth": 0.25,
                    "cyclical_trend_breadth_change_63": -0.20,
                    "qqq_spy_63": 0.045,
                    "hyg_lqd_63": -0.005,
                    "iwm_spy_63": -0.020,
                },
            ),
            "liquid_leadership",
        ),
        (views.dual_momentum, _primary_state(base, {}), "dual_momentum"),
        (
            views.inflation_rotation,
            _primary_state(
                base | {"TLT": -1.2, "IEF": -0.8, "AGG": -0.7, "LQD": -0.8},
                {"hyg_lqd_63": -0.010, "dbc_ief_63": 0.070, "dbc_agg_63": 0.060},
            ),
            "inflation_rotation",
        ),
        (views.risk_adjusted_momentum, _primary_state(base, {}), "risk_adj_momentum"),
        (
            views.credit_switch,
            _primary_state(
                base | {"HYG": 0.9, "IWM": 0.7},
                {"spy_drawdown_252": -0.02, "spy_vol_z": 0.0, "risky_trend_breadth": 0.75, "hyg_lqd_63": 0.030, "iwm_spy_63": 0.020},
            ),
            "credit_switch",
        ),
        (
            views.international_rotation,
            _primary_state(
                base | {"EEM": 0.8, "EFA": -0.4, "UUP": -0.6},
                {"eem_efa_63": 0.060, "dollar_trend": -0.020, "dollar_mom_3_0": -0.010},
            ),
            "international_rotation",
        ),
        (
            views.reflation_breadth,
            _primary_state(
                base | {"IWM": 1.0, "HYG": 0.9, "VNQ": 0.8, "DBC": 0.9, "EEM": 0.7},
                {"iwm_spy_63": 0.040, "hyg_lqd_63": 0.025, "risky_trend_breadth": 0.80, "spy_drawdown_252": -0.03},
            ),
            "reflation_breadth",
        ),
        (
            views.growth_duration,
            _primary_state(
                base | {"IEF": 0.8, "AGG": 0.7, "TLT": 0.6, "DBC": -0.8, "HYG": -0.4, "IWM": -0.5},
                {"qqq_spy_63": 0.030, "hyg_lqd_63": -0.010, "stock_bond_corr_126": -0.05},
            ),
            "growth_duration",
        ),
        (
            views.correlation_stress,
            _primary_state(
                base | {"SPY": -0.4, "QQQ": -0.3, "IWM": -1.1, "EEM": -0.9, "HYG": -0.8, "VNQ": -0.7, "SHY": 0.4, "GLD": 0.5},
                {"avg_risky_corr_126": 0.75, "spy_vol_z": 1.20, "risky_trend_breadth": 0.20, "stock_bond_corr_126": 0.30},
            ),
            "correlation_stress",
        ),
        (
            views.duration_quality,
            _primary_state(
                base | {"TLT": -1.0, "IEF": -0.2, "AGG": 0.2, "SHY": 0.5, "HYG": -0.8},
                {"stock_bond_corr_126": 0.35, "hyg_lqd_63": -0.025},
            ),
            "duration_quality",
        ),
    ]

    for fn, state, expected_family in scenarios:
        _assert_view(fn(state, roles, settings), expected_family)


def _sector_roles() -> dict[str, object]:
    return {
        "assets": SECTOR_ASSETS,
        "growth": ["XLK", "XLC", "XLY"],
        "defensive": ["XLP", "XLU", "XLV"],
        "cyclical": ["XLB", "XLE", "XLF", "XLI"],
        "credit_sensitive": ["XLF", "XLY", "XLI"],
        "inflation_beneficiaries": ["XLE", "XLB"],
        "rate_sensitive": ["XLU", "XLV", "XLRE"],
        "small_cap_sensitive": ["XLI", "XLF", "XLY"],
        "quality_defensive": ["XLV", "XLP"],
        "sleeve_map": {asset: asset for asset in SECTOR_ASSETS},
    }


def _sector_settings() -> views.ViewSettings:
    return views.ViewSettings(
        family_q_caps=views.DEFAULT_SECTOR_FAMILY_Q_CAPS,
        family_display_names=views.DEFAULT_SECTOR_DISPLAY_NAMES,
        assets=SECTOR_ASSETS,
        entry_z=0.35,
    )


def _sector_state(
    scores: dict[str, float],
    means: dict[str, float],
    values: dict[str, float] | None = None,
) -> SimpleNamespace:
    all_scores = {asset: 0.0 for asset in SECTOR_ASSETS + SECTOR_CONTEXT}
    all_scores.update(scores)
    all_means = {asset: 0.0 for asset in all_scores}
    all_means.update(means)
    returns = _constant_returns(all_means)
    signal_table = _signal_table(all_scores)
    return SimpleNamespace(
        signal_table=signal_table,
        signals=signal_table,
        values=values or {},
        market_state=values or {},
        signal_returns=returns,
        returns=returns,
    )


def test_sector_view_rules_cover_momentum_defensive_credit_and_macro_rotations() -> None:
    settings = _sector_settings()
    roles = _sector_roles()

    high_growth = {"XLK": 1.5, "XLC": 1.2, "XLY": 1.0, "XLP": -0.6, "XLU": -0.8, "XLV": -0.5}
    cyclicals = {"XLB": 1.0, "XLE": 1.1, "XLF": 1.2, "XLI": 0.9, "XLP": -0.7, "XLU": -0.8, "XLV": -0.6}
    defensive = {"SPY": -0.8, "XLK": -0.7, "XLC": -0.5, "XLY": -0.8, "XLB": -0.6, "XLE": -0.7, "XLF": -0.9, "XLP": 0.8, "XLU": 0.7, "XLV": 0.9}

    scenarios = [
        (
            views.sector_momentum,
            _sector_state(
                {"XLK": 1.8, "XLE": 1.4, "XLF": 1.1, "XLP": -1.2, "XLU": -1.0, "XLV": -0.8},
                {},
            ),
            "sector_momentum",
        ),
        (
            views.growth_leadership,
            _sector_state(high_growth | {"QQQ": 1.0, "SPY": 0.3}, {"XLK": 0.0012, "XLC": 0.0010, "XLY": 0.0009, "XLP": -0.0006, "XLU": -0.0005, "XLV": -0.0004}, {"qqq_spy_63": 0.050}),
            "growth_leadership",
        ),
        (
            views.defensive_rotation,
            _sector_state(defensive, {"XLP": 0.0010, "XLU": 0.0009, "XLV": 0.0008, "XLK": -0.0008, "XLC": -0.0007, "XLY": -0.0009, "XLB": -0.0005, "XLE": -0.0006, "XLF": -0.0007}, {"spy_drawdown_252": -0.12}),
            "defensive_rotation",
        ),
        (
            views.cyclical_breadth,
            _sector_state(cyclicals | {"SPY": 0.7}, {"XLB": 0.0010, "XLE": 0.0012, "XLF": 0.0011, "XLI": 0.0009, "XLP": -0.0005, "XLU": -0.0004, "XLV": -0.0005}),
            "cyclical_breadth",
        ),
        (
            views.credit_beta,
            _sector_state(cyclicals | {"HYG": 1.0}, {}, {"hyg_lqd_63": 0.035, "hyg_shy_63": 0.045}),
            "credit_beta",
        ),
        (
            views.inflation_beneficiaries,
            _sector_state({"XLE": 1.3, "XLB": 1.1, "XLK": -0.6, "XLY": -0.4, "DBC": 1.2, "GLD": 0.8}, {"XLE": 0.0012, "XLB": 0.0010, "XLK": -0.0004, "XLY": -0.0005}, {"dollar_mom_3_0": -0.010}),
            "inflation_beneficiaries",
        ),
        (
            views.duration_sensitive,
            _sector_state({"XLU": 1.1, "XLV": 0.8, "XLRE": 0.9, "XLF": -0.6, "XLE": -0.5}, {"TLT": 0.0010, "IEF": 0.0008, "SHY": 0.0001, "XLU": 0.0011, "XLV": 0.0008, "XLRE": 0.0009, "XLF": -0.0003, "XLE": -0.0004}),
            "duration_sensitive",
        ),
        (
            views.small_cap_risk_on,
            _sector_state(cyclicals | {"IWM": 1.2}, {"XLI": 0.0011, "XLF": 0.0010, "XLY": 0.0009, "XLP": -0.0005, "XLU": -0.0004, "XLV": -0.0006}, {"iwm_spy_63": 0.050}),
            "small_cap_risk_on",
        ),
        (
            views.quality_defensive,
            _sector_state({"XLV": 1.0, "XLP": 0.9, "XLE": -0.8, "XLF": -0.7, "XLY": -0.6}, {"XLV": 0.0010, "XLP": 0.0008, "XLE": -0.0006, "XLF": -0.0005, "XLY": -0.0004}, {"spy_vol_z": 1.1, "sector_avg_corr_126": 0.70}),
            "quality_defensive",
        ),
    ]

    for fn, state, expected_family in scenarios:
        _assert_view(fn(state, roles, settings), expected_family)


def test_sector_reversal_uses_recent_dispersion_with_medium_term_filter() -> None:
    settings = _sector_settings()
    roles = _sector_roles()
    scores = {asset: 0.0 for asset in SECTOR_ASSETS + SECTOR_CONTEXT}
    state = _sector_state(scores, {})
    returns = state.returns.copy()
    returns.loc[:, SECTOR_ASSETS] = 0.0001
    returns.loc[returns.index[-21]:, ["XLK", "XLE"]] = 0.0060
    returns.loc[returns.index[-21]:, ["XLU", "XLV"]] = -0.0030
    state.returns = returns
    state.signal_returns = returns
    state.values = {"sector_21d_dispersion": 0.055}
    state.market_state = state.values

    view = views.sector_reversal(state, roles, settings)

    _assert_view(view, "sector_reversal")
    assert set(view.long_assets).issubset(set(SECTOR_ASSETS))
    assert set(view.short_assets).issubset(set(SECTOR_ASSETS))
