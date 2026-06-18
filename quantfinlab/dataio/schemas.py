"""Source registry for the dataio loaders.

Each schema describes the conventions of a raw data source so that a
single generalized loader can ingest it. Callers either pick a
registered ``source`` name or pass an explicit ``column_map``.
"""

from __future__ import annotations

from types import MappingProxyType

_RATE_SOURCES: dict[str, dict[str, object]] = {
    # US Treasury par-yield CSV (Fed/Treasury direct download).
    # Columns like '1 mo', '3 mo', '1 yr', '10 yr' ... values in percent.
    "us_treasury": {
        "date_col": "date",
        "skip_banner": False,
        "na_values": ("",),
        "percent": True,
    },
    # Japan MOF par-yield CSV. First line is a banner ("Interest Rate ...
    # (Unit : %)") that must be skipped. Missing values appear as "-".
    # Tenor column names are already in compact form ('1Y', '2Y', ...).
    "japan_mof": {
        "date_col": "Date",
        "skip_banner": True,
        "na_values": ("-",),
        "percent": True,
    },
}


_PANEL_SOURCES: dict[str, dict[str, object]] = {
    # Root-level source-centered panels with '<TICKER>__<field>' columns.
    # The field suffix may be lower or upper case; the loader normalizes
    # to lower. Used by ETF CSVs and Stooq Parquet panels.
    "yfinance_export": {
        "format": "wide_suffix",
        "suffix": "__",
        "date_col": "date",
    },
    "nasdaq_close_volume": {
        "format": "wide_suffix",
        "suffix": "__",
        "date_col": "date",
    },
    "hkex_close_volume": {
        "format": "wide_suffix",
        "suffix": "__",
        "date_col": "date",
    },
}


_OPTION_CHAIN_SOURCES: dict[str, dict[str, object]] = {
    # OptionsDX / CBOE-style SPX end-of-day chain (parquet).
    "optionsdx_spx": {
        "schema": "wide_call_put",  # one row per (date, expiry, strike)
        "profile": "spx_optiondx",
        "underlying_default": "SPX",
        "annualization_days": 365.25,
    },
    "optionsdx_spy": {
        "schema": "wide_call_put",
        "profile": "optionsdx_equity",
        "underlying_default": "SPY",
        "annualization_days": 365.25,
    },
    "optionsdx_qqq": {
        "schema": "wide_call_put",
        "profile": "optionsdx_equity",
        "underlying_default": "QQQ",
        "annualization_days": 365.25,
    },
    # Deribit BTC end-of-day chain via OptionsDX (parquet, long-form).
    "btc_deribit": {
        "schema": "long",  # one row per leg with option_right C/P
        "profile": "btc_deribit",
        "underlying_default": "BTC",
        "annualization_days": 365.0,
    },
    # NSE NIFTY long-form daily option chain (parquet).
    "nse_nifty": {
        "schema": "long",
        "profile": "nse_nifty",
        "underlying_default": "NIFTY",
        "annualization_days": 365.0,
    },
}


RATE_SOURCES: MappingProxyType[str, dict[str, object]] = MappingProxyType(_RATE_SOURCES)
PANEL_SOURCES: MappingProxyType[str, dict[str, object]] = MappingProxyType(_PANEL_SOURCES)
OPTION_CHAIN_SOURCES: MappingProxyType[str, dict[str, object]] = MappingProxyType(_OPTION_CHAIN_SOURCES)


def get_rate_source(name: str) -> dict[str, object]:
    if name not in _RATE_SOURCES:
        raise ValueError(f"Unknown rate source {name!r}; known: {sorted(_RATE_SOURCES)}")
    return dict(_RATE_SOURCES[name])


def get_panel_source(name: str) -> dict[str, object]:
    if name not in _PANEL_SOURCES:
        raise ValueError(f"Unknown panel source {name!r}; known: {sorted(_PANEL_SOURCES)}")
    return dict(_PANEL_SOURCES[name])


def get_option_chain_source(name: str) -> dict[str, object]:
    if name not in _OPTION_CHAIN_SOURCES:
        raise ValueError(
            f"Unknown option-chain source {name!r}; known: {sorted(_OPTION_CHAIN_SOURCES)}"
        )
    return dict(_OPTION_CHAIN_SOURCES[name])


__all__ = [
    "OPTION_CHAIN_SOURCES",
    "PANEL_SOURCES",
    "RATE_SOURCES",
    "get_option_chain_source",
    "get_panel_source",
    "get_rate_source",
]
