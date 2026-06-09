"""Data I/O subpackage: raw-to-analysis loaders and cleaning filters.

Owns all raw file ingestion that the project notebooks share. Loaders
return normalized panels with stable schemas regardless of input source.
Filter functions are small composable steps to be chained via ``df.pipe``.
"""

from __future__ import annotations

from .equity_ohlcv import load_ohlcv
from .option_chain import (
    combine_optionsdx_texts,
    filter_atm_window,
    filter_liquidity,
    filter_valid_quotes,
    load_option_chain,
    load_optionsdx_equity_pairs,
    load_spx_option_pairs,
    pair_calls_puts,
)
from .panel import (
    align_panels,
    load_vix,
    load_yfinance_panel,
    prices_to_returns_panel,
    vix_feature_frame,
)
from .macro import (
    clean_monthly_index,
    load_macro_factors,
    load_nfci,
    macro_availability_table,
)
from .rates import (
    load_par_yield_curve,
    tenor_first_valid,
    tenor_label_to_years,
)
from .schemas import OPTION_CHAIN_SOURCES, PANEL_SOURCES, RATE_SOURCES

__all__ = [
    "OPTION_CHAIN_SOURCES",
    "PANEL_SOURCES",
    "RATE_SOURCES",
    "align_panels",
    "combine_optionsdx_texts",
    "filter_atm_window",
    "filter_liquidity",
    "filter_valid_quotes",
    "clean_monthly_index",
    "load_macro_factors",
    "load_ohlcv",
    "load_nfci",
    "load_vix",
    "vix_feature_frame",
    "load_option_chain",
    "load_optionsdx_equity_pairs",
    "load_par_yield_curve",
    "load_spx_option_pairs",
    "load_yfinance_panel",
    "macro_availability_table",
    "pair_calls_puts",
    "prices_to_returns_panel",
    "tenor_first_valid",
    "tenor_label_to_years",
]
