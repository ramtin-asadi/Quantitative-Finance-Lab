"""Data I/O subpackage: raw-to-analysis loaders and cleaning filters.

Owns all raw file ingestion that the project notebooks share. Loaders
return normalized panels with stable schemas regardless of input source.
Filter functions are small composable steps to be chained via ``df.pipe``.
"""

from __future__ import annotations

from .equity_ohlcv import load_ohlcv
from .option_chain import (
    filter_atm_window,
    filter_liquidity,
    filter_valid_quotes,
    load_option_chain,
    load_spx_option_pairs,
    pair_calls_puts,
)
from .panel import (
    align_panels,
    load_yfinance_panel,
    prices_to_returns_panel,
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
    "filter_atm_window",
    "filter_liquidity",
    "filter_valid_quotes",
    "load_ohlcv",
    "load_option_chain",
    "load_par_yield_curve",
    "load_spx_option_pairs",
    "load_yfinance_panel",
    "pair_calls_puts",
    "prices_to_returns_panel",
    "tenor_first_valid",
    "tenor_label_to_years",
]
