"""Data I/O subpackage: raw-to-analysis loaders and cleaning filters.

Owns all raw file ingestion that the project notebooks share. Loaders
return normalized panels with stable schemas regardless of input source.
Filter functions are small composable steps to be chained via ``df.pipe``.
"""

from __future__ import annotations

from .credit import read_credit_facts, read_credit_market, read_sec_filings, read_structured_credit
from .equity_ohlcv import load_ohlcv
from .fundamentals import read_sec_facts, read_sec_metadata, read_statement_values
from .macro import (
                       clean_monthly_index,
                       load_acm_term_premium,
                       load_macro_factors,
                       load_nfci,
                       macro_availability_table,
                       read_boc_market,
                       read_macro_forecasts,
                       read_mpt_bins,
)
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
                       load_hkex_close_volume,
                       load_nasdaq_close_volume,
                       load_vix,
                       load_yfinance_panel,
                       prices_to_returns_panel,
                       read_equity_history,
                       vix_feature_frame,
)
from .rates import (
                       load_par_yield_curve,
                       read_boc_zero_curve,
                       read_credit_curves,
                       risk_free_returns,
                       tenor_first_valid,
                       tenor_label_to_years,
)
from .realtime import (
                       read_alfred,
                       read_fred_vintages,
                       read_philly_releases,
                       read_philly_vintages,
                       read_statscan_vintages,
                       statscan_catalog,
)
from .schemas import OPTION_CHAIN_SOURCES, PANEL_SOURCES, RATE_SOURCES

__all__ = [
    "read_statement_values", "read_credit_curves", "read_boc_zero_curve",
    "read_macro_forecasts", "read_boc_market", "read_mpt_bins",
    "read_sec_filings", "read_credit_facts", "read_credit_market", "read_structured_credit",
    "read_alfred", "read_fred_vintages", "read_philly_vintages", "read_philly_releases",
    "read_statscan_vintages", "statscan_catalog",
    "OPTION_CHAIN_SOURCES",
    "PANEL_SOURCES",
    "RATE_SOURCES",
    "align_panels",
    "combine_optionsdx_texts",
    "filter_atm_window",
    "filter_liquidity",
    "filter_valid_quotes",
    "clean_monthly_index",
    "load_acm_term_premium",
    "load_hkex_close_volume",
    "load_macro_factors",
    "load_nasdaq_close_volume",
    "load_ohlcv",
    "load_nfci",
    "load_vix",
    "vix_feature_frame",
    "load_option_chain",
    "load_optionsdx_equity_pairs",
    "load_par_yield_curve",
    "risk_free_returns",
    "load_spx_option_pairs",
    "load_yfinance_panel",
    "macro_availability_table",
    "pair_calls_puts",
    "prices_to_returns_panel",
    "read_equity_history",
    "read_sec_facts",
    "read_sec_metadata",
    "tenor_first_valid",
    "tenor_label_to_years",
]
