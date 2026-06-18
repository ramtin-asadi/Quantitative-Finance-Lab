from __future__ import annotations

import pytest

from quantfinlab.dataio.schemas import (
    OPTION_CHAIN_SOURCES,
    PANEL_SOURCES,
    RATE_SOURCES,
    get_option_chain_source,
    get_panel_source,
    get_rate_source,
)


def test_schema_getters_return_copies_and_reject_unknown_sources() -> None:
    rate = get_rate_source("us_treasury")
    rate["percent"] = False
    assert RATE_SOURCES["us_treasury"]["percent"] is True

    panel = get_panel_source("yfinance_export")
    option = get_option_chain_source("btc_deribit")
    assert panel["format"] == "wide_suffix"
    assert option["underlying_default"] == "BTC"

    with pytest.raises(ValueError, match="Unknown rate source"):
        get_rate_source("made_up")
    with pytest.raises(ValueError, match="Unknown panel source"):
        get_panel_source("made_up")
    with pytest.raises(ValueError, match="Unknown option-chain source"):
        get_option_chain_source("made_up")
    with pytest.raises(TypeError):
        RATE_SOURCES["new"] = {}
    assert "optionsdx_spx" in OPTION_CHAIN_SOURCES
    assert "nasdaq_close_volume" in PANEL_SOURCES
