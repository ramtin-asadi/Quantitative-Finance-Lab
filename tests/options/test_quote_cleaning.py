from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.options.quote_cleaning import (
    add_moneyness,
    add_time_to_expiry,
    attach_spot_from_series,
    clean_option_quotes,
    convert_quotes_to_usd_equivalent,
    detect_option_price_unit,
    ensure_option_mid_quotes,
    extract_spot_series,
    pair_put_call_quotes,
    surface_common_support,
    surface_ready_quotes,
    surface_support_by_date,
    wide_option_chain_to_long,
)
from tests.synthetic.generators import option_quotes, option_surface_quotes


def test_wide_chain_conversion_mid_spot_and_usd_conversion_helpers() -> None:
    raw = pd.DataFrame(
        {
            "quote_date": ["2024-01-02"],
            "expire_date": ["2024-02-16"],
            "strike": [100.0],
            "underlying_last": [100.5],
            "c_bid": [2.0],
            "c_ask": [2.4],
            "p_bid": [1.7],
            "p_ask": [2.1],
            "c_iv": [24.0],
            "p_iv": [25.0],
        }
    )

    long = wide_option_chain_to_long(raw, underlying_default="SYN")
    mid = ensure_option_mid_quotes(long.drop(columns=["mid"], errors="ignore"))
    dated = attach_spot_from_series(mid.assign(spot=np.nan), pd.Series([99.0, 100.5], index=pd.to_datetime(["2024-01-01", "2024-01-02"])))
    timed = add_moneyness(add_time_to_expiry(dated))

    assert set(long["option_type"]) == {"call", "put"}
    assert np.isclose(mid.loc[mid["option_type"].eq("call"), "mid"].iloc[0], 2.2)
    assert dated["spot"].eq(100.5).all()
    assert timed["tau"].gt(0).all()
    assert extract_spot_series(timed).iloc[0] == 100.5

    base_quotes = pd.DataFrame({"mid": [0.01], "bid": [0.009], "ask": [0.011], "spot": [50_000.0], "currency": ["BTC"]})
    converted = convert_quotes_to_usd_equivalent(base_quotes, unit="auto")
    assert detect_option_price_unit(base_quotes) == "base"
    assert np.isclose(converted.loc[0, "mid"], 500.0)


def test_cleaning_pairs_and_surface_readiness_preserve_liquid_synthetic_panel() -> None:
    cleaned, report = clean_option_quotes(option_quotes(strikes=(90.0, 95.0, 100.0, 105.0, 110.0)), min_pairs_per_expiry=2, closest_atm_pairs=3)
    pairs = pair_put_call_quotes(cleaned)
    ready = surface_ready_quotes(option_surface_quotes())
    support = surface_support_by_date(ready)
    common = surface_common_support(ready, min_support_share=1.0)

    assert not cleaned.empty
    assert report.iloc[-1]["step"] == "final rows"
    assert len(pairs) >= 3
    assert ready["surface_weight"].between(0.05, 20.0).all()
    assert support.loc[0, "quotes"] == len(ready)
    assert common["support_mask"].shape == common["support_share"].shape
