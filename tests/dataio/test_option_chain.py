from __future__ import annotations

import pandas as pd

from quantfinlab.dataio.option_chain import (
    filter_atm_window,
    filter_liquidity,
    filter_valid_quotes,
    load_option_chain,
    load_optionsdx_equity_pairs,
    load_spx_option_pairs,
    pair_calls_puts,
)
from tests.synthetic.generators import option_quotes


def test_load_option_chain_normalizes_deribit_long_csv(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-02 16:00:00", "2024-01-02 16:00:00"],
            "instrument_name": ["BTC-26JAN24-42000-C", "BTC-26JAN24-42000-P"],
            "underlying_value": [40_000.0, 40_000.0],
            "bid": [900.0, 2_750.0],
            "ask": [1_100.0, 2_950.0],
            "volume": [4, 5],
            "open_interest": [100, 110],
            "iv": [55.0, 60.0],
        }
    )
    path = tmp_path / "btc.csv"
    raw.to_csv(path, index=False)

    out = load_option_chain(path, source="btc_deribit")

    assert set(out["option_type"]) == {"call", "put"}
    assert out["underlying"].eq("BTC").all()
    assert out["mid"].tolist() == [1_000.0, 2_850.0]
    assert out["iv"].between(0.50, 0.61).all()
    assert out["tau"].gt(0.0).all()
    assert out.attrs["annualization_days"] == 365.0


def test_option_chain_filters_and_pairs_complete_liquid_atm_quotes() -> None:
    option_chain = option_quotes()
    broken = option_chain.copy()
    broken.loc[0, "ask"] = broken.loc[0, "bid"] - 0.01
    broken = pd.concat([broken, option_chain.iloc[[0]].assign(strike=130.0, option_type="call")], ignore_index=True)

    valid = filter_valid_quotes(broken, require_pair=True)
    liquid = filter_liquidity(valid, max_rel_spread=0.10, tau_min_days=20, tau_max_days=80)
    atm = filter_atm_window(liquid, k_over_s_range=(0.85, 1.15), top_n_per_expiry=2, min_pairs_per_group=2)
    paired = pair_calls_puts(atm)

    assert paired[["c_bid", "p_bid", "c_mid", "p_mid"]].notna().all().all()
    assert paired["strike"].between(90.0, 110.0).all()
    assert len(paired) == 2


def test_optionsdx_wide_pair_loaders_keep_liquid_atm_rows(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "quote_date": ["2024-01-02", "2024-01-02", "2024-01-02"],
            "quote_readtime": ["2024-01-02 16:00:00"] * 3,
            "expire_date": ["2024-02-16"] * 3,
            "underlying_last": [100.0, 100.0, 100.0],
            "strike": [95.0, 100.0, 120.0],
            "c_bid": [6.0, 3.0, 0.5],
            "c_ask": [6.2, 3.2, 0.8],
            "p_bid": [0.8, 2.8, 19.5],
            "p_ask": [1.0, 3.0, 20.0],
            "c_volume": [10, 20, 5],
            "p_volume": [8, 18, 4],
        }
    )
    path = tmp_path / "spx.csv"
    raw.to_csv(path, index=False)

    spx = load_spx_option_pairs(
        path,
        max_rel_spread=0.25,
        tau_min_days=20,
        tau_max_days=80,
        k_over_s_range=(0.90, 1.05),
        top_n_per_expiry=2,
        min_pairs_per_expiry=1,
    )
    spy = load_optionsdx_equity_pairs(
        path,
        source="optionsdx_spy",
        max_rel_spread=0.25,
        tau_min_days=20,
        tau_max_days=80,
        k_over_s_range=(0.90, 1.05),
        top_n_per_expiry=2,
        min_pairs_per_expiry=1,
    )

    assert spx["underlying"].eq("SPX").all()
    assert spy["underlying"].eq("SPY").all()
    assert spx["strike"].tolist() == [95.0, 100.0]
    assert spx["liq_score"].gt(0.0).all()
    assert spx.attrs["annualization_days"] == 365.25
