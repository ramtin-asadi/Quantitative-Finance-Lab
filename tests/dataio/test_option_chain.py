from __future__ import annotations

import pandas as pd

from quantfinlab.dataio.option_chain import (
    filter_atm_window,
    filter_liquidity,
    filter_valid_quotes,
    load_option_chain,
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
