from __future__ import annotations

import pandas as pd
import pytest

from quantfinlab.dataio.equity_ohlcv import load_ohlcv


def test_load_ohlcv_prefers_adjusted_close_and_keeps_last_duplicate(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "Open": [101.0, 99.0, 99.5],
            "Close": [102.0, 100.0, 101.0],
            "Adj Close": [101.5, 99.5, 100.5],
            "Volume": [1_100, 1_000, 1_050],
        }
    )
    path = tmp_path / "ohlcv.csv"
    raw.to_csv(path, index=False)

    out = load_ohlcv(path, fields=("close", "raw_close", "volume"))

    assert list(out.columns) == ["close", "raw_close", "volume"]
    assert list(out.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert out.loc[pd.Timestamp("2024-01-02"), "close"] == 100.5
    assert out.loc[pd.Timestamp("2024-01-02"), "raw_close"] == 101.0


def test_load_ohlcv_rejects_missing_requested_field(tmp_path) -> None:
    path = tmp_path / "minimal.csv"
    pd.DataFrame({"date": ["2024-01-02"], "close": [100.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="field 'volume'"):
        load_ohlcv(path, fields=("volume",))
