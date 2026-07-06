from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.dataio.panel import (
    align_panels,
    load_vix,
    load_yfinance_panel,
    prices_to_returns_panel,
    vix_feature_frame,
)


def test_load_yfinance_panel_sorts_dedupes_and_filters_tickers(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-02", "2024-01-04"],
            "AAA__close": [101.0, 100.0, 100.5, 102.0],
            "BBB__close": [201.0, 200.0, 200.5, 202.0],
            "AAA__volume": [1_010, 1_000, 1_005, 1_020],
            "IGN__open": [10.0, 10.0, 10.5, 11.0],
        }
    )
    path = tmp_path / "panel.csv"
    raw.to_csv(path, index=False)

    panels = load_yfinance_panel(
        path,
        fields=("close", "volume", "open"),
        tickers=["aaa"],
        source=None,
    )

    close = panels["close"]
    assert list(close.index) == list(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert list(close.columns) == ["AAA"]
    assert close.loc[pd.Timestamp("2024-01-02"), "AAA"] == 100.5
    assert panels["volume"].loc[pd.Timestamp("2024-01-02"), "AAA"] == 1_005
    assert panels["open"].shape[1] == 0


def test_align_panels_and_returns_panel_keep_shape_without_lookahead() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    close = pd.DataFrame({"AAA": [100.0, np.nan, 102.0], "BBB": [50.0, 51.0, 52.0]}, index=idx)
    volume = pd.DataFrame({"AAA": [10.0, 11.0], "CCC": [5.0, 6.0]}, index=idx[:2])

    close_aligned, volume_aligned = align_panels(close, volume, how="inner")
    assert list(close_aligned.columns) == ["AAA"]
    assert list(close_aligned.index) == list(idx[:2])
    assert volume_aligned.loc[pd.Timestamp("2024-01-03"), "AAA"] == 11.0

    returns = prices_to_returns_panel(close, ffill_limit=1, fill_isolated_with=0.0)
    assert list(returns.index) == list(close.index[1:])
    assert returns.loc[pd.Timestamp("2024-01-04"), "AAA"] == pytest.approx(0.02)
    assert returns.loc[pd.Timestamp("2024-01-03"), "AAA"] == 0.0


def test_panel_loader_merges_multiple_files_lowercases_and_builds_vix_features(tmp_path) -> None:
    first = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "AAA__close": [100.0, 101.0],
            "AAA__volume": [1000, 1100],
        }
    )
    second = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "BBB__close": [50.0, 51.0],
            "BBB__volume": [2000, 2100],
        }
    )
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    first.to_csv(path_a, index=False)
    second.to_csv(path_b, index=False)
    vix_path = tmp_path / "vix.csv"
    pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=80),
            "VIX": np.linspace(12.0, 25.0, 80),
        }
    ).to_csv(vix_path, index=False)

    panels = load_yfinance_panel([path_a, path_b], fields=("close", "volume"), tickers=["bbb", "aaa"], lowercase=True)
    target_index = pd.bdate_range("2024-01-02", periods=85)
    vix = load_vix(vix_path, index=target_index, ffill_limit=5)
    features = vix_feature_frame(vix, index=target_index)

    assert list(panels["close"].columns) == ["bbb", "aaa"]
    assert panels["volume"].loc[pd.Timestamp("2024-01-03"), "bbb"] == 2100
    assert vix.index.equals(target_index)
    assert {"vix_z_20", "vix_ma_ratio_63", "vix_pct_252"}.issubset(features.columns)
    assert np.isfinite(features.to_numpy()).all()
