import pandas as pd
import pytest

from quantfinlab.dataio.realtime import read_fred_vintages, read_statscan_vintages
from quantfinlab.macro.realtime import vintage_asof


def test_fred_snapshot_label_is_not_release_day(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    path = tmp_path / "fred.parquet"
    pd.DataFrame({"panel": ["MD", "MD"], "series_id": ["x", "x"],
                   "vintage_date": pd.to_datetime(["2020-01-01", "2020-02-01"]),
                   "observation_date": pd.to_datetime(["2019-12-01", "2019-12-01"]),
                   "value": [100., 999.]}).to_parquet(path)
    known = read_fred_vintages(path, as_of="2020-02-15")
    assert known["value"].tolist() == [100.]
    assert known["available_at"].iloc[0] == pd.Timestamp("2020-02-01")


def test_forward_snapshot_cannot_backfill_historical_information(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    path = tmp_path / "statscan.parquet"
    pd.DataFrame({"table_id": ["x"], "series_title": ["Canada | Retail"],
                   "reference_date": pd.to_datetime(["2020-01-01"]),
                   "release_date": pd.to_datetime(["2026-08-10"]),
                   "snapshot_date": pd.to_datetime(["2026-08-29"]), "value": [100.],
                   "unit": ["Dollars"], "scalar_factor": ["millions"],
                   "source_kind": ["forward_snapshot"]}).to_parquet(path)
    x = read_statscan_vintages(path, {"retail": {"table_id": "x", "series_title": "Canada | Retail"}})
    assert vintage_asof(x, "2026-08-20").empty
    assert len(vintage_asof(x, "2026-08-30")) == 1
