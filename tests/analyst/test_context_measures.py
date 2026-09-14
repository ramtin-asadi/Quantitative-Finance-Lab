from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quantfinlab.analyst.context import ContextRegistry
from quantfinlab.analyst.documents import bounded_excerpt
from quantfinlab.analyst.events import event_evidence, positioning_records
from quantfinlab.analyst.evidence import (
    context_evidence,
    context_text,
    evidence_row,
    merge_evidence,
)
from quantfinlab.analyst.macro import release_reaction
from quantfinlab.analyst.market import curve_shape, market_moves, relative_moves, risk_measures
from quantfinlab.analyst.schemas import ContextSnapshot


def test_cot_directional_net_excludes_spreading_positions():
    text = "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE Code-067651\nFUTURES ONLY POSITIONS AS OF 09/08/26\nOPEN INTEREST: 1,000\nCOMMITMENTS\n 350 210 600 300 400 650 610 350 390\nCHANGES FROM 09/01/26 (CHANGE IN OPEN INTEREST: 10)\n 20 -10 50 2 3 22 -7 -12 17"
    rows = positioning_records(SimpleNamespace(text=text))
    assert rows[0]["net"] == 140 and rows[0]["net_change"] == 30
    assert rows[0]["spreading"] == 600


def test_energy_event_uses_narrative_without_unlabeled_table_cells():
    document = SimpleNamespace(source="eia", document_id="energy", source_url="https://www.eia.gov/test",
        available_at="2026-09-12T00:00:00Z", title="Weekly petroleum highlights",
        text="Weekly Petroleum Status Report\nFor the week ending September 04, 2026, refineries processed 17.6 million barrels per day.\nCommercial crude oil inventories decreased 0.4 million barrels to 424.1 million barrels.\nHighlights\nRefinery Activity\n97.6 97.2 95.1")
    rows = event_evidence(document)
    assert len(rows) == 2 and "424.1" in rows[1]["text"]
    assert all("97.6" not in row["text"] and "Highlights" not in row["text"] for row in rows)


def test_release_window_respects_publication_and_price_availability():
    prices = pd.DataFrame({"SPY": [100.0, 110.0, 121.0]}, index=pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-14"]))
    morning = release_reaction(prices, published_at="2026-09-11T12:30:00Z", as_of="2026-09-14T23:00:00Z")
    assert morning["returns_percent"]["SPY"] == 10 and morning["after_date"] == "2026-09-11"
    evening = release_reaction(prices, published_at="2026-09-11T21:00:00Z", as_of="2026-09-14T23:00:00Z")
    assert evening["before_date"] == "2026-09-11" and evening["after_date"] == "2026-09-14"
    assert release_reaction(prices, published_at="2026-09-11T12:30:00Z", as_of="2026-09-11T19:00:00Z") is None
    assert release_reaction(prices, published_at=None, as_of="2026-09-14T23:00:00Z") is None


def test_curve_shape_uses_basis_points_and_percent():
    rates = pd.DataFrame({name: [0.04, 0.04] for name in ["3M", "2Y", "5Y", "10Y", "30Y"]})
    rates.loc[1, "10Y"] = 0.05
    shape = curve_shape(rates)
    assert shape.loc[0, "level_percent"] == pytest.approx(4)
    assert shape.loc[1, "2s10s_bp"] == pytest.approx(100)
    assert shape.loc[1, "curvature_bp"] == pytest.approx(-100)


def test_return_and_relative_move_units():
    index = pd.date_range("2025-01-01", periods=300)
    prices = pd.DataFrame({"SPY": 100.0, "QQQ": 100.0}, index=index)
    prices.iloc[-1] = [101, 103]
    moves = market_moves(prices)
    gaps = relative_moves(prices, pairs=[("QQQ", "SPY")])
    assert moves.loc["SPY", "return_1d"] == pytest.approx(0.01)
    assert gaps.loc["QQQ_minus_SPY", "return_gap_1d_pp"] == pytest.approx(2.0)


def test_beta_is_consistent_with_scaled_returns():
    returns = np.sin(np.arange(300)) * 0.01
    prices = pd.DataFrame({"SPY": 100 * np.cumprod(1 + returns), "levered": 100 * np.cumprod(1 + 2 * returns)})
    risk = risk_measures(prices)
    assert risk.loc["levered", "beta_spy_63d"] == pytest.approx(2.0)
    assert risk.loc["levered", "correlation_spy_63d"] == pytest.approx(1.0)


def test_context_formatting_distinguishes_percentiles_and_changes():
    snapshot = ContextSnapshot(name="financial_conditions", as_of="2026-09-01T00:00:00Z",
        latest_data_at="2026-08-31T00:00:00Z", available_at="2026-08-31T12:00:00Z",
        freshness="current", measures={"fci_percentile": 0.81, "change_pp": 0.25}, dependencies=[])
    text = context_text(snapshot)
    assert "81.00 percentile" in text and "0.25 percentage points" in text
    rows, statuses = context_evidence([snapshot], count=len, budget=10000)
    assert len(rows) == 1 and statuses[0]["freshness"] == "current"
    assert not context_evidence([snapshot], count=len, budget=10)[0]


def test_evidence_union_preserves_order_and_checks_time():
    row = evidence_row("Verified reported facts.", key="one", available_at="2026-08-31T00:00:00Z", source="sec", title="Release")
    assert merge_evidence([row], [row], as_of="2026-09-01T00:00:00Z") == [row]
    with pytest.raises(ValueError, match="after"):
        merge_evidence([row], as_of="2026-08-30T00:00:00Z")
    with pytest.raises(ValueError, match="hash"):
        merge_evidence([{**row, "text": "Changed facts"}], as_of="2026-09-01T00:00:00Z")


def test_stale_snapshot_keeps_its_status_but_not_current_prompt_values():
    snapshot = ContextSnapshot(name="factors", as_of="2026-09-01T00:00:00Z",
        latest_data_at="2026-05-01T00:00:00Z", available_at="2026-05-01T00:00:00Z",
        freshness="stale", measures={"beta": 1.4}, dependencies=[])
    evidence, statuses = context_evidence([snapshot], count=len, budget=10000)
    assert not evidence
    assert statuses[0]["freshness"] == "stale"
    assert any("omitted" in note for note in statuses[0]["notes"])


def test_excerpts_stay_within_token_budget():
    text = "A company reported stronger operating cash generation and lower financing costs."
    result = bounded_excerpt(text, lambda value: len(value.split()), budget=6)
    assert result == "A company reported stronger operating cash"


def test_registry_dispatches_public_builders_without_hidden_methods(tmp_path):
    registry = ContextRegistry(tmp_path)
    assert registry.builders["market"].__name__ == "market_context"
    assert not hasattr(registry, "_market")
