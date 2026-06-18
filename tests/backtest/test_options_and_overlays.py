from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.backtest import options, overlays


def _overlay_quotes() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=6)
    expiry = pd.Timestamp("2024-02-16")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        spot = 100.0 + i
        for option_type, strike, delta in (("call", 105.0, 0.30), ("put", 95.0, -0.25)):
            mid = 2.0 + 0.05 * i + (0.20 if option_type == "put" else 0.0)
            rows.append(
                {
                    "date": date,
                    "timestamp": date,
                    "expiry": expiry,
                    "strike": strike,
                    "option_type": option_type,
                    "bid": mid - 0.05,
                    "ask": mid + 0.05,
                    "mid": mid,
                    "spot": spot,
                    "dte": (expiry - date).days,
                    "dte_days": (expiry - date).days,
                    "dte_calendar": (expiry - date).days,
                    "moneyness": strike / spot,
                    "delta": delta,
                    "gamma": 0.02,
                    "vega": 8.0,
                    "theta": -0.02,
                    "rho": 0.01,
                    "rel_spread": 0.02,
                    "relative_spread": 0.02,
                    "volume": 100 + i,
                    "assignment_risk": 0.35 if option_type == "call" else 0.0,
                    "days_to_next_dividend": 3,
                    "contract_key": f"{option_type}_{expiry:%Y-%m-%d}_{strike}",
                }
            )
    return pd.DataFrame(rows)


def _hedge_path() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    dates = pd.bdate_range("2024-01-02", periods=8)
    expiry = pd.Timestamp("2024-02-16")
    rows = []
    for i, date in enumerate(dates):
        spot = 100.0 + i
        mid = 4.0 + 0.10 * i
        rows.append(
            {
                "date": date,
                "timestamp": date,
                "expiry": expiry,
                "strike": 100.0,
                "option_type": "call",
                "spot": spot,
                "bid": mid - 0.05,
                "ask": mid + 0.05,
                "mid": mid,
                "delta": 0.55,
                "gamma": 0.02,
                "vega": 8.0,
                "theta": -0.02,
                "rho": 0.01,
                "dte": (expiry - date).days,
                "dte_days": (expiry - date).days,
                "rel_spread": 0.02,
                "volume": 100,
                "contract_key": "call_2024-02-16_100",
            }
        )
    schedule = pd.DataFrame(
        {
            "entry_date": [dates[0]],
            "contract_key": ["call_2024-02-16_100"],
            "quantity": [1.0],
            "label": ["test"],
            "max_hold_days": [3],
        }
    )
    spot = pd.Series(100.0 + np.arange(len(dates)), index=dates)
    return pd.DataFrame(rows), schedule, spot


def test_option_accounting_helpers_and_trade_ledger() -> None:
    row = {"bid": 1.90, "ask": 2.10, "mid": 2.00}

    assert options.compute_option_mark_to_market_pnl(2.4, 2.0, quantity=3, multiplier=100) == pytest.approx(120.0)
    assert options.compute_hedge_pnl(102.0, 100.0, hedge_units=-5.0) == pytest.approx(-10.0)
    assert options.apply_hedge_transaction_costs(10.0, 100.0, trading_cost_bps=2.0, half_spread=0.01) == pytest.approx(0.30)
    assert options.option_fill_price(row, side=1, action="open") == pytest.approx(2.10)
    assert options.option_fill_price(row, side=-1, action="close") == pytest.approx(2.10)
    assert options.mark_option_position(row, quantity=-2, multiplier=100) == pytest.approx(-400.0)
    assert options.settle_option_expiry("call", spot=105.0, strike=100.0, quantity=2, multiplier=100) == pytest.approx(1000.0)
    assert options.close_option_position(row, quantity=2, multiplier=100) == pytest.approx(380.0)

    ledger = options.option_trade_ledger([{"date": "2024-01-02", "quantity": "2", "price": "1.5"}])
    assert pd.api.types.is_datetime64_any_dtype(ledger["date"])
    assert ledger.loc[0, "quantity"] == pytest.approx(2.0)


def test_scheduled_option_hedge_smoke_and_diagnostics_tables() -> None:
    path, schedule, spot = _hedge_path()

    result = options.run_scheduled_option_hedging_backtest(
        path,
        schedule,
        spot_series=spot,
        strategies=("unhedged",),
        option_multiplier=100.0,
        contract_size=100.0,
        max_hold_days=3,
    )
    residual_delta = options.rolling_residual_delta(result, window=2)
    residual_vega = options.rolling_residual_vega(result, window=2)

    assert result["summary"].loc[0, "status"] == "ok"
    assert not result["components"].empty
    assert not residual_delta.empty
    assert not residual_vega.empty
    assert options.hedging_diagnostics(result).loc[0, "n_option_book_rows"] == len(path)
    assert options.summarize_hedging_backtest(result).equals(result["summary"])
    assert options.hedge_trade_ledger(result).equals(result["trades"])


def test_option_schedule_matching_and_hedge_comparison_helpers() -> None:
    quotes = _overlay_quotes()
    source = pd.DataFrame(
        {
            "entry_date": [quotes["date"].min()],
            "contract_key": ["call_2024-02-16_105.0"],
            "quantity": [-1.0],
            "label": ["source"],
        }
    )

    matched = options.matched_option_schedule(source, quotes, min_future_marks=2)
    hedge_book = options.hedge_book_from_schedules(quotes, [matched], lookahead_days=10)
    comparison = options.scheduled_hedge_comparison(
        {
            "run": {
                "summary": pd.DataFrame({"strategy": ["unhedged"], "status": ["ok"], "total_pnl": [1.0], "traded_notional": [10.0], "total_costs": [0.1]}),
                "components": pd.DataFrame(
                    {
                        "strategy": ["unhedged"],
                        "episode_id": [1],
                        "date": [quotes["date"].min()],
                        "main_pos": [1.0],
                        "option_mid": [2.0],
                        "vega_after": [8.0],
                        "delta_after": [0.4],
                    }
                ),
            }
        }
    )

    assert not matched.empty
    assert matched.loc[0, "quantity"] == pytest.approx(-1.0)
    assert set(hedge_book["contract_key"]) == set(matched["contract_key"])
    assert comparison.loc[0, "run"] == "run"
    assert comparison.loc[0, "pnl_per_traded_notional"] == pytest.approx(0.1)


def test_overlay_schedules_backtest_and_summary_tables() -> None:
    quotes = _overlay_quotes()
    underlying = pd.Series(100.0 + np.arange(6), index=pd.bdate_range("2024-01-02", periods=6))

    covered = overlays.covered_call_schedule(quotes, contracts=1.0, min_dte=20, max_dte=60, rebalance_every=3)
    protective = overlays.protective_put_schedule(quotes, contracts=1.0, min_dte=20, max_dte=75, rebalance_every=3)
    collar = overlays.collar_schedule(quotes, contracts=1.0, rebalance_every=3)
    rolls = overlays.boundary_roll_schedule(quotes, threshold=0.30)
    marked = overlays.mark_book_for_schedules(quotes, {"covered": covered})
    defense = overlays.assignment_defense_actions(quotes, threshold=0.30, ex_div_days=7)
    result = overlays.run_overlay_backtest({"covered": covered}, quotes, underlying, initial_nav=10_000.0, shares=10.0)
    summary = overlays.overlay_summary(result)
    mechanics = overlays.overlay_mechanics_table(result, shares=10.0)
    deciles = overlays.pnl_by_vrp_decile(pd.DataFrame({"vrp_rank": [0.1, 0.8], "net_pnl": [1.0, -0.5]}), n_deciles=2)

    assert not covered.empty
    assert not protective.empty
    assert len(collar) == len(covered) + len(protective)
    assert not rolls.empty
    assert set(marked["contract_key"]).issubset(set(covered["contract_key"]))
    assert not defense.empty
    assert summary.loc[0, "strategy"] == "covered"
    assert mechanics.loc[0, "opens"] >= 1
    assert deciles["n_trades"].sum() == 2
