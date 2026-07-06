from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantfinlab.backtest import options as option_backtest, overlays
from tests.synthetic.generators import option_surface_quotes


def _quotes_for_backtest() -> pd.DataFrame:
    quotes = option_surface_quotes(
        dates=("2024-01-02", "2024-01-03", "2024-01-04"),
        tau_days=(30, 45),
        k_values=(-0.10, 0.0, 0.10),
    ).copy()
    quotes["dte"] = quotes["dte_days"]
    quotes["assignment_risk"] = np.where(quotes["option_type"].eq("call"), 0.35, 0.05)
    quotes["days_to_next_dividend"] = 5
    quotes["contract_multiplier"] = 100.0
    return quotes


def test_option_cashflow_helpers_and_ledger_use_bid_ask_direction() -> None:
    row = {"bid": 2.00, "ask": 2.20, "mid": 2.10}
    ledger = option_backtest.option_trade_ledger(
        [
            {
                "date": "2024-01-02",
                "entry_date": "2024-01-02",
                "exit_date": "2024-01-05",
                "expiry": "2024-02-16",
                "quantity": "2",
                "price": "2.10",
                "cashflow": "-420",
                "spread_cost": "10",
                "pnl": "25",
            }
        ]
    )

    assert option_backtest.compute_option_mark_to_market_pnl(2.4, 2.1, quantity=2, multiplier=100) == pytest.approx(60.0)
    assert option_backtest.compute_hedge_pnl(101.0, 100.0, hedge_units=-25.0) == pytest.approx(-25.0)
    assert option_backtest.apply_hedge_transaction_costs(10, 100, trading_cost_bps=2, half_spread=0.01) == pytest.approx(0.30)
    assert option_backtest.hedging_drawdown(pd.Series([1.0, 1.1, 1.05])).iloc[-1] == pytest.approx(-0.05)
    assert option_backtest.option_fill_price(row, side=1.0, action="open") == pytest.approx(2.20)
    assert option_backtest.option_fill_price(row, side=1.0, action="close") == pytest.approx(2.00)
    assert option_backtest.option_fill_price(row, side=-1.0, action="open") == pytest.approx(2.00)
    assert option_backtest.close_option_position(row, quantity=-2.0) == pytest.approx(-440.0)
    assert option_backtest.mark_option_position(row, quantity=2.0) == pytest.approx(420.0)
    assert option_backtest.settle_option_expiry("call", 110.0, 100.0, 2.0) == pytest.approx(2000.0)
    assert option_backtest.settle_option_expiry("put", 90.0, 100.0, -1.0) == pytest.approx(-1000.0)
    assert pd.api.types.is_datetime64_any_dtype(ledger["date"])
    assert ledger.loc[0, "quantity"] == pytest.approx(2.0)


def test_matched_option_schedule_book_and_comparison_tables() -> None:
    quotes = _quotes_for_backtest()
    first = quotes.iloc[0]
    entry_schedule = pd.DataFrame(
        [
            {
                "entry_date": first["date"],
                "contract_key": first["contract_key"],
                "quantity": -1.0,
                "max_hold_days": 3,
            }
        ]
    )
    matched = option_backtest.matched_option_schedule(
        entry_schedule,
        quotes,
        same_option_type=True,
        target_abs_delta=abs(float(first["delta"])),
        min_future_marks=1,
    )
    book = option_backtest.hedge_book_from_schedules(quotes, [matched], lookahead_days=3)
    comparison = option_backtest.scheduled_hedge_comparison(
        {
            "base": {
                "summary": pd.DataFrame(
                    [
                        {
                            "strategy": "matched",
                            "total_pnl": 25.0,
                            "total_costs": 2.0,
                            "traded_notional": 500.0,
                        }
                    ]
                ),
                "components": pd.DataFrame(
                    [
                        {
                            "strategy": "matched",
                            "episode_id": 1,
                            "date": first["date"],
                            "main_pos": -1.0,
                            "option_mid": 2.0,
                            "vega_after": 30.0,
                            "delta_after": 0.15,
                        }
                    ]
                ),
            }
        }
    )
    exposures = {
        "exposures": pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "strategy": ["matched", "matched"],
                "residual_delta_equiv": [0.20, -0.10],
                "residual_vega_equiv": [5.0, -3.0],
            }
        ),
        "diagnostics": {"n_missing_option_marks": 0},
        "trades": matched,
        "summary": comparison,
    }

    assert not matched.empty
    assert matched.loc[0, "quantity"] < 0.0
    assert not book.empty
    assert comparison.loc[0, "pnl_per_premium"] > 0.0
    assert option_backtest.rolling_residual_delta(exposures).iloc[-1, 0] >= 0.0
    assert option_backtest.rolling_residual_vega(exposures).iloc[-1, 0] >= 0.0
    assert option_backtest.hedging_diagnostics(exposures).loc[0, "n_missing_option_marks"] == 0
    assert option_backtest.hedge_trade_ledger(exposures).equals(matched)
    assert option_backtest.summarize_hedging_backtest(exposures).equals(comparison)


def test_scheduled_option_hedging_backtest_opens_delta_hedge_and_summarizes() -> None:
    dates = pd.bdate_range("2024-01-02", periods=6)
    expiry = pd.Timestamp("2024-02-16")
    rows = []
    for i, date in enumerate(dates):
        mid = 4.0 + 0.15 * i
        spot = 100.0 + 0.7 * i
        rows.append(
            {
                "date": date,
                "timestamp": date + pd.Timedelta(hours=16),
                "expiry": expiry,
                "strike": 100.0,
                "option_type": "call",
                "contract_key": "call_2024-02-16_100",
                "bid": mid - 0.05,
                "ask": mid + 0.05,
                "mid": mid,
                "half_spread": 0.05,
                "spot": spot,
                "delta": 0.55 + 0.01 * i,
                "gamma": 0.025,
                "vega": 12.0 - 0.2 * i,
                "rate": 0.03,
                "dte": (expiry - date).days,
                "price_residual": 0.40 - 0.05 * i,
            }
        )
    option_path = pd.DataFrame(rows)
    entry_schedule = pd.DataFrame(
        [
            {
                "entry_date": dates[0],
                "contract_key": "call_2024-02-16_100",
                "quantity": 1.0,
                "label": "residual_call",
                "max_hold_days": 3,
                "exit_on_convergence": True,
                "entry_residual": 0.40,
                "entry_total_error": 0.15,
            }
        ]
    )
    spot_series = option_path.set_index("date")["spot"]

    result = option_backtest.run_scheduled_option_hedging_backtest(
        option_path,
        entry_schedule,
        spot_series=spot_series,
        strategies=("unhedged", "delta", "unsupported"),
        delta_band=0.05,
        delta_cooldown_days=0,
        trading_cost_bps=0.0,
        use_bid_ask_costs=False,
        option_multiplier=100.0,
        missing_option_mark="error",
    )

    assert {"nav", "returns", "pnl", "components", "exposures", "trades", "summary", "diagnostics"}.issubset(result)
    assert {"unhedged", "delta"}.issubset(result["nav"].columns)
    assert result["diagnostics"]["n_opened_episodes"] == 2
    assert result["diagnostics"]["n_unique_main_contracts"] == 1
    assert result["trades"]["instrument"].isin(["main_option", "underlying"]).all()
    assert result["summary"].set_index("strategy").loc["unsupported", "status"] == "skipped"
    assert option_backtest.summarize_hedging_backtest(result).equals(result["summary"])


def test_overlay_schedules_marking_and_summary_tables() -> None:
    quotes = _quotes_for_backtest()
    underlying = quotes.groupby("date")["spot"].median()
    covered = overlays.covered_call_schedule(quotes, contracts=2.0, rebalance_every=2)
    protective = overlays.protective_put_schedule(quotes, contracts=1.0, rebalance_every=2)
    collar = overlays.collar_schedule(quotes, contracts=1.0, rebalance_every=2)
    rolls = overlays.boundary_roll_schedule(quotes, threshold=0.30)
    marked = overlays.mark_book_for_schedules(quotes, {"covered": covered, "protective": protective})
    defense = overlays.assignment_defense_actions(quotes, threshold=0.30, ex_div_days=7)
    nav = pd.DataFrame({"covered": [1_000_000.0, 1_000_200.0, 999_800.0]}, index=underlying.index)
    trades = pd.DataFrame(
        {
            "strategy": ["covered", "covered", "covered"],
            "event": ["open", "roll_close", "expiry_settlement"],
            "cashflow": [200.0, -110.0, 0.0],
            "spread_cost": [1.0, 1.2, 0.0],
            "holding_days": [0, 12, 30],
            "dte_days": [45, 33, 0],
            "moneyness": [1.02, 1.01, 1.00],
            "price": [2.0, 1.1, 0.0],
            "vrp_rank": [0.1, 0.5, 0.9],
            "net_pnl": [20.0, -5.0, 10.0],
        },
        index=nav.index,
    )
    results = {
        "nav": nav,
        "drawdown": nav - nav.cummax(),
        "trades": trades,
        "call_holdings": pd.DataFrame({"covered": [2.0, 1.0, 0.0]}, index=nav.index),
        "put_holdings": pd.DataFrame({"covered": [0.0, 0.0, 0.0]}, index=nav.index),
    }

    summary = overlays.overlay_summary(results)
    mechanics = overlays.overlay_mechanics_table(results, shares=10.0, dividends=pd.Series(0.10, index=nav.index))
    deciles = overlays.pnl_by_vrp_decile(trades, n_deciles=5)

    assert not covered.empty
    assert not protective.empty
    assert {"collar_call", "collar_put"}.issubset(set(collar["label"]))
    assert rolls["roll"].all()
    assert not marked.empty
    assert not defense.empty
    assert overlays.size_long_straddle_by_premium_budget(100_000, 2.0, 2.0, budget_frac=0.01)[0] > 0.0
    assert overlays.size_short_straddle_by_margin_cap(100_000, 100.0, 2.0, 2.0)[0] > 0.0
    assert summary.loc[0, "strategy"] == "covered"
    assert mechanics.loc[0, "opens"] == 1
    assert deciles["n_trades"].sum() == len(trades)
