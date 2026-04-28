from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.options import quote_cleaning


def _as_price_series(underlying: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(underlying, pd.DataFrame):
        series = quote_cleaning.prepare_underlying_series(underlying)
    else:
        series = pd.Series(underlying, copy=True)
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = pd.to_numeric(series, errors="coerce")
    series = series.dropna().sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).last()
    return series[series > 0]


def prepare_option_book(option_quotes: pd.DataFrame) -> pd.DataFrame:
    book = quote_cleaning.wide_option_chain_to_long(option_quotes, include_greeks=False)
    book = quote_cleaning.ensure_option_mid_quotes(book)
    book = quote_cleaning.convert_quotes_to_usd_equivalent(book, unit="auto")
    if "dte" not in book.columns or "tau" not in book.columns:
        book = quote_cleaning.add_time_to_expiry(book, annualization_days=365.25)
    book["date"] = pd.to_datetime(book["date"], errors="coerce").dt.normalize()
    book["expiry"] = pd.to_datetime(book["expiry"], errors="coerce").dt.normalize()
    book["option_type"] = book["option_type"].map(quote_cleaning.parse_option_type)
    for col in ["strike", "bid", "ask", "mid", "spot", "dte"]:
        if col in book.columns:
            book[col] = pd.to_numeric(book[col], errors="coerce")
    book = book.dropna(subset=["date", "expiry", "strike", "option_type", "mid"])
    book["strike_key"] = book["strike"].round(8)
    return book.sort_values(["expiry", "strike_key", "option_type", "date"]).reset_index(drop=True)


def select_atm_straddle(panel: pd.DataFrame, min_dte: int = 21, max_dte: int = 45) -> pd.DataFrame:
    data = panel.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["expiry"] = pd.to_datetime(data["expiry"], errors="coerce").dt.normalize()
    if "dte_calendar" not in data.columns and "dte" in data.columns:
        data["dte_calendar"] = data["dte"]
    data["dte_calendar"] = pd.to_numeric(data["dte_calendar"], errors="coerce")
    data = data[(data["dte_calendar"] >= min_dte) & (data["dte_calendar"] <= max_dte)].copy()
    if data.empty:
        return data
    data["_dte_score"] = (data["dte_calendar"] - 30.0).abs()
    q_col = "quote_quality_score" if "quote_quality_score" in data.columns else "_dte_score"
    return data.sort_values(["date", "_dte_score", q_col]).groupby("date", as_index=False).head(1).drop(columns="_dte_score")


def size_long_straddle_by_premium_budget(
    nav: float,
    call_ask: float,
    put_ask: float,
    *,
    contract_multiplier: float = 100.0,
    budget_frac: float = 0.005,
    fractional_units: bool = False,
    max_units: float | None = None,
) -> tuple[float, float, str]:
    unit_premium = (float(call_ask) + float(put_ask)) * float(contract_multiplier)
    if not np.isfinite(unit_premium) or unit_premium <= 0:
        return 0.0, unit_premium, "invalid_long_premium"
    units = float(nav) * float(budget_frac) / unit_premium
    if not fractional_units:
        units = np.floor(units)
    if max_units is not None:
        units = min(units, float(max_units))
    return float(units), float(unit_premium), "" if units > 0 else "long_budget_too_small"


def size_short_straddle_by_margin_cap(
    nav: float,
    spot: float,
    call_mid: float,
    put_mid: float,
    *,
    contract_multiplier: float = 100.0,
    short_margin_spot_frac: float = 0.15,
    short_margin_budget_frac: float = 0.02,
    max_short_notional_frac: float = 1.0,
    fractional_units: bool = False,
    max_units: float | None = None,
) -> tuple[float, float, float, str]:
    unit_margin = float(contract_multiplier) * (
        float(short_margin_spot_frac) * float(spot) + float(call_mid) + float(put_mid)
    )
    unit_notional = float(spot) * float(contract_multiplier)
    if not np.isfinite(unit_margin) or unit_margin <= 0 or not np.isfinite(unit_notional) or unit_notional <= 0:
        return 0.0, unit_margin, unit_notional, "invalid_short_cap"
    units_margin = float(nav) * float(short_margin_budget_frac) / unit_margin
    units_notional = float(nav) * float(max_short_notional_frac) / unit_notional
    units = min(units_margin, units_notional)
    if not fractional_units:
        units = np.floor(units)
    if max_units is not None:
        units = min(units, float(max_units))
    return float(units), float(unit_margin), float(unit_notional), "" if units > 0 else "short_cap_too_small"


def find_straddle_exit(
    option_book: pd.DataFrame,
    *,
    entry_date: pd.Timestamp,
    target_exit_date: pd.Timestamp,
    expiry: pd.Timestamp,
    strike: float,
    max_exit_lag: int = 2,
) -> tuple[pd.Series | None, pd.Series | None, str]:
    strike_key = round(float(strike), 8)
    subset = option_book[
        (option_book["expiry"] == pd.Timestamp(expiry).normalize())
        & (option_book["strike_key"] == strike_key)
        & (option_book["date"] > pd.Timestamp(entry_date).normalize())
        & (option_book["date"] >= pd.Timestamp(target_exit_date).normalize())
    ].copy()
    if subset.empty:
        return None, None, "missing_exit_contract"
    dates = pd.DatetimeIndex(sorted(subset["date"].unique()))
    target = pd.Timestamp(target_exit_date).normalize()
    target_pos = dates.searchsorted(target, side="left")
    if target_pos >= len(dates):
        return None, None, "missing_exit_within_lag"
    exit_date = dates[target_pos]
    quote_dates = pd.DatetimeIndex(sorted(option_book["date"].dropna().unique()))
    target_rank = quote_dates.searchsorted(target, side="left")
    exit_rank = quote_dates.searchsorted(pd.Timestamp(exit_date).normalize(), side="left")
    if exit_rank - target_rank > int(max_exit_lag):
        return None, None, "missing_exit_within_lag"
    day = subset[subset["date"] == exit_date]
    call = day[day["option_type"] == "call"]
    put = day[day["option_type"] == "put"]
    if call.empty or put.empty:
        return None, None, "missing_exit_leg"
    return call.iloc[0], put.iloc[0], "exact_exit" if exit_date == pd.Timestamp(target_exit_date).normalize() else "nearest_later_exit"


def backtest_straddle_overlay(
    signal_panel: pd.DataFrame,
    option_quotes: pd.DataFrame,
    underlying: pd.Series | pd.DataFrame,
    *,
    strategy_name: str,
    signal_rule,
    initial_nav: float = 1_000_000,
    contract_multiplier: float = 100.0,
    holding_days: int = 5,
    allow_overlap: bool = False,
    fractional_units: bool = False,
    max_units: float | None = None,
    long_premium_budget_frac: float = 0.005,
    short_margin_spot_frac: float = 0.15,
    short_margin_budget_frac: float = 0.02,
    max_short_notional_frac: float = 1.0,
    max_exit_lag: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = _as_price_series(underlying)
    dates = pd.Index(prices.index)
    book = prepare_option_book(option_quotes)
    panel = select_atm_straddle(signal_panel).sort_values("date").reset_index(drop=True)
    trades: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    open_until = pd.Timestamp.min
    nav = float(initial_nav)

    for _, row in panel.iterrows():
        entry_date = pd.Timestamp(row["date"]).normalize()
        if (not allow_overlap) and entry_date <= open_until:
            skipped.append({"strategy": strategy_name, "date": entry_date, "skip_reason": "open_position"})
            continue
        decision = signal_rule(row)
        side = decision.get("side", "cash") if isinstance(decision, dict) else decision
        if side not in {"long", "short"}:
            skip_reason = decision.get("skip_reason", "no_signal") if isinstance(decision, dict) else "no_signal"
            skipped.append({"strategy": strategy_name, "date": entry_date, "skip_reason": skip_reason})
            continue
        if entry_date not in dates:
            skipped.append({"strategy": strategy_name, "date": entry_date, "skip_reason": "missing_underlying_date"})
            continue
        pos = dates.get_loc(entry_date)
        if pos + int(holding_days) >= len(dates):
            skipped.append({"strategy": strategy_name, "date": entry_date, "skip_reason": "not_enough_future_dates"})
            continue
        target_exit_date = pd.Timestamp(dates[pos + int(holding_days)]).normalize()

        call_bid, call_ask = float(row["call_bid"]), float(row["call_ask"])
        put_bid, put_ask = float(row["put_bid"]), float(row["put_ask"])
        call_mid, put_mid = float(row["call_mid"]), float(row["put_mid"])
        spot = float(row["spot"])
        if side == "long":
            units, unit_cap, skip = size_long_straddle_by_premium_budget(
                nav,
                call_ask,
                put_ask,
                contract_multiplier=contract_multiplier,
                budget_frac=long_premium_budget_frac,
                fractional_units=fractional_units,
                max_units=max_units,
            )
            margin_proxy = np.nan
            notional = units * spot * contract_multiplier
            entry_price = call_ask + put_ask
            entry_cashflow = -units * entry_price * contract_multiplier
            cap_used = units * unit_cap
        else:
            units, margin_proxy_unit, notional_unit, skip = size_short_straddle_by_margin_cap(
                nav,
                spot,
                call_mid,
                put_mid,
                contract_multiplier=contract_multiplier,
                short_margin_spot_frac=short_margin_spot_frac,
                short_margin_budget_frac=short_margin_budget_frac,
                max_short_notional_frac=max_short_notional_frac,
                fractional_units=fractional_units,
                max_units=max_units,
            )
            margin_proxy = units * margin_proxy_unit
            notional = units * notional_unit
            entry_price = call_bid + put_bid
            entry_cashflow = units * entry_price * contract_multiplier
            cap_used = margin_proxy
        if units <= 0:
            skipped.append({"strategy": strategy_name, "date": entry_date, "side": side, "skip_reason": skip})
            continue

        exit_call, exit_put, exit_flag = find_straddle_exit(
            book,
            entry_date=entry_date,
            target_exit_date=target_exit_date,
            expiry=pd.Timestamp(row["expiry"]),
            strike=float(row["strike"]),
            max_exit_lag=max_exit_lag,
        )
        if exit_call is None or exit_put is None:
            skipped.append({"strategy": strategy_name, "date": entry_date, "side": side, "skip_reason": exit_flag})
            continue

        exit_date = pd.Timestamp(exit_call["date"]).normalize()
        if side == "long":
            exit_price = float(exit_call["bid"]) + float(exit_put["bid"])
            exit_cashflow = units * exit_price * contract_multiplier
        else:
            exit_price = float(exit_call["ask"]) + float(exit_put["ask"])
            exit_cashflow = -units * exit_price * contract_multiplier
        pnl = entry_cashflow + exit_cashflow
        nav += pnl
        open_until = exit_date
        denom = cap_used if side == "short" else abs(entry_cashflow)
        trades.append(
            {
                "strategy": strategy_name,
                "side": side,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "expiry": pd.Timestamp(row["expiry"]).normalize(),
                "strike": float(row["strike"]),
                "dte_entry": float(row.get("dte_calendar", np.nan)),
                "dte_exit": float(exit_call.get("dte", np.nan)),
                "spot_entry": spot,
                "spot_exit": float(exit_call.get("spot", np.nan)),
                "atm_iv_entry": float(row.get("atm_iv_mid", np.nan)),
                "forecast_vol_entry": float(row.get("forecast_vol_ann", np.nan)),
                "forecast_var_entry": float(row.get("forecast_var_ann", np.nan)),
                "vrp_var": float(row.get("vrp_var", np.nan)),
                "vrp_z": float(row.get("vrp_z", np.nan)),
                "vrp_rank": float(row.get("vrp_rank", np.nan)),
                "selected_model": row.get("selected_model", np.nan),
                "units": float(units),
                "contract_multiplier": float(contract_multiplier),
                "entry_call_price": call_ask if side == "long" else call_bid,
                "entry_put_price": put_ask if side == "long" else put_bid,
                "exit_call_price": float(exit_call["bid"] if side == "long" else exit_call["ask"]),
                "exit_put_price": float(exit_put["bid"] if side == "long" else exit_put["ask"]),
                "entry_cashflow": float(entry_cashflow),
                "exit_cashflow": float(exit_cashflow),
                "gross_pnl": float(pnl),
                "net_pnl": float(pnl),
                "premium_paid_or_received": float(abs(entry_cashflow) if side == "long" else entry_cashflow),
                "margin_proxy": float(margin_proxy) if np.isfinite(margin_proxy) else np.nan,
                "notional_exposure": float(notional),
                "cap_used": float(cap_used),
                "return_on_premium_or_margin": float(pnl / denom) if denom and np.isfinite(denom) else np.nan,
                "entry_flag": "entered",
                "exit_flag": exit_flag,
                "skip_reason": "",
            }
        )

    trades_df = pd.DataFrame(trades)
    skipped_df = pd.DataFrame(skipped)
    if trades_df.empty:
        equity = pd.DataFrame({"date": panel["date"].drop_duplicates().sort_values()})
        equity["strategy_pnl"] = 0.0
        equity["nav"] = float(initial_nav)
    else:
        eq_dates = dates[(dates >= trades_df["entry_date"].min()) & (dates <= trades_df["exit_date"].max())]
        equity = pd.DataFrame({"date": eq_dates})
        pnl_by_date = trades_df.groupby("exit_date")["net_pnl"].sum()
        equity["strategy_pnl"] = equity["date"].map(pnl_by_date).fillna(0.0)
        equity["nav"] = float(initial_nav) + equity["strategy_pnl"].cumsum()
    equity["running_max_nav"] = equity["nav"].cummax()
    equity["drawdown"] = equity["nav"] / equity["running_max_nav"] - 1.0
    equity["strategy"] = strategy_name
    return trades_df, equity, skipped_df


def summarize_overlay_trades(backtests: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], initial_nav: float = 1_000_000) -> pd.DataFrame:
    rows = []
    for name, (trades, equity, _) in backtests.items():
        nav = equity["nav"] if not equity.empty else pd.Series([initial_nav])
        daily = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            "strategy": name,
            "total_net_pnl": float(nav.iloc[-1] - initial_nav),
            "return_on_initial_nav": float(nav.iloc[-1] / initial_nav - 1.0),
            "ann_vol": float(daily.std(ddof=1) * np.sqrt(252.0)) if len(daily) > 1 else np.nan,
            "sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252.0)) if len(daily) > 1 and daily.std(ddof=1) > 0 else np.nan,
            "max_drawdown": float(equity["drawdown"].min()) if "drawdown" in equity else np.nan,
            "n_trades": int(len(trades)),
        }
        if not trades.empty:
            row.update(
                {
                    "hit_rate": float((trades["net_pnl"] > 0).mean()),
                    "avg_trade_pnl": float(trades["net_pnl"].mean()),
                    "median_trade_pnl": float(trades["net_pnl"].median()),
                    "best_trade": float(trades["net_pnl"].max()),
                    "worst_trade": float(trades["net_pnl"].min()),
                    "avg_units": float(trades["units"].mean()),
                    "avg_premium_paid_or_received": float(trades["premium_paid_or_received"].mean()),
                    "avg_margin_proxy": float(trades["margin_proxy"].mean()),
                    "avg_notional_exposure": float(trades["notional_exposure"].mean()),
                    "avg_vrp_rank": float(trades["vrp_rank"].mean()),
                    "avg_vrp_z": float(trades["vrp_z"].mean()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def pnl_by_vrp_decile(trades: pd.DataFrame, n_deciles: int = 10) -> pd.DataFrame:
    if trades.empty or "vrp_rank" not in trades.columns:
        return pd.DataFrame()
    data = trades.replace([np.inf, -np.inf], np.nan).dropna(subset=["vrp_rank", "net_pnl"]).copy()
    if data.empty:
        return pd.DataFrame()
    data["vrp_decile"] = pd.cut(data["vrp_rank"], bins=np.linspace(0, 1, n_deciles + 1), include_lowest=True)
    return data.groupby("vrp_decile", observed=True).agg(
        n_trades=("net_pnl", "size"),
        total_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean"),
        hit_rate=("net_pnl", lambda x: float((x > 0).mean())),
    ).reset_index()


__all__ = [
    "backtest_straddle_overlay",
    "find_straddle_exit",
    "pnl_by_vrp_decile",
    "prepare_option_book",
    "select_atm_straddle",
    "size_long_straddle_by_premium_budget",
    "size_short_straddle_by_margin_cap",
    "summarize_overlay_trades",
]
