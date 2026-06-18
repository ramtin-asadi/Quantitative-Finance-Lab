from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.options import quote_cleaning


def _as_price_series(underlying: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(underlying, pd.DataFrame):
        data = underlying.copy()
        lookup = {str(c).strip().lower().replace(" ", "_"): c for c in data.columns}
        date_col = lookup.get("date", data.columns[0])
        price_col = None
        for candidate in ["adj_close", "close", "price", "spot"]:
            if candidate in lookup:
                price_col = lookup[candidate]
                break
        if price_col is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("Could not infer a numeric underlying price column.")
            price_col = numeric_cols[0]
        series = pd.Series(
            pd.to_numeric(data[price_col], errors="coerce").to_numpy(dtype=float),
            index=pd.to_datetime(data[date_col], errors="coerce").dt.normalize(),
            name=str(price_col),
        )
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


def _contract_key_from_book(book: pd.DataFrame) -> pd.Series:
    expiry = pd.to_datetime(book["expiry"], errors="coerce").dt.strftime("%Y-%m-%d")
    return book["option_type"].astype(str) + "_" + expiry + "_" + pd.to_numeric(book["strike"], errors="coerce").round(6).astype(str)


def _numeric_column(frame: pd.DataFrame, names, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def _rebalance_dates(dates: pd.Series, rebalance_every: int | None = None, dates_allowed=None) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if dates_allowed is not None:
        allowed = pd.Index(pd.to_datetime(pd.Series(dates_allowed), errors="coerce").dt.normalize().dropna().unique())
        return d.isin(allowed)
    if rebalance_every is None or int(rebalance_every) <= 1:
        return pd.Series(True, index=d.index)
    keep = []
    last = None
    for x in d:
        if pd.isna(x):
            keep.append(False)
        elif last is None or (x - last).days >= int(rebalance_every):
            keep.append(True)
            last = x
        else:
            keep.append(False)
    return pd.Series(keep, index=d.index)


def covered_call_schedule(
    quotes: pd.DataFrame,
    *,
    target_delta: float = 0.30,
    min_dte: int = 21,
    max_dte: int = 60,
    contracts: float = 1.0,
    rebalance_every: int | None = None,
    dates_allowed=None,
    risk_weight: float = 0.0,
) -> pd.DataFrame:
    q = quotes.copy()
    if "contract_key" not in q.columns:
        q["contract_key"] = _contract_key_from_book(q)
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q.get("dte", q.get("tau", 0.0) * 365.25), errors="coerce")
    q = q[q["option_type"].astype(str).str.lower().str.startswith("c") & q["dte_days"].between(min_dte, max_dte)].copy()
    if q.empty:
        return pd.DataFrame(columns=["entry_date", "contract_key", "quantity", "label"])
    q["delta_score"] = (_numeric_column(q, ("delta",), target_delta).abs() - float(target_delta)).abs()
    q["spread_score"] = _numeric_column(q, ("rel_spread", "relative_spread"), 0.0)
    q["selection_score"] = q["delta_score"] + q["spread_score"] + float(risk_weight) * _numeric_column(q, ("assignment_risk",), 0.0)
    picks = q.sort_values(["date", "selection_score", "spread_score"]).groupby("date", as_index=False).head(1)
    picks = picks.loc[_rebalance_dates(picks["date"], rebalance_every, dates_allowed)].copy()
    return pd.DataFrame({"entry_date": pd.to_datetime(picks["date"]).dt.normalize(), "contract_key": picks["contract_key"], "quantity": -abs(float(contracts)), "label": "covered_call"})


def protective_put_schedule(
    quotes: pd.DataFrame,
    *,
    target_delta: float = 0.25,
    min_dte: int = 21,
    max_dte: int = 75,
    contracts: float = 1.0,
    rebalance_every: int | None = None,
    dates_allowed=None,
    value_weight: float = 0.0,
) -> pd.DataFrame:
    q = quotes.copy()
    if "contract_key" not in q.columns:
        q["contract_key"] = _contract_key_from_book(q)
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q.get("dte", q.get("tau", 0.0) * 365.25), errors="coerce")
    q = q[q["option_type"].astype(str).str.lower().str.startswith("p") & q["dte_days"].between(min_dte, max_dte)].copy()
    if q.empty:
        return pd.DataFrame(columns=["entry_date", "contract_key", "quantity", "label"])
    q["delta_score"] = (_numeric_column(q, ("delta",), -target_delta).abs() - float(target_delta)).abs()
    edge = _numeric_column(q, ("american_tree_price", "model_price"), 0.0) - _numeric_column(q, ("ask", "mid"), 0.0)
    q["cost_score"] = _numeric_column(q, ("mid",), 0.0) + _numeric_column(q, ("rel_spread", "relative_spread"), 0.0) - float(value_weight) * edge
    picks = q.sort_values(["date", "delta_score", "cost_score"]).groupby("date", as_index=False).head(1)
    picks = picks.loc[_rebalance_dates(picks["date"], rebalance_every, dates_allowed)].copy()
    return pd.DataFrame({"entry_date": pd.to_datetime(picks["date"]).dt.normalize(), "contract_key": picks["contract_key"], "quantity": abs(float(contracts)), "label": "protective_put"})


def collar_schedule(
    quotes: pd.DataFrame,
    *,
    put_delta: float = 0.25,
    call_delta: float = 0.25,
    contracts: float = 1.0,
    rebalance_every: int | None = None,
    dates_allowed=None,
    call_risk_weight: float = 0.0,
    put_value_weight: float = 0.0,
) -> pd.DataFrame:
    put = protective_put_schedule(quotes, target_delta=put_delta, contracts=contracts, rebalance_every=rebalance_every, dates_allowed=dates_allowed, value_weight=put_value_weight)
    call = covered_call_schedule(quotes, target_delta=call_delta, contracts=contracts, rebalance_every=rebalance_every, dates_allowed=dates_allowed, risk_weight=call_risk_weight)
    if put.empty and call.empty:
        return pd.DataFrame(columns=["entry_date", "contract_key", "quantity", "label"])
    out = pd.concat([put.assign(label="collar_put"), call.assign(label="collar_call")], ignore_index=True)
    return out.sort_values(["entry_date", "label"]).reset_index(drop=True)


def boundary_roll_schedule(quotes: pd.DataFrame, *, risk_col: str = "assignment_risk", threshold: float = 1.0) -> pd.DataFrame:
    q = quotes.copy()
    if "contract_key" not in q.columns:
        q["contract_key"] = _contract_key_from_book(q)
    if risk_col not in q.columns:
        q[risk_col] = pd.to_numeric(q.get("roll_urgency", 0.0), errors="coerce").fillna(0.0)
    q = q[pd.to_numeric(q[risk_col], errors="coerce").fillna(0.0) >= float(threshold)].copy()
    return pd.DataFrame({"date": pd.to_datetime(q.get("date"), errors="coerce").dt.normalize(), "contract_key": q.get("contract_key"), "roll": True, "risk": q[risk_col]})


def run_overlay_backtest(
    schedules: dict[str, pd.DataFrame],
    quotes: pd.DataFrame,
    underlying: pd.Series,
    *,
    initial_nav: float = 1_000_000.0,
    contract_multiplier: float = 100.0,
    shares: float = 0.0,
    dividends: pd.Series | None = None,
    assignment_risk_threshold: float = 0.25,
    assignment_defense_strategies: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict:
    prices = pd.Series(underlying).copy()
    prices.index = pd.to_datetime(prices.index, errors="coerce").normalize()
    dates = pd.Index(sorted(prices.dropna().index.unique()))
    clean_schedules = {}
    schedules_have_terms = False
    for name, schedule in schedules.items():
        if schedule is None:
            clean_schedules[name] = schedule
            continue
        sched = schedule.copy()
        if {"expiry", "strike", "option_type"}.issubset(sched.columns):
            sched["contract_key"] = _contract_key_from_book(sched)
            schedules_have_terms = True
        clean_schedules[name] = sched
    schedules = clean_schedules
    nav_frames = []
    cash_frames = []
    mark_frames = []
    holding_frames = []
    call_frames = []
    put_frames = []
    trade_rows = []
    book = quotes.copy()
    if schedules_have_terms and {"expiry", "strike", "option_type"}.issubset(book.columns) or "contract_key" not in book.columns:
        book["contract_key"] = _contract_key_from_book(book)
    scheduled_keys = set()
    for schedule in schedules.values():
        if schedule is not None and not schedule.empty and "contract_key" in schedule.columns:
            scheduled_keys.update(schedule["contract_key"].astype(str).dropna().unique().tolist())
    if scheduled_keys:
        book = book[book["contract_key"].astype(str).isin(scheduled_keys)].copy()
    keep_cols = [
        "date",
        "expiry",
        "contract_key",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "spot",
        "dte_days",
        "moneyness",
        "contract_multiplier",
        "contract_size",
        "assignment_risk",
        "days_to_next_dividend",
    ]
    book = book[[c for c in keep_cols if c in book.columns]].copy()
    book["date"] = pd.to_datetime(book["date"], errors="coerce").dt.normalize()
    book["expiry"] = pd.to_datetime(book["expiry"], errors="coerce").dt.normalize()
    if dividends is None:
        div_series = pd.Series(0.0, index=dates)
    else:
        div_series = pd.Series(dividends).copy()
        div_series.index = pd.to_datetime(div_series.index, errors="coerce").normalize()
        div_series = div_series.groupby(level=0).sum().reindex(dates).fillna(0.0)
    book_by_key = {str(k): g.sort_values("date") for k, g in book.groupby("contract_key", sort=False)}
    date_key = {(pd.Timestamp(r["date"]).normalize(), str(r["contract_key"])): r for _, r in book.iterrows()}
    if assignment_defense_strategies is None:
        defense_names = {str(name) for name in schedules if ("boundary" in str(name).lower() or "aware" in str(name).lower())}
    else:
        defense_names = {str(name) for name in assignment_defense_strategies}
    for name, schedule in schedules.items():
        cash = float(initial_nav)
        if float(shares) != 0.0 and len(dates):
            cash -= float(shares) * float(prices.loc[dates[0]])
        positions: list[dict] = []
        nav_values = []
        cash_values = []
        mark_values = []
        holding_values = []
        call_values = []
        put_values = []
        if schedule is None or schedule.empty:
            for d in dates:
                cash += float(shares) * float(div_series.loc[d])
                nav_values.append(cash + float(shares) * float(prices.loc[d]))
                cash_values.append(cash)
                mark_values.append(0.0)
                holding_values.append(len(positions))
                call_values.append(0.0)
                put_values.append(0.0)
            nav = pd.Series(nav_values, index=dates, name=name)
            nav_frames.append(nav)
            cash_frames.append(pd.Series(cash_values, index=dates, name=name))
            mark_frames.append(pd.Series(mark_values, index=dates, name=name))
            holding_frames.append(pd.Series(holding_values, index=dates, name=name))
            call_frames.append(pd.Series(call_values, index=dates, name=name))
            put_frames.append(pd.Series(put_values, index=dates, name=name))
            continue
        sched = schedule.copy()
        sched["entry_date"] = pd.to_datetime(sched["entry_date"], errors="coerce").dt.normalize()
        sched_by_date = {d: g for d, g in sched.groupby("entry_date", sort=False)}
        for d in dates:
            cash += float(shares) * float(div_series.loc[d])
            new_positions = []
            for pos in positions:
                key = str(pos["contract_key"])
                mark = date_key.get((d, key))
                expiry = pd.Timestamp(pos["expiry"]).normalize()
                if d >= expiry:
                    settle = max(float(prices.loc[d]) - float(pos["strike"]), 0.0) if str(pos["option_type"]).lower().startswith("c") else max(float(pos["strike"]) - float(prices.loc[d]), 0.0)
                    cashflow = float(pos["quantity"]) * settle * float(pos["multiplier"])
                    cash += cashflow
                    trade_rows.append({"strategy": name, "date": d, "event": "expiry_settlement", "contract_key": key, "quantity": pos["quantity"], "cashflow": cashflow, "price": settle, "entry_date": pos.get("entry_date"), "holding_days": (d - pd.Timestamp(pos.get("entry_date", d))).days, "entry_price": pos.get("entry_price"), "strike": pos.get("strike"), "expiry": pos.get("expiry"), "option_type": pos.get("option_type"), "label": pos.get("label")})
                    continue
                if mark is not None:
                    risk = float(pd.to_numeric(mark.get("assignment_risk", 0.0), errors="coerce"))
                    days = float(pd.to_numeric(mark.get("days_to_next_dividend", np.inf), errors="coerce"))
                    if str(name) in defense_names and float(pos["quantity"]) < 0 and str(pos["option_type"]).lower().startswith("c") and risk >= float(assignment_risk_threshold) and days <= 7.0:
                        price = float(pd.to_numeric(mark.get("ask", mark.get("mid")), errors="coerce"))
                        cashflow = float(pos["quantity"]) * price * float(pos["multiplier"])
                        cash += cashflow
                        pnl = cashflow + float(pos["entry_cashflow"])
                        trade_rows.append({"strategy": name, "date": d, "event": "assignment_defense_close", "contract_key": key, "quantity": pos["quantity"], "cashflow": cashflow, "price": price, "pnl": pnl, "entry_date": pos.get("entry_date"), "holding_days": (d - pd.Timestamp(pos.get("entry_date", d))).days, "entry_price": pos.get("entry_price"), "strike": pos.get("strike"), "expiry": pos.get("expiry"), "option_type": pos.get("option_type"), "label": pos.get("label")})
                        continue
                    pos["last_mid"] = float(pd.to_numeric(mark.get("mid", pos.get("last_mid", 0.0)), errors="coerce"))
                new_positions.append(pos)
            positions = new_positions
            if d in sched_by_date:
                for _, row in sched_by_date[d].iterrows():
                    key = str(row["contract_key"])
                    entry = date_key.get((d, key))
                    if entry is None:
                        marks = book_by_key.get(key)
                        if marks is None:
                            continue
                        future = marks[marks["date"] >= d]
                        if future.empty:
                            continue
                        entry = future.iloc[0]
                    label = str(row.get("label", key))
                    keep = []
                    for pos in positions:
                        if str(pos.get("label")) == label:
                            mark = date_key.get((d, str(pos["contract_key"])))
                            if mark is not None:
                                close_price = float(pd.to_numeric(mark.get("bid" if pos["quantity"] > 0 else "ask", mark.get("mid")), errors="coerce"))
                                cashflow = float(pos["quantity"]) * close_price * float(pos["multiplier"])
                                cash += cashflow
                                pnl = cashflow + float(pos["entry_cashflow"])
                                trade_rows.append({"strategy": name, "date": d, "event": "roll_close", "contract_key": pos["contract_key"], "quantity": pos["quantity"], "cashflow": cashflow, "price": close_price, "pnl": pnl, "entry_date": pos.get("entry_date"), "holding_days": (d - pd.Timestamp(pos.get("entry_date", d))).days, "entry_price": pos.get("entry_price"), "strike": pos.get("strike"), "expiry": pos.get("expiry"), "option_type": pos.get("option_type"), "label": pos.get("label")})
                        else:
                            keep.append(pos)
                    positions = keep
                    qty = float(row.get("quantity", 1.0))
                    price = float(pd.to_numeric(entry.get("ask" if qty > 0 else "bid", entry.get("mid")), errors="coerce"))
                    mult = float(entry.get("contract_multiplier", entry.get("contract_size", contract_multiplier)))
                    cashflow = -qty * price * mult
                    cash += cashflow
                    positions.append({"contract_key": key, "quantity": qty, "entry_cashflow": cashflow, "entry_price": price, "entry_date": d, "expiry": entry["expiry"], "strike": float(entry["strike"]), "option_type": str(entry["option_type"]), "multiplier": mult, "last_mid": float(entry.get("mid", price)), "label": label})
                    trade_rows.append({"strategy": name, "date": d, "event": "open", "contract_key": key, "quantity": qty, "cashflow": cashflow, "price": price, "spread_cost": abs(float(entry.get("ask", price)) - float(entry.get("bid", price))) * 0.5 * abs(qty) * mult, "entry_date": d, "holding_days": 0, "entry_price": price, "strike": float(entry["strike"]), "expiry": entry["expiry"], "option_type": str(entry["option_type"]), "label": label, "dte_days": float(pd.to_numeric(entry.get("dte_days", np.nan), errors="coerce")), "moneyness": float(pd.to_numeric(entry.get("moneyness", np.nan), errors="coerce")), "spot": float(pd.to_numeric(entry.get("spot", np.nan), errors="coerce"))})
            option_mark = 0.0
            for pos in positions:
                mark = date_key.get((d, str(pos["contract_key"])))
                if mark is not None:
                    pos["last_mid"] = float(pd.to_numeric(mark.get("mid", pos.get("last_mid", 0.0)), errors="coerce"))
                option_mark += float(pos["quantity"]) * float(pos.get("last_mid", 0.0)) * float(pos["multiplier"])
            nav_values.append(cash + float(shares) * float(prices.loc[d]) + option_mark)
            cash_values.append(cash)
            mark_values.append(option_mark)
            holding_values.append(len(positions))
            call_values.append(float(sum(abs(float(pos["quantity"])) for pos in positions if str(pos.get("option_type", "")).lower().startswith("c"))))
            put_values.append(float(sum(abs(float(pos["quantity"])) for pos in positions if str(pos.get("option_type", "")).lower().startswith("p"))))
        nav = pd.Series(nav_values, index=dates, name=name)
        nav.name = name
        nav_frames.append(nav)
        cash_frames.append(pd.Series(cash_values, index=dates, name=name))
        mark_frames.append(pd.Series(mark_values, index=dates, name=name))
        holding_frames.append(pd.Series(holding_values, index=dates, name=name))
        call_frames.append(pd.Series(call_values, index=dates, name=name))
        put_frames.append(pd.Series(put_values, index=dates, name=name))
    nav = pd.concat(nav_frames, axis=1) if nav_frames else pd.DataFrame(index=dates)
    cash = pd.concat(cash_frames, axis=1) if cash_frames else pd.DataFrame(index=dates)
    option_marks = pd.concat(mark_frames, axis=1) if mark_frames else pd.DataFrame(index=dates)
    holdings = pd.concat(holding_frames, axis=1) if holding_frames else pd.DataFrame(index=dates)
    call_holdings = pd.concat(call_frames, axis=1) if call_frames else pd.DataFrame(index=dates)
    put_holdings = pd.concat(put_frames, axis=1) if put_frames else pd.DataFrame(index=dates)
    if len(dates) and not nav.empty:
        first = dates[0]
        start_nav = pd.DataFrame({col: float(initial_nav) for col in nav.columns}, index=[first])
        start_cash_value = float(initial_nav) - float(shares) * float(prices.loc[first])
        start_cash = pd.DataFrame({col: start_cash_value for col in nav.columns}, index=[first])
        start_zero = pd.DataFrame({col: 0.0 for col in nav.columns}, index=[first])
        nav = pd.concat([start_nav, nav])
        cash = pd.concat([start_cash, cash]) if not cash.empty else start_cash.copy()
        option_marks = pd.concat([start_zero, option_marks]) if not option_marks.empty else start_zero.copy()
        holdings = pd.concat([start_zero, holdings]) if not holdings.empty else start_zero.copy()
        call_holdings = pd.concat([start_zero, call_holdings]) if not call_holdings.empty else start_zero.copy()
        put_holdings = pd.concat([start_zero, put_holdings]) if not put_holdings.empty else start_zero.copy()
    trade_cols = ["strategy", "date", "event", "contract_key", "quantity", "cashflow", "price", "pnl", "entry_date", "holding_days", "entry_price", "strike", "expiry", "option_type", "label", "dte_days", "moneyness", "spot", "spread_cost"]
    trades = pd.DataFrame(trade_rows, columns=trade_cols)
    drawdown = nav / nav.cummax() - 1.0 if not nav.empty else pd.DataFrame()
    return {"nav": nav, "drawdown": drawdown, "trades": trades, "cash": cash, "option_marks": option_marks, "holdings": holdings, "call_holdings": call_holdings, "put_holdings": put_holdings, "summary": overlay_summary({"nav": nav, "drawdown": drawdown, "trades": trades})}


def mark_book_for_schedules(quotes: pd.DataFrame, schedules: dict[str, pd.DataFrame] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(schedules, dict):
        sched = pd.concat([x for x in schedules.values() if x is not None and not x.empty], ignore_index=True) if schedules else pd.DataFrame()
    else:
        sched = schedules.copy()
    if sched.empty:
        return quotes.iloc[:0].copy()
    q = quotes
    cols = [
        "date",
        "expiry",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "spot",
        "dte_days",
        "moneyness",
        "contract_multiplier",
        "contract_size",
        "assignment_risk",
        "days_to_next_dividend",
    ]
    if {"expiry", "strike", "option_type"}.issubset(sched.columns):
        keys = sched[["expiry", "strike", "option_type"]].drop_duplicates().copy()
        keys["expiry"] = pd.to_datetime(keys["expiry"], errors="coerce").dt.normalize()
        keys["strike"] = pd.to_numeric(keys["strike"], errors="coerce")
        keys["option_type"] = keys["option_type"].astype(str)
        expiry = pd.to_datetime(q["expiry"], errors="coerce").dt.normalize()
        strike = pd.to_numeric(q["strike"], errors="coerce")
        option_type = q["option_type"].astype(str)
        mask = expiry.isin(keys["expiry"].unique()) & strike.isin(keys["strike"].unique()) & option_type.isin(keys["option_type"].unique())
        q = q.loc[mask, [c for c in cols if c in q.columns]].copy()
        q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
        q["expiry"] = pd.to_datetime(q["expiry"], errors="coerce").dt.normalize()
        q["strike"] = pd.to_numeric(q["strike"], errors="coerce")
        q["option_type"] = q["option_type"].astype(str)
        out = q[[c for c in cols if c in q.columns]].merge(keys, on=["expiry", "strike", "option_type"], how="inner")
        out["contract_key"] = _contract_key_from_book(out)
        return out.sort_values(["contract_key", "date"]).reset_index(drop=True)
    if "contract_key" not in q.columns:
        q = q.copy()
        q["contract_key"] = _contract_key_from_book(q)
    keys = set(sched["contract_key"].astype(str).dropna().unique())
    out = q[q["contract_key"].astype(str).isin(keys)].copy()
    cols = ["contract_key"] + [c for c in cols if c in out.columns]
    return out[cols].sort_values(["contract_key", "date"]).reset_index(drop=True)


def assignment_defense_actions(quotes: pd.DataFrame, *, threshold: float = 0.25, ex_div_days: int = 7) -> pd.DataFrame:
    q = quotes.copy()
    risk = pd.to_numeric(q.get("assignment_risk", 0.0), errors="coerce").fillna(0.0)
    days = pd.to_numeric(q.get("days_to_next_dividend", np.inf), errors="coerce").fillna(np.inf)
    calls = q.get("option_type", "").astype(str).str.lower().str.startswith("c")
    out = q.loc[calls & (risk >= float(threshold)) & (days <= int(ex_div_days))].copy()
    if out.empty:
        return pd.DataFrame(columns=["date", "contract_key", "assignment_risk", "days_to_next_dividend"])
    if "contract_key" not in out.columns:
        out["contract_key"] = _contract_key_from_book(out)
    return out[["date", "contract_key", "assignment_risk", "days_to_next_dividend"]].reset_index(drop=True)


def overlay_summary(results: dict) -> pd.DataFrame:
    nav = results.get("nav", pd.DataFrame())
    trades = results.get("trades", pd.DataFrame())
    drawdown = results.get("drawdown", pd.DataFrame())
    rows = []
    for col in nav.columns if isinstance(nav, pd.DataFrame) else []:
        ret = nav[col].pct_change().dropna()
        monthly = nav[col].resample("ME").last().pct_change().dropna() if isinstance(nav.index, pd.DatetimeIndex) else pd.Series(dtype=float)
        t = trades[trades["strategy"].eq(col)].copy() if not trades.empty and "strategy" in trades else pd.DataFrame()
        premium = float(t.loc[t["event"].eq("open"), "cashflow"].sum()) if not t.empty and {"event", "cashflow"}.issubset(t.columns) else np.nan
        spread = float(t.get("spread_cost", pd.Series(dtype=float)).sum()) if not t.empty else np.nan
        rows.append(
            {
                "strategy": col,
                "final_nav": float(nav[col].iloc[-1]),
                "total_pnl": float(nav[col].iloc[-1] - nav[col].iloc[0]),
                "total_return": float(nav[col].iloc[-1] / nav[col].iloc[0] - 1.0),
                "annualized_return": float((nav[col].iloc[-1] / nav[col].iloc[0]) ** (252.0 / max(len(nav[col]) - 1, 1)) - 1.0),
                "max_drawdown": float(drawdown[col].min()) if not drawdown.empty and col in drawdown else np.nan,
                "downside_deviation": float(ret[ret < 0].std(ddof=1) * np.sqrt(252.0)) if (ret < 0).sum() > 1 else np.nan,
                "worst_month": float(monthly.min()) if len(monthly) else np.nan,
                "best_month": float(monthly.max()) if len(monthly) else np.nan,
                "trades": int((trades["strategy"] == col).sum()) if not trades.empty and "strategy" in trades else 0,
                "roll_closes": int(t["event"].eq("roll_close").sum()) if not t.empty and "event" in t else 0,
                "assignment_defense_closes": int(t["event"].eq("assignment_defense_close").sum()) if not t.empty and "event" in t else 0,
                "expiry_settlements": int(t["event"].eq("expiry_settlement").sum()) if not t.empty and "event" in t else 0,
                "net_open_premium_cashflow": premium,
                "spread_cost": spread,
                "ann_vol": float(ret.std(ddof=1) * np.sqrt(252.0)) if len(ret) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def overlay_mechanics_table(results: dict, *, shares: float = 0.0, dividends: pd.Series | None = None) -> pd.DataFrame:
    trades = results.get("trades", pd.DataFrame()).copy()
    nav = results.get("nav", pd.DataFrame())
    calls = results.get("call_holdings", pd.DataFrame())
    puts = results.get("put_holdings", pd.DataFrame())
    if isinstance(nav, pd.DataFrame):
        strategies = list(nav.columns)
    elif not trades.empty and "strategy" in trades:
        strategies = sorted(trades["strategy"].astype(str).unique())
    else:
        strategies = []
    if dividends is None or not isinstance(nav.index, pd.DatetimeIndex):
        dividends_received = 0.0
    else:
        div = pd.Series(dividends).copy()
        div.index = pd.to_datetime(div.index, errors="coerce").normalize()
        div = div.groupby(level=0).sum().reindex(nav.index).fillna(0.0)
        dividends_received = float(div.sum() * float(shares))
    rows = []
    for name in strategies:
        t = trades[trades["strategy"].astype(str).eq(str(name))].copy() if not trades.empty and "strategy" in trades else pd.DataFrame()
        opens = t[t["event"].eq("open")].copy() if not t.empty and "event" in t else pd.DataFrame()
        closes = t[t["event"].isin(["roll_close", "assignment_defense_close"])].copy() if not t.empty and "event" in t else pd.DataFrame()
        c = pd.to_numeric(calls[name], errors="coerce") if isinstance(calls, pd.DataFrame) and name in calls else pd.Series(dtype=float)
        p = pd.to_numeric(puts[name], errors="coerce") if isinstance(puts, pd.DataFrame) and name in puts else pd.Series(dtype=float)
        cash = pd.to_numeric(t.get("cashflow", pd.Series(dtype=float)), errors="coerce") if not t.empty else pd.Series(dtype=float)
        event = t.get("event", pd.Series(dtype=str)) if not t.empty else pd.Series(dtype=str)
        open_cash = pd.to_numeric(opens.get("cashflow", pd.Series(dtype=float)), errors="coerce") if not opens.empty else pd.Series(dtype=float)
        close_cash = pd.to_numeric(closes.get("cashflow", pd.Series(dtype=float)), errors="coerce") if not closes.empty else pd.Series(dtype=float)
        rows.append(
            {
                "strategy": name,
                "opens": int(event.eq("open").sum()) if len(event) else 0,
                "closes": int(event.isin(["roll_close", "assignment_defense_close"]).sum()) if len(event) else 0,
                "roll_closes": int(event.eq("roll_close").sum()) if len(event) else 0,
                "expiry_settlements": int(event.eq("expiry_settlement").sum()) if len(event) else 0,
                "assignment_defense_closes": int(event.eq("assignment_defense_close").sum()) if len(event) else 0,
                "max_active_calls": float(c.max()) if len(c) else 0.0,
                "max_active_puts": float(p.max()) if len(p) else 0.0,
                "average_active_calls": float(c.mean()) if len(c) else 0.0,
                "average_active_puts": float(p.mean()) if len(p) else 0.0,
                "median_holding_days": float(pd.to_numeric(t.get("holding_days", pd.Series(dtype=float)), errors="coerce").replace(0.0, np.nan).median()) if not t.empty else np.nan,
                "median_entry_dte": float(pd.to_numeric(opens.get("dte_days", pd.Series(dtype=float)), errors="coerce").median()) if not opens.empty else np.nan,
                "median_entry_moneyness": float(pd.to_numeric(opens.get("moneyness", pd.Series(dtype=float)), errors="coerce").median()) if not opens.empty else np.nan,
                "median_option_entry_price": float(pd.to_numeric(opens.get("price", pd.Series(dtype=float)), errors="coerce").median()) if not opens.empty else np.nan,
                "p95_option_entry_price": float(pd.to_numeric(opens.get("price", pd.Series(dtype=float)), errors="coerce").quantile(0.95)) if not opens.empty else np.nan,
                "average_premium_per_open": float(open_cash.abs().mean()) if len(open_cash) else 0.0,
                "total_premium_received": float(open_cash[open_cash > 0.0].sum()) if len(open_cash) else 0.0,
                "total_close_cost": float(-close_cash[close_cash < 0.0].sum()) if len(close_cash) else 0.0,
                "total_spread_cost": float(pd.to_numeric(t.get("spread_cost", pd.Series(dtype=float)), errors="coerce").sum()) if not t.empty else 0.0,
                "total_dividends_received": dividends_received,
                "net_trade_cashflow": float(cash.sum()) if len(cash) else 0.0,
            }
        )
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
    "boundary_roll_schedule",
    "collar_schedule",
    "covered_call_schedule",
    "find_straddle_exit",
    "mark_book_for_schedules",
    "pnl_by_vrp_decile",
    "prepare_option_book",
    "protective_put_schedule",
    "run_overlay_backtest",
    "assignment_defense_actions",
    "select_atm_straddle",
    "size_long_straddle_by_premium_budget",
    "size_short_straddle_by_margin_cap",
    "summarize_overlay_trades",
    "overlay_summary",
    "overlay_mechanics_table",
]
