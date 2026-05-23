from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.options import quote_cleaning


def compute_option_mark_to_market_pnl(
    current_price: float,
    previous_price: float | None,
    quantity: float = 1.0,
    multiplier: float = 1.0,
) -> float:
    if previous_price is None or not np.isfinite(previous_price):
        return 0.0
    return (float(current_price) - float(previous_price)) * float(quantity) * float(multiplier)


def compute_hedge_pnl(
    current_spot: float,
    previous_spot: float | None,
    hedge_units: float,
) -> float:
    if previous_spot is None or not np.isfinite(previous_spot):
        return 0.0
    return float(hedge_units) * (float(current_spot) - float(previous_spot))


def apply_hedge_transaction_costs(
    trade_units: float,
    price: float,
    trading_cost_bps: float = 1.0,
    half_spread: float = 0.0,
) -> float:
    notional_cost = abs(float(trade_units)) * float(price) * float(trading_cost_bps) * 1e-4
    spread_cost = abs(float(trade_units)) * float(max(half_spread, 0.0))
    return notional_cost + spread_cost


def hedging_drawdown(nav: pd.Series) -> pd.Series:
    nav = pd.Series(nav, dtype=float)
    return nav - nav.cummax()


def _contract_key(row: pd.Series) -> str:
    expiry = pd.Timestamp(row.get("expiry")).strftime("%Y-%m-%d") if pd.notna(row.get("expiry")) else "NA"
    return f"{row.get('option_type', row.get('cp', 'option'))}_{expiry}_{float(row.get('strike', np.nan)):.8g}"


def _dedupe_contract_book(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns or "contract_key" not in frame.columns:
        return frame
    sort_cols = [c for c in ["date", "expiry", "strike", "timestamp"] if c in frame.columns]
    out = frame.sort_values(sort_cols).copy() if sort_cols else frame.copy()
    return out.drop_duplicates(["date", "contract_key"], keep="last").copy()


def _add_future_quote_count(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"date", "contract_key"}.issubset(frame.columns):
        return frame
    global_dates = pd.Index(pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values().unique())
    date_rank = {pd.Timestamp(date): i for i, date in enumerate(global_dates)}
    pieces = []
    for _, group in frame.sort_values(["contract_key", "date"]).groupby("contract_key", sort=False):
        g = group.sort_values("date").copy()
        n = len(g)
        gaps = np.full(n, np.inf, dtype=float)
        trade_gaps = np.full(n, np.inf, dtype=float)
        if n > 1:
            dates = pd.to_datetime(g["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
            gaps[:-1] = (dates[1:] - dates[:-1]) / np.timedelta64(1, "D")
            ranks = np.array([date_rank.get(pd.Timestamp(date), np.nan) for date in dates], dtype=float)
            trade_gaps[:-1] = ranks[1:] - ranks[:-1]
        consecutive = np.ones(n, dtype=float)
        for i in range(n - 2, -1, -1):
            if np.isfinite(trade_gaps[i]) and trade_gaps[i] == 1.0:
                consecutive[i] = consecutive[i + 1] + 1.0
        g["future_quote_count"] = (n - np.arange(n, dtype=float)).astype(float)
        g["next_quote_gap_days"] = gaps
        g["next_quote_gap_trade_days"] = trade_gaps
        g["future_consecutive_quote_days"] = consecutive
        pieces.append(g)
    return pd.concat(pieces, ignore_index=False).sort_values(["date", "contract_key"]).copy()


def _loc_contract(book: pd.DataFrame, contract_id: str | None) -> pd.Series | None:
    if contract_id is None or book.empty or contract_id not in book.index:
        return None
    row = book.loc[contract_id]
    if isinstance(row, pd.DataFrame):
        sort_cols = [c for c in ["timestamp", "expiry", "strike"] if c in row.columns]
        row = row.sort_values(sort_cols).iloc[-1] if sort_cols else row.iloc[-1]
    return row


def _finite_mid(row: pd.Series | None) -> float:
    if row is None or "mid" not in row.index:
        return np.nan
    value = pd.to_numeric(pd.Series([row["mid"]]), errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else np.nan


def _minimal_option_row(mid: float) -> pd.Series:
    return pd.Series({"mid": float(mid)})


def _resolve_option_mark(
    row: pd.Series | None,
    last_mid: float,
    *,
    policy: str,
    diagnostics: dict[str, Any],
    strategy: str,
    role: str,
    contract_id: str | None,
    date: pd.Timestamp,
) -> tuple[float, pd.Series | None, bool, bool]:
    mid = _finite_mid(row)
    if np.isfinite(mid):
        return mid, row, False, False

    diagnostics["n_missing_option_marks"] = int(diagnostics.get("n_missing_option_marks", 0)) + 1
    message = (
        f"Missing {role} option mark for strategy={strategy!r}, "
        f"contract={contract_id!r}, date={pd.Timestamp(date).date()}."
    )
    if policy == "error":
        diagnostics["n_missing_option_mark_errors"] = int(diagnostics.get("n_missing_option_mark_errors", 0)) + 1
        raise ValueError(message + " Set missing_option_mark='skip_day' or 'stale_with_warning' to override.")
    if policy == "skip_day":
        diagnostics["n_missing_option_mark_skip_days"] = int(diagnostics.get("n_missing_option_mark_skip_days", 0)) + 1
        return np.nan, row, False, True
    if policy == "stale_with_warning":
        if not np.isfinite(last_mid):
            diagnostics["n_missing_option_mark_errors"] = int(diagnostics.get("n_missing_option_mark_errors", 0)) + 1
            raise ValueError(message + " No finite stale mark is available.")
        diagnostics["n_stale_option_marks"] = int(diagnostics.get("n_stale_option_marks", 0)) + 1
        warnings.warn(message + f" Using stale mark {float(last_mid):.6g}.", RuntimeWarning, stacklevel=2)
        return float(last_mid), row, True, False
    raise ValueError("missing_option_mark must be 'error', 'skip_day', or 'stale_with_warning'.")


def _merge_greeks(option_path: pd.DataFrame, greeks: pd.DataFrame | None) -> pd.DataFrame:
    out = option_path.copy()
    if {"delta", "vega", "gamma"}.issubset(out.columns) or greeks is None or greeks.empty:
        return out
    greek_cols = [
        c
        for c in [
            "source_index",
            "date",
            "expiry",
            "strike",
            "option_type",
            "delta",
            "gamma",
            "vega",
            "theta",
            "rho",
            "delta_mid",
            "gamma_mid",
            "vega_mid",
        ]
        if c in greeks.columns
    ]
    g = greeks[greek_cols].copy()
    if "source_index" in out.columns and "source_index" in g.columns:
        merged = out.merge(g.drop_duplicates("source_index"), on="source_index", how="left", suffixes=("", "_g"))
    else:
        keys = [c for c in ["date", "expiry", "strike", "option_type"] if c in out.columns and c in g.columns]
        merged = out.merge(g.drop_duplicates(keys), on=keys, how="left", suffixes=("", "_g"))
    for greek in ["delta", "gamma", "vega", "theta", "rho"]:
        gcol = f"{greek}_g"
        if greek not in merged.columns and gcol in merged.columns:
            merged[greek] = merged[gcol]
        elif gcol in merged.columns:
            merged[greek] = merged[greek].combine_first(merged[gcol])
    for greek in ["delta", "gamma", "vega"]:
        if greek not in merged.columns and f"{greek}_mid" in merged.columns:
            merged[greek] = merged[f"{greek}_mid"]
    return merged


def _datetime64ns(values) -> pd.Series:
    return pd.Series(pd.to_datetime(values, errors="coerce")).astype("datetime64[ns]")


def _align_spot(path: pd.DataFrame, spot_series: pd.Series | None) -> pd.Series:
    if "spot" in path.columns and path["spot"].notna().any():
        return pd.to_numeric(path["spot"], errors="coerce")
    if spot_series is None or spot_series.empty:
        raise ValueError("spot_series is required when option_path has no spot column.")
    spot = spot_series.copy()
    spot.index = pd.DatetimeIndex(pd.to_datetime(spot.index, errors="coerce")).astype("datetime64[ns]")
    left = path[["date"]].reset_index().rename(columns={"index": "_row"})
    left["date"] = _datetime64ns(left["date"]).to_numpy()
    left = left.sort_values("date")
    right = spot.rename("spot").reset_index().rename(columns={"index": "date"})
    right["date"] = _datetime64ns(right["date"]).to_numpy()
    right = right.sort_values("date")
    matched = pd.merge_asof(left, right, on="date", direction="backward").set_index("_row")["spot"]
    return matched.reindex(path.index)


def _align_market_series(dates: pd.Series, series: pd.Series | None, default: float = 0.0) -> pd.Series:
    idx = _datetime64ns(dates)
    if series is None or len(series) == 0:
        return pd.Series(default, index=idx)
    s = pd.Series(series).copy()
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index, errors="coerce")).astype("datetime64[ns]")
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    if s.empty:
        return pd.Series(default, index=idx)
    left = pd.DataFrame({"date": idx}).reset_index()
    left["date"] = _datetime64ns(left["date"]).to_numpy()
    left = left.sort_values("date")
    right = s.rename("value").reset_index().rename(columns={"index": "date"})
    right["date"] = _datetime64ns(right["date"]).to_numpy()
    right = right.sort_values("date")
    matched = pd.merge_asof(left, right, on="date", direction="backward").set_index("index")["value"]
    return matched.reindex(range(len(idx))).fillna(default).set_axis(idx)


def _option_half_spread(row: pd.Series | None) -> float:
    if row is None:
        return 0.0
    if "half_spread" in row.index and np.isfinite(row["half_spread"]):
        return float(max(row["half_spread"], 0.0))
    if "spread" in row.index and np.isfinite(row["spread"]):
        return 0.5 * float(max(row["spread"], 0.0))
    bid = float(row["bid"]) if "bid" in row.index and np.isfinite(row["bid"]) else np.nan
    ask = float(row["ask"]) if "ask" in row.index and np.isfinite(row["ask"]) else np.nan
    if np.isfinite(bid) and np.isfinite(ask):
        return 0.5 * float(max(ask - bid, 0.0))
    return 0.0


def _option_trade_cost(
    trade_contracts: float,
    row: pd.Series | None,
    *,
    option_multiplier: float,
    trading_cost_bps: float,
    use_bid_ask_costs: bool,
) -> float:
    if row is None:
        return 0.0
    mid = float(row.get("mid", np.nan))
    notional_cost = abs(float(trade_contracts)) * max(mid, 0.0) * float(option_multiplier) * float(trading_cost_bps) * 1e-4
    spread_cost = (
        abs(float(trade_contracts)) * _option_half_spread(row) * float(option_multiplier)
        if use_bid_ask_costs
        else 0.0
    )
    return notional_cost + spread_cost


def _quantize_step(value: float, step: float) -> float:
    step = float(max(step, 1e-12))
    return float(np.round(float(value) / step) * step)


def _band_target_residual(residual: float, outer_band: float, inner_band: float) -> float | None:
    residual = float(residual)
    outer_band = float(max(outer_band, 1e-12))
    inner_band = float(max(min(inner_band, outer_band), 0.0))
    if abs(residual) <= outer_band:
        return None
    return float(np.sign(residual) * inner_band)


def _trade_share(
    cash: float,
    current_pos: float,
    target_pos: float,
    price: float,
    *,
    trading_cost_bps: float,
) -> tuple[float, float, float, float]:
    delta_q = float(target_pos - current_pos)
    notional = abs(delta_q) * float(price)
    cash -= delta_q * float(price)
    cost = notional * float(trading_cost_bps) * 1e-4
    cash -= cost
    return cash, float(target_pos), cost, notional


def _trade_option_cash(
    cash: float,
    current_pos: float,
    target_pos: float,
    price: float,
    row: pd.Series | None,
    *,
    option_multiplier: float,
    trading_cost_bps: float,
    use_bid_ask_costs: bool,
) -> tuple[float, float, float, float]:
    delta_q = float(target_pos - current_pos)
    notional = abs(delta_q) * float(price) * float(option_multiplier)
    cash -= delta_q * float(price) * float(option_multiplier)
    cost = _option_trade_cost(
        delta_q,
        row,
        option_multiplier=option_multiplier,
        trading_cost_bps=trading_cost_bps,
        use_bid_ask_costs=use_bid_ask_costs,
    )
    cash -= cost
    return cash, float(target_pos), cost, notional


def _row_greek(row: pd.Series, base: str) -> float:
    for col in [base, f"{base}_mid", f"{base}_spot_mid", f"{base}_spot"]:
        if col in row.index and np.isfinite(row[col]):
            return float(row[col])
    if base == "delta" and "delta_spot_mid" in row.index and np.isfinite(row["delta_spot_mid"]):
        return float(row["delta_spot_mid"])
    if base == "gamma" and "gamma_spot_mid" in row.index and np.isfinite(row["gamma_spot_mid"]):
        return float(row["gamma_spot_mid"])
    if base == "vega" and "vega_mid" in row.index and np.isfinite(row["vega_mid"]):
        return float(row["vega_mid"])
    return 0.0


def _under_equiv_per_share(underlying_level: float, hedge_price: float, hedge_beta: float) -> float:
    hedge_price = float(max(hedge_price, 1e-12))
    hedge_beta = float(np.clip(hedge_beta, 0.90, 1.10)) if np.isfinite(hedge_beta) else 1.0
    underlying_level = float(max(underlying_level, 1e-12))
    return float(hedge_beta * hedge_price / underlying_level)


def _target_under_pos_for_residual(
    option_delta_equiv: float,
    target_residual: float,
    underlying_level: float,
    hedge_price: float,
    hedge_beta: float,
) -> float:
    equiv = _under_equiv_per_share(underlying_level, hedge_price, hedge_beta)
    return float((float(target_residual) - float(option_delta_equiv)) / equiv)


def _prepare_hedge_book(
    option_book: pd.DataFrame | None,
    *,
    greeks: pd.DataFrame | None,
    spot_series: pd.Series | None,
    quote_price_unit: str,
    pnl_mode: str,
    contract_size: float,
) -> pd.DataFrame:
    if option_book is None or option_book.empty:
        return pd.DataFrame()
    book = _merge_greeks(option_book, greeks).copy()
    if "date" not in book.columns:
        return pd.DataFrame()
    book["date"] = pd.to_datetime(book["date"], errors="coerce").dt.normalize()
    book = book.dropna(subset=["date"]).sort_values(["date", "expiry", "strike"] if "expiry" in book.columns else ["date"])
    book["spot"] = _align_spot(book, spot_series).to_numpy(dtype=float)
    if str(pnl_mode).lower() == "usd_equivalent":
        book = quote_cleaning.convert_quotes_to_usd_equivalent(
            book,
            spot_col="spot",
            price_cols=("bid", "ask", "mid", "last", "mark", "half_spread"),
            unit=quote_price_unit,
            contract_size=contract_size,
        )
    book["contract_key"] = book.apply(_contract_key, axis=1)
    for col in ["mid", "delta", "gamma", "vega", "dte", "rel_spread", "volume", "half_spread"]:
        if col in book.columns:
            book[col] = pd.to_numeric(book[col], errors="coerce")
    if "dte" not in book.columns and {"date", "expiry"}.issubset(book.columns):
        book["dte"] = (pd.to_datetime(book["expiry"], errors="coerce") - book["date"]).dt.days
    if "rel_spread" not in book.columns and {"bid", "ask", "mid"}.issubset(book.columns):
        book["rel_spread"] = (book["ask"] - book["bid"]) / book["mid"].replace(0, np.nan)
    if "volume" not in book.columns:
        book["volume"] = 0.0
    book["abs_delta"] = pd.to_numeric(book.get("delta"), errors="coerce").abs()
    return _add_future_quote_count(_dedupe_contract_book(book))


def _pick_vega_hedge(day: pd.DataFrame, main_row: pd.Series, *, max_dte: float = 120.0) -> pd.Series | None:
    if day.empty:
        return None
    main_key = str(main_row.get("contract_key", ""))
    candidates = day[day["contract_key"].astype(str).ne(main_key)].copy()
    opt_col = "option_type" if "option_type" in candidates.columns else "cp" if "cp" in candidates.columns else None
    main_type = str(main_row.get(opt_col, "")).lower() if opt_col else ""
    if opt_col and main_type:
        same_type = candidates[candidates[opt_col].astype(str).str.lower().eq(main_type)].copy()
        if not same_type.empty:
            candidates = same_type
    candidates = candidates[
        (pd.to_numeric(candidates.get("mid"), errors="coerce") > 0)
        & np.isfinite(pd.to_numeric(candidates.get("vega"), errors="coerce"))
        & (pd.to_numeric(candidates.get("vega"), errors="coerce").abs() > 1e-8)
        & np.isfinite(pd.to_numeric(candidates.get("delta"), errors="coerce"))
    ].copy()
    if candidates.empty:
        return None
    future_col = "future_consecutive_quote_days" if "future_consecutive_quote_days" in candidates.columns else "future_quote_count"
    if future_col in candidates.columns:
        durable = candidates[pd.to_numeric(candidates[future_col], errors="coerce") >= 5].copy()
        if not durable.empty:
            candidates = durable
    main_dte = float(main_row.get("dte", np.nan))
    if np.isfinite(main_dte) and "dte" in candidates.columns:
        dated = candidates[
            (pd.to_numeric(candidates["dte"], errors="coerce") >= max(main_dte + 14.0, 45.0))
            & (pd.to_numeric(candidates["dte"], errors="coerce") <= max_dte)
        ].copy()
        if not dated.empty:
            candidates = dated
    main_abs_delta = abs(float(main_row.get("delta", 0.0)))
    target_dte = max(float(main_row.get("dte", 60.0)) + 21.0, 60.0)
    if "future_quote_count" not in candidates.columns:
        candidates["future_quote_count"] = 0.0
    if "future_consecutive_quote_days" not in candidates.columns:
        candidates["future_consecutive_quote_days"] = candidates["future_quote_count"]
    candidates["score"] = (
        0.80 * (pd.to_numeric(candidates.get("abs_delta"), errors="coerce") - main_abs_delta).abs().fillna(0.5)
        + 0.020 * (pd.to_numeric(candidates.get("dte"), errors="coerce") - target_dte).abs().fillna(30.0)
        + 1.75 * pd.to_numeric(candidates.get("rel_spread"), errors="coerce").fillna(0.0)
        - 0.00020 * pd.to_numeric(candidates.get("volume"), errors="coerce").fillna(0.0)
        - 0.012 * pd.to_numeric(candidates.get("future_quote_count"), errors="coerce").fillna(0.0)
        - 0.020 * pd.to_numeric(candidates.get("future_consecutive_quote_days"), errors="coerce").fillna(0.0)
    )
    return candidates.sort_values("score").iloc[0]


def _pick_main_contract(
    day: pd.DataFrame,
    *,
    preferred_option_type: str = "call",
    side: float = 1.0,
    target_abs_delta: float = 0.50,
    entry_dte_range: tuple[float, float] | None = None,
    min_future_quote_days: int = 5,
) -> str | None:
    if day.empty:
        return None
    d = day.copy()
    opt_col = "option_type" if "option_type" in d.columns else "cp" if "cp" in d.columns else None
    if opt_col:
        pref = str(preferred_option_type).lower()
        typed = d[d[opt_col].astype(str).str.lower().str.startswith(pref[0])].copy()
        if not typed.empty:
            d = typed
    d = d[
        (pd.to_numeric(d.get("mid"), errors="coerce") > 0)
        & np.isfinite(pd.to_numeric(d.get("delta"), errors="coerce"))
        & np.isfinite(pd.to_numeric(d.get("vega"), errors="coerce"))
        & np.isfinite(pd.to_numeric(d.get("gamma"), errors="coerce"))
    ].copy()
    if d.empty:
        return None
    if entry_dte_range is not None and "dte" in d.columns:
        lo, hi = entry_dte_range
        ranged = d[(pd.to_numeric(d["dte"], errors="coerce") >= lo) & (pd.to_numeric(d["dte"], errors="coerce") <= hi)].copy()
        if not ranged.empty:
            d = ranged
    future_col = "future_consecutive_quote_days" if "future_consecutive_quote_days" in d.columns else "future_quote_count"
    if future_col in d.columns and int(min_future_quote_days) > 1:
        durable = d[pd.to_numeric(d[future_col], errors="coerce") >= int(min_future_quote_days)].copy()
        if not durable.empty:
            d = durable
    d["abs_delta"] = pd.to_numeric(d.get("delta"), errors="coerce").abs()
    delta_band = d[(d["abs_delta"] >= 0.25) & (d["abs_delta"] <= 0.75)].copy()
    if not delta_band.empty:
        d = delta_band
    if side > 0 and opt_col:
        if str(preferred_option_type).lower().startswith("c"):
            pos = d[pd.to_numeric(d.get("delta"), errors="coerce") > 0].copy()
        else:
            pos = d[pd.to_numeric(d.get("delta"), errors="coerce") < 0].copy()
        if not pos.empty:
            d = pos
    if "rel_spread" not in d.columns:
        d["rel_spread"] = 0.0
    if "volume" not in d.columns:
        d["volume"] = 0.0
    if "future_quote_count" not in d.columns:
        d["future_quote_count"] = 0.0
    if "future_consecutive_quote_days" not in d.columns:
        d["future_consecutive_quote_days"] = d["future_quote_count"]
    dte = pd.to_numeric(d.get("dte", pd.Series(35.0, index=d.index)), errors="coerce")
    d["entry_score"] = (
        -2.0 * (d["abs_delta"] - float(target_abs_delta)).abs()
        - 0.07 * (dte - 35.0).abs().fillna(0.0)
        - 1.75 * pd.to_numeric(d["rel_spread"], errors="coerce").fillna(0.0)
        + 0.00015 * pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
        + 0.015 * pd.to_numeric(d.get("future_quote_count"), errors="coerce").fillna(0.0)
        + 0.030 * pd.to_numeric(d.get("future_consecutive_quote_days"), errors="coerce").fillna(0.0)
    )
    return str(d.sort_values(["entry_score", "volume"], ascending=[False, False]).iloc[0]["contract_key"])


def run_delta_hedge_backtest(**kwargs: Any) -> dict:
    return run_option_hedging_backtest(strategies=("delta",), **kwargs)


def run_delta_vega_hedge_backtest(**kwargs: Any) -> dict:
    return run_option_hedging_backtest(strategies=("delta_vega",), **kwargs)


def _prepare_entry_schedule(entry_schedule: pd.DataFrame) -> pd.DataFrame:
    if entry_schedule is None or entry_schedule.empty:
        return pd.DataFrame()
    schedule = entry_schedule.copy()
    if "entry_date" not in schedule.columns:
        raise ValueError("entry_schedule must contain an 'entry_date' column.")
    if "contract_key" not in schedule.columns:
        raise ValueError("entry_schedule must contain a 'contract_key' column.")
    schedule["entry_date"] = pd.to_datetime(schedule["entry_date"], errors="coerce").dt.normalize()
    schedule = schedule.dropna(subset=["entry_date", "contract_key"]).copy()
    schedule["contract_key"] = schedule["contract_key"].astype(str)
    if "quantity" not in schedule.columns:
        schedule["quantity"] = 1.0
    schedule["quantity"] = pd.to_numeric(schedule["quantity"], errors="coerce").fillna(0.0)
    if "label" not in schedule.columns:
        schedule["label"] = "scheduled"
    if "max_hold_days" not in schedule.columns:
        schedule["max_hold_days"] = np.nan
    if "exit_on_convergence" not in schedule.columns:
        schedule["exit_on_convergence"] = False
    if "exit_on_sign_flip" not in schedule.columns:
        schedule["exit_on_sign_flip"] = False
    schedule = schedule[schedule["quantity"].ne(0.0)].copy()
    sort_cols = [c for c in ["entry_date", "label", "entry_score"] if c in schedule.columns]
    if sort_cols:
        schedule = schedule.sort_values(sort_cols)
    return schedule.reset_index(drop=True)


def run_scheduled_option_hedging_backtest(
    option_path: pd.DataFrame,
    entry_schedule: pd.DataFrame,
    spot_series: pd.Series | None = None,
    greeks: pd.DataFrame | None = None,
    strategies: list[str] | tuple[str, ...] = ("unhedged", "delta"),
    delta_band: float = 0.05,
    trading_cost_bps: float = 1.0,
    use_bid_ask_costs: bool = True,
    option_multiplier: float = 1.0,
    delta_inner_band: float | None = None,
    delta_share_lot: float | None = None,
    delta_cooldown_days: int = 1,
    exit_dte_days: float = 7.0,
    hedge_price_series: pd.Series | None = None,
    hedge_dividend_series: pd.Series | None = None,
    hedge_beta_series: pd.Series | None = None,
    hedge_symbol: str = "underlying",
    valuation_currency: str = "USD",
    quote_price_unit: str = "auto",
    pnl_mode: str = "usd_equivalent",
    annualization_days: float = 365.0,
    contract_size: float = 1.0,
    missing_option_mark: str = "error",
    max_hold_days: int | float | None = None,
    calendar: str = "business",
) -> dict:
    """
    Run a hedging backtest from an explicit option entry schedule.

    The scheduled engine uses the same option book preparation, mark handling,
    bid/ask costs, SPY/underlying hedge mechanics, and summary format as
    ``run_option_hedging_backtest``. The selection difference is that the entry
    contract comes from ``entry_schedule`` instead of the internal contract
    picker.
    """
    if option_path.empty:
        raise ValueError("option_path is empty.")
    schedule = _prepare_entry_schedule(entry_schedule)
    if schedule.empty:
        raise ValueError("entry_schedule is empty after cleaning.")
    if max_hold_days is not None:
        mh = pd.to_numeric(schedule.get("max_hold_days", np.nan), errors="coerce")
        schedule["max_hold_days"] = mh.fillna(float(max_hold_days))
    missing_policy = str(missing_option_mark).lower()
    if missing_policy not in {"error", "skip_day", "stale_with_warning"}:
        raise ValueError("missing_option_mark must be 'error', 'skip_day', or 'stale_with_warning'.")

    schedule_keys = set(schedule["contract_key"].astype(str))
    max_hold = pd.to_numeric(schedule.get("max_hold_days", pd.Series(np.nan, index=schedule.index)), errors="coerce")
    hold_days = float(max_hold.replace([np.inf, -np.inf], np.nan).max()) if max_hold.notna().any() else 30.0
    if not np.isfinite(hold_days):
        hold_days = 30.0
    start_date = pd.to_datetime(schedule["entry_date"], errors="coerce").min() - pd.Timedelta(days=3)
    end_date = pd.to_datetime(schedule["entry_date"], errors="coerce").max() + pd.Timedelta(days=max(10.0, hold_days + float(exit_dte_days) + 5.0))
    raw_option_path = option_path.copy()
    if "contract_key" in raw_option_path.columns:
        raw_option_path = raw_option_path[raw_option_path["contract_key"].astype(str).isin(schedule_keys)].copy()
    if "date" in raw_option_path.columns:
        raw_dates = pd.to_datetime(raw_option_path["date"], errors="coerce").dt.normalize()
        raw_option_path = raw_option_path[raw_dates.between(start_date, end_date)].copy()
    raw_greeks = greeks
    if raw_greeks is not None and not raw_greeks.empty and "contract_key" in raw_greeks.columns:
        raw_greeks = raw_greeks[raw_greeks["contract_key"].astype(str).isin(schedule_keys)].copy()
        if "date" in raw_greeks.columns:
            g_dates = pd.to_datetime(raw_greeks["date"], errors="coerce").dt.normalize()
            raw_greeks = raw_greeks[g_dates.between(start_date, end_date)].copy()

    path = _merge_greeks(raw_option_path, raw_greeks).copy()
    path["date"] = pd.to_datetime(path["date"], errors="coerce").dt.normalize()
    path = path.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    path["spot"] = _align_spot(path, spot_series).to_numpy(dtype=float)
    input_price_unit_detected = (
        path["price_unit_detected"].dropna().iloc[0]
        if "price_unit_detected" in path.columns and path["price_unit_detected"].notna().any()
        else "unknown"
    )
    if str(pnl_mode).lower() == "usd_equivalent":
        path = quote_cleaning.convert_quotes_to_usd_equivalent(
            path,
            spot_col="spot",
            price_cols=("bid", "ask", "mid", "last", "mark", "half_spread"),
            unit=quote_price_unit,
            contract_size=contract_size,
        )
    path["contract_key"] = path.apply(_contract_key, axis=1)
    for col in ["mid", "delta", "gamma", "vega", "rate", "dte", "rel_spread", "volume", "half_spread"]:
        if col in path.columns:
            path[col] = pd.to_numeric(path[col], errors="coerce")
    for col in ["mid", "delta", "gamma", "vega"]:
        if col not in path.columns:
            path[col] = np.nan
    if "dte" not in path.columns and {"date", "expiry"}.issubset(path.columns):
        path["dte"] = (pd.to_datetime(path["expiry"], errors="coerce") - path["date"]).dt.days
    if "rel_spread" not in path.columns and {"bid", "ask", "mid"}.issubset(path.columns):
        path["rel_spread"] = (path["ask"] - path["bid"]) / path["mid"].replace(0, np.nan)
    path = _add_future_quote_count(_dedupe_contract_book(path))

    main_books = {
        pd.Timestamp(day): grp.set_index("contract_key", drop=False)
        for day, grp in path.groupby("date", sort=True)
    }
    trade_dates = sorted(main_books)
    if not trade_dates:
        raise ValueError("No dated option rows are available after hedging preparation.")
    date_index = pd.Series(pd.DatetimeIndex(trade_dates), index=pd.DatetimeIndex(trade_dates), name="date")
    hedge_prices = _align_market_series(
        date_index,
        hedge_price_series if hedge_price_series is not None else spot_series,
        default=np.nan,
    )
    hedge_dividends = _align_market_series(date_index, hedge_dividend_series, default=0.0)
    hedge_betas = _align_market_series(date_index, hedge_beta_series, default=1.0)
    if hedge_prices.isna().any():
        spot_by_date = path.groupby("date")["spot"].median()
        hedge_prices = hedge_prices.combine_first(spot_by_date).ffill().bfill()

    delta_inner = float(delta_inner_band) if delta_inner_band is not None else 0.25 * float(delta_band)
    share_lot = 0.0 if delta_share_lot is None else float(delta_share_lot)
    trade_date_index = pd.Index(pd.to_datetime(trade_dates))

    diagnostics: dict[str, Any] = {
        "valuation_currency": valuation_currency,
        "pnl_mode": pnl_mode,
        "price_unit_detected": input_price_unit_detected
        if str(input_price_unit_detected).lower() != "unknown"
        else (
            path.get("price_unit_detected", pd.Series(["unknown"])).dropna().iloc[0]
            if "price_unit_detected" in path.columns and path["price_unit_detected"].notna().any()
            else "unknown"
        ),
        "annualization_days": float(annualization_days),
        "contract_size": float(contract_size),
        "calendar": calendar,
        "missing_option_mark_policy": missing_policy,
        "n_option_book_rows": len(path),
        "n_scheduled_entries": len(schedule),
        "n_opened_episodes": 0,
        "n_missing_entry_contracts": 0,
        "n_missing_option_marks": 0,
        "n_missing_option_mark_errors": 0,
        "n_missing_option_mark_skip_days": 0,
        "n_stale_option_marks": 0,
        "n_exit_max_hold": 0,
        "n_exit_dte": 0,
        "n_exit_convergence": 0,
        "n_exit_sign_flip": 0,
        "n_exit_missing_mark": 0,
    }

    nav_frames: list[pd.Series] = []
    return_frames: list[pd.Series] = []
    pnl_frames: list[pd.Series] = []
    component_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    trade_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_contracts: set[str] = set()

    def record_trade(
        date_: pd.Timestamp,
        strategy_: str,
        episode_id_: int,
        instrument: str,
        contract_key: str | None,
        units: float,
        price: float,
        transaction_cost: float,
        traded_notional: float,
    ) -> None:
        if abs(float(units)) <= 1e-14 and float(traded_notional) <= 0.0:
            return
        trade_rows.append(
            {
                "date": date_,
                "strategy": strategy_,
                "episode_id": int(episode_id_),
                "instrument": instrument,
                "contract_key": contract_key,
                "trade_units": float(units),
                "price": float(price),
                "transaction_cost": float(transaction_cost),
                "traded_notional": float(traded_notional),
            }
        )

    for strategy in strategies:
        if strategy not in {"unhedged", "delta", "none"}:
            summary_rows.append(
                {
                    "strategy": strategy,
                    "status": "skipped",
                    "total_pnl": np.nan,
                    "skipped_reason": "scheduled backtest currently supports unhedged and delta strategies.",
                }
            )
            continue

        mode = "none" if strategy in {"unhedged", "none"} else "delta"
        rows: list[dict[str, Any]] = []

        for entry_i, entry in schedule.iterrows():
            entry_date = pd.Timestamp(entry["entry_date"])
            contract_id = str(entry["contract_key"])
            start_pos = int(trade_date_index.searchsorted(entry_date, side="left"))
            if start_pos >= len(trade_date_index):
                diagnostics["n_missing_entry_contracts"] += 1
                continue
            start_date = pd.Timestamp(trade_date_index[start_pos])
            start_book = main_books.get(start_date, pd.DataFrame())
            start_row = _loc_contract(start_book, contract_id)
            if start_row is None:
                diagnostics["n_missing_entry_contracts"] += 1
                continue

            episode_id = int(len(rows) + entry_i)
            q_main = 0.0
            q_under = 0.0
            cash = 0.0
            last_main_mid = np.nan
            last_hedge_px = np.nan
            prev_equity = 0.0
            prev_date = None
            prev_r = 0.0
            days_since_under_rehedge = 0
            days_held = 0
            entry_residual = float(entry.get("entry_residual", np.nan))
            entry_total_error = float(entry.get("entry_total_error", np.nan))
            max_hold = float(entry.get("max_hold_days", np.nan))
            exit_on_convergence = bool(entry.get("exit_on_convergence", False))
            exit_on_sign_flip = bool(entry.get("exit_on_sign_flip", False))
            exit_after_row = False
            exit_reason = "none"

            diagnostics["n_opened_episodes"] += 1
            selected_contracts.add(contract_id)

            for date in trade_date_index[start_pos:]:
                date = pd.Timestamp(date)
                day = main_books.get(date, pd.DataFrame())
                row = _loc_contract(day, contract_id)
                hedge_px = float(hedge_prices.loc[date])
                hedge_dividend = float(hedge_dividends.loc[date])
                hedge_beta = float(hedge_betas.loc[date])
                day_cost = 0.0
                day_turnover = 0.0
                option_trade_notional = 0.0
                under_trade_notional = 0.0
                skipped_day = 0
                stale_option_mark = 0

                if row is None and q_main == 0.0:
                    diagnostics["n_missing_entry_contracts"] += 1
                    break
                main_mid, row, stale, skip = _resolve_option_mark(
                    row,
                    last_main_mid,
                    policy=missing_policy,
                    diagnostics=diagnostics,
                    strategy=strategy,
                    role="scheduled main",
                    contract_id=contract_id,
                    date=date,
                )
                stale_option_mark += int(stale)
                if skip:
                    diagnostics["n_exit_missing_mark"] += 1
                    break

                prev_main_value = q_main * last_main_mid * float(option_multiplier) if q_main != 0.0 and np.isfinite(last_main_mid) else 0.0
                prev_under_value = q_under * last_hedge_px if q_under != 0.0 and np.isfinite(last_hedge_px) else 0.0

                cash_before_accrual = cash
                if prev_date is not None:
                    dt_year = max((date - pd.Timestamp(prev_date)).days / float(annualization_days), 0.0)
                    cash *= np.exp(prev_r * dt_year)
                    if q_under != 0.0:
                        cash += q_under * hedge_dividend
                cash_accrual_pnl = cash - cash_before_accrual

                main_value_pre = q_main * main_mid * float(option_multiplier) if q_main != 0.0 else 0.0
                under_value_pre = q_under * hedge_px
                main_option_pnl = main_value_pre - prev_main_value
                underlying_hedge_pnl = under_value_pre - prev_under_value

                if q_main == 0.0:
                    old_q = q_main
                    cash, q_main, cost_i, turnover_i = _trade_option_cash(
                        cash,
                        q_main,
                        float(entry["quantity"]),
                        main_mid,
                        row,
                        option_multiplier=option_multiplier,
                        trading_cost_bps=trading_cost_bps,
                        use_bid_ask_costs=use_bid_ask_costs,
                    )
                    day_cost += cost_i
                    day_turnover += turnover_i
                    option_trade_notional += turnover_i
                    record_trade(date, strategy, episode_id, "main_option", contract_id, q_main - old_q, main_mid, cost_i, turnover_i)

                underlying_level = float(row["spot"]) if row is not None and "spot" in row.index else hedge_px
                option_delta_equiv = float(option_multiplier) * q_main * _row_greek(row, "delta")
                option_gamma_equiv = float(option_multiplier) * q_main * _row_greek(row, "gamma")
                option_vega_equiv = float(option_multiplier) * q_main * _row_greek(row, "vega")
                share_delta_equiv_pre = q_under * _under_equiv_per_share(underlying_level, hedge_px, hedge_beta)
                residual_delta_equiv_pre = option_delta_equiv + share_delta_equiv_pre

                if mode == "delta":
                    desired_delta_residual = _band_target_residual(residual_delta_equiv_pre, float(delta_band), delta_inner)
                    should_trade_under = desired_delta_residual is not None and days_since_under_rehedge >= int(delta_cooldown_days)
                    if desired_delta_residual is not None:
                        target_q_under = _target_under_pos_for_residual(
                            option_delta_equiv,
                            desired_delta_residual,
                            underlying_level,
                            hedge_px,
                            hedge_beta,
                        )
                        if share_lot > 0:
                            target_q_under = float(np.round(target_q_under / share_lot) * share_lot)
                    else:
                        target_q_under = q_under
                    min_trade = share_lot if share_lot > 0 else 1e-12
                    if should_trade_under and abs(target_q_under - q_under) >= min_trade:
                        old_under = q_under
                        cash, q_under, cost_i, turnover_i = _trade_share(
                            cash,
                            q_under,
                            target_q_under,
                            hedge_px,
                            trading_cost_bps=trading_cost_bps,
                        )
                        day_cost += cost_i
                        day_turnover += turnover_i
                        under_trade_notional += turnover_i
                        days_since_under_rehedge = 0
                        record_trade(date, strategy, episode_id, "underlying", None, q_under - old_under, hedge_px, cost_i, turnover_i)
                    else:
                        days_since_under_rehedge += 1
                elif q_under != 0.0:
                    old_under = q_under
                    cash, q_under, cost_i, turnover_i = _trade_share(
                        cash,
                        q_under,
                        0.0,
                        hedge_px,
                        trading_cost_bps=trading_cost_bps,
                    )
                    day_cost += cost_i
                    day_turnover += turnover_i
                    under_trade_notional += turnover_i
                    record_trade(date, strategy, episode_id, "underlying", None, q_under - old_under, hedge_px, cost_i, turnover_i)

                dte_value = float(row.get("dte", np.nan)) if row is not None and np.isfinite(row.get("dte", np.nan)) else np.nan
                current_residual = float(row.get("price_residual", np.nan)) if row is not None and "price_residual" in row.index else np.nan
                exit_after_row = False
                exit_reason = "none"
                if np.isfinite(max_hold) and days_held >= max_hold:
                    exit_after_row = True
                    exit_reason = "max_hold"
                    diagnostics["n_exit_max_hold"] += 1
                if not exit_after_row and np.isfinite(dte_value) and dte_value <= float(exit_dte_days):
                    exit_after_row = True
                    exit_reason = "exit_dte"
                    diagnostics["n_exit_dte"] += 1
                if not exit_after_row and exit_on_convergence and np.isfinite(current_residual):
                    hurdle = abs(entry_total_error) if np.isfinite(entry_total_error) and entry_total_error > 0 else 0.0
                    if abs(current_residual) <= hurdle:
                        exit_after_row = True
                        exit_reason = "convergence"
                        diagnostics["n_exit_convergence"] += 1
                if not exit_after_row and exit_on_sign_flip and np.isfinite(entry_residual) and np.isfinite(current_residual):
                    if np.sign(current_residual) != 0.0 and np.sign(entry_residual) != 0.0 and np.sign(current_residual) != np.sign(entry_residual):
                        exit_after_row = True
                        exit_reason = "sign_flip"
                        diagnostics["n_exit_sign_flip"] += 1

                if exit_after_row:
                    if q_under != 0.0:
                        old_under = q_under
                        cash, q_under, cost_i, turnover_i = _trade_share(
                            cash,
                            q_under,
                            0.0,
                            hedge_px,
                            trading_cost_bps=trading_cost_bps,
                        )
                        day_cost += cost_i
                        day_turnover += turnover_i
                        under_trade_notional += turnover_i
                        record_trade(date, strategy, episode_id, "underlying", None, q_under - old_under, hedge_px, cost_i, turnover_i)
                    if q_main != 0.0:
                        old_q = q_main
                        cash, q_main, cost_i, turnover_i = _trade_option_cash(
                            cash,
                            q_main,
                            0.0,
                            main_mid,
                            row,
                            option_multiplier=option_multiplier,
                            trading_cost_bps=trading_cost_bps,
                            use_bid_ask_costs=use_bid_ask_costs,
                        )
                        day_cost += cost_i
                        day_turnover += turnover_i
                        option_trade_notional += turnover_i
                        record_trade(date, strategy, episode_id, "main_option", contract_id, q_main - old_q, main_mid, cost_i, turnover_i)

                share_delta_equiv = q_under * _under_equiv_per_share(underlying_level, hedge_px, hedge_beta)
                residual_delta_equiv = option_delta_equiv + share_delta_equiv
                main_val = q_main * main_mid * float(option_multiplier) if q_main != 0.0 and np.isfinite(main_mid) else 0.0
                under_val = q_under * hedge_px
                equity = cash + main_val + under_val
                daily_pnl = equity - prev_equity
                gross_pnl_before_costs = main_option_pnl + underlying_hedge_pnl + cash_accrual_pnl
                denom = max(abs(prev_equity), abs(main_val) + abs(under_val), 1.0)

                rows.append(
                    {
                        "date": date,
                        "trade_date": date,
                        "strategy": strategy,
                        "mode": mode,
                        "episode_id": int(episode_id),
                        "entry_date": entry_date,
                        "entry_label": entry.get("label", "scheduled"),
                        "entry_score": entry.get("entry_score", np.nan),
                        "entry_residual": entry_residual,
                        "entry_total_error": entry_total_error,
                        "entry_z": entry.get("entry_z", np.nan),
                        "equity": equity,
                        "nav": equity,
                        "net_equity": equity,
                        "cumulative_pnl": equity,
                        "daily_pnl": daily_pnl,
                        "net_pnl": daily_pnl,
                        "gross_pnl_before_costs": gross_pnl_before_costs,
                        "main_option_pnl": main_option_pnl,
                        "vega_option_pnl": 0.0,
                        "underlying_hedge_pnl": underlying_hedge_pnl,
                        "cash_accrual_pnl": cash_accrual_pnl,
                        "option_pnl": main_option_pnl,
                        "hedge_pnl": underlying_hedge_pnl,
                        "cost": day_cost,
                        "transaction_costs": day_cost,
                        "turnover": day_turnover,
                        "option_turnover": option_trade_notional,
                        "underlying_turnover": under_trade_notional,
                        "roll": int(date == start_date),
                        "roll_reason": "scheduled_entry" if date == start_date else exit_reason,
                        "exit_reason": exit_reason,
                        "skipped_day": skipped_day,
                        "stale_option_mark": stale_option_mark,
                        "hedge_symbol": hedge_symbol,
                        "hedge_beta": hedge_beta,
                        "hedge_price": hedge_px,
                        "hedge_dividend": hedge_dividend,
                        "cash": cash,
                        "under_pos": q_under,
                        "main_pos": q_main,
                        "hedge_pos": 0.0,
                        "hedge_units": q_under,
                        "vega_hedge_units": 0.0,
                        "main_contract": contract_id,
                        "hedge_contract": None,
                        "vega_hedge_contract": None,
                        "residual_delta_equiv": residual_delta_equiv,
                        "residual_delta_equiv_pre": residual_delta_equiv_pre,
                        "residual_gamma_equiv": option_gamma_equiv,
                        "residual_vega_equiv": option_vega_equiv,
                        "delta_before": residual_delta_equiv_pre,
                        "delta_after": residual_delta_equiv,
                        "gamma_exposure": option_gamma_equiv,
                        "vega_before": option_vega_equiv,
                        "vega_after": option_vega_equiv,
                        "trade_count": int(day_turnover > 0.0),
                        "traded_notional": day_turnover,
                        "option_mid": float(main_mid) if np.isfinite(main_mid) else np.nan,
                        "vega_option_mid": np.nan,
                        "spot": underlying_level,
                        "contract_key": contract_id,
                        "valuation_currency": valuation_currency,
                        "pnl_mode": pnl_mode,
                        "price_unit_detected": diagnostics["price_unit_detected"],
                        "annualization_days": float(annualization_days),
                        "returns": daily_pnl / denom,
                    }
                )

                prev_equity = equity
                prev_date = date
                prev_r = float(row["rate"]) if row is not None and "rate" in row.index and np.isfinite(row["rate"]) else 0.0
                last_main_mid = float(main_mid) if q_main != 0.0 and np.isfinite(main_mid) else np.nan
                last_hedge_px = float(hedge_px)
                days_held += 1
                if exit_after_row:
                    break

        result = pd.DataFrame(rows)
        if result.empty:
            summary_rows.append(
                {
                    "strategy": strategy,
                    "status": "skipped",
                    "total_pnl": np.nan,
                    "skipped_reason": "no scheduled entries could be opened.",
                }
            )
            continue

        daily = result.groupby("date", sort=True).agg(
            net_pnl=("net_pnl", "sum"),
            gross_pnl_before_costs=("gross_pnl_before_costs", "sum"),
            main_option_pnl=("main_option_pnl", "sum"),
            vega_option_pnl=("vega_option_pnl", "sum"),
            underlying_hedge_pnl=("underlying_hedge_pnl", "sum"),
            cash_accrual_pnl=("cash_accrual_pnl", "sum"),
            option_pnl=("option_pnl", "sum"),
            hedge_pnl=("hedge_pnl", "sum"),
            transaction_costs=("transaction_costs", "sum"),
            cost=("cost", "sum"),
            turnover=("turnover", "sum"),
            traded_notional=("traded_notional", "sum"),
            option_turnover=("option_turnover", "sum"),
            underlying_turnover=("underlying_turnover", "sum"),
            trade_count=("trade_count", "sum"),
            delta_before=("delta_before", "mean"),
            delta_after=("delta_after", "mean"),
            gamma_exposure=("gamma_exposure", "mean"),
            vega_before=("vega_before", "mean"),
            vega_after=("vega_after", "mean"),
            residual_delta_equiv=("residual_delta_equiv", "mean"),
            residual_gamma_equiv=("residual_gamma_equiv", "mean"),
            residual_vega_equiv=("residual_vega_equiv", "mean"),
            hedge_units=("hedge_units", "sum"),
            vega_hedge_units=("vega_hedge_units", "sum"),
            hedge_price=("hedge_price", "last"),
            spot=("spot", "last"),
        ).reset_index()
        daily["strategy"] = strategy
        daily["mode"] = mode
        daily["nav"] = daily["net_pnl"].cumsum()
        daily["equity"] = daily["nav"]
        daily["net_equity"] = daily["nav"]
        daily["cumulative_pnl"] = daily["nav"]
        daily["daily_pnl"] = daily["net_pnl"]
        daily["returns"] = daily["daily_pnl"] / daily["nav"].shift(1).abs().clip(lower=1.0).fillna(1.0)
        daily["drawdown"] = hedging_drawdown(daily["nav"])
        daily["vega_hedge_contract"] = None
        nav_frames.append(daily.set_index("date")["nav"].rename(strategy))
        return_frames.append(daily.set_index("date")["returns"].rename(strategy))
        pnl_frames.append(daily.set_index("date")["net_pnl"].rename(strategy))
        component_frames.append(result.assign(strategy=strategy))
        exposure_frames.append(
            result[
                [
                    "date",
                    "strategy",
                    "episode_id",
                    "entry_label",
                    "delta_before",
                    "delta_after",
                    "gamma_exposure",
                    "vega_before",
                    "vega_after",
                    "residual_delta_equiv",
                    "residual_gamma_equiv",
                    "residual_vega_equiv",
                    "hedge_units",
                    "vega_hedge_units",
                    "vega_hedge_contract",
                ]
            ],
        )
        summary_rows.append(_summarize_one(strategy, daily))

    nav = pd.concat(nav_frames, axis=1) if nav_frames else pd.DataFrame()
    returns = pd.concat(return_frames, axis=1) if return_frames else pd.DataFrame()
    pnl = pd.concat(pnl_frames, axis=1) if pnl_frames else pd.DataFrame()
    components = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    trades = pd.DataFrame(trade_rows)
    summary = pd.DataFrame(summary_rows)
    diagnostics["n_unique_main_contracts"] = len(selected_contracts)
    diagnostics["strategies_run"] = list(nav.columns)

    return {
        "nav": nav,
        "returns": returns,
        "pnl": pnl,
        "components": components,
        "exposures": exposures,
        "trades": trades,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def run_option_hedging_backtest(
    option_path: pd.DataFrame,
    spot_series: pd.Series | None = None,
    greeks: pd.DataFrame | None = None,
    strategies: list[str] | tuple[str, ...] = ("unhedged", "delta"),
    delta_band: float = 0.05,
    vega_band: float = 0.10,
    trading_cost_bps: float = 1.0,
    use_bid_ask_costs: bool = True,
    option_quantity: float = 1.0,
    option_multiplier: float = 1.0,
    vega_hedge_path: pd.DataFrame | None = None,
    vega_inner_ratio: float = 0.50,
    vega_contract_step: float = 0.25,
    max_vega_contracts: float = 1.5,
    vega_rehedge_every: int = 5,
    vega_min_hold_days: int = 5,
    delta_inner_band: float | None = None,
    delta_share_lot: float | None = None,
    delta_cooldown_days: int = 1,
    entry_dte_range: tuple[float, float] | None = None,
    exit_dte_days: float = 7.0,
    preferred_option_type: str = "call",
    target_abs_delta: float = 0.50,
    hedge_price_series: pd.Series | None = None,
    hedge_dividend_series: pd.Series | None = None,
    hedge_beta_series: pd.Series | None = None,
    hedge_symbol: str = "underlying",
    valuation_currency: str = "USD",
    quote_price_unit: str = "auto",
    pnl_mode: str = "usd_equivalent",
    annualization_days: float = 365.0,
    contract_size: float = 1.0,
    on_missing_vega_hedge: str = "skip",
    missing_option_mark: str = "error",
    min_future_quote_days: int = 5,
) -> dict:
    """
    Run a small options-specific hedging backtest.

    Options hedging requires option mark-to-market, hedge P&L, Greek exposure
    tracking, rebalancing bands, transaction costs, and bid/ask costs. It is
    intentionally separate from a stock portfolio rebalance engine.
    """
    if option_path.empty:
        raise ValueError("option_path is empty.")
    missing_policy = str(missing_option_mark).lower()
    if missing_policy not in {"error", "skip_day", "stale_with_warning"}:
        raise ValueError("missing_option_mark must be 'error', 'skip_day', or 'stale_with_warning'.")

    path = _merge_greeks(option_path, greeks)
    path = path.copy()
    path["date"] = pd.to_datetime(path["date"], errors="coerce").dt.normalize()
    path = path.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    path["spot"] = _align_spot(path, spot_series).to_numpy(dtype=float)
    input_price_unit_detected = (
        path["price_unit_detected"].dropna().iloc[0]
        if "price_unit_detected" in path.columns and path["price_unit_detected"].notna().any()
        else "unknown"
    )
    if str(pnl_mode).lower() == "usd_equivalent":
        path = quote_cleaning.convert_quotes_to_usd_equivalent(
            path,
            spot_col="spot",
            price_cols=("bid", "ask", "mid", "last", "mark", "half_spread"),
            unit=quote_price_unit,
            contract_size=contract_size,
        )
    path["contract_key"] = path.apply(_contract_key, axis=1)
    for col in ["mid", "delta", "gamma", "vega"]:
        if col not in path.columns:
            path[col] = np.nan
        path[col] = pd.to_numeric(path[col], errors="coerce")
    if "dte" not in path.columns and {"date", "expiry"}.issubset(path.columns):
        path["dte"] = (pd.to_datetime(path["expiry"], errors="coerce") - path["date"]).dt.days
    if "rel_spread" not in path.columns and {"bid", "ask", "mid"}.issubset(path.columns):
        path["rel_spread"] = (path["ask"] - path["bid"]) / path["mid"].replace(0, np.nan)
    path = _add_future_quote_count(_dedupe_contract_book(path))

    hedge_source = vega_hedge_path
    if hedge_source is None and greeks is not None and len(greeks) > len(path):
        hedge_source = greeks
    hedge_book = _prepare_hedge_book(
        hedge_source,
        greeks=greeks,
        spot_series=spot_series,
        quote_price_unit=quote_price_unit,
        pnl_mode=pnl_mode,
        contract_size=contract_size,
    )
    hedge_books = {
        pd.Timestamp(day): grp.set_index("contract_key", drop=False)
        for day, grp in hedge_book.groupby("date", sort=True)
    } if not hedge_book.empty else {}
    main_books = {
        pd.Timestamp(day): grp.set_index("contract_key", drop=False)
        for day, grp in path.groupby("date", sort=True)
    }
    trade_dates = sorted(main_books)
    if not trade_dates:
        raise ValueError("No dated option rows are available after hedging preparation.")

    date_index = pd.Series(pd.DatetimeIndex(trade_dates), index=pd.DatetimeIndex(trade_dates), name="date")
    hedge_prices = _align_market_series(
        date_index,
        hedge_price_series if hedge_price_series is not None else spot_series,
        default=np.nan,
    )
    hedge_dividends = _align_market_series(date_index, hedge_dividend_series, default=0.0)
    hedge_betas = _align_market_series(date_index, hedge_beta_series, default=1.0)
    if hedge_prices.isna().any():
        spot_by_date = path.groupby("date")["spot"].median()
        hedge_prices = hedge_prices.combine_first(spot_by_date).ffill().bfill()
    delta_inner = float(delta_inner_band) if delta_inner_band is not None else 0.25 * float(delta_band)
    share_lot = 0.0 if delta_share_lot is None else float(delta_share_lot)
    entry_range = entry_dte_range

    diagnostics: dict[str, Any] = {
        "valuation_currency": valuation_currency,
        "pnl_mode": pnl_mode,
        "price_unit_detected": input_price_unit_detected
        if str(input_price_unit_detected).lower() != "unknown"
        else (
            path.get("price_unit_detected", pd.Series(["unknown"])).dropna().iloc[0]
            if "price_unit_detected" in path.columns and path["price_unit_detected"].notna().any()
            else "unknown"
        ),
        "annualization_days": float(annualization_days),
        "contract_size": float(contract_size),
        "missing_option_mark_policy": missing_policy,
        "n_missing_option_marks": 0,
        "n_missing_option_mark_errors": 0,
        "n_missing_option_mark_skip_days": 0,
        "n_stale_option_marks": 0,
        "n_missing_contract_rolls": 0,
        "n_exit_dte_rolls": 0,
        "n_start_rolls": 0,
    }
    diagnostics["n_vega_candidate_rows"] = len(hedge_book)
    diagnostics["vega_hedge_candidate_rows"] = len(hedge_book)
    if "delta_vega" in strategies and not hedge_books:
        diagnostics["delta_vega_skipped_reason"] = "no vega hedge option book was provided."
        if on_missing_vega_hedge != "skip":
            raise ValueError("delta_vega requested but no vega hedge option book was provided.")

    nav_frames: list[pd.Series] = []
    return_frames: list[pd.Series] = []
    pnl_frames: list[pd.Series] = []
    component_frames: list[pd.DataFrame] = []
    exposure_frames: list[pd.DataFrame] = []
    trade_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    selected_main_contract_days = 0
    selected_main_contracts: set[str] = set()

    for strategy in strategies:
        if strategy == "delta_vega" and "delta_vega_skipped_reason" in diagnostics:
            summary_rows.append(
                {
                    "strategy": strategy,
                    "status": "skipped",
                    "total_pnl": np.nan,
                    "skipped_reason": diagnostics["delta_vega_skipped_reason"],
                }
            )
            continue

        mode = "none" if strategy == "unhedged" else strategy
        q_main = 0.0
        q_hedge = 0.0
        q_under = 0.0
        cash = 0.0
        main_id = None
        hedge_id = None
        last_main_mid = np.nan
        last_hedge_mid = np.nan
        last_hedge_px = np.nan
        prev_equity = 0.0
        prev_date = None
        prev_r = 0.0
        episode_id = -1
        days_since_under_rehedge = 0
        days_since_option_rehedge = 0
        days_in_episode = 0
        rows: list[dict[str, Any]] = []

        def record_trade(
            date_: pd.Timestamp,
            instrument: str,
            contract_key: str | None,
            units: float,
            price: float,
            transaction_cost: float,
            traded_notional: float,
            strategy_: str = strategy,
        ) -> None:
            if abs(float(units)) <= 1e-14 and float(traded_notional) <= 0.0:
                return
            trade_rows.append(
                {
                    "date": date_,
                    "strategy": strategy_,
                    "instrument": instrument,
                    "contract_key": contract_key,
                    "trade_units": float(units),
                    "price": float(price),
                    "transaction_cost": float(transaction_cost),
                    "traded_notional": float(traded_notional),
                }
            )

        for date in trade_dates:
            day = main_books[date]
            hedge_day_book = hedge_books.get(pd.Timestamp(date), day)
            hedge_px = float(hedge_prices.loc[pd.Timestamp(date)])
            hedge_dividend = float(hedge_dividends.loc[pd.Timestamp(date)])
            hedge_beta = float(hedge_betas.loc[pd.Timestamp(date)])
            day_cost = 0.0
            day_turnover = 0.0
            option_trade_notional = 0.0
            under_trade_notional = 0.0
            roll = False
            roll_reason = "none"
            skipped_day = 0
            stale_option_mark = 0

            current_main_row = _loc_contract(day, main_id)
            current_hedge_row = _loc_contract(hedge_day_book, hedge_id)
            current_main_mid = np.nan
            current_hedge_mid = np.nan

            if q_main != 0.0:
                current_main_mid, current_main_row, stale, skip = _resolve_option_mark(
                    current_main_row,
                    last_main_mid,
                    policy=missing_policy,
                    diagnostics=diagnostics,
                    strategy=strategy,
                    role="main",
                    contract_id=main_id,
                    date=pd.Timestamp(date),
                )
                stale_option_mark += int(stale)
                if skip:
                    skipped_day = 1
            if q_hedge != 0.0:
                current_hedge_mid, current_hedge_row, stale, skip = _resolve_option_mark(
                    current_hedge_row,
                    last_hedge_mid,
                    policy=missing_policy,
                    diagnostics=diagnostics,
                    strategy=strategy,
                    role="vega hedge",
                    contract_id=hedge_id,
                    date=pd.Timestamp(date),
                )
                stale_option_mark += int(stale)
                skipped_day = max(skipped_day, int(skip))
            if skipped_day:
                continue

            prev_main_value = q_main * last_main_mid * float(option_multiplier) if q_main != 0.0 and np.isfinite(last_main_mid) else 0.0
            prev_vega_value = q_hedge * last_hedge_mid * float(option_multiplier) if q_hedge != 0.0 and np.isfinite(last_hedge_mid) else 0.0
            prev_under_value = q_under * last_hedge_px if q_under != 0.0 and np.isfinite(last_hedge_px) else 0.0

            cash_before_accrual = cash
            if prev_date is not None:
                dt_year = max((pd.Timestamp(date) - pd.Timestamp(prev_date)).days / float(annualization_days), 0.0)
                cash *= np.exp(prev_r * dt_year)
                if q_under != 0.0:
                    cash += q_under * hedge_dividend
            cash_accrual_pnl = cash - cash_before_accrual

            main_value_pre = q_main * current_main_mid * float(option_multiplier) if q_main != 0.0 else 0.0
            vega_value_pre = q_hedge * current_hedge_mid * float(option_multiplier) if q_hedge != 0.0 else 0.0
            under_value_pre = q_under * hedge_px
            main_option_pnl = main_value_pre - prev_main_value
            vega_option_pnl = vega_value_pre - prev_vega_value
            underlying_hedge_pnl = under_value_pre - prev_under_value

            if main_id is None or q_main == 0.0:
                roll = True
                roll_reason = "start"
            elif current_main_row is None:
                roll = True
                roll_reason = "missing_contract"
            else:
                dte_value = pd.to_numeric(pd.Series([current_main_row.get("dte", np.nan)]), errors="coerce").iloc[0]
                if np.isfinite(dte_value) and float(dte_value) < float(exit_dte_days):
                    roll = True
                    roll_reason = "exit_dte"

            if roll:
                if roll_reason == "start":
                    diagnostics["n_start_rolls"] = int(diagnostics.get("n_start_rolls", 0)) + 1
                elif roll_reason == "missing_contract":
                    diagnostics["n_missing_contract_rolls"] = int(diagnostics.get("n_missing_contract_rolls", 0)) + 1
                elif roll_reason == "exit_dte":
                    diagnostics["n_exit_dte_rolls"] = int(diagnostics.get("n_exit_dte_rolls", 0)) + 1

                if q_main != 0.0:
                    px_main = float(current_main_mid)
                    row_main_for_cost = current_main_row if current_main_row is not None else _minimal_option_row(px_main)
                    old_q = q_main
                    cash, q_main, cost_i, turnover_i = _trade_option_cash(
                        cash,
                        q_main,
                        0.0,
                        px_main,
                        row_main_for_cost,
                        option_multiplier=option_multiplier,
                        trading_cost_bps=trading_cost_bps,
                        use_bid_ask_costs=use_bid_ask_costs,
                    )
                    day_cost += cost_i
                    day_turnover += turnover_i
                    option_trade_notional += turnover_i
                    record_trade(date, "main_option", main_id, q_main - old_q, px_main, cost_i, turnover_i)

                if q_hedge != 0.0:
                    px_h = float(current_hedge_mid)
                    row_h_for_cost = current_hedge_row if current_hedge_row is not None else _minimal_option_row(px_h)
                    old_q = q_hedge
                    cash, q_hedge, cost_i, turnover_i = _trade_option_cash(
                        cash,
                        q_hedge,
                        0.0,
                        px_h,
                        row_h_for_cost,
                        option_multiplier=option_multiplier,
                        trading_cost_bps=trading_cost_bps,
                        use_bid_ask_costs=use_bid_ask_costs,
                    )
                    day_cost += cost_i
                    day_turnover += turnover_i
                    option_trade_notional += turnover_i
                    record_trade(date, "vega_option", hedge_id, q_hedge - old_q, px_h, cost_i, turnover_i)

                if q_under != 0.0:
                    old_under = q_under
                    cash, q_under, cost_i, turnover_i = _trade_share(
                        cash,
                        q_under,
                        0.0,
                        hedge_px,
                        trading_cost_bps=trading_cost_bps,
                    )
                    day_cost += cost_i
                    day_turnover += turnover_i
                    under_trade_notional += turnover_i
                    record_trade(date, "underlying", None, q_under - old_under, hedge_px, cost_i, turnover_i)

                main_id = _pick_main_contract(
                    day,
                    preferred_option_type=preferred_option_type,
                    side=np.sign(option_quantity) if option_quantity != 0 else 1.0,
                    target_abs_delta=target_abs_delta,
                    entry_dte_range=entry_range,
                    min_future_quote_days=min_future_quote_days,
                )
                hedge_id = None
                q_hedge = 0.0
                q_under = 0.0
                days_since_under_rehedge = 0
                days_since_option_rehedge = 0
                days_in_episode = 0

                if main_id is not None:
                    episode_id += 1
                    main_row = _loc_contract(day, main_id)
                    main_mid = _finite_mid(main_row)
                    if not np.isfinite(main_mid):
                        main_id = None
                        q_main = 0.0
                        main_row = None
                    else:
                        old_q = q_main
                        cash, q_main, cost_i, turnover_i = _trade_option_cash(
                            cash,
                            q_main,
                            float(option_quantity),
                            main_mid,
                            main_row,
                            option_multiplier=option_multiplier,
                            trading_cost_bps=trading_cost_bps,
                            use_bid_ask_costs=use_bid_ask_costs,
                        )
                        day_cost += cost_i
                        day_turnover += turnover_i
                        option_trade_notional += turnover_i
                        record_trade(date, "main_option", main_id, q_main - old_q, main_mid, cost_i, turnover_i)
                        current_main_row = main_row
                        current_main_mid = main_mid
                        if mode == "delta_vega":
                            picked = _pick_vega_hedge(hedge_day_book, main_row)
                            hedge_id = None if picked is None else str(picked["contract_key"])
                else:
                    q_main = 0.0

            if main_id is not None and q_main != 0.0:
                main_row = _loc_contract(day, main_id)
                main_mid = _finite_mid(main_row)
            else:
                main_row = None
                main_mid = np.nan

            hedge_row = _loc_contract(hedge_day_book, hedge_id)
            hedge_mid = _finite_mid(hedge_row)
            if q_hedge != 0.0 and not np.isfinite(hedge_mid) and np.isfinite(current_hedge_mid):
                hedge_mid = current_hedge_mid

            underlying_level = float(main_row["spot"]) if main_row is not None and "spot" in main_row.index else hedge_px

            main_delta_equiv = 0.0
            main_gamma_equiv = 0.0
            main_vega_equiv = 0.0
            if main_row is not None:
                main_delta_equiv = float(option_multiplier) * q_main * _row_greek(main_row, "delta")
                main_gamma_equiv = float(option_multiplier) * q_main * _row_greek(main_row, "gamma")
                main_vega_equiv = float(option_multiplier) * q_main * _row_greek(main_row, "vega")

            if mode == "delta_vega" and main_row is not None:
                if hedge_id is None:
                    picked = _pick_vega_hedge(hedge_day_book, main_row)
                    hedge_id = None if picked is None else str(picked["contract_key"])
                    hedge_row = _loc_contract(hedge_day_book, hedge_id)
                    hedge_mid = _finite_mid(hedge_row)
                if hedge_row is not None and np.isfinite(hedge_mid):
                    hedge_vega_unit = float(option_multiplier) * _row_greek(hedge_row, "vega")
                    residual_vega_pre = main_vega_equiv + hedge_vega_unit * q_hedge
                    vega_outer_band = float(vega_band) * max(abs(main_vega_equiv), 1e-8)
                    vega_inner_band = float(vega_inner_ratio) * max(abs(main_vega_equiv), 1e-8)
                    desired_vega_residual = _band_target_residual(residual_vega_pre, vega_outer_band, vega_inner_band)
                    allow_vega_trade = roll or (
                        days_in_episode >= int(vega_min_hold_days)
                        and days_since_option_rehedge >= int(vega_rehedge_every)
                        and desired_vega_residual is not None
                    )
                    if (
                        allow_vega_trade
                        and desired_vega_residual is not None
                        and np.isfinite(hedge_vega_unit)
                        and abs(hedge_vega_unit) > 1e-8
                    ):
                        target_q_hedge = (desired_vega_residual - main_vega_equiv) / hedge_vega_unit
                        target_q_hedge = float(np.clip(target_q_hedge, -max_vega_contracts, max_vega_contracts))
                        target_q_hedge = _quantize_step(target_q_hedge, vega_contract_step)
                        if abs(target_q_hedge - q_hedge) >= float(vega_contract_step):
                            old_q = q_hedge
                            cash, q_hedge, cost_i, turnover_i = _trade_option_cash(
                                cash,
                                q_hedge,
                                target_q_hedge,
                                hedge_mid,
                                hedge_row,
                                option_multiplier=option_multiplier,
                                trading_cost_bps=trading_cost_bps,
                                use_bid_ask_costs=use_bid_ask_costs,
                            )
                            day_cost += cost_i
                            day_turnover += turnover_i
                            option_trade_notional += turnover_i
                            days_since_option_rehedge = 0
                            record_trade(date, "vega_option", hedge_id, q_hedge - old_q, hedge_mid, cost_i, turnover_i)
                        else:
                            days_since_option_rehedge += 1
                    else:
                        days_since_option_rehedge += 1
                else:
                    days_since_option_rehedge += 1
            else:
                days_since_option_rehedge += 1

            option_delta_equiv = main_delta_equiv
            option_gamma_equiv = main_gamma_equiv
            option_vega_equiv = main_vega_equiv
            hedge_row = _loc_contract(hedge_day_book, hedge_id)
            hedge_mid = _finite_mid(hedge_row)
            if q_hedge != 0.0 and not np.isfinite(hedge_mid) and np.isfinite(current_hedge_mid):
                hedge_mid = current_hedge_mid
            if hedge_row is not None:
                option_delta_equiv += float(option_multiplier) * q_hedge * _row_greek(hedge_row, "delta")
                option_gamma_equiv += float(option_multiplier) * q_hedge * _row_greek(hedge_row, "gamma")
                option_vega_equiv += float(option_multiplier) * q_hedge * _row_greek(hedge_row, "vega")

            share_delta_equiv_pre = q_under * _under_equiv_per_share(underlying_level, hedge_px, hedge_beta)
            residual_delta_equiv_pre = option_delta_equiv + share_delta_equiv_pre
            target_q_under = q_under
            if mode == "none" or main_row is None:
                should_trade_under = abs(q_under) > 0.0
                target_q_under = 0.0
            else:
                desired_delta_residual = _band_target_residual(residual_delta_equiv_pre, float(delta_band), delta_inner)
                should_trade_under = roll or (
                    desired_delta_residual is not None and days_since_under_rehedge >= int(delta_cooldown_days)
                )
                if desired_delta_residual is not None:
                    target_q_under = _target_under_pos_for_residual(
                        option_delta_equiv,
                        desired_delta_residual,
                        underlying_level,
                        hedge_px,
                        hedge_beta,
                    )
                    if share_lot > 0:
                        target_q_under = float(np.round(target_q_under / share_lot) * share_lot)

            min_trade = share_lot if share_lot > 0 else 1e-12
            if should_trade_under and abs(target_q_under - q_under) >= min_trade:
                old_under = q_under
                cash, q_under, cost_i, turnover_i = _trade_share(
                    cash,
                    q_under,
                    target_q_under,
                    hedge_px,
                    trading_cost_bps=trading_cost_bps,
                )
                day_cost += cost_i
                day_turnover += turnover_i
                under_trade_notional += turnover_i
                days_since_under_rehedge = 0
                record_trade(date, "underlying", None, q_under - old_under, hedge_px, cost_i, turnover_i)
            else:
                days_since_under_rehedge += 1

            share_delta_equiv = q_under * _under_equiv_per_share(underlying_level, hedge_px, hedge_beta)
            residual_delta_equiv = option_delta_equiv + share_delta_equiv
            main_val = q_main * main_mid * float(option_multiplier) if main_row is not None and np.isfinite(main_mid) else 0.0
            hedge_val = q_hedge * hedge_mid * float(option_multiplier) if q_hedge != 0.0 and np.isfinite(hedge_mid) else 0.0
            under_val = q_under * hedge_px
            equity = cash + main_val + hedge_val + under_val
            daily_pnl = equity - prev_equity
            gross_pnl_before_costs = main_option_pnl + vega_option_pnl + underlying_hedge_pnl + cash_accrual_pnl
            denom = max(abs(prev_equity), abs(main_val) + abs(hedge_val) + abs(under_val), 1.0)

            if main_id is not None:
                selected_main_contract_days += 1
                selected_main_contracts.add(str(main_id))

            rows.append(
                {
                    "date": date,
                    "trade_date": date,
                    "strategy": strategy,
                    "mode": mode,
                    "episode_id": int(episode_id),
                    "equity": equity,
                    "nav": equity,
                    "net_equity": equity,
                    "cumulative_pnl": equity,
                    "daily_pnl": daily_pnl,
                    "net_pnl": daily_pnl,
                    "gross_pnl_before_costs": gross_pnl_before_costs,
                    "main_option_pnl": main_option_pnl,
                    "vega_option_pnl": vega_option_pnl,
                    "underlying_hedge_pnl": underlying_hedge_pnl,
                    "cash_accrual_pnl": cash_accrual_pnl,
                    "option_pnl": main_option_pnl + vega_option_pnl,
                    "hedge_pnl": underlying_hedge_pnl,
                    "cost": day_cost,
                    "transaction_costs": day_cost,
                    "turnover": day_turnover,
                    "option_turnover": option_trade_notional,
                    "underlying_turnover": under_trade_notional,
                    "roll": int(roll),
                    "roll_reason": roll_reason,
                    "skipped_day": skipped_day,
                    "stale_option_mark": stale_option_mark,
                    "hedge_symbol": hedge_symbol,
                    "hedge_beta": hedge_beta,
                    "hedge_price": hedge_px,
                    "hedge_dividend": hedge_dividend,
                    "cash": cash,
                    "under_pos": q_under,
                    "main_pos": q_main,
                    "hedge_pos": q_hedge,
                    "hedge_units": q_under,
                    "vega_hedge_units": q_hedge,
                    "main_contract": main_id,
                    "hedge_contract": hedge_id,
                    "vega_hedge_contract": hedge_id,
                    "residual_delta_equiv": residual_delta_equiv,
                    "residual_delta_equiv_pre": residual_delta_equiv_pre,
                    "residual_gamma_equiv": option_gamma_equiv,
                    "residual_vega_equiv": option_vega_equiv,
                    "delta_before": residual_delta_equiv_pre,
                    "delta_after": residual_delta_equiv,
                    "gamma_exposure": option_gamma_equiv,
                    "vega_before": main_vega_equiv,
                    "vega_after": option_vega_equiv,
                    "trade_count": int(day_turnover > 0.0),
                    "traded_notional": day_turnover,
                    "option_mid": float(main_mid) if np.isfinite(main_mid) else np.nan,
                    "vega_option_mid": float(hedge_mid) if np.isfinite(hedge_mid) else np.nan,
                    "spot": underlying_level,
                    "contract_key": main_id,
                    "valuation_currency": valuation_currency,
                    "pnl_mode": pnl_mode,
                    "price_unit_detected": diagnostics["price_unit_detected"],
                    "annualization_days": float(annualization_days),
                    "returns": daily_pnl / denom,
                }
            )

            prev_equity = equity
            prev_date = date
            prev_r = float(main_row["rate"]) if main_row is not None and "rate" in main_row.index and np.isfinite(main_row["rate"]) else 0.0
            last_main_mid = float(main_mid) if q_main != 0.0 and np.isfinite(main_mid) else np.nan
            last_hedge_mid = float(hedge_mid) if q_hedge != 0.0 and np.isfinite(hedge_mid) else np.nan
            last_hedge_px = float(hedge_px)
            days_in_episode += 1

        result = pd.DataFrame(rows)
        if result.empty:
            summary_rows.append(
                {
                    "strategy": strategy,
                    "status": "skipped",
                    "total_pnl": np.nan,
                    "skipped_reason": "no hedgeable days after missing-mark policy.",
                }
            )
            continue
        result["drawdown"] = hedging_drawdown(result["nav"])
        nav_frames.append(result.set_index("date")["nav"].rename(strategy))
        return_frames.append(result.set_index("date")["returns"].rename(strategy))
        pnl_frames.append(result.set_index("date")["net_pnl"].rename(strategy))
        component_frames.append(result.assign(strategy=strategy))
        exposure_frames.append(
            result[
                [
                    "date",
                    "strategy",
                    "delta_before",
                    "delta_after",
                    "gamma_exposure",
                    "vega_before",
                    "vega_after",
                    "residual_delta_equiv",
                    "residual_gamma_equiv",
                    "residual_vega_equiv",
                    "hedge_units",
                    "vega_hedge_units",
                    "vega_hedge_contract",
                ]
            ],
        )
        summary_rows.append(_summarize_one(strategy, result))

    nav = pd.concat(nav_frames, axis=1) if nav_frames else pd.DataFrame()
    returns = pd.concat(return_frames, axis=1) if return_frames else pd.DataFrame()
    pnl = pd.concat(pnl_frames, axis=1) if pnl_frames else pd.DataFrame()
    components = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    exposures = pd.concat(exposure_frames, ignore_index=True) if exposure_frames else pd.DataFrame()
    trades = pd.DataFrame(trade_rows)
    summary = pd.DataFrame(summary_rows)
    diagnostics["n_option_book_rows"] = len(path)
    diagnostics["n_selected_main_contract_days"] = int(selected_main_contract_days)
    diagnostics["n_unique_main_contracts"] = len(selected_main_contracts)
    diagnostics["strategies_run"] = list(nav.columns)
    diagnostics["delta_vega_skipped"] = "delta_vega_skipped_reason" in diagnostics
    diagnostics["vega_hedge_trade_count"] = int((trades.get("instrument") == "vega_option").sum()) if not trades.empty else 0

    return {
        "nav": nav,
        "returns": returns,
        "pnl": pnl,
        "components": components,
        "exposures": exposures,
        "trades": trades,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def rolling_residual_delta(hedge_results: dict, window: int = 20) -> pd.DataFrame:
    exposures = hedge_results.get("exposures", pd.DataFrame())
    value_col = "residual_delta_equiv" if "residual_delta_equiv" in exposures.columns else "delta_after"
    if exposures.empty or value_col not in exposures.columns:
        return pd.DataFrame()
    out = (
        exposures.assign(abs_delta=lambda x: pd.to_numeric(x[value_col], errors="coerce").abs())
        .pivot_table(index="date", columns="strategy", values="abs_delta", aggfunc="mean")
        .sort_index()
    )
    return out.rolling(int(window), min_periods=1).mean()


def rolling_residual_vega(hedge_results: dict, window: int = 20) -> pd.DataFrame:
    exposures = hedge_results.get("exposures", pd.DataFrame())
    value_col = "residual_vega_equiv" if "residual_vega_equiv" in exposures.columns else "vega_after"
    if exposures.empty or value_col not in exposures.columns:
        return pd.DataFrame()
    out = (
        exposures.assign(abs_vega=lambda x: pd.to_numeric(x[value_col], errors="coerce").abs())
        .pivot_table(index="date", columns="strategy", values="abs_vega", aggfunc="mean")
        .sort_index()
    )
    return out.rolling(int(window), min_periods=1).mean()


def hedging_diagnostics(results: dict) -> pd.DataFrame:
    diagnostics = dict(results.get("diagnostics", {}))
    if not diagnostics:
        return pd.DataFrame()
    return pd.DataFrame([diagnostics])


def _summarize_one(strategy: str, result: pd.DataFrame) -> dict[str, Any]:
    pnl = result["net_pnl"].to_numpy(dtype=float)
    finite = pnl[np.isfinite(pnl)]
    nav = result["nav"]
    return {
        "strategy": strategy,
        "status": "ok",
        "n_days": len(result),
        "total_pnl": float(nav.iloc[-1]) if len(nav) else np.nan,
        "mean_daily_pnl": float(np.nanmean(finite)) if len(finite) else np.nan,
        "std_daily_pnl": float(np.nanstd(finite, ddof=1)) if len(finite) >= 2 else np.nan,
        "total_costs": float(np.nansum(result["transaction_costs"])),
        "trade_count": int(np.nansum(result["trade_count"])),
        "traded_notional": float(np.nansum(result["traded_notional"])),
        "max_drawdown": float(np.nanmin(result["drawdown"])) if len(result) else np.nan,
        "mean_abs_delta_after": float(np.nanmean(np.abs(result["delta_after"]))),
        "mean_abs_vega_after": float(np.nanmean(np.abs(result["vega_after"]))),
    }


def summarize_hedging_backtest(results: dict) -> pd.DataFrame:
    return results.get("summary", pd.DataFrame())


def hedge_trade_ledger(results: dict) -> pd.DataFrame:
    return results.get("trades", pd.DataFrame())


def matched_option_schedule(
    entry_schedule: pd.DataFrame,
    option_quotes: pd.DataFrame,
    *,
    same_date: bool = True,
    same_option_type: bool = True,
    same_quantity_sign: bool = True,
    target_abs_delta: float = 0.50,
    dte_tolerance_days: float = 7.0,
    min_future_marks: int = 3,
    selector_name: str = "matched_atm_fixed_3d",
) -> pd.DataFrame:
    schedule = _prepare_entry_schedule(entry_schedule)
    cols = [
        "entry_date",
        "contract_key",
        "quantity",
        "label",
        "entry_score",
        "entry_residual",
        "entry_total_error",
        "entry_z",
        "max_hold_days",
        "exit_on_convergence",
        "exit_on_sign_flip",
    ]
    if schedule.empty or option_quotes.empty:
        return pd.DataFrame(columns=cols)
    book = option_quotes.copy()
    book["date"] = pd.to_datetime(book["date"], errors="coerce").dt.normalize()
    if "dte" not in book.columns and "dte_days" in book.columns:
        book["dte"] = book["dte_days"]
    if "rel_spread" not in book.columns and "relative_spread" in book.columns:
        book["rel_spread"] = book["relative_spread"]
    if "half_spread" not in book.columns and {"bid", "ask"}.issubset(book.columns):
        book["half_spread"] = 0.5 * (pd.to_numeric(book["ask"], errors="coerce") - pd.to_numeric(book["bid"], errors="coerce")).clip(lower=0.0)
    if "contract_key" not in book.columns:
        book["contract_key"] = book.apply(_contract_key, axis=1)
    book = _add_future_quote_count(_dedupe_contract_book(book))
    by_date = {pd.Timestamp(d).normalize(): g.copy() for d, g in book.groupby("date", sort=False)}
    rows: list[dict[str, Any]] = []
    for _, entry in schedule.iterrows():
        d = pd.Timestamp(entry["entry_date"]).normalize()
        day = by_date.get(d)
        if day is None or day.empty:
            continue
        source = book[book["contract_key"].astype(str).eq(str(entry["contract_key"]))].sort_values("date")
        source = source[source["date"] <= d]
        source_row = source.iloc[-1] if not source.empty else None
        candidates = day.copy()
        if same_option_type and source_row is not None and "option_type" in candidates.columns:
            candidates = candidates[candidates["option_type"].astype(str).str.lower().eq(str(source_row.get("option_type", "")).lower())].copy()
        candidates = candidates[
            (pd.to_numeric(candidates.get("mid"), errors="coerce") > 0)
            & np.isfinite(pd.to_numeric(candidates.get("delta"), errors="coerce"))
        ].copy()
        if candidates.empty:
            continue
        if source_row is not None and "dte" in candidates.columns:
            target_dte = float(source_row.get("dte", source_row.get("dte_days", 35.0)))
            close = candidates[(pd.to_numeric(candidates["dte"], errors="coerce") - target_dte).abs() <= float(dte_tolerance_days)].copy()
            if close.empty:
                close = candidates[(pd.to_numeric(candidates["dte"], errors="coerce") - target_dte).abs() <= 2.0 * float(dte_tolerance_days)].copy()
            if not close.empty:
                candidates = close
        future_col = "future_consecutive_quote_days" if "future_consecutive_quote_days" in candidates.columns else "future_quote_count"
        if future_col in candidates.columns:
            durable = candidates[pd.to_numeric(candidates[future_col], errors="coerce") >= int(min_future_marks)].copy()
            if not durable.empty:
                candidates = durable
        candidates["abs_delta"] = pd.to_numeric(candidates["delta"], errors="coerce").abs()
        dte = pd.to_numeric(candidates.get("dte", pd.Series(35.0, index=candidates.index)), errors="coerce")
        rel = pd.to_numeric(candidates.get("rel_spread", pd.Series(0.0, index=candidates.index)), errors="coerce").fillna(0.0)
        future = pd.to_numeric(candidates.get(future_col, pd.Series(0.0, index=candidates.index)), errors="coerce").fillna(0.0) if future_col in candidates.columns else pd.Series(0.0, index=candidates.index)
        candidates["entry_score"] = -2.0 * (candidates["abs_delta"] - float(target_abs_delta)).abs() - 0.03 * (dte - dte.median()).abs().fillna(0.0) - 1.5 * rel + 0.02 * future
        row = candidates.sort_values("entry_score", ascending=False).iloc[0]
        quantity = float(entry["quantity"])
        if same_quantity_sign:
            quantity = np.sign(quantity) * abs(float(entry.get("quantity", 1.0)))
        rows.append(
            {
                "entry_date": d,
                "contract_key": row["contract_key"],
                "quantity": quantity,
                "label": selector_name,
                "entry_score": float(row["entry_score"]),
                "entry_residual": np.nan,
                "entry_total_error": np.nan,
                "entry_z": np.nan,
                "max_hold_days": entry.get("max_hold_days", np.nan),
                "exit_on_convergence": False,
                "exit_on_sign_flip": False,
            },
        )
    return pd.DataFrame(rows, columns=cols)


def hedge_book_from_schedules(
    option_quotes: pd.DataFrame,
    schedules: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
    *,
    lookahead_days: int = 12,
) -> pd.DataFrame:
    if option_quotes.empty:
        return option_quotes.copy()
    valid = [s for s in schedules if s is not None and not s.empty and "contract_key" in s.columns]
    if not valid:
        return option_quotes.head(0).copy()
    schedule = pd.concat(valid, ignore_index=True)
    keys = set(schedule["contract_key"].astype(str))
    first = pd.to_datetime(schedule["entry_date"], errors="coerce").min() - pd.Timedelta(days=3)
    last = pd.to_datetime(schedule["entry_date"], errors="coerce").max() + pd.Timedelta(days=int(lookahead_days))
    book = option_quotes.copy()
    book["date"] = pd.to_datetime(book["date"], errors="coerce").dt.normalize()
    if "contract_key" not in book.columns:
        book["contract_key"] = book.apply(_contract_key, axis=1)
    out = book[book["contract_key"].astype(str).isin(keys) & book["date"].between(first, last)].copy()
    if "dte" not in out.columns and "dte_days" in out.columns:
        out["dte"] = out["dte_days"]
    if "rel_spread" not in out.columns and "relative_spread" in out.columns:
        out["rel_spread"] = out["relative_spread"]
    if "half_spread" not in out.columns and {"bid", "ask"}.issubset(out.columns):
        out["half_spread"] = 0.5 * (pd.to_numeric(out["ask"], errors="coerce") - pd.to_numeric(out["bid"], errors="coerce")).clip(lower=0.0)
    if "timestamp" not in out.columns:
        out["timestamp"] = out["date"]
    return out.sort_values(["date", "contract_key"]).reset_index(drop=True)


def scheduled_hedge_comparison(
    results: dict[str, dict],
    *,
    normalize_by: tuple[str, ...] = ("premium", "traded_notional", "initial_vega"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run, result in results.items():
        summary = summarize_hedging_backtest(result).copy()
        components = result.get("components", pd.DataFrame()).copy()
        if summary.empty:
            rows.append({"run": run, "strategy": "none", "status": "skipped"})
            continue
        for _, row in summary.iterrows():
            out = row.to_dict()
            out["run"] = run
            c = components[components["strategy"].eq(row.get("strategy"))].copy() if not components.empty and "strategy" in components.columns else pd.DataFrame()
            if not c.empty:
                first = c.sort_values(["episode_id", "date"]).groupby("episode_id", as_index=False).head(1)
                premium = float(np.nansum(first["main_pos"].abs() * first["option_mid"].abs()))
                initial_vega = float(np.nansum(np.abs(first.get("vega_after", 0.0))))
                avg_delta = float(np.nanmean(np.abs(c.get("delta_after", np.nan))))
            else:
                premium = np.nan
                initial_vega = np.nan
                avg_delta = np.nan
            out["entry_premium"] = premium
            out["initial_vega"] = initial_vega
            out["average_abs_delta"] = avg_delta
            out["pnl_per_premium"] = out.get("total_pnl", np.nan) / max(abs(premium), 1e-12) if np.isfinite(premium) else np.nan
            out["pnl_per_traded_notional"] = out.get("total_pnl", np.nan) / max(abs(out.get("traded_notional", np.nan)), 1e-12) if np.isfinite(out.get("traded_notional", np.nan)) else np.nan
            out["pnl_per_initial_vega"] = out.get("total_pnl", np.nan) / max(abs(initial_vega), 1e-12) if np.isfinite(initial_vega) else np.nan
            out["cost_pct_premium"] = out.get("total_costs", np.nan) / max(abs(premium), 1e-12) if np.isfinite(premium) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def option_fill_price(row: pd.Series | dict, side: float, action: str = "open") -> float:
    data = row if isinstance(row, pd.Series) else pd.Series(row)
    bid = float(pd.to_numeric(data.get("bid", np.nan), errors="coerce"))
    ask = float(pd.to_numeric(data.get("ask", np.nan), errors="coerce"))
    mid = float(pd.to_numeric(data.get("mid", 0.5 * (bid + ask)), errors="coerce"))
    if not np.isfinite(bid) or not np.isfinite(ask):
        return mid
    if float(side) > 0:
        return ask if str(action).lower().startswith("open") else bid
    return bid if str(action).lower().startswith("open") else ask


def mark_option_position(row: pd.Series | dict, quantity: float, *, multiplier: float = 100.0, price_col: str = "mid") -> float:
    data = row if isinstance(row, pd.Series) else pd.Series(row)
    price = float(pd.to_numeric(data.get(price_col, data.get("mid", np.nan)), errors="coerce"))
    return float(quantity) * price * float(multiplier)


def settle_option_expiry(option_type: str, spot: float, strike: float, quantity: float, *, multiplier: float = 100.0) -> float:
    if str(option_type).lower().startswith("c"):
        payoff = max(float(spot) - float(strike), 0.0)
    else:
        payoff = max(float(strike) - float(spot), 0.0)
    return float(quantity) * payoff * float(multiplier)


def close_option_position(row: pd.Series | dict, quantity: float, *, multiplier: float = 100.0) -> float:
    price = option_fill_price(row, quantity, action="close")
    return float(quantity) * price * float(multiplier)


def option_trade_ledger(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(rows).copy()
    if out.empty:
        return out
    for col in ["date", "entry_date", "exit_date", "expiry"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    for col in ["quantity", "price", "cashflow", "spread_cost", "pnl"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


__all__ = [
    "apply_hedge_transaction_costs",
    "compute_hedge_pnl",
    "compute_option_mark_to_market_pnl",
    "hedge_book_from_schedules",
    "hedge_trade_ledger",
    "hedging_diagnostics",
    "close_option_position",
    "mark_option_position",
    "hedging_drawdown",
    "matched_option_schedule",
    "rolling_residual_delta",
    "rolling_residual_vega",
    "run_delta_hedge_backtest",
    "run_delta_vega_hedge_backtest",
    "run_option_hedging_backtest",
    "run_scheduled_option_hedging_backtest",
    "scheduled_hedge_comparison",
    "summarize_hedging_backtest",
    "option_fill_price",
    "option_trade_ledger",
    "settle_option_expiry",
]
