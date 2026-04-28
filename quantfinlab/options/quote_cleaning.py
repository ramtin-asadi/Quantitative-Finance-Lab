from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

_DATE_COLUMNS = ["Date", "date", "TIMESTAMP", "timestamp", "Trade Date", "TRADE_DATE", "QUOTE_DATE", "QUOTE_READTIME"]
_EXPIRY_COLUMNS = ["Expiry", "EXPIRY_DT", "expiry", "expiryDate", "Expiry Date", "EXPIRY_DATE"]
_STRIKE_COLUMNS = ["Strike", "STRIKE_PR", "strike", "strikePrice", "STRIKE_PRICE", "strike_price"]
_TYPE_COLUMNS = ["OptionType", "OPTION_TYP", "option_type", "type", "CE", "PE", "CALL", "PUT", "OPTION_RIGHT"]
_UNDERLYING_COLUMNS = ["Symbol", "SYMBOL", "underlying", "instrument", "INSTRUMENT", "symbol", "BASE_CURRENCY"]
_SPOT_COLUMNS = [
    "Spot",
    "spot",
    "underlying_price",
    "Close_Underlying",
    "Futures Price",
    "FUTURE_PRICE",
    "underlying_value",
    "underlying_last",
    "UNDERLYING_PRICE",
]
_BID_COLUMNS = ["Bid", "bid", "best_bid", "BID", "BID_PRICE"]
_ASK_COLUMNS = ["Ask", "ask", "best_ask", "ASK", "ASK_PRICE"]
_LAST_COLUMNS = ["LTP", "ltp", "last", "Last", "LAST_PRICE"]
_MARK_COLUMNS = ["mark", "Mark", "MARK", "MARK_PRICE"]
_CLOSE_COLUMNS = ["Close", "close", "SETTLE_PR", "settlement", "option_close", "settle_price"]
_VOLUME_COLUMNS = ["Volume", "volume", "CONTRACTS", "no_of_contracts", "VOLUME"]
_OI_COLUMNS = ["OI", "OPEN_INT", "open_interest", "open_int", "OPEN_INTEREST"]
_INSTRUMENT_COLUMNS = ["instrument_name", "INSTRUMENT_NAME", "instrument", "symbol"]
_TIMESTAMP_COLUMNS = ["timestamp", "quote_readtime", "QUOTE_READTIME", "quote_unixtime", "QUOTE_UNIXTIME"]
_IV_COLUMNS = ["iv", "implied_volatility", "MARK_IV", "mark_iv"]
_GREEK_COLUMNS = {
    "delta": ["delta", "DELTA"],
    "gamma": ["gamma", "GAMMA"],
    "vega": ["vega", "VEGA"],
    "theta": ["theta", "THETA"],
    "rho": ["rho", "RHO"],
}


def _to_datetime_ns(values) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").astype("datetime64[ns]")


def _normalize_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^\[|\]$", "", text).strip()
    text = text.replace("[", "").replace("]", "")
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _normalized_lookup(columns: Iterable[Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for col in columns:
        lookup[_normalize_key(col)] = col
    return lookup


def _find_column(columns: Iterable[Any], candidates: Iterable[str]) -> Any | None:
    lookup = _normalized_lookup(columns)
    for candidate in candidates:
        key = _normalize_key(candidate)
        if key in lookup:
            return lookup[key]
    return None


def parse_option_type(x: Any) -> str | float:
    """Map common option type labels to 'call' or 'put'."""
    if pd.isna(x):
        return np.nan
    text = str(x).strip().upper()
    if text in {"CE", "C", "CALL"}:
        return "call"
    if text in {"PE", "P", "PUT"}:
        return "put"
    return np.nan


def _parse_btc_deribit_instrument(values: pd.Series) -> pd.DataFrame:
    text = values.astype(str).str.strip()
    parts = text.str.extract(
        r"^(?P<underlying>[A-Z0-9]+)-(?P<expiry_token>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>[0-9.]+)-(?P<option_code>[CP])$",
        flags=re.IGNORECASE,
    )
    expiry = pd.to_datetime(parts["expiry_token"], format="%d%b%y", errors="coerce")
    out = pd.DataFrame(index=values.index)
    out["underlying_from_instrument"] = parts["underlying"].str.upper()
    out["expiry_from_instrument"] = expiry
    out["strike_from_instrument"] = pd.to_numeric(parts["strike"], errors="coerce")
    out["option_type_from_instrument"] = parts["option_code"].map({"C": "call", "P": "put", "c": "call", "p": "put"})
    return out


def normalize_option_quote_schema(
    raw: pd.DataFrame,
    profile: str = "spx_optiondx",
    underlying_default: str | None = None,
) -> pd.DataFrame:
    """Normalize already-loaded option quote tables to long-form columns."""
    if raw.empty:
        return raw.copy()

    profile_key = str(profile or "spx_optiondx").lower()
    data = raw.copy()
    out = data.copy()

    mapping = {
        "date": _find_column(data.columns, _DATE_COLUMNS),
        "timestamp": _find_column(data.columns, _TIMESTAMP_COLUMNS),
        "expiry": _find_column(data.columns, _EXPIRY_COLUMNS),
        "strike": _find_column(data.columns, _STRIKE_COLUMNS),
        "option_type": _find_column(data.columns, _TYPE_COLUMNS),
        "underlying": _find_column(data.columns, _UNDERLYING_COLUMNS),
        "instrument_name": _find_column(data.columns, _INSTRUMENT_COLUMNS),
        "spot": _find_column(data.columns, _SPOT_COLUMNS),
        "bid": _find_column(data.columns, _BID_COLUMNS),
        "ask": _find_column(data.columns, _ASK_COLUMNS),
        "last": _find_column(data.columns, _LAST_COLUMNS),
        "mark": _find_column(data.columns, _MARK_COLUMNS),
        "close": _find_column(data.columns, _CLOSE_COLUMNS),
        "volume": _find_column(data.columns, _VOLUME_COLUMNS),
        "open_interest": _find_column(data.columns, _OI_COLUMNS),
        "iv": _find_column(data.columns, _IV_COLUMNS),
    }
    for greek, candidates in _GREEK_COLUMNS.items():
        mapping[greek] = _find_column(data.columns, candidates)

    for standard, source in mapping.items():
        if source is not None and standard not in out.columns:
            out[standard] = data[source]

    if "instrument_name" in out.columns:
        parsed = _parse_btc_deribit_instrument(out["instrument_name"])
        if "underlying" not in out.columns:
            out["underlying"] = parsed["underlying_from_instrument"]
        else:
            out["underlying"] = out["underlying"].combine_first(parsed["underlying_from_instrument"])
        if "expiry" not in out.columns:
            out["expiry"] = parsed["expiry_from_instrument"]
        else:
            out["expiry"] = out["expiry"].combine_first(parsed["expiry_from_instrument"])
        if "strike" not in out.columns:
            out["strike"] = parsed["strike_from_instrument"]
        else:
            out["strike"] = pd.to_numeric(out["strike"], errors="coerce").combine_first(parsed["strike_from_instrument"])
        if "option_type" not in out.columns:
            out["option_type"] = parsed["option_type_from_instrument"]
        else:
            out["option_type"] = out["option_type"].combine_first(parsed["option_type_from_instrument"])

    if "underlying" not in out.columns and underlying_default is not None:
        out["underlying"] = underlying_default
    elif underlying_default is not None:
        out["underlying"] = out["underlying"].fillna(underlying_default)

    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].map(parse_option_type)

    for col in ["date", "timestamp", "expiry"]:
        if col in out.columns:
            if col == "timestamp" and pd.api.types.is_numeric_dtype(out[col]):
                out[col] = pd.to_datetime(out[col], unit="s", errors="coerce").astype("datetime64[ns]")
            else:
                out[col] = _to_datetime_ns(out[col])
    if "date" not in out.columns and "timestamp" in out.columns:
        out["date"] = out["timestamp"].dt.normalize()
    if "timestamp" not in out.columns and "date" in out.columns:
        out["timestamp"] = out["date"]

    numeric_cols = [
        "strike",
        "spot",
        "bid",
        "ask",
        "last",
        "mark",
        "close",
        "mid",
        "volume",
        "open_interest",
        "iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "iv" in out.columns:
        med_iv = float(np.nanmedian(out["iv"])) if out["iv"].notna().any() else np.nan
        if np.isfinite(med_iv) and med_iv > 5.0:
            out["iv"] = out["iv"] / 100.0
    if profile_key == "btc_deribit":
        out["underlying"] = out.get("underlying", underlying_default or "BTC")
        out["quote_profile"] = "btc_deribit"
    else:
        out["quote_profile"] = profile_key

    preferred = [
        "date",
        "timestamp",
        "expiry",
        "strike",
        "option_type",
        "underlying",
        "instrument_name",
        "spot",
        "bid",
        "ask",
        "last",
        "mark",
        "close",
        "mid",
        "volume",
        "open_interest",
        "iv",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "quote_profile",
    ]
    remaining = [c for c in out.columns if c not in preferred]
    return out[[c for c in preferred if c in out.columns] + remaining]


def normalize_spx_option_schema(raw: pd.DataFrame, underlying_default: str | None = "SPX") -> pd.DataFrame:
    return normalize_option_quote_schema(raw, profile="spx_optiondx", underlying_default=underlying_default)


def normalize_btc_deribit_option_schema(raw: pd.DataFrame, underlying_default: str | None = "BTC") -> pd.DataFrame:
    return normalize_option_quote_schema(raw, profile="btc_deribit", underlying_default=underlying_default)


def wide_option_chain_to_long(
    raw: pd.DataFrame,
    *,
    underlying_default: str | None = "SPX",
    include_greeks: bool = True,
) -> pd.DataFrame:
    """
    Convert one-row-per-strike call/put chains into the long option quote schema.

    The SPX OptionDx files used in Projects 4 and 5 store call and put quotes in
    separate ``c_*`` and ``p_*`` columns on the same row. Most reusable option
    helpers expect one row per option contract, so this function is the adapter
    between the vendor-shaped file and the Project 4 cleaning, parity, IV, and
    Greek modules.
    """
    if raw.empty:
        return raw.copy()

    cols = set(raw.columns)
    if {"option_type", "bid", "ask"}.issubset(cols):
        return normalize_option_quote_schema(raw, underlying_default=underlying_default)
    if not ({"c_bid", "c_ask", "p_bid", "p_ask"} & cols):
        return normalize_option_quote_schema(raw, underlying_default=underlying_default)

    data = raw.copy()
    if "source_index" not in data.columns:
        data["source_index"] = data.index

    common_candidates = {
        "date": ["quote_date", "date", "QUOTE_DATE"],
        "timestamp": ["quote_readtime", "timestamp", "QUOTE_READTIME", "quote_unixtime", "QUOTE_UNIXTIME"],
        "expiry": ["expire_date", "expiry", "EXPIRY_DT", "expiryDate"],
        "strike": ["strike", "STRIKE_PR", "STRIKE_PRICE"],
        "spot": ["underlying_last", "underlying_price", "spot", "Spot"],
        "underlying": ["underlying", "symbol", "Symbol", "SYMBOL"],
        "dte": ["dte", "DTE"],
    }

    common: dict[str, Any] = {"source_index": "source_index"}
    for standard, candidates in common_candidates.items():
        source = _find_column(data.columns, candidates)
        if source is not None:
            common[standard] = source

    option_maps = {
        "call": {
            "bid": "c_bid",
            "ask": "c_ask",
            "last": "c_last",
            "volume": "c_volume",
            "open_interest": "c_open_interest",
            "iv": "c_iv",
            "delta": "c_delta",
            "gamma": "c_gamma",
            "vega": "c_vega",
            "theta": "c_theta",
            "rho": "c_rho",
        },
        "put": {
            "bid": "p_bid",
            "ask": "p_ask",
            "last": "p_last",
            "volume": "p_volume",
            "open_interest": "p_open_interest",
            "iv": "p_iv",
            "delta": "p_delta",
            "gamma": "p_gamma",
            "vega": "p_vega",
            "theta": "p_theta",
            "rho": "p_rho",
        },
    }

    pieces: list[pd.DataFrame] = []
    for option_type, mapping in option_maps.items():
        frame = pd.DataFrame(index=data.index)
        for standard, source in common.items():
            frame[standard] = data[source]
        frame["option_type"] = option_type
        for standard, source in mapping.items():
            if source not in data.columns:
                continue
            if not include_greeks and standard in _GREEK_COLUMNS:
                continue
            frame[standard] = data[source]
        pieces.append(frame)

    out = pd.concat(pieces, ignore_index=True)
    if underlying_default is not None:
        if "underlying" not in out.columns:
            out["underlying"] = underlying_default
        else:
            out["underlying"] = out["underlying"].fillna(underlying_default)
    return normalize_option_quote_schema(out, underlying_default=underlying_default)


def ensure_option_mid_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    """Create a usable mid quote from bid/ask, close, or last prices."""
    out = quotes.copy()
    for col in ["bid", "ask", "last", "mark", "close", "mid"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "mid" not in out.columns:
        out["mid"] = np.nan
    valid_mid = out["mid"].notna() & (out["mid"] > 0)

    if {"bid", "ask"}.issubset(out.columns):
        bid = out["bid"]
        ask = out["ask"]
        spread_ok = bid.notna() & ask.notna() & (bid >= 0) & (ask >= bid)
        out.loc[~valid_mid & spread_ok, "mid"] = 0.5 * (bid[spread_ok] + ask[spread_ok])

    fallback_cols = [c for c in ["mark", "close", "last"] if c in out.columns]
    for col in fallback_cols:
        valid_mid = out["mid"].notna() & (out["mid"] > 0)
        fallback = out[col].notna() & (out[col] > 0)
        out.loc[~valid_mid & fallback, "mid"] = out.loc[~valid_mid & fallback, col]

    if {"bid", "ask"}.issubset(out.columns):
        out["spread"] = out["ask"] - out["bid"]
        out["half_spread"] = 0.5 * out["spread"]
        out["rel_spread"] = out["spread"] / out["mid"].replace(0, np.nan)
    elif "rel_spread" not in out.columns:
        out["spread"] = np.nan
        out["half_spread"] = np.nan
        out["rel_spread"] = np.nan

    return out


def attach_spot_from_series(
    quotes: pd.DataFrame,
    spot_series: pd.Series,
    date_col: str = "date",
    spot_col: str = "spot",
    method: str = "previous",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Attach spot values using same-day or previous available observations only."""
    if method != "previous":
        raise ValueError("Only method='previous' is supported.")
    if spot_series.empty:
        raise ValueError("spot_series is empty; no spot can be attached.")

    out = quotes.copy()
    out[date_col] = _to_datetime_ns(out[date_col])
    if spot_col not in out.columns:
        out[spot_col] = np.nan
    out[spot_col] = pd.to_numeric(out[spot_col], errors="coerce")

    needs_spot = overwrite | out[spot_col].isna() | (out[spot_col] <= 0)
    if needs_spot.any():
        spot = spot_series.copy()
        spot.index = pd.DatetimeIndex(pd.to_datetime(spot.index, errors="coerce")).astype("datetime64[ns]")
        spot = spot.sort_index()
        spot = pd.to_numeric(spot, errors="coerce").dropna()
        if spot.empty:
            raise ValueError("spot_series has no numeric values.")

        left = (
            out.loc[needs_spot, [date_col]]
            .reset_index()
            .sort_values(date_col)
            .rename(columns={"index": "_row"})
        )
        right = spot.rename("_spot").reset_index().rename(columns={"index": date_col})
        matched = pd.merge_asof(left, right.sort_values(date_col), on=date_col, direction="backward")
        matched = matched.set_index("_row")["_spot"]
        out.loc[matched.index, spot_col] = matched

    valid = out[spot_col].notna() & (out[spot_col] > 0)
    if not valid.any():
        raise ValueError("No valid spot values could be attached to option quotes.")
    return out


def prepare_underlying_series(
    raw: pd.DataFrame,
    date_col: str = "date",
    price_col: str | None = None,
    name: str | None = None,
) -> pd.Series:
    """Build a clean daily underlying price series from an already-loaded table."""
    if raw.empty:
        raise ValueError("raw underlying table is empty.")
    data = raw.copy()
    lookup = _normalized_lookup(data.columns)
    date_source = data.columns[0] if date_col not in data.columns else date_col
    if _normalize_key(date_col) in lookup:
        date_source = lookup[_normalize_key(date_col)]
    if price_col is None:
        for candidate in ["adj_close", "Adj Close", "close", "Close", "price", "spot"]:
            key = _normalize_key(candidate)
            if key in lookup:
                price_col = lookup[key]
                break
    elif _normalize_key(price_col) in lookup:
        price_col = lookup[_normalize_key(price_col)]
    if price_col is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Could not infer a numeric underlying price column.")
        price_col = numeric_cols[0]
    dates = pd.to_datetime(data[date_source], errors="coerce").dt.normalize()
    values = pd.to_numeric(data[price_col], errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=dates, name=name or str(price_col))
    series = series[~series.index.isna()].dropna().sort_index()
    series = series[series > 0]
    if series.index.has_duplicates:
        series = series.groupby(level=0).last()
    if series.empty:
        raise ValueError("No positive underlying prices remain after cleaning.")
    return series


def extract_spot_series(quotes: pd.DataFrame, date_col: str = "date", spot_col: str = "spot") -> pd.Series:
    """Extract one spot observation per quote date from normalized quotes."""
    if date_col not in quotes.columns or spot_col not in quotes.columns:
        raise ValueError(f"quotes must contain {date_col!r} and {spot_col!r}.")
    data = quotes[[date_col, spot_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data[spot_col] = pd.to_numeric(data[spot_col], errors="coerce")
    data = data.dropna()
    data = data[data[spot_col] > 0]
    if data.empty:
        raise ValueError("No usable spot observations found.")
    return data.groupby(date_col)[spot_col].median().sort_index().rename("spot")


def detect_option_price_unit(quotes: pd.DataFrame) -> str:
    """Detect whether option prices are USD, BTC/base, or unknown."""
    cols = {_normalize_key(c): c for c in quotes.columns}
    for key, col in cols.items():
        if key in {"price_unit", "premium_unit", "quote_price_unit", "price_unit_detected"}:
            vals = quotes[col].dropna().astype(str).str.lower()
            if vals.str.contains("usd|usdc|usdt|dollar").any():
                return "usd"
            if vals.str.contains("btc|base|coin|underlying").any():
                return "base"
        if key in {"currency", "valuation_currency", "quote_currency"}:
            vals = quotes[col].dropna().astype(str).str.upper()
            if vals.eq("USD").any() or vals.isin(["USDC", "USDT"]).any():
                return "usd"
            if vals.eq("BTC").any():
                return "base"

    price_cols = [c for c in ["mid", "mark", "last", "bid", "ask"] if c in quotes.columns]
    if not price_cols:
        return "unknown"
    prices = pd.concat([pd.to_numeric(quotes[c], errors="coerce") for c in price_cols], axis=0)
    prices = prices.replace([np.inf, -np.inf], np.nan).dropna()
    prices = prices[prices > 0]
    if prices.empty:
        return "unknown"
    median_price = float(prices.median())
    if "spot" in quotes.columns:
        spot = pd.to_numeric(quotes["spot"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        spot = spot[spot > 0]
        if not spot.empty:
            median_spot = float(spot.median())
            ratio = median_price / max(median_spot, 1e-12)
            if median_price < 10.0 and ratio < 0.02 and median_spot > 1000.0:
                return "base"
            if median_price > 50.0 or ratio > 0.02:
                return "usd"
    if median_price > 50.0:
        return "usd"
    return "unknown"


def convert_quotes_to_usd_equivalent(
    quotes: pd.DataFrame,
    spot_col: str = "spot",
    price_cols: tuple[str, ...] = ("bid", "ask", "mid", "last", "mark"),
    unit: str = "auto",
    contract_size: float = 1.0,
) -> pd.DataFrame:
    """Convert BTC/base-denominated option premiums to USD-equivalent terms."""
    out = quotes.copy()
    detected = detect_option_price_unit(out) if str(unit).lower() == "auto" else str(unit).lower()
    if detected == "btc":
        detected = "base"
    if detected not in {"usd", "base", "unknown"}:
        raise ValueError("unit must be one of {'auto', 'usd', 'base', 'btc', 'unknown'}.")

    if spot_col not in out.columns and detected == "base":
        raise ValueError(f"{spot_col!r} is required to convert base premiums to USD.")
    spot = pd.to_numeric(out[spot_col], errors="coerce") if spot_col in out.columns else pd.Series(np.nan, index=out.index)
    scale = spot * float(contract_size)
    for col in price_cols:
        if col not in out.columns:
            continue
        raw_col = f"{col}_raw"
        if raw_col not in out.columns:
            out[raw_col] = out[col]
        if detected == "base":
            raw = pd.to_numeric(out[raw_col], errors="coerce")
            out[col] = raw * scale
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["valuation_currency"] = "USD"
    out["price_unit_detected"] = detected
    out["contract_size"] = float(contract_size)
    return out


def add_time_to_expiry(
    quotes: pd.DataFrame,
    date_col: str = "date",
    expiry_col: str = "expiry",
    annualization_days: float = 365.0,
) -> pd.DataFrame:
    out = quotes.copy()
    out[date_col] = _to_datetime_ns(out[date_col])
    out[expiry_col] = _to_datetime_ns(out[expiry_col])
    dte = (out[expiry_col] - out[date_col]).dt.total_seconds() / 86400.0
    out["dte"] = dte
    out["tau"] = dte / float(annualization_days)
    out["annualization_days"] = float(annualization_days)
    return out


def add_moneyness(
    quotes: pd.DataFrame,
    spot_col: str = "spot",
    strike_col: str = "strike",
) -> pd.DataFrame:
    out = quotes.copy()
    spot = pd.to_numeric(out[spot_col], errors="coerce")
    strike = pd.to_numeric(out[strike_col], errors="coerce")
    out["moneyness"] = strike / spot.replace(0, np.nan)
    out["log_moneyness"] = np.log(out["moneyness"].where(out["moneyness"] > 0))
    return out


def split_calls_puts(quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = quotes.copy()
    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].map(parse_option_type)
    return out[out["option_type"] == "call"].copy(), out[out["option_type"] == "put"].copy()


def pair_put_call_quotes(
    quotes: pd.DataFrame,
    on: tuple[str, ...] = ("date", "expiry", "strike"),
    price_col: str = "mid",
) -> pd.DataFrame:
    calls, puts = split_calls_puts(quotes)
    common = [c for c in on if c in quotes.columns]
    keep = list(dict.fromkeys([*common, price_col, "bid", "ask", "spot", "tau", "rate", "moneyness"]))
    calls = calls[[c for c in keep if c in calls.columns]].rename(
        columns={c: f"call_{c}" for c in keep if c not in common}
    )
    puts = puts[[c for c in keep if c in puts.columns]].rename(
        columns={c: f"put_{c}" for c in keep if c not in common}
    )
    return calls.merge(puts, on=common, how="inner")


def _closest_atm_pairs_impl(quotes: pd.DataFrame, n_pairs: int | None = 25) -> pd.DataFrame:
    if n_pairs is None or n_pairs <= 0 or quotes.empty:
        return quotes.copy()
    out = quotes.copy()
    if "log_moneyness" not in out.columns:
        out = add_moneyness(out)
    group_cols = [c for c in ["date", "expiry"] if c in out.columns]
    if not group_cols:
        return out
    strike_score = (
        out.groupby([*group_cols, "strike"], dropna=False)["log_moneyness"]
        .apply(lambda s: float(np.nanmedian(np.abs(s))))
        .rename("atm_score")
        .reset_index()
    )
    strike_score["atm_rank"] = strike_score.groupby(group_cols)["atm_score"].rank(
        method="first",
        ascending=True,
    )
    selected = strike_score[strike_score["atm_rank"] <= int(n_pairs)]
    return out.merge(selected[[*group_cols, "strike"]], on=[*group_cols, "strike"], how="inner")


def closest_atm_pairs(quotes: pd.DataFrame, n_pairs: int = 25) -> pd.DataFrame:
    return _closest_atm_pairs_impl(quotes, n_pairs=n_pairs)


def _count_pairs_per_group(quotes: pd.DataFrame) -> pd.Series:
    if quotes.empty:
        return pd.Series(dtype=int)
    group_cols = [c for c in ["date", "expiry"] if c in quotes.columns]
    if len(group_cols) < 2:
        return pd.Series(dtype=int)
    grouped = quotes.groupby([*group_cols, "strike"])["option_type"].nunique()
    paired_strikes = grouped[grouped >= 2].reset_index().groupby(group_cols).size()
    return paired_strikes


def clean_option_quotes(
    quotes: pd.DataFrame,
    *,
    min_dte: int = 7,
    max_dte: int = 120,
    moneyness_range: tuple[float, float] = (0.85, 1.15),
    max_relative_spread: float = 0.20,
    closest_atm_pairs: int | None = 25,
    min_pairs_per_expiry: int = 10,
    annualization_days: float = 365.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize and filter a liquid near-ATM option panel."""
    report: list[dict[str, Any]] = []

    def add_step(step: str, frame: pd.DataFrame, previous: int | None) -> int:
        rows = len(frame)
        report.append({"step": step, "rows": rows, "removed": None if previous is None else previous - rows})
        return rows

    raw_rows = add_step("raw rows", quotes, None)
    out = normalize_option_quote_schema(quotes)
    out = ensure_option_mid_quotes(out)
    raw_rows = add_step("after schema normalization", out, raw_rows)

    if "tau" not in out.columns or "dte" not in out.columns:
        out = add_time_to_expiry(out, annualization_days=annualization_days)
    if "moneyness" not in out.columns:
        out = add_moneyness(out)

    type_ok = out["option_type"].isin(["call", "put"])
    price_ok = (
        type_ok
        & (pd.to_numeric(out["strike"], errors="coerce") > 0)
        & (pd.to_numeric(out["spot"], errors="coerce") > 0)
        & (pd.to_numeric(out["tau"], errors="coerce") > 0)
        & (pd.to_numeric(out["mid"], errors="coerce") > 0)
    )
    out = out.loc[price_ok].copy()
    raw_rows = add_step("after positive price filter", out, raw_rows)

    if {"bid", "ask"}.issubset(out.columns):
        bid = pd.to_numeric(out["bid"], errors="coerce")
        ask = pd.to_numeric(out["ask"], errors="coerce")
        both = bid.notna() & ask.notna()
        spread_ok = pd.Series(True, index=out.index)
        mid = pd.to_numeric(out["mid"], errors="coerce")
        rel = (ask - bid) / mid.replace(0, np.nan)
        spread_ok.loc[both] = (
            (bid[both] >= 0)
            & (ask[both] > bid[both])
            & (mid[both] >= bid[both])
            & (mid[both] <= ask[both])
            & (rel[both] <= max_relative_spread)
        )
        out = out.loc[spread_ok].copy()
    raw_rows = add_step("after bid/ask/spread filter", out, raw_rows)

    out = out.loc[(out["dte"] >= min_dte) & (out["dte"] <= max_dte)].copy()
    raw_rows = add_step("after DTE filter", out, raw_rows)

    lo, hi = moneyness_range
    out = out.loc[(out["moneyness"] >= lo) & (out["moneyness"] <= hi)].copy()
    raw_rows = add_step("after moneyness filter", out, raw_rows)

    out = _closest_atm_pairs_impl(out, n_pairs=closest_atm_pairs)
    raw_rows = add_step("after ATM-pair selection", out, raw_rows)

    if min_pairs_per_expiry and min_pairs_per_expiry > 0 and not out.empty:
        counts = _count_pairs_per_group(out)
        if not counts.empty:
            group_cols = ["date", "expiry"]
            keep_groups = counts[counts >= min_pairs_per_expiry].reset_index()[group_cols]
            out = out.merge(keep_groups, on=group_cols, how="inner")
    add_step("final rows", out, raw_rows)

    out = out.sort_values(["date", "expiry", "strike", "option_type"]).reset_index(drop=True)
    return out, pd.DataFrame(report)


def filter_liquid_atm_panel(quotes: pd.DataFrame, **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    return clean_option_quotes(quotes, **kwargs)


def _contract_key(frame: pd.DataFrame) -> pd.Series:
    expiry = pd.to_datetime(frame["expiry"]).dt.strftime("%Y-%m-%d")
    strike = pd.to_numeric(frame["strike"], errors="coerce").round(8).astype(str)
    return frame["option_type"].astype(str) + "_" + expiry + "_" + strike


def select_hedging_option_path(
    quotes: pd.DataFrame,
    min_path_length: int = 20,
    preferred_option_type: str = "call",
    dte_range: tuple[int, int] = (21, 60),
    moneyness_range: tuple[float, float] = (0.95, 1.05),
) -> pd.DataFrame:
    """Select a fixed or rolling near-ATM option path for hedging diagnostics."""
    if quotes.empty:
        raise ValueError("No usable option quotes are available for hedging path selection.")

    out = quotes.copy()
    if "source_index" not in out.columns:
        out["source_index"] = out.index
    if "dte" not in out.columns or "tau" not in out.columns:
        out = add_time_to_expiry(out)
    if "moneyness" not in out.columns:
        out = add_moneyness(out)
    out["option_type"] = out["option_type"].map(parse_option_type)
    out["mid"] = pd.to_numeric(out["mid"], errors="coerce")

    dte_lo, dte_hi = dte_range
    mon_lo, mon_hi = moneyness_range
    base = out[
        (out["option_type"] == parse_option_type(preferred_option_type))
        & (out["mid"] > 0)
        & (out["dte"] >= dte_lo)
        & (out["dte"] <= dte_hi)
        & (out["moneyness"] >= mon_lo)
        & (out["moneyness"] <= mon_hi)
    ].copy()

    if base.empty:
        base = out[(out["option_type"] == parse_option_type(preferred_option_type)) & (out["mid"] > 0)].copy()
    if base.empty:
        raise ValueError("No usable option quotes are available for hedging path selection.")

    base["contract_key"] = _contract_key(base)
    base["atm_score"] = np.abs(np.log(base["moneyness"].clip(lower=1e-12)))
    if "rel_spread" not in base.columns:
        base["rel_spread"] = np.nan

    scored = (
        base.groupby("contract_key")
        .agg(
            n_dates=("date", "nunique"),
            median_atm_score=("atm_score", "median"),
            median_rel_spread=("rel_spread", "median"),
        )
        .reset_index()
    )
    fixed = scored[scored["n_dates"] >= min_path_length]
    if not fixed.empty:
        chosen = fixed.sort_values(["median_atm_score", "median_rel_spread", "n_dates"], ascending=[True, True, False]).iloc[0]
        path = base[base["contract_key"] == chosen["contract_key"]].sort_values("date").copy()
        path["path_mode"] = "fixed_contract"
        path["rolling_path"] = False
        return path.reset_index(drop=True)

    rolling = (
        base.sort_values(["date", "atm_score", "rel_spread", "dte"], ascending=[True, True, True, False])
        .groupby("date", as_index=False)
        .head(1)
        .sort_values("date")
        .copy()
    )
    if rolling.empty:
        raise ValueError("No usable rolling option path could be selected.")
    rolling["path_mode"] = "rolling_near_atm"
    rolling["rolling_path"] = True
    rolling["path_length_warning"] = len(rolling) < min_path_length
    return rolling.reset_index(drop=True)


__all__ = [
    "add_moneyness",
    "add_time_to_expiry",
    "attach_spot_from_series",
    "clean_option_quotes",
    "closest_atm_pairs",
    "convert_quotes_to_usd_equivalent",
    "detect_option_price_unit",
    "ensure_option_mid_quotes",
    "extract_spot_series",
    "filter_liquid_atm_panel",
    "normalize_btc_deribit_option_schema",
    "normalize_option_quote_schema",
    "normalize_spx_option_schema",
    "pair_put_call_quotes",
    "parse_option_type",
    "prepare_underlying_series",
    "select_hedging_option_path",
    "split_calls_puts",
    "wide_option_chain_to_long",
]
