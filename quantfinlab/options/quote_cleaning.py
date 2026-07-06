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
    """Normalize common option-type labels to ``'call'`` or ``'put'``.

    Parameters
    ----------
    x : Any
        Raw option type value. Supported labels include ``'C'``, ``'CE'``, ``'CALL'``,
        ``'P'``, ``'PE'``, and ``'PUT'`` in any capitalization.

    Returns
    -------
    str or float
        ``'call'`` for call labels, ``'put'`` for put labels, and ``nan`` for missing
        or unrecognized inputs.
    """

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
    """Normalize an already-loaded option quote table to a standard long-form schema.

    The function resolves common date, expiry, strike, option-type, underlying, spot,
    quote, open-interest, implied-volatility, and Greek column aliases. It also parses
    instrument names when available, normalizes option types, coerces numeric fields,
    standardizes dates to pandas datetime values, and records a quote-profile label.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw option quote table.
    profile : str, default='spx_optiondx'
        Source/profile hint used for schema-specific parsing and metadata.
    underlying_default : str, optional
        Underlying symbol used when the raw table does not provide one.

    Returns
    -------
    pandas.DataFrame
        Normalized quote table with preferred columns first and remaining original
        columns preserved afterward.

    Notes
    -----
    The function does not apply liquidity or moneyness filters. Use the dedicated
    cleaning pipeline for quote validation and near-ATM selection.
    """

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
    """Normalize an SPX-style option quote table to the standard long-form schema.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw option quote table.
    underlying_default : str, optional, default='SPX'
        Underlying label used when missing from the raw data.

    Returns
    -------
    pandas.DataFrame
        Normalized quote table.
    """

    return normalize_option_quote_schema(raw, profile="spx_optiondx", underlying_default=underlying_default)


def normalize_btc_deribit_option_schema(raw: pd.DataFrame, underlying_default: str | None = "BTC") -> pd.DataFrame:
    """Normalize a BTC Deribit-style option quote table to the standard long-form schema.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw option quote table.
    underlying_default : str, optional, default='BTC'
        Underlying label used when missing from the raw data.

    Returns
    -------
    pandas.DataFrame
        Normalized quote table with parsed instrument information where possible.
    """

    return normalize_option_quote_schema(raw, profile="btc_deribit", underlying_default=underlying_default)


def wide_option_chain_to_long(
    raw: pd.DataFrame,
    *,
    underlying_default: str | None = "SPX",
    include_greeks: bool = True,
) -> pd.DataFrame:
    """Convert a wide call/put option chain into one row per option contract.

    Wide vendor chains often store call and put quotes for the same strike on a
    single row using prefixes such as ``c_`` and ``p_``. This function splits those
    legs into separate call and put rows, preserves common quote metadata, optionally
    keeps Greeks, and returns the standard long-form quote schema.

    Parameters
    ----------
    raw : pandas.DataFrame
        Raw option chain. If it already contains long-form ``option_type``, ``bid``,
        and ``ask`` columns, it is passed through schema normalization.
    underlying_default : str, optional, default='SPX'
        Underlying symbol used when missing.
    include_greeks : bool, default=True
        If False, skip call/put Greek columns during the wide-to-long conversion.

    Returns
    -------
    pandas.DataFrame
        Long-form normalized quote table.
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
    """Create or repair usable option mid quotes and spread diagnostics.

    The function coerces quote columns to numeric form, fills missing positive mid
    prices from valid bid/ask pairs, then from mark, close, or last prices when
    available. If bid and ask are available, spread, half-spread, and relative-spread
    columns are attached.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with a usable ``mid`` column and spread diagnostics when
        possible.
    """

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
    """Attach underlying spot prices to option quotes using previous-available observations.

    The function performs an as-of merge from a spot time series onto quote dates,
    never using future spot observations. Existing positive spot values are preserved
    unless ``overwrite=True``.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    spot_series : pandas.Series
        Spot price series indexed by date or timestamp.
    date_col : str, default='date'
        Quote date column.
    spot_col : str, default='spot'
        Output spot column.
    method : {'previous'}, default='previous'
        As-of matching method. Only previous-available matching is supported.
    overwrite : bool, default=False
        If True, replace existing spot values.

    Returns
    -------
    pandas.DataFrame
        Quote table with attached spot values.

    Raises
    ------
    ValueError
        If the method is unsupported, the spot series is empty, or no valid spot value
        can be attached.
    """

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


def extract_spot_series(quotes: pd.DataFrame, date_col: str = "date", spot_col: str = "spot") -> pd.Series:
    """Extract one representative spot observation per quote date.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Normalized quote table.
    date_col : str, default='date'
        Quote date column.
    spot_col : str, default='spot'
        Spot price column.

    Returns
    -------
    pandas.Series
        Date-indexed median spot series named ``'spot'``.

    Raises
    ------
    ValueError
        If required columns are missing or no positive spot observations are found.
    """

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
    """Infer whether option premiums are quoted in USD-like currency or base-asset units.

    The function first checks explicit unit or currency columns. If they are absent,
    it uses price magnitudes and the ratio of option premium to spot to distinguish
    USD-like prices from base-denominated premiums.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table containing price columns and, ideally, spot or currency
        metadata.

    Returns
    -------
    {'usd', 'base', 'unknown'}
        Detected premium unit.

    Notes
    -----
    The heuristic is designed for practical data cleaning. For high-stakes valuation,
    explicit source metadata should be preferred over automatic detection.
    """

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
    """Convert base-denominated option premiums to USD-equivalent prices.

    If premiums are detected or declared as base-asset units, selected price columns
    are multiplied by spot and contract size. Original values are preserved in
    ``<column>_raw`` columns before conversion.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    spot_col : str, default='spot'
        Spot price column used for base-to-USD conversion.
    price_cols : tuple[str, ...], default=('bid', 'ask', 'mid', 'last', 'mark')
        Price columns to convert.
    unit : {'auto', 'usd', 'base', 'btc', 'unknown'}, default='auto'
        Premium-unit assumption. ``'auto'`` uses automatic detection.
    contract_size : float, default=1.0
        Contract multiplier applied during conversion.

    Returns
    -------
    pandas.DataFrame
        Quote table with USD-equivalent price columns, raw-price backups,
        ``valuation_currency``, ``price_unit_detected``, and ``contract_size``.

    Raises
    ------
    ValueError
        If base-unit conversion is requested without a spot column or if ``unit`` is
        invalid.
    """

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
    """Add calendar days to expiry and year-fraction maturity columns to option quotes.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    date_col : str, default='date'
        Valuation date column.
    expiry_col : str, default='expiry'
        Expiry date column.
    annualization_days : float, default=365.0
        Number of days used to convert calendar DTE to year fraction.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with ``dte``, ``tau``, and ``annualization_days`` columns.
    """

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
    """Add strike-over-spot moneyness and log-moneyness columns.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    spot_col : str, default='spot'
        Spot price column.
    strike_col : str, default='strike'
        Strike price column.

    Returns
    -------
    pandas.DataFrame
        Copy of ``quotes`` with ``moneyness`` and ``log_moneyness`` columns.
    """

    out = quotes.copy()
    spot = pd.to_numeric(out[spot_col], errors="coerce")
    strike = pd.to_numeric(out[strike_col], errors="coerce")
    out["moneyness"] = strike / spot.replace(0, np.nan)
    out["log_moneyness"] = np.log(out["moneyness"].where(out["moneyness"] > 0))
    return out


def split_calls_puts(quotes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a quote table into call and put subsets.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table containing an ``option_type`` column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(calls, puts)`` after option-type normalization.
    """

    out = quotes.copy()
    if "option_type" in out.columns:
        out["option_type"] = out["option_type"].map(parse_option_type)
    return out[out["option_type"] == "call"].copy(), out[out["option_type"] == "put"].copy()


def pair_put_call_quotes(
    quotes: pd.DataFrame,
    on: tuple[str, ...] = ("date", "expiry", "strike"),
    price_col: str = "mid",
) -> pd.DataFrame:
    """Pair call and put quotes that share the same date, expiry, and strike keys.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Long-form option quote table.
    on : tuple[str, ...], default=('date', 'expiry', 'strike')
        Columns used as pair keys.
    price_col : str, default='mid'
        Price column included in the paired output.

    Returns
    -------
    pandas.DataFrame
        Wide paired table containing call-prefixed and put-prefixed columns merged on
        the common keys.
    """

    calls, puts = split_calls_puts(quotes)
    common = [c for c in on if c in quotes.columns]
    keep = list(dict.fromkeys([*common, price_col, "bid", "ask", "spot", "forward", "tau", "rate", "moneyness"]))
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
    """Select the closest near-the-money call/put pairs per date and expiry.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Long-form option quote table.
    n_pairs : int, default=25
        Number of strike pairs to retain per date-expiry group.

    Returns
    -------
    pandas.DataFrame
        Filtered quote table restricted to the closest available ATM pairs.
    """

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
    """Normalize and filter an option quote panel to a liquid near-ATM dataset.

    The cleaning pipeline standardizes schema, builds mid quotes, adds expiry and
    moneyness fields, removes invalid quotes, filters bid-ask and relative-spread
    outliers, restricts DTE and moneyness ranges, selects closest ATM pairs, and
    requires a minimum number of pairs per expiry.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Raw or normalized option quote table.
    min_dte : int, default=7
        Minimum calendar days to expiry.
    max_dte : int, default=120
        Maximum calendar days to expiry.
    moneyness_range : tuple[float, float], default=(0.85, 1.15)
        Inclusive strike-over-spot moneyness range.
    max_relative_spread : float, default=0.20
        Maximum allowed relative bid-ask spread for quotes with bid and ask.
    closest_atm_pairs : int or None, default=25
        Number of closest ATM pairs to retain per date-expiry group. If None, skip
        this selection step.
    min_pairs_per_expiry : int, default=10
        Minimum number of complete call/put strike pairs required for a date-expiry
        group to remain.
    annualization_days : float, default=365.0
        Days per year used to compute ``tau``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Cleaned quote table and a stepwise cleaning report with row counts and
        removals.
    """

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
    """Compatibility alias for the liquid near-ATM option quote cleaning pipeline.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    **kwargs
        Keyword arguments forwarded to the cleaning pipeline.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Cleaned quote table and cleaning report.
    """

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
    """Select a fixed-contract or rolling near-ATM option path for hedging diagnostics.

    The function first searches for a single contract with enough observations inside
    the requested DTE and moneyness ranges. If no fixed contract has sufficient
    history, it constructs a rolling path by selecting the best near-ATM contract on
    each date.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Option quote table.
    min_path_length : int, default=20
        Minimum number of dates required for a fixed-contract path.
    preferred_option_type : str, default='call'
        Preferred option type for the path.
    dte_range : tuple[int, int], default=(21, 60)
        Calendar-DTE range used for the initial path search.
    moneyness_range : tuple[float, float], default=(0.95, 1.05)
        Strike-over-spot moneyness range used for the initial path search.

    Returns
    -------
    pandas.DataFrame
        Selected path with ``path_mode`` and ``rolling_path`` metadata columns.

    Raises
    ------
    ValueError
        If no usable fixed or rolling path can be selected.
    """

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


def _surface_pdf(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.exp(-0.5 * arr * arr) / np.sqrt(2.0 * np.pi)


def surface_ready_quotes(
    quotes,
    date_col="date",
    expiry_col="expiry",
    option_type_col="option_type",
    strike_col="strike",
    spot_col="spot",
    forward_col="forward",
    tau_col="tau",
    rate_col="rate",
    discount_col="discount_factor",
    iv_bid_col="iv_bid",
    iv_mid_col="iv_mid",
    iv_ask_col="iv_ask",
    rel_spread_col="relative_spread",
    out_k_col="k",
    out_k_spot_col="k_spot",
    out_weight_col="surface_weight",
    min_iv=0.02,
    max_iv=2.00,
    min_tau=7 / 365.25,
    max_tau=180 / 365.25,
    min_weight=0.05,
    max_weight=20.0,
):
    """Prepare option quotes for volatility-surface fitting.

    The function validates core option inputs, filters implausible maturities and
    volatilities, computes forward and spot log-moneyness, total variance, log total
    variance, IV uncertainty, and a robust surface-fitting weight that combines
    spread quality, vega information, wing distance, and uncertainty penalties.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Quote table containing option type, strike, spot, forward, maturity, IV, and
        quote-quality columns.
    date_col, expiry_col, option_type_col, strike_col, spot_col, forward_col : str
        Column names for date, expiry, option type, strike, spot, and forward.
    tau_col : str, default='tau'
        Time-to-expiry column in years.
    rate_col : str, default='rate'
        Continuous rate column.
    discount_col : str, default='discount_factor'
        Discount-factor column.
    iv_bid_col, iv_mid_col, iv_ask_col : str
        Bid, mid, and ask implied-volatility columns.
    rel_spread_col : str, default='relative_spread'
        Relative-spread column used for weights.
    out_k_col : str, default='k'
        Output forward log-moneyness column.
    out_k_spot_col : str, default='k_spot'
        Output spot log-moneyness column.
    out_weight_col : str, default='surface_weight'
        Output fitting-weight column.
    min_iv, max_iv : float
        Allowed implied-volatility range.
    min_tau, max_tau : float
        Allowed maturity range in years.
    min_weight, max_weight : float
        Bounds applied to the final surface-fitting weights.

    Returns
    -------
    pandas.DataFrame
        Surface-ready quote table sorted by date, expiry, strike, and option type.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    out = quotes.copy()
    for col in [date_col, expiry_col]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").astype("datetime64[ns]")
    if option_type_col in out.columns:
        out[option_type_col] = out[option_type_col].map(parse_option_type)

    for col in [strike_col, spot_col, forward_col, tau_col, rate_col, discount_col, iv_bid_col, iv_mid_col, iv_ask_col]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    rel_col = rel_spread_col if rel_spread_col in out.columns else "rel_spread" if "rel_spread" in out.columns else rel_spread_col
    if rel_col not in out.columns:
        if {"bid", "ask", "mid"}.issubset(out.columns):
            out[rel_col] = (pd.to_numeric(out["ask"], errors="coerce") - pd.to_numeric(out["bid"], errors="coerce")) / pd.to_numeric(
                out["mid"],
                errors="coerce",
            ).replace(0, np.nan)
        else:
            out[rel_col] = np.nan
    out[rel_spread_col] = pd.to_numeric(out[rel_col], errors="coerce")

    required = [date_col, expiry_col, option_type_col, strike_col, spot_col, forward_col, tau_col, iv_mid_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"surface_ready_quotes is missing required columns: {missing}")

    mask = (
        out[option_type_col].isin(["call", "put"])
        & np.isfinite(out[strike_col])
        & np.isfinite(out[spot_col])
        & np.isfinite(out[forward_col])
        & np.isfinite(out[tau_col])
        & np.isfinite(out[iv_mid_col])
        & (out[strike_col] > 0)
        & (out[spot_col] > 0)
        & (out[forward_col] > 0)
        & (out[tau_col] >= float(min_tau))
        & (out[tau_col] <= float(max_tau))
        & (out[iv_mid_col] >= float(min_iv))
        & (out[iv_mid_col] <= float(max_iv))
    )
    if rate_col in out.columns:
        mask &= np.isfinite(out[rate_col])
    if discount_col in out.columns:
        mask &= np.isfinite(out[discount_col]) & (out[discount_col] > 0)
    if {"bid", "ask"}.issubset(out.columns):
        bid = pd.to_numeric(out["bid"], errors="coerce")
        ask = pd.to_numeric(out["ask"], errors="coerce")
        mask &= bid.notna() & ask.notna() & (bid >= 0) & (ask >= bid)

    out = out.loc[mask].copy()
    out[out_k_col] = np.log(out[strike_col] / out[forward_col])
    out[out_k_spot_col] = np.log(out[strike_col] / out[spot_col])
    out["total_variance"] = out[iv_mid_col] * out[iv_mid_col] * out[tau_col]
    out = out[np.isfinite(out["total_variance"]) & (out["total_variance"] > 0)].copy()
    out["log_total_variance"] = np.log(out["total_variance"].clip(lower=1e-12))

    sqrt_tau = np.sqrt(out[tau_col].clip(lower=1e-10))
    sigma = out[iv_mid_col].clip(lower=1e-8)
    d1 = (-out[out_k_col] + 0.5 * sigma * sigma * out[tau_col]) / (sigma * sqrt_tau)
    df = out[discount_col] if discount_col in out.columns else 1.0
    dollar_vega = pd.Series(np.asarray(df, dtype=float) * out[forward_col].to_numpy(dtype=float) * _surface_pdf(d1) * sqrt_tau, index=out.index)

    if iv_bid_col in out.columns and iv_ask_col in out.columns:
        iv_width = (out[iv_ask_col] - out[iv_bid_col]).where(lambda x: np.isfinite(x) & (x > 0))
    else:
        iv_width = pd.Series(np.nan, index=out.index, dtype=float)
    if {"bid", "ask"}.issubset(out.columns):
        price_spread = (pd.to_numeric(out["ask"], errors="coerce") - pd.to_numeric(out["bid"], errors="coerce")).clip(lower=0.0)
        iv_from_price = price_spread / dollar_vega.replace(0, np.nan)
    else:
        iv_from_price = pd.Series(np.nan, index=out.index, dtype=float)
    out["iv_uncertainty"] = iv_width.combine_first(iv_from_price)
    unc_med = float(np.nanmedian(out["iv_uncertainty"])) if out["iv_uncertainty"].notna().any() else 0.025
    out["iv_uncertainty"] = out["iv_uncertainty"].fillna(unc_med).clip(lower=0.003, upper=0.50)

    spread = out[rel_spread_col].copy()
    spread_med = float(np.nanmedian(spread[(spread > 0) & np.isfinite(spread)])) if spread.notna().any() else 0.08
    spread = spread.fillna(spread_med).clip(lower=0.003, upper=1.0)
    spread_weight = 1.0 / (spread.to_numpy(dtype=float) ** 2 + 0.02**2)
    spread_weight /= max(float(np.nanmedian(spread_weight[np.isfinite(spread_weight)])), 1e-12)

    info = dollar_vega.to_numpy(dtype=float)
    info_med = float(np.nanmedian(info[np.isfinite(info) & (info > 0)])) if np.isfinite(info).any() else 1.0
    info_weight = np.clip((info / max(info_med, 1e-12)) ** 0.45, 0.20, 3.00)
    wing_weight = np.exp(-np.maximum(np.abs(out[out_k_col].to_numpy(dtype=float)) - 0.10, 0.0) / 0.45)
    unc = out["iv_uncertainty"].to_numpy(dtype=float)
    unc_med = float(np.nanmedian(unc[np.isfinite(unc) & (unc > 0)])) if np.isfinite(unc).any() else 0.025
    uncertainty_penalty = np.clip(unc / max(unc_med, 1e-12), 0.50, 3.50) ** -0.35
    weight = spread_weight * info_weight * wing_weight * uncertainty_penalty
    weight /= max(float(np.nanmedian(weight[np.isfinite(weight)])), 1e-12)
    out[out_weight_col] = np.clip(weight, float(min_weight), float(max_weight))

    if "dte_days" not in out.columns:
        if "dte_calendar" in out.columns:
            out["dte_days"] = out["dte_calendar"]
        elif "dte" in out.columns:
            out["dte_days"] = out["dte"]
        else:
            out["dte_days"] = out[tau_col] * 365.25
    return out.sort_values([date_col, expiry_col, strike_col, option_type_col]).reset_index(drop=True)


def surface_support_by_date(
    quotes: pd.DataFrame,
    *,
    date_col: str = "date",
    k_col: str = "k",
    tau_col: str = "tau",
    k_quantiles: tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    """Summarize observed moneyness and maturity support by quote date.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Surface-ready quote table.
    date_col : str, default='date'
        Date column.
    k_col : str, default='k'
        Log-moneyness column.
    tau_col : str, default='tau'
        Time-to-expiry column in years.
    k_quantiles : tuple[float, float], default=(0.01, 0.99)
        Quantiles used to define robust lower and upper moneyness support.

    Returns
    -------
    pandas.DataFrame
        Per-date support table with quote count, min/max and quantile moneyness, and
        min/max maturity.
    """

    if quotes.empty:
        return pd.DataFrame(columns=[date_col, "quotes", "k_min", "k_max", "k_lo", "k_hi", "tau_min", "tau_max"])
    data = quotes[[date_col, k_col, tau_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce").dt.normalize()
    data[k_col] = pd.to_numeric(data[k_col], errors="coerce")
    data[tau_col] = pd.to_numeric(data[tau_col], errors="coerce")
    data = data[np.isfinite(data[k_col]) & np.isfinite(data[tau_col])].copy()
    grouped = data.groupby(date_col)
    out = grouped.agg(
        quotes=(k_col, "size"),
        k_min=(k_col, "min"),
        k_max=(k_col, "max"),
        k_lo=(k_col, lambda x: float(np.nanquantile(x, k_quantiles[0]))),
        k_hi=(k_col, lambda x: float(np.nanquantile(x, k_quantiles[1]))),
        tau_min=(tau_col, "min"),
        tau_max=(tau_col, "max"),
    ).reset_index()
    return out.sort_values(date_col).reset_index(drop=True)


def surface_common_support(
    quotes: pd.DataFrame,
    *,
    date_col: str = "date",
    k_col: str = "k",
    tau_col: str = "tau",
    k_grid=None,
    tau_grid=None,
    min_support_share: float = 0.85,
    k_quantiles: tuple[float, float] = (0.01, 0.99),
) -> dict:
    """Build a common historical volatility-surface support mask.

    The function evaluates how often each node of a candidate ``k``/``tau`` grid lies
    inside the robust quote support of each date and marks nodes that meet a minimum
    support-share threshold.

    Parameters
    ----------
    quotes : pandas.DataFrame
        Surface-ready quote table.
    date_col : str, default='date'
        Date column.
    k_col : str, default='k'
        Log-moneyness column.
    tau_col : str, default='tau'
        Maturity column in years.
    k_grid : array-like, optional
        Candidate log-moneyness grid. If omitted, a grid is derived from median
        historical support quantiles.
    tau_grid : array-like, optional
        Candidate maturity grid. If omitted, a grid is derived from median historical
        maturity support.
    min_support_share : float, default=0.85
        Minimum fraction of dates that must support a node.
    k_quantiles : tuple[float, float], default=(0.01, 0.99)
        Robust support quantiles used by date.

    Returns
    -------
    dict
        Dictionary containing ``k``, ``tau``, ``support_share``, ``support_mask``, and
        the per-date ``support_by_date`` table.
    """

    support = surface_support_by_date(quotes, date_col=date_col, k_col=k_col, tau_col=tau_col, k_quantiles=k_quantiles)
    if k_grid is None:
        k_grid = np.linspace(float(support["k_lo"].median()), float(support["k_hi"].median()), 41)
    if tau_grid is None:
        tau_grid = np.linspace(float(support["tau_min"].median()), float(support["tau_max"].median()), 21)
    k_grid = np.asarray(k_grid, dtype=float)
    tau_grid = np.asarray(tau_grid, dtype=float)
    kk, tt = np.meshgrid(k_grid, tau_grid)
    masks = []
    for _, row in support.iterrows():
        masks.append((kk >= row["k_lo"]) & (kk <= row["k_hi"]) & (tt >= row["tau_min"]) & (tt <= row["tau_max"]))
    share = np.mean(np.asarray(masks, dtype=bool), axis=0) if masks else np.zeros_like(kk, dtype=float)
    mask = share >= float(min_support_share)
    return {"k": k_grid, "tau": tau_grid, "support_share": share, "support_mask": mask, "support_by_date": support}


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
    "select_hedging_option_path",
    "surface_common_support",
    "surface_ready_quotes",
    "surface_support_by_date",
    "split_calls_puts",
    "wide_option_chain_to_long",
]
