from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.fixed_income import discounting

from . import rates_dividends
from .quote_cleaning import pair_put_call_quotes


def _datetime_keys_ns(frame: pd.DataFrame, keys: tuple[str, ...] = ("date", "expiry")) -> pd.DataFrame:
    out = frame.copy()
    for key in keys:
        if key in out.columns:
            out[key] = pd.to_datetime(out[key], errors="coerce").astype("datetime64[ns]")
    return out


def infer_forward_from_put_call_pair(
    call_mid,
    put_mid,
    strike,
    rate,
    tau,
):
    """Infer F from C - P = DF * (F - K)."""
    df = discounting.discount_factor_from_rate(rate, tau)
    return strike + (np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)) / np.asarray(
        df,
        dtype=float,
    )


def put_call_parity_residual(
    call_price,
    put_price,
    strike,
    forward,
    rate,
    tau,
) -> np.ndarray:
    df = discounting.discount_factor_from_rate(rate, tau)
    return np.asarray(call_price, dtype=float) - np.asarray(put_price, dtype=float) - np.asarray(
        df,
        dtype=float,
    ) * (np.asarray(forward, dtype=float) - np.asarray(strike, dtype=float))


def robust_forward_by_group(pairs: pd.DataFrame) -> pd.Series:
    """Robust weighted median forward estimate for one date/expiry group."""
    if pairs.empty:
        return pd.Series(dtype=float)
    fwd = pd.to_numeric(pairs["pair_forward"], errors="coerce")
    if "pair_weight" in pairs.columns:
        weights = pd.to_numeric(pairs["pair_weight"], errors="coerce")
    else:
        weights = pd.Series(1.0, index=pairs.index)
    mask = fwd.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        forward = float(fwd.median())
    else:
        x = fwd[mask].to_numpy(dtype=float)
        w = weights[mask].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        w = w[order]
        forward = float(x[np.searchsorted(np.cumsum(w), 0.5 * w.sum(), side="left")])
    residual = pairs["pair_forward"] - forward
    return pd.Series(
        {
            "forward": forward,
            "parity_error_median": float(np.nanmedian(residual)),
            "parity_error_mad": float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))),
            "parity_error_iqr": float(np.nanquantile(residual, 0.75) - np.nanquantile(residual, 0.25))
            if residual.notna().sum() >= 2
            else np.nan,
            "n_pairs": int(fwd.notna().sum()),
        },
    )


def infer_forwards_from_put_call_parity(
    quotes: pd.DataFrame,
    rates: pd.Series | pd.DataFrame | None = None,
    price_col: str = "mid",
) -> pd.DataFrame:
    """Pair calls and puts, infer parity forwards, and aggregate by date/expiry."""
    data = _datetime_keys_ns(quotes)
    if rates is not None and "rate" not in data.columns:
        if isinstance(rates, pd.DataFrame):
            cols = [c for c in ["date", "expiry", "rate"] if c in rates.columns]
            rate_frame = _datetime_keys_ns(rates[cols].drop_duplicates())
            data = data.merge(rate_frame, on=[c for c in ["date", "expiry"] if c in cols], how="left")
        else:
            rate_series = rates.copy()
            rate_series.index = pd.to_datetime(rate_series.index)
            data = data.sort_values("date")
            data["rate"] = pd.merge_asof(
                data[["date"]].reset_index().sort_values("date"),
                rate_series.rename("rate").reset_index().rename(columns={"index": "date"}).sort_values("date"),
                on="date",
                direction="backward",
            ).set_index("index")["rate"]

    pairs = pair_put_call_quotes(data, price_col=price_col)
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "expiry",
                "tau",
                "spot",
                "rate",
                "forward",
                "implied_carry",
                "n_pairs",
                "parity_error_median",
                "parity_error_mad",
                "parity_error_iqr",
            ],
        )

    if "call_rate" in pairs.columns:
        pairs["rate"] = pairs["call_rate"].combine_first(pairs.get("put_rate"))
    elif "rate" not in pairs.columns:
        pairs["rate"] = np.nan
    if "call_tau" in pairs.columns:
        pairs["tau"] = pairs["call_tau"].combine_first(pairs.get("put_tau"))
    if "call_spot" in pairs.columns:
        pairs["spot"] = pairs["call_spot"].combine_first(pairs.get("put_spot"))

    df = discounting.discount_factor_from_rate(pairs["rate"], pairs["tau"])
    pairs["pair_forward"] = pairs["strike"] + (pairs[f"call_{price_col}"] - pairs[f"put_{price_col}"]) / df

    if {"call_bid", "call_ask", "put_bid", "put_ask"}.issubset(pairs.columns):
        spread = (pairs["call_ask"] - pairs["call_bid"]) + (pairs["put_ask"] - pairs["put_bid"])
        pairs["pair_weight"] = 1.0 / spread.clip(lower=1e-8)
    else:
        atm = np.abs(np.log(pd.to_numeric(pairs["strike"], errors="coerce") / pd.to_numeric(pairs["spot"], errors="coerce")))
        pairs["pair_weight"] = 1.0 / atm.clip(lower=0.01)

    group_cols = ["date", "expiry"]
    grouped = pairs.groupby(group_cols, dropna=False).apply(robust_forward_by_group, include_groups=False)
    grouped = grouped.reset_index()
    meta = (
        pairs.groupby(group_cols)
        .agg(tau=("tau", "median"), spot=("spot", "median"), rate=("rate", "median"))
        .reset_index()
    )
    out = grouped.merge(meta, on=group_cols, how="left")
    out["implied_carry"] = rates_dividends.infer_carry_from_forward(out["spot"], out["forward"], out["tau"])
    return out[
        [
            "date",
            "expiry",
            "tau",
            "spot",
            "rate",
            "forward",
            "implied_carry",
            "n_pairs",
            "parity_error_median",
            "parity_error_mad",
            "parity_error_iqr",
        ]
    ].sort_values(["date", "expiry"]).reset_index(drop=True)


def choose_liquid_single_day(
    quotes: pd.DataFrame,
    min_pairs: int = 20,
    prefer_dte_range: tuple[int, int] = (21, 60),
) -> pd.Timestamp:
    """Choose one quote date with enough near-dated put-call pairs."""
    if quotes.empty or "date" not in quotes.columns:
        raise ValueError("quotes must contain date rows.")
    data = quotes.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if "dte" not in data.columns and {"date", "expiry"}.issubset(data.columns):
        data["dte"] = (pd.to_datetime(data["expiry"], errors="coerce") - data["date"]).dt.days
    lo, hi = prefer_dte_range
    preferred = data[(pd.to_numeric(data.get("dte"), errors="coerce") >= lo) & (pd.to_numeric(data.get("dte"), errors="coerce") <= hi)]
    candidates = preferred if not preferred.empty else data
    pairs = pair_put_call_quotes(candidates)
    counts = candidates.groupby("date").size() if pairs.empty else pairs.groupby("date").size()
    counts = counts[counts >= int(min_pairs)]
    if counts.empty:
        counts = (pairs.groupby("date").size() if not pairs.empty else candidates.groupby("date").size())
    if counts.empty:
        raise ValueError("No liquid quote date could be selected.")
    return pd.Timestamp(counts.sort_values(ascending=False).index[0]).normalize()


def infer_single_day_forward_curve(
    quotes: pd.DataFrame,
    price_col: str = "mid",
    rate_col: str = "rate",
) -> pd.DataFrame:
    """Infer the parity forward curve for one quote date."""
    data = quotes.copy()
    if rate_col != "rate" and rate_col in data.columns:
        data["rate"] = data[rate_col]
    return infer_forwards_from_put_call_parity(data, price_col=price_col)


def parity_error_table(
    quotes: pd.DataFrame,
    forward_table: pd.DataFrame | None = None,
    price_col: str = "mid",
) -> pd.DataFrame:
    data = quotes.copy()
    data = _datetime_keys_ns(data)
    if forward_table is not None and "forward" not in data.columns:
        fwd = _datetime_keys_ns(forward_table[["date", "expiry", "forward"]])
        data = data.merge(fwd, on=["date", "expiry"], how="left")
    pairs = pair_put_call_quotes(data, price_col=price_col)
    if pairs.empty:
        return pairs
    for col in ["rate", "tau", "forward"]:
        call_col = f"call_{col}"
        put_col = f"put_{col}"
        if call_col in pairs.columns:
            pairs[col] = pairs[call_col].combine_first(pairs.get(put_col))
    pairs["parity_residual"] = put_call_parity_residual(
        pairs[f"call_{price_col}"],
        pairs[f"put_{price_col}"],
        pairs["strike"],
        pairs["forward"],
        pairs["rate"],
        pairs["tau"],
    )
    if "call_moneyness" in pairs.columns:
        pairs["moneyness"] = pairs["call_moneyness"]
    return pairs


__all__ = [
    "choose_liquid_single_day",
    "infer_forward_from_put_call_pair",
    "infer_forwards_from_put_call_parity",
    "infer_single_day_forward_curve",
    "parity_error_table",
    "put_call_parity_residual",
    "robust_forward_by_group",
]
