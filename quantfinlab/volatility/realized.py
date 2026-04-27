from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return prices.pct_change()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return np.log(prices / prices.shift(1))


def realized_volatility(
    returns: pd.Series | pd.DataFrame,
    annualization: int = 252,
) -> float | pd.Series:
    vol = returns.std(skipna=True) * np.sqrt(float(annualization))
    return vol


def rolling_realized_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 21,
    annualization: int = 252,
) -> pd.Series | pd.DataFrame:
    return returns.rolling(int(window)).std() * np.sqrt(float(annualization))


def realized_volatility_table(
    returns: pd.Series,
    windows: list[int] | tuple[int, ...] = (21, 63),
    annualization: int = 252,
) -> pd.DataFrame:
    out = pd.DataFrame(index=pd.to_datetime(returns.index))
    for window in windows:
        out[f"rv_{int(window)}"] = rolling_realized_volatility(returns, int(window), annualization)
    out["rv_full"] = realized_volatility(returns, annualization)
    return out


def align_realized_to_option_expiries(
    realized_vol: pd.DataFrame | pd.Series,
    option_table: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    rv = realized_vol.copy()
    if isinstance(rv, pd.Series):
        rv = rv.to_frame("realized_vol")
    rv.index = pd.DatetimeIndex(pd.to_datetime(rv.index, errors="coerce")).astype("datetime64[ns]")
    rv = rv.sort_index()
    left = option_table[[date_col]].copy()
    left[date_col] = pd.to_datetime(left[date_col], errors="coerce").astype("datetime64[ns]")
    left = left.reset_index().sort_values(date_col).rename(columns={"index": "_row"})
    right = rv.reset_index().rename(columns={"index": date_col}).sort_values(date_col)
    matched = pd.merge_asof(left, right, on=date_col, direction="backward").set_index("_row")
    out = option_table.copy()
    for col in rv.columns:
        out[col] = matched[col].reindex(out.index)
    return out


def compare_realized_implied_vol(
    realized_vol: pd.DataFrame | pd.Series,
    iv_table: pd.DataFrame,
    date_col: str = "date",
    iv_col: str = "iv_mid",
) -> pd.DataFrame:
    out = align_realized_to_option_expiries(realized_vol, iv_table, date_col=date_col)
    rv_cols = [c for c in out.columns if str(c).startswith("rv_") or c == "realized_vol"]
    if rv_cols:
        out["realized_vol"] = out[rv_cols[0]]
    out["implied_vol"] = pd.to_numeric(out[iv_col], errors="coerce") if iv_col in out.columns else np.nan
    out["iv_minus_rv"] = out["implied_vol"] - out["realized_vol"]
    return out


def compare_realized_implied_vol_summary(rv_iv: pd.DataFrame) -> pd.DataFrame:
    diff = pd.to_numeric(rv_iv.get("iv_minus_rv"), errors="coerce")
    return pd.DataFrame(
        [
            {
                "n": int(diff.notna().sum()),
                "median_iv_minus_rv": float(np.nanmedian(diff)),
                "mean_iv_minus_rv": float(np.nanmean(diff)),
                "p10_iv_minus_rv": float(np.nanquantile(diff.dropna(), 0.10)) if diff.notna().any() else np.nan,
                "p90_iv_minus_rv": float(np.nanquantile(diff.dropna(), 0.90)) if diff.notna().any() else np.nan,
            },
        ],
    )


__all__ = [
    "align_realized_to_option_expiries",
    "compare_realized_implied_vol",
    "compare_realized_implied_vol_summary",
    "log_returns",
    "realized_volatility",
    "realized_volatility_table",
    "rolling_realized_volatility",
    "simple_returns",
]
