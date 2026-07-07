from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


def _to_datetime_index(index_like: pd.Index | Sequence[pd.Timestamp | str]) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index_like))
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        raise InputError("Date index is empty.")
    return idx.sort_values().unique()


def _sanitize_frame(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise InputError(f"{name} must be a pandas DataFrame.")
    if df.empty:
        raise InputError(f"{name} is empty.")
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.isna()].sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out.columns = [str(c).strip() for c in out.columns]
    if out.columns.duplicated().any():
        out = out.T.groupby(level=0).last().T
    out = out.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _resolve_date_on_or_before(index: pd.DatetimeIndex, dt: pd.Timestamp | str) -> pd.Timestamp | None:
    d = pd.Timestamp(dt)
    if d in index:
        return d
    pos = int(index.searchsorted(d, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(index[pos])


def prices_to_returns(
    prices: pd.DataFrame,
    *,
    kind: Literal["simple", "log"] = "simple",
    drop_all_nan: bool = True,
    dtype: str | np.dtype = np.float64,
) -> pd.DataFrame:
    """Convert a price panel into simple or log returns.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price panel with dates in rows and assets in columns.
    kind : {"simple", "log"}, default="simple"
        Return calculation. ``"simple"`` uses percentage change; ``"log"`` uses
        log price differences.
    drop_all_nan : bool, default=True
        Whether to drop rows where all returns are missing.
    dtype : str or numpy.dtype, default=numpy.float64
        Output dtype.

    Returns
    -------
    pandas.DataFrame
        Return panel with the same columns as ``prices``.

    Raises
    ------
    InputError
        If an unsupported return kind is supplied.

    Notes
    -----
    Infinite values are replaced with ``NaN`` before optional row dropping.
    """

    px = _sanitize_frame(prices, name="prices")
    if kind == "simple":
        returns = px.pct_change(fill_method=None)
    elif kind == "log":
        returns = np.log(px / px.shift(1))
    else:
        raise InputError(f"Unsupported return kind: {kind!r}.")
    returns = returns.replace([np.inf, -np.inf], np.nan)
    if drop_all_nan:
        returns = returns.dropna(how="all")
    return returns.astype(dtype)


def make_rebalance_dates(
    index: pd.Index | Sequence[pd.Timestamp | str],
    *,
    freq: str = "M",
    min_history_days: int = 0,
) -> pd.DatetimeIndex:
    """Build rebalance dates from a trading-day index.

    The function groups dates by a pandas frequency and selects the last
    available trading date in each period.

    Parameters
    ----------
    index : pandas.Index or sequence of pandas.Timestamp or str
        Trading-day index.
    freq : str, default="M"
        Rebalance frequency. Common aliases such as ``"M"``, ``"Q"``, and
        ``"Y"`` are mapped to pandas month/quarter/year-end aliases.
    min_history_days : int, default=0
        Minimum number of historical observations required before a rebalance
        date is allowed.

    Returns
    -------
    pandas.DatetimeIndex
        Sorted unique rebalance dates.

    Raises
    ------
    InputError
        If ``min_history_days`` exceeds the available index length.

    Notes
    -----
    The returned dates are actual dates from the input index, not calendar period
    ends that may fall on non-trading days.
    """

    idx = _to_datetime_index(index)
    freq_norm = str(freq).upper().strip()
    freq_alias = {"M": "ME", "Q": "QE", "Y": "YE", "A": "YE"}
    grouped = (
        pd.Series(idx, index=idx)
        .groupby(pd.Grouper(freq=freq_alias.get(freq_norm, freq)))
        .last()
        .dropna()
    )
    rebalance_dates = pd.DatetimeIndex(grouped.values).sort_values().unique()
    if min_history_days > 0:
        if min_history_days >= len(idx):
            raise InputError("min_history_days is larger than the index length.")
        rebalance_dates = rebalance_dates[rebalance_dates >= idx[int(min_history_days)]]
    return rebalance_dates


def first_valid_close_volume_date(close: pd.DataFrame, volume: pd.DataFrame) -> pd.Series:
    first_close = close.apply(pd.Series.first_valid_index)
    first_volume = volume.apply(pd.Series.first_valid_index)
    return pd.concat([first_close, first_volume], axis=1).max(axis=1)


def clean_close_volume_panels(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    start: str | pd.Timestamp | None = "2016-01-01",
    min_price: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align and lightly clean close/volume panels after notebook data loading.

    Data loading stays visible in the notebook; this helper starts from already-built
    close and volume DataFrames.
    """
    close_clean = _sanitize_frame(close, name="close")
    volume_clean = _sanitize_frame(volume, name="volume")

    close_clean = close_clean.where(close_clean > 0)
    volume_clean = volume_clean.where(volume_clean >= 0)
    if min_price is not None:
        close_clean = close_clean.where(close_clean >= float(min_price))

    if start is not None:
        start_ts = pd.Timestamp(start)
        close_clean = close_clean.loc[close_clean.index >= start_ts]
        volume_clean = volume_clean.loc[volume_clean.index >= start_ts]

    idx = close_clean.index.intersection(volume_clean.index)
    cols = close_clean.columns.intersection(volume_clean.columns)
    close_clean = close_clean.loc[idx, cols]
    volume_clean = volume_clean.loc[idx, cols]

    valid_cols = close_clean.notna().any(axis=0) & volume_clean.notna().any(axis=0)
    close_clean = close_clean.loc[:, valid_cols]
    volume_clean = volume_clean.loc[:, valid_cols]

    sorted_cols = sorted([str(c) for c in close_clean.columns])
    close_clean = close_clean.reindex(columns=sorted_cols)
    volume_clean = volume_clean.reindex(columns=sorted_cols)
    if close_clean.empty or volume_clean.empty or close_clean.shape[1] < 2:
        raise InputError("Not enough valid assets after close/volume cleaning.")
    return close_clean.astype(float), volume_clean.astype(float)


def select_liquid_universe(
    dt: pd.Timestamp | str,
    *,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    close_prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    top_n: int = 100,
    liquidity_lookback: int | None = None,
    liq_lookback: int | None = None,
    min_listing_days: int = 252,
    min_obs: int = 200,
    min_price: float | None = None,
    first_date: pd.Series | None = None,
) -> tuple[list[str], pd.Series]:
    """Select the most liquid seasoned assets at a rebalance date.

    The universe is ranked by average dollar volume over a trailing lookback
    window after applying seasoning, observation-count, positive-volume, and
    optional minimum-price filters.

    Parameters
    ----------
    dt : pandas.Timestamp or str
        Rebalance date. The latest available date on or before ``dt`` is used.
    close : pandas.DataFrame, optional
        Close-price panel.
    volume : pandas.DataFrame, optional
        Volume panel.
    close_prices, volumes : pandas.DataFrame, optional
        Aliases for ``close`` and ``volume``.
    top_n : int, default=100
        Maximum number of assets selected.
    liquidity_lookback : int, optional
        Lookback window for average dollar volume.
    liq_lookback : int, optional
        Alias for ``liquidity_lookback``.
    min_listing_days : int, default=252
        Minimum days since first valid price/volume observation.
    min_obs : int, default=200
        Minimum valid dollar-volume observations in the lookback window.
    min_price : float, optional
        Minimum close price filter.
    first_date : pandas.Series, optional
        Precomputed first valid close/volume date per asset.

    Returns
    -------
    tickers : list of str
        Selected asset tickers.
    avg_dollar_volume : pandas.Series
        Average dollar volume for the selected assets.

    Raises
    ------
    InputError
        If close/volume panels are missing or configuration values are invalid.

    Notes
    -----
    If insufficient history or no eligible assets are available, the function
    returns an empty list and empty Series rather than raising.
    """

    close_panel = close if close is not None else close_prices
    volume_panel = volume if volume is not None else volumes
    if close_panel is None or volume_panel is None:
        raise InputError("close/volume panels are required.")
    if top_n <= 0:
        raise InputError("top_n must be positive.")

    lookback = int(liquidity_lookback if liquidity_lookback is not None else liq_lookback or 252)
    if lookback <= 0 or min_listing_days <= 0 or min_obs <= 0:
        raise InputError("liquidity_lookback, min_listing_days and min_obs must be positive.")

    cp = _sanitize_frame(close_panel, name="close")
    vv = _sanitize_frame(volume_panel, name="volume")

    common_idx = cp.index.intersection(vv.index)
    common_cols = cp.columns.intersection(vv.columns)
    if len(common_idx) == 0 or len(common_cols) == 0:
        raise InputError("close and volume must overlap on index and columns.")
    cp = cp.loc[common_idx, common_cols]
    vv = vv.loc[common_idx, common_cols]
    if min_price is not None:
        cp = cp.where(cp >= float(min_price))

    idx = pd.DatetimeIndex(cp.index)
    d_eff = _resolve_date_on_or_before(idx, dt)
    if d_eff is None:
        return [], pd.Series(dtype=float)

    pos = int(idx.get_loc(d_eff))
    need = max(lookback, int(min_listing_days))
    if pos < need:
        return [], pd.Series(dtype=float)

    fdate = first_date if first_date is not None else first_valid_close_volume_date(cp, vv)
    cutoff_date = idx[pos - int(min_listing_days)]
    seasoned = (fdate.notna()) & (fdate <= cutoff_date)
    seasoned = seasoned.reindex(cp.columns).fillna(False)
    cols = cp.columns[seasoned.values]
    if len(cols) == 0:
        return [], pd.Series(dtype=float)

    start = pos - lookback
    dollar_volume = cp.iloc[start:pos][cols] * vv.iloc[start:pos][cols]
    obs_ok = dollar_volume.notna().sum(axis=0) >= int(min_obs)
    positive_ok = (dollar_volume > 0).sum(axis=0) >= int(min_obs)
    selected = dollar_volume.columns[obs_ok & positive_ok]
    if len(selected) == 0:
        return [], pd.Series(dtype=float)

    adv = (
        dollar_volume[selected]
        .mean(axis=0, skipna=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if adv.empty:
        return [], pd.Series(dtype=float)
    top = adv.nlargest(min(int(top_n), len(adv)))
    return top.index.tolist(), top.astype(float)


def build_liquid_universe_by_date(
    *,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    close_prices: pd.DataFrame | None = None,
    volumes: pd.DataFrame | None = None,
    rebalance_dates: Sequence[pd.Timestamp | str],
    top_n: int = 100,
    liquidity_lookback: int = 252,
    liq_lookback: int | None = None,
    min_listing_days: int = 252,
    min_obs: int = 200,
    min_price: float | None = None,
) -> dict[pd.Timestamp, dict[str, object]]:
    """Build liquidity-filtered universes for multiple rebalance dates.

    Parameters
    ----------
    close : pandas.DataFrame, optional
        Close-price panel.
    volume : pandas.DataFrame, optional
        Volume panel.
    close_prices, volumes : pandas.DataFrame, optional
        Aliases for ``close`` and ``volume``.
    rebalance_dates : sequence of pandas.Timestamp or str
        Dates for which universes should be selected.
    top_n : int, default=100
        Maximum number of assets per date.
    liquidity_lookback : int, default=252
        Lookback window for average dollar volume.
    liq_lookback : int, optional
        Alias for ``liquidity_lookback``.
    min_listing_days : int, default=252
        Minimum seasoning requirement.
    min_obs : int, default=200
        Minimum valid observations in the lookback window.
    min_price : float, optional
        Minimum price filter.

    Returns
    -------
    dict
        Mapping from rebalance date to a dictionary containing selected
        ``tickers`` and ``avg_dollar_volume``.

    Raises
    ------
    InputError
        If close/volume panels are missing.

    Notes
    -----
    Dates with no eligible universe are omitted from the returned mapping.
    """

    close_panel = close if close is not None else close_prices
    volume_panel = volume if volume is not None else volumes
    if close_panel is None or volume_panel is None:
        raise InputError("close/volume panels are required.")

    cp = _sanitize_frame(close_panel, name="close")
    vv = _sanitize_frame(volume_panel, name="volume")
    idx = cp.index.intersection(vv.index)
    cols = cp.columns.intersection(vv.columns)
    cp = cp.loc[idx, cols]
    vv = vv.loc[idx, cols]
    first_date = first_valid_close_volume_date(cp, vv)
    lookback = int(liq_lookback if liq_lookback is not None else liquidity_lookback)

    out: dict[pd.Timestamp, dict[str, object]] = {}
    for raw_dt in pd.to_datetime(list(rebalance_dates)):
        dt = pd.Timestamp(raw_dt)
        tickers, adv = select_liquid_universe(
            dt,
            close=cp,
            volume=vv,
            top_n=top_n,
            liquidity_lookback=lookback,
            min_listing_days=min_listing_days,
            min_obs=min_obs,
            min_price=min_price,
            first_date=first_date,
        )
        if tickers:
            out[dt] = {"tickers": tickers, "avg_dollar_volume": adv}
    return out


__all__ = [
    "build_liquid_universe_by_date",
    "clean_close_volume_panels",
    "first_valid_close_volume_date",
    "make_rebalance_dates",
    "prices_to_returns",
    "select_liquid_universe",
]
