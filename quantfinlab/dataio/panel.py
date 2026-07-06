"""Multi-asset wide panel loaders.

``load_yfinance_panel`` ingests source-centered root files with
``<TICKER>__<field>`` columns, including ETF CSVs and Stooq Parquet
close/volume panels. It returns ``dict[field_name -> wide DataFrame]``
with a sorted, deduplicated ``DatetimeIndex`` and tickers as columns.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..portfolio.universe import prices_to_returns
from .schemas import get_panel_source


def _resolve_data_file(path: str | Path, filename: str) -> Path:
    p = Path(path)
    return p / filename if p.is_dir() else p


def _read_table(path: Path, *, header: int | list[int] = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, header=header, low_memory=False)
    raise ValueError(f"Unsupported panel file format: {path.suffix}")


def _resolve_date_column(columns: Iterable[str], expected: str) -> str | None:
    cols = list(columns)
    lower = [str(c).lower() for c in cols]
    target = str(expected).lower()
    if target in lower:
        return cols[lower.index(target)]
    for cand in ("date", "trade_date", "datetime"):
        if cand in lower:
            return cols[lower.index(cand)]
    return None


def _load_wide_suffix(
    path: Path,
    *,
    suffix: str,
    date_col: str,
    fields: tuple[str, ...],
    tickers: list[str] | None,
) -> dict[str, pd.DataFrame]:
    raw = _read_table(path, header=0)
    raw = raw.rename(columns={c: str(c) for c in raw.columns})
    real_date = _resolve_date_column(raw.columns, date_col)
    if real_date is None:
        raise ValueError(f"Panel file {path.name} has no date column (looked for {date_col!r}).")
    dates = pd.to_datetime(raw[real_date], errors="coerce")
    raw = raw.drop(columns=[real_date])
    raw.index = dates

    field_set = {f.lower() for f in fields}
    out: dict[str, dict[str, pd.Series]] = {f: {} for f in field_set}
    for col in raw.columns:
        s = str(col)
        if suffix not in s:
            continue
        ticker, field = s.rsplit(suffix, 1)
        field_lc = field.lower()
        if field_lc not in field_set:
            continue
        out[field_lc][ticker] = raw[col]

    panels: dict[str, pd.DataFrame] = {}
    for field_lc, ticker_map in out.items():
        if not ticker_map:
            panels[field_lc] = pd.DataFrame(index=raw.index.dropna().sort_values().unique())
            continue
        df = pd.DataFrame(ticker_map)
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(axis=1, how="all").reindex(columns=sorted(df.columns))
        if tickers is not None:
            keep = [t for t in tickers if t in df.columns]
            df = df.loc[:, keep]
        panels[field_lc] = df
    return panels


def _load_multi_header(
    path: Path,
    *,
    date_col: str,
    fields: tuple[str, ...],
    tickers: list[str] | None,
) -> dict[str, pd.DataFrame]:
    raw = _read_table(path, header=[0, 1])
    raw.columns = pd.MultiIndex.from_tuples(
        [(str(a).strip(), str(b).strip()) for a, b in raw.columns]
    )

    date_cols = [c for c in raw.columns if str(c[0]).lower() == date_col.lower()]
    if not date_cols:
        date_cols = [c for c in raw.columns if str(c[0]).lower() == "date"]
    if not date_cols:
        raise ValueError(f"Multi-header panel {path.name} has no date column.")
    dates = pd.to_datetime(raw[date_cols[0]], errors="coerce")

    panels: dict[str, pd.DataFrame] = {}
    for field in fields:
        field_lc = field.lower()
        col_match = [c for c in raw.columns if str(c[1]).lower() == field_lc]
        if not col_match:
            panels[field_lc] = pd.DataFrame(index=pd.DatetimeIndex([]))
            continue
        df = raw.loc[:, col_match].copy()
        df.columns = [str(c[0]).strip() for c in col_match]
        if df.columns.duplicated().any():
            df = df.T.groupby(level=0).last().T
        df.index = dates
        df = df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        df = df[~df.index.isna()]
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.dropna(axis=1, how="all").reindex(columns=sorted(df.columns))
        if tickers is not None:
            keep = [t for t in tickers if t in df.columns]
            df = df.loc[:, keep]
        panels[field_lc] = df
    return panels


def load_yfinance_panel(
    path: str | Path | Sequence[str | Path],
    *,
    fields: tuple[str, ...] = ("close", "volume", "dividends", "stock_splits"),
    tickers: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    lowercase: bool = False,
    date_col: str | None = None,
    suffix: str = "__",
    source: str | None = "yfinance_export",
) -> dict[str, pd.DataFrame]:
    """Load one or more wide multi-asset price panels.

    Parameters
    ----------
    path : str, pathlib.Path, or sequence of path-like
        CSV or Parquet file, or sequence of files, containing wide multi-asset
        fields.
    fields : tuple of str, default ("close", "volume", "dividends", "stock_splits")
        Field names to extract.
    tickers : list of str or None, optional
        Optional ticker whitelist. Matching is case-insensitive.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound.
    end : str, pandas.Timestamp, or None, optional
        Optional upper date bound.
    lowercase : bool, default False
        If ``True``, normalize ticker labels to lowercase.
    date_col : str or None, optional
        Explicit date column override.
    suffix : str, default "__"
        Field suffix separator used for wide-suffix schemas.
    source : str or None, default "yfinance_export"
        Registered source schema. Use ``None`` to bypass the registry and rely on
        explicit ``date_col`` and ``suffix`` settings.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from lowercase field name to a wide DataFrame indexed by sorted
        dates with tickers as columns.

    Raises
    ------
    ValueError
        If a requested file does not exist or the registered panel format is
        unsupported.

    Notes
    -----
    When a sequence of paths is supplied, panels are loaded separately and merged
    field by field. Duplicate ticker columns keep the last occurrence. Missing
    fields return empty DataFrames.
    """

    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, Path)):
        merged: dict[str, pd.DataFrame] = {}
        for one_path in path:
            part = load_yfinance_panel(
                one_path,
                fields=fields,
                tickers=None,
                start=start,
                end=end,
                lowercase=lowercase,
                date_col=date_col,
                suffix=suffix,
                source=source,
            )
            for field, frame in part.items():
                if field not in merged:
                    merged[field] = frame
                else:
                    merged[field] = pd.concat([merged[field], frame], axis=1)
                    merged[field] = merged[field].loc[:, ~merged[field].columns.duplicated(keep="last")]
        wanted = [str(t).strip() for t in tickers or [] if str(t).strip()]
        for field, frame in list(merged.items()):
            panel = frame.reindex(columns=sorted(frame.columns))
            if wanted:
                lookup = {str(c).lower(): c for c in panel.columns}
                keep = []
                seen = set()
                for ticker in wanted:
                    key = ticker.lower() if lowercase else ticker.lower()
                    col = key if lowercase and key in panel.columns else lookup.get(ticker.lower())
                    if col is not None and col not in seen:
                        keep.append(col)
                        seen.add(col)
                panel = panel.reindex(columns=keep)
            merged[field] = panel
        return merged

    p = Path(path)
    if not p.exists():
        raise ValueError(f"Panel file does not exist: {p}")

    cfg: dict[str, object] = {}
    if source is not None:
        cfg = get_panel_source(source)

    fmt = str(cfg.get("format", "wide_suffix"))
    eff_date_col = date_col or str(cfg.get("date_col", "date"))
    eff_suffix = str(cfg.get("suffix", suffix)) if "suffix" in cfg else suffix

    if fmt == "multi_header":
        panels = _load_multi_header(p, date_col=eff_date_col, fields=fields, tickers=None)
    elif fmt == "wide_suffix":
        panels = _load_wide_suffix(
            p, suffix=eff_suffix, date_col=eff_date_col, fields=fields, tickers=None
        )
    else:
        raise ValueError(f"Unsupported panel format {fmt!r} for source {source!r}")

    wanted = [str(t).strip() for t in tickers or [] if str(t).strip()]
    out: dict[str, pd.DataFrame] = {}
    for field, df in panels.items():
        panel = df.copy()
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()
        panel.columns = [str(c).strip().lower() if lowercase else str(c).strip() for c in panel.columns]
        if panel.columns.duplicated().any():
            panel = panel.T.groupby(level=0).last().T
        if start is not None:
            panel = panel.loc[panel.index >= pd.Timestamp(start)]
        if end is not None:
            panel = panel.loc[panel.index <= pd.Timestamp(end)]
        if wanted:
            if lowercase:
                keep = []
                seen: set[str] = set()
                for ticker in wanted:
                    t = ticker.lower()
                    if t in panel.columns and t not in seen:
                        keep.append(t)
                        seen.add(t)
            else:
                lookup = {str(c).lower(): c for c in panel.columns}
                keep = []
                seen = set()
                for ticker in wanted:
                    col = lookup.get(ticker.lower())
                    if col is not None and col not in seen:
                        keep.append(col)
                        seen.add(col)
            panel = panel.reindex(columns=keep)
        else:
            panel = panel.reindex(columns=sorted(panel.columns))
        out[field] = panel
    return out


def load_nasdaq_close_volume(
    path: str | Path,
    *,
    tickers: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    lowercase: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load a NASDAQ close/volume panel.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the panel file or a directory containing the expected close/volume
        Parquet file.
    tickers : list of str or None, optional
        Optional ticker whitelist.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound.
    end : str, pandas.Timestamp, or None, optional
        Optional upper date bound.
    lowercase : bool, default False
        If ``True``, normalize ticker labels to lowercase.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing ``"close"`` and ``"volume"`` DataFrames.
    """

    p = _resolve_data_file(path, "nasdaq_close_volume.parquet")
    return load_yfinance_panel(
        p,
        fields=("close", "volume"),
        tickers=tickers,
        start=start,
        end=end,
        lowercase=lowercase,
        source="nasdaq_close_volume",
    )


def load_hkex_close_volume(
    path: str | Path,
    *,
    tickers: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    lowercase: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load an HKEX close/volume panel.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the panel file or a directory containing the expected close/volume
        Parquet file.
    tickers : list of str or None, optional
        Optional ticker whitelist.
    start : str, pandas.Timestamp, or None, optional
        Optional lower date bound.
    end : str, pandas.Timestamp, or None, optional
        Optional upper date bound.
    lowercase : bool, default False
        If ``True``, normalize ticker labels to lowercase.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary containing ``"close"`` and ``"volume"`` DataFrames.
    """

    p = _resolve_data_file(path, "hkex_close_volume.parquet")
    return load_yfinance_panel(
        p,
        fields=("close", "volume"),
        tickers=tickers,
        start=start,
        end=end,
        lowercase=lowercase,
        source="hkex_close_volume",
    )


def align_panels(*panels: pd.DataFrame, how: str = "inner") -> tuple[pd.DataFrame, ...]:
    """Align multiple wide panels on shared or combined dates and columns.

    Parameters
    ----------
    *panels : pandas.DataFrame
        Wide DataFrames to align.
    how : {"inner", "outer"}, default "inner"
        ``"inner"`` uses the intersection of indices and columns. Any other value
        uses the union.

    Returns
    -------
    tuple[pandas.DataFrame, ...]
        Reindexed panels in the order of the non-None inputs. Empty input returns
        an empty tuple.

    Notes
    -----
    ``None`` inputs are ignored when computing the alignment set and are not
    included in the returned tuple.
    """

    if not panels:
        return ()
    valid = [p for p in panels if p is not None]
    if not valid:
        return tuple(panels)
    idx = valid[0].index
    cols = valid[0].columns
    for p in valid[1:]:
        idx = idx.intersection(p.index) if how == "inner" else idx.union(p.index)
        cols = cols.intersection(p.columns) if how == "inner" else cols.union(p.columns)
    aligned = tuple(p.reindex(index=idx, columns=cols) for p in valid)
    return aligned


def prices_to_returns_panel(
    close: pd.DataFrame,
    *,
    kind: str = "simple",
    ffill_limit: int | None = 3,
    fill_isolated_with: float | None = 0.0,
) -> pd.DataFrame:
    """Convert a wide price panel to a wide return panel.

    Parameters
    ----------
    close : pandas.DataFrame
        Price panel indexed by date with assets in columns.
    kind : str, default "simple"
        Return convention passed to the underlying price-to-return routine.
    ffill_limit : int or None, default 3
        Optional maximum number of consecutive missing prices to forward-fill
        before computing returns.
    fill_isolated_with : float or None, default 0.0
        Value used to fill remaining missing returns after return calculation. Use
        ``None`` to preserve NaNs.

    Returns
    -------
    pandas.DataFrame
        Return panel aligned to the input price index.

    Notes
    -----
    Infinite returns are replaced by NaN before optional filling. The default
    settings are designed to avoid isolated holiday gaps from propagating through
    daily panels.
    """

    px = close.copy()
    if ffill_limit is not None and ffill_limit > 0:
        px = px.ffill(limit=ffill_limit)
    rets = prices_to_returns(px, kind=kind)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    if fill_isolated_with is not None:
        rets = rets.where(rets.notna(), fill_isolated_with)
    return rets


def load_vix(
    path: str | Path,
    *,
    index: pd.Index | None = None,
    ffill_limit: int | None = None,
) -> pd.Series:
    """Load a saved VIX close series and optionally align it to a target index.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV file containing a date column and a VIX close column.
    index : pandas.Index or None, optional
        Optional target index. When supplied, the VIX series is forward-filled onto
        this index.
    ffill_limit : int or None, optional
        Maximum number of consecutive missing values to forward-fill during
        alignment.

    Returns
    -------
    pandas.Series
        VIX close series named ``"VIX"``.

    Notes
    -----
    The value column is selected as ``"VIX"`` when present; otherwise the first
    numeric column is used. Duplicate dates keep the last observation.
    """

    frame = pd.read_csv(Path(path))
    date_col = _resolve_date_column(frame.columns, "Date") or frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    value_col = "VIX" if "VIX" in frame.columns else frame.select_dtypes("number").columns[0]
    vix = pd.to_numeric(frame[value_col], errors="coerce").rename("VIX")
    vix = vix[~vix.index.duplicated(keep="last")]
    if index is not None:
        target = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
        vix = vix.reindex(vix.index.union(target)).ffill(limit=ffill_limit).reindex(target)
    return vix


def vix_feature_frame(vix: pd.Series, *, index: pd.Index | None = None) -> pd.DataFrame:
    """Transform a VIX series into bounded volatility-regime features.

    Parameters
    ----------
    vix : pandas.Series
        Raw VIX level series.
    index : pandas.Index or None, optional
        Optional target index. When supplied, features are forward-filled onto this
        index.

    Returns
    -------
    pandas.DataFrame
        Feature frame with columns ``vix_z_20``, ``vix_ma_ratio_63``, and
        ``vix_pct_252``.

    Notes
    -----
    The features are a 20-day rolling z-score, a 63-day moving-average ratio, and a
    252-day rolling percentile rank. Missing and infinite values are replaced with
    zero after feature construction.
    """

    v = pd.to_numeric(pd.Series(vix), errors="coerce").astype(float)
    v.index = pd.to_datetime(v.index)
    v = v.sort_index().ffill()
    roll20 = v.rolling(20, min_periods=10)
    z20 = ((v - roll20.mean()) / roll20.std(ddof=1)).clip(-4.0, 4.0)
    ma63 = v.rolling(63, min_periods=21).mean()
    ratio63 = (v / ma63.replace(0.0, np.nan)).clip(0.25, 4.0)
    pct252 = v.rolling(252, min_periods=63).rank(pct=True)
    out = pd.DataFrame({"vix_z_20": z20, "vix_ma_ratio_63": ratio63, "vix_pct_252": pct252})
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if index is not None:
        target = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
        out = out.reindex(out.index.union(target)).ffill().reindex(target).fillna(0.0)
    return out


__all__ = [
    "align_panels",
    "load_hkex_close_volume",
    "load_nasdaq_close_volume",
    "load_vix",
    "load_yfinance_panel",
    "prices_to_returns_panel",
    "vix_feature_frame",
]
