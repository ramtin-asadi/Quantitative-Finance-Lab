"""Multi-asset wide panel loaders.

A single ``load_yfinance_panel`` ingests two on-disk shapes:

* yfinance-style export with ``<TICKER>__<field>`` columns
  (NB6 ETF_data.csv, NB3 nasdaq_all_close_volume.parquet)
* Stooq HKEX multi-header CSV with ticker on header level 0 and
  field name (``Close`` / ``Volume``) on header level 1 (NB2 / NB3).

Both shapes return the same ``dict[field_name -> wide DataFrame]`` with
DatetimeIndex (sorted, deduplicated) and tickers as columns.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..portfolio.universe import prices_to_returns
from .schemas import get_panel_source


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
    path: str | Path,
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
    """Load a wide multi-asset panel into ``{field -> DataFrame}``.

    Returns a dict mapping the lowercased field name to a wide DataFrame
    with a sorted, deduplicated ``DatetimeIndex`` and tickers as columns.
    Coerces all values to numeric, drops empty columns, sorts columns
    alphabetically, and (optionally) restricts to ``tickers``.

    Parameters
    ----------
    path
        Path to a CSV or Parquet file.
    fields
        Field names to extract. Names are matched case-insensitively.
        Missing fields become empty DataFrames in the output.
    tickers
        Optional whitelist of ticker symbols. Matching is case-insensitive.
    start, end
        Optional date cutoff applied after parsing.
    lowercase
        If true, normalize ticker columns to lowercase. This is useful when
        combining ETF and equity panels in notebooks.
    source
        Registered schema in :data:`PANEL_SOURCES`. ``None`` skips the
        registry; use ``date_col``/``suffix`` to drive the loader.
    """
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


def align_panels(*panels: pd.DataFrame, how: str = "inner") -> tuple[pd.DataFrame, ...]:
    """Align several wide panels on the index/columns intersection or union."""
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
    """Wrap :func:`portfolio.universe.prices_to_returns` with NB6 idioms.

    Optionally forward-fills price gaps within ``ffill_limit`` and fills
    any remaining post-inception NaNs with ``fill_isolated_with`` (so
    isolated holiday gaps don't propagate). Returns a returns DataFrame
    aligned to ``close``'s index.
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
    """Load a saved CBOE VIX close series and optionally align it to a target index.

    The CSV is expected to have a date column and a single ``VIX`` close column
    (as written by ``yfinance``). When ``index`` is supplied the series is
    forward-filled onto that index so it can be merged with a daily return panel.
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
    """Distil a raw VIX series into stationary RL-observation features.

    Returns three columns: a 20-day rolling z-score of the level, the ratio of
    VIX to its 63-day moving average (term-structure proxy), and the 252-day
    percentile rank of the level. All are bounded and free of raw macro data.
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
    "load_vix",
    "load_yfinance_panel",
    "prices_to_returns_panel",
    "vix_feature_frame",
]
