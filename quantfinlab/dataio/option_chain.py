"""Option-chain loaders and composable filters.

A single :func:`load_option_chain` ingests three on-disk shapes used by
the project notebooks:

* OptionsDX SPX EOD parquet (wide call/put per row),
* OptionsDX / Deribit BTC EOD parquet (long form with ``option_right``),
* NSE NIFTY long-form daily option chain parquet (option_type ``CE``/``PE``).

All three return a normalized **long** DataFrame: one row per option
leg with stable column names. The filters operate on the long form so
they compose via :py:meth:`pandas.DataFrame.pipe`. :func:`pair_calls_puts`
collapses the long form into a wide ``(date, expiry, strike)`` table
with paired call/put quote columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..options.quote_cleaning import (
    closest_atm_pairs,
    ensure_option_mid_quotes,
    normalize_option_quote_schema,
    wide_option_chain_to_long,
)
from .schemas import get_option_chain_source

_REQUIRED_LONG_COLUMNS = ("date", "expiry", "strike", "option_type", "underlying")


def _clean_optionsdx_columns(columns) -> list[str]:
    return [str(c).strip().strip("[]").strip().lower() for c in columns]


def _normalize_optionsdx_text_frame(raw: pd.DataFrame, source_file: Path) -> pd.DataFrame:
    out = raw.copy()
    out.columns = _clean_optionsdx_columns(out.columns)
    for col in ("quote_readtime", "quote_date", "expire_date"):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in out.columns:
        if col in {"quote_readtime", "quote_date", "expire_date", "c_size", "p_size"}:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["source_file"] = source_file.name
    out["source_month"] = source_file.stem.split("_")[-1]
    return out


def combine_optionsdx_texts(
    files: str | Path | list[str | Path],
    output_path: str | Path,
    *,
    compression: str = "zstd",
) -> pd.DataFrame:
    """Combine one or more OptionsDX text files into a single Parquet file.

    Parameters
    ----------
    files : str, pathlib.Path, or list of path-like
        File path, glob-like path, directory, or explicit list of raw text files to
        combine. If a directory is supplied, all ``*.txt`` files in that directory
        are used.
    output_path : str or pathlib.Path
        Destination Parquet path. Parent directories are created automatically.
    compression : str, default "zstd"
        Parquet compression codec passed to the Parquet writer.

    Returns
    -------
    pandas.DataFrame
        Summary table with one row per source file and the number of normalized
        rows written from that file.

    Raises
    ------
    ValueError
        If no input files are found.

    Notes
    -----
    The output is written incrementally through a Parquet writer to avoid holding
    all normalized monthly files in memory at once. The normalized output includes
    source-file and source-month traceability columns where available.
    """

    if isinstance(files, (str, Path)):
        p = Path(files)
        file_list = sorted(p.glob("*.txt")) if p.is_dir() else sorted(p.parent.glob(p.name))
    else:
        file_list = [Path(f) for f in files]
    if not file_list:
        raise ValueError("No OptionsDX text files were provided.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    rows = []
    try:
        for path in file_list:
            raw = pd.read_csv(path, skipinitialspace=True, low_memory=False)
            frame = _normalize_optionsdx_text_frame(raw, path)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression=compression)
            writer.write_table(table)
            rows.append({"source_file": path.name, "rows": int(len(frame))})
    finally:
        if writer is not None:
            writer.close()
    return pd.DataFrame(rows)


def _read_chain(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False, usecols=columns)
    raise ValueError(f"Unsupported option-chain file format: {path.suffix}")


def _add_tau(df: pd.DataFrame, *, annualization_days: float) -> pd.DataFrame:
    out = df
    if "expiry" not in out.columns or "date" not in out.columns:
        return out
    expiry = pd.to_datetime(out["expiry"], errors="coerce")
    quote = pd.to_datetime(out.get("timestamp", out["date"]), errors="coerce")
    seconds = (expiry - quote).dt.total_seconds()
    tau = seconds / (float(annualization_days) * 86400.0)
    out = out.assign(tau=tau.astype(float))
    out["dte_calendar"] = tau * float(annualization_days)
    return out


def _add_quote_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    bid = pd.to_numeric(out.get("bid"), errors="coerce") if "bid" in out.columns else None
    ask = pd.to_numeric(out.get("ask"), errors="coerce") if "ask" in out.columns else None
    if bid is not None and ask is not None:
        spread = (ask - bid).clip(lower=0.0)
        mid = out["mid"] if "mid" in out.columns else 0.5 * (bid + ask)
        mid = pd.to_numeric(mid, errors="coerce")
        rel = spread / mid.where(mid > 0)
        out = out.assign(spread=spread, rel_spread=rel)
        if "mid" not in out.columns:
            out["mid"] = mid
    if "spot" in out.columns and "strike" in out.columns:
        spot = pd.to_numeric(out["spot"], errors="coerce")
        strike = pd.to_numeric(out["strike"], errors="coerce")
        out = out.assign(k_over_s=(strike / spot.where(spot > 0)).astype(float))
    if "spread" in out.columns:
        out["liq_score"] = 1.0 / (1.0 + out["spread"].fillna(np.inf))
    return out


def load_option_chain(
    path: str | Path,
    *,
    source: str = "optionsdx_spx",
    columns: list[str] | None = None,
    annualization_days: float | None = None,
) -> pd.DataFrame:
    """Load an option-chain file into a normalized long quote table.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV or Parquet option-chain file.
    source : str, default "optionsdx_spx"
        Registered source schema used to determine whether the raw file is wide
        call/put format or already close to long quote format.
    columns : list of str or None, optional
        Optional column subset to read from the source file.
    annualization_days : float or None, optional
        Denominator used to convert calendar time-to-expiry into year fractions.
        If omitted, the registered source default is used.

    Returns
    -------
    pandas.DataFrame
        Long-form option quote table with normalized date, expiry, strike,
        option-type, underlying, spot, quote, Greek, maturity, spread, moneyness,
        liquidity, and quote-profile columns where available. Metadata attributes
        include ``chain_source`` and ``annualization_days``.

    Raises
    ------
    ValueError
        If the file does not exist, normalization produces an empty table, or
        required normalized columns are missing.

    Notes
    -----
    The returned ``tau`` is measured in years. Quote-mid, spread, relative spread,
    moneyness, and liquidity diagnostics are added after schema normalization.
    """

    p = Path(path)
    if not p.exists():
        raise ValueError(f"Option-chain file does not exist: {p}")

    cfg = get_option_chain_source(source)
    schema = str(cfg["schema"])
    profile = str(cfg["profile"])
    underlying_default = cfg.get("underlying_default")
    eff_ann = float(annualization_days if annualization_days is not None else cfg["annualization_days"])

    raw = _read_chain(p, columns=columns)

    if schema == "wide_call_put":
        long = wide_option_chain_to_long(raw, underlying_default=underlying_default)
    else:
        long = normalize_option_quote_schema(raw, profile=profile, underlying_default=underlying_default)

    if long.empty:
        raise ValueError(f"Option chain at {p} normalized to empty DataFrame.")

    missing = [c for c in _REQUIRED_LONG_COLUMNS if c not in long.columns]
    if missing:
        raise ValueError(
            f"Option chain at {p} is missing required columns after normalization: {missing}"
        )

    long = ensure_option_mid_quotes(long)
    long = _add_tau(long, annualization_days=eff_ann)
    long = _add_quote_metrics(long)
    long = long.reset_index(drop=True)
    long.attrs["chain_source"] = source
    long.attrs["annualization_days"] = eff_ann
    return long


def load_spx_option_pairs(
    path: str | Path,
    *,
    annualization_days: float = 365.25,
    max_rel_spread: float = 0.30,
    tau_min_days: float = 7.0,
    tau_max_days: float = 180.0,
    k_over_s_range: tuple[float, float] = (0.70, 1.42),
    top_n_per_expiry: int | None = 70,
    min_pairs_per_expiry: int = 6,
) -> pd.DataFrame:
    """Load wide SPX option quotes as paired call/put rows.

    Parameters
    ----------
    path : str or pathlib.Path
        OptionsDX-style wide quote file containing call and put columns on the same
        date-expiry-strike row.
    annualization_days : float, default 365.25
        Number of calendar days used to convert time to expiry into years.
    max_rel_spread : float, default 0.30
        Maximum allowed relative bid-ask spread for both call and put legs.
    tau_min_days : float, default 7.0
        Minimum calendar days to expiry.
    tau_max_days : float, default 180.0
        Maximum calendar days to expiry.
    k_over_s_range : tuple of float, default (0.70, 1.42)
        Inclusive strike-to-spot moneyness range.
    top_n_per_expiry : int or None, default 70
        If positive, keep the closest-to-the-money rows per date-expiry group.
    min_pairs_per_expiry : int, default 6
        Minimum number of paired strikes required for a date-expiry group.

    Returns
    -------
    pandas.DataFrame
        Filtered wide quote table with paired call/put bid, ask, mid, spread,
        relative-spread, volume, moneyness, time-to-expiry, and liquidity columns.

    Raises
    ------
    ValueError
        If the source file does not exist.

    Notes
    -----
    The function keeps the input in paired wide form for efficiency. This avoids
    doubling very large quote files before basic liquidity and ATM-window filters
    are applied.
    """

    p = Path(path)
    if not p.exists():
        raise ValueError(f"Option-chain file does not exist: {p}")

    columns = [
        "quote_date",
        "quote_readtime",
        "expire_date",
        "underlying_last",
        "strike",
        "c_bid",
        "c_ask",
        "p_bid",
        "p_ask",
        "c_volume",
        "p_volume",
    ]
    raw = _read_chain(p, columns=columns)
    out = raw.copy()

    for col in ("quote_date", "quote_readtime", "expire_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["date"] = out["quote_date"].dt.normalize()
    out["timestamp"] = pd.to_datetime(out["quote_readtime"], errors="coerce")
    out["expiry"] = out["expire_date"].dt.normalize()
    out["tau"] = (
        (out["expiry"] - out["timestamp"]).dt.total_seconds()
        / (float(annualization_days) * 86400.0)
    )
    out["dte_calendar"] = out["tau"] * float(annualization_days)
    out["underlying"] = "SPX"
    out["spot"] = pd.to_numeric(out["underlying_last"], errors="coerce")
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")

    for col in ("c_bid", "c_ask", "p_bid", "p_ask", "c_volume", "p_volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["c_mid"] = 0.5 * (out["c_bid"] + out["c_ask"])
    out["p_mid"] = 0.5 * (out["p_bid"] + out["p_ask"])
    out["c_spread"] = (out["c_ask"] - out["c_bid"]).clip(lower=0.0)
    out["p_spread"] = (out["p_ask"] - out["p_bid"]).clip(lower=0.0)
    out["c_rel_spread"] = out["c_spread"] / out["c_mid"].where(out["c_mid"] > 0)
    out["p_rel_spread"] = out["p_spread"] / out["p_mid"].where(out["p_mid"] > 0)
    out["volume_sum"] = out["c_volume"].fillna(0.0) + out["p_volume"].fillna(0.0)
    out["spread_sum"] = out["c_spread"] + out["p_spread"]
    out["liq_score"] = np.sqrt(out["volume_sum"] + 1.0) / (out["spread_sum"] + 1e-4)
    out["k_over_s"] = out["strike"] / out["spot"].where(out["spot"] > 0)

    valid = (
        out["date"].notna()
        & out["timestamp"].notna()
        & out["expiry"].notna()
        & (out["spot"] > 0)
        & (out["strike"] > 0)
        & (out["tau"] > 0)
        & (out["c_bid"] >= 0)
        & (out["p_bid"] >= 0)
        & (out["c_ask"] >= out["c_bid"])
        & (out["p_ask"] >= out["p_bid"])
        & (out["c_mid"] > 0)
        & (out["p_mid"] > 0)
    )
    liquid = (
        out["c_rel_spread"].le(float(max_rel_spread))
        & out["p_rel_spread"].le(float(max_rel_spread))
        & out["dte_calendar"].between(float(tau_min_days), float(tau_max_days))
    )
    lo, hi = float(k_over_s_range[0]), float(k_over_s_range[1])
    moneyness = out["k_over_s"].between(lo, hi)
    out = out.loc[valid & liquid & moneyness].copy()

    if top_n_per_expiry and top_n_per_expiry > 0 and not out.empty:
        out["atm_dist_s"] = np.abs(np.log(out["k_over_s"].where(out["k_over_s"] > 0)))
        out["atm_rank"] = out.groupby(["date", "expiry"])["atm_dist_s"].rank(
            method="first",
            ascending=True,
        )
        out = out.loc[out["atm_rank"] <= int(top_n_per_expiry)].copy()

    if min_pairs_per_expiry and min_pairs_per_expiry > 0 and not out.empty:
        counts = out.groupby(["date", "expiry"]).size()
        keep = counts[counts >= int(min_pairs_per_expiry)].index
        idx = out.set_index(["date", "expiry"]).index
        out = out.loc[idx.isin(keep)].copy()

    keep_cols = [
        "date",
        "timestamp",
        "expiry",
        "underlying",
        "spot",
        "strike",
        "tau",
        "dte_calendar",
        "k_over_s",
        "c_bid",
        "c_ask",
        "c_mid",
        "c_spread",
        "c_rel_spread",
        "c_volume",
        "p_bid",
        "p_ask",
        "p_mid",
        "p_spread",
        "p_rel_spread",
        "p_volume",
        "volume_sum",
        "spread_sum",
        "liq_score",
    ]
    out = out[keep_cols].sort_values(["date", "expiry", "strike"]).reset_index(drop=True)
    out.attrs["chain_source"] = "optionsdx_spx"
    out.attrs["annualization_days"] = float(annualization_days)
    return out


def load_optionsdx_equity_pairs(
    path: str | Path,
    *,
    source: str = "optionsdx_spy",
    annualization_days: float = 365.25,
    max_rel_spread: float = 0.35,
    tau_min_days: float = 7.0,
    tau_max_days: float = 180.0,
    k_over_s_range: tuple[float, float] = (0.70, 1.35),
    top_n_per_expiry: int | None = 70,
    min_pairs_per_expiry: int = 6,
) -> pd.DataFrame:
    """Load OptionsDX-style equity option quotes as paired call/put rows.

    Parameters
    ----------
    path : str or pathlib.Path
        Wide OptionsDX-style quote file.
    source : str, default "optionsdx_spy"
        Registered equity-option source used to set source metadata and default
        underlying label.
    annualization_days : float, default 365.25
        Number of calendar days used to convert time to expiry into years.
    max_rel_spread : float, default 0.35
        Maximum allowed relative bid-ask spread for both legs.
    tau_min_days : float, default 7.0
        Minimum calendar days to expiry.
    tau_max_days : float, default 180.0
        Maximum calendar days to expiry.
    k_over_s_range : tuple of float, default (0.70, 1.35)
        Inclusive strike-to-spot moneyness range.
    top_n_per_expiry : int or None, default 70
        Number of closest-to-the-money paired strikes to keep per date-expiry group.
    min_pairs_per_expiry : int, default 6
        Minimum paired strikes required for each date-expiry group.

    Returns
    -------
    pandas.DataFrame
        Filtered paired quote panel with call/put quote columns, moneyness,
        maturity, and liquidity diagnostics. Metadata attributes include the source
        name and annualization convention.

    Notes
    -----
    This helper reuses the paired-quote filtering logic and then assigns the
    underlying label from the registered source configuration.
    """

    cfg = get_option_chain_source(source)
    underlying = str(cfg.get("underlying_default", ""))
    out = load_spx_option_pairs(
        path,
        annualization_days=annualization_days,
        max_rel_spread=max_rel_spread,
        tau_min_days=tau_min_days,
        tau_max_days=tau_max_days,
        k_over_s_range=k_over_s_range,
        top_n_per_expiry=top_n_per_expiry,
        min_pairs_per_expiry=min_pairs_per_expiry,
    )
    out["underlying"] = underlying
    out.attrs["chain_source"] = source
    out.attrs["annualization_days"] = float(annualization_days)
    return out


def filter_valid_quotes(
    df: pd.DataFrame,
    *,
    min_bid: float = 0.0,
    require_pair: bool = True,
) -> pd.DataFrame:
    """Filter long option quotes for basic quote validity.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form option quote table.
    min_bid : float, default 0.0
        Minimum allowed bid.
    require_pair : bool, default True
        If ``True``, keep only date-expiry-strike groups that contain both a call
        and a put.

    Returns
    -------
    pandas.DataFrame
        Filtered quote table with the index reset.

    Notes
    -----
    Rows are retained only when bid and ask are finite, bid is at least
    ``min_bid``, ask is not below bid, and mid is strictly positive. Pair filtering
    is applied only when the required grouping columns are present.
    """

    out = df.copy()
    bid = pd.to_numeric(out.get("bid"), errors="coerce") if "bid" in out.columns else pd.Series(np.nan, index=out.index)
    ask = pd.to_numeric(out.get("ask"), errors="coerce") if "ask" in out.columns else pd.Series(np.nan, index=out.index)
    mid = pd.to_numeric(out.get("mid"), errors="coerce") if "mid" in out.columns else pd.Series(np.nan, index=out.index)
    mask = (
        bid.notna() & ask.notna()
        & (bid >= float(min_bid))
        & (ask >= bid)
        & (mid > 0.0)
    )
    out = out.loc[mask].copy()

    if require_pair and {"date", "expiry", "strike", "option_type"}.issubset(out.columns):
        groups = out.groupby(["date", "expiry", "strike"], dropna=False)["option_type"].nunique()
        complete = groups[groups >= 2].index
        idx = out.set_index(["date", "expiry", "strike"]).index
        out = out.loc[idx.isin(complete)].copy()
    return out.reset_index(drop=True)


def filter_liquidity(
    df: pd.DataFrame,
    *,
    max_rel_spread: float = 0.20,
    tau_min_days: float = 7.0,
    tau_max_days: float = 120.0,
) -> pd.DataFrame:
    """Filter option quotes by spread and time-to-expiry bounds.

    Parameters
    ----------
    df : pandas.DataFrame
        Option quote table containing ``rel_spread`` and either ``dte_calendar`` or
        ``tau`` where available.
    max_rel_spread : float, default 0.20
        Maximum allowed relative bid-ask spread.
    tau_min_days : float, default 7.0
        Minimum calendar days to expiry.
    tau_max_days : float, default 120.0
        Maximum calendar days to expiry.

    Returns
    -------
    pandas.DataFrame
        Liquidity-filtered quote table with the index reset.

    Notes
    -----
    If ``dte_calendar`` is missing but ``tau`` is present, days-to-expiry are
    computed using the ``annualization_days`` DataFrame attribute, defaulting to
    365.25.
    """

    out = df.copy()
    if "rel_spread" in out.columns:
        rs = pd.to_numeric(out["rel_spread"], errors="coerce")
        out = out.loc[rs.notna() & (rs <= float(max_rel_spread))]
    if "dte_calendar" in out.columns:
        dte = pd.to_numeric(out["dte_calendar"], errors="coerce")
    elif "tau" in out.columns:
        ann = float(df.attrs.get("annualization_days", 365.25))
        dte = pd.to_numeric(out["tau"], errors="coerce") * ann
    else:
        dte = None
    if dte is not None:
        out = out.loc[dte.notna() & (dte >= float(tau_min_days)) & (dte <= float(tau_max_days))]
    return out.reset_index(drop=True)


def filter_atm_window(
    df: pd.DataFrame,
    *,
    k_over_s_range: tuple[float, float] = (0.85, 1.15),
    top_n_per_expiry: int = 25,
    min_pairs_per_group: int = 10,
) -> pd.DataFrame:
    """Restrict option quotes to an at-the-money moneyness window.

    Parameters
    ----------
    df : pandas.DataFrame
        Option quote table.
    k_over_s_range : tuple of float, default (0.85, 1.15)
        Inclusive strike-to-spot range to keep.
    top_n_per_expiry : int, default 25
        Number of closest-to-the-money paired strikes to keep per date-expiry group
        when the necessary columns are available.
    min_pairs_per_group : int, default 10
        Minimum number of complete call/put pairs required per date-expiry group.

    Returns
    -------
    pandas.DataFrame
        Filtered quote table with the index reset.

    Notes
    -----
    The function first applies the explicit moneyness interval, then optionally
    keeps the closest ATM pairs, and finally drops date-expiry groups with too few
    complete pairs.
    """

    out = df.copy()
    if "k_over_s" in out.columns:
        kos = pd.to_numeric(out["k_over_s"], errors="coerce")
        lo, hi = float(k_over_s_range[0]), float(k_over_s_range[1])
        out = out.loc[kos.notna() & (kos >= lo) & (kos <= hi)]

    if {"date", "expiry", "strike", "option_type", "spot"}.issubset(out.columns) and top_n_per_expiry:
        out = closest_atm_pairs(out, n_pairs=int(top_n_per_expiry))

    if {"date", "expiry"}.issubset(out.columns) and min_pairs_per_group > 0:
        counts = (
            out.groupby(["date", "expiry", "strike"], dropna=False)["option_type"].nunique()
            .groupby(level=[0, 1]).apply(lambda s: int((s >= 2).sum()))
        )
        keep = counts[counts >= int(min_pairs_per_group)].index
        idx = out.set_index(["date", "expiry"]).index
        out = out.loc[idx.isin(keep)]
    return out.reset_index(drop=True)


def pair_calls_puts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-form call and put quotes into paired wide rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form quote table containing ``date``, ``expiry``, ``strike``, and
        ``option_type``. Optional leg-level columns include bid, ask, mid, volume,
        spread, and relative spread.

    Returns
    -------
    pandas.DataFrame
        Wide table indexed by rows rather than a MultiIndex, with common contract
        columns and prefixed call/put leg columns such as ``c_mid`` and ``p_mid``.

    Raises
    ------
    ValueError
        If ``option_type`` is missing.

    Notes
    -----
    If multiple rows exist for the same date-expiry-strike-leg combination, the
    last row is retained. The function does not require both legs to be present;
    missing legs remain NaN in the paired output.
    """

    if df.empty:
        return df.copy()
    if "option_type" not in df.columns:
        raise ValueError("pair_calls_puts requires an 'option_type' column.")

    keys = ["date", "expiry", "strike"]
    keep_one = ["underlying", "spot", "tau", "dte_calendar", "k_over_s"]
    keep_one = [c for c in keep_one if c in df.columns]
    leg_value_cols = [c for c in ("bid", "ask", "mid", "volume", "spread", "rel_spread")
                      if c in df.columns]

    base = (
        df[keys + keep_one]
        .drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )

    legs = df[keys + ["option_type"] + leg_value_cols].copy()
    legs["option_type"] = legs["option_type"].astype(str).str.lower()
    pivots = []
    for opt in ("call", "put"):
        sub = legs.loc[legs["option_type"].eq(opt)].copy()
        if sub.empty:
            continue
        sub = sub.drop(columns=["option_type"]).drop_duplicates(subset=keys, keep="last")
        prefix = "c_" if opt == "call" else "p_"
        rename = {c: f"{prefix}{c}" for c in leg_value_cols}
        sub = sub.rename(columns=rename)
        pivots.append(sub)

    paired = base
    for piv in pivots:
        paired = paired.merge(piv, on=keys, how="left")

    return paired.reset_index(drop=True)


__all__ = [
    "filter_atm_window",
    "filter_liquidity",
    "filter_valid_quotes",
    "combine_optionsdx_texts",
    "load_option_chain",
    "load_optionsdx_equity_pairs",
    "load_spx_option_pairs",
    "pair_calls_puts",
]
