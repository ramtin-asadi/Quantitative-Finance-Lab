from __future__ import annotations

import io
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache" / "sec_enrichment"
SUB_CACHE_DIR = CACHE_DIR / "fsds_submissions"
ARCHIVE_MANIFEST_PATH = CACHE_DIR / "fsds_archives.json"
SIC_CODES_PATH = CACHE_DIR / "sic_codes.parquet"

FSDS_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/"
    "financial-statement-data-sets"
)
FSDS_DOCUMENTATION = "https://www.sec.gov/files/fsds.pdf"
SIC_CODES_URL = (
    "https://www.sec.gov/search-filings/"
    "standard-industrial-classification-sic-code-list"
)
COMPANY_FACTS_BULK_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)

SHARE_CONCEPTS = [
    "dei:EntityCommonStockSharesOutstanding",
    "us-gaap:CommonStockSharesOutstanding",
]
SHARE_CONCEPT_RANK = {
    "dei:EntityCommonStockSharesOutstanding": 2,
    "us-gaap:CommonStockSharesOutstanding": 1,
}
DEFAULT_MAX_SHARE_AGE_DAYS = 400
MIN_REFERENCE_MARKET_CAP = 50_000_000.0
MAX_REFERENCE_MARKET_CAP = 10_000_000_000_000.0
SCALE_LOWER_BOUND = 0.01
SCALE_UPPER_BOUND = 100.0
ISOLATED_RATIO_LOWER = 0.20
ISOLATED_RATIO_UPPER = 5.0

LEGACY_ENRICHMENT_COLUMNS = [
    "sec_cik",
    "sec_sic",
    "sec_sic_office",
    "sec_sic_industry",
    "sec_sic_filed_date",
    "sec_shares_concept",
    "sec_shares_outstanding",
    "sec_shares_period_end",
    "sec_shares_filed_date",
    "sec_shares_age_days",
    "sec_shares_split_factor",
    "sec_shares_outstanding_price_basis",
    "sec_market_cap_estimate",
]
OUTPUT_COLUMNS = ["industry", "market_cap"]


def log(message: str, indent: int = 0) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {'  ' * indent}{message}", flush=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SecClient:
    """Small retrying SEC client that stays below the published request ceiling."""

    def __init__(self, identity: str, min_interval_seconds: float = 0.12) -> None:
        if not identity.strip():
            raise ValueError(
                "SEC access requires --identity or EDGAR_IDENTITY with a name "
                "and contact email. This is a User-Agent, not an API key."
            )
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": identity.strip(),
                "Accept-Encoding": "identity",
            }
        )
        self.min_interval_seconds = min_interval_seconds
        self.last_request_at = 0.0

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: tuple[int, int] = (15, 120),
    ) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(5):
            wait = self.min_interval_seconds - (time.monotonic() - self.last_request_at)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session.get(url, headers=headers, timeout=timeout)
                self.last_request_at = time.monotonic()
                if response.status_code in {403, 429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"SEC returned HTTP {response.status_code}", response=response
                    )
                response.raise_for_status()
                return response
            except (requests.RequestException, OSError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"SEC request failed for {url}: {last_error}") from last_error


class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP file backed by cached Range requests for ZIP member access."""

    def __init__(
        self,
        client: SecClient,
        url: str,
        block_size: int = 2 * 1024 * 1024,
    ) -> None:
        super().__init__()
        self.client = client
        self.url = url
        self.block_size = block_size
        self.position = 0
        self.blocks: dict[int, bytes] = {}

        response = client.get(url, headers={"Range": "bytes=0-0"})
        if response.status_code != 206:
            raise RuntimeError(
                f"{url} did not honor HTTP Range requests (HTTP "
                f"{response.status_code}); refusing to download the full archive."
            )
        content_range = response.headers.get("Content-Range", "")
        match = re.search(r"/(\d+)$", content_range)
        if not match:
            raise RuntimeError(f"Missing archive length in Content-Range: {content_range}")
        self.length = int(match.group(1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if new_position < 0:
            raise ValueError("Negative seek position")
        self.position = min(new_position, self.length)
        return self.position

    def _get_block(self, block_index: int) -> bytes:
        cached = self.blocks.get(block_index)
        if cached is not None:
            return cached
        start = block_index * self.block_size
        end = min(start + self.block_size, self.length) - 1
        response = self.client.get(
            self.url,
            headers={"Range": f"bytes={start}-{end}"},
        )
        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206 for {self.url}, received {response.status_code}"
            )
        expected = end - start + 1
        if len(response.content) != expected:
            raise RuntimeError(
                f"Short Range response for {self.url}: "
                f"{len(response.content):,} of {expected:,} bytes"
            )
        self.blocks[block_index] = response.content
        return response.content

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        remaining = size
        chunks: list[bytes] = []
        while remaining:
            block_index = self.position // self.block_size
            block_offset = self.position % self.block_size
            block = self._get_block(block_index)
            take = min(remaining, len(block) - block_offset)
            chunks.append(block[block_offset : block_offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def atomic_write_parquet(
    table: pa.Table,
    path: Path,
    *,
    compression_level: int = 9,
    row_group_size: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            compression_level=compression_level,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=row_group_size,
        )
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def discover_fsds_archives(client: SecClient) -> list[dict[str, str]]:
    response = client.get(FSDS_PAGE)
    matches = re.findall(
        r"""href=["']([^"']*?(\d{4}q[1-4]\.zip))["']""",
        response.text,
        flags=re.IGNORECASE,
    )
    archives: dict[str, str] = {}
    for href, filename in matches:
        quarter = filename[:-4].lower()
        if int(quarter[:4]) >= 2009:
            archives[quarter] = urljoin(FSDS_PAGE, href)
    if not archives:
        raise RuntimeError(f"No quarterly FSDS archives found at {FSDS_PAGE}")
    result = [
        {"quarter": quarter, "url": archives[quarter]}
        for quarter in sorted(archives)
    ]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_MANIFEST_PATH.write_text(
        json.dumps(
            {
                "source": FSDS_PAGE,
                "retrieved_at_utc": utc_now().isoformat(),
                "archives": result,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def cached_fsds_archives() -> list[dict[str, str]]:
    if not ARCHIVE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Offline mode requires {ARCHIVE_MANIFEST_PATH}. Run once online first."
        )
    manifest = json.loads(ARCHIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    return list(manifest["archives"])


def read_fsds_sub_member(client: SecClient, url: str, quarter: str) -> pd.DataFrame:
    remote_file = HTTPRangeFile(client, url)
    with zipfile.ZipFile(remote_file) as archive:
        matches = [
            name for name in archive.namelist() if Path(name).name.lower() == "sub.txt"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one sub.txt in {quarter}, found {len(matches)}"
            )
        with archive.open(matches[0]) as member:
            frame = pd.read_csv(
                member,
                sep="\t",
                usecols=["adsh", "cik", "name", "sic", "form", "filed", "accepted"],
                dtype={
                    "adsh": "string",
                    "cik": "Int64",
                    "name": "string",
                    "sic": "Int64",
                    "form": "string",
                    "filed": "string",
                    "accepted": "string",
                },
                low_memory=False,
            )
    frame["filed_date"] = pd.to_datetime(
        frame.pop("filed"), format="%Y%m%d", errors="coerce"
    )
    frame["accepted"] = frame["accepted"].str.replace(r"\.0$", "", regex=True)
    frame["quarter"] = quarter
    frame = frame.dropna(subset=["adsh", "cik", "filed_date"]).reset_index(drop=True)
    return frame


def load_fsds_submissions(
    ciks: set[int],
    *,
    client: SecClient | None,
    refresh: bool,
    offline: bool,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    archives = cached_fsds_archives() if offline else discover_fsds_archives(client)
    SUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    selected: list[pd.DataFrame] = []
    for number, item in enumerate(archives, start=1):
        quarter = item["quarter"]
        cache_path = SUB_CACHE_DIR / f"{quarter}_sub.parquet"
        if refresh or not cache_path.exists():
            if offline:
                raise FileNotFoundError(f"Offline mode requires {cache_path}")
            log(
                f"{number}/{len(archives)} SEC {quarter}: reading only sub.txt "
                "with HTTP ranges",
                2,
            )
            frame = read_fsds_sub_member(client, item["url"], quarter)
            atomic_write_parquet(
                pa.Table.from_pandas(frame, preserve_index=False),
                cache_path,
                compression_level=6,
            )
        else:
            log(f"{number}/{len(archives)} SEC {quarter}: cached", 2)
            frame = pq.read_table(cache_path).to_pandas()
        selected.append(frame.loc[frame["cik"].isin(ciks)])
    if not selected:
        raise RuntimeError("No SEC FSDS submission data was loaded")
    return pd.concat(selected, ignore_index=True), archives


def load_sic_code_list(
    *,
    client: SecClient | None,
    refresh: bool,
    offline: bool,
) -> pd.DataFrame:
    if SIC_CODES_PATH.exists() and not refresh:
        return pq.read_table(SIC_CODES_PATH).to_pandas()
    if offline:
        raise FileNotFoundError(f"Offline mode requires {SIC_CODES_PATH}")
    response = client.get(SIC_CODES_URL)
    tables = pd.read_html(io.StringIO(response.text), header=None)
    if not tables:
        raise RuntimeError(f"No SIC table found at {SIC_CODES_URL}")
    codes = tables[0].copy()
    if str(codes.iloc[0, 0]).strip().lower() == "sic code":
        codes = codes.iloc[1:].copy()
    codes.columns = ["sec_sic", "sec_sic_office", "sec_sic_industry"]
    codes["sec_sic"] = pd.to_numeric(codes["sec_sic"], errors="coerce").astype(
        "Int64"
    )
    codes["sec_sic_office"] = codes["sec_sic_office"].astype("string").str.strip()
    codes["sec_sic_industry"] = (
        codes["sec_sic_industry"].astype("string").str.strip()
    )
    codes = codes.dropna(subset=["sec_sic"]).drop_duplicates("sec_sic")
    codes = codes.sort_values("sec_sic", ignore_index=True)
    atomic_write_parquet(
        pa.Table.from_pandas(codes, preserve_index=False),
        SIC_CODES_PATH,
        compression_level=6,
    )
    return codes


def load_ticker_cik_mapping(path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "cik",
        "mapping_valid_from",
        "mapping_valid_to",
        "mapping_source",
        "mapping_confidence",
    ]
    dataset = ds.dataset(path, format="parquet")
    pieces: list[pd.DataFrame] = []
    for batch in dataset.scanner(columns=columns, batch_size=262_144).to_batches():
        pieces.append(batch.to_pandas().drop_duplicates())
    mapping = pd.concat(pieces, ignore_index=True).drop_duplicates()
    duplicates = mapping.groupby("ticker", observed=True)["cik"].nunique()
    if (duplicates > 1).any():
        tickers = duplicates.loc[duplicates > 1].index.tolist()
        raise RuntimeError(f"Ambiguous ticker-CIK mappings: {tickers}")
    return mapping.sort_values("ticker", ignore_index=True)


def load_share_candidates(path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "cik",
        "concept",
        "value",
        "unit",
        "period_type",
        "period_end",
        "filed_date",
        "form_type",
        "accession",
        "mapping_valid_from",
        "mapping_valid_to",
        "is_amendment",
        "filing_version",
    ]
    dataset = ds.dataset(path, format="parquet")
    shares = dataset.to_table(
        columns=columns,
        filter=ds.field("concept").isin(SHARE_CONCEPTS),
    ).to_pandas()
    valid = (
        shares["unit"].str.lower().eq("shares")
        & shares["period_type"].eq("instant")
        & np.isfinite(shares["value"])
        & shares["value"].gt(0)
        & shares["period_end"].le(shares["filed_date"])
        & shares["filed_date"].ge(shares["mapping_valid_from"])
        & shares["filed_date"].le(shares["mapping_valid_to"])
    )
    shares = shares.loc[valid].copy()
    shares["concept_rank"] = shares["concept"].map(SHARE_CONCEPT_RANK)

    # A filing can repeat older balance dates. Keep the newest observation date
    # for each concept, then prefer the DEI cover-page concept over US-GAAP.
    shares = shares.sort_values(
        [
            "ticker",
            "filed_date",
            "concept",
            "period_end",
            "filing_version",
            "accession",
        ]
    ).drop_duplicates(["ticker", "filed_date", "concept"], keep="last")
    shares = shares.sort_values(
        ["ticker", "filed_date", "concept_rank"]
    ).drop_duplicates(["ticker", "filed_date"], keep="last")
    return shares.reset_index(drop=True)


def future_split_factors(
    ticker: str,
    filed_dates: pd.Series,
    market: pd.DataFrame,
) -> np.ndarray:
    splits = market.loc[
        market["ticker"].eq(ticker)
        & market["stock_splits"].gt(0)
        & market["stock_splits"].ne(1),
        ["date", "stock_splits"],
    ].sort_values("date")
    if splits.empty:
        return np.ones(len(filed_dates), dtype="float64")
    split_dates = splits["date"].to_numpy(dtype="datetime64[ns]")
    split_values = splits["stock_splits"].to_numpy(dtype="float64")
    suffix_products = np.cumprod(split_values[::-1])[::-1]
    positions = split_dates.searchsorted(
        filed_dates.to_numpy(dtype="datetime64[ns]"), side="right"
    )
    factors = np.ones(len(positions), dtype="float64")
    has_future_split = positions < len(split_values)
    factors[has_future_split] = suffix_products[positions[has_future_split]]
    return factors


def next_available_close(
    ticker: str,
    filed_dates: pd.Series,
    market: pd.DataFrame,
) -> np.ndarray:
    prices = market.loc[
        market["ticker"].eq(ticker), ["date", "close"]
    ].sort_values("date")
    dates = prices["date"].to_numpy(dtype="datetime64[ns]")
    closes = prices["close"].to_numpy(dtype="float64")
    positions = dates.searchsorted(
        filed_dates.to_numpy(dtype="datetime64[ns]"), side="left"
    )
    result = np.full(len(positions), np.nan, dtype="float64")
    available = positions < len(closes)
    result[available] = closes[positions[available]]
    return result


def clean_share_events(
    candidates: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    shares = candidates.copy()
    shares["sec_shares_split_factor"] = 1.0
    shares["reference_close"] = np.nan
    for ticker, indices in shares.groupby("ticker", observed=True).groups.items():
        shares.loc[indices, "sec_shares_split_factor"] = future_split_factors(
            ticker, shares.loc[indices, "filed_date"], market
        )
        shares.loc[indices, "reference_close"] = next_available_close(
            ticker, shares.loc[indices, "filed_date"], market
        )

    shares["sec_shares_outstanding_price_basis"] = (
        shares["value"] * shares["sec_shares_split_factor"]
    )
    shares["reference_market_cap"] = (
        shares["reference_close"]
        * shares["sec_shares_outstanding_price_basis"]
    )
    reference_cap_ok = shares["reference_market_cap"].between(
        MIN_REFERENCE_MARKET_CAP, MAX_REFERENCE_MARKET_CAP
    )

    ticker_median = (
        shares.loc[reference_cap_ok]
        .groupby("ticker", observed=True)["sec_shares_outstanding_price_basis"]
        .median()
    )
    relative_scale = shares["sec_shares_outstanding_price_basis"] / shares[
        "ticker"
    ].map(ticker_median)
    scale_ok = relative_scale.between(SCALE_LOWER_BOUND, SCALE_UPPER_BOUND)
    keep = reference_cap_ok & scale_ok
    cleaned = shares.loc[keep].sort_values(["ticker", "filed_date"]).copy()

    previous = cleaned.groupby("ticker", observed=True)[
        "sec_shares_outstanding_price_basis"
    ].shift(1)
    following = cleaned.groupby("ticker", observed=True)[
        "sec_shares_outstanding_price_basis"
    ].shift(-1)
    current = cleaned["sec_shares_outstanding_price_basis"]
    current_vs_previous = current / previous
    current_vs_following = current / following
    following_vs_previous = following / previous
    isolated = (
        previous.notna()
        & following.notna()
        & (
            current_vs_previous.lt(ISOLATED_RATIO_LOWER)
            | current_vs_previous.gt(ISOLATED_RATIO_UPPER)
        )
        & (
            current_vs_following.lt(ISOLATED_RATIO_LOWER)
            | current_vs_following.gt(ISOLATED_RATIO_UPPER)
        )
        & following_vs_previous.between(
            ISOLATED_RATIO_LOWER, ISOLATED_RATIO_UPPER
        )
    )
    cleaned = cleaned.loc[~isolated].copy()

    cleaned["change_ratio"] = cleaned.groupby("ticker", observed=True)[
        "sec_shares_outstanding_price_basis"
    ].pct_change(fill_method=None) + 1
    unresolved_extreme_changes = int(
        (
            cleaned["change_ratio"].lt(ISOLATED_RATIO_LOWER)
            | cleaned["change_ratio"].gt(ISOLATED_RATIO_UPPER)
        ).sum()
    )
    diagnostics = {
        "raw_selected_share_events": int(len(shares)),
        "raw_selected_share_tickers": int(shares["ticker"].nunique()),
        "reference_market_cap_outliers_removed": int((~reference_cap_ok).sum()),
        "within_ticker_scale_outliers_removed": int(
            (reference_cap_ok & ~scale_ok).sum()
        ),
        "isolated_share_spikes_removed": int(isolated.sum()),
        "clean_share_events": int(len(cleaned)),
        "clean_share_tickers": int(cleaned["ticker"].nunique()),
        "unresolved_outside_0_2x_to_5x_share_changes": (
            unresolved_extreme_changes
        ),
        "share_concept_counts": {
            str(key): int(value)
            for key, value in cleaned["concept"].value_counts().items()
        },
    }
    columns = [
        "ticker",
        "cik",
        "concept",
        "value",
        "period_end",
        "filed_date",
        "sec_shares_split_factor",
        "sec_shares_outstanding_price_basis",
    ]
    return cleaned[columns].sort_values(
        ["ticker", "filed_date"], ignore_index=True
    ), diagnostics


def build_sic_events(
    submissions: pd.DataFrame,
    codes: pd.DataFrame,
) -> pd.DataFrame:
    events = submissions.dropna(subset=["cik", "sic", "filed_date"]).copy()
    events["accepted_sort"] = events["accepted"].fillna("")
    events = events.sort_values(
        ["cik", "filed_date", "accepted_sort", "adsh"]
    ).drop_duplicates(["cik", "filed_date"], keep="last")
    events = events.rename(
        columns={"sic": "sec_sic", "filed_date": "sec_sic_filed_date"}
    )
    events = events.merge(codes, on="sec_sic", how="left", validate="many_to_one")
    return events[
        [
            "cik",
            "sec_sic",
            "sec_sic_office",
            "sec_sic_industry",
            "sec_sic_filed_date",
        ]
    ].sort_values(["cik", "sec_sic_filed_date"], ignore_index=True)


def assign_cik(
    market: pd.DataFrame,
    mapping: pd.DataFrame,
) -> np.ndarray:
    by_ticker = mapping.set_index("ticker")
    ciks = market["ticker"].map(by_ticker["cik"]).astype("Float64")
    valid_from = market["ticker"].map(by_ticker["mapping_valid_from"])
    valid_to = market["ticker"].map(by_ticker["mapping_valid_to"])
    valid = market["date"].ge(valid_from) & market["date"].le(valid_to)
    result = ciks.to_numpy(dtype="float64", na_value=np.nan)
    result[~valid.to_numpy()] = np.nan
    return result


def empty_enrichment_arrays(row_count: int) -> dict[str, np.ndarray]:
    missing_dates = np.full(row_count, np.datetime64("NaT"), dtype="datetime64[ns]")
    return {
        "sec_cik": np.full(row_count, np.nan, dtype="float64"),
        "sec_sic": np.full(row_count, np.nan, dtype="float64"),
        "sec_sic_office": np.full(row_count, None, dtype="object"),
        "sec_sic_industry": np.full(row_count, None, dtype="object"),
        "sec_sic_filed_date": missing_dates.copy(),
        "sec_shares_concept": np.full(row_count, None, dtype="object"),
        "sec_shares_outstanding": np.full(row_count, np.nan, dtype="float64"),
        "sec_shares_period_end": missing_dates.copy(),
        "sec_shares_filed_date": missing_dates.copy(),
        "sec_shares_age_days": np.full(row_count, np.nan, dtype="float64"),
        "sec_shares_split_factor": np.full(row_count, np.nan, dtype="float64"),
        "sec_shares_outstanding_price_basis": np.full(
            row_count, np.nan, dtype="float64"
        ),
        "sec_market_cap_estimate": np.full(row_count, np.nan, dtype="float64"),
    }


def assign_point_in_time_enrichment(
    market: pd.DataFrame,
    mapping: pd.DataFrame,
    shares: pd.DataFrame,
    sic_events: pd.DataFrame,
    *,
    max_share_age_days: int,
) -> dict[str, np.ndarray]:
    arrays = empty_enrichment_arrays(len(market))
    arrays["sec_cik"] = assign_cik(market, mapping)

    # Strictly earlier filing dates make each value safe for any trading time on
    # the labeled market date, including the market open.
    for ticker, positions in market.groupby(
        "ticker", observed=True, sort=False
    ).indices.items():
        events = shares.loc[shares["ticker"].eq(ticker)]
        if events.empty:
            continue
        positions = np.asarray(positions)
        dates = market.iloc[positions]["date"].to_numpy(dtype="datetime64[ns]")
        filed = events["filed_date"].to_numpy(dtype="datetime64[ns]")
        event_positions = filed.searchsorted(dates, side="left") - 1
        has_event = event_positions >= 0
        if not has_event.any():
            continue
        target_positions = positions[has_event]
        source_positions = event_positions[has_event]
        age_days = (
            dates[has_event] - filed[source_positions]
        ) / np.timedelta64(1, "D")
        matching_cik = (
            arrays["sec_cik"][target_positions]
            == events["cik"].to_numpy(dtype="float64")[source_positions]
        )
        usable = (age_days <= max_share_age_days) & matching_cik
        target_positions = target_positions[usable]
        source_positions = source_positions[usable]
        age_days = age_days[usable]
        if not len(target_positions):
            continue
        arrays["sec_shares_concept"][target_positions] = events[
            "concept"
        ].to_numpy(dtype="object")[source_positions]
        arrays["sec_shares_outstanding"][target_positions] = events[
            "value"
        ].to_numpy(dtype="float64")[source_positions]
        arrays["sec_shares_period_end"][target_positions] = events[
            "period_end"
        ].to_numpy(dtype="datetime64[ns]")[source_positions]
        arrays["sec_shares_filed_date"][target_positions] = filed[source_positions]
        arrays["sec_shares_age_days"][target_positions] = age_days
        arrays["sec_shares_split_factor"][target_positions] = events[
            "sec_shares_split_factor"
        ].to_numpy(dtype="float64")[source_positions]
        arrays["sec_shares_outstanding_price_basis"][target_positions] = events[
            "sec_shares_outstanding_price_basis"
        ].to_numpy(dtype="float64")[source_positions]

    arrays["sec_market_cap_estimate"] = (
        market["close"].to_numpy(dtype="float64")
        * arrays["sec_shares_outstanding_price_basis"]
    )

    valid_cik_positions = np.flatnonzero(np.isfinite(arrays["sec_cik"]))
    if len(valid_cik_positions):
        cik_values = arrays["sec_cik"].astype("float64")
        for cik in np.unique(cik_values[valid_cik_positions]).astype("int64"):
            positions = np.flatnonzero(cik_values == cik)
            events = sic_events.loc[sic_events["cik"].eq(cik)]
            if events.empty:
                continue
            dates = market.iloc[positions]["date"].to_numpy(dtype="datetime64[ns]")
            filed = events["sec_sic_filed_date"].to_numpy(dtype="datetime64[ns]")
            event_positions = filed.searchsorted(dates, side="left") - 1
            usable = event_positions >= 0
            target_positions = positions[usable]
            source_positions = event_positions[usable]
            arrays["sec_sic"][target_positions] = events["sec_sic"].to_numpy(
                dtype="float64", na_value=np.nan
            )[source_positions]
            arrays["sec_sic_office"][target_positions] = events[
                "sec_sic_office"
            ].to_numpy(dtype="object")[source_positions]
            arrays["sec_sic_industry"][target_positions] = events[
                "sec_sic_industry"
            ].to_numpy(dtype="object")[source_positions]
            arrays["sec_sic_filed_date"][target_positions] = filed[source_positions]
    return arrays


def arrays_to_arrow(arrays: dict[str, np.ndarray]) -> list[tuple[str, pa.Array]]:
    return [
        (
            "industry",
            pa.array(arrays["sec_sic_industry"], type=pa.string()),
        ),
        (
            "market_cap",
            pa.array(arrays["sec_market_cap_estimate"], type=pa.float64()),
        ),
    ]


def validate_enrichment(
    market: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    share_diagnostics: dict[str, Any],
    *,
    max_share_age_days: int,
) -> dict[str, Any]:
    dates = market["date"].to_numpy(dtype="datetime64[ns]")
    members = market["is_sp500_member"].to_numpy(dtype=bool)
    shares_available = np.isfinite(arrays["sec_shares_outstanding"])
    sic_available = np.isfinite(arrays["sec_sic"])
    cik_available = np.isfinite(arrays["sec_cik"])
    cap_available = np.isfinite(arrays["sec_market_cap_estimate"])
    recent_cutoff = market["date"].max() - pd.DateOffset(years=2)
    recent_member = members & market["date"].ge(recent_cutoff).to_numpy()
    since_2012_member = members & market["date"].ge("2012-01-01").to_numpy()

    share_filed = arrays["sec_shares_filed_date"]
    sic_filed = arrays["sec_sic_filed_date"]
    same_day_or_future_share_filings = int(
        (shares_available & (share_filed >= dates)).sum()
    )
    same_day_or_future_sic_filings = int(
        (sic_available & (sic_filed >= dates)).sum()
    )
    invalid_share_age = int(
        (
            shares_available
            & (
                (arrays["sec_shares_age_days"] < 0)
                | (arrays["sec_shares_age_days"] > max_share_age_days)
            )
        ).sum()
    )
    invalid_share_values = int(
        (
            shares_available
            & (
                (arrays["sec_shares_outstanding"] <= 0)
                | (arrays["sec_shares_outstanding_price_basis"] <= 0)
                | (arrays["sec_shares_split_factor"] <= 0)
            )
        ).sum()
    )
    recomputed_cap = (
        market["close"].to_numpy(dtype="float64")
        * arrays["sec_shares_outstanding_price_basis"]
    )
    inconsistent_cap = int(
        (
            cap_available
            & ~np.isclose(
                arrays["sec_market_cap_estimate"],
                recomputed_cap,
                rtol=1e-12,
                atol=0,
                equal_nan=True,
            )
        ).sum()
    )

    recent_share_coverage = float(shares_available[recent_member].mean())
    recent_sic_coverage = float(sic_available[recent_member].mean())
    recent_cik_coverage = float(cik_available[recent_member].mean())
    since_2012_share_coverage = float(shares_available[since_2012_member].mean())
    failures: list[str] = []
    if same_day_or_future_share_filings:
        failures.append(
            f"{same_day_or_future_share_filings} share values were not filed "
            "strictly before the market date"
        )
    if same_day_or_future_sic_filings:
        failures.append(
            f"{same_day_or_future_sic_filings} SIC values were not filed "
            "strictly before the market date"
        )
    if invalid_share_age:
        failures.append(f"{invalid_share_age} share values violated the age limit")
    if invalid_share_values:
        failures.append(f"{invalid_share_values} share values were nonpositive")
    if inconsistent_cap:
        failures.append(f"{inconsistent_cap} market caps failed recomputation")
    if recent_cik_coverage < 0.95:
        failures.append(
            f"Recent member CIK coverage is below 95% ({recent_cik_coverage:.2%})"
        )
    if recent_sic_coverage < 0.90:
        failures.append(
            f"Recent member SIC coverage is below 90% ({recent_sic_coverage:.2%})"
        )
    if recent_share_coverage < 0.85:
        failures.append(
            "Recent member shares/market-cap coverage is below 85% "
            f"({recent_share_coverage:.2%})"
        )

    valid_caps = arrays["sec_market_cap_estimate"][cap_available]
    result = {
        "status": "pass" if not failures else "fail",
        "rows": int(len(market)),
        "max_share_age_days": int(max_share_age_days),
        "point_in_time_availability_rule": "source filed date < market date",
        "rows_with_cik": int(cik_available.sum()),
        "rows_with_sic": int(sic_available.sum()),
        "rows_with_shares_and_market_cap": int(shares_available.sum()),
        "tickers_with_sic": int(market.loc[sic_available, "ticker"].nunique()),
        "tickers_with_shares_and_market_cap": int(
            market.loc[shares_available, "ticker"].nunique()
        ),
        "recent_member_cik_coverage": round(recent_cik_coverage, 6),
        "recent_member_sic_coverage": round(recent_sic_coverage, 6),
        "recent_member_share_market_cap_coverage": round(
            recent_share_coverage, 6
        ),
        "since_2012_member_share_market_cap_coverage": round(
            since_2012_share_coverage, 6
        ),
        "same_day_or_future_share_filings": same_day_or_future_share_filings,
        "same_day_or_future_sic_filings": same_day_or_future_sic_filings,
        "invalid_share_age_rows": invalid_share_age,
        "invalid_share_value_rows": invalid_share_values,
        "inconsistent_market_cap_rows": inconsistent_cap,
        "market_cap_min": float(np.min(valid_caps)) if len(valid_caps) else None,
        "market_cap_median": (
            float(np.median(valid_caps)) if len(valid_caps) else None
        ),
        "market_cap_max": float(np.max(valid_caps)) if len(valid_caps) else None,
        "share_event_cleaning": share_diagnostics,
        "failures": failures,
    }
    return result


def enrich_market_file(
    *,
    market_path: Path,
    fundamentals_path: Path,
    output_path: Path,
    identity: str,
    refresh_sic: bool,
    offline: bool,
    max_share_age_days: int,
    validate_only: bool,
) -> dict[str, Any]:
    if not market_path.exists():
        raise FileNotFoundError(market_path)
    if not fundamentals_path.exists():
        raise FileNotFoundError(
            f"{fundamentals_path} is required because it already contains "
            "the point-in-time SEC share facts and ticker-CIK mapping."
        )
    if max_share_age_days <= 0:
        raise ValueError("--max-share-age-days must be positive")

    log("1/6 Load existing market rows and SEC ticker-CIK mapping")
    market_table = pq.read_table(market_path)
    existing_enrichment = [
        column
        for column in LEGACY_ENRICHMENT_COLUMNS + OUTPUT_COLUMNS
        if column in market_table.column_names
    ]
    if existing_enrichment:
        market_table = market_table.drop(existing_enrichment)
    required_market_columns = [
        "date",
        "ticker",
        "close",
        "stock_splits",
        "is_sp500_member",
    ]
    market = market_table.select(required_market_columns).to_pandas()
    mapping = load_ticker_cik_mapping(fundamentals_path)
    log(
        f"{len(market):,} unchanged market rows; "
        f"{len(mapping):,} ticker-CIK mappings",
        1,
    )

    log("2/6 Select and clean existing SEC shares facts")
    share_candidates = load_share_candidates(fundamentals_path)
    share_events, share_diagnostics = clean_share_events(share_candidates, market)
    log(json.dumps(share_diagnostics, indent=2, sort_keys=True), 1)

    log("3/6 Load point-in-time SEC filing SIC history")
    client = None if offline else SecClient(identity)
    submissions, archives = load_fsds_submissions(
        set(mapping["cik"].astype(int)),
        client=client,
        refresh=refresh_sic,
        offline=offline,
    )
    codes = load_sic_code_list(
        client=client,
        refresh=refresh_sic,
        offline=offline,
    )
    sic_events = build_sic_events(submissions, codes)
    log(
        f"{len(sic_events):,} CIK/filing-date SIC observations from "
        f"{archives[0]['quarter']} through {archives[-1]['quarter']}",
        1,
    )

    log("4/6 Assign only information filed before each market date")
    arrays = assign_point_in_time_enrichment(
        market,
        mapping,
        share_events,
        sic_events,
        max_share_age_days=max_share_age_days,
    )

    log("5/6 Validate enrichment")
    validation = validate_enrichment(
        market,
        arrays,
        share_diagnostics,
        max_share_age_days=max_share_age_days,
    )
    log(json.dumps(validation, indent=2, sort_keys=True), 1)
    if validation["status"] != "pass":
        raise RuntimeError(
            "SEC enrichment validation failed; parquet was not replaced. "
            + "; ".join(validation["failures"])
        )
    if validate_only:
        log("Validation passed; --validate-only left the parquet unchanged.")
        return validation

    log("6/6 Atomically append columns to the market parquet")
    enriched_table = market_table
    for name, column in arrays_to_arrow(arrays):
        enriched_table = enriched_table.append_column(name, column)
    metadata = dict(enriched_table.schema.metadata or {})
    additions = {
        "schema_version": "1.1.0",
        "sec_enriched_at_utc": utc_now().isoformat(),
        "sec_enrichment_sources": json.dumps(
            {
                "company_facts_bulk": COMPANY_FACTS_BULK_URL,
                "financial_statement_data_sets": FSDS_PAGE,
                "financial_statement_data_sets_documentation": FSDS_DOCUMENTATION,
                "sic_code_list": SIC_CODES_URL,
            },
            sort_keys=True,
        ),
        "sec_enrichment_method": (
            "CIK/SIC and shares are assigned only when source filed_date is "
            "strictly earlier than market date. Shares older than "
            f"{max_share_age_days} days are null. Issuer-reported shares are "
            "normalized for Yahoo's current split basis before close-times-shares."
        ),
        "sec_enrichment_validation": json.dumps(validation, sort_keys=True),
        "sec_identity": (
            "SEC identifying User-Agent configured at runtime; value not persisted"
        ),
    }
    existing_notes = metadata.get(b"notes", b"").decode(errors="replace")
    additions["notes"] = (
        existing_notes
        + " SEC enrichment is point-in-time and filing-derived. "
        "market_cap is an issuer-level estimate, not vendor float market cap; "
        "industry is the filing-date SEC SIC title. See "
        "data/sp500_market/README.md."
    ).strip()
    metadata.update({key.encode(): value.encode() for key, value in additions.items()})
    enriched_table = enriched_table.replace_schema_metadata(metadata)
    atomic_write_parquet(
        enriched_table,
        output_path,
        compression_level=9,
        row_group_size=250_000,
    )
    written = pq.ParquetFile(output_path)
    if written.metadata.num_rows != len(market):
        raise RuntimeError("Written parquet row count changed unexpectedly")
    log(
        f"Wrote {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB, "
        f"{written.metadata.num_rows:,} rows x "
        f"{len(written.schema_arrow.names)} columns)"
    )
    return validation
