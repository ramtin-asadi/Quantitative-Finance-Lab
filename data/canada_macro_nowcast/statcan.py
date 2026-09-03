from __future__ import annotations

import hashlib
import itertools
import json
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import STATCAN_CURRENT, STATCAN_REALTIME

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "statcan"
STATE = CACHE / "sources.json"
REALTIME_OUTPUT = DATA / "canada_statscan_realtime.parquet"
CURRENT_OUTPUT = DATA / "canada_statscan_current.parquet"
WDS = "https://www150.statcan.gc.ca/t1/wds/rest"
_METADATA_CACHE: dict[str, dict | None] = {}
_HIERARCHY_CACHE: dict[tuple[str, str, int], set[str] | None] = {}
_STATE_LOCK = threading.Lock()

SCHEMA = pa.schema(
    [
        pa.field("table_id", pa.string(), nullable=False),
        pa.field("table_title", pa.string(), nullable=False),
        pa.field("source_kind", pa.string(), nullable=False),
        pa.field("reference_date", pa.timestamp("ns"), nullable=False),
        pa.field("release_date", pa.timestamp("ns")),
        pa.field("snapshot_date", pa.timestamp("ns")),
        pa.field("vector", pa.string()),
        pa.field("coordinate", pa.string()),
        pa.field("series_title", pa.string(), nullable=False),
        pa.field("geography", pa.string()),
        pa.field("seasonal_adjustment", pa.string()),
        pa.field("prices", pa.string()),
        pa.field("estimate", pa.string()),
        pa.field("industry", pa.string()),
        pa.field("product", pa.string()),
        pa.field("trade", pa.string()),
        pa.field("basis", pa.string()),
        pa.field("principal_statistic", pa.string()),
        pa.field("alternative_measure", pa.string()),
        pa.field("sales_measure", pa.string()),
        pa.field("adjustment", pa.string()),
        pa.field("account_flow", pa.string()),
        pa.field("current_account", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("unit_id", pa.int32()),
        pa.field("scalar_factor", pa.string()),
        pa.field("scalar_id", pa.int16()),
        pa.field("value", pa.float64(), nullable=False),
        pa.field("status", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("terminated", pa.int8()),
        pa.field("decimals", pa.int8()),
        pa.field("source_file", pa.string(), nullable=False),
    ]
)

FIXED_COLUMNS = {
    "REF_DATE",
    "GEO",
    "DGUID",
    "UOM",
    "UOM_ID",
    "SCALAR_FACTOR",
    "SCALAR_ID",
    "VECTOR",
    "COORDINATE",
    "VALUE",
    "STATUS",
    "SYMBOL",
    "TERMINATED",
    "DECIMALS",
    "Release",
}


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    client = requests.Session()
    client.headers["User-Agent"] = "Quantitative-Finance-Lab Canada nowcast builder"
    client.mount("https://", HTTPAdapter(max_retries=retry))
    return client


def load_state() -> dict[str, dict]:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state: dict[str, dict]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def metadata_path(pid: str) -> Path:
    return CACHE / f"{pid}-metadata.json"


def get_metadata(config: dict, *, refresh: bool = False) -> dict | None:
    if config["pid"] in _METADATA_CACHE and not refresh:
        return _METADATA_CACHE[config["pid"]]
    path = metadata_path(config["pid"])
    if path.exists() and not refresh:
        payload = json.loads(path.read_text())
        result = payload if payload.get("productId") else None
        _METADATA_CACHE[config["pid"]] = result
        return result
    response = make_session().post(
        f"{WDS}/getCubeMetadata",
        json=[{"productId": int(config["pid"])}],
        timeout=120,
    )
    response.raise_for_status()
    item = response.json()[0]
    payload = (
        item.get("object") if item.get("status") == "SUCCESS" else {"error": item.get("object")}
    )
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    result = payload if isinstance(payload, dict) and payload.get("productId") else None
    _METADATA_CACHE[config["pid"]] = result
    return result


def archive_path(pid: str) -> Path:
    return CACHE / f"{pid}-eng.zip"


def valid_archive(path: Path) -> bool:
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any(
            Path(name).name.lower() == f"{path.stem.removesuffix('-eng')}.csv"
            for name in archive.namelist()
        )


def download_archive(config: dict, *, force: bool = False) -> Path:
    pid = config["pid"]
    path = archive_path(pid)
    if path.exists() and valid_archive(path) and not force:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    url = f"https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip"
    response = make_session().get(url, stream=True, timeout=(30, 1200))
    response.raise_for_status()
    expected = int(response.headers.get("Content-Length", 0) or 0)
    temporary = path.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    received = 0
    with temporary.open("wb") as output:
        for block in response.iter_content(chunk_size=1024 * 1024):
            if not block:
                continue
            output.write(block)
            digest.update(block)
            received += len(block)
            if expected and received // (50 * 1024 * 1024) != (received - len(block)) // (
                50 * 1024 * 1024
            ):
                print(
                    f"StatsCan {config['table_id']}: {received / 1e6:.0f}/{expected / 1e6:.0f} MB"
                )
    if expected and received != expected:
        temporary.unlink(missing_ok=True)
        raise IOError(
            f"Incomplete StatsCan archive for {config['table_id']}: {received} of {expected} bytes"
        )
    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Invalid StatsCan ZIP for {config['table_id']}")
    with zipfile.ZipFile(temporary) as archive:
        expected_member = f"{pid}.csv"
        member_exists = any(Path(name).name == expected_member for name in archive.namelist())
    if not member_exists:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"StatsCan ZIP lacks {expected_member} for {config['table_id']}")
    temporary.replace(path)
    with _STATE_LOCK:
        state = load_state()
        state[pid] = {
            "table_id": config["table_id"],
            "url": url,
            "bytes": received,
            "sha256": digest.hexdigest(),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
    return path


def dimension(config: dict, prefix: str) -> dict | None:
    metadata = get_metadata(config)
    if not metadata:
        return None
    prefix = prefix.casefold()
    return next(
        (
            item
            for item in metadata.get("dimension", [])
            if item.get("dimensionNameEn", "").casefold().startswith(prefix)
        ),
        None,
    )


def hierarchy_members(config: dict, prefix: str, max_depth: int) -> set[str] | None:
    key = (config["pid"], prefix, max_depth)
    if key in _HIERARCHY_CACHE:
        return _HIERARCHY_CACHE[key]
    item = dimension(config, prefix)
    if not item:
        _HIERARCHY_CACHE[key] = None
        return None
    members = item.get("member", [])
    parents = {int(member["memberId"]): member.get("parentMemberId") for member in members}

    def depth(member_id: int) -> int:
        result = 0
        seen = set()
        parent = parents.get(member_id)
        while parent not in (None, "", 0, "0"):
            parent = int(parent)
            if parent in seen or parent not in parents:
                break
            seen.add(parent)
            result += 1
            parent = parents.get(parent)
        return result

    result = {
        member["memberNameEn"] for member in members if depth(int(member["memberId"])) <= max_depth
    }
    _HIERARCHY_CACHE[key] = result
    return result


def match_column(columns: list[str], name: str) -> str | None:
    folded = name.casefold()
    if folded == "geography" and "GEO" in columns:
        return "GEO"
    return next((column for column in columns if column.casefold().startswith(folded)), None)


def filter_chunk(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    columns = list(frame.columns)
    for requested, values in config.get("equals", {}).items():
        column = match_column(columns, requested)
        if column:
            keep &= frame[column].isin(values)
    for requested, pattern in config.get("regex", {}).items():
        column = match_column(columns, requested)
        if column:
            keep &= frame[column].astype("string").str.contains(pattern, regex=True, na=False)
    for requested, max_depth in config.get("hierarchy", {}).items():
        column = match_column(columns, requested)
        allowed = hierarchy_members(config, requested, max_depth)
        if column and allowed:
            member_name = (
                frame[column].astype("string").str.replace(r"\s+\[[^\]]+\]$", "", regex=True)
            )
            keep &= member_name.isin(allowed)
    return frame.loc[keep].copy()


def semantic_column(name: str) -> str | None:
    folded = name.casefold()
    if name == "GEO":
        return "geography"
    if folded == "seasonal adjustment":
        return "seasonal_adjustment"
    if folded == "prices":
        return "prices"
    if folded in {"estimate", "estimates"}:
        return "estimate"
    if folded.startswith("north american industry classification system"):
        return "industry"
    if (
        folded.startswith("north american product classification system")
        or folded == "products and product groups"
    ):
        return "product"
    if folded == "trade":
        return "trade"
    if folded == "basis":
        return "basis"
    if folded == "principal statistics":
        return "principal_statistic"
    if folded == "alternative measures":
        return "alternative_measure"
    if folded in {"sales", "sales, price and volume"}:
        return "sales_measure"
    if folded == "adjustments":
        return "adjustment"
    if folded == "receipts, payments and balances":
        return "account_flow"
    if folded == "current account":
        return "current_account"
    return None


def text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


def numeric_series(frame: pd.DataFrame, column: str, dtype: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype=dtype)
    return pd.to_numeric(frame[column], errors="coerce").astype(dtype)


def normalize(
    frame: pd.DataFrame,
    config: dict,
    *,
    source_file: str,
    snapshot_date: pd.Timestamp | None = None,
    source_release_date: pd.Timestamp | None = None,
) -> pa.Table | None:
    frame = filter_chunk(frame, config)
    frame["VALUE"] = pd.to_numeric(frame.get("VALUE"), errors="coerce")
    frame["REF_DATE"] = pd.to_datetime(frame.get("REF_DATE"), errors="coerce")
    frame = frame.dropna(subset=["VALUE", "REF_DATE"])
    if frame.empty:
        return None
    dimensions = [
        column
        for column in frame.columns
        if column not in FIXED_COLUMNS and not column.startswith("Unnamed:")
    ]
    semantic = {column: semantic_column(column) for column in dimensions + ["GEO"]}
    title_columns = [column for column in ["GEO", *dimensions] if column in frame]
    title_parts = frame[title_columns].fillna("").astype("string")
    title = title_parts.iloc[:, 0]
    for column in title_parts.columns[1:]:
        title = title.str.cat(title_parts[column], sep=" | ")
    title = title.str.replace(r"(?: \| )+\Z", "", regex=True)
    output = pd.DataFrame(index=frame.index)
    output["table_id"] = config["table_id"]
    output["table_title"] = config["title"]
    output["source_kind"] = config.get(
        "source_kind",
        "archived_realtime_vintage" if config.get("archived") else "realtime_vintage",
    )
    output["reference_date"] = frame["REF_DATE"]
    if "Release" in frame:
        output["release_date"] = pd.to_datetime(frame["Release"], errors="coerce")
    else:
        output["release_date"] = source_release_date
    output["snapshot_date"] = snapshot_date
    output["vector"] = text_series(frame, "VECTOR")
    output["coordinate"] = text_series(frame, "COORDINATE")
    output["series_title"] = title
    for target in (
        "geography",
        "seasonal_adjustment",
        "prices",
        "estimate",
        "industry",
        "product",
        "trade",
        "basis",
        "principal_statistic",
        "alternative_measure",
        "sales_measure",
        "adjustment",
        "account_flow",
        "current_account",
    ):
        source = next((name for name, mapped in semantic.items() if mapped == target), None)
        output[target] = text_series(frame, source) if source else pd.NA
    output["unit"] = text_series(frame, "UOM")
    output["unit_id"] = numeric_series(frame, "UOM_ID", "Int32")
    output["scalar_factor"] = text_series(frame, "SCALAR_FACTOR")
    output["scalar_id"] = numeric_series(frame, "SCALAR_ID", "Int16")
    output["value"] = frame["VALUE"].astype(float)
    output["status"] = text_series(frame, "STATUS")
    output["symbol"] = text_series(frame, "SYMBOL")
    output["terminated"] = numeric_series(frame, "TERMINATED", "Int8")
    output["decimals"] = numeric_series(frame, "DECIMALS", "Int8")
    output["source_file"] = source_file
    return pa.Table.from_pandas(output, schema=SCHEMA, preserve_index=False, safe=False)


def data_member(path: Path, pid: str) -> str:
    with zipfile.ZipFile(path) as archive:
        exact = f"{pid}.csv"
        return next(name for name in archive.namelist() if Path(name).name.lower() == exact.lower())


def tables_from_archive(
    config: dict,
    *,
    snapshot_date: pd.Timestamp | None = None,
    source_release_date: pd.Timestamp | None = None,
) -> Iterator[pa.Table]:
    path = archive_path(config["pid"])
    member = data_member(path, config["pid"])
    with zipfile.ZipFile(path) as archive, archive.open(member) as source:
        chunks = pd.read_csv(source, chunksize=300_000, low_memory=False)
        for chunk in chunks:
            table = normalize(
                chunk,
                config,
                source_file=path.name,
                snapshot_date=snapshot_date,
                source_release_date=source_release_date,
            )
            if table is not None and table.num_rows:
                yield table


def metadata_timestamp(config: dict) -> pd.Timestamp | None:
    metadata = get_metadata(config, refresh=True)
    if not metadata:
        return None
    return pd.to_datetime(metadata.get("releaseTime"), errors="coerce")


def output_metadata(name: str) -> dict[bytes, bytes]:
    return {
        b"dataset": name.encode(),
        b"source_url": b"https://www150.statcan.gc.ca/t1/wds/rest",
        b"generated_at_utc": datetime.now(timezone.utc).isoformat().encode(),
        b"timing": (
            b"release_date is the StatsCan Release dimension; snapshot_date is only "
            b"used for the post-archive retail snapshot series"
        ),
    }


def build_realtime(*, force_download: bool = False) -> None:
    for config in STATCAN_REALTIME:
        get_metadata(config)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda config: download_archive(config, force=force_download),
                STATCAN_REALTIME,
            )
        )
    temporary = REALTIME_OUTPUT.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA.with_metadata(
            output_metadata("Selected Statistics Canada real-time macro vintages")
        ),
        compression="zstd",
    )
    rows = 0
    try:
        for config in STATCAN_REALTIME:
            table_rows = 0
            for table in tables_from_archive(config):
                writer.write_table(table, row_group_size=250_000)
                rows += table.num_rows
                table_rows += table.num_rows
            print(f"StatsCan {config['table_id']}: kept {table_rows:,} rows")
    finally:
        writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Statistics Canada real-time build produced no rows.")
    temporary.replace(REALTIME_OUTPUT)
    print(
        f"wrote {REALTIME_OUTPUT} rows={rows:,} "
        f"size_mb={REALTIME_OUTPUT.stat().st_size / 1e6:.1f}"
    )


def build_current(*, force_download: bool = False) -> None:
    snapshot_date = pd.Timestamp.today().normalize()
    release_dates = {}
    for config in STATCAN_CURRENT:
        release_dates[config["pid"]] = metadata_timestamp(config)
    preserved: list[pa.RecordBatch] = []
    existing_releases: dict[str, set[pd.Timestamp]] = {}
    if CURRENT_OUTPUT.exists():
        old = pq.ParquetFile(CURRENT_OUTPUT)
        for batch in old.iter_batches(batch_size=250_000):
            table = pa.Table.from_batches([batch])
            kinds = table["source_kind"].to_pylist()
            table_ids = table["table_id"].to_pylist()
            releases = table["release_date"].to_pylist()
            for kind, table_id, release in zip(kinds, table_ids, releases):
                if kind == "forward_snapshot" and release is not None:
                    existing_releases.setdefault(table_id, set()).add(pd.Timestamp(release))
            keep = pa.array([kind == "forward_snapshot" for kind in kinds])
            filtered = table.filter(keep)
            if filtered.num_rows:
                preserved.extend(filtered.to_batches())
        old.close()
    for config in STATCAN_CURRENT:
        release = release_dates[config["pid"]]
        already_stored = (
            config["source_kind"] == "forward_snapshot"
            and release is not None
            and release in existing_releases.get(config["table_id"], set())
        )
        if not already_stored:
            download_archive(config, force=force_download)
    temporary = CURRENT_OUTPUT.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary,
        SCHEMA.with_metadata(
            output_metadata("Statistics Canada non-revised CPI and retail snapshots")
        ),
        compression="zstd",
    )
    rows = 0
    try:
        for batch in preserved:
            writer.write_batch(batch)
            rows += batch.num_rows
        for config in STATCAN_CURRENT:
            is_snapshot = config["source_kind"] == "forward_snapshot"
            release = release_dates[config["pid"]]
            if (
                is_snapshot
                and release is not None
                and release in existing_releases.get(config["table_id"], set())
            ):
                print(f"StatsCan {config['table_id']}: source release already snapshotted")
                continue
            table_rows = 0
            for table in tables_from_archive(
                config,
                snapshot_date=snapshot_date if is_snapshot else None,
                source_release_date=release if is_snapshot else None,
            ):
                writer.write_table(table, row_group_size=250_000)
                rows += table.num_rows
                table_rows += table.num_rows
            print(f"StatsCan {config['table_id']}: kept {table_rows:,} rows")
    finally:
        writer.close()
    if rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Statistics Canada current build produced no rows.")
    temporary.replace(CURRENT_OUTPUT)
    print(
        f"wrote {CURRENT_OUTPUT} rows={rows:,} "
        f"size_mb={CURRENT_OUTPUT.stat().st_size / 1e6:.1f}"
    )


def selected_member_ids(config: dict, metadata: dict) -> list[list[dict]]:
    dimensions = metadata.get("dimension", [])
    choices = []
    for item in dimensions:
        name = item["dimensionNameEn"]
        members = item.get("member", [])
        exact = next(
            (
                values
                for requested, values in config.get("equals", {}).items()
                if name.casefold().startswith(requested.casefold())
            ),
            None,
        )
        hierarchy = next(
            (
                depth
                for requested, depth in config.get("hierarchy", {}).items()
                if name.casefold().startswith(requested.casefold())
            ),
            None,
        )
        if exact is not None:
            selected = [member for member in members if member["memberNameEn"] in exact]
        elif hierarchy is not None:
            allowed = hierarchy_members(config, name, hierarchy) or set()
            selected = [member for member in members if member["memberNameEn"] in allowed]
        else:
            selected = members
        choices.append(selected)
    return choices


def new_release_combinations(
    config: dict,
    last_release: pd.Timestamp | None,
) -> list[dict]:
    metadata = get_metadata(config, refresh=True)
    if not metadata:
        return []
    dimensions = metadata.get("dimension", [])
    release_position = next(
        (
            index
            for index, item in enumerate(dimensions)
            if item["dimensionNameEn"].casefold() == "release"
        ),
        None,
    )
    if release_position is None:
        return []
    choices = selected_member_ids(config, metadata)
    choices[release_position] = [
        member
        for member in choices[release_position]
        if last_release is None
        or pd.to_datetime(member["memberNameEn"], errors="coerce") > last_release
    ]
    combinations = []
    for members in itertools.product(*choices):
        coordinate = ["0"] * 10
        values = {}
        for item, member in zip(dimensions, members):
            coordinate[int(item["dimensionPositionId"]) - 1] = str(member["memberId"])
            values[item["dimensionNameEn"]] = member["memberNameEn"]
        combinations.append({"coordinate": ".".join(coordinate), "values": values})
    return combinations


def batches(items: list, size: int = 100):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_increment(
    config: dict,
    last_release: pd.Timestamp | None,
) -> list[pa.Table]:
    combinations = new_release_combinations(config, last_release)
    if not combinations:
        return []
    client = make_session()
    coordinate_map = {item["coordinate"]: item for item in combinations}
    vectors = []
    for group in batches(combinations):
        response = client.post(
            f"{WDS}/getSeriesInfoFromCubePidCoord",
            json=[
                {"productId": int(config["pid"]), "coordinate": item["coordinate"]}
                for item in group
            ],
            timeout=120,
        )
        response.raise_for_status()
        for item in response.json():
            if item.get("status") != "SUCCESS":
                continue
            obj = item["object"]
            combo = coordinate_map.get(obj["coordinate"])
            if combo:
                vectors.append({**obj, **combo})
    tables = []
    for group in batches(vectors):
        response = client.post(
            f"{WDS}/getDataFromVectorsAndLatestNPeriods",
            json=[{"vectorId": int(item["vectorId"]), "latestN": 1000} for item in group],
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        by_vector = {int(item["vectorId"]): item for item in group}
        for item in payload:
            if item.get("status") != "SUCCESS":
                continue
            obj = item["object"]
            info = by_vector.get(int(obj["vectorId"]))
            if not info:
                continue
            points = pd.DataFrame(obj.get("vectorDataPoint", []))
            if points.empty:
                continue
            frame = pd.DataFrame(
                {
                    "REF_DATE": points["refPer"],
                    "VALUE": points["value"],
                    "VECTOR": str(obj["vectorId"]),
                    "COORDINATE": info["coordinate"],
                    "UOM_ID": info.get("memberUomCode"),
                    "SCALAR_ID": info.get("scalarFactorCode"),
                    "DECIMALS": info.get("decimals"),
                    "STATUS": points.get("statusCode"),
                    "SYMBOL": points.get("symbolCode"),
                }
            )
            for name, value in info["values"].items():
                frame[name] = value
            if "Geography" in frame:
                frame["GEO"] = frame.pop("Geography")
            table = normalize(
                frame,
                config,
                source_file="StatsCan WDS incremental API",
            )
            if table is not None:
                tables.append(table)
    return tables


def append_realtime_updates() -> int:
    if not REALTIME_OUTPUT.exists():
        build_realtime()
        return 0
    parquet = pq.ParquetFile(REALTIME_OUTPUT)
    names = parquet.schema_arrow.names
    table_index = names.index("table_id")
    release_index = names.index("release_date")
    known: dict[str, pd.Timestamp] = {}
    for position in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(position)
        table_stats = row_group.column(table_index).statistics
        release_stats = row_group.column(release_index).statistics
        if (
            not table_stats
            or not release_stats
            or not table_stats.has_min_max
            or not release_stats.has_min_max
        ):
            continue
        table_id = str(table_stats.max)
        release = pd.Timestamp(release_stats.max)
        if table_id not in known or release > known[table_id]:
            known[table_id] = release
    additions = []
    for config in STATCAN_REALTIME:
        if config.get("archived"):
            continue
        fresh = fetch_increment(config, known.get(config["table_id"]))
        additions.extend(fresh)
        if fresh:
            print(f"StatsCan {config['table_id']}: {sum(x.num_rows for x in fresh):,} new rows")
    if not additions:
        parquet.close()
        print("Statistics Canada real-time vintages are current.")
        return 0
    affected = {
        table_id for table in additions for table_id in table["table_id"].unique().to_pylist()
    }
    units: dict[tuple[str, str], tuple] = {}
    columns = [
        "table_id",
        "series_title",
        "unit",
        "unit_id",
        "scalar_factor",
        "scalar_id",
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=500_000):
        values = batch.to_pydict()
        for table_id, title, unit, unit_id, scalar, scalar_id in zip(
            *(values[column] for column in columns)
        ):
            if (
                table_id in affected
                and title is not None
                and (unit is not None or scalar is not None)
            ):
                units.setdefault(
                    (table_id, title),
                    (unit, unit_id, scalar, scalar_id),
                )
    enriched = []
    for table in additions:
        frame = table.to_pandas()
        source = [
            units.get((table_id, title), (None, None, None, None))
            for table_id, title in zip(frame["table_id"], frame["series_title"])
        ]
        for position, column in enumerate(("unit", "unit_id", "scalar_factor", "scalar_id")):
            replacement = pd.Series(
                [value[position] for value in source],
                index=frame.index,
            )
            frame[column] = frame[column].fillna(replacement)
        enriched.append(
            pa.Table.from_pandas(
                frame,
                schema=SCHEMA,
                preserve_index=False,
                safe=False,
            )
        )
    additions = enriched
    temporary = REALTIME_OUTPUT.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(
        temporary, SCHEMA.with_metadata(parquet.schema_arrow.metadata), compression="zstd"
    )
    try:
        for batch in parquet.iter_batches(batch_size=250_000):
            writer.write_batch(batch)
        for table in additions:
            writer.write_table(table)
    finally:
        writer.close()
        parquet.close()
    temporary.replace(REALTIME_OUTPUT)
    rows = sum(table.num_rows for table in additions)
    print(f"appended {rows:,} StatsCan real-time rows")
    return rows
