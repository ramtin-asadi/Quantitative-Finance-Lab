"""Incrementally update the repository macro-factor CSVs.

Unlike ``download.py``, this updater does not request the complete history.
It downloads a short recent overlap (six months by default), merges successful
observations into the existing files, appends newly completed months, validates
the result, and atomically replaces an output only after the checks pass.

The overlap deliberately captures normal revisions to recent macro releases.
Older rows are never changed by this script.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import download as base


def issue(
    issues: list[dict[str, str]],
    *,
    country: str,
    source: str,
    series: str,
    requested_id: str,
    message: str,
) -> None:
    issues.append(
        {
            "country": country,
            "source": source,
            "series": series,
            "requested_id": requested_id,
            "used_id": "",
            "message": message,
        }
    )


def fetch_fred(
    series: dict[str, str],
    *,
    country: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    frames = []
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "quantitative-finance-lab/1.0"}
    )
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        status=1,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry,
            pool_connections=1,
            pool_maxsize=1,
        ),
    )

    # A single persistent connection is intentionally used. FRED throttles
    # concurrent graph requests; sequential keep-alive requests complete more
    # reliably than a large worker pool while still downloading only the short
    # recent overlap.
    series_items = list(series.items())
    consecutive_failures = 0
    for position, (name, series_id) in enumerate(series_items, start=1):
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv?"
            f"id={series_id}&cosd={start.date()}&coed={end.date()}"
        )
        try:
            response = session.get(url, timeout=(5, 15))
            response.raise_for_status()
            raw = pd.read_csv(io.StringIO(response.text))
            if raw.shape[1] < 2:
                raise ValueError("FRED response has fewer than two columns")
            values = raw.iloc[:, :2].copy()
            values.columns = ["date", name]
            values["date"] = pd.to_datetime(
                values["date"],
                errors="coerce",
            )
            values[name] = pd.to_numeric(
                values[name],
                errors="coerce",
            )
            frames.append(
                values.dropna(subset=["date"])
                .set_index("date")[name]
                .replace([np.inf, -np.inf], np.nan)
                .rename(name)
            )
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            issue(
                issues,
                country=country,
                source="fred",
                series=name,
                requested_id=series_id,
                message=str(exc),
            )
            if consecutive_failures >= 3:
                for skipped_name, skipped_id in series_items[position:]:
                    issue(
                        issues,
                        country=country,
                        source="fred",
                        series=skipped_name,
                        requested_id=skipped_id,
                        message=(
                            "Skipped after three consecutive FRED failures; "
                            "existing stored values were preserved."
                        ),
                    )
                break
        if position % 10 == 0 or position == len(series_items):
            print(
                f"FRED {country}: processed {position}/{len(series_items)} series",
                flush=True,
            )
    session.close()
    return (
        pd.concat(frames, axis=1).sort_index()
        if frames
        else pd.DataFrame()
    )


def fetch_statcan(
    *,
    latest_n: int,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    payload = [
        {"vectorId": int(vector_id), "latestN": int(latest_n)}
        for vector_id in base.ca_statcan_vectors.values()
    ]
    try:
        response = requests.post(
            "https://www150.statcan.gc.ca/t1/wds/rest/"
            "getDataFromVectorsAndLatestNPeriods",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        issue(
            issues,
            country="ca",
            source="statcan",
            series="all_statcan_vectors",
            requested_id=",".join(
                str(value)
                for value in base.ca_statcan_vectors.values()
            ),
            message=str(exc),
        )
        return pd.DataFrame()

    reverse = {
        int(vector_id): name
        for name, vector_id in base.ca_statcan_vectors.items()
    }
    series_list = []
    for item in data:
        obj = item.get("object", {}) if isinstance(item, dict) else {}
        vector_id = int(obj.get("vectorId", 0) or 0)
        name = reverse.get(vector_id)
        points = obj.get("vectorDataPoint", [])
        if name is None or not points:
            continue
        frame = pd.DataFrame(points)
        frame["date"] = pd.to_datetime(frame["refPer"], errors="coerce")
        frame[name] = pd.to_numeric(frame["value"], errors="coerce")
        series_list.append(
            frame.dropna(subset=["date"])
            .set_index("date")[name]
            .replace([np.inf, -np.inf], np.nan)
            .rename(name)
        )
    return (
        pd.concat(series_list, axis=1).sort_index()
        if series_list
        else pd.DataFrame()
    )


def fetch_boc(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    series_ids = ",".join(base.ca_boc_series.values())
    url = (
        "https://www.bankofcanada.ca/valet/observations/"
        f"{series_ids}/json?start_date={start.date()}&end_date={end.date()}"
    )
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        issue(
            issues,
            country="ca",
            source="boc",
            series="all_boc_series",
            requested_id=series_ids,
            message=str(exc),
        )
        return pd.DataFrame()

    reverse = {
        series_id: name
        for name, series_id in base.ca_boc_series.items()
    }
    rows = []
    for observation in data.get("observations", []):
        row = {
            "date": pd.to_datetime(
                observation.get("d"),
                errors="coerce",
            )
        }
        for series_id, name in reverse.items():
            row[name] = pd.to_numeric(
                observation.get(series_id, {}).get("v"),
                errors="coerce",
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .dropna(subset=["date"])
        .set_index("date")
        .sort_index()
        if rows
        else pd.DataFrame()
    )


def read_output(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index)
    return (
        frame.sort_index()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def merge_increment(
    old: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    overlap_start: pd.Timestamp,
    minimum_series: int,
) -> pd.DataFrame:
    combined = old.copy()
    if not fresh.empty:
        index = combined.index.union(fresh.index).sort_values()
        columns = combined.columns.union(fresh.columns)
        combined = combined.reindex(index=index, columns=columns)
        for column in fresh:
            incoming = fresh[column].dropna()
            combined.loc[incoming.index, column] = incoming

    new_rows = combined.index > old.index.max()
    if bool(new_rows.any()):
        sufficient = combined.loc[new_rows].notna().sum(axis=1).ge(
            minimum_series
        )
        reject = sufficient.index[~sufficient]
        combined = combined.drop(index=reject)

    if not combined.index.is_monotonic_increasing:
        raise ValueError("updated macro index is not sorted")
    if combined.index.has_duplicates:
        raise ValueError("updated macro index contains duplicates")
    if not set(old.columns).issubset(combined.columns):
        raise ValueError("incremental update removed existing columns")
    if combined.index.max() < old.index.max():
        raise ValueError("incremental update shortened the dataset")

    frozen_index = old.index[old.index < overlap_start]
    pd.testing.assert_frame_equal(
        combined.reindex(index=frozen_index, columns=old.columns),
        old.reindex(index=frozen_index, columns=old.columns),
        check_dtype=False,
        check_freq=False,
    )
    return combined


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index_label="date")
    temporary.replace(path)


def backfill_missing_fred_history(
    country: str,
    series_names: list[str],
    *,
    end: pd.Timestamp,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fill historical holes for explicitly requested FRED series only."""

    path = base.us_output if country == "us" else base.ca_output
    mapping = (
        base.us_fred_series
        if country == "us"
        else base.ca_fred_series
    )
    selected = {
        name: mapping[name]
        for name in series_names
        if name in mapping
    }
    if not selected:
        return read_output(path), {}

    old = read_output(path)
    raw = fetch_fred(
        selected,
        country=country,
        start=old.index.min().to_period("M").to_timestamp(),
        end=end,
        issues=issues,
    )
    fresh = base.monthly_table(
        [raw],
        start=str(old.index.min().date()),
        end=str(end.date()),
    )
    combined = old.copy()
    filled: dict[str, int] = {}
    for name in selected:
        if name not in fresh:
            filled[name] = 0
            continue
        incoming = fresh[name].reindex(combined.index)
        missing = combined[name].isna() & incoming.notna()
        combined.loc[missing, name] = incoming.loc[missing]
        filled[name] = int(missing.sum())

    untouched = [column for column in old if column not in selected]
    pd.testing.assert_frame_equal(
        combined[untouched],
        old[untouched],
        check_dtype=False,
        check_freq=False,
    )
    if combined.index.has_duplicates or not combined.index.is_monotonic_increasing:
        raise ValueError("historical backfill damaged the macro index")
    atomic_csv(combined, path)
    return combined, filled


def update_country(
    country: str,
    *,
    end: pd.Timestamp,
    overlap_months: int,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, dict]:
    path = base.us_output if country == "us" else base.ca_output
    old = read_output(path)
    overlap_start = (
        old.index.max() - pd.DateOffset(months=int(overlap_months))
    ).to_period("M").to_timestamp()

    if country == "us":
        raw = fetch_fred(
            base.us_fred_series,
            country="us",
            start=overlap_start,
            end=end,
            issues=issues,
        )
        minimum_series = 20
    else:
        raw = pd.concat(
            [
                fetch_fred(
                    base.ca_fred_series,
                    country="ca",
                    start=overlap_start,
                    end=end,
                    issues=issues,
                ),
                fetch_statcan(
                    latest_n=max(18, overlap_months + 6),
                    issues=issues,
                ),
                fetch_boc(
                    start=overlap_start,
                    end=end,
                    issues=issues,
                ),
            ],
            axis=1,
        )
        minimum_series = 12

    fresh = base.monthly_table(
        [raw],
        start=str(overlap_start.date()),
        end=str(end.date()),
    )
    combined = merge_increment(
        old,
        fresh,
        overlap_start=overlap_start,
        minimum_series=minimum_series,
    )
    atomic_csv(combined, path)
    return combined, {
        "country": country,
        "old_end": old.index.max(),
        "new_end": combined.index.max(),
        "old_rows": len(old),
        "new_rows": len(combined),
        "download_start": overlap_start,
        "download_end": end,
        "successful_series": int(raw.notna().any().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--country",
        choices=("us", "ca", "all"),
        default="all",
    )
    parser.add_argument("--overlap-months", type=int, default=6)
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--backfill-series",
        nargs="*",
        default=[],
        metavar="NAME",
        help=(
            "Optionally fill existing historical holes for named FRED series; "
            "the normal update remains recent-overlap-only."
        ),
    )
    args = parser.parse_args()
    if args.overlap_months < 1:
        raise ValueError("--overlap-months must be positive")

    countries = ("us", "ca") if args.country == "all" else (args.country,)
    end = pd.Timestamp(args.end)
    issues: list[dict[str, str]] = []
    summaries = []
    outputs: dict[str, pd.DataFrame] = {}
    for country in countries:
        outputs[country], summary = update_country(
            country,
            end=end,
            overlap_months=args.overlap_months,
            issues=issues,
        )
        summaries.append(summary)
        requested_backfills = [
            name
            for name in args.backfill_series
            if name
            in (
                base.us_fred_series
                if country == "us"
                else base.ca_fred_series
            )
        ]
        if requested_backfills:
            outputs[country], filled = backfill_missing_fred_history(
                country,
                requested_backfills,
                end=end,
                issues=issues,
            )
            summary["historical_values_filled"] = int(sum(filled.values()))
            summary["backfilled_series"] = ",".join(
                name for name, count in filled.items() if count > 0
            )

    for country, path in (
        ("us", base.us_output),
        ("ca", base.ca_output),
    ):
        if country not in outputs and path.exists():
            outputs[country] = read_output(path)
    pd.concat(
        [
            base.availability_summary(country, outputs[country])
            for country in ("us", "ca")
            if country in outputs
        ],
        ignore_index=True,
    ).to_csv(base.summary_output, index=False)
    pd.DataFrame(
        issues,
        columns=[
            "country",
            "source",
            "series",
            "requested_id",
            "used_id",
            "message",
        ],
    ).to_csv(base.issues_output, index=False)

    summary_frame = pd.DataFrame(summaries)
    print(summary_frame.to_string(index=False))
    print(f"endpoint issues: {len(issues)}")
    if issues:
        print(
            "Some endpoints failed; successful series were merged and all "
            "existing data was preserved.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
