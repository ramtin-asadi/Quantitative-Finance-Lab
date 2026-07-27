"""Parquet readers for point-in-time SEC fundamentals.

The functions in this module stop at the data boundary: they expose source
metadata, concept and ticker/CIK catalogs, and normalized fact rows. Financial
statement reconstruction and scoring belong to :mod:`quantfinlab.fundamentals`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_CONCEPT_COLUMNS = ("concept", "label", "unit", "period_type")
_MAPPING_COLUMNS = (
    "ticker",
    "cik",
    "entity_name",
    "mapping_source",
    "mapping_confidence",
    "mapping_valid_from",
    "mapping_valid_to",
)
_FACT_COLUMNS = (
    "ticker",
    "cik",
    "entity_name",
    "mapping_valid_from",
    "mapping_valid_to",
    "concept",
    "label",
    "value",
    "unit",
    "period_type",
    "period_start",
    "period_end",
    "fiscal_year",
    "fiscal_period",
    "filed_date",
    "form_type",
    "accession",
    "statement_type",
    "taxonomy",
    "is_annual_filing",
    "is_amendment",
    "filing_version",
)
_DATE_COLUMNS = (
    "mapping_valid_from",
    "mapping_valid_to",
    "period_start",
    "period_end",
    "filed_date",
)
_EXACT_VERSION_KEY = (
    "cik",
    "concept",
    "unit",
    "period_start",
    "period_end",
    "filed_date",
    "accession",
    "filing_version",
)


def _decode_parquet_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, object]:
    decoded: dict[str, object] = {}
    for raw_key, raw_value in (metadata or {}).items():
        key = raw_key.decode("utf-8", errors="replace")
        value = raw_value.decode("utf-8", errors="replace")
        try:
            decoded[key] = json.loads(value)
        except (TypeError, ValueError):
            decoded[key] = value
    return decoded


def _require_validation_pass(metadata: dict[str, object], *, path: Path) -> None:
    result = metadata.get("validation")
    if not isinstance(result, dict):
        raise ValueError(f"{path.name} has no usable 'validation' Parquet metadata.")
    if str(result.get("status", "")).lower() != "pass":
        failures = result.get("failures", [])
        raise ValueError(f"{path.name} failed source validation: {failures}")


def _require_columns(
    available: Iterable[str],
    requested: Sequence[str],
    *,
    path: Path,
) -> None:
    available_set = set(available)
    missing = [column for column in requested if column not in available_set]
    if missing:
        raise ValueError(f"{path.name} is missing requested columns: {missing}")


def read_sec_metadata(
    path: str | Path,
    *,
    include_concepts: bool = True,
    include_mappings: bool = True,
    validate: bool = True,
    batch_size: int = 524_288,
) -> dict[str, object]:
    """Read SEC source metadata and optional compact catalogs.

    Parameters
    ----------
    path : str or pathlib.Path
        Point-in-time SEC fundamentals Parquet file.
    include_concepts : bool, default True
        Include a concept/label/unit/period-type catalog with row counts.
    include_mappings : bool, default True
        Include distinct point-in-time ticker-to-CIK mappings.
    validate : bool, default True
        Require the Parquet ``validation`` metadata to report ``"pass"``.
    batch_size : int, default 524288
        Scanner batch size used while constructing catalogs.

    Returns
    -------
    dict[str, object]
        Decoded ``metadata`` and, when requested, ``concepts`` and ``mappings``
        DataFrames.

    Notes
    -----
    Catalogs are aggregated batch by batch so the full fact table is never
    materialized. PyArrow is imported only when the function is called.
    """

    p = Path(path)
    if not p.exists():
        raise ValueError(f"SEC fundamentals file does not exist: {p}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError("read_sec_metadata requires the optional 'pyarrow' package.") from exc

    parquet = pq.ParquetFile(p)
    metadata = _decode_parquet_metadata(parquet.schema_arrow.metadata)
    if validate:
        _require_validation_pass(metadata, path=p)

    result: dict[str, object] = {"metadata": metadata}
    if not include_concepts and not include_mappings:
        return result

    dataset = ds.dataset(p, format="parquet")
    if include_concepts:
        _require_columns(dataset.schema.names, _CONCEPT_COLUMNS, path=p)
        catalog_parts: list[pd.DataFrame] = []
        scanner = dataset.scanner(
            columns=list(_CONCEPT_COLUMNS),
            batch_size=batch_size,
            use_threads=True,
        )
        batch = None
        frame = None
        part = None
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            part = (
                frame.groupby(list(_CONCEPT_COLUMNS), dropna=False)
                .size()
                .rename("rows")
                .reset_index()
            )
            catalog_parts.append(part)
        if catalog_parts:
            concepts = (
                pd.concat(catalog_parts, ignore_index=True)
                .groupby(list(_CONCEPT_COLUMNS), dropna=False)["rows"]
                .sum()
                .reset_index()
                .sort_values("rows", ascending=False, kind="stable")
                .reset_index(drop=True)
            )
        else:
            concepts = pd.DataFrame(columns=[*_CONCEPT_COLUMNS, "rows"])
        result["concepts"] = concepts
        catalog_parts.clear()
        del scanner, batch, frame, part

    if include_mappings:
        _require_columns(dataset.schema.names, _MAPPING_COLUMNS, path=p)
        mapping_parts: list[pd.DataFrame] = []
        scanner = dataset.scanner(
            columns=list(_MAPPING_COLUMNS),
            batch_size=batch_size,
            use_threads=True,
        )
        batch = None
        for batch in scanner.to_batches():
            mapping_parts.append(batch.to_pandas().drop_duplicates())
        if mapping_parts:
            mappings = pd.concat(mapping_parts, ignore_index=True).drop_duplicates()
            for column in ("mapping_valid_from", "mapping_valid_to"):
                mappings[column] = pd.to_datetime(mappings[column])
            mappings = mappings.reset_index(drop=True)
        else:
            mappings = pd.DataFrame(columns=_MAPPING_COLUMNS)
        result["mappings"] = mappings
        mapping_parts.clear()
        del scanner, batch

    pa.default_memory_pool().release_unused()
    return result


def read_sec_facts(
    path: str | Path,
    *,
    concepts: Iterable[str] | None = None,
    ciks: Iterable[int] | None = None,
    forms: Iterable[str] | None = None,
    period_start: str | pd.Timestamp | None = None,
    columns: Sequence[str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Read a filtered, normalized slice of point-in-time SEC facts.

    Parameters
    ----------
    path : str or pathlib.Path
        Point-in-time SEC fundamentals Parquet file.
    concepts : iterable of str or None, optional
        Concepts to retain.
    ciks : iterable of int or None, optional
        Issuer CIKs to retain.
    forms : iterable of str or None, optional
        Filing forms to retain. Values are normalized to uppercase.
    period_start : str, pandas.Timestamp, or None, optional
        Inclusive modeling start applied to ``period_end``.
    columns : sequence of str or None, optional
        Fact columns to materialize. Defaults to the Project 21 fact boundary.
    validate : bool, default True
        Require passing source metadata before scanning facts.

    Returns
    -------
    pandas.DataFrame
        Filtered facts with normalized dates, form types, taxonomy, and values.
        Nonfinite values, filings before their period end, and duplicate exact
        filing versions are removed when their required columns are present.

    Notes
    -----
    Despite the argument name used by the project notebook, ``period_start`` is
    deliberately pushed down against ``period_end``. Instant balance-sheet facts
    have a null SEC ``period_start`` and must remain eligible.
    """

    p = Path(path)
    if not p.exists():
        raise ValueError(f"SEC fundamentals file does not exist: {p}")

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError("read_sec_facts requires the optional 'pyarrow' package.") from exc

    parquet = pq.ParquetFile(p)
    metadata = _decode_parquet_metadata(parquet.schema_arrow.metadata)
    if validate:
        _require_validation_pass(metadata, path=p)

    dataset = ds.dataset(p, format="parquet")
    selected_columns = list(dict.fromkeys(columns or _FACT_COLUMNS))
    _require_columns(dataset.schema.names, selected_columns, path=p)

    expression = None

    def add_filter(candidate) -> None:
        nonlocal expression
        expression = candidate if expression is None else expression & candidate

    if concepts is not None:
        concept_values = [str(concept) for concept in concepts]
        add_filter(ds.field("concept").isin(concept_values))
    if ciks is not None:
        cik_values = [int(cik) for cik in ciks]
        add_filter(ds.field("cik").isin(cik_values))
    if forms is not None:
        form_values = [str(form).upper().strip() for form in forms]
        add_filter(ds.field("form_type").isin(form_values))
    if period_start is not None:
        add_filter(ds.field("period_end") >= pd.Timestamp(period_start))

    table = dataset.to_table(columns=selected_columns, filter=expression)
    facts = table.to_pandas()
    del table
    pa.default_memory_pool().release_unused()
    for column in _DATE_COLUMNS:
        if column in facts:
            facts[column] = pd.to_datetime(facts[column])
    if "form_type" in facts:
        facts["form_type"] = facts["form_type"].str.upper().str.strip()
    if "taxonomy" in facts:
        facts["taxonomy"] = facts["taxonomy"].str.lower().str.strip()
    if "value" in facts:
        facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
        facts = facts[np.isfinite(facts["value"])]
    if {"filed_date", "period_end"}.issubset(facts.columns):
        facts = facts[facts["filed_date"] >= facts["period_end"]]

    exact_key = list(_EXACT_VERSION_KEY)
    if set(exact_key).issubset(facts.columns):
        facts = (
            facts.sort_values(exact_key)
            .drop_duplicates(exact_key, keep="last")
            .reset_index(drop=True)
        )
    else:
        facts = facts.reset_index(drop=True)
    return facts


__all__ = ["read_sec_facts", "read_sec_metadata"]
