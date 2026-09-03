from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import companyfacts
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from event_reviews import BANKRUPTCY_REVIEWS

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
SUBMISSIONS = HERE / "cache" / "submissions"
FUNDAMENTALS = DATA / "sp500_fundamentals.parquet"
OUTPUT = DATA / "sec_credit.parquet"

RELEVANT_FORMS = {
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "8-K",
    "8-K/A",
    "NT 10-K",
    "NT 10-Q",
    "25",
    "25-NSE",
    "15-12B",
    "15-12G",
    "15-15D",
}

MAPPING_COLUMNS = [
    "ticker",
    "mapping_source",
    "mapping_confidence",
    "member_first_date",
    "member_last_date",
    "mapping_valid_from",
    "mapping_valid_to",
]
EVENT_FLAGS = [
    "is_bankruptcy_or_receivership",
    "is_financial_obligation_trigger",
    "is_exit_or_disposal",
    "is_material_impairment",
    "is_delisting_or_listing_failure",
    "is_nonreliance",
    "is_late_filing",
    "is_deregistration",
    "is_registrant_bankruptcy_event",
]
SUBMISSION_COLUMNS = [
    "accepted_at",
    "report_date",
    "form_items",
    "primary_document",
    "file_number",
    "film_number",
    "filing_size",
    "bankruptcy_scope",
    "bankruptcy_review_note",
    "is_xbrl",
    "is_inline_xbrl",
    *EVENT_FLAGS,
]
OUTPUT_COLUMNS = [
    "record_type",
    "ticker",
    "submission_tickers",
    "cik",
    "entity_name",
    "sec_sic",
    "sec_industry",
    "entity_type",
    "is_sp500_issuer",
    "credit_group",
    "mapping_source",
    "mapping_confidence",
    "member_first_date",
    "member_last_date",
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
    "accepted_at",
    "report_date",
    "form_type",
    "accession",
    "form_items",
    "primary_document",
    "file_number",
    "film_number",
    "filing_size",
    "bankruptcy_scope",
    "bankruptcy_review_note",
    "statement_type",
    "taxonomy",
    "data_quality",
    "confidence_score",
    "is_annual_filing",
    "is_amendment",
    "filing_version",
    "is_xbrl",
    "is_inline_xbrl",
    *EVENT_FLAGS,
]

STRING_COLUMNS = {
    "record_type",
    "ticker",
    "submission_tickers",
    "entity_name",
    "sec_industry",
    "entity_type",
    "credit_group",
    "mapping_source",
    "mapping_confidence",
    "concept",
    "label",
    "unit",
    "period_type",
    "fiscal_period",
    "form_type",
    "accession",
    "form_items",
    "primary_document",
    "file_number",
    "film_number",
    "bankruptcy_scope",
    "bankruptcy_review_note",
    "statement_type",
    "taxonomy",
    "data_quality",
}
DATE_COLUMNS = {
    "member_first_date",
    "member_last_date",
    "mapping_valid_from",
    "mapping_valid_to",
    "period_start",
    "period_end",
    "filed_date",
    "accepted_at",
    "report_date",
}
BOOL_COLUMNS = {
    "is_sp500_issuer",
    "is_annual_filing",
    "is_amendment",
    "is_xbrl",
    "is_inline_xbrl",
    *EVENT_FLAGS,
}

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field(column, pa.string())
        if column in STRING_COLUMNS
        else pa.field(column, pa.timestamp("ns"))
        if column in DATE_COLUMNS
        else pa.field(column, pa.bool_())
        if column in BOOL_COLUMNS
        else pa.field(column, pa.int64())
        if column in {"cik", "filing_size"}
        else pa.field(column, pa.int32())
        if column in {"sec_sic", "fiscal_year", "filing_version"}
        else pa.field(column, pa.float32())
        if column == "confidence_score"
        else pa.field(column, pa.float64())
        for column in OUTPUT_COLUMNS
    ]
)


def columnar_frame(payload: dict[str, Any]) -> pd.DataFrame:
    columns = {key: value for key, value in payload.items() if isinstance(value, list)}
    return pd.DataFrame(columns) if columns else pd.DataFrame()


def item_flag(items: pd.Series, code: str) -> pd.Series:
    normalized = items.fillna("").astype(str).str.replace(" ", "", regex=False)
    escaped = code.replace(".", r"\.")
    return normalized.str.contains(rf"(?:^|,){escaped}(?:,|$)", regex=True)


def safe_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    parsed = parsed.where(parsed.between(pd.Timestamp("1900-01-01"), pd.Timestamp("2100-12-31")))
    return parsed.astype("datetime64[ns]")


def load_entity(cik: int) -> tuple[dict[str, Any], dict[str, Any]]:
    current_path = SUBMISSIONS / f"CIK{cik:010d}.json"
    if not current_path.exists():
        raise FileNotFoundError(current_path)
    current = orjson.loads(current_path.read_bytes())
    entity = {
        "cik": cik,
        "entity_name": str(current.get("name", "")),
        "entity_type": str(current.get("entityType", "")).lower(),
        "sec_sic": pd.to_numeric(current.get("sic"), errors="coerce"),
        "sec_industry": str(current.get("sicDescription", "")),
        "tickers": [str(value).upper() for value in current.get("tickers", []) if value],
    }
    return current, entity


def load_submission(
    cik: int,
    start: pd.Timestamp,
    current: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if current is None:
        current, entity = load_entity(cik)
    else:
        entity = {
            "cik": cik,
            "entity_name": str(current.get("name", "")),
            "entity_type": str(current.get("entityType", "")).lower(),
            "sec_sic": pd.to_numeric(current.get("sic"), errors="coerce"),
            "sec_industry": str(current.get("sicDescription", "")),
            "tickers": [str(value).upper() for value in current.get("tickers", []) if value],
        }
    frames: list[pd.DataFrame] = []
    recent = columnar_frame(current.get("filings", {}).get("recent", {}))
    if not recent.empty:
        frames.append(recent)
    for item in current.get("filings", {}).get("files", []):
        if item.get("filingTo", "") < start.strftime("%Y-%m-%d"):
            continue
        path = SUBMISSIONS / str(item.get("name", ""))
        if not path.exists():
            raise FileNotFoundError(f"Missing cached SEC history segment: {path}")
        history = columnar_frame(orjson.loads(path.read_bytes()))
        if not history.empty:
            frames.append(history)
    filings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    required = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "fileNumber",
        "filmNumber",
        "items",
        "size",
        "isXBRL",
        "isInlineXBRL",
        "primaryDocument",
    ]
    for column in required:
        if column not in filings:
            filings[column] = pd.NA
    filings["filed_date"] = safe_datetime(filings["filingDate"])
    filings = filings.loc[
        filings["filed_date"].ge(start) & filings["form"].isin(RELEVANT_FORMS)
    ].copy()
    filings["cik"] = cik
    filings["accepted_at"] = pd.to_datetime(
        filings["acceptanceDateTime"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    filings["report_date"] = safe_datetime(filings["reportDate"])
    filings["form_type"] = filings["form"].fillna("").astype(str)
    filings["accession"] = filings["accessionNumber"].fillna("").astype(str)
    filings["form_items"] = filings["items"].fillna("").astype(str)
    filings["primary_document"] = filings["primaryDocument"].fillna("").astype(str)
    filings["file_number"] = filings["fileNumber"].fillna("").astype(str)
    filings["film_number"] = filings["filmNumber"].fillna("").astype(str)
    filings["filing_size"] = pd.to_numeric(filings["size"], errors="coerce").astype("Int64")
    filings["is_xbrl"] = (
        pd.to_numeric(filings["isXBRL"], errors="coerce").fillna(0).astype(bool)
    )
    filings["is_inline_xbrl"] = (
        pd.to_numeric(filings["isInlineXBRL"], errors="coerce").fillna(0).astype(bool)
    )
    filings["is_bankruptcy_or_receivership"] = item_flag(filings["items"], "1.03")
    filings["is_financial_obligation_trigger"] = item_flag(filings["items"], "2.04")
    filings["is_exit_or_disposal"] = item_flag(filings["items"], "2.05")
    filings["is_material_impairment"] = item_flag(filings["items"], "2.06")
    filings["is_delisting_or_listing_failure"] = item_flag(filings["items"], "3.01")
    filings["is_nonreliance"] = item_flag(filings["items"], "4.02")
    filings["is_late_filing"] = filings["form_type"].isin({"NT 10-K", "NT 10-Q"})
    filings["is_deregistration"] = filings["form_type"].isin(
        {"25", "25-NSE", "15-12B", "15-12G", "15-15D"}
    )
    reviews = pd.DataFrame.from_dict(
        BANKRUPTCY_REVIEWS,
        orient="index",
        columns=[
            "bankruptcy_scope",
            "is_registrant_bankruptcy_event",
            "bankruptcy_review_note",
        ],
    )
    filings = filings.merge(reviews, left_on="accession", right_index=True, how="left")
    unreviewed = filings["is_bankruptcy_or_receivership"] & filings["bankruptcy_scope"].isna()
    filings.loc[unreviewed, "bankruptcy_scope"] = "unreviewed"
    filings["bankruptcy_scope"] = filings["bankruptcy_scope"].fillna("")
    filings["bankruptcy_review_note"] = filings["bankruptcy_review_note"].fillna("")
    filings["is_registrant_bankruptcy_event"] = (
        filings["is_registrant_bankruptcy_event"].fillna(False).astype(bool)
    )
    filings = filings.sort_values(["filed_date", "accepted_at"]).drop_duplicates(
        ["cik", "accession"], keep="last"
    )
    return filings.reset_index(drop=True), entity


def sp500_mapping() -> dict[int, dict[str, Any]]:
    if not FUNDAMENTALS.exists():
        return {}
    columns = ["cik", "entity_name", *MAPPING_COLUMNS]
    mapping = pd.read_parquet(FUNDAMENTALS, columns=columns).drop_duplicates()
    mapping["mapping_valid_to"] = pd.to_datetime(mapping["mapping_valid_to"], errors="coerce")
    mapping["member_last_date"] = pd.to_datetime(mapping["member_last_date"], errors="coerce")
    mapping["sort_date"] = mapping["mapping_valid_to"].fillna(
        mapping["member_last_date"].fillna(pd.Timestamp.min)
    )
    mapping = mapping.sort_values(["cik", "sort_date", "ticker"]).drop_duplicates(
        "cik", keep="last"
    )
    return {
        int(row.cik): {column: getattr(row, column) for column in columns if column != "cik"}
        for row in mapping.itertuples(index=False)
    }


def issuer_mapping(entity: dict[str, Any], sp500: dict[int, dict[str, Any]]) -> dict[str, Any]:
    cik = int(entity["cik"])
    historical = sp500.get(cik)
    tickers = list(dict.fromkeys(entity["tickers"]))
    if historical:
        result = {column: historical.get(column) for column in MAPPING_COLUMNS}
        result["entity_name"] = entity["entity_name"] or historical.get("entity_name", "")
        result["is_sp500_issuer"] = True
        if result.get("ticker") and result["ticker"] not in tickers:
            tickers.insert(0, str(result["ticker"]))
    else:
        result = {
            "ticker": tickers[0] if tickers else "",
            "entity_name": entity["entity_name"],
            "mapping_source": "sec_submissions_current" if tickers else "cik_only",
            "mapping_confidence": "source_reported" if tickers else "cik_identifier",
            "member_first_date": pd.NaT,
            "member_last_date": pd.NaT,
            "mapping_valid_from": pd.NaT,
            "mapping_valid_to": pd.NaT,
            "is_sp500_issuer": False,
        }
    result["submission_tickers"] = "|".join(tickers)
    return result


def normalize_output(frame: pd.DataFrame) -> pa.Table:
    frame = frame.reindex(columns=OUTPUT_COLUMNS)
    for column in STRING_COLUMNS:
        frame[column] = frame[column].fillna("").astype(str)
    for column in DATE_COLUMNS:
        frame[column] = safe_datetime(frame[column])
    for column in BOOL_COLUMNS:
        frame[column] = frame[column].fillna(False).astype(bool)
    frame["cik"] = pd.to_numeric(frame["cik"], errors="raise").astype("int64")
    frame["sec_sic"] = pd.to_numeric(frame["sec_sic"], errors="raise").astype("int32")
    frame["fiscal_year"] = pd.to_numeric(frame["fiscal_year"], errors="coerce").astype("Int32")
    frame["filing_version"] = (
        pd.to_numeric(frame["filing_version"], errors="coerce").fillna(1).astype("int32")
    )
    frame["filing_size"] = pd.to_numeric(frame["filing_size"], errors="coerce").astype("Int64")
    frame["confidence_score"] = pd.to_numeric(
        frame["confidence_score"], errors="coerce"
    ).astype("Float32")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype("float64")
    return pa.Table.from_pandas(frame, schema=OUTPUT_SCHEMA, preserve_index=False, safe=False)


def output_chunks(
    facts: pd.DataFrame,
    filings: pd.DataFrame,
    entity: dict[str, Any],
    mapping: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fact_rows = facts.merge(
        filings[["accession", *SUBMISSION_COLUMNS]],
        on="accession",
        how="left",
        validate="many_to_one",
    )
    fact_rows["record_type"] = "fact"
    for key, value in mapping.items():
        fact_rows[key] = value
    fact_rows["sec_sic"] = int(entity["sec_sic"])
    fact_rows["sec_industry"] = entity["sec_industry"]
    fact_rows["entity_type"] = entity["entity_type"]
    for flag in EVENT_FLAGS:
        fact_rows[flag] = fact_rows[flag].fillna(False).astype(bool)
    fact_rows["is_xbrl"] = fact_rows["is_xbrl"].fillna(True).astype(bool)
    fact_rows["is_inline_xbrl"] = fact_rows["is_inline_xbrl"].fillna(False).astype(bool)

    filing_rows = filings.copy()
    for key, value in mapping.items():
        filing_rows[key] = value
    filing_rows["entity_name"] = mapping["entity_name"]
    filing_rows["sec_sic"] = int(entity["sec_sic"])
    filing_rows["sec_industry"] = entity["sec_industry"]
    filing_rows["entity_type"] = entity["entity_type"]
    filing_rows["record_type"] = "filing"
    filing_rows["credit_group"] = filing_rows[EVENT_FLAGS[:6]].any(axis=1).map(
        {True: "filing_event", False: "filing_metadata"}
    )
    filing_rows["concept"] = ""
    filing_rows["label"] = ""
    filing_rows["value"] = np.nan
    filing_rows["unit"] = ""
    filing_rows["period_type"] = ""
    filing_rows["period_start"] = pd.NaT
    filing_rows["period_end"] = pd.NaT
    filing_rows["fiscal_year"] = pd.NA
    filing_rows["fiscal_period"] = ""
    filing_rows["statement_type"] = "submission"
    filing_rows["taxonomy"] = "sec-submission"
    filing_rows["data_quality"] = "source_metadata"
    filing_rows["confidence_score"] = np.float32(1.0)
    filing_rows["is_annual_filing"] = filing_rows["form_type"].isin({"10-K", "10-K/A"})
    filing_rows["is_amendment"] = filing_rows["form_type"].str.endswith("/A")
    filing_rows["filing_version"] = 1
    return fact_rows, filing_rows


def build(start: str) -> None:
    start_date = pd.Timestamp(start)
    ciks = companyfacts.eligible_ciks()
    if not SUBMISSIONS.exists():
        raise FileNotFoundError(f"{SUBMISSIONS} is missing. Run data/sec_credit/download.py first.")
    sp500 = sp500_mapping()
    temporary = OUTPUT.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()

    counts = {
        "candidate_ciks": len(ciks),
        "ciks": 0,
        "sp500_ciks": 0,
        "fact_rows": 0,
        "filing_rows": 0,
        "bankruptcy_filings": 0,
        "registrant_bankruptcy_events": 0,
        "obligation_trigger_filings": 0,
        "excluded_financial": 0,
        "excluded_nonoperating": 0,
        "excluded_missing_sic": 0,
    }
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    writer = pq.ParquetWriter(
        temporary,
        OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
    )
    try:
        for number, cik in enumerate(ciks, start=1):
            current, entity = load_entity(cik)
            if pd.isna(entity["sec_sic"]):
                counts["excluded_missing_sic"] += 1
                continue
            if 6000 <= int(entity["sec_sic"]) <= 6999:
                counts["excluded_financial"] += 1
                continue
            if entity["entity_type"] not in {"", "operating"}:
                counts["excluded_nonoperating"] += 1
                continue
            filings, entity = load_submission(cik, start_date, current=current)
            facts = pd.read_parquet(companyfacts.fact_cache_path(cik))
            for column in ["period_start", "period_end", "filed_date"]:
                facts[column] = safe_datetime(facts[column])
            facts = facts.loc[
                facts["period_end"].ge(start_date)
                & facts["filed_date"].ge(facts["period_end"])
                & (facts["period_start"].isna() | facts["period_start"].le(facts["period_end"]))
            ].copy()
            if facts.empty or filings.empty:
                continue
            if set(facts["cik"].astype(int).unique()) != {cik}:
                raise ValueError(f"Company Facts cache identity mismatch for CIK {cik}")
            if set(filings["cik"].astype(int).unique()) != {cik}:
                raise ValueError(f"Submissions cache identity mismatch for CIK {cik}")
            mapping = issuer_mapping(entity, sp500)
            fact_rows, filing_rows = output_chunks(facts, filings, entity, mapping)
            fact_key = [
                "cik",
                "concept",
                "unit",
                "period_start",
                "period_end",
                "filed_date",
                "accession",
                "filing_version",
            ]
            if fact_rows.duplicated(fact_key).any():
                raise ValueError(f"Duplicate fact versions remain for CIK {cik}")
            if filing_rows.duplicated(["cik", "accession"]).any():
                raise ValueError(f"Duplicate filing rows remain for CIK {cik}")
            if not np.isfinite(fact_rows["value"]).all():
                raise ValueError(f"Non-finite fact values remain for CIK {cik}")

            combined = pd.concat([fact_rows, filing_rows], ignore_index=True)
            pending.append(combined)
            pending_rows += len(combined)
            counts["ciks"] += 1
            counts["sp500_ciks"] += int(mapping["is_sp500_issuer"])
            counts["fact_rows"] += len(fact_rows)
            counts["filing_rows"] += len(filing_rows)
            counts["bankruptcy_filings"] += int(
                filing_rows["is_bankruptcy_or_receivership"].sum()
            )
            counts["registrant_bankruptcy_events"] += int(
                filing_rows["is_registrant_bankruptcy_event"].sum()
            )
            counts["obligation_trigger_filings"] += int(
                filing_rows["is_financial_obligation_trigger"].sum()
            )
            if pending_rows >= 250_000:
                writer.write_table(normalize_output(pd.concat(pending, ignore_index=True)))
                pending = []
                pending_rows = 0
            if number % 100 == 0 or number == len(ciks):
                print(
                    f"build {number:,}/{len(ciks):,} kept={counts['ciks']:,} "
                    f"rows={counts['fact_rows'] + counts['filing_rows']:,}",
                    flush=True,
                )
        if pending:
            writer.write_table(normalize_output(pd.concat(pending, ignore_index=True)))
        if not counts["ciks"] or not counts["fact_rows"] or not counts["filing_rows"]:
            raise RuntimeError("SEC credit output requires eligible issuers, facts, and filings")
        writer.add_key_value_metadata(
            {
                "dataset": "Point-in-time SEC credit facts and filing events for U.S. nonfinancial operating companies",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": "existing SEC Company Facts cache plus targeted per-CIK Submissions JSON",
                "start": start,
                "rows": str(counts["fact_rows"] + counts["filing_rows"]),
                **{key: str(value) for key, value in counts.items()},
                "point_in_time_rule": "Use accepted_at when present; otherwise use filed_date.",
                "company_requirements": json.dumps(
                    {
                        "us_gaap": True,
                        "minimum_periodic_filings": 8,
                        "minimum_history_years": 2.0,
                        "minimum_asset_periods": 4,
                        "minimum_liability_or_equity_periods": 4,
                        "excluded_sic_range": "6000-6999",
                        "entity_type": "operating or source blank",
                    },
                    sort_keys=True,
                ),
            }
        )
    finally:
        writer.close()
    temporary.replace(OUTPUT)
    print(
        f"wrote {OUTPUT} rows={counts['fact_rows'] + counts['filing_rows']:,} "
        f"ciks={counts['ciks']:,} size_mb={OUTPUT.stat().st_size / 1e6:.1f}"
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=companyfacts.DEFAULT_START)
    args = parser.parse_args()
    build(args.start)


if __name__ == "__main__":
    main()
