from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MARKET_SCRIPT = DATA_DIR / "sp500_market" / "update.py"
FUNDAMENTALS_SCRIPT = DATA_DIR / "sp500_fundamentals" / "update.py"
MARKET_BUILDER = DATA_DIR / "sp500_market" / "download.py"
MARKET_PATH = DATA_DIR / "sp500_market_data.parquet"
FUNDAMENTALS_PATH = DATA_DIR / "sp500_fundamentals.parquet"


def run_step(label: str, command: list[str], environment: dict[str, str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def decoded_metadata(path: Path) -> dict[str, str]:
    parquet = pq.ParquetFile(path)
    metadata = parquet.schema_arrow.metadata or {}
    parquet.close()
    return {key.decode(): value.decode() for key, value in metadata.items()}


def validation_summary(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    metadata = decoded_metadata(path)
    validation = json.loads(metadata.get("validation", "{}"))
    if validation.get("status") != "pass":
        raise RuntimeError(f"{path.name} does not have passing validation metadata")
    summary = {
        "path": str(path.relative_to(ROOT)),
        "rows": parquet.metadata.num_rows,
        "columns": len(parquet.schema_arrow.names),
        "size_mb": round(path.stat().st_size / 1_000_000, 1),
        "validation_status": validation["status"],
        "date_min": validation.get("date_min") or validation.get("period_end_min"),
        "date_max": validation.get("date_max") or validation.get("filed_date_max"),
    }
    parquet.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally maintain both S&P datasets: prices first, then new SEC "
            "filings, then point-in-time market-cap/industry enrichment."
        )
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("EDGAR_IDENTITY", ""),
        help="SEC User-Agent identity. Prefer the EDGAR_IDENTITY environment variable.",
    )
    parser.add_argument("--market-end", help="Exclusive Yahoo end date (YYYY-MM-DD).")
    parser.add_argument("--market-overlap-days", type=int, default=14)
    parser.add_argument("--sec-index-overlap-days", type=int, default=10)
    parser.add_argument(
        "--market-offline",
        action="store_true",
        help="Reassemble the recent market window from caches without contacting Yahoo.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    identity = " ".join(args.identity.strip().split())
    if "@" not in identity or " " not in identity:
        raise ValueError(
            "Set EDGAR_IDENTITY to a descriptive name and email before running updates."
        )
    for path in [MARKET_PATH, FUNDAMENTALS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run the full builders once first.")

    environment = os.environ.copy()
    environment["EDGAR_IDENTITY"] = identity

    market_command = [
        sys.executable,
        str(MARKET_SCRIPT),
        "--overlap-days",
        str(args.market_overlap_days),
    ]
    if args.market_end:
        market_command.extend(["--end", args.market_end])
    if args.market_offline:
        market_command.append("--offline")
    if args.dry_run:
        market_command.append("--dry-run")
    run_step("1/3 Incremental market and membership update", market_command, environment)

    fundamentals_command = [
        sys.executable,
        str(FUNDAMENTALS_SCRIPT),
        "--index-overlap-days",
        str(args.sec_index_overlap_days),
    ]
    if args.dry_run:
        fundamentals_command.append("--dry-run")
    run_step("2/3 Selective SEC Company Facts update", fundamentals_command, environment)

    if args.dry_run:
        print(
            "\nDry run complete. No output Parquet file was replaced and SEC enrichment "
            "was not run.",
            flush=True,
        )
        return 0

    run_step(
        "3/3 Recompute point-in-time industry and market cap",
        [sys.executable, str(MARKET_BUILDER), "--enrich-only"],
        environment,
    )

    summaries = [
        validation_summary(MARKET_PATH),
        validation_summary(FUNDAMENTALS_PATH),
    ]
    print("\n=== Incremental update complete ===", flush=True)
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
