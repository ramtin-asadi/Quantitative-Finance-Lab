"""Run the repository's bounded, source-specific incremental updaters."""

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from .inference import save_json
from .sources import source_folders

logger = logging.getLogger(__name__)
structured_sources = {"market": "core_cross_asset_etfs", "rates": "us_treasury_yields", "factors": "factor_proxy_etfs",
    "fundamentals": "sp500_fundamentals",
    "issuer_prices": "sp500_market", "credit": "fed_credit", "cmdi": "nyfed_cmdi",
    "finra": "finra_credit", "macro": "alfred_realtime", "high_frequency": "macro_high_frequency",
    "gdpnow": "gdpnow", "policy": "atlanta_mpt", "financial_conditions": "macro_factors"}


def update_sources(config, *, identity, sources=(), tickers=(), structured=(), limit=4,
                   force=False, max_age_hours=20):
    if not identity or "@" not in identity:
        raise ValueError("Supply an identifying name and contact email for source updates.")
    if limit < 1 or limit > 30:
        raise ValueError("Choose between 1 and 30 documents per source.")
    unknown = set(sources) - source_folders.keys()
    unknown_structured = set(structured) - structured_sources.keys()
    if unknown or unknown_structured:
        raise ValueError(f"Unknown update sources: {sorted(unknown | unknown_structured)}")
    if "sec" in sources and not tickers:
        raise ValueError("SEC document updates require selected tickers.")
    environment = os.environ.copy()
    environment["EDGAR_IDENTITY"] = identity
    environment["PYTHONUNBUFFERED"] = "1"
    commands = []
    for name in structured:
        script = config.root / "data" / structured_sources[name] / "update.py"
        if not script.exists():
            script = script.with_name("download.py")
        commands.append(("structured-" + name, [sys.executable, str(script)]))
    for source in dict.fromkeys([*sources, *(["sec"] if tickers else [])]):
        command = [sys.executable, str(config.root / "data" / source_folders[source] / "update.py"),
                   "--root", str(config.root), "--limit", str(limit)]
        if source == "sec":
            for ticker in tickers:
                command += ["--ticker", ticker]
        if source == "bls":
            command += ["--start-year", str(datetime.now().year)]
        if source == "gdelt":
            command += ["--transport", "gal", "--minutes", "30"]
        commands.append((source, command))
    rows = []
    for name, command in commands:
        key = name + ("-" + "-".join(sorted(t.upper() for t in tickers)) if name == "sec" else "")
        receipt = config.workspace / "updates/orchestrator" / (key + ".json")
        saved = json.loads(receipt.read_text()) if receipt.exists() else {}
        due = not saved or datetime.now(timezone.utc) - datetime.fromisoformat(saved["finished_at"]) > timedelta(hours=max_age_hours)
        if not force and not due and saved.get("returncode") == 0:
            rows.append({"source": name, "status": "recent update reused", "finished_at": saved["finished_at"]})
            continue
        log_path = config.workspace / "updates" / (key + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Updating %s; progress and source errors are saved in %s", name, log_path)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log_path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=config.root, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, creationflags=flags, timeout=1800)
        row = {"source": name, "finished_at": datetime.now(timezone.utc).isoformat(),
               "returncode": result.returncode, "status": "updated" if result.returncode == 0 else "failed",
               "log": str(log_path.relative_to(config.root))}
        save_json(receipt, row)
        rows.append(row)
        if result.returncode:
            logger.warning("Source %s failed; inspect %s", name, log_path)
    save_json(config.workspace / "updates/last_update.json", rows)
    if any(row["status"] == "updated" for row in rows):
        save_json(config.workspace / "current_cutoff.json",
                  {"as_of": datetime.now(timezone.utc).replace(microsecond=0).isoformat()})
    return rows
