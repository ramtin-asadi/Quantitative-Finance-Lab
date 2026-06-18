from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
from time import sleep
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError, ParserError


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = Path(__file__).resolve().parent / "raw"
FIELD_ORDER = ("close", "volume")
HKEX_SECURITIES_URL = (
    "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
)


def ticker_from_path(path: Path, market: str) -> str:
    name = path.name.lower()
    if market == "us":
        return name.replace(".us.txt", "").upper()
    code = name.replace(".hk.txt", "")
    if code.isdigit() and len(code) <= 4:
        code = code.zfill(4)
    return f"{code.upper()}.HK"


def stock_code_from_path(path: Path) -> str:
    code = path.name.lower().replace(".hk.txt", "")
    if not code.isdigit():
        return code.upper()
    return code.zfill(4) if len(code) <= 4 else code


def normalize_col(name: object) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def col_number(ref: str) -> int:
    letters = "".join(ch for ch in str(ref) if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + ord(ch.upper()) - ord("A") + 1
    return max(value - 1, 0)


def read_xlsx_simple(content: bytes) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(BytesIO(content)) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                texts = [node.text or "" for node in item.findall(".//a:t", ns)]
                shared.append("".join(texts))
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in root.findall(".//a:sheetData/a:row", ns):
        values: list[str] = []
        for cell in row.findall("a:c", ns):
            idx = col_number(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                texts = [node.text or "" for node in cell.findall(".//a:t", ns)]
                value = "".join(texts)
            else:
                raw = cell.find("a:v", ns)
                value = "" if raw is None or raw.text is None else raw.text
                if cell_type == "s" and value.isdigit():
                    value = shared[int(value)]
            values[idx] = value
        if any(str(v).strip() for v in values):
            rows.append(values)
    header_idx = None
    for i, row in enumerate(rows):
        normalized = [normalize_col(v) for v in row]
        if "stockcode" in normalized and (
            "nameofsecurities" in normalized or "nameofsecurity" in normalized
        ):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find HKEX security-name header row in downloaded workbook")
    header = rows[header_idx]
    width = len(header)
    records = [row[:width] + [""] * max(0, width - len(row)) for row in rows[header_idx + 1:]]
    return pd.DataFrame(records, columns=header)


def clean_security_name(name: object) -> str:
    text = " ".join(str(name).replace("__", " ").split())
    return text.strip()


def download_hkex_name_lookup() -> dict[str, str]:
    errors: list[str] = []
    for attempt in range(1, 4):
        req = Request(
            HKEX_SECURITIES_URL,
            headers={"User-Agent": "Mozilla/5.0 quantfinlab-data-builder/1.0"},
        )
        try:
            with urlopen(req, timeout=120) as response:
                raw = read_xlsx_simple(response.read())
            break
        except Exception as exc:
            errors.append(f"attempt {attempt}: {exc}")
            if attempt == 3:
                raise RuntimeError(
                    "Could not download HKEX List of Securities from the official URL. "
                    + " | ".join(errors)
                ) from exc
            sleep(5 * attempt)
    raw.columns = [str(c).strip() for c in raw.columns]
    lookup = {normalize_col(c): c for c in raw.columns}
    code_col = lookup.get("stockcode") or lookup.get("code") or raw.columns[0]
    name_col = (
        lookup.get("nameofsecurities")
        or lookup.get("nameofsecurity")
        or lookup.get("securityname")
        or raw.columns[1]
    )
    mapping: dict[str, str] = {}
    for _, row in raw.iterrows():
        match = pd.Series([row[code_col]]).astype(str).str.extract(r"(\d+)", expand=False).iloc[0]
        if pd.isna(match):
            continue
        code = str(int(match)).zfill(4) if str(match).isdigit() and len(str(int(match))) <= 4 else str(match)
        name = clean_security_name(row[name_col])
        if name:
            mapping[code] = name
    return mapping


def unique_name_map(codes: list[str], names_by_code: dict[str, str]) -> dict[str, str]:
    base_names = {code: names_by_code[code] for code in codes if code in names_by_code}
    counts = Counter(base_names.values())
    seen: Counter[str] = Counter()
    out: dict[str, str] = {}
    for code in sorted(base_names):
        name = base_names[code]
        if counts[name] == 1:
            out[code] = name
            continue
        seen[name] += 1
        out[code] = f"{name} [{seen[name]}]"
    return out


def discover(market: str) -> list[Path]:
    suffix = "*.us.txt" if market == "us" else "*.hk.txt"
    files = sorted(RAW.rglob(suffix))
    if not files:
        raise FileNotFoundError(f"No Stooq {market} raw files found under {RAW}")
    return files


def read_one(path: Path, label: str) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path)
    except (EmptyDataError, ParserError) as exc:
        print(f"skip {path.name}: {exc}")
        return None
    frame.columns = [str(c).strip().strip("<>") for c in frame.columns]
    required = {"DATE", "CLOSE", "VOL"}
    if not required.issubset(frame.columns):
        print(f"skip {path.name}: missing {sorted(required - set(frame.columns))}")
        return None
    out = frame[["DATE", "CLOSE", "VOL"]].copy()
    out["DATE"] = pd.to_datetime(out["DATE"].astype(str), format="%Y%m%d", errors="coerce")
    out = out.dropna(subset=["DATE"]).drop_duplicates("DATE", keep="last").sort_values("DATE")
    if out.empty:
        return None
    out = out.set_index("DATE")
    out[f"{label}__close"] = pd.to_numeric(out["CLOSE"], errors="coerce")
    out[f"{label}__volume"] = pd.to_numeric(out["VOL"], errors="coerce")
    return out[[f"{label}__close", f"{label}__volume"]]


def build(market: str, output_name: str) -> int:
    files = discover(market)
    labels_by_code: dict[str, str] = {}
    if market == "hk":
        names_by_code = download_hkex_name_lookup()
        codes = [stock_code_from_path(path) for path in files]
        labels_by_code = unique_name_map(codes, names_by_code)
        missing = sorted(set(codes) - set(labels_by_code))
        if missing:
            print(
                "skip HKEX files without names in the current HKEX securities list: "
                f"{len(missing):,}"
            )
    frames = []
    seen = set()
    for i, path in enumerate(files, 1):
        if market == "hk":
            code = stock_code_from_path(path)
            label = labels_by_code.get(code)
            if label is None:
                continue
        else:
            label = ticker_from_path(path, market)
        if label in seen:
            continue
        seen.add(label)
        frame = read_one(path, label)
        if frame is not None:
            frames.append(frame)
        if i % 500 == 0:
            print(f"processed {i:,}/{len(files):,} files")
    if not frames:
        raise RuntimeError(f"No readable Stooq {market} files under {RAW}")
    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    ordered = []
    labels = sorted({c.rsplit("__", 1)[0] for c in combined.columns})
    for label in labels:
        for field in FIELD_ORDER:
            col = f"{label}__{field}"
            if col in combined.columns:
                ordered.append(col)
    combined = combined[ordered].replace([np.inf, -np.inf], np.nan)
    out = combined.reset_index().rename(columns={"DATE": "date", "index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    output = DATA / output_name
    out.to_parquet(output, index=False, compression="zstd")
    print(f"wrote {output} rows={len(out):,} securities={len(labels):,} cols={len(out.columns):,}")
    return 0

if __name__ == "__main__":
    raise SystemExit(build("hk", "hkex_close_volume.parquet"))
