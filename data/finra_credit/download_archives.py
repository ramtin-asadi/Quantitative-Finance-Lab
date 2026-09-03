from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
PAGE = "https://www.finra.org/finra-data/browse-catalog/structured-product-activity-reports-and-tables/historic-reports"
ARCHIVE_PATTERN = re.compile(r"HISTORIC_SPREPORTS-(\d{6})\.zip$", re.IGNORECASE)


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.headers["User-Agent"] = "Quantitative-Finance-Lab FINRA archive downloader"
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def discover(session: requests.Session) -> list[tuple[str, str]]:
    response = session.get(PAGE, timeout=120)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    archives = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(response.url, anchor["href"])
        match = ARCHIVE_PATTERN.search(url)
        if match:
            archives.append((match.group(1), url))
    archives = sorted(set(archives))
    if not archives:
        raise RuntimeError("FINRA's historic-report page exposed no monthly ZIP links.")
    return archives


def download(start: str | None, end: str | None, latest_only: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    session = make_session()
    archives = discover(session)
    if start:
        archives = [item for item in archives if item[0] >= start.replace("-", "")]
    if end:
        archives = [item for item in archives if item[0] <= end.replace("-", "")]
    if latest_only and archives:
        archives = [archives[-1]]
    if not archives:
        raise ValueError("No FINRA archives fall inside the requested month range.")

    print(f"archives selected={len(archives):,}")
    downloaded = 0
    for month, url in archives:
        path = RAW / f"HISTORIC_SPREPORTS-{month}.zip"
        if path.exists() and path.stat().st_size > 10_000:
            print(f"cached {path.name}")
            continue
        temporary = path.with_suffix(".zip.tmp")
        with session.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        if not zipfile.is_zipfile(temporary):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"FINRA response is not a ZIP archive: {url}")
        temporary.replace(path)
        downloaded += 1
        print(f"downloaded {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    print(f"archives downloaded={downloaded:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="First YYYY-MM archive to keep.")
    parser.add_argument("--end", help="Last YYYY-MM archive to keep.")
    parser.add_argument("--latest-only", action="store_true")
    args = parser.parse_args()
    download(args.start, args.end, args.latest_only)


if __name__ == "__main__":
    main()
