from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(name: str, *args: str) -> None:
    subprocess.run([sys.executable, str(HERE / name), *args], cwd=ROOT, check=True)


def main() -> None:
    run("download_archives.py", "--latest-only")
    if os.getenv("FINRA_CLIENT_ID") and os.getenv("FINRA_CLIENT_SECRET"):
        run("download_api.py", "--update")
    else:
        print("FINRA API update skipped: FINRA_CLIENT_ID/FINRA_CLIENT_SECRET are not set")
    run("build.py")


if __name__ == "__main__":
    main()
