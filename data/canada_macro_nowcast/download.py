from __future__ import annotations

import argparse

from boc import build_bos, build_market, build_mps
from statcan import build_current, build_realtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("all", "statcan", "boc"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.component in {"all", "statcan"}:
        build_realtime(force_download=args.force)
        build_current(force_download=args.force)
    if args.component in {"all", "boc"}:
        build_market()
        build_bos()
        build_mps()


if __name__ == "__main__":
    main()
