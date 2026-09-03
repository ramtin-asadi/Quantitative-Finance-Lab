from __future__ import annotations

from boc import build_bos, build_market, build_mps
from statcan import append_realtime_updates, build_current


def main() -> None:
    append_realtime_updates()
    build_current(force_download=True)
    build_market(update=True)
    build_bos(update=True)
    build_mps(update=True)


if __name__ == "__main__":
    main()
