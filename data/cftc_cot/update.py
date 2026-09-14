import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantfinlab.analyst.cli import source_main

if __name__ == "__main__":
    source_main("cftc", update=True)
