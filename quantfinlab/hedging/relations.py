from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


@dataclass(frozen=True)
class rel:
    """A target asset and one or more hedge assets."""

    name: str
    target: str
    hedges: list[str]
    desc: str = ""
    pair: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip().lower())
        object.__setattr__(self, "target", str(self.target).strip().lower())
        object.__setattr__(self, "hedges", [str(x).strip().lower() for x in self.hedges])
        if self.pair is not None:
            object.__setattr__(self, "pair", tuple(str(x).strip().lower() for x in self.pair))

    @property
    def assets(self) -> list[str]:
        return [self.target, *self.hedges]

    @property
    def single(self) -> bool:
        return len(self.hedges) == 1

    @property
    def hedge_label(self) -> str:
        return "+".join(self.hedges)


def rel_tickers(rels: Iterable[rel]) -> list[str]:
    """Unique tickers used by a relationship list."""
    out: list[str] = []
    for r in rels:
        for ticker in r.assets:
            t = str(ticker).strip().lower()
            if t not in out:
                out.append(t)
    return out


def filter_rels(rels: Sequence[rel], columns: Sequence[str]) -> tuple[list[rel], dict[str, list[str]]]:
    """Keep relationships whose target and hedge tickers are all available."""
    available = {str(c).strip().lower() for c in columns}
    kept: list[rel] = []
    missing: dict[str, list[str]] = {}
    for r in rels:
        miss = [t for t in r.assets if t not in available]
        if miss:
            missing[r.name] = miss
        else:
            kept.append(r)
    return kept, missing


def rel_table(rels: Sequence[rel], columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Compact relationship table with optional availability flag."""
    available = None if columns is None else {str(c).strip().lower() for c in columns}
    rows = []
    for r in rels:
        included = True if available is None else all(t in available for t in r.assets)
        rows.append(
            {
                "relationship": r.name,
                "target": r.target,
                "hedges": ", ".join(r.hedges),
                "residual_pair": "" if r.pair is None else " / ".join(r.pair),
                "included": bool(included),
            }
        )
    return pd.DataFrame(rows)


def hedge_proxy_ret(ret: pd.DataFrame, r: rel) -> pd.Series:
    """Equal-weight hedge proxy return for diagnostics and beta reduction."""
    if not isinstance(ret, pd.DataFrame):
        raise InputError("ret must be a pandas DataFrame.")
    panel = ret.copy()
    panel.columns = [str(c).strip().lower() for c in panel.columns]
    missing = [h for h in r.hedges if h not in panel.columns]
    if missing:
        raise InputError(f"Missing hedge returns for {r.name}: {missing}")
    proxy = panel[r.hedges].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).mean(axis=1)
    proxy.name = f"{r.name}_hedge_proxy"
    return proxy


__all__ = ["filter_rels", "hedge_proxy_ret", "rel", "rel_table", "rel_tickers"]
