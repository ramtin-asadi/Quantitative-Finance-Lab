from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


@dataclass(frozen=True)
class rel:
    """Define a hedge relationship between one target asset and one or more hedges.

    Attributes
    ----------
    name : str
        Relationship identifier. It is normalized to lowercase.
    target : str
        Target asset to hedge. It is normalized to lowercase.
    hedges : list of str
        Hedge assets. Each name is normalized to lowercase.
    desc : str, default=""
        Optional human-readable description.
    pair : tuple of str, optional
        Optional residual-trading pair identifier.

    Properties
    ----------
    assets : list of str
        Target followed by hedge assets.
    single : bool
        True when the relationship has exactly one hedge asset.
    hedge_label : str
        Hedge tickers joined by ``"+"``.
    """

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
    """Filter hedge relationships by available columns.

    Parameters
    ----------
    rels : sequence of rel
        Candidate relationships.
    columns : sequence of str
        Available asset columns.

    Returns
    -------
    tuple
        ``(kept, missing)`` where ``kept`` is a list of complete relationships and
        ``missing`` maps relationship names to missing asset tickers.
    """

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
    """Build a compact relationship table.

    Parameters
    ----------
    rels : sequence of rel
        Relationships to summarize.
    columns : sequence of str, optional
        Optional available columns used to mark whether each relationship is
        included.

    Returns
    -------
    pandas.DataFrame
        Table with relationship name, target, hedge list, residual-pair label, and
        inclusion flag.
    """

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
    """Compute an equal-weight hedge-proxy return series.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel containing all hedge assets.
    r : rel
        Relationship defining the hedge assets.

    Returns
    -------
    pandas.Series
        Equal-weight average return of the hedge assets, named after the
        relationship.

    Raises
    ------
    InputError
        If ``ret`` is not a DataFrame or any hedge asset is missing.
    """

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
