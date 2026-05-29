from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


def _is_date_token(value: str) -> bool:
    text = str(value).strip()
    return (len(text) == 6 or len(text) == 8) and text.isdigit()


def _parse_date_token(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 6:
        return pd.to_datetime(text + "01", format="%Y%m%d") + pd.offsets.MonthEnd(0)
    return pd.to_datetime(text, format="%Y%m%d")


def _read_first_table(path: str | Path) -> pd.DataFrame:
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header_idx = None
    for i, line in enumerate(lines[:-1]):
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 2:
            continue
        next_token = lines[i + 1].split(",", 1)[0].strip()
        if _is_date_token(next_token):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No monthly Fama-French table found in {path}.")

    rows = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        token = line.split(",", 1)[0].strip()
        if not _is_date_token(token):
            break
        rows.append(line)

    df = pd.read_csv(StringIO("\n".join(rows)))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].map(_parse_date_token)
    df = df.set_index("date").sort_index()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0, -999.99], np.nan)
    return df / 100.0


def load_ff_factors(path, columns=None):
    out = _read_first_table(path)
    if columns is not None:
        out = out.loc[:, [c for c in columns if c in out.columns]]
    return out


def load_ff_momentum(path, name="MOM"):
    out = _read_first_table(path)
    if out.empty:
        return pd.Series(dtype=float, name=name)
    series = out.iloc[:, 0].copy()
    series.name = name
    return series


def load_ff_industries(path):
    return _read_first_table(path)


__all__ = [
    "load_ff_factors",
    "load_ff_industries",
    "load_ff_momentum",
]
