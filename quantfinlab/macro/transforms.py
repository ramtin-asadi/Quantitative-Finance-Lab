"""Source transformation codes and training-only macro scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fred_transform(values: pd.DataFrame, codes=None) -> pd.DataFrame:
    """Apply FRED-MD transformation codes 1-7, with log/percent changes scaled by 100."""
    codes = values.attrs["transformation"] if codes is None else codes
    result = {}
    for name in values:
        x, code = values[name].astype(float), int(codes[name])
        if code == 1:
            y = x
        elif code == 2:
            y = x.diff()
        elif code == 3:
            y = x.diff().diff()
        elif code == 4:
            y = np.log(x.where(x > 0))
        elif code == 5:
            y = 100 * np.log(x.where(x > 0)).diff()
        elif code == 6:
            y = 100 * np.log(x.where(x > 0)).diff().diff()
        elif code == 7:
            y = 100 * x.pct_change(fill_method=None).diff()
        else:
            raise ValueError(f"Unknown transformation code {code} for {name}.")
        result[name] = y
    return pd.DataFrame(result)


def growth_rate(values, *, periods=1, scale=100, method="log", annualization=1):
    """Explicit log, percentage or level growth on a regular time index."""
    if method == "log":
        return scale * np.log(values.where(values > 0)).diff(periods)
    if method == "percent":
        return scale * ((values / values.shift(periods)) ** annualization - 1)
    if method == "difference":
        return scale * values.diff(periods)
    if method == "level":
        return values.copy()
    raise ValueError("Unknown growth method.")


def robust_standardize(values: pd.DataFrame, training_end, *, cap=8.0) -> tuple:
    """Median/MAD scaling estimated through training_end; preserve ragged missing data."""
    train = values.loc[:pd.Timestamp(training_end)]
    location = train.median()
    scale = 1.4826 * (train - location).abs().median()
    scale = scale.where(scale.gt(0) & np.isfinite(scale), train.std(ddof=0))
    scale = scale.where(scale.gt(0) & np.isfinite(scale), 1)
    z = (values - location) / scale
    capped = z.abs().gt(cap).mean()
    return z.clip(-cap, cap), location, scale, capped
