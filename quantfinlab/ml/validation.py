"""Chronological splitting with explicit outcome-availability dates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def available_labels(origins, available_at, cutoff, *, observed=None) -> np.ndarray:
    """Mask labels fully known before cutoff; origins alone do not establish maturity."""
    dates, known = pd.to_datetime(origins), pd.to_datetime(available_at)
    mask = (dates < pd.Timestamp(cutoff)) & (known < pd.Timestamp(cutoff))
    if observed is not None:
        mask &= np.asarray(observed, dtype=bool)
    return np.asarray(mask)


def release_splits(origins, available_at, cutoffs, *, observed=None,
                    window=None):
    """Yield cutoff, training indices and forecast indices without mixing date groups.

    The final forecast interval is open-ended. ``window`` is an optional pandas
    DateOffset or Timedelta; credit horizons may use calendar months.
    """
    dates = pd.DatetimeIndex(origins)
    cuts = pd.DatetimeIndex(cutoffs).sort_values().unique()
    for j, date in enumerate(cuts):
        train = available_labels(dates, available_at, date, observed=observed)
        if window is not None:
            train &= dates >= date - window
        test = dates >= date
        if j + 1 < len(cuts):
            test &= dates < cuts[j + 1]
        yield date, np.flatnonzero(train), np.flatnonzero(test)
