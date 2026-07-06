from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.contracts import Curve
from .discounting import curve_value_table


def forward_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
) -> pd.DataFrame:
    """Evaluate instantaneous forward rates from fitted curves.

    Parameters
    ----------
    curves : dict[str, Curve]
        Mapping from method name to fitted curve.
    grid : numpy.ndarray or None, optional
        Explicit maturity grid in years.
    t_min : float, default 1/12
        Minimum maturity when generating a grid.
    t_max : float, default 30.0
        Maximum maturity when generating a grid.
    points : int, default 400
        Number of grid points when ``grid`` is omitted.

    Returns
    -------
    pandas.DataFrame
        Forward-rate table indexed by maturity in years with one column per method.
    """

    return curve_value_table(curves, value="forward", grid=grid, t_min=t_min, t_max=t_max, points=points)

__all__ = ["forward_curve_table"]
