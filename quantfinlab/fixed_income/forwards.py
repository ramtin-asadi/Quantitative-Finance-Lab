from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import Curve
from .discounting import curve_value_table


def forward_curve_table(
    curves: dict[str, Curve],
    *,
    grid: np.ndarray | None = None,
    t_min: float = 1 / 12,
    t_max: float = 30.0,
    points: int = 400,
) -> pd.DataFrame:
    return curve_value_table(curves, value="forward", grid=grid, t_min=t_min, t_max=t_max, points=points)

__all__ = ["forward_curve_table"]
