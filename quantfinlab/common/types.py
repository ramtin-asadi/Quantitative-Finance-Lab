from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
import pandas as pd

ArrayLike: TypeAlias = np.ndarray | pd.Series | list[float] | tuple[float, ...]
DFCallable: TypeAlias = Callable[[np.ndarray], np.ndarray]
SeriesOrFrame: TypeAlias = pd.Series | pd.DataFrame

__all__ = ["ArrayLike", "DFCallable", "SeriesOrFrame"]
