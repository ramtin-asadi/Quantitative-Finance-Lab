from __future__ import annotations

import pandas as pd

from ..common.results import SimpleBacktestResult
from ..fixed_income import duration_overlay, ladder


def run_bond_ladder_backtest(
    par_yields: pd.DataFrame,
    *,
    duration_target: float | None = None,
    overlay_fn=None,
    **kwargs,
) -> SimpleBacktestResult:
    if duration_target is not None and overlay_fn is None:
        overlay_fn = duration_overlay.duration_switch_overlay
    return ladder.run_ladder_backtest(
        par_yields,
        duration_target=duration_target,
        overlay_fn=overlay_fn,
        **kwargs,
    )


__all__ = ["run_bond_ladder_backtest"]
