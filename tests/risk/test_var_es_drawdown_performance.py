from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.risk.drawdown import (
    avg_recovery_time,
    drawdown_episodes,
    drawdown_episodes_table,
    drawdown_series,
    drawdown_summary_table,
    ulcer_index,
)
from quantfinlab.risk.es import cornish_fisher_es, filtered_historical_es, historical_es
from quantfinlab.risk.performance import (
    make_returns_panel,
    nav_series,
    performance_table,
    rolling_volatility,
    sortino_ratio,
)
from quantfinlab.risk.var import (
    cf_var_es,
    fhs_var_es,
    hist_var_es,
    historical_var,
    rolling_var,
    var_es_table,
)
from tests.synthetic.generators import return_panel


def test_var_es_wrappers_and_rolling_forecasts_are_positive_for_left_tail_losses() -> None:
    panel = return_panel(n=90, assets=("AAA", "BBB", "CCC"))
    returns = panel["AAA"]

    hist_var, hist_es = hist_var_es(returns, alpha=0.10)
    cf_var, cf_es = cf_var_es(returns, alpha=0.10, n_sim=5_000, seed=3)
    fhs_var, fhs_es = fhs_var_es(returns, alpha=0.10, lam=0.92)
    rolling = rolling_var(returns, alpha=0.10, lookback=20, method="hist")
    table = var_es_table(panel[["AAA", "BBB"]], alpha=0.10, methods=("hist", "fhs"))

    assert hist_var > 0
    assert hist_es >= hist_var
    assert historical_var(returns, alpha=0.10) == hist_var
    assert historical_es(returns, alpha=0.10) == hist_es
    assert cornish_fisher_es(returns, alpha=0.10, n_sim=5_000, seed=3) == cf_es
    assert filtered_historical_es(returns, alpha=0.10, lam=0.92) == fhs_es
    assert np.isfinite([cf_var, cf_es, fhs_var, fhs_es]).all()
    assert rolling.notna().sum() == len(returns) - 20
    assert {"hist_var10", "hist_es10", "fhs_var10", "fhs_es10"}.issubset(table.columns)


def test_drawdown_and_performance_tables_summarize_synthetic_return_panel() -> None:
    panel = return_panel(n=80, assets=("AAA", "BBB"))
    returns = panel["AAA"]

    nav = nav_series(returns, start_value=100.0)
    dd = drawdown_series(nav, input_kind="nav")
    episodes = drawdown_episodes(returns)
    perf = performance_table(panel)
    vol = rolling_volatility(panel, windows=(5, 10))
    joined = make_returns_panel({"a": panel["AAA"], "b": panel["BBB"]})
    dd_summary = drawdown_summary_table(panel)
    dd_top = drawdown_episodes_table(panel, top_n=2)

    assert nav.iloc[0] == 100.0 * (1.0 + returns.iloc[0])
    assert dd.max() <= 0.0
    assert len(episodes) >= 1
    recovered = pd.Series([0.02, -0.05, 0.04, 0.02, -0.01, 0.02], index=returns.index[:6])
    assert avg_recovery_time(recovered) >= 1.0
    assert ulcer_index(returns) > 0.0
    assert {"ann_return", "ann_vol", "sharpe", "sortino"}.issubset(perf.columns)
    assert np.isfinite(sortino_ratio(returns))
    assert ("AAA", "vol_5") in vol.columns
    assert joined.shape == (80, 2)
    assert dd_summary.loc["AAA", "max_dd"] < 0.0
    assert set(dd_top["object"]).issubset({"AAA", "BBB"})
