"""Small calculations used to explain a financial market snapshot."""

import numpy as np
import pandas as pd

from quantfinlab.ml.features import drawdown_level, realized_vol
from quantfinlab.risk.capm import rolling_beta_corr


def market_moves(prices, *, horizons=(1, 5, 21, 63), lookback=252):
    rows = []
    for asset in prices:
        values = prices[asset].dropna()
        if len(values) <= max(horizons):
            continue
        returns = values.pct_change(fill_method=None)
        history = returns.iloc[-lookback-1:-1]
        deviation = history.std()
        rows.append({"asset": asset, "data_date": values.index[-1], "close": values.iloc[-1],
                     **{f"return_{days}d": values.iloc[-1] / values.iloc[-days-1] - 1 for days in horizons},
                     "move_z": (returns.iloc[-1] - history.mean()) / deviation if deviation > 0 else np.nan})
    return pd.DataFrame(rows).set_index("asset") if rows else pd.DataFrame()


def risk_measures(prices, *, benchmark="SPY", window=63):
    returns = prices.pct_change(fill_method=None)
    rows = []
    for asset in prices:
        beta, correlation = rolling_beta_corr(returns[asset], returns[benchmark], window=window)
        rows.append({"asset": asset, "realized_vol_21d": realized_vol(returns[asset], 21).iloc[-1],
                     "drawdown_252d": drawdown_level(prices[asset], 252).iloc[-1],
                     f"beta_{benchmark.lower()}_{window}d": beta.iloc[-1],
                     f"correlation_{benchmark.lower()}_{window}d": correlation.iloc[-1]})
    return pd.DataFrame(rows).set_index("asset")


def curve_moves(rates, *, tenors=("3M", "2Y", "5Y", "10Y", "30Y"), lookback=252):
    rows = []
    for tenor in tenors:
        values = rates[tenor].dropna()
        changes = values.diff() * 10000
        history = changes.iloc[-lookback-1:-1]
        deviation = history.std()
        rows.append({"tenor": tenor, "yield_percent": values.iloc[-1] * 100,
                     "change_1d_bp": changes.iloc[-1], "change_5d_bp": (values.iloc[-1] - values.iloc[-6]) * 10000,
                     "move_z": (changes.iloc[-1] - history.mean()) / deviation if deviation > 0 else np.nan})
    return pd.DataFrame(rows).set_index("tenor")


def curve_shape(rates):
    return pd.DataFrame({"2s10s_bp": (rates["10Y"] - rates["2Y"]) * 10000,
                         "5s30s_bp": (rates["30Y"] - rates["5Y"]) * 10000,
                         "3m10y_bp": (rates["10Y"] - rates["3M"]) * 10000,
                         "level_percent": rates[["2Y", "5Y", "10Y", "30Y"]].mean(axis=1) * 100,
                         "curvature_bp": (2 * rates["5Y"] - rates["2Y"] - rates["10Y"]) * 10000})


def relative_moves(prices, *, pairs=(("QQQ", "SPY"), ("HYG", "LQD"), ("TLT", "IEF"), ("GLD", "SPY"), ("DBC", "UUP"))):
    returns = prices.pct_change(fill_method=None)
    rows = []
    for left, right in pairs:
        if left not in prices or right not in prices:
            continue
        gap = returns[left] - returns[right]
        history = gap.iloc[-253:-1]
        deviation = history.std()
        rows.append({"pair": left + "_minus_" + right, "return_gap_1d_pp": gap.iloc[-1] * 100,
                     "return_gap_5d_pp": ((prices[left].iloc[-1] / prices[left].iloc[-6] - 1)
                                           - (prices[right].iloc[-1] / prices[right].iloc[-6] - 1)) * 100,
                     "gap_z": (gap.iloc[-1] - history.mean()) / deviation if deviation > 0 else np.nan,
                     "correlation_63d": returns[left].tail(63).corr(returns[right].tail(63))})
    return pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()


def snapshot_records(frame):
    return {str(key): {str(name): round(float(value), 6) for name, value in row.items()
                      if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
            for key, row in frame.to_dict("index").items()}
