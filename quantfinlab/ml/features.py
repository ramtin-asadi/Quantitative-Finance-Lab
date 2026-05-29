from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def total_return(prices: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    px = pd.Series(prices) if not isinstance(prices, (pd.Series, pd.DataFrame)) else prices
    return px.astype(float) / px.astype(float).shift(int(window)) - 1.0


def skip_return(
    prices: pd.Series | pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
) -> pd.Series | pd.DataFrame:
    px = pd.Series(prices) if not isinstance(prices, (pd.Series, pd.DataFrame)) else prices
    return px.astype(float).shift(int(skip)) / px.astype(float).shift(int(lookback)) - 1.0


def future_return(prices: pd.Series | pd.DataFrame, horizon: int = 21) -> pd.Series | pd.DataFrame:
    px = pd.Series(prices) if not isinstance(prices, (pd.Series, pd.DataFrame)) else prices
    return px.astype(float).shift(-int(horizon)) / px.astype(float) - 1.0


def relative_return(
    prices_a: pd.Series,
    prices_b: pd.Series,
    window: int,
) -> pd.Series:
    return total_return(prices_a, window) - total_return(prices_b, window)


def realized_vol(
    returns: pd.Series | pd.DataFrame,
    window: int,
    annualization: float = 252.0,
) -> pd.Series | pd.DataFrame:
    r = pd.Series(returns) if not isinstance(returns, (pd.Series, pd.DataFrame)) else returns
    return r.astype(float).rolling(int(window)).std(ddof=1) * np.sqrt(float(annualization))


def drawdown_level(prices: pd.Series | pd.DataFrame, window: int = 252) -> pd.Series | pd.DataFrame:
    px = pd.Series(prices) if not isinstance(prices, (pd.Series, pd.DataFrame)) else prices
    high = px.astype(float).rolling(int(window), min_periods=max(2, int(window) // 4)).max()
    return px.astype(float) / high - 1.0


def drawdown_change(
    prices_or_drawdown: pd.Series | pd.DataFrame,
    drawdown_window: int | None = None,
    change_window: int | None = None,
    *,
    window: int | None = None,
) -> pd.Series | pd.DataFrame:
    x = (
        pd.Series(prices_or_drawdown)
        if not isinstance(prices_or_drawdown, (pd.Series, pd.DataFrame))
        else prices_or_drawdown
    )
    if change_window is not None:
        dd = drawdown_level(x, int(252 if drawdown_window is None else drawdown_window))
        shift = int(change_window)
    else:
        dd = x.astype(float)
        shift = int(window if window is not None else (21 if drawdown_window is None else drawdown_window))
    return dd.astype(float) - dd.astype(float).shift(shift)


def rolling_pair_corr(
    returns_or_a: pd.DataFrame | pd.Series,
    asset_a: str | pd.Series,
    asset_b: str | int | None = None,
    window: int = 252,
) -> pd.Series:
    if isinstance(returns_or_a, pd.DataFrame):
        if asset_b is None:
            raise ValueError("asset_b is required when the first argument is a DataFrame.")
        return returns_or_a[str(asset_a)].astype(float).rolling(int(window)).corr(
            returns_or_a[str(asset_b)].astype(float)
        )
    if isinstance(asset_a, pd.Series):
        win = int(asset_b) if isinstance(asset_b, (int, np.integer)) else int(window)
        return returns_or_a.astype(float).rolling(win).corr(asset_a.astype(float))
    raise ValueError("Pass either (returns, asset_a, asset_b) or (series_a, series_b).")


def rolling_avg_corr(returns: pd.DataFrame, window: int = 252) -> pd.Series:
    r = returns.astype(float)
    cols = list(r.columns)
    vals = []
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            vals.append(r[left].rolling(int(window)).corr(r[right]))
    if not vals:
        return pd.Series(np.nan, index=r.index, name="avg_corr")
    return pd.concat(vals, axis=1).mean(axis=1).rename("avg_corr")


def breadth(prices: pd.DataFrame, window: int = 63, assets: Sequence[str] | None = None) -> pd.Series:
    cols = list(assets) if assets is not None else list(prices.columns)
    ret = total_return(prices[cols], int(window))
    return ret.gt(0.0).mean(axis=1).rename(f"breadth_{window}")


def dispersion(prices: pd.DataFrame, window: int = 63, assets: Sequence[str] | None = None) -> pd.Series:
    cols = list(assets) if assets is not None else list(prices.columns)
    ret = total_return(prices[cols], int(window))
    return ret.std(axis=1, ddof=1).rename(f"dispersion_{window}")


def feature_vif(x: pd.DataFrame) -> pd.DataFrame:
    z = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    rows = []
    for col in z.columns:
        others = [c for c in z.columns if c != col]
        if not others:
            rows.append({"feature": col, "r2": np.nan, "vif": np.nan})
            continue
        y = z[col].to_numpy(dtype=float)
        X = z[others].to_numpy(dtype=float)
        r2 = 1.0 if np.nanstd(y) <= 1e-14 else float(LinearRegression().fit(X, y).score(X, y))
        vif = np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "r2": r2, "vif": vif})
    out = pd.DataFrame(rows).set_index("feature")
    return out.sort_values("vif", ascending=False)


def pca_tables(
    x: pd.DataFrame,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = x.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    n = min(z.shape)
    if n_components is None:
        n_components = n
    n_components = int(max(1, min(int(n_components), n)))
    arr = StandardScaler().fit_transform(z)
    pca = PCA(n_components=n_components, random_state=0).fit(arr)
    pcs = [f"PC{i + 1}" for i in range(n_components)]
    explained = pd.DataFrame(
        {
            "component": pcs,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    ).set_index("component")
    loadings = pd.DataFrame(pca.components_.T, index=z.columns, columns=pcs)
    return explained, loadings


__all__ = [
    "breadth",
    "dispersion",
    "drawdown_change",
    "drawdown_level",
    "feature_vif",
    "future_return",
    "pca_tables",
    "realized_vol",
    "relative_return",
    "rolling_avg_corr",
    "rolling_pair_corr",
    "skip_return",
    "total_return",
]
