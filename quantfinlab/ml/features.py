from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


DEFAULT_ASSET_GROUPS = {
    "SPY": "us_equity",
    "QQQ": "us_equity",
    "IWM": "us_equity",
    "EFA": "intl_equity",
    "EEM": "intl_equity",
    "VNQ": "real_assets",
    "DBC": "real_assets",
    "GLD": "real_assets",
    "IEF": "rates",
    "TLT": "rates",
    "LQD": "credit",
    "HYG": "credit",
    "XLB": "sector",
    "XLC": "sector",
    "XLE": "sector",
    "XLF": "sector",
    "XLI": "sector",
    "XLK": "sector",
    "XLP": "sector",
    "XLU": "sector",
    "XLV": "sector",
    "XLY": "sector",
}


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


def forward_excess_return(
    prices: pd.Series | pd.DataFrame,
    cash_prices: pd.Series | None = None,
    *,
    horizon: int = 21,
    rf_daily: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Forward log excess return over cash or a constant daily risk-free rate."""
    px = pd.Series(prices) if not isinstance(prices, (pd.Series, pd.DataFrame)) else prices
    forward = np.log(px.astype(float).shift(-int(horizon)) / px.astype(float))
    if cash_prices is not None:
        cash = pd.Series(cash_prices, dtype=float).reindex(px.index)
        cash_forward = np.log(cash.shift(-int(horizon)) / cash)
        return forward.sub(cash_forward, axis=0)
    rf = 0.0 if rf_daily is None else float(rf_daily)
    return forward - int(horizon) * np.log1p(rf)


def ex_ante_vol(
    returns: pd.Series | pd.DataFrame,
    *,
    lookback: int = 63,
    horizon: int = 21,
) -> pd.Series | pd.DataFrame:
    """Trailing daily volatility scaled to a forward horizon."""
    r = pd.Series(returns) if not isinstance(returns, (pd.Series, pd.DataFrame)) else returns
    return r.astype(float).rolling(int(lookback), min_periods=int(lookback)).std(ddof=1) * np.sqrt(int(horizon))


def vol_scaled_return(
    returns: pd.Series | pd.DataFrame,
    sigma: pd.Series | pd.DataFrame,
    *,
    clip: float | None = None,
) -> pd.Series | pd.DataFrame:
    """Scale return-like labels by an ex-ante volatility estimate."""
    out = returns.astype(float).div(sigma.astype(float).replace(0.0, np.nan))
    return out.clip(-float(clip), float(clip)) if clip is not None else out


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


def _as_frame(data: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    out = pd.DataFrame(data).copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    if columns is not None:
        keep = [c for c in columns if c in out.columns]
        out = out.reindex(columns=keep)
    return out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _rolling_sharpe(
    returns: pd.Series,
    window: int,
    *,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
) -> pd.Series:
    r = pd.Series(returns, dtype=float)
    ex = r - float(rf_daily)
    mean = ex.rolling(int(window), min_periods=max(10, int(window) // 3)).mean()
    vol = r.rolling(int(window), min_periods=max(10, int(window) // 3)).std(ddof=1)
    return mean.div(vol.replace(0.0, np.nan)) * np.sqrt(float(annualization))


def _volume_zscore(volume: pd.Series, window: int = 63) -> pd.Series:
    v = np.log1p(pd.Series(volume, dtype=float))
    mean = v.rolling(int(window), min_periods=max(10, int(window) // 3)).mean()
    std = v.rolling(int(window), min_periods=max(10, int(window) // 3)).std(ddof=1)
    return (v - mean).div(std.replace(0.0, np.nan))


def build_asset_feature_block(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    *,
    assets: Sequence[str] | None = None,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Build the Project 19 asset-date candidate feature block.

    The output is long-form with columns ``date`` and ``asset`` plus the
    per-asset features used by the forecasting notebook.
    """
    cols = list(assets) if assets is not None else list(close.columns)
    px_all = _as_frame(close).ffill(limit=3)
    px = px_all.reindex(columns=cols)
    ret_all = (
        _as_frame(returns).reindex(index=px_all.index)
        if returns is not None
        else px_all.pct_change(fill_method=None)
    )
    ret = ret_all.reindex(columns=cols)
    vol_panel = _as_frame(volume, cols) if volume is not None and not volume.empty else None

    rows: list[pd.DataFrame] = []
    for asset in cols:
        if asset not in px.columns:
            continue
        p = px[asset].astype(float)
        r = ret[asset].astype(float) if asset in ret.columns else p.pct_change(fill_method=None)
        frame = pd.DataFrame(index=px.index)
        frame["r_1"] = r
        for window in (5, 21, 63, 126):
            frame[f"r_{window}"] = total_return(p, window)
        frame["skip_r_21_252"] = skip_return(p, lookback=252, skip=21)
        for window in (21, 63, 126):
            frame[f"vol_{window}"] = realized_vol(r, window, annualization=annualization)
        frame["down_vol_63"] = (
            r.where(r < 0.0)
            .rolling(63, min_periods=21)
            .std(ddof=1)
            .mul(np.sqrt(float(annualization)))
        )
        frame["drawdown_126"] = drawdown_level(p, 126)
        frame["drawdown_252"] = drawdown_level(p, 252)
        frame["sharpe_63"] = _rolling_sharpe(
            r, 63, rf_daily=rf_daily, annualization=annualization
        )
        frame["sharpe_126"] = _rolling_sharpe(
            r, 126, rf_daily=rf_daily, annualization=annualization
        )
        frame["skew_63"] = r.rolling(63, min_periods=42).skew()
        frame["autocorr_63"] = r.rolling(63, min_periods=42).corr(r.shift(1))
        vol_21_daily = r.rolling(21, min_periods=15).std(ddof=1)
        frame["vol_of_vol_63"] = vol_21_daily.rolling(63, min_periods=42).std(ddof=1) * np.sqrt(float(annualization))
        frame["downside_asymmetry_63"] = frame["down_vol_63"].div(frame["vol_63"].replace(0.0, np.nan))
        trend_inputs = pd.concat(
            [frame["r_5"], frame["r_21"], frame["r_63"], frame["r_126"]],
            axis=1,
        )
        frame["trend_consistency"] = trend_inputs.gt(0.0).mean(axis=1)
        if "SPY" in ret_all.columns and "SPY" in px_all.columns:
            spy_r = ret_all["SPY"].astype(float)
            spy_var = spy_r.rolling(63, min_periods=42).var(ddof=1)
            beta_spy = r.rolling(63, min_periods=42).cov(spy_r).div(spy_var.replace(0.0, np.nan))
            frame["corr_spy_63"] = r.rolling(63, min_periods=42).corr(spy_r)
            frame["beta_spy_63"] = beta_spy
            frame["resid_mom_63_spy"] = frame["r_63"] - beta_spy * total_return(px_all["SPY"], 63)
        for ref in ("IEF", "TLT", "GLD"):
            if ref in ret_all.columns:
                frame[f"corr_{ref.lower()}_63"] = r.rolling(63, min_periods=42).corr(ret_all[ref].astype(float))
        if vol_panel is not None and asset in vol_panel.columns:
            frame["volume_z_63"] = _volume_zscore(vol_panel[asset], 63)
        else:
            frame["volume_z_63"] = np.nan
        frame.insert(0, "asset", asset)
        frame.insert(0, "date", frame.index)
        rows.append(frame.reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=["date", "asset"])
    return _add_cross_sectional_asset_features(pd.concat(rows, ignore_index=True))


asset_return_features = build_asset_feature_block


def _cross_sectional_zscore(series: pd.Series) -> pd.Series:
    s = pd.Series(series, dtype=float)
    std = s.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def _add_cross_sectional_asset_features(data: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(data).copy()
    if out.empty or not {"date", "asset"}.issubset(out.columns):
        return out
    out["date"] = pd.to_datetime(out["date"])
    group = out["asset"].astype(str).map(DEFAULT_ASSET_GROUPS).fillna("other")
    feature_cols = [
        "r_21",
        "r_63",
        "r_126",
        "skip_r_21_252",
        "sharpe_126",
        "drawdown_126",
        "vol_63",
        "trend_consistency",
        "resid_mom_63_spy",
    ]
    grouped = out.assign(_asset_group=group)
    for col in feature_cols:
        if col not in out.columns:
            continue
        out[f"xs_z_{col}"] = out.groupby("date", sort=False)[col].transform(_cross_sectional_zscore)
        group_mean = grouped.groupby(["date", "_asset_group"], sort=False)[col].transform("mean")
        out[f"group_rel_{col}"] = out[col] - group_mean
    return out.replace([np.inf, -np.inf], np.nan)


def _maybe_relative(close: pd.DataFrame, left: str, right: str, window: int) -> pd.Series:
    if left in close.columns and right in close.columns:
        return relative_return(close[left], close[right], window)
    return pd.Series(np.nan, index=close.index, dtype=float)


def build_cross_asset_feature_block(
    close: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    *,
    assets: Sequence[str] | None = None,
    cash_ticker: str = "SHY",
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    """Build Project 16-style cross-asset and market-regime features."""
    cols = list(assets) if assets is not None else list(close.columns)
    px = _as_frame(close).ffill(limit=3)
    ret = _as_frame(returns) if returns is not None else px.pct_change(fill_method=None)
    use_assets = [c for c in cols if c in px.columns]

    out = pd.DataFrame(index=px.index)
    out["breadth_21"] = breadth(px, 21, use_assets)
    out["breadth_63"] = breadth(px, 63, use_assets)
    out["dispersion_21"] = dispersion(px, 21, use_assets)
    out["dispersion_63"] = dispersion(px, 63, use_assets)
    out["rolling_avg_corr_63"] = rolling_avg_corr(ret[use_assets], 63) if use_assets else np.nan
    out["rolling_avg_corr_126"] = (
        rolling_avg_corr(ret[use_assets], 126) if use_assets else np.nan
    )
    if benchmark_ticker in px.columns:
        out["spy_r_63"] = total_return(px[benchmark_ticker], 63)
    else:
        out["spy_r_63"] = np.nan
    out["qqq_spy_63"] = _maybe_relative(px, "QQQ", "SPY", 63)
    out["hyg_lqd_63"] = _maybe_relative(px, "HYG", "LQD", 63)
    out["hyg_lqd_change_63"] = out["hyg_lqd_63"].diff(63)
    out["tlt_ief_63"] = _maybe_relative(px, "TLT", "IEF", 63)
    out["tlt_shy_63"] = _maybe_relative(px, "TLT", cash_ticker, 63)
    out["gld_spy_126"] = _maybe_relative(px, "GLD", "SPY", 126)
    out["dbc_ief_126"] = _maybe_relative(px, "DBC", "IEF", 126)
    risk_cols = [c for c in ["SPY", "QQQ", "IWM", "EFA", "EEM", "HYG"] if c in px.columns]
    defensive_cols = [c for c in ["IEF", "TLT", "LQD", "GLD", cash_ticker] if c in px.columns]
    risk_ret = total_return(px[risk_cols], 63).mean(axis=1) if risk_cols else np.nan
    defensive_ret = (
        total_return(px[defensive_cols], 63).mean(axis=1) if defensive_cols else np.nan
    )
    out["risk_defensive_spread_63"] = risk_ret - defensive_ret
    vol_21 = realized_vol(ret[use_assets], 21) if use_assets else pd.DataFrame(index=px.index)
    vol_63 = realized_vol(ret[use_assets], 63) if use_assets else pd.DataFrame(index=px.index)
    out["avg_vol_change_63"] = vol_21.mean(axis=1) - vol_63.mean(axis=1)
    out.index.name = "date"
    return out.replace([np.inf, -np.inf], np.nan)


context_return_features = build_cross_asset_feature_block


def build_fci_feature_block(
    macro_factors: pd.DataFrame | None = None,
    nfci: pd.DataFrame | None = None,
    *,
    index: Sequence[pd.Timestamp] | pd.Index | None = None,
    min_history: int = 60,
    release_lag_months: int = 1,
) -> pd.DataFrame:
    """Build aligned financial-condition features from available macro data."""
    parts: list[pd.DataFrame | pd.Series] = []
    if macro_factors is not None and not macro_factors.empty:
        from quantfinlab.macro import indicators
        from quantfinlab.macro.models import economic_fci, fci_change, fci_percentile

        factors = _as_frame(macro_factors)
        signal_list = [
            indicators.inflation_level_pressure(factors, min_history=min_history),
            indicators.inflation_impulse(factors, min_history=min_history),
            indicators.inflation_acceleration(factors, min_history=min_history),
            indicators.inflation_diffusion(factors, min_history=min_history),
            indicators.policy_tightness(factors, min_history=min_history),
            indicators.policy_shock(factors, min_history=min_history),
            indicators.growth_momentum_stress(factors, min_history=min_history),
            indicators.growth_acceleration_stress(factors, min_history=min_history),
            indicators.growth_breadth_stress(factors, min_history=min_history),
            indicators.survey_warning(factors, min_history=min_history),
            indicators.labor_cooling(factors, min_history=min_history),
            indicators.housing_impulse_stress(factors, min_history=min_history),
            indicators.external_demand_stress(factors, min_history=min_history),
            indicators.external_vulnerability(factors, min_history=min_history),
        ]
        signals = pd.concat(signal_list, axis=1)
        signals["stress_breadth"] = indicators.stress_breadth(
            signals, min_history=min_history
        )
        signals["severe_stress_breadth"] = indicators.severe_stress_breadth(
            signals, min_history=min_history
        )
        signals["stagflation_pressure"] = indicators.stagflation_pressure(signals)
        signals["goldilocks_support"] = indicators.goldilocks_support(signals)
        blocks = indicators.condition_blocks(signals)
        fci = economic_fci(blocks, min_history=min_history)
        macro_out = pd.DataFrame(
            {
                "fci_level": fci,
                "fci_percentile": fci_percentile(fci, min_history=min_history),
                "fci_change_21": fci_change(fci, periods=1),
                "fci_change_63": fci_change(fci, periods=3),
                "stress_breadth": signals["stress_breadth"],
                "policy_pressure": blocks.get("policy_rate_pressure_block"),
                "inflation_pressure": blocks.get("inflation_pressure_block"),
                "growth_pressure": blocks.get("growth_recession_block"),
            }
        )
        macro_visible = macro_out.shift(int(release_lag_months))
        parts.append(macro_visible)

    if nfci is not None and not nfci.empty:
        n = _as_frame(nfci)
        nfci_out = pd.DataFrame(index=n.index)
        if "NFCI" in n.columns:
            nfci_out["nfci_level"] = n["NFCI"]
            nfci_out["nfci_change_21"] = n["NFCI"].diff(1)
            nfci_out["nfci_change_63"] = n["NFCI"].diff(3)
        for col in ["Risk", "Credit", "Leverage", "Nonfinancial_Leverage"]:
            if col in n.columns:
                nfci_out[f"nfci_{col.lower()}"] = n[col]
        parts.append(nfci_out)

    if parts:
        out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    else:
        out = pd.DataFrame()

    if index is not None:
        idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
        out = out.reindex(idx.union(out.index)).sort_index().ffill().reindex(idx)
        out.index.name = "date"
    return out


fci_feature_frame = build_fci_feature_block


def assemble_forecasting_table(
    base: pd.DataFrame,
    asset_features: pd.DataFrame,
    cross_features: pd.DataFrame | None = None,
    fci_features: pd.DataFrame | None = None,
    regime_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge target/base rows with asset, cross-asset, FCI, and regime features."""
    data = pd.DataFrame(base).copy()
    data["date"] = pd.to_datetime(data["date"])
    asset = pd.DataFrame(asset_features).copy()
    asset["date"] = pd.to_datetime(asset["date"])
    out = data.merge(asset, on=["date", "asset"], how="left")
    for frame in [cross_features, fci_features, regime_features]:
        if frame is None or frame.empty:
            continue
        f = pd.DataFrame(frame).copy()
        if "date" not in f.columns:
            f = f.reset_index()
        f["date"] = pd.to_datetime(f["date"])
        out = out.merge(f, on="date", how="left")
    out = out.sort_values(["date", "asset"]).reset_index(drop=True)
    return add_forecasting_feature_upgrades(out)


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(numerator, dtype=float).div(pd.Series(denominator, dtype=float).replace(0.0, np.nan))


def add_forecasting_feature_upgrades(data: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional, relative, and regime-interaction features for Project 19.

    These transformations use only date-t information.  They intentionally avoid
    target columns except for ``sigma_21``, which is a trailing risk estimate
    computed at the forecast date.
    """
    out = pd.DataFrame(data).copy()
    if out.empty or "date" not in out.columns or "asset" not in out.columns:
        return out
    out["date"] = pd.to_datetime(out["date"])

    rank_cols = [
        "r_21",
        "r_63",
        "r_126",
        "skip_r_21_252",
        "vol_63",
        "sharpe_126",
        "drawdown_126",
        "volume_z_63",
        "trend_consistency",
        "resid_mom_63_spy",
        "corr_spy_63",
        "corr_ief_63",
        "corr_tlt_63",
        "corr_gld_63",
        "vol_of_vol_63",
        "downside_asymmetry_63",
    ]
    for col in rank_cols:
        if col in out.columns:
            out[f"rank_{col}"] = out.groupby("date")[col].rank(pct=True)

    if {"r_21", "sigma_21"}.issubset(out.columns):
        out["mom_21_sigma21"] = _safe_div(out["r_21"], out["sigma_21"])
    if {"r_63", "vol_63"}.issubset(out.columns):
        out["mom_63_vol63"] = _safe_div(out["r_63"], out["vol_63"])
    if {"r_126", "vol_126"}.issubset(out.columns):
        out["mom_126_vol126"] = _safe_div(out["r_126"], out["vol_126"])
    if {"skip_r_21_252", "vol_126"}.issubset(out.columns):
        out["skip_21_252_vol126"] = _safe_div(out["skip_r_21_252"], out["vol_126"])

    if {"r_5", "r_21"}.issubset(out.columns):
        out["trend_5_21"] = out["r_5"] - out["r_21"]
    if {"r_21", "r_63"}.issubset(out.columns):
        out["trend_21_63"] = out["r_21"] - out["r_63"]
    if {"r_63", "r_126"}.issubset(out.columns):
        out["trend_63_126"] = out["r_63"] - out["r_126"]

    by_date = out.groupby("date", sort=False)
    if "r_63" in out.columns:
        avg = by_date["r_63"].transform("mean")
        out["rel_r_63_avg"] = out["r_63"] - avg
    if "r_126" in out.columns:
        avg = by_date["r_126"].transform("mean")
        out["rel_r_126_avg"] = out["r_126"] - avg
    if "vol_63" in out.columns:
        avg = by_date["vol_63"].transform("mean")
        out["rel_vol_63_avg"] = _safe_div(out["vol_63"], avg)
    if "drawdown_126" in out.columns:
        avg = by_date["drawdown_126"].transform("mean")
        out["rel_drawdown_126_avg"] = out["drawdown_126"] - avg
    if "SPY" in set(out["asset"].astype(str)):
        spy = (
            out.loc[out["asset"].astype(str).eq("SPY"), ["date", "r_63", "r_126"]]
            .drop_duplicates("date")
            .rename(columns={"r_63": "spy_asset_r_63", "r_126": "spy_asset_r_126"})
        )
        out = out.merge(spy, on="date", how="left")
        if {"r_63", "spy_asset_r_63"}.issubset(out.columns):
            out["rel_r_63_spy"] = out["r_63"] - out["spy_asset_r_63"]
        if {"r_126", "spy_asset_r_126"}.issubset(out.columns):
            out["rel_r_126_spy"] = out["r_126"] - out["spy_asset_r_126"]

    targeted = {
        "r_63_x_fci_percentile": ("r_63", "fci_percentile"),
        "vol_63_x_stress_breadth": ("vol_63", "stress_breadth"),
        "sharpe_126_x_p_risk_on": ("sharpe_126", "p_risk_on"),
        "drawdown_126_x_p_defensive": ("drawdown_126", "p_defensive"),
        "mom_63_vol63_x_p_risk_on": ("mom_63_vol63", "p_risk_on"),
        "skip_21_252_vol126_x_fci": ("skip_21_252_vol126", "fci_percentile"),
        "rank_r_63_x_p_risk_on": ("rank_r_63", "p_risk_on"),
        "rank_vol_63_x_p_defensive": ("rank_vol_63", "p_defensive"),
        "rel_r_63_avg_x_fci": ("rel_r_63_avg", "fci_percentile"),
        "trend_21_63_x_stress": ("trend_21_63", "stress_breadth"),
    }
    for new_col, (left, right) in targeted.items():
        if left in out.columns and right in out.columns:
            out[new_col] = out[left].astype(float) * out[right].astype(float)

    return out.replace([np.inf, -np.inf], np.nan)


def clean_feature_columns(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    max_missing: float = 0.25,
    min_std: float = 1e-8,
    max_abs_corr: float = 0.98,
) -> list[str]:
    """Apply the first-stage Project 19 feature cleaning screen."""
    x = pd.DataFrame(data[list(feature_cols)]).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    missing = x.isna().mean()
    cols = missing[missing <= float(max_missing)].index.tolist()
    if not cols:
        return []
    std = x[cols].std(skipna=True, ddof=0)
    cols = std[std > float(min_std)].index.tolist()
    if len(cols) <= 1:
        return cols
    corr = x[cols].fillna(x[cols].median()).corr().abs()
    keep: list[str] = []
    for col in cols:
        if all(float(corr.loc[col, prev]) < float(max_abs_corr) for prev in keep):
            keep.append(col)
    return keep


def feature_availability_by_date(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    target_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize daily cross-section and feature availability."""
    frame = pd.DataFrame(data).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "row_count",
                "asset_count",
                "feature_coverage",
                "min_asset_feature_coverage",
                "target_complete",
            ]
        )
    frame[date_col] = pd.to_datetime(frame[date_col])
    cols = [c for c in feature_cols if c in frame.columns]
    by_date = frame.groupby(date_col, sort=True)
    out = pd.DataFrame(
        {
            "row_count": by_date.size(),
            "asset_count": by_date[asset_col].nunique()
            if asset_col in frame.columns
            else by_date.size(),
        }
    )
    if cols:
        feature_present = frame[cols].apply(pd.to_numeric, errors="coerce").notna()
        daily_feature = feature_present.groupby(frame[date_col], sort=True).mean()
        row_feature_coverage = feature_present.mean(axis=1)
        out["feature_coverage"] = daily_feature.mean(axis=1)
        out["min_asset_feature_coverage"] = row_feature_coverage.groupby(
            frame[date_col], sort=True
        ).min()
        out["min_feature_cross_section_coverage"] = daily_feature.min(axis=1)
    else:
        out["feature_coverage"] = np.nan
        out["min_asset_feature_coverage"] = np.nan
        out["min_feature_cross_section_coverage"] = np.nan
    target_cols = [c for c in (target_cols or []) if c in frame.columns]
    if target_cols:
        target_complete = frame[target_cols].notna().all(axis=1)
        out["target_complete"] = target_complete.groupby(frame[date_col], sort=True).mean()
    else:
        out["target_complete"] = np.nan
    return out


def trim_feature_table_by_availability(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    target_cols: Sequence[str] | None = None,
    min_feature_coverage: float = 0.75,
    min_asset_count: int | None = None,
    min_target_complete: float = 1.0,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.DataFrame]:
    """Trim leading dates before the table is suitable for model training."""
    availability = feature_availability_by_date(
        data,
        feature_cols,
        date_col=date_col,
        asset_col=asset_col,
        target_cols=target_cols,
    )
    if availability.empty:
        return pd.DataFrame(data).copy(), None, availability
    eligible = availability["feature_coverage"] >= float(min_feature_coverage)
    if min_asset_count is not None:
        eligible &= availability["asset_count"] >= int(min_asset_count)
    if target_cols:
        eligible &= availability["target_complete"] >= float(min_target_complete)
    eligible_dates = availability.index[eligible.fillna(False)]
    first_date = pd.Timestamp(eligible_dates.min()) if len(eligible_dates) else None
    out = pd.DataFrame(data).copy()
    if first_date is not None:
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.loc[out[date_col] >= first_date].copy()
    return out, first_date, availability


def _regime_feature_matrix(close: pd.DataFrame, returns: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    x = build_cross_asset_feature_block(close, returns, assets=assets)
    extra = pd.DataFrame(index=close.index)
    if "SPY" in close.columns:
        extra["spy_126"] = total_return(close["SPY"], 126)
        extra["spy_skip_21_252"] = skip_return(close["SPY"], 252, 21)
    if "HYG" in close.columns and "LQD" in close.columns:
        extra["hyg_lqd_126"] = relative_return(close["HYG"], close["LQD"], 126)
    if "TLT" in close.columns and "IEF" in close.columns:
        extra["tlt_ief_126"] = relative_return(close["TLT"], close["IEF"], 126)
    return pd.concat([x, extra], axis=1).replace([np.inf, -np.inf], np.nan)


def _future_regime_label(
    close: pd.DataFrame,
    *,
    benchmark_ticker: str,
    cash_ticker: str,
    horizon: int,
) -> pd.Series:
    bench = close[benchmark_ticker].astype(float)
    if cash_ticker in close.columns:
        cash = close[cash_ticker].astype(float)
        score = np.log(bench.shift(-int(horizon)) / bench) - np.log(
            cash.shift(-int(horizon)) / cash
        )
    else:
        score = np.log(bench.shift(-int(horizon)) / bench)
    return score.rename("future_risk_score")


def _make_classifier(name: str, random_state: int = 42):
    key = str(name).strip().lower().replace("_", "").replace("-", "")
    if key in {"logistic", "logisticregression", "logit"}:
        return LogisticRegression(max_iter=2500, class_weight="balanced", C=0.75)
    if key in {"gradientboosting", "gradientboostingclassifier", "gb", "gbm"}:
        return GradientBoostingClassifier(
            n_estimators=220,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=25,
            random_state=int(random_state),
        )
    raise ValueError("model must be LogisticRegression or GradientBoosting.")


def regime_probability_features(
    close: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    *,
    assets: Sequence[str],
    cash_ticker: str = "SHY",
    benchmark_ticker: str = "SPY",
    model: str = "LogisticRegression",
    horizon: int = 21,
    train_days: int = 1260,
    min_train: int = 504,
    rebalance_dates: Sequence[pd.Timestamp | str] | None = None,
    output: str = "features",
    max_weight: float = 0.35,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Walk-forward risk-on/neutral/defensive probabilities or blended weights."""
    px = _as_frame(close).ffill(limit=3)
    ret = _as_frame(returns) if returns is not None else px.pct_change(fill_method=None)
    if benchmark_ticker not in px.columns:
        raise ValueError(f"{benchmark_ticker!r} is required for regime probabilities.")
    use_assets = [a for a in assets if a in px.columns]
    x_all = _regime_feature_matrix(px, ret, use_assets)
    y_score = _future_regime_label(
        px, benchmark_ticker=benchmark_ticker, cash_ticker=cash_ticker, horizon=horizon
    )

    if rebalance_dates is None:
        dates = pd.DatetimeIndex(x_all.index)
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(list(rebalance_dates))).sort_values().unique()
        dates = dates[dates.isin(x_all.index)]

    def _fit_probability_for_date(dt) -> pd.Series | None:
        cutoff = pd.Timestamp(dt) - pd.tseries.offsets.BDay(int(horizon))
        hist_raw = pd.concat([x_all.loc[:cutoff], y_score.loc[:cutoff]], axis=1)
        hist_raw = hist_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_risk_score"])
        current = x_all.loc[[dt]].replace([np.inf, -np.inf], np.nan)
        usable = [
            c
            for c in x_all.columns
            if hist_raw[c].notna().mean() > 0.80 and np.isfinite(current[c].iloc[0])
        ]
        if len(usable) < 4:
            return None
        hist = hist_raw[usable + ["future_risk_score"]].dropna()
        if len(hist) > int(train_days):
            hist = hist.tail(int(train_days))
        if len(hist) < int(min_train):
            return None
        q_low, q_high = hist["future_risk_score"].quantile([1.0 / 3.0, 2.0 / 3.0])
        y = pd.Series(1, index=hist.index, dtype=int)
        y.loc[hist["future_risk_score"] >= q_high] = 0
        y.loc[hist["future_risk_score"] <= q_low] = 2
        if y.nunique() < 3:
            return None
        scaler = StandardScaler()
        x_train = scaler.fit_transform(hist[usable].astype(float))
        x_now = scaler.transform(current[usable].astype(float))
        clf = _make_classifier(model, random_state=random_state)
        clf.fit(x_train, y)
        raw = pd.Series(0.0, index=[0, 1, 2], dtype=float)
        p = clf.predict_proba(x_now)[0]
        for cls, val in zip(clf.classes_, p, strict=False):
            raw.loc[int(cls)] = float(val)
        raw = raw / float(raw.sum()) if float(raw.sum()) > 1e-12 else raw.add(1.0 / 3.0)
        return pd.Series(
            {
                "p_risk_on": raw.loc[0],
                "p_neutral": raw.loc[1],
                "p_defensive": raw.loc[2],
                "regime_confidence": float(raw.max()),
            },
            name=pd.Timestamp(dt),
        )

    n_jobs_int = int(n_jobs or 1)
    if n_jobs_int > 1 and len(dates) > 1:
        try:
            from joblib import Parallel, delayed

            prob_rows = Parallel(n_jobs=n_jobs_int, prefer="threads", batch_size=1)(
                delayed(_fit_probability_for_date)(dt) for dt in dates
            )
        except Exception:
            prob_rows = [_fit_probability_for_date(dt) for dt in dates]
    else:
        prob_rows = [_fit_probability_for_date(dt) for dt in dates]
    prob_rows = [row for row in prob_rows if row is not None]
    probabilities = pd.DataFrame(prob_rows)
    probabilities.index.name = "date"
    if output.lower().startswith("feature") or output.lower().startswith("prob"):
        return probabilities

    from quantfinlab.portfolio.sizing import cap_weights

    weight_rows: list[pd.Series] = []
    for dt, p in probabilities.iterrows():
        hist_ret = ret[use_assets].loc[:dt].tail(252).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        hist_px = px[use_assets].loc[:dt]
        mom = total_return(hist_px, min(126, max(21, len(hist_px) - 1))).iloc[-1].reindex(use_assets)
        vol = hist_ret.tail(126).std(ddof=1).replace(0.0, np.nan)
        mom_z = (mom - mom.mean()) / (mom.std(ddof=0) if mom.std(ddof=0) > 1e-12 else 1.0)
        vol_z = (vol - vol.mean()) / (vol.std(ddof=0) if vol.std(ddof=0) > 1e-12 else 1.0)
        risk_score = (mom_z - 0.25 * vol_z).fillna(0.0).clip(lower=0.0)
        if float(risk_score.sum()) <= 1e-12:
            risk_w = pd.Series(1.0 / len(use_assets), index=use_assets)
        else:
            risk_w = risk_score / float(risk_score.sum())
        neutral_w = pd.Series(1.0 / len(use_assets), index=use_assets)
        defensive_raw = (-vol_z).fillna(0.0)
        for ticker in ["IEF", "TLT", "LQD", "GLD", cash_ticker]:
            if ticker in defensive_raw.index:
                defensive_raw.loc[ticker] += 0.75
        defensive_raw = defensive_raw.clip(lower=0.0)
        defensive_w = (
            defensive_raw / float(defensive_raw.sum())
            if float(defensive_raw.sum()) > 1e-12
            else neutral_w.copy()
        )
        w = (
            float(p["p_risk_on"]) * risk_w
            + float(p["p_neutral"]) * neutral_w
            + float(p["p_defensive"]) * defensive_w
        )
        w = cap_weights(w, max_weight=max_weight).rename(pd.Timestamp(dt))
        weight_rows.append(w)
    return pd.DataFrame(weight_rows).fillna(0.0)


__all__ = [
    "assemble_forecasting_table",
    "add_forecasting_feature_upgrades",
    "asset_return_features",
    "build_asset_feature_block",
    "build_cross_asset_feature_block",
    "build_fci_feature_block",
    "breadth",
    "clean_feature_columns",
    "dispersion",
    "drawdown_change",
    "drawdown_level",
    "ex_ante_vol",
    "feature_vif",
    "feature_availability_by_date",
    "fci_feature_frame",
    "forward_excess_return",
    "future_return",
    "context_return_features",
    "pca_tables",
    "realized_vol",
    "regime_probability_features",
    "relative_return",
    "rolling_avg_corr",
    "rolling_pair_corr",
    "skip_return",
    "total_return",
    "trim_feature_table_by_availability",
    "vol_scaled_return",
]
