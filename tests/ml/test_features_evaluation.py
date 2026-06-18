from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from quantfinlab.ml import evaluation, features
from tests.synthetic.generators import price_panel, volume_panel

pytestmark = pytest.mark.ml


ASSETS = ("SPY", "QQQ", "IWM", "IEF", "TLT", "GLD", "HYG", "LQD", "SHY")


def _market_panel(n: int = 360) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = price_panel(n=n, assets=ASSETS)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    volume = volume_panel(n=n, assets=ASSETS)
    return close, returns, volume


def test_return_volatility_and_cross_sectional_feature_primitives() -> None:
    close, returns, _ = _market_panel()
    spy = close["SPY"]

    total = features.total_return(spy, 21)
    future = features.future_return(spy, 10)
    excess = features.forward_excess_return(spy, horizon=10, rf_daily=0.0001)
    sigma = features.ex_ante_vol(returns["SPY"], lookback=63, horizon=21)
    scaled = features.vol_scaled_return(future, sigma, clip=4.0)
    pair_corr = features.rolling_pair_corr(returns, "SPY", "QQQ", window=63)
    avg_corr = features.rolling_avg_corr(returns[["SPY", "QQQ", "IWM"]], window=63)
    vif = features.feature_vif(returns[["SPY", "QQQ", "IWM"]].tail(160))
    explained, loadings = features.pca_tables(returns[["SPY", "QQQ", "IWM"]].tail(160), n_components=2)

    assert total.iloc[21] == pytest.approx(spy.iloc[21] / spy.iloc[0] - 1.0)
    assert future.dropna().iloc[0] == pytest.approx(spy.iloc[10] / spy.iloc[0] - 1.0)
    assert excess.dropna().abs().mean() > 0.0
    assert sigma.dropna().gt(0.0).all()
    assert scaled.dropna().between(-4.0, 4.0).all()
    assert features.relative_return(close["SPY"], close["SHY"], 21).notna().any()
    assert features.realized_vol(returns["SPY"], 63).dropna().gt(0.0).all()
    assert features.drawdown_level(spy, 126).dropna().le(0.0).all()
    assert features.drawdown_change(spy, drawdown_window=126, change_window=21).notna().any()
    assert pair_corr.dropna().between(-1.0, 1.0).all()
    assert avg_corr.dropna().between(-1.0, 1.0).all()
    assert features.breadth(close, 21, ASSETS[:-1]).dropna().between(0.0, 1.0).all()
    assert features.dispersion(close, 21, ASSETS[:-1]).dropna().ge(0.0).all()
    assert {"r2", "vif"}.issubset(vif.columns)
    assert explained["cumulative"].iloc[-1] <= 1.0 + 1e-12
    assert loadings.shape == (3, 2)


def test_feature_blocks_merge_clean_and_trim_forecasting_table() -> None:
    close, returns, volume = _market_panel()
    asset_list = ASSETS[:-1]
    asset_block = features.build_asset_feature_block(close, volume, returns, assets=asset_list)
    context = features.build_cross_asset_feature_block(close, returns, assets=asset_list, cash_ticker="SHY")
    months = pd.date_range(close.index[0], periods=18, freq="ME")
    nfci = pd.DataFrame(
        {
            "NFCI": np.linspace(-0.5, 0.5, len(months)),
            "Risk": np.linspace(0.1, 0.3, len(months)),
            "Credit": np.linspace(-0.2, 0.2, len(months)),
        },
        index=months,
    )
    fci = features.build_fci_feature_block(nfci=nfci, index=close.index)
    recent_dates = close.index[-14:]
    base = (
        asset_block.loc[asset_block["date"].isin(recent_dates), ["date", "asset", "r_21", "vol_21"]]
        .rename(columns={"r_21": "y", "vol_21": "sigma_21"})
        .dropna()
    )

    table = features.assemble_forecasting_table(base, asset_block, context, fci)
    numeric_features = [
        c
        for c in table.columns
        if c not in {"date", "asset", "y"} and pd.api.types.is_numeric_dtype(table[c])
    ]
    clean = features.clean_feature_columns(table, numeric_features, max_missing=0.80, max_abs_corr=0.995)
    availability = features.feature_availability_by_date(table, clean[:8], target_cols=["y"])
    trimmed, first_date, full_availability = features.trim_feature_table_by_availability(
        table,
        clean[:8],
        target_cols=["y"],
        min_feature_coverage=0.40,
        min_asset_count=len(asset_list),
    )

    assert {"date", "asset", "r_21", "vol_63", "xs_z_r_63"}.issubset(asset_block.columns)
    assert {"breadth_21", "rolling_avg_corr_63", "risk_defensive_spread_63"}.issubset(context.columns)
    assert {"nfci_level", "nfci_risk", "nfci_credit"}.issubset(fci.columns)
    assert {"rank_r_21", "mom_21_sigma21", "rel_r_63_avg", "breadth_21", "nfci_level"}.issubset(table.columns)
    assert len(clean) >= 8
    assert availability["asset_count"].max() == len(asset_list)
    assert not full_availability.empty
    assert first_date is not None
    assert trimmed["date"].min() >= first_date


def _forecast_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    assets = ("A", "B", "C", "D")
    for day_i, dt in enumerate(pd.bdate_range("2024-01-01", periods=40)):
        for asset_i, asset in enumerate(assets):
            signal = 0.10 * day_i + 0.50 * asset_i
            y = 0.02 * signal + 0.01 * np.sin(day_i + asset_i)
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "x1": signal,
                    "x2": asset_i,
                    "y": y,
                    "pred_good": y + 0.001 * np.cos(day_i),
                    "pred_bad": -y,
                    "q10": y - 0.02,
                    "q50": y,
                    "q90": y + 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_forecast_evaluation_tables_rank_buckets_and_walkforward_predictions() -> None:
    frame = _forecast_frame()

    point = evaluation.forecast_metrics(frame, y_col="y", prediction_cols=["pred_good", "pred_bad"])
    ranks = evaluation.rank_metrics(
        frame,
        date_col="date",
        asset_col="asset",
        y_col="y",
        prediction_cols=["pred_good"],
        top_frac=0.25,
    )
    buckets = evaluation.forecast_buckets(frame, date_col="date", y_col="y", score_col="pred_good", n_buckets=4)
    q_metrics = evaluation.quantile_metrics(frame, y_col="y", quantile_sets={"model": ("q10", "q50", "q90")})
    rolling_ic = evaluation.rolling_rank_ic(
        frame,
        date_col="date",
        asset_col="asset",
        y_col="y",
        pred_col="pred_good",
        window=5,
    )
    predictions = evaluation.walkforward_tabular_predictions(
        frame[["x1", "x2"]],
        frame["y"],
        frame["date"],
        frame["asset"],
        refit_dates=[frame["date"].iloc[80], frame["date"].iloc[120]],
        prediction_dates=pd.DatetimeIndex(frame["date"].unique()[20:35]),
        estimators={"linear": LinearRegression()},
        train_window=30,
        horizon=2,
        min_train=20,
        n_jobs=1,
    )

    assert point.loc["pred_good", "MAE"] < point.loc["pred_bad", "MAE"]
    assert ranks.loc["pred_good", "mean_rank_ic"] > 0.95
    assert buckets["mean"].is_monotonic_increasing
    assert q_metrics.loc["model", "coverage_80"] == pytest.approx(1.0)
    assert rolling_ic.dropna().gt(0.95).all()
    assert {"date", "asset", "linear"}.issubset(predictions.columns)
    assert len(predictions) == 60


def test_policy_and_stress_evaluation_summary_tables() -> None:
    dates = pd.bdate_range("2024-01-01", periods=60)
    returns = pd.DataFrame(
        {
            "benchmark": np.sin(np.arange(len(dates))) / 1000.0,
            "strategy": np.sin(np.arange(len(dates)) + 0.2) / 1000.0 + 0.0002,
        },
        index=dates,
    )
    weights = pd.DataFrame(
        {"A": [0.30, 0.40, 0.35], "B": [0.40, 0.30, 0.35], "SHY": [0.30, 0.30, 0.30]},
        index=dates[:3],
    )
    screen = evaluation.feature_screen_table(
        pd.Series({"mom": 0.4, "vol": 0.2}),
        pd.Series({"mom": 1.0, "vol": -3.0}),
        top_n=2,
    )
    active = evaluation.active_performance_table(strategy_returns=returns, benchmark="benchmark")
    diagnostics = evaluation.policy_diagnostics_table(weights_by_strategy={"strategy": weights}, cost_bps=5.0)
    stress = evaluation.stress_active_table(
        strategy_returns=returns,
        benchmark="benchmark",
        windows={"sample": (dates[0], dates[-1])},
    )

    assert screen.index.tolist() == ["mom", "vol"]
    assert active.loc["strategy", "Information Ratio"] > 0.0
    assert diagnostics.loc["strategy", "Avg Risky Exposure"] == pytest.approx(0.70)
    assert ("sample", "strategy") in stress.index
    assert stress.loc[("sample", "strategy"), "Active Return"] > 0.0
