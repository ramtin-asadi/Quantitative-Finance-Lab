from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StateTables:
    dates: pd.DatetimeIndex
    assets: list[str]
    cash_ticker: str
    asset_state: np.ndarray
    global_state: np.ndarray
    prior_weights: np.ndarray
    returns_period: pd.DataFrame
    prior_names: list[str]
    asset_feature_names: list[str]
    global_feature_names: list[str]
    portfolio_feature_names: list[str]
    prior_weight_frames: dict[str, pd.DataFrame]
    active_benchmark: str | None = None
    returns_daily: pd.DataFrame | None = None
    daily_windows: tuple[pd.DataFrame, ...] = ()
    daily_windows_np: tuple[np.ndarray, ...] = ()

    @property
    def n_dates(self) -> int:
        return int(len(self.dates))

    @property
    def n_assets(self) -> int:
        return int(len(self.assets))

    @property
    def n_asset_features(self) -> int:
        return int(self.asset_state.shape[-1])

    @property
    def n_global_features(self) -> int:
        return int(self.global_state.shape[-1])

    @property
    def n_portfolio_features(self) -> int:
        return int(len(self.portfolio_feature_names))

    @property
    def columns(self) -> list[str]:
        return [*self.assets, self.cash_ticker]

    def copy_with(
        self,
        *,
        asset_state: np.ndarray | None = None,
        global_state: np.ndarray | None = None,
        prior_weights: np.ndarray | None = None,
    ) -> StateTables:
        return replace(
            self,
            asset_state=np.array(self.asset_state if asset_state is None else asset_state, copy=True),
            global_state=np.array(self.global_state if global_state is None else global_state, copy=True),
            prior_weights=np.array(self.prior_weights if prior_weights is None else prior_weights, copy=True),
        )

    def period_indices(self, period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None) -> np.ndarray:
        if period is None:
            return np.arange(self.n_dates)
        start, end = period
        mask = (self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end))
        return np.where(mask)[0]


def make_decision_dates(
    index: Sequence[pd.Timestamp | str] | pd.Index,
    *,
    freq: str = "W-FRI",
    min_history_days: int = 756,
) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index)).sort_values().unique()
    if len(idx) == 0:
        return idx
    start_pos = min(max(int(min_history_days), 0), max(len(idx) - 1, 0))
    eligible = idx[start_pos:]
    if len(eligible) == 0:
        return pd.DatetimeIndex([])
    sampled = pd.Series(eligible, index=eligible).resample(freq).last().dropna()
    out = pd.DatetimeIndex(sampled.astype("datetime64[ns]").to_numpy()).sort_values().unique()
    return out[out.isin(idx)]


def _slug(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace(".", "_")
    )


def _as_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _align_weight_frame(
    weights: pd.DataFrame | Mapping[str, float],
    *,
    decision_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    cash_ticker: str | None = None,
    fallback: str = "equal",
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).sort_values().unique()
    asset_list = list(assets)
    cols = asset_list + ([cash_ticker] if cash_ticker is not None and cash_ticker not in asset_list else [])
    if len(dates) == 0:
        return pd.DataFrame(columns=cols, dtype=float)
    if fallback == "zero":
        base = pd.DataFrame(0.0, index=dates, columns=cols, dtype=float)
    else:
        base = pd.DataFrame(1.0 / max(len(asset_list), 1), index=dates, columns=asset_list, dtype=float)
        if cash_ticker is not None and cash_ticker not in base.columns:
            base[cash_ticker] = 0.0
    W = pd.DataFrame(weights).copy()
    if W.empty:
        return base.reindex(columns=cols).fillna(0.0)
    W.index = pd.to_datetime(W.index)
    W = W.sort_index().reindex(columns=cols)
    W = W.reindex(W.index.union(dates)).sort_index().ffill().reindex(dates)
    W = W.fillna(base)
    risky_sum = W[asset_list].sum(axis=1).clip(lower=0.0)
    total_sum = W.sum(axis=1)
    bad = total_sum.abs() <= 1e-12
    if bool(bad.any()):
        W.loc[bad, asset_list] = 1.0 / max(len(asset_list), 1)
        if cash_ticker is not None and cash_ticker in W.columns:
            W.loc[bad, cash_ticker] = 0.0
        total_sum = W.sum(axis=1)
    W = W.div(total_sum.replace(0.0, np.nan), axis=0).fillna(base)
    if cash_ticker is not None and cash_ticker in W.columns:
        risky_sum = W[asset_list].sum(axis=1).clip(0.0, 1.0)
        W[asset_list] = W[asset_list].div(risky_sum.replace(0.0, np.nan), axis=0).fillna(0.0)
        W[asset_list] = W[asset_list].mul(risky_sum, axis=0)
        W[cash_ticker] = 1.0 - W[asset_list].sum(axis=1)
    return W.reindex(columns=cols).fillna(0.0)


def align_weight_priors(
    weights: Mapping[str, pd.DataFrame | Mapping[str, float]],
    *,
    decision_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    cash_ticker: str | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        str(name): _align_weight_frame(
            frame,
            decision_dates=decision_dates,
            assets=assets,
            cash_ticker=cash_ticker,
        )
        for name, frame in weights.items()
    }


def _period_returns(
    returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    R = _as_datetime_index(returns).reindex(columns=list(columns)).fillna(0.0)
    idx = pd.DatetimeIndex(R.index)
    rows: list[pd.Series] = []
    row_dates: list[pd.Timestamp] = []
    for i, dt in enumerate(decision_dates):
        start_pos = int(idx.searchsorted(pd.Timestamp(dt), side="right"))
        if start_pos >= len(idx):
            continue
        if i + 1 < len(decision_dates):
            end_pos = int(idx.searchsorted(pd.Timestamp(decision_dates[i + 1]), side="right"))
        else:
            end_pos = min(start_pos + 5, len(idx))
        if end_pos <= start_pos:
            continue
        window = R.iloc[start_pos:end_pos].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        period = (1.0 + window).prod(axis=0) - 1.0
        rows.append(period.rename(pd.Timestamp(dt)))
        row_dates.append(pd.Timestamp(dt))
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(row_dates))
    return out.reindex(columns=list(columns)).fillna(0.0)


def _period_daily_windows(
    returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    *,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, tuple[pd.DataFrame, ...]]:
    R = _as_datetime_index(returns).reindex(columns=list(columns)).fillna(0.0)
    idx = pd.DatetimeIndex(R.index)
    rows: list[pd.Series] = []
    row_dates: list[pd.Timestamp] = []
    windows: list[pd.DataFrame] = []
    for i, dt in enumerate(decision_dates):
        start_pos = int(idx.searchsorted(pd.Timestamp(dt), side="right"))
        if start_pos >= len(idx):
            continue
        if i + 1 < len(decision_dates):
            end_pos = int(idx.searchsorted(pd.Timestamp(decision_dates[i + 1]), side="right"))
        else:
            end_pos = min(start_pos + 5, len(idx))
        if end_pos <= start_pos:
            continue
        window = R.iloc[start_pos:end_pos].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        period = (1.0 + window).prod(axis=0) - 1.0
        rows.append(period.rename(pd.Timestamp(dt)))
        row_dates.append(pd.Timestamp(dt))
        windows.append(window.reindex(columns=list(columns)).copy())
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(row_dates)).reindex(columns=list(columns)).fillna(0.0)
    return out, tuple(windows)


def _pivot_asset_feature(
    long_frame: pd.DataFrame,
    *,
    feature: str,
    dates: pd.DatetimeIndex,
    assets: Sequence[str],
) -> pd.DataFrame:
    if long_frame.empty or feature not in long_frame.columns:
        return pd.DataFrame(0.0, index=dates, columns=list(assets), dtype=float)
    f = pd.DataFrame(long_frame[["date", "asset", feature]]).copy()
    f["date"] = pd.to_datetime(f["date"])
    piv = f.pivot_table(index="date", columns="asset", values=feature, aggfunc="last")
    piv = piv.sort_index().reindex(columns=list(assets))
    piv = piv.reindex(piv.index.union(dates)).sort_index().ffill().reindex(dates)
    return piv.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _numeric_feature_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    cols = []
    for col in frame.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(str(col))
    return cols


LEAKY_FORECAST_COLUMNS = {
    "y",
    "target",
    "label",
    "future_return",
    "fwd",
    "fwd_return",
    "forward_return",
    "r_ex_21",
    "z_21",
    "y_alpha",
}


def _combine_global_features(
    frames: Sequence[pd.DataFrame | None],
    *,
    dates: pd.DatetimeIndex,
    returns: pd.DataFrame,
    assets: Sequence[str],
    benchmark_ticker: str | None,
) -> pd.DataFrame:
    parts = []
    for frame in frames:
        if frame is None or pd.DataFrame(frame).empty:
            continue
        f = pd.DataFrame(frame).copy()
        if "date" in f.columns:
            f["date"] = pd.to_datetime(f["date"])
            f = f.drop_duplicates("date").set_index("date")
        f.index = pd.to_datetime(f.index)
        parts.append(f.sort_index())
    R = _as_datetime_index(returns)
    use_bench = benchmark_ticker if benchmark_ticker in R.columns else (list(assets)[0] if assets else R.columns[0])
    extra = pd.DataFrame(index=R.index)
    extra["rolling_market_vol_63"] = R[use_bench].rolling(63, min_periods=21).std(ddof=1) * np.sqrt(252.0)
    if len(assets) > 1:
        corr_vals = []
        for i, left in enumerate(assets):
            if left not in R.columns:
                continue
            for right in list(assets)[i + 1 :]:
                if right in R.columns:
                    corr_vals.append(R[left].rolling(63, min_periods=42).corr(R[right]))
        extra["rolling_avg_corr_63_state"] = pd.concat(corr_vals, axis=1).mean(axis=1) if corr_vals else 0.0
    nav = (1.0 + R[use_bench].fillna(0.0)).cumprod()
    extra["benchmark_drawdown"] = nav / nav.cummax() - 1.0
    parts.append(extra)
    if parts:
        out = pd.concat(parts, axis=1)
        out = out.loc[:, ~out.columns.duplicated()].copy()
    else:
        out = pd.DataFrame(index=dates)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().reindex(out.index.union(dates)).ffill().reindex(dates)
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return out.fillna(0.0)


def build_state_tables(
    *,
    asset_features: pd.DataFrame,
    context_features: pd.DataFrame | None = None,
    fci_features: pd.DataFrame | None = None,
    regime_features: pd.DataFrame | None = None,
    vix_features: pd.DataFrame | None = None,
    forecast_features: pd.DataFrame | None = None,
    prior_weights: Mapping[str, pd.DataFrame] | None = None,
    include_prior_weights: bool = False,
    returns: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    cash_ticker: str = "SHY",
    benchmark_ticker: str | None = "SPY",
    active_benchmark: str = "Forecast-Gated MaxSharpe",
) -> StateTables:
    asset_list = list(assets)
    columns = asset_list + ([cash_ticker] if cash_ticker not in asset_list else [])
    dates_raw = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).sort_values().unique()
    period, daily_windows = _period_daily_windows(returns, dates_raw, columns=columns)
    dates = pd.DatetimeIndex(period.index).sort_values().unique()
    if len(dates) == 0:
        raise ValueError("No decision dates align to the return index.")

    asset_long = pd.DataFrame(asset_features).copy()
    if "date" not in asset_long.columns or "asset" not in asset_long.columns:
        raise ValueError("asset_features must contain date and asset columns.")
    asset_long["date"] = pd.to_datetime(asset_long["date"])
    asset_long = asset_long[asset_long["asset"].isin(asset_list)]

    if forecast_features is not None and not pd.DataFrame(forecast_features).empty:
        forecast = pd.DataFrame(forecast_features).copy()
        forecast["date"] = pd.to_datetime(forecast["date"])
        forecast = forecast[forecast["asset"].isin(asset_list)]
        forecast_exclude = {"date", "asset"}
        forecast_exclude |= {c for c in forecast.columns if str(c).lower() in LEAKY_FORECAST_COLUMNS}
        keep = ["date", "asset", *_numeric_feature_columns(forecast, exclude=forecast_exclude)]
        asset_long = asset_long.merge(
            forecast[keep].drop_duplicates(["date", "asset"]),
            on=["date", "asset"],
            how="left",
            suffixes=("", "_forecast"),
        )

    feature_cols = _numeric_feature_columns(asset_long, exclude={"date", "asset"})
    blocks = []
    names = []
    for feature in feature_cols:
        blocks.append(_pivot_asset_feature(asset_long, feature=feature, dates=dates, assets=asset_list))
        names.append(feature)

    aligned_priors = align_weight_priors(
        prior_weights or {},
        decision_dates=dates,
        assets=asset_list,
        cash_ticker=cash_ticker,
    )
    prior_names = list(aligned_priors)
    prior_arrays = []
    for name, frame in aligned_priors.items():
        risky = frame.reindex(index=dates, columns=asset_list).fillna(0.0)
        if include_prior_weights:
            blocks.append(risky)
            names.append(f"prior_{_slug(name)}")
        prior_arrays.append(risky.to_numpy(dtype=np.float32))

    prev_weight = pd.DataFrame(0.0, index=dates, columns=asset_list, dtype=float)
    blocks.append(prev_weight)
    names.append("previous_weight")

    if not blocks:
        blocks.append(pd.DataFrame(0.0, index=dates, columns=asset_list, dtype=float))
        names.append("constant")
    asset_state = np.stack([b.reindex(index=dates, columns=asset_list).to_numpy(dtype=np.float32) for b in blocks], axis=2)
    asset_state = np.nan_to_num(asset_state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    global_frame = _combine_global_features(
        [context_features, fci_features, regime_features],
        dates=dates,
        returns=returns,
        assets=asset_list,
        benchmark_ticker=benchmark_ticker,
    )
    if include_prior_weights:
        for name, frame in aligned_priors.items():
            clean = _slug(name)
            W = frame.reindex(index=dates, columns=columns).ffill().fillna(0.0)
            risky = W.reindex(columns=asset_list).fillna(0.0)
            global_frame[f"prior_{clean}_risky_exposure"] = risky.sum(axis=1)
            if cash_ticker in W.columns:
                global_frame[f"prior_{clean}_cash"] = W[cash_ticker].astype(float)
    if vix_features is not None and not pd.DataFrame(vix_features).empty:
        vix_frame = pd.DataFrame(vix_features).copy()
        vix_frame.index = pd.to_datetime(vix_frame.index)
        vix_frame = vix_frame.sort_index()
        vix_frame = vix_frame.reindex(vix_frame.index.union(dates)).ffill().reindex(dates)
        vix_frame = vix_frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for col in vix_frame.columns:
            global_frame[str(col)] = vix_frame[col].to_numpy(dtype=float)
    global_state = global_frame.to_numpy(dtype=np.float32)
    global_state = np.nan_to_num(global_state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if prior_arrays:
        prior_arr = np.stack(prior_arrays, axis=2).astype(np.float32)
    else:
        prior_arr = np.zeros((len(dates), len(asset_list), 0), dtype=np.float32)

    portfolio_names = [f"prev_{a}" for a in asset_list] + [
        "previous_risky_exposure",
        "previous_turnover",
        "current_drawdown",
        "rolling_portfolio_vol",
        "recent_portfolio_return",
    ]
    return StateTables(
        dates=dates,
        assets=asset_list,
        cash_ticker=cash_ticker,
        asset_state=asset_state,
        global_state=global_state,
        prior_weights=prior_arr,
        returns_period=period.reindex(index=dates, columns=columns).fillna(0.0),
        prior_names=prior_names,
        asset_feature_names=names,
        global_feature_names=list(global_frame.columns),
        portfolio_feature_names=portfolio_names,
        prior_weight_frames=aligned_priors,
        active_benchmark=active_benchmark if active_benchmark in aligned_priors else None,
        returns_daily=_as_datetime_index(returns).reindex(columns=columns).fillna(0.0),
        daily_windows=daily_windows,
        daily_windows_np=tuple(
            np.ascontiguousarray(w.reindex(columns=columns).to_numpy(dtype=np.float64))
            for w in daily_windows
        ),
    )


def portfolio_state_vector(
    w_prev: Sequence[float] | np.ndarray,
    *,
    previous_turnover: float = 0.0,
    current_drawdown: float = 0.0,
    rolling_portfolio_vol: float = 0.0,
    recent_portfolio_return: float = 0.0,
) -> np.ndarray:
    w = np.asarray(w_prev, dtype=np.float32).reshape(-1)
    risky = float(np.clip(w[:-1].sum() if len(w) > 1 else w.sum(), 0.0, 1.0))
    tail = np.asarray(
        [
            risky,
            float(previous_turnover),
            float(current_drawdown),
            float(rolling_portfolio_vol),
            float(recent_portfolio_return),
        ],
        dtype=np.float32,
    )
    return np.concatenate([w[:-1] if len(w) > 1 else w, tail]).astype(np.float32)


def action_to_weights(
    action,
    *,
    min_exposure: float = 0.50,
    max_exposure: float = 1.00,
    max_weight: float = 0.35,
    active_scale: float = 0.25,
    action_mode: str = "softmax",
):
    mode = str(action_mode).lower().replace("-", "_")
    if hasattr(action, "detach"):
        import torch

        raw = action
        logits = raw[..., :-1]
        exposure_logit = raw[..., -1]
        exposure = float(min_exposure) + (float(max_exposure) - float(min_exposure)) * torch.sigmoid(exposure_logit)
        if mode in {"softmax", "absolute"}:
            risky = torch.softmax(logits, dim=-1) * exposure.unsqueeze(-1)
        else:
            n = logits.shape[-1]
            base = torch.full_like(logits, 1.0 / max(int(n), 1))
            tilt = torch.tanh(logits)
            tilt = tilt - tilt.mean(dim=-1, keepdim=True)
            risky = base * exposure.unsqueeze(-1) + float(active_scale) * tilt
            risky = risky.clamp_min(0.0)
            risky_sum = risky.sum(dim=-1, keepdim=True)
            risky = torch.where(risky_sum > 1e-8, risky / risky_sum * exposure.unsqueeze(-1), base * exposure.unsqueeze(-1))
        cap = float(max_weight)
        for _ in range(8):
            capped = torch.clamp(risky, max=cap)
            excess = (risky - capped).clamp_min(0.0).sum(dim=-1, keepdim=True)
            room = (cap - capped).clamp_min(0.0)
            room_sum = room.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            risky = capped + excess * room / room_sum
            if not bool((risky > cap + 1e-7).any()):
                break
        risky_sum = risky.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        cash = 1.0 - risky_sum
        return torch.cat([risky, cash], dim=-1)

    raw = np.asarray(action, dtype=float)
    logits = raw[..., :-1]
    exposure = float(min_exposure) + (float(max_exposure) - float(min_exposure)) / (1.0 + np.exp(-raw[..., -1]))
    if mode in {"softmax", "absolute"}:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        risky = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True) * np.expand_dims(exposure, axis=-1)
    else:
        n = logits.shape[-1]
        base = np.full_like(logits, 1.0 / max(int(n), 1), dtype=float)
        tilt = np.tanh(logits)
        tilt = tilt - tilt.mean(axis=-1, keepdims=True)
        risky = base * np.expand_dims(exposure, axis=-1) + float(active_scale) * tilt
        risky = np.maximum(risky, 0.0)
        risky_sum = risky.sum(axis=-1, keepdims=True)
        risky = np.where(risky_sum > 1e-12, risky / np.maximum(risky_sum, 1e-12) * np.expand_dims(exposure, axis=-1), base * np.expand_dims(exposure, axis=-1))
    cap = float(max_weight)
    for _ in range(20):
        capped = np.minimum(risky, cap)
        excess = np.maximum(risky - capped, 0.0).sum(axis=-1, keepdims=True)
        room = np.maximum(cap - capped, 0.0)
        room_sum = np.maximum(room.sum(axis=-1, keepdims=True), 1e-12)
        risky = capped + excess * room / room_sum
        if not np.any(risky > cap + 1e-10):
            break
    cash = 1.0 - risky.sum(axis=-1, keepdims=True)
    return np.concatenate([risky, cash], axis=-1)


def portfolio_turnover(w_new: Sequence[float], w_prev: Sequence[float]) -> float:
    new = np.asarray(w_new, dtype=float).reshape(-1)
    prev = np.asarray(w_prev, dtype=float).reshape(-1)
    if prev.size and float(prev.sum()) > 1e-12:
        prev = prev / float(prev.sum())
    return 0.5 * float(np.abs(new - prev).sum())


def portfolio_step_return(
    w_new: Sequence[float],
    r_next: Sequence[float],
    *,
    w_prev: Sequence[float] | None = None,
    cost_bps: float = 10.0,
) -> tuple[float, float, float]:
    w = np.asarray(w_new, dtype=float).reshape(-1)
    r = np.asarray(r_next, dtype=float).reshape(-1)
    turnover = portfolio_turnover(w, w_prev) if w_prev is not None else 0.0
    cost = float(cost_bps) / 10000.0 * turnover
    gross = float(np.dot(w, r))
    return gross - cost, turnover, cost


def portfolio_step_path_return(
    w_new: Sequence[float],
    daily_returns: pd.DataFrame | np.ndarray,
    *,
    w_prev: Sequence[float] | None = None,
    cost_bps: float = 10.0,
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror ``run_weights_backtest(..., weight_timing='next_close')`` for one action window."""
    w = np.asarray(w_new, dtype=float).reshape(-1)
    if w.size == 0:
        raise ValueError("w_new is empty.")
    total = float(w.sum())
    if total > 1e-12:
        w = w / total
    if isinstance(daily_returns, pd.DataFrame):
        R = daily_returns.to_numpy(dtype=float)
    else:
        R = np.asarray(daily_returns, dtype=float)
    if R.ndim == 1:
        R = R.reshape(1, -1)
    if R.shape[1] != w.size:
        raise ValueError("daily return window column count must match weights.")

    if w_prev is None:
        prev = np.zeros_like(w)
    else:
        prev = np.asarray(w_prev, dtype=float).reshape(-1)
        if prev.size != w.size:
            prev = np.resize(prev, w.size)
        if float(prev.sum()) > 1e-12:
            prev = prev / float(prev.sum())
    turnover = 0.5 * float(np.abs(w - prev).sum())
    cost = float(cost_bps) / 10000.0 * turnover

    net = max(1.0 - cost, 1e-12)
    weights = w.copy()
    daily_net_returns: list[float] = []
    nav_path: list[float] = []
    prev_net = 1.0
    for k, r_today in enumerate(R):
        r = np.nan_to_num(np.asarray(r_today, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        port_ret = float(np.dot(weights, r))
        net *= 1.0 + port_ret
        daily_net_returns.append(float(net / prev_net - 1.0) if k == 0 else port_ret)
        nav_path.append(float(net))
        grossed = weights * (1.0 + r)
        gross_sum = float(grossed.sum())
        weights = grossed / gross_sum if gross_sum > 0 and np.isfinite(gross_sum) else np.zeros_like(weights)
        prev_net = net
    if len(nav_path) == 0:
        nav_path.append(float(net))
    return (
        float(net - 1.0),
        float(turnover),
        float(cost),
        weights.astype(float),
        np.asarray(daily_net_returns, dtype=float),
        np.asarray(nav_path, dtype=float),
    )


def _policy_weights(policy, asset_x, global_x, portfolio_x, *, device=None, deterministic: bool = True):
    import torch

    dev = device or next(policy.parameters()).device
    ax = torch.as_tensor(asset_x[None, :, :], dtype=torch.float32, device=dev)
    gx = torch.as_tensor(global_x[None, :], dtype=torch.float32, device=dev)
    px = torch.as_tensor(portfolio_x[None, :], dtype=torch.float32, device=dev)
    with torch.no_grad():
        if hasattr(policy, "deterministic_weights"):
            w = policy.deterministic_weights(ax, gx, px)
        else:
            out = policy(ax, gx, px)
            raw = out[0] if isinstance(out, tuple) else out
            w = action_to_weights(
                raw,
                min_exposure=getattr(policy, "min_exposure", 0.70),
                max_exposure=getattr(policy, "max_exposure", 1.00),
                max_weight=getattr(policy, "max_weight", 0.35),
            )
    return w.detach().cpu().numpy().reshape(-1)


def rollout_weights(
    policy,
    state: StateTables,
    *,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    device=None,
    initial_weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    idx = state.period_indices(period)
    if len(idx) == 0:
        return pd.DataFrame(columns=state.columns)
    w_prev = (
        np.asarray(initial_weights, dtype=float)
        if initial_weights is not None
        else np.r_[np.repeat(1.0 / state.n_assets, state.n_assets), 0.0]
    )
    nav = 1.0
    peak = 1.0
    returns_hist: list[float] = []
    prev_turnover = 0.0
    rows = []
    hidden = None
    recurrent = policy.__class__.__name__.lower().startswith("recurrent")
    for j in idx:
        asset_x = np.array(state.asset_state[j], copy=True)
        if "previous_weight" in state.asset_feature_names:
            asset_x[:, state.asset_feature_names.index("previous_weight")] = w_prev[: state.n_assets]
        rolling_vol = float(np.std(returns_hist[-13:], ddof=1) * np.sqrt(52.0)) if len(returns_hist) > 2 else 0.0
        p_state = portfolio_state_vector(
            w_prev,
            previous_turnover=prev_turnover,
            current_drawdown=nav / peak - 1.0,
            rolling_portfolio_vol=rolling_vol,
            recent_portfolio_return=returns_hist[-1] if returns_hist else 0.0,
        )
        if recurrent:
            import torch

            dev = device or next(policy.parameters()).device
            ax = torch.as_tensor(asset_x[None, :, :], dtype=torch.float32, device=dev)
            gx = torch.as_tensor(state.global_state[j][None, :], dtype=torch.float32, device=dev)
            px = torch.as_tensor(p_state[None, :], dtype=torch.float32, device=dev)
            with torch.no_grad():
                _, _, _, _, weights_t, hidden = policy.act(ax, gx, px, hidden=hidden, deterministic=True)
                hidden = tuple(h.detach() for h in hidden)
            w = weights_t.detach().cpu().numpy().reshape(-1)
        else:
            w = _policy_weights(policy, asset_x, state.global_state[j], p_state, device=device)
        rows.append(pd.Series(w, index=state.columns, name=state.dates[j]))
        if state.daily_windows:
            ret, prev_turnover, _, w_end, _, _ = portfolio_step_path_return(
                w,
                state.daily_windows[int(j)],
                w_prev=w_prev,
                cost_bps=10.0,
            )
        else:
            ret, prev_turnover, _ = portfolio_step_return(w, state.returns_period.iloc[j].to_numpy(), w_prev=w_prev)
            w_end = w
        nav *= 1.0 + ret
        peak = max(peak, nav)
        returns_hist.append(ret)
        w_prev = w_end
    return pd.DataFrame(rows).fillna(0.0)


def policy_weight_frame(
    *,
    policy,
    state: StateTables,
    period: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
    assets: Sequence[str] | None = None,
    cash_ticker: str | None = None,
    device=None,
) -> pd.DataFrame:
    W = rollout_weights(policy, state, period=period, device=device)
    cols = list(assets or state.assets)
    cash = cash_ticker or state.cash_ticker
    if cash in W.columns and cash not in cols:
        cols.append(cash)
    return W.reindex(columns=cols).fillna(0.0)


def tcnrank_forecast_features(
    *,
    close: pd.DataFrame,
    returns: pd.DataFrame,
    assets: Sequence[str],
    volume: pd.DataFrame | None = None,
    cash_ticker: str = "SHY",
    benchmark_ticker: str = "SPY",
    model_dir: str | Path,
    decision_dates: Sequence[pd.Timestamp | str] | None = None,
    fci_features: pd.DataFrame | None = None,
    regime_features: pd.DataFrame | None = None,
    rf_daily: float = 0.0,
    horizon: int = 21,
    vol_lookback: int = 63,
    lookback: int = 21,
    train_end: str | pd.Timestamp = "2017-12-31",
    valid_start: str | pd.Timestamp = "2018-01-01",
    valid_end: str | pd.Timestamp = "2020-12-31",
    epochs: int = 24,
    batch_size: int = 512,
    force_retrain: bool = False,
    device=None,
) -> pd.DataFrame:
    """Train or load the compact TCNRank forecast table used by Project 20."""
    from quantfinlab.ml.features import (
        assemble_forecasting_table,
        build_asset_feature_block,
        build_cross_asset_feature_block,
        clean_feature_columns,
        trim_feature_table_by_availability,
    )
    from quantfinlab.ml.sequence_models import TcnForecast, torch_predictions, train_torch_model

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    cache_path = model_path / "tcnrank_forecast_features.pkl"
    if cache_path.exists() and not bool(force_retrain):
        cached = pd.read_pickle(cache_path)
        if {"date", "asset", "tcn_alpha", "tcn_rank"}.issubset(cached.columns):
            have_assets = set(cached["asset"].astype(str).unique())
            if set(map(str, assets)).issubset(have_assets):
                return cached[cached["asset"].isin(list(assets))].copy()

    asset_list = list(assets)
    px = _as_datetime_index(close).ffill(limit=3)
    R = _as_datetime_index(returns).reindex(px.index).fillna(0.0)
    r_log = np.log(px[asset_list] / px[asset_list].shift(1))
    fwd = np.log(px[asset_list].shift(-int(horizon)) / px[asset_list])
    sigma = r_log.rolling(int(vol_lookback), min_periods=int(vol_lookback)).std(ddof=1) * np.sqrt(int(horizon))
    z = (fwd - int(horizon) * np.log1p(float(rf_daily))).div(sigma.replace(0.0, np.nan)).clip(-4.0, 4.0)
    y_alpha = z.sub(z.median(axis=1), axis=0).clip(-4.0, 4.0)
    target = (
        pd.concat({"sigma_21": sigma, "z_21": z, "y_alpha": y_alpha}, axis=1)
        .stack(level=1)
        .rename_axis(["date", "asset"])
        .reset_index()
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["sigma_21", "z_21", "y_alpha"])
    )
    counts = target.groupby("date")["asset"].nunique()
    target = target[target["date"].isin(counts[counts.eq(len(asset_list))].index)].copy()

    asset_x = build_asset_feature_block(px, volume, R, assets=asset_list, rf_daily=rf_daily)
    context_x = build_cross_asset_feature_block(
        px,
        R,
        assets=asset_list,
        cash_ticker=cash_ticker,
        benchmark_ticker=benchmark_ticker,
    )
    data = assemble_forecasting_table(target, asset_x, context_x, fci_features, regime_features)
    data = data[data["asset"].isin(asset_list)].replace([np.inf, -np.inf], np.nan)
    data = data.sort_values(["date", "asset"]).reset_index(drop=True)
    feature_cols = [c for c in data.columns if c not in {"date", "asset", "sigma_21", "z_21", "y_alpha"}]
    stage1 = clean_feature_columns(data, feature_cols, max_missing=0.35, max_abs_corr=0.985)
    data, _, _ = trim_feature_table_by_availability(
        data,
        stage1,
        target_cols=["y_alpha", "sigma_21"],
        min_feature_coverage=0.70,
        min_asset_count=len(asset_list),
        min_target_complete=1.0,
    )
    features = clean_feature_columns(data, feature_cols, max_missing=0.35, max_abs_corr=0.985)
    for col in features:
        data[col] = data.groupby("asset", sort=False)[col].transform(lambda s: s.ffill())
    med = data.loc[pd.to_datetime(data["date"]) <= pd.Timestamp(train_end), features].median()
    data[features] = data[features].fillna(med).fillna(0.0)
    asset_code = {asset: i for i, asset in enumerate(asset_list)}
    data["asset_id"] = data["asset"].map(asset_code).astype(int)

    import torch

    torch.manual_seed(42)
    model = TcnForecast(
        n_features=len(features),
        n_assets=len(asset_list),
        embedding_dim=4,
        channels=(64, 64, 64),
        kernel_size=3,
        output_size=1,
        dropout=0.08,
    )
    model_file = model_path / "tcnrank.pt"
    if model_file.exists() and not bool(force_retrain):
        model.load_state_dict(torch.load(model_file, map_location=device or "cpu"))
    else:
        model, history = train_torch_model(
            model,
            data=data,
            features=features,
            target="y_alpha",
            asset_col="asset_id",
            date_col="date",
            lookback=int(lookback),
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            epochs=int(epochs),
            batch_size=int(batch_size),
            lr=7.5e-4,
            weight_decay=1e-4,
            patience=max(5, min(10, int(epochs) // 2)),
            loss_name="huber",
            early_stop_metric="composite",
            device=device,
        )
        torch.save(model.state_dict(), model_file)
        history.to_csv(model_path / "tcnrank_history.csv", index=False)

    pred = torch_predictions(
        model,
        data=data,
        features=features,
        asset_col="asset_id",
        date_col="date",
        lookback=int(lookback),
        batch_size=int(batch_size),
        device=device,
    )
    out = data.loc[pred.index, ["date", "asset", "sigma_21", "z_21", "y_alpha"]].copy()
    out["tcn_alpha_raw"] = pd.Series(pred, index=pred.index).to_numpy(dtype=float)
    out["tcn_alpha"] = out["tcn_alpha_raw"] - out.groupby("date")["tcn_alpha_raw"].transform("median")
    out["tcn_rank"] = out.groupby("date")["tcn_alpha"].rank(pct=True)
    scale = out.groupby("date")["tcn_alpha"].transform(lambda s: s.abs().median())
    out["tcn_confidence"] = out["tcn_alpha"].abs().div(scale.replace(0.0, np.nan)).clip(0.0, 3.0).fillna(0.0) / 3.0
    if decision_dates is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(list(decision_dates))).sort_values().unique()
        out = out[out["date"].isin(dates)].copy()
    out = out.sort_values(["date", "asset"]).reset_index(drop=True)
    out.to_pickle(cache_path)
    return out


def blend_policy_weights(
    weight_frames: Mapping[str, pd.DataFrame],
    *,
    blend_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Blend several policy weight frames into one renormalised ensemble frame.

    With ``blend_weights=None`` the policies are averaged equally; otherwise each
    frame is scaled by its (clipped, non-negative) blend weight before the rows
    are renormalised to sum to one.
    """
    frames = {str(k): pd.DataFrame(v).copy() for k, v in weight_frames.items() if v is not None and not pd.DataFrame(v).empty}
    if not frames:
        return pd.DataFrame()
    if blend_weights is None:
        blend = {k: 1.0 for k in frames}
    else:
        blend = {k: max(0.0, float(blend_weights.get(k, 0.0))) for k in frames}
    total = sum(blend.values())
    if total <= 0:
        blend = {k: 1.0 for k in frames}
        total = float(len(frames))
    columns = sorted({c for f in frames.values() for c in f.columns})
    index = sorted({d for f in frames.values() for d in f.index})
    blended = pd.DataFrame(0.0, index=pd.DatetimeIndex(pd.to_datetime(index)), columns=columns, dtype=float)
    for name, frame in frames.items():
        f = frame.reindex(index=blended.index, columns=columns).fillna(0.0)
        blended = blended + (blend[name] / total) * f
    row_sum = blended.sum(axis=1).replace(0.0, np.nan)
    return blended.div(row_sum, axis=0).fillna(0.0)


__all__ = [
    "StateTables",
    "action_to_weights",
    "align_weight_priors",
    "blend_policy_weights",
    "build_state_tables",
    "make_decision_dates",
    "policy_weight_frame",
    "portfolio_state_vector",
    "portfolio_step_path_return",
    "portfolio_step_return",
    "portfolio_turnover",
    "rollout_weights",
    "tcnrank_forecast_features",
]
