from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quantfinlab.portfolio.covariance import ledoit_wolf_covariance, make_psd


def normalize_long_only(
    weights: pd.Series,
    *,
    min_weight: float = 0.0,
    max_weight: float | Mapping[str, float] | pd.Series | None = None,
) -> pd.Series:
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    w = w.clip(lower=float(min_weight))
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0 / len(w), index=w.index, dtype=float)
    else:
        w = w / float(w.sum())
    if max_weight is None:
        return w
    return cap_weights(w, max_weight=max_weight, min_weight=min_weight)


def _cap_series(index: pd.Index, max_weight: float | Mapping[str, float] | pd.Series) -> pd.Series:
    if isinstance(max_weight, (Mapping, pd.Series)):
        caps = pd.Series(max_weight, dtype=float).reindex(index).fillna(1.0)
    else:
        caps = pd.Series(float(max_weight), index=index, dtype=float)
    if float(caps.sum()) < 1.0:
        caps[:] = max(float(caps.max()), 1.0 / max(len(caps), 1))
    return caps.clip(lower=0.0)


def cap_weights(
    weights: pd.Series | pd.DataFrame,
    *,
    max_weight: float | Mapping[str, float] | pd.Series = 0.35,
    min_weight: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """Long-only cap-and-renormalize for a Series or each row of a DataFrame."""
    if isinstance(weights, pd.DataFrame):
        return weights.apply(
            lambda row: cap_weights(row, max_weight=max_weight, min_weight=min_weight),
            axis=1,
        )
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    caps = _cap_series(w.index, max_weight)
    w = w.clip(lower=float(min_weight), upper=caps)
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0 / len(w), index=w.index, dtype=float).clip(upper=caps)
    else:
        w = w / float(w.sum())
    for _ in range(50):
        over = w > caps + 1e-12
        if not bool(over.any()):
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = (caps[~over] - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += excess * room / float(room.sum())
    w = w.clip(lower=float(min_weight), upper=caps)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else w


def smooth_weights(weights: pd.DataFrame, *, strength: float = 0.35) -> pd.DataFrame:
    """Blend each rebalance with the previous rebalance to reduce turnover."""
    W = pd.DataFrame(weights).copy().astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if W.empty:
        return W
    alpha = float(np.clip(strength, 0.0, 1.0))
    rows = []
    prev = None
    for dt, row in W.iterrows():
        use = row if prev is None else (1.0 - alpha) * row + alpha * prev
        use = normalize_long_only(use)
        rows.append(use.rename(dt))
        prev = use
    return pd.DataFrame(rows).fillna(0.0)


def kelly_weight_vector(
    mu,
    cov,
    *,
    kelly_fraction: float = 0.25,
    max_weight: float = 0.35,
    ridge: float = 1e-6,
) -> pd.Series:
    """Long-only fractional Kelly weights from excess return and covariance.

    The output is a continuous risky sleeve that may sum below 1.0.  That keeps
    ``kelly_fraction`` meaningful and lets the caller assign residual capital
    to cash/SHY instead of forcing every positive forecast to be fully invested.
    """
    mu_s = pd.Series(mu, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    labels = mu_s.index
    cov_df = pd.DataFrame(cov, index=labels, columns=labels).astype(float)
    S = make_psd(cov_df.to_numpy(dtype=float), eps=float(ridge))
    try:
        raw = np.linalg.solve(S + float(ridge) * np.eye(len(labels)), mu_s.to_numpy(dtype=float))
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(S + float(ridge) * np.eye(len(labels)), mu_s.to_numpy(dtype=float), rcond=None)[0]
    w_raw = pd.Series(float(kelly_fraction) * raw, index=labels, dtype=float)
    w_raw = w_raw.clip(lower=0.0)
    if float(w_raw.sum()) <= 1e-12:
        positive = mu_s.clip(lower=0.0)
        if float(positive.sum()) <= 1e-12:
            return pd.Series(0.0, index=labels, dtype=float)
        w_raw = positive / float(positive.sum()) * min(float(kelly_fraction), 1.0)
    w_raw = w_raw.clip(upper=float(max_weight))
    if float(w_raw.sum()) > 1.0:
        return cap_weights(w_raw, max_weight=max_weight)
    return w_raw


def _soft_confidence_blend(
    c_width,
    c_model=None,
    c_nll=None,
    *,
    index: pd.Index | None = None,
    floor: float = 0.50,
) -> pd.Series:
    """Project 19 confidence blend used for alpha sizing.

    The blend is intentionally soft.  Width, model disagreement, and NLL
    uncertainty should scale alpha influence, not erase it through repeated
    multiplication.
    """
    idx = index if index is not None else pd.Series(c_width).index
    cw = pd.Series(c_width, dtype=float).reindex(idx).fillna(0.0).clip(0.0, 1.0)
    cm = (
        pd.Series(c_model, dtype=float).reindex(idx).fillna(1.0).clip(0.0, 1.0)
        if c_model is not None
        else pd.Series(1.0, index=idx, dtype=float)
    )
    cn = (
        pd.Series(c_nll, dtype=float).reindex(idx).fillna(1.0).clip(0.0, 1.0)
        if c_nll is not None
        else pd.Series(1.0, index=idx, dtype=float)
    )
    c_total = 0.50 + 0.50 * (0.50 * cw + 0.30 * cm + 0.20 * cn)
    return c_total.clip(float(floor), 1.0)


def _confidence_from_row(
    row: pd.DataFrame,
    alpha_horizon: pd.Series,
    *,
    width_col: str | None = None,
    confidence_cols: Sequence[str] | None = None,
) -> pd.Series:
    idx = alpha_horizon.index
    if "c_total" in row.columns:
        return pd.to_numeric(row["c_total"], errors="coerce").reindex(idx).fillna(0.75).clip(0.50, 1.0)
    if "c_width" in row.columns:
        c_width = pd.to_numeric(row["c_width"], errors="coerce").reindex(idx)
    elif width_col is not None and width_col in row.columns:
        width = pd.to_numeric(row[width_col], errors="coerce").reindex(idx).abs()
        c_width = alpha_horizon.abs().div(alpha_horizon.abs() + width.replace(0.0, np.nan))
    else:
        c_width = pd.Series(1.0, index=idx, dtype=float)
    c_model = pd.to_numeric(row["c_model"], errors="coerce").reindex(idx) if "c_model" in row.columns else None
    c_nll = pd.to_numeric(row["c_nll"], errors="coerce").reindex(idx) if "c_nll" in row.columns else None
    if confidence_cols:
        extras = [
            pd.to_numeric(row[col], errors="coerce").reindex(idx)
            for col in confidence_cols
            if col in row.columns and col not in {"c_width", "c_model", "c_nll", "c_total"}
        ]
        if extras and c_nll is None:
            c_nll = pd.concat(extras, axis=1).mean(axis=1)
    return _soft_confidence_blend(c_width, c_model, c_nll, index=idx)


def weights_from_forecasts(
    forecast: pd.DataFrame,
    *,
    date_col: str,
    asset_col: str,
    mu_col: str,
    returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    sigma_col: str | None = None,
    mu_is_z: bool | None = None,
    lookback: int = 252,
    horizon: int = 21,
    kelly_fraction: float = 0.25,
    max_weight: float = 0.35,
    ridge: float = 1e-6,
    cash_asset: str | None = None,
    confidence_cols: Sequence[str] | None = None,
    base_gross: float = 1.0,
    min_gross: float = 0.20,
    max_gross: float = 1.0,
    target_vol: float | None = None,
    top_k: int | None = None,
    cov_model: str = "ledoit_wolf",
) -> pd.DataFrame:
    """Convert a long forecast table into rebalance-date Kelly weights.

    When ``cash_asset`` is supplied and present in ``returns``, risky weights
    are allowed to sum below 1.0 and residual capital is assigned to cash.
    This is the preferred way to model underinvestment with the existing
    backtest engine, which otherwise normalizes active weights.
    """
    f = pd.DataFrame(forecast).copy()
    f[date_col] = pd.to_datetime(f[date_col])
    R = pd.DataFrame(returns).copy().astype(float)
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()
    assets = list(R.columns)
    cash_label = str(cash_asset) if cash_asset is not None else None
    risky_assets = [a for a in assets if str(a) != cash_label]
    rows = []
    for raw_dt in pd.to_datetime(list(rebalance_dates)):
        dt = pd.Timestamp(raw_dt)
        row = f[f[date_col].eq(dt)]
        if row.empty:
            continue
        row = row.drop_duplicates(asset_col).set_index(asset_col).reindex(risky_assets)
        if mu_col not in row.columns:
            continue
        mu = pd.to_numeric(row[mu_col], errors="coerce").astype(float)
        use_z = (
            bool(mu_is_z)
            if mu_is_z is not None
            else bool(sigma_col is not None and str(mu_col).lower().startswith(("z", "q")))
        )
        if use_z and sigma_col is not None and sigma_col in row.columns:
            mu = mu * pd.to_numeric(row[sigma_col], errors="coerce").astype(float)
        active_assets = list(risky_assets)
        if top_k is not None and int(top_k) > 0 and int(top_k) < len(risky_assets):
            active_assets = list(mu.sort_values(ascending=False).head(int(top_k)).index)
            mu = mu.where(mu.index.isin(active_assets), 0.0)
        active_mu = mu.reindex(active_assets).fillna(0.0)
        if active_mu.clip(lower=0.0).sum() <= 1e-12:
            risky = pd.Series(0.0, index=risky_assets, dtype=float, name=dt)
            if cash_label is not None and cash_label in assets:
                full = risky.reindex(assets).fillna(0.0)
                full.loc[cash_label] = 1.0
                rows.append(full.rename(dt))
            else:
                rows.append(risky)
            continue
        window = R.loc[:dt, active_assets].tail(int(lookback)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if len(window) < max(60, int(lookback) // 4):
            continue
        if str(cov_model).lower().replace("-", "_") in {"ledoit_wolf", "lw", "ledoitwolf"}:
            cov_daily = ledoit_wolf_covariance(window, annualization=1.0, return_df=True)
        else:
            cov_daily = window.cov().reindex(index=active_assets, columns=active_assets).fillna(0.0)
        cov = pd.DataFrame(cov_daily, index=active_assets, columns=active_assets).fillna(0.0) * float(horizon)
        w_active = kelly_weight_vector(
            active_mu,
            cov,
            kelly_fraction=kelly_fraction,
            max_weight=max_weight,
            ridge=ridge,
        )
        w = pd.Series(0.0, index=risky_assets, dtype=float)
        w.loc[w_active.index] = w_active
        gross_conf = 1.0
        if confidence_cols:
            confidence = _confidence_from_row(row, mu.reindex(risky_assets).fillna(0.0), confidence_cols=confidence_cols)
            if confidence.notna().any():
                gross_conf = float(np.clip(confidence.mean(), 0.0, 1.0))
        gross = float(np.clip(float(base_gross) * gross_conf, float(min_gross), float(max_gross)))
        if target_vol is not None:
            cov_ann = pd.DataFrame(cov, index=active_assets, columns=active_assets) / float(horizon) * 252.0
            w_for_vol = w.reindex(active_assets).fillna(0.0)
            vol = float(np.sqrt(max(float(w_for_vol.to_numpy() @ cov_ann.to_numpy() @ w_for_vol.to_numpy()), 0.0)))
            if np.isfinite(vol) and vol > 1e-12:
                gross *= min(float(target_vol) / vol, 1.0)
        risky = (w * gross).clip(lower=0.0)
        risky = risky.rename(dt)
        if cash_label is not None and cash_label in assets:
            full = risky.reindex(assets).fillna(0.0)
            full.loc[cash_label] = max(0.0, 1.0 - float(full.drop(labels=[cash_label], errors="ignore").sum()))
            rows.append(full.rename(dt))
        else:
            rows.append(risky)
    return pd.DataFrame(rows).fillna(0.0)


def forecast_kelly_weight_frame(
    forecast: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    mu_col: str,
    returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    sigma_col: str | None = "sigma_21",
    mu_is_z: bool | None = True,
    assets: Sequence[str] | None = None,
    kelly_fraction: float = 0.50,
    max_weight: float = 0.35,
    top_k: int | None = None,
    smooth: float = 0.10,
    target_vol: float | None = None,
    cash_asset: str | None = None,
    confidence_cols: Sequence[str] | None = None,
    base_gross: float = 1.0,
    min_gross: float = 0.20,
    max_gross: float = 1.0,
    cov_model: str = "ledoit_wolf",
    lookback: int = 252,
    horizon: int = 21,
) -> pd.DataFrame:
    """Forecast-to-Kelly wrapper used by the Project 19 notebooks.

    The default is a continuous long-only risky sleeve.  Supplying ``cash_asset``
    keeps residual capital in cash after confidence or volatility scaling.
    """
    R = pd.DataFrame(returns).copy()
    cols = list(assets) if assets is not None else [c for c in R.columns if str(c) != str(cash_asset)]
    W = weights_from_forecasts(
        forecast,
        date_col=date_col,
        asset_col=asset_col,
        mu_col=mu_col,
        sigma_col=sigma_col,
        mu_is_z=mu_is_z,
        returns=R,
        rebalance_dates=rebalance_dates,
        kelly_fraction=kelly_fraction,
        max_weight=max_weight,
        cash_asset=cash_asset,
        confidence_cols=confidence_cols,
        base_gross=base_gross,
        min_gross=min_gross,
        max_gross=max_gross,
        target_vol=target_vol,
        top_k=top_k,
        cov_model=cov_model,
        lookback=lookback,
        horizon=horizon,
    )
    if W.empty:
        return W
    cash_label = str(cash_asset) if cash_asset is not None else None
    risky_cols = [c for c in W.columns if str(c) != cash_label]
    risky = W.reindex(columns=risky_cols).fillna(0.0)
    risky_gross = risky.sum(axis=1).clip(0.0, 1.0)
    if smooth and float(smooth) > 0:
        risky = smooth_weights(risky, strength=float(smooth))
    risky = cap_weights(risky, max_weight=max_weight)
    if cash_label is not None and cash_label in W.columns:
        risky = risky.mul(risky_gross.reindex(risky.index).fillna(1.0), axis=0)
        risky[cash_label] = 1.0 - risky.drop(columns=[cash_label], errors="ignore").sum(axis=1)
        return risky.reindex(columns=cols + ([cash_label] if cash_label not in cols else [])).fillna(0.0).sort_index()
    return risky.reindex(columns=cols).fillna(0.0).sort_index()


def ml_alpha_maxsharpe_weight_frame(
    forecast: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    alpha_col: str,
    returns: pd.DataFrame,
    rebalance_dates: Sequence[pd.Timestamp | str],
    sigma_col: str | None = "sigma_21",
    alpha_is_z: bool = True,
    width_col: str | None = "width",
    confidence_cols: Sequence[str] | None = None,
    prior_model: str = "momentum",
    lambda_alpha: float = 0.10,
    assets: Sequence[str] | None = None,
    cash_asset: str | None = None,
    prior_lookback: int = 252,
    cov_lookback: int = 756,
    horizon: int = 21,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
    max_weight: float = 0.35,
    turnover_penalty_bps: float = 10.0,
) -> pd.DataFrame:
    """MaxSharpe weights whose expected returns are adjusted by ML alpha.

    The optimizer remains a standard Ledoit-Wolf MaxSharpe allocator.  The ML
    signal enters only as a centered, prior-scaled active-return adjustment:

    ``mu_final = mu_prior + lambda_alpha * confidence * alpha_scaled``.
    """
    from quantfinlab.portfolio import expected_returns, optimizers

    f = pd.DataFrame(forecast).copy()
    f[date_col] = pd.to_datetime(f[date_col])
    R = pd.DataFrame(returns).copy().astype(float)
    R.index = pd.to_datetime(R.index)
    R = R.sort_index()
    asset_list = list(assets) if assets is not None else list(R.columns)
    cash_label = str(cash_asset) if cash_asset is not None else None
    risky_assets = [a for a in asset_list if str(a) != cash_label]
    if not risky_assets:
        return pd.DataFrame(columns=asset_list)

    rows = []
    prev = None
    for raw_dt in pd.to_datetime(list(rebalance_dates)):
        dt = pd.Timestamp(raw_dt)
        row = f[f[date_col].eq(dt)].drop_duplicates(asset_col).set_index(asset_col).reindex(risky_assets)
        if row.empty or alpha_col not in row.columns:
            continue
        cov_window = R.loc[:dt, risky_assets].tail(int(cov_lookback)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        mu_window = R.loc[:dt, risky_assets].tail(int(prior_lookback)).replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if len(cov_window) < max(126, int(cov_lookback) // 4) or len(mu_window) < max(63, int(prior_lookback) // 4):
            continue

        cov_ann = ledoit_wolf_covariance(cov_window, annualization=float(annualization), return_df=True)
        prior_name = str(prior_model).lower().replace("-", "_").replace(" ", "_")
        if prior_name in {"bayes_stein", "bayesstein", "shrinkage"}:
            mu_prior = expected_returns.bayes_stein_mu(
                mu_window,
                cov_ann=cov_ann,
                rf_daily=rf_daily,
                annualization=float(annualization),
                return_series=True,
            )
        elif prior_name in {"bayes_stein_momentum", "bayesstein_momentum"}:
            mu_prior = expected_returns.bayes_stein_momentum_mu(
                mu_window,
                cov_ann=cov_ann,
                rf_daily=rf_daily,
                annualization=float(annualization),
                return_series=True,
            )
        else:
            mu_prior = expected_returns.momentum_mu(
                mu_window,
                cov_ann=cov_ann,
                annualization=float(annualization),
                return_series=True,
            )
        mu_prior = pd.Series(mu_prior, index=risky_assets, dtype=float).reindex(risky_assets).fillna(0.0)

        alpha = pd.to_numeric(row[alpha_col], errors="coerce").astype(float).reindex(risky_assets)
        if bool(alpha_is_z) and sigma_col is not None and sigma_col in row.columns:
            alpha_horizon = alpha * pd.to_numeric(row[sigma_col], errors="coerce").astype(float).reindex(risky_assets)
        else:
            alpha_horizon = alpha
        alpha_horizon = alpha_horizon.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        confidence = _confidence_from_row(
            row,
            alpha_horizon,
            width_col=width_col,
            confidence_cols=confidence_cols,
        ).replace([np.inf, -np.inf], np.nan).clip(0.50, 1.0).fillna(0.75)
        alpha_ann = alpha_horizon * float(annualization) / max(float(horizon), 1.0)
        alpha_ann = alpha_ann - float(alpha_ann.median())
        alpha_abs = float(alpha_ann.abs().median())
        prior_abs = float(mu_prior.abs().median())
        if np.isfinite(alpha_abs) and alpha_abs > 1e-12 and np.isfinite(prior_abs) and prior_abs > 1e-12:
            alpha_ann = alpha_ann * (prior_abs / alpha_abs)
        mu_final = mu_prior + float(lambda_alpha) * confidence * alpha_ann

        w_arr = optimizers.max_sharpe_slsqp(
            mu_excess_ann=mu_final.reindex(risky_assets).to_numpy(dtype=float),
            cov_ann=pd.DataFrame(cov_ann, index=risky_assets, columns=risky_assets).to_numpy(dtype=float),
            w_prev=prev,
            w_min=0.0,
            w_max=float(max_weight),
            long_only=True,
            turnover_penalty_bps=float(turnover_penalty_bps),
            raise_on_fail=False,
        )
        if w_arr is None:
            w = pd.Series(1.0 / len(risky_assets), index=risky_assets, dtype=float)
        else:
            w = pd.Series(w_arr, index=risky_assets, dtype=float).clip(lower=0.0)
            w = cap_weights(w, max_weight=max_weight)
        prev = w.reindex(risky_assets).fillna(0.0).to_numpy(dtype=float)
        if cash_label is not None and cash_label in asset_list:
            w = w.reindex(asset_list).fillna(0.0)
            w.loc[cash_label] = max(0.0, 1.0 - float(w.drop(labels=[cash_label], errors="ignore").sum()))
        rows.append(w.reindex(asset_list).fillna(0.0).rename(dt))
    return pd.DataFrame(rows).fillna(0.0).sort_index()


def forecast_gated_maxsharpe_weight_frame(*args, **kwargs) -> pd.DataFrame:
    """Forecast-gated MaxSharpe allocator.

    This is the Project 19-facing name for the allocator implemented by
    :func:`ml_alpha_maxsharpe_weight_frame`: a standard MaxSharpe optimizer
    whose expected-return vector is adjusted by scaled ML active alpha and
    soft forecast confidence.
    """
    return ml_alpha_maxsharpe_weight_frame(*args, **kwargs)


def align_weight_frame(
    weights: pd.DataFrame | Mapping[str, float],
    *,
    target_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    fallback_dates: Sequence[pd.Timestamp | str] | None = None,
    cash_asset: str | None = None,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Align a weight table to target dates with equal-weight fallback."""
    target_idx = pd.DatetimeIndex(pd.to_datetime(list(target_dates))).sort_values().unique()
    if len(target_idx) == 0 and fallback_dates is not None:
        target_idx = pd.DatetimeIndex(pd.to_datetime(list(fallback_dates))).sort_values().unique()
    asset_list = list(assets)
    cols = asset_list + ([cash_asset] if cash_asset is not None and cash_asset not in asset_list else [])
    if len(target_idx) == 0:
        return pd.DataFrame(columns=cols, dtype=float)
    fallback = pd.DataFrame(1.0 / len(asset_list), index=target_idx, columns=asset_list, dtype=float)
    if cash_asset is not None and cash_asset not in fallback.columns:
        fallback[cash_asset] = 0.0
    W = pd.DataFrame(weights).copy()
    if W.empty:
        return fallback.reindex(columns=cols).fillna(0.0)
    W.index = pd.to_datetime(W.index)
    W = W.sort_index().reindex(columns=cols)
    W = W.reindex(target_idx).ffill()
    if W.empty or W.dropna(how="all").empty:
        return fallback.reindex(columns=cols).fillna(0.0)
    W = W.fillna(0.0)
    row_sum = W.sum(axis=1)
    zero_rows = row_sum.abs() <= 1e-12
    if bool(zero_rows.any()):
        W.loc[zero_rows, asset_list] = 1.0 / len(asset_list)
        if cash_asset is not None and cash_asset in W.columns:
            W.loc[zero_rows, cash_asset] = 0.0
        row_sum = W.sum(axis=1)
    W = W.div(row_sum.replace(0.0, np.nan), axis=0).fillna(0.0)
    if max_weight is not None:
        risky = cap_weights(W[asset_list], max_weight=max_weight)
        if cash_asset is not None and cash_asset in W.columns:
            risky[cash_asset] = 1.0 - risky.sum(axis=1)
        W = risky.reindex(columns=cols).fillna(0.0)
    return W.reindex(columns=cols).fillna(0.0).sort_index()


def rank_signal_weight_frame(
    forecast: pd.DataFrame,
    *,
    date_col: str = "date",
    asset_col: str = "asset",
    score_col: str,
    vol_col: str = "vol_63",
    rebalance_dates: Sequence[pd.Timestamp | str] | None = None,
    assets: Sequence[str] | None = None,
    top_k: int = 5,
    max_weight: float = 0.35,
    gross: float = 1.0,
    score_power: float = 1.0,
    vol_power: float = 1.0,
    smooth: float = 0.10,
    cash_asset: str | None = None,
) -> pd.DataFrame:
    """Build top-k long-only weights from cross-sectional forecast ranks.

    This is intentionally rank based: it uses the ordering of a noisy signal
    rather than treating small forecast levels as calibrated expected returns.
    """
    f = pd.DataFrame(forecast).copy()
    f[date_col] = pd.to_datetime(f[date_col])
    asset_list = list(assets) if assets is not None else sorted(f[asset_col].dropna().astype(str).unique())
    dates = pd.to_datetime(list(rebalance_dates)) if rebalance_dates is not None else pd.DatetimeIndex(f[date_col].drop_duplicates())
    rows = []
    for raw_dt in pd.DatetimeIndex(dates).sort_values().unique():
        dt = pd.Timestamp(raw_dt)
        row = f[f[date_col].eq(dt)].drop_duplicates(asset_col).set_index(asset_col).reindex(asset_list)
        if row.empty or score_col not in row.columns:
            continue
        score = pd.to_numeric(row[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if score.notna().sum() < 2:
            continue
        k = max(1, min(int(top_k), len(asset_list)))
        keep = score.sort_values(ascending=False).head(k).index
        selected = score.reindex(keep).fillna(score.median())
        intensity = (selected - selected.min()).clip(lower=0.0).add(0.10).pow(float(score_power))
        if vol_col in row.columns:
            vol = pd.to_numeric(row.reindex(keep)[vol_col], errors="coerce").replace(0.0, np.nan)
            inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan)
            inv_vol = inv_vol.fillna(inv_vol.median() if inv_vol.notna().any() else 1.0).pow(float(vol_power))
        else:
            inv_vol = pd.Series(1.0, index=keep)
        raw = intensity * inv_vol
        if float(raw.sum()) <= 1e-12:
            raw = pd.Series(1.0, index=keep)
        w = cap_weights(raw.reindex(asset_list).fillna(0.0), max_weight=max_weight) * float(gross)
        if cash_asset is not None:
            full_index = asset_list + ([cash_asset] if cash_asset not in asset_list else [])
            w = w.reindex(full_index).fillna(0.0)
            w.loc[cash_asset] = max(0.0, 1.0 - float(w.drop(labels=[cash_asset], errors="ignore").sum()))
        rows.append(w.rename(dt))
    W = pd.DataFrame(rows).fillna(0.0)
    if W.empty:
        return W
    risky_cols = [c for c in W.columns if str(c) != str(cash_asset)]
    if smooth and float(smooth) > 0:
        risky = smooth_weights(W[risky_cols], strength=float(smooth))
        risky = cap_weights(risky, max_weight=max_weight)
        gross_path = W[risky_cols].sum(axis=1).reindex(risky.index).fillna(float(gross)).clip(0.0, 1.0)
        risky = risky.mul(gross_path, axis=0)
        if cash_asset is not None:
            risky[cash_asset] = 1.0 - risky.sum(axis=1)
        W = risky.reindex(columns=W.columns).fillna(0.0)
    return W.sort_index()


def gated_blend_weight_frame(
    base_weights: pd.DataFrame,
    overlay_weights: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    score_col: str,
    date_col: str = "date",
    asset_col: str = "asset",
    assets: Sequence[str] | None = None,
    alpha: float = 0.10,
    keep_k: int = 8,
    weak_scale: float = 0.25,
    max_weight: float = 0.35,
) -> pd.DataFrame:
    """Gate a strong base allocator with forecast ranks and a small overlay."""
    base = pd.DataFrame(base_weights).copy()
    overlay = pd.DataFrame(overlay_weights).copy()
    base.index = pd.to_datetime(base.index)
    overlay.index = pd.to_datetime(overlay.index)
    f = pd.DataFrame(forecast).copy()
    f[date_col] = pd.to_datetime(f[date_col])
    asset_list = list(assets) if assets is not None else list(base.columns)
    idx = pd.DatetimeIndex(overlay.index).sort_values().unique()
    B = base.sort_index().reindex(idx).ffill().reindex(columns=asset_list).fillna(0.0)
    overlay_aligned = overlay.sort_index().reindex(idx).ffill().reindex(columns=asset_list).fillna(0.0)
    rows = []
    for dt, base_row in B.iterrows():
        row = f[f[date_col].eq(pd.Timestamp(dt))].drop_duplicates(asset_col).set_index(asset_col).reindex(asset_list)
        if row.empty or score_col not in row.columns:
            gated = base_row
        else:
            score = pd.to_numeric(row[score_col], errors="coerce").fillna(0.0)
            gate = (score.rank(pct=True) >= (1.0 - float(keep_k) / max(len(asset_list), 1))).astype(float)
            gated = base_row * (float(weak_scale) + (1.0 - float(weak_scale)) * gate)
        gated = cap_weights(gated.reindex(asset_list).fillna(0.0), max_weight=max_weight)
        blended = (1.0 - float(alpha)) * gated + float(alpha) * overlay_aligned.loc[dt].reindex(asset_list).fillna(0.0)
        rows.append(cap_weights(blended, max_weight=max_weight).rename(pd.Timestamp(dt)))
    return pd.DataFrame(rows).fillna(0.0).sort_index()


def tcnrank_kelly_weight_frame(
    *,
    forecast_features: pd.DataFrame,
    returns: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    cash_ticker: str | None = None,
    score_col: str = "tcn_alpha",
    sigma_col: str | None = "sigma_21",
    kelly_fraction: float = 0.50,
    max_weight: float = 0.35,
    target_vol: float | None = 0.12,
    smooth: float = 0.10,
    horizon: int = 21,
    lookback: int = 252,
) -> pd.DataFrame:
    """Project 20 wrapper for TCNRank fractional-Kelly weights."""
    return forecast_kelly_weight_frame(
        forecast_features,
        date_col="date",
        asset_col="asset",
        mu_col=score_col,
        returns=returns,
        rebalance_dates=decision_dates,
        sigma_col=sigma_col,
        mu_is_z=False,
        assets=list(assets),
        kelly_fraction=kelly_fraction,
        max_weight=max_weight,
        smooth=smooth,
        target_vol=target_vol,
        cash_asset=cash_ticker,
        confidence_cols=[c for c in ["tcn_confidence"] if c in forecast_features.columns],
        horizon=horizon,
        lookback=lookback,
    )


def tcnrank_maxsharpe_weight_frame(
    *,
    forecast_features: pd.DataFrame,
    returns: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp | str],
    assets: Sequence[str],
    cash_ticker: str | None = None,
    score_col: str = "tcn_alpha",
    sigma_col: str | None = "sigma_21",
    lambda_alpha: float = 0.12,
    max_weight: float = 0.35,
    turnover_penalty_bps: float = 10.0,
    rf_daily: float = 0.0,
    annualization: float = 252.0,
    horizon: int = 21,
) -> pd.DataFrame:
    """Project 20 wrapper for forecast-integrated MaxSharpe weights."""
    return ml_alpha_maxsharpe_weight_frame(
        forecast_features,
        date_col="date",
        asset_col="asset",
        alpha_col=score_col,
        returns=returns,
        rebalance_dates=decision_dates,
        sigma_col=sigma_col,
        alpha_is_z=False,
        width_col=None,
        confidence_cols=[c for c in ["tcn_confidence"] if c in forecast_features.columns],
        prior_model="momentum",
        lambda_alpha=lambda_alpha,
        assets=list(assets),
        cash_asset=cash_ticker,
        horizon=horizon,
        rf_daily=rf_daily,
        annualization=annualization,
        max_weight=max_weight,
        turnover_penalty_bps=turnover_penalty_bps,
    )


__all__ = [
    "align_weight_frame",
    "cap_weights",
    "forecast_gated_maxsharpe_weight_frame",
    "forecast_kelly_weight_frame",
    "kelly_weight_vector",
    "ml_alpha_maxsharpe_weight_frame",
    "normalize_long_only",
    "gated_blend_weight_frame",
    "rank_signal_weight_frame",
    "smooth_weights",
    "tcnrank_kelly_weight_frame",
    "tcnrank_maxsharpe_weight_frame",
    "weights_from_forecasts",
]
