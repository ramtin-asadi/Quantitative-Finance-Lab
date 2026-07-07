from __future__ import annotations

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError
from quantfinlab.hedging.relations import rel


def _ret_panel(ret: pd.DataFrame, r: rel) -> pd.DataFrame:
    if not isinstance(ret, pd.DataFrame):
        raise InputError("ret must be a pandas DataFrame.")
    out = ret.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    missing = [c for c in r.assets if c not in out.columns]
    if missing:
        raise InputError(f"Missing returns for {r.name}: {missing}")
    return out[r.assets].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _empty_beta(index: pd.Index, r: rel) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=pd.DatetimeIndex(index), columns=r.hedges, dtype=float)


def _ols_fit(y: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    xmat = np.column_stack([np.ones(len(y), dtype=float), x])
    coef = np.linalg.lstsq(xmat, y, rcond=None)[0]
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def ols_beta(ret: pd.DataFrame, r: rel, *, n_train: int = 504) -> pd.DataFrame:
    """Estimate a static OLS hedge beta after an initial training window.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel.
    r : rel
        Relationship defining the target and hedge assets.
    n_train : int, default=504
        Initial training-window length.

    Returns
    -------
    pandas.DataFrame
        Beta path indexed like the relationship return panel. Rows before the end
        of the training window are missing; later rows contain the static beta.

    Notes
    -----
    For multi-hedge relationships, the beta vector is estimated by multivariate
    least squares.
    """

    panel = _ret_panel(ret, r)
    out = _empty_beta(panel.index, r)
    z = panel.dropna()
    if len(z) < int(n_train):
        return out
    train = z.iloc[: int(n_train)]
    _, beta = _ols_fit(train[r.target].to_numpy(), train[r.hedges].to_numpy())
    out.loc[z.index[int(n_train) - 1] :, r.hedges] = beta
    return out


def roll_beta(ret: pd.DataFrame, r: rel, *, win: int = 252, n_train: int = 504) -> pd.DataFrame:
    """Estimate rolling OLS hedge betas.

    At each date after the initial training and rolling-window requirements are
    met, the function estimates hedge betas using only the trailing window ending at
    that date.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel.
    r : rel
        Relationship defining target and hedge assets.
    win : int, default=252
        Rolling estimation window.
    n_train : int, default=504
        Minimum initial history before rolling estimates are emitted.

    Returns
    -------
    pandas.DataFrame
        Rolling beta path with one column per hedge asset.

    Notes
    -----
    The output is a desired beta path. Trading timing and lagging are handled by the
    backtest layer.
    """

    panel = _ret_panel(ret, r)
    out = _empty_beta(panel.index, r)
    z = panel.dropna()
    w = int(win)
    start = max(int(n_train), w)
    if len(z) < start or w < 3:
        return out
    for i in range(start - 1, len(z)):
        sample = z.iloc[i - w + 1 : i + 1]
        _, beta = _ols_fit(sample[r.target].to_numpy(), sample[r.hedges].to_numpy())
        out.loc[z.index[i], r.hedges] = beta
    return out


def _kalman_run(
    y: np.ndarray,
    x: np.ndarray,
    state0: np.ndarray,
    p0: np.ndarray,
    q_diag: np.ndarray,
    obs_var: float,
) -> tuple[np.ndarray, float]:
    p = len(state0)
    state = state0.astype(float).copy()
    pcov = np.asarray(p0, dtype=float).copy()
    qcov = np.diag(np.asarray(q_diag, dtype=float))
    states = np.full((len(y), p), np.nan, dtype=float)
    loglik = 0.0
    for i in range(len(y)):
        h = np.r_[1.0, x[i]]
        pcov = pcov + qcov
        pred = float(h @ state)
        s = float(h @ pcov @ h.T + obs_var)
        if s <= 1e-12 or not np.isfinite(s):
            states[i] = state
            continue
        err = float(y[i] - pred)
        loglik += -0.5 * (np.log(2.0 * np.pi * s) + err * err / s)
        k = (pcov @ h.T) / s
        state = state + k * err
        pcov = (np.eye(p) - np.outer(k, h)) @ pcov
        states[i] = state
    return states, float(loglik)


def _kalman_calibration(train: pd.DataFrame, r: rel) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    y = train[r.target].to_numpy(dtype=float)
    x = train[r.hedges].to_numpy(dtype=float)
    init_n = max(min(len(train) // 4, 126), min(30, len(train)))
    _, beta0 = _ols_fit(y[:init_n], x[:init_n])
    intercept0 = float(np.nanmean(y[:init_n] - x[:init_n] @ beta0))
    state0 = np.r_[intercept0, beta0]
    h0 = np.column_stack([np.ones(len(train)), x])
    resid = y - h0 @ state0
    resid_var = max(float(np.nanvar(resid, ddof=1)), 1e-8)
    p0 = np.eye(len(state0)) * max(resid_var, 1e-6)

    beta_scale = np.r_[0.05, np.maximum(np.abs(beta0), 0.25)]
    q_grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    r_grid = [0.50, 1.0, 2.0, 4.0]
    best_ll = -np.inf
    best_q = np.r_[1e-7, np.repeat(1e-5, len(state0) - 1)]
    best_r = resid_var
    for q_mult in q_grid:
        q_diag = np.maximum(q_mult * beta_scale**2, 1e-12)
        for r_mult in r_grid:
            obs_var = max(resid_var * r_mult, 1e-10)
            _, ll = _kalman_run(y, x, state0, p0, q_diag, obs_var)
            jump_penalty = float(np.sum(q_diag[1:])) / max(obs_var, 1e-12)
            ll_adj = ll - 0.5 * jump_penalty
            if ll_adj > best_ll:
                best_ll = ll_adj
                best_q = q_diag
                best_r = obs_var
    return state0, p0, best_q, best_r


def ridge_beta(
    ret: pd.DataFrame,
    r: rel,
    *,
    win: int = 252,
    alpha: float = 10.0,
    n_train: int = 504,
) -> pd.DataFrame:
    """Estimate rolling ridge-regression hedge betas.

    The function centers the target and hedge returns, standardizes hedge features,
    applies ridge shrinkage in standardized space, and converts coefficients back
    to return units. This is useful when hedge assets are correlated or rolling OLS
    coefficients are unstable.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel.
    r : rel
        Relationship defining target and hedge assets.
    win : int, default=252
        Rolling estimation window.
    alpha : float, default=10.0
        Ridge penalty strength. Larger values shrink betas more strongly.
    n_train : int, default=504
        Minimum initial history before estimates are emitted.

    Returns
    -------
    pandas.DataFrame
        Rolling ridge beta path.

    Notes
    -----
    The regression includes a centered intercept implicitly through demeaning.
    """

    panel = _ret_panel(ret, r)
    out = _empty_beta(panel.index, r)
    z = panel.dropna()
    w = int(win)
    start = max(int(n_train), w)
    if len(z) < start or w < 3:
        return out
    a = max(float(alpha), 0.0)
    for i in range(start - 1, len(z)):
        sample = z.iloc[i - w + 1 : i + 1]
        x = sample[r.hedges].to_numpy(dtype=float)
        y = sample[r.target].to_numpy(dtype=float)
        x_mean = x.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        x_std = x.std(axis=0, ddof=1)
        y_std = float(y.std(ddof=1))
        x_std = np.where(x_std > 1e-12, x_std, 1.0)
        if not np.isfinite(y_std) or y_std <= 1e-12:
            continue
        xs = (x - x_mean) / x_std
        ys = (y - y_mean) / y_std
        p = xs.shape[1]
        b_std = np.linalg.pinv(xs.T @ xs + a * np.eye(p)) @ (xs.T @ ys)
        beta = b_std * y_std / x_std
        out.loc[z.index[i], r.hedges] = beta
    return out


def kf_beta(
    ret: pd.DataFrame,
    r: rel,
    *,
    n_train: int = 504,
    q: float | None = None,
    r_mult: float | None = None,
) -> pd.DataFrame:
    """Estimate time-varying hedge betas with a Kalman filter.

    The filter is calibrated using an initial training period and then run through
    the full relationship history. The state contains an intercept and one beta per
    hedge asset; only beta columns are returned.

    Parameters
    ----------
    ret : pandas.DataFrame
        Return panel.
    r : rel
        Relationship defining target and hedge assets.
    n_train : int, default=504
        Initial calibration window.
    q : float, optional
        Override for state-noise intensity. If omitted, the function uses the
        training calibration.
    r_mult : float, optional
        Multiplier applied to the observation variance estimate.

    Returns
    -------
    pandas.DataFrame
        Filtered beta path with missing values before the end of the training
        window.

    Notes
    -----
    The calibration uses only the initial training sample. The filter update at each
    date uses information available through that date.
    """

    panel = _ret_panel(ret, r)
    out = _empty_beta(panel.index, r)
    z = panel.dropna()
    n = int(n_train)
    if len(z) < n:
        return out

    train = z.iloc[:n]
    state0, p0, q_diag, obs_var = _kalman_calibration(train, r)
    if q is not None:
        q_diag = np.r_[max(float(q) * 0.01, 1e-12), np.repeat(max(float(q), 1e-12), len(state0) - 1)]
    if r_mult is not None:
        y_train = train[r.target].to_numpy(dtype=float)
        x_train = train[r.hedges].to_numpy(dtype=float)
        h_train = np.column_stack([np.ones(len(train)), x_train])
        obs_var = max(float(np.nanvar(y_train - h_train @ state0, ddof=1)) * float(r_mult), 1e-10)

    y_all = z[r.target].to_numpy(dtype=float)
    x_all = z[r.hedges].to_numpy(dtype=float)
    states, _ = _kalman_run(y_all, x_all, state0, p0, q_diag, obs_var)
    beta_states = states[:, 1:]
    out.loc[z.index, r.hedges] = beta_states
    out.loc[z.index[: n - 1], r.hedges] = np.nan
    return out


__all__ = ["kf_beta", "ols_beta", "ridge_beta", "roll_beta"]
