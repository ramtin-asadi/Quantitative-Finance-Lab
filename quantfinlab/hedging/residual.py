from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError


def _statsmodels_tools():
    try:
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise ImportError(
            "Residual hedge trading requires statsmodels. Install quantfinlab[hedging] "
            "or quantfinlab[volatility]."
        ) from exc
    return adfuller, coint


def _price_panel(px: pd.DataFrame, target: str, hedge: str) -> pd.DataFrame:
    if not isinstance(px, pd.DataFrame):
        raise InputError("px must be a pandas DataFrame.")
    out = px.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.columns = [str(c).strip().lower() for c in out.columns]
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    missing = [c for c in [target, hedge] if c not in out.columns]
    if missing:
        raise InputError(f"Missing prices: {missing}")
    return out[[target, hedge]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    xmat = np.column_stack([np.ones(len(x), dtype=float), x])
    coef = np.linalg.lstsq(xmat, y, rcond=None)[0]
    return float(coef[0]), float(coef[1])


def price_ols_beta(px: pd.DataFrame, target: str, hedge: str, *, n_train: int = 504) -> pd.DataFrame:
    """Static log-price OLS alpha/beta after the initial training window."""
    p = _price_panel(px, target, hedge)
    out = pd.DataFrame(np.nan, index=p.index, columns=["alpha", "beta"], dtype=float)
    z = np.log(p[p > 0]).dropna()
    if len(z) < int(n_train):
        return out
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    a, b = _ols_alpha_beta(z[target].iloc[: int(n_train)].to_numpy(), z[hedge].iloc[: int(n_train)].to_numpy())
    out.loc[z.index[int(n_train) - 1] :, ["alpha", "beta"]] = [a, b]
    return out


def roll_price_beta(
    px: pd.DataFrame,
    target: str,
    hedge: str,
    *,
    win: int = 252,
    n_train: int = 504,
) -> pd.DataFrame:
    """Rolling log-price OLS alpha/beta."""
    p = _price_panel(px, target, hedge)
    out = pd.DataFrame(np.nan, index=p.index, columns=["alpha", "beta"], dtype=float)
    z = np.log(p[p > 0]).dropna()
    w = int(win)
    start = max(int(n_train), w)
    if len(z) < start or w < 3:
        return out
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    for i in range(start - 1, len(z)):
        sample = z.iloc[i - w + 1 : i + 1]
        a, b = _ols_alpha_beta(sample[target].to_numpy(), sample[hedge].to_numpy())
        out.loc[z.index[i], ["alpha", "beta"]] = [a, b]
    return out


def kf_price_beta(
    px: pd.DataFrame,
    target: str,
    hedge: str,
    *,
    n_train: int = 504,
    q: float | None = None,
    r_mult: float | None = None,
) -> pd.DataFrame:
    """Filtered Kalman log-price alpha/beta with training-only Q/R calibration."""
    p = _price_panel(px, target, hedge)
    out = pd.DataFrame(np.nan, index=p.index, columns=["alpha", "beta"], dtype=float)
    z = np.log(p[p > 0]).dropna()
    n = int(n_train)
    if len(z) < n:
        return out
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    a, b = _ols_alpha_beta(z[target].iloc[:n].to_numpy(), z[hedge].iloc[:n].to_numpy())
    state = np.array([a, b], dtype=float)
    y_train = z[target].iloc[:n].to_numpy(dtype=float)
    x_train = z[hedge].iloc[:n].to_numpy(dtype=float)
    h0 = np.column_stack([np.ones(n), x_train])
    resid = y_train - h0 @ state
    resid_var = max(float(np.nanvar(resid, ddof=1)), 1e-8)
    p_cov0 = np.eye(2) * max(resid_var, 1e-6)

    def run(y, x, q_diag, obs_var):
        st = state.copy()
        pc = p_cov0.copy()
        ll = 0.0
        states = []
        qc = np.diag(q_diag)
        for yy, xx in zip(y, x, strict=False):
            h = np.array([1.0, float(xx)])
            pc = pc + qc
            pred = float(h @ st)
            s = float(h @ pc @ h.T + obs_var)
            if s > 1e-12 and np.isfinite(s):
                err = float(yy - pred)
                ll += -0.5 * (np.log(2.0 * np.pi * s) + err * err / s)
                k = (pc @ h.T) / s
                st = st + k * err
                pc = (np.eye(2) - np.outer(k, h)) @ pc
            states.append(st.copy())
        return np.asarray(states), float(ll)

    q_grid = [1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5]
    r_grid = [0.5, 1.0, 2.0, 4.0]
    best_ll = -np.inf
    best_q = np.array([1e-7, 1e-5])
    best_r = resid_var
    for q_mult in q_grid:
        q_diag = np.array([q_mult * 0.01, q_mult * max(abs(state[1]), 0.25) ** 2])
        for rm in r_grid:
            obs_var = resid_var * rm
            _, ll = run(y_train, x_train, q_diag, obs_var)
            if ll > best_ll:
                best_ll = ll
                best_q = q_diag
                best_r = obs_var

    if q is not None:
        best_q = np.array([max(float(q) * 0.01, 1e-12), max(float(q), 1e-12)])
    if r_mult is not None:
        best_r = max(resid_var * float(r_mult), 1e-10)

    out.loc[z.index[n - 1], ["alpha", "beta"]] = state
    states, _ = run(
        z[target].to_numpy(dtype=float),
        z[hedge].to_numpy(dtype=float),
        best_q,
        best_r,
    )
    out.loc[z.index, ["alpha", "beta"]] = states
    out.loc[z.index[: n - 1], ["alpha", "beta"]] = np.nan
    return out


def log_spread(px: pd.DataFrame, target: str, hedge: str, beta) -> pd.Series:
    """Log target minus alpha and beta times log hedge."""
    p = _price_panel(px, target, hedge)
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    lp = np.log(p[p > 0])
    if isinstance(beta, pd.DataFrame):
        b = beta.copy()
        b.index = pd.to_datetime(b.index)
        b = b.sort_index().reindex(lp.index).ffill()
        alpha = pd.to_numeric(b.get("alpha", 0.0), errors="coerce")
        coef = pd.to_numeric(b["beta"], errors="coerce")
    else:
        alpha = 0.0
        coef = float(beta)
    spread = lp[target] - alpha - coef * lp[hedge]
    spread.name = f"{target}_{hedge}_spread"
    return spread.replace([np.inf, -np.inf], np.nan)


def eg_test(y, x) -> float:
    """Engle-Granger cointegration p-value."""
    _, coint = _statsmodels_tools()
    z = pd.concat([pd.Series(y), pd.Series(x)], axis=1).dropna()
    if len(z) < 30:
        return float("nan")
    try:
        return float(coint(z.iloc[:, 0], z.iloc[:, 1])[1])
    except Exception:
        return float("nan")


def adf_test(series) -> float:
    """ADF stationarity p-value."""
    adfuller, _ = _statsmodels_tools()
    s = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return float("nan")
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return float("nan")


def half_life(series) -> float:
    """AR(1) mean-reversion half-life in observations."""
    s = pd.to_numeric(pd.Series(series), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        return float("nan")
    y = s.diff().dropna()
    x = s.shift(1).reindex(y.index)
    z = pd.concat([y, x], axis=1).dropna()
    if len(z) < 30:
        return float("nan")
    a, b = _ols_alpha_beta(z.iloc[:, 0].to_numpy(), z.iloc[:, 1].to_numpy())
    if not np.isfinite(b) or b >= 0:
        return float("inf")
    return float(-math.log(2.0) / b)


def z_signal(
    spread,
    *,
    z_win: int = 126,
    z_in: float = 2.0,
    z_out: float = 0.5,
    z_stop: float = 3.5,
    z_cool: int = 5,
) -> pd.DataFrame:
    """Rolling z-score signal, lagged one day for execution."""
    s = pd.to_numeric(pd.Series(spread), errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = s.rolling(int(z_win)).mean()
    std = s.rolling(int(z_win)).std(ddof=1)
    z = (s - mean) / std.replace(0.0, np.nan)
    state = []
    pos = 0.0
    cool = 0
    for val in z:
        if not np.isfinite(val):
            state.append(pos)
            continue
        if abs(val) > float(z_stop):
            pos = 0.0
            cool = int(z_cool)
        elif cool > 0:
            pos = 0.0
            cool -= 1
        elif pos == 0.0:
            if val > float(z_in):
                pos = -1.0
            elif val < -float(z_in):
                pos = 1.0
        elif abs(val) < float(z_out):
            pos = 0.0
        state.append(pos)
    signal = pd.Series(state, index=s.index, name="signal").shift(1).fillna(0.0)
    return pd.DataFrame({"z": z, "signal": signal})


def spread_w(signal, beta, target: str, hedge: str, tickers: Sequence[str]) -> pd.DataFrame:
    """Convert long/short spread signal to target and hedge weights."""
    sig = pd.Series(signal, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = [str(c).strip().lower() for c in tickers]
    target = str(target).strip().lower()
    hedge = str(hedge).strip().lower()
    missing = [c for c in [target, hedge] if c not in cols]
    if missing:
        raise InputError(f"Missing tickers: {missing}")
    if isinstance(beta, pd.DataFrame):
        b = pd.to_numeric(beta["beta"], errors="coerce")
        b.index = pd.to_datetime(beta.index)
        b = b.sort_index().reindex(sig.index).ffill()
    else:
        b = pd.Series(float(beta), index=sig.index)
    out = pd.DataFrame(0.0, index=sig.index, columns=cols, dtype=float)
    out.loc[:, target] = sig
    out.loc[:, hedge] = -sig * b.fillna(0.0)
    return out


def resid_gate(
    rows,
    *,
    max_eg_p: float = 0.10,
    max_adf_p: float = 0.10,
    min_half_life: float = 2.0,
    max_half_life: float = 63.0,
    min_spread_vol: float = 0.005,
    min_trades: int = 3,
    max_cost_drag: float = 0.10,
    max_break_p: float = 0.20,
    max_beta_turnover: float = 0.25,
) -> pd.DataFrame:
    """Apply residual-trading eligibility filters to a gate table."""
    tab = pd.DataFrame(rows).copy()
    if tab.empty:
        tab["eligible"] = []
        return tab
    for col in ["eg_p", "adf_p", "half_life", "spread_vol", "trades", "cost_drag", "cost_drag_ann", "beta_turnover"]:
        if col not in tab.columns:
            tab[col] = np.nan
    source = tab.get("beta_source", pd.Series("", index=tab.index)).astype(str).str.lower()
    cost_col = "cost_drag_ann" if tab["cost_drag_ann"].notna().any() else "cost_drag"
    eg_ok = (tab["eg_p"] <= float(max_eg_p)) | (~source.str.contains("static"))
    ok = (
        eg_ok
        & (tab["adf_p"] <= float(max_adf_p))
        & (tab["half_life"] >= float(min_half_life))
        & (tab["half_life"] <= float(max_half_life))
        & (tab["spread_vol"] >= float(min_spread_vol))
        & (tab["trades"] >= int(min_trades))
        & (tab[cost_col].fillna(0.0) <= float(max_cost_drag))
        & (tab["beta_turnover"].fillna(0.0) <= float(max_beta_turnover))
    )
    if "break_p" in tab.columns:
        ok = ok & (tab["break_p"].fillna(1.0) <= float(max_break_p))
    tab["eligible"] = ok.fillna(False).astype(bool)
    return tab


__all__ = [
    "adf_test",
    "eg_test",
    "half_life",
    "kf_price_beta",
    "log_spread",
    "price_ols_beta",
    "resid_gate",
    "roll_price_beta",
    "spread_w",
    "z_signal",
]
