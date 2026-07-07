from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantfinlab.backtest.hedging import run_spread_backtest
from quantfinlab.common.errors import InputError
from quantfinlab.risk import total_return


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
    """Estimate a static log-price hedge ratio for a residual spread.

    The function fits ``log(target) = alpha + beta * log(hedge)`` on the initial
    training window and then holds the estimated alpha and beta constant for the
    rest of the sample.

    Parameters
    ----------
    px : pandas.DataFrame
        Price panel.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    n_train : int, default=504
        Initial training-window length.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``alpha`` and ``beta`` columns indexed like the price panel.
        Rows before the training estimate are missing.
    """

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
    """Estimate rolling log-price alpha/beta for a residual spread.

    Parameters
    ----------
    px : pandas.DataFrame
        Price panel.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    win : int, default=252
        Rolling window.
    n_train : int, default=504
        Minimum initial history before estimates are emitted.

    Returns
    -------
    pandas.DataFrame
        Rolling ``alpha`` and ``beta`` estimates.
    """

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
    """Estimate time-varying log-price alpha/beta with a Kalman filter.

    The function initializes alpha and beta from an initial log-price OLS fit,
    calibrates process and observation noise on the training period, and filters the
    full spread relationship.

    Parameters
    ----------
    px : pandas.DataFrame
        Price panel.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    n_train : int, default=504
        Training period used for initialization and noise calibration.
    q : float, optional
        Override for beta process noise.
    r_mult : float, optional
        Multiplier for observation variance.

    Returns
    -------
    pandas.DataFrame
        Filtered ``alpha`` and ``beta`` path with missing values before the end of
        the training window.
    """

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
    """Compute a log-price residual spread.

    The spread is defined as ``log(target) - alpha - beta * log(hedge)``. ``beta``
    may be a scalar or a DataFrame containing time-varying ``alpha`` and ``beta``
    columns.

    Parameters
    ----------
    px : pandas.DataFrame
        Price panel.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    beta : float or pandas.DataFrame
        Scalar beta or time-varying alpha/beta table.

    Returns
    -------
    pandas.Series
        Residual log spread.
    """

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
    """Create a lagged residual-spread trading signal from rolling z-scores.

    The signal enters short-spread positions when the z-score is high, long-spread
    positions when the z-score is low, exits near zero, and optionally stops out
    during extreme deviations. The final signal is lagged one day for execution.

    Parameters
    ----------
    spread : array-like
        Residual spread series.
    z_win : int, default=126
        Rolling window for mean and standard deviation.
    z_in : float, default=2.0
        Entry threshold.
    z_out : float, default=0.5
        Exit threshold.
    z_stop : float, default=3.5
        Stop threshold that forces the signal flat.
    z_cool : int, default=5
        Cooldown length after a stop event.

    Returns
    -------
    pandas.DataFrame
        Table with rolling ``z`` and lagged executable ``signal`` columns.

    Notes
    -----
    A positive signal represents long residual spread exposure. A negative signal
    represents short residual spread exposure.
    """

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
    """Convert a residual-spread signal into target and hedge weights.

    Parameters
    ----------
    signal : array-like
        Spread signal, typically from ``z_signal``.
    beta : float or pandas.DataFrame
        Scalar beta or time-varying beta table.
    target : str
        Target asset.
    hedge : str
        Hedge asset.
    tickers : sequence of str
        Full ticker list for the output frame.

    Returns
    -------
    pandas.DataFrame
        Weight table with target weight equal to the signal and hedge weight equal
        to ``-signal * beta``.

    Raises
    ------
    InputError
        If the target or hedge ticker is missing from ``tickers``.
    """

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
    """Apply eligibility filters to residual spread-trading candidates.

    The gate combines cointegration, stationarity, half-life, spread volatility,
    trade count, cost drag, structural-break, and beta-turnover filters into a
    single boolean ``eligible`` column.

    Parameters
    ----------
    rows : array-like or pandas.DataFrame
        Candidate gate rows.
    max_eg_p : float, default=0.10
        Maximum Engle-Granger p-value for static beta sources.
    max_adf_p : float, default=0.10
        Maximum ADF p-value for spread stationarity.
    min_half_life, max_half_life : float
        Acceptable half-life range.
    min_spread_vol : float, default=0.005
        Minimum annualized spread-change volatility.
    min_trades : int, default=3
        Minimum number of signal entries.
    max_cost_drag : float, default=0.10
        Maximum cost drag.
    max_break_p : float, default=0.20
        Maximum structural-break p-value when available.
    max_beta_turnover : float, default=0.25
        Maximum average absolute beta turnover.

    Returns
    -------
    pandas.DataFrame
        Gate table with an ``eligible`` boolean column.
    """

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


def residual_backtest_grid(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    pairs: Sequence[tuple[str, str, str]],
    beta_sources: dict[str, object],
    *,
    ann: float = 252.0,
    cost_bps: float = 5.0,
    z_win: int = 126,
    z_in: float = 2.0,
    z_out: float = 0.5,
    z_stop: float = 3.5,
    z_cool: int = 5,
    gate_kwargs: dict[str, float] | None = None,
):
    """Build residual-spread signals, backtest them, and evaluate eligibility.

    For each candidate pair and beta source, the function estimates the residual
    spread, builds a z-score signal, runs a spread backtest, and records diagnostic
    statistics used by the residual gate.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price panel.
    returns : pandas.DataFrame
        Return panel.
    pairs : sequence of tuple
        Tuples of ``(pair_name, target, hedge)``.
    beta_sources : dict
        Mapping from beta-source name to a callable that returns beta parameters
        for a target/hedge pair.
    ann : float, default=252.0
        Annualization factor.
    cost_bps : float, default=5.0
        Transaction cost in basis points applied by the spread backtest.
    z_win, z_in, z_out, z_stop, z_cool
        Signal-generation parameters passed to ``z_signal``.
    gate_kwargs : dict, optional
        Overrides passed to ``resid_gate``.

    Returns
    -------
    tuple
        ``(gate, backtests, signals, metadata)`` where ``gate`` is the eligibility
        table, ``backtests`` maps strategy keys to results, ``signals`` maps keys to
        signal tables, and ``metadata`` stores pair/beta-source identifiers.

    Notes
    -----
    Each beta source callable should return a compatible scalar or time-varying
    beta table for the requested pair.
    """

    gate_rows = []
    backtests = {}
    signals = {}
    metadata = {}

    for pair_name, target, hedge in pairs:
        p = _price_panel(prices, target, hedge)
        lp = np.log(p[p > 0]).dropna()
        eg_p = eg_test(lp[target], lp[hedge])
        for beta_source, fn in beta_sources.items():
            params = fn(target, hedge)
            spread = log_spread(prices, target, hedge, params)
            sig = z_signal(spread, z_win=z_win, z_in=z_in, z_out=z_out, z_stop=z_stop, z_cool=z_cool)
            key = f"{pair_name} | {beta_source}"
            res = run_spread_backtest(
                returns,
                params,
                sig["signal"],
                target=target,
                hedge=hedge,
                cost_bps=cost_bps,
                name=key,
            )

            sdrop = spread.dropna()
            split = max(len(sdrop) // 2, 30)
            first, second = sdrop.iloc[:split], sdrop.iloc[split:]
            trades = int((sig["signal"].ne(0) & sig["signal"].shift(1).fillna(0).eq(0)).sum())
            cost_drag = max(total_return(res.gross_values) - total_return(res.net_values), 0.0)
            gate_rows.append({
                "pair": pair_name, "beta_source": beta_source, "eg_p": eg_p,
                "adf_p": adf_test(spread), "half_life": half_life(spread),
                "spread_vol": spread.diff().std() * np.sqrt(float(ann)), "trades": trades,
                "cost_drag": cost_drag, "cost_drag_ann": res.cost.mean() * float(ann) if len(res.cost) else 0.0,
                "break_p": max(adf_test(first), adf_test(second)) if len(second) >= 30 else np.nan,
                "beta_turnover": params["beta"].dropna().diff().abs().mean(), "key": key})
            backtests[key] = res
            signals[key] = sig
            metadata[key] = {"pair": pair_name, "beta_source": beta_source}

    gate = resid_gate(gate_rows, **dict(gate_kwargs or {}))
    return gate, backtests, signals, metadata


__all__ = [
    "adf_test",
    "eg_test",
    "half_life",
    "kf_price_beta",
    "log_spread",
    "price_ols_beta",
    "resid_gate",
    "residual_backtest_grid",
    "roll_price_beta",
    "spread_w",
    "z_signal",
]
