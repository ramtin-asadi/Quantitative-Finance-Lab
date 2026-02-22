from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import BacktestResult, InputError, PortfolioState, RiskReportArtifacts

try:  # optional
    from IPython.display import display as ipy_display
except Exception:  # pragma: no cover
    ipy_display = None

try:  # optional
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None

DEFAULT_ANNUALIZATION = 252.0
VAR_BACKTEST_METHODS = ("hist", "cf", "fhs")


def _to_numeric_series(
    x: pd.Series | Sequence[float] | np.ndarray,
    *,
    name: str = "series",
) -> pd.Series:
    if isinstance(x, pd.Series):
        s = x.copy()
    else:
        s = pd.Series(np.asarray(x, dtype=float))
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise InputError(f"{name} is empty after numeric cleaning.")
    return s.astype(float)


def _to_datetime_if_possible(s: pd.Series) -> pd.Series:
    idx = pd.Index(s.index)
    if isinstance(idx, pd.DatetimeIndex):
        out = s.copy()
        out = out[~out.index.isna()].sort_index()
        return out
    dt = pd.to_datetime(idx, errors="coerce")
    if dt.notna().all():
        out = s.copy()
        out.index = pd.DatetimeIndex(dt)
        out = out[~out.index.isna()].sort_index()
        return out
    return s


def _coerce_objects(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    dropna: bool = True,
) -> dict[str, pd.Series]:
    if isinstance(objects, pd.DataFrame):
        data = {str(c): objects[c] for c in objects.columns}
    elif isinstance(objects, Mapping):
        data = {str(k): v for k, v in objects.items()}
    else:
        raise InputError("objects must be a mapping of name -> returns series, or a DataFrame.")
    if not data:
        raise InputError("objects is empty.")
    out: dict[str, pd.Series] = {}
    for name, val in data.items():
        s = _to_numeric_series(val, name=f"objects[{name!r}]")
        s = _to_datetime_if_possible(s)
        if dropna:
            s = s.dropna()
        if s.empty:
            continue
        out[name] = s.astype(float)
    if not out:
        raise InputError("No non-empty object series remain after cleaning.")
    return out


def _align_pair(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(z) == 0:
        raise InputError("Series do not overlap after alignment.")
    return z


def _normalize_alpha(alpha: float) -> float:
    a = float(alpha)
    if not (0.0 < a < 0.5):
        raise InputError("alpha must be in (0, 0.5).")
    return a


def _normalize_var_methods(
    *,
    method: str | None = None,
    methods: Sequence[str] | None = None,
) -> list[str]:
    if methods is None:
        base = [method or "hist"]
    else:
        base = list(methods)
        if len(base) == 0:
            raise InputError("methods must contain at least one method.")
    out = [str(m).strip().lower() for m in base]
    valid = set(VAR_BACKTEST_METHODS)
    unknown = [m for m in out if m not in valid]
    if unknown:
        raise InputError(f"Unknown VaR method(s): {unknown}. Valid methods: {sorted(valid)}.")
    return out


def _as_result_mapping(result: BacktestResult | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(result, BacktestResult):
        return result.as_dict()
    if not isinstance(result, Mapping):
        raise InputError("result must be a BacktestResult or dict-like mapping.")
    return result


def _as_state_mapping(state: PortfolioState | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(state, PortfolioState):
        return state.as_dict()
    if isinstance(state, Mapping):
        return state
    raise InputError("state must be PortfolioState or dict-like.")


def nav_series(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    start_value: float = 1.0,
) -> pd.Series:
    if start_value <= 0:
        raise InputError("start_value must be positive.")
    r = _to_numeric_series(returns, name="returns").fillna(0.0)
    return float(start_value) * (1.0 + r).cumprod()


def drawdown_series(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> pd.Series:
    x = _to_numeric_series(returns_or_nav, name="returns_or_nav")
    kind = str(input_kind).strip().lower()
    if kind in {"returns", "ret", "r"}:
        nav = (1.0 + x).cumprod()
    elif kind in {"nav", "equity", "value"}:
        nav = x
    else:
        raise InputError("input_kind must be either 'returns' or 'nav'.")
    return nav / nav.cummax() - 1.0


def ulcer_index(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> float:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    return float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float))))) if len(dd) else float("nan")


def drawdown_episodes(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> pd.DataFrame:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    if dd.empty:
        return pd.DataFrame(columns=["start", "end", "depth", "duration"])
    in_dd = False
    start_i = 0
    rows: list[tuple[Any, Any, float, int]] = []
    vals = dd.to_numpy(dtype=float)
    for i, v in enumerate(vals):
        if v < 0 and not in_dd:
            in_dd = True
            start_i = i
        if v >= -1e-15 and in_dd:
            seg = dd.iloc[start_i:i]
            if len(seg):
                rows.append((seg.index[0], seg.index[-1], float(seg.min()), int(len(seg))))
            in_dd = False
    if in_dd:
        seg = dd.iloc[start_i:]
        if len(seg):
            rows.append((seg.index[0], seg.index[-1], float(seg.min()), int(len(seg))))
    return pd.DataFrame(rows, columns=["start", "end", "depth", "duration"])


def avg_recovery_time(
    returns_or_nav: pd.Series | Sequence[float] | np.ndarray,
    *,
    input_kind: str = "returns",
) -> float:
    dd = drawdown_series(returns_or_nav, input_kind=input_kind)
    if dd.empty:
        return float("nan")
    rec_times: list[int] = []
    in_dd = False
    start = 0
    for i, v in enumerate(dd.to_numpy(dtype=float)):
        if v < 0 and not in_dd:
            in_dd = True
            start = i
        if v >= -1e-15 and in_dd:
            rec_times.append(i - start)
            in_dd = False
    return float(np.mean(rec_times)) if rec_times else float("nan")


def sortino_ratio(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> float:
    r = _to_numeric_series(returns, name="returns")
    ex = r - float(rf_daily)
    dn = np.minimum(ex.to_numpy(dtype=float), 0.0)
    den = float(np.sqrt(np.mean(np.square(dn))))
    if den <= 1e-12:
        return float("nan")
    return float((ex.mean() / den) * math.sqrt(float(annualization)))


def hist_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    q = float(r.quantile(a))
    tail = r[r <= q]
    es = float(tail.mean()) if len(tail) else q
    return -q, -es


def cf_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_sim: int = 70_000,
    seed: int = 7,
) -> tuple[float, float]:
    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    if len(r) < 10:
        return float("nan"), float("nan")
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 1e-12:
        return float("nan"), float("nan")
    s = float(r.skew())
    k = float(r.kurt())
    z = NormalDist().inv_cdf(a)
    zc = z + (z**2 - 1.0) * s / 6.0 + (z**3 - 3.0 * z) * k / 24.0 - (2.0 * z**3 - 5.0 * z) * (s**2) / 36.0
    q = mu + sd * zc

    rng = np.random.default_rng(int(seed))
    zs = rng.standard_normal(int(n_sim))
    za = (
        zs
        + (zs**2 - 1.0) * s / 6.0
        + (zs**3 - 3.0 * zs) * k / 24.0
        - (2.0 * zs**3 - 5.0 * zs) * (s**2) / 36.0
    )
    rs = mu + sd * za
    tail = rs[rs <= q]
    es = float(np.mean(tail)) if len(tail) else float(q)
    return -float(q), -float(es)


def fhs_var_es(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lam: float = 0.94,
) -> tuple[float, float]:
    a = _normalize_alpha(alpha)
    if not (0.0 < float(lam) < 1.0):
        raise InputError("lam must be in (0, 1).")
    r = _to_numeric_series(returns, name="returns")
    if len(r) < 10:
        return float("nan"), float("nan")
    mu = float(r.mean())
    e = r - mu
    sig = np.zeros(len(e), dtype=float)
    sig[0] = max(float(e.std(ddof=1)), 1e-6)
    ev = e.to_numpy(dtype=float)
    for t in range(1, len(e)):
        sig[t] = math.sqrt(float(lam) * sig[t - 1] ** 2 + (1.0 - float(lam)) * ev[t - 1] ** 2)
    z = ev / np.where(sig > 1e-12, sig, np.nan)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return float("nan"), float("nan")
    qz = float(np.quantile(z, a))
    tail = z[z <= qz]
    ez = float(np.mean(tail)) if len(tail) else qz
    sn = float(sig[-1])
    return -(mu + sn * qz), -(mu + sn * ez)

def _rolling_var_quantile(
    returns: pd.Series,
    *,
    alpha: float,
    lookback: int,
    method: Literal["hist", "cf", "fhs"],
    cf_n_sim: int = 15_000,
    cf_seed: int = 7,
    fhs_lambda: float = 0.94,
) -> pd.Series:
    if lookback < 20:
        raise InputError("lookback must be at least 20.")
    r = _to_numeric_series(returns, name="returns")
    if len(r) < lookback + 1:
        return pd.Series(dtype=float)
    method_norm = str(method).strip().lower()
    if method_norm not in set(VAR_BACKTEST_METHODS):
        raise InputError("method must be one of {'hist', 'cf', 'fhs'}.")
    if method_norm == "hist":
        # One-step-ahead VaR forecast (no look-ahead): estimate at t-1, test at t.
        return r.rolling(int(lookback), min_periods=int(lookback)).quantile(float(alpha)).shift(1)

    idx = r.index
    q = pd.Series(np.nan, index=idx, dtype=float)
    # One-step-ahead VaR forecast (no look-ahead): estimate from [t-lookback, t-1].
    for i in range(int(lookback), len(r)):
        window = r.iloc[i - int(lookback) : i]
        if method_norm == "cf":
            v, _ = cf_var_es(window, alpha=alpha, n_sim=cf_n_sim, seed=cf_seed)
        elif method_norm == "fhs":
            v, _ = fhs_var_es(window, alpha=alpha, lam=fhs_lambda)
        q.iloc[i] = -float(v) if np.isfinite(v) else np.nan
    return q


def rolling_var(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lookback: int = 252,
    method: Literal["hist", "cf", "fhs"] = "hist",
    cf_n_sim: int = 15_000,
    cf_seed: int = 7,
    fhs_lambda: float = 0.94,
) -> pd.Series:
    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    return _rolling_var_quantile(
        r,
        alpha=a,
        lookback=int(lookback),
        method=method,
        cf_n_sim=int(cf_n_sim),
        cf_seed=int(cf_seed),
        fhs_lambda=float(fhs_lambda),
    )


def longest_true_streak(mask: Sequence[bool] | np.ndarray | pd.Series) -> int:
    arr = np.asarray(mask, dtype=bool)
    best = 0
    cur = 0
    for v in arr:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def kupiec_test(breach: Sequence[bool] | np.ndarray | pd.Series, *, alpha: float = 0.05) -> tuple[float, float]:
    a = _normalize_alpha(alpha)
    b = np.asarray(breach, dtype=bool)
    n = int(len(b))
    x = int(np.sum(b))
    if n == 0:
        return float("nan"), float("nan")
    eps = 1e-12
    ph = x / n
    ph = min(max(ph, eps), 1 - eps)
    ll0 = (n - x) * math.log1p(-a) + x * math.log(a)
    ll1 = (n - x) * math.log1p(-ph) + x * math.log(ph)
    lr = -2.0 * (ll0 - ll1)
    p = float(1.0 - chi2.cdf(lr, 1)) if chi2 is not None else float("nan")
    return float(lr), p


def christoffersen_independence(
    breach: Sequence[bool] | np.ndarray | pd.Series,
) -> tuple[float, float]:
    b = np.asarray(breach, dtype=int)
    if len(b) < 3:
        return float("nan"), float("nan")
    b0 = b[:-1]
    b1 = b[1:]
    n00 = int(np.sum((b0 == 0) & (b1 == 0)))
    n01 = int(np.sum((b0 == 0) & (b1 == 1)))
    n10 = int(np.sum((b0 == 1) & (b1 == 0)))
    n11 = int(np.sum((b0 == 1) & (b1 == 1)))
    eps = 1e-12
    pi01 = n01 / (n00 + n01 + eps)
    pi11 = n11 / (n10 + n11 + eps)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11 + eps)
    pi01 = min(max(pi01, eps), 1 - eps)
    pi11 = min(max(pi11, eps), 1 - eps)
    pi = min(max(pi, eps), 1 - eps)
    ll0 = (n00 + n10) * math.log1p(-pi) + (n01 + n11) * math.log(pi)
    ll1 = (
        n00 * math.log1p(-pi01)
        + n01 * math.log(pi01)
        + n10 * math.log1p(-pi11)
        + n11 * math.log(pi11)
    )
    lr = -2.0 * (ll0 - ll1)
    p = float(1.0 - chi2.cdf(lr, 1)) if chi2 is not None else float("nan")
    return float(lr), p


def quantile_loss(
    returns: pd.Series | np.ndarray | Sequence[float],
    quantile_forecast: pd.Series | np.ndarray | Sequence[float],
    *,
    alpha: float = 0.05,
) -> float:
    """
    Mean pinball loss for lower-tail quantile forecasts. Lower is better.
    """
    a = _normalize_alpha(alpha)
    y = pd.to_numeric(pd.Series(returns), errors="coerce")
    q = pd.to_numeric(pd.Series(quantile_forecast), errors="coerce")
    z = pd.concat([y.rename("ret"), q.rename("q")], axis=1).dropna()
    if z.empty:
        return float("nan")
    e = z["ret"] - z["q"]
    # For alpha-quantile: rho_alpha(e) = e*(alpha - 1{e<0})
    loss = e * (a - (e < 0.0).astype(float))
    return float(np.mean(loss.to_numpy(dtype=float)))


def breach_stats(
    returns: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    lookback: int = 252,
    method: Literal["hist", "cf", "fhs"] = "hist",
) -> dict[str, Any]:
    a = _normalize_alpha(alpha)
    r = _to_numeric_series(returns, name="returns")
    q = rolling_var(r, alpha=a, lookback=int(lookback), method=method)
    z = pd.concat([r.rename("ret"), q.rename("var_q")], axis=1).dropna()
    if z.empty:
        return {
            "series": z,
            "breach": pd.Series(dtype=bool),
            "count": 0,
            "rate": float("nan"),
            "longest_streak": 0,
            "avg_gap": float("nan"),
            "med_gap": float("nan"),
            "kupiec_lr": float("nan"),
            "kupiec_p": float("nan"),
            "christ_lr": float("nan"),
            "christ_p": float("nan"),
        }
    br = z["ret"] < z["var_q"]
    lr_uc, p_uc = kupiec_test(br, alpha=a)
    lr_ind, p_ind = christoffersen_independence(br)
    loc = np.flatnonzero(br.to_numpy(dtype=bool))
    gaps = np.diff(loc) if len(loc) >= 2 else np.array([], dtype=int)
    return {
        "series": z,
        "breach": br,
        "count": int(br.sum()),
        "rate": float(br.mean()),
        "coverage_error": float(br.mean() - a),
        "abs_coverage_error": float(abs(br.mean() - a)),
        "longest_streak": longest_true_streak(br),
        "avg_gap": float(np.mean(gaps)) if len(gaps) else float("nan"),
        "med_gap": float(np.median(gaps)) if len(gaps) else float("nan"),
        "kupiec_lr": float(lr_uc),
        "kupiec_p": float(p_uc),
        "christ_lr": float(lr_ind),
        "christ_p": float(p_ind),
        "quantile_loss": quantile_loss(z["ret"], z["var_q"], alpha=a),
    }


def performance_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    ann = float(annualization)
    for name, r in obj.items():
        nav = nav_series(r)
        n = len(r)
        years = n / ann if ann > 0 else float("nan")
        ann_ret = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else float("nan")
        dvol = float(r.std(ddof=1)) if n > 1 else float("nan")
        ann_vol = dvol * math.sqrt(ann) if np.isfinite(dvol) else float("nan")
        sharpe = (
            float((r.mean() - float(rf_daily)) / dvol * math.sqrt(ann))
            if np.isfinite(dvol) and dvol > 1e-12
            else float("nan")
        )
        rows.append(
            {
                "object": name,
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": sortino_ratio(r, rf_daily=rf_daily, annualization=annualization),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()


def tail_shape_table(objects: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        q05 = float(r.quantile(0.05)) if len(r) else float("nan")
        q95 = float(r.quantile(0.95)) if len(r) else float("nan")
        tail_ratio = float(abs(q95 / q05)) if abs(q05) > 1e-12 else float("nan")
        rows.append(
            {
                "object": name,
                "skew": float(r.skew()) if len(r) else float("nan"),
                "excess_kurtosis": float(r.kurt()) if len(r) else float("nan"),
                "tail_ratio_95_05": tail_ratio,
                "worst_1d": float(r.min()) if len(r) else float("nan"),
                "worst_5d_avg": float(r.nsmallest(5).mean()) if len(r) >= 5 else float("nan"),
                "worst_10d_avg": float(r.nsmallest(10).mean()) if len(r) >= 10 else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()


def drawdown_summary_table(objects: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        dd = drawdown_series(r, input_kind="returns")
        ep = drawdown_episodes(r, input_kind="returns")
        rows.append(
            {
                "object": name,
                "max_dd": float(dd.min()) if len(dd) else float("nan"),
                "longest_dd_days": int(ep["duration"].max()) if len(ep) else 0,
                "avg_recovery_days": avg_recovery_time(r, input_kind="returns"),
                "ulcer_index": ulcer_index(r, input_kind="returns"),
            }
        )
    return pd.DataFrame(rows).set_index("object").sort_index()


def drawdown_episodes_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    top_n: int = 1,
) -> pd.DataFrame:
    if top_n <= 0:
        raise InputError("top_n must be positive.")
    obj = _coerce_objects(objects)
    rows: list[pd.DataFrame] = []
    for name, r in obj.items():
        ep = drawdown_episodes(r, input_kind="returns").sort_values("depth")
        ep = ep.head(int(top_n)).copy()
        if ep.empty:
            continue
        ep.insert(0, "object", name)
        rows.append(ep)
    if not rows:
        return pd.DataFrame(columns=["object", "start", "end", "depth", "duration"])
    return pd.concat(rows, axis=0).reset_index(drop=True)


def var_es_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    alpha: float = 0.05,
    methods: Sequence[str] = ("hist", "cf", "fhs"),
) -> pd.DataFrame:
    a = _normalize_alpha(alpha)
    obj = _coerce_objects(objects)
    methods_norm = [str(m).strip().lower() for m in methods]
    if not methods_norm:
        raise InputError("methods must contain at least one method.")
    valid = {"hist", "cf", "fhs"}
    unknown = [m for m in methods_norm if m not in valid]
    if unknown:
        raise InputError(f"Unknown VaR/ES method(s): {unknown}")
    p = int(round(a * 100))
    rows: list[dict[str, Any]] = []
    for name, r in obj.items():
        row: dict[str, Any] = {"object": name}
        if "hist" in methods_norm:
            v, e = hist_var_es(r, alpha=a)
            row[f"hist_var{p}"] = v
            row[f"hist_es{p}"] = e
        if "cf" in methods_norm:
            v, e = cf_var_es(r, alpha=a)
            row[f"cf_var{p}"] = v
            row[f"cf_es{p}"] = e
        if "fhs" in methods_norm:
            v, e = fhs_var_es(r, alpha=a)
            row[f"fhs_var{p}"] = v
            row[f"fhs_es{p}"] = e
        rows.append(row)
    return pd.DataFrame(rows).set_index("object").sort_index()

def var_backtest_details(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    alpha: float = 0.05,
    method: Literal["hist", "cf", "fhs"] = "hist",
    lookback: int = 252,
) -> dict[str, dict[str, Any]]:
    obj = _coerce_objects(objects)
    return {
        name: breach_stats(r, alpha=alpha, lookback=int(lookback), method=method)
        for name, r in obj.items()
    }


def _rank_var_backtest_accuracy(tbl: pd.DataFrame) -> pd.DataFrame:
    out = tbl.copy()
    out["accuracy_rank"] = np.nan
    out["accuracy_score"] = np.nan
    out["is_best"] = False
    if not isinstance(out.index, pd.MultiIndex) or out.index.nlevels < 2:
        out["accuracy_rank"] = 1.0
        out["accuracy_score"] = 1.0
        out["is_best"] = True
        return out

    for obj_name, g in out.groupby(level=0, sort=False):
        abs_cov = g["abs_coverage_error"].astype(float)
        qloss = g["quantile_loss"].astype(float)
        kup = g["kupiec_p"].astype(float).fillna(-np.inf)
        chrp = g["christoffersen_p"].astype(float).fillna(-np.inf)
        r_abs = abs_cov.rank(ascending=True, method="min", na_option="bottom")
        r_ql = qloss.rank(ascending=True, method="min", na_option="bottom")
        r_k = kup.rank(ascending=False, method="min")
        r_c = chrp.rank(ascending=False, method="min")
        rank_sum = (r_abs + r_ql + r_k + r_c).astype(float)
        acc_rank = rank_sum.rank(ascending=True, method="min")
        score = 1.0 / (1.0 + rank_sum)
        out.loc[g.index, "accuracy_rank"] = acc_rank.to_numpy(dtype=float)
        out.loc[g.index, "accuracy_score"] = score.to_numpy(dtype=float)

        sort_df = pd.DataFrame(
            {
                "rank_sum": rank_sum,
                "abs_cov": abs_cov,
                "qloss": qloss,
                "kupiec_p": kup,
                "christ_p": chrp,
                "method_name": [str(idx[1]) for idx in g.index],
            },
            index=g.index,
        )
        best_idx = sort_df.sort_values(
            by=["rank_sum", "abs_cov", "qloss", "kupiec_p", "christ_p", "method_name"],
            ascending=[True, True, True, False, False, True],
        ).index[0]
        out.loc[best_idx, "is_best"] = True
    return out


def var_backtest_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    alpha: float = 0.05,
    method: Literal["hist", "cf", "fhs"] = "hist",
    methods: Sequence[str] | None = None,
    lookback: int = 252,
) -> pd.DataFrame:
    a = _normalize_alpha(alpha)
    methods_norm = _normalize_var_methods(method=method, methods=methods)
    obj = _coerce_objects(objects)

    rows: list[dict[str, Any]] = []
    for m in methods_norm:
        details = {name: breach_stats(r, alpha=a, lookback=int(lookback), method=m) for name, r in obj.items()}
        for name, st in details.items():
            rows.append(
                {
                    "object": str(name),
                    "method": str(m),
                    "breach_count": st["count"],
                    "breach_rate": st["rate"],
                    "coverage_error": st["coverage_error"],
                    "abs_coverage_error": st["abs_coverage_error"],
                    "longest_breach_streak": st["longest_streak"],
                    "avg_gap_days": st["avg_gap"],
                    "kupiec_p": st["kupiec_p"],
                    "christoffersen_p": st["christ_p"],
                    "quantile_loss": st["quantile_loss"],
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "breach_count",
                "breach_rate",
                "coverage_error",
                "abs_coverage_error",
                "longest_breach_streak",
                "avg_gap_days",
                "kupiec_p",
                "christoffersen_p",
                "quantile_loss",
                "accuracy_rank",
                "accuracy_score",
                "is_best",
            ]
        )

    if len(methods_norm) == 1 and methods is None:
        out = out.drop(columns=["method"]).set_index("object").sort_index()
        return _rank_var_backtest_accuracy(out)

    out = out.set_index(["object", "method"]).sort_index()
    return _rank_var_backtest_accuracy(out)


def best_var_methods(var_backtest_tbl: pd.DataFrame) -> dict[str, str]:
    """
    Return best VaR backtest method by object from a var_backtest_table output.
    """
    if var_backtest_tbl is None or var_backtest_tbl.empty:
        return {}
    idx = var_backtest_tbl.index
    if isinstance(idx, pd.MultiIndex) and idx.nlevels >= 2:
        tbl = var_backtest_tbl.copy()
        if "is_best" in tbl.columns and tbl["is_best"].any():
            best = tbl[tbl["is_best"]]
        elif "accuracy_rank" in tbl.columns:
            best = tbl.sort_values(["accuracy_rank", "abs_coverage_error", "quantile_loss"]).groupby(level=0).head(1)
        else:
            best = tbl.groupby(level=0).head(1)
        return {str(k[0]): str(k[1]) for k in best.index}
    if "method" in var_backtest_tbl.columns:
        m = str(var_backtest_tbl["method"].iloc[0]) if len(var_backtest_tbl) else "hist"
        return {str(k): m for k in var_backtest_tbl.index}
    return {str(k): "hist" for k in var_backtest_tbl.index}


def _window_returns(
    series: pd.Series,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    return series.loc[(series.index >= start) & (series.index <= end)].dropna()


def stress_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    windows: Mapping[str, tuple[str | pd.Timestamp, str | pd.Timestamp]],
    worst_only: bool = True,
    worst_by: Literal["cum_return", "max_dd", "worst_day", "worst_week"] = "cum_return",
) -> pd.DataFrame:
    if not windows:
        raise InputError("windows cannot be empty.")
    valid_worst_by = {"cum_return", "max_dd", "worst_day", "worst_week"}
    if worst_by not in valid_worst_by:
        raise InputError(f"worst_by must be one of {sorted(valid_worst_by)}.")
    obj = _coerce_objects(objects)
    rows: list[dict[str, Any]] = []
    for wname, (start, end) in windows.items():
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if e < s:
            s, e = e, s
        for name, r0 in obj.items():
            r = _to_datetime_if_possible(r0)
            x = _window_returns(r, start=s, end=e)
            if len(x) == 0:
                continue
            nav = nav_series(x)
            dd = nav / nav.cummax() - 1.0
            has_dates = isinstance(x.index, pd.DatetimeIndex)
            worst_week = x.resample("W-FRI").sum().min() if has_dates and len(x) > 5 else float("nan")
            rows.append(
                {
                    "window": str(wname),
                    "object": str(name),
                    "cum_return": float(nav.iloc[-1] - 1.0),
                    "max_dd": float(dd.min()),
                    "worst_day": float(x.min()),
                    "worst_week": float(worst_week) if np.isfinite(worst_week) else float("nan"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["object", "cum_return", "max_dd", "worst_day", "worst_week"])
    out = pd.DataFrame(rows)
    if not bool(worst_only):
        return out.set_index("window").sort_values(["window", "object"])

    # One worst stress scenario row per object; keep the source window for context.
    out = out.sort_values([worst_by, "window"]).groupby("object", as_index=False).first()
    return out.set_index("object").sort_index()


def capm_ols(
    y_excess: pd.Series | Sequence[float] | np.ndarray,
    x_excess: pd.Series | Sequence[float] | np.ndarray,
) -> tuple[float, float, float]:
    y = _to_numeric_series(y_excess, name="y_excess")
    x = _to_numeric_series(x_excess, name="x_excess")
    z = _align_pair(y, x)
    if len(z) < 3:
        return float("nan"), float("nan"), float("nan")
    xv = z["x"].to_numpy(dtype=float)
    yv = z["y"].to_numpy(dtype=float)
    xmat = np.column_stack([np.ones(len(xv), dtype=float), xv])
    coef = np.linalg.lstsq(xmat, yv, rcond=None)[0]
    alpha = float(coef[0])
    beta = float(coef[1])
    yhat = xmat @ coef
    ssr = float(np.sum((yv - yhat) ** 2))
    sst = float(np.sum((yv - np.mean(yv)) ** 2))
    r2 = 1.0 - ssr / sst if sst > 1e-12 else float("nan")
    return alpha, beta, float(r2)


def rolling_beta_corr(
    returns: pd.Series | Sequence[float] | np.ndarray,
    market_ret: pd.Series | Sequence[float] | np.ndarray,
    *,
    window: int,
) -> tuple[pd.Series, pd.Series]:
    if int(window) < 5:
        raise InputError("window must be >= 5.")
    r = _to_numeric_series(returns, name="returns")
    m = _to_numeric_series(market_ret, name="market_ret")
    z = _align_pair(r, m)
    beta = z["y"].rolling(int(window)).cov(z["x"]) / z["x"].rolling(int(window)).var()
    corr = z["y"].rolling(int(window)).corr(z["x"])
    beta.name = f"beta_{int(window)}"
    corr.name = f"corr_{int(window)}"
    return beta, corr


def _normalize_windows(rolling: int | Sequence[int] | None) -> list[int]:
    if rolling is None:
        return [126, 252]
    if isinstance(rolling, int):
        vals = [126, int(rolling)]
    else:
        vals = [int(v) for v in rolling if int(v) > 1]
        if not vals:
            vals = [126, 252]
    out = sorted(set(vals))
    return out


def capm_table(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    market_ret: pd.Series | Sequence[float] | np.ndarray,
    rf_daily: float = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    rolling: int | Sequence[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    obj = _coerce_objects(objects)
    m = _to_numeric_series(market_ret, name="market_ret")
    m = _to_datetime_if_possible(m)
    windows = _normalize_windows(rolling)
    rows: list[dict[str, Any]] = []
    roll: dict[str, pd.DataFrame] = {}
    ann = float(annualization)

    for name, r in obj.items():
        y = _to_datetime_if_possible(r)
        y_ex = y - float(rf_daily)
        m_ex = m - float(rf_daily)
        z_ex = _align_pair(y_ex, m_ex)
        alpha_d, beta, r2 = capm_ols(z_ex["y"], z_ex["x"])
        alpha_ann = (1.0 + alpha_d) ** ann - 1.0 if alpha_d > -0.999 else float("nan")

        z_raw = _align_pair(y, m)
        active = z_raw["y"] - z_raw["x"]
        has_var = len(active) > 1 and active.std(ddof=1) > 1e-12
        te = float(active.std(ddof=1) * math.sqrt(ann)) if has_var else float("nan")
        ir = float(active.mean() / active.std(ddof=1) * math.sqrt(ann)) if has_var else float("nan")

        up = z_raw["x"] > 0
        dn = z_raw["x"] < 0
        up_den = float(z_raw.loc[up, "x"].mean()) if up.any() else float("nan")
        dn_den = float(z_raw.loc[dn, "x"].mean()) if dn.any() else float("nan")
        up_cap = (
            float(z_raw.loc[up, "y"].mean() / up_den)
            if up.sum() > 10 and np.isfinite(up_den) and abs(up_den) > 1e-12
            else float("nan")
        )
        dn_cap = (
            float(z_raw.loc[dn, "y"].mean() / dn_den)
            if dn.sum() > 10 and np.isfinite(dn_den) and abs(dn_den) > 1e-12
            else float("nan")
        )

        vy = float(np.var(z_ex["y"].to_numpy(dtype=float), ddof=1)) if len(z_ex) > 1 else float("nan")
        vm = float(np.var(z_ex["x"].to_numpy(dtype=float), ddof=1)) if len(z_ex) > 1 else float("nan")
        sys_share = ((beta**2) * vm / vy) if np.isfinite(vy) and vy > 1e-12 and np.isfinite(vm) else float("nan")

        rows.append(
            {
                "object": name,
                "alpha_daily": alpha_d,
                "alpha_ann": alpha_ann,
                "beta": beta,
                "r2": r2,
                "tracking_error": te,
                "information_ratio": ir,
                "up_capture": up_cap,
                "down_capture": dn_cap,
                "systematic_var_share": sys_share,
            }
        )

        roll_cols: dict[str, pd.Series] = {}
        for w in windows:
            b, c = rolling_beta_corr(y_ex, m_ex, window=int(w))
            roll_cols[f"beta_{w}"] = b
            roll_cols[f"corr_{w}"] = c
        roll[name] = pd.DataFrame(roll_cols)

    capm = pd.DataFrame(rows).set_index("object").sort_index()
    return capm, roll


def corr_matrix(
    objects: Mapping[str, Any] | pd.DataFrame,
    *,
    min_periods: int = 20,
) -> pd.DataFrame:
    obj = _coerce_objects(objects)
    mat = pd.concat({k: v for k, v in obj.items()}, axis=1)
    return mat.corr(min_periods=int(min_periods))

def vol_contribution(
    weights: pd.Series | Sequence[float] | np.ndarray,
    cov: np.ndarray,
    *,
    index: Sequence[str] | None = None,
) -> pd.Series:
    w = np.asarray(weights, dtype=float).reshape(-1)
    S = np.asarray(cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("cov must be a square matrix.")
    if S.shape[0] != w.shape[0]:
        raise InputError("weights length must match cov shape.")
    S = 0.5 * (S + S.T)
    m = S @ w
    var = float(w @ m)
    vol = math.sqrt(max(var, 1e-18))
    rc = (w * m) / vol
    labels = [str(i) for i in index] if index is not None else [f"a{i}" for i in range(len(w))]
    return pd.Series(rc, index=labels, dtype=float)


def scenario_es_contribution(
    returns_window: pd.DataFrame,
    weights: pd.Series | Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
) -> pd.Series:
    a = _normalize_alpha(alpha)
    if not isinstance(returns_window, pd.DataFrame) or returns_window.empty:
        raise InputError("returns_window must be a non-empty DataFrame.")
    x = (
        returns_window.copy()
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna(axis=0, how="any")
    )
    if x.empty:
        raise InputError("returns_window is empty after cleaning.")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if x.shape[1] != w.shape[0]:
        raise InputError("returns_window columns must match weights length.")
    rp = x.to_numpy(dtype=float) @ w
    q = float(np.quantile(rp, a))
    mask = rp <= q
    if not np.any(mask):
        mask[np.argmin(rp)] = True
    contrib = -np.mean(x.to_numpy(dtype=float)[mask] * w, axis=0)
    return pd.Series(contrib, index=[str(c) for c in x.columns], dtype=float)


def _resolve_state_date(cache: Mapping[Any, Any], dt: pd.Timestamp) -> pd.Timestamp | None:
    idx = pd.DatetimeIndex(pd.to_datetime(list(cache.keys()))).sort_values().unique()
    if len(idx) == 0:
        return None
    if dt in idx:
        return pd.Timestamp(dt)
    pos = int(idx.searchsorted(dt, side="right")) - 1
    if pos < 0:
        return None
    return pd.Timestamp(idx[pos])


def _weights_state_from_spec(
    spec: Mapping[str, Any],
    *,
    date: pd.Timestamp | None = None,
) -> tuple[pd.Series, Mapping[str, Any], pd.Timestamp]:
    result = spec.get("backtest", spec.get("result"))
    cache = spec.get("state_cache", spec.get("cache"))
    if result is None or cache is None:
        raise InputError("Portfolio spec requires 'backtest' (or 'result') and 'state_cache' (or 'cache').")
    res_map = _as_result_mapping(result)
    wdf = pd.DataFrame(res_map.get("weights"))
    if wdf.empty:
        raise InputError("Portfolio result has empty weights.")
    wdf.index = pd.to_datetime(wdf.index)
    dt = pd.Timestamp(date) if date is not None else pd.Timestamp(wdf.index[-1])
    st_dt = _resolve_state_date(cache, dt)
    if st_dt is None:
        raise InputError("Could not resolve state date from state cache.")
    state = _as_state_mapping(cache[st_dt])
    tickers = [str(t) for t in state.get("tickers", [])]
    if not tickers:
        raise InputError("State is missing tickers.")
    if dt not in wdf.index:
        pos = int(wdf.index.searchsorted(dt, side="right")) - 1
        if pos < 0:
            raise InputError("No weights available on or before requested date.")
        dt = pd.Timestamp(wdf.index[pos])
    w = wdf.loc[dt].reindex(tickers).fillna(0.0).astype(float)
    s = float(w.sum())
    if not np.isfinite(s) or abs(s) <= 1e-12:
        raise InputError("Resolved weights sum to zero.")
    w = w / s
    return w, state, pd.Timestamp(dt)


def portfolio_contribution_snapshot(
    portfolio_spec: Mapping[str, Any],
    *,
    cov_key: str | None = None,
    es_alpha: float = 0.05,
    date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (volatility contribution, scenario-ES contribution) for one portfolio snapshot.

    If a returns window is unavailable in the state cache, ES contribution is returned
    as an all-NaN series (same index as vol contribution) instead of raising.
    """
    w, state, _ = _weights_state_from_spec(portfolio_spec, date=date)
    ck = str(cov_key or portfolio_spec.get("cov_key", "ledoitwolf"))
    cov_map = state.get("cov_ann_map", {})
    if ck not in cov_map:
        low = {str(k).lower(): k for k in cov_map}
        if ck.lower() in low:
            ck = low[ck.lower()]
        else:
            raise InputError(f"cov_key {ck!r} not found in state covariance map.")
    cov = np.asarray(cov_map[ck], dtype=float)
    vol_rc = vol_contribution(w.to_numpy(dtype=float), cov, index=w.index).sort_values(ascending=False)

    window = state.get("window")
    if window is None:
        meta = state.get("metadata")
        if isinstance(meta, Mapping):
            window = meta.get("window")
    if window is None:
        es_rc = pd.Series(np.nan, index=vol_rc.index, dtype=float)
        return vol_rc, es_rc
    if not isinstance(window, pd.DataFrame):
        window = pd.DataFrame(window, columns=w.index)
    x = window.reindex(columns=w.index)
    try:
        es_rc = scenario_es_contribution(x, w.to_numpy(dtype=float), alpha=es_alpha).sort_values(ascending=False)
    except Exception:
        es_rc = pd.Series(np.nan, index=vol_rc.index, dtype=float)
    return vol_rc, es_rc


def attribution_tables(
    portfolios: Mapping[str, Any],
    *,
    es_alpha: float = 0.05,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not portfolios:
        raise InputError("portfolios cannot be empty.")
    if top_k <= 0:
        raise InputError("top_k must be positive.")
    vol_map: dict[str, pd.Series] = {}
    es_map: dict[str, pd.Series] = {}
    overlap_rows: list[dict[str, Any]] = []
    for pname, raw in portfolios.items():
        if isinstance(raw, Mapping):
            spec = raw
        elif isinstance(raw, tuple) and len(raw) >= 2:
            spec = {"backtest": raw[0], "state_cache": raw[1], "cov_key": raw[2] if len(raw) > 2 else None}
        else:
            raise InputError("Each portfolio entry must be a mapping or tuple.")
        vol_rc, es_rc = portfolio_contribution_snapshot(spec, es_alpha=es_alpha)
        vol_map[str(pname)] = vol_rc
        es_map[str(pname)] = es_rc
        es_rank = es_rc.dropna()
        overlap = len(set(vol_rc.head(int(top_k)).index).intersection(set(es_rank.head(int(top_k)).index)))
        overlap_rows.append({"portfolio": str(pname), f"top{int(top_k)}_overlap_count": int(overlap)})
    vol_tbl = pd.DataFrame.from_dict(vol_map, orient="index").sort_index(axis=0)
    es_tbl = pd.DataFrame.from_dict(es_map, orient="index").sort_index(axis=0)
    overlap_tbl = pd.DataFrame(overlap_rows).set_index("portfolio").sort_index()
    return vol_tbl, es_tbl, overlap_tbl


def _display_table(df: pd.DataFrame, *, round_digits: int = 4) -> None:
    show = df.round(int(round_digits)) if isinstance(df, pd.DataFrame) else df
    if ipy_display is not None:
        ipy_display(show)
    else:  # pragma: no cover
        print(show)


def executive_bullets(
    *,
    perf_tbl: pd.DataFrame | None = None,
    dd_tbl: pd.DataFrame | None = None,
    var_tbl: pd.DataFrame | None = None,
    capm_tbl: pd.DataFrame | None = None,
    var_bt_tbl: pd.DataFrame | None = None,
) -> list[str]:
    bullets: list[str] = []
    if perf_tbl is not None and not perf_tbl.empty and "sharpe" in perf_tbl.columns:
        best = str(perf_tbl["sharpe"].idxmax())
        bullets.append(f"{best} has the highest realized Sharpe ratio.")
    if dd_tbl is not None and not dd_tbl.empty and "max_dd" in dd_tbl.columns:
        best_dd = str(dd_tbl["max_dd"].idxmax())
        bullets.append(f"Least severe maximum drawdown: {best_dd} ({dd_tbl.loc[best_dd, 'max_dd']:.2%}).")
    if var_tbl is not None and not var_tbl.empty:
        es_cols = [c for c in var_tbl.columns if c.startswith("hist_es")]
        if es_cols:
            c = es_cols[0]
            low_tail = str(var_tbl[c].idxmin())
            bullets.append(f"Lower historical ES tail risk: {low_tail} ({var_tbl.loc[low_tail, c]:.2%}).")
    if capm_tbl is not None and not capm_tbl.empty and "beta" in capm_tbl.columns:
        hi_beta = str(capm_tbl["beta"].idxmax())
        lo_beta = str(capm_tbl["beta"].idxmin())
        bullets.append(
            f"Highest market beta: {hi_beta} ({capm_tbl.loc[hi_beta, 'beta']:.2f}); "
            f"lowest: {lo_beta} ({capm_tbl.loc[lo_beta, 'beta']:.2f})."
        )
    if var_bt_tbl is not None and not var_bt_tbl.empty:
        bt_eval = var_bt_tbl
        if isinstance(bt_eval.index, pd.MultiIndex) and bt_eval.index.nlevels >= 2:
            if "is_best" in bt_eval.columns and bt_eval["is_best"].any():
                bt_eval = bt_eval[bt_eval["is_best"]].copy()
            elif "accuracy_rank" in bt_eval.columns:
                bt_eval = (
                    bt_eval.sort_values(["accuracy_rank", "abs_coverage_error", "quantile_loss"])
                    .groupby(level=0)
                    .head(1)
                )
        issue = []
        if "kupiec_p" in bt_eval.columns:
            issue.extend([str(i[0] if isinstance(i, tuple) else i) for i, v in bt_eval["kupiec_p"].items() if np.isfinite(v) and v < 0.05])
        if "christoffersen_p" in bt_eval.columns:
            issue.extend(
                [str(i[0] if isinstance(i, tuple) else i) for i, v in bt_eval["christoffersen_p"].items() if np.isfinite(v) and v < 0.05]
            )
        issue = sorted(set(issue))
        if issue:
            bullets.append("Potential VaR model instability (p<0.05): " + ", ".join(issue) + ".")
    return bullets


def risk_report(
    *,
    objects: Mapping[str, Any] | pd.DataFrame,
    market_ret: pd.Series | Sequence[float] | np.ndarray | None = None,
    rf_daily: float = 0.0,
    portfolios: Mapping[str, Any] | None = None,
    include: Mapping[str, bool] | None = None,
    var_settings: Mapping[str, Any] | None = None,
    backtest_settings: Mapping[str, Any] | None = None,
    rolling_settings: Mapping[str, Any] | None = None,
    stress_settings: Mapping[str, Any] | None = None,
    attribution_settings: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
) -> RiskReportArtifacts:
    from . import plots as pl

    obj = _coerce_objects(objects)
    names = list(obj.keys())

    include_cfg = {
        "performance_tables": True,
        "shape_tables": True,
        "drawdowns": True,
        "drawdown_episodes": True,
        "var_es": True,
        "var_backtest": True,
        "stress": True,
        "capm": True,
        "rolling_beta": True,
        "correlation": True,
        "attribution": True,
        "exec_bullets": True,
    }
    if include:
        include_cfg.update({str(k): bool(v) for k, v in include.items()})

    var_cfg = {"alpha": 0.05, "methods": ["hist", "cf", "fhs"], "lookback": 252}
    if var_settings:
        var_cfg.update(dict(var_settings))
    bt_cfg = {
        "alpha": 0.05,
        "methods": list(VAR_BACKTEST_METHODS),
        "lookback": 252,
        # Method used in VaR breach plots. "best" chooses the highest-ranked method per object.
        "plot_method": "best",
    }
    if backtest_settings:
        bt_cfg.update(dict(backtest_settings))
    bt_methods = _normalize_var_methods(
        method=(None if "methods" in bt_cfg and bt_cfg.get("methods") is not None else bt_cfg.get("method")),
        methods=bt_cfg.get("methods"),
    )
    bt_plot_method = str(bt_cfg.get("plot_method", "best")).strip().lower()
    roll_cfg = {"vol_windows": [20, 60, 252], "beta_windows": [126, 252]}
    if rolling_settings:
        roll_cfg.update(dict(rolling_settings))
    stress_cfg = {
        "windows": {
            "2018_q4": ("2018-10-01", "2018-12-31"),
            "2020_covid": ("2020-02-20", "2020-04-30"),
            "2022_inflation": ("2022-01-03", "2022-10-31"),
        }
    }
    if stress_settings:
        stress_cfg.update(dict(stress_settings))
    attr_cfg = {"es_alpha": 0.05, "top_k": 10}
    if attribution_settings:
        attr_cfg.update(dict(attribution_settings))
    layout_cfg = {"ncols": 2, "sharex": True, "sharey": True}
    if layout:
        layout_cfg.update(dict(layout))
    output_cfg = {
        "round_tables": 4,
        "print_exec_bullets": True,
        "display_tables": True,
        "show_figures": True,
        # Optional controls to display only a subset of computed tables.
        "display_table_keys": None,
        "hide_table_keys": [],
    }
    if output:
        output_cfg.update(dict(output))

    tables: dict[str, pd.DataFrame] = {}
    series: dict[str, Any] = {}
    figures: dict[str, list[Any]] = {}
    texts: dict[str, list[str]] = {}

    if include_cfg["performance_tables"]:
        tables["performance"] = performance_table(obj, rf_daily=rf_daily, annualization=DEFAULT_ANNUALIZATION)
    if include_cfg["shape_tables"]:
        tables["shape"] = tail_shape_table(obj)
    if include_cfg["drawdowns"]:
        tables["drawdown_summary"] = drawdown_summary_table(obj)
    if include_cfg["drawdown_episodes"]:
        tables["drawdown_episodes"] = drawdown_episodes_table(obj, top_n=1)
    if include_cfg["var_es"]:
        tables["var_es"] = var_es_table(obj, alpha=var_cfg["alpha"], methods=var_cfg["methods"])
    if include_cfg["var_backtest"]:
        tables["var_backtest"] = var_backtest_table(
            obj,
            alpha=bt_cfg["alpha"],
            methods=bt_methods,
            lookback=int(bt_cfg["lookback"]),
        )
        if len(bt_methods) == 1:
            series["var_backtest_detail"] = var_backtest_details(
                obj,
                alpha=bt_cfg["alpha"],
                method=bt_methods[0],
                lookback=int(bt_cfg["lookback"]),
            )
        else:
            series["var_backtest_detail"] = {
                m: var_backtest_details(
                    obj,
                    alpha=bt_cfg["alpha"],
                    method=m,
                    lookback=int(bt_cfg["lookback"]),
                )
                for m in bt_methods
            }
        series["var_backtest_best_method"] = best_var_methods(tables["var_backtest"])
    if include_cfg["stress"]:
        stress_worst_only = bool(stress_cfg.get("worst_only", True))
        stress_worst_by = str(stress_cfg.get("worst_by", "cum_return"))
        tables["stress"] = stress_table(
            obj,
            windows=stress_cfg["windows"],
            worst_only=stress_worst_only,
            worst_by=stress_worst_by,
        )
        # Keep full window-level stress for stress subplot section.
        series["stress_full"] = stress_table(
            obj,
            windows=stress_cfg["windows"],
            worst_only=False,
            worst_by=stress_worst_by,
        )
    if include_cfg["capm"] and market_ret is not None:
        capm_tbl, capm_roll = capm_table(
            obj,
            market_ret=market_ret,
            rf_daily=rf_daily,
            rolling=roll_cfg.get("beta_windows", [126, 252]),
        )
        tables["capm"] = capm_tbl
        series["capm_roll"] = capm_roll
    if include_cfg["correlation"]:
        tables["corr"] = corr_matrix(obj)
    if include_cfg["attribution"] and portfolios:
        vtbl, etbl, otbl = attribution_tables(
            portfolios,
            es_alpha=float(attr_cfg["es_alpha"]),
            top_k=int(attr_cfg["top_k"]),
        )
        tables["attribution_vol"] = vtbl
        tables["attribution_es"] = etbl
        tables["attribution_overlap"] = otbl

    if include_cfg["exec_bullets"]:
        texts["exec_bullets"] = executive_bullets(
            perf_tbl=tables.get("performance"),
            dd_tbl=tables.get("drawdown_summary"),
            var_tbl=tables.get("var_es"),
            capm_tbl=tables.get("capm"),
            var_bt_tbl=tables.get("var_backtest"),
        )

    if bool(output_cfg["display_tables"]):
        show_keys_raw = output_cfg.get("display_table_keys")
        hide_keys = {str(k) for k in output_cfg.get("hide_table_keys", [])}
        if show_keys_raw is None:
            show_keys = list(tables.keys())
        else:
            show_keys = [str(k) for k in show_keys_raw if str(k) in tables]
        for key in show_keys:
            if key in hide_keys:
                continue
            _display_table(tables[key], round_digits=int(output_cfg["round_tables"]))
        if bool(output_cfg["print_exec_bullets"]) and texts.get("exec_bullets"):
            for b in texts["exec_bullets"]:
                print(f"- {b}")

    ncols = max(int(layout_cfg["ncols"]), 1)

    def _grid_size(
        n_items: int,
        *,
        ncols_use: int,
        panel_w: float = 3.8,
        panel_h: float = 2.7,
    ) -> tuple[float, float]:
        cols = max(min(int(ncols_use), max(int(n_items), 1)), 1)
        rows = int(math.ceil(max(int(n_items), 1) / cols))
        width = float(np.clip(panel_w * cols, 8.0, 28.0))
        height = float(np.clip(panel_h * rows, 4.5, 26.0))
        return width, height

    if include_cfg["drawdowns"]:
        fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
        pl.plot_nav_compare(ax[0], obj)
        pl.plot_drawdown_compare_objects(ax[1], obj)
        plt.tight_layout()
        figures.setdefault("drawdown_compare", []).append(fig)
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["drawdowns"]:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        for a, nm in zip(axes, names, strict=False):
            pl.plot_rolling_vol(
                a,
                obj[nm],
                windows=roll_cfg.get("vol_windows", [20, 60, 252]),
                annualization=DEFAULT_ANNUALIZATION,
                name=nm,
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["rolling_vol"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["var_backtest"]:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        best_method_map = series.get("var_backtest_best_method", {}) if isinstance(series.get("var_backtest_best_method", {}), Mapping) else {}
        for a, nm in zip(axes, names, strict=False):
            chosen_method = bt_plot_method
            if bt_plot_method == "best":
                chosen_method = str(best_method_map.get(nm, bt_methods[0]))
            pl.plot_var_backtest(
                a,
                obj[nm],
                alpha=float(bt_cfg["alpha"]),
                lookback=int(bt_cfg["lookback"]),
                method=chosen_method,
                methods=bt_methods,
                name=nm,
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["var_backtest"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["stress"] and "stress_full" in series and isinstance(series["stress_full"], pd.DataFrame) and not series["stress_full"].empty:
        stress_plot_tbl = series["stress_full"]
        windows = [str(w) for w in pd.Index(stress_plot_tbl.index).unique()]
        stress_ncols = min(ncols, max(1, len(windows)))
        fig, axes = pl.auto_grid(
            len(windows),
            ncols=stress_ncols,
            figsize=_grid_size(len(windows), ncols_use=stress_ncols, panel_w=4.0, panel_h=2.9),
            sharex=False,
            sharey=bool(layout_cfg["sharey"]),
        )
        for a, wn in zip(axes, windows, strict=False):
            pl.plot_stress_bar(a, stress_plot_tbl, window=wn, metric="cum_return")
        pl.turn_off_unused_axes(axes, used=len(windows))
        plt.tight_layout()
        figures["stress"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["capm"] and market_ret is not None:
        fig, axes = pl.auto_grid(
            len(names),
            ncols=ncols,
            figsize=_grid_size(len(names), ncols_use=ncols),
            sharex=bool(layout_cfg["sharex"]),
            sharey=bool(layout_cfg["sharey"]),
        )
        color_map = pl.make_color_map(names, pl.LAB_COLORS)
        for a, nm in zip(axes, names, strict=False):
            pl.plot_capm_scatter(
                a,
                obj[nm],
                market_ret,
                rf_daily=rf_daily,
                name=nm,
                color=color_map.get(nm),
            )
        pl.turn_off_unused_axes(axes, used=len(names))
        plt.tight_layout()
        figures["capm_scatter"] = [fig]
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["rolling_beta"] and "capm_roll" in series:
        beta_windows = [int(v) for v in roll_cfg.get("beta_windows", [126, 252]) if int(v) > 1]
        if beta_windows:
            fig, axes = plt.subplots(len(beta_windows), 1, figsize=(11, 3.2 * len(beta_windows)), sharex=True)
            axes_arr = np.asarray([axes]) if isinstance(axes, plt.Axes) else np.asarray(axes).reshape(-1)
            for a, w in zip(axes_arr, beta_windows, strict=False):
                pl.plot_rolling_beta_compare(a, series["capm_roll"], window=int(w), metric="beta")
            plt.tight_layout()
            figures.setdefault("rolling_beta", []).append(fig)
            if bool(output_cfg["show_figures"]):
                plt.show()

    if include_cfg["correlation"] and "corr" in tables:
        fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))
        pl.plot_corr_heatmap(ax, tables["corr"])
        plt.tight_layout()
        figures.setdefault("correlation", []).append(fig)
        if bool(output_cfg["show_figures"]):
            plt.show()

    if include_cfg["attribution"] and portfolios and "attribution_vol" in tables and "attribution_es" in tables:
        pnames = list(tables["attribution_vol"].index)
        top_k = int(attr_cfg["top_k"])
        fig_vol, axes_vol = pl.auto_grid(
            len(pnames),
            ncols=ncols,
            figsize=_grid_size(len(pnames), ncols_use=ncols),
            sharex=False,
            sharey=False,
        )
        for a, pname in zip(axes_vol, pnames, strict=False):
            pl.plot_top_contrib(
                a,
                tables["attribution_vol"].loc[pname],
                title=f"{pname} - Top Vol Contribution",
                k=top_k,
            )
        pl.turn_off_unused_axes(axes_vol, used=len(pnames))
        plt.tight_layout()

        fig_es, axes_es = pl.auto_grid(
            len(pnames),
            ncols=ncols,
            figsize=_grid_size(len(pnames), ncols_use=ncols),
            sharex=False,
            sharey=False,
        )
        for a, pname in zip(axes_es, pnames, strict=False):
            pl.plot_top_contrib(
                a,
                tables["attribution_es"].loc[pname],
                title=f"{pname} - Top ES Contribution",
                k=top_k,
            )
        pl.turn_off_unused_axes(axes_es, used=len(pnames))
        plt.tight_layout()

        figures["attribution"] = [fig_vol, fig_es]
        if bool(output_cfg["show_figures"]):
            plt.show()
            plt.show()

    return RiskReportArtifacts(tables=tables, figures=figures, series=series, text=texts)


__all__ = [
    "DEFAULT_ANNUALIZATION",
    "VAR_BACKTEST_METHODS",
    "nav_series",
    "drawdown_series",
    "ulcer_index",
    "drawdown_episodes",
    "avg_recovery_time",
    "sortino_ratio",
    "hist_var_es",
    "cf_var_es",
    "fhs_var_es",
    "rolling_var",
    "longest_true_streak",
    "kupiec_test",
    "christoffersen_independence",
    "quantile_loss",
    "breach_stats",
    "performance_table",
    "tail_shape_table",
    "drawdown_summary_table",
    "drawdown_episodes_table",
    "var_es_table",
    "var_backtest_details",
    "var_backtest_table",
    "best_var_methods",
    "stress_table",
    "capm_ols",
    "rolling_beta_corr",
    "capm_table",
    "corr_matrix",
    "vol_contribution",
    "scenario_es_contribution",
    "portfolio_contribution_snapshot",
    "attribution_tables",
    "executive_bullets",
    "risk_report",
]
