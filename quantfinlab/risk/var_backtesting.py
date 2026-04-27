from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from quantfinlab.risk.utils import (
    _coerce_objects,
    _normalize_alpha,
    _normalize_var_methods,
    _to_numeric_series,
)
from quantfinlab.risk.var import rolling_var

try:  # optional
    from scipy.stats import chi2
except Exception:  # pragma: no cover
    chi2 = None

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
    n = len(b)
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

    for _obj_name, g in out.groupby(level=0, sort=False):
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

__all__ = [
    "best_var_methods",
    "breach_stats",
    "christoffersen_independence",
    "kupiec_test",
    "longest_true_streak",
    "quantile_loss",
    "var_backtest_details",
    "var_backtest_table",
]
