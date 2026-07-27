from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from quantfinlab.common.contracts import BacktestResult
from quantfinlab.common.errors import InputError
from quantfinlab.portfolio.attribution import effective_number_of_holdings
from quantfinlab.risk.utils import _excess_returns

DEFAULT_ANNUALIZATION = 252.0
MU_ORDER = ("Momentum", "BayesStein", "BayesSteinMomentum")
COV_ORDER = ("Sample", "LedoitWolf", "OAS", "EWMA")


def calc_drawdown(series: pd.Series) -> pd.Series:
    s = pd.Series(series, copy=False).astype(float)
    if s.empty:
        return s
    return s / s.cummax() - 1.0


def _as_result_obj(x: BacktestResult | Mapping[str, Any]) -> BacktestResult:
    if isinstance(x, BacktestResult):
        return x
    required = ["gross_values", "net_values", "gross_returns", "net_returns", "weights", "turnover", "costs"]
    missing = [k for k in required if k not in x]
    if missing:
        raise InputError(f"Result mapping is missing keys: {missing}")
    return BacktestResult(
        gross_values=pd.Series(x["gross_values"]),
        net_values=pd.Series(x["net_values"]),
        gross_returns=pd.Series(x["gross_returns"]),
        net_returns=pd.Series(x["net_returns"]),
        weights=pd.DataFrame(x["weights"]),
        turnover=pd.Series(x["turnover"]),
        costs=pd.Series(x["costs"]),
        fallbacks=int(x.get("fallbacks", 0)),
        metadata=dict(x.get("metadata", {})) if "metadata" in x else None,
    )


def performance_metrics(
    net_returns: pd.Series,
    net_values: pd.Series,
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> dict[str, float]:
    """Compute standard performance metrics from returns and NAV.

    Parameters
    ----------
    net_returns : pandas.Series
        Net periodic returns.
    net_values : pandas.Series
        Net portfolio value or NAV series.
    rf_daily : float or pandas.Series, default=0.0
        Periodic risk-free rate used in Sharpe and Sortino calculations. A
        Series is aligned to the return dates.
    annualization : float, default=252.0
        Annualization factor for return and volatility metrics.

    Returns
    -------
    dict
        Metrics including CAGR, annualized volatility, Sharpe, maximum drawdown,
        Calmar, and Sortino. Values are ``NaN`` when insufficient data are
        available.

    Notes
    -----
    The CAGR calculation assumes the length of ``net_returns`` represents the
    number of periods and divides by ``annualization`` to obtain years.
    """

    r = pd.Series(net_returns, copy=False).dropna().astype(float)
    v = pd.Series(net_values, copy=False).dropna().astype(float)
    if v.empty:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan, "Calmar": np.nan, "Sortino": np.nan}
    years = len(r) / float(annualization) if len(r) > 0 else np.nan
    cagr = float(v.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(float(annualization))) if len(r) > 1 else np.nan
    excess = _excess_returns(r, rf_daily).dropna()
    sharpe_vol = r.std(ddof=1) if np.isscalar(rf_daily) else excess.std(ddof=1)
    sharpe = (
        float(excess.mean() / sharpe_vol * math.sqrt(float(annualization)))
        if len(excess) > 1 and sharpe_vol > 0
        else np.nan
    )
    drawdown = calc_drawdown(v)
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan
    calmar = float(cagr / abs(max_dd)) if np.isfinite(cagr) and np.isfinite(max_dd) and max_dd < 0 else np.nan
    downside = r[r < 0] if np.isscalar(rf_daily) else excess[excess < 0]
    sortino = (
        float(excess.mean() / downside.std(ddof=1) * math.sqrt(float(annualization)))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else np.nan
    )
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Sortino": sortino,
    }


def result_sharpe(
    result: BacktestResult | Mapping[str, Any],
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    min_obs: int = 50,
) -> float:
    res = _as_result_obj(result)
    r = res.net_returns.dropna().astype(float)
    if len(r) < int(min_obs):
        return float("nan")
    excess = _excess_returns(r, rf_daily).dropna()
    sd = float(r.std(ddof=1) if np.isscalar(rf_daily) else excess.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(excess.mean() / sd * math.sqrt(float(annualization)))


def strategy_family(name: str) -> str:
    text = str(name)
    if text.startswith("MaxSharpe (FrontierGrid)") or text.startswith("FrontierGrid"):
        return "FrontierGrid"
    if text.startswith("MaxSharpe"):
        return "MaxSharpe"
    if text.startswith("RidgeMV") or text.startswith("Ridge MV"):
        return "RidgeMV"
    if text.startswith("MinVar"):
        return "MinVar"
    if text.startswith("MV"):
        return "MV"
    if text.startswith("EW"):
        return "EW"
    return text


def _parse_from_name(name: str) -> tuple[str, str | None, str | None]:
    optimizer = strategy_family(name)
    if "(" not in name or ")" not in name:
        return optimizer, None, None
    if name.startswith("MaxSharpe (FrontierGrid)") and name.count("(") >= 2:
        inner = name[name.rfind("(") + 1 : name.rfind(")")]
    else:
        inner = name[name.find("(") + 1 : name.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    if optimizer in {"MV", "RidgeMV", "MaxSharpe", "FrontierGrid"} and len(parts) >= 2:
        return optimizer, parts[1], parts[0]
    if optimizer == "MinVar" and parts:
        return optimizer, None, parts[0]
    return optimizer, None, None


def parse_strategy_spec(name: str, res: BacktestResult | Mapping[str, Any] | None = None) -> tuple[str, str | None, str | None]:
    if isinstance(res, BacktestResult):
        meta = dict(res.metadata or {})
    elif isinstance(res, Mapping):
        meta = dict(res.get("metadata", {})) if "metadata" in res else dict(res)
    else:
        meta = {}
    optimizer, mu_model, cov_model = _parse_from_name(name)
    return (
        str(meta.get("optimizer", optimizer)),
        meta.get("mu_model", mu_model),
        meta.get("cov_model", meta.get("cov_key", cov_model)),
    )


def strategy_display_label(name: str, res: BacktestResult | Mapping[str, Any] | None = None) -> str:
    optimizer, mu_model, cov_model = parse_strategy_spec(name, res)
    if optimizer in {"FrontierGrid", "MaxSharpe (FrontierGrid)"}:
        return f"FrontierGrid [{cov_model}, {mu_model}]"
    if mu_model in (None, "-"):
        return f"{optimizer} [{cov_model}]" if cov_model not in (None, "-") else optimizer
    return f"{optimizer} [{cov_model}, {mu_model}]"


def build_metrics_table(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
    rows = []
    for name, raw in results.items():
        res = _as_result_obj(raw)
        rows.append({"Strategy": str(name), **performance_metrics(res.net_returns, res.net_values, rf_daily=rf_daily, annualization=annualization)})
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


def build_trade_table(results: Mapping[str, BacktestResult | Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, raw in results.items():
        res = _as_result_obj(raw)
        if not res.weights.empty:
            eff_n = effective_number_of_holdings(res.weights)
            avg_eff_n = float(pd.Series(eff_n).mean())
            avg_hhi = float((res.weights.astype(float).fillna(0.0) ** 2).sum(axis=1).mean())
        else:
            avg_hhi = np.nan
            avg_eff_n = np.nan
        final_value = float(res.net_values.iloc[-1]) if not res.net_values.empty else np.nan
        total_cost = float(res.costs.sum()) if not res.costs.empty else 0.0
        rows.append(
            {
                "Strategy": str(name),
                "Avg Turnover": float(res.turnover.mean()) if not res.turnover.empty else 0.0,
                "Total Turnover": float(res.turnover.sum()) if not res.turnover.empty else 0.0,
                "Total Costs": total_cost,
                "Cost Drag": (total_cost / final_value) if final_value and final_value > 0 else np.nan,
                "Avg HHI": avg_hhi,
                "Effective N": avg_eff_n,
                "Fallbacks": int(res.fallbacks),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy") if rows else pd.DataFrame()


def build_strategy_summary(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> pd.DataFrame:
    """Build a combined strategy summary table from backtest results.

    The function joins performance metrics, trade/turnover statistics, and
    strategy metadata parsed from strategy names or result metadata.

    Parameters
    ----------
    results : mapping
        Mapping from strategy name to backtest result object or compatible
        dictionary.
    rf_daily : float or pandas.Series, default=0.0
        Periodic risk-free rate used in performance metrics.
    annualization : float, default=252.0
        Annualization factor.

    Returns
    -------
    pandas.DataFrame
        Strategy-indexed summary table with optimizer, expected-return model,
        covariance model, performance metrics, turnover, cost drag, effective
        number of holdings, and fallback counts where available.

    Notes
    -----
    This table is intended for notebook and report display rather than for
    low-level backtest execution.
    """

    metrics = build_metrics_table(results, rf_daily=rf_daily, annualization=annualization)
    trade = build_trade_table(results)
    meta_rows = []
    for name, raw in results.items():
        optimizer, mu_model, cov_model = parse_strategy_spec(name, raw)
        meta_rows.append(
            {
                "Strategy": str(name),
                "Optimizer": optimizer,
                "Mu model": mu_model if mu_model is not None else "-",
                "Covariance model": cov_model if cov_model is not None else "-",
            }
        )
    meta = pd.DataFrame(meta_rows).set_index("Strategy") if meta_rows else pd.DataFrame()
    if meta.empty:
        return meta
    out = meta.join(metrics, how="left")
    if not trade.empty:
        out = out.join(trade[["Avg Turnover", "Total Turnover", "Cost Drag", "Effective N", "Fallbacks"]], how="left")
        out = out.rename(columns={"Avg Turnover": "Turnover"})
    return out


def summarize_results(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return performance and trade summary tables for a result mapping.

    Parameters
    ----------
    results : mapping
        Mapping from strategy name to backtest result object or compatible
        dictionary.
    rf_daily : float or pandas.Series, default=0.0
        Periodic risk-free rate used in performance metrics.
    annualization : float, default=252.0
        Annualization factor.

    Returns
    -------
    metrics : pandas.DataFrame
        Strategy performance metrics.
    trades : pandas.DataFrame
        Turnover, cost, effective-number, and fallback diagnostics.

    Notes
    -----
    This is a compact convenience wrapper around the metrics and trade-table
    builders.
    """

    return (
        build_metrics_table(results, rf_daily=rf_daily, annualization=annualization),
        build_trade_table(results),
    )


def best_strategy_by_sharpe(
    results: Mapping[str, BacktestResult | Mapping[str, Any]],
    *,
    rf_daily: float | pd.Series = 0.0,
    annualization: float = DEFAULT_ANNUALIZATION,
    min_obs: int = 50,
) -> tuple[str, dict[str, float]]:
    """Select the strategy with the highest Sharpe ratio.

    Parameters
    ----------
    results : mapping
        Mapping from strategy name to backtest result object or compatible
        dictionary.
    rf_daily : float or pandas.Series, default=0.0
        Periodic risk-free rate.
    annualization : float, default=252.0
        Annualization factor.
    min_obs : int, default=50
        Minimum number of return observations required for a valid Sharpe.

    Returns
    -------
    best_name : str
        Name of the best strategy.
    sharpes : dict
        Mapping from strategy name to Sharpe ratio.

    Notes
    -----
    Strategies with invalid Sharpe values are treated as worse than finite
    strategies.
    """

    sharpes = {
        name: result_sharpe(res, rf_daily=rf_daily, annualization=annualization, min_obs=min_obs)
        for name, res in results.items()
    }
    best = max(sharpes, key=lambda k: -np.inf if np.isnan(sharpes[k]) else sharpes[k])
    return str(best), sharpes


def _sort_candidates(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if out.empty:
        return out
    sort_cols = [metric]
    ascending = [False]
    if "Max Drawdown" in out.columns:
        sort_cols.append("Max Drawdown")
        ascending.append(False)
    if "Turnover" in out.columns:
        sort_cols.append("Turnover")
        ascending.append(True)
    return out.sort_values(sort_cols, ascending=ascending)


def filter_grid(
    results: pd.DataFrame,
    *,
    optimizer: str | Sequence[str] | None = None,
    mu_model: str | Sequence[str] | None = None,
    cov_model: str | Sequence[str] | None = None,
    include_frontier: bool = True,
) -> pd.DataFrame:
    df = results.copy()
    if optimizer is not None:
        vals = {optimizer} if isinstance(optimizer, str) else set(optimizer)
        df = df[df["Optimizer"].isin(vals)]
    if mu_model is not None:
        vals = {mu_model} if isinstance(mu_model, str) else set(mu_model)
        df = df[df["Mu model"].isin(vals)]
    if cov_model is not None:
        vals = {cov_model} if isinstance(cov_model, str) else set(cov_model)
        df = df[df["Covariance model"].isin(vals)]
    if not include_frontier:
        df = df[~df["Optimizer"].isin({"FrontierGrid", "MaxSharpe (FrontierGrid)"})]
    return df


def select_best_maxsharpe_combination(results: pd.DataFrame, *, metric: str = "Sharpe") -> pd.Series | None:
    df = filter_grid(results, optimizer="MaxSharpe", include_frontier=False)
    df = _sort_candidates(df, metric)
    if df.empty:
        return None
    return df.iloc[0]


def select_best_grid_strategy(
    grid,
    *,
    optimizer: str,
    metric: str = "Sharpe",
) -> str:
    """Select the strongest strategy for one optimizer in a walk-forward grid.

    Parameters
    ----------
    grid
        Walk-forward grid result with aligned ``results`` and ``diagnostics``
        tables.
    optimizer : str
        Optimizer name recorded in the grid diagnostics.
    metric : str, default "Sharpe"
        Result column to maximize.

    Returns
    -------
    str
        Strategy name for the highest metric value.

    Raises
    ------
    InputError
        If the optimizer has no strategy with an observed metric value.
    """

    names = grid.diagnostics.index[
        grid.diagnostics["Optimizer"].eq(optimizer)
    ]
    scores = grid.results.reindex(names)[metric].dropna()
    if scores.empty:
        raise InputError(
            f"No {optimizer!r} grid strategy has an observed {metric!r} value."
        )
    return str(scores.idxmax())


def append_or_select_frontiergrid(finalists: Sequence[str], results: pd.DataFrame) -> list[str]:
    out = list(dict.fromkeys(str(x) for x in finalists))
    frontier = results[results["Optimizer"].isin({"FrontierGrid", "MaxSharpe (FrontierGrid)"})]
    for name in frontier.index:
        if name not in out:
            out.append(str(name))
            break
    return out


def with_baseline(
    strategies: Sequence[str],
    *,
    available: Sequence[str] | pd.Index | None = None,
    baseline: str = "EW",
) -> list[str]:
    """Return strategies with a benchmark baseline first when it is available."""
    out = list(dict.fromkeys(str(x) for x in strategies))
    base = str(baseline)
    if available is not None and base not in {str(x) for x in available}:
        return out
    if base in out:
        out.remove(base)
    return [base, *out]


def select_finalists(
    results: pd.DataFrame,
    *,
    minvar_n: int = 2,
    mv_n: int = 2,
    ridge_n: int = 1,
    maxsharpe_n: int = 1,
    include_frontier: bool = True,
    include_baseline: bool = True,
    baseline: str = "EW",
    metric: str = "Sharpe",
) -> list[str]:
    """Notebook 2 finalist rule: EW baseline plus top families and FrontierGrid."""
    picks: list[str] = []
    family_specs = [
        ("MinVar", minvar_n),
        ("MV", mv_n),
        ("RidgeMV", ridge_n),
        ("MaxSharpe", maxsharpe_n),
    ]
    for family, n in family_specs:
        df = _sort_candidates(filter_grid(results, optimizer=family, include_frontier=False), metric)
        picks.extend(str(x) for x in df.head(int(n)).index)
    picks = list(dict.fromkeys(picks))
    if include_frontier:
        picks = append_or_select_frontiergrid(picks, results)
    if include_baseline:
        picks = with_baseline(picks, available=results.index, baseline=baseline)
    return picks


def fixed_mu_covariance_comparison(
    results: pd.DataFrame,
    *,
    mu_model: str = "BayesSteinMomentum",
    optimizer: str = "MV",
    cov_models: Sequence[str] = COV_ORDER,
) -> list[str]:
    """All covariance models for one fixed mu model and optimizer. No best-picking."""
    out = []
    df = filter_grid(results, optimizer=optimizer, mu_model=mu_model)
    for cov_model in cov_models:
        names = df[df["Covariance model"].eq(cov_model)].index.tolist()
        out.extend(str(x) for x in names)
    return out


def fixed_cov_mu_comparison(
    results: pd.DataFrame,
    *,
    cov_model: str = "EWMA",
    optimizers: Sequence[str] = ("MV", "MaxSharpe"),
    mu_models: Sequence[str] = MU_ORDER,
) -> list[str]:
    """All mu models for a fixed covariance and optimizer list. No best-picking."""
    out = []
    df = filter_grid(results, optimizer=list(optimizers), cov_model=cov_model)
    for optimizer in optimizers:
        for mu_model in mu_models:
            names = df[df["Optimizer"].eq(optimizer) & df["Mu model"].eq(mu_model)].index.tolist()
            out.extend(str(x) for x in names)
    return out


def finalist_summary(
    results: pd.DataFrame,
    finalists: Sequence[str],
    *,
    baseline: str = "EW",
    include_baseline: bool = True,
) -> pd.DataFrame:
    names = with_baseline(finalists, available=results.index, baseline=baseline) if include_baseline else list(finalists)
    present = [name for name in names if name in results.index]
    out = results.loc[present].copy()
    out.insert(0, "Label", [strategy_display_label(name, out.loc[name].to_dict()) for name in out.index])
    return out


def comparison_summary(results: pd.DataFrame, strategies: Sequence[str]) -> pd.DataFrame:
    present = [name for name in strategies if name in results.index]
    return results.loc[present].copy()


__all__ = [
    "COV_ORDER",
    "MU_ORDER",
    "append_or_select_frontiergrid",
    "best_strategy_by_sharpe",
    "build_metrics_table",
    "build_strategy_summary",
    "build_trade_table",
    "calc_drawdown",
    "comparison_summary",
    "filter_grid",
    "finalist_summary",
    "fixed_cov_mu_comparison",
    "fixed_mu_covariance_comparison",
    "parse_strategy_spec",
    "performance_metrics",
    "result_sharpe",
    "select_best_grid_strategy",
    "select_best_maxsharpe_combination",
    "select_finalists",
    "strategy_display_label",
    "strategy_family",
    "summarize_results",
    "with_baseline",
]
