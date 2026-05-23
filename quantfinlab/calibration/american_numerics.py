from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantfinlab.options.rates_dividends import add_discount_factors, attach_rates
from quantfinlab.options.american import (
    assignment_risk,
    boundary_distance,
    european_tree_batch,
    model_disagreement,
    pricing_error,
    roll_signal,
    tree_batch,
    tree_boundary,
)


def _settings_match(meta_path: Path, settings: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("settings") == settings


def _write_meta(meta_path: Path, settings: dict, rows: int) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"settings": settings, "rows": int(rows), "created_utc": pd.Timestamp.now("UTC").isoformat()}, indent=2), encoding="utf-8")


def _flags(option_type) -> np.ndarray:
    text = np.asarray(option_type).astype(str)
    return np.where(np.char.startswith(np.char.lower(text), "c"), 1, -1).astype(np.int32)


def prepare_american_quotes(
    quotes: pd.DataFrame,
    underlying: pd.DataFrame | pd.Series,
    curve_panel: pd.DataFrame,
    *,
    min_dte: float = 7.0,
    max_dte: float = 180.0,
    moneyness_range: tuple[float, float] = (0.65, 1.45),
    max_rel_spread: float = 0.35,
    min_sigma: float = 0.03,
    max_sigma: float = 2.50,
    annualization_days: float = 365.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q = quotes.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    q["expiry"] = pd.to_datetime(q["expiry"], errors="coerce").dt.normalize()
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q.get("dte_calendar", q["tau"] * annualization_days), errors="coerce")
    if "moneyness" not in q.columns:
        q["moneyness"] = pd.to_numeric(q["strike"], errors="coerce") / pd.to_numeric(q["spot"], errors="coerce")
    q["log_moneyness"] = np.log(q["moneyness"].where(q["moneyness"] > 0.0))
    q["relative_spread"] = pd.to_numeric(q.get("relative_spread", q.get("rel_spread")), errors="coerce")
    q["rel_spread"] = q["relative_spread"]
    q["iv_mid"] = pd.to_numeric(q.get("iv_mid", q.get("iv")), errors="coerce")
    steps = []
    def add_step(name, frame):
        steps.append({"step": name, "rows": len(frame), "removed": np.nan if not steps else steps[-1]["rows"] - len(frame)})
    add_step("raw rows", q)
    q = q[q["date"].notna() & q["expiry"].notna() & pd.to_numeric(q["spot"], errors="coerce").gt(0) & pd.to_numeric(q["strike"], errors="coerce").gt(0)].copy()
    add_step("after schema normalization", q)
    q = q[pd.to_numeric(q["bid"], errors="coerce").gt(0) & pd.to_numeric(q["ask"], errors="coerce").gt(0) & pd.to_numeric(q["mid"], errors="coerce").gt(0) & (pd.to_numeric(q["ask"], errors="coerce") >= pd.to_numeric(q["bid"], errors="coerce"))].copy()
    add_step("after positive quote filter", q)
    q = q[q["relative_spread"].le(float(max_rel_spread))].copy()
    add_step("after spread filter", q)
    q = q[q["dte_days"].between(float(min_dte), float(max_dte))].copy()
    add_step("after DTE filter", q)
    q = q[q["moneyness"].between(float(moneyness_range[0]), float(moneyness_range[1]))].copy()
    add_step("after moneyness filter", q)
    q = q[q["iv_mid"].between(float(min_sigma), float(max_sigma))].copy()
    add_step("after IV/sigma filter", q)
    q["m_bucket"] = pd.cut(q["moneyness"], np.linspace(float(moneyness_range[0]), float(moneyness_range[1]), 17), include_lowest=True)
    smooth = q.groupby(["date", "expiry", "m_bucket"], observed=True)["iv_mid"].median().rename("sigma_smooth").reset_index()
    q = q.merge(smooth, on=["date", "expiry", "m_bucket"], how="left")
    q["sigma_used"] = (0.75 * q["iv_mid"] + 0.25 * q["sigma_smooth"].fillna(q["iv_mid"])).clip(float(min_sigma), float(max_sigma))
    q = attach_rates(q, curve_panel=curve_panel)
    q = add_discount_factors(q)
    u = underlying.copy()
    if isinstance(u, pd.Series):
        u = u.to_frame("close")
    date_col = "Date" if "Date" in u.columns else "date" if "date" in u.columns else None
    if date_col is not None:
        u[date_col] = pd.to_datetime(u[date_col], errors="coerce").dt.normalize()
        u = u.set_index(date_col)
    u.index = pd.to_datetime(u.index, errors="coerce").normalize()
    u = u.sort_index()
    div_col = "Dividends" if "Dividends" in u.columns else "dividend" if "dividend" in u.columns else None
    if div_col is None:
        div_events = pd.DataFrame(columns=["date", "dividend"])
    else:
        div_events = u[[div_col]].rename(columns={div_col: "dividend"})
        div_events = div_events[pd.to_numeric(div_events["dividend"], errors="coerce").fillna(0.0) > 0.0].reset_index(names="date")
    div_dates = pd.to_datetime(div_events["date"], errors="coerce").to_numpy("datetime64[ns]") if not div_events.empty else np.array([], dtype="datetime64[ns]")
    div_amt = pd.to_numeric(div_events["dividend"], errors="coerce").to_numpy(float) if not div_events.empty else np.array([], dtype=float)
    q_dates = q["date"].to_numpy("datetime64[ns]")
    q_exp = q["expiry"].to_numpy("datetime64[ns]")
    q_rate = pd.to_numeric(q["rate"], errors="coerce").fillna(0.0).to_numpy(float)
    next_div = np.zeros(len(q), dtype=float)
    days_to = np.full(len(q), np.nan, dtype=float)
    div_in_life = np.zeros(len(q), dtype=float)
    pv = np.zeros(len(q), dtype=float)
    for d, amount in zip(div_dates, div_amt):
        after_date = d > q_dates
        before_exp = d <= q_exp
        inside = after_date & before_exp
        div_in_life[inside] += amount
        days = (d - q_dates).astype("timedelta64[D]").astype(float)
        pv[inside] += amount * np.exp(-q_rate[inside] * days[inside] / annualization_days)
        next_mask = after_date & (np.isnan(days_to) | (days < days_to))
        next_div[next_mask] = amount
        days_to[next_mask] = days[next_mask]
    q["next_dividend"] = next_div
    q["days_to_next_dividend"] = days_to
    q["dividend_in_life"] = div_in_life
    q["pv_dividends"] = pv
    spot = pd.to_numeric(q["spot"], errors="coerce")
    tau = pd.to_numeric(q["tau"], errors="coerce")
    adj = ((spot - q["pv_dividends"]).clip(lower=1e-6) / spot).clip(lower=1e-8, upper=1.0)
    q["dividend_yield"] = (-np.log(adj) / tau.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    intrinsic = np.where(q["option_type"].astype(str).str.startswith("c"), np.maximum(spot - q["strike"], 0.0), np.maximum(q["strike"] - spot, 0.0))
    q["time_value"] = pd.to_numeric(q["mid"], errors="coerce") - intrinsic
    q = q.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", "expiry", "spot", "strike", "tau", "rate", "sigma_used", "mid"]).copy()
    add_step("final rows", q)
    audit = pd.DataFrame(steps)
    return q.reset_index(drop=True), audit, div_events


def select_teaching_contracts(quotes: pd.DataFrame) -> pd.DataFrame:
    q = quotes.copy()
    q["date"] = pd.to_datetime(q["date"], errors="coerce").dt.normalize()
    if "dte_days" not in q.columns:
        q["dte_days"] = pd.to_numeric(q["tau"], errors="coerce") * 365.25
    if "moneyness" not in q.columns:
        q["moneyness"] = pd.to_numeric(q["strike"], errors="coerce") / pd.to_numeric(q["spot"], errors="coerce")
    div = pd.to_numeric(q.get("dividend_in_life", 0.0), errors="coerce").fillna(0.0)
    time_value = pd.to_numeric(q.get("time_value", q.get("mid", 0.0)), errors="coerce").fillna(0.0)
    specs = [
        ("atm_put_30_60", q["option_type"].astype(str).str.startswith("p") & q["dte_days"].between(30, 60), 1.00, 45.0, 0.0),
        ("itm_put_30_90", q["option_type"].astype(str).str.startswith("p") & q["dte_days"].between(30, 90) & q["moneyness"].between(1.06, 1.20), 1.12, 60.0, 0.0),
        ("otm_put_60_120", q["option_type"].astype(str).str.startswith("p") & q["dte_days"].between(60, 120) & q["moneyness"].between(0.82, 0.95), 0.90, 90.0, 0.0),
        ("atm_call_30_60", q["option_type"].astype(str).str.startswith("c") & q["dte_days"].between(30, 60), 1.00, 45.0, 0.0),
        ("dividend_call", q["option_type"].astype(str).str.startswith("c") & q["dte_days"].between(7, 45) & (div > 0.0), 0.96, 25.0, 1.0),
    ]
    rows = []
    used = set()
    for label, mask, m0, d0, div_bonus in specs:
        sub = q.loc[mask].copy()
        if sub.empty:
            continue
        score = (sub["moneyness"] - m0).abs() + (sub["dte_days"] - d0).abs() / 365.25
        if div_bonus:
            score = score - 0.05 * div.loc[sub.index] + 0.05 * (time_value.loc[sub.index] / pd.to_numeric(sub["spot"], errors="coerce")).fillna(0.0)
        sub["_score"] = score
        row = sub.sort_values(["_score", "rel_spread" if "rel_spread" in sub.columns else "mid"]).iloc[0].copy()
        key = (row["date"], row["expiry"], row["option_type"], float(row["strike"]))
        if key in used:
            continue
        row["contract_role"] = label
        rows.append(row.drop(labels=["_score"], errors="ignore"))
        used.add(key)
    return pd.DataFrame(rows).reset_index(drop=True)


def full_chain_tree_scan(
    quotes: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    settings: dict | None = None,
    sigma_col: str = "sigma_used",
    q_col: str = "dividend_yield",
    steps: int = 300,
    tree_type: str = "crr",
    engine: str = "cpp",
    chunk_size: int = 25000,
) -> pd.DataFrame:
    settings = dict(settings or {})
    settings.update({"method": "tree_full_chain", "steps": int(steps), "tree_type": str(tree_type), "engine": str(engine), "sigma_col": sigma_col, "q_col": q_col})
    cache = Path(cache_path) if cache_path is not None else None
    meta = cache.with_suffix(".meta.json") if cache is not None else None
    if cache is not None and cache.exists() and meta is not None and _settings_match(meta, settings):
        return pd.read_parquet(cache)
    q = quotes.copy().reset_index(drop=True)
    sigma = pd.to_numeric(q[sigma_col], errors="coerce").to_numpy(float)
    div = pd.to_numeric(q[q_col], errors="coerce").fillna(0.0).to_numpy(float) if q_col in q.columns else np.zeros(len(q), dtype=float)
    rates = pd.to_numeric(q.get("rate", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    s = pd.to_numeric(q["spot"], errors="coerce").to_numpy(float)
    k = pd.to_numeric(q["strike"], errors="coerce").to_numpy(float)
    tau = pd.to_numeric(q["tau"], errors="coerce").to_numpy(float)
    opt = q["option_type"].to_numpy()
    american = np.empty(len(q), dtype=float)
    european = np.empty(len(q), dtype=float)
    t0 = time.perf_counter()
    for start in range(0, len(q), int(chunk_size)):
        stop = min(start + int(chunk_size), len(q))
        sl = slice(start, stop)
        american[sl] = tree_batch(s[sl], k[sl], rates[sl], div[sl], sigma[sl], tau[sl], opt[sl], steps=steps, tree_type=tree_type, american=True, engine=engine)
        european[sl] = european_tree_batch(s[sl], k[sl], rates[sl], div[sl], sigma[sl], tau[sl], opt[sl], steps=steps, tree_type=tree_type, engine=engine)
    elapsed = time.perf_counter() - t0
    keep = [c for c in ["date", "expiry", "option_type", "spot", "strike", "tau", "dte_days", "moneyness", "log_moneyness", "mid", "bid", "ask", sigma_col, "rate", q_col, "relative_spread", "rel_spread", "next_dividend", "days_to_next_dividend", "dividend_in_life", "pv_dividends"] if c in q.columns]
    out = q[keep].copy()
    out = out.rename(columns={sigma_col: "sigma_used", q_col: "dividend_yield"})
    out["european_tree_price"] = european
    out["american_tree_price"] = american
    out["american_premium"] = american - european
    out["pricing_error"] = pricing_error(american, q["mid"])
    out["abs_pricing_error"] = np.abs(out["pricing_error"])
    out["tree_runtime_sec_total"] = elapsed
    out["tree_contracts_per_sec"] = len(out) / max(elapsed, 1e-12)
    out["tree_steps"] = int(steps)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
        if meta is not None:
            _write_meta(meta, settings, len(out))
    return out


def pde_regime_grid(
    quotes: pd.DataFrame,
    *,
    cache_path: str | Path | None = None,
    settings: dict | None = None,
    dte_bins=(7, 14, 30, 60, 90, 120, 180),
    moneyness_bins=(0.65, 0.80, 0.90, 0.97, 1.03, 1.10, 1.25, 1.45),
    sigma_bins=(0.03, 0.15, 0.22, 0.32, 0.50, 2.50),
) -> pd.DataFrame:
    settings = dict(settings or {})
    settings.update({"method": "pde_regime_grid", "dte_bins": list(dte_bins), "moneyness_bins": list(moneyness_bins), "sigma_bins": list(sigma_bins)})
    cache = Path(cache_path) if cache_path is not None else None
    meta = cache.with_suffix(".meta.json") if cache is not None else None
    if cache is not None and cache.exists() and meta is not None and _settings_match(meta, settings):
        return pd.read_parquet(cache)
    q = quotes.copy().reset_index(drop=True)
    q["dte_bucket"] = pd.cut(q["dte_days"], dte_bins, include_lowest=True)
    q["moneyness_bucket"] = pd.cut(q["moneyness"], moneyness_bins, include_lowest=True)
    q["sigma_bucket"] = pd.cut(q["sigma_used"], sigma_bins, include_lowest=True)
    q["dividend_bucket"] = np.where(pd.to_numeric(q.get("dividend_in_life", 0.0), errors="coerce").fillna(0.0) > 0.0, "dividend", "none")
    q["ex_div_bucket"] = pd.cut(pd.to_numeric(q.get("days_to_next_dividend", np.inf), errors="coerce").fillna(np.inf), [-1, 7, 21, 10_000], labels=["0_7", "8_21", "none"], include_lowest=True)
    keys = ["option_type", "dte_bucket", "moneyness_bucket", "sigma_bucket", "dividend_bucket", "ex_div_bucket"]
    coverage = q.groupby(keys, observed=True).agg(cell_rows=("mid", "size"), median_dte=("dte_days", "median"), median_moneyness=("moneyness", "median"), median_sigma=("sigma_used", "median"), median_spread=("rel_spread", "median")).reset_index()
    q = q.merge(coverage[keys + ["cell_rows", "median_dte", "median_moneyness", "median_sigma"]], on=keys, how="left")
    q["_distance"] = (q["dte_days"] - q["median_dte"]).abs() / 365.25 + (q["moneyness"] - q["median_moneyness"]).abs() + (q["sigma_used"] - q["median_sigma"]).abs()
    medoids = q.sort_values(keys + ["_distance", "rel_spread", "date"]).drop_duplicates(subset=keys, keep="first").drop(columns=["_distance"])
    for col in ["dte_bucket", "moneyness_bucket", "sigma_bucket", "ex_div_bucket"]:
        medoids[col] = medoids[col].astype(str)
    for col in medoids.columns:
        if isinstance(medoids[col].dtype, pd.CategoricalDtype) or isinstance(medoids[col].dtype, pd.IntervalDtype):
            medoids[col] = medoids[col].astype(str)
    medoids["coverage_rows"] = medoids["cell_rows"]
    medoids["coverage_pct"] = medoids["coverage_rows"] / len(q)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        medoids.to_parquet(cache, index=False)
        if meta is not None:
            _write_meta(meta, settings, len(medoids))
    return medoids.reset_index(drop=True)


def method_comparison(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    data = pd.DataFrame(rows).copy()
    if data.empty:
        return data
    for col in ["tree_price", "pde_price", "lsm_price", "runtime_sec", "american_premium", "model_disagreement", "pricing_error"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def method_disagreement_table(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    price_cols = [c for c in ["tree_price", "pde_price", "lsm_price"] if c in out.columns]
    if price_cols:
        out["model_disagreement"] = model_disagreement(*[out[c].to_numpy(float) for c in price_cols])
    return out


def american_scan_summary(scan: pd.DataFrame) -> pd.DataFrame:
    q = scan.copy()
    return pd.DataFrame(
        [
            {
                "rows": len(q),
                "dates": q["date"].nunique() if "date" in q else np.nan,
                "expiries": q["expiry"].nunique() if "expiry" in q else np.nan,
                "contracts": q[["expiry", "option_type", "strike"]].drop_duplicates().shape[0] if {"expiry", "option_type", "strike"}.issubset(q.columns) else np.nan,
                "median_american_premium": float(pd.to_numeric(q.get("american_premium"), errors="coerce").median()),
                "median_abs_pricing_error": float(pd.to_numeric(q.get("abs_pricing_error"), errors="coerce").median()),
                "contracts_per_sec": float(pd.to_numeric(q.get("tree_contracts_per_sec"), errors="coerce").median()),
            }
        ]
    )


def overlay_candidates(quotes: pd.DataFrame) -> pd.DataFrame:
    out = assignment_risk(quotes)
    out = roll_signal(out)
    spread = pd.to_numeric(out.get("rel_spread", out.get("relative_spread", 0.0)), errors="coerce").fillna(0.0)
    premium = pd.to_numeric(out.get("american_premium", 0.0), errors="coerce").fillna(0.0)
    disagreement = pd.to_numeric(out.get("model_disagreement", 0.0), errors="coerce").fillna(0.0)
    out["candidate_score"] = premium - 0.5 * spread - 0.25 * disagreement
    return out.reset_index(drop=True)


def method_summary(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    return method_comparison(rows)


def speed_table(methods: dict[str, callable], sizes: list[int], repeats: int = 3) -> pd.DataFrame:
    rows = []
    for name, fn in methods.items():
        for n in sizes:
            times = []
            for _ in range(int(repeats)):
                t0 = time.perf_counter()
                fn(int(n))
                times.append(time.perf_counter() - t0)
            elapsed = float(np.median(times))
            rows.append({"method": name, "n": int(n), "runtime_sec": elapsed, "items_per_sec": n / max(elapsed, 1e-12)})
    return pd.DataFrame(rows)


__all__ = [
    "american_scan_summary",
    "full_chain_tree_scan",
    "method_comparison",
    "method_disagreement_table",
    "method_summary",
    "overlay_candidates",
    "pde_regime_grid",
    "prepare_american_quotes",
    "select_teaching_contracts",
    "speed_table",
]
