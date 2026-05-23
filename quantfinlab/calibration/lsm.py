from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quantfinlab.numerics.monte_carlo import gbm_paths, payoff_paths


_LSM_NUMBA = None
_LSM_EVAL_NUMBA = None


def _resolve_engine(engine: str) -> str:
    key = str(engine).lower()
    if key in {"numpy", "python"}:
        return "numpy"
    if key == "numba":
        return "numba"
    if key in {"cpp", "c++"}:
        return "cpp"
    raise ValueError("engine must be one of {'numpy', 'numba', 'cpp'}.")


def _get_lsm_numba():
    global _LSM_NUMBA, _LSM_EVAL_NUMBA
    if _LSM_NUMBA is not None and _LSM_EVAL_NUMBA is not None:
        return _LSM_NUMBA, _LSM_EVAL_NUMBA
    try:
        from numba import njit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba LSM engine requested but Numba is not available.") from exc

    @njit(cache=True)
    def payoff_nb(s, k, flag):
        if flag > 0:
            return max(s - k, 0.0)
        return max(k - s, 0.0)

    @njit(cache=True)
    def solve_nb(a, b, n):
        aa = a.copy()
        bb = b.copy()
        for p in range(n):
            pivot = p
            best = abs(aa[p, p])
            for i in range(p + 1, n):
                val = abs(aa[i, p])
                if val > best:
                    best = val
                    pivot = i
            if best < 1e-12:
                continue
            if pivot != p:
                for j in range(p, n):
                    tmp = aa[p, j]
                    aa[p, j] = aa[pivot, j]
                    aa[pivot, j] = tmp
                tmpb = bb[p]
                bb[p] = bb[pivot]
                bb[pivot] = tmpb
            diag = aa[p, p]
            for j in range(p, n):
                aa[p, j] /= diag
            bb[p] /= diag
            for i in range(n):
                if i == p:
                    continue
                f = aa[i, p]
                for j in range(p, n):
                    aa[i, j] -= f * aa[p, j]
                bb[i] -= f * bb[p]
        return bb

    @njit(cache=True)
    def eval_poly_nb(beta, x):
        out = 0.0
        p = 1.0
        for j in range(beta.size):
            out += beta[j] * p
            p *= x
        return out

    @njit(cache=True)
    def train_nb(paths, strike, r, tau, flag, degree):
        n_paths = paths.shape[0]
        steps = paths.shape[1] - 1
        pcols = degree + 1
        dt = tau / steps
        disc = np.exp(-r * dt)
        cash = np.empty(n_paths, dtype=np.float64)
        exercise = np.empty(n_paths, dtype=np.int64)
        coeffs = np.zeros((steps + 1, pcols), dtype=np.float64)
        for i in range(n_paths):
            cash[i] = payoff_nb(paths[i, steps], strike, flag)
            exercise[i] = steps
        for t in range(steps - 1, 0, -1):
            for i in range(n_paths):
                cash[i] *= disc
            ata = np.zeros((pcols, pcols), dtype=np.float64)
            aty = np.zeros(pcols, dtype=np.float64)
            count = 0
            for i in range(n_paths):
                ex = payoff_nb(paths[i, t], strike, flag)
                if ex <= 0.0:
                    continue
                count += 1
                x = np.log(max(paths[i, t], 1e-300) / strike)
                basis = np.empty(pcols, dtype=np.float64)
                basis[0] = 1.0
                for j in range(1, pcols):
                    basis[j] = basis[j - 1] * x
                for a in range(pcols):
                    aty[a] += basis[a] * cash[i]
                    for b in range(pcols):
                        ata[a, b] += basis[a] * basis[b]
            beta = np.zeros(pcols, dtype=np.float64)
            if count >= pcols:
                for j in range(pcols):
                    ata[j, j] += 1e-10
                beta = solve_nb(ata, aty, pcols)
            for j in range(pcols):
                coeffs[t, j] = beta[j]
            for i in range(n_paths):
                ex = payoff_nb(paths[i, t], strike, flag)
                if ex <= 0.0:
                    continue
                x = np.log(max(paths[i, t], 1e-300) / strike)
                cont = eval_poly_nb(beta, x)
                if ex > cont:
                    cash[i] = ex
                    exercise[i] = t
        price = 0.0
        for i in range(n_paths):
            price += cash[i] * disc
        return price / n_paths, exercise, coeffs

    @njit(cache=True)
    def eval_nb(paths, strike, r, tau, flag, coeffs):
        n_paths = paths.shape[0]
        steps = paths.shape[1] - 1
        dt = tau / steps
        pcols = coeffs.shape[1]
        exercise = np.empty(n_paths, dtype=np.int64)
        total = 0.0
        for i in range(n_paths):
            cash = payoff_nb(paths[i, steps], strike, flag) * np.exp(-r * tau)
            et = steps
            for t in range(1, steps):
                ex = payoff_nb(paths[i, t], strike, flag)
                if ex <= 0.0:
                    continue
                x = np.log(max(paths[i, t], 1e-300) / strike)
                beta = coeffs[t]
                cont = eval_poly_nb(beta, x)
                if ex > cont:
                    cash = ex * np.exp(-r * dt * t)
                    et = t
                    break
            total += cash
            exercise[i] = et
        return total / n_paths, exercise

    _LSM_NUMBA = train_nb
    _LSM_EVAL_NUMBA = eval_nb
    return _LSM_NUMBA, _LSM_EVAL_NUMBA


def basis_matrix(x, degree: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cols = [np.ones_like(x)]
    for p in range(1, int(degree) + 1):
        cols.append(cols[-1] * x)
    return np.column_stack(cols)


def lsm_train(paths: np.ndarray, strike: float, r: float, tau: float, option_type: str = "put", degree: int = 3, engine: str = "numba") -> dict:
    resolved = _resolve_engine(engine)
    flag = 1 if str(option_type).lower().startswith("c") else -1
    if resolved == "cpp":
        from quantfinlab import _kernels

        return _kernels.lsm_backward(np.asarray(paths, dtype=float), float(strike), float(r), float(tau), int(flag), int(degree))
    if resolved == "numba":
        try:
            train_nb, _ = _get_lsm_numba()
            price, exercise, coeffs = train_nb(np.asarray(paths, dtype=np.float64), float(strike), float(r), float(tau), int(flag), int(degree))
            return {"price": float(price), "exercise_time": np.asarray(exercise), "coefficients": np.asarray(coeffs)}
        except Exception:
            if str(engine).lower() == "numba":
                raise
    p = np.asarray(paths, dtype=float)
    n_paths, cols = p.shape
    steps = cols - 1
    dt = float(tau) / steps
    disc = np.exp(-float(r) * dt)
    payoff = payoff_paths(p, strike, option_type)
    cashflow = payoff[:, -1].copy()
    exercise_time = np.full(n_paths, steps, dtype=int)
    coeffs = np.zeros((steps + 1, int(degree) + 1), dtype=float)
    for t in range(steps - 1, 0, -1):
        cashflow *= disc
        itm = payoff[:, t] > 0
        if itm.sum() >= degree + 1:
            x = np.log(np.maximum(p[itm, t], 1e-300) / float(strike))
            b = basis_matrix(x, degree)
            beta, *_ = np.linalg.lstsq(b, cashflow[itm], rcond=None)
            coeffs[t, : len(beta)] = beta
            cont = basis_matrix(np.log(np.maximum(p[:, t], 1e-300) / float(strike)), degree) @ beta
            exercise = itm & (payoff[:, t] > cont)
            cashflow[exercise] = payoff[exercise, t]
            exercise_time[exercise] = t
    price = float(np.mean(cashflow * disc))
    return {"price": price, "exercise_time": exercise_time, "coefficients": coeffs}


def lsm_value(paths: np.ndarray, strike: float, r: float, tau: float, option_type: str, coefficients: np.ndarray, engine: str = "numba") -> dict:
    resolved = _resolve_engine(engine)
    flag = 1 if str(option_type).lower().startswith("c") else -1
    if resolved == "cpp":
        from quantfinlab import _kernels

        return _kernels.lsm_eval_policy(np.asarray(paths, dtype=float), float(strike), float(r), float(tau), int(flag), np.asarray(coefficients, dtype=float))
    if resolved == "numba":
        try:
            _, eval_nb = _get_lsm_numba()
            price, exercise = eval_nb(np.asarray(paths, dtype=np.float64), float(strike), float(r), float(tau), int(flag), np.asarray(coefficients, dtype=np.float64))
            return {"price": float(price), "exercise_time": np.asarray(exercise)}
        except Exception:
            if str(engine).lower() == "numba":
                raise
    p = np.asarray(paths, dtype=float)
    coeffs = np.asarray(coefficients, dtype=float)
    steps = p.shape[1] - 1
    dt = float(tau) / steps
    pay = payoff_paths(p, strike, option_type)
    exercise_time = np.full(p.shape[0], steps, dtype=int)
    cash = pay[:, -1] * np.exp(-float(r) * tau)
    degree = coeffs.shape[1] - 1
    for t in range(1, steps):
        x = np.log(np.maximum(p[:, t], 1e-300) / float(strike))
        cont = basis_matrix(x, degree) @ coeffs[t]
        exercise = (pay[:, t] > 0) & (pay[:, t] > cont) & (exercise_time == steps)
        cash[exercise] = pay[exercise, t] * np.exp(-float(r) * dt * t)
        exercise_time[exercise] = t
    return {"price": float(np.mean(cash)), "exercise_time": exercise_time}


def lsm_crossfit(
    s0: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: str = "put",
    *,
    steps: int = 50,
    paths: int = 20000,
    degree: int = 3,
    seed: int = 7,
    engine: str = "numba",
) -> dict:
    resolved = _resolve_engine(engine)
    train_paths = gbm_paths(s0, r, q, sigma, tau, steps=steps, paths=paths, seed=seed, engine=resolved if resolved != "cpp" else "cpp")
    eval_paths = gbm_paths(s0, r, q, sigma, tau, steps=steps, paths=paths, seed=seed + 1009, engine=resolved if resolved != "cpp" else "cpp")
    train = lsm_train(train_paths, strike, r, tau, option_type, degree, engine=resolved)
    val = lsm_value(eval_paths, strike, r, tau, option_type, train["coefficients"], engine=resolved)
    return {
        "train_price": float(train["price"]),
        "evaluation_price": float(val["price"]),
        "coefficients": np.asarray(train["coefficients"], dtype=float),
        "train_exercise_time": np.asarray(train["exercise_time"]),
        "evaluation_exercise_time": np.asarray(val["exercise_time"]),
        "steps": int(steps),
        "paths": int(paths),
        "degree": int(degree),
        "engine_used": resolved,
    }


def policy_gap(lower_price: float, reference_price: float) -> float:
    return float(reference_price) - float(lower_price)


def exercise_boundary_from_policy(coefficients: np.ndarray, strike: float, option_type: str = "put", x_grid=None) -> pd.DataFrame:
    coeffs = np.asarray(coefficients, dtype=float)
    if x_grid is None:
        x_grid = np.linspace(-0.45, 0.45, 401)
    rows = []
    degree = coeffs.shape[1] - 1
    s_grid = float(strike) * np.exp(x_grid)
    if str(option_type).lower().startswith("c"):
        payoff = np.maximum(s_grid - strike, 0.0)
    else:
        payoff = np.maximum(strike - s_grid, 0.0)
    b = basis_matrix(x_grid, degree)
    for t in range(1, coeffs.shape[0] - 1):
        cont = b @ coeffs[t]
        ex = payoff > cont
        boundary = np.nan
        if ex.any():
            boundary = float(s_grid[ex][0] if str(option_type).lower().startswith("c") else s_grid[ex][-1])
        rows.append({"step": t, "boundary": boundary})
    return pd.DataFrame(rows)


def lsm_boundary(coefficients: np.ndarray, strike: float, option_type: str = "put", x_grid=None) -> pd.DataFrame:
    return exercise_boundary_from_policy(coefficients, strike, option_type, x_grid=x_grid)


def lsm_regime_grid(
    quotes: pd.DataFrame,
    *,
    dte_bins=(7, 21, 45, 75, 120, 180),
    moneyness_bins=(0.65, 0.85, 0.95, 1.03, 1.12, 1.45),
    sigma_bins=(0.03, 0.18, 0.28, 0.45, 2.50),
    ex_div_bins=(-1, 7, 21, 10000),
) -> pd.DataFrame:
    q = quotes.copy().reset_index(drop=True)
    q["dte_bucket"] = pd.cut(q["dte_days"], dte_bins, include_lowest=True)
    q["moneyness_bucket"] = pd.cut(q["moneyness"], moneyness_bins, include_lowest=True)
    q["sigma_bucket"] = pd.cut(q["sigma_used"], sigma_bins, include_lowest=True)
    q["dividend_bucket"] = np.where(pd.to_numeric(q.get("dividend_in_life", 0.0), errors="coerce").fillna(0.0) > 0.0, "dividend", "none")
    q["ex_div_bucket"] = pd.cut(pd.to_numeric(q.get("days_to_next_dividend", 10000.0), errors="coerce").fillna(10000.0), ex_div_bins, labels=["0_7", "8_21", "none"], include_lowest=True)
    keys = ["option_type", "dte_bucket", "moneyness_bucket", "sigma_bucket", "dividend_bucket", "ex_div_bucket"]
    coverage = q.groupby(keys, observed=True).agg(cell_rows=("mid", "size"), median_dte=("dte_days", "median"), median_moneyness=("moneyness", "median"), median_sigma=("sigma_used", "median")).reset_index()
    q = q.merge(coverage, on=keys, how="left")
    q["_distance"] = (q["dte_days"] - q["median_dte"]).abs() / 365.25 + (q["moneyness"] - q["median_moneyness"]).abs() + (q["sigma_used"] - q["median_sigma"]).abs()
    out = q.sort_values(keys + ["_distance", "rel_spread", "date"]).drop_duplicates(subset=keys, keep="first").drop(columns=["_distance"]).reset_index(drop=True)
    out["coverage_rows"] = out["cell_rows"]
    out["coverage_pct"] = out["coverage_rows"] / len(q)
    return out


def lsm_regime_map(
    regime_quotes: pd.DataFrame,
    *,
    paths: int = 64000,
    steps: int = 40,
    degree: int = 3,
    engine: str = "cpp",
    seed: int = 11,
    repeats: int = 1,
) -> pd.DataFrame:
    rows = []
    for i, row in regime_quotes.reset_index(drop=True).iterrows():
        resolved = _resolve_engine(engine)
        train_paths = gbm_paths(
            float(row["spot"]),
            float(row.get("rate", 0.0)),
            float(row.get("dividend_yield", 0.0)),
            float(row["sigma_used"]),
            float(row["tau"]),
            steps=steps,
            paths=paths,
            seed=seed + int(i) * 17,
            engine=resolved,
        )
        train = lsm_train(
            train_paths,
            float(row["strike"]),
            float(row.get("rate", 0.0)),
            float(row["tau"]),
            str(row["option_type"]),
            degree=degree,
            engine=resolved,
        )
        prices = []
        exercise_rates = []
        n_repeats = max(1, int(repeats))
        for rep in range(n_repeats):
            eval_paths = gbm_paths(
                float(row["spot"]),
                float(row.get("rate", 0.0)),
                float(row.get("dividend_yield", 0.0)),
                float(row["sigma_used"]),
                float(row["tau"]),
                steps=steps,
                paths=paths,
                seed=seed + int(i) * 17 + 1009 + rep * 7919,
                engine=resolved,
            )
            val = lsm_value(eval_paths, float(row["strike"]), float(row.get("rate", 0.0)), float(row["tau"]), str(row["option_type"]), train["coefficients"], engine=resolved)
            prices.append(float(val["price"]))
            exercise_rates.append(float(np.mean(np.asarray(val["exercise_time"]) < int(steps))))
        price_arr = np.asarray(prices, dtype=float)
        exercise_arr = np.asarray(exercise_rates, dtype=float)
        price_std = float(price_arr.std(ddof=1)) if len(price_arr) > 1 else 0.0
        price_se = float(price_std / np.sqrt(len(price_arr))) if len(price_arr) > 1 else 0.0
        out = {
            "date": row.get("date"),
            "expiry": row.get("expiry"),
            "option_type": row.get("option_type"),
            "strike": float(row["strike"]),
            "spot": float(row["spot"]),
            "tau": float(row["tau"]),
            "dte_days": float(row["dte_days"]),
            "moneyness": float(row["moneyness"]),
            "sigma_used": float(row["sigma_used"]),
            "rate": float(row.get("rate", 0.0)),
            "dividend_yield": float(row.get("dividend_yield", 0.0)),
            "coverage_rows": int(row.get("coverage_rows", row.get("cell_rows", 1))),
            "train_price": float(train["price"]),
            "lsm_price": float(price_arr.mean()),
            "lsm_price_std": price_std,
            "lsm_price_se": price_se,
            "lsm_ci_low": float(price_arr.mean() - 1.96 * price_se),
            "lsm_ci_high": float(price_arr.mean() + 1.96 * price_se),
            "exercise_probability": float(exercise_arr.mean()),
            "exercise_probability_std": float(exercise_arr.std(ddof=1)) if len(exercise_arr) > 1 else 0.0,
            "paths": int(paths),
            "steps": int(steps),
            "degree": int(degree),
            "repeats": n_repeats,
            "engine": engine,
        }
        for col in ["dte_bucket", "moneyness_bucket", "sigma_bucket", "dividend_bucket", "ex_div_bucket"]:
            if col in row.index:
                out[col] = str(row[col]) if col.endswith("_bucket") else row[col]
        rows.append(out)
    return pd.DataFrame(rows)


def lsm_stability(
    s0: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: str = "put",
    *,
    path_counts=(16000, 64000, 128000),
    degrees=(2, 3, 4),
    steps: int = 40,
    engine: str = "cpp",
) -> pd.DataFrame:
    rows = []
    for paths in path_counts:
        for degree in degrees:
            t0 = time.perf_counter()
            res = lsm_crossfit(s0, strike, r, q, sigma, tau, option_type, steps=steps, paths=int(paths), degree=int(degree), engine=engine)
            elapsed = time.perf_counter() - t0
            rows.append(
                {
                    "paths": int(paths),
                    "degree": int(degree),
                    "evaluation_price": float(res["evaluation_price"]),
                    "train_price": float(res["train_price"]),
                    "exercise_probability": float(np.mean(np.asarray(res["evaluation_exercise_time"]) < int(steps))),
                    "runtime_sec": elapsed,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "basis_matrix",
    "exercise_boundary_from_policy",
    "lsm_boundary",
    "lsm_crossfit",
    "lsm_regime_grid",
    "lsm_regime_map",
    "lsm_stability",
    "lsm_train",
    "lsm_value",
    "policy_gap",
]
