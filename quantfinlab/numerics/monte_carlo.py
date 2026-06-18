from __future__ import annotations

import numpy as np

_GBM_NUMBA = None


def _resolve_engine(engine: str) -> str:
    key = str(engine).lower()
    if key in {"numpy", "python"}:
        return "numpy"
    if key == "numba":
        return "numba"
    if key in {"cpp", "c++"}:
        return "cpp"
    raise ValueError("engine must be one of {'numpy', 'numba', 'cpp'}.")


def _get_gbm_numba():
    global _GBM_NUMBA
    if _GBM_NUMBA is not None:
        return _GBM_NUMBA
    try:
        from numba import njit
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Numba GBM engine requested but Numba is not available.") from exc

    @njit(cache=True)
    def paths_nb(s0, r, q, sigma, tau, steps, z):
        n_paths = z.shape[0]
        out = np.empty((n_paths, steps + 1), dtype=np.float64)
        dt = tau / steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)
        for i in range(n_paths):
            out[i, 0] = s0
            for j in range(1, steps + 1):
                out[i, j] = out[i, j - 1] * np.exp(drift + vol * z[i, j - 1])
        return out

    _GBM_NUMBA = paths_nb
    return _GBM_NUMBA


def antithetic_normals(paths: int, steps: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    half = max(int(paths) // 2, 1)
    z = rng.standard_normal((half, int(steps)))
    return np.vstack([z, -z])


def gbm_paths(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    *,
    steps: int = 50,
    paths: int = 10000,
    seed: int = 7,
    engine: str = "numba",
) -> np.ndarray:
    resolved = _resolve_engine(engine)
    if resolved == "cpp":
        from quantfinlab import _kernels

        return np.asarray(_kernels.gbm_paths_antithetic(s0, r, q, sigma, tau, int(steps), int(paths), int(seed)), dtype=float)
    z = antithetic_normals(paths, steps, seed)
    if resolved == "numba":
        try:
            return _get_gbm_numba()(float(s0), float(r), float(q), float(sigma), float(tau), int(steps), np.asarray(z, dtype=np.float64))
        except Exception:
            if str(engine).lower() == "numba":
                raise
    n_paths = z.shape[0]
    out = np.empty((n_paths, int(steps) + 1), dtype=float)
    out[:, 0] = float(s0)
    dt = float(tau) / int(steps)
    drift = (float(r) - float(q) - 0.5 * float(sigma) ** 2) * dt
    vol = float(sigma) * np.sqrt(dt)
    for i in range(1, int(steps) + 1):
        out[:, i] = out[:, i - 1] * np.exp(drift + vol * z[:, i - 1])
    return out


def payoff_paths(paths: np.ndarray, strike: float, option_type: str = "put") -> np.ndarray:
    p = np.asarray(paths, dtype=float)
    if str(option_type).lower().startswith("c"):
        return np.maximum(p - float(strike), 0.0)
    return np.maximum(float(strike) - p, 0.0)


__all__ = ["antithetic_normals", "gbm_paths", "payoff_paths"]
