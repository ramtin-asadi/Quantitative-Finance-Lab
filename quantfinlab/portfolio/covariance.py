from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from quantfinlab.common.errors import InputError

try:
    from sklearn.covariance import OAS, LedoitWolf
except Exception:  # pragma: no cover
    LedoitWolf = None
    OAS = None

DEFAULT_ANNUALIZATION = 252.0
CovarianceMethod = Literal["Sample", "LedoitWolf", "OAS", "EWMA"]


def _sanitize_returns(window: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(window, pd.DataFrame):
        raise InputError("window must be a pandas DataFrame.")
    if window.empty:
        raise InputError("window is empty.")
    out = window.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=0, how="any")
    if out.shape[0] < 3 or out.shape[1] < 2:
        raise InputError("window must have at least 3 rows and 2 assets after cleaning.")
    return out


def make_psd(sigma: np.ndarray | pd.DataFrame, *, eps: float = 1e-10) -> np.ndarray:
    """Project a square covariance matrix to PSD space via eigenvalue flooring."""
    S = np.asarray(sigma, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise InputError("sigma must be a square matrix.")
    S = 0.5 * (S + S.T)
    vals, vecs = np.linalg.eigh(S)
    vals = np.maximum(vals, float(eps))
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def _finalize_covariance(
    cov_daily: np.ndarray,
    columns: pd.Index,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    ridge: float = 1e-10,
    psd: bool = True,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    cov_ann = float(annualization) * np.asarray(cov_daily, dtype=float)
    cov_ann = 0.5 * (cov_ann + cov_ann.T)
    if ridge > 0:
        cov_ann = cov_ann + float(ridge) * np.eye(cov_ann.shape[0])
    if psd:
        cov_ann = make_psd(cov_ann, eps=psd_eps)
    if return_df:
        return pd.DataFrame(cov_ann, index=columns, columns=columns)
    return cov_ann


def sample_covariance(
    window: pd.DataFrame,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    ridge: float = 1e-10,
    psd: bool = True,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    R = _sanitize_returns(window)
    cov_daily = np.cov(R.to_numpy(dtype=float), rowvar=False, ddof=1).astype(float)
    return _finalize_covariance(
        cov_daily,
        R.columns,
        annualization=annualization,
        ridge=ridge,
        psd=psd,
        psd_eps=psd_eps,
        return_df=return_df,
    )


def ledoit_wolf_covariance(
    window: pd.DataFrame,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    ridge: float = 1e-10,
    psd: bool = True,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    if LedoitWolf is None:
        raise ImportError("scikit-learn is required for Ledoit-Wolf covariance.")
    R = _sanitize_returns(window)
    cov_daily = LedoitWolf().fit(R.to_numpy(dtype=float)).covariance_.astype(float)
    return _finalize_covariance(
        cov_daily,
        R.columns,
        annualization=annualization,
        ridge=ridge,
        psd=psd,
        psd_eps=psd_eps,
        return_df=return_df,
    )


def oas_covariance(
    window: pd.DataFrame,
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    ridge: float = 1e-10,
    psd: bool = True,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    if OAS is None:
        raise ImportError("scikit-learn is required for OAS covariance.")
    R = _sanitize_returns(window)
    cov_daily = OAS().fit(R.to_numpy(dtype=float)).covariance_.astype(float)
    return _finalize_covariance(
        cov_daily,
        R.columns,
        annualization=annualization,
        ridge=ridge,
        psd=psd,
        psd_eps=psd_eps,
        return_df=return_df,
    )


def ewma_covariance(
    window: pd.DataFrame | np.ndarray,
    *,
    lam: float = 0.94,
    annualization: float | None = None,
    ridge: float = 0.0,
    psd: bool = False,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """
    Estimate EWMA covariance.

    If annualization is None, returns the daily covariance. Passing a DataFrame
    preserves labels when return_df=True.
    """
    if not (0 < lam < 1):
        raise InputError("EWMA lambda must be in (0, 1).")
    if isinstance(window, pd.DataFrame):
        R = _sanitize_returns(window)
        x = R.to_numpy(dtype=float)
        columns = R.columns
    else:
        x = np.asarray(window, dtype=float)
        if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
            raise InputError("window must have shape (T, n) with T>=2.")
        x = x[np.all(np.isfinite(x), axis=1)]
        columns = pd.Index([f"a{i}" for i in range(x.shape[1])])
    x = x - x.mean(axis=0, keepdims=True)
    T, n = x.shape
    S = np.zeros((n, n), dtype=float)
    alpha = 1.0 - float(lam)
    for t in range(T):
        xt = x[t][:, None]
        S = float(lam) * S + alpha * (xt @ xt.T)
    scale = 1.0 - (float(lam) ** max(T, 1))
    if scale > 1e-12:
        S = S / scale
    ann = 1.0 if annualization is None else float(annualization)
    return _finalize_covariance(
        S,
        columns,
        annualization=ann,
        ridge=ridge,
        psd=psd,
        psd_eps=psd_eps,
        return_df=return_df,
    )


def normalize_covariance_method(method: str) -> CovarianceMethod:
    key = str(method).strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "sample": "Sample",
        "samplecov": "Sample",
        "samplecovariance": "Sample",
        "lw": "LedoitWolf",
        "ledoitwolf": "LedoitWolf",
        "ledoitwolfcovariance": "LedoitWolf",
        "oas": "OAS",
        "ewma": "EWMA",
    }
    if key not in aliases:
        raise InputError(f"Unknown covariance method: {method!r}.")
    return aliases[key]  # type: ignore[return-value]


def estimate_covariance(
    window: pd.DataFrame,
    method: str = "LedoitWolf",
    *,
    annualization: float = DEFAULT_ANNUALIZATION,
    ewma_lambda: float = 0.94,
    lam: float | None = None,
    ridge: float = 1e-10,
    psd: bool = True,
    psd_eps: float = 1e-10,
    return_df: bool = False,
) -> np.ndarray | pd.DataFrame:
    """Estimate annualized covariance using a supported Notebook 2 model label."""
    method_norm = normalize_covariance_method(method)
    if method_norm == "Sample":
        return sample_covariance(
            window,
            annualization=annualization,
            ridge=ridge,
            psd=psd,
            psd_eps=psd_eps,
            return_df=return_df,
        )
    if method_norm == "LedoitWolf":
        return ledoit_wolf_covariance(
            window,
            annualization=annualization,
            ridge=ridge,
            psd=psd,
            psd_eps=psd_eps,
            return_df=return_df,
        )
    if method_norm == "OAS":
        return oas_covariance(
            window,
            annualization=annualization,
            ridge=ridge,
            psd=psd,
            psd_eps=psd_eps,
            return_df=return_df,
        )
    return ewma_covariance(
        window,
        lam=float(lam if lam is not None else ewma_lambda),
        annualization=annualization,
        ridge=ridge,
        psd=psd,
        psd_eps=psd_eps,
        return_df=return_df,
    )


def estimate_covariance_map(
    window: pd.DataFrame,
    methods: tuple[str, ...] | list[str] = ("Sample", "LedoitWolf", "OAS", "EWMA"),
    **kwargs,
) -> dict[str, np.ndarray | pd.DataFrame]:
    return {normalize_covariance_method(m): estimate_covariance(window, m, **kwargs) for m in methods}


cov_estimate = estimate_covariance


__all__ = [
    "DEFAULT_ANNUALIZATION",
    "cov_estimate",
    "estimate_covariance",
    "estimate_covariance_map",
    "ewma_covariance",
    "ledoit_wolf_covariance",
    "make_psd",
    "normalize_covariance_method",
    "oas_covariance",
    "sample_covariance",
]
