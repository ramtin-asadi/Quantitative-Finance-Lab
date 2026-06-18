from __future__ import annotations

from typing import Any

import numpy as np

def american_tree_batch(
    s: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    option_type: np.ndarray,
    steps: int = 200,
    tree_type: int = 0,
    american: bool = True,
) -> np.ndarray: ...


def american_tree_boundary(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: int,
    steps: int = 200,
    tree_type: int = 0,
    american: bool = True,
) -> dict[str, Any]: ...


def american_pde_psor(
    s: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    option_type: int,
    s_steps: int = 160,
    t_steps: int = 120,
    s_max_mult: float = 3.0,
    omega: float = 1.35,
    tol: float = 1e-7,
    max_iter: int = 5000,
    american: bool = True,
) -> dict[str, Any]: ...


def american_pde_psor_batch(
    s: np.ndarray,
    k: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    option_type: np.ndarray,
    s_steps: int = 100,
    t_steps: int = 80,
    s_max_mult: float = 3.0,
    omega: float = 1.35,
    tol: float = 1e-7,
    max_iter: int = 3000,
    american: bool = True,
) -> dict[str, Any]: ...


def gbm_paths_antithetic(
    s0: float,
    r: float,
    q: float,
    sigma: float,
    tau: float,
    steps: int,
    paths: int,
    seed: int = 7,
) -> np.ndarray: ...


def lsm_backward(
    paths: np.ndarray,
    strike: float,
    r: float,
    tau: float,
    option_type: int,
    degree: int = 3,
) -> dict[str, Any]: ...


def lsm_eval_policy(
    paths: np.ndarray,
    strike: float,
    r: float,
    tau: float,
    option_type: int,
    coefficients: np.ndarray,
) -> dict[str, Any]: ...


def fft_prices(
    model_id: int,
    params: np.ndarray,
    s: float,
    r: float,
    q: float,
    tau: float,
    alpha: float = 1.5,
    n: int = 256,
    eta: float = 0.25,
    option_type: int = 1,
) -> dict[str, Any]: ...


def direct_prices(
    model_id: int,
    params: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    s: float,
    r: float,
    q: float,
    n_terms: int = 512,
    u_max: float = 120.0,
    option_type: int = 1,
) -> np.ndarray: ...


def cos_prices(
    model_id: int,
    params: np.ndarray,
    strikes: np.ndarray,
    tau: np.ndarray,
    s: float,
    r: float,
    q: float,
    n_terms: int = 256,
    truncation_width: float = 100.0,
    option_type: int = 1,
) -> np.ndarray: ...
