from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import t as student_t
from sklearn.covariance import LedoitWolf

from quantfinlab.common.errors import InputError

if TYPE_CHECKING:
    import networkx as nx
else:
    try:
        import networkx as nx
    except Exception:  # pragma: no cover
        nx = None


def _require_networkx():
    if nx is None:  # pragma: no cover
        raise ImportError("networkx is required for portfolio network functions. Install quantfinlab[network].")
    return nx


def _clean_returns(returns: pd.DataFrame, *, min_rows: int = 20, min_cols: int = 2) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame) or returns.empty:
        raise InputError("returns must be a non-empty DataFrame.")
    out = returns.copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = out.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if out.shape[0] < int(min_rows) or out.shape[1] < int(min_cols):
        raise InputError("returns has too little usable history after cleaning.")
    return out.astype(float)


def _matrix_frame(x, *, index: pd.Index | None = None, name: str = "matrix") -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        out = x.copy().astype(float)
        out.index = out.index.astype(str)
        out.columns = out.columns.astype(str)
        common = out.index.intersection(out.columns)
        out = out.loc[common, common]
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise InputError(f"{name} must be a square matrix.")
        labels = pd.Index(index if index is not None else [f"a{i}" for i in range(arr.shape[0])]).astype(str)
        out = pd.DataFrame(arr, index=labels, columns=labels)
    if out.empty:
        raise InputError(f"{name} is empty.")
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return 0.5 * (out + out.T)


def _cap_normalize(weights: pd.Series, *, max_weight: float = 1.0, min_weight: float = 0.0) -> pd.Series:
    w = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if w.empty:
        return w
    lo = max(float(min_weight), 0.0)
    cap = max(float(max_weight), 1.0 / len(w))
    w = w.clip(lower=lo)
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0, index=w.index, dtype=float)
    w = w / float(w.sum())
    for _ in range(50):
        over = w > cap + 1e-12
        if not bool(over.any()):
            break
        extra = float((w[over] - cap).sum())
        w[over] = cap
        room = (cap - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += extra * room / float(room.sum())
    w = w.clip(lower=lo, upper=cap)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(w), index=w.index)


def scale_01(values: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Min-max scale a Series or each DataFrame column to [0, 1]."""
    if isinstance(values, pd.DataFrame):
        return values.apply(scale_01)
    s = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(0.0, index=s.index, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if hi - lo <= 1e-12:
        return pd.Series(0.5, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def shrink_corr(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance converted to a correlation matrix."""
    r = _clean_returns(returns)
    cov = LedoitWolf().fit(r.to_numpy(dtype=float)).covariance_.astype(float)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-16))
    corr = cov / np.outer(sd, sd)
    corr = np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=r.columns.astype(str), columns=r.columns.astype(str))


def corr_distance(corr: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Correlation distance d_ij = sqrt(2 * (1 - rho_ij))."""
    c = _matrix_frame(corr, name="corr")
    arr = np.clip(c.to_numpy(dtype=float), -1.0, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - arr), 0.0, 4.0))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=c.index, columns=c.columns)


def pseudo_observations(returns: pd.DataFrame) -> pd.DataFrame:
    """Rank-transform returns into copula pseudo-observations in (0, 1)."""
    r = _clean_returns(returns)
    u = r.rank(axis=0, method="average") / (len(r) + 1.0)
    return u.clip(1e-6, 1.0 - 1e-6)


def kendall_to_t_copula_corr(pseudo: pd.DataFrame) -> pd.DataFrame:
    """Estimate t-copula correlation from Kendall tau via rho = sin(pi*tau/2)."""
    u = _clean_returns(pseudo, min_rows=20)
    tau = u.corr(method="kendall").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rho = np.sin(0.5 * math.pi * tau.to_numpy(dtype=float))
    rho = np.clip(0.5 * (rho + rho.T), -0.995, 0.995)
    np.fill_diagonal(rho, 1.0)
    return pd.DataFrame(rho, index=u.columns.astype(str), columns=u.columns.astype(str))


def _pair_sample(n: int, max_pairs: int | None) -> list[tuple[int, int]]:
    pairs = list(combinations(range(int(n)), 2))
    if max_pairs is None or len(pairs) <= int(max_pairs):
        return pairs
    loc = np.linspace(0, len(pairs) - 1, int(max_pairs), dtype=int)
    return [pairs[i] for i in np.unique(loc)]


def student_t_copula_loglik(
    pseudo: pd.DataFrame,
    rho: pd.DataFrame | np.ndarray,
    nu: float,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> float:
    """Average bivariate Student-t copula log likelihood across selected pairs."""
    if float(nu) <= 2.0:
        raise InputError("nu must be greater than 2.")
    u = _clean_returns(pseudo, min_rows=20)
    rho_df = _matrix_frame(rho, index=u.columns, name="rho").reindex(index=u.columns.astype(str), columns=u.columns.astype(str))
    x = student_t.ppf(u.clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float), df=float(nu))
    rho_arr = rho_df.to_numpy(dtype=float)
    pairs_use = list(pairs) if pairs is not None else _pair_sample(x.shape[1], None)
    if not pairs_use:
        return float("nan")

    ll_total = 0.0
    count = 0
    uni_const = gammaln((float(nu) + 1.0) / 2.0) - gammaln(float(nu) / 2.0) - 0.5 * math.log(float(nu) * math.pi)
    bi_const = gammaln((float(nu) + 2.0) / 2.0) - gammaln(float(nu) / 2.0) - math.log(float(nu) * math.pi)
    for i, j in pairs_use:
        r = float(np.clip(rho_arr[i, j], -0.995, 0.995))
        det = max(1.0 - r * r, 1e-12)
        xi, xj = x[:, i], x[:, j]
        q = (xi * xi - 2.0 * r * xi * xj + xj * xj) / det
        log_bi = bi_const - 0.5 * math.log(det) - 0.5 * (float(nu) + 2.0) * np.log1p(q / float(nu))
        log_u1 = uni_const - 0.5 * (float(nu) + 1.0) * np.log1p((xi * xi) / float(nu))
        log_u2 = uni_const - 0.5 * (float(nu) + 1.0) * np.log1p((xj * xj) / float(nu))
        vals = log_bi - log_u1 - log_u2
        ll_total += float(np.nanmean(vals))
        count += 1
    return ll_total / max(count, 1)


def select_t_copula_nu(
    pseudo: pd.DataFrame,
    *,
    nu_grid: Sequence[float] = (3, 4, 5, 7, 10, 15, 25, 50),
    max_pairs: int | None = 40,
) -> float:
    """Choose a global t-copula degrees-of-freedom value by pair log likelihood."""
    u = _clean_returns(pseudo, min_rows=20)
    rho = kendall_to_t_copula_corr(u)
    pairs = _pair_sample(u.shape[1], max_pairs)
    scores = {
        float(nu): student_t_copula_loglik(u, rho, float(nu), pairs=pairs)
        for nu in nu_grid
        if float(nu) > 2.0
    }
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        return 10.0
    return float(max(finite, key=finite.get))


def student_t_tail_dependence(rho: pd.DataFrame | np.ndarray, nu: float) -> pd.DataFrame:
    """Symmetric lower-tail dependence of a Student-t copula."""
    r = _matrix_frame(rho, name="rho")
    arr = np.clip(r.to_numpy(dtype=float), -0.995, 0.995)
    arg = -np.sqrt(np.maximum((float(nu) + 1.0) * (1.0 - arr) / np.maximum(1.0 + arr, 1e-10), 0.0))
    tail = 2.0 * student_t.cdf(arg, df=float(nu) + 1.0)
    tail = np.clip(np.nan_to_num(tail, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    np.fill_diagonal(tail, 1.0)
    return pd.DataFrame(tail, index=r.index, columns=r.columns)


def dependence_to_distance(dependence: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Convert a nonnegative dependence strength matrix to a distance matrix."""
    dep = _matrix_frame(dependence, name="dependence")
    arr = np.clip(dep.to_numpy(dtype=float), 0.0, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - arr), 0.0, 2.0))
    np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=dep.index, columns=dep.columns)


def dense_network(dependence: pd.DataFrame | np.ndarray, *, distance: pd.DataFrame | np.ndarray | None = None) -> nx.Graph:
    nx_mod = _require_networkx()
    dep = _matrix_frame(dependence, name="dependence")
    dist = _matrix_frame(distance, index=dep.index, name="distance") if distance is not None else dependence_to_distance(dep)
    graph = nx_mod.Graph()
    graph.add_nodes_from(dep.index)
    for i, a in enumerate(dep.index):
        for j in range(i + 1, len(dep.index)):
            b = dep.index[j]
            signed = float(dep.iat[i, j])
            strength = max(signed, 0.0)
            graph.add_edge(a, b, weight=strength, signed_weight=signed, distance=max(float(dist.iat[i, j]), 1e-8))
    return graph


def pmfg_network(dependence: pd.DataFrame | np.ndarray, *, distance: pd.DataFrame | np.ndarray | None = None) -> nx.Graph:
    nx_mod = _require_networkx()
    dep = _matrix_frame(dependence, name="dependence")
    dist = _matrix_frame(distance, index=dep.index, name="distance") if distance is not None else dependence_to_distance(dep)
    graph = nx_mod.Graph()
    graph.add_nodes_from(dep.index)
    target_edges = max(0, 3 * (len(dep.index) - 2))
    edges = []
    for i, a in enumerate(dep.index):
        for j in range(i + 1, len(dep.index)):
            b = dep.index[j]
            signed = float(dep.iat[i, j])
            edges.append((max(signed, 0.0), signed, a, b, max(float(dist.iat[i, j]), 1e-8)))
    edges.sort(key=lambda x: x[0], reverse=True)

    embedding = None
    faces: list[set[str]] = []

    def update_faces(planar_embedding):
        seen = set()
        out = []
        for u in planar_embedding.nodes():
            for v in planar_embedding.neighbors(u):
                if (u, v) in seen:
                    continue
                mark = set()
                out.append(set(planar_embedding.traverse_face(u, v, mark)))
                seen.update(mark)
        return out

    for strength, signed, a, b, d in edges:
        if graph.number_of_edges() >= target_edges:
            break
        if (
            embedding is not None
            and nx_mod.has_path(graph, a, b)
            and not any(a in face and b in face for face in faces)
        ):
            continue
        graph.add_edge(a, b, weight=strength, signed_weight=signed, distance=d)
        is_planar, new_embedding = nx_mod.check_planarity(graph, counterexample=False)
        if is_planar:
            embedding = new_embedding
            faces = update_faces(embedding)
        else:
            graph.remove_edge(a, b)

    edge_lookup = {
        frozenset((a, b)): (strength, signed, a, b, d)
        for strength, signed, a, b, d in edges
    }
    while graph.number_of_edges() < target_edges and faces:
        feasible = []
        for face in faces:
            for a, b in combinations(sorted(face), 2):
                if not graph.has_edge(a, b):
                    feasible.append(edge_lookup[frozenset((a, b))])
        if not feasible:
            break
        strength, signed, a, b, d = max(feasible, key=lambda row: row[0])
        graph.add_edge(a, b, weight=strength, signed_weight=signed, distance=d)
        is_planar, embedding = nx_mod.check_planarity(graph, counterexample=False)
        if not is_planar:
            graph.remove_edge(a, b)
            break
        faces = update_faces(embedding)
    return graph


def mst_network(dependence: pd.DataFrame | np.ndarray, *, distance: pd.DataFrame | np.ndarray | None = None) -> nx.Graph:
    nx_mod = _require_networkx()
    dep = _matrix_frame(dependence, name="dependence")
    dist = _matrix_frame(distance, index=dep.index, name="distance") if distance is not None else dependence_to_distance(dep)
    candidate = nx_mod.Graph()
    candidate.add_nodes_from(dep.index)
    for i, left in enumerate(dep.index):
        for j in range(i + 1, len(dep.index)):
            right = dep.index[j]
            signed = float(dep.iat[i, j])
            candidate.add_edge(
                left,
                right,
                weight=max(signed, 0.0),
                signed_weight=signed,
                distance=max(float(dist.iat[i, j]), 1e-8),
            )
    return nx_mod.minimum_spanning_tree(candidate, weight="distance")


def combined_centrality(scores: pd.DataFrame) -> pd.Series:
    """Rank-combine centrality columns into a scaled score."""
    if scores.empty:
        return pd.Series(dtype=float)
    vals = scores.apply(scale_01).rank(pct=True).mean(axis=1)
    return scale_01(vals)


def centrality_table(graph: nx.Graph) -> pd.DataFrame:
    """Scaled degree/strength, eigenvector, betweenness, closeness, and combined scores."""
    nx_mod = _require_networkx()
    nodes = list(graph.nodes())
    if not nodes:
        return pd.DataFrame()
    raw_degree = pd.Series(dict(graph.degree()), dtype=float).reindex(nodes).fillna(0.0)
    strength = pd.Series(dict(graph.degree(weight="weight")), dtype=float).reindex(nodes).fillna(0.0)
    try:
        eig = nx_mod.eigenvector_centrality_numpy(graph, weight="weight")
    except Exception:
        eig = nx_mod.eigenvector_centrality(graph, weight="weight", max_iter=1000, tol=1e-7)
    bet_k = min(16, len(nodes)) if len(nodes) > 32 else None
    bet = nx_mod.betweenness_centrality(
        graph,
        k=bet_k,
        seed=17 if bet_k is not None else None,
        weight="distance",
        normalized=True,
    )
    close = nx_mod.closeness_centrality(graph, distance="distance")
    out = pd.DataFrame(
        {
            "raw_degree": raw_degree,
            "strength": strength,
            "degree": scale_01(strength),
            "eigenvector": scale_01(pd.Series(eig, dtype=float)),
            "betweenness": scale_01(pd.Series(bet, dtype=float)),
            "closeness": scale_01(pd.Series(close, dtype=float)),
        }
    ).reindex(nodes)
    out["combined"] = combined_centrality(out[["degree", "eigenvector", "betweenness", "closeness"]])
    return out


def central_peripheral_weights(
    score: pd.Series,
    *,
    returns: pd.DataFrame | None = None,
    side: str = "central",
    n_stocks: int = 20,
    max_weight: float = 0.10,
) -> pd.Series:
    """Equal-weight the top names in the supplied centrality or periphery score."""
    side_norm = str(side).lower()
    if side_norm not in {"central", "peripheral", "periphery"}:
        raise InputError("side must be central or peripheral.")
    s = pd.Series(score, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if returns is not None:
        s = s.reindex(returns.columns).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    names = list(s.sort_values(ascending=False).head(int(n_stocks)).index)
    raw = pd.Series(1.0, index=names, dtype=float)
    return _cap_normalize(raw, max_weight=max_weight)


def network_score(
    peripheral: pd.Series,
    *,
    returns: pd.DataFrame,
    momentum_window: int = 126,
    momentum_skip: int = 0,
    volatility_window: int = 126,
    drawdown_window: int = 252,
    periphery_weight: float = 0.35,
    momentum_weight: float = 0.35,
    volatility_weight: float = -0.20,
    drawdown_weight: float = -0.10,
) -> pd.Series:
    """Score peripheral, positive-trend, lower-risk stocks for the diversifier."""
    r = _clean_returns(returns, min_rows=max(20, min(momentum_window, volatility_window, drawdown_window) // 2))
    periph = scale_01(pd.Series(peripheral, dtype=float).reindex(r.columns).fillna(0.0))
    momentum_end = -int(momentum_skip) if int(momentum_skip) > 0 else None
    mom_window = r.iloc[:momentum_end].tail(int(momentum_window) - int(momentum_skip))
    mom = (1.0 + mom_window).prod(axis=0) - 1.0
    vol = r.tail(int(volatility_window)).std(axis=0, ddof=1) * math.sqrt(252.0)
    nav = (1.0 + r.tail(int(drawdown_window))).cumprod(axis=0)
    dd = (nav / nav.cummax(axis=0) - 1.0).min(axis=0).abs()
    score = (
        float(periphery_weight) * scale_01(periph)
        + float(momentum_weight) * scale_01(mom)
        + float(volatility_weight) * scale_01(vol)
        + float(drawdown_weight) * scale_01(dd)
    )
    return pd.Series(score, index=r.columns, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def network_diversifier_weights(
    score: pd.Series,
    *,
    returns: pd.DataFrame,
    n_stocks: int = 25,
    max_weight: float = 0.10,
) -> pd.Series:
    """Top-score portfolio with positive-score/inverse-vol weights and cap redistribution."""
    r = _clean_returns(returns)
    s = pd.Series(score, dtype=float).reindex(r.columns).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return pd.Series(1.0 / r.shape[1], index=r.columns, dtype=float)
    chosen = list(s.sort_values(ascending=False).head(int(n_stocks)).index)
    vol = r[chosen].std(axis=0, ddof=1).replace(0.0, np.nan).fillna(r[chosen].std().median())
    positive = s.reindex(chosen) - min(float(s.reindex(chosen).min()), 0.0) + 1e-6
    raw = positive / vol.clip(lower=1e-8)
    return _cap_normalize(raw, max_weight=max_weight)


def pairwise_corr_for_weights(
    returns: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    lookback: int = 126,
    min_weight: float = 1e-5,
) -> pd.Series:
    """Rolling average pairwise correlation among active holdings at each weight date."""
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    r = r.sort_index()
    wdf = weights.copy()
    wdf.index = pd.to_datetime(wdf.index)
    rows = {}
    for dt, row in wdf.iterrows():
        active = row[row.abs() > float(min_weight)].index.intersection(r.columns)
        if len(active) < 2:
            rows[pd.Timestamp(dt)] = np.nan
            continue
        window = r.loc[:dt, active].tail(int(lookback)).dropna(axis=0, how="any")
        if window.shape[0] < max(20, int(lookback) // 4):
            rows[pd.Timestamp(dt)] = np.nan
            continue
        corr = window.corr().to_numpy(dtype=float)
        tri = corr[np.triu_indices_from(corr, k=1)]
        rows[pd.Timestamp(dt)] = float(np.nanmean(tri)) if len(tri) else np.nan
    return pd.Series(rows, name="avg_pairwise_corr").sort_index()


def network_summary(
    *,
    corr: pd.DataFrame,
    tail: pd.DataFrame,
    centrality: pd.DataFrame,
    nu: float,
) -> pd.Series:
    """Compact diagnostics for a dependence network window."""
    c = _matrix_frame(corr, name="corr")
    tdep = _matrix_frame(tail, name="tail")
    tri = np.triu_indices_from(c, k=1)
    return pd.Series(
        {
            "nu": float(nu),
            "avg_corr": float(np.nanmean(c.to_numpy(dtype=float)[tri])),
            "avg_tail": float(np.nanmean(tdep.to_numpy(dtype=float)[tri])),
            "median_centrality": float(centrality["combined"].median()) if "combined" in centrality else np.nan,
            "top_centrality": float(centrality["combined"].max()) if "combined" in centrality else np.nan,
        }
    )


__all__ = [
    "central_peripheral_weights",
    "centrality_table",
    "combined_centrality",
    "corr_distance",
    "dense_network",
    "dependence_to_distance",
    "kendall_to_t_copula_corr",
    "mst_network",
    "network_diversifier_weights",
    "network_score",
    "network_summary",
    "pairwise_corr_for_weights",
    "pmfg_network",
    "pseudo_observations",
    "scale_01",
    "select_t_copula_nu",
    "shrink_corr",
    "student_t_copula_loglik",
    "student_t_tail_dependence",
]
