from __future__ import annotations

from collections.abc import Mapping

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from quantfinlab.portfolio.covariance import make_psd


def _cap_series(index, w_max):
    idx = pd.Index(index)
    if isinstance(w_max, (pd.Series, Mapping)):
        caps = pd.Series(w_max, dtype=float).reindex(idx).fillna(1.0)
    else:
        caps = pd.Series(float(w_max), index=idx, dtype=float)
    if float(caps.sum()) < 1.0 - 1e-12:
        caps = pd.Series(max(float(caps.max()), 1.0 / len(idx)), index=idx, dtype=float)
    return caps.clip(lower=0.0)


def _clean_weights(values, index, *, w_min=0.0, w_max=1.0):
    idx = pd.Index(index)
    caps = _cap_series(idx, w_max)
    w = pd.Series(values, index=idx, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=float(w_min), upper=caps)
    if float(w.sum()) <= 1e-12:
        w = pd.Series(1.0 / len(idx), index=idx, dtype=float)
    else:
        w = w / float(w.sum())
    for _ in range(50):
        over = w > caps + 1e-12
        if not bool(over.any()):
            break
        extra = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = (caps[~over] - w[~over]).clip(lower=0.0)
        if float(room.sum()) <= 1e-12:
            break
        w.loc[room.index] += extra * room / float(room.sum())
    w = w.clip(lower=float(w_min), upper=caps)
    return w / float(w.sum()) if float(w.sum()) > 1e-12 else pd.Series(1.0 / len(idx), index=idx)


def _solve(problem):
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver not in installed:
            continue
        try:
            problem.solve(solver=solver, verbose=False)
            if problem.status in {"optimal", "optimal_inaccurate"}:
                return problem.status
        except Exception:
            pass
    return str(problem.status)


def _cov_to_corr(cov_ann):
    cov = make_psd(np.asarray(cov_ann, dtype=float), eps=1e-10)
    diag = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(diag, diag)
    corr = np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def corr_to_distance(corr):
    arr = np.asarray(corr, dtype=float)
    arr = np.clip(arr, -1.0, 1.0)
    return np.sqrt(np.clip((1.0 - arr) / 2.0, 0.0, 1.0))


def _linkage(cov_ann, linkage_method="average"):
    corr = _cov_to_corr(cov_ann)
    dist = corr_to_distance(corr)
    link = hierarchy.linkage(squareform(dist, checks=False), method=str(linkage_method))
    return corr, dist, link


def cluster_labels(cov_ann, *, tickers=None, n_clusters=3, linkage_method="average"):
    labels = pd.Index(tickers if tickers is not None else [f"a{i}" for i in range(np.asarray(cov_ann).shape[0])])
    _, _, link = _linkage(cov_ann, linkage_method=linkage_method)
    vals = hierarchy.fcluster(link, t=int(n_clusters), criterion="maxclust")
    return pd.Series(vals, index=labels, name="cluster")


def cluster_membership_table(cov_ann, *, tickers=None, n_clusters=3, linkage_method="average"):
    s = cluster_labels(cov_ann, tickers=tickers, n_clusters=n_clusters, linkage_method=linkage_method)
    return pd.DataFrame({"asset": s.index, "cluster": s.values}).sort_values(["cluster", "asset"]).reset_index(drop=True)


def _cluster_variance(cov, items):
    sub = cov[np.ix_(items, items)]
    diag = np.maximum(np.diag(sub), 1e-12)
    ivp = 1.0 / diag
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov_ann, *, tickers=None, linkage_method="average", w_min=0.0, w_max=0.40):
    """Compute hierarchical risk parity portfolio weights.

    The function builds a correlation-distance hierarchy from the covariance
    matrix, orders assets by the hierarchical clustering leaves, and recursively
    allocates capital between clusters using inverse cluster variance.

    Parameters
    ----------
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    tickers : sequence of str, optional
        Asset labels. Generated labels are used when omitted.
    linkage_method : str, default="average"
        Hierarchical clustering linkage method.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.Series
        HRP weights indexed by asset.

    Notes
    -----
    The covariance matrix is projected to positive-semidefinite form before
    clustering and allocation.
    """

    labels = pd.Index(tickers if tickers is not None else [f"a{i}" for i in range(np.asarray(cov_ann).shape[0])])
    cov = make_psd(np.asarray(cov_ann, dtype=float), eps=1e-10)
    _, _, link = _linkage(cov, linkage_method=linkage_method)
    order = list(hierarchy.leaves_list(link))
    alloc = pd.Series(0.0, index=range(len(labels)), dtype=float)

    def split(items, weight):
        if len(items) == 1:
            alloc.loc[items[0]] += weight
            return
        k = len(items) // 2
        left, right = items[:k], items[k:]
        lv = _cluster_variance(cov, left)
        rv = _cluster_variance(cov, right)
        lw = rv / (lv + rv) if lv + rv > 1e-18 else 0.5
        split(left, weight * lw)
        split(right, weight * (1.0 - lw))

    split(order, 1.0)
    return _clean_weights(pd.Series(alloc.values, index=labels), labels, w_min=w_min, w_max=w_max)


def _mv_weights(mu, cov, *, cap=0.40, lam=3.0):
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    cov_arr = make_psd(np.asarray(cov, dtype=float), eps=1e-10)
    n = len(mu_arr)
    cap_use = max(float(cap), 1.0 / n)
    w = cp.Variable(n)
    problem = cp.Problem(cp.Maximize(mu_arr @ w - 0.5 * float(lam) * cp.quad_form(w, cp.psd_wrap(cov_arr))), [cp.sum(w) == 1.0, w >= 0.0, w <= cap_use])
    status = _solve(problem)
    if w.value is None or status not in {"optimal", "optimal_inaccurate"}:
        return np.ones(n, dtype=float) / n
    return _clean_weights(w.value, range(n), w_max=cap_use).to_numpy(dtype=float)


def nco_mv_weights(
    cov_ann,
    mu_ann,
    *,
    tickers=None,
    n_clusters=3,
    inner_lambda=3.0,
    outer_lambda=3.0,
    cluster_cap=0.75,
    linkage_method="average",
    w_min=0.0,
    w_max=0.40,
):
    """Compute nested-clustered mean-variance portfolio weights.

    The function clusters assets, solves a mean-variance allocation within each
    cluster, builds a cluster-level covariance/return problem from those inner
    portfolios, and then allocates across clusters.

    Parameters
    ----------
    cov_ann : array-like or pandas.DataFrame
        Annualized covariance matrix.
    mu_ann : array-like or pandas.Series
        Annualized expected returns.
    tickers : sequence of str, optional
        Asset labels.
    n_clusters : int, default=3
        Number of clusters used for nested optimization.
    inner_lambda : float, default=3.0
        Risk-aversion parameter for within-cluster optimization.
    outer_lambda : float, default=3.0
        Risk-aversion parameter for across-cluster optimization.
    cluster_cap : float, default=0.75
        Maximum allocation to any single cluster.
    linkage_method : str, default="average"
        Hierarchical clustering linkage method.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.Series
        Nested-clustered portfolio weights indexed by asset.

    Notes
    -----
    Nested optimization can reduce estimation error by separating within-cluster
    selection from between-cluster capital allocation.
    """

    labels = pd.Index(tickers if tickers is not None else pd.Series(mu_ann).index)
    cov = pd.DataFrame(make_psd(np.asarray(cov_ann, dtype=float), eps=1e-10), index=labels, columns=labels)
    mu = pd.Series(mu_ann, dtype=float).reindex(labels).fillna(0.0)
    clusters = cluster_labels(cov, tickers=labels, n_clusters=n_clusters, linkage_method=linkage_method)
    cluster_ids = sorted(pd.unique(clusters))
    inner = {}
    cluster_mu = []
    for cid in cluster_ids:
        names = list(clusters[clusters.eq(cid)].index)
        inner_cap = max(float(w_max), 1.0 / len(names))
        vals = _mv_weights(mu.reindex(names).values, cov.loc[names, names].values, cap=inner_cap, lam=inner_lambda)
        inner[cid] = pd.Series(vals, index=names, dtype=float)
        cluster_mu.append(float(mu.reindex(names).values @ vals))
    cluster_cov = np.zeros((len(cluster_ids), len(cluster_ids)), dtype=float)
    for i, ci in enumerate(cluster_ids):
        ni = list(inner[ci].index)
        wi = inner[ci].to_numpy(dtype=float)
        for j, cj in enumerate(cluster_ids):
            nj = list(inner[cj].index)
            wj = inner[cj].to_numpy(dtype=float)
            cluster_cov[i, j] = float(wi @ cov.loc[ni, nj].to_numpy(dtype=float) @ wj)
    outer = _mv_weights(cluster_mu, cluster_cov, cap=cluster_cap, lam=outer_lambda)
    out = pd.Series(0.0, index=labels, dtype=float)
    for j, cid in enumerate(cluster_ids):
        out.loc[inner[cid].index] = outer[j] * inner[cid]
    return _clean_weights(out, labels, w_min=w_min, w_max=w_max)


def hrp_weight_frame(cache, rebalance_dates, *, cov_model, linkage_method="average", w_min=0.0, w_max=0.40):
    """Build a rebalance-date panel of hierarchical risk parity weights.

    Parameters
    ----------
    cache : mapping
        Rebalance-state cache containing tickers and covariance matrices.
    rebalance_dates : sequence
        Candidate rebalance dates.
    cov_model : str
        Covariance model key to extract from each cached state.
    linkage_method : str, default="average"
        Hierarchical clustering linkage method.
    w_min : float, default=0.0
        Minimum per-asset weight.
    w_max : float, default=0.40
        Maximum per-asset weight.

    Returns
    -------
    pandas.DataFrame
        Weight frame indexed by rebalance date.

    Notes
    -----
    Dates not present in the cache are skipped.
    """

    rows = []
    for dt in [pd.Timestamp(x) for x in rebalance_dates if pd.Timestamp(x) in cache][:-1]:
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        w = hrp_weights(state["cov_ann_map"][cov_model], tickers=tickers, linkage_method=linkage_method, w_min=w_min, w_max=w_max)
        rows.append(w.rename(pd.Timestamp(dt)))
    return pd.DataFrame(rows).fillna(0.0)


def nco_mv_weight_frame(cache, rebalance_dates, *, cov_model, mu_model, n_clusters=3, inner_lambda=3.0, outer_lambda=3.0, cluster_cap=0.75, linkage_method="average", w_min=0.0, w_max=0.40):
    rows = []
    for dt in [pd.Timestamp(x) for x in rebalance_dates if pd.Timestamp(x) in cache][:-1]:
        state = cache[pd.Timestamp(dt)]
        tickers = list(state.get("tickers", state["R_cov"].columns))
        mu = state["mu_ann_map"][cov_model][mu_model].reindex(tickers).fillna(0.0)
        w = nco_mv_weights(state["cov_ann_map"][cov_model], mu, tickers=tickers, n_clusters=n_clusters, inner_lambda=inner_lambda, outer_lambda=outer_lambda, cluster_cap=cluster_cap, linkage_method=linkage_method, w_min=w_min, w_max=w_max)
        rows.append(w.rename(pd.Timestamp(dt)))
    return pd.DataFrame(rows).fillna(0.0)


__all__ = [
    "cluster_labels",
    "cluster_membership_table",
    "corr_to_distance",
    "hrp_weight_frame",
    "hrp_weights",
    "nco_mv_weight_frame",
    "nco_mv_weights",
]
