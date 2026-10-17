"""Label-free measurements the Hedonic calibration search takes on an instance and on every cover it returns.

* :func:`active_density` and :func:`internal_densities` are the density scales of an instance: the edge density over
  the vertices that have an edge, and the internal edge density of each reference community. The search's first
  resolution bracket is read from the second (``[rho_in / 32, rho_in]``, ``rho_in`` its median), and every probe is
  also reported in units of both.
* :class:`CoverAudit` is a vectorized regret audit of a labelled cover under the unit-l2 CPM payoff (every vertex of
  ``k`` labels takes intensity ``1/sqrt(k)`` in each). Exact for ``gamma >= 0``, unit vertex weights, isolation
  enabled (a fresh label of gain 0 is always available) and any cap ``M`` (``None`` = uncapped): the current labels,
  the labels held by a neighbour and one fresh label contain a best response for every ``gamma >= 0``, so the
  candidate set does not depend on ``gamma``. Every label gain is affine in ``gamma``, ``g_vc = L_vc - gamma R_vc``,
  so each vertex regret is a convex piecewise-linear function of ``gamma`` and its stable set is an interval; two
  endpoint audits therefore certify a whole window (:meth:`CoverAudit.window_robust_fraction`).
  It mirrors ``robustness.best_response`` but works on CSR arrays, one call for every vertex.
* :func:`golden_min` is a golden-section minimizer for a unimodal function (the mean regret of a fixed cover is convex
  in ``gamma``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- density scales
def active_density(graph) -> tuple[float, int]:
    """Edge density over the vertices with at least one edge, and their number."""
    deg = np.asarray(graph.degree())
    n_act = int((deg > 0).sum())
    if n_act < 2:
        raise ValueError("the graph needs at least one edge")
    return 2.0 * graph.ecount() / (n_act * (n_act - 1)), n_act


def _incidence(n: int, communities) -> sp.csr_matrix:
    """The vertex x community 0/1 matrix of a cover (a repeated member counts once)."""
    sizes = np.fromiter((len(c) for c in communities), dtype=np.int64, count=len(communities))
    rows = np.fromiter((v for c in communities for v in c), dtype=np.int64, count=int(sizes.sum()))
    cols = np.repeat(np.arange(len(communities), dtype=np.int64), sizes)
    B = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(communities)))
    B.data[:] = 1.0  # duplicate (vertex, community) entries were summed
    return B


def _adjacency(graph) -> sp.csr_matrix:
    n = graph.vcount()
    edges = np.asarray(graph.get_edgelist(), dtype=np.int64).reshape(-1, 2)
    return sp.csr_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                         shape=(n, n))


def internal_densities(graph, gt) -> np.ndarray:
    """Internal edge density of every reference community: its edges over its vertex pairs (1 for a single vertex).

    The edges inside community ``c`` are ``(B^T A B)_cc / 2`` for the incidence matrix ``B``, computed as the column
    sums of ``(A B) * B``: one sparse product for all communities instead of one induced subgraph per community. The
    graph must be simple (no loops, no parallel edges), as every graph the search loads is."""
    gt = [list(c) for c in gt]
    if not gt:
        return np.empty(0)
    B = _incidence(graph.vcount(), gt)
    inside = np.asarray((_adjacency(graph) @ B).multiply(B).sum(axis=0)).ravel() / 2.0
    sizes = np.asarray(B.sum(axis=0)).ravel()
    pairs = sizes * (sizes - 1) / 2.0
    return np.where(sizes > 1, inside / np.where(pairs > 0, pairs, 1.0), 1.0)


# --------------------------------------------------------------------------- the regret audit
class CoverAudit:
    def __init__(self, graph, rows):
        n = graph.vcount()
        if len(rows) != n:
            raise ValueError("rows must have one entry per vertex")
        k = np.fromiter((len(r) for r in rows), dtype=np.int64, count=n)
        if (k < 1).any():
            raise ValueError("every vertex needs at least one label")
        labels = np.fromiter((c for r in rows for c in r), dtype=np.int64, count=int(k.sum()))
        _, labels = np.unique(labels, return_inverse=True)
        owner = np.repeat(np.arange(n), k)
        inten = 1.0 / np.sqrt(k)
        n_labels = int(labels.max()) + 1
        F = sp.csr_matrix((inten[owner], (owner, labels)), shape=(n, n_labels))
        mass = np.asarray(F.sum(axis=0)).ravel()
        A = _adjacency(graph)
        L = (A @ F).tocsr()                       # neighbour support L_vc
        own = sp.csr_matrix((np.ones(len(owner)), (owner, labels)), shape=(n, n_labels))
        pattern = (L + own).tocsr()               # candidate labels: support or own
        pattern.sort_indices()
        L = (L + 0 * pattern).tocsr(); L.sort_indices()
        own = (own + 0 * pattern).tocsr(); own.sort_indices()
        Fi = (F + 0 * pattern).tocsr(); Fi.sort_indices()
        self.n, self.k, self.n_labels = n, k, n_labels
        self.indptr = pattern.indptr
        self.row = np.repeat(np.arange(n), np.diff(pattern.indptr))
        self.label = pattern.indices
        self.L = _aligned(L, pattern)
        self.own = _aligned(own, pattern) > 0.5
        self.R = mass[self.label] - _aligned(Fi, pattern)   # mass of the others
        self.inv_sqrt_k = 1.0 / np.sqrt(k)

    def evaluate(self, gamma, cap=None):
        """Per-vertex current utility, best utility, regret, best size, and the crowding exposure of the current and
        best actions (their gamma slopes). ``gamma`` is a scalar or a per-vertex array."""
        gam = np.broadcast_to(np.asarray(gamma, dtype=float), (self.n,))
        g = self.L - gam[self.row] * self.R
        cur = np.bincount(self.row, weights=np.where(self.own, g, 0.0), minlength=self.n) * self.inv_sqrt_k
        cur_slope = np.bincount(self.row, weights=np.where(self.own, self.R, 0.0),
                                minlength=self.n) * self.inv_sqrt_k
        order = np.lexsort((-g, self.row))
        gs, rs, row = g[order], self.R[order], self.row[order]
        start = self.indptr[:-1]
        rank = np.arange(len(order)) - start[row] + 1
        cg = np.cumsum(gs); cr = np.cumsum(rs)
        base_g = np.r_[0.0, cg][start]; base_r = np.r_[0.0, cr][start]
        pref_g = cg - base_g[row]; pref_r = cr - base_r[row]
        util = pref_g / np.sqrt(rank)
        if cap is not None:
            util = np.where(rank <= cap, util, -np.inf)
        best = np.full(self.n, -np.inf)              # the best sorted prefix of each row
        np.maximum.at(best, row, util)
        is_best = util >= best[row] - 1e-15
        first = np.full(self.n, -1)
        idx = np.nonzero(is_best)[0]
        first_idx = np.full(self.n, np.iinfo(np.int64).max)  # the smallest rank attaining the maximum
        np.minimum.at(first_idx, row[idx], idx)
        has = first_idx < np.iinfo(np.int64).max
        first[has] = first_idx[has]
        best_size = np.where(has, rank[np.maximum(first, 0)], 1)
        best_slope = np.where(has, pref_r[np.maximum(first, 0)] / np.sqrt(best_size), 0.0)
        fresh = best < 0.0                           # a fresh singleton (gain 0) is better
        best = np.where(fresh, 0.0, best)
        best_size = np.where(fresh, 1, best_size)
        best_slope = np.where(fresh, 0.0, best_slope)
        regret = np.maximum(best - cur, 0.0)
        return dict(current=cur, best=best, regret=regret, best_size=best_size,
                    current_slope=cur_slope, best_slope=best_slope)

    def stable(self, gamma, cap=None, atol=1e-10, rtol=1e-9):
        ev = self.evaluate(gamma, cap)
        return ev["regret"] <= atol + rtol * np.maximum(1.0, np.abs(ev["current"]))

    def mean_regret(self, gamma, cap=None, vertices=None):
        r = self.evaluate(gamma, cap)["regret"]
        return float(r[vertices].mean() if vertices is not None else r.mean())

    def window_robust_fraction(self, gamma, factor, cap=None, vertices=None):
        """Fraction of vertices stable on [gamma/factor, gamma*factor]; by the resolution-interval theorem two endpoint
        audits certify the window."""
        lo = self.stable(gamma / factor, cap)
        hi = self.stable(min(gamma * factor, 1.0), cap)
        both = lo & hi
        return float(both[vertices].mean() if vertices is not None else both.mean())


def _aligned(M, pattern):
    """Data of M on the sparsity pattern of ``pattern`` (same shape, both sorted CSR, M's pattern a subset of
    pattern's after adding 0*pattern)."""
    M = M.tocsr(); M.sort_indices()
    if M.nnz == pattern.nnz and np.array_equal(M.indices, pattern.indices):
        return np.asarray(M.data, dtype=float)
    out = np.zeros(pattern.nnz)
    key_p = (pattern.indices.astype(np.int64)
             + np.repeat(np.arange(pattern.shape[0]), np.diff(pattern.indptr)) * pattern.shape[1])
    key_m = M.indices.astype(np.int64) + np.repeat(np.arange(M.shape[0]), np.diff(M.indptr)) * M.shape[1]
    out[np.searchsorted(key_p, key_m)] = M.data
    return out


def golden_min(f, lo, hi, tol=1e-3, max_iter=60):
    """Golden-section minimization of a unimodal (here: convex) ``f`` on [lo, hi]: (x, f(x), trace)."""
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    trace = [(c, fc), (d, fd)]
    for _ in range(max_iter):
        if b - a < tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a); fc = f(c); trace.append((c, fc))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a); fd = f(d); trace.append((d, fd))
    x = c if fc <= fd else d
    return x, min(fc, fd), trace
