"""Label-free measurements the Hedonic calibration search takes on an instance and on every cover it returns.

* :func:`active_density` and :func:`internal_densities` are the density scales of an instance: the edge density over
  the vertices that have an edge, and the internal edge density of each reference community. The search's first
  resolution bracket is read from the second (``[rho_in / 32, rho_in]``, ``rho_in`` its median), and every probe is
  also reported in units of both.
* :class:`CoverAudit` is the regret audit of a labelled cover under the unit-l2 CPM payoff (every vertex of ``k``
  labels takes intensity ``1/sqrt(k)`` in each), computed by the C audit (``stability.prepare``, the code behind
  ``Game.regrets``). Exact for ``gamma`` in [0, 1], unit vertex weights, isolation enabled (a fresh label of gain 0
  is always available) and any cap ``M`` (``None`` = uncapped). Every label gain is affine in ``gamma``,
  ``g_vc = L_vc - gamma R_vc``, so each vertex regret is a convex function of ``gamma``; two endpoint audits
  therefore certify a whole window (:meth:`CoverAudit.window_robust_fraction`).
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
    """Per-vertex regrets of one labelled cover (``rows``: one label list per vertex, any integer labels)."""

    def __init__(self, graph, rows, *, atol: float = 1e-10, rtol: float = 1e-9):
        from hedonic.experiments.overlapping import stability as st

        n = graph.vcount()
        if len(rows) != n:
            raise ValueError("rows must have one entry per vertex")
        k = np.fromiter((len(set(r)) for r in rows), dtype=np.int64, count=n)
        if (k < 1).any():
            raise ValueError("every vertex needs at least one label")
        names = sorted({int(c) for r in rows for c in r})
        index = {name: i for i, name in enumerate(names)}
        self.n, self.k, self.n_labels = n, k, len(names)
        self.atol, self.rtol = float(atol), float(rtol)
        self._cover = st.prepare(graph, [[index[int(c)] for c in r] for r in rows])
        self._vertices = np.arange(n, dtype=np.int64)

    def _cap(self, cap) -> int:
        """``None`` (uncapped): every label and the fresh one fit in a deviation."""
        return self.n_labels + 1 if cap is None else int(cap)

    def evaluate(self, gamma, cap=None):
        """Per-vertex ``regret`` (best utility minus current, 0 when at most 1e-15) and stability ``tolerance`` at the
        scalar resolution ``gamma``, with deviations of at most ``cap`` labels."""
        grid = np.array([float(gamma)])
        regret, tolerance = self._cover.regrets(self._vertices, grid, self._cap(cap), True, atol=self.atol,
                                                rtol=self.rtol)
        return dict(regret=regret[:, 0], tolerance=tolerance[:, 0])

    def stable(self, gamma, cap=None):
        ev = self.evaluate(gamma, cap)
        return ev["regret"] <= ev["tolerance"]

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
