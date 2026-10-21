"""The detector and the stability audit over NumPy arrays, without igraph.

:func:`detect` is :meth:`hedonic.Game.community_hedonic` for a graph given as
a vertex count and an edge list: the same parameters, defaults, checks and
core call, so the two return the same bits for the same graph, weights,
options and seed. :func:`stable_intervals` and :func:`regrets` are the audit
of :meth:`hedonic.Game.stable_intervals` and :meth:`hedonic.Game.regrets`.
Only NumPy is imported, here and on every call; igraph is never loaded.

A graph is ``n`` vertices ``0 .. n - 1`` and ``edges``, either

* an ``(m, 2)`` integer array (or a list of pairs): edge ``e`` joins
  ``edges[e, 0]`` and ``edges[e, 1]``, in that order of edges; or
* a CSR adjacency ``(indptr, indices)`` or ``(indptr, indices, data)`` of a
  symmetric ``n x n`` matrix: each undirected edge is stored twice, as
  ``(i, j)`` and ``(j, i)``, and the edges are the entries with ``i <= j`` in
  row-major order, ``data`` their weights (an explicit zero is an edge of
  weight 0). A matrix stored as one triangle is ``A + A.T`` first.

The graph is undirected, so a pair's order does not matter; parallel edges
add up; a self-loop is refused by the detector. ``edges`` given as ``(m, 2)``
follows the edge order of ``igraph.Graph(n, edges)``, which is what makes
``detect(n, edges, ...)`` and ``Game(igraph.Graph(n, edges)).community_hedonic(...)``
identical (the visiting order depends on the edge order, not on each pair's).

``from_networkx`` and ``from_scipy`` build an :class:`ArrayGraph` (vertex count,
edges, weights and the original nodes) from those libraries, imported only
when called.

An unseeded call (``seed=None``) draws from Python's ``random`` module, as
``Game`` does with python-igraph's default generator.
"""

from __future__ import annotations

import random as _random
import functools
from dataclasses import dataclass
from itertools import chain
from numbers import Integral
from typing import Any, NamedTuple

import numpy as np

from . import _audit, _core
from ._game.detection import Source, plan, rng_label
from ._game.helpers import core_versions, real_vector

__all__ = [
    "ArrayGraph",
    "Result",
    "detect",
    "from_networkx",
    "from_scipy",
    "regrets",
    "stable_intervals",
]


# ---- The graph -------------------------------------------------------------------


def _integer(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _vertex_count(n) -> int:
    if not _integer(n) or n < 0:
        raise ValueError(f"n must be a non-negative integer (the vertex count), got {n!r}")
    return int(n)


def _integer_array(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return array.astype(np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must hold integers, got an array of dtype {array.dtype}")
    return array


def _is_csr(edges, n: int) -> bool:
    """Whether ``edges`` is ``(indptr, indices[, data])``: a tuple whose first
    entry is one-dimensional with n + 1 entries (a tuple of vertex pairs is
    that only when n = 1, where every pair is a self-loop)."""
    return (
        isinstance(edges, tuple) and len(edges) in (2, 3)
        and all(np.ndim(part) == 1 for part in edges)
        and len(edges[0]) == n + 1
    )


def _csr_edges(n: int, indptr, indices, data):
    """The edges (i <= j, row-major) and weights of a symmetric CSR adjacency."""
    indptr = _integer_array(indptr, "indptr")
    indices = _integer_array(indices, "indices")
    if indptr[0] != 0 or indptr[-1] != len(indices) or (np.diff(indptr) < 0).any():
        raise ValueError(
            "indptr must start at 0, never decrease and end at len(indices) "
            f"({len(indices)}), as a CSR matrix's does"
        )
    if len(indices) and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"indices must be columns 0 .. {n - 1} of the {n} x {n} adjacency")
    # np.repeat and np.bincount take counts and labels as the platform's index type
    # (int32 on 32-bit builds: Pyodide, win32), never int64 cast unsafely.
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr).astype(np.intp))
    columns = indices.astype(np.int64)
    if data is not None:
        data = real_vector(data, len(indices), "data", per_vertex=False)
    upper, lower = rows < columns, rows > columns
    # Symmetric: the entries below the diagonal mirror those above, weights included.
    mine = (rows[upper], columns[upper]) + ((data[upper],) if data is not None else ())
    mirror = (columns[lower], rows[lower]) + ((data[lower],) if data is not None else ())
    a, b = (np.lexsort(part[::-1]) for part in (mine, mirror))
    if len(a) != len(b) or any(not np.array_equal(x[a], y[b]) for x, y in zip(mine, mirror)):
        raise ValueError(
            "the CSR adjacency must be symmetric (each undirected edge stored as "
            "(i, j) and (j, i), with the same weight); for a matrix stored as one "
            "triangle, pass A + A.T"
        )
    kept = rows <= columns
    return rows[kept], columns[kept], None if data is None else data[kept]


def _edge_arrays(n: int, edges):
    """``edges`` as ``(first, second, data)``: int64 endpoints with
    ``first <= second``, as igraph stores an undirected edge, and the CSR's
    weights (None for an edge list)."""
    if _is_csr(edges, n):
        first, second, data = _csr_edges(n, *edges, *([None] * (3 - len(edges))))
        return np.ascontiguousarray(first), np.ascontiguousarray(second), data
    array = np.asarray(edges)
    if array.size == 0:
        array = array.reshape(0, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            "edges must be an (m, 2) array of vertex pairs, or a CSR adjacency "
            f"(indptr, indices[, data]) with n + 1 = {n + 1} entries in indptr; "
            f"got an array of shape {array.shape} (two endpoint arrays u, v are "
            "np.column_stack((u, v)))"
        )
    array = _integer_array(array, "edges")
    outside = (array < 0) | (array >= n)
    if outside.any():
        e = int(np.argmax(outside.any(axis=1)))
        raise ValueError(
            f"edge {e} = ({array[e, 0]}, {array[e, 1]}) has a vertex outside 0 .. {n - 1}"
        )
    array = array.astype(np.int64, copy=False)
    first = np.ascontiguousarray(np.minimum(array[:, 0], array[:, 1]))
    second = np.ascontiguousarray(np.maximum(array[:, 0], array[:, 1]))
    return first, second, None


def _given_weights(data, edge_weights):
    if data is not None and edge_weights is not None:
        raise ValueError("pass the weights as the CSR's data or as edge_weights, not both")
    return data if data is not None else edge_weights


def _edge_values(values, m: int, nonnegative: bool) -> np.ndarray:
    if isinstance(values, str):
        raise ValueError(
            "edge_weights must be numbers, one per edge; attribute names need an "
            "igraph graph (hedonic.Game)"
        )
    return real_vector(values, m, "edge_weights", per_vertex=False, nonnegative=nonnegative)


def _node_values(values, n: int) -> np.ndarray | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(
            "node_weights must be numbers, one per vertex; attribute names need an "
            "igraph graph (hedonic.Game)"
        )
    return real_vector(values, n, "node_weights", per_vertex=True, nonnegative=True)


class _Edges:
    """The graph as the core's binding reads an ``igraph.Graph``: ``vcount``,
    ``ecount``, ``get_edgelist`` and, without copying the edges, the head of
    an ``igraph_t`` behind ``__graph_as_capsule`` (``n``, undirected, and the
    ``from``/``to`` vectors, which here point into the two endpoint arrays).
    The binding checks that head against ``es[e].tuple`` and falls back to
    ``get_edgelist`` when it does not match, so this is an optimization only."""

    def __init__(self, n: int, first: np.ndarray, second: np.ndarray):
        self.n, self.first, self.second = n, first, second
        self._head = None

    def vcount(self) -> int:
        return self.n

    def ecount(self) -> int:
        return len(self.first)

    def is_directed(self) -> bool:
        return False

    def get_edgelist(self) -> list[tuple[int, int]]:
        return list(zip(self.first.tolist(), self.second.tolist()))

    @property
    def es(self):
        return _EdgeTuples(self.first, self.second)


class _EdgeTuples:
    def __init__(self, first, second):
        self.first, self.second = first, second

    def __len__(self):
        return len(self.first)

    def __getitem__(self, e):
        return _EdgeTuple((int(self.first[e]), int(self.second[e])))


class _EdgeTuple(NamedTuple):
    tuple: tuple


@functools.cache
def _capsule_types():
    """The head of igraph's ``igraph_t`` (igraph 0.10 and 1.0) that the binding
    reads, and ``PyCapsule_New``."""
    import ctypes

    class Vector(ctypes.Structure):
        _fields_ = [("stor_begin", ctypes.c_void_p), ("stor_end", ctypes.c_void_p),
                    ("end", ctypes.c_void_p)]

    class Head(ctypes.Structure):
        _fields_ = [("n", ctypes.c_int64), ("directed", ctypes.c_ubyte),
                    ("source", Vector), ("target", Vector)]

    new = ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.c_void_p, ctypes.c_char_p,
                            ctypes.c_void_p)(("PyCapsule_New", ctypes.pythonapi))
    return Vector, Head, new, ctypes.addressof


def _graph_as_capsule(self):
    # The name would be mangled in the class body (it starts with two
    # underscores and does not end with them), so it is set below.
    Vector, Head, new, address = _capsule_types()

    def vector(array):
        begin = array.ctypes.data
        end = begin + array.nbytes
        return Vector(begin, end, end)

    # igraph_edge() reports an undirected edge as (to, from): to is the
    # smaller end, which the binding reads first. The head lives as long as
    # this object, which the binding holds for the call.
    self._head = Head(self.n, 0, vector(self.second), vector(self.first))
    return new(address(self._head), None, None)


setattr(_Edges, "__graph_as_capsule", _graph_as_capsule)


# ---- Results ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Result:
    """What :func:`detect` returns.

    ``membership`` is each vertex's community (a partition) or None (a cover).
    Both give ``offsets`` and ``labels``, the rows of a CSR array: vertex
    ``v`` belongs to ``labels[offsets[v]:offsets[v + 1]]`` (sorted in a
    cover; for a partition ``labels`` is ``membership``). Labels are
    ``0 .. community_count - 1``. ``quality`` and ``equilibrium_slack`` are
    those of :meth:`hedonic.Game.community_hedonic` (its
    ``_hedonic_native_quality`` and ``_hedonic_provenance["equilibrium_slack"]``),
    and ``provenance`` its ``_hedonic_provenance`` (``versions`` without
    igraph). A result is a start: pass it as ``initial_membership``.
    """

    membership: np.ndarray | None
    offsets: np.ndarray
    labels: np.ndarray
    community_count: int
    quality: float
    equilibrium_slack: float
    provenance: dict

    @property
    def overlapping(self) -> bool:
        return self.membership is None

    def rows(self) -> list[list[int]]:
        """One list of labels per vertex: ``Game.memberships`` after the same call."""
        labels = self.labels.tolist()
        bounds = self.offsets.tolist()
        return [labels[bounds[v]:bounds[v + 1]] for v in range(len(bounds) - 1)]

    def communities(self) -> list[np.ndarray]:
        """The vertices of each community, by label, in vertex order (the
        clusters of the ``VertexClustering`` or ``VertexCover`` Game returns)."""
        n = len(self.offsets) - 1
        vertex = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.offsets).astype(np.intp))
        order = np.argsort(self.labels, kind="stable")
        sizes = np.bincount(self.labels.astype(np.intp), minlength=self.community_count)
        return np.split(vertex[order], np.cumsum(sizes)[:-1])

    def __repr__(self) -> str:
        kind = "cover" if self.overlapping else "partition"
        return (
            f"Result({kind} of {len(self.offsets) - 1} vertices into "
            f"{self.community_count} communities, quality={self.quality!r}, "
            f"equilibrium_slack={self.equilibrium_slack!r})"
        )


def _state(membership):
    """A :class:`Result` as a start or an audited cover; anything else as given."""
    if isinstance(membership, Result):
        return membership.membership if membership.membership is not None else membership.rows()
    return membership


def _random_module():
    return _random


# ---- The detector ----------------------------------------------------------------


def detect(
    n,
    edges,
    *,
    initial_membership=None,
    max_communities: int | None = None,
    max_memberships: int = 1,
    resolution: float | None = None,
    allow_isolation: bool = False,
    edge_weights=None,
    seed: int | None = None,
    node_weights=None,
    max_total_communities: int | None = None,
    n_communities: int | None = None,
) -> Result:
    """Best-response local moving of the CPM hedonic game on a graph of ``n``
    vertices and ``edges`` (see the module), to a certified equilibrium.

    Every option is that of :meth:`hedonic.Game.community_hedonic`, with its
    default and its checks: ``max_memberships=1`` gives a partition, ``M > 1``
    a cover with at most ``M`` communities per vertex (``-1``: no practical
    cap); ``resolution=None`` is the weighted density; ``initial_membership``
    is flat labels, per-vertex rows (a list, or an integer NumPy array) or a
    :class:`Result`; ``max_communities`` draws a random disjoint start from
    ``seed`` (``hedonic.utils.sample_uniform_ints``); ``edge_weights`` (one
    per edge, in edge order; a CSR's ``data`` gives them instead) and
    ``node_weights`` (one per vertex) are numbers, not attribute names.

    For the same graph (the same edges in the same order), weights, options
    and seed, the result has the bits of
    ``Game(igraph.Graph(n, edges)).community_hedonic(...)``: membership or
    rows, quality, slack and provenance (but ``versions`` and the result type).
    """
    n = _vertex_count(n)
    if n < 1:
        raise ValueError("detect requires a nonempty graph")
    first, second, data = _edge_arrays(n, edges)
    loops = first == second
    if loops.any():
        e = int(np.argmax(loops))
        raise ValueError(
            "detect requires a graph without self-loops (a loop ties a vertex to "
            f"no one else); edge {e} joins vertex {int(first[e])} to itself"
        )
    m = len(first)
    source = Source(
        n=n,
        m=m,
        edge_vector=lambda values, nonnegative: _edge_values(values, m, nonnegative),
        node_vector=lambda values: _node_values(values, n),
        edge_pairs=lambda: zip(first.tolist(), second.tolist()),
        caller="detect",
    )
    call = plan(
        source,
        _state(initial_membership),
        max_communities=max_communities,
        max_memberships=max_memberships,
        resolution=resolution,
        allow_isolation=allow_isolation,
        edge_weights=_given_weights(data, edge_weights),
        seed=seed,
        node_weights=node_weights,
        max_total_communities=max_total_communities,
        n_communities=n_communities,
        default_rng=_random_module,
    )
    rng = _random.Random(call.seed) if call.seed is not None else _random
    out = _core.detect(
        _Edges(n, first, second),
        rng,
        call.resolution,
        max_memberships=call.max_memberships,
        start=call.start,
        allow_isolation=call.allow_isolation,
        edge_weights=call.edge_weights,
        node_weights=call.node_weights,
        max_total_communities=call.native_max_total,
        n_communities=call.n_communities,
    )
    if not call.overlapping:
        membership_list, count, quality, slack = out
        membership = np.array(membership_list, dtype=np.int64)
        offsets, labels = np.arange(n + 1, dtype=np.int64), membership
    else:
        rows, _clusters, count, quality, slack = out
        membership = None
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.fromiter(map(len, rows), dtype=np.int64, count=n), out=offsets[1:])
        labels = np.fromiter(chain.from_iterable(rows), dtype=np.int64, count=int(offsets[-1]))
    provenance = call.provenance(core_versions(), rng_label(call.seed, _random_module),
                                 len(np.unique(labels)), slack)
    return Result(membership, offsets, labels, int(count), quality, slack, provenance)


# ---- The stability audit ---------------------------------------------------------


def _audit_state(n, edges, memberships, max_memberships, edge_weights, vertices, caller):
    n = _vertex_count(n)
    first, second, data = _edge_arrays(n, edges)
    weights = _given_weights(data, edge_weights)
    selected = _audit.vertex_indices(n, vertices)
    if memberships is None:
        raise ValueError(f"{caller} needs a cover: memberships (rows, flat labels or a Result)")
    m = len(first)
    state, cap = _audit.state(
        _Edges(n, first, second), n, _state(memberships), max_memberships,
        lambda: None if weights is None else _edge_values(weights, m, True),
    )
    return state, cap, selected


def stable_intervals(
    n, edges, memberships, *, max_memberships=None, allow_isolation=False,
    edge_weights=None, atol=1e-10, rtol=1e-9, vertices=None,
):
    """:meth:`hedonic.Game.stable_intervals` of ``memberships`` (per-vertex
    rows, flat labels or a :class:`Result`) on the graph ``n``, ``edges``:
    each selected vertex's sorted, disjoint, closed stable intervals of the
    resolution in [0, 1]."""
    state, cap, selected = _audit_state(n, edges, memberships, max_memberships,
                                        edge_weights, vertices, "stable_intervals")
    return _audit.intervals(state, selected, cap, allow_isolation, atol, rtol, True)


def regrets(
    n, edges, resolutions, memberships, *, max_memberships=None,
    allow_isolation=False, edge_weights=None, atol=1e-10, rtol=1e-9, vertices=None,
):
    """:meth:`hedonic.Game.regrets` of ``memberships`` on the graph ``n``,
    ``edges``: a :class:`hedonic.stability.RegretResult` of
    ``(len(vertices), len(resolutions))`` arrays."""
    gammas = _audit.resolution_grid(resolutions)
    state, cap, selected = _audit_state(n, edges, memberships, max_memberships,
                                        edge_weights, vertices, "regrets")
    return _audit.regret_arrays(state, selected, gammas, cap, allow_isolation, atol, rtol, True)


# ---- Adapters --------------------------------------------------------------------


class ArrayGraph(NamedTuple):
    """A graph as this module takes it: ``n`` vertices, ``edges`` an
    ``(m, 2)`` int64 array, optional ``edge_weights`` and ``node_weights``
    (float64), and ``nodes``, the source graph's node of each vertex (None
    when the vertices are the source's own indices)."""

    n: int
    edges: np.ndarray
    edge_weights: np.ndarray | None = None
    node_weights: np.ndarray | None = None
    nodes: list | None = None

    def detect(self, **options) -> Result:
        """:func:`detect` on this graph and its weights (``options`` may replace them)."""
        weights = {"edge_weights": self.edge_weights, "node_weights": self.node_weights}
        return detect(self.n, self.edges, **{**weights, **options})

    def stable_intervals(self, memberships, **options):
        """:func:`stable_intervals` on this graph and its edge weights."""
        return stable_intervals(self.n, self.edges, memberships,
                                **{"edge_weights": self.edge_weights, **options})

    def regrets(self, resolutions, memberships, **options):
        """:func:`regrets` on this graph and its edge weights."""
        return regrets(self.n, self.edges, resolutions, memberships,
                       **{"edge_weights": self.edge_weights, **options})


def _attribute_values(values: list[Any], given: bool, name: str, attribute, what: str,
                      per_vertex: bool):
    if not given:
        return None
    where = f"{name}: the {what} attribute {attribute!r} at {what} " + "{}"
    return real_vector(values, len(values), name, per_vertex=per_vertex,
                       nonnegative=per_vertex, where=where)


def from_networkx(graph, weight: str | None = "weight", node_weight: str | None = None) -> ArrayGraph:
    """An :class:`ArrayGraph` of an undirected NetworkX graph (``Graph`` or ``MultiGraph``).

    Vertex ``i`` is ``nodes[i]``, the ``i``-th node of ``graph.nodes`` (the
    order nodes were added). The edges follow ``graph.edges``, one per
    parallel edge of a multigraph. ``weight`` names the edge attribute read
    as weights: an edge without it weighs 1 (NetworkX's convention), and when
    no edge has it the graph is unweighted (``edge_weights`` None); None
    ignores weights. ``node_weight`` names a node attribute the same way
    (None, the default: unit vertex weights). Self-loops are kept, and
    :func:`detect` refuses them (``graph.remove_edges_from(nx.selfloop_edges(graph))``).
    """
    if graph.is_directed():
        raise ValueError(
            "from_networkx needs an undirected graph; convert it first "
            "(graph.to_undirected())"
        )
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    pairs, weights, weighted = [], [], False
    for u, v, attributes in graph.edges(data=True):
        pairs.append((index[u], index[v]))
        if weight is not None:
            weighted = weighted or weight in attributes
            weights.append(attributes.get(weight, 1.0))
    masses, massed = [], False
    if node_weight is not None:
        for node in nodes:
            attributes = graph.nodes[node]
            massed = massed or node_weight in attributes
            masses.append(attributes.get(node_weight, 1.0))
    return ArrayGraph(
        n=len(nodes),
        edges=np.array(pairs, dtype=np.int64).reshape(-1, 2),
        edge_weights=_attribute_values(weights, weighted, "edge_weights", weight, "edge", False),
        node_weights=_attribute_values(masses, massed, "node_weights", node_weight, "vertex", True),
        nodes=nodes,
    )


def from_scipy(matrix, weighted: bool = True) -> ArrayGraph:
    """An :class:`ArrayGraph` of a symmetric SciPy sparse adjacency matrix (or array).

    Vertex ``i`` is row and column ``i``. The edges are the stored entries
    with ``i <= j`` in row-major order of the matrix's CSR form (the CSR
    ``edges`` of :func:`detect`), and their values are the weights unless
    ``weighted=False`` (booleans weigh 1). The lower triangle must mirror
    the upper one; an explicit zero is an edge of weight 0
    (``matrix.eliminate_zeros()`` drops them).
    """
    import scipy.sparse as sparse

    if not sparse.issparse(matrix):
        raise TypeError(f"from_scipy takes a SciPy sparse matrix or array, got {type(matrix).__name__}")
    rows, columns = matrix.shape
    if rows != columns:
        raise ValueError(f"an adjacency matrix is square, got shape {matrix.shape}")
    csr = matrix.tocsr()
    data = None
    if weighted:
        data = csr.data.astype(np.float64) if csr.data.dtype == np.bool_ else csr.data
    first, second, values = _csr_edges(rows, csr.indptr, csr.indices, data)
    return ArrayGraph(rows, np.column_stack([first, second]), values, None, None)
