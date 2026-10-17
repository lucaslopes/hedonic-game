"""The CPM hedonic game, disjoint and overlapping.

The detector behind :meth:`hedonic.Game.community_hedonic`.
``community_hedonic(graph, ...)`` runs best-response local moving on an
undirected ``igraph.Graph`` until no vertex has an improving move. With
``max_memberships=1`` it returns a partition, a Nash equilibrium of the
CPM hedonic game (Felipe, Avrachenkov & Menasche, Physica A 680:130989, 2025);
with ``max_memberships > 1`` it returns a cover of the unit-l2 overlapping
game, every vertex in at most ``max_memberships`` communities. A move is
taken only when its computed gain exceeds a rigorous bound of its rounding
error, so every move raises the exact potential and every call terminates;
the result reports ``equilibrium_slack``, a bound of what any vertex could
still gain by deviating (0 when the arithmetic is exact; ``native/README.md``).

The C core (``native/`` in the repository, the extension ``hedonic._core``)
has no igraph dependency: it takes an edge list and draws random numbers from
``rng.getrandbits(32)``. The binding copies the edge list out of the igraph
graph (through python-igraph's ``__graph_as_capsule``, checked against
``get_edgelist``'s order, falling back to ``get_edgelist``); igraph is
otherwise used only for the result types.
"""

from __future__ import annotations

import math as _math
import random as _random
from array import array as _array

from ._core import core_version
from ._core import cover as _cover
from ._core import detect as _detect
from ._core import partition as _partition

__all__ = [
    "community_hedonic",
    "cover",
    "core_version",
    "default_rng",
    "partition",
]

def partition(n, edges, rng, **kwargs):
    """The raw partition call: see ``help(hedonic._core.partition)``."""
    return _partition(n, edges, rng, **kwargs)


def cover(n, edges, rng, **kwargs):
    """The raw cover call: see ``help(hedonic._core.cover)``."""
    return _cover(n, edges, rng, **kwargs)


def default_rng():
    """The generator igraph would draw from now: its current Python generator
    when the binding exposes ``igraph.get_random_number_generator``, else the
    ``random`` module (python-igraph's default)."""
    try:
        import igraph
    except ImportError:  # pragma: no cover - igraph is optional
        return _random
    getter = getattr(igraph, "get_random_number_generator", None)
    if getter is not None:
        current = getter()
        if hasattr(current, "getrandbits"):
            return current
    return _random


def _edge_weight_buffer(graph, values):
    """One real weight per edge whose total is finite (signs are allowed)."""
    if values is None:
        return None
    values = graph.es[values] if isinstance(values, str) else list(values)
    if len(values) != graph.ecount():
        raise ValueError("edge_weights must have one value per graph edge")
    buffer = _array("d")
    total = 0.0
    for value in values:
        if isinstance(value, bool):
            raise ValueError("edge_weights must be real numbers")
        weight = float(value)
        total += weight
        if not _math.isfinite(total):
            raise ValueError("edge_weights must be finite")
        buffer.append(weight)
    return buffer


def _node_weight_buffer(graph, values):
    """One finite, non-negative weight per vertex."""
    if values is None:
        return None
    values = graph.vs[values] if isinstance(values, str) else list(values)
    if len(values) != graph.vcount():
        raise ValueError("node_weights must have one value per vertex")
    buffer = _array("d")
    for value in values:
        if isinstance(value, bool):
            raise ValueError("node_weights must be real numbers")
        weight = float(value)
        if not _math.isfinite(weight) or weight < 0:
            raise ValueError("node_weights must be finite and non-negative")
        buffer.append(weight)
    return buffer


def community_hedonic(
    graph,
    *,
    resolution=None,
    max_memberships=1,
    initial_membership=None,
    allow_isolation=False,
    edge_weights=None,
    node_weights=None,
    max_total_communities=None,
    n_communities=None,
    seed=None,
    rng=None,
):
    """Best-response local moving of the CPM hedonic game on ``graph``, until
    every vertex has been visited without a certainly improving move since
    the last move. Every accepted move raises the exact potential (its gain
    beats a rigorous bound of its rounding error), and the result carries
    ``equilibrium_slack``: no unilateral deviation raises a vertex's exact
    payoff by more than slack x |W| (0 when the arithmetic is exact, e.g.
    unit weights at resolution 0, 1/2 or 1: an exact Nash equilibrium).

    resolution: CPM gamma; the graph density if None.
    max_memberships: 1 for a partition; M > 1 for a cover with at most M
        communities per vertex; -1 for no practical cap (the vertex count).
    initial_membership: start state, flat labels or per-vertex rows; None
        starts from singletons (or, under a count limit, v mod K). Labels are
        non-negative integers contiguous from 0 (igraph's convention), rows
        are non-empty, flat and nested rows are not mixed, and a row holds at
        most max_memberships distinct labels (a repeated label counts once).
    allow_isolation: may a vertex move to a new, empty community.
    edge_weights, node_weights: sequences or attribute names; None for ones.
    max_total_communities, n_communities: at most / exactly K occupied
        communities, enforced at every move.
    seed: draw the visiting order from random.Random(seed); rng: any object
        with getrandbits() instead; neither: default_rng().
    """
    if graph.is_directed():
        raise ValueError("community_hedonic requires an undirected graph")
    n = graph.vcount()
    if max_memberships == -1:
        max_memberships = n
    if not isinstance(max_memberships, int) or isinstance(max_memberships, bool) \
            or max_memberships < 1:
        raise ValueError("max_memberships must be an integer >= 1, or -1")
    if seed is not None and rng is not None:
        raise ValueError("pass seed or rng, not both")
    if seed is not None:
        rng = _random.Random(seed)
    elif rng is None:
        rng = default_rng()
    if resolution is None:
        res = graph.density()
        if not _math.isfinite(res):
            raise ValueError("graph density is not a finite resolution; pass resolution= explicitly")
    else:
        res = float(resolution)
        if not _math.isfinite(res):
            raise ValueError("resolution must be a finite real number")

    edge_buffer = _edge_weight_buffer(graph, edge_weights)
    node_buffer = _node_weight_buffer(graph, node_weights)
    overlapping = max_memberships > 1
    # The C side reads the edges straight from the igraph graph, checks the
    # start (see the docstring) and builds the result lists.
    out = _detect(
        graph,
        rng,
        res,
        max_memberships=max_memberships,
        start=initial_membership,
        allow_isolation=bool(allow_isolation),
        edge_weights=edge_buffer,
        node_weights=node_buffer,
        max_total_communities=max_total_communities,
        n_communities=n_communities,
    )

    try:
        from igraph import VertexClustering, VertexCover
    except ImportError:  # pragma: no cover - igraph is optional
        return out

    if not overlapping:
        membership, nb_clusters, quality, slack = out
        modularity_params = {"resolution": res}
        if edge_weights is not None:
            modularity_params["weights"] = edge_weights
        return VertexClustering(graph, membership,
                                params={"quality": quality, "equilibrium_slack": slack},
                                modularity_params=modularity_params)
    rows, clusters, nb_clusters, quality, slack = out
    result = VertexCover(graph, clusters)
    result._params = {"quality": quality, "equilibrium_slack": slack}
    # The per-vertex rows, so callers need not rebuild them from the clusters.
    result._hedonic_rows = rows
    return result
