"""Argument checks and runtime information for :class:`hedonic.Game`."""

from __future__ import annotations

import math

import igraph
from igraph import Graph


# ``None`` is a meaningful explicit value for ``community_hedonic`` (it asks
# for the default singleton start), so the omitted argument needs its own
# sentinel.  Keeping this private avoids exposing an implementation detail in
# the public API while still allowing a persistent ``Game.memberships`` state.
class _UnsetType:
    __slots__ = ()


_UNSET = _UnsetType()


def total_edge_weight(graph: Graph, edge_weights=None) -> float:
    """Return the undirected total edge weight ``W`` used by ``Φ_γ = Φ̃_γ / W``.

    ``edge_weights is None`` means every edge has weight one, matching the
    native call.  An explicit sequence or edge-attribute name is summed.
    """
    n_edges = int(graph.ecount())
    if n_edges == 0:
        return 0.0
    if edge_weights is None:
        return float(n_edges)
    if isinstance(edge_weights, str):
        values = graph.es[edge_weights]
    else:
        values = edge_weights
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
        if not math.isfinite(total):
            raise ValueError("edge_weights must be finite")
    if count != n_edges:
        raise ValueError("edge_weights must have one value per graph edge")
    return total


def node_weight_values(graph: Graph, node_weights) -> list[float] | None:
    """Validate vertex weights (finite, >= 0)."""
    if node_weights is None:
        return None
    values = graph.vs[node_weights] if isinstance(node_weights, str) else list(node_weights)
    if len(values) != graph.vcount():
        raise ValueError("node_weights must have one value per vertex")
    weights = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError("node_weights must be real numbers")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("node_weights must be finite and non-negative")
        weights.append(weight)
    return weights


def check_start_counts(
    rows: list[list[int]],
    source: str,
    max_total_communities: int | None,
    n_communities: int | None,
) -> None:
    """Reject a start state that violates a community-count limit.

    The native layer rejects such a start as well; checking here names the
    conflicting option instead of surfacing a generic error.
    """
    occupied = len({label for row in rows for label in row})
    if max_total_communities is not None and occupied > max_total_communities:
        raise ValueError(
            f"Invalid {source}: it occupies {occupied} communities, more than "
            f"max_total_communities={max_total_communities}"
        )
    if n_communities is not None and occupied != n_communities:
        raise ValueError(
            f"Invalid {source}: it occupies {occupied} communities, but "
            f"n_communities={n_communities} requires exactly that many"
        )


def runtime_versions() -> dict[str, str | None]:
    """Versions of the packages that ran a detector call."""
    try:
        from importlib.metadata import version

        hedonic_version = version("hedonic")
    except Exception:  # pragma: no cover - metadata absent in a bare checkout
        hedonic_version = None
    from .. import _core

    return {
        "hedonic": hedonic_version,
        "hedonic_core": _core.core_version,
        "python_igraph": getattr(igraph, "__version__", None),
    }
