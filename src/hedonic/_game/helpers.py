"""Graph, token, trace, and runtime helpers for :class:`hedonic.Game`."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import igraph
from igraph import Graph

# lucas-igraph uses 64-bit ``igraph_integer_t`` for vertex/edge counts.
_IGRAPH_INTEGER_MAX = 2**63 - 1

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


def token_graph_preflight(
    n_vertices: int,
    n_edges: int,
    max_memberships: int,
    memberships: Sequence[Sequence[int]] | None = None,
) -> dict[str, int | bool]:
    """Estimate the frozen-multiplicity token graph before native expansion.

    Each original undirected edge ``uv`` becomes ``k_u k_v`` token edges.
    The bound uses the membership cap rather than a post-move cover, so it is
    a memory/integer preflight, not a prediction of the accepted-move count.
    Local-moving-only runs never materialize this graph.
    """
    cap = int(max_memberships)
    if n_vertices < 0 or n_edges < 0 or cap < 1:
        raise ValueError("n_vertices, n_edges must be nonnegative and cap >= 1")
    token_count = (
        sum(len(row) for row in memberships)
        if memberships is not None
        else n_vertices
    )
    token_count_bound = n_vertices * cap
    overflow = False
    if cap > _IGRAPH_INTEGER_MAX // max(cap, 1):
        overflow = True
        max_token_edges = _IGRAPH_INTEGER_MAX
    elif n_edges > _IGRAPH_INTEGER_MAX // (cap * cap):
        overflow = True
        max_token_edges = _IGRAPH_INTEGER_MAX
    else:
        max_token_edges = n_edges * cap * cap
    return {
        "n_vertices": int(n_vertices),
        "n_edges": int(n_edges),
        "max_memberships": cap,
        "initial_token_count": int(token_count),
        "token_count_bound": int(token_count_bound),
        "max_token_edges": int(max_token_edges),
        "estimated_token_edge_bytes": int(24 * max_token_edges),
        "integer_overflow": overflow,
    }


def membership_incidence_trace(
    memberships: Sequence[Sequence[int]],
) -> dict[str, int | bool | list]:
    """Count labelled incidences, intra-row duplicates, and collision lists.

    Native token projection reports collisions only as a boolean. This
    wrapper-side trace records the labelled state, including every
    ``(vertex, label, token_count)`` intra-row collision. It cannot recover
    token-graph collisions that the C ABI discards after Leiden merges.
    """
    duplicate_row_count = 0
    incidence_count = 0
    unique_incidence_count = 0
    collisions: list[dict[str, int]] = []
    multiplicities: list[int] = []
    for vertex, row in enumerate(memberships):
        labels = [int(label) for label in row]
        incidence_count += len(labels)
        unique_incidence_count += len(set(labels))
        multiplicities.append(len(set(labels)))
        counts: dict[int, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        if len(labels) != len(counts):
            duplicate_row_count += 1
        for label, count in sorted(counts.items()):
            if count > 1:
                collisions.append(
                    {"vertex": vertex, "label": label, "token_count": count}
                )
    return {
        "n_vertices": len(memberships),
        "incidence_count": incidence_count,
        "unique_incidence_count": unique_incidence_count,
        "duplicate_labels": duplicate_row_count > 0,
        "duplicate_row_count": duplicate_row_count,
        "collisions": collisions,
        "multiplicities": multiplicities,
    }


def membership_multiplicity_drops(
    start: Sequence[Sequence[int]],
    returned: Sequence[Sequence[int]],
) -> list[dict[str, int]]:
    """Rows whose unique-label count fell between start and return."""
    drops: list[dict[str, int]] = []
    for vertex, (start_row, returned_row) in enumerate(zip(start, returned, strict=True)):
        k_start = len({int(label) for label in start_row})
        k_returned = len({int(label) for label in returned_row})
        if k_returned < k_start:
            drops.append(
                {
                    "vertex": vertex,
                    "k_start": k_start,
                    "k_returned": k_returned,
                }
            )
    return drops


@contextmanager
def seeded_igraph_rng(seed) -> Iterator[None]:
    """Use ``random.Random(seed)`` as igraph's generator inside the block.

    Every stochastic native choice of ``community_leiden`` (vertex visiting
    order, refinement randomness, random ties of the multilevel phase) draws
    from igraph's generator, so seeding it makes a call a deterministic
    function of its inputs on a given build.  The previous generator is
    restored afterwards, including the C-level default. Bindings without
    ``igraph.get_random_number_generator`` cannot restore an arbitrary
    caller's generator and reject seeded calls before changing it.
    """
    getter = getattr(igraph, "get_random_number_generator", None)
    if getter is None:
        raise RuntimeError(
            "seeded calls require igraph.get_random_number_generator to restore "
            "the caller's generator; lucas-igraph 1.0.0.5 or later is required"
        )
    previous = getter()
    igraph.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        igraph.set_random_number_generator(previous)


def node_weight_values(graph: Graph, node_weights) -> list[float] | None:
    """Validate vertex weights for the unit-l2 CPM domain (finite, >= 0)."""
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
    conflicting option instead of surfacing a generic igraph error.
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
    """Versions of the packages that actually ran a detector call."""
    try:
        from importlib.metadata import version

        hedonic_version = version("hedonic")
    except Exception:  # pragma: no cover - metadata absent in a bare checkout
        hedonic_version = None
    native = getattr(igraph, "_igraph", None)
    return {
        "hedonic": hedonic_version,
        "python_igraph": getattr(igraph, "__version__", None),
        "igraph_c": getattr(native, "__igraph_version__", None),
    }
