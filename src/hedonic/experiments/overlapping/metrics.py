"""Cover metrics for the experiments: the library's :mod:`hedonic.metrics`, plus two experiment-only helpers.

Every score (F1, ONMI, Omega, one-to-one matching, diagnostics) is computed by
:mod:`hedonic.metrics`; this module re-exports it under the name experiments
import (private helpers included), so there is one implementation of each
metric. Only the overlapping CPM quality of a community list and the
binary join/leave equilibrium check of the small-graph smoke test, which the
library does not have, are defined here.
"""

from __future__ import annotations

from collections import defaultdict

from hedonic.metrics import *  # noqa: F401,F403 - the library's metrics are this module's
from hedonic.metrics import (  # noqa: F401 - private helpers experiments and their tests use
    GAME_EVALUATION_METRICS,
    MAX_DENSE_MATCHING_CELLS,
    SCORE_DEFINITION,
    SCORE_DEFINITION_VERSION,
    MatchingWeight,
    SingletonMode,
    _best_match_scores,
    _best_match_scores_from_intersections,
    _binary_entropy,
    _categorical_entropy,
    _check_count,
    _check_vertices,
    _cover_sets,
    _is_label,
    _node_metrics_from_matches,
    _one_to_one_matches,
    _one_to_one_metrics_from_matches,
    _pair_conditional_entropy,
    _pair_scores,
    _size_weighted_f1_from_scores,
    _sparse_best_conditional_entropy,
    _symmetric_metrics_from_scores,
)


def quality_overlapping_cpm(
    g,
    cover: list[list[int]],
    resolution: float,
    weights: list[float] | None = None,
) -> float:
    """Overlapping CPM quality Q = (1/2m) Σ_c [e_c − γ · C(N_c, 2)]."""
    if weights is None:
        w = [1.0] * g.ecount()
    else:
        w = list(weights)

    total_w = sum(w)
    if total_w == 0:
        return 0.0

    # Accumulate internal edge weight through memberships.  This is equivalent
    # to the former community-by-edge scan but avoids O(|C| * m) work.
    community_sets = [set(members) for members in cover]
    vertex_communities: dict[int, set[int]] = defaultdict(set)
    for community_index, members in enumerate(community_sets):
        for vertex in members:
            vertex_communities[vertex].add(community_index)
    internal_weights = [0.0] * len(community_sets)
    for edge_index, (source, target) in enumerate(g.get_edgelist()):
        for community_index in vertex_communities[source] & vertex_communities[target]:
            internal_weights[community_index] += w[edge_index]

    quality = sum(
        internal_weights[community_index]
        - resolution * len(members) * (len(members) - 1) / 2.0
        for community_index, members in enumerate(community_sets)
        if members
    )

    return quality / (2.0 * total_w)


def in_equilibrium_overlapping(
    g,
    cover: list[list[int]],
    resolution: float,
) -> bool:
    """Nash check for binary join/leave overlapping hedonic incentives."""
    n = g.vcount()
    v_comms: list[set[int]] = [set() for _ in range(n)]
    comm_sets = []
    for c_idx, members in enumerate(cover):
        ms = set(members)
        comm_sets.append(ms)
        for v in members:
            v_comms[v].add(c_idx)

    for v in range(n):
        deg_vc: dict[int, float] = defaultdict(float)
        for u in g.neighbors(v):
            for c in v_comms[u]:
                if u != v:
                    deg_vc[c] += 1.0

        for c_idx, ms in enumerate(comm_sets):
            Nc = len(ms)
            ew = deg_vc.get(c_idx, 0.0)
            if c_idx in v_comms[v]:
                if ew - resolution * (Nc - 1) < 0:
                    return False
            else:
                if ew - resolution * Nc > 0:
                    return False
    return True
