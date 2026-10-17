"""Detector orchestration for :class:`hedonic.Game`.

Every call runs best-response local moving in this package's C core
(:mod:`hedonic._native`) until no vertex has an improving move, for partitions
and covers alike.
"""

from __future__ import annotations

import math
import random as _random
import sys
from numbers import Integral, Real

import numpy as np

from .. import _native
from ..utils import sample_uniform_ints
from .helpers import (
    ARITHMETIC_BOUND,
    check_start_counts,
    node_weight_values as _node_weight_values,
    runtime_versions,
    sequential_sum,
    weight_vector,
)


def _integer(value) -> bool:
    """Whether ``value`` is an integer (Python or NumPy), not a bool."""
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _positive_or_none(name: str, value) -> int | None:
    if value is None:
        return None
    if not _integer(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _preferred_labels(game, rows, cap, resolution, edge_weights, node_weights):
    """Project a start to the cap: each vertex keeps its ``cap`` preferred labels.

    A vertex with more than ``cap`` labels ranks them by the payoff of
    belonging to each one alone, evaluated on the start as given: the weight
    of its edges to the label's other members less ``resolution`` times its
    weight times their total weight (the CPM payoff of that community in the
    disjoint game). Ties go to the larger community, then the smaller label.
    Every vertex chooses against the same start, so two neighbours can keep
    different labels of a community they shared; the detector then moves
    from the projected state to an equilibrium. Labels nobody keeps are
    dropped and the rest renumbered in order. Returns the rows and the number
    of vertices that had to choose.
    """
    weights = node_weights if node_weights is not None else [1.0] * game.vcount()
    mass: dict[int, float] = {}
    size: dict[int, int] = {}
    for vertex, row in enumerate(rows):
        for label in row:
            mass[label] = mass.get(label, 0.0) + weights[vertex]
            size[label] = size.get(label, 0) + 1
    shared: dict[int, dict[int, float]] = {
        vertex: dict.fromkeys(row, 0.0)
        for vertex, row in enumerate(rows)
        if len(row) > cap
    }
    for index, (a, b) in enumerate(game.get_edgelist()):
        weight = 1.0 if edge_weights is None else float(edge_weights[index])
        for x, y in ((a, b), (b, a)):
            gains = shared.get(x)
            if gains is not None:
                for label in rows[y]:
                    if label in gains:
                        gains[label] += weight
    kept_rows = []
    for vertex, row in enumerate(rows):
        gains = shared.get(vertex)
        if gains is None:
            kept_rows.append(row)
            continue
        own = weights[vertex]
        ranked = sorted(
            row,
            key=lambda label: (
                -(gains[label] - resolution * own * (mass[label] - own)),
                -size[label],
                label,
            ),
        )
        kept_rows.append(ranked[:cap])
    kept = {label: i for i, label in enumerate(sorted({l for row in kept_rows for l in row}))}
    return [[kept[label] for label in row] for row in kept_rows], len(shared)


def community_hedonic(
    game,
    initial_membership: list[int] | list[list[int]] | None = None,
    *,
    max_communities: int | None = None,
    max_memberships: int = 1,
    resolution: float | None = None,
    allow_isolation: bool = False,
    edge_weights=None,
    seed: int | None = None,
    node_weights=None,
    max_total_communities: int | None = None,
    n_communities: int | None = None,
):
    """Run the detector for ``game`` (see :meth:`hedonic.Game.community_hedonic`)."""
    if game.vcount() < 1:
        raise ValueError("community_hedonic requires a nonempty graph")
    if game.is_directed():
        raise ValueError(
            "community_hedonic requires an undirected graph; "
            "project directed data before calling the detector"
        )
    if game.ecount() and any(game.is_loop()):
        raise ValueError(
            "community_hedonic requires a graph without self-loops (a loop "
            "ties a vertex to no one else); remove them with "
            "g.simplify(multiple=False, loops=True)"
        )
    if not _integer(max_memberships) or (max_memberships < 1 and max_memberships != -1):
        raise ValueError("max_memberships must be an integer >= 1, or -1")
    requested_max_memberships = max_memberships = int(max_memberships)
    if max_memberships == -1 or max_memberships > game.vcount():
        # No vertex can hold more labels than there are vertices.
        max_memberships = game.vcount()
    overlapping = max_memberships > 1
    max_total_communities = _positive_or_none("max_total_communities", max_total_communities)
    n_communities = _positive_or_none("n_communities", n_communities)
    if max_communities is not None:
        max_communities = _positive_or_none("max_communities", max_communities)
        if max_communities > 2**63:
            raise ValueError("max_communities must be at most 2**63")
    if (
        max_total_communities is not None
        and n_communities is not None
        and n_communities > max_total_communities
    ):
        raise ValueError("n_communities must not exceed max_total_communities")
    # A state occupies at most n x max_memberships communities.
    capacity = game.vcount() * max_memberships
    if n_communities is not None and n_communities > capacity:
        if not overlapping:
            bound = f"the vertex count ({capacity})"
        else:
            bound = (
                f"the vertex count times max_memberships ({game.vcount()} x "
                f"{max_memberships} = {capacity}), the most communities a cover "
                "can occupy"
                + ("" if requested_max_memberships == max_memberships else
                   f" (max_memberships={requested_max_memberships} is capped at "
                   "the vertex count)")
            )
        raise ValueError(f"n_communities={n_communities} exceeds {bound}")
    # A larger limit binds nothing; the core takes 64-bit counts.
    native_max_total = (
        None if max_total_communities is None else min(max_total_communities, capacity)
    )
    count_constrained = max_total_communities is not None or n_communities is not None
    if seed is not None:
        if not _integer(seed):
            raise ValueError("seed must be an integer or None")
        seed = int(seed)
    if not isinstance(allow_isolation, (bool, np.bool_)):
        raise ValueError("allow_isolation must be True or False")
    allow_isolation = bool(allow_isolation)
    if resolution is not None:
        if not isinstance(resolution, Real) or isinstance(resolution, (bool, np.bool_)):
            raise ValueError("resolution must be a finite real number or None")
        try:
            res = float(resolution)
        except OverflowError:
            res = math.inf
        if not math.isfinite(res):
            raise ValueError("resolution must be a finite real number")

    # Read the weights once, as float64 arrays (an iterator would be empty
    # the second time); a cover takes non-negative edge weights only.
    if edge_weights is not None:
        edge_weights = weight_vector(
            game, edge_weights, "edge_weights", per_vertex=False, nonnegative=overlapping
        )
    if game.ecount() == 0:
        weight_total = 0.0
    elif edge_weights is None:
        weight_total = float(game.ecount())
    else:
        weight_total = sequential_sum(edge_weights)
    if game.ecount() == 0:
        raise ValueError(
            f"community_hedonic needs a graph with edges (this one has "
            f"{game.vcount()} vertices and none): the quality, the potential "
            "divided by the total edge weight W, is undefined when W = 0"
        )
    if weight_total <= 0:
        raise ValueError(
            "community_hedonic requires a positive total edge weight W "
            f"(here W = {weight_total:g}); the quality, the potential divided "
            "by W, is undefined otherwise"
        )
    if weight_total < sys.float_info.min:
        raise ValueError(
            f"the total edge weight W = {weight_total:g} is below the smallest "
            "normal double (about 2.2e-308); scale the weights up (the game is "
            "unchanged when every weight and the resolution are scaled by the "
            "same factor)"
        )
    node_weight_values = _node_weight_values(game, node_weights)

    if resolution is None:
        # The weighted density: W over the number of vertex pairs (the
        # density of a simple unweighted graph).
        n = game.vcount()
        res = weight_total / (n * (n - 1) / 2) if n > 1 else 0.0
    # Every crowding sum is at most |resolution| x N^2 x max_memberships for
    # the total vertex weight N (the core's own check, named here).
    with np.errstate(over="ignore"):
        mass = (
            float(game.vcount()) if node_weight_values is None
            else float(node_weight_values.sum())
        )
    scale = max(mass, 1.0)
    if not abs(res) <= ARITHMETIC_BOUND / scale / scale / max_memberships:
        raise ValueError(
            "the crowding term would overflow: |resolution| x N^2 x "
            f"max_memberships, with |resolution| = {abs(res):g}, the total "
            f"vertex weight N = {mass:g} and max_memberships = {max_memberships}, "
            f"must stay below about {ARITHMETIC_BOUND:.3g}"
        )

    projected = None
    start_seed_used = None
    candidate = initial_membership
    if candidate is not None:
        source = "initial_membership"
        try:
            rows = game._normalize_membership(
                candidate, collapse_duplicate_labels=True, renumber_labels=True,
                name=source,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if message.startswith(source + " "):
                message = message[len(source) + 1:]
            raise ValueError(f"Invalid {source}: {message}") from exc
        if any(len(row) > max_memberships for row in rows):
            rows, projected = _preferred_labels(
                game, rows, max_memberships, res, edge_weights, node_weight_values
            )
        check_start_counts(rows, source, max_total_communities, n_communities, projected)
        membership_vector = rows if overlapping else [row[0] for row in rows]
        max_communities_effect = "none: a start state was supplied"
    elif count_constrained:
        # The native layer builds the deterministic feasible start
        # (vertex v in community v mod K) for count-constrained calls.
        membership_vector = None
        max_communities_effect = "none: count limits select the default start"
    elif overlapping:
        membership_vector = [[v] for v in range(game.vcount())]
        max_communities_effect = "none: overlapping singleton start"
    elif max_communities is None:
        membership_vector = list(range(game.vcount()))
        max_communities_effect = "none: disjoint singleton start"
    else:
        # The start is drawn from the call's seed (its absolute value, as
        # random.Random reads an integer seed), or, without one, from a
        # seed drawn from the generator the visiting order uses.
        start_seed = (
            abs(seed) if seed is not None
            else _native.default_rng().randint(0, 2**63 - 1)
        )
        membership_vector = sample_uniform_ints(
            game.vcount(), max_communities - 1, start_seed
        ).tolist()
        start_seed_used = start_seed
        # A draw can leave labels unused, and the detector takes labels
        # contiguous from 0: renumber the drawn labels in order (the
        # identity when every label was drawn).
        drawn = sorted(set(membership_vector))
        if len(drawn) != max_communities:
            index = {label: i for i, label in enumerate(drawn)}
            membership_vector = [index[label] for label in membership_vector]
        max_communities_effect = "random disjoint start"

    result = _native.community_hedonic(
        game,
        resolution=res,
        max_memberships=max_memberships,
        initial_membership=membership_vector,
        allow_isolation=allow_isolation,
        edge_weights=edge_weights,
        node_weights=node_weight_values,
        max_total_communities=native_max_total,
        n_communities=n_communities,
        seed=seed,
    )

    # A partition carries its quality as an attribute (igraph copies the
    # params of a VertexClustering); a cover keeps it in ``_params``.
    params = getattr(result, "_params", None)
    if isinstance(params, dict) and "quality" in params:
        setattr(result, "_hedonic_native_quality", params["quality"])
        slack = params.get("equilibrium_slack")
    else:
        if getattr(result, "quality", None) is not None:
            setattr(result, "_hedonic_native_quality", result.quality)
        slack = getattr(result, "equilibrium_slack", None)
    # The native cover carries its per-vertex rows; taking them saves
    # rebuilding them from the clusters (result.membership).
    native_rows = result.__dict__.pop("_hedonic_rows", None)
    if not overlapping:
        returned_rows = [[label] for label in result.membership]
    elif native_rows is not None:
        returned_rows = native_rows
    else:
        returned_rows = [[int(label) for label in labels] for labels in result.membership]

    occupied = len({label for row in returned_rows for label in row})
    setattr(
        result,
        "_hedonic_provenance",
        {
            "versions": runtime_versions(),
            "resolution": res,
            "max_memberships": requested_max_memberships,
            "effective_max_memberships": max_memberships,
            "count_constraint": (
                "exact" if n_communities is not None
                else "at_most" if max_total_communities is not None
                else "none"
            ),
            "max_total_communities": max_total_communities,
            "n_communities": n_communities,
            "max_communities": max_communities,
            "max_communities_effect": max_communities_effect,
            # The seed of a random max_communities start (drawn when the
            # call has none), so the start can be rebuilt.
            "start_seed": start_seed_used,
            "initialization": (
                "initial_membership, projected to max_memberships"
                if projected is not None
                else "initial_membership" if candidate is not None
                else "default"
            ),
            "projected_vertices": projected or 0,
            "occupied_communities": occupied,
            "allow_isolation": bool(allow_isolation),
            "edge_weighted": edge_weights is not None,
            "node_weighted": node_weight_values is not None,
            "total_edge_weight": weight_total,
            "seed": seed,
            "rng": (
                "seeded" if seed is not None
                else "random module" if _native.default_rng() is _random
                else "igraph generator"
            ),
            # The run ends when every vertex has been visited without a
            # certainly improving move since the last move: an equilibrium
            # over the declared action space (with a count limit, over the
            # feasible unilateral moves). No such deviation raises a
            # vertex's exact payoff by more than equilibrium_slack x |W|
            # (normalized like the quality; 0 when every computation was
            # exact, and NaN when W = 0).
            "equilibrium_notion": (
                "count_constrained" if count_constrained
                else "cap_constrained" if overlapping
                else "unconstrained_partition"
            ),
            "equilibrium_slack": slack,
        },
    )

    # Keep the returned state in the canonical representation; a later call
    # continues from it when given initial_membership=game.memberships.
    game._memberships = returned_rows

    return result
