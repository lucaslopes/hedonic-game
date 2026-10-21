"""Detector orchestration for :class:`hedonic.Game` and :mod:`hedonic.arrays`.

Every call runs best-response local moving in this package's core
(:mod:`hedonic._native`) until no vertex has an improving move, for partitions
and covers alike. :func:`plan` is the part both callers share: every check of
the arguments, the resolution and the start state, over a :class:`Source`
(what the call reads of its graph). This module does not import igraph.
"""

from __future__ import annotations

import math
import random as _random
import sys
from dataclasses import dataclass
from itertools import chain
from numbers import Integral, Real
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np

from .. import _native
from ..utils import sample_uniform_ints
from .helpers import (
    ARITHMETIC_BOUND,
    check_start_counts,
    gc_paused,
    node_weight_values as _node_weight_values,
    runtime_versions,
    sequential_sum,
    weight_vector,
)
from .memberships import normalize_rows


def _integer(value) -> bool:
    """Whether ``value`` is an integer (Python or NumPy), not a bool."""
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _positive_or_none(name: str, value) -> int | None:
    if value is None:
        return None
    if not _integer(value) or value < 1:
        raise ValueError(f"{name} must be a positive integer or None")
    return int(value)


def _preferred_labels(n, edge_pairs, rows, cap, resolution, edge_weights, node_weights):
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
    weights = node_weights if node_weights is not None else [1.0] * n
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
    for index, (a, b) in enumerate(edge_pairs()):
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


class Source(NamedTuple):
    """What :func:`plan` reads of a graph: an ``igraph.Graph`` (``Game``) or arrays.

    ``edge_vector(values, nonnegative)`` and ``node_vector(values)`` turn the
    caller's weights into checked float64 arrays (``node_vector`` returns None
    for None); ``edge_pairs()`` lists the edges as ``(u, v)`` pairs in edge
    order (read only to project a start to the cap); ``caller`` names the
    function in messages.
    """

    n: int
    m: int
    edge_vector: Callable[[Any, bool], np.ndarray]
    node_vector: Callable[[Any], np.ndarray | None]
    edge_pairs: Callable[[], Iterable[tuple[int, int]]]
    caller: str


def game_source(game) -> Source:
    """The :class:`Source` of an ``igraph.Graph``: weights may be attribute names."""
    return Source(
        n=game.vcount(),
        m=game.ecount(),
        edge_vector=lambda values, nonnegative: weight_vector(
            game, values, "edge_weights", per_vertex=False, nonnegative=nonnegative
        ),
        node_vector=lambda values: _node_weight_values(game, values),
        edge_pairs=game.get_edgelist,
        caller="community_hedonic",
    )


@dataclass
class Plan:
    """A checked detector call: the core's arguments and what the provenance records."""

    resolution: float
    requested_max_memberships: int
    max_memberships: int
    overlapping: bool
    max_total_communities: int | None
    native_max_total: int | None
    n_communities: int | None
    count_constrained: bool
    max_communities: int | None
    max_communities_effect: str
    seed: int | None
    allow_isolation: bool
    edge_weights: np.ndarray | None
    node_weights: np.ndarray | None
    weight_total: float
    start: list | None
    start_given: bool
    projected: int | None
    start_seed: int | None

    def provenance(self, versions: dict, rng: str, occupied: int, slack) -> dict:
        """``_hedonic_provenance`` of the call whose result occupies ``occupied``
        communities with ``slack``."""
        return {
            "versions": versions,
            "resolution": self.resolution,
            "max_memberships": self.requested_max_memberships,
            "effective_max_memberships": self.max_memberships,
            "count_constraint": (
                "exact" if self.n_communities is not None
                else "at_most" if self.max_total_communities is not None
                else "none"
            ),
            "max_total_communities": self.max_total_communities,
            "n_communities": self.n_communities,
            "max_communities": self.max_communities,
            "max_communities_effect": self.max_communities_effect,
            # The seed of a random max_communities start (drawn when the
            # call has none), so the start can be rebuilt.
            "start_seed": self.start_seed,
            "initialization": (
                "initial_membership, projected to max_memberships"
                if self.projected is not None
                else "initial_membership" if self.start_given
                else "default"
            ),
            "projected_vertices": self.projected or 0,
            "occupied_communities": occupied,
            "allow_isolation": bool(self.allow_isolation),
            "edge_weighted": self.edge_weights is not None,
            "node_weighted": self.node_weights is not None,
            "total_edge_weight": self.weight_total,
            "seed": self.seed,
            "rng": rng,
            # The run ends when every vertex has been visited without a
            # certainly improving move since the last move: an equilibrium
            # over the declared action space (with a count limit, over the
            # feasible unilateral moves). No such deviation raises a
            # vertex's exact payoff by more than equilibrium_slack x |W|
            # (normalized like the quality; 0 when every computation was
            # exact, and NaN when W = 0).
            "equilibrium_notion": (
                "count_constrained" if self.count_constrained
                else "cap_constrained" if self.overlapping
                else "unconstrained_partition"
            ),
            "equilibrium_slack": slack,
        }


def rng_label(seed, default_rng) -> str:
    """The provenance's ``rng``: where the visiting order was drawn from."""
    return (
        "seeded" if seed is not None
        else "random module" if default_rng() is _random
        else "igraph generator"
    )


def plan(
    source: Source,
    initial_membership,
    *,
    max_communities,
    max_memberships,
    resolution,
    allow_isolation,
    edge_weights,
    seed,
    node_weights,
    max_total_communities,
    n_communities,
    default_rng: Callable[[], Any],
) -> Plan:
    """Check a detector call on a nonempty, undirected graph without
    self-loops (the caller checks those) and build its start.

    ``default_rng()`` is the generator of an unseeded call; a random
    ``max_communities`` start draws its seed from it.
    """
    n = source.n
    if not _integer(max_memberships) or (max_memberships < 1 and max_memberships != -1):
        raise ValueError("max_memberships must be an integer >= 1, or -1")
    requested_max_memberships = max_memberships = int(max_memberships)
    if max_memberships == -1 or max_memberships > n:
        # No vertex can hold more labels than there are vertices.
        max_memberships = n
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
    capacity = n * max_memberships
    if n_communities is not None and n_communities > capacity:
        if not overlapping:
            bound = f"the vertex count ({capacity})"
        else:
            bound = (
                f"the vertex count times max_memberships ({n} x "
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
        edge_weights = source.edge_vector(edge_weights, overlapping)
    if source.m == 0:
        weight_total = 0.0
    elif edge_weights is None:
        weight_total = float(source.m)
    else:
        weight_total = sequential_sum(edge_weights)
    if source.m == 0:
        raise ValueError(
            f"{source.caller} needs a graph with edges (this one has "
            f"{n} vertices and none): the quality, the potential "
            "divided by the total edge weight W, is undefined when W = 0"
        )
    if weight_total <= 0:
        raise ValueError(
            f"{source.caller} requires a positive total edge weight W "
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
    node_weight_values = source.node_vector(node_weights)

    if resolution is None:
        # The weighted density: W over the number of vertex pairs (the
        # density of a simple unweighted graph).
        res = weight_total / (n * (n - 1) / 2) if n > 1 else 0.0
    # Every crowding sum is at most |resolution| x N^2 x max_memberships for
    # the total vertex weight N (the core's own check, named here).
    with np.errstate(over="ignore"):
        mass = (
            float(n) if node_weight_values is None
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
        source_name = "initial_membership"
        try:
            rows = normalize_rows(
                candidate, n, collapse_duplicate_labels=True, renumber_labels=True,
                name=source_name,
            )
        except (TypeError, ValueError) as exc:
            message = str(exc)
            if message.startswith(source_name + " "):
                message = message[len(source_name) + 1:]
            raise ValueError(f"Invalid {source_name}: {message}") from exc
        if any(len(row) > max_memberships for row in rows):
            rows, projected = _preferred_labels(
                n, source.edge_pairs, rows, max_memberships, res, edge_weights,
                node_weight_values,
            )
        check_start_counts(rows, source_name, max_total_communities, n_communities, projected)
        membership_vector = rows if overlapping else [row[0] for row in rows]
        max_communities_effect = "none: a start state was supplied"
    elif count_constrained:
        # The native layer builds the deterministic feasible start
        # (vertex v in community v mod K) for count-constrained calls.
        membership_vector = None
        max_communities_effect = "none: count limits select the default start"
    elif overlapping:
        # The core starts from singletons when given no start (v mod n = v),
        # the state [[v] for v in range(n)] would give it, without n rows.
        membership_vector = None
        max_communities_effect = "none: overlapping singleton start"
    elif max_communities is None:
        # The same singletons as list(range(n)), built by the core.
        membership_vector = None
        max_communities_effect = "none: disjoint singleton start"
    else:
        # The start is drawn from the call's seed (its absolute value, as
        # random.Random reads an integer seed), or, without one, from a
        # seed drawn from the generator the visiting order uses.
        start_seed = (
            abs(seed) if seed is not None
            else default_rng().randint(0, 2**63 - 1)
        )
        membership_vector = sample_uniform_ints(n, max_communities - 1, start_seed).tolist()
        start_seed_used = start_seed
        # A draw can leave labels unused, and the detector takes labels
        # contiguous from 0: renumber the drawn labels in order (the
        # identity when every label was drawn).
        drawn = sorted(set(membership_vector))
        if len(drawn) != max_communities:
            index = {label: i for i, label in enumerate(drawn)}
            membership_vector = [index[label] for label in membership_vector]
        max_communities_effect = "random disjoint start"

    return Plan(
        resolution=res,
        requested_max_memberships=requested_max_memberships,
        max_memberships=max_memberships,
        overlapping=overlapping,
        max_total_communities=max_total_communities,
        native_max_total=native_max_total,
        n_communities=n_communities,
        count_constrained=count_constrained,
        max_communities=max_communities,
        max_communities_effect=max_communities_effect,
        seed=seed,
        allow_isolation=allow_isolation,
        edge_weights=edge_weights,
        node_weights=node_weight_values,
        weight_total=weight_total,
        start=membership_vector,
        start_given=candidate is not None,
        projected=projected,
        start_seed=start_seed_used,
    )


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
    # is_simple() is one pass in igraph; is_loop() builds a bool per edge, so
    # it is asked only of a graph with loops or multiple edges.
    if game.ecount() and not game.is_simple() and any(game.is_loop()):
        raise ValueError(
            "community_hedonic requires a graph without self-loops (a loop "
            "ties a vertex to no one else); remove them with "
            "g.simplify(multiple=False, loops=True)"
        )
    call = plan(
        game_source(game),
        initial_membership,
        max_communities=max_communities,
        max_memberships=max_memberships,
        resolution=resolution,
        allow_isolation=allow_isolation,
        edge_weights=edge_weights,
        seed=seed,
        node_weights=node_weights,
        max_total_communities=max_total_communities,
        n_communities=n_communities,
        default_rng=_native.default_rng,
    )

    result = _native.community_hedonic(
        game,
        resolution=call.resolution,
        max_memberships=call.max_memberships,
        initial_membership=call.start,
        allow_isolation=call.allow_isolation,
        edge_weights=call.edge_weights,
        node_weights=call.node_weights,
        max_total_communities=call.native_max_total,
        n_communities=call.n_communities,
        seed=call.seed,
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
    if not call.overlapping:
        membership = result.membership
        with gc_paused():
            returned_rows = [[label] for label in membership]
        occupied = len(set(membership))
    else:
        if native_rows is not None:
            returned_rows = native_rows
        else:
            with gc_paused():
                returned_rows = [[int(label) for label in labels] for labels in result.membership]
        occupied = len(set(chain.from_iterable(returned_rows)))
    setattr(
        result,
        "_hedonic_provenance",
        call.provenance(runtime_versions(), rng_label(call.seed, _native.default_rng),
                        occupied, slack),
    )

    # Keep the returned state in the canonical representation; a later call
    # continues from it when given initial_membership=game.memberships.
    game._memberships = returned_rows

    return result
