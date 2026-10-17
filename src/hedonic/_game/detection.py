"""Detector orchestration for :class:`hedonic.Game`.

Every call runs best-response local moving in this package's C core
(:mod:`hedonic._native`) until no vertex has an improving move, for partitions
and covers alike.
"""

from __future__ import annotations

import math
import random as _random

from .. import _native
from ..utils import sample_uniform_ints
from .helpers import (
    _UNSET,
    _UnsetType,
    check_start_counts,
    node_weight_values as _node_weight_values,
    runtime_versions,
    total_edge_weight,
)


def _positive_or_none(name: str, value) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer or None")


def community_hedonic(
    game,
    initial_membership: list[int] | list[list[int]] | None | _UnsetType = _UNSET,
    max_communities: int | None = None,
    max_memberships: int = 1,
    resolution: float | None = None,
    allow_isolation: bool = False,
    *,
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
    if (
        not isinstance(max_memberships, int)
        or isinstance(max_memberships, bool)
        or (max_memberships < 1 and max_memberships != -1)
    ):
        raise ValueError("max_memberships must be an integer >= 1, or -1")
    requested_max_memberships = max_memberships
    if max_memberships == -1:
        max_memberships = game.vcount()
    _positive_or_none("max_total_communities", max_total_communities)
    _positive_or_none("n_communities", n_communities)
    if (
        max_total_communities is not None
        and n_communities is not None
        and n_communities > max_total_communities
    ):
        raise ValueError("n_communities must not exceed max_total_communities")
    count_constrained = max_total_communities is not None or n_communities is not None
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("seed must be an integer or None")

    weight_total = total_edge_weight(game, edge_weights)
    if weight_total <= 0:
        raise ValueError(
            "community_hedonic requires positive total edge weight; "
            "the normalized potential is undefined when W=0"
        )
    node_weight_values = _node_weight_values(game, node_weights)

    overlapping = max_memberships > 1
    if n_communities is not None:
        capacity = game.vcount() * max_memberships
        if n_communities > capacity:
            raise ValueError(
                f"n_communities={n_communities} exceeds the "
                + ("vertex count" if not overlapping else
                   "vertex count times max_memberships")
                + f" ({capacity}) with max_memberships={requested_max_memberships}"
            )
    if resolution is None:
        res = game.density()
        if not math.isfinite(res):
            raise ValueError(
                "graph density is not a finite resolution; "
                "pass resolution= explicitly"
            )
    else:
        res = float(resolution)
        if not math.isfinite(res):
            raise ValueError("resolution must be a finite real number")

    # Omitted and explicit ``None`` are intentionally different.  An
    # omitted argument reuses a valid persistent state; explicit ``None``
    # always requests the default singleton initialization.
    use_loaded = initial_membership is _UNSET
    candidate = game.memberships if use_loaded else initial_membership
    if use_loaded and candidate == []:
        candidate = None

    if candidate is not None:
        source = "memberships" if use_loaded else "initial_membership"
        try:
            rows = game._normalize_membership(
                candidate,
                max_memberships=max_memberships,
                collapse_duplicate_labels=overlapping,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {source}: {exc}") from exc
        if not overlapping and any(len(row) != 1 for row in rows):
            raise ValueError(
                f"Invalid {source}: overlapping rows require max_memberships > 1"
            )
        check_start_counts(rows, source, max_total_communities, n_communities)
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
        if not isinstance(max_communities, int) or max_communities <= 0:
            raise ValueError(
                "max_communities must be a positive integer when provided"
            )
        membership_vector = sample_uniform_ints(
            game.vcount(), max_communities - 1, 42 if seed is None else seed
        ).tolist()
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
        max_total_communities=max_total_communities,
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
            "initialization": (
                "loaded memberships" if use_loaded and candidate is not None
                else "initial_membership" if candidate is not None
                else "default"
            ),
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

    # Persist the returned state in the canonical representation so a
    # subsequent call without ``initial_membership`` continues from this
    # equilibrium.
    game._memberships = returned_rows

    return result
