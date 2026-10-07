"""Native detector orchestration for :class:`hedonic.Game`."""

from __future__ import annotations

import math
import time

from ..utils import sample_uniform_ints
from .helpers import (
    _UNSET,
    _UnsetType,
    check_start_counts,
    membership_incidence_trace,
    membership_multiplicity_drops,
    node_weight_values as _node_weight_values,
    runtime_versions,
    seeded_igraph_rng,
    token_graph_preflight,
    total_edge_weight,
)


def community_hedonic(
    game,
    initial_membership: list[int] | list[list[int]] | None | _UnsetType = _UNSET,
    max_communities: int | None = None,
    max_memberships: int = 1,
    n_iterations: int = -1,
    resolution: float | None = None,
    allow_isolation: bool = False,
    local_move_only: bool = True,
    edge_weights=None,
    seed: int | None = None,
    beta: float = 0.01,
    *,
    node_weights=None,
    max_total_communities: int | None = None,
    n_communities: int | None = None,
    debug_trace: bool | str = False,
    phase_policy: str = "direct",
    algorithm_identity: str,
):
    """Run the native ``community_leiden`` call for ``game``."""
    if phase_policy not in ("direct", "disjoint_then_overlap"):
        raise ValueError(
            'phase_policy must be "direct" or "disjoint_then_overlap"'
        )
    if phase_policy == "disjoint_then_overlap":
        return disjoint_then_overlap(
            game,
            algorithm_identity=algorithm_identity,
            initial_membership=initial_membership,
            max_communities=max_communities,
            max_memberships=max_memberships,
            n_iterations=n_iterations,
            resolution=resolution,
            allow_isolation=allow_isolation,
            local_move_only=local_move_only,
            edge_weights=edge_weights,
            seed=seed,
            beta=beta,
            node_weights=node_weights,
            max_total_communities=max_total_communities,
            n_communities=n_communities,
            debug_trace=debug_trace,
        )
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
    for name, value in (
        ("max_total_communities", max_total_communities),
        ("n_communities", n_communities),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if (
        max_total_communities is not None
        and n_communities is not None
        and n_communities > max_total_communities
    ):
        raise ValueError("n_communities must not exceed max_total_communities")
    count_constrained = max_total_communities is not None or n_communities is not None
    if isinstance(debug_trace, str):
        if debug_trace not in ("full", "counters"):
            raise ValueError('debug_trace must be a boolean, "full" or "counters"')
        trace_level = debug_trace
    else:
        trace_level = "full" if debug_trace else None
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
        try:
            rows = game._normalize_membership(
                candidate,
                max_memberships=max_memberships,
                collapse_duplicate_labels=overlapping,
            )
        except (TypeError, ValueError) as exc:
            source = "memberships" if use_loaded else "initial_membership"
            raise ValueError(f"Invalid {source}: {exc}") from exc
        source = "memberships" if use_loaded else "initial_membership"
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
    else:
        if overlapping:
            # Singleton cover: vertex v alone in community v
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
            max_communities_effect = "random disjoint start"

    preflight = token_graph_preflight(
        game.vcount(),
        game.ecount(),
        max_memberships,
        membership_vector if overlapping else None,
    )
    if overlapping and membership_vector is None and n_communities is not None:
        # The native default gives each vertex one membership and adds
        # the remaining labels round-robin when exact K exceeds n.
        preflight["initial_token_count"] = max(game.vcount(), n_communities)
    if overlapping and not local_move_only and preflight["integer_overflow"]:
        raise OverflowError(
            "token-graph expansion would overflow a 64-bit igraph integer"
        )

    start_rows = (
        [list(row) for row in membership_vector]
        if overlapping and membership_vector is not None
        else None
    )
    if start_rows is None:
        start_trace = None
    elif candidate is not None:
        # Record the caller's original incidences, including repeated
        # labels, while native code receives the normalized unique rows.
        trace_rows = (
            candidate
            if isinstance(candidate[0], (list, tuple))
            else [[label] for label in candidate]
        )
        start_trace = membership_incidence_trace(trace_rows)
    else:
        start_trace = membership_incidence_trace(start_rows)

    leiden_kwargs = dict(
        initial_membership=membership_vector,
        n_iterations=n_iterations,
        resolution=res,
        allow_isolation=allow_isolation,
        local_move_only=local_move_only,
        weights=edge_weights,
        max_memberships=max_memberships,
        beta=beta,
    )
    # New native controls are passed only when used, so an old binding
    # keeps working for old calls and fails clearly for new ones.
    if node_weight_values is not None:
        leiden_kwargs["node_weights"] = node_weight_values
    if max_total_communities is not None:
        leiden_kwargs["max_total_communities"] = max_total_communities
    if n_communities is not None:
        leiden_kwargs["n_communities"] = n_communities
    if trace_level is not None:
        leiden_kwargs["debug_trace"] = trace_level
    try:
        if seed is None:
            result = game.community_leiden(**leiden_kwargs)
        else:
            with seeded_igraph_rng(seed):
                result = game.community_leiden(**leiden_kwargs)
    except TypeError as exc:
        if count_constrained and "unexpected keyword" in str(exc):
            raise RuntimeError(
                "the installed igraph binding does not support community-count "
                "constraints; lucas-igraph 1.0.0.5 or later is required"
            ) from exc
        raise

    setattr(result, "_hedonic_token_preflight", preflight)
    setattr(result, "_hedonic_original_edge_weight", weight_total)
    setattr(result, "_hedonic_algorithm_identity", algorithm_identity)
    native_quality = getattr(result, "_params", None)
    if isinstance(native_quality, dict) and "quality" in native_quality:
        setattr(result, "_hedonic_native_quality", native_quality["quality"])
    trace = native_quality.get("debug_trace") if isinstance(native_quality, dict) else None
    if trace_level is not None and isinstance(trace, dict):
        # The binding records the generator in effect; the facade adds the
        # seed that selected it and the algorithm identity.
        trace["seed_context"] = {
            "seed": seed,
            "rng": "random.Random(seed)" if seed is not None
            else "caller igraph generator",
        }
        trace["algorithm_identity"] = algorithm_identity
    returned_rows = (
        [[int(label) for label in labels] for labels in result.membership]
        if overlapping
        else [[int(label)] for label in result.membership]
    )
    if start_trace is not None:
        setattr(result, "_hedonic_start_membership_trace", start_trace)
        setattr(
            result,
            "_hedonic_returned_membership_trace",
            membership_incidence_trace(returned_rows),
        )
        setattr(
            result,
            "_hedonic_multiplicity_drops",
            membership_multiplicity_drops(start_rows, returned_rows),
        )

    if n_iterations < 0:
        # Keep the native membership vectors available to experiment
        # layers without changing the public return type.  Negative
        # iterations are the native equilibrium run; this is provenance
        # for that single native call, not a Python cleanup pass.
        setattr(result, "_hedonic_raw_memberships", [list(row) for row in returned_rows])

    occupied = len({label for row in returned_rows for label in row})
    setattr(
        result,
        "_hedonic_provenance",
        {
            "algorithm_identity": algorithm_identity,
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
            "debug_trace": trace_level,
            "phase_policy": "direct",
            "initialization": (
                "loaded memberships" if use_loaded and candidate is not None
                else "initial_membership" if candidate is not None
                else "default"
            ),
            "occupied_communities": occupied,
            "allow_isolation": bool(allow_isolation),
            "local_move_only": bool(local_move_only),
            "n_iterations": int(n_iterations),
            "beta": float(beta),
            "edge_weighted": edge_weights is not None,
            "node_weighted": node_weight_values is not None,
            "seed": seed,
            "rng": "seeded" if seed is not None else "caller igraph generator",
            # A negative budget ends with a complete local-moving sweep
            # that accepted no move: tolerance-level stationarity over the
            # native candidate set, which covers the declared action space
            # (Proposition "sparse candidate sets suffice"; with a count
            # constraint, the feasible unilateral actions). It is not an
            # independent audit; see experiments.overlapping.robustness.
            "native_certificate_sweep": n_iterations < 0,
            "equilibrium_notion": (
                "count_constrained" if count_constrained
                else "cap_constrained" if overlapping
                else "unconstrained_partition"
            ),
        },
    )

    # Persist the returned state in the canonical representation so a
    # subsequent call without ``initial_membership`` naturally continues
    # from this equilibrium.
    game._memberships = returned_rows

    return result


def disjoint_then_overlap(
    game,
    *,
    initial_membership,
    max_memberships,
    debug_trace,
    algorithm_identity,
    **shared,
):
    """``phase_policy="disjoint_then_overlap"``: a disjoint warm start.

    Stage 1 is an ordinary disjoint call; stage 2 is an ordinary
    overlapping call whose ``initial_membership`` is the stage-1
    partition. Both stages receive the same shared options, so the
    policy adds no hidden parameter; only the start of stage 2 changes.
    """
    if max_memberships == 1:
        raise ValueError(
            'phase_policy="disjoint_then_overlap" requires max_memberships > 1 '
            "(with max_memberships=1 the disjoint stage is the whole result)"
        )
    if initial_membership is not _UNSET and initial_membership is not None:
        raise ValueError(
            'phase_policy="disjoint_then_overlap" builds the overlapping start '
            "from its disjoint stage; do not also pass initial_membership"
        )
    started = time.perf_counter()
    partition = game.community_hedonic(
        initial_membership=None, max_memberships=1, debug_trace=False,
        phase_policy="direct", **shared,
    )
    stage_seconds = time.perf_counter() - started
    labels = [int(label) for label in partition.membership]
    shared.pop("max_communities")  # initialization-only; used by stage 1
    cover = game.community_hedonic(
        initial_membership=labels, max_memberships=max_memberships,
        debug_trace=debug_trace, phase_policy="direct", **shared,
    )
    stage_provenance = partition._hedonic_provenance
    setattr(cover, "_hedonic_disjoint_stage", partition)
    cover._hedonic_provenance["phase_policy"] = "disjoint_then_overlap"
    cover._hedonic_provenance["initialization"] = "disjoint_stage"
    cover._hedonic_provenance["disjoint_stage"] = {
        "occupied_communities": stage_provenance["occupied_communities"],
        "native_quality": getattr(partition, "_hedonic_native_quality", None),
        "max_communities_effect": stage_provenance["max_communities_effect"],
        "seconds": stage_seconds,
    }
    return cover
