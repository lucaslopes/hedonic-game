"""Versioned boundary instances for exact rational DNN certificates."""

from __future__ import annotations

from typing import Any

from hedonic.experiments.overlapping._integrity_cases import (
    guard_trace_violations,
    trace_requires_label_counts,
)
from hedonic.experiments.overlapping.dnn_certificate import (
    LOCKED_INSTANCES as NUMERICAL_INSTANCES,
    LockedInstance,
)


RATIONAL_INSTANCES: dict[str, LockedInstance] = {
    **NUMERICAL_INSTANCES,
    "collision_path3": LockedInstance(
        name="collision_path3",
        n_vertices=3,
        edges=((0, 1), (0, 2)),
        edge_weights=(1.0, 1.0),
        vertex_weights=(1.0, 1.0, 1.0),
        resolution=0.0,
        max_labels=3,
        max_memberships=2,
    ),
    "duplicate_cap4": LockedInstance(
        name="duplicate_cap4",
        n_vertices=4,
        edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        edge_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        vertex_weights=(1.0, 1.0, 1.0, 1.0),
        resolution=0.2,
        max_labels=3,
        max_memberships=3,
    ),
}

INSTANCE_BOUNDARY_TAGS: dict[str, tuple[str, ...]] = {
    "path4": ("fractional_mass", "cap_two"),
    "bow_tie5": ("dense", "shared_articulation"),
    "weighted_bridge5": ("weighted", "fractional_mass", "three_label_bank"),
    "collision_path3": (
        "projection_collision",
        "fractional_mass",
        "three_active_initial_labels",
    ),
    "duplicate_cap4": (
        "duplicate_community_bodies",
        "cap_equals_label_bank",
        "fractional_sqrt3",
    ),
}


def run_projection_collision_boundary(
    instance: LockedInstance,
    *,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Run the opt-in raw binding trace for a registered collision boundary."""

    if "projection_collision" not in INSTANCE_BOUNDARY_TAGS[instance.name]:
        return None

    import inspect
    import random

    import igraph as ig

    if "debug_trace" not in inspect.signature(ig.Graph.community_leiden).parameters:
        return {
            "required_for_candidate_artifact": True,
            "available": False,
            "reason": "installed binding does not expose debug_trace",
            "ok": False,
        }
    graph = ig.Graph(n=instance.n_vertices, edges=list(instance.edges), directed=False)
    graph.es["weight"] = list(instance.edge_weights)
    initial = [[vertex] for vertex in range(instance.n_vertices)]
    ig.set_random_number_generator(random.Random(seed))
    result = graph.community_leiden(
        objective_function="CPM",
        weights="weight",
        resolution=instance.resolution,
        max_memberships=instance.max_memberships,
        initial_membership=initial,
        n_iterations=2,
        allow_isolation=True,
        local_move_only=False,
        debug_trace=True,
    )
    trace = result._params["debug_trace"]
    moves = trace["moves"]
    projections = trace["projections"]
    checks = {
        "accepted_moves_present": bool(moves),
        "move_deltas_match": all(row["abs_error"] <= row["tolerance"] for row in moves),
        "projection_checkpoints_present": bool(projections),
        "collision_observed": any(row["collision_count"] > 0 for row in projections),
        "token_identity_matches": all(
            row["token_identity_abs_error"] <= 1e-12 for row in projections
        ),
        # Every guard decision must follow the post-local rule of lucas-igraph
        # 1.0.0.5. Which branch this instance takes is recorded below but not
        # required: at resolution 0 local moving already reaches the optimum,
        # and the token proposal only ties it (it is kept because it merges
        # the duplicate labels).
        "guard_decisions_follow_post_local_rule": bool(projections)
        and not guard_trace_violations(
            projections, n_vertices=instance.n_vertices,
            require_label_counts=trace_requires_label_counts(),
        ),
    }
    observations = {
        "guard_accept_observed": any(row["accepted"] for row in projections),
        "guard_restore_observed": any(not row["accepted"] for row in projections),
        "guard_label_counts_verified": bool(projections) and not guard_trace_violations(
            projections, n_vertices=instance.n_vertices, require_label_counts=True
        ),
    }
    return {
        "required_for_candidate_artifact": True,
        "available": True,
        "seed": seed,
        "initial_memberships_by_vertex": initial,
        "final_memberships_by_vertex": [
            list(map(int, labels)) for labels in result.membership
        ],
        "collision_count": sum(int(row["collision_count"]) for row in projections),
        "trace": trace,
        "checks": checks,
        "observations": observations,
        "ok": all(checks.values()),
    }
