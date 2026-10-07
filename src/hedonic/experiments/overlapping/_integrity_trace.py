"""Bounded native accepted-move and token-projection trace fixture."""

from __future__ import annotations

import math
import random
from typing import Any

import igraph as ig

from hedonic.experiments.overlapping._integrity_cases import (
    _nested_memberships,
    _native_quality,
    _scale_tolerance,
    guard_trace_violations,
    trace_requires_label_counts,
)


def _debug_trace_fixture() -> dict[str, Any]:
    graph = ig.Graph(
        n=11,
        edges=[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (3, 4),
            (5, 6),
            (5, 7),
            (5, 8),
            (5, 9),
            (6, 7),
            (6, 8),
            (6, 9),
            (7, 8),
            (7, 9),
            (8, 9),
            (10, 0),
            (10, 1),
            (10, 2),
            (10, 5),
            (10, 6),
            (10, 7),
        ],
    )
    context = {
        "seed": 20260912,
        "n": graph.vcount(),
        "edges": [list(edge) for edge in graph.get_edgelist()],
        "resolution": 0.2,
        "max_memberships": 3,
        "n_iterations": 2,
        "allow_isolation": True,
        "local_move_only": False,
    }
    ig.set_random_number_generator(random.Random(context["seed"]))
    result = graph.community_leiden(
        objective_function="CPM",
        resolution=context["resolution"],
        max_memberships=context["max_memberships"],
        n_iterations=context["n_iterations"],
        allow_isolation=context["allow_isolation"],
        local_move_only=context["local_move_only"],
        debug_trace=True,
    )
    trace = (getattr(result, "_params", None) or {}).get("debug_trace")
    violations: list[str] = []
    if not isinstance(trace, dict):
        return {
            "name": "native_accepted_move_and_projection_trace",
            "context": context,
            "available": False,
            "violations": ["missing_debug_trace"],
            "ok": False,
        }
    moves = trace.get("moves") or []
    projections = trace.get("projections") or []
    if not moves:
        violations.append("no_accepted_moves")
    if not projections:
        violations.append("no_projection_checkpoints")
    violations.extend(guard_trace_violations(
        projections, n_vertices=graph.vcount(),
        require_label_counts=trace_requires_label_counts(),
    ))
    for move in moves:
        if float(move["predicted_delta"]) <= 0.0:
            violations.append("nonpositive_predicted_move_delta")
        if float(move["direct_delta"]) <= 0.0:
            violations.append("nonpositive_direct_move_delta")
        if float(move["abs_error"]) > float(move["tolerance"]):
            violations.append("accepted_move_delta_mismatch")
        if float(move["quality_after"]) < float(move["quality_before"]):
            violations.append("accepted_move_direct_decrease")
    for projection in projections:
        identity_error = abs(
            float(projection["original_unnormalized"])
            - float(projection["token_initial_unnormalized"])
        )
        if not math.isclose(
            identity_error,
            float(projection["token_identity_abs_error"]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            violations.append("token_identity_error_misreported")
        if identity_error > _scale_tolerance(
            float(projection["original_unnormalized"]),
            float(projection["token_initial_unnormalized"]),
        ):
            violations.append("token_unnormalized_identity_mismatch")
        if float(projection["quality_committed"]) < float(
            projection["quality_before"]
        ) - _scale_tolerance(
            float(projection["quality_committed"]),
            float(projection["quality_before"]),
        ):
            violations.append("projection_guard_quality_decrease")
    if not any(int(row["collision_count"]) > 0 for row in projections):
        violations.append("no_projection_collision")
    if not any(bool(row["accepted"]) for row in projections):
        violations.append("no_accepted_projection")
    if not any(not bool(row["accepted"]) for row in projections):
        violations.append("no_restored_projection")
    if not any(
        abs(float(row["original_weight"]) - float(row["token_weight"]))
        > _scale_tolerance(float(row["original_weight"]), float(row["token_weight"]))
        for row in projections
    ):
        violations.append("normalization_denominators_not_distinguished")
    return {
        "name": "native_accepted_move_and_projection_trace",
        "context": context,
        "available": True,
        "final_memberships": _nested_memberships(result, 3),
        "final_quality": _native_quality(result),
        "move_count": len(moves),
        "projection_count": len(projections),
        "guard_label_counts_verified": bool(projections) and all(
            "labels_local" in row and "labels_proposed" in row for row in projections
        ) and not guard_trace_violations(
            projections, n_vertices=graph.vcount(), require_label_counts=True
        ),
        "max_move_delta_error": max(
            (float(row["abs_error"]) for row in moves), default=0.0
        ),
        "max_token_identity_error": max(
            (float(row["token_identity_abs_error"]) for row in projections),
            default=0.0,
        ),
        "collision_count": sum(int(row["collision_count"]) for row in projections),
        "trace": trace,
        "violations": sorted(set(violations)),
        "ok": not violations,
    }
