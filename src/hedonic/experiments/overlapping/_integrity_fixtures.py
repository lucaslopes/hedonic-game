"""Fixed domain, representation, and normalization integrity fixtures."""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

import igraph as ig
import numpy as np

from hedonic.experiments.overlapping import unit_l2_oracle as oracle
from hedonic.experiments.overlapping._integrity_cases import (
    ABS_TOLERANCE,
    REL_TOLERANCE,
    SCHEMA_VERSION,
    _adjacency,
    _nested_memberships,
    _native_quality,
    _scale_tolerance,
    _validate_rows,
)
from hedonic.experiments.overlapping._integrity_trace import _debug_trace_fixture


def _positive_guard_fixture() -> dict[str, Any]:
    graph = ig.Graph(
        n=4,
        edges=[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
    )
    weights = [0.25, 2.0, 0.1, 1.0, 1.0]
    initial = [[1, 2], [0], [0, 1, 2], [0, 1, 2]]
    common = {
        "objective_function": "CPM",
        "weights": weights,
        "resolution": 0.5,
        "beta": 0.01,
        "max_memberships": 3,
        "initial_membership": initial,
        "allow_isolation": False,
        "local_move_only": False,
    }
    baseline = graph.community_leiden(n_iterations=0, **common)
    ig.set_random_number_generator(random.Random(1452719858))
    candidate = graph.community_leiden(n_iterations=1, **common)
    before = _native_quality(baseline)
    after = _native_quality(candidate)
    tolerance = _scale_tolerance(before, after)
    return {
        "name": "positive_budget_original_quality_guard",
        "before": before,
        "after": after,
        "delta": after - before,
        "restored_memberships": candidate.membership == baseline.membership,
        "tolerance": tolerance,
        "ok": after + tolerance >= before,
    }


def _canonical_bodies(rows: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    labels = sorted({int(label) for row in rows for label in row})
    return sorted(
        {
            tuple(vertex for vertex, row in enumerate(rows) if label in row)
            for label in labels
        }
    )


def _weighted_quality_fixture() -> dict[str, Any]:
    graph = ig.Graph(
        n=4,
        edges=[(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
    )
    edge_weights = [0.25, 2.0, 0.1, 1.0, 1.0]
    node_weights = np.asarray([1.0, 2.0, 0.5, 1.5])
    initial = [[1, 2], [0], [0, 1, 2], [0, 1, 2]]
    result = graph.community_leiden(
        objective_function="CPM",
        weights=edge_weights,
        node_weights=node_weights.tolist(),
        resolution=0.5,
        beta=0.01,
        max_memberships=3,
        initial_membership=initial,
        n_iterations=0,
        allow_isolation=False,
        local_move_only=False,
    )
    rows = _nested_memberships(result, 3)
    adjacency = np.zeros((4, 4), dtype=float)
    for (first, second), weight in zip(graph.get_edgelist(), edge_weights):
        adjacency[first, second] = adjacency[second, first] = weight
    native_quality = _native_quality(result)
    direct_quality = oracle.original_normalized_quality(
        adjacency,
        node_weights,
        0.5,
        rows,
        max(label for row in rows for label in row) + 1,
    )
    error = abs(native_quality - direct_quality)
    tolerance = _scale_tolerance(native_quality, direct_quality)
    return {
        "name": "weighted_heterogeneous_membership_normalization",
        "memberships": rows,
        "native_quality": native_quality,
        "direct_quality": direct_quality,
        "error": error,
        "tolerance": tolerance,
        "ok": not _validate_rows(rows, 3) and error <= tolerance,
    }


def _label_permutation_fixture() -> dict[str, Any]:
    graph = ig.Graph.Ring(5)
    original = [[0], [0, 1], [1], [0], [1]]
    permuted = [[1], [0, 1], [0], [1], [0]]
    common = {
        "objective_function": "CPM",
        "max_memberships": 2,
        "n_iterations": 0,
        "resolution": 0.3,
        "allow_isolation": False,
        "local_move_only": True,
    }
    first = graph.community_leiden(initial_membership=original, **common)
    second = graph.community_leiden(initial_membership=permuted, **common)
    first_rows = _nested_memberships(first, 2)
    second_rows = _nested_memberships(second, 2)
    first_quality = _native_quality(first)
    second_quality = _native_quality(second)
    tolerance = _scale_tolerance(first_quality, second_quality)
    return {
        "name": "label_permutation_objective_and_bodies",
        "first_memberships": first_rows,
        "second_memberships": second_rows,
        "first_bodies": _canonical_bodies(first_rows),
        "second_bodies": _canonical_bodies(second_rows),
        "first_quality": first_quality,
        "second_quality": second_quality,
        "tolerance": tolerance,
        "ok": (
            not _validate_rows(first_rows, 2)
            and not _validate_rows(second_rows, 2)
            and _canonical_bodies(first_rows) == _canonical_bodies(second_rows)
            and abs(first_quality - second_quality) <= tolerance
        ),
    }


def _duplicate_body_fixture() -> dict[str, Any]:
    graph = ig.Graph.Ring(4)
    initial = [[0, 1] for _ in range(4)]
    result = graph.community_leiden(
        objective_function="CPM",
        initial_membership=initial,
        max_memberships=2,
        n_iterations=0,
        resolution=0.5,
        allow_isolation=False,
        local_move_only=True,
    )
    rows = _nested_memberships(result, 2)
    projected = [[0] for _ in range(4)]
    adjacency = _adjacency(4, graph.get_edgelist())
    before = oracle.original_normalized_quality(adjacency, np.ones(4), 0.5, rows, 2)
    after = oracle.original_normalized_quality(adjacency, np.ones(4), 0.5, projected, 1)
    native_quality = _native_quality(result)
    tolerance = _scale_tolerance(before, after, native_quality)
    raw_bodies = [
        tuple(vertex for vertex, row in enumerate(rows) if label in row)
        for label in range(2)
    ]
    return {
        "name": "duplicate_community_body_projection",
        "native_memberships": rows,
        "projected_memberships": projected,
        "duplicate_bodies_present": len(set(raw_bodies)) < len(raw_bodies),
        "native_quality": native_quality,
        "direct_before": before,
        "direct_after": after,
        "tolerance": tolerance,
        "ok": (
            len(set(raw_bodies)) < len(raw_bodies)
            and abs(before - after) <= tolerance
            and abs(native_quality - before) <= tolerance
        ),
    }


def _negative_resolution_fixture() -> dict[str, Any]:
    graph = ig.Graph.Ring(4)
    initial = [[0], [0, 1], [1], [0]]
    result = graph.community_leiden(
        objective_function="CPM",
        initial_membership=initial,
        max_memberships=2,
        n_iterations=0,
        resolution=-0.1,
        allow_isolation=False,
        local_move_only=True,
    )
    rows = _nested_memberships(result, 2)
    adjacency = _adjacency(4, graph.get_edgelist())
    direct_quality = oracle.original_normalized_quality(
        adjacency, np.ones(4), -0.1, rows, 2
    )
    native_quality = _native_quality(result)
    tolerance = _scale_tolerance(native_quality, direct_quality)
    return {
        "name": "finite_negative_resolution_extension",
        "resolution": -0.1,
        "memberships": rows,
        "native_quality": native_quality,
        "direct_quality": direct_quality,
        "tolerance": tolerance,
        "ok": (
            math.isfinite(native_quality)
            and math.isfinite(direct_quality)
            and abs(native_quality - direct_quality) <= tolerance
        ),
    }


def _invalid_input_fixture() -> dict[str, dict[str, Any]]:
    graph = ig.Graph(n=2, edges=[(0, 1)])
    looped_graph = ig.Graph(n=2, edges=[(0, 0), (0, 1)])
    directed_graph = ig.Graph(n=2, edges=[(0, 1)], directed=True)
    edgeless_graph = ig.Graph(n=2)
    invalid = {
        "nan_resolution": lambda: graph.community_leiden(
            max_memberships=2, resolution=math.nan
        ),
        "negative_weight": lambda: graph.community_leiden(
            max_memberships=2, weights=[-1.0]
        ),
        "infinite_weight": lambda: graph.community_leiden(
            max_memberships=2, weights=[math.inf]
        ),
        "negative_node_weight": lambda: graph.community_leiden(
            max_memberships=2, node_weights=[1.0, -1.0]
        ),
        "nonfinite_node_weight": lambda: graph.community_leiden(
            max_memberships=2, node_weights=[1.0, math.inf]
        ),
        "nan_beta": lambda: graph.community_leiden(max_memberships=2, beta=math.nan),
        "negative_beta": lambda: graph.community_leiden(max_memberships=2, beta=-0.01),
        "cap_above_n": lambda: graph.community_leiden(max_memberships=3),
        "zero_cap": lambda: graph.community_leiden(max_memberships=0),
        "zero_total_weight": lambda: graph.community_leiden(
            max_memberships=2, weights=[0.0]
        ),
        "looped_graph": lambda: looped_graph.community_leiden(max_memberships=2),
        "directed_graph": lambda: directed_graph.community_leiden(max_memberships=2),
        "edgeless_graph": lambda: edgeless_graph.community_leiden(max_memberships=2),
    }
    results = {}
    for name, operation in invalid.items():
        try:
            operation()
        except Exception as error:
            results[name] = {
                "rejected": True,
                "exception": type(error).__name__,
                "message": str(error),
            }
        else:
            results[name] = {"rejected": False}
    return results


def _contract_fixtures(
    include_positive_budget: bool,
    *,
    require_domain_rejections: bool,
    require_debug_trace: bool,
) -> dict[str, Any]:
    from hedonic import Game
    from hedonic.experiments.overlapping.native_differential import ce1_records

    cases: dict[str, Any] = {}
    if include_positive_budget:
        cases["positive_guard"] = _positive_guard_fixture()
    if require_debug_trace:
        try:
            cases["debug_trace"] = _debug_trace_fixture()
        except Exception as error:
            cases["debug_trace"] = {
                "name": "native_accepted_move_and_projection_trace",
                "available": False,
                "exception": type(error).__name__,
                "message": str(error),
                "violations": ["debug_trace_exception"],
                "ok": False,
            }
    else:
        cases["debug_trace"] = {
            "name": "native_accepted_move_and_projection_trace",
            "available": False,
            "status": "not_requested",
            "violations": [],
            "ok": True,
        }
    invalid_results = _invalid_input_fixture()
    cases["invalid_inputs"] = invalid_results
    ce1 = ce1_records(seed=20260907)
    cases["complete_response_ce1"] = {
        "records": ce1,
        "ok": all(
            row["oracles_agree"]
            and row["escaped_trap"]
            and row["native_certified_false_on_trap"]
            for row in ce1
        ),
    }
    wrapper_graph = Game(ig.Graph.Ring(6))
    flat = [0, 0, 1, 1, 2, 2]
    nested = [[label] for label in flat]
    wrapper_args = {
        "max_memberships": 2,
        "n_iterations": -1,
        "resolution": 0.2,
        "allow_isolation": True,
        "local_move_only": True,
    }
    ig.set_random_number_generator(random.Random(918))
    flat_result = wrapper_graph.community_hedonic(
        initial_membership=flat, **wrapper_args
    )
    ig.set_random_number_generator(random.Random(918))
    nested_result = wrapper_graph.community_hedonic(
        initial_membership=nested, **wrapper_args
    )
    flat_rows = _nested_memberships(flat_result, 2)
    nested_rows = _nested_memberships(nested_result, 2)
    cases["flat_nested_wrapper_equivalence"] = {
        "flat": flat_rows,
        "nested": nested_rows,
        "ok": flat_rows == nested_rows,
    }
    duplicate_graph = ig.Graph.Ring(4)
    duplicate_result = duplicate_graph.community_leiden(
        objective_function="CPM",
        initial_membership=[[0, 0], [1], [2], [3]],
        max_memberships=2,
        n_iterations=-1,
        resolution=0.5,
        allow_isolation=True,
        local_move_only=True,
    )
    duplicate_rows = _nested_memberships(duplicate_result, 2)
    cases["duplicate_start_cleanup"] = {
        "final": duplicate_rows,
        "ok": not _validate_rows(duplicate_rows, 2),
    }
    cases["weighted_quality"] = _weighted_quality_fixture()
    cases["label_permutation"] = _label_permutation_fixture()
    cases["duplicate_body_projection"] = _duplicate_body_fixture()
    cases["negative_resolution_extension"] = _negative_resolution_fixture()
    expected_invalid = all(row["rejected"] for row in invalid_results.values())
    domain_ok = expected_invalid if require_domain_rejections else True
    positive_ok = not include_positive_budget or cases["positive_guard"]["ok"]
    complete_response_ok = cases["complete_response_ce1"]["ok"]
    representation_ok = (
        cases["flat_nested_wrapper_equivalence"]["ok"]
        and cases["duplicate_start_cleanup"]["ok"]
        and cases["weighted_quality"]["ok"]
        and cases["label_permutation"]["ok"]
        and cases["duplicate_body_projection"]["ok"]
        and cases["negative_resolution_extension"]["ok"]
    )
    trace_ok = cases["debug_trace"]["ok"]
    return {
        "schema_version": SCHEMA_VERSION,
        "tolerance": {"atol": ABS_TOLERANCE, "rtol": REL_TOLERANCE},
        "include_positive_budget": include_positive_budget,
        "domain_rejection_required": require_domain_rejections,
        "debug_trace_required": require_debug_trace,
        "cases": cases,
        "ok": (
            domain_ok
            and positive_ok
            and complete_response_ok
            and representation_ok
            and trace_ok
        ),
    }
