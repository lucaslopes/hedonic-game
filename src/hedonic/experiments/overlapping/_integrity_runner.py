"""Graph-batch shard execution for the native integrity grid."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import igraph as ig

from hedonic.experiments.overlapping._integrity_cases import (
    PROTOCOL_NAME,
    SCHEMA_VERSION,
    _adjacency,
    _canonical_json,
    _disjoint_modes,
    _overlap_modes,
    _run_case,
    expected_calls_per_graph,
    graph_edges,
    graph_pairs,
    iter_disjoint_states,
    iter_q2_states,
)


MAX_FAILURE_DETAILS_PER_SHARD = 100


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _shard_specs(min_n: int, max_n: int, batch_graphs: int) -> list[dict[str, int]]:
    specs = []
    for n_vertices in range(min_n, max_n + 1):
        final_mask = 1 << len(graph_pairs(n_vertices))
        for first_mask in range(1, final_mask, batch_graphs):
            specs.append(
                {
                    "n": n_vertices,
                    "first_mask": first_mask,
                    "stop_mask": min(final_mask, first_mask + batch_graphs),
                }
            )
    return specs


def _shard_id(spec: dict[str, int]) -> str:
    return f"n{spec['n']}-m{spec['first_mask']:04x}-{spec['stop_mask'] - 1:04x}"


def _valid_shard(path: Path, run_identity: str, spec: dict[str, int]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("protocol_name") == PROTOCOL_NAME
        and payload.get("run_identity") == run_identity
        and payload.get("shard_id") == _shard_id(spec)
        and payload.get("status") == "complete"
        and payload.get("observed_calls") == payload.get("expected_calls")
    )


def _failure_rank(failure: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(failure.get("n", 10**9)),
        len(failure.get("edges") or []),
        int(failure.get("graph_mask", 10**9)),
        str(failure.get("initial_membership_sha256") or ""),
        str(failure.get("mode") or ""),
        float(failure.get("gamma", math.inf)),
    )


def _record_outcome(
    outcome: dict[str, Any],
    *,
    digest: Any,
    failures: list[dict[str, Any]],
    maxima: dict[str, float],
) -> tuple[int, float]:
    stable = outcome["stable"]
    digest.update(_canonical_json(stable))
    maxima["quality_error"] = max(
        maxima["quality_error"], float(stable.get("quality_error", 0.0))
    )
    maxima["initial_decrease"] = max(
        maxima["initial_decrease"], float(stable.get("initial_decrease", 0.0))
    )
    maxima["terminal_regret"] = max(
        maxima["terminal_regret"], float(stable.get("max_regret", 0.0))
    )
    failure = outcome["failure"]
    if failure is not None and len(failures) < MAX_FAILURE_DETAILS_PER_SHARD:
        failures.append(failure)
    return int(failure is not None), float(outcome["elapsed_seconds"])


def _run_shard(
    spec: dict[str, int],
    *,
    run_identity: str,
    gammas: Sequence[float],
    include_positive_budget: bool,
) -> dict[str, Any]:
    n_vertices = spec["n"]
    overlap_modes = _overlap_modes(include_positive_budget)
    disjoint_modes = _disjoint_modes()
    expected_per_graph = expected_calls_per_graph(
        n_vertices,
        gammas,
        include_positive_budget=include_positive_budget,
    )
    expected = (spec["stop_mask"] - spec["first_mask"]) * expected_per_graph
    digest = hashlib.sha256()
    failures: list[dict[str, Any]] = []
    failure_count = 0
    observed = 0
    total_case_seconds = 0.0
    maxima = {
        "quality_error": 0.0,
        "initial_decrease": 0.0,
        "terminal_regret": 0.0,
    }
    started = time.monotonic()

    for graph_mask in range(spec["first_mask"], spec["stop_mask"]):
        edges = graph_edges(n_vertices, graph_mask)
        graph = ig.Graph(n=n_vertices, edges=edges, directed=False)
        adjacency = _adjacency(n_vertices, edges)

        for state in iter_q2_states(n_vertices):
            for gamma in gammas:
                for mode in overlap_modes:
                    outcome = _run_case(
                        graph=graph,
                        adjacency=adjacency,
                        graph_mask=graph_mask,
                        initial_rows=state,
                        cap=2,
                        gamma=gamma,
                        mode=mode,
                    )
                    failed, elapsed = _record_outcome(
                        outcome,
                        digest=digest,
                        failures=failures,
                        maxima=maxima,
                    )
                    observed += 1
                    failure_count += failed
                    total_case_seconds += elapsed

        for flat_state in iter_disjoint_states(n_vertices):
            state = tuple((label,) for label in flat_state)
            for gamma in gammas:
                for mode in disjoint_modes:
                    outcome = _run_case(
                        graph=graph,
                        adjacency=adjacency,
                        graph_mask=graph_mask,
                        initial_rows=state,
                        cap=1,
                        gamma=gamma,
                        mode=mode,
                    )
                    failed, elapsed = _record_outcome(
                        outcome,
                        digest=digest,
                        failures=failures,
                        maxima=maxima,
                    )
                    observed += 1
                    failure_count += failed
                    total_case_seconds += elapsed

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_name": PROTOCOL_NAME,
        "run_identity": run_identity,
        "shard_id": _shard_id(spec),
        "status": "complete",
        "spec": spec,
        "expected_calls": expected,
        "observed_calls": observed,
        "failure_count": failure_count,
        "failures": failures,
        "failure_details_truncated": failure_count > len(failures),
        "maxima": maxima,
        "case_result_sha256": digest.hexdigest(),
        "case_seconds": total_case_seconds,
        "wall_seconds": time.monotonic() - started,
        "completed_utc": _utc_now(),
    }
