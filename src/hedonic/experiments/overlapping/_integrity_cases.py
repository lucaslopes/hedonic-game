"""Enumeration and single-case checks for the native integrity grid."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import igraph as ig
import numpy as np

from hedonic.experiments.overlapping import unit_l2_oracle as oracle


SCHEMA_VERSION = 1
PROTOCOL_NAME = "unit-l2-native-integrity-v1"
DEFAULT_GAMMAS = (0.0, 1.0)
ABS_TOLERANCE = 1e-10
REL_TOLERANCE = 1e-9

OVERLAP_NEGATIVE_MODES = (
    ("negative_local_closed", -1, True, False),
    ("negative_local_open", -1, True, True),
    ("negative_multilevel_closed", -1, False, False),
    ("negative_multilevel_open", -1, False, True),
)
OVERLAP_POSITIVE_MODES = (
    ("positive_multilevel_closed", 1, False, False),
    ("positive_multilevel_open", 1, False, True),
)
DISJOINT_MODES = OVERLAP_NEGATIVE_MODES
Q2_ACTIONS = ((0,), (1,), (0, 1))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def native_is_improvement(candidate: float, current: float) -> bool:
    """Mirror of the native ``igraph_i_leiden_overlap_is_improvement``.

    The overlapping mover accepts a candidate only
    when it exceeds the current value by more than 64 * epsilon * max(1, |a|,
    |b|); ties within that margin are rejected.
    """

    candidate, current = float(candidate), float(current)
    if math.isnan(candidate) or math.isnan(current):
        return False
    if not (math.isfinite(candidate) and math.isfinite(current)):
        return candidate > current
    margin = 64.0 * sys.float_info.epsilon * max(1.0, abs(candidate), abs(current))
    return candidate - current > margin


def trace_requires_label_counts() -> bool:
    """The 1.0.0.5 binding contract requires the extended native trace."""

    version = re.match(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)", ig.__version__)
    return bool(version and tuple(map(int, version.groups())) >= (1, 0, 0, 5))


def guard_decision_violations(
    projection: dict[str, Any],
    *,
    require_label_counts: bool = False,
    tie_budget_remaining: int | None = None,
) -> list[str]:
    """Check one projection checkpoint against the post-local guard rule.

    From lucas-igraph 1.0.0.5 each multilevel iteration compares its token
    proposal with the cover reached by that iteration's local moving. It keeps
    a proposal that improves on that cover beyond the native margin, or one
    within the margin that occupies fewer labels (token projection merges
    duplicate community bodies); otherwise it restores the local cover.
    (1.0.0.4 compared with the cover before the iteration, so it could keep a
    proposal worse than the local-moving state and discard the local
    improvement on rejection.) Legacy traces lack label counts and support
    only the quality check. New candidates require the count columns. With
    the remaining tie budget supplied, the acceptance decision is checked in
    both directions; a single row alone cannot tell whether it was exhausted.
    """

    projected = float(projection["quality_projected"])
    after_local = float(projection["quality_after_local"])
    committed = float(projection["quality_committed"])
    violations = []
    if not all(math.isfinite(value) for value in (projected, after_local, committed)):
        return ["projection_guard_nonfinite_quality"]
    counts = [projection.get(name) for name in ("labels_local", "labels_proposed")]
    has_counts = all(value is not None for value in counts)
    valid_counts = has_counts and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0 and int(value) == value
        for value in counts
    )
    if has_counts and not valid_counts:
        violations.append("projection_guard_invalid_label_counts")
    elif not has_counts and (require_label_counts or any(value is not None for value in counts)):
        violations.append("projection_guard_missing_label_counts")

    improves = native_is_improvement(projected, after_local)
    worsens = native_is_improvement(after_local, projected)
    accepted = bool(projection["accepted"])
    mismatch = (accepted and worsens) or (not accepted and improves)
    if valid_counts and not improves and not worsens:
        reduces_labels = counts[1] < counts[0]
        if accepted:
            mismatch = not reduces_labels or tie_budget_remaining == 0
        elif tie_budget_remaining is not None:
            mismatch = reduces_labels and tie_budget_remaining > 0
    if mismatch:
        violations.append("projection_guard_decision_mismatch")
    expected = projected if projection["accepted"] else after_local
    if abs(committed - expected) > _scale_tolerance(committed, expected):
        violations.append("projection_guard_commit_mismatch")
    return violations


def guard_trace_violations(
    projections: Sequence[dict[str, Any]],
    *,
    n_vertices: int,
    require_label_counts: bool = False,
) -> list[str]:
    """Audit the full projection transcript, including the per-call tie budget."""

    violations = []
    tie_budget = n_vertices
    stopped = False
    for iteration, projection in enumerate(projections):
        if (require_label_counts or "iteration" in projection) and projection.get("iteration") != iteration:
            violations.append("projection_trace_iteration_mismatch")
        if stopped:
            violations.append("projection_after_guard_rejection")
        violations.extend(guard_decision_violations(
            projection, require_label_counts=require_label_counts,
            tie_budget_remaining=tie_budget,
        ))
        if projection["accepted"]:
            if not native_is_improvement(
                projection["quality_projected"], projection["quality_after_local"]
            ):
                tie_budget = max(0, tie_budget - 1)
        else:
            stopped = True
    return sorted(set(violations))


def _scale_tolerance(*values: float) -> float:
    return ABS_TOLERANCE + REL_TOLERANCE * max(
        (abs(float(value)) for value in values), default=0.0
    )


def graph_pairs(n_vertices: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(n_vertices), 2))


def graph_edges(n_vertices: int, mask: int) -> list[tuple[int, int]]:
    pairs = graph_pairs(n_vertices)
    if mask < 0 or mask >= 1 << len(pairs):
        raise ValueError("graph mask is outside the labelled simple-graph range")
    return [edge for index, edge in enumerate(pairs) if mask & (1 << index)]


def graph_mask_count(n_vertices: int) -> int:
    """Number of supported nonzero-edge labelled graph masks."""
    return (1 << len(graph_pairs(n_vertices))) - 1


def valid_q2_state_count(n_vertices: int) -> int:
    """Every cap-two state except the non-contiguous all-label-1 profile."""
    return 3**n_vertices - 1


def valid_disjoint_state_count(n_vertices: int) -> int:
    """Every binary flat state except the non-contiguous all-label-1 profile."""
    return 2**n_vertices - 1


def iter_q2_states(n_vertices: int) -> Iterable[tuple[tuple[int, ...], ...]]:
    for code in range(3**n_vertices):
        value = code
        rows: list[tuple[int, ...]] = []
        uses_zero = False
        for _ in range(n_vertices):
            action = Q2_ACTIONS[value % 3]
            rows.append(action)
            uses_zero = uses_zero or 0 in action
            value //= 3
        if uses_zero:
            yield tuple(rows)


def iter_disjoint_states(n_vertices: int) -> Iterable[tuple[int, ...]]:
    for code in range(2**n_vertices):
        labels = tuple((code >> vertex) & 1 for vertex in range(n_vertices))
        if 0 in labels:
            yield labels


def _state_hash(state: Sequence[Sequence[int]]) -> str:
    return _sha256_bytes(_canonical_json([list(row) for row in state]))


def _case_key(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload))[:24]


def _case_seed(case_key: str) -> int:
    return int(case_key[:16], 16) % (2**31 - 1)


def _adjacency(n_vertices: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n_vertices, n_vertices), dtype=float)
    for first, second in edges:
        matrix[first, second] = matrix[second, first] = 1.0
    return matrix


def _native_quality(result: Any) -> float:
    params = getattr(result, "_params", None)
    if isinstance(params, dict) and "quality" in params:
        return float(params["quality"])
    value = getattr(result, "quality", None)
    if value is None:
        raise ValueError("native result does not expose quality")
    return float(value)


def _nested_memberships(result: Any, cap: int) -> list[list[int]]:
    memberships = result.membership
    if cap == 1:
        return [[int(label)] for label in memberships]
    return [[int(label) for label in row] for row in memberships]


def _validate_rows(rows: Sequence[Sequence[int]], cap: int) -> list[str]:
    violations: list[str] = []
    for vertex, row in enumerate(rows):
        labels = [int(label) for label in row]
        if not labels:
            violations.append(f"empty_row:{vertex}")
        if len(labels) > cap:
            violations.append(f"cap_exceeded:{vertex}")
        if len(labels) != len(set(labels)):
            violations.append(f"duplicate_label:{vertex}")
        if labels != sorted(labels):
            violations.append(f"unsorted_row:{vertex}")
        if any(label < 0 for label in labels):
            violations.append(f"negative_label:{vertex}")
    used = sorted({label for row in rows for label in row})
    if used and used != list(range(used[-1] + 1)):
        violations.append("noncontiguous_labels")
    return violations


def _normalized_quality(
    adjacency: np.ndarray,
    gamma: float,
    rows: Sequence[Sequence[int]],
) -> float:
    label_count = max(label for row in rows for label in row) + 1
    return oracle.original_normalized_quality(
        adjacency,
        np.ones(adjacency.shape[0]),
        gamma,
        rows,
        label_count,
    )


def _max_prefix_regret(
    adjacency: np.ndarray,
    gamma: float,
    rows: Sequence[Sequence[int]],
    cap: int,
    *,
    allow_isolation: bool,
) -> tuple[float, dict[str, Any] | None]:
    active_count = max(label for row in rows for label in row) + 1
    label_count = active_count + int(allow_isolation)
    weights = np.ones(adjacency.shape[0])
    maximum = 0.0
    witness = None
    for vertex, current in enumerate(rows):
        gains = oracle.label_gains(adjacency, weights, gamma, rows, label_count, vertex)
        best, best_utility, regret = oracle.prefix_best_response(gains, current, cap)
        if regret > maximum:
            maximum = float(regret)
            witness = {
                "vertex": vertex,
                "current": list(current),
                "best": list(best),
                "best_utility": float(best_utility),
                "regret": float(regret),
                "tolerance": _scale_tolerance(best_utility, best_utility - regret),
                "gains": [float(value) for value in gains],
            }
    return maximum, witness


@dataclass(frozen=True)
class Mode:
    name: str
    n_iterations: int
    local_move_only: bool
    allow_isolation: bool
    require_terminal_equilibrium: bool


def _overlap_modes(include_positive_budget: bool) -> tuple[Mode, ...]:
    modes = tuple(
        Mode(name, iterations, local, isolation, True)
        for name, iterations, local, isolation in OVERLAP_NEGATIVE_MODES
    )
    if include_positive_budget:
        modes += tuple(
            Mode(name, iterations, local, isolation, False)
            for name, iterations, local, isolation in OVERLAP_POSITIVE_MODES
        )
    return modes


def _disjoint_modes() -> tuple[Mode, ...]:
    return tuple(
        Mode(name, iterations, local, isolation, True)
        for name, iterations, local, isolation in DISJOINT_MODES
    )


def _run_case(
    *,
    graph: ig.Graph,
    adjacency: np.ndarray,
    graph_mask: int,
    initial_rows: Sequence[Sequence[int]],
    cap: int,
    gamma: float,
    mode: Mode,
) -> dict[str, Any]:
    n_vertices = graph.vcount()
    case_descriptor = {
        "protocol": PROTOCOL_NAME,
        "n": n_vertices,
        "graph_mask": graph_mask,
        "initial_membership_sha256": _state_hash(initial_rows),
        "cap": cap,
        "gamma": gamma,
        "mode": mode.name,
    }
    key = _case_key(case_descriptor)
    initial_quality = _normalized_quality(adjacency, gamma, initial_rows)
    violations: list[str] = []
    failure: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        ig.set_random_number_generator(random.Random(_case_seed(key)))
        initial_membership: Any
        if cap == 1:
            initial_membership = [int(row[0]) for row in initial_rows]
        else:
            initial_membership = [list(row) for row in initial_rows]
        result = graph.community_leiden(
            objective_function="CPM",
            resolution=gamma,
            beta=0.01,
            max_memberships=cap,
            initial_membership=initial_membership,
            n_iterations=mode.n_iterations,
            allow_isolation=mode.allow_isolation,
            local_move_only=mode.local_move_only,
        )
        final_rows = _nested_memberships(result, cap)
        violations.extend(_validate_rows(final_rows, cap))
        direct_quality = _normalized_quality(adjacency, gamma, final_rows)
        native_quality = _native_quality(result)
        quality_error = abs(direct_quality - native_quality)
        decrease = initial_quality - direct_quality
        quality_tolerance = _scale_tolerance(direct_quality, native_quality)
        monotonicity_tolerance = _scale_tolerance(initial_quality, direct_quality)
        if not math.isfinite(native_quality) or not math.isfinite(direct_quality):
            violations.append("nonfinite_quality")
        if quality_error > quality_tolerance:
            violations.append("native_quality_mismatch")
        if decrease > monotonicity_tolerance:
            violations.append("original_quality_decrease")
        max_regret = 0.0
        witness = None
        terminal_tolerance = ABS_TOLERANCE
        if mode.require_terminal_equilibrium:
            max_regret, witness = _max_prefix_regret(
                adjacency,
                gamma,
                final_rows,
                cap,
                allow_isolation=mode.allow_isolation,
            )
            terminal_tolerance = (
                float(witness["tolerance"]) if witness else ABS_TOLERANCE
            )
            if max_regret > terminal_tolerance:
                violations.append("positive_terminal_regret")
        stable_result = {
            "case_key": key,
            "final_membership_sha256": _state_hash(final_rows),
            "native_quality": native_quality,
            "direct_quality": direct_quality,
            "quality_error": quality_error,
            "quality_tolerance": quality_tolerance,
            "initial_decrease": max(0.0, decrease),
            "monotonicity_tolerance": monotonicity_tolerance,
            "max_regret": max_regret,
            "terminal_tolerance": terminal_tolerance,
            "violations": sorted(set(violations)),
        }
        if violations:
            failure = {
                **case_descriptor,
                **stable_result,
                "edges": [list(edge) for edge in graph.get_edgelist()],
                "initial_memberships": [list(row) for row in initial_rows],
                "final_memberships": final_rows,
                "regret_witness": witness,
            }
    except Exception as error:
        stable_result = {
            "case_key": key,
            "exception": type(error).__name__,
            "message": str(error),
            "violations": ["native_exception"],
        }
        failure = {
            **case_descriptor,
            **stable_result,
            "edges": [list(edge) for edge in graph.get_edgelist()],
            "initial_memberships": [list(row) for row in initial_rows],
        }
    return {
        "stable": stable_result,
        "failure": failure,
        "elapsed_seconds": time.perf_counter() - started,
    }


def expected_calls_per_graph(
    n_vertices: int,
    gammas: Sequence[float],
    *,
    include_positive_budget: bool,
) -> int:
    overlap = (
        valid_q2_state_count(n_vertices)
        * len(gammas)
        * len(_overlap_modes(include_positive_budget))
    )
    disjoint = (
        valid_disjoint_state_count(n_vertices) * len(gammas) * len(_disjoint_modes())
    )
    return overlap + disjoint


def expected_total_calls(
    min_n: int,
    max_n: int,
    gammas: Sequence[float],
    *,
    include_positive_budget: bool,
) -> int:
    return sum(
        graph_mask_count(n)
        * expected_calls_per_graph(
            n, gammas, include_positive_budget=include_positive_budget
        )
        for n in range(min_n, max_n + 1)
    )
