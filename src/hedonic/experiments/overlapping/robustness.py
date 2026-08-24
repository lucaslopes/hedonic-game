"""Unit-ell2 overlapping robustness and equilibrium diagnostics.

This module deliberately lives in ``hedonic.experiments``.  The core
``Game`` API delegates detection to the native binding; these helpers provide
an independent, auditable implementation of the local utility used by the
overlapping paper.  In particular, :func:`audit_cover` must not be replaced by
``metrics.in_equilibrium_overlapping``: that historical helper checks a
different binary join/leave objective.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any


FRESH_LABEL = -1


def _as_int_member(member: Any) -> int:
    try:
        value = int(member)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid vertex id {member!r}") from exc
    return value


def canonicalize_cover(
    cover: Sequence[Sequence[int]],
    n_vertices: int | None = None,
) -> list[list[int]]:
    """Return a deterministic, label-invariant cover representation.

    Empty and duplicate communities/members are removed.  Communities are
    sorted by their sorted member tuple, so hashes and comparisons do not
    depend on native label ordering.
    """
    communities: set[tuple[int, ...]] = set()
    for community in cover:
        members = sorted({_as_int_member(member) for member in community})
        if n_vertices is not None and any(
            member < 0 or member >= n_vertices for member in members
        ):
            raise ValueError("cover contains a vertex outside the graph")
        if members:
            communities.add(tuple(members))
    return [list(members) for members in sorted(communities)]


def cover_hash(cover: Sequence[Sequence[int]], n_vertices: int | None = None) -> str:
    """Hash a canonical cover with SHA-256."""
    canonical = canonicalize_cover(cover, n_vertices=n_vertices)
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def cover_to_vertex_memberships(
    cover: Sequence[Sequence[int]],
    n_vertices: int,
    *,
    require_covered: bool = True,
) -> list[list[int]]:
    """Convert community lists to contiguous per-vertex membership labels."""
    canonical = canonicalize_cover(cover, n_vertices=n_vertices)
    memberships: list[list[int]] = [[] for _ in range(n_vertices)]
    for community_id, community in enumerate(canonical):
        for vertex in community:
            memberships[vertex].append(community_id)
    if require_covered:
        uncovered = [vertex for vertex, labels in enumerate(memberships) if not labels]
        if uncovered:
            preview = ", ".join(map(str, uncovered[:8]))
            suffix = "..." if len(uncovered) > 8 else ""
            raise ValueError(
                f"cover does not cover every vertex ({len(uncovered)} missing: "
                f"{preview}{suffix}); apply an explicit completion policy"
            )
    return memberships


def vertex_memberships_to_cover(
    memberships: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Convert per-vertex memberships to community lists."""
    if not memberships:
        return []
    max_label = max((int(label) for labels in memberships for label in labels), default=-1)
    cover = [[] for _ in range(max_label + 1)]
    for vertex, labels in enumerate(memberships):
        if not labels:
            raise ValueError(f"vertex {vertex} has no membership")
        for label in sorted({int(label) for label in labels}):
            if label < 0:
                raise ValueError("membership labels must be non-negative")
            while label >= len(cover):
                cover.append([])
            cover[label].append(vertex)
    return [community for community in cover if community]


def canonicalize_memberships(
    memberships: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Canonicalize a per-vertex cover while preserving vertex identities."""
    return cover_to_vertex_memberships(
        canonicalize_cover(vertex_memberships_to_cover(memberships), len(memberships)),
        len(memberships),
        require_covered=True,
    )


def _edge_weights(graph, edge_weights: Sequence[float] | None) -> list[float]:
    if edge_weights is not None:
        weights = [float(value) for value in edge_weights]
    elif "weight" in graph.edge_attributes():
        weights = [float(value) for value in graph.es["weight"]]
    else:
        weights = [1.0] * graph.ecount()
    if len(weights) != graph.ecount():
        raise ValueError("edge_weights must have one value per graph edge")
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("edge weights must be finite and non-negative")
    return weights


@dataclass
class FractionalState:
    """Cached local state for a fixed cover."""

    memberships: list[tuple[int, ...]]
    masses: list[float]
    support: list[dict[int, float]]
    intensities: list[dict[int, float]]
    lightest_labels: list[int]
    edge_weights: list[float]

    @property
    def n_vertices(self) -> int:
        return len(self.memberships)

    @property
    def n_communities(self) -> int:
        return len(self.masses)


def build_fractional_state(
    graph,
    memberships: Sequence[Sequence[int]],
    *,
    edge_weights: Sequence[float] | None = None,
) -> FractionalState:
    """Build masses and sparse neighbor support for a valid cover."""
    n = graph.vcount()
    normalized: list[tuple[int, ...]] = []
    max_label = -1
    for vertex, labels in enumerate(memberships):
        unique = tuple(sorted({int(label) for label in labels}))
        if not unique:
            raise ValueError(f"vertex {vertex} has no membership")
        if unique[0] < 0:
            raise ValueError("membership labels must be non-negative")
        max_label = max(max_label, unique[-1])
        normalized.append(unique)
    if len(normalized) != n:
        raise ValueError("memberships length must equal graph.vcount()")
    if max_label < 0:
        raise ValueError("cover must contain at least one community")
    labels = {label for vertex_labels in normalized for label in vertex_labels}
    if labels != set(range(max_label + 1)):
        raise ValueError("membership labels must be contiguous from zero")

    intensities: list[dict[int, float]] = []
    masses = [0.0] * (max_label + 1)
    for vertex_labels in normalized:
        intensity = 1.0 / math.sqrt(len(vertex_labels))
        row = {label: intensity for label in vertex_labels}
        intensities.append(row)
        for label in vertex_labels:
            masses[label] += intensity

    weights = _edge_weights(graph, edge_weights)
    support: list[dict[int, float]] = [defaultdict(float) for _ in range(n)]
    for edge_index, (first, second) in enumerate(graph.get_edgelist()):
        weight = weights[edge_index]
        if weight == 0:
            continue
        for label, intensity in intensities[first].items():
            support[second][label] += weight * intensity
        for label, intensity in intensities[second].items():
            support[first][label] += weight * intensity

    lightest_labels = sorted(range(len(masses)), key=lambda label: (masses[label], label))
    return FractionalState(
        memberships=normalized,
        masses=masses,
        support=[dict(row) for row in support],
        intensities=intensities,
        lightest_labels=lightest_labels,
        edge_weights=weights,
    )


def _candidate_labels(
    state: FractionalState,
    vertex: int,
    gamma: float,
    max_memberships: int,
    allow_isolation: bool,
    *,
    dense: bool,
) -> list[int]:
    """Return an exact candidate-label superset for prefix optimization.

    In dense mode this is every active label.  In sparse mode, neighbor/current
    labels plus the lightest-mass labels suffice: unseen labels have no support
    and therefore non-positive score for gamma >= 0.  If all scores are
    non-positive, the best action has cardinality one, so the lightest unseen
    label is enough.
    """
    n_communities = state.n_communities
    if dense or n_communities <= max(256, max_memberships * 8):
        labels = set(range(n_communities))
    else:
        labels = set(state.memberships[vertex])
        labels.update(state.support[vertex])
        labels.update(state.lightest_labels[: max_memberships + len(labels) + 1])
    if allow_isolation:
        labels.add(FRESH_LABEL)
    return sorted(labels)


def _label_score(
    state: FractionalState,
    vertex: int,
    label: int,
    gamma: float,
) -> float:
    if label == FRESH_LABEL:
        return 0.0
    support = state.support[vertex].get(label, 0.0)
    own_intensity = state.intensities[vertex].get(label, 0.0)
    crowding = state.masses[label] - own_intensity
    return support - float(gamma) * crowding


def _utility_for_labels(
    state: FractionalState,
    vertex: int,
    labels: Sequence[int],
    gamma: float,
) -> float:
    if not labels:
        raise ValueError("a membership action must be non-empty")
    return sum(_label_score(state, vertex, label, gamma) for label in labels) / math.sqrt(
        len(labels)
    )


def best_response(
    state: FractionalState,
    vertex: int,
    gamma: float,
    max_memberships: int,
    allow_isolation: bool,
    *,
    dense: bool = False,
) -> dict[str, Any]:
    """Return the exact best prefix response and regret for one vertex."""
    if not 0 <= vertex < state.n_vertices:
        raise IndexError("vertex outside state")
    if max_memberships < 1:
        raise ValueError("max_memberships must be positive")
    current = state.memberships[vertex]
    if len(current) > max_memberships:
        raise ValueError("current membership exceeds max_memberships")
    labels = _candidate_labels(
        state,
        vertex,
        gamma,
        max_memberships,
        allow_isolation,
        dense=dense,
    )
    scored = sorted(
        ((label, _label_score(state, vertex, label, gamma)) for label in labels),
        key=lambda item: (-item[1], item[0]),
    )
    best_utility = -math.inf
    best_labels: tuple[int, ...] = ()
    prefix = 0.0
    for cardinality, (label, score) in enumerate(scored[:max_memberships], 1):
        prefix += score
        utility = prefix / math.sqrt(cardinality)
        if utility > best_utility + 1e-15:
            best_utility = utility
            best_labels = tuple(item[0] for item in scored[:cardinality])
    if not best_labels:
        raise ValueError("no admissible membership action")
    current_utility = _utility_for_labels(state, vertex, current, gamma)
    regret = best_utility - current_utility
    return {
        "vertex": int(vertex),
        "gamma": float(gamma),
        "current_memberships": list(current),
        "best_memberships": list(best_labels),
        "best_utility": float(best_utility),
        "current_utility": float(current_utility),
        "regret": float(regret),
        "candidate_label_count": len(labels),
        "dense_candidates": bool(dense or state.n_communities <= max(256, max_memberships * 8)),
    }


def _tolerance(value: float, atol: float, rtol: float) -> float:
    return float(atol + rtol * max(1.0, abs(value)))


def _stable(diagnostic: dict[str, Any], atol: float, rtol: float) -> bool:
    return diagnostic["regret"] <= _tolerance(diagnostic["current_utility"], atol, rtol)


def stability_interval_for_vertex(
    state: FractionalState,
    vertex: int,
    center: float,
    max_memberships: int,
    allow_isolation: bool,
    *,
    atol: float = 1e-10,
    rtol: float = 1e-9,
    iterations: int = 45,
    dense: bool = False,
) -> tuple[float | None, float | None]:
    """Approximate the stable resolution interval containing ``center``.

    Stability is an intersection of affine half-spaces in one dimension, so
    the set is an interval.  This helper is intentionally optional; the exact
    full-range certificate uses endpoint checks directly.
    """
    if not 0.0 <= center <= 1.0:
        raise ValueError("center must lie in [0, 1]")
    at_center = best_response(
        state, vertex, center, max_memberships, allow_isolation, dense=dense
    )
    if not _stable(at_center, atol, rtol):
        return None, None

    def stable_at(value: float) -> bool:
        return _stable(
            best_response(
                state, vertex, value, max_memberships, allow_isolation, dense=dense
            ),
            atol,
            rtol,
        )

    if stable_at(0.0):
        lower = 0.0
    else:
        low, high = 0.0, center
        for _ in range(iterations):
            middle = (low + high) / 2.0
            if stable_at(middle):
                high = middle
            else:
                low = middle
        lower = high

    if stable_at(1.0):
        upper = 1.0
    else:
        low, high = center, 1.0
        for _ in range(iterations):
            middle = (low + high) / 2.0
            if stable_at(middle):
                low = middle
            else:
                high = middle
        upper = low
    return float(lower), float(upper)


def audit_cover(
    graph,
    memberships: Sequence[Sequence[int]],
    *,
    max_memberships: int,
    allow_isolation: bool,
    gamma: float | None = None,
    interval: tuple[float, float] = (0.0, 1.0),
    atol: float = 1e-10,
    rtol: float = 1e-9,
    vertices: Iterable[int] | None = None,
    compute_intervals: bool = False,
    dense: bool = False,
    edge_weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Audit endpoint robustness and pointwise equilibrium residuals."""
    lower, upper = (float(interval[0]), float(interval[1]))
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("interval must satisfy 0 <= lower <= upper <= 1")
    state = build_fractional_state(graph, memberships, edge_weights=edge_weights)
    selected_gamma = graph.density() if gamma is None else float(gamma)
    if selected_gamma < 0 or selected_gamma > 1:
        raise ValueError("gamma must lie in [0, 1] for robustness auditing")
    selected_vertices = list(range(state.n_vertices) if vertices is None else vertices)
    if any(vertex < 0 or vertex >= state.n_vertices for vertex in selected_vertices):
        raise ValueError("vertices contains an index outside the graph")
    if not selected_vertices:
        raise ValueError("at least one vertex is required for an audit")

    endpoint_diagnostics = [
        [
            best_response(
                state,
                vertex,
                endpoint,
                max_memberships,
                allow_isolation,
                dense=dense,
            )
            for endpoint in (lower, upper)
        ]
        for vertex in selected_vertices
    ]
    selected_diagnostics = [
        best_response(
            state,
            vertex,
            selected_gamma,
            max_memberships,
            allow_isolation,
            dense=dense,
        )
        for vertex in selected_vertices
    ]
    robust_flags = [
        _stable(pair[0], atol, rtol) and _stable(pair[1], atol, rtol)
        for pair in endpoint_diagnostics
    ]
    selected_flags = [
        _stable(diagnostic, atol, rtol) for diagnostic in selected_diagnostics
    ]
    selected_tolerances = [
        _tolerance(float(diagnostic["current_utility"]), atol, rtol)
        for diagnostic in selected_diagnostics
    ]
    positive_regrets = [max(0.0, float(diagnostic["regret"])) for diagnostic in selected_diagnostics]
    interval_bounds: list[tuple[float | None, float | None]] = []
    if compute_intervals:
        interval_bounds = [
            stability_interval_for_vertex(
                state,
                vertex,
                selected_gamma,
                max_memberships,
                allow_isolation,
                atol=atol,
                rtol=rtol,
                dense=dense,
            )
            for vertex in selected_vertices
        ]
    valid_bounds = [bound for bound in interval_bounds if bound[0] is not None]
    equilibrium_lower = (
        max(float(bound[0]) for bound in valid_bounds)
        if valid_bounds and len(valid_bounds) == len(selected_vertices)
        else None
    )
    equilibrium_upper = (
        min(float(bound[1]) for bound in valid_bounds)
        if valid_bounds and len(valid_bounds) == len(selected_vertices)
        else None
    )
    return {
        "robust_fraction_gamma_0_1": sum(robust_flags) / len(robust_flags),
        "robust_vertex_count_gamma_0_1": int(sum(robust_flags)),
        "stable_fraction_at_resolution": sum(selected_flags) / len(selected_flags),
        "stable_vertex_count_at_resolution": int(sum(selected_flags)),
        "mean_positive_regret_at_resolution": sum(positive_regrets) / len(positive_regrets),
        "max_positive_regret_at_resolution": max(positive_regrets, default=0.0),
        "max_stability_tolerance_at_resolution": max(
            selected_tolerances, default=float(atol + rtol)
        ),
        "p95_positive_regret_at_resolution": _percentile(positive_regrets, 0.95),
        "profitable_vertex_count_at_resolution": int(
            sum(value > _tolerance(diagnostic["current_utility"], atol, rtol)
                for value, diagnostic in zip(positive_regrets, selected_diagnostics))
        ),
        "is_local_equilibrium_at_resolution": all(selected_flags),
        "near_tolerance_vertex_count": int(
            sum(
                abs(float(diagnostic["regret"]))
                <= 10.0 * _tolerance(diagnostic["current_utility"], atol, rtol)
                for diagnostic in selected_diagnostics
            )
        ),
        "n_vertices_scored": len(selected_vertices),
        "n_vertices_graph": state.n_vertices,
        "n_communities": state.n_communities,
        "max_memberships": int(max_memberships),
        "cap_saturated_fraction": sum(
            len(state.memberships[vertex]) >= max_memberships
            for vertex in selected_vertices
        )
        / len(selected_vertices),
        "cap_saturated_vertex_count": int(
            sum(len(state.memberships[vertex]) >= max_memberships for vertex in selected_vertices)
        ),
        "allow_isolation": bool(allow_isolation),
        "gamma": selected_gamma,
        "interval": [lower, upper],
        "robustness_atol": float(atol),
        "robustness_rtol": float(rtol),
        "equilibrium_interval_lower": equilibrium_lower,
        "equilibrium_interval_upper": equilibrium_upper,
        "equilibrium_interval_width": (
            equilibrium_upper - equilibrium_lower
            if equilibrium_lower is not None and equilibrium_upper is not None
            else None
        ),
        "vertex_diagnostics": (
            {
                str(vertex): {
                    "endpoint": pair,
                    "selected": selected,
                    "robust_gamma_0_1": robust,
                }
                for vertex, pair, selected, robust in zip(
                    selected_vertices,
                    endpoint_diagnostics,
                    selected_diagnostics,
                    robust_flags,
                )
            }
            if compute_intervals
            else None
        ),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def fractional_phi(
    graph,
    memberships: Sequence[Sequence[int]],
    gamma: float,
    *,
    edge_weights: Sequence[float] | None = None,
) -> float:
    r"""Compute the normalized unit-$\ell_2$ fractional CPM potential."""
    state = build_fractional_state(graph, memberships, edge_weights=edge_weights)
    total_weight = sum(state.edge_weights)
    if total_weight <= 0:
        return 0.0
    internal = 0.0
    for edge_index, (first, second) in enumerate(graph.get_edgelist()):
        shared = set(state.memberships[first]).intersection(state.memberships[second])
        if not shared:
            continue
        weight = state.edge_weights[edge_index]
        for label in shared:
            internal += weight * state.intensities[first][label] * state.intensities[second][label]
    penalty = sum(mass * mass for mass in state.masses)
    return float((2.0 * internal - float(gamma) * penalty) / (2.0 * total_weight))


def exhaustive_best_response(
    state: FractionalState,
    vertex: int,
    gamma: float,
    max_memberships: int,
    allow_isolation: bool,
) -> dict[str, Any]:
    """Slow exhaustive oracle for tiny-graph tests and diagnostics."""
    labels = list(range(state.n_communities))
    if allow_isolation:
        labels.append(FRESH_LABEL)
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for cardinality in range(1, max_memberships + 1):
        for selected in itertools.combinations(labels, cardinality):
            candidates.append(
                (_utility_for_labels(state, vertex, selected, gamma), tuple(selected))
            )
    utility, selected = max(candidates, key=lambda item: (item[0], tuple(-x for x in item[1])))
    current = state.memberships[vertex]
    current_utility = _utility_for_labels(state, vertex, current, gamma)
    return {
        "vertex": int(vertex),
        "gamma": float(gamma),
        "current_memberships": list(current),
        "best_memberships": list(selected),
        "best_utility": float(utility),
        "current_utility": float(current_utility),
        "regret": float(utility - current_utility),
        "candidate_label_count": len(labels),
        "dense_candidates": True,
    }


def perturb_cover_incidence(
    cover: Sequence[Sequence[int]],
    n_vertices: int,
    *,
    swaps: int,
    seed: int,
    max_attempts: int | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Perturb a cover with degree-preserving bipartite incidence switches."""
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    canonical = canonicalize_cover(cover, n_vertices=n_vertices)
    memberships = cover_to_vertex_memberships(canonical, n_vertices)
    incidence = {(vertex, label) for vertex, labels in enumerate(memberships) for label in labels}
    community_sets = [set(community) for community in canonical]
    rng = random.Random(int(seed))
    attempts_limit = max_attempts if max_attempts is not None else max(100, swaps * 50)
    successful = 0
    attempts = 0
    while successful < swaps and attempts < attempts_limit:
        attempts += 1
        first, second = rng.sample(sorted(incidence), 2) if len(incidence) >= 2 else (None, None)
        if first is None:
            break
        vertex_a, label_a = first
        vertex_b, label_b = second
        if vertex_a == vertex_b or label_a == label_b:
            continue
        cross_a = (vertex_a, label_b)
        cross_b = (vertex_b, label_a)
        if cross_a in incidence or cross_b in incidence:
            continue
        # A switch must preserve the cover as a labelled incidence structure.
        # Reject switches that would make two communities identical; otherwise
        # the label-invariant canonicalizer would silently drop one and break
        # the promised community-size and incidence-count invariants.
        candidate_a = (community_sets[label_a] - {vertex_a}) | {vertex_b}
        candidate_b = (community_sets[label_b] - {vertex_b}) | {vertex_a}
        if candidate_a == candidate_b:
            continue
        unchanged = {
            frozenset(community_sets[label])
            for label in range(len(community_sets))
            if label not in {label_a, label_b}
        }
        if frozenset(candidate_a) in unchanged or frozenset(candidate_b) in unchanged:
            continue
        incidence.remove(first)
        incidence.remove(second)
        incidence.add(cross_a)
        incidence.add(cross_b)
        community_sets[label_a] = candidate_a
        community_sets[label_b] = candidate_b
        successful += 1

    perturbed_raw = [[] for _ in canonical]
    for vertex, label in sorted(incidence):
        perturbed_raw[label].append(vertex)
    # Preserve the original community labels for the incidence-distance
    # calculation.  cover_to_vertex_memberships canonicalizes and relabels
    # community bodies, which would turn a two-edge switch into an unrelated
    # label-permutation distance.
    raw_memberships = [[] for _ in range(n_vertices)]
    for vertex, label in sorted(incidence):
        raw_memberships[vertex].append(label)
    perturbed = canonicalize_cover(perturbed_raw, n_vertices=n_vertices)
    initial_pairs = set((vertex, label) for vertex, labels in enumerate(memberships) for label in labels)
    # Row degrees are vertex-specific invariants; community sizes are compared
    # as a multiset because the returned canonical cover relabels bodies.
    realized_vertex_f1 = _incidence_f1(memberships, raw_memberships)
    metadata = {
        "requested_swaps": int(swaps),
        "successful_swaps": int(successful),
        "attempts": int(attempts),
        "max_attempts": int(attempts_limit),
        "seed": int(seed),
        "realized_incidence_distance": float(1.0 - realized_vertex_f1),
        "realized_incidence_f1": float(realized_vertex_f1),
        "vertex_membership_counts_preserved": list(map(len, memberships))
        == list(map(len, raw_memberships)),
        "community_sizes_preserved": sorted(map(len, canonical))
        == sorted(map(len, perturbed)),
        "initial_incidence_count": len(initial_pairs),
        "final_incidence_count": sum(map(len, perturbed)),
    }
    return perturbed, metadata


def _incidence_f1(
    first: Sequence[Sequence[int]],
    second: Sequence[Sequence[int]],
) -> float:
    """Return incidence F1 when the community labels are held fixed.

    Incidence switches intentionally preserve the original community labels,
    so this is the correct realized perturbation distance.  Label-invariant
    output-cover comparisons use ``metrics.evaluate_cover`` separately.
    """
    first_pairs = {
        (vertex, int(label))
        for vertex, labels in enumerate(first)
        for label in labels
    }
    second_pairs = {
        (vertex, int(label))
        for vertex, labels in enumerate(second)
        for label in labels
    }
    denominator = len(first_pairs) + len(second_pairs)
    return 2.0 * len(first_pairs & second_pairs) / denominator if denominator else 1.0


def cover_incidence_f1(
    first: Sequence[Sequence[int]],
    second: Sequence[Sequence[int]],
) -> float:
    """Return one-to-one label-invariant incidence F1 for tiny calibrations.

    The routine deliberately uses permutation search and is therefore only a
    tiny-instance oracle.  Large experiments should use the audited matching
    implementation in ``overlapping.metrics``.
    """
    first_cover = canonicalize_cover(first)
    second_cover = canonicalize_cover(second)
    first_sizes = [set(community) for community in first_cover]
    second_sizes = [set(community) for community in second_cover]
    if not first_sizes and not second_sizes:
        return 1.0
    if len(first_sizes) <= len(second_sizes):
        best = 0
        for chosen in itertools.permutations(range(len(second_sizes)), len(first_sizes)):
            best = max(
                best,
                sum(
                    len(first_sizes[index].intersection(second_sizes[target]))
                    for index, target in enumerate(chosen)
                ),
            )
    else:
        best = 0
        for chosen in itertools.permutations(range(len(first_sizes)), len(second_sizes)):
            best = max(
                best,
                sum(
                    len(first_sizes[target].intersection(second_sizes[index]))
                    for index, target in enumerate(chosen)
                ),
            )
    total = sum(map(len, first_sizes)) + sum(map(len, second_sizes))
    return 2.0 * best / total if total else 1.0


def enumerate_valid_covers(
    n_vertices: int,
    n_communities: int,
    max_memberships: int,
    *,
    max_candidates: int | None = None,
) -> Iterable[list[list[int]]]:
    """Enumerate label-invariant equal-intensity covers for tiny graphs.

    Every vertex receives a nonempty subset of the requested labels, each
    requested label must occur, and no vertex exceeds the cap.  The generator
    deduplicates label permutations after enumeration and is intentionally not
    suitable for real networks.
    """
    if n_vertices < 1 or n_communities < 1:
        raise ValueError("n_vertices and n_communities must be positive")
    if max_memberships < 1:
        raise ValueError("max_memberships must be positive")
    cap = min(max_memberships, n_communities)
    actions = tuple(
        tuple(selected)
        for cardinality in range(1, cap + 1)
        for selected in itertools.combinations(range(n_communities), cardinality)
    )
    seen: set[str] = set()
    yielded = 0
    for assignment in itertools.product(actions, repeat=n_vertices):
        if any(
            not any(label in labels for labels in assignment)
            for label in range(n_communities)
        ):
            continue
        canonical = canonicalize_memberships(assignment)
        digest = cover_hash(vertex_memberships_to_cover(canonical), n_vertices)
        if digest in seen:
            continue
        seen.add(digest)
        yield canonical
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def nearest_equilibrium_tiny(
    graph,
    ground_truth_cover: Sequence[Sequence[int]],
    *,
    gamma: float,
    n_communities: int | None = None,
    max_memberships: int | None = None,
    allow_isolation: bool = False,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    """Find the nearest certified equilibrium by exhaustive tiny enumeration."""
    gt_cover = canonicalize_cover(ground_truth_cover, graph.vcount())
    labels = n_communities if n_communities is not None else len(gt_cover)
    cap = (
        max_memberships
        if max_memberships is not None
        else max(1, max(map(len, cover_to_vertex_memberships(gt_cover, graph.vcount()))))
    )
    best: dict[str, Any] | None = None
    equilibria = 0
    candidates = 0
    for memberships in enumerate_valid_covers(
        graph.vcount(), labels, cap, max_candidates=max_candidates
    ):
        candidates += 1
        audit = audit_cover(
            graph,
            memberships,
            max_memberships=cap,
            allow_isolation=allow_isolation,
            gamma=gamma,
            interval=(0.0, 1.0),
            dense=True,
        )
        if not audit["is_local_equilibrium_at_resolution"]:
            continue
        equilibria += 1
        cover = vertex_memberships_to_cover(memberships)
        distance = 1.0 - cover_incidence_f1(cover, gt_cover)
        candidate = {
            "cover": cover,
            "memberships": memberships,
            "distance_to_ground_truth": distance,
            "audit": audit,
        }
        if best is None or distance < best["distance_to_ground_truth"] - 1e-15:
            best = candidate
    return {
        "status": "completed" if best is not None else "no_equilibrium_found",
        "candidate_count": candidates,
        "equilibrium_count": equilibria,
        "nearest": best,
        "gamma": float(gamma),
        "n_communities": int(labels),
        "max_memberships": int(cap),
        "allow_isolation": bool(allow_isolation),
    }


__all__ = [
    "FRESH_LABEL",
    "FractionalState",
    "audit_cover",
    "best_response",
    "build_fractional_state",
    "canonicalize_cover",
    "canonicalize_memberships",
    "cover_hash",
    "cover_incidence_f1",
    "cover_to_vertex_memberships",
    "enumerate_valid_covers",
    "exhaustive_best_response",
    "fractional_phi",
    "nearest_equilibrium_tiny",
    "perturb_cover_incidence",
    "stability_interval_for_vertex",
    "vertex_memberships_to_cover",
]
