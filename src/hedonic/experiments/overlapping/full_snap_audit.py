"""Independent unit-l2 objective and prefix-response audit for full-snap-v1.

This is deliberately separate from the historical binary-cover quality helpers.
It supports unweighted simple undirected graphs, no global count constraint, and
nonempty rows. Numerical audits are evidence, never exact rational certificates.
"""
from __future__ import annotations

from collections import Counter
import math
import random
from typing import Sequence


def memberships(cover: Sequence[Sequence[int]], n: int) -> list[set[int]]:
    rows: list[set[int]] = [set() for _ in range(n)]
    for label, body in enumerate(cover):
        if not body or len(set(body)) != len(body):
            raise ValueError("empty community or repeated vertex in detector cover")
        for vertex in body:
            if not isinstance(vertex, int) or not 0 <= vertex < n:
                raise ValueError("invalid detector vertex")
            rows[vertex].add(label)
    return rows


def unit_l2_audit(graph, cover, gamma: float, cap: int, *, max_vertices=32,
                  max_label_visits=2_000_000, seed=0) -> dict:
    """Audit arbitrary nonempty row replacements by independent sorted prefixes.

    Holding other rows fixed, row utility is sum(a[c] for c in S)/sqrt(|S|).
    Sorting coefficients and maximizing all feasible prefix lengths is therefore
    a full best response. Up to ``cap`` new empty labels are included: an audit
    violation is a real deviation in the unconstrained dynamic-label model. The
    negative-cap Fable contract resolves a finite label bank B=n for these
    singleton-start experiments, also bounding occupied labels.
    """
    if graph.is_directed() or not graph.is_simple() or not math.isfinite(gamma):
        raise ValueError("audit requires simple undirected graph and finite gamma")
    n, m = graph.vcount(), graph.ecount()
    rows = memberships(cover, n)
    if any(not row for row in rows):
        return {"status": "unsupported_uncovered_vertices", "objective": None,
                "certificate": "none", "reason": "unit-l2 rows require coverage"}
    unlimited = cap < 0
    if unlimited and len(cover) > n:
        raise ValueError("negative-cap output exceeds finite label bank n")
    cap = n if unlimited else cap
    fresh_labels = max(0, n - len(cover)) if unlimited else cap
    if cap < 1 and n:
        raise ValueError("cap must be positive or negative")
    if any(len(row) > cap for row in rows):
        raise ValueError("detector output exceeds per-row cap")
    scale = [1 / math.sqrt(len(row)) for row in rows]
    mass = [math.fsum(scale[v] for v in body) for body in cover]
    edge_sum = math.fsum(len(rows[e.source] & rows[e.target]) *
                         scale[e.source] * scale[e.target] for e in graph.es)
    mass_square = math.fsum(value * value for value in mass)
    phi = edge_sum - gamma * (mass_square - n) / 2
    quality = (2 * edge_sum - gamma * mass_square) / (2 * m) if m else None
    # Each sampled row scans every occupied label, not a neighbour-only subset.
    affordable = max_label_visits // max(1, len(cover) + cap)
    count = min(n, max_vertices, affordable)
    vertices = sorted(random.Random(seed).sample(range(n), count))
    max_gain = 0.0
    witness = None
    max_margin = 0.0
    violating_vertices = 0
    for vertex in vertices:
        coefficients = [-gamma * value for value in mass]
        for label in rows[vertex]:
            coefficients[label] += gamma * scale[vertex]
        for neighbour in graph.neighbors(vertex):
            for label in rows[neighbour]:
                coefficients[label] += scale[neighbour]
        current = math.fsum(coefficients[c] for c in rows[vertex]) * scale[vertex]
        # A lone vertex may put itself into several distinct new communities.
        # Do not deduplicate supports or assume a single fresh label suffices.
        ordered = sorted(coefficients + [0.0] * fresh_labels, reverse=True)
        prefix, best, best_size = 0.0, -math.inf, 0
        for size, value in enumerate(ordered[:cap], 1):
            prefix += value
            candidate = prefix / math.sqrt(size)
            if candidate > best:
                best, best_size = candidate, size
        gain = best - current
        margin = 64 * math.ulp(1.0) * max(1.0, abs(best), abs(current))
        max_margin = max(max_margin, margin)
        violating_vertices += gain > margin
        if gain > max_gain:
            max_gain = gain
            witness = {"vertex": vertex, "best_size": best_size,
                       "current_utility": current, "best_utility": best}
    complete = count == n
    return {
        "status": "full_numerical_audit" if complete else "sampled_vertices_full_actions",
        "certificate": "none_exact", "objective": {"unit_l2_pair_potential": phi,
            "unit_l2_native_quality_with_self_term": quality},
        "action_scope": "all_nonempty_subsets_up_to_cap_of_occupied_and_fresh_labels",
        "per_row_cap": cap, "label_bank": n if unlimited else None, "vertices_audited": count, "vertices_total": n,
        "audited_vertex_ids": vertices, "unverified_vertex_count": n - count,
        "max_label_visits": max_label_visits, "seed": seed,
        "max_unilateral_gain_observed": max_gain, "witness": witness,
        "violating_vertices": violating_vertices, "largest_numerical_margin": max_margin,
        "numerically_stationary_in_declared_scope": bool(count and not violating_vertices),
        "unverified_remainder": ("floating-point error, no exact certificate" if complete else
            "all unaudited vertices; floating-point error; no exact certificate"),
    }


def omega_pair_agreement(predicted, truth, n, *, sample_size=10000, seed=0):
    """Raw and chance-adjusted Omega from identical uniform pair samples.

    Pair sampling is with replacement; all graph vertices are eligible, including
    those absent from metadata. No dense vertex-pair matrix is constructed.
    """
    if sample_size <= 0:
        raise ValueError("positive Omega sample size required")
    pred_rows, true_rows = memberships(predicted, n), memberships(truth, n)
    if n < 2:
        return {"raw": 1.0, "adjusted": 1.0, "pairs": 0, "sampling": "vacuous"}
    rng = random.Random(seed)
    pred_hist, true_hist = Counter(), Counter()
    observed = 0
    for _ in range(sample_size):
        a = rng.randrange(n)
        b = rng.randrange(n - 1)
        b += b >= a
        kp = len(pred_rows[a] & pred_rows[b])
        kt = len(true_rows[a] & true_rows[b])
        pred_hist[kp] += 1
        true_hist[kt] += 1
        observed += kp == kt
    raw = observed / sample_size
    expected = sum(v * true_hist[k] for k, v in pred_hist.items()) / sample_size**2
    return {"raw": raw, "adjusted": (raw - expected) / (1 - expected)
            if expected < 1 else 1.0, "expected": expected, "pairs": sample_size,
            "sampling": "uniform_unordered_pairs_with_replacement", "seed": seed}
