"""The ground-truth disjoint oracle: a supervised control that reads the reference cover but may answer only a partition.

It measures how much of a benchmark score an overlap-blind answer can reach when it knows the answer: every covered
vertex is placed in exactly one of its own reference communities, so each answer block is a subset of one reference
community. Three assignment rules give starting partitions:

``intact``    pack the reference communities greedily, most intact first (the share of a community's members still
              unassigned; smaller first on a tie), each taking all of its unassigned members;
``largest``   every vertex joins its largest reference community;
``smallest``  every vertex joins its smallest reference community.

Each start is improved by single-vertex moves between a vertex's own communities under a separable surrogate of the
symmetric best-match F1, ``(F/|C| + F/|P|) / 2`` with ``F`` the summed F1 of each block against its own community, ``|C|``
the reference communities and ``|P|`` the nonempty blocks. The answer is the candidate (start or improved start) with
the highest exact symmetric F1 against the reference. It is the best of this family, a lower bound on the best
disjoint score rather than a proven optimum; it is deterministic and ignores the graph's edges.

Vertices in no reference community are left out of the answer (the paper's scoring filter drops them anyway).
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Any

from hedonic.experiments.overlapping.metrics import symmetric_best_match_f1

RULES = ("intact", "largest", "smallest")
MAX_PASSES = 50


def _communities(reference: Sequence[Sequence[int]]) -> list[list[int]]:
    """The reference as distinct, nonempty, sorted vertex lists (duplicates carry no extra evidence)."""
    return [list(c) for c in sorted({tuple(sorted({int(v) for v in community})) for community in reference} - {()})]


def _memberships(communities: list[list[int]]) -> dict[int, list[int]]:
    member_of: dict[int, list[int]] = {}
    for index, community in enumerate(communities):
        for vertex in community:
            member_of.setdefault(vertex, []).append(index)
    return member_of


def start(rule: str, communities: list[list[int]], member_of: dict[int, list[int]]) -> dict[int, int]:
    """The starting assignment vertex -> community index of one rule."""
    sizes = [len(c) for c in communities]
    if rule == "largest":
        return {v: min(cs, key=lambda c: (-sizes[c], c)) for v, cs in member_of.items()}
    if rule == "smallest":
        return {v: min(cs, key=lambda c: (sizes[c], c)) for v, cs in member_of.items()}
    if rule != "intact":
        raise ValueError(f"unknown rule {rule!r}; choose from {', '.join(RULES)}")
    remaining = list(sizes)
    heap = [(-1.0, sizes[c], c) for c in range(len(communities))]
    heapq.heapify(heap)
    assignment: dict[int, int] = {}
    while heap:
        key, size, c = heapq.heappop(heap)
        if remaining[c] == 0:
            continue
        share = remaining[c] / size
        if -key != share:  # stale: other communities took some of its members since it was pushed
            heapq.heappush(heap, (-share, size, c))
            continue
        for v in communities[c]:
            if v not in assignment:
                assignment[v] = c
                for other in member_of[v]:
                    remaining[other] -= 1
    return assignment


def _f1(size: int, held: int) -> float:
    return 2.0 * held / (size + held) if held else 0.0


def improve(assignment: dict[int, int], communities: list[list[int]], member_of: dict[int, list[int]],
            max_passes: int = MAX_PASSES) -> tuple[dict[int, int], dict[str, Any]]:
    """Single-vertex moves between a vertex's own communities that raise the surrogate, until none does."""
    assignment = dict(assignment)
    sizes = [len(c) for c in communities]
    held = [0] * len(communities)
    for c in assignment.values():
        held[c] += 1
    total = sum(_f1(s, h) for s, h in zip(sizes, held))
    blocks = sum(1 for h in held if h)
    n_ref = len(communities)

    def surrogate(f: float, p: int) -> float:
        return 0.5 * (f / n_ref + (f / p if p else 0.0))

    movable = sorted(v for v, cs in member_of.items() if len(cs) > 1)
    moves = passes = 0
    before = surrogate(total, blocks)
    for passes in range(1, max_passes + 1):
        moved = 0
        for v in movable:
            a = assignment[v]
            leave = _f1(sizes[a], held[a] - 1) - _f1(sizes[a], held[a])
            current = surrogate(total, blocks)
            best, best_value = None, current
            for b in member_of[v]:
                if b == a:
                    continue
                f = total + leave + _f1(sizes[b], held[b] + 1) - _f1(sizes[b], held[b])
                p = blocks - (held[a] == 1) + (held[b] == 0)
                value = surrogate(f, p)
                if value > best_value + 1e-15 * max(1.0, abs(best_value)):
                    best, best_value = b, value
            if best is not None:
                total += leave + _f1(sizes[best], held[best] + 1) - _f1(sizes[best], held[best])
                blocks += -(held[a] == 1) + (held[best] == 0)
                held[a] -= 1
                held[best] += 1
                assignment[v] = best
                moved += 1
        moves += moved
        if not moved:
            break
    return assignment, {"moves": moves, "passes": passes, "surrogate_before": before,
                        "surrogate_after": surrogate(total, blocks)}


def _blocks(assignment: dict[int, int]) -> list[list[int]]:
    grouped: dict[int, list[int]] = {}
    for v, c in assignment.items():
        grouped.setdefault(c, []).append(v)
    return sorted(sorted(block) for block in grouped.values())


def oracle_partition(reference: Sequence[Sequence[int]], *, rules: Sequence[str] = RULES,
                     max_passes: int = MAX_PASSES) -> tuple[list[list[int]], dict[str, Any]]:
    """The best disjoint answer of the family and a report of every candidate (rule, improved, symmetric F1)."""
    communities = _communities(reference)
    if not communities:
        return [], {"chosen": None, "candidates": [], "reference_communities": 0, "covered_vertices": 0}
    member_of = _memberships(communities)
    candidates = []
    best: tuple[float, list[list[int]], dict[str, Any]] | None = None
    for rule in rules:
        initial = start(rule, communities, member_of)
        improved, search = improve(initial, communities, member_of, max_passes=max_passes)
        for label, assignment, details in ((rule, initial, {}), (f"{rule}+moves", improved, search)):
            blocks = _blocks(assignment)
            f1 = symmetric_best_match_f1(blocks, communities)
            entry = {"candidate": label, "f1": f1, "blocks": len(blocks), **details}
            candidates.append(entry)
            if best is None or f1 > best[0]:
                best = (f1, blocks, entry)
    assert best is not None
    return best[1], {"chosen": best[2]["candidate"], "f1": best[0], "candidates": candidates,
                     "reference_communities": len(communities), "covered_vertices": len(member_of),
                     "multi_membership_vertices": sum(1 for cs in member_of.values() if len(cs) > 1)}
