"""Evaluation helpers for overlapping community covers (experiments only).

The historical ``f1`` returned by :func:`evaluate_cover` is the symmetric
best-match community F1.  It remains available for compatibility and is also
reported under the explicit name ``symmetric_best_match_f1``.  New code can
additionally use one-to-one matching, node-membership multilabel scores, and
cover diagnostics from the same function.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Literal

import numpy as np

SingletonMode = Literal["all", "size_ge_2"]
MatchingWeight = Literal["f1", "jaccard"]
MAX_DENSE_MATCHING_CELLS = 5_000_000


def cover_quality(result) -> float | None:
    """Extract Leiden quality from a VertexClustering or VertexCover."""
    q = getattr(result, "quality", None)
    if q is not None:
        return float(q)
    params = getattr(result, "_params", None) or {}
    if "quality" in params:
        return float(params["quality"])
    return None


def partition_to_cover_lists(partition) -> list[list[int]]:
    """Convert a VertexClustering (or flat membership) to community lists."""
    if hasattr(partition, "membership") and partition.membership is not None:
        mem = partition.membership
        # Overlapping VertexCover: membership is list[list[int]]
        if mem and isinstance(mem[0], (list, tuple)):
            return [list(c) for c in partition]
        n_comms = max(mem) + 1 if mem else 0
        return [
            [v for v, c in enumerate(mem) if c == ci]
            for ci in range(n_comms)
        ]
    return [list(c) for c in partition]


def flat_membership_to_cover_init(membership: list[int]) -> list[list[int]]:
    """Disjoint membership vector → overlapping initial_membership format."""
    return [[int(c)] for c in membership]


def singleton_cover(n: int) -> list[list[int]]:
    return [[v] for v in range(n)]


def grand_coalition_cover(n: int) -> list[list[int]]:
    return [list(range(n))]


def total_overlap_cover(n: int, n_communities: int) -> list[list[int]]:
    full = list(range(n))
    return [list(full) for _ in range(max(1, n_communities))]


def _cover_sets(
    cover: Sequence[Sequence[int]],
    singleton_mode: SingletonMode,
) -> list[set[int]]:
    if singleton_mode not in ("all", "size_ge_2"):
        raise ValueError("singleton_mode must be 'all' or 'size_ge_2'")
    minimum_size = 1 if singleton_mode == "all" else 2
    return [members for c in cover if len(members := set(c)) >= minimum_size]


def _pair_scores(
    pred_sets: Sequence[set[int]],
    gt_sets: Sequence[set[int]],
    weight: MatchingWeight,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """Positive pair scores, built from an inverted index (no dense matrix)."""
    if weight not in ("f1", "jaccard"):
        raise ValueError("matching weight must be 'f1' or 'jaccard'")
    vertex_to_gt: dict[int, list[int]] = defaultdict(list)
    for gi, members in enumerate(gt_sets):
        for vertex in members:
            vertex_to_gt[vertex].append(gi)

    intersections: dict[tuple[int, int], int] = defaultdict(int)
    for pi, members in enumerate(pred_sets):
        for vertex in members:
            for gi in vertex_to_gt.get(vertex, ()):
                intersections[(pi, gi)] += 1

    scores: dict[tuple[int, int], float] = {}
    for (pi, gi), inter in intersections.items():
        if weight == "f1":
            score = 2.0 * inter / (len(pred_sets[pi]) + len(gt_sets[gi]))
        else:
            score = inter / (len(pred_sets[pi]) + len(gt_sets[gi]) - inter)
        scores[(pi, gi)] = float(score)
    return scores, dict(intersections)


def _one_to_one_matches(
    pred_sets: Sequence[set[int]],
    gt_sets: Sequence[set[int]],
    *,
    weight: MatchingWeight = "f1",
    pair_data: tuple[
        dict[tuple[int, int], float], dict[tuple[int, int], int]
    ] | None = None,
) -> list[tuple[int, int, float, int]]:
    """Maximum-weight one-to-one matches over positive-overlap components.

    The positive-overlap bipartite graph is solved component by component with
    :func:`scipy.optimize.linear_sum_assignment`. Components too large for a
    bounded dense matrix use SciPy's exact sparse full bipartite matcher with
    one zero-score dummy prediction per GT community. Both paths solve the same
    optional maximum-weight one-to-one assignment.
    """
    from scipy.optimize import linear_sum_assignment

    scores, intersections = (
        pair_data if pair_data is not None else _pair_scores(pred_sets, gt_sets, weight)
    )
    if not scores:
        return []

    pred_to_gt: dict[int, set[int]] = defaultdict(set)
    gt_to_pred: dict[int, set[int]] = defaultdict(set)
    for pi, gi in scores:
        pred_to_gt[pi].add(gi)
        gt_to_pred[gi].add(pi)

    remaining = set(pred_to_gt)
    matches: list[tuple[int, int, float, int]] = []
    while remaining:
        start = remaining.pop()
        component_pred = {start}
        component_gt: set[int] = set()
        queue: deque[tuple[str, int]] = deque([("p", start)])
        while queue:
            side, idx = queue.popleft()
            if side == "p":
                for gi in pred_to_gt[idx] - component_gt:
                    component_gt.add(gi)
                    queue.append(("g", gi))
            else:
                for pi in gt_to_pred[idx] - component_pred:
                    component_pred.add(pi)
                    remaining.discard(pi)
                    queue.append(("p", pi))

        pred_ids = sorted(component_pred)
        gt_ids = sorted(component_gt)
        pred_pos = {idx: pos for pos, idx in enumerate(pred_ids)}
        gt_pos = {idx: pos for pos, idx in enumerate(gt_ids)}
        n_cells = len(pred_ids) * len(gt_ids)
        if n_cells <= MAX_DENSE_MATCHING_CELLS:
            matrix = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
            for pi in pred_ids:
                for gi in pred_to_gt[pi] & component_gt:
                    matrix[pred_pos[pi], gt_pos[gi]] = scores[(pi, gi)]
            rows, cols = linear_sum_assignment(-matrix)
            assignments = zip(rows.tolist(), cols.tolist())
        else:
            from scipy import sparse
            from scipy.sparse.csgraph import min_weight_full_bipartite_matching

            sparse_rows: list[int] = []
            sparse_cols: list[int] = []
            sparse_costs: list[float] = []
            for pi in pred_ids:
                for gi in pred_to_gt[pi] & component_gt:
                    sparse_rows.append(pred_pos[pi])
                    sparse_cols.append(gt_pos[gi])
                    # Positive costs are required because explicit sparse
                    # zeros are removed. Minimizing 2-score maximizes score.
                    sparse_costs.append(2.0 - scores[(pi, gi)])
            # A private dummy prediction lets each GT remain unmatched at
            # score zero while retaining a sparse full-matching formulation.
            for col in range(len(gt_ids)):
                sparse_rows.append(len(pred_ids) + col)
                sparse_cols.append(col)
                sparse_costs.append(2.0)
            cost_matrix = sparse.csr_matrix(
                (sparse_costs, (sparse_rows, sparse_cols)),
                shape=(len(pred_ids) + len(gt_ids), len(gt_ids)),
            )
            rows, cols = min_weight_full_bipartite_matching(cost_matrix)
            assignments = (
                (row, col)
                for row, col in zip(rows.tolist(), cols.tolist())
                if row < len(pred_ids)
            )
        for row, col in assignments:
            pi, gi = pred_ids[row], gt_ids[col]
            score = scores.get((pi, gi), 0.0)
            if score > 0.0:
                matches.append((pi, gi, score, intersections[(pi, gi)]))
    return sorted(matches)


def one_to_one_community_metrics(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    matching_weight: MatchingWeight = "f1",
    singleton_mode: SingletonMode = "all",
) -> dict[str, float | int | str]:
    """One-to-one community precision/recall/F1 with zero unmatched scores.

    Matched-pair precision is averaged over *all predicted* communities and
    matched-pair recall over *all GT* communities.  Thus an unmatched predicted
    community contributes zero precision and an unmatched GT community
    contributes zero recall.  ``matching_f1`` is their harmonic mean.
    """
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    matches = _one_to_one_matches(pred_sets, gt_sets, weight=matching_weight)
    return _one_to_one_metrics_from_matches(
        pred_sets, gt_sets, matches, matching_weight
    )


def _one_to_one_metrics_from_matches(
    pred_sets: Sequence[set[int]],
    gt_sets: Sequence[set[int]],
    matches: Sequence[tuple[int, int, float, int]],
    matching_weight: MatchingWeight,
) -> dict[str, float | int | str]:
    precision = (
        sum(inter / len(pred_sets[pi]) for pi, _gi, _w, inter in matches)
        / len(pred_sets)
        if pred_sets
        else 0.0
    )
    recall = (
        sum(inter / len(gt_sets[gi]) for _pi, gi, _w, inter in matches)
        / len(gt_sets)
        if gt_sets
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_weight = (
        sum(match[2] for match in matches) / max(len(pred_sets), len(gt_sets))
        if pred_sets or gt_sets
        else 1.0
    )
    return {
        "matching_precision": float(precision),
        "matching_recall": float(recall),
        "matching_f1": float(f1),
        "matching_mean_weight": float(mean_weight),
        "matching_weight": matching_weight,
        "n_matched_communities": len(matches),
        "n_unmatched_predicted_comms": len(pred_sets) - len(matches),
        "n_unmatched_gt_comms": len(gt_sets) - len(matches),
    }


def node_membership_multilabel_metrics(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    matching_weight: MatchingWeight = "f1",
    singleton_mode: SingletonMode = "all",
) -> dict[str, float]:
    """Multilabel membership scores after one-to-one community alignment.

    Micro scores count vertex-community assignments.  Macro F1 is the standard
    label-wise multilabel macro average over aligned GT labels plus unmatched
    predicted labels; unmatched labels therefore receive F1 zero.
    """
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    matches = _one_to_one_matches(pred_sets, gt_sets, weight=matching_weight)
    return _node_metrics_from_matches(pred_sets, gt_sets, matches)


def _node_metrics_from_matches(
    pred_sets: Sequence[set[int]],
    gt_sets: Sequence[set[int]],
    matches: Sequence[tuple[int, int, float, int]],
) -> dict[str, float]:
    true_positive = sum(inter for _pi, _gi, _w, inter in matches)
    predicted_positive = sum(len(c) for c in pred_sets)
    gt_positive = sum(len(c) for c in gt_sets)
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / gt_positive if gt_positive else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    n_labels = len(pred_sets) + len(gt_sets) - len(matches)
    macro_sum = sum(
        2.0 * inter / (len(pred_sets[pi]) + len(gt_sets[gi]))
        for pi, gi, _w, inter in matches
    )
    macro_f1 = macro_sum / n_labels if n_labels else 1.0
    return {
        "node_micro_precision": float(precision),
        "node_micro_recall": float(recall),
        "node_micro_f1": float(f1),
        "node_macro_f1": float(macro_f1),
    }


def _best_match_scores(
    source: Sequence[set[int]],
    target: Sequence[set[int]],
) -> list[tuple[float, float, float, float]]:
    _weights, intersections = _pair_scores(source, target, "f1")
    return _best_match_scores_from_intersections(source, target, intersections)


def _best_match_scores_from_intersections(
    source: Sequence[set[int]],
    target: Sequence[set[int]],
    intersections: dict[tuple[int, int], int],
    *,
    reverse: bool = False,
) -> list[tuple[float, float, float, float]]:
    """Directional best scores from sparse positive community intersections."""
    best_scores = [(0.0, 0.0, 0.0, 0.0) for _ in source]
    for (left, right), inter in intersections.items():
        source_idx, target_idx = (right, left) if reverse else (left, right)
        precision = inter / len(source[source_idx])
        recall = inter / len(target[target_idx])
        f1 = 2.0 * inter / (len(source[source_idx]) + len(target[target_idx]))
        jaccard = inter / (
            len(source[source_idx]) + len(target[target_idx]) - inter
        )
        if f1 > best_scores[source_idx][0]:
            best_scores[source_idx] = (f1, precision, recall, jaccard)
    return best_scores


def symmetric_best_match_f1(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    singleton_mode: SingletonMode = "all",
) -> float:
    """Historical macro best-match F1, averaged in both directions."""
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    pred_scores = _best_match_scores(pred_sets, gt_sets)
    gt_scores = _best_match_scores(gt_sets, pred_sets)
    pred_mean = float(np.mean([x[0] for x in pred_scores])) if pred_scores else 0.0
    gt_mean = float(np.mean([x[0] for x in gt_scores])) if gt_scores else 0.0
    return (pred_mean + gt_mean) / 2.0


def symmetric_best_match_metrics(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    singleton_mode: SingletonMode = "all",
) -> dict[str, float | int]:
    """Historical best-match metrics without computing Hungarian alignment."""
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    _weights, intersections = _pair_scores(pred_sets, gt_sets, "f1")
    pred_scores = _best_match_scores_from_intersections(
        pred_sets, gt_sets, intersections
    )
    gt_scores = _best_match_scores_from_intersections(
        gt_sets, pred_sets, intersections, reverse=True
    )
    return _symmetric_metrics_from_scores(
        pred_scores, gt_scores, len(pred_sets), len(gt_sets)
    )


def _symmetric_metrics_from_scores(
    pred_scores: Sequence[tuple[float, float, float, float]],
    gt_scores: Sequence[tuple[float, float, float, float]],
    n_predicted: int,
    n_gt: int,
) -> dict[str, float | int]:
    pred_f1 = float(np.mean([x[0] for x in pred_scores])) if pred_scores else 0.0
    gt_f1 = float(np.mean([x[0] for x in gt_scores])) if gt_scores else 0.0
    pred_jaccard = (
        float(np.mean([x[3] for x in pred_scores])) if pred_scores else 0.0
    )
    gt_jaccard = float(np.mean([x[3] for x in gt_scores])) if gt_scores else 0.0
    return {
        "f1": (pred_f1 + gt_f1) / 2.0,
        "symmetric_best_match_f1": (pred_f1 + gt_f1) / 2.0,
        "jaccard": (pred_jaccard + gt_jaccard) / 2.0,
        "symmetric_best_match_jaccard": (pred_jaccard + gt_jaccard) / 2.0,
        "precision": (
            float(np.mean([x[1] for x in pred_scores])) if pred_scores else 0.0
        ),
        "recall": (
            float(np.mean([x[1] for x in gt_scores])) if gt_scores else 0.0
        ),
        "n_predicted_comms": n_predicted,
        "n_gt_comms": n_gt,
    }


def size_weighted_community_f1(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    singleton_mode: SingletonMode = "all",
) -> float:
    """Symmetric best-match F1 weighted by source-community size."""
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)

    _weights, intersections = _pair_scores(pred_sets, gt_sets, "f1")
    pred_scores = _best_match_scores_from_intersections(
        pred_sets, gt_sets, intersections
    )
    gt_scores = _best_match_scores_from_intersections(
        gt_sets, pred_sets, intersections, reverse=True
    )

    return _size_weighted_f1_from_scores(
        pred_sets, gt_sets, pred_scores, gt_scores
    )


def _size_weighted_f1_from_scores(
    pred_sets: Sequence[set[int]],
    gt_sets: Sequence[set[int]],
    pred_scores: Sequence[tuple[float, float, float, float]],
    gt_scores: Sequence[tuple[float, float, float, float]],
) -> float:
    def directional(
        source: Sequence[set[int]],
        scores: Sequence[tuple[float, float, float, float]],
    ) -> float:
        denominator = sum(len(c) for c in source)
        if denominator == 0:
            return 0.0
        return sum(len(c) * score[0] for c, score in zip(source, scores)) / denominator

    return float(
        (directional(pred_sets, pred_scores) + directional(gt_sets, gt_scores))
        / 2.0
    )


def cover_diagnostics(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    n_vertices: int,
    *,
    singleton_mode: SingletonMode = "all",
) -> dict[str, float | int | None]:
    """Descriptive cover statistics after applying ``singleton_mode``."""
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)

    def stats(cover: Sequence[set[int]], prefix: str) -> dict[str, float | int]:
        sizes = [len(c) for c in cover]
        covered = set().union(*cover) if cover else set()
        memberships = sum(sizes)
        return {
            f"{prefix}_community_count": len(cover),
            f"{prefix}_singleton_fraction": (
                sum(size == 1 for size in sizes) / len(sizes) if sizes else 0.0
            ),
            f"{prefix}_vertices_covered_fraction": (
                len(covered) / n_vertices if n_vertices > 0 else 0.0
            ),
            f"{prefix}_average_community_size": (
                float(np.mean(sizes)) if sizes else 0.0
            ),
            f"{prefix}_median_community_size": (
                float(np.median(sizes)) if sizes else 0.0
            ),
            f"{prefix}_average_memberships_per_vertex": (
                memberships / n_vertices if n_vertices > 0 else 0.0
            ),
        }

    result = {**stats(pred_sets, "predicted"), **stats(gt_sets, "gt")}
    n_gt = len(gt_sets)
    result["community_count_ratio"] = len(pred_sets) / n_gt if n_gt else None
    # Compatibility aliases used by existing experiments.
    result["n_predicted_comms"] = len(pred_sets)
    result["n_gt_comms"] = n_gt
    result["singleton_fraction"] = result["predicted_singleton_fraction"]
    result["vertices_covered_fraction"] = result[
        "predicted_vertices_covered_fraction"
    ]
    result["average_community_size"] = result["predicted_average_community_size"]
    result["median_community_size"] = result["predicted_median_community_size"]
    result["average_memberships_per_vertex"] = result[
        "predicted_average_memberships_per_vertex"
    ]
    return result


def evaluate_cover(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    n_vertices: int,
    compute_omega: bool = True,
    *,
    singleton_mode: SingletonMode = "all",
    matching_weight: MatchingWeight = "f1",
    omega_sample_size: int = 100_000,
    omega_seed: int = 0,
) -> dict:
    """Evaluate a cover with legacy and recommended overlap-aware metrics."""
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    pair_data = _pair_scores(pred_sets, gt_sets, matching_weight)
    _weights, intersections = pair_data
    pred_scores = _best_match_scores_from_intersections(
        pred_sets, gt_sets, intersections
    )
    gt_scores = _best_match_scores_from_intersections(
        gt_sets, pred_sets, intersections, reverse=True
    )
    legacy = _symmetric_metrics_from_scores(
        pred_scores, gt_scores, len(pred_sets), len(gt_sets)
    )
    matches = _one_to_one_matches(
        pred_sets,
        gt_sets,
        weight=matching_weight,
        pair_data=pair_data,
    )

    omega = (
        omega_index(
            pred_sets,
            gt_sets,
            n_vertices,
            sample_size=omega_sample_size,
            seed=omega_seed,
        )
        if compute_omega
        else None
    )
    result = {
        **legacy,
        "omega": omega,
        "omega_method": "sampled_pairwise" if compute_omega else None,
        "omega_sample_size": int(omega_sample_size) if compute_omega else None,
        "omega_seed": int(omega_seed) if compute_omega else None,
        "size_weighted_community_f1": _size_weighted_f1_from_scores(
            pred_sets, gt_sets, pred_scores, gt_scores
        ),
        "singleton_mode": singleton_mode,
    }
    result.update(
        _one_to_one_metrics_from_matches(
            pred_sets, gt_sets, matches, matching_weight
        )
    )
    result.update(_node_metrics_from_matches(pred_sets, gt_sets, matches))
    result.update(cover_diagnostics(pred_sets, gt_sets, n_vertices))
    return result


def omega_index(
    pred: Sequence[Sequence[int]],
    gt: Sequence[Sequence[int]],
    n: int,
    *,
    sample_size: int = 100_000,
    seed: int = 0,
) -> float:
    """Sampled Omega approximation with no dense ``n × n`` allocation.

    ``sample_size`` unordered vertex pairs are drawn uniformly with replacement.
    Memory is O(total memberships + n + sample_size), making the metric safe for
    full DBLP.  Use a fixed ``seed`` for reproducible rescoring.
    """
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    def vertex_memberships(cover: Sequence[Sequence[int]]) -> list[set[int]]:
        memberships = [set() for _ in range(n)]
        for community, members in enumerate(cover):
            for vertex in set(members):
                if 0 <= vertex < n:
                    memberships[vertex].add(community)
        return memberships

    pred_memberships = vertex_memberships(pred)
    gt_memberships = vertex_memberships(gt)
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n, size=int(sample_size), dtype=np.int64)
    second = rng.integers(0, n - 1, size=int(sample_size), dtype=np.int64)
    # Map [0, n-1) onto all vertices except ``first`` without rejection.
    second += second >= first
    k_pred = np.fromiter(
        (len(pred_memberships[u] & pred_memberships[v]) for u, v in zip(first, second)),
        dtype=np.int32,
        count=int(sample_size),
    )
    k_gt = np.fromiter(
        (len(gt_memberships[u] & gt_memberships[v]) for u, v in zip(first, second)),
        dtype=np.int32,
        count=int(sample_size),
    )
    max_k = int(max(k_pred.max(), k_gt.max())) + 1
    observed = np.count_nonzero(k_pred == k_gt) / sample_size
    n_pred_k = np.bincount(k_pred, minlength=max_k).astype(np.float64)
    n_gt_k = np.bincount(k_gt, minlength=max_k).astype(np.float64)
    expected = float(np.sum(n_pred_k * n_gt_k) / (sample_size ** 2))

    if abs(1.0 - expected) < 1e-10:
        return 1.0
    return float((observed - expected) / (1.0 - expected))


def quality_overlapping_cpm(
    g,
    cover: list[list[int]],
    resolution: float,
    weights: list[float] | None = None,
) -> float:
    """Overlapping CPM quality Q = (1/2m) Σ_c [e_c − γ · C(N_c, 2)]."""
    if weights is None:
        w = [1.0] * g.ecount()
    else:
        w = list(weights)

    total_w = sum(w)
    if total_w == 0:
        return 0.0

    quality = 0.0
    for members in cover:
        Nc = len(members)
        if Nc == 0:
            continue
        member_set = set(members)
        ec = 0.0
        for e_idx, (src, tgt) in enumerate(g.get_edgelist()):
            if src in member_set and tgt in member_set:
                ec += w[e_idx]
        quality += ec - resolution * Nc * (Nc - 1) / 2.0

    return quality / (2.0 * total_w)


def in_equilibrium_overlapping(
    g,
    cover: list[list[int]],
    resolution: float,
) -> bool:
    """Nash check for binary join/leave overlapping hedonic incentives."""
    n = g.vcount()
    v_comms: list[set[int]] = [set() for _ in range(n)]
    comm_sets = []
    for c_idx, members in enumerate(cover):
        ms = set(members)
        comm_sets.append(ms)
        for v in members:
            v_comms[v].add(c_idx)

    for v in range(n):
        deg_vc: dict[int, float] = defaultdict(float)
        for u in g.neighbors(v):
            for c in v_comms[u]:
                if u != v:
                    deg_vc[c] += 1.0

        for c_idx, ms in enumerate(comm_sets):
            Nc = len(ms)
            ew = deg_vc.get(c_idx, 0.0)
            if c_idx in v_comms[v]:
                if ew - resolution * (Nc - 1) < 0:
                    return False
            else:
                if ew - resolution * Nc > 0:
                    return False
    return True
