"""Evaluation helpers for overlapping community covers (experiments only).

The historical ``f1`` returned by :func:`evaluate_cover` is the symmetric
best-match community F1.  It remains available for compatibility and is also
reported under the explicit name ``symmetric_best_match_f1``.  New code can
additionally use one-to-one matching, node-membership multilabel scores, and
cover diagnostics from the same function.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from itertools import combinations
from collections.abc import Sequence
import math
from typing import Any, Literal

import numpy as np

SCORE_DEFINITION_VERSION = "canonical_unique_vertex_set_cover_v1"
SCORE_DEFINITION = {
    "version": SCORE_DEFINITION_VERSION,
    "canonicalization": "unique_sorted_vertex_sets",
    "duplicate_communities": "collapsed_before_scoring",
    "matching_weight_default": "f1",
    "matching": "one_to_one_maximum_weight_assignment",
    "matching_precision_denominator": "canonical_predicted_communities",
    "matching_recall_denominator": "canonical_ground_truth_communities",
    "node_micro_denominator": "canonical_vertex_community_incidences",
    "node_macro_denominator": "aligned_gt_labels_plus_unmatched_predicted_labels",
    "singleton_mode_default": "all",
    "singleton_mode_size_ge_2": "drop_communities_of_size_1_before_scoring",
    "omega": "sampled_pairwise_never_dense_n_by_n",
    "empty_predicted_or_gt": "precision_or_recall_zero_when_its_denominator_is_zero",
    "both_empty_matching_mean_weight": 1.0,
    "uncovered_vertices": "absent_from_incidence_numerators_and_counted_in_coverage_diagnostics",
}
SingletonMode = Literal["all", "size_ge_2"]
MatchingWeight = Literal["f1", "jaccard"]
MAX_DENSE_MATCHING_CELLS = 5_000_000

# These are the scalar names accepted by ``Game.evaluate_against`` and are
# useful to notebooks that want the same compact score vector as the smoke
# report.  Keeping the contract beside the metric implementation prevents
# report/tutorial cells from duplicating it.
GAME_EVALUATION_METRICS: tuple[str, ...] = (
    "f1",
    "one_to_one_f1",
    "jaccard",
    "omega",
    "node_micro_f1",
    "size_weighted_community_f1",
)


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


def cover_to_memberships(
    cover: Sequence[Sequence[int]], n_vertices: int
) -> list[list[int]]:
    """Convert community lists to the canonical per-vertex Game state.

    Vertices absent from a predicted cover receive deterministic singleton
    labels.  This is the same representation persisted by
    :meth:`hedonic.Game.community_hedonic`, making the helper convenient for
    notebooks and experiment adapters that already have a cover in list form.
    """
    rows = [[] for _ in range(int(n_vertices))]
    for community_id, community in enumerate(cover):
        for vertex in sorted(set(map(int, community))):
            if 0 <= vertex < n_vertices:
                rows[vertex].append(community_id)
    next_label = len(cover)
    for row in rows:
        if not row:
            row.append(next_label)
            next_label += 1
    labels = sorted({label for row in rows for label in row})
    remap = {old: new for new, old in enumerate(labels)}
    return [[remap[label] for label in row] for row in rows]


def score_cover(
    graph: Any,
    ground_truth: Sequence[Sequence[int]],
    cover: Sequence[Sequence[int]],
    *,
    metrics: Sequence[str] = GAME_EVALUATION_METRICS,
    omega_sample_size: int = 10_000,
    seed: int = 7,
) -> dict[str, float | None]:
    """Score a cover through the public :class:`hedonic.Game` facade.

    The import is local so the metrics package remains usable without making
    the core package eagerly import the optional experiment machinery.
    """
    from hedonic import Game

    game = Game(graph)
    game.memberships = cover_to_memberships(cover, graph.vcount())
    return {
        metric: game.evaluate_against(
            ground_truth,
            method=metric,
            omega_sample_size=omega_sample_size,
            omega_seed=seed,
        )
        for metric in metrics
    }


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
    # Treat a cover as a set of vertex sets.  Duplicate detector labels and
    # repeated members are serialization artifacts, not additional evidence;
    # retaining them would inflate community-weighted metric denominators.
    canonical = {
        tuple(sorted({int(member) for member in community}))
        for community in cover
    }
    return [set(members) for members in sorted(canonical) if len(members) >= minimum_size]


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


def _binary_entropy(probability: float) -> float:
    """Return binary Shannon entropy in bits for a probability in [0, 1]."""
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * math.log2(probability)
        - (1.0 - probability) * math.log2(1.0 - probability)
    )


def _categorical_entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy in bits for a finite probability vector."""
    return float(
        -sum(probability * math.log2(probability) for probability in probabilities if probability > 0.0)
    )


def _pair_conditional_entropy(
    source_size: int,
    target_size: int,
    intersection_size: int,
    universe_size: int,
) -> float:
    """Compute the LFK conditional entropy for one community pair.

    This is the pairwise definition used by the original overlapping-NMI
    implementation (Lancichinetti--Fortunato--Kertész).  The branch selecting
    the smaller conditional direction is part of the reference implementation
    and avoids a directional bias for communities with very different sizes.
    """
    if universe_size <= 0:
        return 0.0
    n = float(universe_size)
    a = (universe_size - source_size - target_size + intersection_size) / n
    b = (target_size - intersection_size) / n
    c = (source_size - intersection_size) / n
    d = intersection_size / n
    # Rounding at the integer boundary can otherwise produce log2(-0.0).
    a, b, c, d = (max(0.0, min(1.0, value)) for value in (a, b, c, d))
    if _binary_entropy(a) + _binary_entropy(d) > _binary_entropy(b) + _binary_entropy(c):
        joint = _categorical_entropy((a, b, c, d))
        return max(0.0, joint - _binary_entropy(target_size / n))
    return _binary_entropy(source_size / n)


def _sparse_best_conditional_entropy(
    source_sets: Sequence[set[int]],
    target_sets: Sequence[set[int]],
    intersections: dict[tuple[int, int], int],
    universe_size: int,
) -> list[float]:
    """Average best-match conditional entropy without a dense pair matrix.

    Positive intersections are enumerated through a vertex inverted index.  A
    source community can also match a target with no shared vertices, so the
    lower envelope of those zero-intersection pairs is considered by target
    community size.  The resulting score is equivalent to the reference
    implementation for set covers, while avoiding the quadratic community-pair
    allocation that is infeasible on DBLP and the other SNAP archives.
    """
    if not source_sets:
        return 0.0
    target_sizes = [len(community) for community in target_sets]
    size_counts = Counter(target_sizes)
    unique_target_sizes = np.asarray(sorted(size_counts), dtype=np.int64)
    target_size_counts = np.asarray(
        [size_counts[int(size)] for size in unique_target_sizes], dtype=np.int64
    )
    conditional_cache: dict[tuple[int, int, int], float] = {}

    def conditional(source_size: int, target_size: int, intersection: int) -> float:
        key = (int(source_size), int(target_size), int(intersection))
        value = conditional_cache.get(key)
        if value is None:
            value = _pair_conditional_entropy(
                source_size, target_size, intersection, universe_size
            )
            conditional_cache[key] = value
        return value

    overlapped_sizes_by_source: dict[int, Counter[int]] = defaultdict(Counter)
    positive_pairs_by_source: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (source_index, target_index), intersection in intersections.items():
        if intersection > 0:
            overlapped_sizes_by_source[source_index][
                len(target_sets[target_index])
            ] += 1
            positive_pairs_by_source[source_index].append((target_index, intersection))

    best_values: list[float] = []
    for source_index, source in enumerate(source_sets):
        source_size = len(source)
        best = float("inf")
        # There is a zero-intersection candidate for a size if at least one
        # target community of that size is not present in the positive pair
        # list for this source community.
        overlapped_sizes = overlapped_sizes_by_source.get(source_index, Counter())
        if unique_target_sizes.size:
            available = np.asarray(
                [
                    int(count) > int(overlapped_sizes.get(int(size), 0))
                    for size, count in zip(unique_target_sizes, target_size_counts)
                ],
                dtype=bool,
            )
            for target_size in unique_target_sizes[available].tolist():
                best = min(
                    best,
                    conditional(source_size, int(target_size), 0),
                )
        for target_index, intersection in positive_pairs_by_source.get(source_index, ()):
            best = min(
                best,
                conditional(source_size, len(target_sets[target_index]), int(intersection)),
            )
        # An empty target cover is handled by the public function.  This guard
        # keeps malformed/degenerate covers from propagating infinity.
        best_values.append(0.0 if not math.isfinite(best) else best)
    return best_values


def overlapping_normalized_mutual_information_lfk(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    n_vertices: int | None = None,
) -> float:
    """Compute the LFK overlapping normalized mutual information (ONMI).

    The CoDeSEG paper delegates ONMI to the LFK/OvpNMI implementation.  This
    experiment-layer implementation keeps the same pairwise entropy definition
    but enumerates only positive community intersections, so it remains usable
    on the full SNAP covers.  ``n_vertices`` controls the universe used for
    entropy; omit it to reproduce the paper's evaluation convention of using
    the union of vertices appearing in either cover.
    """
    pred_sets = _cover_sets(predicted, "all")
    gt_sets = _cover_sets(ground_truth, "all")
    if not pred_sets and not gt_sets:
        return 1.0
    if not pred_sets or not gt_sets:
        return 0.0
    if pred_sets == gt_sets:
        return 1.0

    if n_vertices is None:
        universe = set().union(*(pred_sets + gt_sets))
        universe_size = len(universe)
    else:
        universe_size = int(n_vertices)
    if universe_size <= 0:
        return 0.0

    def positive_intersections(
        left: Sequence[set[int]], right: Sequence[set[int]]
    ) -> dict[tuple[int, int], int]:
        vertex_to_right: dict[int, list[int]] = defaultdict(list)
        for right_index, community in enumerate(right):
            for vertex in community:
                vertex_to_right[vertex].append(right_index)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for left_index, community in enumerate(left):
            for vertex in community:
                for right_index in vertex_to_right.get(vertex, ()):
                    counts[(left_index, right_index)] += 1
        return dict(counts)

    pred_to_gt = positive_intersections(pred_sets, gt_sets)
    gt_to_pred = {
        (gt_index, pred_index): intersection
        for (pred_index, gt_index), intersection in pred_to_gt.items()
    }
    pred_conditional = _sparse_best_conditional_entropy(
        pred_sets, gt_sets, pred_to_gt, universe_size
    )
    gt_conditional = _sparse_best_conditional_entropy(
        gt_sets, pred_sets, gt_to_pred, universe_size
    )

    # LFK normalizes each directional conditional entropy by the binary
    # entropy of its source community before averaging over communities.
    pred_normalized = 0.0
    for community, conditional in zip(pred_sets, pred_conditional):
        entropy = _binary_entropy(len(community) / universe_size)
        pred_normalized += 1.0 if entropy == 0.0 else conditional / entropy
    gt_normalized = 0.0
    for community, conditional in zip(gt_sets, gt_conditional):
        entropy = _binary_entropy(len(community) / universe_size)
        gt_normalized += 1.0 if entropy == 0.0 else conditional / entropy
    value = 1.0 - 0.5 * (
        pred_normalized / len(pred_sets) + gt_normalized / len(gt_sets)
    )
    # Numerical round-off in very small/large communities can move the value a
    # few ulps outside the metric's closed interval.
    return float(max(0.0, min(1.0, value)))


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
        membership_counts = Counter(vertex for community in cover for vertex in community)
        overlapping_nodes = sum(count > 1 for count in membership_counts.values())
        return {
            f"{prefix}_community_count": len(cover),
            f"{prefix}_singleton_count": sum(size == 1 for size in sizes),
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
            f"{prefix}_p95_community_size": (
                float(np.percentile(sizes, 95)) if sizes else 0.0
            ),
            f"{prefix}_max_community_size": max(sizes) if sizes else 0,
            f"{prefix}_average_memberships_per_vertex": (
                memberships / n_vertices if n_vertices > 0 else 0.0
            ),
            f"{prefix}_average_memberships_per_covered_vertex": (
                memberships / len(covered) if covered else 0.0
            ),
            f"{prefix}_max_memberships_per_vertex": max(
                membership_counts.values(), default=0
            ),
            f"{prefix}_overlapping_node_count": overlapping_nodes,
            f"{prefix}_overlapping_node_fraction": (
                overlapping_nodes / len(covered) if covered else 0.0
            ),
        }

    result = {**stats(pred_sets, "predicted"), **stats(gt_sets, "gt")}
    n_gt = len(gt_sets)
    result["community_count_ratio"] = len(pred_sets) / n_gt if n_gt else None
    # Compatibility aliases used by existing experiments.
    result["n_predicted_comms"] = len(pred_sets)
    result["n_gt_comms"] = n_gt
    result["singleton_fraction"] = result["predicted_singleton_fraction"]
    result["singleton_count"] = result["predicted_singleton_count"]
    result["vertices_covered_fraction"] = result[
        "predicted_vertices_covered_fraction"
    ]
    result["average_community_size"] = result["predicted_average_community_size"]
    result["median_community_size"] = result["predicted_median_community_size"]
    result["average_memberships_per_vertex"] = result[
        "predicted_average_memberships_per_vertex"
    ]
    return result


def structural_overlap_metrics(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
    *,
    singleton_mode: SingletonMode = "all",
    max_pair_events: int = 100_000,
) -> dict[str, float | int | bool]:
    """Bounded structural overlap metrics, separate from recovery accuracy.

    ``inclusion_rate`` is the mean fraction of each predicted community covered
    by its best ground-truth match.  ``coverage_rate`` is its inverse
    ground-truth direction.  ``overlapping_rate`` is the mean normalized
    intersection (intersection / smaller community) across predicted community
    pairs that share a node.  ``distribution_rate`` is the fraction of
    predicted memberships assigned to nodes with multiple memberships.

    Pair intersections are accumulated from an inverted index and stop after
    ``max_pair_events`` events; this deliberately avoids dense community-pair
    matrices on the largest SNAP covers.  The truncation flag must accompany
    comparisons where this bound is reached.
    """
    if max_pair_events <= 0:
        raise ValueError("max_pair_events must be positive")
    pred_sets = _cover_sets(predicted, singleton_mode)
    gt_sets = _cover_sets(ground_truth, singleton_mode)
    _scores, intersections = _pair_scores(pred_sets, gt_sets, "f1")
    pred_best = [0.0] * len(pred_sets)
    gt_best = [0.0] * len(gt_sets)
    for (pi, gi), inter in intersections.items():
        pred_best[pi] = max(pred_best[pi], inter / len(pred_sets[pi]))
        gt_best[gi] = max(gt_best[gi], inter / len(gt_sets[gi]))

    memberships: dict[int, list[int]] = defaultdict(list)
    for ci, community in enumerate(pred_sets):
        for vertex in community:
            memberships[vertex].append(ci)
    pair_intersections: dict[tuple[int, int], int] = defaultdict(int)
    events = 0
    truncated = False
    for community_ids in memberships.values():
        if len(community_ids) < 2:
            continue
        for first, second in combinations(community_ids, 2):
            if events >= max_pair_events:
                truncated = True
                break
            pair_intersections[(first, second)] += 1
            events += 1
        if truncated:
            break
    normalized_pair_overlaps = [
        intersection / min(len(pred_sets[first]), len(pred_sets[second]))
        for (first, second), intersection in pair_intersections.items()
    ]
    membership_total = sum(len(community) for community in pred_sets)
    memberships_on_overlapping_nodes = sum(
        len(community_ids)
        for community_ids in memberships.values()
        if len(community_ids) > 1
    )
    return {
        "inclusion_rate": float(np.mean(pred_best)) if pred_best else 0.0,
        "coverage_rate": float(np.mean(gt_best)) if gt_best else 0.0,
        "overlapping_rate": (
            float(np.mean(normalized_pair_overlaps))
            if normalized_pair_overlaps
            else 0.0
        ),
        "distribution_rate": (
            memberships_on_overlapping_nodes / membership_total
            if membership_total
            else 0.0
        ),
        "overlap_pair_count": len(pair_intersections),
        "overlap_pair_events": events,
        "overlap_pair_events_truncated": truncated,
    }


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
        "score_definition_version": SCORE_DEFINITION_VERSION,
        # One-sided best-match F1 in each direction; their mean is ``f1``.
        # reference_to_detected_f1 is the "average F1" of NEO-K-Means/NISE/SSE.
        "reference_to_detected_f1": (sum(sc[0] for sc in gt_scores) / len(gt_scores)) if gt_scores else 0.0,
        "detected_to_reference_f1": (sum(sc[0] for sc in pred_scores) / len(pred_scores)) if pred_scores else 0.0,
        "nf1": normalized_f1_rossetti(predicted, ground_truth),
    }
    result.update(
        _one_to_one_metrics_from_matches(
            pred_sets, gt_sets, matches, matching_weight
        )
    )
    result.update(_node_metrics_from_matches(pred_sets, gt_sets, matches))
    result.update(cover_diagnostics(pred_sets, gt_sets, n_vertices))
    result.update(
        structural_overlap_metrics(
            pred_sets, gt_sets, singleton_mode=singleton_mode
        )
    )
    return result


def normalized_f1_rossetti(
    predicted: Sequence[Sequence[int]],
    ground_truth: Sequence[Sequence[int]],
) -> float:
    """NF1 of Rossetti, Pappalardo & Rinzivillo (2016), as used by ANGEL.

    A faithful port of the reference implementation (``NF1`` in CDlib 0.4.0,
    ``cdlib/evaluation/internal/NF1.py``), including its conventions: each
    detected community is matched to the reference communities with the largest
    member count among its vertices (ties give several matches); a vertex is
    counted only in the first detected community in which it appears; each
    match's F1 is rounded to two decimals. NF1 = mean F1 x coverage / redundancy,
    where coverage is the fraction of reference communities matched and
    redundancy is the number of detected communities per matched reference
    community.
    """
    gt = [list(c) for c in ground_truth]
    if not gt:
        return 0.0
    node_to_gt: dict[int, list[int]] = defaultdict(list)
    for cid, nodes in enumerate(gt):
        for v in nodes:
            node_to_gt[v].append(cid)
    seen: set[int] = set()
    matched: set[int] = set()
    f1s: list[float] = []
    communities = [list(c) for c in predicted]
    for nodes in communities:
        counts: Counter = Counter()
        for v in nodes:
            if v not in seen:
                seen.add(v)
                for cid in node_to_gt.get(v, ()):
                    counts[cid] += 1
        if not counts or not nodes:
            continue
        best = max(counts.values())
        for cid, p in counts.items():
            if p == best:
                matched.add(cid)
                precision, recall = p / len(nodes), p / len(gt[cid])
                f1s.append(float("%.2f" % (2 * precision * recall / (precision + recall))))
    if not f1s or not matched:
        return 0.0
    coverage = len(matched) / len(gt)
    redundancy = len(communities) / len(matched)
    return (sum(f1s) / len(f1s)) * coverage / redundancy


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

    # Accumulate internal edge weight through memberships.  This is equivalent
    # to the former community-by-edge scan but avoids O(|C| * m) work.
    community_sets = [set(members) for members in cover]
    vertex_communities: dict[int, set[int]] = defaultdict(set)
    for community_index, members in enumerate(community_sets):
        for vertex in members:
            vertex_communities[vertex].add(community_index)
    internal_weights = [0.0] * len(community_sets)
    for edge_index, (source, target) in enumerate(g.get_edgelist()):
        for community_index in vertex_communities[source] & vertex_communities[target]:
            internal_weights[community_index] += w[edge_index]

    quality = sum(
        internal_weights[community_index]
        - resolution * len(members) * (len(members) - 1) / 2.0
        for community_index, members in enumerate(community_sets)
        if members
    )

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
