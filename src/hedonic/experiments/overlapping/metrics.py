"""Evaluation helpers for overlapping community covers (experiments only)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


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


def evaluate_cover(
    predicted: list[list[int]],
    ground_truth: list[list[int]],
    n_vertices: int,
    compute_omega: bool = True,
) -> dict:
    """F1 / Jaccard / precision / recall / optional Omega between covers."""
    pred_sets = [set(c) for c in predicted if c]
    gt_sets = [set(c) for c in ground_truth if c]

    def best_match_f1_precision_recall(A_sets, B_sets):
        total_f1 = total_p = total_r = 0.0
        for a in A_sets:
            best_f1 = best_p = best_r = 0.0
            for b in B_sets:
                inter = len(a & b)
                if inter == 0:
                    continue
                p = inter / len(a)
                r = inter / len(b)
                f1 = 2 * p * r / (p + r)
                if f1 > best_f1:
                    best_f1, best_p, best_r = f1, p, r
            total_f1 += best_f1
            total_p += best_p
            total_r += best_r
        if not A_sets:
            return 0.0, 0.0, 0.0
        n = len(A_sets)
        return total_f1 / n, total_p / n, total_r / n

    def best_match_jaccard(A_sets, B_sets):
        total = 0.0
        for a in A_sets:
            best = 0.0
            for b in B_sets:
                inter = len(a & b)
                union = len(a | b)
                j = inter / union if union > 0 else 0.0
                if j > best:
                    best = j
            total += best
        return total / len(A_sets) if A_sets else 0.0

    f1_p2g, precision, _ = best_match_f1_precision_recall(pred_sets, gt_sets)
    f1_g2p, recall, _ = best_match_f1_precision_recall(gt_sets, pred_sets)
    f1 = (f1_p2g + f1_g2p) / 2.0
    jaccard = (
        best_match_jaccard(pred_sets, gt_sets)
        + best_match_jaccard(gt_sets, pred_sets)
    ) / 2.0

    omega = (
        omega_index(predicted, ground_truth, n_vertices) if compute_omega else None
    )

    return {
        "f1": f1,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "omega": omega,
        "n_predicted_comms": len(pred_sets),
        "n_gt_comms": len(gt_sets),
    }


def omega_index(
    pred: list[list[int]],
    gt: list[list[int]],
    n: int,
) -> float:
    """Vectorized Omega index for overlapping covers."""
    from scipy import sparse

    def co_count(cover, n):
        c_count = len(cover)
        if c_count == 0:
            return np.zeros((n, n), dtype=np.int32)
        rows, cols = [], []
        for c, members in enumerate(cover):
            ms = list(members)
            rows.extend(ms)
            cols.extend([c] * len(ms))
        if not rows:
            return np.zeros((n, n), dtype=np.int32)
        data = np.ones(len(rows), dtype=np.int32)
        m = sparse.csr_matrix((data, (rows, cols)), shape=(n, c_count))
        density = len(rows) / (n * c_count)
        if density > 0.05:
            md = m.toarray().astype(np.float64)
            co = (md @ md.T).astype(np.int32)
        else:
            co = (m @ m.T).toarray().astype(np.int32)
        np.fill_diagonal(co, 0)
        return co

    co_pred = co_count(pred, n)
    co_gt = co_count(gt, n)

    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return 1.0

    iu, iv = np.triu_indices(n, k=1)
    k_pred = co_pred[iu, iv]
    k_gt = co_gt[iu, iv]
    max_k = int(max(k_pred.max(), k_gt.max())) + 1

    observed = np.count_nonzero(k_pred == k_gt) / n_pairs
    n_pred_k = np.bincount(k_pred, minlength=max_k).astype(np.float64)
    n_gt_k = np.bincount(k_gt, minlength=max_k).astype(np.float64)
    expected = float(np.sum(n_pred_k * n_gt_k) / (n_pairs ** 2))

    if abs(1.0 - expected) < 1e-10:
        return 1.0
    return (observed - expected) / (1.0 - expected)


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
