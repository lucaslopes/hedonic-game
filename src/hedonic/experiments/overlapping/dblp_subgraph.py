"""
DBLP subgraph experiment around ground-truth communities.

Migrated from tmp/hedonic-overlapping/scripts/subgraph_experiment.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from hedonic.experiments.config import DBLP_DIR
from hedonic.experiments.overlapping.dblp_full import load_dblp

ALL_METHODS = [
    "leiden",
    "hedonic_v1",
    "hedonic_v2",
    "singleton",
    "grand_coalition",
    "total_overlap",
]
METHODS = list(ALL_METHODS)


def log(msg, t0=None):
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(f"{msg}{elapsed}", flush=True)


def extract_subgraph(g, seed_nodes, levels):
    """Extract subgraph with `levels` hops around seed_nodes.

    Returns (subgraph, old2new, nodes_sorted).
    """
    nodes = set(seed_nodes)
    frontier = set(seed_nodes)

    for _ in range(levels):
        new_frontier = set()
        for v in frontier:
            for u in g.neighbors(v):
                if u not in nodes:
                    new_frontier.add(u)
        nodes.update(new_frontier)
        frontier = new_frontier

    nodes_sorted = sorted(nodes)
    old2new = {old: new for new, old in enumerate(nodes_sorted)}
    subg = g.induced_subgraph(nodes_sorted)
    return subg, old2new, nodes_sorted


def resolve_max_memberships(max_memberships, n_gt_communities: int) -> int:
    """Use explicit K, else number of ground-truth communities (≥ 1)."""
    if max_memberships is not None:
        return max(1, int(max_memberships))
    return max(1, int(n_gt_communities))


def build_covers(subg, n_gt_in_subgraph, resolution, n_iterations, max_memberships):
    """Generate reference covers via Game.community_hedonic(max_memberships=...).

    Always run community_hedonic with n_iterations negative (e.g. -1) so local
    moving continues until equilibrium. When max_memberships is None, K is set
    to the number of ground-truth communities present in the subgraph.
    """
    from hedonic import Game
    from hedonic.experiments.overlapping.metrics import (
        grand_coalition_cover,
        partition_to_cover_lists,
        singleton_cover,
        total_overlap_cover,
    )

    n_sub = subg.vcount()
    game = Game(subg)
    covers, timings = {}, {}
    k = resolve_max_memberships(max_memberships, n_gt_in_subgraph)

    if resolution is None:
        resolution = game.density()

    # Baseline: full Leiden-style run through community_hedonic (disjoint)
    t = time.time()
    part = game.community_hedonic(
        resolution=resolution,
        n_iterations=-1,
        max_memberships=1,
        local_move_only=False,
    )
    t_leiden = time.time() - t
    if "leiden" in METHODS:
        covers["leiden"] = partition_to_cover_lists(part)
        timings["leiden"] = t_leiden

    if "hedonic_v1" in METHODS:
        # Primary exploratory: local-moving-only overlapping hedonic
        t = time.time()
        cover_obj = game.community_hedonic(
            resolution=resolution,
            n_iterations=n_iterations,
            max_memberships=k,
            local_move_only=True,
            initial_membership=list(part.membership),
        )
        covers["hedonic_v1"] = partition_to_cover_lists(cover_obj)
        timings["hedonic_v1"] = time.time() - t

    if "hedonic_v2" in METHODS:
        # Full multi-phase overlapping via community_hedonic
        t = time.time()
        cover_obj_v2 = game.community_hedonic(
            resolution=resolution,
            max_memberships=k,
            n_iterations=n_iterations,
            local_move_only=False,
        )
        covers["hedonic_v2"] = partition_to_cover_lists(cover_obj_v2)
        timings["hedonic_v2"] = time.time() - t

    if "singleton" in METHODS:
        covers["singleton"] = singleton_cover(n_sub)
    if "grand_coalition" in METHODS:
        covers["grand_coalition"] = grand_coalition_cover(n_sub)
    if "total_overlap" in METHODS:
        covers["total_overlap"] = total_overlap_cover(
            n_sub, max(1, n_gt_in_subgraph)
        )

    return covers, timings, resolution, k


def score_covers(covers, gt_target, n_sub, compute_omega=False):
    from hedonic.experiments.overlapping.metrics import evaluate_cover

    return {
        name: evaluate_cover(
            cover, gt_target, n_sub, compute_omega=compute_omega
        )
        for name, cover in covers.items()
    }


def run_one(
    g_full, gt_community, all_gt, levels, resolution, n_iterations, max_memberships, comm_idx
):
    t0 = time.time()
    subg, old2new, new2old = extract_subgraph(g_full, gt_community, levels)
    n_sub = subg.vcount()
    m_sub = subg.ecount()
    log(f"  subgraph: {n_sub:,} nodes, {m_sub:,} edges  [{time.time() - t0:.1f}s]")

    if n_sub < 3 or m_sub == 0:
        log("  subgraph too small, skipping.")
        return None, None

    gt_target = [[old2new[v] for v in gt_community if v in old2new]]

    node_set = set(new2old)
    n_gt_in_subgraph = sum(
        1 for comm in all_gt if len([v for v in comm if v in node_set]) >= 2
    )

    covers, timings, resolution, k = build_covers(
        subg, n_gt_in_subgraph, resolution, n_iterations, max_memberships
    )
    metrics = score_covers(covers, gt_target, n_sub, compute_omega=False)

    for name in METHODS:
        m = metrics[name]
        log(
            f"  {name:16s}  {len(covers[name]):5,} communities  F1={m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"[{timings.get(name, 0.0):.1f}s]"
        )

    has_delta1 = "leiden" in METHODS and "hedonic_v1" in METHODS
    has_delta2 = "leiden" in METHODS and "hedonic_v2" in METHODS
    delta_f1 = (
        metrics["hedonic_v1"]["f1"] - metrics["leiden"]["f1"] if has_delta1 else None
    )
    delta_f1_v2 = (
        metrics["hedonic_v2"]["f1"] - metrics["leiden"]["f1"] if has_delta2 else None
    )
    msg1 = f"ΔF1 (v1-Leiden) = {delta_f1:+.4f}" if has_delta1 else "ΔF1 (v1-Leiden) = n/a"
    msg2 = f"ΔF1 (v2-Leiden) = {delta_f1_v2:+.4f}" if has_delta2 else "ΔF1 (v2-Leiden) = n/a"
    log(f"  {msg1}   {msg2}")
    log(f"  max_memberships={k}  (GT communities in subgraph={n_gt_in_subgraph})")

    result = {
        "community_idx": comm_idx,
        "gt_size": len(gt_community),
        "subgraph_nodes": n_sub,
        "subgraph_edges": m_sub,
        "levels": levels,
        "resolution": resolution,
        "n_iterations": n_iterations,
        "n_gt_in_subgraph": n_gt_in_subgraph,
        "max_memberships": k,
        **{
            name: {
                **metrics[name],
                "n_communities": len(covers[name]),
                "time_s": timings.get(name, 0.0),
            }
            for name in METHODS
        },
        "delta_f1": delta_f1,
        "delta_f1_v2": delta_f1_v2,
    }
    cover_record = {
        "community_idx": comm_idx,
        "gt_size": len(gt_community),
        "subgraph_nodes": n_sub,
        "subgraph_edges": m_sub,
        "levels": levels,
        "resolution": resolution,
        "n_iterations": n_iterations,
        "n_gt_in_subgraph": n_gt_in_subgraph,
        "max_memberships": k,
        "covers": covers,
        "ground_truth": gt_target,
    }
    return result, cover_record


def find_overlapping_communities(
    gt, min_overlap=2, min_overlap_ratio=0.0, min_size=5, max_size=200
):
    """Return (idx, partners) for GT communities with real overlap."""
    vertex_to_comms = {}
    for i, comm in enumerate(gt):
        for v in comm:
            vertex_to_comms.setdefault(v, []).append(i)

    gt_sizes = [len(c) for c in gt]
    result = []
    for i, comm in enumerate(gt):
        if not (min_size <= len(comm) <= max_size):
            continue
        partners = {}
        for v in comm:
            for j in vertex_to_comms.get(v, []):
                if j != i:
                    partners[j] = partners.get(j, 0) + 1
        valid = {
            j: cnt
            for j, cnt in partners.items()
            if cnt >= min_overlap
            and cnt / min(len(comm), gt_sizes[j]) >= min_overlap_ratio
        }
        if valid:
            result.append((i, valid))
    return result


def run_one_overlapping(
    g_full, gt, comm_idx, partners, levels, resolution, n_iterations, max_memberships
):
    seed_nodes = set(gt[comm_idx])

    t0 = time.time()
    subg, old2new, new2old = extract_subgraph(g_full, seed_nodes, levels)
    n_sub = subg.vcount()
    m_sub = subg.ecount()
    log(
        f"  subgraph: {n_sub:,} nodes, {m_sub:,} edges  "
        f"(seed={len(seed_nodes)}, {len(partners)} partners, {levels}-hop)  "
        f"[{time.time() - t0:.1f}s]"
    )

    if n_sub < 3 or m_sub == 0:
        log("  subgraph too small, skipping.")
        return None, None

    node_set = set(new2old)
    n_gt_in_subgraph = sum(
        1 for comm in gt if len([v for v in comm if v in node_set]) >= 2
    )

    gt_target = [[old2new[v] for v in gt[comm_idx] if v in old2new]]

    covers, timings, resolution, k = build_covers(
        subg, n_gt_in_subgraph, resolution, n_iterations, max_memberships
    )
    metrics = score_covers(covers, gt_target, n_sub, compute_omega=False)

    for name in METHODS:
        m = metrics[name]
        log(
            f"  {name:16s}  {len(covers[name]):5,} communities  F1={m['f1']:.4f}  "
            f"[{timings.get(name, 0.0):.1f}s]"
        )

    has_delta1 = "leiden" in METHODS and "hedonic_v1" in METHODS
    has_delta2 = "leiden" in METHODS and "hedonic_v2" in METHODS
    delta_f1 = (
        metrics["hedonic_v1"]["f1"] - metrics["leiden"]["f1"] if has_delta1 else None
    )
    delta_f1_v2 = (
        metrics["hedonic_v2"]["f1"] - metrics["leiden"]["f1"] if has_delta2 else None
    )
    shared = sum(partners.values())
    msg1 = f"ΔF1(v1)={delta_f1:+.4f}" if has_delta1 else "ΔF1(v1)=n/a"
    msg2 = f"ΔF1(v2)={delta_f1_v2:+.4f}" if has_delta2 else "ΔF1(v2)=n/a"
    log(
        f"  {msg1}  {msg2}  partners={len(partners)}  shared_nodes={shared}"
    )
    log(f"  max_memberships={k}  (GT communities in subgraph={n_gt_in_subgraph})")

    result = {
        "community_idx": comm_idx,
        "gt_size": len(gt[comm_idx]),
        "n_partners": len(partners),
        "shared_nodes": shared,
        "subgraph_nodes": n_sub,
        "subgraph_edges": m_sub,
        "levels": levels,
        "resolution": resolution,
        "n_iterations": n_iterations,
        "n_gt_in_subgraph": n_gt_in_subgraph,
        "max_memberships": k,
        **{
            name: {
                **metrics[name],
                "n_communities": len(covers[name]),
                "time_s": timings.get(name, 0.0),
            }
            for name in METHODS
        },
        "delta_f1": delta_f1,
        "delta_f1_v2": delta_f1_v2,
    }
    cover_record = {
        "community_idx": comm_idx,
        "gt_size": len(gt[comm_idx]),
        "n_partners": len(partners),
        "shared_nodes": shared,
        "subgraph_nodes": n_sub,
        "subgraph_edges": m_sub,
        "levels": levels,
        "resolution": resolution,
        "n_iterations": n_iterations,
        "n_gt_in_subgraph": n_gt_in_subgraph,
        "max_memberships": k,
        "covers": covers,
        "ground_truth": gt_target,
    }
    return result, cover_record


def main(argv=None):
    parser = argparse.ArgumentParser(description="DBLP subgraph overlapping experiment")
    parser.add_argument("--data_dir", default=str(DBLP_DIR))
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument(
        "--density_resolution",
        action="store_true",
        help="Ignore --resolution and use each subgraph's density as γ",
    )
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=-1,
        help=(
            "Leiden/hedonic iteration budget for overlapping methods. "
            "Use a negative value (default -1) to iterate until equilibrium."
        ),
    )
    parser.add_argument(
        "--max_memberships",
        type=int,
        default=None,
        help=(
            "Max communities per vertex for hedonic_v1/v2. "
            "Default: number of ground-truth communities present in each subgraph "
            "(≥2 nodes of a GT community inside the L-hop window)."
        ),
    )
    parser.add_argument(
        "--methods",
        default=",".join(ALL_METHODS),
        help="Comma-separated subset of: " + ",".join(ALL_METHODS),
    )
    parser.add_argument(
        "--n_communities",
        type=int,
        default=20,
        help="Number of GT communities to test",
    )
    parser.add_argument(
        "--community_idx",
        type=int,
        default=None,
        help="Specific GT community index",
    )
    parser.add_argument(
        "--overlapping_only",
        action="store_true",
        help="Select only communities with real GT overlap",
    )
    parser.add_argument(
        "--min_overlap",
        type=int,
        default=2,
        help="Min shared nodes to count as overlap",
    )
    parser.add_argument(
        "--min_overlap_ratio",
        type=float,
        default=0.0,
        help="Min overlap coefficient: intersection/min(|i|,|j|)",
    )
    parser.add_argument("--output", default="results/subgraph_experiment.json")
    parser.add_argument(
        "--covers_cache",
        default=None,
        help="Pickle cache path for raw covers "
        "(default: same stem as --output with _covers.pkl)",
    )
    args = parser.parse_args(argv)
    resolution = None if args.density_resolution else args.resolution

    selected = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = [m for m in selected if m not in ALL_METHODS]
    if invalid:
        parser.error(f"Invalid --methods: {invalid}. Options: {ALL_METHODS}")
    METHODS[:] = selected

    log("=" * 55)
    log("  DBLP Subgraph Experiment")
    log("=" * 55)
    log(f"  levels        : {args.levels}")
    log(
        f"  resolution    : "
        f"{'subgraph density' if resolution is None else f'{resolution:.2e}'}"
    )
    log(f"  n_iterations  : {args.n_iterations}")
    mm_log = (
        "auto (n GT communities in subgraph)"
        if args.max_memberships is None
        else str(args.max_memberships)
    )
    log(f"  max_memberships: {mm_log}")
    log(f"  methods       : {METHODS}")
    if args.n_iterations >= 0:
        log(
            "  WARNING: n_iterations >= 0 may stop before equilibrium; "
            "prefer -1 for paper reproduction."
        )
    log("=" * 55)

    cache = Path(args.data_dir) / "dblp.pkl"
    if cache.exists():
        log(f"[load] {cache} …")
        t0 = time.time()
        with open(cache, "rb") as f:
            g, gt, node_map = pickle.load(f)
        log(f"[load] {g.vcount():,} nodes, {g.ecount():,} edges, {len(gt):,} GT", t0)
    else:
        g, gt, node_map = load_dblp(args.data_dir)

    if not args.overlapping_only:
        if args.community_idx is not None:
            indices = [args.community_idx]
        else:
            candidates = [i for i, c in enumerate(gt) if 5 <= len(c) <= 200]
            rng = np.random.default_rng(42)
            indices = rng.choice(
                candidates,
                size=min(args.n_communities, len(candidates)),
                replace=False,
            ).tolist()

    results = []
    covers_cache = []

    if args.overlapping_only:
        log(
            f"\n[select] Finding communities with real overlap "
            f"(min_overlap={args.min_overlap}) …"
        )
        t_sel = time.time()
        overlapping_list = find_overlapping_communities(
            gt,
            min_overlap=args.min_overlap,
            min_overlap_ratio=args.min_overlap_ratio,
            min_size=5,
            max_size=200,
        )
        log(
            f"[select] {len(overlapping_list):,} overlapping communities  "
            f"[{time.time() - t_sel:.1f}s]"
        )
        rng = np.random.default_rng(42)
        chosen = rng.choice(
            len(overlapping_list),
            size=min(args.n_communities, len(overlapping_list)),
            replace=False,
        )
        overlapping_sample = [overlapping_list[i] for i in chosen]
        log(f"[select] Sampled {len(overlapping_sample)} communities\n")

        for rank, (idx, partners) in enumerate(overlapping_sample):
            log(
                f"[{rank + 1}/{len(overlapping_sample)}] community {idx}  "
                f"({len(gt[idx])} GT nodes, {len(partners)} partners)"
            )
            t_comm = time.time()
            result, cover_record = run_one_overlapping(
                g,
                gt,
                idx,
                partners,
                args.levels,
                resolution,
                args.n_iterations,
                args.max_memberships,
            )
            if result:
                results.append(result)
                covers_cache.append(cover_record)
            log(f"  total: {time.time() - t_comm:.1f}s\n")
    else:
        gamma_log = (
            "subgraph density" if resolution is None else f"{resolution:.2e}"
        )
        log(f"\n[exp] {len(indices)} communities  levels={args.levels}  γ={gamma_log}\n")

        for rank, idx in enumerate(indices):
            gt_comm = gt[idx]
            log(f"[{rank + 1}/{len(indices)}] community {idx}  ({len(gt_comm)} GT nodes)")
            t_comm = time.time()
            result, cover_record = run_one(
                g,
                gt_comm,
                gt,
                args.levels,
                resolution,
                args.n_iterations,
                args.max_memberships,
                idx,
            )
            if result:
                results.append(result)
                covers_cache.append(cover_record)
            log(f"  total: {time.time() - t_comm:.1f}s\n")

    if results:
        log("=" * 55)
        for name in METHODS:
            f1s = [r[name]["f1"] for r in results]
            log(f"  {name:16s}  mean F1 = {np.mean(f1s):.4f}")
        if "leiden" in METHODS and "hedonic_v1" in METHODS:
            n_improved_v1 = sum(1 for r in results if r["delta_f1"] >= 0)
            log(
                f"  mean ΔF1 (v1-Leiden) : "
                f"{np.mean([r['delta_f1'] for r in results]):+.4f}"
                f"   improved in {n_improved_v1}/{len(results)}"
            )
        if "leiden" in METHODS and "hedonic_v2" in METHODS:
            n_improved_v2 = sum(1 for r in results if r["delta_f1_v2"] >= 0)
            log(
                f"  mean ΔF1 (v2-Leiden) : "
                f"{np.mean([r['delta_f1_v2'] for r in results]):+.4f}"
                f"   improved in {n_improved_v2}/{len(results)}"
            )
        log("=" * 55)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"\n[done] Results in {args.output}")

    covers_cache_path = args.covers_cache or str(
        Path(args.output).with_name(Path(args.output).stem + "_covers.pkl")
    )
    Path(covers_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(covers_cache_path, "wb") as f:
        pickle.dump(covers_cache, f)
    log(f"[done] Covers cache in {covers_cache_path}")


if __name__ == "__main__":
    main()
