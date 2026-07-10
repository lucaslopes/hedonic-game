"""
DBLP full-graph overlapping validation experiment.

Migrated from tmp/hedonic-overlapping/scripts/dblp_experiment.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.config import DBLP_DIR
from hedonic.experiments.overlapping.metrics import (
    cover_quality,
    evaluate_cover,
    in_equilibrium_overlapping,
    partition_to_cover_lists,
    quality_overlapping_cpm,
)


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(f"{msg}{elapsed}", flush=True)


def _evaluate_no_omega(predicted, ground_truth, n_vertices):
    return evaluate_cover(
        predicted, ground_truth, n_vertices, compute_omega=False
    )


def load_dblp(data_dir: str | Path | None = None):
    """Load DBLP graph and ground-truth communities.

    Supports:
      - data_dir/dblp.pkl cache (g, gt, node_map)
      - data_dir/pkl/com-dblp.ungraph.pkl + com-dblp.all.cmty.pkl
      - data_dir/raw/*.gz or data_dir/*.gz
    """
    data_dir = Path(data_dir) if data_dir is not None else DBLP_DIR
    cache_file = data_dir / "dblp.pkl"
    pkl_dir = data_dir / "pkl"
    raw_dir = data_dir / "raw"

    if cache_file.exists():
        log(f"[load] Loading cache {cache_file} …")
        t = time.time()
        with open(cache_file, "rb") as f:
            g, gt, node_map = pickle.load(f)
        log(
            f"[load] Cache OK — {g.vcount():,} nodes, {g.ecount():,} edges, "
            f"{len(gt):,} GT communities",
            t,
        )
        return g, gt, node_map

    ungraph_pkl = pkl_dir / "com-dblp.ungraph.pkl"
    cmty_pkl = pkl_dir / "com-dblp.all.cmty.pkl"
    if ungraph_pkl.exists() and cmty_pkl.exists():
        log(f"[load] Loading pkl/ caches under {pkl_dir} …")
        t = time.time()
        with open(ungraph_pkl, "rb") as f:
            g = pickle.load(f)
        with open(cmty_pkl, "rb") as f:
            gt = pickle.load(f)
        node_map = {}
        log(
            f"[load] pkl OK — {g.vcount():,} nodes, {g.ecount():,} edges, "
            f"{len(gt):,} GT communities",
            t,
        )
        return g, gt, node_map

    edge_candidates = [
        raw_dir / "com-dblp.ungraph.txt.gz",
        data_dir / "com-dblp.ungraph.txt.gz",
    ]
    cmty_candidates = [
        raw_dir / "com-dblp.all.cmty.txt.gz",
        raw_dir / "com-dblp.cmty.txt.gz",
        data_dir / "com-dblp.cmty.txt.gz",
        data_dir / "com-dblp.all.cmty.txt.gz",
    ]
    edge_file = next((p for p in edge_candidates if p.exists()), None)
    cmty_file = next((p for p in cmty_candidates if p.exists()), None)
    if edge_file is None or cmty_file is None:
        raise FileNotFoundError(
            f"Could not find DBLP edge/community files under {data_dir}. "
            f"Set HEDONIC_DBLP_DIR or pass --data_dir."
        )

    log("[load] Reading edges from .gz …")
    t_total = time.time()
    edges, node_set = [], set()
    with gzip.open(edge_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = map(int, line.split())
            edges.append((u, v))
            node_set.update([u, v])
    log(f"[load] {len(node_set):,} nodes, {len(edges):,} edges read", t_total)

    log("[load] Remapping ids and building igraph …")
    t = time.time()
    node_list = sorted(node_set)
    node_map = {old: new for new, old in enumerate(node_list)}
    edges_r = [(node_map[u], node_map[v]) for u, v in edges]
    g = ig.Graph(n=len(node_list), edges=edges_r, directed=False)
    log("[load] Graph built", t)

    log("[load] Reading ground-truth communities …")
    t = time.time()
    gt = []
    with gzip.open(cmty_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            members = [
                node_map[int(x)] for x in line.split() if int(x) in node_map
            ]
            if len(members) >= 2:
                gt.append(members)
    log(f"[load] {len(gt):,} ground-truth communities", t)

    log("[load] Saving cache …")
    t = time.time()
    with open(cache_file, "wb") as f:
        pickle.dump((g, gt, node_map), f)
    log(f"[load] Cache saved to {cache_file}", t)

    log("[load] Load complete", t_total)
    return g, gt, node_map


def resolve_max_memberships(max_memberships, n_gt_communities: int) -> int:
    """Use explicit K, else number of ground-truth communities (≥ 1)."""
    if max_memberships is not None:
        return max(1, int(max_memberships))
    return max(1, int(n_gt_communities))


def run_experiment(
    g: ig.Graph,
    gt: list,
    resolution: float,
    n_iter: int,
    max_memberships: int | None = None,
):
    log(f"\n[exp] Creating Game (n={g.vcount():,}, m={g.ecount():,}) …")
    t_total = time.time()
    game = Game(g)
    n = game.vcount()
    k = resolve_max_memberships(max_memberships, len(gt))
    results = {
        "params": {
            "resolution": resolution,
            "n_iterations": n_iter,
            "max_memberships": k,
            "n_gt_communities": len(gt),
        }
    }

    log(f"\n[1/3] Hedonic non-overlapping  γ={resolution:.2e} …")
    t0 = time.time()
    leiden_part = game.community_hedonic(
        resolution=resolution,
        n_iterations=-1,
        max_memberships=1,
        only_local_moving=False,
    )
    t_leiden = time.time() - t0
    leiden_cover = partition_to_cover_lists(leiden_part)
    log(f"[1/3] {len(leiden_cover):,} communities in {t_leiden:.1f}s")

    log("[1/3] Metrics vs ground truth …")
    t = time.time()
    metrics_leiden = evaluate_cover(leiden_cover, gt, n, compute_omega=False)
    metrics_leiden["time_s"] = t_leiden
    metrics_leiden["n_communities"] = len(leiden_cover)
    results["leiden_nonoverlapping"] = metrics_leiden
    log(
        f"[1/3] F1={metrics_leiden['f1']:.4f}  "
        f"Jaccard={metrics_leiden['jaccard']:.4f}",
        t,
    )

    log(
        f"\n[2/3] Hedonic overlapping  γ={resolution:.2e}  "
        f"n_iter={n_iter}  max_memberships={k} (n_GT={len(gt)}) …"
    )
    t0 = time.time()
    hedonic_cover_obj = game.community_hedonic(
        resolution=resolution,
        n_iterations=n_iter,
        max_memberships=k,
        only_local_moving=True,
        initial_membership=list(leiden_part.membership),
    )
    t_hedonic = time.time() - t0
    cover_lists = partition_to_cover_lists(hedonic_cover_obj)
    log(f"[2/3] {len(cover_lists):,} communities in {t_hedonic:.1f}s")

    log("[2/3] Metrics vs ground truth …")
    t = time.time()
    metrics_hedonic = evaluate_cover(cover_lists, gt, n, compute_omega=False)
    metrics_hedonic["time_s"] = t_hedonic
    metrics_hedonic["n_communities"] = len(cover_lists)
    metrics_hedonic["max_memberships"] = k
    results["hedonic_overlapping"] = metrics_hedonic
    log(
        f"[2/3] F1={metrics_hedonic['f1']:.4f}  "
        f"Jaccard={metrics_hedonic['jaccard']:.4f}",
        t,
    )

    log("\n[3/3] Nash equilibrium check …")
    t = time.time()
    is_eq = in_equilibrium_overlapping(game, cover_lists, resolution)
    results["hedonic_overlapping"]["in_equilibrium"] = is_eq
    log(f"[3/3] {'In Nash equilibrium' if is_eq else 'Not in equilibrium'}", t)

    log("\n[exp] CPM quality …")
    t = time.time()
    q_leiden = cover_quality(leiden_part)
    if q_leiden is None:
        q_leiden = quality_overlapping_cpm(game, leiden_cover, resolution)
    q_hedonic = cover_quality(hedonic_cover_obj)
    if q_hedonic is None:
        q_hedonic = quality_overlapping_cpm(game, cover_lists, resolution)
    results["leiden_nonoverlapping"]["quality"] = q_leiden
    results["hedonic_overlapping"]["quality"] = q_hedonic
    log(f"[exp] Q non-overlapping : {q_leiden:.6f}", t)
    log(f"[exp] Q overlapping     : {q_hedonic:.6f}")
    log(
        f"[exp] ΔQ = {q_hedonic - q_leiden:+.6f}  "
        f"{'improved' if q_hedonic >= q_leiden else 'worsened'}"
    )
    log("\n[exp] Experiment complete", t_total)
    return results


def resolution_sweep(game, gt, resolutions, n_iter, max_memberships: int | None = None):
    n = game.vcount()
    k = resolve_max_memberships(max_memberships, len(gt))
    sweep = []
    log(
        f"\n[sweep] {len(resolutions)} resolutions: "
        f"{resolutions[0]:.1e} → {resolutions[-1]:.1e}  "
        f"max_memberships={k}"
    )

    for i, res in enumerate(resolutions):
        log(f"\n[sweep {i + 1}/{len(resolutions)}] γ={res:.5e} …")
        t0 = time.time()

        log("  Hedonic non-overlapping …")
        t = time.time()
        part = game.community_hedonic(
            resolution=res,
            n_iterations=-1,
            max_memberships=1,
            only_local_moving=False,
        )
        log(f"  {max(part.membership) + 1:,} communities in {time.time() - t:.1f}s")

        log("  Hedonic overlapping …")
        t = time.time()
        cover_obj = game.community_hedonic(
            resolution=res,
            n_iterations=n_iter,
            max_memberships=k,
            only_local_moving=True,
            initial_membership=list(part.membership),
        )
        cover = partition_to_cover_lists(cover_obj)
        log(f"  {len(cover):,} communities in {time.time() - t:.1f}s")

        log("  Metrics …")
        t = time.time()
        metrics = _evaluate_no_omega(cover, gt, n)
        metrics["resolution"] = res
        metrics["max_memberships"] = k
        metrics["n_iterations"] = n_iter
        q = cover_quality(cover_obj)
        if q is None:
            q = quality_overlapping_cpm(game, cover, res)
        metrics["quality"] = q
        sweep.append(metrics)
        log(
            f"  F1={metrics['f1']:.4f}  Jaccard={metrics['jaccard']:.4f}  "
            f"Q={metrics['quality']:.6f}  total={time.time() - t0:.1f}s",
            t,
        )

    return sweep


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DBLP overlapping community experiment"
    )
    parser.add_argument("--data_dir", default=str(DBLP_DIR))
    parser.add_argument("--resolution", type=float, default=1e-4)
    parser.add_argument(
        "--n_iterations",
        type=int,
        default=-1,
        help=(
            "Overlapping local-moving iteration budget. "
            "Use a negative value (default -1) to iterate until equilibrium."
        ),
    )
    parser.add_argument(
        "--max_memberships",
        type=int,
        default=None,
        help=(
            "Max communities per vertex (overlapping when > 1). "
            "Default: number of ground-truth communities on the full graph."
        ),
    )
    parser.add_argument("--resolution_sweep", action="store_true")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args(argv)

    log("=" * 55)
    log("  DBLP Overlapping Hedonic Game Experiment")
    log("=" * 55)
    log(f"  data_dir   : {args.data_dir}")
    log(f"  output     : {args.output}")
    log(f"  sweep      : {args.resolution_sweep}")
    log(f"  resolution : {args.resolution:.2e}")
    log(f"  n_iterations : {args.n_iterations}")
    mm_log = (
        "auto (n GT communities)"
        if args.max_memberships is None
        else str(args.max_memberships)
    )
    log(f"  max_memberships : {mm_log}")
    if args.n_iterations >= 0:
        log(
            "  WARNING: n_iterations >= 0 may stop before equilibrium; "
            "prefer -1 for paper reproduction."
        )
    log("=" * 55)

    t_start = time.time()
    g, gt, _ = load_dblp(args.data_dir)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.resolution_sweep:
        game = Game(g)
        resolutions = np.logspace(-2, 0, 10)
        sweep = resolution_sweep(
            game, gt, resolutions, args.n_iterations, args.max_memberships
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"resolution_sweep": sweep}, f, indent=2)
    else:
        results = run_experiment(
            g, gt, args.resolution, args.n_iterations, args.max_memberships
        )
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    log(f"\n[done] Results saved to {args.output}")
    log("[done] Total time", t_start)


if __name__ == "__main__":
    main()
