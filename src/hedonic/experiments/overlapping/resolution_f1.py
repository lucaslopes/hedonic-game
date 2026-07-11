"""Full-DBLP F1 vs resolution for multi-phase overlapping community_hedonic.

Sweeps resolution γ over [0, 1], runs::

    community_hedonic(
        n_iterations=-1,
        only_local_moving=False,
        allow_isolation=True,
        max_memberships=K,
    )

where ``K`` is the number of ground-truth communities with more than one node.
Each γ is repeated over multiple seeds (seed-dependent random initial membership)
so F1 can be summarized with a confidence interval. Writes JSON + a line plot
(resolution on x, F1 on y, CI band/error bars).

CLI::

    hedonic-exp overlapping-resolution --smoke --output_dir /tmp/res-f1
    hedonic-exp overlapping-resolution --resolutions 0:1:11 --seeds 0-4 \\
        --output_dir .../dblp_resolution_f1
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.config import DBLP_DIR
from hedonic.experiments.overlapping.dblp_full import load_dblp
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    partition_to_cover_lists,
)
from hedonic.utils import sample_uniform_ints

# Fixed algorithm flags (not CLI knobs) — match the experiment contract.
N_ITERATIONS = -1
ONLY_LOCAL_MOVING = False
ALLOW_ISOLATION = True

DEFAULT_OUTPUT_DIR = Path("overlapping_resolution_f1_results")
DEFAULT_RESOLUTIONS = "0:1:11"  # linspace 0..1 inclusive, 11 points
DEFAULT_SEEDS = "0-4"  # five seeds for CI
DEFAULT_CI_LEVEL = 0.95


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(f"{msg}{elapsed}", flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_gt_communities_gt1(gt: Sequence[Sequence[int]]) -> int:
    """Count ground-truth communities with more than one node."""
    return sum(1 for c in gt if len(c) > 1)


def filter_gt_communities_gt1(
    gt: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Keep only GT communities with more than one node."""
    return [list(c) for c in gt if len(c) > 1]


def resolve_max_memberships(
    max_memberships: int | None,
    gt: Sequence[Sequence[int]],
) -> int:
    """Explicit K, else # GT communities with size > 1 (at least 1)."""
    if max_memberships is not None:
        return max(1, int(max_memberships))
    return max(1, count_gt_communities_gt1(gt))


def parse_resolutions(spec: str) -> list[float]:
    """Parse resolution grid.

    Formats
    -------
    * ``start:stop:n`` — ``np.linspace(start, stop, n)`` (inclusive ends)
    * comma list — ``0,0.25,0.5,0.75,1``
    """
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty resolution spec")
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"resolution range must be start:stop:n, got {spec!r}"
            )
        start, stop, n = float(parts[0]), float(parts[1]), int(parts[2])
        if n < 1:
            raise ValueError("resolution count n must be >= 1")
        return [float(x) for x in np.linspace(start, stop, n)]
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"no resolutions parsed from {spec!r}")
    return vals


def parse_seeds(spec: str) -> list[int]:
    """Parse seed list: ``0-4`` → 0..4, or comma list ``0,1,7``."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("empty seed spec")
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        start, stop = int(a.strip()), int(b.strip())
        if stop < start:
            raise ValueError(f"seed range end < start: {spec!r}")
        return list(range(start, stop + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def seeded_initial_membership(
    n_vertices: int,
    n_communities: int,
    seed: int,
) -> list[int]:
    """Seed-dependent random disjoint labels in ``[0, K)``, labels contiguous.

    Used as overlapping init (flat vector expanded to singleton lists per
    vertex) so multi-seed F1 CIs are not identical duplicates when the
    Leiden binding does not take an RNG seed.
    """
    k = max(1, int(n_communities))
    if k == 1:
        return [0] * n_vertices
    raw = sample_uniform_ints(n_vertices, k - 1, seed).tolist()
    # sample_uniform_ints draws [0, k-1] inclusive when K=k-1 → [0, k-1]
    uniq = sorted(set(raw))
    if len(uniq) == k and uniq[0] == 0 and uniq[-1] == k - 1:
        return [int(x) for x in raw]
    remap = {old: new for new, old in enumerate(uniq)}
    return [remap[int(x)] for x in raw]


def mean_ci(
    values: Sequence[float],
    *,
    confidence: float = DEFAULT_CI_LEVEL,
) -> dict[str, float]:
    """Mean and two-sided CI half-width over seed replicates."""
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_half": float("nan"),
            "n": 0,
            "confidence": float(confidence),
        }
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "se": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "ci_half": 0.0,
            "n": 1,
            "confidence": float(confidence),
        }
    std = float(np.std(arr, ddof=1))
    se = std / float(np.sqrt(n))
    try:
        from scipy import stats

        tcrit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except Exception:  # pragma: no cover - scipy always present with experiments
        tcrit = 1.96
    half = tcrit * se
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_half": half,
        "n": n,
        "confidence": float(confidence),
    }


# ---------------------------------------------------------------------------
# Single run + sweep
# ---------------------------------------------------------------------------


def run_one(
    game: Game,
    gt: Sequence[Sequence[int]],
    *,
    resolution: float,
    seed: int,
    max_memberships: int,
    n_iterations: int = N_ITERATIONS,
    only_local_moving: bool = ONLY_LOCAL_MOVING,
    allow_isolation: bool = ALLOW_ISOLATION,
) -> dict[str, Any]:
    """One (resolution, seed) detection + F1 vs GT (no Omega)."""
    n = game.vcount()
    k = max(1, int(max_memberships))
    init = seeded_initial_membership(n, k, seed)
    t0 = time.perf_counter()
    cover_obj = game.community_hedonic(
        resolution=float(resolution),
        n_iterations=int(n_iterations),
        only_local_moving=bool(only_local_moving),
        allow_isolation=bool(allow_isolation),
        max_memberships=k,
        initial_membership=init,
        seed=int(seed),
    )
    elapsed = time.perf_counter() - t0
    cover = partition_to_cover_lists(cover_obj)
    metrics = evaluate_cover(cover, list(gt), n, compute_omega=False)
    return {
        "resolution": float(resolution),
        "seed": int(seed),
        "f1": float(metrics["f1"]),
        "jaccard": float(metrics["jaccard"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "n_predicted_comms": int(metrics["n_predicted_comms"]),
        "n_gt_comms": int(metrics["n_gt_comms"]),
        "max_memberships": k,
        "n_iterations": int(n_iterations),
        "only_local_moving": bool(only_local_moving),
        "allow_isolation": bool(allow_isolation),
        "wallclock_s": float(elapsed),
        "n_vertices": n,
        "n_edges": int(game.ecount()),
    }


def aggregate_runs(
    runs: Sequence[dict[str, Any]],
    *,
    confidence: float = DEFAULT_CI_LEVEL,
) -> list[dict[str, Any]]:
    """Group by resolution; mean F1 ± CI over seeds."""
    by_res: dict[float, list[dict[str, Any]]] = {}
    for r in runs:
        by_res.setdefault(float(r["resolution"]), []).append(r)

    out: list[dict[str, Any]] = []
    for res in sorted(by_res):
        group = by_res[res]
        f1s = [float(g["f1"]) for g in group]
        stats = mean_ci(f1s, confidence=confidence)
        out.append(
            {
                "resolution": res,
                "f1_mean": stats["mean"],
                "f1_std": stats["std"],
                "f1_se": stats["se"],
                "f1_ci_low": stats["ci_low"],
                "f1_ci_high": stats["ci_high"],
                "f1_ci_half": stats["ci_half"],
                "f1_samples": f1s,
                "seeds": [int(g["seed"]) for g in group],
                "n_seeds": stats["n"],
                "confidence": stats["confidence"],
                "jaccard_mean": float(
                    np.mean([float(g["jaccard"]) for g in group])
                ),
                "wallclock_s_mean": float(
                    np.mean([float(g["wallclock_s"]) for g in group])
                ),
                "max_memberships": int(group[0]["max_memberships"]),
                "n_iterations": int(group[0]["n_iterations"]),
                "only_local_moving": bool(group[0]["only_local_moving"]),
                "allow_isolation": bool(group[0]["allow_isolation"]),
            }
        )
    return out


@dataclass
class ResolutionF1Result:
    """Full experiment payload (serializable)."""

    runs: list[dict[str, Any]] = field(default_factory=list)
    aggregated: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "runs": self.runs,
            "aggregated": self.aggregated,
        }


def run_resolution_f1_experiment(
    game: Game,
    gt: Sequence[Sequence[int]],
    *,
    resolutions: Sequence[float],
    seeds: Sequence[int],
    max_memberships: int | None = None,
    confidence: float = DEFAULT_CI_LEVEL,
) -> ResolutionF1Result:
    """Sweep γ × seeds on a fixed graph; aggregate F1 CIs."""
    gt_eval = filter_gt_communities_gt1(gt)
    k = resolve_max_memberships(max_memberships, gt)
    n_res = len(list(resolutions))
    n_seeds = len(list(seeds))

    log("=" * 55)
    log("  Overlapping resolution × F1 (full multi-phase)")
    log("=" * 55)
    log(f"  n_vertices      : {game.vcount():,}")
    log(f"  n_edges         : {game.ecount():,}")
    log(f"  n_gt_gt1        : {count_gt_communities_gt1(gt)}")
    log(f"  max_memberships : {k}")
    log(f"  n_iterations    : {N_ITERATIONS}")
    log(f"  only_local_moving: {ONLY_LOCAL_MOVING}")
    log(f"  allow_isolation : {ALLOW_ISOLATION}")
    log(f"  resolutions     : {list(resolutions)}")
    log(f"  seeds           : {list(seeds)}")
    log(f"  CI level        : {confidence}")
    log("=" * 55)

    runs: list[dict[str, Any]] = []
    total = n_res * n_seeds
    done = 0
    t_all = time.time()
    for res in resolutions:
        for seed in seeds:
            done += 1
            log(
                f"\n[{done}/{total}] γ={float(res):.6g}  seed={seed}  K={k} …"
            )
            t0 = time.time()
            rec = run_one(
                game,
                gt_eval,
                resolution=float(res),
                seed=int(seed),
                max_memberships=k,
            )
            runs.append(rec)
            log(
                f"  F1={rec['f1']:.4f}  Jaccard={rec['jaccard']:.4f}  "
                f"comms={rec['n_predicted_comms']}  "
                f"wall={rec['wallclock_s']:.3f}s",
                t0,
            )

    aggregated = aggregate_runs(runs, confidence=confidence)
    meta = {
        "n_iterations": N_ITERATIONS,
        "only_local_moving": ONLY_LOCAL_MOVING,
        "allow_isolation": ALLOW_ISOLATION,
        "max_memberships": k,
        "max_memberships_rule": "n_gt_communities_size_gt_1",
        "n_gt_communities_raw": len(gt),
        "n_gt_communities_gt1": count_gt_communities_gt1(gt),
        "resolutions": [float(r) for r in resolutions],
        "seeds": [int(s) for s in seeds],
        "confidence": float(confidence),
        "n_runs": len(runs),
        "n_vertices": int(game.vcount()),
        "n_edges": int(game.ecount()),
        "wallclock_total_s": float(time.time() - t_all),
    }
    log(f"\n[done] {len(runs)} runs in {meta['wallclock_total_s']:.1f}s")
    for row in aggregated:
        log(
            f"  γ={row['resolution']:.4g}  "
            f"F1={row['f1_mean']:.4f}  "
            f"CI95=[{row['f1_ci_low']:.4f}, {row['f1_ci_high']:.4f}]  "
            f"n={row['n_seeds']}"
        )
    return ResolutionF1Result(runs=runs, aggregated=aggregated, meta=meta)


# ---------------------------------------------------------------------------
# Smoke graph (no DBLP)
# ---------------------------------------------------------------------------


def build_smoke_instance(
    *,
    n_blocks: int = 3,
    block_size: int = 8,
    p_in: float = 0.45,
    p_out: float = 0.05,
    seed: int = 0,
) -> tuple[Game, list[list[int]]]:
    """Tiny SBM + planted block GT for CI / smoke (no DBLP)."""
    block_sizes = [block_size] * n_blocks
    pref = [
        [p_in if i == j else p_out for j in range(n_blocks)]
        for i in range(n_blocks)
    ]
    g = ig.Graph.SBM(pref, block_sizes, directed=False)
    g.simplify()
    gt: list[list[int]] = []
    start = 0
    for sz in block_sizes:
        gt.append(list(range(start, start + sz)))
        start += sz
    # One singleton GT community (must be ignored for K)
    gt.append([0])
    return Game(g), gt


# ---------------------------------------------------------------------------
# I/O + plot
# ---------------------------------------------------------------------------


def save_results(result: ResolutionF1Result, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def load_results(path: Path | str) -> ResolutionF1Result:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ResolutionF1Result(
        runs=list(data.get("runs", [])),
        aggregated=list(data.get("aggregated", [])),
        meta=dict(data.get("meta", {})),
    )


def plot_resolution_f1(
    result: ResolutionF1Result | dict[str, Any] | Sequence[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Line plot: resolution (x) vs F1 (y) with CI error bars / band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(result, ResolutionF1Result):
        rows = result.aggregated
        meta = result.meta
    elif isinstance(result, dict) and "aggregated" in result:
        rows = result["aggregated"]
        meta = result.get("meta", {})
    else:
        rows = list(result)
        meta = {}

    rows = sorted(rows, key=lambda r: float(r["resolution"]))
    xs = [float(r["resolution"]) for r in rows]
    ys = [float(r["f1_mean"]) for r in rows]
    yerr_lo = [max(0.0, ys[i] - float(rows[i]["f1_ci_low"])) for i in range(len(rows))]
    yerr_hi = [max(0.0, float(rows[i]["f1_ci_high"]) - ys[i]) for i in range(len(rows))]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, "o-", color="C0", linewidth=2, markersize=6, label="F1 mean")
    ax.fill_between(
        xs,
        [ys[i] - yerr_lo[i] for i in range(len(xs))],
        [ys[i] + yerr_hi[i] for i in range(len(xs))],
        color="C0",
        alpha=0.2,
        label="CI band",
    )
    ax.errorbar(
        xs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="none",
        ecolor="C0",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_xlabel("Resolution γ")
    ax.set_ylabel("F1 (vs ground truth)")
    conf = meta.get("confidence", DEFAULT_CI_LEVEL)
    k = meta.get("max_memberships", "?")
    ax.set_title(
        title
        or (
            f"Overlapping multi-phase hedonic: F1 vs γ "
            f"(K={k}, {conf:.0%} CI over seeds)"
        )
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-resolution",
        description=(
            "Full-DBLP (or --smoke) F1 vs resolution for multi-phase "
            "community_hedonic: n_iterations=-1, only_local_moving=False, "
            "allow_isolation=True, max_memberships=#GT communities with "
            "size>1. Multi-seed F1 confidence intervals + line plot."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default=str(DBLP_DIR),
        help="DBLP root (pkl/raw); ignored with --smoke",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for results JSON + plot",
    )
    parser.add_argument(
        "--resolutions",
        default=DEFAULT_RESOLUTIONS,
        help=(
            "Resolution grid: start:stop:n (linspace) or comma list. "
            "Default 0:1:11 covers γ from 0 to 1 inclusive."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=DEFAULT_SEEDS,
        help="Seed list: a-b inclusive range or comma list (multi-seed CI)",
    )
    parser.add_argument(
        "--max_memberships",
        type=int,
        default=None,
        help=(
            "Cap on communities per vertex. Default: number of ground-truth "
            "communities with more than one node."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CI_LEVEL,
        help="Two-sided CI level for F1 over seeds (e.g. 0.95)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny synthetic SBM only (no DBLP); CI-friendly",
    )
    parser.add_argument(
        "--smoke-blocks",
        type=int,
        default=3,
        help="Planted blocks for --smoke SBM",
    )
    parser.add_argument(
        "--smoke-block-size",
        type=int,
        default=8,
        help="Nodes per block for --smoke SBM",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format",
    )
    parser.add_argument(
        "--results-name",
        default="resolution_f1.json",
        help="Results filename under --output_dir",
    )
    parser.add_argument(
        "--plot-name",
        default=None,
        help="Plot filename (default: resolution_f1.<format>)",
    )
    args = parser.parse_args(argv)

    try:
        resolutions = parse_resolutions(args.resolutions)
        seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        log(f"[error] {exc}")
        return 2

    if not (0.0 < args.confidence < 1.0):
        log("[error] --confidence must be in (0, 1)")
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Tight defaults for smoke if user left production grid
        if args.resolutions == DEFAULT_RESOLUTIONS:
            resolutions = parse_resolutions("0,0.5,1")
        if args.seeds == DEFAULT_SEEDS:
            seeds = parse_seeds("0,1")
        log(
            f"[smoke] synthetic SBM  blocks={args.smoke_blocks}  "
            f"block_size={args.smoke_block_size}"
        )
        game, gt = build_smoke_instance(
            n_blocks=args.smoke_blocks,
            block_size=args.smoke_block_size,
            seed=0,
        )
    else:
        g, gt, _node_map = load_dblp(args.data_dir)
        game = Game(g)

    result = run_resolution_f1_experiment(
        game,
        gt,
        resolutions=resolutions,
        seeds=seeds,
        max_memberships=args.max_memberships,
        confidence=args.confidence,
    )
    result.meta["smoke"] = bool(args.smoke)
    result.meta["data_dir"] = None if args.smoke else str(args.data_dir)
    result.meta["resolutions_spec"] = args.resolutions
    result.meta["seeds_spec"] = args.seeds

    results_path = out_dir / args.results_name
    save_results(result, results_path)
    log(f"\n[done] Results → {results_path}")

    plot_name = args.plot_name or f"resolution_f1.{args.plot_format}"
    plot_path = out_dir / plot_name
    plot_resolution_f1(result, plot_path)
    log(f"[done] Plot    → {plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
