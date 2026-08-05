"""Wallclock scaling of overlapping hedonic as subnetworks grow toward full DBLP.

Times two ``community_hedonic`` configurations to equilibrium on increasingly
large induced subnetworks (L-hop around a GT seed on DBLP, or synthetic SBMs):

* ``local_move_only=True``  (hedonic local-moving only)
* ``local_move_only=False`` (full multi-phase Leiden-style)

Shared settings (required for the experiment):

* ``resolution`` = edge density of the subnetwork under test
* ``max_memberships`` = number of GT communities with ≥2 nodes in the window
* ``allow_isolation=True``
* ``n_iterations=-1`` (until equilibrium)

Growth stops per configuration when a wallclock ``--timeout`` is exceeded so
the full network need not finish. Results JSON + a two-line size-vs-time plot
are written for hardware-limit inspection.

CLI::

    hedonic-exp overlapping-scale --smoke --output_dir /tmp/scale
    hedonic-exp overlapping-scale --timeout 30 --max-levels 6 --output_dir ...
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.config import DBLP_DIR
from hedonic.experiments.overlapping.dblp_full import load_dblp
from hedonic.experiments.overlapping.dblp_subgraph import extract_subgraph
from hedonic.experiments.overlapping.protocol import current_experiment_identity

# Fixed algorithm flags (not CLI knobs) — match the experiment contract.
N_ITERATIONS = -1
ALLOW_ISOLATION = True
LOCAL_MOVE_ONLY_VARIANTS: tuple[bool, bool] = (True, False)

# CLI names → local_move_only flags.
VARIANT_CHOICES = ("both", "local", "full")
VARIANT_TO_FLAGS: dict[str, tuple[bool, ...]] = {
    "both": (True, False),
    "local": (True,),  # local_move_only=True
    "full": (False,),  # local_move_only=False (full multi-phase)
}

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_LEVELS = 8
DEFAULT_OUTPUT_DIR = Path("overlapping_scale_results")


def parse_variants(name: str) -> tuple[bool, ...]:
    """Map CLI ``--variant`` to ``local_move_only`` flags."""
    key = (name or "both").strip().lower()
    if key not in VARIANT_TO_FLAGS:
        raise ValueError(
            f"unknown variant {name!r}; choose one of {VARIANT_CHOICES}"
        )
    return VARIANT_TO_FLAGS[key]


def log(msg: str, t0: float | None = None) -> None:
    elapsed = f"  [{time.time() - t0:.1f}s]" if t0 else ""
    print(f"{msg}{elapsed}", flush=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def count_gt_communities_in_nodes(
    node_ids: Sequence[int],
    all_gt: Sequence[Sequence[int]],
    *,
    min_nodes: int = 2,
) -> int:
    """Count GT communities with at least ``min_nodes`` vertices in ``node_ids``."""
    node_set = set(node_ids)
    return sum(
        1 for comm in all_gt if sum(1 for v in comm if v in node_set) >= min_nodes
    )


def remap_gt_to_subgraph(
    all_gt: Sequence[Sequence[int]],
    old2new: dict[int, int],
    *,
    min_nodes: int = 2,
) -> list[list[int]]:
    """Map full-graph GT communities into subgraph vertex ids (drop tiny remnants)."""
    remapped: list[list[int]] = []
    for comm in all_gt:
        local = [old2new[v] for v in comm if v in old2new]
        if len(local) >= min_nodes:
            remapped.append(local)
    return remapped


@dataclass
class SubgraphPoint:
    """One induced subnetwork in the growth series."""

    n_nodes: int
    n_edges: int
    density: float
    max_memberships: int
    n_gt_in_subgraph: int
    levels: int | None = None
    seed_community_idx: int | None = None
    label: str = ""
    # Edge list for process isolation (undirected simple graph).
    edges: list[tuple[int, int]] = field(default_factory=list, repr=False)


def _edge_list(g: ig.Graph) -> list[tuple[int, int]]:
    return [(int(e.source), int(e.target)) for e in g.es]


def subgraph_point_from_graph(
    subg: ig.Graph,
    n_gt: int,
    *,
    levels: int | None = None,
    seed_community_idx: int | None = None,
    label: str = "",
) -> SubgraphPoint:
    game = Game(subg)
    density = float(game.density())
    k = max(1, int(n_gt))
    return SubgraphPoint(
        n_nodes=int(subg.vcount()),
        n_edges=int(subg.ecount()),
        density=density,
        max_memberships=k,
        n_gt_in_subgraph=int(n_gt),
        levels=levels,
        seed_community_idx=seed_community_idx,
        label=label or f"n={subg.vcount()}",
        edges=_edge_list(subg),
    )


# ---------------------------------------------------------------------------
# Timing (hard timeout via child process when requested)
# ---------------------------------------------------------------------------


def _worker_community_hedonic(payload: dict[str, Any], queue: mp.Queue) -> None:
    """Child process: run one community_hedonic and put timing result on queue."""
    try:
        n = int(payload["n_nodes"])
        edges = payload["edges"]
        max_memberships = int(payload["max_memberships"])
        local_move_only = bool(payload["local_move_only"])
        g = Game(ig.Graph(n=n, edges=edges, directed=False))
        # Simple undirected graphs may have multi-edges if input did; simplify.
        g.simplify()
        resolution = float(g.density())
        t0 = time.perf_counter()
        g.community_hedonic(
            resolution=resolution,
            max_memberships=max_memberships,
            local_move_only=local_move_only,
            allow_isolation=ALLOW_ISOLATION,
            n_iterations=N_ITERATIONS,
        )
        elapsed = time.perf_counter() - t0
        queue.put(
            {
                "ok": True,
                "wallclock_s": elapsed,
                "resolution": resolution,
                "n_nodes": n,
                "n_edges": int(g.ecount()),
                "local_move_only": local_move_only,
                "max_memberships": max_memberships,
                "timed_out": False,
                "error": None,
            }
        )
    except Exception as exc:  # pragma: no cover - surfaced to parent
        queue.put(
            {
                "ok": False,
                "wallclock_s": None,
                "resolution": None,
                "n_nodes": payload.get("n_nodes"),
                "n_edges": None,
                "local_move_only": payload.get("local_move_only"),
                "max_memberships": payload.get("max_memberships"),
                "timed_out": False,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        )


def time_community_hedonic(
    point: SubgraphPoint,
    *,
    local_move_only: bool,
    timeout_s: float | None = None,
    use_process: bool = True,
) -> dict[str, Any]:
    """Time one equilibrium run; optional hard wallclock timeout via process.

    Parameters
    ----------
    point :
        Subnetwork descriptor (nodes/edges/K).
    local_move_only :
        True → local-moving only; False → full multi-phase.
    timeout_s :
        If set and the run exceeds this many seconds, mark ``timed_out`` and
        (when ``use_process``) terminate the worker. Soft mode (in-process)
        still finishes the call then marks timeout if elapsed > budget.
    use_process :
        Hard-kill timeout requires a child process. In-process is better for
        tiny unit tests and environments where spawn is costly.
    """
    base = {
        "n_nodes": point.n_nodes,
        "n_edges": point.n_edges,
        "density": point.density,
        "max_memberships": point.max_memberships,
        "n_gt_in_subgraph": point.n_gt_in_subgraph,
        "local_move_only": local_move_only,
        "allow_isolation": ALLOW_ISOLATION,
        "n_iterations": N_ITERATIONS,
        "levels": point.levels,
        "seed_community_idx": point.seed_community_idx,
        "label": point.label,
        "resolution": point.density,
        "timed_out": False,
        "error": None,
        "wallclock_s": None,
    }

    if point.n_nodes < 2 or point.n_edges < 0:
        base["error"] = "subgraph too small"
        return base

    payload = {
        "n_nodes": point.n_nodes,
        "edges": point.edges,
        "max_memberships": point.max_memberships,
        "local_move_only": local_move_only,
    }

    if use_process and timeout_s is not None and timeout_s > 0:
        ctx = mp.get_context("spawn")
        queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_worker_community_hedonic, args=(payload, queue))
        proc.start()
        proc.join(timeout_s)
        if proc.is_alive():
            proc.terminate()
            proc.join(5.0)
            if proc.is_alive():  # pragma: no cover
                proc.kill()
                proc.join(1.0)
            base["timed_out"] = True
            base["wallclock_s"] = float(timeout_s)
            base["error"] = f"timeout after {timeout_s}s"
            return base
        if queue.empty():
            base["error"] = "worker exited without result"
            base["timed_out"] = True
            base["wallclock_s"] = float(timeout_s) if timeout_s else None
            return base
        result = queue.get()
        base["wallclock_s"] = result.get("wallclock_s")
        base["resolution"] = result.get("resolution", point.density)
        base["error"] = result.get("error")
        base["timed_out"] = bool(result.get("timed_out", False))
        if result.get("n_edges") is not None:
            base["n_edges"] = result["n_edges"]
        return base

    # In-process (soft timeout): always finish, then flag if over budget.
    g = Game(ig.Graph(n=point.n_nodes, edges=point.edges, directed=False))
    g.simplify()
    resolution = float(g.density())
    t0 = time.perf_counter()
    g.community_hedonic(
        resolution=resolution,
        max_memberships=point.max_memberships,
        local_move_only=local_move_only,
        allow_isolation=ALLOW_ISOLATION,
        n_iterations=N_ITERATIONS,
    )
    elapsed = time.perf_counter() - t0
    base["wallclock_s"] = elapsed
    base["resolution"] = resolution
    base["n_edges"] = int(g.ecount())
    if timeout_s is not None and timeout_s > 0 and elapsed > timeout_s:
        base["timed_out"] = True
        base["error"] = f"soft timeout: {elapsed:.3f}s > {timeout_s}s"
    return base


# ---------------------------------------------------------------------------
# Growth series builders
# ---------------------------------------------------------------------------


def _gt_vertices_in_graph(
    community: Sequence[int], n_vertices: int
) -> list[int]:
    """Keep only GT member ids that exist on the loaded graph (0..n-1).

    Some DBLP community pickles still hold original SNAP node ids; after the
    graph is renumbered those ids are invalid and must be dropped.
    """
    return [int(v) for v in community if 0 <= int(v) < n_vertices]


def build_dblp_hop_series(
    g_full: ig.Graph,
    all_gt: Sequence[Sequence[int]],
    *,
    seed_community_idx: int | None = None,
    max_levels: int = DEFAULT_MAX_LEVELS,
    min_gt_size: int = 5,
    max_gt_size: int = 80,
    rng: np.random.Generator | None = None,
) -> list[SubgraphPoint]:
    """Grow L-hop windows around one GT seed; one point per hop level with growth."""
    rng = rng or np.random.default_rng(42)
    n_full = int(g_full.vcount())

    def _eligible(i: int) -> bool:
        seed_ids = _gt_vertices_in_graph(all_gt[i], n_full)
        return min_gt_size <= len(seed_ids) <= max_gt_size

    candidates = [i for i in range(len(all_gt)) if _eligible(i)]
    if not candidates:
        # Fall back: any community with at least 2 in-graph vertices.
        candidates = [
            i
            for i in range(len(all_gt))
            if len(_gt_vertices_in_graph(all_gt[i], n_full)) >= 2
        ]
    if not candidates:
        raise ValueError(
            "No GT communities with vertices inside the loaded graph "
            f"(n={n_full}). Check DBLP pkl / node remapping."
        )

    if seed_community_idx is None:
        seed_community_idx = int(rng.choice(candidates))
    if seed_community_idx < 0 or seed_community_idx >= len(all_gt):
        raise IndexError(
            f"seed_community_idx={seed_community_idx} out of range "
            f"(0..{len(all_gt) - 1})"
        )

    seed = _gt_vertices_in_graph(all_gt[seed_community_idx], n_full)
    if len(seed) < 2:
        raise ValueError(
            f"seed community {seed_community_idx} has fewer than 2 vertices "
            f"in-graph (raw size {len(all_gt[seed_community_idx])}). "
            "Pick another --community_idx or let the default filter choose."
        )

    # Only count GT communities whose members use in-graph ids (same filter).
    gt_in_graph = [
        _gt_vertices_in_graph(c, n_full) for c in all_gt
    ]
    gt_in_graph = [c for c in gt_in_graph if len(c) >= 2]

    points: list[SubgraphPoint] = []
    prev_n = 0
    for level in range(1, max_levels + 1):
        subg, old2new, nodes_sorted = extract_subgraph(g_full, seed, level)
        n = subg.vcount()
        if n <= prev_n and points:
            # No further expansion (component exhausted).
            break
        n_gt = count_gt_communities_in_nodes(nodes_sorted, gt_in_graph)
        points.append(
            subgraph_point_from_graph(
                subg,
                n_gt,
                levels=level,
                seed_community_idx=seed_community_idx,
                label=f"L{level}_comm{seed_community_idx}",
            )
        )
        prev_n = n
        # Full graph reached.
        if n >= n_full:
            break
    return points


def build_synthetic_size_series(
    sizes: Sequence[int] = (24, 48, 96),
    *,
    n_blocks: int = 4,
    p_in: float = 0.35,
    p_out: float = 0.05,
    seed: int = 0,
) -> list[SubgraphPoint]:
    """Increasing SBM graphs with planted block GT (CI / smoke, no DBLP)."""
    points: list[SubgraphPoint] = []
    for n in sizes:
        n = int(n)
        if n < n_blocks * 2:
            continue
        block = n // n_blocks
        block_sizes = [block] * n_blocks
        block_sizes[-1] += n - sum(block_sizes)
        pref = [
            [p_in if i == j else p_out for j in range(n_blocks)]
            for i in range(n_blocks)
        ]
        # lucas-igraph: SBM(pref_matrix, block_sizes, directed=...)
        g = ig.Graph.SBM(pref, block_sizes, directed=False)
        # Planted blocks as GT communities (global ids 0..n-1).
        gt: list[list[int]] = []
        start = 0
        for sz in block_sizes:
            gt.append(list(range(start, start + sz)))
            start += sz
        n_gt = len(gt)
        points.append(
            subgraph_point_from_graph(
                g,
                n_gt,
                levels=None,
                seed_community_idx=None,
                label=f"synth_n{n}",
            )
        )
    return points


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------


@dataclass
class ScaleResult:
    """Full experiment output (serializable)."""

    points: list[dict[str, Any]]
    meta: dict[str, Any]

    def completed_for(self, local_move_only: bool) -> list[dict[str, Any]]:
        return [
            p
            for p in self.points
            if p.get("local_move_only") is local_move_only
            and not p.get("timed_out")
            and p.get("wallclock_s") is not None
            and p.get("error") is None
        ]


def run_scale_experiment(
    series: Sequence[SubgraphPoint],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    use_process: bool = True,
    variants: Sequence[bool] = LOCAL_MOVE_ONLY_VARIANTS,
) -> ScaleResult:
    """Time both local_move_only settings on each size; stop a line on timeout.

    Growth of a configuration stops after the first timed-out (or failed) point
    for that configuration; the other line may continue further.
    """
    active = {bool(v): True for v in variants}
    records: list[dict[str, Any]] = []

    log("=" * 55)
    log("  Overlapping complexity scale")
    log("=" * 55)
    unlimited = timeout_s is None or timeout_s <= 0
    log(f"  sizes         : {[p.n_nodes for p in series]}")
    log(f"  timeout_s     : {'none (unlimited)' if unlimited else timeout_s}")
    log(f"  variants      : local_move_only in {list(variants)}")
    log(f"  n_iterations  : {N_ITERATIONS}")
    log(f"  allow_isolation: {ALLOW_ISOLATION}")
    log(f"  resolution    : density (per subgraph)")
    log(f"  max_memberships: n GT communities in subgraph")
    log("=" * 55)

    for point in series:
        if not any(active.values()):
            log("[stop] both configurations timed out / inactive — end growth")
            break
        log(
            f"\n[size] n={point.n_nodes:,}  m={point.n_edges:,}  "
            f"K={point.max_memberships}  dens={point.density:.4g}  "
            f"({point.label})"
        )
        for only_lm in variants:
            only_lm = bool(only_lm)
            if not active.get(only_lm, False):
                log(f"  local_move_only={only_lm}: skipped (already stopped)")
                continue
            t0 = time.time()
            rec = time_community_hedonic(
                point,
                local_move_only=only_lm,
                timeout_s=None if unlimited else timeout_s,
                use_process=use_process and not unlimited,
            )
            records.append(rec)
            status = "TIMEOUT" if rec["timed_out"] else (
                "ERROR" if rec.get("error") else "ok"
            )
            wc = rec.get("wallclock_s")
            wc_s = f"{wc:.4f}s" if isinstance(wc, (int, float)) else "n/a"
            log(
                f"  local_move_only={str(only_lm):5s}  {status:7s}  "
                f"wallclock={wc_s}  [{time.time() - t0:.1f}s wall]"
            )
            if rec["timed_out"] or rec.get("error"):
                active[only_lm] = False
                log(f"  → stop further growth for local_move_only={only_lm}")

    meta = {
        "experiment_identity": current_experiment_identity(),
        "timeout_s": None if unlimited else timeout_s,
        "timeout_unlimited": unlimited,
        "n_iterations": N_ITERATIONS,
        "allow_isolation": ALLOW_ISOLATION,
        "resolution_rule": "subgraph_density",
        "max_memberships_rule": "n_gt_communities_in_subgraph_ge2",
        "variants": list(variants),
        "n_series_points": len(series),
        "n_records": len(records),
        "series_labels": [p.label for p in series],
        "series_sizes": [p.n_nodes for p in series],
    }
    return ScaleResult(points=records, meta=meta)


# ---------------------------------------------------------------------------
# I/O + plot
# ---------------------------------------------------------------------------


def save_results(result: ScaleResult, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": result.meta, "points": result.points}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_results(path: Path | str) -> ScaleResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ScaleResult(points=data["points"], meta=data.get("meta", {}))


def plot_complexity_scale(
    result: ScaleResult | dict[str, Any] | Sequence[dict[str, Any]],
    output_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Two-line plot: network size (x) vs wallclock to equilibrium (y).

    One line for ``local_move_only=True``, one for ``False``. Timed-out
    points are omitted from the line (may end earlier).
    """
    # Non-interactive backend before pyplot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if isinstance(result, ScaleResult):
        points = result.points
    elif isinstance(result, dict) and "points" in result:
        points = result["points"]
    else:
        points = list(result)

    series: dict[bool, list[tuple[int, float]]] = {True: [], False: []}
    for p in points:
        if p.get("timed_out") or p.get("error") or p.get("wallclock_s") is None:
            continue
        olm = bool(p["local_move_only"])
        series[olm].append((int(p["n_nodes"]), float(p["wallclock_s"])))

    for key in series:
        series[key].sort(key=lambda t: t[0])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    styles = {
        True: ("o-", "local_move_only=True"),
        False: ("s--", "local_move_only=False"),
    }
    for olm, (style, label) in styles.items():
        pts = series[olm]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, style, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel("Wallclock time to equilibrium (s)")
    ax.set_title(
        title
        or "Overlapping hedonic scaling (density γ, K = #GT, allow_isolation)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-scale",
        description=(
            "Wallclock complexity of overlapping community_hedonic as "
            "subnetworks grow toward the full graph. Times local_move_only "
            "True vs False with density resolution, GT-based max_memberships, "
            "allow_isolation=True, n_iterations=-1. Stops growth on --timeout."
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
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=(
            "Wallclock seconds per (size, local_move_only) run; "
            "growth for that line stops after a timeout. "
            "Use 0 or a negative value for no timeout (run until equilibrium "
            "even on the full hop ladder / near-full graph)."
        ),
    )
    parser.add_argument(
        "--variant",
        choices=VARIANT_CHOICES,
        default="both",
        help=(
            "Which local_move_only setting(s) to time: "
            "'local' = True (local-moving only), "
            "'full' = False (full multi-phase refine+aggregate), "
            "'both' = two lines (default)"
        ),
    )
    parser.add_argument(
        "--max-levels",
        type=int,
        default=DEFAULT_MAX_LEVELS,
        help="Max L-hop expansion levels from the seed GT community (DBLP)",
    )
    parser.add_argument(
        "--community_idx",
        type=int,
        default=None,
        help="GT community index used as hop seed (default: random eligible)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for seed-community selection / synthetic graphs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic increasing SBMs only (no DBLP); CI-friendly",
    )
    parser.add_argument(
        "--sizes",
        default="24,48,96",
        help="Comma-separated node counts for --smoke synthetic series",
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Time in-process (soft timeout) instead of hard process kill",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Figure format",
    )
    parser.add_argument(
        "--results-name",
        default="complexity_scale.json",
        help="Results filename under --output_dir",
    )
    parser.add_argument(
        "--plot-name",
        default=None,
        help="Plot filename (default: complexity_scale.<format>)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_process = not args.no_process

    if args.smoke:
        sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
        log(f"[smoke] synthetic SBM series sizes={sizes}")
        series = build_synthetic_size_series(sizes, seed=args.seed)
    else:
        g, gt, _node_map = load_dblp(args.data_dir)
        series = build_dblp_hop_series(
            g,
            gt,
            seed_community_idx=args.community_idx,
            max_levels=args.max_levels,
            rng=np.random.default_rng(args.seed),
        )
        if not series:
            log("[error] empty hop series — check seed community / DBLP data")
            return 1
        log(
            f"[dblp] seed community={series[0].seed_community_idx}  "
            f"levels → sizes {[p.n_nodes for p in series]}"
        )

    if not series:
        log("[error] no subgraphs to time")
        return 1

    variants = parse_variants(args.variant)
    # <= 0 → no wallclock budget (never mark timed_out / never kill worker).
    timeout_s = None if args.timeout is not None and args.timeout <= 0 else args.timeout
    if timeout_s is None:
        log("[run] timeout disabled (run each size to equilibrium)")
        use_process = False  # no need for process isolation without a budget
    result = run_scale_experiment(
        series,
        timeout_s=timeout_s if timeout_s is not None else 0.0,
        use_process=use_process,
        variants=variants,
    )
    result.meta["smoke"] = bool(args.smoke)
    result.meta["data_dir"] = None if args.smoke else str(args.data_dir)
    result.meta["seed"] = args.seed
    result.meta["use_process"] = use_process
    result.meta["variant"] = args.variant

    results_path = out_dir / args.results_name
    save_results(result, results_path)
    log(f"\n[done] Results → {results_path}")

    plot_name = args.plot_name or f"complexity_scale.{args.plot_format}"
    plot_path = out_dir / plot_name
    plot_complexity_scale(result, plot_path)
    log(f"[done] Plot    → {plot_path}")

    # Summary table
    log("\nCompleted points (no timeout):")
    for olm in (True, False):
        done = result.completed_for(olm)
        if not done:
            log(f"  local_move_only={olm}: (none)")
            continue
        for p in done:
            log(
                f"  local_move_only={olm}: n={p['n_nodes']:,}  "
                f"t={p['wallclock_s']:.4f}s  K={p['max_memberships']}"
            )
        last = done[-1]
        log(
            f"  → max completed size for local_move_only={olm}: "
            f"n={last['n_nodes']:,}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
