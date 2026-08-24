"""Controlled overlap ablations on an explicitly LFR-derived construction.

NetworkX implements the *disjoint* LFR benchmark.  This module uses that
heavy-tailed graph and primary partition as a base, then deterministically
adds secondary memberships and seeded within-secondary-community edges.  It
must not be described as the canonical overlapping LFR benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import platform
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import igraph as ig

from hedonic import Game
from hedonic.experiments.config import OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    partition_to_cover_lists,
)


CONSTRUCTION_LABEL = "LFR-derived controlled overlap"
CONSTRUCTION_CAVEAT = (
    "NetworkX LFR generates a disjoint base. Secondary memberships and their "
    "reinforcing edges are added by this experiment; this is not canonical "
    "overlapping LFR."
)
SCHEMA_VERSION = 2
ENSURE_EQUILIBRIUM = True
ALLOW_ISOLATION = True
NEUTRAL_START_SEED_OFFSET = 91_001
_LFR_BASE_CACHE: dict[tuple, tuple[object, int, int]] = {}
IMPLEMENTATION_SOURCE_FILES = (
    "src/hedonic/experiments/overlapping/controlled_overlap.py",
    "src/hedonic/Game.py",
    "src/hedonic/experiments/overlapping/metrics.py",
)


@dataclass(frozen=True)
class ControlledCover:
    """One generated graph, cover, and fully recorded construction metadata."""

    graph: ig.Graph
    cover: list[list[int]]
    primary_labels: list[int]
    overlapping_vertices: tuple[int, ...]
    metadata: dict


def _csv_numbers(value: str, cast):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _extract_primary_cover(nx_graph) -> tuple[list[list[int]], list[int]]:
    communities = {
        frozenset(int(member) for member in nx_graph.nodes[vertex]["community"])
        for vertex in nx_graph.nodes
    }
    ordered = sorted(communities, key=lambda members: (min(members), len(members)))
    cover = [sorted(members) for members in ordered]
    labels = [-1] * nx_graph.number_of_nodes()
    for community_id, members in enumerate(cover):
        for vertex in members:
            if labels[vertex] != -1:
                raise ValueError("NetworkX LFR base unexpectedly contains overlap")
            labels[vertex] = community_id
    if any(label < 0 for label in labels):
        raise ValueError("NetworkX LFR base left vertices without a community")
    return cover, labels


def generate_controlled_cover(
    *,
    n: int,
    mu: float,
    overlap_fraction: float,
    memberships_per_overlapping_vertex: int,
    secondary_edge_probability: float,
    seed: int,
    average_degree: int = 5,
    min_community: int = 20,
    max_community: int = 50,
    max_lfr_attempts: int = 20,
    lfr_max_iters: int = 500,
) -> ControlledCover:
    """Generate a disjoint LFR base, then add controlled secondary overlap.

    LFR generation can fail for a particular seed/degree sequence.  Retry
    seeds are deterministic and the successful effective seed is recorded.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - experiments extra supplies it
        raise RuntimeError("networkx is required for controlled overlap") from exc

    if n < 4:
        raise ValueError("n must be at least 4")
    if not 0 <= mu <= 1:
        raise ValueError("mu must be in [0, 1]")
    if not 0 <= overlap_fraction <= 1:
        raise ValueError("overlap_fraction must be in [0, 1]")
    if memberships_per_overlapping_vertex < 2:
        raise ValueError("memberships_per_overlapping_vertex must be at least 2")
    if not 0 <= secondary_edge_probability <= 1:
        raise ValueError("secondary_edge_probability must be in [0, 1]")
    if min_community > max_community:
        raise ValueError("min_community cannot exceed max_community")

    cache_key = (
        n,
        float(mu),
        int(seed),
        int(average_degree),
        int(min_community),
        int(max_community),
        int(max_lfr_attempts),
        int(lfr_max_iters),
    )
    cached = _LFR_BASE_CACHE.get(cache_key)
    if cached is not None:
        base, effective_seed, lfr_attempts = cached
    else:
        base = None
        effective_seed = None
        failures: list[str] = []
        for attempt in range(max_lfr_attempts):
            candidate_seed = int(seed) + attempt * 1_000_003
            try:
                base = nx.LFR_benchmark_graph(
                    n,
                    tau1=2.5,
                    tau2=1.5,
                    mu=float(mu),
                    average_degree=int(average_degree),
                    min_community=int(min_community),
                    max_community=int(max_community),
                    max_iters=int(lfr_max_iters),
                    seed=candidate_seed,
                )
                effective_seed = candidate_seed
                break
            except nx.ExceededMaxIterations as exc:
                failures.append(type(exc).__name__)
        if base is None or effective_seed is None:
            raise RuntimeError(
                f"LFR base failed after {max_lfr_attempts} deterministic attempts: "
                f"{failures[-3:]}"
            )
        lfr_attempts = len(failures) + 1
        _LFR_BASE_CACHE[cache_key] = (base, effective_seed, lfr_attempts)

    primary_cover, primary_labels = _extract_primary_cover(base)
    if len(primary_cover) < 2:
        raise RuntimeError("controlled overlap requires at least two primary communities")
    requested_memberships = int(memberships_per_overlapping_vertex)
    realized_memberships = min(requested_memberships, len(primary_cover))

    rng = random.Random(int(seed) ^ 0x5EEDC0DE)
    vertices = list(range(n))
    rng.shuffle(vertices)
    overlap_count = min(n, max(0, round(n * float(overlap_fraction))))
    overlapping_vertices = tuple(sorted(vertices[:overlap_count]))

    gt_sets = [set(members) for members in primary_cover]
    secondary_assignments: dict[int, list[int]] = {}
    added_edges: set[tuple[int, int]] = set()
    existing_edges = {
        tuple(sorted((int(first), int(second))))
        for first, second in base.edges()
        if first != second
    }
    for vertex in overlapping_vertices:
        choices = [
            community_id
            for community_id in range(len(primary_cover))
            if community_id != primary_labels[vertex]
        ]
        rng.shuffle(choices)
        targets = choices[: realized_memberships - 1]
        secondary_assignments[vertex] = sorted(targets)
        for target in targets:
            gt_sets[target].add(vertex)
            candidates = [member for member in primary_cover[target] if member != vertex]
            chosen: list[int] = []
            for member in candidates:
                if rng.random() < secondary_edge_probability:
                    chosen.append(member)
            # A positive reinforcement control always realizes at least one
            # corresponding edge per secondary membership.
            if secondary_edge_probability > 0 and candidates and not chosen:
                chosen = [candidates[rng.randrange(len(candidates))]]
            for member in chosen:
                edge = tuple(sorted((vertex, member)))
                if edge not in existing_edges:
                    added_edges.add(edge)

    all_edges = existing_edges | added_edges
    graph = ig.Graph(n=n, edges=sorted(all_edges), directed=False)
    cover = [sorted(members) for members in gt_sets]

    vertex_memberships = [set() for _ in range(n)]
    for community_id, members in enumerate(cover):
        for vertex in members:
            vertex_memberships[vertex].add(community_id)
    shared_edges = sum(
        bool(vertex_memberships[first] & vertex_memberships[second])
        for first, second in graph.get_edgelist()
    )
    realized_mixing = (
        1.0 - shared_edges / graph.ecount() if graph.ecount() else 0.0
    )
    metadata = {
        "construction_label": CONSTRUCTION_LABEL,
        "construction_caveat": CONSTRUCTION_CAVEAT,
        "n": n,
        "tau1": 2.5,
        "tau2": 1.5,
        "mu_base_lfr": float(mu),
        "average_degree_requested": int(average_degree),
        "min_community": int(min_community),
        "max_community": int(max_community),
        "seed_requested": int(seed),
        "lfr_effective_seed": effective_seed,
        "lfr_attempts": lfr_attempts,
        "lfr_max_iters_per_attempt": int(lfr_max_iters),
        "overlap_fraction_requested": float(overlap_fraction),
        "overlap_fraction_realized": overlap_count / n,
        "overlapping_vertex_count": overlap_count,
        "memberships_per_overlapping_vertex_requested": requested_memberships,
        "memberships_per_overlapping_vertex_realized": realized_memberships,
        "secondary_edge_probability": float(secondary_edge_probability),
        "secondary_assignments": {
            str(vertex): targets for vertex, targets in secondary_assignments.items()
        },
        "base_edge_count": len(existing_edges),
        "secondary_edges_added": len(added_edges),
        "final_edge_count": graph.ecount(),
        "community_count": len(cover),
        "realized_edge_mixing": realized_mixing,
        "directed": False,
    }
    return ControlledCover(
        graph=graph,
        cover=cover,
        primary_labels=primary_labels,
        overlapping_vertices=overlapping_vertices,
        metadata=metadata,
    )


def _resolve_caps(specs: Sequence[str], gt_cap: int) -> list[tuple[str, int]]:
    resolved: list[tuple[str, int]] = []
    for spec in specs:
        normalized = spec.strip().lower()
        value = gt_cap if normalized == "gt" else int(normalized)
        if value < 1:
            raise ValueError("max-membership caps must be positive")
        pair = (normalized, value)
        if pair not in resolved:
            resolved.append(pair)
    return resolved


def _neutral_disjoint_start(graph: ig.Graph, resolution: float, seed: int) -> list[int]:
    ig.set_random_number_generator(random.Random(seed))
    partition = Game(graph).community_hedonic(
        resolution=resolution,
        max_memberships=1,
        local_move_only=False,
        n_iterations=-1,
        seed=seed,
        allow_isolation=ALLOW_ISOLATION,
        ensure_equilibrium=ENSURE_EQUILIBRIUM,
    )
    return [int(label) for label in partition.membership]


def _detector_worker(
    queue,
    graph: ig.Graph,
    *,
    resolution: float,
    cap: int,
    local: bool,
    initial_membership,
    seed: int,
) -> None:
    """Run one equilibrium detector in an isolated process."""
    try:
        ig.set_random_number_generator(random.Random(seed))
        started = time.perf_counter()
        result = Game(graph).community_hedonic(
            resolution=resolution,
            max_memberships=cap,
            local_move_only=local,
            n_iterations=-1,
            allow_isolation=ALLOW_ISOLATION,
            initial_membership=initial_membership,
            seed=seed,
            ensure_equilibrium=ENSURE_EQUILIBRIUM,
        )
        queue.put(
            {
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - started,
                "predicted_cover": partition_to_cover_lists(result),
            }
        )
    except BaseException:
        queue.put(
            {
                "status": "failed",
                "elapsed_seconds": None,
                "error": traceback.format_exc(),
                "predicted_cover": None,
            }
        )


def _run_detector_isolated(
    graph: ig.Graph,
    *,
    resolution: float,
    cap: int,
    local: bool,
    initial_membership,
    seed: int,
    timeout_seconds: float,
) -> dict:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    context = mp.get_context("fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_detector_worker,
        args=(queue, graph),
        kwargs={
            "resolution": resolution,
            "cap": cap,
            "local": local,
            "initial_membership": initial_membership,
            "seed": seed,
        },
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)
        queue.close()
        queue.join_thread()
        return {
            "status": "timeout",
            "elapsed_seconds": float(timeout_seconds),
            "error": f"detector exceeded {timeout_seconds:g}s hard timeout",
            "predicted_cover": None,
        }
    try:
        record = queue.get(timeout=1.0)
    except Exception:
        record = {
            "status": "failed",
            "elapsed_seconds": None,
            "error": f"detector exited with code {process.exitcode} without a record",
            "predicted_cover": None,
        }
    queue.close()
    queue.join_thread()
    return record


def run_ablations(
    instance: ControlledCover,
    *,
    phases: Sequence[str],
    cap_specs: Sequence[str],
    starts: Sequence[str],
    resolution_multipliers: Sequence[float],
    seed: int,
    compute_omega: bool = False,
    timeout_seconds: float = 5.0,
) -> list[dict]:
    """Run the locked Cartesian ablation grid for one generated instance."""
    allowed_phases = {"local", "multiphase"}
    allowed_starts = {"singleton", "neutral-disjoint", "gt-primary"}
    if not set(phases) <= allowed_phases:
        raise ValueError(f"phases must be drawn from {sorted(allowed_phases)}")
    if not set(starts) <= allowed_starts:
        raise ValueError(f"starts must be drawn from {sorted(allowed_starts)}")

    graph = instance.graph
    density = graph.density()
    gt_cap = max(
        sum(vertex in community for community in instance.cover)
        for vertex in range(graph.vcount())
    )
    caps = _resolve_caps(cap_specs, gt_cap)
    rows: list[dict] = []
    for multiplier in resolution_multipliers:
        resolution = min(float(density) * float(multiplier), 1.0)
        neutral_start = None
        if "neutral-disjoint" in starts:
            neutral_start = _neutral_disjoint_start(
                graph, resolution, seed + NEUTRAL_START_SEED_OFFSET
            )
        for phase in phases:
            for cap_label, cap in caps:
                for start in starts:
                    # Paired ablations share one detector seed per generated
                    # graph. Only the declared ablation axis changes.
                    run_seed = int(seed)
                    initial_membership = None
                    if start == "gt-primary":
                        initial_membership = list(instance.primary_labels)
                    elif start == "neutral-disjoint":
                        initial_membership = list(neutral_start or [])
                    print(
                        f"  run phase={phase} cap={cap_label} start={start} "
                        f"resolution_x={float(multiplier):g} seed={run_seed}",
                        flush=True,
                    )
                    detector = _run_detector_isolated(
                        graph,
                        resolution=resolution,
                        cap=cap,
                        local=phase == "local",
                        initial_membership=initial_membership,
                        seed=run_seed,
                        timeout_seconds=timeout_seconds,
                    )
                    status = detector["status"]
                    elapsed = detector["elapsed_seconds"]
                    predicted = detector["predicted_cover"]
                    print(f"    {status} in {elapsed}s", flush=True)
                    metrics = (
                        evaluate_cover(
                            predicted,
                            instance.cover,
                            graph.vcount(),
                            compute_omega=compute_omega,
                            omega_seed=run_seed,
                        )
                        if status == "completed"
                        else {}
                    )
                    rows.append(
                        {
                            "status": status,
                            "error": detector.get("error"),
                            "phase": phase,
                            "local_move_only": phase == "local",
                            "allow_isolation": ALLOW_ISOLATION,
                            "ensure_equilibrium": ENSURE_EQUILIBRIUM,
                            "max_memberships_spec": cap_label,
                            "max_memberships": cap,
                            "initialization": start,
                            "initialization_supervision": (
                                "ground-truth primary labels"
                                if start == "gt-primary"
                                else "none"
                            ),
                            "resolution_multiplier": float(multiplier),
                            "resolution": resolution,
                            "seed": run_seed,
                            "elapsed_seconds": elapsed,
                            "predicted_cover": predicted,
                            **metrics,
                        }
                    )
    return rows


def _summary(rows: Sequence[dict], *, keys: Sequence[str] | None = None) -> list[dict]:
    if keys is None:
        keys = (
            "phase",
            "max_memberships_spec",
            "initialization",
            "resolution_multiplier",
        )
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    summary: list[dict] = []
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        completed = [value for value in values if value["status"] == "completed"]

        def aggregate(metric: str) -> dict[str, float | None]:
            observations = [float(value[metric]) for value in completed]
            if not observations:
                return {
                    f"mean_{metric}": None,
                    f"sd_{metric}": None,
                    f"ci95_low_{metric}": None,
                    f"ci95_high_{metric}": None,
                }
            mean = statistics.fmean(observations)
            standard_deviation = (
                statistics.stdev(observations) if len(observations) > 1 else 0.0
            )
            half_width = 1.96 * standard_deviation / (len(observations) ** 0.5)
            return {
                f"mean_{metric}": mean,
                f"sd_{metric}": standard_deviation,
                f"ci95_low_{metric}": mean - half_width,
                f"ci95_high_{metric}": mean + half_width,
            }

        summary.append(
            {
                **dict(zip(keys, group)),
                "n_expected": len(values),
                "n_completed": len(completed),
                "n_timeout": sum(value["status"] == "timeout" for value in values),
                "n_failed": sum(value["status"] == "failed" for value in values),
                **aggregate("matching_f1"),
                **aggregate("node_micro_f1"),
                **aggregate("elapsed_seconds"),
            }
        )
    return summary


def _write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields = [
        "graph_seed",
        "mu_base_lfr",
        "overlap_fraction_realized",
        "memberships_per_overlapping_vertex_realized",
        "secondary_edge_probability",
        "status",
        "error",
        "phase",
        "max_memberships_spec",
        "max_memberships",
        "initialization",
        "initialization_supervision",
        "resolution_multiplier",
        "resolution",
        "seed",
        "elapsed_seconds",
        "matching_f1",
        "node_micro_f1",
        "size_weighted_community_f1",
        "predicted_community_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_json(value: object) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _implementation_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment_identity() -> dict[str, str | None]:
    """Return the implementation-relevant runtime distribution versions."""
    lucas_igraph = _distribution_version("lucas-igraph")
    return {
        "python": platform.python_version(),
        "networkx": _distribution_version("networkx"),
        "lucas_igraph_distribution": lucas_igraph,
        # The released wheel is named lucas-igraph but exposes the ``igraph``
        # import package. Preserve both names to make migrations auditable.
        "igraph_distribution": _distribution_version("igraph") or lucas_igraph,
        "hedonic": _distribution_version("hedonic"),
    }


def _implementation_identity(
    *, environment: dict[str, str | None] | None = None
) -> dict:
    """Bind experiment output to every local source file it directly depends on.

    Paths are repository-relative so the identity is independent of the
    checkout location. The aggregate digest covers both the source map and the
    implementation-relevant package versions.
    """
    repository_root = Path(__file__).resolve().parents[4]
    source_files = {
        relative: hashlib.sha256((repository_root / relative).read_bytes()).hexdigest()
        for relative in IMPLEMENTATION_SOURCE_FILES
    }
    descriptor = {
        "algorithm": "sha256",
        "path_scope": "repository-relative",
        "source_files": source_files,
        "environment": dict(
            _environment_identity() if environment is None else environment
        ),
    }
    return {
        **descriptor,
        "identity_sha256": _sha256_json(descriptor),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "LFR-derived controlled overlap study (not canonical overlapping LFR): "
            "cap, initialization, phase, and resolution ablations"
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OVERLAPPING_ARTIFACTS_DIR
        / "controlled_overlap"
        / "controlled_overlap.json",
        help=(
            "JSON output path (default: "
            "artifacts/overlapping/controlled_overlap/controlled_overlap.json)"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="tiny deterministic grid")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--mus", default="0.2,0.4")
    parser.add_argument("--overlap-fractions", default="0.1,0.3")
    parser.add_argument("--overlap-memberships", default="2")
    parser.add_argument("--secondary-edge-probabilities", default="0.1,0.3")
    parser.add_argument("--graph-seeds", default="0")
    parser.add_argument("--average-degree", type=int, default=5)
    parser.add_argument("--min-community", type=int, default=15)
    parser.add_argument("--max-community", type=int, default=30)
    parser.add_argument("--lfr-max-iters", type=int, default=500)
    parser.add_argument("--phases", default="local,multiphase")
    parser.add_argument("--max-memberships", default="1,2,gt")
    parser.add_argument("--starts", default="singleton,neutral-disjoint,gt-primary")
    parser.add_argument("--resolution-multipliers", default="1,10")
    parser.add_argument("--timeout-per-run", type=float, default=5.0)
    parser.add_argument("--omega", action="store_true", help="compute sampled Omega")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.n = 80
        args.mus = "0.2"
        args.overlap_fractions = "0.2"
        args.overlap_memberships = "2"
        args.secondary_edge_probabilities = "0.2"
        args.graph_seeds = "0"
        args.average_degree = 5
        args.min_community = 15
        args.max_community = 30
        args.phases = "local,multiphase"
        args.max_memberships = "1,gt"
        args.starts = "singleton,gt-primary"
        args.resolution_multipliers = "1,10"

    mus = _csv_numbers(args.mus, float)
    overlap_fractions = _csv_numbers(args.overlap_fractions, float)
    overlap_memberships = _csv_numbers(args.overlap_memberships, int)
    secondary_probabilities = _csv_numbers(args.secondary_edge_probabilities, float)
    graph_seeds = _csv_numbers(args.graph_seeds, int)
    phases = _csv_numbers(args.phases, str)
    cap_specs = _csv_numbers(args.max_memberships, str)
    starts = _csv_numbers(args.starts, str)
    multipliers = _csv_numbers(args.resolution_multipliers, float)

    environment = _environment_identity()
    implementation_identity = _implementation_identity(environment=environment)
    protocol = {
        "implementation_identity": implementation_identity,
        "construction": {
            "label": CONSTRUCTION_LABEL,
            "n": int(args.n),
            "tau1": 2.5,
            "tau2": 1.5,
            "mus": mus,
            "overlap_fractions": overlap_fractions,
            "memberships_per_overlapping_vertex": overlap_memberships,
            "secondary_edge_probabilities": secondary_probabilities,
            "graph_seeds": graph_seeds,
            "average_degree": int(args.average_degree),
            "min_community": int(args.min_community),
            "max_community": int(args.max_community),
            "max_lfr_attempts": 20,
            "lfr_max_iters_per_attempt": int(args.lfr_max_iters),
            "undirected": True,
        },
        "detector": {
            "api": "Game.community_hedonic",
            "phases": phases,
            "max_memberships_specs": cap_specs,
            "initializations": starts,
            "resolution_multipliers": multipliers,
            "n_iterations": -1,
            "allow_isolation": ALLOW_ISOLATION,
            "ensure_equilibrium": ENSURE_EQUILIBRIUM,
            "paired_detector_seed_rule": "detector_seed = graph_seed",
            "neutral_disjoint_seed_offset": NEUTRAL_START_SEED_OFFSET,
            "hard_timeout_seconds_per_cell": float(args.timeout_per_run),
            "timeout_process_isolation": "forked child; terminate then kill/reap",
        },
        "metrics": {
            "evaluate_cover": True,
            "compute_omega": bool(args.omega),
            "omega_sample_size": 100_000 if args.omega else None,
            "summary_ci": "mean +/- 1.96 * sample_sd / sqrt(n)",
        },
    }

    instances: list[dict] = []
    flat_rows: list[dict] = []
    total_instances = (
        len(graph_seeds)
        * len(mus)
        * len(overlap_fractions)
        * len(overlap_memberships)
        * len(secondary_probabilities)
    )
    instance_index = 0
    for graph_seed in graph_seeds:
        for mu in mus:
            for fraction in overlap_fractions:
                for memberships in overlap_memberships:
                    for probability in secondary_probabilities:
                        instance = generate_controlled_cover(
                            n=args.n,
                            mu=mu,
                            overlap_fraction=fraction,
                            memberships_per_overlapping_vertex=memberships,
                            secondary_edge_probability=probability,
                            seed=graph_seed,
                            average_degree=args.average_degree,
                            min_community=args.min_community,
                            max_community=args.max_community,
                            lfr_max_iters=args.lfr_max_iters,
                        )
                        rows = run_ablations(
                            instance,
                            phases=phases,
                            cap_specs=cap_specs,
                            starts=starts,
                            resolution_multipliers=multipliers,
                            seed=graph_seed,
                            compute_omega=args.omega,
                            timeout_seconds=args.timeout_per_run,
                        )
                        instances.append({"construction": instance.metadata, "runs": rows})
                        for row in rows:
                            flat_rows.append(
                                {
                                    "graph_seed": graph_seed,
                                    **instance.metadata,
                                    **row,
                                }
                            )
                        instance_index += 1
                        print(
                            f"completed instance {instance_index}/{total_instances}: "
                            f"seed={graph_seed} mu={mu:g} overlap={fraction:g} "
                            f"memberships={memberships} secondary_p={probability:g}",
                            flush=True,
                        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "construction_label": CONSTRUCTION_LABEL,
        "construction_caveat": CONSTRUCTION_CAVEAT,
        # Compatibility digest retained for readers of schema-v2 artifacts.
        "implementation_sha256": _implementation_sha256(),
        "implementation_identity": implementation_identity,
        "protocol_sha256": _sha256_json(protocol),
        "protocol": protocol,
        "environment": environment,
        "experiment_design": {
            "detector_api": "Game.community_hedonic",
            "n_iterations": -1,
            "ablation_axes": [
                "max_memberships",
                "initialization",
                "local_move_only",
                "resolution_multiplier",
            ],
            "gt_informed_controls": ["gt-primary", "max_memberships=gt"],
            "paired_detector_seed_rule": "detector_seed = graph_seed",
            "neutral_disjoint_seed_offset": NEUTRAL_START_SEED_OFFSET,
            "allow_isolation": ALLOW_ISOLATION,
            "ensure_equilibrium": ENSURE_EQUILIBRIUM,
            "summary_uncertainty": (
                "sample standard deviation and two-sided normal-approximation "
                "95% CI (mean +/- 1.96 * sd / sqrt(n)) across generated instances"
            ),
        },
        "instances": instances,
        "summary": _summary(flat_rows),
        "condition_summary": _summary(
            flat_rows,
            keys=(
                "mu_base_lfr",
                "overlap_fraction_requested",
                "secondary_edge_probability",
                "phase",
                "max_memberships_spec",
                "initialization",
                "resolution_multiplier",
            ),
        ),
    }
    args.output = expand_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    _write_csv(csv_path, flat_rows)
    print(
        f"wrote {len(flat_rows)} runs from {len(instances)} instance(s) to "
        f"{args.output} and {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
