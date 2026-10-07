"""Noisy-cover tracking controls and robustness/recovery triangle (TKT-12).

This module is a graph-level paired experiment.  A planted cover is perturbed
with incidence-preserving double-edge switches, and every control receives the
same perturbed cover, label universe, resolution and graph.  The controls are
Mirror, one synchronous proposal sweep, one sequential/full sweep,
local-to-convergence, and multiphase-to-cleanup.  Requested and achieved
incidence distances are recorded separately because sparse covers can make a
target unattainable.

The tracking arm is intentionally supervised (it starts from a supplied
reference cover) and must not be conflated with the metadata-free TKT-11
benchmark.  ``analyze_robustness_recovery`` keeps graph identity as the
independent unit and reports raw plus condition-stratified associations; a
negative association is a valid result.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.config import OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.metrics import evaluate_cover, partition_to_cover_lists
from hedonic.experiments.overlapping.robustness import (
    audit_cover,
    canonicalize_cover,
    cover_hash,
    cover_to_vertex_memberships,
    perturb_cover_incidence,
    vertex_memberships_to_cover,
)
from hedonic.experiments.overlapping.overlap_lfr import generate_overlapping_lfr, graph_sha256


PROTOCOL_VERSION = "tracking-triangle-v2"
SCHEMA_VERSION = 2
DEFAULT_DISTANCES = (0.0, 0.02, 0.10, 0.30, 0.60)
DEFAULT_VARIANTS = ("mirror", "one_sweep_synchronous", "one_sweep_full", "local", "multiphase")

# The canonical tracking arm is the eight-cell, 30-graph subset specified in
# E1-B: every mixing value crossed with a zero-overlap arm and one positive
# overlap arm.  Keeping this policy in code makes selection from the separate
# TKT-11 28-cell ledger deterministic and auditable.
TRACKING_CONDITION_POLICY = "mixing_x_zero_or_overlap03_m2"
TRACKING_CONDITIONS = tuple(
    {"mixing": float(mixing), "overlap_fraction": float(overlap_fraction), "overlap_multiplicity": int(multiplicity)}
    for mixing in (0.1, 0.3, 0.5, 0.7)
    for overlap_fraction, multiplicity in ((0.0, 1), (0.3, 2))
)
TRACKING_GRAPHS_PER_CONDITION = 30
TRACKING_PILOT_GRAPHS = 20


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    """Return the checked-out source revision without requiring Git metadata."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    return revision or None


def environment_receipt() -> dict[str, Any]:
    """Capture the dependency/source closure needed to interpret a ledger."""
    distributions: dict[str, str | None] = {}
    for name in ("hedonic", "lucas-igraph", "igraph", "numpy", "scipy"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    root = Path(__file__).resolve().parents[4]
    lock_files: dict[str, str | None] = {}
    for name in ("pyproject.toml", "uv.lock"):
        candidate = root / name
        lock_files[name] = _sha256_file(candidate) if candidate.is_file() else None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_revision": _git_revision(),
        "source_file": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "distributions": distributions,
        "lock_files": lock_files,
        "thread_controls": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if os.environ.get(key) is not None
        },
    }


def _peak_rss_bytes() -> int | None:
    """Best-effort process peak RSS, normalized across Darwin/Linux."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if platform.system().lower() == "darwin":
        return value
    return value * 1024


def _condition_matches(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        return (
            math.isclose(float(left.get("mixing")), float(right.get("mixing")), rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                float(left.get("overlap_fraction")),
                float(right.get("overlap_fraction")),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and int(left.get("overlap_multiplicity", 1)) == int(right.get("overlap_multiplicity", 1))
        )
    except (TypeError, ValueError):
        return False


def _condition_label(condition: dict[str, Any]) -> str:
    return "m{:.3g}_o{:.3g}_k{}".format(
        float(condition["mixing"]),
        float(condition["overlap_fraction"]),
        int(condition.get("overlap_multiplicity", 1)),
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Durably replace *path* without exposing a partially written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    try:
        directory_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Directory fsync is not available on every supported filesystem; the
        # atomic rename remains the important invariant.
        pass


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def incidence_count(cover: Sequence[Sequence[int]]) -> int:
    return sum(len(set(map(int, body))) for body in cover)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def _graph_from_record(ledger_path: Path, record: dict[str, Any]) -> tuple[ig.Graph, list[list[int]], Path]:
    """Load and verify one graph archive referenced by a TKT-11 ledger row."""
    graph_hash_value = record.get("graph_hash")
    if not graph_hash_value:
        raise ValueError("canonical graph record has no graph_hash")
    archive_value = record.get("graph_path") or record.get("archive")
    if archive_value:
        archive = Path(str(archive_value)).expanduser()
        if not archive.is_absolute():
            archive = ledger_path.parent / archive
    else:
        archive = ledger_path.parent / "graphs" / f"{graph_hash_value}.json"
    if not archive.is_file():
        raise ValueError(f"canonical graph archive is missing: {archive}")
    payload = _load_json(archive)
    if not isinstance(payload, dict):
        raise ValueError(f"canonical graph archive must be an object: {archive}")
    try:
        edges = [tuple(map(int, edge)) for edge in payload["edges"]]
        metadata_payload = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        n_vertices = int(payload.get("n", metadata_payload.get("n", 0)) or 0)
        if n_vertices < 1:
            n_vertices = max((max(edge) for edge in edges), default=-1) + 1
        cover = canonicalize_cover(payload["cover"], n_vertices=n_vertices)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed canonical graph archive {archive}: {exc}") from exc
    graph = ig.Graph(n=n_vertices, edges=edges, directed=False)
    actual_graph_hash = graph_sha256(graph)
    if actual_graph_hash != str(graph_hash_value):
        raise ValueError(
            f"canonical graph hash mismatch for {archive}: expected {graph_hash_value}, got {actual_graph_hash}"
        )
    expected_cover_hash = record.get("cover_hash") or payload.get("cover_hash")
    actual_cover_hash = cover_hash(cover, n_vertices)
    if expected_cover_hash and actual_cover_hash != str(expected_cover_hash):
        raise ValueError(
            f"canonical cover hash mismatch for {archive}: expected {expected_cover_hash}, got {actual_cover_hash}"
        )
    return graph, cover, archive


def load_tracking_graph_ledger(
    ledger: str | Path,
    *,
    profile: str = "standard",
    max_graphs: int | None = None,
) -> list[dict[str, Any]]:
    """Load the selected canonical TKT-11 graph subset with hash checks.

    The tracking arm never regenerates a graph from a cover.  It consumes the
    immutable graph archives produced by ``overlapping-lfr`` and verifies the
    graph and planted-cover hashes before any detector work starts.
    """
    ledger_path = expand_path(ledger).resolve()
    if not ledger_path.is_file():
        raise ValueError(f"tracking graph ledger is missing: {ledger_path}")
    payload = _load_json(ledger_path)
    if not isinstance(payload, list):
        raise ValueError("tracking graph ledger must be a JSON list (the TKT-11 graphs.json artifact)")
    selected: list[dict[str, Any]] = []
    seen_slots: set[tuple[int, int]] = set()
    seen_graph_hashes: set[str] = set()
    for record_value in payload:
        if not isinstance(record_value, dict) or record_value.get("status", "completed") != "completed":
            continue
        condition = record_value.get("condition")
        if not isinstance(condition, dict):
            continue
        condition_index = next(
            (index for index, expected in enumerate(TRACKING_CONDITIONS) if _condition_matches(condition, expected)),
            None,
        )
        if condition_index is None:
            continue
        try:
            graph_index = int(record_value.get("graph_index_within_condition", len(selected)))
        except (TypeError, ValueError) as exc:
            raise ValueError("canonical tracking graph record has an invalid graph index") from exc
        slot = (int(condition_index), graph_index)
        if slot in seen_slots:
            raise ValueError(f"canonical tracking ledger contains duplicate condition/graph slot: {slot}")
        seen_slots.add(slot)
        # Canonical evidence must come from the official LFRbenchmarks arm.
        policy = str(record_value.get("generator_policy") or (record_value.get("metadata") or {}).get("generator_policy") or "")
        if profile in {"pilot", "standard"} and policy != "official_binary_required":
            raise ValueError(
                "tracking canonical input is not an official LFRbenchmarks record: "
                f"condition={condition} policy={policy!r}"
            )
        graph, cover, archive = _graph_from_record(ledger_path, record_value)
        graph_hash_value = graph_sha256(graph)
        if graph_hash_value in seen_graph_hashes:
            raise ValueError(
                "canonical tracking ledger contains duplicate graph hash: "
                f"{graph_hash_value}"
            )
        seen_graph_hashes.add(graph_hash_value)
        selected.append(
            {
                "graph": graph,
                "cover": cover,
                "graph_id": f"{_condition_label(TRACKING_CONDITIONS[condition_index])}_g{graph_index:03d}",
                "condition": dict(TRACKING_CONDITIONS[condition_index]),
                "condition_index": int(condition_index),
                "graph_index_within_condition": graph_index,
                "graph_seed": int(record_value.get("graph_seed", 0)),
                "graph_hash": graph_hash_value,
                "cover_hash": cover_hash(cover, graph.vcount()),
                "archive": str(archive),
                "source_record": record_value,
            }
        )
    selected.sort(key=lambda row: (int(row["condition_index"]), int(row["graph_index_within_condition"])))
    if max_graphs is not None:
        selected = selected[: max(0, int(max_graphs))]
    expected_per_condition = TRACKING_GRAPHS_PER_CONDITION if str(profile).lower() == "standard" else None
    if str(profile).lower() == "standard" and max_graphs is None:
        counts = {index: 0 for index in range(len(TRACKING_CONDITIONS))}
        for row in selected:
            counts[int(row["condition_index"])] += 1
        missing = {index: count for index, count in counts.items() if count != expected_per_condition}
        if missing:
            raise ValueError(
                "canonical tracking ledger is incomplete for the registered 240-graph subset: "
                + json.dumps(missing, sort_keys=True)
            )
    if not selected:
        raise ValueError("canonical tracking ledger contains no selected TKT-12 graph records")
    return selected


def perturb_cover(
    cover: Sequence[Sequence[int]],
    n_vertices: int,
    *,
    target_distance: float,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return a size/multiplicity-preserving noisy cover and receipt."""
    if not 0.0 <= float(target_distance) <= 1.0:
        raise ValueError("target_distance must be in [0, 1]")
    canonical = canonicalize_cover(cover, n_vertices=n_vertices)
    total = incidence_count(canonical)
    requested_swaps = int(round(float(target_distance) * total / 2.0))
    if target_distance == 0.0:
        perturbed = canonical
        metadata = {
            "requested_distance": 0.0,
            "requested_swaps": 0,
            "successful_swaps": 0,
            "achieved_distance": 0.0,
            "achieved_incidence_f1": 1.0,
            "vertex_membership_counts_preserved": True,
            "community_sizes_preserved": True,
            "perturbation_status": "untouched_reference",
            "seed": int(seed),
            "initial_incidence_count": total,
            "final_incidence_count": total,
        }
        return perturbed, metadata
    perturbed, metadata = perturb_cover_incidence(
        canonical,
        n_vertices,
        swaps=requested_swaps,
        seed=int(seed),
    )
    metadata = {
        **metadata,
        "requested_distance": float(target_distance),
        "achieved_distance": float(metadata.get("realized_incidence_distance", 0.0)),
        "perturbation_status": (
            "target_attained"
            if float(target_distance) == 0.0
            or (requested_swaps > 0 and int(metadata.get("successful_swaps", 0)) >= requested_swaps)
            else "target_unattainable_within_attempt_budget"
        ),
    }
    return perturbed, metadata


def mirror_cover(noisy_cover: Sequence[Sequence[int]]) -> tuple[list[list[int]], dict[str, Any]]:
    """No-change control: return the supplied cover byte-for-byte canonically."""
    cover = canonicalize_cover(noisy_cover)
    return cover, {
        "variant": "mirror",
        "commit_policy": "no_change",
        "n_proposals": 0,
        "accepted_moves": 0,
        "metadata_free": False,
    }


def _normalize_memberships(memberships: Any, n_vertices: int) -> list[list[int]]:
    if not isinstance(memberships, list) or len(memberships) != n_vertices:
        raise ValueError("native membership rows must contain one row per vertex")
    labels = sorted({int(label) for row in memberships for label in row})
    remap = {old: new for new, old in enumerate(labels)}
    rows: list[list[int]] = []
    for row in memberships:
        normalized = sorted({remap[int(label)] for label in row})
        if not normalized:
            raise ValueError("native membership rows cannot be empty")
        rows.append(normalized)
    return rows


def _hedonic_one_run(
    graph: ig.Graph,
    noisy_cover: Sequence[Sequence[int]],
    *,
    variant: str,
    cap: int,
    gamma: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Run one hedonic control and return scoring cover + exact rows."""
    memberships = cover_to_vertex_memberships(noisy_cover, graph.vcount(), require_covered=True)
    if variant == "one_sweep_synchronous":
        local_move_only, n_iterations = True, 1
        commit_policy = "synchronous_proposals_guarded"
    elif variant == "one_sweep_full":
        local_move_only, n_iterations = False, 1
        commit_policy = "sequential_full_best_response_one_sweep"
    elif variant == "local":
        local_move_only, n_iterations = True, -1
        commit_policy = "sequential_local_to_convergence"
    elif variant == "multiphase":
        local_move_only, n_iterations = False, -1
        commit_policy = "leiden_refine_aggregate_then_cleanup"
    else:
        raise ValueError(f"unknown hedonic tracking variant {variant!r}")
    ig.set_random_number_generator(random.Random(int(seed)))
    started = time.perf_counter()
    result = Game(graph).community_hedonic(
        initial_membership=memberships,
        max_memberships=int(cap),
        resolution=float(gamma),
        local_move_only=local_move_only,
        n_iterations=n_iterations,
        allow_isolation=True,
        beta=0.01,
    )
    elapsed = time.perf_counter() - started
    final = getattr(result, "membership", None)
    if final and isinstance(final[0], (list, tuple)):
        final_rows = _normalize_memberships([list(row) for row in final], graph.vcount())
    else:
        final_rows = _normalize_memberships([[int(label)] for label in final], graph.vcount())
    return vertex_memberships_to_cover(final_rows), final_rows, {
        "variant": variant,
        "commit_policy": commit_policy,
        "local_move_only": local_move_only,
        "n_iterations": n_iterations,
        "allow_isolation": True,
        "accepted_moves": getattr(result, "_hedonic_accepted_moves", None),
        "projection_events": getattr(result, "_hedonic_projection_events", None),
        "runtime_seconds": elapsed,
    }


def _pearson(values_x: Sequence[float], values_y: Sequence[float]) -> float | None:
    if len(values_x) < 2 or len(values_y) != len(values_x):
        return None
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def tracking_row_key(row: dict[str, Any]) -> tuple[str, str, float, int, int] | None:
    """Stable idempotency key for one graph/perturbation/control observation."""
    graph_hash_value = row.get("graph_hash")
    variant = row.get("variant")
    if not graph_hash_value or variant is None:
        return None
    try:
        return (
            str(graph_hash_value),
            str(variant),
            round(float(row.get("requested_incidence_distance", 0.0)), 12),
            int(row.get("perturbation_seed", 0)),
            int(row.get("optimizer_seed", 0)),
        )
    except (TypeError, ValueError):
        return None


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _mean_interval(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None]}
    mean = statistics.fmean(finite)
    if len(finite) < 2:
        return {"n": len(finite), "mean": mean, "sd": 0.0, "ci95": [mean, mean]}
    sd = statistics.stdev(finite)
    half = 1.96 * sd / math.sqrt(len(finite))
    return {"n": len(finite), "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def run_tracking_graph(
    graph: ig.Graph,
    ground_truth_cover: Sequence[Sequence[int]],
    *,
    graph_id: str,
    graph_seed: int,
    condition: dict[str, Any] | None = None,
    condition_index: int | None = None,
    graph_index_within_condition: int | None = None,
    input_archive: str | None = None,
    distance_levels: Sequence[float] = DEFAULT_DISTANCES,
    perturbation_seeds: Sequence[int] = (0, 1, 2),
    optimizer_seeds: Sequence[int] = (0, 1),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    cap: int = 4,
    gamma: float | None = None,
) -> list[dict[str, Any]]:
    """Run all tracking variants for one graph, paired by graph/perturbation."""
    gt = canonicalize_cover(ground_truth_cover, graph.vcount())
    selected_gamma = float(graph.density() if gamma is None else gamma)
    rows: list[dict[str, Any]] = []
    for distance_index, requested_distance in enumerate(distance_levels):
        seeds = (int(perturbation_seeds[0]),) if float(requested_distance) == 0.0 else tuple(int(seed) for seed in perturbation_seeds)
        for perturbation_seed in seeds:
            noisy, perturbation = perturb_cover(
                gt,
                graph.vcount(),
                target_distance=float(requested_distance),
                seed=perturbation_seed,
            )
            noisy_hash = cover_hash(noisy, graph.vcount())
            for variant in variants:
                if variant == "mirror":
                    variant_seeds = (0,)
                else:
                    variant_seeds = tuple(int(seed) for seed in optimizer_seeds)
                for optimizer_seed in variant_seeds:
                    started = time.perf_counter()
                    status = "completed"
                    predicted: list[list[int]] | None = None
                    final_rows: list[list[int]] | None = None
                    metadata: dict[str, Any] = {}
                    error = None
                    try:
                        if variant == "mirror":
                            predicted, metadata = mirror_cover(noisy)
                            final_rows = cover_to_vertex_memberships(predicted, graph.vcount())
                        else:
                            predicted, final_rows, metadata = _hedonic_one_run(
                                graph,
                                noisy,
                                variant=variant,
                                cap=int(cap),
                                gamma=selected_gamma,
                                seed=int(optimizer_seed),
                            )
                    except BaseException as exc:
                        status = "failed"
                        error = f"{type(exc).__name__}: {exc}"
                        metadata = {"traceback": traceback.format_exc()}
                    record: dict[str, Any] = {
                        "schema_version": SCHEMA_VERSION,
                        "protocol_version": PROTOCOL_VERSION,
                        "graph_id": str(graph_id),
                        "graph_hash": graph_sha256(graph),
                        "graph_instance_hash": graph_sha256(graph),
                        "parent_graph_hash": graph_sha256(graph),
                        "graph_seed": int(graph_seed),
                        "ground_truth_cover_hash": cover_hash(gt, graph.vcount()),
                        "cover_parent_hash": cover_hash(gt, graph.vcount()),
                        "input_archive": input_archive,
                        "condition": dict(condition) if condition is not None else None,
                        "condition_index": condition_index,
                        "graph_index_within_condition": graph_index_within_condition,
                        "noisy_cover_hash": noisy_hash,
                        "distance_index": int(distance_index),
                        "requested_incidence_distance": float(requested_distance),
                        "achieved_incidence_distance": float(perturbation["achieved_distance"]),
                        "perturbation_seed": int(perturbation_seed),
                        "perturbation": perturbation,
                        "variant": str(variant),
                        "optimizer_seed": int(optimizer_seed),
                        "seed_nesting": {
                            "graph_seed": int(graph_seed),
                            "perturbation_seed": int(perturbation_seed),
                            "optimizer_seed": int(optimizer_seed),
                        },
                        "resolution": selected_gamma,
                        "max_memberships": int(cap),
                        "same_initial_cover_across_variants": True,
                        "metadata_free": False,
                        "status": status,
                        "error": error,
                        "stage_seconds": {"detector": float(metadata.get("runtime_seconds", time.perf_counter() - started))},
                        "predicted_cover": predicted,
                        "variant_metadata": metadata,
                    }
                    if status == "completed" and predicted is not None:
                        try:
                            score_gt = evaluate_cover(predicted, gt, graph.vcount(), compute_omega=False)
                            score_noisy = evaluate_cover(predicted, noisy, graph.vcount(), compute_omega=False)
                            record["metrics"] = {
                                "ground_truth": score_gt,
                                "reference": score_noisy,
                                "recovery_f1": score_gt.get("matching_f1"),
                                "reference_retention_f1": score_noisy.get("matching_f1"),
                                "reference_distance": 1.0 - float(score_noisy.get("matching_f1", 0.0)),
                            }
                            if final_rows is not None:
                                audit = audit_cover(
                                    graph,
                                    final_rows,
                                    max_memberships=int(cap),
                                    allow_isolation=True,
                                    gamma=selected_gamma,
                                    compute_intervals=False,
                                )
                                record["robustness"] = {
                                    "fixed_profile": audit.get("stable_fraction_at_resolution"),
                                    "endpoint": audit.get("robust_fraction_gamma_0_1"),
                                    "mean_positive_regret": audit.get("mean_positive_regret_at_resolution"),
                                    "max_positive_regret": audit.get("max_positive_regret_at_resolution"),
                                    "profitable_vertex_count": audit.get("profitable_vertex_count_at_resolution"),
                                    "accepted_moves": metadata.get("accepted_moves"),
                                    "projection_events": metadata.get("projection_events"),
                                }
                        except BaseException as exc:
                            record["status"] = "failed_scoring"
                            record["error"] = f"{type(exc).__name__}: {exc}"
                            record["traceback"] = traceback.format_exc()
                    else:
                        record["metrics"] = None
                        record["robustness"] = None
                    rows.append(record)
    return rows


def analyze_robustness_recovery(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate TKT-12/13 graph-level associations, preserving failures."""
    completed = [row for row in rows if row.get("status") == "completed" and isinstance(row.get("metrics"), dict)]
    by_graph_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in completed:
        by_graph_variant[(str(row.get("graph_id")), str(row.get("variant")))].append(row)
    graph_means: list[dict[str, Any]] = []
    for (graph_id, variant), values in sorted(by_graph_variant.items()):
        condition = next((value.get("condition") for value in values if isinstance(value.get("condition"), dict)), None)
        graph_means.append(
            {
                "graph_id": graph_id,
                "variant": variant,
                "n_rows": len(values),
                "condition": condition,
                "mean_recovery_f1": float(np.mean([_numeric((v.get("metrics") or {}).get("recovery_f1")) for v in values])),
                "mean_reference_retention_f1": float(np.mean([_numeric((v.get("metrics") or {}).get("reference_retention_f1")) for v in values])),
                "mean_fixed_profile_robustness": float(np.mean([_numeric((v.get("robustness") or {}).get("fixed_profile")) for v in values])),
                "mean_endpoint_robustness": float(np.mean([_numeric((v.get("robustness") or {}).get("endpoint")) for v in values])),
            }
        )
    raw_x = [entry["mean_fixed_profile_robustness"] for entry in graph_means]
    raw_y = [entry["mean_recovery_f1"] for entry in graph_means]
    raw_retention = [entry["mean_reference_retention_f1"] for entry in graph_means]
    # Condition-stratified residual association removes variant-level mean
    # differences; graph identity remains the independent block.
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in graph_means:
        by_variant[entry["variant"]].append(entry)
    residual_x: list[float] = []
    residual_y: list[float] = []
    for values in by_variant.values():
        mean_x = statistics.fmean(v["mean_fixed_profile_robustness"] for v in values)
        mean_y = statistics.fmean(v["mean_recovery_f1"] for v in values)
        residual_x.extend(v["mean_fixed_profile_robustness"] - mean_x for v in values)
        residual_y.extend(v["mean_recovery_f1"] - mean_y for v in values)

    # Condition summaries are deliberately based on graph means, never on
    # pooled optimizer/perturbation rows.  This keeps graph identity as the
    # independent unit and makes uncertainty auditable.
    by_condition_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in graph_means:
        condition = entry.get("condition") or {}
        by_condition_variant[_condition_label(condition) if condition else "unknown", str(entry["variant"])].append(entry)
    condition_summaries: list[dict[str, Any]] = []
    for (condition_label, variant), values in sorted(by_condition_variant.items()):
        condition_summaries.append(
            {
                "condition": condition_label,
                "variant": variant,
                "n_graphs": len(values),
                "recovery_f1": _mean_interval([_numeric(value["mean_recovery_f1"]) for value in values]),
                "reference_retention_f1": _mean_interval([_numeric(value["mean_reference_retention_f1"]) for value in values]),
                "fixed_profile_robustness": _mean_interval([_numeric(value["mean_fixed_profile_robustness"]) for value in values]),
                "endpoint_robustness": _mean_interval([_numeric(value["mean_endpoint_robustness"]) for value in values]),
            }
        )

    # Paired graph-level differences are a separate estimand from raw
    # association.  The mirror baseline is retained even when a detector
    # fails, so no failed run is converted into an invented score.
    by_graph: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for entry in graph_means:
        by_graph[str(entry["graph_id"])][str(entry["variant"])] = entry
    paired_differences: list[dict[str, Any]] = []
    for variant in sorted({entry["variant"] for entry in graph_means if entry["variant"] != "mirror"}):
        deltas = [
            _numeric(by_variant_rows[variant]["mean_recovery_f1"]) - _numeric(by_variant_rows["mirror"]["mean_recovery_f1"])
            for by_variant_rows in by_graph.values()
            if variant in by_variant_rows and "mirror" in by_variant_rows
        ]
        paired_differences.append({"variant": variant, "against": "mirror", "recovery_f1": _mean_interval(deltas)})
    return {
        "protocol_version": PROTOCOL_VERSION,
        "independent_unit": "graph_id (optimizer/perturbation rows averaged within graph)",
        "n_input_rows": len(rows),
        "n_completed_rows": len(completed),
        "status_counts": {status: sum(str(row.get("status")) == status for row in rows) for status in sorted({str(row.get("status")) for row in rows})},
        "graph_means": graph_means,
        "condition_summaries": condition_summaries,
        "paired_differences": paired_differences,
        "completed_graphs": len({str(row.get("graph_id")) for row in completed}),
        "raw_associations": {
            "fixed_profile_robustness_vs_recovery_f1": _pearson(raw_x, raw_y),
            "fixed_profile_robustness_vs_reference_retention": _pearson(raw_x, raw_retention),
        },
        "condition_stratified_associations": {
            "fixed_profile_robustness_vs_recovery_f1": _pearson(residual_x, residual_y),
            "interpretation": "descriptive; zero or negative association is retained",
        },
    }


def _compact_tracking_row(row: dict[str, Any], shard: str | None = None) -> dict[str, Any]:
    """Return the durable index representation of a full result row.

    Final covers live in per-graph compressed shards.  The index keeps all
    scores, statuses, provenance, and audit summaries while avoiding a
    multi-gigabyte repeated JSONL file for the 28,080-row registered arm.
    """
    compact = dict(row)
    compact.pop("predicted_cover", None)
    if shard is not None:
        compact["result_shard"] = shard
    return compact


def _write_result_shard(output: Path, token: str, rows: Sequence[dict[str, Any]]) -> str:
    relative = Path("result_shards") / f"{token}.json.gz"
    payload = _json_bytes({"schema_version": SCHEMA_VERSION, "rows": list(rows)})
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    _atomic_bytes(output / relative, compressed)
    return str(relative)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"cannot resume malformed tracking results.jsonl line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"cannot resume non-object tracking results.jsonl line {line_number}")
        rows.append(value)
    return rows


def _load_result_shard(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read tracking result shard {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"tracking result shard has invalid schema: {path}")
    if any(not isinstance(row, dict) for row in payload["rows"]):
        raise ValueError(f"tracking result shard contains a non-object row: {path}")
    return [dict(row) for row in payload["rows"]]


def _dedupe_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate the compact result index and reject duplicate keys.

    Tracking rows are persisted as an idempotent index, not an event stream.
    Treating malformed rows or duplicate keys as ignorable would let a damaged
    checkpoint look complete after reload, and conflicting rows could change
    the graph-level means without changing the apparent row count.
    """
    unique: dict[tuple[str, str, float, int, int], dict[str, Any]] = {}
    for row in rows:
        key = tracking_row_key(row)
        if key is None:
            raise ValueError("tracking checkpoint contains a row without a stable graph/variant key")
        if key in unique:
            raise ValueError(f"tracking checkpoint contains duplicate row key: {key}")
        unique[key] = dict(row)
    return list(unique.values())


def _index_graph_records(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate the graph index before using graph IDs as idempotency keys."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("tracking graph index contains a non-object record")
        graph_id = str(record.get("graph_id") or "")
        if not graph_id:
            raise ValueError("tracking graph index contains a record without graph_id")
        if graph_id in indexed:
            raise ValueError(f"tracking graph index contains duplicate graph_id: {graph_id}")
        indexed[graph_id] = dict(record)
    return indexed


def load_tracking_artifacts(output_dir: str | Path, *, include_shards: bool = False) -> dict[str, Any]:
    """Reload and validate a tracking run without rerunning detectors."""
    output = expand_path(output_dir)
    config = _load_json(output / "config.json") if (output / "config.json").is_file() else {}
    manifest = _load_json(output / "manifest.json") if (output / "manifest.json").is_file() else {}
    progress = _load_json(output / "progress.json") if (output / "progress.json").is_file() else {}
    graphs = _load_json(output / "graphs.json") if (output / "graphs.json").is_file() else []
    if not isinstance(graphs, list):
        raise ValueError("tracking graph index must be a JSON list")
    graph_records = [dict(row) for row in graphs]
    _index_graph_records(graph_records)
    rows = _dedupe_rows(_read_jsonl(output / "results.jsonl"))
    recomputed = analyze_robustness_recovery(rows)
    stored = _load_json(output / "robustness_recovery_analysis.json") if (output / "robustness_recovery_analysis.json").is_file() else None
    if stored is not None and stored.get("n_input_rows") != recomputed.get("n_input_rows"):
        raise ValueError("tracking analysis row count does not match the durable result index")
    result = {
        "config": config,
        "manifest": manifest,
        "progress": progress,
        "graphs": graph_records,
        "rows": rows,
        "analysis": recomputed,
        "analysis_on_disk": stored,
    }
    if include_shards:
        full_rows: list[dict[str, Any]] = []
        for shard in sorted((output / "result_shards").glob("*.json.gz")):
            full_rows.extend(_load_result_shard(shard))
        shard_rows = _dedupe_rows(full_rows)
        index_keys = {tracking_row_key(row) for row in rows}
        shard_keys = {tracking_row_key(row) for row in shard_rows}
        if index_keys != shard_keys:
            raise ValueError("tracking result shards do not match the durable result index")
        result["shard_rows"] = shard_rows
    return result


def _tracking_launch_command(
    *, output_dir: Path, profile: str, graph_ledger: Path | None, resume: bool,
) -> str:
    command = ["hedonic-exp", "overlapping-tracking", "--profile", str(profile)]
    if graph_ledger is not None:
        command.extend(["--graph-ledger", str(graph_ledger)])
    command.extend(["--output-dir", str(output_dir)])
    if resume:
        command.append("--resume")
    return " ".join(command)


def build_tracking_preflight(
    *,
    output_dir: str | Path,
    profile: str = "standard",
    graph_ledger: str | Path | None = None,
    max_graphs: int | None = None,
    distance_levels: Sequence[float] | None = None,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    perturbation_seeds: Sequence[int] = (0, 1, 2),
    optimizer_seeds: Sequence[int] = (0, 1),
    cap: int = 4,
    pilot_receipt: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Build a no-detector machine-readable launch receipt."""
    profile = str(profile).lower()
    output = expand_path(output_dir).resolve()
    if profile not in {"smoke", "pilot", "standard"}:
        raise ValueError("profile must be smoke, pilot, or standard")
    distances = tuple(DEFAULT_DISTANCES[:3] if profile == "smoke" and distance_levels is None else DEFAULT_DISTANCES if distance_levels is None else distance_levels)
    selected_graphs: list[dict[str, Any]] = []
    blocked_reason: str | None = None
    ledger_path = expand_path(graph_ledger).resolve() if graph_ledger is not None else None
    if profile in {"pilot", "standard"} and ledger_path is None:
        blocked_reason = "canonical_graph_ledger_required"
    elif ledger_path is not None:
        try:
            selected_graphs = load_tracking_graph_ledger(ledger_path, profile=profile, max_graphs=max_graphs)
        except ValueError as exc:
            blocked_reason = str(exc)
    else:
        n_graphs = 2 if max_graphs is None else max(0, int(max_graphs))
        selected_graphs = [{"graph_id": f"g{index:04d}"} for index in range(n_graphs)]
    if profile == "pilot" and max_graphs is None and ledger_path is not None:
        # Pilot is the registered 20-graph implementation/resource check.
        selected_graphs = selected_graphs[:TRACKING_PILOT_GRAPHS]
    rows_per_graph = (1 if 0.0 in {float(value) for value in distances} else 0) * sum(
        1 if str(variant) == "mirror" else len(tuple(int(seed) for seed in optimizer_seeds))
        for variant in variants
    ) + sum(
        1
        for value in distances
        if float(value) != 0.0
    ) * len(tuple(int(seed) for seed in perturbation_seeds)) * sum(
        1 if str(variant) == "mirror" else len(tuple(int(seed) for seed in optimizer_seeds))
        for variant in variants
    )
    pilot = None
    if pilot_receipt is not None:
        try:
            pilot = _load_json(expand_path(pilot_receipt))
        except ValueError as exc:
            blocked_reason = blocked_reason or f"invalid_pilot_receipt: {exc}"
    estimate = {"available": False, "seconds_per_graph": None, "estimated_seconds": None}
    if isinstance(pilot, dict):
        seconds_per_graph = _numeric(pilot.get("seconds_per_graph"), default=float("nan"))
        if math.isfinite(seconds_per_graph) and seconds_per_graph > 0:
            estimate = {
                "available": True,
                "seconds_per_graph": seconds_per_graph,
                "estimated_seconds": seconds_per_graph * len(selected_graphs),
                "source": str(expand_path(pilot_receipt).resolve()),
            }
    if blocked_reason is None and profile == "standard" and graph_ledger is None:
        blocked_reason = "canonical_graph_ledger_required"
    planned_graphs = (
        240
        if profile == "standard" and max_graphs is None
        else TRACKING_PILOT_GRAPHS
        if profile == "pilot" and max_graphs is None
        else max(0, int(max_graphs))
        if max_graphs is not None
        else len(selected_graphs)
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "preflight_status": "ready" if blocked_reason is None else "blocked",
        # A preflight is detector-free by contract.  Keep explicit integer
        # counters so receipt verifiers cannot infer zero work from a blocked
        # status or confuse an omitted field with a successful launch.
        "no_production_graphs_launched": True,
        "production_graphs_launched": 0,
        "production_result_rows": 0,
        "blocked_reason": blocked_reason,
        "profile": profile,
        "condition_policy": TRACKING_CONDITION_POLICY,
        "registered_conditions": list(TRACKING_CONDITIONS),
        "selected_graphs": len(selected_graphs),
        "expected_graphs": int(planned_graphs),
        "distance_levels": [float(value) for value in distances],
        "variants": [str(value) for value in variants],
        "perturbation_seeds": [int(value) for value in perturbation_seeds],
        "optimizer_seeds": [int(value) for value in optimizer_seeds],
        "seed_plan": {
            "graph_seed": "inherited verbatim from canonical TKT-11 graph ledger",
            "perturbation_seed": "declared seed per graph/distance; seed 0 only at zero distance",
            "optimizer_seed": "declared nested seed per graph/perturbation/variant",
        },
        "max_memberships": int(cap),
        "rows_per_graph": int(rows_per_graph),
        "expected_result_rows": int(planned_graphs * rows_per_graph),
        "graph_ledger": str(ledger_path) if ledger_path is not None else None,
        "graph_ledger_sha256": _sha256_file(ledger_path) if ledger_path is not None and ledger_path.is_file() else None,
        "canonical_input_policy": "official TKT-11 LFRbenchmarks archives" if profile in {"pilot", "standard"} else "explicit compatibility smoke fixture",
        "environment": environment_receipt(),
        "pilot_estimate": estimate,
        "launch_command": _tracking_launch_command(output_dir=output, profile=profile, graph_ledger=ledger_path, resume=resume),
        "resume": bool(resume),
        "atomic_persistence": "atomic per-graph gzip shards + durable JSONL index + progress.json",
    }
    return receipt


def run_tracking_study(
    *,
    output_dir: str | Path,
    profile: str = "smoke",
    max_graphs: int | None = None,
    distance_levels: Sequence[float] | None = None,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    cap: int = 4,
    graph_ledger: str | Path | None = None,
    perturbation_seeds: Sequence[int] = (0, 1, 2),
    optimizer_seeds: Sequence[int] = (0, 1),
    resume: bool = False,
    pilot_receipt: str | Path | None = None,
    stop_after_graphs: int | None = None,
) -> dict[str, Any]:
    """Run a resumable tracking study with canonical-input and provenance gates.

    ``profile=standard`` and ``profile=pilot`` require a completed TKT-11
    ``graphs.json`` ledger produced by the official LFRbenchmarks adapter.
    ``profile=smoke`` alone may use the explicit compatibility fixture.  Every
    completed graph writes a compressed full-state shard and an atomic index
    checkpoint before the next graph starts; resumption is idempotent by the
    graph/distance/perturbation/variant/optimizer key.
    """
    profile = str(profile).lower()
    if profile not in {"smoke", "pilot", "standard"}:
        raise ValueError("profile must be smoke, pilot, or standard")
    if int(cap) < 1:
        raise ValueError("cap must be positive")
    output = expand_path(output_dir).resolve()
    preflight = build_tracking_preflight(
        output_dir=output,
        profile=profile,
        graph_ledger=graph_ledger,
        max_graphs=max_graphs,
        distance_levels=distance_levels,
        variants=variants,
        perturbation_seeds=perturbation_seeds,
        optimizer_seeds=optimizer_seeds,
        cap=cap,
        pilot_receipt=pilot_receipt,
        resume=True,
    )
    _atomic_json(output / "preflight.json", preflight)
    if preflight["preflight_status"] != "ready":
        config = {
            **preflight,
            "profile": profile,
            "failure_policy": "fail closed before detector execution",
            "status": "blocked",
        }
        _atomic_json(output / "config.json", config)
        progress = {"schema_version": SCHEMA_VERSION, "status": "blocked", "completed_graphs": 0, "rows_persisted": 0}
        _atomic_json(output / "progress.json", progress)
        manifest = {**config, "study_status": "blocked", "result_rows": 0, "status_counts": {}}
        _atomic_json(output / "manifest.json", manifest)
        return manifest

    distances = tuple(float(value) for value in preflight["distance_levels"])
    selected_graphs: list[dict[str, Any]]
    if graph_ledger is not None:
        selected_graphs = load_tracking_graph_ledger(graph_ledger, profile=profile, max_graphs=max_graphs)
        if profile == "pilot" and max_graphs is None:
            selected_graphs = selected_graphs[:TRACKING_PILOT_GRAPHS]
    else:
        selected_graphs = []

    # Existing checkpoints are loaded from the compact index.  A malformed
    # line is fatal: silently dropping it could turn an interrupted run into a
    # false complete ledger.
    existing_rows = _dedupe_rows(_read_jsonl(output / "results.jsonl")) if resume else []
    rows_by_key = {tracking_row_key(row): row for row in existing_rows if tracking_row_key(row) is not None}
    existing_graph_records: list[dict[str, Any]] = []
    if resume and (output / "graphs.json").is_file():
        value = _load_json(output / "graphs.json")
        if not isinstance(value, list):
            raise ValueError("tracking graph index must be a JSON list")
        existing_graph_records = list(value)
    graph_records_by_id = _index_graph_records(existing_graph_records)
    started_total = time.perf_counter()
    processed_this_run = 0
    resume_count = int((_load_json(output / "progress.json") if resume and (output / "progress.json").is_file() else {}).get("resume_count", 0)) + (1 if resume else 0)
    config = {
        **preflight,
        "schema_version": SCHEMA_VERSION,
        "profile": profile,
        "status": "running",
        "failure_policy": "retain generation, perturbation, detector, audit, and scoring failures",
        "supervision": "tracking starts from supplied cover; separate from metadata-free TKT-11",
        "independent_unit": "graph_id",
        "resolution_rule": "graph density fixed before test scoring",
        "perturbation": "degree-preserving bipartite incidence double-edge switches",
        "resume_count": resume_count,
        "environment": environment_receipt(),
    }

    def checkpoint(status: str, current_graph: str | None = None) -> dict[str, Any]:
        analysis = analyze_robustness_recovery(list(rows_by_key.values()))
        status_counts: dict[str, int] = {}
        for row in rows_by_key.values():
            label = str(row.get("status", "failed"))
            status_counts[label] = status_counts.get(label, 0) + 1
        complete_graph_ids = sorted(
            str(record.get("graph_id"))
            for record in graph_records_by_id.values()
            if record.get("status") == "completed"
        )
        elapsed = time.perf_counter() - started_total
        resource_pilot = {
            "protocol_version": PROTOCOL_VERSION,
            "profile": profile,
            "completed_graphs": len(complete_graph_ids),
            "result_rows": len(rows_by_key),
            "wall_seconds": elapsed,
            "seconds_per_graph": elapsed / len(complete_graph_ids) if complete_graph_ids else None,
            "peak_rss_bytes": _peak_rss_bytes(),
            "measurement_scope": "parent process wall clock and process peak RSS; use only for ETA planning",
        }
        progress = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "status": status,
            "resume_count": resume_count,
            "expected_graphs": len(selected_graphs) if selected_graphs else int(preflight.get("expected_graphs", 0)),
            "completed_graphs": len(complete_graph_ids),
            "completed_graph_ids": complete_graph_ids,
            "rows_persisted": len(rows_by_key),
            "current_graph": current_graph,
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": _peak_rss_bytes(),
            "updated_unix": time.time(),
        }
        config_snapshot = {**config, "status": status, "graphs_attempted": len(graph_records_by_id), "result_rows": len(rows_by_key)}
        _atomic_json(output / "config.json", config_snapshot)
        _atomic_json(output / "graphs.json", list(graph_records_by_id.values()))
        _write_jsonl(output / "results.jsonl", list(rows_by_key.values()))
        _write_csv(output / "results.csv", list(rows_by_key.values()))
        _atomic_json(output / "robustness_recovery_analysis.json", analysis)
        if profile in {"smoke", "pilot"} and complete_graph_ids:
            _atomic_json(output / "pilot_receipt.json", resource_pilot)
        manifest = {
            **config_snapshot,
            "study_status": status,
            "result_rows": len(rows_by_key),
            "status_counts": status_counts,
            "analysis": analysis,
            "graph_seed_count": len({record.get("graph_seed") for record in graph_records_by_id.values()}),
            "config_sha256": _sha256_bytes(_json_bytes(config_snapshot)),
            "resource_pilot": resource_pilot,
        }
        _atomic_json(output / "progress.json", progress)
        _atomic_json(output / "manifest.json", manifest)
        return manifest

    # A source-ledger run has all input graphs available before detector work;
    # smoke remains a deterministic compatibility fixture for CI only.
    if graph_ledger is None:
        n_graphs = 2 if max_graphs is None else max(0, int(max_graphs))
        n = 80
        for graph_index in range(n_graphs):
            graph_seed = 10_000 + graph_index
            instance = generate_overlapping_lfr(
                n=n,
                mixing=(0.1, 0.3, 0.5, 0.7)[graph_index % 4],
                overlap_fraction=(0.0, 0.1, 0.3, 0.5)[graph_index % 4],
                overlap_multiplicity=2 if graph_index % 2 else 4,
                average_degree=min(20, n - 1),
                max_degree=min(100, n - 1),
                min_community=min(20, max(2, n // 4)),
                max_community=min(100, max(2, n // 2)),
                seed=graph_seed,
            )
            selected_graphs.append(
                {
                    "graph": instance.graph,
                    "cover": instance.cover,
                    "graph_id": f"g{graph_index:04d}",
                    "condition": None,
                    "condition_index": None,
                    "graph_index_within_condition": graph_index,
                    "graph_seed": graph_seed,
                    "graph_hash": instance.graph_hash,
                    "cover_hash": instance.cover_hash,
                    "archive": None,
                    "metadata": instance.metadata,
                }
            )

    expected_graph_count = len(selected_graphs)
    for graph_index, source in enumerate(selected_graphs):
        graph_id = str(source["graph_id"])
        if graph_id in graph_records_by_id and graph_records_by_id[graph_id].get("status") == "completed":
            # A completed graph checkpoint is idempotent only when all its
            # expected result keys are present.  Otherwise resume continues
            # from the graph rather than silently claiming completion.
            graph_rows = [row for row in rows_by_key.values() if str(row.get("graph_id")) == graph_id]
            if graph_rows:
                continue
        try:
            full_rows = run_tracking_graph(
                source["graph"],
                source["cover"],
                graph_id=graph_id,
                graph_seed=int(source["graph_seed"]),
                condition=source.get("condition"),
                condition_index=source.get("condition_index"),
                graph_index_within_condition=source.get("graph_index_within_condition"),
                input_archive=source.get("archive"),
                distance_levels=distances,
                perturbation_seeds=tuple(int(seed) for seed in perturbation_seeds),
                optimizer_seeds=tuple(int(seed) for seed in optimizer_seeds),
                variants=tuple(str(value) for value in variants),
                cap=int(cap),
            )
            shard_token = f"{graph_index:04d}_{str(source['graph_hash'])[:16]}"
            shard = _write_result_shard(output, shard_token, full_rows)
            for row in full_rows:
                compact = _compact_tracking_row(row, shard=shard)
                key = tracking_row_key(compact)
                if key is not None:
                    rows_by_key[key] = compact
            graph_records_by_id[graph_id] = {
                "graph_id": graph_id,
                "graph_seed": int(source["graph_seed"]),
                "graph_hash": str(source["graph_hash"]),
                "cover_hash": str(source["cover_hash"]),
                "condition": source.get("condition"),
                "condition_index": source.get("condition_index"),
                "graph_index_within_condition": source.get("graph_index_within_condition"),
                "input_archive": source.get("archive"),
                "generator_policy": (source.get("metadata") or {}).get("generator_policy", "canonical_ledger"),
                "result_shard": shard,
                "status": "completed",
            }
        except BaseException as exc:
            graph_records_by_id[graph_id] = {
                "graph_id": graph_id,
                "graph_seed": int(source.get("graph_seed", 0)),
                "graph_hash": source.get("graph_hash"),
                "cover_hash": source.get("cover_hash"),
                "condition": source.get("condition"),
                "condition_index": source.get("condition_index"),
                "graph_index_within_condition": source.get("graph_index_within_condition"),
                "input_archive": source.get("archive"),
                "status": "tracking_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        processed_this_run += 1
        status = "running"
        if stop_after_graphs is not None and processed_this_run >= int(stop_after_graphs):
            status = "interrupted"
        checkpoint(status, current_graph=graph_id)
        if status == "interrupted":
            break
    complete = sum(record.get("status") == "completed" for record in graph_records_by_id.values())
    if complete >= expected_graph_count and expected_graph_count > 0:
        final_status = "completed_with_failures" if any(record.get("status") != "completed" for record in graph_records_by_id.values()) else "completed"
    elif stop_after_graphs is not None and processed_this_run >= int(stop_after_graphs):
        final_status = "interrupted"
    else:
        final_status = "completed_with_failures"
    return checkpoint(final_status)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, _json_bytes(payload))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join((json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8") for row in rows)
    _atomic_bytes(path, payload)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = ["graph_id", "graph_hash", "ground_truth_cover_hash", "noisy_cover_hash", "requested_incidence_distance", "achieved_incidence_distance", "perturbation_seed", "variant", "optimizer_seed", "resolution", "max_memberships", "status", "recovery_f1", "reference_retention_f1", "reference_distance", "fixed_profile_robustness", "endpoint_robustness", "error"]
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        metrics = row.get("metrics") or {}
        robustness = row.get("robustness") or {}
        writer.writerow({**row, "recovery_f1": metrics.get("recovery_f1"), "reference_retention_f1": metrics.get("reference_retention_f1"), "reference_distance": metrics.get("reference_distance"), "fixed_profile_robustness": robustness.get("fixed_profile"), "endpoint_robustness": robustness.get("endpoint")})
    _atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Noisy-cover tracking controls and robustness/recovery triangle (Astra TKT-12)")
    parser.add_argument("--profile", choices=("smoke", "pilot", "standard"), default="smoke")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=OVERLAPPING_ARTIFACTS_DIR / "tracking_triangle")
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--distance-levels", default=None, help="comma-separated incidence distances")
    parser.add_argument("--perturbation-seeds", default="0,1,2", help="comma-separated perturbation seeds")
    parser.add_argument("--optimizer-seeds", default="0,1", help="comma-separated optimizer seeds")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--cap", type=int, default=4, help="maximum memberships for hedonic tracking variants")
    parser.add_argument(
        "--graph-ledger",
        type=Path,
        default=None,
        help="TKT-11 graphs.json ledger; required for pilot/standard canonical runs",
    )
    parser.add_argument("--pilot-receipt", type=Path, default=None, help="bounded pilot manifest used for ETA estimation")
    parser.add_argument("--resume", action="store_true", help="resume from atomic per-graph shards and results.jsonl")
    parser.add_argument("--preflight", action="store_true", help="write preflight.json and perform no detector work")
    return parser


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one integer seed")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    distances = None if args.distance_levels is None else tuple(float(item.strip()) for item in args.distance_levels.split(",") if item.strip())
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    perturbation_seeds = _parse_int_list(args.perturbation_seeds)
    optimizer_seeds = _parse_int_list(args.optimizer_seeds)
    if args.preflight:
        receipt = build_tracking_preflight(
            output_dir=args.output_dir,
            profile=args.profile,
            graph_ledger=args.graph_ledger,
            max_graphs=args.max_graphs,
            distance_levels=distances,
            variants=variants,
            perturbation_seeds=perturbation_seeds,
            optimizer_seeds=optimizer_seeds,
            cap=args.cap,
            pilot_receipt=args.pilot_receipt,
            resume=True,
        )
        output = expand_path(args.output_dir)
        _atomic_json(output / "preflight.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0 if receipt["preflight_status"] == "ready" else 2
    manifest = run_tracking_study(
        output_dir=args.output_dir,
        profile=args.profile,
        max_graphs=args.max_graphs,
        distance_levels=distances,
        variants=variants,
        cap=args.cap,
        graph_ledger=args.graph_ledger,
        perturbation_seeds=perturbation_seeds,
        optimizer_seeds=optimizer_seeds,
        resume=args.resume,
        pilot_receipt=args.pilot_receipt,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest.get("study_status") in {"completed", "completed_with_failures", "interrupted"} else 2


__all__ = [
    "DEFAULT_DISTANCES",
    "DEFAULT_VARIANTS",
    "TRACKING_CONDITIONS",
    "TRACKING_CONDITION_POLICY",
    "TRACKING_GRAPHS_PER_CONDITION",
    "analyze_robustness_recovery",
    "build_tracking_preflight",
    "incidence_count",
    "load_tracking_artifacts",
    "load_tracking_graph_ledger",
    "main",
    "mirror_cover",
    "perturb_cover",
    "run_tracking_graph",
    "run_tracking_study",
    "tracking_row_key",
]
