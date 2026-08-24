"""Ground-truth robustness and GT-seeded equilibrium experiment.

The command is intentionally separate from the locked 125-condition SNAP
benchmark.  It audits supplied overlapping covers, starts the native
``Game.community_hedonic`` detector from the full nested GT membership, and
stores both the returned cover and an independent unit-ell2 equilibrium audit.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import tomllib
from typing import Any, Iterable

import igraph as ig

from hedonic import Game
from hedonic.experiments.config import (
    DEFAULT_NETWORKS_DIR,
    OVERLAPPING_ARTIFACTS_DIR,
    expand_path,
)
from hedonic.experiments.overlapping.execution import run_in_subprocess
from hedonic.experiments.overlapping.ground_truth_data import (
    load_prepared_dataset,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
)
from hedonic.experiments.overlapping.protocol import current_experiment_identity
from hedonic.experiments.overlapping.robustness import (
    audit_cover,
    canonicalize_cover,
    cover_hash,
    cover_to_vertex_memberships,
    fractional_phi,
    perturb_cover_incidence,
    vertex_memberships_to_cover,
)
from hedonic.experiments.overlapping.snap import UnsupportedCoverVariant


SCHEMA_VERSION = 3
PROTOCOL_NAME = "canonical_unique_cover_v3"
DEFAULT_CONFIG = Path("configs/overlapping-ground-truth.toml")
DEFAULT_OUTPUT_DIR = OVERLAPPING_ARTIFACTS_DIR / "ground_truth_robustness_v3"
DEFAULT_DATASETS = ("amazon", "dblp", "livejournal", "youtube")
PROTOCOL_LOCK_PATH = (
    Path(__file__).resolve().parents[4]
    / "configs"
    / "overlapping-ground-truth-protocol.lock.json"
)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _scientific_grid(options: dict[str, Any]) -> dict[str, Any]:
    """Return every scientific axis that defines the substantive ledger."""
    grid = {
        "datasets": list(options["datasets"]),
        "cover": str(options["cover"]),
        "completion_policy": str(options["policy"]),
        "max_nodes": options["max_nodes"],
        "phases": list(options["phases"]),
        "action_policies": list(options["isolation_policies"]),
        "detector_seeds": list(options["detector_seeds"]),
        "perturbation_seeds": list(options["perturbation_seeds"]),
        "zero_distance_perturbation_seed_policy": "first_seed_only",
        "perturbation_design": [
            {
                "target_incidence_distance": float(distance),
                "swap_count_rule": "round_half_up(target_distance*incidences/2)",
                "seeds": _perturbation_seeds_for_distance(
                    options, float(distance)
                ),
            }
            for distance in options["perturbation_distances"]
        ],
        "resolution_multipliers": [float(value) for value in options["multipliers"]],
        "perturbation_incidence_distances": [
            float(value) for value in options["perturbation_distances"]
        ],
        "robustness_profile": [float(value) for value in options["profile_grid"]],
        "omega": bool(options["omega"]),
        "omega_sample_size": int(options["omega_sample_size"]),
        "timeout_seconds": float(options["timeout_seconds"]),
        "robustness_atol": float(options["atol"]),
        "robustness_rtol": float(options["rtol"]),
        "dense_oracle": bool(options["dense"]),
        "analysis_graph_pipeline": (
            "bounded_then_common_undirected_simple_then_covered_induced_v1"
        ),
        "cap_rule": "max(2, supplied_cover_max_memberships_per_node)",
        "initialization_rule": (
            "complete_canonical_supplied_cover_with_degree_preserving_incidence_switches"
        ),
        "detector_parameters": {
            "n_iterations": -1,
            "beta": 0.01,
            "phase_mapping": {
                "local": {"local_move_only": True},
                "multiphase": {"local_move_only": False},
            },
            "policy_mapping": {
                "fixed_labels": {
                    "allow_isolation": False,
                    "ensure_equilibrium": False,
                },
                "open_labels": {
                    "allow_isolation": True,
                    "ensure_equilibrium": True,
                },
            },
            "exact_pre_and_final_memberships_required": True,
        },
        "certificate_semantics": {
            "independent_complete_action_audit": True,
            "target": "exact_labeled_final_membership_state",
            "canonical_unique_body_projection_is_scoring_only": True,
            "successful_perturbation_swaps_must_equal_requested": True,
        },
        "terminal_outcome_policy": {
            "allowed_statuses": ["unsupported_cleanup"],
            "native_label_capacity_is_explicit": True,
            "terminal_outcomes_are_not_scored_returns": True,
        },
    }
    grid["expected_detector_conditions"] = sum(
        1
        for _dataset in options["datasets"]
        for _phase in options["phases"]
        for _policy in options["isolation_policies"]
        for _multiplier in options["multipliers"]
        for _seed in options["detector_seeds"]
        for distance in options["perturbation_distances"]
        for _perturbation_seed in _perturbation_seeds_for_distance(
            options, distance
        )
    )
    return grid


def _protocol_identity(
    config_path: str | Path | None,
    scientific_grid: dict[str, Any],
    *,
    config_sha256: str | None,
) -> dict[str, Any]:
    """Bind GT evidence to its reviewed config and implementation lock."""
    lock_bytes = PROTOCOL_LOCK_PATH.read_bytes()
    lock = json.loads(lock_bytes)
    implementation_identity = current_experiment_identity(
        PROTOCOL_LOCK_PATH, lock_bytes=lock_bytes
    )
    repository = PROTOCOL_LOCK_PATH.resolve().parents[1]
    expected_config = repository / str(
        lock.get("config_path", "configs/overlapping-ground-truth.toml")
    )
    actual_config = Path(config_path).expanduser().resolve() if config_path else None
    canonical_grid = lock.get("canonical_grid")
    effective_grid_sha256 = _json_hash(scientific_grid)
    canonical_grid_sha256 = (
        _json_hash(canonical_grid) if isinstance(canonical_grid, dict) else None
    )
    actual_runtime_environment = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    expected_runtime_environment = lock.get("runtime_environment") or {}
    return {
        "schema_version": int(lock.get("schema_version", -1)),
        "protocol_name": lock.get("protocol_name"),
        "protocol_lock_sha256": implementation_identity.get(
            "protocol_lock_sha256"
        ),
        "implementation_identity": implementation_identity,
        "tracked_files_sha256": _json_hash(
            implementation_identity.get("tracked_files", {})
        ),
        "tracked_files_match_lock": implementation_identity.get(
            "tracked_files_match_lock"
        ),
        "native_dependency_identity_matches_lock": (
            implementation_identity.get("lucas_igraph", {}).get(
                "package_identity_matches_lock"
            )
            and implementation_identity.get("scientific_dependencies_match_lock")
        ),
        "runtime_environment": actual_runtime_environment,
        "expected_runtime_environment": expected_runtime_environment,
        "runtime_environment_matches_lock": (
            actual_runtime_environment == expected_runtime_environment
        ),
        "config_path": str(actual_config) if actual_config else None,
        "expected_config_path": str(expected_config.resolve()),
        "config_path_matches_lock": actual_config == expected_config.resolve(),
        "config_sha256": config_sha256,
        "expected_config_sha256": lock.get("tracked_files", {}).get(
            str(lock.get("config_path", "configs/overlapping-ground-truth.toml"))
        ),
        "effective_grid": scientific_grid,
        "effective_grid_sha256": effective_grid_sha256,
        "canonical_grid_sha256": canonical_grid_sha256,
        "effective_grid_matches_lock": (
            effective_grid_sha256 == canonical_grid_sha256
        ),
        "dataset_content_identities": lock.get("dataset_content_identities", {}),
    }


def _assert_locked_dataset_identity(
    prepared,
    gt_cover: list[list[int]],
    gt_memberships: list[list[int]],
    options: dict[str, Any],
) -> None:
    """Require exact reviewed graph/metadata bytes for the canonical grid."""
    if options["smoke"] or not options["protocol_identity"].get(
        "effective_grid_matches_lock"
    ):
        return
    graph = prepared.dataset.graph
    actual = {
        "schema_version": 2,
        "analysis_graph_policy": (
            "bounded_then_common_undirected_simple_then_covered_induced_v1"
        ),
        "completion_policy": prepared.policy,
        "max_nodes": options["max_nodes"],
        "graph_sha256": prepared.graph_identity,
        "ground_truth_cover_sha256": prepared.ground_truth_identity,
        "n": graph.vcount(),
        "m": graph.ecount(),
        "directed": graph.is_directed(),
        "ground_truth_community_count": len(gt_cover),
        "ground_truth_max_memberships_per_node": max(map(len, gt_memberships)),
    }
    key = f"{prepared.dataset.name}/{prepared.dataset.cover_variant}"
    expected = options["protocol_identity"].get(
        "dataset_content_identities", {}
    ).get(key)
    if actual != expected:
        raise ValueError(
            f"prepared dataset identity differs from the reviewed lock for {key}"
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_gzip_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), sort_keys=True)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _read_gzip_json(path: Path) -> list[list[int]] | None:
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, list) else None


def _parse_csv(value: str | None, cast, *, default: Iterable[Any] = ()) -> list[Any]:
    if value is None:
        return list(default)
    result: list[Any] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if raw:
            result.append(cast(raw))
    if not result:
        raise ValueError("expected at least one comma-separated value")
    return result


def _parse_range(value: str | None, *, default: Iterable[int] = ()) -> list[int]:
    if value is None:
        return list(default)
    result: list[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" in raw and raw.count("-") == 1:
            first, last = raw.split("-", 1)
            start, stop = int(first), int(last)
            step = 1 if stop >= start else -1
            result.extend(range(start, stop + step, step))
        else:
            result.append(int(raw))
    if not result:
        raise ValueError("expected at least one integer seed")
    return result


def _parse_profile(value: str) -> list[float]:
    """Parse ``start:stop:count`` or comma-separated resolution values."""
    if ":" in value:
        fields = value.split(":")
        if len(fields) != 3:
            raise ValueError("profile must be start:stop:count")
        start, stop, count = float(fields[0]), float(fields[1]), int(fields[2])
        if count < 2:
            return [start]
        step = (stop - start) / (count - 1)
        return [start + index * step for index in range(count)]
    return _parse_csv(value, float)


def _cover_artifact_path(output_dir: Path, digest: str) -> Path:
    return output_dir / "covers" / f"{digest}.json.gz"


def _persist_cover(output_dir: Path, cover: list[list[int]]) -> str:
    canonical = canonicalize_cover(cover)
    digest = cover_hash(canonical)
    path = _cover_artifact_path(output_dir, digest)
    if _load_cover_artifact(output_dir, digest) != canonical:
        _atomic_gzip_json(path, canonical)
    return digest


def _load_cover_artifact(output_dir: Path, digest: str) -> list[list[int]] | None:
    cover = _read_gzip_json(_cover_artifact_path(output_dir, digest))
    if cover is None:
        return None
    try:
        return cover if cover_hash(cover) == digest else None
    except (TypeError, ValueError):
        return None


def _membership_artifact_path(output_dir: Path, digest: str) -> Path:
    return output_dir / "raw_memberships" / f"{digest}.json.gz"


def _raw_cover_hash(cover: list[list[int]], n_vertices: int | None = None) -> str:
    """Hash a cover while preserving duplicate communities as a multiset."""
    communities: list[tuple[int, ...]] = []
    for community in cover:
        members = tuple(sorted({int(member) for member in community}))
        if n_vertices is not None and any(
            member < 0 or member >= n_vertices for member in members
        ):
            raise ValueError("cover contains a vertex outside the graph")
        if members:
            communities.append(members)
    payload = [list(members) for members in sorted(communities)]
    return _json_hash(payload)


def _persist_memberships(
    output_dir: Path,
    memberships: list[list[int]],
    n_vertices: int,
) -> str:
    """Persist raw native memberships without globally merging communities."""
    raw_cover = vertex_memberships_to_cover(memberships)
    digest = _raw_cover_hash(raw_cover, n_vertices)
    path = _membership_artifact_path(output_dir, digest)
    if _load_membership_artifact(output_dir, digest) is None:
        _atomic_gzip_json(path, memberships)
    return digest


def _load_membership_artifact(
    output_dir: Path,
    digest: str,
) -> list[list[int]] | None:
    memberships = _read_gzip_json(_membership_artifact_path(output_dir, digest))
    if memberships is None:
        return None
    try:
        raw_cover = vertex_memberships_to_cover(memberships)
        actual = _raw_cover_hash(raw_cover, len(memberships))
    except (TypeError, ValueError):
        return None
    return memberships if actual == digest else None


def _normalize_raw_memberships(
    memberships: Any,
    n_vertices: int,
    cap: int,
) -> list[list[int]]:
    """Validate raw native rows and compact labels without merging communities."""
    if not isinstance(memberships, list) or len(memberships) != n_vertices:
        raise ValueError("raw memberships must contain one row per graph vertex")
    label_map: dict[int, int] = {}
    normalized: list[list[int]] = []
    for labels in memberships:
        if not isinstance(labels, (list, tuple)) or not labels:
            raise ValueError("every raw membership row must be non-empty")
        unique = sorted({int(label) for label in labels})
        if unique[0] < 0:
            raise ValueError("raw membership labels must be non-negative")
        if len(unique) > cap:
            raise ValueError("detector returned a cover above the configured cap")
        row: list[int] = []
        for label in unique:
            if label not in label_map:
                label_map[label] = len(label_map)
            row.append(label_map[label])
        normalized.append(row)
    if not label_map:
        raise ValueError("raw memberships contain no community labels")
    return normalized


def _resume_artifacts_match(
    output_dir: Path,
    record: dict[str, Any],
    *,
    n_vertices: int,
    cap: int,
) -> bool:
    """Verify every exact-state artifact before accepting a cached return."""
    final_membership_hash = record.get("final_membership_hash")
    pre_cleanup_hash = record.get("pre_cleanup_membership_hash")
    final_cover_hash = record.get("final_cover_hash")
    if not all(
        isinstance(value, str) and value
        for value in (final_membership_hash, pre_cleanup_hash, final_cover_hash)
    ):
        return False
    final_memberships = _load_membership_artifact(
        output_dir, str(final_membership_hash)
    )
    pre_cleanup_memberships = _load_membership_artifact(
        output_dir, str(pre_cleanup_hash)
    )
    final_cover = _load_cover_artifact(output_dir, str(final_cover_hash))
    if (
        final_memberships is None
        or pre_cleanup_memberships is None
        or final_cover is None
    ):
        return False
    try:
        normalized = _normalize_raw_memberships(final_memberships, n_vertices, cap)
        _normalize_raw_memberships(pre_cleanup_memberships, n_vertices, cap)
        projected = canonicalize_cover(
            vertex_memberships_to_cover(normalized), n_vertices
        )
    except (TypeError, ValueError):
        return False
    return (
        projected == final_cover
        and cover_hash(projected, n_vertices) == final_cover_hash
        and record.get("raw_membership_hash") == final_membership_hash
    )


def _detector_worker(
    graph,
    initial_membership: list[list[int]],
    max_memberships: int,
    gamma: float,
    phase: str,
    allow_isolation: bool,
    seed: int,
) -> dict[str, Any]:
    """Top-level worker target so multiprocessing spawn can pickle it."""
    ig.set_random_number_generator(random.Random(int(seed)))
    result = Game(graph).community_hedonic(
        initial_membership=initial_membership,
        max_memberships=int(max_memberships),
        resolution=float(gamma),
        local_move_only=phase == "local",
        allow_isolation=bool(allow_isolation),
        n_iterations=-1,
        seed=int(seed),
        beta=0.01,
        ensure_equilibrium=bool(allow_isolation),
    )
    preserved = getattr(result, "_hedonic_raw_memberships", None)
    membership = result.membership
    if preserved is not None:
        raw_memberships = [list(map(int, labels)) for labels in preserved]
    else:
        if membership and isinstance(membership[0], (list, tuple)):
            raw_memberships = [list(map(int, labels)) for labels in membership]
        else:
            raw_memberships = [[int(label)] for label in membership]
    if membership and isinstance(membership[0], (list, tuple)):
        final_memberships = [list(map(int, labels)) for labels in membership]
    else:
        final_memberships = [[int(label)] for label in membership]
    return {
        "raw_memberships": raw_memberships,
        "final_memberships": final_memberships,
        "ensure_equilibrium": bool(allow_isolation),
    }


def _condition_key(payload: dict[str, Any]) -> str:
    return _json_hash(payload)[:24]


def _condition_axis_payload(
    *,
    dataset: str,
    cover: str,
    completion_policy: str,
    phase: str,
    action_policy: str,
    resolution_multiplier: float,
    detector_seed: int,
    perturbation_target_distance: float,
    perturbation_seed: int,
) -> dict[str, Any]:
    """Host-independent identity of one factorial condition."""
    return {
        "dataset": str(dataset),
        "cover": str(cover),
        "completion_policy": str(completion_policy),
        "phase": str(phase),
        "action_policy": str(action_policy),
        "resolution_multiplier": float(resolution_multiplier),
        "detector_seed": int(detector_seed),
        "perturbation_target_distance": float(perturbation_target_distance),
        "perturbation_seed": int(perturbation_seed),
    }


def _condition_axis_key(**kwargs: Any) -> str:
    return _json_hash(_condition_axis_payload(**kwargs))


def _expected_condition_axis_keys(options: dict[str, Any]) -> set[str]:
    if options["audit_only"]:
        return set()
    return {
        _condition_axis_key(
            dataset=dataset,
            cover=options["cover"],
            completion_policy=options["policy"],
            phase=phase,
            action_policy=policy,
            resolution_multiplier=multiplier,
            detector_seed=seed,
            perturbation_target_distance=distance,
            perturbation_seed=perturbation_seed,
        )
        for dataset in options["datasets"]
        for phase in options["phases"]
        for policy in options["isolation_policies"]
        for multiplier in options["multipliers"]
        for seed in options["detector_seeds"]
        for distance in options["perturbation_distances"]
        for perturbation_seed in _perturbation_seeds_for_distance(
            options, distance
        )
    }


def _perturbation_seeds_for_distance(
    options: dict[str, Any], distance: float
) -> list[int]:
    """Avoid five identical pseudo-replicates at zero perturbation."""
    seeds = list(options["perturbation_seeds"])
    return seeds[:1] if float(distance) == 0.0 else seeds


def _requested_swaps_for_distance(
    target_distance: float, memberships: list[list[int]]
) -> int:
    """Map a comparable incidence distance to a dataset-specific swap count."""
    target = float(target_distance)
    if not 0.0 <= target <= 1.0:
        raise ValueError("perturbation incidence distance must lie in [0, 1]")
    incidence_count = sum(map(len, memberships))
    if target == 0.0 or incidence_count == 0:
        return 0
    return max(1, int(math.floor(target * incidence_count / 2.0 + 0.5)))


def _condition_path(output_dir: Path, job_key: str, condition_key: str) -> Path:
    return output_dir / "runs" / job_key / f"{condition_key}.json"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _initialization_distance(
    initial_metrics: dict[str, Any],
    final_metrics: dict[str, Any],
    transition_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transition_metrics = transition_metrics or {}
    return {
        "initial_node_micro_f1": initial_metrics.get("node_micro_f1"),
        "final_node_micro_f1": final_metrics.get("node_micro_f1"),
        "distance_to_ground_truth": (
            None
            if final_metrics.get("node_micro_f1") is None
            else 1.0 - float(final_metrics["node_micro_f1"])
        ),
        "initial_to_final_node_micro_f1": transition_metrics.get("node_micro_f1"),
        "initial_to_final_distance": (
            None
            if transition_metrics.get("node_micro_f1") is None
            else 1.0 - float(transition_metrics["node_micro_f1"])
        ),
        "membership_count_delta": (
            final_metrics.get("predicted_average_memberships_per_vertex", 0.0)
            - initial_metrics.get("predicted_average_memberships_per_vertex", 0.0)
        ),
    }


def _cover_change_diagnostics(
    initial_memberships: list[list[int]],
    final_memberships: list[list[int]],
) -> dict[str, Any]:
    """Summarize how a returned cover moved from its warm start."""
    if len(initial_memberships) != len(final_memberships):
        raise ValueError("initial and final memberships have different lengths")
    initial_sets = [set(labels) for labels in initial_memberships]
    final_sets = [set(labels) for labels in final_memberships]
    changed = [first != second for first, second in zip(initial_sets, final_sets)]
    added = sum(len(second - first) for first, second in zip(initial_sets, final_sets))
    removed = sum(len(first - second) for first, second in zip(initial_sets, final_sets))
    return {
        "changed_vertex_fraction": sum(changed) / len(changed) if changed else 0.0,
        "changed_vertex_count": int(sum(changed)),
        "memberships_added": int(added),
        "memberships_removed": int(removed),
        "initial_membership_count": int(sum(map(len, initial_sets))),
        "final_membership_count": int(sum(map(len, final_sets))),
    }


def _score_cover_record(
    *,
    record: dict[str, Any],
    path: Path,
    prepared,
    options: dict[str, Any],
    gt_cover: list[list[int]],
    initial_cover: list[list[int]],
    initial_memberships: list[list[int]],
    final_cover: list[list[int]] | None,
    final_memberships: list[list[int]] | None,
    cap: int,
    policy: str,
    seed: int,
    gamma: float,
    initial_audit: dict[str, Any],
    rescored: bool = False,
    pre_cleanup_memberships: list[list[int]] | None = None,
    audit_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Validate and score a detector cover, shared by fresh and rescore runs."""
    graph = prepared.dataset.graph
    try:
        normalization_source = "raw_memberships"
        if final_memberships is not None:
            raw_memberships = _normalize_raw_memberships(
                final_memberships, graph.vcount(), cap
            )
            raw_cover = vertex_memberships_to_cover(raw_memberships)
        elif final_cover is not None:
            # Compatibility path for old records. New protocol runs always
            # persist raw memberships and never enter this branch.
            normalization_source = "canonical_cover_legacy"
            final_cover = canonicalize_cover(final_cover, graph.vcount())
            raw_cover = final_cover
            raw_memberships = cover_to_vertex_memberships(
                final_cover, graph.vcount(), require_covered=True
            )
        else:
            raise ValueError("detector returned neither raw memberships nor a cover")

        final_cover = canonicalize_cover(raw_cover, graph.vcount())
        canonical_memberships = cover_to_vertex_memberships(
            final_cover, graph.vcount(), require_covered=True
        )
        raw_digest = _raw_cover_hash(raw_cover, graph.vcount())
        normalization_changed = raw_digest != cover_hash(final_cover, graph.vcount())
        duplicate_community_count = len(raw_cover) - len(
            {tuple(sorted(community)) for community in raw_cover}
        )
    except Exception as exc:
        record.update(
            {
                "status": "invalid_cover" if not rescored else "rescore_invalid_cover",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_applicable": False,
            }
        )
        _atomic_json(path, record)
        return record, []

    final_digest = _persist_cover(options["output_dir"], final_cover)
    raw_membership_digest = _persist_memberships(
        options["output_dir"], raw_memberships, graph.vcount()
    )
    pre_cleanup_digest = (
        _persist_memberships(
            options["output_dir"], pre_cleanup_memberships, graph.vcount()
        )
        if pre_cleanup_memberships is not None
        else None
    )
    scoring_started = time.monotonic()
    omega_kwargs = {
        "compute_omega": options["omega"],
        "omega_sample_size": options["omega_sample_size"],
        "omega_seed": seed,
    }
    initial_metrics = evaluate_cover(initial_cover, gt_cover, graph.vcount(), **omega_kwargs)
    final_metrics = evaluate_cover(final_cover, gt_cover, graph.vcount(), **omega_kwargs)
    transition_metrics = evaluate_cover(
        final_cover,
        initial_cover,
        graph.vcount(),
        compute_omega=False,
    )
    metrics_runtime = time.monotonic() - scoring_started
    allow_isolation = policy == "open_labels"
    audit_started = time.monotonic()

    def run_audit(
        memberships: list[list[int]],
        membership_digest: str,
        isolation: bool,
    ) -> dict[str, Any]:
        cache_key = (
            prepared.graph_identity,
            membership_digest,
            int(cap),
            float(gamma),
            bool(isolation),
            float(options["atol"]),
            float(options["rtol"]),
            bool(options["dense"]),
        )
        if audit_cache is not None and cache_key in audit_cache:
            return audit_cache[cache_key]
        result = audit_cover(
            graph,
            memberships,
            max_memberships=cap,
            allow_isolation=isolation,
            gamma=gamma,
            interval=(0.0, 1.0),
            atol=options["atol"],
            rtol=options["rtol"],
            dense=options["dense"],
        )
        if audit_cache is not None:
            audit_cache[cache_key] = result
        return result

    fixed_audit = run_audit(raw_memberships, raw_membership_digest, False)
    open_audit = run_audit(raw_memberships, raw_membership_digest, True)
    selected_audit = open_audit if allow_isolation else fixed_audit
    canonical_selected_audit = (
        run_audit(
            canonical_memberships,
            final_digest,
            allow_isolation,
        )
        if normalization_changed
        else selected_audit
    )
    audit_runtime = time.monotonic() - audit_started
    initial_phi = fractional_phi(graph, initial_memberships, gamma)
    final_phi = fractional_phi(graph, raw_memberships, gamma)
    canonical_phi = fractional_phi(graph, canonical_memberships, gamma)
    record.update(
        {
            "protocol_identity": options["protocol_identity"],
            "status": (
                "completed"
                if selected_audit["is_local_equilibrium_at_resolution"]
                else "completed_non_equilibrium"
            ),
            "equilibrium_status": (
                "verified"
                if selected_audit["is_local_equilibrium_at_resolution"]
                else "stationary_not_equilibrium"
            ),
            "runtime_applicable": selected_audit["is_local_equilibrium_at_resolution"],
            "final_cover_hash": final_digest,
            "raw_cover_hash": raw_digest,
            "raw_membership_hash": raw_membership_digest,
            "final_membership_hash": raw_membership_digest,
            "pre_cleanup_membership_hash": pre_cleanup_digest,
            "normalization_source": normalization_source,
            "normalization_changed": normalization_changed,
            "duplicate_community_count": duplicate_community_count,
            "final_metrics": final_metrics,
            "initial_metrics": initial_metrics,
            "transition_metrics": transition_metrics,
            "distance": _initialization_distance(
                initial_metrics, final_metrics, transition_metrics
            ),
            "cover_change": _cover_change_diagnostics(initial_memberships, raw_memberships),
            "robustness": {
                "selected_policy": selected_audit,
                "fixed_labels": fixed_audit,
                "open_labels": open_audit,
                "canonical_selected_policy": canonical_selected_audit,
            },
            "fractional_phi_initial": initial_phi,
            "fractional_phi_final": final_phi,
            "fractional_phi_delta": final_phi - initial_phi,
            "fractional_phi_canonical": canonical_phi,
            "fractional_phi_normalization_delta": canonical_phi - final_phi,
            "metrics_runtime_seconds": metrics_runtime,
            "robustness_audit_runtime_seconds": audit_runtime,
            "scoring_runtime_seconds": time.monotonic() - scoring_started,
        }
    )
    if rescored:
        record["rescored_at"] = time.time()
    _atomic_json(path, record)
    return record, final_cover


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    configured_path = args.config or os.getenv("HEDONIC_CONFIG") or DEFAULT_CONFIG
    config_path = expand_path(configured_path)
    config: dict[str, Any] = {}
    config_sha256: str | None = None
    if config_path.is_file():
        config_bytes = config_path.read_bytes()
        config = tomllib.loads(config_bytes.decode("utf-8"))
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    section = config.get("overlapping_ground_truth_robustness", {})
    if not isinstance(section, dict):
        section = {}
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}

    def choose(cli_value, key: str, default):
        return cli_value if cli_value is not None else section.get(key, default)

    smoke = bool(args.smoke or args.profile == "smoke")
    if smoke:
        datasets = ["amazon"]
        cover = "top5000"
        phases = ["local"] if args.phases is None else _parse_csv(args.phases, str)
        isolation = (
            ["fixed_labels"]
            if args.isolation_policies is None
            else _parse_csv(args.isolation_policies, str)
        )
        detector_seeds = [0] if args.seeds is None else _parse_range(args.seeds)
        perturbation_seeds = (
            [0]
            if args.perturbation_seeds is None
            else _parse_range(args.perturbation_seeds)
        )
        multipliers = (
            [1.0]
            if args.resolution_multipliers is None
            else _parse_csv(args.resolution_multipliers, float)
        )
        perturbation_distances = (
            [0.0]
            if args.perturbation_distances is None
            else _parse_csv(args.perturbation_distances, float)
        )
        max_nodes = None
    else:
        jobs = section.get("jobs", [])
        configured_datasets = [
            str(job.get("dataset"))
            for job in jobs
            if isinstance(job, dict) and job.get("dataset")
        ]
        datasets = _parse_csv(
            args.datasets,
            str,
            default=configured_datasets or list(DEFAULT_DATASETS),
        )
        cover = choose(args.cover, "cover", "top5000")
        phases = _parse_csv(
            args.phases,
            str,
            default=section.get("phases", ["local", "multiphase"]),
        )
        isolation = _parse_csv(
            args.isolation_policies,
            str,
            default=section.get("isolation_policies", ["fixed_labels", "open_labels"]),
        )
        detector_seeds = _parse_range(
            args.seeds,
            default=_parse_range(str(section.get("seeds", "0-4"))),
        )
        perturbation_seeds = _parse_range(
            args.perturbation_seeds,
            default=_parse_range(
                str(section.get("perturbation_seeds", section.get("seeds", "0-4")))
            ),
        )
        multipliers = _parse_csv(
            args.resolution_multipliers,
            float,
            default=section.get("resolution_multipliers", [1, 10, 100]),
        )
        perturbation_distances = _parse_csv(
            args.perturbation_distances,
            float,
            default=section.get(
                "perturbation_incidence_distances", [0.0]
            ),
        )
        max_nodes = int(choose(args.max_nodes, "max_nodes", 0) or 0) or None

    perturbation_distances = [float(value) for value in perturbation_distances]
    if (
        not perturbation_distances
        or len(set(perturbation_distances)) != len(perturbation_distances)
        or any(not 0.0 <= value <= 1.0 for value in perturbation_distances)
    ):
        raise ValueError(
            "perturbation incidence distances must be distinct values in [0, 1]"
        )

    data_root = args.data_root
    if data_root is None:
        data_root = (
            os.getenv("HEDONIC_NETWORKS_DIR")
            or section.get("data_root")
            or paths.get("networks_dir")
            or DEFAULT_NETWORKS_DIR
        )
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            os.getenv("HEDONIC_OUTPUT_DIR")
            or section.get("output_dir")
            or paths.get("output_dir")
            or DEFAULT_OUTPUT_DIR
        )
    output_dir = expand_path(output_dir)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path) if config_path.is_file() else None,
        "config_sha256": config_sha256,
        "section": section,
        "smoke": smoke,
        "datasets": datasets,
        "cover": str(cover),
        "data_root": str(expand_path(data_root)),
        "output_dir": output_dir,
        "policy": args.uncovered_policy or section.get("uncovered_policy", "covered-induced"),
        "max_nodes": max_nodes,
        "phases": phases,
        "isolation_policies": isolation,
        "seeds": detector_seeds,
        "detector_seeds": detector_seeds,
        "perturbation_seeds": perturbation_seeds,
        "multipliers": multipliers,
        "perturbation_distances": perturbation_distances,
        "profile_grid": _parse_profile(
            args.robustness_profile
            or section.get("robustness_profile", "0:1:11")
        ),
        "omega": bool(args.omega or section.get("omega", False)),
        "omega_sample_size": int(
            args.omega_sample_size
            if args.omega_sample_size != 100_000
            else section.get("omega_sample_size", 100_000)
        ),
        "timeout_seconds": float(
            args.timeout_per_run
            if args.timeout_per_run is not None
            else section.get("timeout_per_run", 120.0 if smoke else 3600.0)
        ),
        "atol": float(
            args.robustness_atol
            if args.robustness_atol != 1e-10
            else section.get("robustness_atol", 1e-10)
        ),
        "rtol": float(
            args.robustness_rtol
            if args.robustness_rtol != 1e-9
            else section.get("robustness_rtol", 1e-9)
        ),
        "dense": bool(args.dense_oracle),
        "audit_only": bool(args.audit_only),
        "rescore_only": bool(args.rescore_only),
        "resume": bool(args.resume) and not bool(args.force),
    }


def _audit_ground_truth(
    prepared,
    options: dict[str, Any],
) -> dict[str, Any]:
    graph = prepared.dataset.graph
    memberships = cover_to_vertex_memberships(
        prepared.dataset.cover, graph.vcount(), require_covered=True
    )
    max_gt = max(len(labels) for labels in memberships)
    cap = max(2, max_gt)
    density = graph.density()
    policies: dict[str, Any] = {}
    for policy in ("fixed_labels", "open_labels"):
        allow_isolation = policy == "open_labels"
        profile = []
        for gamma in options["profile_grid"]:
            profile.append(
                {
                    "gamma": float(gamma),
                    **audit_cover(
                        graph,
                        memberships,
                        max_memberships=cap,
                        allow_isolation=allow_isolation,
                        gamma=float(gamma),
                        interval=(0.0, 1.0),
                        atol=options["atol"],
                        rtol=options["rtol"],
                        dense=options["dense"],
                    ),
                }
            )
        policies[policy] = {
            "allow_isolation": allow_isolation,
            "cap": cap,
            "profile": profile,
            "at_density": audit_cover(
                graph,
                memberships,
                max_memberships=cap,
                allow_isolation=allow_isolation,
                gamma=min(density, 1.0),
                interval=(0.0, 1.0),
                atol=options["atol"],
                rtol=options["rtol"],
                dense=options["dense"],
            ),
        }
        # Keep the short historical key for consumers of the first smoke
        # artifact, while making the operating-resolution meaning explicit.
        policies[policy]["endpoint"] = policies[policy]["at_density"]
    cap_plus_one = {}
    for policy in ("fixed_labels", "open_labels"):
        cap_plus_one[policy] = audit_cover(
            graph,
            memberships,
            max_memberships=cap + 1,
            allow_isolation=policy == "open_labels",
            gamma=min(density, 1.0),
            interval=(0.0, 1.0),
            atol=options["atol"],
            rtol=options["rtol"],
            dense=options["dense"],
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "protocol_identity": options["protocol_identity"],
        "dataset": prepared.dataset.name,
        "cover": prepared.dataset.cover_variant,
        "policy": prepared.policy,
        "graph_identity": prepared.graph_identity,
        "ground_truth_identity": prepared.ground_truth_identity,
        "n_vertices": graph.vcount(),
        "n_edges": graph.ecount(),
        "density": density,
        "ground_truth_max_memberships": max_gt,
        "effective_max_memberships": cap,
        "policies": policies,
        "cap_plus_one": cap_plus_one,
        "report": prepared.dataset.report,
    }


def _build_condition(
    *,
    prepared,
    options: dict[str, Any],
    gt_memberships: list[list[int]],
    initial_cover_identity: str,
    perturbation: dict[str, Any],
    phase: str,
    policy: str,
    multiplier: float,
    seed: int,
) -> dict[str, Any]:
    graph = prepared.dataset.graph
    allow_isolation = policy == "open_labels"
    cap = max(2, max(len(labels) for labels in gt_memberships))
    gamma = min(graph.density() * float(multiplier), 1.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "dataset": prepared.dataset.name,
        "cover": prepared.dataset.cover_variant,
        "completion_policy": prepared.policy,
        "graph_identity": prepared.graph_identity,
        "ground_truth_identity": prepared.ground_truth_identity,
        "initial_cover_identity": initial_cover_identity,
        "perturbation": perturbation,
        "phase": phase,
        "policy": policy,
        "allow_isolation": allow_isolation,
        "max_memberships": cap,
        "resolution_multiplier": float(multiplier),
        "gamma": gamma,
        "seed": int(seed),
        "n_iterations": -1,
        "beta": 0.01,
        "ensure_equilibrium": bool(allow_isolation),
        "raw_memberships": True,
        "robustness_atol": options["atol"],
        "robustness_rtol": options["rtol"],
        "metric_sampling": {
            "omega": bool(options["omega"]),
            "omega_sample_size": int(options["omega_sample_size"]),
        },
    }


def _run_record(
    *,
    prepared,
    options: dict[str, Any],
    gt_cover: list[list[int]],
    gt_memberships: list[list[int]],
    initial_cover: list[list[int]],
    initial_memberships: list[list[int]],
    perturbation: dict[str, Any],
    phase: str,
    policy: str,
    multiplier: float,
    seed: int,
    initial_audit: dict[str, Any],
    audit_cache: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[list[int]] | None]:
    graph = prepared.dataset.graph
    allow_isolation = policy == "open_labels"
    cap = max(2, max(len(labels) for labels in gt_memberships))
    gamma = min(graph.density() * float(multiplier), 1.0)
    initial_digest = cover_hash(initial_cover, graph.vcount())
    condition = _build_condition(
        prepared=prepared,
        options=options,
        gt_memberships=gt_memberships,
        initial_cover_identity=initial_digest,
        perturbation=perturbation,
        phase=phase,
        policy=policy,
        multiplier=multiplier,
        seed=seed,
    )
    condition_key = _condition_key(condition)
    condition_axis_key = _condition_axis_key(
        dataset=prepared.dataset.name,
        cover=prepared.dataset.cover_variant,
        completion_policy=prepared.policy,
        phase=phase,
        action_policy=policy,
        resolution_multiplier=multiplier,
        detector_seed=seed,
        perturbation_target_distance=perturbation.get(
            "target_incidence_distance", 0.0
        ),
        perturbation_seed=perturbation.get("seed", 0),
    )
    output_dir: Path = options["output_dir"]
    path = _condition_path(
        output_dir,
        f"{prepared.dataset.name}-{prepared.dataset.cover_variant}",
        condition_key,
    )
    expected_identity = _json_hash(condition)
    existing = _read_json(path)
    if (
        options["resume"]
        and existing is not None
        and existing.get("condition_identity") == expected_identity
        and existing.get("condition_key") == condition_key
        and existing.get("condition_axis_key") == condition_axis_key
        and existing.get("protocol_identity") == options["protocol_identity"]
        and existing.get("status") in {"completed", "completed_non_equilibrium"}
        and _resume_artifacts_match(
            output_dir,
            existing,
            n_vertices=graph.vcount(),
            cap=cap,
        )
    ):
        final_memberships = _load_membership_artifact(
            output_dir, str(existing["final_membership_hash"])
        )
        pre_cleanup_memberships = _load_membership_artifact(
            output_dir, str(existing["pre_cleanup_membership_hash"])
        )
        final_cover = _load_cover_artifact(
            output_dir, str(existing["final_cover_hash"])
        )
        if (
            final_memberships is not None
            and pre_cleanup_memberships is not None
            and final_cover is not None
        ):
            rescored, _ = _score_cover_record(
                record=existing,
                path=path,
                prepared=prepared,
                options=options,
                gt_cover=gt_cover,
                initial_cover=initial_cover,
                initial_memberships=initial_memberships,
                final_cover=final_cover,
                final_memberships=final_memberships,
                cap=cap,
                policy=policy,
                seed=seed,
                gamma=gamma,
                initial_audit=initial_audit,
                rescored=True,
                pre_cleanup_memberships=pre_cleanup_memberships,
                audit_cache=audit_cache,
            )
            return rescored, None

    outcome = run_in_subprocess(
        _detector_worker,
        graph,
        initial_memberships,
        cap,
        gamma,
        phase,
        allow_isolation,
        int(seed),
        timeout_seconds=options["timeout_seconds"],
        packet_dir=output_dir / "worker_packets",
    )
    base_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "condition_key": condition_key,
        "condition_axis_key": condition_axis_key,
        "condition_identity": expected_identity,
        "condition": condition,
        "protocol_identity": options["protocol_identity"],
        "dataset": prepared.dataset.name,
        "cover": prepared.dataset.cover_variant,
        "policy": prepared.policy,
        "source_covered_vertices": len(prepared.source_covered_vertices),
        "synthetic_completion_count": prepared.completion_count,
        "initial_cover_hash": initial_digest,
        "ground_truth_cover_hash": cover_hash(gt_cover, graph.vcount()),
        "initial_audit": initial_audit,
        "perturbation": perturbation,
        "detector_runtime_seconds": outcome.runtime_seconds,
        "detector_peak_rss_bytes": outcome.peak_rss_bytes,
        "status": outcome.status,
        "error": outcome.error,
        "runtime_applicable": outcome.status == "ok",
    }
    if outcome.status != "ok" or not isinstance(outcome.payload, dict):
        if (
            outcome.status == "failed"
            and isinstance(outcome.error, str)
            and "ensure_equilibrium cleanup cannot encode" in outcome.error
        ):
            base_record["status"] = "unsupported_cleanup"
            base_record["error_kind"] = "native_label_capacity"
        _atomic_json(path, base_record)
        return base_record, None

    raw_memberships = outcome.payload.get("raw_memberships")
    final_memberships = outcome.payload.get("final_memberships")
    if not isinstance(raw_memberships, list) or not isinstance(final_memberships, list):
        base_record.update(
            {
                "status": "invalid_worker_payload",
                "error": "worker did not return raw and final memberships",
                "runtime_applicable": False,
            }
        )
        _atomic_json(path, base_record)
        return base_record, None

    scored, final_cover = _score_cover_record(
        record=base_record,
        path=path,
        prepared=prepared,
        options=options,
        gt_cover=gt_cover,
        initial_cover=initial_cover,
        initial_memberships=initial_memberships,
        final_cover=None,
        final_memberships=final_memberships,
        cap=cap,
        policy=policy,
        seed=seed,
        gamma=gamma,
        initial_audit=initial_audit,
        pre_cleanup_memberships=raw_memberships,
        audit_cache=audit_cache,
    )
    return scored, final_cover or None


def _rescore_dataset_records(
    *,
    prepared,
    options: dict[str, Any],
    gt_cover: list[list[int]],
    gt_memberships: list[list[int]],
) -> list[dict[str, Any]]:
    """Re-score persisted covers without ever launching a detector."""
    output_dir: Path = options["output_dir"]
    job_dir = output_dir / "runs" / f"{prepared.dataset.name}-{options['cover']}"
    if not job_dir.is_dir():
        return []
    graph = prepared.dataset.graph
    records: list[dict[str, Any]] = []
    audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    initial_audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for path in sorted(job_dir.glob("*.json")):
        record = _read_json(path)
        if not record:
            continue
        condition = record.get("condition", {})
        perturbation = condition.get("perturbation", {})
        try:
            phase = str(condition["phase"])
            policy = str(condition["policy"])
            multiplier = float(condition["resolution_multiplier"])
            seed = int(condition["seed"])
            target_distance = float(
                perturbation["target_incidence_distance"]
            )
            requested_swaps = int(perturbation["requested_swaps"])
            perturbation_seed = int(perturbation["seed"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            phase not in options["phases"]
            or policy not in options["isolation_policies"]
            or multiplier not in options["multipliers"]
            or seed not in options["detector_seeds"]
            or target_distance not in options["perturbation_distances"]
            or perturbation_seed
            not in _perturbation_seeds_for_distance(options, target_distance)
        ):
            continue

        # Reconstruct the complete detector input from the locked factorial
        # axes.  Rescoring must never promote a self-consistent but altered
        # record merely because its high-level axes happen to be in range.
        expected_requested_swaps = _requested_swaps_for_distance(
            target_distance, gt_memberships
        )
        if requested_swaps != expected_requested_swaps:
            continue
        expected_initial_cover, expected_perturbation = perturb_cover_incidence(
            gt_cover,
            graph.vcount(),
            swaps=expected_requested_swaps,
            seed=perturbation_seed,
        )
        expected_perturbation["target_incidence_distance"] = target_distance
        if expected_perturbation["successful_swaps"] != expected_requested_swaps:
            continue
        initial_cover = canonicalize_cover(
            expected_initial_cover, graph.vcount()
        )
        initial_digest = cover_hash(initial_cover, graph.vcount())
        expected_condition = _build_condition(
            prepared=prepared,
            options=options,
            gt_memberships=gt_memberships,
            initial_cover_identity=initial_digest,
            perturbation=expected_perturbation,
            phase=phase,
            policy=policy,
            multiplier=multiplier,
            seed=seed,
        )
        expected_identity = _json_hash(expected_condition)
        expected_key = _condition_key(expected_condition)
        expected_axis_key = _condition_axis_key(
            dataset=prepared.dataset.name,
            cover=options["cover"],
            completion_policy=prepared.policy,
            phase=phase,
            action_policy=policy,
            resolution_multiplier=multiplier,
            detector_seed=seed,
            perturbation_target_distance=target_distance,
            perturbation_seed=perturbation_seed,
        )
        if (
            record.get("protocol_identity") != options["protocol_identity"]
            or record.get("dataset") != prepared.dataset.name
            or record.get("cover") != options["cover"]
            or condition != expected_condition
            or record.get("condition_identity") != expected_identity
            or record.get("condition_key") != expected_key
            or record.get("condition_axis_key") != expected_axis_key
            or record.get("initial_cover_hash") != initial_digest
            or record.get("perturbation") != expected_perturbation
        ):
            continue
        if (
            record.get("status") == "unsupported_cleanup"
            and record.get("error_kind") == "native_label_capacity"
        ):
            # This is a registered terminal condition: the native cleanup
            # cannot encode the full returned label bank.  Preserve it during
            # detector-free replay instead of treating the missing cover as a
            # corrupt artifact or silently rerunning a different algorithm.
            record["rescored_at"] = time.time()
            _atomic_json(path, record)
            records.append(record)
            continue
        initial_memberships = cover_to_vertex_memberships(
            initial_cover, graph.vcount()
        )
        cap = int(expected_condition["max_memberships"])
        gamma = float(expected_condition["gamma"])
        final_hash = record.get("final_cover_hash")
        raw_membership_hash = record.get("final_membership_hash")
        pre_cleanup_hash = record.get("pre_cleanup_membership_hash")
        final_memberships = (
            _load_membership_artifact(output_dir, str(raw_membership_hash))
            if isinstance(raw_membership_hash, str)
            else None
        )
        pre_cleanup_memberships = (
            _load_membership_artifact(output_dir, str(pre_cleanup_hash))
            if isinstance(pre_cleanup_hash, str)
            else None
        )
        final_cover = (
            _load_cover_artifact(output_dir, str(final_hash))
            if isinstance(final_hash, str)
            else None
        )
        if (
            final_memberships is None
            or pre_cleanup_memberships is None
            or final_cover is None
            or not _resume_artifacts_match(
                output_dir,
                record,
                n_vertices=graph.vcount(),
                cap=cap,
            )
        ):
            record.update(
                {
                    "status": "rescore_missing_cover",
                    "error": (
                        "exact final/pre-cleanup membership and canonical-cover "
                        "artifacts are all required"
                    ),
                    "runtime_applicable": False,
                    "rescored_at": time.time(),
                }
            )
            _atomic_json(path, record)
            records.append(record)
            continue
        initial_audit_key = (
            prepared.graph_identity,
            initial_digest,
            cap,
            policy == "open_labels",
            gamma,
            float(options["atol"]),
            float(options["rtol"]),
            bool(options["dense"]),
        )
        initial_audit = initial_audit_cache.get(initial_audit_key)
        if initial_audit is None:
            initial_audit = audit_cover(
                graph,
                initial_memberships,
                max_memberships=cap,
                allow_isolation=policy == "open_labels",
                gamma=gamma,
                interval=(0.0, 1.0),
                atol=options["atol"],
                rtol=options["rtol"],
                dense=options["dense"],
            )
            initial_audit_cache[initial_audit_key] = initial_audit
        rescored, _ = _score_cover_record(
            record=record,
            path=path,
            prepared=prepared,
            options=options,
            gt_cover=gt_cover,
            initial_cover=initial_cover,
            initial_memberships=initial_memberships,
            final_cover=final_cover,
            final_memberships=final_memberships,
            cap=cap,
            policy=policy,
            seed=seed,
            gamma=gamma,
            initial_audit=initial_audit,
            rescored=True,
            pre_cleanup_memberships=pre_cleanup_memberships,
            audit_cache=audit_cache,
        )
        records.append(rescored)
    return records


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    final_metrics = record.get("final_metrics", {})
    robustness = record.get("robustness", {})
    selected = robustness.get("selected_policy", {}) if isinstance(robustness, dict) else {}
    condition = record.get("condition", {})
    protocol_identity = record.get("protocol_identity", {})
    return {
        "condition_key": record.get("condition_key"),
        "condition_axis_key": record.get("condition_axis_key"),
        "condition_identity": record.get("condition_identity"),
        "dataset": record.get("dataset"),
        "cover": record.get("cover"),
        "phase": condition.get("phase"),
        "protocol": condition.get("protocol"),
        "graph_identity": condition.get("graph_identity"),
        "ground_truth_identity": condition.get("ground_truth_identity"),
        "initial_cover_identity": condition.get("initial_cover_identity"),
        "protocol_identity_sha256": _json_hash(protocol_identity),
        "protocol_lock_sha256": protocol_identity.get("protocol_lock_sha256"),
        "tracked_files_sha256": protocol_identity.get("tracked_files_sha256"),
        "effective_grid_sha256": protocol_identity.get("effective_grid_sha256"),
        "canonical_grid_sha256": protocol_identity.get("canonical_grid_sha256"),
        "canonical_grid": protocol_identity.get("effective_grid_matches_lock"),
        # Keep the two independent policy axes explicit in flat exports.  The
        # top-level policy is the partial-cover completion rule, whereas the
        # condition policy controls whether fresh/isolation labels are legal.
        "policy": record.get("policy"),
        "completion_policy": record.get("policy"),
        "action_policy": condition.get("policy"),
        "gamma": condition.get("gamma"),
        "resolution_multiplier": condition.get("resolution_multiplier"),
        "seed": condition.get("seed"),
        "perturbation_target_distance": record.get("perturbation", {}).get(
            "target_incidence_distance"
        ),
        "perturbation_swaps": record.get("perturbation", {}).get("requested_swaps"),
        "perturbation_successful_swaps": record.get("perturbation", {}).get(
            "successful_swaps"
        ),
        "perturbation_seed": record.get("perturbation", {}).get("seed"),
        "perturbation_distance": record.get("perturbation", {}).get("realized_incidence_distance"),
        "status": record.get("status"),
        "equilibrium_status": record.get("equilibrium_status"),
        "detector_runtime_seconds": record.get("detector_runtime_seconds"),
        "detector_peak_rss_bytes": record.get("detector_peak_rss_bytes"),
        "metrics_runtime_seconds": record.get("metrics_runtime_seconds"),
        "robustness_audit_runtime_seconds": record.get(
            "robustness_audit_runtime_seconds"
        ),
        "scoring_runtime_seconds": record.get("scoring_runtime_seconds"),
        "runtime_applicable": record.get("runtime_applicable"),
        "fractional_phi_delta": record.get("fractional_phi_delta"),
        "fractional_phi_normalization_delta": record.get(
            "fractional_phi_normalization_delta"
        ),
        "accuracy_node_micro_f1": final_metrics.get("node_micro_f1"),
        "accuracy_matching_f1": final_metrics.get("matching_f1"),
        "accuracy_omega": final_metrics.get("omega"),
        "distance_to_ground_truth": record.get("distance", {}).get("distance_to_ground_truth"),
        "initial_to_final_node_micro_f1": record.get("distance", {}).get(
            "initial_to_final_node_micro_f1"
        ),
        "changed_vertex_fraction": record.get("cover_change", {}).get(
            "changed_vertex_fraction"
        ),
        "memberships_added": record.get("cover_change", {}).get("memberships_added"),
        "memberships_removed": record.get("cover_change", {}).get(
            "memberships_removed"
        ),
        "robust_fraction_gamma_0_1": selected.get("robust_fraction_gamma_0_1"),
        "stable_fraction_at_resolution": selected.get("stable_fraction_at_resolution"),
        "max_positive_regret": selected.get("max_positive_regret_at_resolution"),
        "cap_saturated_fraction": selected.get("cap_saturated_fraction"),
        "equilibrium_interval_width": selected.get("equilibrium_interval_width"),
        "final_cover_hash": record.get("final_cover_hash"),
        "raw_cover_hash": record.get("raw_cover_hash"),
        "raw_membership_hash": record.get("raw_membership_hash"),
        "final_membership_hash": record.get("final_membership_hash") or record.get(
            "raw_membership_hash"
        ),
        "pre_cleanup_membership_hash": record.get("pre_cleanup_membership_hash"),
        "normalization_changed": record.get("normalization_changed"),
        "duplicate_community_count": record.get("duplicate_community_count"),
    }


def _publication_evidence_index(
    output_dir: Path,
    rows: list[dict[str, Any]],
    protocol_identity: dict[str, Any],
) -> dict[str, Any]:
    """Bind and semantically verify every detailed record and state artifact."""
    reasons: list[str] = []
    expected_paths: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        cover = str(row.get("cover", ""))
        condition_key = str(row.get("condition_key", ""))
        if not dataset or not cover or not condition_key:
            reasons.append("flat_row_missing_record_identity")
            continue
        relative = f"runs/{dataset}-{cover}/{condition_key}.json"
        if relative in expected_paths:
            reasons.append(f"duplicate_run_record_path:{relative}")
        expected_paths[relative] = row
    actual_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "runs").glob("*/*.json")
    }
    if actual_paths != set(expected_paths):
        reasons.append("run_record_path_set_mismatch")

    run_hashes: dict[str, str] = {}
    record_checks: list[tuple[str, str, str, str, bool]] = []
    terminal_policy = (
        protocol_identity.get("effective_grid", {})
        .get("terminal_outcome_policy", {})
    )
    terminal_statuses = set(terminal_policy.get("allowed_statuses", ()))
    cover_digests: set[str] = set()
    membership_digests: set[str] = set()
    final_membership_digests: set[str] = set()
    for relative, row in sorted(expected_paths.items()):
        path = output_dir / relative
        try:
            encoded = path.read_bytes()
            record = json.loads(encoded)
        except (OSError, json.JSONDecodeError):
            reasons.append(f"invalid_run_record:{relative}")
            continue
        if not isinstance(record, dict):
            reasons.append(f"non_object_run_record:{relative}")
            continue
        run_hashes[relative] = hashlib.sha256(encoded).hexdigest()
        if (
            record.get("protocol_identity") != protocol_identity
            or str(record.get("condition_key")) != str(row.get("condition_key"))
            or str(record.get("condition_axis_key"))
            != str(row.get("condition_axis_key"))
            or str(record.get("dataset")) != str(row.get("dataset"))
            or str(record.get("cover")) != str(row.get("cover"))
            or str(record.get("status")) != str(row.get("status"))
        ):
            reasons.append(f"run_record_flat_row_mismatch:{relative}")
            continue
        status = str(record.get("status"))
        if status in terminal_statuses:
            if (
                status != "unsupported_cleanup"
                or record.get("error_kind") != "native_label_capacity"
                or not terminal_policy.get("native_label_capacity_is_explicit")
            ):
                reasons.append(f"invalid_terminal_status:{relative}")
            # A registered terminal outcome is a condition-level result, not
            # a returned cover.  It must not be forced through the exact-state
            # artifact checks below.
            continue
        selected = (record.get("robustness") or {}).get("selected_policy") or {}
        record_checks.append(
            (
                str(record.get("condition_key")),
                str(record.get("final_cover_hash")),
                str(record.get("final_membership_hash")),
                status,
                selected.get("is_local_equilibrium_at_resolution") is True,
            )
        )
        for key in (
            "initial_cover_hash",
            "ground_truth_cover_hash",
            "final_cover_hash",
        ):
            value = record.get(key)
            if isinstance(value, str):
                cover_digests.add(value)
            else:
                reasons.append(f"missing_{key}:{relative}")
        for key in (
            "final_membership_hash",
            "pre_cleanup_membership_hash",
        ):
            value = record.get(key)
            if isinstance(value, str):
                membership_digests.add(value)
                if key == "final_membership_hash":
                    final_membership_digests.add(value)
            else:
                reasons.append(f"missing_{key}:{relative}")

    artifact_hashes: dict[str, str] = {}
    valid_cover_digests: set[str] = set()
    membership_projection_digests: dict[str, str] = {}
    for digest in sorted(cover_digests):
        relative = f"covers/{digest}.json.gz"
        path = output_dir / relative
        value = _load_cover_artifact(output_dir, digest)
        file_digest = _file_hash(path)
        if value is None or file_digest is None:
            reasons.append(f"invalid_cover_artifact:{relative}")
            continue
        valid_cover_digests.add(digest)
        artifact_hashes[relative] = file_digest
    for digest in sorted(membership_digests):
        relative = f"raw_memberships/{digest}.json.gz"
        path = output_dir / relative
        value = _load_membership_artifact(output_dir, digest)
        file_digest = _file_hash(path)
        if value is None or file_digest is None:
            reasons.append(f"invalid_membership_artifact:{relative}")
            continue
        if digest in final_membership_digests:
            try:
                membership_projection_digests[digest] = cover_hash(
                    canonicalize_cover(
                        vertex_memberships_to_cover(value), len(value)
                    )
                )
            except (TypeError, ValueError):
                reasons.append(f"invalid_membership_projection:{relative}")
                continue
        artifact_hashes[relative] = file_digest

    for condition_key, final_digest, membership_digest, status, certified in record_checks:
        if (
            final_digest in valid_cover_digests
            and membership_projection_digests.get(membership_digest)
            != final_digest
        ):
            reasons.append(f"final_projection_mismatch:{condition_key}")
        if (status == "completed") != certified:
            reasons.append(f"certificate_status_mismatch:{condition_key}")

    return {
        "valid": not reasons,
        "reasons": reasons[:50],
        "run_records_sha256": run_hashes,
        "referenced_artifacts_sha256": artifact_hashes,
    }


def _write_plots(
    output_dir: Path,
    gt_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    protocol_identity: dict[str, Any],
    expected_condition_axis_keys: set[str],
) -> list[str]:
    # Manuscript-specific figures are generated by the private research
    # checkout. The public experiment records auditable JSON/CSV artifacts and
    # deliberately keeps plotting code out of the reusable detector package.
    del output_dir, gt_rows, rows, protocol_identity, expected_condition_axis_keys
    return []


def _assert_output_safe(output_dir: Path, data_root: str | Path) -> None:
    """Keep experiment artifacts out of raw SNAP archives and locked evidence."""
    output = output_dir.expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    if output == data or data in output.parents:
        raise ValueError(
            f"refusing to write experiment artifacts inside raw network data: {output}"
        )
    locked_evidence = (Path("artifacts") / "evidence" / "overlapping_communities").resolve()
    if output == locked_evidence or locked_evidence in output.parents:
        raise ValueError(
            "refusing to write into the locked overlapping-paper evidence tree"
        )


def _assert_no_foreign_run_records(
    output_dir: Path, protocol_identity: dict[str, Any]
) -> None:
    """Fail closed instead of mixing legacy/subset records in one run root."""
    runs = output_dir / "runs"
    if not runs.is_dir():
        return
    foreign: list[str] = []
    for path in sorted(runs.glob("*/*.json")):
        record = _read_json(path)
        if record is None or record.get("protocol_identity") != protocol_identity:
            foreign.append(path.relative_to(output_dir).as_posix())
            if len(foreign) >= 5:
                break
    if foreign:
        raise ValueError(
            "output directory contains foreign or legacy run records; choose "
            f"a fresh output directory (examples: {', '.join(foreign)})"
        )


class _OutputLock:
    def __init__(self, stream) -> None:
        self.stream = stream

    def close(self) -> None:
        if self.stream.closed:
            return
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()

    def __del__(self) -> None:
        self.close()


def _acquire_output_lock(output_dir: Path) -> _OutputLock:
    """Hold a nonblocking process lock for the complete output mutation."""
    path = output_dir / ".ground_truth_robustness.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(
            f"another ground-truth robustness process owns {output_dir}"
        ) from exc
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return _OutputLock(stream)


def run_experiment(args: argparse.Namespace) -> int:
    options = _effective_config(args)
    protocol_identity = _protocol_identity(
        options.get("config_path"),
        _scientific_grid(options),
        config_sha256=options.get("config_sha256"),
    )
    options["protocol_identity"] = protocol_identity
    if not options["smoke"] and not (
        protocol_identity["schema_version"] == 3
        and protocol_identity["protocol_name"]
        == "overlapping-ground-truth-robustness-v3"
        and protocol_identity["tracked_files_match_lock"]
        and protocol_identity["native_dependency_identity_matches_lock"]
        and protocol_identity["runtime_environment_matches_lock"]
        and protocol_identity["config_path_matches_lock"]
        and protocol_identity["config_sha256"]
        == protocol_identity["expected_config_sha256"]
    ):
        raise ValueError(
            "ground-truth robustness code/config does not match the reviewed "
            "protocol lock; refresh the lock only after scientific review"
        )
    output_dir: Path = options["output_dir"]
    canonical_output_dir = expand_path(
        options["section"].get("output_dir", DEFAULT_OUTPUT_DIR)
    ).resolve()
    if (
        options["smoke"] or options["audit_only"]
    ) and (
        args.output_dir is None
        or output_dir.resolve() == canonical_output_dir
    ):
        raise ValueError(
            "smoke and audit-only modes require an explicit, non-default "
            "--output-dir"
        )
    if (
        not options["smoke"]
        and not protocol_identity["effective_grid_matches_lock"]
        and (
            args.output_dir is None
            or output_dir.resolve() == canonical_output_dir
        )
    ):
        raise ValueError(
            "non-canonical scientific overrides require an explicit, "
            "non-default --output-dir"
        )
    _assert_output_safe(output_dir, options["data_root"])
    output_dir.mkdir(parents=True, exist_ok=True)
    # The live file object deliberately remains referenced until this function
    # returns or unwinds; closing it releases the advisory lock automatically.
    _output_lock = _acquire_output_lock(output_dir)
    # Invalidate any prior claim fragments before touching the ledger.  The
    # strict report builder is the only component allowed to set this true
    # after validating current coverage and bound artifact bytes.
    _atomic_text(
        output_dir / "paper" / "gt_publication_ready.tex",
        "\\GTResultsReadyfalse\n",
    )
    _assert_no_foreign_run_records(output_dir, protocol_identity)
    expected_axis_keys = _expected_condition_axis_keys(options)
    expected = len(expected_axis_keys)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "protocol_identity": protocol_identity,
        "canonical_grid": protocol_identity["effective_grid_matches_lock"],
        "started_at": time.time(),
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in options.items()
            if key != "section"
        },
        "environment": {
            "python": platform.python_version(),
            "hedonic": _distribution_version("hedonic") or "local",
            "lucas_igraph": _distribution_version("lucas-igraph"),
            "igraph": getattr(ig, "__version__", None),
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
            "networkx": _distribution_version("networkx"),
            "platform": platform.platform(),
        },
    }
    _atomic_json(output_dir / "plan.json", manifest)
    all_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    for dataset_name in options["datasets"]:
        try:
            prepared = load_prepared_dataset(
                dataset_name,
                cover_variant=options["cover"],
                data_root=options["data_root"],
                policy=options["policy"],
                max_nodes=options["max_nodes"],
                smoke=options["smoke"],
            )
        except Exception as exc:
            status = (
                "unsupported_cover_variant"
                if isinstance(exc, UnsupportedCoverVariant)
                else "data_unavailable"
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            _atomic_json(
                output_dir / "ground_truth" / f"{dataset_name}-{options['cover']}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": dataset_name,
                    "cover": options["cover"],
                    "status": status,
                    "protocol": PROTOCOL_NAME,
                    "protocol_identity": protocol_identity,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            print(f"[{status}] {dataset_name}/{options['cover']}: {exc}")
            continue

        graph = prepared.dataset.graph
        gt_cover = canonicalize_cover(prepared.dataset.cover, graph.vcount())
        gt_memberships = cover_to_vertex_memberships(gt_cover, graph.vcount())
        _assert_locked_dataset_identity(
            prepared, gt_cover, gt_memberships, options
        )
        final_audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        initial_audit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        gt_digest = _persist_cover(output_dir, gt_cover)
        gt_audit_started = time.monotonic()
        gt_audit = _audit_ground_truth(prepared, options)
        gt_audit["audit_runtime_seconds"] = time.monotonic() - gt_audit_started
        gt_reference_metrics = evaluate_cover(
            gt_cover,
            gt_cover,
            graph.vcount(),
            compute_omega=options["omega"],
            omega_sample_size=options["omega_sample_size"],
            omega_seed=0,
        )
        gt_audit["reference"] = {
            "accuracy": gt_reference_metrics,
            "detector_runtime_seconds": None,
            "detector_peak_rss_bytes": None,
            "runtime_applicable": False,
            "equilibrium_status": "not_a_detector_run",
        }
        gt_path = output_dir / "ground_truth" / f"{dataset_name}-{options['cover']}.json"
        _atomic_json(
            gt_path,
            {
                **gt_audit,
                "protocol": PROTOCOL_NAME,
                "status": "completed",
                "ground_truth_cover_hash": gt_digest,
            },
        )
        fixed_endpoint = gt_audit["policies"]["fixed_labels"]["endpoint"]
        gt_rows.append(
            {
                "dataset": dataset_name,
                "cover": options["cover"],
                "n_vertices": graph.vcount(),
                "n_edges": graph.ecount(),
                "fixed_labels_robust_fraction": fixed_endpoint["robust_fraction_gamma_0_1"],
                "open_labels_robust_fraction": gt_audit["policies"]["open_labels"]["endpoint"][
                    "robust_fraction_gamma_0_1"
                ],
                "accuracy_node_micro_f1": gt_reference_metrics.get("node_micro_f1"),
                "detector_runtime_seconds": None,
                "audit_runtime_seconds": gt_audit.get("audit_runtime_seconds"),
                "ground_truth_cover_hash": gt_digest,
                "graph_identity": prepared.graph_identity,
                "ground_truth_identity": prepared.ground_truth_identity,
                "protocol_identity_sha256": _json_hash(protocol_identity),
                "protocol_lock_sha256": protocol_identity.get(
                    "protocol_lock_sha256"
                ),
                "tracked_files_sha256": protocol_identity.get(
                    "tracked_files_sha256"
                ),
                "effective_grid_sha256": protocol_identity.get(
                    "effective_grid_sha256"
                ),
            }
        )
        if options["audit_only"]:
            status_counts["completed_ground_truth_audit"] = status_counts.get(
                "completed_ground_truth_audit", 0
            ) + 1
            continue
        if options["rescore_only"]:
            rescored_records = _rescore_dataset_records(
                prepared=prepared,
                options=options,
                gt_cover=gt_cover,
                gt_memberships=gt_memberships,
            )
            for record in rescored_records:
                all_rows.append(_flatten_record(record))
                status = str(record.get("status", "failed"))
                status_counts[status] = status_counts.get(status, 0) + 1
                print(
                    f"[{status}] {dataset_name}/{options['cover']} "
                    f"rescore-only condition={record.get('condition_key', '?')}",
                    flush=True,
                )
            continue

        for target_distance in options["perturbation_distances"]:
            requested_swaps = _requested_swaps_for_distance(
                target_distance, gt_memberships
            )
            for perturbation_seed in _perturbation_seeds_for_distance(
                options, target_distance
            ):
                initial_cover, perturbation = perturb_cover_incidence(
                    gt_cover,
                    graph.vcount(),
                    swaps=requested_swaps,
                    seed=int(perturbation_seed),
                )
                perturbation["target_incidence_distance"] = float(
                    target_distance
                )
                if perturbation["successful_swaps"] != requested_swaps:
                    raise RuntimeError(
                        "incidence perturbation did not reach its registered "
                        f"target: requested {requested_swaps}, achieved "
                        f"{perturbation['successful_swaps']}"
                    )
                initial_cover = canonicalize_cover(initial_cover, graph.vcount())
                initial_memberships = cover_to_vertex_memberships(
                    initial_cover, graph.vcount()
                )
                initial_digest = _persist_cover(output_dir, initial_cover)
                # The initialization audit uses the detector's policy and the
                # first operating resolution; all final covers are audited
                # under both policies below.
                cap = max(2, max(len(labels) for labels in gt_memberships))
                for phase in options["phases"]:
                    if phase not in {"local", "multiphase"}:
                        raise ValueError("phase must be local or multiphase")
                    for policy in options["isolation_policies"]:
                        if policy not in {"fixed_labels", "open_labels"}:
                            raise ValueError(
                                "isolation policy must be fixed_labels or open_labels"
                            )
                        for multiplier in options["multipliers"]:
                            operating_gamma = min(
                                graph.density() * float(multiplier), 1.0
                            )
                            initial_audit_key = (
                                prepared.graph_identity,
                                initial_digest,
                                int(cap),
                                bool(policy == "open_labels"),
                                float(operating_gamma),
                                float(options["atol"]),
                                float(options["rtol"]),
                                bool(options["dense"]),
                            )
                            if initial_audit_key not in initial_audit_cache:
                                initial_audit_cache[initial_audit_key] = audit_cover(
                                    graph,
                                    initial_memberships,
                                    max_memberships=cap,
                                    allow_isolation=policy == "open_labels",
                                    gamma=operating_gamma,
                                    interval=(0.0, 1.0),
                                    atol=options["atol"],
                                    rtol=options["rtol"],
                                    dense=options["dense"],
                                )
                            initial_audit = initial_audit_cache[initial_audit_key]
                            for seed in options["detector_seeds"]:
                                record, _final = _run_record(
                                    prepared=prepared,
                                    options=options,
                                    gt_cover=gt_cover,
                                    gt_memberships=gt_memberships,
                                    initial_cover=initial_cover,
                                    initial_memberships=initial_memberships,
                                    perturbation=perturbation,
                                    phase=phase,
                                    policy=policy,
                                    multiplier=float(multiplier),
                                    seed=int(seed),
                                    initial_audit=initial_audit,
                                    audit_cache=final_audit_cache,
                                )
                                all_rows.append(_flatten_record(record))
                                status = str(record.get("status", "failed"))
                                status_counts[status] = status_counts.get(status, 0) + 1
                                print(
                                    f"[{status}] {dataset_name}/{options['cover']} "
                                    f"phase={phase} policy={policy} gamma×={float(multiplier):g} "
                                    f"seed={seed} init={initial_digest[:8]}",
                                    flush=True,
                                )

    observed_axis_keys = [
        str(row.get("condition_axis_key"))
        for row in all_rows
        if isinstance(row.get("condition_axis_key"), str)
    ]
    observed_axis_set = set(observed_axis_keys)
    missing_axis_keys = expected_axis_keys - observed_axis_set
    unexpected_axis_keys = observed_axis_set - expected_axis_keys
    duplicate_condition_count = len(observed_axis_keys) - len(observed_axis_set)
    identity_rows_match = all(
        row.get("protocol_lock_sha256")
        == protocol_identity.get("protocol_lock_sha256")
        and row.get("tracked_files_sha256")
        == protocol_identity.get("tracked_files_sha256")
        and row.get("effective_grid_sha256")
        == protocol_identity.get("effective_grid_sha256")
        and row.get("canonical_grid")
        == protocol_identity.get("effective_grid_matches_lock")
        for row in all_rows
    )
    condition_coverage_complete = bool(expected_axis_keys) and (
        not missing_axis_keys
        and not unexpected_axis_keys
        and duplicate_condition_count == 0
        and len(all_rows) == expected
        and identity_rows_match
    )
    terminal_statuses = set(
        protocol_identity.get("effective_grid", {})
        .get("terminal_outcome_policy", {})
        .get("allowed_statuses", ())
    )
    result_statuses = {"completed", "completed_non_equilibrium"} | terminal_statuses
    all_results_available = condition_coverage_complete and all(
        row.get("status") in result_statuses for row in all_rows
    )
    expected_ground_truth_paths = {
        f"ground_truth/{dataset}-{options['cover']}.json"
        for dataset in options["datasets"]
    }
    actual_ground_truth_paths = {
        path.relative_to(output_dir).as_posix()
        for path in (output_dir / "ground_truth").glob("*.json")
    }
    ground_truth_artifact_set_complete = (
        actual_ground_truth_paths == expected_ground_truth_paths
    )
    canonical_protocol_complete = bool(
        protocol_identity.get("effective_grid_matches_lock")
        and len(gt_rows) == len(options["datasets"])
        and ground_truth_artifact_set_complete
        and all_results_available
    )
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL_NAME,
        "expected_detector_conditions": expected,
        "observed_detector_records": len(all_rows),
        "unique_observed_conditions": len(observed_axis_set),
        "duplicate_condition_count": duplicate_condition_count,
        "missing_condition_count": len(missing_axis_keys),
        "unexpected_condition_count": len(unexpected_axis_keys),
        "expected_condition_axis_keys_sha256": _json_hash(
            sorted(expected_axis_keys)
        ),
        "observed_condition_axis_keys_sha256": _json_hash(
            sorted(observed_axis_set)
        ),
        "missing_condition_examples": sorted(missing_axis_keys)[:20],
        "unexpected_condition_examples": sorted(unexpected_axis_keys)[:20],
        "protocol_identity": protocol_identity,
        "status_counts": status_counts,
        "terminal_statuses": sorted(
            status for status in status_counts if status not in {
                "completed", "completed_non_equilibrium"
            }
        ),
        "ground_truth_instances": len(gt_rows),
        "ground_truth_artifact_set_complete": ground_truth_artifact_set_complete,
        "condition_coverage_complete": condition_coverage_complete,
        "all_results_available": all_results_available,
        "canonical_protocol_complete": canonical_protocol_complete,
        "complete": canonical_protocol_complete,
    }
    results_jsonl_path = output_dir / "results.jsonl"
    results_csv_path = output_dir / "results.csv"
    ground_truth_summary_path = output_dir / "ground_truth_summary.csv"
    _append_jsonl(results_jsonl_path, all_rows)
    _write_csv(results_csv_path, all_rows)
    _write_csv(ground_truth_summary_path, gt_rows)
    evidence_index = _publication_evidence_index(
        output_dir, all_rows, protocol_identity
    )
    if not evidence_index["valid"]:
        canonical_protocol_complete = False
        coverage["canonical_protocol_complete"] = False
        coverage["complete"] = False
    ground_truth_record_hashes = {
        path.relative_to(output_dir).as_posix(): _file_hash(path)
        for path in sorted((output_dir / "ground_truth").glob("*.json"))
    }
    artifact_sha256 = {
        "results.jsonl": _file_hash(results_jsonl_path),
        "results.csv": _file_hash(results_csv_path),
        "ground_truth_summary.csv": _file_hash(ground_truth_summary_path),
    }
    coverage["artifact_sha256"] = artifact_sha256
    coverage["ground_truth_records_sha256"] = ground_truth_record_hashes
    coverage["detailed_evidence_valid"] = evidence_index["valid"]
    coverage["detailed_evidence_reasons"] = evidence_index["reasons"]
    coverage["run_records_sha256"] = evidence_index["run_records_sha256"]
    coverage["referenced_artifacts_sha256"] = evidence_index[
        "referenced_artifacts_sha256"
    ]
    plots = (
        _write_plots(
            output_dir,
            gt_rows,
            all_rows,
            protocol_identity=protocol_identity,
            expected_condition_axis_keys=expected_axis_keys,
        )
        if canonical_protocol_complete
        else []
    )
    plot_hashes: dict[str, str | None] = {}
    for value in plots:
        plot_path = Path(value)
        if not plot_path.is_absolute():
            plot_path = output_dir / plot_path
        try:
            relative = plot_path.resolve().relative_to(output_dir.resolve()).as_posix()
        except ValueError:
            canonical_protocol_complete = False
            coverage["canonical_protocol_complete"] = False
            coverage["complete"] = False
            continue
        plot_hashes[relative] = _file_hash(plot_path)
    expected_plot_paths = {
        "plots/gt_robustness_overview.png",
        "plots/gt_robustness_overview.pdf",
        "plots/gt_equilibrium_tradeoff.png",
        "plots/gt_equilibrium_tradeoff.pdf",
    }
    if canonical_protocol_complete and (
        set(plot_hashes) != expected_plot_paths
        or any(not isinstance(value, str) for value in plot_hashes.values())
    ):
        canonical_protocol_complete = False
        coverage["canonical_protocol_complete"] = False
        coverage["complete"] = False
    coverage["plot_sha256"] = plot_hashes
    _atomic_json(output_dir / "coverage_report.json", coverage)
    manifest.update(
        {
            "finished_at": time.time(),
            "expected_detector_conditions": expected,
            "status_counts": status_counts,
            "ground_truth_instances": len(gt_rows),
            "ground_truth_artifact_set_complete": ground_truth_artifact_set_complete,
            "detailed_evidence_valid": evidence_index["valid"],
            "plot_paths": plots,
            "condition_coverage_complete": condition_coverage_complete,
            "all_results_available": all_results_available,
            "canonical_protocol_complete": canonical_protocol_complete,
            "artifact_sha256": artifact_sha256,
            "ground_truth_records_sha256": ground_truth_record_hashes,
            "run_records_sha256": evidence_index["run_records_sha256"],
            "referenced_artifacts_sha256": evidence_index[
                "referenced_artifacts_sha256"
            ],
            "plot_sha256": plot_hashes,
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    _output_lock.close()
    if (
        not options["smoke"]
        and not options["audit_only"]
        and protocol_identity.get("effective_grid_matches_lock")
        and not canonical_protocol_complete
    ):
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit supplied overlapping ground truth and find community_hedonic "
            "equilibria seeded from the full GT cover."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default=None)
    parser.add_argument("--smoke", action="store_true", help="Use the built-in six-vertex cover")
    parser.add_argument("--datasets", "--dataset", dest="datasets", default=None)
    parser.add_argument("--cover", default=None, choices=("top5000", "all"))
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=None)
    parser.add_argument("--max-nodes", "--max_nodes", dest="max_nodes", type=int, default=None)
    parser.add_argument(
        "--uncovered-policy",
        choices=("covered-induced", "singleton"),
        default=None,
        help="How partial metadata covers become valid GT starts (default: covered-induced)",
    )
    parser.add_argument("--phases", default=None, help="Comma-separated local,multiphase")
    parser.add_argument(
        "--isolation-policies",
        default=None,
        help="Comma-separated fixed_labels,open_labels",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Detector seeds: comma-separated integers or ranges such as 0-4",
    )
    parser.add_argument(
        "--perturbation-seeds",
        "--perturbation_seeds",
        dest="perturbation_seeds",
        default=None,
        help="Independent incidence-perturbation seeds (defaults to detector seeds)",
    )
    parser.add_argument("--resolution-multipliers", default=None)
    parser.add_argument(
        "--perturbation-distances",
        dest="perturbation_distances",
        default=None,
        help=(
            "Comma-separated target labeled-incidence distances in [0,1]; "
            "each is converted to a dataset-scaled degree-preserving swap count"
        ),
    )
    parser.add_argument("--robustness-profile", default=None, help="Grid such as 0:1:11")
    parser.add_argument("--robustness-atol", type=float, default=1e-10)
    parser.add_argument("--robustness-rtol", type=float, default=1e-9)
    parser.add_argument("--timeout-per-run", "--timeout_per_run", dest="timeout_per_run", type=float, default=None)
    parser.add_argument("--omega", action="store_true")
    parser.add_argument("--omega-sample-size", type=int, default=100_000)
    parser.add_argument("--dense-oracle", action="store_true", help="Use every active label in the audit oracle")
    parser.add_argument("--audit-only", action="store_true", help="Only compute GT robustness; do not run a detector")
    parser.add_argument("--rescore-only", action="store_true", help="Re-score persisted final covers without detection")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Ignore compatible run caches")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_only and args.rescore_only:
        raise SystemExit("--audit-only and --rescore-only are mutually exclusive")
    return run_experiment(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
