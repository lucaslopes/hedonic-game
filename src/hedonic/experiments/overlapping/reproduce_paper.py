"""TOML-driven, tmux-orchestrated reproduction of the overlapping SNAP paper.

The lower-level ``overlapping-benchmark`` command owns one output directory and
therefore deliberately runs its conditions serially.  This module parallelizes
only independent dataset/cover *shards*.  Every shard has a separate output
tree, which keeps the benchmark's manifests, logs, summaries, and atomic run
records free of concurrent-writer races.  A coordinator then merges the run
records, regenerates paper plots and tables, and (when the full protocol has
no retryable records) compiles the manuscript.

All normal experiment parameters live in ``[overlapping_paper]`` in an
externally supplied TOML file.  The optional one-command entry point is::

    hedonic-exp reproduce-overlapping-paper

The public checkout does not ship a manuscript configuration.  A private
research checkout can supply the full protocol and manuscript directory when
that workflow is required.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments import config as experiment_config
from hedonic.experiments.overlapping import benchmark
from hedonic.experiments.overlapping import protocol
from hedonic.experiments.overlapping.methods import METHODS
from hedonic.experiments.overlapping.snap import (
    bounded_induced_dataset,
    common_undirected_analysis_dataset,
    load_snap_dataset,
    network_names,
    smoke_dataset,
)


PAPER_SCHEMA_VERSION = 4
MIN_MAC_RESERVE_GB = 20.0
DEFAULT_MAC_RESERVE_GB = 24.0
DEFAULT_SAFETY_FACTOR = 1.5
DEFAULT_METHODS = (
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "cpm",
    "demon",
)
DEFAULT_JOBS = (
    ("amazon", "top5000"),
    ("dblp", "top5000"),
    ("livejournal", "top5000"),
    ("youtube", "top5000"),
    ("wikipedia", "all"),
)
# Resource exhaustion is deterministic for a fixed condition and is never
# retried automatically. A detector exception is the only potentially
# transient outcome, and even that defaults to zero retries in TOML.
RETRYABLE_STATUSES = {"failed"}
NON_RETRYABLE_FAILURES = {
    "timeout",
    "memory_limit",
    "oom",
    "skipped_not_scalable",
    "skipped_external_unchanged",
}
_DATASET_WEIGHT = {
    "dblp": 1,
    "amazon": 2,
    "youtube": 3,
    "livejournal": 5,
    "wikipedia": 5,
}
_PAPER_METRICS = (
    "symmetric_best_match_f1",
    "matching_f1",
    "node_membership_micro_f1",
    "size_weighted_f1",
    "omega",
    "runtime_seconds",
    "predicted_overlapping_node_fraction",
    "coverage_rate",
    "inclusion_rate",
    "cpm_overlapping_quality",
)
_PAPER_METRIC_ALIASES = {
    # Current protocol records bind the nested ``metrics`` mapping with a
    # content digest.  These are semantic aliases within that mapping only;
    # unbound top-level fields must never shadow audited values.
    "node_membership_micro_f1": ("node_membership_micro_f1", "node_micro_f1"),
    "size_weighted_f1": ("size_weighted_f1", "size_weighted_community_f1"),
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _as_list(value: Any, field: str) -> list[str]:
    if isinstance(value, str):
        result = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        result = [str(part).strip() for part in value if str(part).strip()]
    else:
        raise ValueError(f"overlapping_paper.{field} must be a string or TOML array")
    if not result:
        raise ValueError(f"overlapping_paper.{field} must not be empty")
    return result


def _path(value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"overlapping_paper.{field} must be a non-empty path")
    path = Path(os.path.expanduser(str(value))).expanduser()
    if not path.is_absolute():
        path = experiment_config.PROJECT_ROOT / path
    return path.resolve()


def _physical_memory_bytes() -> int | None:
    """Return installed RAM without introducing an optional dependency."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and pages > 0:
            return page_size * pages
    except (AttributeError, OSError, ValueError):
        pass
    # macOS exposes this even when POSIX sysconf does not.
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        value = int(output)
        return value if value > 0 else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _resolve_worker_limit(value: Any, *, jobs: int, cap: int) -> int:
    if jobs < 1:
        raise ValueError("At least one paper job is required")
    cpu_count = os.cpu_count() or 1
    if cap < 1:
        raise ValueError("overlapping_paper.max_parallel_cap must be >= 1")
    if isinstance(value, str) and value.strip().lower() == "auto":
        return max(1, min(jobs, cpu_count, cap))
    try:
        workers = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "overlapping_paper.max_parallel_workers must be a positive integer or 'auto'"
        ) from exc
    if workers < 1:
        raise ValueError("overlapping_paper.max_parallel_workers must be >= 1")
    if workers > cpu_count:
        raise ValueError(
            f"Requested {workers} workers but only {cpu_count} logical CPUs are available"
        )
    if workers > cap:
        raise ValueError(
            f"Requested {workers} workers but max_parallel_cap is {cap}; raise the cap explicitly"
        )
    return min(workers, jobs)


def _parse_jobs(raw: Any) -> list[dict[str, Any]]:
    source = raw if raw is not None else [
        {"dataset": dataset, "cover": cover} for dataset, cover in DEFAULT_JOBS
    ]
    if not isinstance(source, list) or not source:
        raise ValueError("overlapping_paper.jobs must be a non-empty TOML array of tables")
    valid_datasets = set(network_names())
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            raise ValueError("Each overlapping_paper.jobs item must be a TOML table")
        dataset = str(item.get("dataset", "")).strip().lower()
        cover = str(item.get("cover", "")).strip().lower()
        if dataset not in valid_datasets:
            raise ValueError(f"Unknown paper job dataset: {dataset!r}")
        if cover not in {"all", "top5000"}:
            raise ValueError(f"Paper job {dataset!r} has invalid cover {cover!r}")
        if dataset == "wikipedia" and cover != "all":
            raise ValueError("Wikipedia only has the supplied 'all' category cover")
        name = str(item.get("name") or f"{dataset}-{cover}").strip()
        if not name or name in seen:
            raise ValueError("Paper job names must be unique and non-empty")
        seen.add(name)
        weight = item.get("weight", _DATASET_WEIGHT.get(dataset, 1))
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Paper job {name!r} has invalid weight") from exc
        if weight <= 0:
            raise ValueError(f"Paper job {name!r} weight must be positive")
        jobs.append({"name": name, "dataset": dataset, "cover": cover, "weight": weight, "index": index})
    return jobs


def _memory_budget_bytes(value: Any, reserve_gb: float) -> int:
    """Resolve a usable experiment-memory budget, leaving OS headroom."""
    if reserve_gb < 0:
        raise ValueError("overlapping_paper.memory_reserve_gb must be >= 0")
    if isinstance(value, str) and value.strip().lower() == "auto":
        physical = _physical_memory_bytes()
        if physical is None:
            raise ValueError(
                "Could not determine physical memory; set overlapping_paper.memory_budget_gb explicitly"
            )
        budget = physical - int(reserve_gb * (1024**3))
    else:
        try:
            budget = int(float(value) * (1024**3))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "overlapping_paper.memory_budget_gb must be a positive GiB value or 'auto'"
            ) from exc
    if budget <= 0:
        raise ValueError("The memory budget must be positive after the OS reserve")
    return budget


def _historical_peaks(output_dir: Path | None, job: dict[str, Any], methods: list[str]) -> tuple[dict[str, int], bool]:
    """Read observed process-tree peaks without treating old limits as peaks."""
    if output_dir is None:
        return {}, True
    shard = output_dir / "shards" / str(job["name"])
    peaks: dict[str, int] = {}
    uncertain = False
    for record in benchmark._completed_records(shard):
        method = str(record.get("method", ""))
        if method not in methods:
            continue
        memory = record.get("memory") or record.get("detector_memory")
        observed = memory.get("observed_peak_rss_bytes") if isinstance(memory, dict) else None
        if isinstance(observed, (int, float)) and observed > 0:
            peaks[method] = max(peaks.get(method, 0), int(observed))
        elif record.get("status") in {"memory_limit", "oom", "timeout", "failed"}:
            uncertain = True
    if not peaks:
        uncertain = True
    return peaks, uncertain


def _job_memory_profile(
    job: dict[str, Any],
    *,
    data_root: Path,
    profile: str,
    max_nodes: int,
    methods: list[str],
    memory_budget_bytes: int,
    safety_factor: float,
    configured_membership_cap: int | None,
    output_dir: Path | None = None,
    locked_dataset_identities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate one job's peak and cap the hedonic workspace before launch.

    The lucas-igraph overlap workspace scales as ``n * max_memberships``.
    Treating the number of ground-truth communities as that capacity made the
    old full LiveJournal and Wikipedia jobs require roughly 150 and 232 GiB.
    The correct automatic capacity is the maximum *per-node* GT membership.
    """
    dataset = (
        smoke_dataset(str(job["dataset"]), cover_variant=str(job["cover"]))
        if profile == "smoke"
        else load_snap_dataset(
            str(job["dataset"]), cover_variant=str(job["cover"]), data_root=data_root
        )
    )
    dataset = common_undirected_analysis_dataset(
        bounded_induced_dataset(dataset, max_nodes)
    )
    report = dataset.report
    if profile != "smoke":
        dataset_key = f"{job['dataset']}/{job['cover']}"
        expected_content = (locked_dataset_identities or {}).get(dataset_key)
        if not isinstance(expected_content, dict):
            raise ValueError(
                f"Protocol lock has no dataset identity for {dataset_key}"
            )
        if report.get("content_identity") != expected_content:
            raise ValueError(
                f"{dataset_key} bounded graph/ground-truth content does not "
                "match the reviewed protocol lock"
            )
    n, m = int(report["n"]), int(report["m"])
    overlap = report.get("overlap_statistics", {})
    content = report.get("content_identity") or {}
    memberships = int(overlap.get("memberships", 0))
    community_count = int(content.get("ground_truth_community_count", len(dataset.cover)))
    gt_membership_cap = max(
        1, int(content.get("ground_truth_max_memberships_per_node", 1))
    )
    requested_cap = min(
        gt_membership_cap,
        configured_membership_cap if configured_membership_cap is not None else gt_membership_cap,
    )
    requested_cap = max(2, requested_cap)

    gib = 1024**3
    historical_peaks, history_uncertain = _historical_peaks(output_dir, job, methods)
    # Parent graph + forked child graph bookkeeping, cover lists, and native
    # igraph allocator headroom. NetworkX baselines are estimated separately
    # with a deliberately pessimistic 512 B/edge because dict-backed adjacency
    # dominates their full-graph peak.
    igraph_bytes = 2 * (n * 64 + m * 32) + memberships * 24 + 2 * gib
    networkx_bytes = n * 192 + m * 512 + 2 * gib
    baseline_bytes = igraph_bytes + networkx_bytes if {"cpm", "demon"} & set(methods) else 0
    hedonic_base_bytes = igraph_bytes
    workspace_per_membership = n * 8
    raw_room = int(memory_budget_bytes / safety_factor) - hedonic_base_bytes
    fit_cap = raw_room // workspace_per_membership if workspace_per_membership else requested_cap
    if {name for name in methods if name.startswith("hedonic_")}:
        if fit_cap < 2:
            raise ValueError(
                f"{job['name']} cannot fit even two hedonic memberships within the "
                f"{memory_budget_bytes / gib:.1f} GiB budget; increase the budget or use --max_nodes"
            )
        effective_cap = min(requested_cap, int(fit_cap))
        hedonic_bytes = hedonic_base_bytes + workspace_per_membership * effective_cap
    else:
        effective_cap = requested_cap
        hedonic_bytes = hedonic_base_bytes
    method_estimates: dict[str, int] = {}
    for method in methods:
        if method.startswith("hedonic_"):
            method_estimates[method] = hedonic_bytes
        elif method == "demon":
            method_estimates[method] = int(networkx_bytes * 1.25)
        else:
            method_estimates[method] = baseline_bytes or igraph_bytes
    raw_peak = max(method_estimates.values(), default=igraph_bytes)
    unbounded_estimated_peak = max(
        math.ceil(raw_peak * safety_factor),
        max((int(value * 1.25) for value in historical_peaks.values()), default=0),
    )
    # The hard detector cap is the safe budget even when the conservative
    # unbounded estimate is larger. Such a job is still planned as a solo wave
    # and any hit is reported as memory_limit/non-scalable; it is never allowed
    # to borrow the OS reserve or another worker's budget.
    estimated_peak = min(unbounded_estimated_peak, memory_budget_bytes)
    parent_allowance = n * 64 + m * 32 + gib
    detector_limit = max(gib // 2, estimated_peak - parent_allowance)
    detector_limit = min(detector_limit, estimated_peak)
    # Do not retain any full graph while profiling the next job.
    del dataset
    gc.collect()
    return {
        **job,
        "max_memberships": effective_cap,
        "dataset_content_identity": report.get("content_identity"),
        "dataset_metadata_identity": protocol.dataset_metadata_identity(report),
        "memory": {
            "n": n,
            "m": m,
            "ground_truth_community_count": community_count,
            "ground_truth_max_memberships_per_node": gt_membership_cap,
            "configured_membership_cap": configured_membership_cap,
            "effective_max_memberships": effective_cap,
            "bytes_per_vertex_membership": 8,
            "naive_community_count_workspace_bytes": n * community_count * 8,
            "effective_workspace_bytes": workspace_per_membership * effective_cap,
            "igraph_parent_child_bytes": igraph_bytes,
            "networkx_baseline_bytes": networkx_bytes if baseline_bytes else 0,
            "safety_factor": safety_factor,
            "estimated_peak_bytes": estimated_peak,
            "unbounded_estimated_peak_bytes": unbounded_estimated_peak,
            "parent_memory_allowance_bytes": parent_allowance,
            "detector_memory_limit_bytes": detector_limit,
            "memory_estimate_clamped_to_hard_budget": unbounded_estimated_peak > memory_budget_bytes,
            "method_estimates_bytes": method_estimates,
            "historical_observed_peaks_bytes": historical_peaks,
            "history_uncertain": history_uncertain,
            "schedule_reason": (
                "livejournal_must_be_serialized"
                if job["dataset"] == "livejournal"
                else "unbounded_estimate_exceeds_budget_hard_cap_applied"
                if unbounded_estimated_peak > memory_budget_bytes
                else "historical_peak_missing_or_incomplete"
                if history_uncertain
                else "graph_and_method_estimate"
            ),
        },
    }


def _schedule_memory_waves(
    jobs: list[dict[str, Any]], *, memory_budget_bytes: int, worker_limit: int
) -> list[list[dict[str, Any]]]:
    """Pack independent jobs into waves whose estimated peaks fit the budget."""
    # Start with the least demanding dataset so the protocol validates its
    # full record/merge path before reaching the largest memory-risk jobs.
    # LiveJournal remains a dedicated wave regardless of this ordering.
    remaining = sorted(
        jobs,
        key=lambda job: (int(job["memory"]["estimated_peak_bytes"]), int(job["index"])),
    )
    waves: list[list[dict[str, Any]]] = []
    while remaining:
        wave: list[dict[str, Any]] = []
        used = 0
        for job in list(remaining):
            estimate = int(job["memory"]["estimated_peak_bytes"])
            if job.get("dataset") == "livejournal" and wave:
                continue
            if any(item.get("dataset") == "livejournal" for item in wave):
                continue
            if len(wave) < worker_limit and used + estimate <= memory_budget_bytes:
                wave.append(job)
                remaining.remove(job)
                used += estimate
        if not wave:  # guarded above, retained as an invariant check
            job = remaining[0]
            raise ValueError(f"No memory-safe schedule can place {job['name']}")
        waves.append(wave)
    return waves


def _load_options(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config_path = experiment_config.find_config_file(
        args.config, search_cwd=True
    )
    if config_path is None:
        raise ValueError(
            "No TOML configuration found. Run from the repository root or pass --config."
        )
    encoded_config = config_path.read_bytes()
    toml = tomllib.loads(encoded_config.decode("utf-8"))
    if not isinstance(toml, dict):
        raise ValueError(f"TOML root must be a table: {config_path}")
    # Apply path settings from the exact byte snapshot whose digest is carried
    # into the plan.  Reopening through load_config_file would permit a swap
    # between parsing and hashing.
    experiment_config._LOADED_TOML = toml
    experiment_config._LOADED_TOML_PATH = config_path
    extracted_paths = experiment_config._paths_from_toml(toml)
    experiment_config.apply_path_overrides(
        **{
            key: value
            for key, value in extracted_paths.items()
            if value is not None
        }
    )
    experiment_config._apply_env_overrides()
    config_sha256 = hashlib.sha256(encoded_config).hexdigest()
    raw = toml.get("overlapping_paper")
    if not isinstance(raw, dict):
        raise ValueError(
            f"{config_path} needs an [overlapping_paper] table for this command"
        )
    encoded_protocol_lock = protocol.LOCK_PATH.read_bytes()
    locked_protocol = protocol._parse_protocol_lock(encoded_protocol_lock)
    experiment_identity = protocol.current_experiment_identity(
        protocol.LOCK_PATH, lock_bytes=encoded_protocol_lock
    )
    locked_dataset_identities = locked_protocol.get(
        "dataset_content_identities"
    ) or {}
    methods = _as_list(raw.get("methods", list(DEFAULT_METHODS)), "methods")
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown configured methods: {', '.join(unknown)}")
    if "hedonic_local" in methods:
        raise ValueError(
            "The paper protocol excludes hedonic_local; use the three hedonic_multiphase "
            "resolution variants with cpm,demon"
        )
    if "hedonic_multiphase" not in methods:
        raise ValueError("The paper protocol must include hedonic_multiphase")
    raw_not_rerun_external_methods = raw.get("not_rerun_external_methods", [])
    not_rerun_external_methods = (
        []
        if raw_not_rerun_external_methods in (None, "", [])
        else _as_list(raw_not_rerun_external_methods, "not_rerun_external_methods")
    )
    unknown_not_rerun = sorted(set(not_rerun_external_methods) - set(methods))
    if unknown_not_rerun:
        raise ValueError(
            "overlapping_paper.not_rerun_external_methods must be a subset of methods: "
            + ", ".join(unknown_not_rerun)
        )
    unsupported_not_rerun = sorted(
        set(not_rerun_external_methods) - benchmark.NON_SCALABLE_BASELINES
    )
    if unsupported_not_rerun:
        raise ValueError(
            "overlapping_paper.not_rerun_external_methods is limited to external "
            "baselines cpm,demon: "
            + ", ".join(unsupported_not_rerun)
        )

    profile = str(raw.get("profile", "full")).strip().lower()
    if profile not in benchmark.PROFILE_DEFAULTS:
        raise ValueError("overlapping_paper.profile must be smoke, standard, or full")
    jobs = _parse_jobs(raw.get("jobs"))
    cap = int(raw.get("max_parallel_cap", 2))
    worker_setting = args.workers if args.workers is not None else raw.get("max_parallel_workers", "auto")
    worker_limit = _resolve_worker_limit(worker_setting, jobs=len(jobs), cap=cap)
    if args.session is not None:
        session = args.session
    else:
        session = str(raw.get("tmux_session", "hedonic-overlapping-paper")).strip()
    if not session:
        raise ValueError("overlapping_paper.tmux_session must be non-empty")
    if any(character.isspace() for character in session) or ":" in session:
        raise ValueError("tmux session names must not contain whitespace or ':'")

    data_root = _path(raw.get("data_root", experiment_config.NETWORKS_DIR), "data_root")
    output_dir = _path(raw.get("output_dir"), "output_dir")
    paper_dir = _path(raw.get("paper_dir"), "paper_dir")
    if not (paper_dir / "main.tex").is_file():
        raise ValueError(f"overlapping_paper.paper_dir has no main.tex: {paper_dir}")
    expected_artifact_dir = (experiment_config.PAPER_ARTIFACTS_DIR / "full").resolve()
    legacy_artifact_dir = (paper_dir / "artifacts" / "full").resolve()
    artifact_root = experiment_config.PAPER_ARTIFACTS_DIR.resolve()
    output_is_registered_artifact = output_dir == expected_artifact_dir or artifact_root in output_dir.parents
    if not output_is_registered_artifact and output_dir != legacy_artifact_dir:
        raise ValueError(
            "overlapping_paper.output_dir must be "
            "under artifacts/papers/overlapping_communities; the legacy "
            "paper_dir/artifacts/full path is accepted only for isolated "
            "backward-compatible runs"
        )
    if output_dir == legacy_artifact_dir and output_dir != expected_artifact_dir:
        print(
            "[paper] warning: using legacy paper_dir/artifacts/full output; "
            "new runs should use artifacts/papers/overlapping_communities/full",
            file=sys.stderr,
        )
    benchmark._safe_output_dir(output_dir, data_root)

    max_nodes = int(raw.get("max_nodes", 0))
    timeout = float(raw.get("timeout_per_run", 3600))
    if timeout <= 0:
        raise ValueError("overlapping_paper.timeout_per_run must be positive")

    reserve_gb = float(raw.get("memory_reserve_gb", DEFAULT_MAC_RESERVE_GB))
    if profile == "full" and reserve_gb < MIN_MAC_RESERVE_GB:
        raise ValueError(
            f"Full paper protocol must reserve at least {MIN_MAC_RESERVE_GB:.0f} GiB for macOS; "
            "use 20--24 GiB or more"
        )
    memory_budget_bytes = _memory_budget_bytes(raw.get("memory_budget_gb", "auto"), reserve_gb)
    safety_factor = float(raw.get("memory_safety_factor", DEFAULT_SAFETY_FACTOR))
    if safety_factor < 1:
        raise ValueError("overlapping_paper.memory_safety_factor must be >= 1")
    membership_cap_raw = raw.get("max_memberships")
    if membership_cap_raw is None:
        membership_cap = None
    else:
        membership_cap = int(membership_cap_raw)
        if membership_cap < 1:
            raise ValueError("overlapping_paper.max_memberships must be >= 1")
    profiled_jobs = [
        _job_memory_profile(
            job,
            data_root=data_root,
            profile=profile,
            max_nodes=max_nodes,
            methods=methods,
            memory_budget_bytes=memory_budget_bytes,
            safety_factor=safety_factor,
            configured_membership_cap=membership_cap,
            output_dir=output_dir,
            locked_dataset_identities=locked_dataset_identities,
        )
        for job in jobs
    ]
    if isinstance(worker_setting, str) and worker_setting.strip().lower() == "auto" and any(
        bool(job["memory"].get("history_uncertain")) for job in profiled_jobs
    ):
        worker_limit = 1
    waves = _schedule_memory_waves(
        profiled_jobs, memory_budget_bytes=memory_budget_bytes, worker_limit=worker_limit
    )
    workers = max(len(wave) for wave in waves)

    retries = int(raw.get("retry_attempts", 0))
    if retries < 0:
        raise ValueError("overlapping_paper.retry_attempts must be >= 0")
    if retries > 1:
        raise ValueError("overlapping_paper.retry_attempts must be 0 or 1")
    poll_seconds = float(raw.get("poll_seconds", 10))
    if poll_seconds <= 0:
        raise ValueError("overlapping_paper.poll_seconds must be positive")
    if not experiment_identity["tracked_files_match_lock"]:
        raise ValueError("Protocol-locked code/config hashes do not match; refresh and review the lock")
    if not experiment_identity["lucas_igraph"]["package_identity_matches_lock"]:
        raise ValueError("installed lucas-igraph package does not match the protocol-locked release")
    if not experiment_identity["scientific_dependencies_match_lock"]:
        raise ValueError(
            "installed numerical packages do not match the protocol-locked implementations"
        )
    selected_external_mismatches = [
        method
        for method in methods
        if method in benchmark.NON_SCALABLE_BASELINES
        and (
            not isinstance(
                experiment_identity["external_dependencies"].get(method), dict
            )
            or (
                experiment_identity["external_dependencies"][method].get(
                    "actual_version"
                )
                is not None
                and not all(
                    experiment_identity["external_dependencies"][method].get(field)
                    is True
                    for field in (
                        "version_matches_lock",
                        "package_lock_matches_lock",
                        "distribution_tree_matches_lock",
                        "implementation_matches_lock",
                    )
                )
            )
        )
    ]
    if selected_external_mismatches:
        raise ValueError(
            "installed external baseline packages do not match the protocol-locked "
            "implementations: " + ", ".join(selected_external_mismatches)
        )
    options = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "experiment_identity": experiment_identity,
        "profile": profile,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "paper_dir": str(paper_dir),
        "tmux_session": session,
        "methods": methods,
        "not_rerun_external_methods": not_rerun_external_methods,
        "seeds": str(raw.get("seeds", "0-4")),
        "resolutions": str(raw.get("resolutions", "auto")),
        "timeout_per_run": timeout,
        "timeout_by_method": benchmark._parse_timeout_map(
            raw.get("timeout_by_method", raw.get("timeout_per_method")),
            "overlapping_paper.timeout_by_method",
        ),
        "timeout_by_dataset": benchmark._parse_timeout_map(
            raw.get("timeout_by_dataset"), "overlapping_paper.timeout_by_dataset"
        ),
        "max_nodes": max_nodes,
        "omega": bool(raw.get("omega", True)),
        "omega_sample_size": int(raw.get("omega_sample_size", 100_000)),
        "resume": bool(raw.get("resume", True)),
        "retry_attempts": retries,
        "retry_policy": {
            "default_attempts": retries,
            "timeout_memory_limit_oom_max_attempts": 0,
            "failed_max_attempts": retries,
            "baseline_resource_failures": "skipped_not_scalable_without_repeat",
            "external_baseline_policy": (
                "skipped_external_unchanged_without_detector_invocation"
            ),
        },
        "compile_paper": bool(raw.get("compile_paper", True)),
        "poll_seconds": poll_seconds,
        "memory_budget_bytes": memory_budget_bytes,
        "memory_reserve_gb": reserve_gb,
        "memory_safety_factor": safety_factor,
        "configured_max_memberships": membership_cap,
        "max_parallel_cap": cap,
        "worker_limit": worker_limit,
        "workers": workers,
        "jobs": profiled_jobs,
        "waves": waves,
    }
    return options, config_path


def _plan_path(output_dir: Path) -> Path:
    return output_dir / "orchestration" / "plan.json"


def _state_path(output_dir: Path, worker_index: int) -> Path:
    return output_dir / "orchestration" / "workers" / f"worker_{worker_index}.json"


def _make_plan(options: dict[str, Any]) -> dict[str, Any]:
    workers = int(options["workers"])
    assignments = [
        wave + [None] * (workers - len(wave)) for wave in options["waves"]
    ]
    gib = 1024**3
    wave_reports = []
    for index, wave in enumerate(options["waves"]):
        used = sum(int(job["memory"]["estimated_peak_bytes"]) for job in wave)
        wave_reports.append(
            {
                "index": index,
                "jobs": [job["name"] for job in wave],
                "estimated_peak_bytes": used,
                "estimated_peak_gib": round(used / gib, 3),
                "unbounded_estimated_peak_bytes": sum(
                    int(job["memory"].get("unbounded_estimated_peak_bytes", job["memory"]["estimated_peak_bytes"]))
                    for job in wave
                ),
                "budget_bytes": int(options["memory_budget_bytes"]),
                "fits_budget": used <= int(options["memory_budget_bytes"]),
                "unbounded_estimate_fits_budget": all(
                    int(job["memory"].get("unbounded_estimated_peak_bytes", job["memory"]["estimated_peak_bytes"]))
                    <= int(options["memory_budget_bytes"])
                    for job in wave
                ),
                "decision": "livejournal_serialized"
                if any(job.get("dataset") == "livejournal" for job in wave)
                else "hard_cap_applied_to_unbounded_estimate"
                if any(job["memory"].get("memory_estimate_clamped_to_hard_budget") for job in wave)
                else "conservative_memory_fit",
            }
        )
    return {
        **options,
        "plan_id": uuid.uuid4().hex,
        "started_at": _timestamp(),
        "assignments": assignments,
        "wave_reports": wave_reports,
        "scheduler_policy": {
            "memory_budget_is_hard_limit": True,
            "livejournal_never_coScheduled": True,
            "uncertain_estimates_force_serial_workers": True,
            "reserve_for_macOS_gib": max(float(options["memory_reserve_gb"]), MIN_MAC_RESERVE_GB),
        },
        "worker_startup_timeout_seconds": max(
            120.0, 4.0 * float(options["poll_seconds"])
        ),
    }


def _write_new_plan(options: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(options["output_dir"])
    plan = _make_plan(options)
    old_plan = _plan_path(output_dir)
    if old_plan.is_file():
        backup = old_plan.with_name(f"plan.previous.{int(time.time())}.json")
        shutil.copy2(old_plan, backup)
    _write_json(_plan_path(output_dir), plan)
    workers = int(plan["workers"])
    for index in range(workers):
        jobs = [
            job["name"]
            for wave in plan["assignments"]
            if index < len(wave) and isinstance((job := wave[index]), dict)
        ]
        _write_json(
            _state_path(output_dir, index),
            {
                "plan_id": plan["plan_id"],
                "worker_index": index,
                "worker_pid": None,
                "status": "pending",
                "jobs": jobs,
                "waves": [
                    {"index": wave_index, "status": "pending"}
                    for wave_index in range(len(plan["assignments"]))
                ],
                "updated_at": _timestamp(),
            },
        )
    return _load_plan(_plan_path(output_dir))


def _load_plan(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        encoded = path.read_bytes()
        plan = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        plan = None
    if not isinstance(plan, dict):
        raise ValueError(f"Paper orchestration plan is missing or invalid: {path}")
    assignments = plan.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(f"Paper orchestration plan has no assignments: {path}")
    plan["_plan_source_path"] = str(path)
    plan["_plan_source_sha256"] = hashlib.sha256(encoded).hexdigest()
    return plan


def _bound_finalization_plan(
    supplied: dict[str, Any],
) -> tuple[dict[str, Any], bytes | None, list[str]]:
    """Bind finalization to the canonical on-disk plan bytes.

    The caller may have loaded a stale, alternate, or subsequently mutated
    plan.  Publication decisions always use the single byte snapshot stored at
    ``<output_dir>/orchestration/plan.json``; any disagreement fails closed.
    """
    failures: list[str] = []
    supplied_public = {
        key: value for key, value in supplied.items() if not key.startswith("_plan_source_")
    }
    try:
        output_dir = Path(str(supplied_public["output_dir"])).expanduser().resolve()
    except (KeyError, TypeError, ValueError):
        return supplied_public, None, ["invalid_supplied_plan_output_dir"]
    canonical_path = _plan_path(output_dir).resolve()
    source_path = supplied.get("_plan_source_path")
    if source_path is None:
        failures.append("missing_supplied_plan_source_identity")
    else:
        try:
            if Path(str(source_path)).expanduser().resolve() != canonical_path:
                failures.append("supplied_plan_path_is_not_canonical")
        except (OSError, TypeError, ValueError):
            failures.append("invalid_supplied_plan_source_path")
    try:
        encoded = canonical_path.read_bytes()
        disk_plan = json.loads(encoded)
        if not isinstance(disk_plan, dict):
            raise TypeError("plan must be a JSON object")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return supplied_public, None, failures + ["canonical_plan_missing_or_invalid"]
    digest = hashlib.sha256(encoded).hexdigest()
    if supplied.get("_plan_source_sha256") != digest:
        failures.append("supplied_plan_sha256_mismatch")
    if supplied_public != disk_plan:
        failures.append("supplied_plan_content_mismatch")
    return disk_plan, encoded, list(dict.fromkeys(failures))


def _benchmark_argv(
    plan: dict[str, Any], job: dict[str, Any], output_dir: Path, *, execution: str = "fresh"
) -> list[str]:
    argv = [
        "--profile",
        str(plan["profile"]),
        "--data_root",
        str(plan["data_root"]),
        "--datasets",
        str(job["dataset"]),
        "--cover",
        str(job["cover"]),
        "--methods",
        ",".join(plan["methods"]),
        "--seeds",
        str(plan["seeds"]),
        "--resolutions",
        str(plan["resolutions"]),
        "--timeout_per_run",
        str(plan["timeout_per_run"]),
        "--timeout_by_method",
        ",".join(f"{name}={seconds}" for name, seconds in plan.get("timeout_by_method", {}).items()),
        "--timeout_by_dataset",
        ",".join(f"{name}={seconds}" for name, seconds in plan.get("timeout_by_dataset", {}).items()),
        "--max_nodes",
        str(plan["max_nodes"]),
        "--max_memberships",
        str(job["max_memberships"]),
        "--memory_limit_gb",
        str(job["memory"]["detector_memory_limit_bytes"] / (1024**3)),
        "--output_dir",
        str(output_dir),
        "--no-plots",
        "--execution",
        execution,
        "--expected_dataset_metadata_sha256",
        str(job["dataset_metadata_identity"]["sha256"]),
    ]
    if plan.get("not_rerun_external_methods"):
        argv.extend(["--skip-methods", ",".join(plan["not_rerun_external_methods"])])
    if bool(plan.get("resume", True)):
        argv.append("--resume")
    if bool(plan.get("omega", True)):
        argv.extend(["--omega", "--omega_sample_size", str(plan["omega_sample_size"])])
    return argv


def _job_retryable_records(
    shard_dir: Path, plan: dict[str, Any], job: dict[str, Any]
) -> int:
    return sum(
        1
        for record in benchmark._completed_records(shard_dir)
        if (
            record.get("status") in RETRYABLE_STATUSES
            and record.get("dataset") == job["dataset"]
            and record.get("cover") == job["cover"]
            and record.get("method") in set(plan["methods"])
        )
    )


def _process_is_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _mark_worker_terminal(
    plan: dict[str, Any], worker_index: int, state: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Atomically turn every unfinished wave into an error terminal state."""
    repaired = dict(state)
    repaired["status"] = "error"
    repaired.setdefault("failures", []).append({"name": "worker", "error": reason})
    waves = list(repaired.get("waves") or [])
    for index, wave in enumerate(waves):
        current = dict(wave) if isinstance(wave, dict) else {"index": index}
        if current.get("status") not in {"completed", "error", "idle"}:
            current["status"] = "error"
            current["reason"] = reason
        waves[index] = current
    repaired["waves"] = waves
    repaired["updated_at"] = _timestamp()
    _write_json(_state_path(Path(plan["output_dir"]), worker_index), repaired)
    return repaired


def _repair_worker_liveness(
    plan: dict[str, Any], states: list[dict[str, Any] | None], *, started: float
) -> list[dict[str, Any] | None]:
    startup_timeout = float(plan.get("worker_startup_timeout_seconds", 120.0))
    for index, state in enumerate(states):
        if state is None or state.get("plan_id") != plan.get("plan_id"):
            continue
        status = state.get("status")
        pid = state.get("worker_pid")
        if status == "running" and not _process_is_alive(pid):
            states[index] = _mark_worker_terminal(
                plan,
                index,
                state,
                "worker process disappeared before finalizing its waves",
            )
        elif status == "pending" and time.monotonic() - started >= startup_timeout:
            states[index] = _mark_worker_terminal(
                plan,
                index,
                state,
                f"worker did not start within {startup_timeout:g} seconds",
            )
    return states


def _wait_for_wave(plan: dict[str, Any], wave_index: int) -> None:
    """Barrier preventing a later wave from overlapping the current one."""
    output_dir = Path(plan["output_dir"])
    worker_count = int(plan["workers"])
    final_statuses = {"completed", "error", "idle"}
    started = time.monotonic()
    while True:
        states = [_read_json(_state_path(output_dir, index)) for index in range(worker_count)]
        states = _repair_worker_liveness(plan, states, started=started)
        if all(
            state is not None
            and state.get("plan_id") == plan["plan_id"]
            and isinstance(state.get("waves"), list)
            and len(state["waves"]) > wave_index
            and (
                state["waves"][wave_index].get("status") in final_statuses
                or state.get("status") == "error"
            )
            for state in states
        ):
            return
        time.sleep(float(plan["poll_seconds"]))


def run_worker(plan: dict[str, Any], worker_index: int, *, synchronize: bool = True) -> int:
    """Execute assigned jobs one memory-budgeted wave at a time."""
    assignments = plan["assignments"]
    worker_count = int(plan["workers"])
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(f"Worker index {worker_index} is outside this plan")
    output_dir = Path(plan["output_dir"])
    planned_jobs = [
        job["name"]
        for wave in assignments
        if worker_index < len(wave) and isinstance((job := wave[worker_index]), dict)
    ]
    state = {
        "plan_id": plan["plan_id"],
        "worker_index": worker_index,
        "worker_pid": os.getpid(),
        "status": "running",
        "jobs": planned_jobs,
        "completed_jobs": [],
        "failures": [],
        "waves": [
            {"index": wave_index, "status": "pending"}
            for wave_index in range(len(assignments))
        ],
        "started_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    _write_json(_state_path(output_dir, worker_index), state)

    for wave_index, wave in enumerate(assignments):
        job = wave[worker_index] if worker_index < len(wave) else None
        state["waves"][wave_index] = {
            "index": wave_index,
            "status": "running" if isinstance(job, dict) else "idle",
            "job": job["name"] if isinstance(job, dict) else None,
        }
        state["updated_at"] = _timestamp()
        _write_json(_state_path(output_dir, worker_index), state)
        if isinstance(job, dict):
            shard_dir = output_dir / "shards" / str(job["name"])
            attempts = 0
            retry_events = 0
            try:
                while True:
                    code = benchmark.main(
                        _benchmark_argv(
                            plan,
                            job,
                            shard_dir,
                            execution="retry" if attempts else "fresh",
                        )
                    )
                    retryable = _job_retryable_records(shard_dir, plan, job)
                    can_retry = (
                        retryable > 0
                        and attempts < min(int(plan["retry_attempts"]), 1)
                    )
                    if not can_retry:
                        break
                    attempts += 1
                    retry_events += 1
                if code != 0:
                    state["failures"].append(
                        {"name": job["name"], "error": f"benchmark returned exit code {code}"}
                    )
                    state["waves"][wave_index]["status"] = "error"
                else:
                    state["completed_jobs"].append(
                        {
                            "name": job["name"],
                            "attempts": attempts + 1,
                            "retry_events": retry_events,
                            "retryable_records": retryable,
                            "retry_policy": "bounded_no_loop",
                        }
                    )
                    state["waves"][wave_index]["status"] = "completed"
            except BaseException as exc:
                state["failures"].append({"name": job["name"], "error": f"{type(exc).__name__}: {exc}"})
                state["waves"][wave_index]["status"] = "error"
        state["updated_at"] = _timestamp()
        _write_json(_state_path(output_dir, worker_index), state)
        if synchronize:
            _wait_for_wave(plan, wave_index)

    state["status"] = "completed" if not state["failures"] else "error"
    state["finished_at"] = _timestamp()
    state["updated_at"] = state["finished_at"]
    _write_json(_state_path(output_dir, worker_index), state)
    return 0 if state["status"] == "completed" else 1


def _selected_record(
    record: dict[str, Any], plan: dict[str, Any], job: dict[str, Any]
) -> bool:
    return (
        record.get("method") in set(plan["methods"])
        and record.get("dataset") == job["dataset"]
        and record.get("cover") == job["cover"]
    )


def _collect_shard_records(plan: dict[str, Any]) -> list[dict[str, Any]]:
    output_dir = Path(plan["output_dir"])
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for job in plan["jobs"]:
        shard_dir = output_dir / "shards" / str(job["name"])
        for record in benchmark._completed_records(shard_dir):
            if not _selected_record(record, plan, job):
                continue
            key = (
                record.get("dataset"),
                record.get("cover"),
                record.get("method"),
                record.get("seed"),
                record.get("resolution"),
            )
            previous = selected.get(key)
            if previous is None or str(record.get("created_at", "")) > str(
                previous.get("created_at", "")
            ):
                selected[key] = record
    records = list(selected.values())
    return sorted(
        records,
        key=lambda record: (
            str(record.get("dataset")),
            str(record.get("cover")),
            str(record.get("method")),
            int(record.get("seed", 0)),
            float(record.get("resolution", 0)),
        ),
    )


def _load_protocol_admissible_records(
    output_dir: Path, protocol_audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load the exact record bytes admitted by reconciliation.

    Condition keys alone are insufficient when duplicate or misplaced shard
    files exist.  Bind publication aggregation to the reconciler's precise
    relative path and SHA-256, and recheck the bytes immediately before use.
    """
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in protocol_audit.get("rows", []):
        if not row.get("admissible"):
            continue
        reference = row.get("record_path")
        expected_sha256 = row.get("record_sha256")
        if not isinstance(reference, str) or not isinstance(expected_sha256, str):
            failures.append("missing_admitted_record_identity")
            continue
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"nonportable_admitted_record_path:{reference}")
            continue
        path = output_dir / relative
        try:
            encoded = path.read_bytes()
        except OSError:
            failures.append(f"missing_admitted_record:{reference}")
            continue
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        if actual_sha256 != expected_sha256:
            failures.append(f"admitted_record_sha256_mismatch:{reference}")
            continue
        try:
            record = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            record = None
        if not isinstance(record, dict):
            failures.append(f"invalid_admitted_record:{reference}")
            continue
        verified_artifacts = row.get("verified_artifacts")
        required_prefixes = {
            "analysis_graph",
            "ground_truth_cover",
            "final_cover",
        }
        if str(record.get("method", "")).startswith("hedonic_"):
            required_prefixes.update(
                {"pre_cleanup_membership", "final_membership"}
            )
        if not isinstance(verified_artifacts, dict) or not required_prefixes.issubset(
            verified_artifacts
        ):
            failures.append(f"missing_admitted_artifact_bindings:{reference}")
            continue
        artifact_root = path.parent
        while artifact_root != artifact_root.parent and artifact_root.name != "runs":
            artifact_root = artifact_root.parent
        if artifact_root.name != "runs":
            failures.append(f"invalid_admitted_artifact_root:{reference}")
            continue
        artifact_root = artifact_root.parent
        artifact_failure = False
        for prefix in sorted(required_prefixes):
            identity = verified_artifacts.get(prefix)
            if not isinstance(identity, dict):
                artifact_failure = True
                failures.append(f"missing_admitted_{prefix}_binding:{reference}")
                continue
            artifact_reference = identity.get("path")
            artifact_sha256 = identity.get("artifact_sha256")
            content_sha256 = identity.get("content_sha256")
            if (
                artifact_reference != record.get(f"{prefix}_artifact")
                or artifact_sha256 != record.get(f"{prefix}_artifact_sha256")
                or content_sha256 != record.get(f"{prefix}_sha256")
                or not isinstance(artifact_reference, str)
            ):
                artifact_failure = True
                failures.append(f"admitted_{prefix}_identity_mismatch:{reference}")
                continue
            artifact_relative = Path(artifact_reference)
            if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
                artifact_failure = True
                failures.append(f"nonportable_admitted_{prefix}_path:{reference}")
                continue
            try:
                artifact_bytes = (artifact_root / artifact_relative).read_bytes()
            except OSError:
                artifact_failure = True
                failures.append(f"missing_admitted_{prefix}:{reference}")
                continue
            if hashlib.sha256(artifact_bytes).hexdigest() != artifact_sha256:
                artifact_failure = True
                failures.append(f"admitted_{prefix}_sha256_mismatch:{reference}")
        if artifact_failure:
            continue
        record["run_path"] = relative.as_posix()
        records.append(record)
    return records, failures


def _collect_execution_counts(plan: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for job in plan["jobs"]:
        manifest = _read_json(Path(plan["output_dir"]) / "shards" / str(job["name"]) / "manifest.json")
        if not manifest:
            continue
        for name, value in (manifest.get("execution_counts") or {}).items():
            try:
                counts[str(name)] += int(value)
            except (TypeError, ValueError):
                continue
    return dict(sorted(counts.items()))


def _expected_count(plan: dict[str, Any]) -> int:
    seeds = benchmark.parse_seeds(str(plan["seeds"])) or []
    resolutions = benchmark.parse_resolutions(str(plan["resolutions"])) or []
    return len(plan["jobs"]) * len(plan["methods"]) * len(seeds) * len(resolutions)


def _condition_summary(plan: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize one honest row for every dataset/method/seed condition."""
    seeds = benchmark.parse_seeds(str(plan["seeds"])) or []
    resolutions = benchmark.parse_resolutions(str(plan["resolutions"])) or []
    rows: list[dict[str, Any]] = []
    for job in plan["jobs"]:
        for method in plan["methods"]:
            for seed in seeds:
                candidates = [
                    record
                    for record in records
                    if record.get("dataset") == job["dataset"]
                    and record.get("cover") == job["cover"]
                    and record.get("method") == method
                    and int(record.get("seed", -1)) == seed
                ]
                if not candidates:
                    for requested in resolutions:
                        rows.append(
                            {
                                "dataset": job["dataset"],
                                "cover": job["cover"],
                                "method": method,
                                "seed": seed,
                                "requested_resolution": requested,
                                "status": "missing",
                                "reason": "no resumable run record",
                            }
                        )
                    continue
                for record in candidates:
                    rows.append(
                        {
                            "dataset": record.get("dataset"),
                            "cover": record.get("cover"),
                            "method": record.get("method"),
                            "seed": record.get("seed"),
                            "requested_resolution": "auto" if "auto" in resolutions else record.get("resolution"),
                            "resolution": record.get("resolution"),
                            "status": record.get("status"),
                            "execution": record.get("execution"),
                            "runtime_seconds": record.get("runtime_seconds"),
                            "failure_kind": record.get("failure_kind"),
                            "resource_status": record.get("resource_status"),
                            "reason": record.get("reason") or record.get("error"),
                        }
                    )
    return rows


def _audit_records(plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(record.get("status", "unknown")) for record in records)
    resource_counts = Counter(
        str(record.get("resource_status"))
        for record in records
        if record.get("resource_status")
    )
    expected = _expected_count(plan)
    retryable = sum(counts[status] for status in RETRYABLE_STATUSES)
    seeds = benchmark.parse_seeds(str(plan["seeds"])) or []
    expected_bases = {
        (job["dataset"], job["cover"], method, seed)
        for job in plan["jobs"]
        for method in plan["methods"]
        for seed in seeds
    }
    expected_per_base = len(benchmark.parse_resolutions(str(plan["resolutions"])) or [])
    observed_bases = Counter(
        (record.get("dataset"), record.get("cover"), record.get("method"), record.get("seed"))
        for record in records
    )
    missing_bases = sorted(
        "/".join(map(str, base))
        for base in expected_bases
        if observed_bases[base] != expected_per_base
    )
    return {
        "profile": plan.get("profile"),
        "expected_records": expected,
        "observed_records": len(records),
        "status_counts": dict(sorted(counts.items())),
        "resource_status_counts": dict(sorted(resource_counts.items())),
        "missing_records": max(0, expected - len(records)),
        "missing_or_incomplete_conditions": missing_bases,
        "retryable_records": retryable,
        "non_completed_records": expected - counts["completed"],
        "failure_records": [
            {
                "dataset": record.get("dataset"),
                "cover": record.get("cover"),
                "method": record.get("method"),
                "seed": record.get("seed"),
                "resolution": record.get("resolution"),
                "status": record.get("status"),
                "resource_status": record.get("resource_status"),
                "failure_kind": record.get("failure_kind"),
                "reason": record.get("reason") or record.get("error"),
                "runtime_seconds": record.get("runtime_seconds"),
                "memory": record.get("memory") or record.get("detector_memory"),
            }
            for record in records
            if record.get("status") != "completed"
        ],
        "execution_counts": dict(
            Counter(str(record.get("execution", "unknown")) for record in records)
        ),
        "ready_for_paper": (
            plan.get("profile") in {"standard", "full"}
            and len(records) == expected
            and not missing_bases
            and counts["completed"] == expected
        ),
    }


def _coverage_report(plan: dict[str, Any], records: list[dict[str, Any]], audit: dict[str, Any]) -> dict[str, Any]:
    """One inspectable coverage row per requested condition and its outcome."""
    rows = _condition_summary(plan, records)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    resource_counts = Counter(
        str(row.get("resource_status")) for row in rows if row.get("resource_status")
    )
    def resource_total(name: str) -> int:
        return sum(
            row.get("status") == name or row.get("resource_status") == name
            for row in rows
        )

    return {
        "schema_version": PAPER_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "expected_records": audit["expected_records"],
        "completed": status_counts["completed"],
        "not_scalable": status_counts["skipped_not_scalable"],
        "not_rerun_external": status_counts["skipped_external_unchanged"],
        "timeout": resource_total("timeout"),
        "oom": resource_total("oom"),
        "memory_limit": resource_total("memory_limit"),
        "missing": status_counts["missing"],
        "status_counts": dict(sorted(status_counts.items())),
        "resource_status_counts": dict(sorted(resource_counts.items())),
        "rows": rows,
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    return mean, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def _record_metric(record: dict[str, Any], metric: str) -> float | None:
    """Read a paper metric exclusively from the protocol-bound mapping."""
    candidates = _PAPER_METRIC_ALIASES.get(metric, (metric,))
    nested = record.get("metrics")
    if not isinstance(nested, dict):
        return None
    for candidate in candidates:
        value = _numeric(nested.get(candidate))
        if value is not None:
            return value
    return None


def _paper_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record.get("dataset")),
                str(record.get("cover")),
                str(record.get("method")),
                float(record.get("resolution", 0)),
            )
        ].append(record)
    result: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        dataset, cover, method, resolution = key
        row: dict[str, Any] = {
            "dataset": dataset,
            "cover": cover,
            "method": method,
            "resolution": resolution,
            "n_records": len(group),
        }
        statuses = Counter(str(record.get("status", "unknown")) for record in group)
        for status, count in sorted(statuses.items()):
            row[f"n_{status}"] = count
        completed = [record for record in group if record.get("status") == "completed"]
        row["n_completed"] = len(completed)
        for metric in _PAPER_METRICS:
            values = [
                value
                for record in completed
                for value in [_record_metric(record, metric)]
                if value is not None
            ]
            mean, ci = _mean_ci(values)
            if mean is not None:
                row[f"mean_{metric}"] = mean
                row[f"ci95_{metric}"] = ci
                row[f"n_{metric}"] = len(values)
        result.append(row)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("&", "\\&")


def _tex_method(method: str) -> str:
    return {
        "hedonic_multiphase": r"\methodmulti{}",
        "hedonic_multiphase_x10": r"\methodmultiTen{}",
        "hedonic_multiphase_x100": r"\methodmultiHundred{}",
        "cpm": r"\methodcpm{}",
        "demon": r"\methoddemon{}",
    }.get(method, r"\texttt{" + _tex_escape(method) + "}")


def _format_metric(row: dict[str, Any], metric: str, *, seconds: bool = False) -> str:
    value = row.get(f"mean_{metric}")
    if not isinstance(value, (int, float)):
        return "--"
    ci = row.get(f"ci95_{metric}")
    if seconds:
        if value < 1:
            text = f"{value:.3f}"
            ci_text = f"{float(ci):.3f}" if isinstance(ci, (int, float)) else None
        else:
            text = f"{value:.1f}"
            ci_text = f"{float(ci):.1f}" if isinstance(ci, (int, float)) else None
    else:
        text = f"{value:.3f}"
        ci_text = f"{float(ci):.3f}" if isinstance(ci, (int, float)) else None
    return text if ci_text is None else f"{text} $\\pm$ {ci_text}"


def _write_paper_tex(output_dir: Path, plan: dict[str, Any], audit: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    complete_rows = [row for row in rows if int(row.get("n_completed", 0)) > 0]
    status_counts = audit["status_counts"]
    max_nodes = int(plan.get("max_nodes", 0) or 0)
    scope_text = (
        f"a deterministic induced-subgraph cap of {max_nodes:,} vertices"
        if max_nodes > 0
        else "the full validated graph"
    )
    lines = [
        "% Generated by hedonic-exp reproduce-overlapping-paper. Do not edit.",
        r"\subsection{Ground-truth recovery}",
        (
            "The reproducibility runner aggregated "
            f"{audit['observed_records']} of {audit['expected_records']} planned run records "
            "from the immutable per-condition caches.  Entries are mean $\\pm$ 95\\% "
            "normal-approximation confidence interval across completed seeds; status counts "
            "remain explicit rather than being silently discarded."
        ),
        (
            "The reported comparison uses "
            + scope_text
            + ".  Hedonic rows use full Leiden followed by local-only cleanup and are "
            "published only when an independent unit-$\\ell_2$ best-response audit verifies "
            "the exact labelled final-membership state at the recorded resolution and "
            "membership cap. Recovery metrics use its bound canonical unique-set projection; "
            "duplicate label bodies, if present, are not claimed to preserve equilibrium "
            "after that scoring projection. "
            "CPM and DEMON have no hedonic-equilibrium claim."
        ),
        r"\begin{table*}[t]",
        r"\centering",
        (
            r"\caption{SNAP recovery results generated from cached run records on "
            + scope_text
            + r". $n$ is the number of completed seeds; \texttt{--} means no completed estimate.}"
        ),
        r"\label{tab:full-recovery}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $n$ & Best-match $F_1$ & Matched $F_1$ & Node micro-$F_1$ & Omega & Runtime (s) \\",
        r"\midrule",
    ]
    for row in rows:
        dataset = _tex_escape(str(row["dataset"]))
        lines.append(
            " & ".join(
                [
                    dataset,
                    _tex_method(str(row["method"])),
                    str(row.get("n_completed", 0)),
                    _format_metric(row, "symmetric_best_match_f1"),
                    _format_metric(row, "matching_f1"),
                    _format_metric(row, "node_membership_micro_f1"),
                    _format_metric(row, "omega"),
                    _format_metric(row, "runtime_seconds", seconds=True),
                ]
            )
            + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            (
                r"\par\smallskip\footnotesize Status counts across all planned records: "
                + ", ".join(f"{_tex_escape(status)}={count}" for status, count in sorted(status_counts.items()))
                + "."
            ),
            (
                r"\par\smallskip\footnotesize Completed estimates are the only values used in "
                r"means. Records marked \texttt{timeout}, \texttt{memory\_limit}, "
                r"\texttt{oom}, \texttt{failed}, \texttt{skipped\_unsupported}, or "
                r"\texttt{skipped\_external\_unchanged} are explicit "
                r"non-results and are reported in the failure report."
            ),
            r"\end{table*}",
            r"\subsection{Overlap structure and CPM quality}",
            (
                "The machine-readable paper summary includes coverage, inclusion, predicted overlap, "
                "community-size, membership, and CPM-quality diagnostics for every completed condition. "
                "CPM quality is deliberately marked unavailable where full-network computation is skipped. "
                "The primary recovery, structure, and runtime scorecard is shown in "
                r"Fig.~\ref{fig:full-overview}."
            ),
            r"\begin{figure*}[t]",
            r"  \centering",
            r"  \includegraphics[width=.96\textwidth]{benchmark_overview.pdf}",
            r"  \caption{Primary recovery, structure, and runtime scorecard on the bounded SNAP comparison. Each point is a mean over completed seeds with a 95\% normal-approximation interval; colors identify methods consistently across panels. The black marks in the overlap panel are the supplied metadata fractions, and the runtime panel uses a logarithmic scale.}",
            r"  \Description{Five panels compare best-match F1, one-to-one matched F1, Omega agreement, predicted overlap, and logarithmic detector runtime for five datasets and five methods.}",
            r"  \label{fig:full-overview}",
            r"\end{figure*}",
            r"\subsection{Runtime and completion}",
            (
                "Runtime figures report detector wall-clock time for completed conditions. "
                "Timeouts, errors, unavailable baselines, and intentionally skipped conditions are retained "
                "in the artifact manifest and are not converted into artificial runtimes."
            ),
        ]
    )
    if not complete_rows:
        lines.insert(2, r"\textbf{No completed full-network records are available yet.}")
    (output_dir / "paper_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_paper_status(output_dir: Path, audit: dict[str, Any], *, partial: bool = False) -> None:
    warning = ""
    if audit["ready_for_paper"]:
        state = r"\smokeresultsfalse"
        label = "complete bounded protocol" if audit.get("profile") == "standard" else "complete full protocol"
    elif partial:
        state = r"\smokeresultstrue"
        label = "EXPLICIT PARTIAL COMPILE: incomplete protocol; inspect failure_report.json"
        warning = (
            r"\newcommand{\partialcompilewarning}{%" "\n"
            r"\par\noindent\colorbox{yellow!35}{\parbox{0.94\linewidth}{%" "\n"
            r"\textbf{EXPLICIT PARTIAL COMPILE.} This PDF contains provisional "
            r"results; inspect \texttt{artifacts/papers/overlapping\_communities/full/failure\_report.json}.}}\par}"
            "\n"
        )
    else:
        state = r"\smokeresultstrue"
        label = "incomplete or non-full protocol"
    (output_dir / "paper_status.tex").write_text(
        "% Generated by hedonic-exp reproduce-overlapping-paper.\n"
        f"% State: {label}.\n{state}\n{warning}",
        encoding="utf-8",
    )


def _compile_paper(plan: dict[str, Any], audit: dict[str, Any]) -> int:
    if not bool(plan.get("compile_paper")):
        return 0
    if not bool(audit["ready_for_paper"]) and not bool(plan.get("compile_partial")):
        return 0
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        print("[paper] latexmk is unavailable; artifacts were generated but PDF was not compiled.", file=sys.stderr)
        return 1
    paper_dir = Path(plan["paper_dir"])
    build_dir = Path(plan["output_dir"]).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            f"-outdir={build_dir}",
            "main.tex",
        ],
        cwd=paper_dir,
        text=True,
    )
    return int(proc.returncode)


def finalize(plan: dict[str, Any]) -> int:
    """Merge shard caches and materialize aggregate, plot, and paper artifacts."""
    plan, bound_plan_bytes, plan_binding_failures = _bound_finalization_plan(plan)
    output_dir = Path(plan["output_dir"])
    records = _collect_shard_records(plan)
    audit = _audit_records(plan, records)
    audit["execution_counts"] = _collect_execution_counts(plan)
    try:
        protocol_audit = protocol.reconcile(
            output_dir,
            Path(plan["config_path"]),
            protocol.LOCK_PATH,
            plan_bytes=bound_plan_bytes,
        )
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        protocol_audit = {
            "ready_for_publication": False,
            "expected_conditions": _expected_count(plan),
            "admissible_records": 0,
            "rejected_records": len(records),
            "missing_records": max(0, _expected_count(plan) - len(records)),
            "rejection_reason_counts": {
                f"reconcile_error:{type(exc).__name__}": len(records) or 1
            },
            "rows": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    if plan_binding_failures:
        protocol_audit["ready_for_publication"] = False
        protocol_audit["finalize_plan_binding_failures"] = plan_binding_failures
        global_reasons = list(protocol_audit.get("global_rejection_reasons") or [])
        global_reasons.extend(
            f"finalize_plan_binding:{reason}" for reason in plan_binding_failures
        )
        protocol_audit["global_rejection_reasons"] = list(
            dict.fromkeys(global_reasons)
        )
        counts = Counter(protocol_audit.get("rejection_reason_counts") or {})
        for reason in plan_binding_failures:
            counts[f"finalize_plan_binding:{reason}"] += 1
        protocol_audit["rejection_reason_counts"] = dict(sorted(counts.items()))
    if plan.get("profile") in {"standard", "full"}:
        publication_records, binding_failures = _load_protocol_admissible_records(
            output_dir, protocol_audit
        )
        if binding_failures:
            protocol_audit["ready_for_publication"] = False
            protocol_audit["finalize_artifact_binding_failures"] = binding_failures
            counts = Counter(protocol_audit.get("rejection_reason_counts") or {})
            counts["finalize_admitted_record_binding_failure"] += len(binding_failures)
            protocol_audit["rejection_reason_counts"] = dict(sorted(counts.items()))
    else:
        # Smoke output remains useful as a diagnostic aggregate, but the
        # publication gate below can never mark it ready.
        publication_records = records
    audit["protocol_reconciliation"] = {
        key: protocol_audit.get(key)
        for key in (
            "ready_for_publication",
            "expected_conditions",
            "present_records",
            "admissible_records",
            "rejected_records",
            "missing_records",
            "rejection_reason_counts",
            "global_rejection_reasons",
            "error",
        )
        if key in protocol_audit
    }
    _write_json(output_dir / "protocol_reconciliation.json", protocol_audit)
    _write_csv(
        output_dir / "protocol_reconciliation.csv",
        [
            {
                **row,
                "rejection_reasons": ";".join(
                    row.get("rejection_reasons") or []
                ),
            }
            for row in protocol_audit.get("rows", [])
        ],
    )
    audit["ready_for_paper"] = bool(
        audit["ready_for_paper"]
        and protocol_audit.get("ready_for_publication")
        and plan.get("profile") in {"standard", "full"}
    )
    rows = benchmark._write_results(output_dir, publication_records)
    benchmark._write_summary(output_dir, rows)
    condition_rows = _condition_summary(plan, records)
    _write_csv(output_dir / "condition_summary.csv", condition_rows)
    _write_json(output_dir / "condition_summary.json", {"rows": condition_rows})
    coverage = _coverage_report(plan, records, audit)
    _write_json(output_dir / "coverage_report.json", coverage)
    _write_csv(output_dir / "coverage_report.csv", coverage["rows"])
    plots = benchmark._write_plots(output_dir, rows)
    paper_rows = _paper_summary(publication_records)
    _write_csv(output_dir / "paper_summary.csv", paper_rows)
    failure_rows = audit["failure_records"]
    protocol_failure_rows = [
        {
            "dataset": row.get("dataset"),
            "cover": row.get("cover"),
            "method": row.get("method"),
            "seed": row.get("seed"),
            "resolution": row.get("resolution"),
            "status": "protocol_rejected",
            "reason": ";".join(row.get("rejection_reasons") or []),
            "record_path": row.get("record_path"),
        }
        for row in protocol_audit.get("rows", [])
        if not row.get("admissible")
    ]
    failure_rows = failure_rows + protocol_failure_rows
    _write_json(
        output_dir / "failure_report.json",
        {
            "schema_version": PAPER_SCHEMA_VERSION,
            "created_at": _timestamp(),
            "complete_protocol": bool(audit["ready_for_paper"]),
            "records": failure_rows,
            "status_counts": audit["status_counts"],
            "protocol_reconciliation": protocol_audit,
        },
    )
    _write_csv(output_dir / "failure_report.csv", failure_rows)
    paper_manifest = {
        "schema_version": PAPER_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "plan_id": plan["plan_id"],
        "config_path": Path(plan["config_path"]).name,
        "config_sha256": plan.get("config_sha256"),
        "experiment_identity": plan["experiment_identity"],
        "methods": plan["methods"],
        "jobs": plan["jobs"],
        "audit": audit,
        "artifacts": {
            "results_csv": "results.csv.gz",
            "summary_csv": "summary.csv",
            "paper_summary_csv": "paper_summary.csv",
            "condition_summary_csv": "condition_summary.csv",
            "condition_summary_json": "condition_summary.json",
            "coverage_report_csv": "coverage_report.csv",
            "coverage_report_json": "coverage_report.json",
            "paper_results_tex": "paper_results.tex",
            "failure_report_json": "failure_report.json",
            "failure_report_csv": "failure_report.csv",
            "protocol_reconciliation_json": "protocol_reconciliation.json",
            "protocol_reconciliation_csv": "protocol_reconciliation.csv",
            "plots": [
                Path(path).relative_to(output_dir).as_posix()
                if Path(path).is_absolute() and output_dir in Path(path).parents
                else str(path)
                for path in plots
            ],
        },
    }
    _write_json(output_dir / "paper_manifest.json", paper_manifest)
    # Keep the conventional top-level manifest name for tooling that already
    # understands benchmark artifacts; paper_manifest.json remains the more
    # explicit stable name for this orchestration layer.
    _write_json(output_dir / "manifest.json", paper_manifest)
    _write_paper_tex(output_dir, plan, audit, paper_rows)
    _write_paper_status(output_dir, audit, partial=bool(plan.get("compile_partial")))
    compile_code = _compile_paper(plan, audit)
    if compile_code != 0:
        # Do not leave a status file that asks TeX to use full results after a
        # failed render. The cached numerical artifacts remain intact.
        (output_dir / "paper_status.tex").write_text(
            "% Generated full artifacts, but manuscript compilation failed.\n"
            "\\smokeresultstrue\n",
            encoding="utf-8",
        )
        return compile_code
    print(f"[finalize] records={audit['observed_records']}/{audit['expected_records']}")
    print(f"[finalize] status_counts={json.dumps(audit['status_counts'], sort_keys=True)}")
    print(f"[finalize] artifacts: {output_dir}")
    if not audit["ready_for_paper"]:
        print(
            "[finalize] full-paper switch remains disabled: inspect failure_report.json; use --compile-partial only with an explicit visible warning.",
            file=sys.stderr,
        )
    return 0


def run_coordinator(plan: dict[str, Any]) -> int:
    """Wait for tmux workers, then run the sole aggregate/paper writer."""
    output_dir = Path(plan["output_dir"])
    worker_count = int(plan["workers"])
    print(f"[coordinator] waiting for {worker_count} worker windows", flush=True)
    started = time.monotonic()
    while True:
        states = [_read_json(_state_path(output_dir, index)) for index in range(worker_count)]
        states = _repair_worker_liveness(plan, states, started=started)
        final = [
            state is not None
            and state.get("plan_id") == plan["plan_id"]
            and state.get("status") in {"completed", "error"}
            for state in states
        ]
        if all(final):
            break
        time.sleep(float(plan["poll_seconds"]))
    return finalize(plan)


def _tmux_command(
    plan_path: Path,
    action: str,
    worker_index: int | None = None,
    window_target: str | None = None,
) -> str:
    argv = [sys.executable, "-m", "hedonic.experiments.CLI", "reproduce-overlapping-paper", "--plan", str(plan_path)]
    if action == "worker":
        argv.extend(["--worker-index", str(worker_index)])
    else:
        argv.append("--coordinator")
    command = "exec " + shlex.join(argv)
    if window_target is not None:
        # A worker can fail before the parent gets a chance to set the option
        # in a second tmux command. Set it from the pane's shell first so the
        # window remains inspectable even for immediate failures.
        target = shlex.quote(window_target)
        return f"tmux set-window-option -t {target} remain-on-exit on; {command}"
    return command


def _print_memory_plan(plan: dict[str, Any]) -> None:
    gib = 1024**3
    print(
        "[plan] memory budget: "
        f"{plan['memory_budget_bytes'] / gib:.1f} GiB "
        f"(reserve {float(plan['memory_reserve_gb']):.1f} GiB; "
        f"safety factor {float(plan['memory_safety_factor']):.2g})"
    )
    for index, wave in enumerate(plan["assignments"], start=1):
        jobs = [job for job in wave if isinstance(job, dict)]
        used = sum(int(job["memory"]["estimated_peak_bytes"]) for job in jobs)
        details = ", ".join(
            f"{job['name']} ({job['memory']['estimated_peak_bytes'] / gib:.1f} GiB, "
            f"K={job['max_memberships']})"
            for job in jobs
        )
        print(f"[plan] wave {index}: {used / gib:.1f} GiB — {details}")


def launch_tmux(plan: dict[str, Any]) -> int:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required; install it or use --no-tmux for a foreground debug run")
    # tmux 3.5a normalizes periods in session names to underscores. Use the
    # normalized target for every subsequent command or later windows fail to
    # resolve even though the initial session was created successfully.
    session = str(plan["tmux_session"]).replace(".", "_")
    exists = subprocess.run(
        ["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0
    if exists:
        raise RuntimeError(
            f"tmux session {session!r} already exists; attach to it or set overlapping_paper.tmux_session to a new name"
        )
    plan_path = _plan_path(Path(plan["output_dir"])).resolve()
    for index in range(int(plan["workers"])):
        target = f"worker-{index + 1}"
        command = _tmux_command(plan_path, "worker", index, target)
        if index == 0:
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-n", target, command], check=True)
        else:
            subprocess.run(["tmux", "new-window", "-d", "-t", session, "-n", target, command], check=True)
    subprocess.run(
        [
            "tmux",
            "new-window",
            "-d",
            "-t",
            session,
            "-n",
            "coordinator",
            _tmux_command(plan_path, "coordinator", window_target="coordinator"),
        ],
        check=True,
    )
    print(f"[launch] tmux session: {session}")
    print(f"[launch] attach with: tmux attach -t {session}")
    print(f"[launch] plan: {plan_path}")
    _print_memory_plan(plan)
    return 0


def run_foreground(plan: dict[str, Any]) -> int:
    """Debug/test mode without tmux; intentionally serial and race-free."""
    code = 0
    for index in range(int(plan["workers"])):
        code = max(code, run_worker(plan, index, synchronize=False))
    finalized = finalize(plan)
    return max(code, finalized)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce a full overlapping-paper protocol from an externally "
            "supplied [overlapping_paper] TOML configuration and manuscript directory. "
            "Launches RAM-bounded tmux workers, "
            "merges their caches, regenerates figures/tables, and compiles only a complete full run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="TOML config (default: configs/hedonic.toml from cwd)")
    parser.add_argument("--workers", type=int, help="Override TOML worker count after CPU/RAM safety checks")
    parser.add_argument("--session", help="Override TOML tmux session name")
    parser.add_argument("--no-tmux", action="store_true", help="Foreground serial debug/test run; normal reproduction uses tmux")
    parser.add_argument("--dry-run", action="store_true", help="Validate TOML and write the balanced worker plan without running it")
    parser.add_argument("--finalize", action="store_true", help="Merge existing shard caches and regenerate paper artifacts only")
    parser.add_argument(
        "--compile-partial",
        action="store_true",
        help="Explicitly compile an incomplete paper with a visible warning in paper_status.tex",
    )
    parser.add_argument("--plan", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.plan:
        plan = _load_plan(experiment_config.expand_path(args.plan).resolve())
        if args.worker_index is not None:
            return run_worker(plan, args.worker_index)
        if args.coordinator:
            return run_coordinator(plan)
        if args.finalize:
            return finalize(plan)
        raise ValueError("Internal --plan requires --worker-index, --coordinator, or --finalize")
    if args.worker_index is not None or args.coordinator:
        raise ValueError("Internal worker/coordinator actions require --plan")

    options, _config_path = _load_options(args)
    options["compile_partial"] = bool(args.compile_partial)
    plan = _write_new_plan(options)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "plan": str(_plan_path(Path(plan["output_dir"]))),
                    "memory_budget_gib": plan["memory_budget_bytes"] / (1024**3),
                    "memory_reserve_gib": plan["memory_reserve_gb"],
                    "workers": plan["workers"],
                    "assignments": plan["assignments"],
                    "wave_reports": plan["wave_reports"],
                    "scheduler_policy": plan["scheduler_policy"],
                },
                indent=2,
            )
        )
        return 0
    if args.finalize:
        return finalize(plan)
    if args.no_tmux:
        return run_foreground(plan)
    return launch_tmux(plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
