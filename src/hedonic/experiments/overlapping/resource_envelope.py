"""Resource-envelope and capacity measurements (Astra TKT-14).

The resource experiment is deliberately a measurement protocol, rather than a
claim about detector scalability.  It records the quantities that determine the
overlapping token path (``n``, ``m``, incidences ``T``, incidence-aware token
edge bounds, cap bounds, projection observations, stage timings, and worker
RSS) and keeps capacity refusals, timeouts, and downstream failures in the
ledger.

The runner has two important operational properties:

* every case is an atomically written shard and can be resumed by its stable
  case key; and
* ``--preflight`` exercises the complete phase graph on a disposable tiny
  fixture and writes a machine-readable receipt without consuming a registered
  case.

The standard grid remains prospective.  The compatibility graph generator is
explicitly identified in every receipt; it is a resource stress fixture, not a
replacement for the canonical official-LFR graph ledger (TKT-11).
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
import tomllib
import traceback
from typing import Any, Iterable, Sequence

import igraph as ig
import numpy as np

from hedonic.experiments.config import OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.overlap_lfr import (
    environment_receipt,
    generate_overlapping_lfr,
    graph_sha256,
)
from hedonic.experiments.overlapping.metrics import evaluate_cover
from hedonic.experiments.overlapping.robustness import (
    audit_cover,
    cover_to_vertex_memberships,
)
from hedonic.Game import membership_incidence_trace, token_graph_preflight


PROTOCOL_VERSION = "resource-envelope-v2"
SCHEMA_VERSION = 2
DEFAULT_CAPS = (2, 4, 8)
DEFAULT_SIZES = (1_000, 5_000, 20_000, 100_000)
DEFAULT_PILOT_SIZES = (100, 500, 1_000)
DEFAULT_SMOKE_SIZES = (40, 80)
DEFAULT_METHODS = ("local", "multiphase", "audit")
DEFAULT_OPTIMIZER_SEEDS = (0, 1)
DEFAULT_TIMEOUT_SECONDS = 60.0
TOKEN_EDGE_BYTES = 24
TOKEN_VERTEX_BYTES = 24
RECORDS_DIRNAME = "records"


class ResourceEnvelopeInterrupted(RuntimeError):
    """Raised by the optional disposable interruption hook after a shard."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _source_receipt(config_path: str | Path | None = None) -> dict[str, Any]:
    """Bind the run to the source files that implement every phase."""
    root = _repository_root()
    relative_paths = (
        "src/hedonic/experiments/overlapping/resource_envelope.py",
        "src/hedonic/experiments/overlapping/execution.py",
        "src/hedonic/experiments/overlapping/overlap_lfr.py",
        "src/hedonic/experiments/overlapping/metrics.py",
        "src/hedonic/experiments/overlapping/robustness.py",
        "src/hedonic/Game.py",
        "pyproject.toml",
        "uv.lock",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        files[relative] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_bytes(path) if path.is_file() else None,
        }
    if config_path is not None:
        path = expand_path(config_path)
        files["config"] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_bytes(path) if path.is_file() else None,
        }
    commit = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            commit = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "repository_root": str(root),
        "git_head": commit,
        "files": files,
        "protocol_version": PROTOCOL_VERSION,
    }


def _dependency_closure() -> dict[str, Any]:
    """Report imports and distribution versions required by the phase graph."""
    root = _repository_root()
    project_dependencies: dict[str, str | None] = {}
    try:
        project_text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        project_text = ""
    # Keep this deliberately small and dependency-free.  Exact pins in the
    # project manifest are launch gates; unpinned scientific packages remain
    # recorded but are not misrepresented as exact locks.
    for distribution, version in re.findall(
        r"[\"']([A-Za-z0-9_.-]+)==([^\"'\s,]+)[\"']", project_text
    ):
        project_dependencies[distribution.lower().replace("_", "-")] = version
    requirements = (
        ("hedonic", "hedonic"),
        # lucas-igraph intentionally provides the ``igraph`` import name;
        # there is no separate ``igraph`` distribution in the locked runtime.
        ("igraph", "lucas-igraph"),
        ("numpy", "numpy"),
    )
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for import_name, distribution in requirements:
        try:
            spec = importlib.util.find_spec(import_name)
        except (ImportError, ValueError):
            spec = None
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        row = {
            "import": import_name,
            "distribution": distribution,
            "version": version,
            "expected_exact_version": project_dependencies.get(distribution.lower().replace("_", "-")),
            "origin": None if spec is None else str(spec.origin),
            "available": spec is not None,
        }
        expected = row["expected_exact_version"]
        row["version_matches_manifest"] = (
            version == expected if expected is not None else version is not None
        )
        rows.append(row)
        if spec is None or version is None or not row["version_matches_manifest"]:
            missing.append(import_name)
    return {
        "required": rows,
        "missing": missing,
        "complete": not missing,
        "policy": "exact pyproject pins are enforced; unpinned dependencies are recorded as present",
    }


def _protected_environment() -> dict[str, Any]:
    names = (
        "HEDONIC_DBLP_DIR",
        "HEDONIC_NETWORKS_DIR",
        "HEDONIC_SYNTHETIC_DIR",
        "HEDONIC_OUTPUT_DIR",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {name: os.environ.get(name) for name in names if name in os.environ}


def _runtime_receipt() -> dict[str, Any]:
    base = environment_receipt()
    base.update(
        {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "environment_variables": _protected_environment(),
            "platform_machine": platform.machine(),
            "platform_system": platform.system(),
        }
    )
    return base


def estimate_token_edges(graph: ig.Graph, memberships: Sequence[Sequence[int]], cap: int) -> dict[str, Any]:
    """Return exact incidence counts and conservative token-edge bounds.

    ``token_edges_incidence_upper_bound`` is an incidence-aware upper bound for
    the initial projection (``sum_e k_u k_v``).  ``token_edges_cap_upper_bound``
    is the cap-only bound used by the native preflight.  Neither is an observed
    post-Leiden edge count.  The old ``token_edges_predicted_from_rows`` key is
    retained as an alias for readers of schema v1 and is explicitly labelled as
    a bound in ``resource_semantics``.
    """
    if int(cap) < 1:
        raise ValueError("cap must be positive")
    rows = [sorted({int(label) for label in row}) for row in memberships]
    if len(rows) != graph.vcount():
        raise ValueError("memberships must contain one row per graph vertex")
    incidence = membership_incidence_trace(rows)
    incidence_edge_bound = sum(
        len(rows[first]) * len(rows[second])
        for first, second in graph.get_edgelist()
    )
    preflight = token_graph_preflight(graph.vcount(), graph.ecount(), int(cap), rows)
    token_vertices = int(incidence["unique_incidence_count"])
    estimated_native_bytes = (
        int(incidence_edge_bound) * TOKEN_EDGE_BYTES
        + int(token_vertices) * TOKEN_VERTEX_BYTES
    )
    cap_edge_bytes = int(preflight["estimated_token_edge_bytes"])
    return {
        "original_vertices": int(graph.vcount()),
        "original_edges": int(graph.ecount()),
        "original_incidences": int(incidence["incidence_count"]),
        "original_unique_incidences": token_vertices,
        "original_max_membership": max(incidence["multiplicities"], default=0),
        "token_vertices_exact": token_vertices,
        "token_vertices_predicted": token_vertices,
        "token_edges_incidence_upper_bound": int(incidence_edge_bound),
        "token_edges_predicted_from_rows": int(incidence_edge_bound),
        "token_edges_cap_upper_bound": int(preflight["max_token_edges"]),
        "token_edges_upper_bound_from_cap": int(preflight["max_token_edges"]),
        "token_edge_bytes_incidence_upper_bound": int(incidence_edge_bound) * TOKEN_EDGE_BYTES,
        "token_edge_bytes_upper_bound": cap_edge_bytes,
        "estimated_native_bytes": int(estimated_native_bytes),
        "preflight": preflight,
        "incidence_trace": incidence,
        "resource_semantics": {
            "token_vertices": "exact unique labelled incidences in supplied start",
            "token_edges_incidence": "upper bound sum_e(k_u*k_v), not observed native count",
            "token_edges_cap": "native cap-only integer/memory upper bound",
            "estimated_native_bytes": "incidence edge bound*24 + token vertices*24; conservative estimate",
        },
    }


def _observation(value: Any) -> dict[str, Any]:
    return {
        "value": value,
        "status": "observed" if value is not None else "not_exposed",
    }


def _native_measure(
    graph: ig.Graph,
    memberships: Sequence[Sequence[int]],
    *,
    cap: int,
    gamma: float,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Run one native call and preserve exposed instrumentation verbatim."""
    import random

    from hedonic import Game

    import igraph as igraph_module

    igraph_module.set_random_number_generator(random.Random(int(seed)))
    initial = [list(map(int, row)) for row in memberships]
    local_only = method == "local"
    started = time.perf_counter()
    result = Game(graph).community_hedonic(
        initial_membership=initial,
        max_memberships=int(cap),
        resolution=float(gamma),
        local_move_only=local_only,
        n_iterations=-1,
        allow_isolation=True,
        beta=0.01,
    )
    elapsed = time.perf_counter() - started
    final = getattr(result, "membership", None)
    if final and isinstance(final[0], (list, tuple)):
        final_rows = [list(map(int, row)) for row in final]
    else:
        final_rows = [[int(label)] for label in (final or [])]
    names = (
        "_hedonic_token_preflight",
        "_hedonic_start_membership_trace",
        "_hedonic_returned_membership_trace",
        "_hedonic_accepted_moves",
        "_hedonic_projection_events",
        "_hedonic_level_work",
        "_hedonic_vertex_visits",
        "_hedonic_commits",
        "_hedonic_sweeps",
        "_hedonic_native_quality",
        "_hedonic_algorithm_identity",
    )
    observed = {name: getattr(result, name, None) for name in names}
    return {
        "status": "completed",
        "runtime_seconds": elapsed,
        "final_memberships": final_rows,
        "pre_cleanup_memberships": getattr(result, "_hedonic_raw_memberships", None),
        "native_token_preflight": observed["_hedonic_token_preflight"],
        "start_membership_trace": observed["_hedonic_start_membership_trace"],
        "returned_membership_trace": observed["_hedonic_returned_membership_trace"],
        "accepted_moves": observed["_hedonic_accepted_moves"],
        "projection_events": observed["_hedonic_projection_events"],
        "work": {
            "per_level": observed["_hedonic_level_work"],
            "vertex_visits": observed["_hedonic_vertex_visits"],
            "commits": observed["_hedonic_commits"],
            "sweeps": observed["_hedonic_sweeps"],
            "status": (
                "observed"
                if any(observed[name] is not None for name in (
                    "_hedonic_level_work",
                    "_hedonic_vertex_visits",
                    "_hedonic_commits",
                    "_hedonic_sweeps",
                ))
                else "not_exposed"
            ),
        },
        "native_quality": observed["_hedonic_native_quality"],
        "algorithm_identity": observed["_hedonic_algorithm_identity"],
        "native_observability": {
            key.removeprefix("_hedonic_"): _observation(value)
            for key, value in observed.items()
        },
    }


def _audit_measure(
    graph: ig.Graph,
    memberships: Sequence[Sequence[int]],
    *,
    cap: int,
    gamma: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = audit_cover(
        graph,
        memberships,
        max_memberships=int(cap),
        allow_isolation=True,
        gamma=float(gamma),
        compute_intervals=False,
    )
    return {"result": result, "runtime_seconds": time.perf_counter() - started}


def _score_measure(graph: ig.Graph, memberships: Sequence[Sequence[int]]) -> dict[str, Any]:
    started = time.perf_counter()
    predicted_cover = [
        [vertex for vertex, row in enumerate(memberships) if label in row]
        for label in sorted({int(label) for row in memberships for label in row})
    ]
    metrics = evaluate_cover(predicted_cover, predicted_cover, graph.vcount(), compute_omega=False)
    return {"metrics": metrics, "runtime_seconds": time.perf_counter() - started}


def _stage_failure(record: dict[str, Any], stage: str, outcome: Any) -> dict[str, Any]:
    status = str(outcome.status)
    record["status"] = f"{stage}_{status}"
    record["error"] = outcome.error or f"{stage} worker ended with status {status}"
    record.setdefault("failure_stage", stage)
    if outcome.peak_rss_bytes is not None:
        memory = record.setdefault("memory", {})
        memory.setdefault("peak_rss_by_stage", {})[stage] = int(outcome.peak_rss_bytes)
        values = [value for value in memory["peak_rss_by_stage"].values() if value is not None]
        memory["peak_rss_bytes"] = max(values, default=None)
    if status == "timeout":
        record.setdefault("capacity", {})["status"] = "timeout"
    return record


def measure_resource_case(
    graph: ig.Graph,
    memberships: Sequence[Sequence[int]],
    *,
    cap: int,
    method: str,
    seed: int = 0,
    gamma: float | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int | None = None,
    max_token_edges: int | None = None,
    max_estimated_bytes: int | None = None,
    max_graph_vertices: int | None = None,
    max_graph_edges: int | None = None,
) -> dict[str, Any]:
    """Measure one case with bounded detector, audit, and scoring stages.

    Worker RSS is post-hoc ``RUSAGE_SELF.ru_maxrss`` from each spawned child;
    it is not a live parent RSS sample.  A killed timeout has no trustworthy
    post-hoc RSS and is recorded as unavailable.  Native projection and token
    counters are ``not_exposed`` when the installed binding does not expose
    them; missing instrumentation is never converted to zero.
    """
    from hedonic.experiments.overlapping.execution import run_in_subprocess

    if str(method) not in {"local", "multiphase", "audit"}:
        raise ValueError("method must be local, multiphase, or audit")
    if int(cap) < 1:
        raise ValueError("cap must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    if memory_limit_bytes is not None and int(memory_limit_bytes) <= 0:
        raise ValueError("memory_limit_bytes must be positive or None")
    selected_gamma = float(graph.density() if gamma is None else gamma)
    estimate = estimate_token_edges(graph, memberships, cap)
    memory: dict[str, Any] = {
        "peak_rss_bytes": None,
        "peak_rss_by_stage": {},
        "limit_bytes": memory_limit_bytes,
        "rss_source": "spawned-child-RUSAGE_SELF.ru_maxrss",
        "rss_scope": "worker_peak_only; not live parent RSS",
        "rss_status": "not_started",
    }
    projection = {
        "events": None,
        "accepted": None,
        "rejected": None,
        "events_status": "not_exposed",
        "accepted_status": "not_exposed",
        "rejected_status": "not_exposed",
    }
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "graph_hash": graph_sha256(graph),
        "n": int(graph.vcount()),
        "m": int(graph.ecount()),
        "cap": int(cap),
        "method": str(method),
        "seed": int(seed),
        "gamma": selected_gamma,
        "timings": {"detector": None, "audit": None, "scoring": None, "end_to_end": None},
        "memory": memory,
        "capacity": {
            "status": "within_integer_bound",
            "preflight_reasons": [],
            "token_graph_preflight": estimate["preflight"],
        },
        "resources": estimate,
        "work": {
            "per_level": None,
            "vertex_visits": None,
            "commits": None,
            "sweeps": None,
            "status": "not_exposed",
        },
        "projection": projection,
        "native_observability": {},
        "status": "pending",
        "error": None,
    }
    started_all = time.perf_counter()
    reasons: list[str] = []
    if bool(estimate["preflight"].get("integer_overflow")):
        reasons.append("integer_overflow")
    if int(estimate["original_max_membership"]) > int(cap):
        reasons.append("initial_membership_exceeds_cap")
    if max_token_edges is not None and int(estimate["token_edges_cap_upper_bound"]) > int(max_token_edges):
        reasons.append("token_edge_bound_exceeds_limit")
    if max_estimated_bytes is not None and int(estimate["estimated_native_bytes"]) > int(max_estimated_bytes):
        reasons.append("estimated_native_bytes_exceeds_limit")
    if memory_limit_bytes is not None and int(estimate["estimated_native_bytes"]) > int(memory_limit_bytes):
        reasons.append("estimated_native_bytes_exceeds_memory_limit")
    if max_graph_vertices is not None and graph.vcount() > int(max_graph_vertices):
        reasons.append("vertex_count_exceeds_limit")
    if max_graph_edges is not None and graph.ecount() > int(max_graph_edges):
        reasons.append("edge_count_exceeds_limit")
    if reasons:
        record["capacity"] = {**record["capacity"], "status": "capacity_refused", "preflight_reasons": reasons}
        record["status"] = "capacity_refused"
        record["memory"]["rss_status"] = "not_started_capacity_refusal"
        record["timings"]["end_to_end"] = time.perf_counter() - started_all
        return record

    if method == "audit":
        detector = {
            "status": "not_run",
            "final_memberships": [list(map(int, row)) for row in memberships],
            "runtime_seconds": 0.0,
        }
    else:
        outcome = run_in_subprocess(
            _native_measure,
            graph,
            [list(map(int, row)) for row in memberships],
            cap=int(cap),
            gamma=selected_gamma,
            method="local" if method == "local" else "multiphase",
            seed=int(seed),
            timeout_seconds=timeout_seconds,
        )
        detector = {
            "status": "completed" if outcome.status == "ok" else outcome.status,
            "error": outcome.error,
            "runtime_seconds": float(outcome.runtime_seconds),
            "peak_rss_bytes": outcome.peak_rss_bytes,
            **(outcome.payload if isinstance(outcome.payload, dict) else {}),
        }
        if outcome.peak_rss_bytes is not None:
            memory["peak_rss_by_stage"]["detector"] = int(outcome.peak_rss_bytes)
            memory["peak_rss_bytes"] = int(outcome.peak_rss_bytes)
        memory["rss_status"] = "observed" if outcome.peak_rss_bytes is not None else "unavailable"
        if detector.get("status") not in {"completed", "not_run", "ok"}:
            record["status"] = str(detector.get("status"))
            record["error"] = detector.get("error")
            record["capacity"]["status"] = "timeout" if detector.get("status") == "timeout" else "detector_failed"
            record["timings"]["detector"] = float(detector.get("runtime_seconds", 0.0))
            record["timings"]["end_to_end"] = time.perf_counter() - started_all
            return record

    record["timings"]["detector"] = float(detector.get("runtime_seconds", 0.0))
    if detector.get("native_observability"):
        record["native_observability"] = detector["native_observability"]
    if detector.get("work"):
        record["work"] = detector["work"]
    final_rows = detector.get("final_memberships") or [list(map(int, row)) for row in memberships]
    if any(len(row) > int(cap) for row in final_rows):
        record["status"] = "capacity_exceeded_returned"
        record["capacity"]["status"] = "capacity_exceeded_returned"
        record["capacity"]["observed_max_membership"] = max((len(row) for row in final_rows), default=0)
        record["timings"]["end_to_end"] = time.perf_counter() - started_all
        return record

    audit_outcome = run_in_subprocess(
        _audit_measure,
        graph,
        final_rows,
        cap=int(cap),
        gamma=selected_gamma,
        timeout_seconds=timeout_seconds,
    )
    record["timings"]["audit"] = float(audit_outcome.runtime_seconds)
    if audit_outcome.peak_rss_bytes is not None:
        memory["peak_rss_by_stage"]["audit"] = int(audit_outcome.peak_rss_bytes)
        memory["peak_rss_bytes"] = max(int(memory["peak_rss_bytes"] or 0), int(audit_outcome.peak_rss_bytes))
    if audit_outcome.status != "ok":
        _stage_failure(record, "audit", audit_outcome)
        record["timings"]["end_to_end"] = time.perf_counter() - started_all
        return record
    audit_result = (audit_outcome.payload or {}).get("result", {})

    score_outcome = run_in_subprocess(
        _score_measure,
        graph,
        final_rows,
        timeout_seconds=timeout_seconds,
    )
    record["timings"]["scoring"] = float(score_outcome.runtime_seconds)
    if score_outcome.peak_rss_bytes is not None:
        memory["peak_rss_by_stage"]["scoring"] = int(score_outcome.peak_rss_bytes)
        memory["peak_rss_bytes"] = max(int(memory["peak_rss_bytes"] or 0), int(score_outcome.peak_rss_bytes))
    if score_outcome.status != "ok":
        _stage_failure(record, "scoring", score_outcome)
        record["timings"]["end_to_end"] = time.perf_counter() - started_all
        return record

    memory["rss_status"] = "observed" if memory["peak_rss_bytes"] is not None else memory["rss_status"]
    if memory_limit_bytes is not None and memory["peak_rss_bytes"] is not None and int(memory["peak_rss_bytes"]) > int(memory_limit_bytes):
        record["status"] = "memory_limit_exceeded"
        record["error"] = f"worker peak RSS {int(memory['peak_rss_bytes'])} exceeds limit {int(memory_limit_bytes)}"
        record["capacity"]["status"] = "memory_limit_exceeded"
        record["timings"]["end_to_end"] = time.perf_counter() - started_all
        return record

    score_result = (score_outcome.payload or {}).get("metrics", {})
    record["metrics"] = score_result
    record["robustness"] = {
        "fixed_profile": audit_result.get("stable_fraction_at_resolution"),
        "mean_positive_regret": audit_result.get("mean_positive_regret_at_resolution"),
        "max_positive_regret": audit_result.get("max_positive_regret_at_resolution"),
    }
    record["resources"]["final_membership_trace"] = membership_incidence_trace(final_rows)
    record["resources"]["actual_token_preflight"] = detector.get("native_token_preflight")
    native_preflight = detector.get("native_token_preflight")
    if isinstance(native_preflight, dict):
        for key in ("token_edges", "token_edge_count", "n_token_edges"):
            if native_preflight.get(key) is not None:
                record["resources"]["token_edges_observed"] = int(native_preflight[key])
                record["resources"]["token_edges_observed_status"] = "observed"
                break
    if "token_edges_observed" not in record["resources"]:
        record["resources"]["token_edges_observed"] = None
        record["resources"]["token_edges_observed_status"] = "not_exposed"
    for output_key, source_key in (("events", "projection_events"), ("accepted", "accepted_moves")):
        value = detector.get(source_key)
        record["projection"][output_key] = value
        record["projection"][f"{output_key}_status"] = "observed" if value is not None else "not_exposed"
    record["projection"]["start_trace"] = detector.get("start_membership_trace")
    record["projection"]["returned_trace"] = detector.get("returned_membership_trace")
    record["capacity"]["status"] = "within_integer_bound"
    record["status"] = "completed"
    record["timings"]["end_to_end"] = time.perf_counter() - started_all
    return record


def fit_resource_models(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit descriptive slopes and robust stage/resource summaries."""
    completed = [row for row in rows if row.get("status") == "completed"]
    terms: dict[str, list[float]] = {"mM": [], "nq": [], "T": [], "Ktok": []}
    detector_times: list[float] = []
    end_to_end_times: list[float] = []
    rss_values: list[float] = []
    for row in completed:
        resources = row.get("resources") or {}
        n, m, cap = float(row.get("n", 0)), float(row.get("m", 0)), float(row.get("cap", 1))
        k = resources.get("token_edges_incidence_upper_bound")
        if k is None or float(k) <= 0:
            k = resources.get("token_edges_cap_upper_bound")
        for key, value in {"mM": m * cap, "nq": n * cap, "T": resources.get("original_unique_incidences", 0), "Ktok": k}.items():
            terms[key].append(max(float(value or 0), 1.0))
        timings = row.get("timings") or {}
        detector_times.append(max(float(timings.get("detector") or 0.0), 1e-9))
        end_to_end_times.append(max(float(timings.get("end_to_end") or 0.0), 1e-9))
        rss = (row.get("memory") or {}).get("peak_rss_bytes")
        if rss is not None:
            rss_values.append(float(rss))
    slopes: dict[str, float | None] = {}
    for key, values in terms.items():
        if len(values) < 2 or len(set(values)) < 2:
            slopes[key] = None
            continue
        slopes[key] = float(np.polyfit(np.log(values), np.log(detector_times), 1)[0])

    def percentile(values: Sequence[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        if low == high:
            return float(ordered[low])
        return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))

    return {
        "independent_unit": "graph case",
        "n_completed": len(completed),
        "descriptive_log_log_slopes": slopes,
        "stage_seconds": {
            "detector_median": statistics.median(detector_times) if detector_times else None,
            "end_to_end_median": statistics.median(end_to_end_times) if end_to_end_times else None,
            "end_to_end_p95": percentile(end_to_end_times, 0.95),
        },
        "rss_bytes": {"max": max(rss_values, default=None), "p95": percentile(rss_values, 0.95)},
        "warning": "descriptive slopes; censored failures are retained but not treated as zero time",
    }


def estimate_full_run(rows: Sequence[dict[str, Any]], *, target_cases: int, effective_workers: int = 1) -> dict[str, Any]:
    """Estimate full-grid wall time from observed bounded case durations."""
    if int(target_cases) < 0:
        raise ValueError("target_cases must be non-negative")
    workers = max(1, int(effective_workers))
    durations = [
        float((row.get("timings") or {}).get("end_to_end"))
        for row in rows
        if row.get("status") == "completed" and float((row.get("timings") or {}).get("end_to_end") or 0.0) > 0
    ]
    rss = [
        float((row.get("memory") or {}).get("peak_rss_bytes"))
        for row in rows
        if row.get("memory", {}).get("peak_rss_bytes") is not None
    ]
    median_seconds = statistics.median(durations) if durations else None
    eta_seconds = None if median_seconds is None else float(target_cases) * median_seconds / workers
    return {
        "target_case_count": int(target_cases),
        "observed_completed_cases": len(durations),
        "effective_workers": workers,
        "observed_case_seconds": {
            "median": median_seconds,
            "p95": float(np.percentile(np.asarray(durations), 95)) if durations else None,
        },
        "throughput_cases_per_second": None if median_seconds in (None, 0) else float(workers / median_seconds),
        "estimated_wall_seconds": eta_seconds,
        "estimated_wall_hours": None if eta_seconds is None else eta_seconds / 3600.0,
        "rss_bytes": {
            "max_observed": max(rss, default=None),
            "p95_observed": float(np.percentile(np.asarray(rss), 95)) if rss else None,
        },
        "status": "measured_pilot_basis" if durations else "unavailable_no_completed_cases",
        "warning": "ETA is extrapolated from bounded observed cases; it is not measured full-grid wall time",
    }


def _profile_sizes(profile: str, sizes: Sequence[int] | None) -> tuple[int, ...]:
    if sizes is not None:
        selected = tuple(int(value) for value in sizes)
    elif profile == "smoke":
        selected = DEFAULT_SMOKE_SIZES
    elif profile == "pilot":
        selected = DEFAULT_PILOT_SIZES
    else:
        selected = DEFAULT_SIZES
    if not selected or any(value < 2 for value in selected):
        raise ValueError("sizes must contain positive graph sizes >= 2")
    return selected


def _build_case_plan(*, profile: str, sizes: Sequence[int], caps: Sequence[int], methods: Sequence[str], optimizer_seeds: Sequence[int], max_cases: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full and selected deterministic plans."""
    full: list[dict[str, Any]] = []
    for size_index, n in enumerate(sizes):
        axis = "dense_overlap_stress" if profile == "smoke" and int(n) == max(sizes) else "sparse_scaling"
        graph_seed = 30_000 + size_index
        for cap in caps:
            for method in methods:
                for optimizer_seed in optimizer_seeds:
                    identity = {
                        "profile": profile,
                        "size_index": size_index,
                        "n": int(n),
                        "graph_seed": graph_seed,
                        "cap": int(cap),
                        "method": str(method),
                        "optimizer_seed": int(optimizer_seed),
                    }
                    full.append({**identity, "axis": axis, "case_key": _json_hash(identity), "case_index": len(full)})
    selected = full if max_cases is None else full[: max(0, int(max_cases))]
    return full, selected


def _generator_kwargs(profile: str, n: int, graph_seed: int) -> dict[str, Any]:
    dense = profile == "smoke" and int(n) >= max(DEFAULT_SMOKE_SIZES)
    return {
        "n": int(n),
        "average_degree": min(20, max(2, int(n) - 1)),
        "max_degree": min(100, max(2, int(n) - 1)),
        "min_community": max(2, min(20, int(n) // 4)),
        "max_community": max(2, min(100, int(n) // 2)),
        "mixing": 0.3,
        "overlap_fraction": 0.5 if dense else 0.1,
        "overlap_multiplicity": 4 if profile in {"smoke", "standard"} else 2,
        "seed": int(graph_seed),
    }


def _case_record_path(output: Path, case: dict[str, Any]) -> Path:
    return output / RECORDS_DIRNAME / f"{int(case['case_index']):05d}-{case['case_key'][:16]}.json"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "case_index", "case_key", "axis", "graph_seed", "optimizer_seed", "graph_hash", "n", "m", "cap", "method", "status", "capacity_status", "refusal_reasons", "peak_rss_bytes", "detector_seconds", "audit_seconds", "scoring_seconds", "end_to_end_seconds", "original_incidences", "original_unique_incidences", "token_edges_incidence_upper_bound", "token_edges_cap_upper_bound", "token_edges_observed", "projection_events", "accepted_moves", "error",
    ]
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        resources, timings = row.get("resources") or {}, row.get("timings") or {}
        memory, capacity, projection = row.get("memory") or {}, row.get("capacity") or {}, row.get("projection") or {}
        output_rows.append({
            **row,
            "capacity_status": capacity.get("status"),
            "refusal_reasons": ";".join(map(str, capacity.get("preflight_reasons", []))),
            "peak_rss_bytes": memory.get("peak_rss_bytes"),
            "detector_seconds": timings.get("detector"),
            "audit_seconds": timings.get("audit"),
            "scoring_seconds": timings.get("scoring"),
            "end_to_end_seconds": timings.get("end_to_end"),
            "original_incidences": resources.get("original_incidences"),
            "original_unique_incidences": resources.get("original_unique_incidences"),
            "token_edges_incidence_upper_bound": resources.get("token_edges_incidence_upper_bound"),
            "token_edges_cap_upper_bound": resources.get("token_edges_cap_upper_bound"),
            "token_edges_observed": resources.get("token_edges_observed"),
            "projection_events": projection.get("events"),
            "accepted_moves": projection.get("accepted"),
        })
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}-{time.time_ns()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _load_shards(
    output: Path,
    *,
    plan: Sequence[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Reload committed shards and reject tampered/out-of-plan records.

    A resumable run must not treat any JSON file under ``records/`` as a
    completed case.  Validate the stable plan identity and protocol fields
    before accepting a shard; otherwise a hand-edited or stale shard could
    silently alter a resource envelope while preserving the config
    fingerprint.
    """
    records_dir = output / RECORDS_DIRNAME
    if not records_dir.exists():
        return {}
    expected = {str(case["case_key"]): case for case in (plan or ())}
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(records_dir.glob("*.json")):
        row = _load_json(path)
        key = str(row.get("case_key", ""))
        if not key:
            raise ValueError(f"record missing case_key: {path}")
        if expected:
            case = expected.get(key)
            if case is None:
                raise ValueError(f"record case_key is outside the requested plan: {key}")
            for field in ("case_index", "axis", "graph_seed", "optimizer_seed", "n", "cap", "method"):
                if str(row.get(field)) != str(case.get(field)):
                    raise ValueError(
                        f"record {path} field {field} does not match requested plan"
                    )
        if int(row.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"record has incompatible schema_version: {path}")
        if str(row.get("protocol_version", "")) != PROTOCOL_VERSION:
            raise ValueError(f"record has incompatible protocol_version: {path}")
        if key in rows:
            raise ValueError(f"duplicate case_key in record shards: {key}")
        rows[key] = row
    return rows


def _config_fingerprint(config: dict[str, Any]) -> str:
    # The fingerprint binds scientific/provenance inputs while deliberately
    # excluding run-local fields (PID, cwd, wall timestamp, and an optional
    # bounded ``max_cases`` truncation).  This lets an interrupted bounded
    # pilot resume into the same registered plan without treating its changed
    # completion count as a protocol drift.
    ignored = {
        "max_cases",
        "resume",
        "launch_command",
        "output_dir",
        "created_utc",
        "selected_case_count",
        "environment",
    }
    return _json_hash({key: value for key, value in config.items() if key not in ignored})


def _write_progress(output: Path, *, status: str, phase: str, active_item: str | None, completed: int, total: int, started_monotonic: float, last_row: dict[str, Any] | None = None) -> None:
    elapsed = max(0.0, time.perf_counter() - started_monotonic)
    rate = completed / elapsed if completed and elapsed > 0 else None
    remaining = max(0, total - completed)
    eta = None if rate in (None, 0) else remaining / rate
    memory = (last_row or {}).get("memory") or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "phase": phase,
        "active_item": active_item,
        "completed_units": int(completed),
        "total_units": int(total),
        "fraction": (completed / total if total else 1.0),
        "elapsed_seconds": elapsed,
        "rate_cases_per_second": rate,
        "eta_seconds": eta,
        "updated_utc": _utc_now(),
        "pid": os.getpid(),
        "resource_metrics": {
            "last_status": (last_row or {}).get("status"),
            "last_peak_rss_bytes": memory.get("peak_rss_bytes"),
            "last_end_to_end_seconds": ((last_row or {}).get("timings") or {}).get("end_to_end"),
        },
    }
    _atomic_json(output / "progress.json", payload)


def _failed_generation_record(case: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "case_index": case["case_index"],
        "case_key": case["case_key"],
        "axis": case["axis"],
        "graph_seed": case["graph_seed"],
        "optimizer_seed": case["optimizer_seed"],
        "n": case["n"],
        "m": None,
        "cap": case["cap"],
        "method": case["method"],
        "seed": case["optimizer_seed"],
        "status": "generation_failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "timings": {"detector": None, "audit": None, "scoring": None, "end_to_end": None},
        "memory": {"peak_rss_bytes": None, "rss_status": "not_started"},
        "capacity": {"status": "generation_failed", "preflight_reasons": []},
        "resources": {},
        "projection": {},
    }


def _base_config(*, output_dir: Path, profile: str, sizes: Sequence[int], caps: Sequence[int], methods: Sequence[str], optimizer_seeds: Sequence[int], max_cases: int | None, timeout_seconds: float | None, memory_limit_bytes: int | None, max_token_edges: int | None, max_estimated_bytes: int | None, max_graph_vertices: int | None, max_graph_edges: int | None, config_path: str | Path | None, launch_command: Sequence[str] | None, planned_case_count: int, selected_case_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "output_dir": str(output_dir),
        "profile": profile,
        "graph_generator_policy": "compatibility_smoke_fixture; official TKT-11 graphs required for canonical evidence",
        "sizes": list(map(int, sizes)),
        "caps": list(map(int, caps)),
        "methods": list(map(str, methods)),
        "optimizer_seeds": list(map(int, optimizer_seeds)),
        "max_cases": max_cases,
        "timeout_seconds": timeout_seconds,
        "memory_limit_bytes": memory_limit_bytes,
        "max_token_edges": max_token_edges,
        "max_estimated_bytes": max_estimated_bytes,
        "max_graph_vertices": max_graph_vertices,
        "max_graph_edges": max_graph_edges,
        "sparse_scaling_and_dense_stress_separate": True,
        "capacity_policy": "typed preflight refusals, timeout, memory, generation, detector, audit, and scoring outcomes retained",
        "planned_case_count": int(planned_case_count),
        "selected_case_count": int(selected_case_count),
        "registered_grid": {"sizes": list(DEFAULT_SIZES), "caps": list(map(int, caps)), "methods": list(map(str, methods)), "optimizer_seeds": list(map(int, optimizer_seeds))},
        "config_path": None if config_path is None else str(expand_path(config_path)),
        "launch_command": None if launch_command is None else list(map(str, launch_command)),
        "created_utc": _utc_now(),
    }


def run_resource_study(
    *,
    output_dir: str | Path,
    profile: str = "smoke",
    sizes: Sequence[int] | None = None,
    caps: Sequence[int] = DEFAULT_CAPS,
    methods: Sequence[str] = DEFAULT_METHODS,
    optimizer_seeds: Sequence[int] = DEFAULT_OPTIMIZER_SEEDS,
    max_cases: int | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int | None = None,
    max_token_edges: int | None = None,
    max_estimated_bytes: int | None = None,
    max_graph_vertices: int | None = None,
    max_graph_edges: int | None = None,
    resume: bool = False,
    config_path: str | Path | None = None,
    launch_command: Sequence[str] | None = None,
    interrupt_after_cases: int | None = None,
) -> dict[str, Any]:
    """Run a bounded or standard grid with atomic case shards and reload."""
    profile = str(profile).lower()
    if profile not in {"smoke", "pilot", "standard"}:
        raise ValueError("profile must be smoke, pilot, or standard")
    selected_sizes = _profile_sizes(profile, sizes)
    selected_caps = tuple(int(value) for value in caps)
    selected_methods = tuple(str(value) for value in methods)
    selected_seeds = tuple(int(value) for value in optimizer_seeds)
    if not selected_caps or any(value < 1 for value in selected_caps):
        raise ValueError("caps must contain positive integers")
    if not selected_methods or any(value not in {"local", "multiphase", "audit"} for value in selected_methods):
        raise ValueError("methods must be drawn from local, multiphase, audit")
    if not selected_seeds:
        raise ValueError("optimizer_seeds must not be empty")
    if interrupt_after_cases is not None and int(interrupt_after_cases) < 1:
        raise ValueError("interrupt_after_cases must be positive when supplied")
    full_plan, selected_plan = _build_case_plan(
        profile=profile,
        sizes=selected_sizes,
        caps=selected_caps,
        methods=selected_methods,
        optimizer_seeds=selected_seeds,
        max_cases=max_cases,
    )
    output = expand_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _base_config(
        output_dir=output,
        profile=profile,
        sizes=selected_sizes,
        caps=selected_caps,
        methods=selected_methods,
        optimizer_seeds=selected_seeds,
        max_cases=max_cases,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        max_token_edges=max_token_edges,
        max_estimated_bytes=max_estimated_bytes,
        max_graph_vertices=max_graph_vertices,
        max_graph_edges=max_graph_edges,
        config_path=config_path,
        launch_command=launch_command,
        planned_case_count=len(full_plan),
        selected_case_count=len(selected_plan),
    )
    config["source"] = _source_receipt(config_path)
    config["environment"] = _runtime_receipt()
    config["dependencies"] = _dependency_closure()
    config["config_fingerprint"] = _config_fingerprint(config)
    existing_config = output / "config.json"
    if existing_config.exists():
        prior = _load_json(existing_config)
        if not resume:
            raise ValueError(f"output already contains config; pass resume=True: {output}")
        if prior.get("config_fingerprint") != config["config_fingerprint"]:
            raise ValueError("resume config fingerprint differs; use a new output root")
    elif resume and (output / RECORDS_DIRNAME).exists() and any((output / RECORDS_DIRNAME).glob("*.json")):
        raise ValueError("cannot resume shards without a prior config.json")
    _atomic_json(existing_config, config)
    shards = _load_shards(output, plan=full_plan) if resume else {}
    unknown_keys = set(shards) - {str(case["case_key"]) for case in full_plan}
    if unknown_keys:
        raise ValueError("resume output contains shards outside the requested plan")
    started = time.perf_counter()
    completed_count = sum(1 for case in selected_plan if case["case_key"] in shards)
    _write_progress(
        output,
        status="running" if completed_count < len(selected_plan) else "resuming",
        phase="collection",
        active_item=None,
        completed=completed_count,
        total=len(selected_plan),
        started_monotonic=started,
    )
    graph_cache: dict[int, Any] = {}
    newly_completed = 0
    for case in selected_plan:
        key = str(case["case_key"])
        if key in shards:
            continue
        try:
            graph_seed = int(case["graph_seed"])
            if graph_seed not in graph_cache:
                graph_cache[graph_seed] = generate_overlapping_lfr(
                    **_generator_kwargs(profile, int(case["n"]), graph_seed)
                )
            instance = graph_cache[graph_seed]
            memberships = cover_to_vertex_memberships(
                instance.cover,
                instance.graph.vcount(),
                require_covered=True,
            )
            row = measure_resource_case(
                instance.graph,
                memberships,
                cap=int(case["cap"]),
                method=str(case["method"]),
                seed=int(case["optimizer_seed"]),
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
                max_token_edges=max_token_edges,
                max_estimated_bytes=max_estimated_bytes,
                max_graph_vertices=max_graph_vertices,
                max_graph_edges=max_graph_edges,
            )
            row.update(
                {
                    "case_index": int(case["case_index"]),
                    "case_key": key,
                    "axis": case["axis"],
                    "graph_seed": int(case["graph_seed"]),
                    "optimizer_seed": int(case["optimizer_seed"]),
                    "generator_metadata": dict(instance.metadata),
                }
            )
        except BaseException as exc:
            row = _failed_generation_record(case, exc)
        shard_path = _case_record_path(output, case)
        _atomic_json(shard_path, row)
        shards[key] = row
        newly_completed += 1
        completed_count += 1
        _write_progress(
            output,
            status="running",
            phase="case_execution",
            active_item=key,
            completed=completed_count,
            total=len(selected_plan),
            started_monotonic=started,
            last_row=row,
        )
        if interrupt_after_cases is not None and newly_completed >= int(interrupt_after_cases):
            _write_progress(
                output,
                status="interrupted",
                phase="case_execution",
                active_item=key,
                completed=completed_count,
                total=len(selected_plan),
                started_monotonic=started,
                last_row=row,
            )
            raise ResourceEnvelopeInterrupted("intentional interruption after committed case shard")

    rows = [shards[str(case["case_key"])] for case in full_plan if str(case["case_key"]) in shards]
    rows.sort(key=lambda row: int(row.get("case_index", 0)))
    _write_jsonl(output / "results.jsonl", rows)
    _write_csv(output / "results.csv", rows)
    models = fit_resource_models(rows)
    registered_target = len(DEFAULT_SIZES) * len(selected_caps) * len(selected_methods) * len(selected_seeds)
    estimate = estimate_full_run(rows, target_cases=registered_target, effective_workers=1)
    status_counts = {
        status: sum(str(row.get("status")) == status for row in rows)
        for status in sorted({str(row.get("status")) for row in rows})
    }
    complete_selected = len(rows) >= len(selected_plan)
    truncated = len(selected_plan) < len(full_plan)
    if profile == "standard" and complete_selected and not truncated:
        terminal_status = "complete"
    elif profile == "pilot":
        terminal_status = "pilot_complete" if complete_selected else "pilot_incomplete"
    else:
        terminal_status = "bounded_complete" if complete_selected else "bounded_incomplete"
    manifest = {
        **config,
        "status": terminal_status,
        "case_count": len(rows),
        "status_counts": status_counts,
        "coverage": {
            "planned_case_count": len(full_plan),
            "selected_case_count": len(selected_plan),
            "committed_case_count": len(rows),
            "truncated_by_max_cases": truncated,
            "missing_case_count": max(0, len(selected_plan) - len(rows)),
            "censored_failures_retained": True,
        },
        "resource_models": models,
        "full_run_estimate": estimate,
        "terminal_reload": False,
        "finished_utc": _utc_now(),
    }
    _atomic_json(output / "resource_models.json", models)
    _atomic_json(output / "manifest.json", manifest)
    reloaded = _load_json(output / "manifest.json")
    if reloaded.get("case_count") != len(rows) or reloaded.get("status") != terminal_status:
        raise RuntimeError("terminal manifest reload did not preserve case count/status")
    manifest["terminal_reload"] = True
    _atomic_json(output / "manifest.json", manifest)
    _write_progress(
        output,
        status=terminal_status,
        phase="terminal_verification",
        active_item=None,
        completed=len(rows),
        total=len(selected_plan),
        started_monotonic=started,
    )
    return manifest


def _load_resource_section(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}
    path = expand_path(config_path)
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"resource config is not valid TOML: {path}") from exc
    section = payload.get("resource_envelope", {})
    if not isinstance(section, dict):
        raise ValueError("[resource_envelope] must be a table")
    return dict(section)


def _csv_values(value: Any, cast: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(cast(item) for item in value)
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


def preflight_resource_study(
    *,
    output_dir: str | Path,
    profile: str = "standard",
    sizes: Sequence[int] | None = None,
    caps: Sequence[int] = DEFAULT_CAPS,
    methods: Sequence[str] = DEFAULT_METHODS,
    optimizer_seeds: Sequence[int] = DEFAULT_OPTIMIZER_SEEDS,
    max_cases: int | None = None,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    memory_limit_bytes: int | None = None,
    max_token_edges: int | None = None,
    max_estimated_bytes: int | None = None,
    max_graph_vertices: int | None = None,
    max_graph_edges: int | None = None,
    config_path: str | Path | None = None,
    launch_command: Sequence[str] | None = None,
    pilot_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fail-closed launch receipt without running production cases."""
    profile = str(profile).lower()
    selected_sizes = _profile_sizes(profile, sizes)
    full_plan, selected_plan = _build_case_plan(
        profile=profile,
        sizes=selected_sizes,
        caps=tuple(caps),
        methods=tuple(methods),
        optimizer_seeds=tuple(optimizer_seeds),
        max_cases=max_cases,
    )
    output = expand_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dependencies = _dependency_closure()
    fixture: dict[str, Any]
    if not dependencies["complete"]:
        fixture = {"status": "not_run_missing_dependency", "missing": dependencies["missing"]}
    else:
        try:
            instance = generate_overlapping_lfr(
                n=16,
                average_degree=3,
                max_degree=8,
                min_community=4,
                max_community=8,
                mixing=0.3,
                overlap_fraction=0.25,
                overlap_multiplicity=2,
                seed=91_001,
            )
            memberships = cover_to_vertex_memberships(instance.cover, 16, require_covered=True)
            fixture_row = measure_resource_case(
                instance.graph,
                memberships,
                cap=2,
                method="local",
                seed=91_002,
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
                max_token_edges=max_token_edges,
                max_estimated_bytes=max_estimated_bytes,
                max_graph_vertices=max_graph_vertices,
                max_graph_edges=max_graph_edges,
            )
            fixture = {
                "status": fixture_row.get("status"),
                "graph_hash": instance.graph_hash,
                "row_status": fixture_row.get("status"),
                "timings": fixture_row.get("timings"),
                "memory": fixture_row.get("memory"),
                "capacity": fixture_row.get("capacity"),
            }
        except BaseException as exc:
            fixture = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
    pilot_estimate = None
    if pilot_receipt is not None:
        pilot_path = expand_path(pilot_receipt)
        pilot = _load_json(pilot_path)
        pilot_estimate = pilot.get("full_run_estimate") or pilot.get("resource_models")
    config = _base_config(
        output_dir=output,
        profile=profile,
        sizes=selected_sizes,
        caps=tuple(caps),
        methods=tuple(methods),
        optimizer_seeds=tuple(optimizer_seeds),
        max_cases=max_cases,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        max_token_edges=max_token_edges,
        max_estimated_bytes=max_estimated_bytes,
        max_graph_vertices=max_graph_vertices,
        max_graph_edges=max_graph_edges,
        config_path=config_path,
        launch_command=launch_command,
        planned_case_count=len(full_plan),
        selected_case_count=len(selected_plan),
    )
    config["source"] = _source_receipt(config_path)
    config["environment"] = _runtime_receipt()
    config["dependencies"] = dependencies
    ready = dependencies["complete"] and fixture.get("status") in {"completed", "capacity_refused"}
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": "ready" if ready else "blocked",
        "production_cases_executed": 0,
        "scope": "dependency closure + disposable end-to-end fixture; no registered case consumed",
        "config": config,
        "config_fingerprint": _config_fingerprint(config),
        "source": config["source"],
        "environment": config["environment"],
        "dependencies": dependencies,
        "fixture": fixture,
        "pilot_receipt": None if pilot_receipt is None else str(expand_path(pilot_receipt)),
        "pilot_estimate": pilot_estimate,
        "planned_case_count": len(full_plan),
        "selected_case_count": len(selected_plan),
        "exact_launch_command": None if launch_command is None else list(map(str, launch_command)),
        "created_utc": _utc_now(),
    }
    _atomic_json(output / "preflight.json", receipt)
    _atomic_json(output / "launch_receipt.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Original/token/audit resource envelope (Astra TKT-14)")
    parser.add_argument("--config", type=Path, default=None, help="optional TOML config; [resource_envelope] values are defaults")
    parser.add_argument("--profile", choices=("smoke", "pilot", "standard"), default=None)
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=None)
    parser.add_argument("--sizes", default=None, help="comma-separated graph sizes")
    parser.add_argument("--caps", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--optimizer-seeds", default=None, help="comma-separated detector seeds")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--max-token-edges", type=int, default=None)
    parser.add_argument("--max-estimated-bytes", type=int, default=None)
    parser.add_argument("--max-graph-vertices", type=int, default=None)
    parser.add_argument("--max-graph-edges", type=int, default=None)
    parser.add_argument("--pilot-receipt", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="resume validated atomic case shards")
    parser.add_argument("--preflight", action="store_true", help="write launch/preflight receipts without registered cases")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    section = _load_resource_section(args.config)
    profile = args.profile or str(section.get("profile", "smoke"))
    output = args.output_dir or section.get("output_dir") or (OVERLAPPING_ARTIFACTS_DIR / "resource_envelope")
    sizes = _csv_values(args.sizes if args.sizes is not None else section.get("sizes"), int)
    caps = _csv_values(args.caps if args.caps is not None else section.get("caps"), int) or DEFAULT_CAPS
    methods = _csv_values(args.methods if args.methods is not None else section.get("methods"), str) or DEFAULT_METHODS
    optimizer_seeds = _csv_values(args.optimizer_seeds if args.optimizer_seeds is not None else section.get("optimizer_seeds"), int) or DEFAULT_OPTIMIZER_SEEDS
    timeout = args.timeout_seconds if args.timeout_seconds is not None else section.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    memory_gb = args.memory_limit_gb
    if memory_gb is None and section.get("memory_limit_gb") is not None:
        memory_gb = float(section["memory_limit_gb"])
    memory = None if memory_gb is None else int(float(memory_gb) * 1024**3)
    max_cases = args.max_cases if args.max_cases is not None else section.get("max_cases")
    launch_command = ["hedonic-exp", "overlapping-resource-envelope", *raw_argv]
    common = {
        "output_dir": output,
        "profile": profile,
        "sizes": sizes,
        "caps": caps,
        "methods": methods,
        "optimizer_seeds": optimizer_seeds,
        "max_cases": max_cases,
        "timeout_seconds": timeout,
        "memory_limit_bytes": memory,
        "max_token_edges": args.max_token_edges,
        "max_estimated_bytes": args.max_estimated_bytes,
        "max_graph_vertices": args.max_graph_vertices,
        "max_graph_edges": args.max_graph_edges,
        "config_path": args.config,
        "launch_command": launch_command,
    }
    if args.preflight:
        receipt = preflight_resource_study(**common, pilot_receipt=args.pilot_receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
        return 0 if receipt["status"] == "ready" else 2
    manifest = run_resource_study(**common, resume=bool(args.resume))
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "DEFAULT_CAPS",
    "DEFAULT_METHODS",
    "DEFAULT_OPTIMIZER_SEEDS",
    "DEFAULT_PILOT_SIZES",
    "DEFAULT_SIZES",
    "DEFAULT_SMOKE_SIZES",
    "ResourceEnvelopeInterrupted",
    "estimate_full_run",
    "estimate_token_edges",
    "fit_resource_models",
    "main",
    "measure_resource_case",
    "preflight_resource_study",
    "run_resource_study",
]
