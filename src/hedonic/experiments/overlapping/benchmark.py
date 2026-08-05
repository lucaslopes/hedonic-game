"""Reproducible multi-network overlapping-community benchmark CLI.

``hedonic-exp overlapping-benchmark`` is intentionally self-contained at the
experiment layer: it loads the five supported SNAP covers, delegates methods
through adapters, writes resumable run records, and emits aggregate tables and
plots.  It never writes inside a raw network directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments.config import OUTPUT_DIR
from hedonic.experiments.overlapping.methods import (
    METHODS,
    effective_resolution,
    method_availability,
    resolve_methods,
    run_method,
    seeded_initial_membership,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    quality_overlapping_cpm,
)
from hedonic.experiments.overlapping.protocol import (
    current_experiment_identity,
    dataset_metadata_identity,
    identity_rejection_reasons,
)
from hedonic.experiments.overlapping.snap import (
    DEFAULT_NETWORKS_DIR,
    SnapDataset,
    SnapLoadError,
    UnsupportedCoverVariant,
    bounded_induced_dataset,
    common_undirected_analysis_dataset,
    load_snap_dataset,
    network_names,
    print_dataset_report,
    smoke_dataset,
)


SCHEMA_VERSION = 2
# Version 6 gives every hedonic method a shared seed-dependent disjoint warm
# start. Singleton-start and warm-start records are different experimental
# conditions and must never share a cache entry.
RUN_PROTOCOL_VERSION = 6
RSS_POLL_SECONDS = 0.05
RSS_RECORD_INTERVAL_SECONDS = 1.0
RESUMABLE_CACHE_STATUSES = {
    "completed",
    "skipped_unsupported",
    "skipped_external_unchanged",
}
RESOURCE_FAILURE_STATUSES = {"memory_limit", "timeout", "oom"}
NON_SCALABLE_BASELINES = {"cpm", "demon"}
SKIPPED_EXTERNAL_STATUS = "skipped_external_unchanged"
PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "smoke": {
        "cover": "all",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": 64,
        "timeout_per_run": 30.0,
        "plots": True,
    },
    "standard": {
        "cover": "top5000",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": 3_000,
        "timeout_per_run": 180.0,
        "plots": True,
    },
    "full": {
        "cover": "top5000",
        "datasets": list(network_names()),
        "methods": [
            "hedonic_multiphase",
            "hedonic_multiphase_x10",
            "hedonic_multiphase_x100",
            "cpm",
            "demon",
        ],
        "seeds": [0],
        "resolutions": ["auto"],
        "max_nodes": None,
        "timeout_per_run": 3_600.0,
        "plots": True,
    },
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _parse_csv(value: str | None, valid: Iterable[str], option: str) -> list[str] | None:
    if value is None:
        return None
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    allowed = set(valid)
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {option}: {', '.join(unknown)}; choose from {', '.join(sorted(allowed))}"
        )
    return values


def parse_seeds(value: str | None) -> list[int] | None:
    """Parse ``0-4`` and comma-list seed syntax without a third-party helper."""
    if value is None:
        return None
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            start_text, stop_text = part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            step = 1 if stop >= start else -1
            values.extend(range(start, stop + step, step))
        else:
            values.append(int(part))
    if not values:
        raise ValueError("--seeds must contain at least one integer")
    return list(dict.fromkeys(values))


def parse_resolutions(value: str | None) -> list[str | float] | None:
    """Parse ``auto``, comma values, or inclusive ``start:stop:count`` grids."""
    if value is None:
        return None
    result: list[str | float] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "auto":
            result.append("auto")
        elif part.count(":") == 2:
            start_text, stop_text, count_text = part.split(":")
            start, stop, count = float(start_text), float(stop_text), int(count_text)
            if count <= 0:
                raise ValueError("resolution range count must be positive")
            if count == 1:
                result.append(start)
            else:
                result.extend(start + (stop - start) * i / (count - 1) for i in range(count))
        else:
            result.append(float(part))
    if not result:
        raise ValueError("--resolutions must contain at least one value")
    return list(dict.fromkeys(result))


def _resolution_label(resolution: float) -> str:
    return f"{resolution:.12g}".replace("-", "m").replace(".", "p")


def _run_path(
    output_dir: Path,
    dataset: str,
    cover: str,
    method: str,
    seed: int,
    resolution: float,
) -> Path:
    return (
        output_dir
        / "runs"
        / dataset
        / cover
        / method
        / f"seed_{seed}"
        / f"resolution_{_resolution_label(resolution)}.json"
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temporary.replace(path)


def _append_log(output_dir: Path, message: str) -> None:
    """Persist concise orchestration events alongside the printed progress."""
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "benchmark.log").open("a", encoding="utf-8") as f:
        f.write(f"{_timestamp()} {message}\n")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _parse_timeout_map(value: Any, field: str) -> dict[str, float]:
    """Parse TOML/CLI timeout maps written as tables or ``name=seconds``."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, str):
        items = (part.split("=", 1) for part in value.split(",") if part.strip())
    else:
        raise ValueError(f"{field} must be a TOML table or comma-list name=seconds")
    result: dict[str, float] = {}
    for item in items:
        try:
            name, seconds = item
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid {field} entry; expected name=positive_seconds") from None
        name = str(name).strip().lower()
        if not name or seconds <= 0:
            raise ValueError(f"Invalid {field} entry; names and seconds must be positive")
        result[name] = seconds
    return result


def _timeout_for(options: dict[str, Any], dataset: str, method: str) -> float:
    """Use the most conservative applicable global, method, and data limit."""
    limits = [float(options["timeout_per_run"])]
    method_limit = options.get("timeout_by_method", {}).get(method)
    dataset_limit = options.get("timeout_by_dataset", {}).get(dataset)
    if method_limit is not None:
        limits.append(float(method_limit))
    if dataset_limit is not None:
        limits.append(float(dataset_limit))
    return min(limits)


def _cache_parameters_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Check the full identity of an experimental condition, excluding outcome."""
    if identity_rejection_reasons(
        existing.get("experiment_identity"), expected.get("experiment_identity", {})
    ):
        return False
    try:
        protocol_version = int(existing.get("protocol_version", -1))
    except (TypeError, ValueError):
        return False
    if protocol_version != RUN_PROTOCOL_VERSION:
        return False
    for key in (
        "dataset",
        "cover",
        "method",
        "seed",
        "max_memberships",
        "allow_isolation",
        "initialization",
    ):
        if existing.get(key) != expected.get(key):
            return False
    for key in ("resolution", "timeout_seconds"):
        try:
            if abs(float(existing.get(key)) - float(expected.get(key))) > 1e-12:
                return False
        except (TypeError, ValueError):
            return False
    if existing.get("memory_limit_bytes") != expected.get("memory_limit_bytes"):
        return False
    if existing.get("dataset_metadata_identity") != expected.get("dataset_metadata_identity"):
        return False
    if existing.get("dataset_report", {}).get("analysis_graph") != expected.get("analysis_graph"):
        return False
    existing_options = existing.get("run_options")
    expected_options = expected.get("run_options")
    if not isinstance(existing_options, dict) or not isinstance(expected_options, dict):
        return False
    for key in ("omega", "omega_sample_size", "external_baseline_policy"):
        if existing_options.get(key) != expected_options.get(key):
            return False
    # The previously shipped coordinator could persist this construction
    # failure after a detector finished.  It is not a detector result, so a
    # post-fix --resume must rerun it while preserving the old JSON as an
    # incompatible-cache backup.
    if "_record() got multiple values for keyword argument 'timeout_seconds'" in str(
        existing.get("error", "")
    ):
        return False
    return True


def _cache_compatible(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Only valid metrics (or an explicit unavailable dependency) can resume."""
    if not _cache_parameters_compatible(existing, expected):
        return False
    status = str(existing.get("status"))
    if status not in RESUMABLE_CACHE_STATUSES:
        return False
    if status == "completed":
        return True
    # This is a capability decision, not a numerical result.  It is safe to
    # retain it only if the record says why the adapter is unavailable.
    return status == "skipped_unsupported" and bool(existing.get("failure_kind") == "unsupported")


def _worker(
    result_path: str,
    method_name: str,
    graph,
    max_memberships: int,
    resolution: float,
    seed: int,
    memory_limit_bytes: int | None,
    initial_membership: list[int] | list[list[int]] | None,
    method_parameters: dict[str, Any] | None,
) -> None:
    """Run one detector and persist its potentially large result off-pipe."""
    # Give the detector and every subprocess it creates a private process
    # group.  Killing only the multiprocessing child is unsafe: NetworkX and
    # external baselines can leave descendants alive after the parent exits.
    try:
        os.setsid()
    except (AttributeError, OSError):  # pragma: no cover - platform-specific
        pass
    memory: dict[str, Any] = {"limit_bytes": memory_limit_bytes}
    try:
        cover, method_meta = run_method(
            METHODS[method_name],
            graph,
            max_memberships=max_memberships,
            resolution=resolution,
            seed=seed,
            parameters=method_parameters,
            initial_membership=initial_membership,
        )
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            child_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            # macOS reports bytes; Linux reports KiB.
            memory["resource_peak_rss_bytes"] = peak if sys.platform == "darwin" else peak * 1024
            memory["resource_children_peak_rss_bytes"] = (
                child_peak if sys.platform == "darwin" else child_peak * 1024
            )
        except (AttributeError, ValueError):  # pragma: no cover - platform-specific
            pass
        _write_worker_packet(
            result_path,
            {"status": "ok", "cover": cover, "method_meta": method_meta, "memory": memory},
        )
    except BaseException as exc:  # child errors must reach a resumable run record
        _write_worker_packet(
            result_path,
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "memory": memory,
            }
        )


def _write_worker_packet(result_path: str, packet: dict[str, Any]) -> None:
    """Atomically write a detector packet without a bounded IPC pipe.

    A full overlapping cover can be hundreds of megabytes. Sending it through
    ``multiprocessing.Queue`` deadlocks when the parent waits for process exit
    before draining the queue: the queue feeder fills its pipe and prevents the
    child from exiting. A private temporary file has no bounded pipe and is
    read only after the child has terminated.
    """
    target = Path(result_path)
    staging = target.with_suffix(target.suffix + ".partial")
    with staging.open("wb") as stream:
        pickle.dump(packet, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(staging, target)


def _process_tree_snapshot(pid: int) -> dict[int, dict[str, int]]:
    """Return ``pid -> {ppid, rss_bytes}`` for the current process table."""
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return {}
    result: dict[int, dict[str, int]] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            child_pid, parent_pid, rss_kib = (int(fields[0]), int(fields[1]), int(fields[2]))
        except ValueError:
            continue
        result[child_pid] = {"ppid": parent_pid, "rss_bytes": max(0, rss_kib) * 1024}
    return result


def _process_tree_pids(pid: int, snapshot: dict[int, dict[str, int]]) -> set[int]:
    """Find the root and all descendants in one process-table snapshot."""
    pids = {pid}
    changed = True
    while changed:
        changed = False
        for child_pid, info in snapshot.items():
            if info.get("ppid") in pids and child_pid not in pids:
                pids.add(child_pid)
                changed = True
    return pids


def _process_tree_rss_bytes(pid: int) -> tuple[int, set[int]]:
    """Sum RSS for an isolated detector and every observable descendant."""
    snapshot = _process_tree_snapshot(pid)
    pids = _process_tree_pids(pid, snapshot)
    return sum(snapshot.get(item, {}).get("rss_bytes", 0) for item in pids), pids


def _process_rss_bytes(pid: int) -> int | None:
    """Backward-compatible root-RSS helper; new enforcement uses process trees."""
    rss, pids = _process_tree_rss_bytes(pid)
    return rss if pid in pids and rss else None


def _terminate_process_tree(process, *, grace_seconds: float = 1.0) -> None:
    """Terminate a detector process group and reap the multiprocessing root."""
    pid = int(process.pid)
    try:
        pgid = os.getpgid(pid)
    except (AttributeError, OSError):
        pgid = None
    private_group = pgid == pid
    if private_group:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
    else:
        # A failed setsid must never turn a kill request into a kill of the
        # coordinator's process group. Best-effort terminate observed
        # descendants individually, then use multiprocessing's root kill.
        _, descendants = _process_tree_rss_bytes(pid)
        for descendant in sorted(descendants - {pid}, reverse=True):
            try:
                os.kill(descendant, signal.SIGTERM)
            except OSError:
                pass
        process.terminate()
    process.join(max(0.05, grace_seconds))
    if process.is_alive():
        if private_group:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        else:
            process.kill() if hasattr(process, "kill") else process.terminate()
        process.join(max(0.05, grace_seconds))


def _classify_exit(exit_code: int | None, *, enforced_reason: str | None = None) -> tuple[str, str]:
    """Classify an exited detector without hiding an OS-level SIGKILL."""
    if enforced_reason:
        return enforced_reason, f"detector terminated by {enforced_reason} enforcement"
    if exit_code == -signal.SIGKILL:
        return "oom", "detector exited with SIGKILL (exit=-9); likely system or allocator OOM"
    if exit_code is None:
        return "failed", "detector subprocess exit code was unavailable"
    return "failed", f"detector subprocess exited without a result (exit={exit_code})"


def _run_with_timeout(
    method_name: str,
    graph,
    *,
    max_memberships: int,
    resolution: float,
    seed: int,
    timeout_seconds: float | None,
    memory_limit_bytes: int | None = None,
    initial_membership: list[int] | list[list[int]] | None = None,
    method_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a detector with process-tree RSS and wall-clock enforcement."""
    if (timeout_seconds is None or timeout_seconds <= 0) and memory_limit_bytes is None:
        try:
            cover, method_meta = run_method(
                METHODS[method_name],
                graph,
                max_memberships=max_memberships,
                resolution=resolution,
                seed=seed,
                parameters=method_parameters,
                initial_membership=initial_membership,
            )
            return {"status": "ok", "cover": cover, "method_meta": method_meta}
        except BaseException as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    # ``fork`` avoids copying bounded igraph inputs through a second pickle on
    # Unix; Windows falls back to spawn. Large results are persisted to a
    # private temporary file because a multiprocessing queue can deadlock once
    # its bounded pipe fills while the parent is waiting for child exit.
    methods = mp.get_all_start_methods()
    context = mp.get_context("fork" if "fork" in methods else "spawn")
    result_dir = tempfile.TemporaryDirectory(prefix="hedonic-detector-")
    result_path = Path(result_dir.name) / "result.pickle"

    def finish(packet: dict[str, Any]) -> dict[str, Any]:
        result_dir.cleanup()
        return packet

    process = context.Process(
        target=_worker,
        args=(
            str(result_path),
            method_name,
            graph,
            max_memberships,
            resolution,
            seed,
            memory_limit_bytes,
            initial_membership,
            method_parameters,
        ),
    )
    started = time.monotonic()
    try:
        process.start()
    except BaseException:
        result_dir.cleanup()
        raise
    peak_rss_bytes = 0
    peak_pids: set[int] = set()
    monitor_samples = 0
    rss_samples: list[dict[str, int | float]] = []
    last_sample_at = -RSS_RECORD_INTERVAL_SECONDS
    termination_reason: str | None = None
    while process.is_alive():
        elapsed = time.monotonic() - started
        rss_bytes, pids = _process_tree_rss_bytes(process.pid)
        monitor_samples += 1
        # Persist a compact, bounded-rate trace.  This gives a reviewer the
        # process-tree observation behind a peak without turning a one-hour
        # run into tens of thousands of JSON entries.
        if elapsed - last_sample_at >= RSS_RECORD_INTERVAL_SECONDS:
            rss_samples.append(
                {
                    "elapsed_seconds": round(elapsed, 6),
                    "rss_bytes": int(rss_bytes),
                    "process_count": len(pids),
                }
            )
            last_sample_at = elapsed
        if rss_bytes:
            if rss_bytes > peak_rss_bytes:
                peak_rss_bytes = rss_bytes
                peak_pids = set(pids)
            if memory_limit_bytes is not None and rss_bytes > memory_limit_bytes:
                termination_reason = "memory_limit"
                _terminate_process_tree(process)
                return finish({
                    "status": "memory_limit",
                    "runtime_seconds": elapsed,
                    "memory": {
                        "limit_bytes": memory_limit_bytes,
                        "enforcement": "process_tree_rss_polling",
                        "observed_peak_rss_bytes": peak_rss_bytes,
                        "peak_process_count": max(1, len(peak_pids)),
                        "monitor_samples": monitor_samples,
                        "monitor_interval_seconds": RSS_POLL_SECONDS,
                        "rss_samples": rss_samples,
                        "termination_reason": termination_reason,
                    },
                    "termination_reason": termination_reason,
                })
        if timeout_seconds is not None and timeout_seconds > 0 and elapsed >= timeout_seconds:
            termination_reason = "timeout"
            _terminate_process_tree(process)
            return finish({
                "status": "timeout",
                "runtime_seconds": elapsed,
                "memory": {
                    "limit_bytes": memory_limit_bytes,
                    "enforcement": "process_tree_rss_polling" if memory_limit_bytes else None,
                    "observed_peak_rss_bytes": peak_rss_bytes,
                    "peak_process_count": max(1, len(peak_pids)),
                    "monitor_samples": monitor_samples,
                    "monitor_interval_seconds": RSS_POLL_SECONDS,
                    "rss_samples": rss_samples,
                    "termination_reason": termination_reason,
                },
                "termination_reason": termination_reason,
            })
        process.join(RSS_POLL_SECONDS)
    elapsed = time.monotonic() - started
    try:
        with result_path.open("rb") as stream:
            packet = pickle.load(stream)
        if not isinstance(packet, dict):
            raise TypeError(f"detector packet must be a dict, got {type(packet).__name__}")
    except Exception as exc:
        status, reason = _classify_exit(process.exitcode)
        return finish({
            "status": status,
            "runtime_seconds": elapsed,
            "error": f"{reason}; result artifact unreadable: {type(exc).__name__}: {exc}",
            "exit_code": process.exitcode,
            "termination_reason": reason,
            "memory": {
                "limit_bytes": memory_limit_bytes,
                "enforcement": "process_tree_rss_polling" if memory_limit_bytes else None,
                "observed_peak_rss_bytes": peak_rss_bytes,
                "peak_process_count": max(1, len(peak_pids)),
                "monitor_samples": monitor_samples,
                "monitor_interval_seconds": RSS_POLL_SECONDS,
                "rss_samples": rss_samples,
            },
        })
    packet["runtime_seconds"] = elapsed
    packet["exit_code"] = process.exitcode
    memory = packet.get("memory")
    if not isinstance(memory, dict):
        memory = {}
        packet["memory"] = memory
    memory.update(
        {
            "limit_bytes": memory_limit_bytes,
            "enforcement": "process_tree_rss_polling" if memory_limit_bytes else None,
            "observed_peak_rss_bytes": max(
                peak_rss_bytes,
                int(memory.get("resource_peak_rss_bytes", 0) or 0),
                int(memory.get("resource_children_peak_rss_bytes", 0) or 0),
            ),
            "peak_process_count": max(1, len(peak_pids)),
            "monitor_samples": monitor_samples,
            "monitor_interval_seconds": RSS_POLL_SECONDS,
            "rss_samples": rss_samples,
            "result_transport": "temporary_pickle_file",
        }
    )
    return finish(packet)


def _record(
    *,
    dataset: str,
    cover: str,
    method: str,
    seed: int,
    resolution: float,
    status: str,
    profile: str,
    dataset_report: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    # Keep identity and outcome fields owned by this constructor.  In
    # particular, ``timeout_seconds`` is the configured limit computed by the
    # coordinator, rather than an incidental value returned by the monitor.
    # This catches future call sites that accidentally try to override a
    # record's identity/status through ``**extra``.
    owned = {
        "schema_version", "protocol_version", "created_at", "dataset", "cover",
        "method", "seed", "resolution", "status", "profile", "dataset_report",
    }
    conflict = owned.intersection(extra)
    if conflict:
        raise ValueError(f"_record extra fields conflict with record fields: {sorted(conflict)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": RUN_PROTOCOL_VERSION,
        "created_at": _timestamp(),
        "dataset": dataset,
        "cover": cover,
        "method": method,
        "seed": seed,
        "resolution": resolution,
        "status": status,
        "profile": profile,
        "dataset_report": dataset_report,
        "dataset_metadata_identity": dataset_metadata_identity(dataset_report),
        "experiment_identity": current_experiment_identity(),
        **extra,
    }


def _completed_records(output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    runs = output_dir / "runs"
    if not runs.is_dir():
        return records
    for path in sorted(runs.rglob("*.json")):
        record = _read_json(path)
        if record is not None:
            record["run_path"] = str(path)
            records.append(record)
    return records


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key not in {"metrics", "dataset_report", "method_metadata", "traceback"}
    }
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        row.update(metrics)
    method_metadata = record.get("method_metadata")
    if isinstance(method_metadata, dict):
        row["method_parameters"] = json.dumps(
            method_metadata.get("parameters", {}), sort_keys=True
        )
    return row


def _write_results(output_dir: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_flatten_record(record) for record in records]
    jsonl = output_dir / "results.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    columns = sorted({key for row in rows for key in row})
    with gzip.open(output_dir / "results.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _summary(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("dataset"), row.get("cover"), row.get("method"), row.get("resolution"), row.get("status"))
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (dataset, cover, method, resolution, status), group in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "dataset": dataset,
            "cover": cover,
            "method": method,
            "resolution": resolution,
            "status": status,
            "n_runs": len(group),
        }
        # Resource and detector failures are diagnostics, never observations.
        # In particular, a timeout duration must not become a mean runtime or
        # a failed record contribute an accidental numeric F1 in a future
        # schema extension.
        if status != "completed":
            summary_rows.append(summary_row)
            continue
        numeric_keys = {
            key
            for row in group
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in group if isinstance(row.get(key), (int, float))]
            if values:
                summary_row[f"mean_{key}"] = sum(values) / len(values)
                summary_row[f"min_{key}"] = min(values)
                summary_row[f"max_{key}"] = max(values)
        summary_rows.append(summary_row)
    return {"schema_version": SCHEMA_VERSION, "groups": summary_rows}, summary_rows


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary, summary_rows = _summary(rows)
    _write_json(output_dir / "summary.json", summary)
    columns = sorted({key for row in summary_rows for key in row})
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def _plot_metric(
    rows: list[dict[str, Any]], *, metric: str, title: str, path: Path
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    usable = [
        row
        for row in rows
        if row.get("status") == "completed" and isinstance(row.get(metric), (int, float))
    ]
    if not usable:
        return False
    labels = [f"{row['dataset']}\n{row['method']}" for row in usable]
    values = [float(row[metric]) for row in usable]
    fig, axis = plt.subplots(figsize=(max(6, len(labels) * 0.8), 4.5))
    axis.bar(range(len(values)), values)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_title(title)
    axis.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    requested = [
        ("symmetric_best_match_f1", "Recovery accuracy by dataset", "accuracy_by_dataset"),
        ("runtime_seconds", "Runtime by dataset", "runtime_by_dataset"),
        ("predicted_overlapping_node_fraction", "Predicted overlap structure", "overlap_structure"),
        ("matching_f1", "One-to-one method comparison", "method_comparison"),
    ]
    written: list[str] = []
    for metric, title, stem in requested:
        if _plot_metric(rows, metric=metric, title=title, path=plots_dir / stem):
            written.extend([str(plots_dir / f"{stem}.png"), str(plots_dir / f"{stem}.pdf")])
    return written


def _safe_output_dir(output_dir: Path, data_root: Path) -> Path:
    output_dir = output_dir.expanduser().resolve()
    data_root = data_root.expanduser().resolve()
    protected_inputs = [
        data_root / name
        for name in ("Amazon", "DBLP", "LiveJournal", "Youtube", "YouTube", "Wikipedia")
    ]
    if output_dir == data_root or any(
        output_dir == protected or protected in output_dir.parents
        for protected in protected_inputs
    ):
        raise ValueError("--output_dir must not overwrite a raw SNAP dataset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_dataset_status_records(
    *,
    output_dir: Path,
    dataset: str,
    cover: str,
    methods,
    seeds: list[int],
    resolutions: list[str | float],
    profile: str,
    status: str,
    reason: str,
) -> list[dict[str, Any]]:
    """Write resumable records when data/cover selection prevents a run."""
    records: list[dict[str, Any]] = []
    for adapter in methods:
        for seed in seeds:
            for requested_resolution in resolutions:
                resolution = 0.0 if requested_resolution == "auto" else float(requested_resolution)
                record = _record(
                    dataset=dataset,
                    cover=cover,
                    method=adapter.name,
                    seed=seed,
                    resolution=resolution,
                    status=status,
                    profile=profile,
                    reason=reason,
                )
                path = _run_path(output_dir, dataset, cover, adapter.name, seed, resolution)
                _write_json(path, record)
                records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible overlapping-community benchmark for Amazon, DBLP, "
            "LiveJournal, YouTube, and Wikipedia SNAP covers. Standard runs "
            "are bounded induced subgraphs; use --profile full to disable that cap."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_root", help="SNAP root (default: HEDONIC_NETWORKS_DIR or ~/Databases/Hedonic/Networks)")
    parser.add_argument("--datasets", help="Comma list: amazon,dblp,livejournal,youtube,wikipedia")
    parser.add_argument("--cover", choices=("all", "top5000"), help="Supplied GT cover variant; Wikipedia provides all categories only")
    parser.add_argument("--methods", help="Comma-list of methods; use --list-methods for details")
    parser.add_argument(
        "--skip-methods",
        help=(
            "Comma-list of external baselines to record as intentionally not rerun; "
            "only cpm and demon are allowed"
        ),
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="standard")
    parser.add_argument("--seeds", help="Comma/range syntax, e.g. 0-4")
    parser.add_argument("--resolutions", help="auto, comma floats, or start:stop:count")
    parser.add_argument("--output_dir", help="Artifact root (default: ~/Databases/Hedonic/experiments/snap_benchmark)")
    parser.add_argument("--resume", action="store_true", help="Skip existing per-run records")
    parser.add_argument("--timeout_per_run", type=float, help="Hard wall-clock limit per detector run (seconds; <=0 disables)")
    parser.add_argument(
        "--timeout_by_method",
        help="Per-method overrides, e.g. demon=900,cpm=1800",
    )
    parser.add_argument(
        "--timeout_by_dataset",
        help="Per-dataset overrides, e.g. livejournal=1800,youtube=1800",
    )
    parser.add_argument("--max_nodes", type=int, help="Deterministic GT-informed induced-subgraph cap; <=0 disables")
    parser.add_argument(
        "--max_memberships",
        type=int,
        help=(
            "Cap memberships per vertex for hedonic methods; the default is the "
            "maximum ground-truth memberships of any node, not the number of communities"
        ),
    )
    parser.add_argument(
        "--memory_limit_gb",
        type=float,
        help=(
            "RSS limit for each isolated detector process; <=0 disables. A parent monitor "
            "sums the detector process tree and terminates an over-limit group. The paper "
            "scheduler supplies a graph-aware value."
        ),
    )
    parser.add_argument("--omega", action="store_true", help="Compute memory-safe sampled Omega")
    parser.add_argument("--omega_sample_size", type=int, default=100_000)
    parser.add_argument("--plots", dest="plots", action="store_true", default=None)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--list-networks", action="store_true", help="List supported overlapping SNAP datasets and exit")
    parser.add_argument("--list-methods", action="store_true", help="List adapters and availability then exit")
    parser.add_argument("--dry-run", action="store_true", help="Inspect/validate selected data and write no detector runs")
    parser.add_argument("--execution", choices=("fresh", "rerun", "retry"), default="fresh", help=argparse.SUPPRESS)
    return parser


def _print_networks() -> None:
    print("Overlapping SNAP benchmarks:")
    print("  amazon      product cover: all, top5000")
    print("  dblp        co-authorship cover: all, top5000")
    print("  livejournal social-community cover: all, top5000")
    print("  youtube     channel cover: all, top5000")
    print("  wikipedia   wiki-topcats category cover: all (directed graph)")
    print("Excluded: email-Eu-core, Cora, and PubMed have disjoint labels; DBLP_CLI is output only.")


def _print_methods() -> None:
    print("Overlapping benchmark methods:")
    for name, info in method_availability().items():
        state = "available" if info["available"] else f"unavailable: {info['reason']}"
        print(f"  {name:20} {state}")
        print(f"    {info['family']} | {info['implementation']}")
        print(f"    parameters={info['parameters']} | {info['scalability']}")
        if info["install_requirement"]:
            print(f"    install: {info['install_requirement']}")


def _effective_options(args: argparse.Namespace) -> dict[str, Any]:
    profile = PROFILE_DEFAULTS[args.profile]
    datasets = _parse_csv(args.datasets, network_names(), "dataset") or profile["datasets"]
    methods = _parse_csv(args.methods, METHODS, "method") or profile["methods"]
    skip_methods = _parse_csv(args.skip_methods, METHODS, "skip-methods") or []
    unknown_skip_methods = sorted(set(skip_methods) - set(methods))
    if unknown_skip_methods:
        raise ValueError(
            "--skip-methods must be a subset of --methods: "
            + ", ".join(unknown_skip_methods)
        )
    unsupported_skip_methods = sorted(set(skip_methods) - NON_SCALABLE_BASELINES)
    if unsupported_skip_methods:
        raise ValueError(
            "--skip-methods is limited to external baselines cpm,demon: "
            + ", ".join(unsupported_skip_methods)
        )
    if args.max_memberships is not None and args.max_memberships < 1:
        raise ValueError("--max_memberships must be >= 1")
    if args.memory_limit_gb is not None and args.memory_limit_gb <= 0:
        raise ValueError("--memory_limit_gb must be positive when supplied")
    timeout_per_run = (
        args.timeout_per_run
        if args.timeout_per_run is not None
        else profile["timeout_per_run"]
    )
    if timeout_per_run <= 0 and not args.dry_run:
        raise ValueError("--timeout_per_run must be positive unless used only for a dry-run")
    return {
        "datasets": datasets,
        "methods": methods,
        "skip_methods": skip_methods,
        "cover": args.cover or profile["cover"],
        "seeds": parse_seeds(args.seeds) or profile["seeds"],
        "resolutions": parse_resolutions(args.resolutions) or profile["resolutions"],
        "max_nodes": args.max_nodes if args.max_nodes is not None else profile["max_nodes"],
        "max_memberships": args.max_memberships,
        "memory_limit_bytes": (
            int(args.memory_limit_gb * (1024**3))
            if args.memory_limit_gb is not None
            else None
        ),
        "timeout_per_run": timeout_per_run,
        "timeout_by_method": _parse_timeout_map(args.timeout_by_method, "--timeout_by_method"),
        "timeout_by_dataset": _parse_timeout_map(args.timeout_by_dataset, "--timeout_by_dataset"),
        "plots": args.plots if args.plots is not None else profile["plots"],
    }


def run_benchmark(args: argparse.Namespace) -> int:
    """Run the configured suite. Public for small-fixture tests and scripts."""
    options = _effective_options(args)
    experiment_identity = current_experiment_identity()
    if not experiment_identity["tracked_files_match_lock"]:
        raise ValueError("Protocol-locked code/config hashes do not match; refresh and review the lock")
    if not experiment_identity["lucas_igraph"]["package_identity_matches_lock"]:
        raise ValueError("installed lucas-igraph package does not match the protocol-locked release")
    selected_methods = resolve_methods(options["methods"])
    data_root = Path(args.data_root).expanduser() if args.data_root else DEFAULT_NETWORKS_DIR
    default_output = Path(OUTPUT_DIR) / "snap_benchmark"
    output_dir = _safe_output_dir(
        Path(args.output_dir).expanduser() if args.output_dir else default_output,
        data_root,
    )
    availability = method_availability()
    _append_log(output_dir, f"start profile={args.profile} datasets={','.join(options['datasets'])}")
    _write_json(output_dir / "method_availability.json", availability)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _timestamp(),
        "command": "hedonic-exp overlapping-benchmark",
        "profile": args.profile,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "options": {**options, "omega": args.omega, "omega_sample_size": args.omega_sample_size},
        "method_availability": availability,
        "experiment_identity": experiment_identity,
        "datasets": {},
        "run_status_counts": {},
        "execution_counts": {"cached": 0, "rerun": 0, "retry": 0, "fresh": 0},
    }
    _write_json(output_dir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    for dataset_name in options["datasets"]:
        try:
            dataset = (
                smoke_dataset(dataset_name, cover_variant=options["cover"])
                if args.profile == "smoke"
                else load_snap_dataset(
                    dataset_name,
                    cover_variant=options["cover"],
                    data_root=data_root,
                )
            )
            dataset = bounded_induced_dataset(dataset, options["max_nodes"])
            dataset = common_undirected_analysis_dataset(dataset)
            print_dataset_report(dataset.report)
            manifest["datasets"][dataset_name] = {"status": "loaded", "report": dataset.report}
        except UnsupportedCoverVariant as exc:
            manifest["datasets"][dataset_name] = {"status": "skipped_unsupported", "reason": str(exc)}
            records.extend(
                _write_dataset_status_records(
                    output_dir=output_dir,
                    dataset=dataset_name,
                    cover=options["cover"],
                    methods=selected_methods,
                    seeds=options["seeds"],
                    resolutions=options["resolutions"],
                    profile=args.profile,
                    status="skipped_unsupported",
                    reason=str(exc),
                )
            )
            _append_log(output_dir, f"dataset={dataset_name} status=skipped reason={exc}")
            continue
        except SnapLoadError as exc:
            manifest["datasets"][dataset_name] = {"status": "data_unavailable", "reason": str(exc)}
            print(f"[dataset] {dataset_name}: unavailable: {exc}", file=sys.stderr)
            records.extend(
                _write_dataset_status_records(
                    output_dir=output_dir,
                    dataset=dataset_name,
                    cover=options["cover"],
                    methods=selected_methods,
                    seeds=options["seeds"],
                    resolutions=options["resolutions"],
                    profile=args.profile,
                    status="data_unavailable",
                    reason=str(exc),
                )
            )
            _append_log(output_dir, f"dataset={dataset_name} status=data_unavailable reason={exc}")
            continue

        if args.dry_run:
            continue
        ground_truth_membership_cap = max(
            1,
            int(
                dataset.report.get("overlap_statistics", {}).get(
                    "max_memberships_per_node", 1
                )
            ),
        )
        configured_membership_cap = options["max_memberships"]
        max_memberships = max(
            2,
            min(
                ground_truth_membership_cap,
                configured_membership_cap
                if configured_membership_cap is not None
                else ground_truth_membership_cap,
            ),
        )
        manifest["datasets"][dataset_name]["membership_capacity"] = {
            "ground_truth_max_memberships_per_node": ground_truth_membership_cap,
            "configured_cap": configured_membership_cap,
            "effective_max_memberships": max_memberships,
        }
        for requested_resolution in options["resolutions"]:
            requested_value = (
                dataset.graph.density()
                if requested_resolution == "auto"
                else float(requested_resolution)
            )
            for adapter in selected_methods:
                resolution = effective_resolution(adapter, dataset.graph, requested_value)
                for seed in options["seeds"]:
                    initialization = (
                        {
                            "kind": "seeded_random_disjoint",
                            "requested_community_count": len(dataset.cover),
                            "seed": int(seed),
                            "shared_across_hedonic_methods": True,
                        }
                        if adapter.name.startswith("hedonic_")
                        else None
                    )
                    path = _run_path(output_dir, dataset.name, options["cover"], adapter.name, seed, resolution)
                    execution_event = str(args.execution)
                    incompatible_cache = False
                    timeout_seconds = _timeout_for(options, dataset.name, adapter.name)
                    expected_cache = {
                        "experiment_identity": experiment_identity,
                        "dataset_metadata_identity": dataset_metadata_identity(dataset.report),
                        "analysis_graph": dataset.report.get("analysis_graph"),
                        "dataset": dataset.name,
                        "cover": options["cover"],
                        "method": adapter.name,
                        "seed": seed,
                        "resolution": resolution,
                        "max_memberships": max_memberships,
                        "allow_isolation": bool(
                            adapter.parameters.get("allow_isolation", False)
                        ),
                        "initialization": initialization,
                        "timeout_seconds": timeout_seconds,
                        "memory_limit_bytes": options["memory_limit_bytes"],
                        "run_options": {
                            "omega": bool(args.omega),
                            "omega_sample_size": int(args.omega_sample_size),
                            "external_baseline_policy": adapter.name in options["skip_methods"],
                        },
                    }
                    if adapter.name in options["skip_methods"]:
                        existing = _read_json(path) if path.is_file() else None
                        if (
                            args.resume
                            and existing is not None
                            and _cache_parameters_compatible(existing, expected_cache)
                            and str(existing.get("status")) == SKIPPED_EXTERNAL_STATUS
                        ):
                            existing["run_path"] = str(path)
                            existing["execution"] = "cached"
                            records.append(existing)
                            manifest["execution_counts"]["cached"] += 1
                            print(
                                f"[resume] {dataset.name}/{adapter.name}/seed={seed}/"
                                f"resolution={resolution:.6g} (external baseline not rerun)"
                            )
                            continue
                        if existing is not None:
                            backup = path.with_name(
                                f"{path.name}.incompatible.{int(time.time() * 1000)}"
                            )
                            try:
                                path.replace(backup)
                            except OSError:
                                pass
                        record = _record(
                            dataset=dataset.name,
                            cover=options["cover"],
                            method=adapter.name,
                            seed=seed,
                            resolution=resolution,
                            status=SKIPPED_EXTERNAL_STATUS,
                            profile=args.profile,
                            dataset_report=dataset.report,
                            max_memberships=max_memberships,
                            ground_truth_max_memberships=ground_truth_membership_cap,
                            allow_isolation=expected_cache["allow_isolation"],
                            initialization=initialization,
                            timeout_seconds=timeout_seconds,
                            memory_limit_bytes=options["memory_limit_bytes"],
                            run_options=expected_cache["run_options"],
                            failure_kind="external_baseline_not_rerun",
                            reason=(
                                "External baseline intentionally not rerun: the release-sensitive "
                                "change is confined to native community_leiden; historical external "
                                "baseline outcomes are retained separately."
                            ),
                            method_metadata={
                                "name": adapter.name,
                                "family": adapter.family,
                                "implementation": adapter.implementation,
                                "parameters": adapter.parameters,
                                "execution": "not_run_external_baseline",
                            },
                        )
                        record["execution"] = "external_baseline_policy"
                        _write_json(path, record)
                        _append_log(
                            output_dir,
                            f"dataset={dataset.name} method={adapter.name} seed={seed} "
                            f"resolution={resolution:.12g} status={record['status']}",
                        )
                        record["run_path"] = str(path)
                        records.append(record)
                        manifest["execution_counts"]["external_baseline_policy"] = (
                            manifest["execution_counts"].get("external_baseline_policy", 0) + 1
                        )
                        print(
                            f"[external-baseline-policy] {dataset.name}/{adapter.name}/seed={seed}/"
                            f"resolution={resolution:.6g}"
                        )
                        continue
                    if args.resume and path.is_file():
                        existing = _read_json(path)
                        if existing is not None and _cache_compatible(existing, expected_cache):
                            existing["run_path"] = str(path)
                            existing["execution"] = "cached"
                            records.append(existing)
                            manifest["execution_counts"]["cached"] += 1
                            print(f"[resume] {dataset.name}/{adapter.name}/seed={seed}/resolution={resolution:.6g}")
                            continue
                        # A compatible resource failure is a policy decision,
                        # not a cached metric.  Do not print [resume] or
                        # launch the same deterministic condition again.
                        if (
                            existing is not None
                            and adapter.name not in NON_SCALABLE_BASELINES
                            and _cache_parameters_compatible(existing, expected_cache)
                            and str(existing.get("status")) in RESOURCE_FAILURE_STATUSES
                        ):
                            existing["run_path"] = str(path)
                            existing["execution"] = "resource_failure_policy"
                            _write_json(path, existing)
                            records.append(existing)
                            manifest["execution_counts"]["resource_failure_policy"] = (
                                manifest["execution_counts"].get("resource_failure_policy", 0) + 1
                            )
                            print(f"[resource-failure] {dataset.name}/{adapter.name}/seed={seed}/resolution={resolution:.6g}")
                            continue
                        if (
                            existing is not None
                            and adapter.name in NON_SCALABLE_BASELINES
                            and _cache_parameters_compatible(existing, expected_cache)
                            and str(existing.get("status"))
                            in RESOURCE_FAILURE_STATUSES | {"skipped_not_scalable"}
                        ):
                            if str(existing.get("status")) != "skipped_not_scalable":
                                backup = path.with_name(
                                    f"{path.name}.incompatible.{int(time.time() * 1000)}"
                                )
                                try:
                                    path.replace(backup)
                                except OSError:
                                    pass
                                existing = _record(
                                    dataset=dataset.name,
                                    cover=options["cover"],
                                    method=adapter.name,
                                    seed=seed,
                                    resolution=resolution,
                                    status="skipped_not_scalable",
                                    profile=args.profile,
                                    dataset_report=dataset.report,
                                    max_memberships=max_memberships,
                                    allow_isolation=expected_cache["allow_isolation"],
                                    initialization=initialization,
                                    timeout_seconds=timeout_seconds,
                                    memory_limit_bytes=options["memory_limit_bytes"],
                                    run_options=expected_cache["run_options"],
                                    failure_kind="resource_exhausted",
                                    resource_status=existing.get("status"),
                                    reason="baseline previously exceeded its identical resource limit",
                                    detector_memory=existing.get("detector_memory") or existing.get("memory"),
                                    runtime_seconds=existing.get("runtime_seconds"),
                                )
                                _write_json(path, existing)
                            existing["run_path"] = str(path)
                            existing["execution"] = "not_scalable_policy"
                            _write_json(path, existing)
                            records.append(existing)
                            manifest["execution_counts"]["not_scalable_policy"] = (
                                manifest["execution_counts"].get("not_scalable_policy", 0) + 1
                            )
                            print(f"[not-scalable] {dataset.name}/{adapter.name}/seed={seed}/resolution={resolution:.6g}")
                            continue
                        if existing is not None:
                            incompatible_cache = True
                            execution_event = "rerun"
                            manifest["execution_counts"]["rerun"] += 1
                            _append_log(
                                output_dir,
                                f"cache_incompatible dataset={dataset.name} method={adapter.name} "
                                f"seed={seed} resolution={resolution:.12g}",
                            )
                        else:
                            execution_event = "fresh"
                            manifest["execution_counts"]["fresh"] += 1
                    if not availability[adapter.name]["available"]:
                        record = _record(
                            dataset=dataset.name,
                            cover=options["cover"],
                            method=adapter.name,
                            seed=seed,
                            resolution=resolution,
                            status="skipped_unsupported",
                            profile=args.profile,
                            dataset_report=dataset.report,
                            reason=availability[adapter.name]["reason"],
                            failure_kind="unsupported",
                            install_requirement=availability[adapter.name]["install_requirement"],
                            max_memberships=max_memberships,
                            allow_isolation=expected_cache["allow_isolation"],
                            initialization=initialization,
                            timeout_seconds=timeout_seconds,
                            memory_limit_bytes=options["memory_limit_bytes"],
                            run_options=expected_cache["run_options"],
                        )
                    else:
                        print(
                            f"[run] {dataset.name}/{adapter.name} seed={seed} "
                            f"resolution={resolution:.6g}",
                            flush=True,
                        )
                        initial_membership = (
                            seeded_initial_membership(
                                dataset.graph.vcount(), len(dataset.cover), seed
                            )
                            if initialization is not None
                            else None
                        )
                        outcome = _run_with_timeout(
                            adapter.name,
                            dataset.graph,
                            max_memberships=max_memberships,
                            resolution=resolution,
                            seed=seed,
                            timeout_seconds=timeout_seconds,
                            memory_limit_bytes=options["memory_limit_bytes"],
                            initial_membership=initial_membership,
                        )
                        del initial_membership
                        if outcome["status"] == "ok":
                            cover = outcome.pop("cover")
                            method_meta = outcome.pop("method_meta")
                            detector_memory = outcome.pop("memory", None)
                            metrics = evaluate_cover(
                                cover,
                                dataset.cover,
                                dataset.graph.vcount(),
                                compute_omega=args.omega,
                                omega_sample_size=args.omega_sample_size,
                                omega_seed=seed,
                            )
                            if dataset.graph.vcount() <= 20_000:
                                metrics["cpm_overlapping_quality"] = quality_overlapping_cpm(
                                    dataset.graph, cover, resolution
                                )
                                metrics["cpm_overlapping_quality_status"] = "computed"
                            else:
                                metrics["cpm_overlapping_quality"] = None
                                metrics["cpm_overlapping_quality_status"] = "skipped_large_graph"
                            metrics["runtime_seconds"] = outcome.get(
                                "runtime_seconds", method_meta["runtime_seconds"]
                            )
                            record = _record(
                                dataset=dataset.name,
                                cover=options["cover"],
                                method=adapter.name,
                                seed=seed,
                                resolution=resolution,
                                status="completed",
                                profile=args.profile,
                                dataset_report=dataset.report,
                                max_memberships=max_memberships,
                                ground_truth_max_memberships=ground_truth_membership_cap,
                                memory_limit_bytes=options["memory_limit_bytes"],
                                timeout_seconds=timeout_seconds,
                                run_options=expected_cache["run_options"],
                                detector_memory=detector_memory,
                                runtime_seconds=metrics["runtime_seconds"],
                                n_iterations=-1,
                                local_move_only=bool(
                                    adapter.parameters.get("local_move_only", False)
                                ),
                                allow_isolation=bool(
                                    adapter.parameters.get("allow_isolation", False)
                                ),
                                initialization=initialization,
                                method_metadata=method_meta,
                                metrics=metrics,
                            )
                        else:
                            outcome_status = str(outcome.pop("status"))
                            failure_kind = "error" if outcome_status == "error" else outcome_status
                            if outcome_status == "error":
                                outcome_status = "failed"
                            # The coordinator owns these two configured
                            # resource fields.  Older/interrupted workers may
                            # still return timeout_seconds, so discard it
                            # rather than passing it once explicitly and again
                            # via **outcome (which caused the reported
                            # TypeError).  Normalize monitor memory under the
                            # stable detector_memory JSON field for all
                            # terminal outcomes.
                            outcome.pop("timeout_seconds", None)
                            outcome.pop("memory_limit_bytes", None)
                            detector_memory = outcome.pop("memory", None)
                            reported_detector_memory = outcome.pop("detector_memory", None)
                            if detector_memory is None:
                                detector_memory = reported_detector_memory
                            resource_status = outcome_status if outcome_status in RESOURCE_FAILURE_STATUSES else None
                            if adapter.name in NON_SCALABLE_BASELINES and resource_status is not None:
                                outcome_status = "skipped_not_scalable"
                                failure_kind = "resource_exhausted"
                            record = _record(
                                dataset=dataset.name,
                                cover=options["cover"],
                                method=adapter.name,
                                seed=seed,
                                resolution=resolution,
                                status=outcome_status,
                                profile=args.profile,
                                dataset_report=dataset.report,
                                max_memberships=max_memberships,
                                ground_truth_max_memberships=ground_truth_membership_cap,
                                allow_isolation=expected_cache["allow_isolation"],
                                initialization=initialization,
                                memory_limit_bytes=options["memory_limit_bytes"],
                                timeout_seconds=timeout_seconds,
                                run_options=expected_cache["run_options"],
                                failure_kind=failure_kind,
                                resource_status=resource_status,
                                detector_memory=detector_memory,
                                **outcome,
                            )
                    if incompatible_cache and path.is_file():
                        backup = path.with_name(
                            f"{path.name}.incompatible.{int(time.time() * 1000)}"
                        )
                        try:
                            path.replace(backup)
                        except OSError:
                            pass
                    record["execution"] = execution_event
                    _write_json(path, record)
                    _append_log(
                        output_dir,
                        f"dataset={dataset.name} method={adapter.name} seed={seed} "
                        f"resolution={resolution:.12g} status={record['status']}",
                    )
                    record["run_path"] = str(path)
                    records.append(record)
    # Reading the run tree includes resumed and previously complete points even
    # when a current invocation selected only a subset.
    all_records = _completed_records(output_dir)
    rows = _write_results(output_dir, all_records)
    _write_summary(output_dir, rows)
    plot_paths = _write_plots(output_dir, rows) if options["plots"] else []
    status_counts = CounterLike(record.get("status", "unknown") for record in all_records)
    manifest["run_status_counts"] = dict(status_counts)
    manifest["finished_at"] = _timestamp()
    manifest["artifacts"] = {
        "manifest": str(output_dir / "manifest.json"),
        "runs": str(output_dir / "runs"),
        "results_jsonl": str(output_dir / "results.jsonl"),
        "results_csv": str(output_dir / "results.csv.gz"),
        "summary_json": str(output_dir / "summary.json"),
        "summary_csv": str(output_dir / "summary.csv"),
        "method_availability": str(output_dir / "method_availability.json"),
        "logs": str(output_dir / "logs"),
        "plots": plot_paths,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _append_log(output_dir, f"finished status_counts={json.dumps(manifest['run_status_counts'], sort_keys=True)}")
    print(f"[done] Benchmark artifacts: {output_dir}")
    for name, path in manifest["artifacts"].items():
        if name != "plots":
            print(f"  {name}: {path}")
    return 0


class CounterLike(dict):
    """Tiny serializable Counter replacement, avoiding another import alias."""

    def __init__(self, values: Iterable[str]) -> None:
        super().__init__()
        for value in values:
            self[value] = self.get(value, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_networks:
        _print_networks()
        return 0
    if args.list_methods:
        _print_methods()
        return 0
    try:
        return run_benchmark(args)
    except (ValueError, SnapLoadError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover - argparse raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
