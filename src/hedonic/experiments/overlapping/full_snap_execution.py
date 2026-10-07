"""All-stage isolated execution and sampled RSS watchdog for full-snap-v1."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _rss(pid):
    # All adapters in this track are in-process libraries; no external children.
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                            capture_output=True, text=True, timeout=5, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ps RSS query failed (rc={result.returncode}): {result.stderr.strip()}")
    return int(result.stdout.strip()) * 1024


def _entry(config, unit_dir):
    from .full_snap import execute_unit
    directory = Path(unit_dir)
    started = time.monotonic()
    try:
        result = execute_unit(config, directory)
    except MemoryError:
        result = {"status": "oom", "error": "worker raised MemoryError"}
    except BaseException as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}",
                  "traceback": traceback.format_exc()}
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result.update(runtime_seconds=time.monotonic() - started,
                  worker_peak_rss_bytes=int(rss if sys.platform == "darwin" else rss * 1024))
    atomic_json(directory / "packet.json", result)


def run_unit(config: dict, directory: Path, *, heartbeat=None):
    """The time/RSS budgets include loading, detection, serialization and scoring.

    RSS is sampled every 0.25s, not a hard allocator limit. Overshoot between
    observations remains possible and is recorded as a policy limitation.
    Native interruption handlers are not used; timeout kills this expendable
    per-method process, retaining only already-committed artifacts.
    """
    if directory.exists() and any(directory.iterdir()):
        # Preserve orphaned partial attempts and never consume their stale packet.
        directory = directory.with_name(directory.name + f"-attempt-{time.time_ns()}")
    directory.mkdir(parents=True, exist_ok=True)
    process = mp.get_context("spawn").Process(target=_entry, args=(config, str(directory)))
    started = time.monotonic()
    peak, last_heartbeat = 0, -1.0
    started_process = False
    result = None
    try:
        process.start()
        started_process = True
        while process.is_alive():
            elapsed = time.monotonic() - started
            try:
                observed = _rss(process.pid)
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                process.join(0)
                if not process.is_alive():
                    break  # RSS query raced with confirmed worker exit.
                result = {"status": "resource_monitor_failed", "error": str(error)}
                break
            peak = max(peak, observed)
            if heartbeat and elapsed - last_heartbeat >= 1:
                heartbeat(elapsed, observed, process.pid)
                last_heartbeat = elapsed
            if observed > config["memory_limit_bytes"]:
                result = {"status": "memory_limit", "error": "sampled RSS exceeded limit"}
                break
            if elapsed > config["timeout_seconds"]:
                result = {"status": "timeout", "error": "all-stage wall-clock limit exceeded"}
                break
            process.join(0.25)
        if result is None:
            packet = directory / "packet.json"
            if packet.is_file():
                result = json.loads(packet.read_text())
                # A worker can finish after its deadline between RSS polls.
                # Preserve its committed artifacts, but do not admit it as a
                # completed within-budget measurement.
                if result.get("status") == "completed" and time.monotonic() - started > config["timeout_seconds"]:
                    result.update(status="timeout", worker_terminal_status="completed",
                                  error="worker completed after the all-stage wall-clock deadline")
            else:
                result = {"status": "worker_exit_unclassified", "exitcode": process.exitcode,
                          "error": "no terminal packet; signal alone does not prove OOM"}
    finally:
        if started_process and process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join(1)
        if started_process:
            process.join()
    result.update(total_wall_seconds=time.monotonic() - started, sampled_peak_rss_bytes=peak,
                  rss_enforcement="0.25s sampled watchdog; possible between-sample overshoot")
    stage = directory / "stage.json"
    result["last_committed_stage"] = json.loads(stage.read_text()) if stage.is_file() else None
    # Covers may have been committed even when scoring/audit subsequently failed.
    from .full_snap import file_identity
    result["artifacts"] = [file_identity(p) for p in sorted(directory.glob("*.json*"))
                           if not p.name.endswith(".tmp") and p.name != "packet.json"]
    return result
