"""Small isolated-process runner for the GT robustness experiment.

The locked SNAP benchmark has its own private runner.  This module intentionally
does not modify or import that private implementation; its file-backed packet
transport avoids a bounded multiprocessing pipe when a detector returns a
large cover.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import resource
import sys
import tempfile
import time
import traceback
from typing import Any, Callable


@dataclass(frozen=True)
class ProcessOutcome:
    status: str
    payload: Any | None
    runtime_seconds: float
    error: str | None = None
    peak_rss_bytes: int | None = None


def _peak_rss_bytes() -> int:
    """Return the worker's peak RSS with the platform's native unit fixed."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the BSDs used by CI conventionally report
    # KiB.  The explicit conversion keeps the artifact schema portable.
    return raw if sys.platform == "darwin" else raw * 1024


def _write_packet(path: str, packet: dict[str, Any]) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(packet, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)


def _worker(
    packet_path: str,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    started = time.monotonic()
    try:
        payload = target(*args, **kwargs)
        packet = {
            "status": "ok",
            "payload": payload,
            "runtime_seconds": time.monotonic() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    except BaseException as exc:  # the parent records detector failures explicitly
        packet = {
            "status": "failed",
            "payload": None,
            "runtime_seconds": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "peak_rss_bytes": _peak_rss_bytes(),
        }
    try:
        _write_packet(packet_path, packet)
    except BaseException:
        # There is no safe channel left if packet serialization itself fails.
        pass


def run_in_subprocess(
    target: Callable[..., Any],
    *args: Any,
    timeout_seconds: float | None = None,
    packet_dir: str | Path | None = None,
    **kwargs: Any,
) -> ProcessOutcome:
    """Run a picklable callable with a hard wall-clock limit."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    directory = Path(packet_dir) if packet_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix="hedonic-gt-worker-",
        suffix=".pkl",
        dir=str(directory) if directory is not None else None,
    )
    os.close(fd)
    packet_path = Path(raw_path)
    context = mp.get_context("spawn")
    process = context.Process(
        target=_worker,
        args=(str(packet_path), target, tuple(args), dict(kwargs)),
    )
    started = time.monotonic()
    started_process = False

    def stop_worker() -> None:
        if not started_process or not process.is_alive():
            return
        process.terminate()
        process.join(1.0)
        if process.is_alive():
            process.kill()
            process.join(1.0)

    try:
        process.start()
        started_process = True
        process.join(timeout_seconds)
        if process.is_alive():
            stop_worker()
            runtime = time.monotonic() - started
            return ProcessOutcome(
                "timeout",
                None,
                runtime,
                "worker exceeded timeout",
                peak_rss_bytes=None,
            )

        runtime = time.monotonic() - started
        if not packet_path.is_file():
            return ProcessOutcome(
                "failed",
                None,
                runtime,
                f"worker exited with code {process.exitcode} without a packet",
                peak_rss_bytes=None,
            )
        with packet_path.open("rb") as stream:
            packet = pickle.load(stream)
        status = str(packet.get("status", "failed"))
        return ProcessOutcome(
            status,
            packet.get("payload"),
            float(packet.get("runtime_seconds", runtime)),
            packet.get("error"),
            int(packet["peak_rss_bytes"])
            if packet.get("peak_rss_bytes") is not None
            else None,
        )
    except (OSError, pickle.PickleError, EOFError) as exc:
        runtime = time.monotonic() - started
        return ProcessOutcome(
            "failed",
            None,
            runtime,
            f"could not read worker packet: {exc}",
            peak_rss_bytes=None,
        )
    finally:
        # Covers KeyboardInterrupt and any parent-side exception as well as
        # ordinary timeout paths: no detached native worker or stale packet is
        # allowed to survive an interrupted resumable run.
        stop_worker()
        packet_path.unlink(missing_ok=True)


__all__ = ["ProcessOutcome", "run_in_subprocess"]
