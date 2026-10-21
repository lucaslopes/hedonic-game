"""The one clock every overlap detector is timed with (``timing_scope = "detector-compute-v1"``).

The benchmark compares methods by detection wall-clock and peak memory, so every method's clock has to cover the
same work. The rule, applied by every adapter through this module and nothing else:

START  The graph is already loaded and has already been converted to the form the detector consumes: the
       ``igraph`` undirected/active-vertex copy, the ``networkx`` graph, the :class:`hedonic.Game`, the JSON request
       of an isolated worker, the edge-list/METIS/``.m`` input files an executable reads. Temporary folders, helper
       scripts and library imports are also done before the clock starts. For an executable the clock starts
       when the process exists (``Popen`` has returned, so the harness's own fork/exec work is not counted).

STOP   The clock stops as soon as the detector has returned its raw result: the call returned (in process), the
       process exited (executable), or the worker's own clock stopped (see below). Reading and parsing the result,
       validating it, canonicalising the cover, scoring (F1, ONMI, Omega), robustness/stability audits, saving covers
       and writing records all come after the clock.

INSIDE Only the detector's own computation. Two cases cannot be cut finer, and both are *counted* and recorded:

       * an executable (CoDeSEG, BigCLAM, FOX, NEO-K-Means, OSLOM, SVI) and the NcGame script read their input
         file and write their cover themselves. That is how the implementation is used, it cannot be separated
         without editing the program, so the clock covers the program from exec to exit (``kind = "process"``).
         The per-process input size is recorded (``input_bytes``) so the share can be judged.
       * Octave/R/Python code the *harness* writes or ships (the CDlib workers, the generated Octave and R
         scripts, the NcGame wrapper) is timed **inside** that code, around the detector call only, after the
         interpreter started, the libraries were imported and the input graph was built. Its start-up,
         imports and JSON transport are an artefact of isolating CDlib in its own environment (a library
         method in this process pays none of them), so they are reported (``isolation_overhead_seconds``) but not
         counted (``kind = "worker"``).

``detection_seconds`` of a record is the sum of the counted regions. ``timing_scope`` names this rule in every
record, and each record carries the regions (``timing.regions``) so a reader can see which kind of clock each
number is.

Peak memory follows the same boundary. ``detector_peak_rss_bytes`` is the peak resident set of the process the
detector ran in: for an executable or worker the OS figure of that process alone (``wait4``; it includes the
process's own copy of the graph, interpreter and imports, which cannot be separated), for an in-process method the
lifetime peak of the harness process **read when the clock stops, before scoring** (comparable between methods only
for runs started in a fresh process, ``--isolate``). ``detector_peak_rss_scope`` says which one it is. The old
``peak_child_rss_bytes`` was ``RUSAGE_CHILDREN``'s running maximum over every child the run had ever waited for, so a
method without a child reported the previous method's peak.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

TIMING_SCOPE = "detector-compute-v1"
TIMING_RULE = (
    "detection_seconds covers the detector's own computation only: the graph is loaded and converted, the input files "
    "are written and the libraries are imported before the clock starts; it stops when the detector returns its raw "
    "result, before the result is parsed, validated, canonicalised, scored, audited or saved. Executables and the "
    "NcGame script are timed from exec to exit (their own file reading and writing is counted); code the harness "
    "ships or generates (CDlib workers, Octave and R scripts, the NcGame wrapper) is timed inside, around the "
    "detector call, so interpreter start-up, imports and JSON transport are not counted."
)
# What a worker or script writes when it times itself.
REPORT_KEY = "detector_seconds"

_FACTOR = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS and KiB elsewhere
_CURRENT: contextvars.ContextVar["DetectorClock | None"] = contextvars.ContextVar("hedonic_detector_clock", default=None)


def _own_usage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_SELF)


def peak_rss_bytes() -> int:
    """The lifetime peak resident set of this process, in bytes."""
    return int(_own_usage().ru_maxrss * _FACTOR)


# On Linux a child's ru_maxrss (wait4) is at least the parent's resident size when it was spawned: exec records the
# high-water mark of the memory it replaces, which is the parent's (vfork) or its copy (fork). So a program started by
# a harness that holds a large graph would be charged for the graph. There the child's own peak is read from
# /proc/<pid>/status (VmHWM, the high-water mark of the new program's memory) while it runs; the reading misses at
# most the growth of its last SAMPLE_SECONDS. macOS reports the program's own peak and needs none of this.
_PROC_HWM = sys.platform.startswith("linux") and os.path.isdir("/proc/self")
SAMPLE_SECONDS = 0.01


class _HighWater:
    """Samples a running child's VmHWM until :meth:`stop`; ``peak`` is the largest value read, in bytes."""

    def __init__(self, pid: int) -> None:
        self.peak: int | None = None
        self._path = f"/proc/{pid}/status"
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            with open(self._path, "rb") as handle:
                for line in handle:
                    if line.startswith(b"VmHWM:"):
                        value = int(line.split()[1]) * 1024
                        self.peak = value if self.peak is None else max(self.peak, value)
                        return
        except (OSError, ValueError, IndexError):
            return  # not started yet, or already a zombie (its memory is gone)

    def _run(self) -> None:
        while not self._done.is_set():
            self._read()
            self._done.wait(SAMPLE_SECONDS)

    def stop(self) -> None:
        self._done.set()
        self._thread.join()


def _child_peak(usage: Any, hwm: "_HighWater | None") -> int:
    """The child's own peak: ru_maxrss where it is the program's (macOS), the sampled VmHWM on Linux."""
    if hwm is not None and hwm.peak is not None:
        return int(hwm.peak)
    return int(usage.ru_maxrss * _FACTOR)


def current_rss_bytes() -> int | None:
    """The resident set of this process now, in bytes (``/proc`` on Linux, ``ps`` elsewhere); None when unreadable."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as handle:
                return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) * 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


@dataclass
class ProcessResult:
    """A finished process, with the clock of its life: ``seconds`` runs from exec to exit."""

    returncode: int
    stdout: str
    stderr: str
    seconds: float
    peak_rss_bytes: int
    user_cpu_seconds: float
    system_cpu_seconds: float
    command: list[str] = field(default_factory=list)
    input_bytes: int | None = None


class DetectorClock:
    """Accumulates the seconds, CPU and memory of one detector run. Only the methods below start it."""

    def __init__(self) -> None:
        self.regions: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.user_cpu_seconds = 0.0
        self.system_cpu_seconds = 0.0
        self.child_user_cpu_seconds = 0.0
        self.child_system_cpu_seconds = 0.0
        self.process_peak_rss_bytes = 0
        self.harness_peak_rss_bytes: int | None = None
        self.rss_before_bytes: int | None = None
        self.stopped_at: float | None = None
        self.memory_stop: dict[str, Any] | None = None  # the watchdog's stop of a process (``memwatch``), if any

    # ------------------------------------------------------------------ what ran
    @property
    def used(self) -> bool:
        """True once some region was counted. A detector that never reached the clock has no timing."""
        return any(region["counted"] for region in self.regions)

    @property
    def in_process(self) -> bool:
        return any(region["kind"] == "in_process" for region in self.regions)

    def _stop(self) -> None:
        self.harness_peak_rss_bytes = peak_rss_bytes()
        self.stopped_at = time.perf_counter()

    # ------------------------------------------------------------------ in process
    @contextlib.contextmanager
    def measure(self, label: str = "detector") -> Iterator[None]:
        """Time a block of this process's own code: the detector call and nothing else."""
        if self.rss_before_bytes is None:
            self.rss_before_bytes = current_rss_bytes()
        usage0 = _own_usage()
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            usage1 = _own_usage()
            self.seconds += seconds
            self.user_cpu_seconds += usage1.ru_utime - usage0.ru_utime
            self.system_cpu_seconds += usage1.ru_stime - usage0.ru_stime
            self.regions.append({"kind": "in_process", "label": label, "seconds": seconds, "counted": True})
            self._stop()

    # ------------------------------------------------------------------ out of process
    def process(
        self,
        command: Sequence[str],
        *,
        label: str,
        cwd: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        stdin_text: str | None = None,
        timeout: float | None = None,
        counted: bool = True,
        input_path: str | os.PathLike | None = None,
    ) -> ProcessResult:
        """Run ``command`` and time it from exec to exit. Output goes to temporary files, so no pipe is ever drained
        inside the clock; ``stdin_text`` is written before the process exists.

        ``counted=False`` records the region without adding it to ``seconds``: a worker that times itself calls
        :meth:`adopt` afterwards. Raises ``subprocess.TimeoutExpired`` (the whole process group is killed) and
        ``OSError`` like ``subprocess.run`` did, after recording the time up to then.

        While it runs, a thread of this process reads the machine's free memory every half second (``memwatch``; the
        program's own process is never touched): below the floor the program's whole tree is killed and
        ``memwatch.MemoryLimitReached`` is raised, after recording the time up to then (``memory_stop``).
        """
        from hedonic.experiments.overlapping import memwatch

        command = [str(part) for part in command]
        with tempfile.TemporaryDirectory(prefix="hedonic-clock-") as scratch:
            out_path, err_path = Path(scratch, "stdout"), Path(scratch, "stderr")
            stdin_handle = None
            if stdin_text is not None:
                in_path = Path(scratch, "stdin")
                in_path.write_text(stdin_text, encoding="utf-8")
                stdin_handle = open(in_path, "rb")
            input_bytes = None
            if input_path is not None:
                with contextlib.suppress(OSError):
                    input_bytes = Path(input_path).stat().st_size
            if self.rss_before_bytes is None:
                self.rss_before_bytes = current_rss_bytes()
            try:
                with open(out_path, "wb") as out, open(err_path, "wb") as err:
                    handle = subprocess.Popen(
                        command, cwd=cwd, env=env, stdin=stdin_handle if stdin_handle else subprocess.DEVNULL,
                        stdout=out, stderr=err, start_new_session=True,
                    )
                    started = time.perf_counter()  # the process exists: its exec has succeeded
                    killed = threading.Event()

                    def kill_group() -> None:
                        killed.set()
                        with contextlib.suppress(OSError):
                            os.killpg(handle.pid, signal.SIGKILL)

                    timer = threading.Timer(timeout, kill_group) if timeout is not None else None
                    if timer is not None:
                        timer.daemon = True
                        timer.start()
                    hwm = _HighWater(handle.pid) if _PROC_HWM else None
                    watch = memwatch.Watch(handle.pid)
                    try:
                        with watch.thread():
                            _pid, status, usage = os.wait4(handle.pid, 0)
                    except BaseException:  # Ctrl-C or an unexpected error: leave nothing running
                        kill_group()
                        with contextlib.suppress(OSError):
                            os.wait4(handle.pid, 0)
                        raise
                    finally:
                        seconds = time.perf_counter() - started
                        if timer is not None:
                            timer.cancel()
                        if hwm is not None:
                            hwm.stop()
                    handle.returncode = os.waitstatus_to_exitcode(status)
            finally:
                if stdin_handle is not None:
                    stdin_handle.close()
            peak = _child_peak(usage, hwm)
            region = {"kind": "process", "label": label, "seconds": seconds, "counted": bool(counted),
                      "peak_rss_bytes": peak, "input_bytes": input_bytes}
            if hwm is not None:
                region["peak_rss_source"] = "proc_vmhwm_sampled"
            if watch.tripped is not None:
                region["stopped_for_memory"] = True
            self.regions.append(region)
            if counted:
                self.seconds += seconds
            self.child_user_cpu_seconds += usage.ru_utime
            self.child_system_cpu_seconds += usage.ru_stime
            self.process_peak_rss_bytes = max(self.process_peak_rss_bytes, peak)
            self._stop()
            if watch.tripped is not None:
                self.memory_stop = dict(watch.tripped)
                raise memwatch.MemoryLimitReached(watch.tripped["free_memory_percent"],
                                                  watch.tripped["memory_floor_percent"])
            if killed.is_set():
                raise subprocess.TimeoutExpired(command, timeout)
            return ProcessResult(
                returncode=handle.returncode,
                stdout=out_path.read_text(encoding="utf-8", errors="replace"),
                stderr=err_path.read_text(encoding="utf-8", errors="replace"),
                seconds=seconds, peak_rss_bytes=peak,
                user_cpu_seconds=float(usage.ru_utime), system_cpu_seconds=float(usage.ru_stime),
                command=command, input_bytes=input_bytes,
            )

    def adopt(self, seconds: Any, process: ProcessResult, *, label: str) -> float:
        """Count the time a worker or script measured around its own detector call.

        The report must be a finite number between 0 and the process's own exec-to-exit time (a worker cannot have
        computed for longer than it lived); anything else is an error rather than a guess.
        """
        try:
            value = float(seconds)
        except (TypeError, ValueError):
            raise ValueError(f"{label} did not report {REPORT_KEY}") from None
        if not (0.0 <= value <= process.seconds + 0.05):
            raise ValueError(f"{label} reported {value} s of detection in a {process.seconds:.3f} s process")
        value = min(value, process.seconds)
        self.seconds += value
        self.regions.append({"kind": "worker", "label": label, "seconds": value, "counted": True,
                             "process_seconds": process.seconds,
                             "isolation_overhead_seconds": process.seconds - value,
                             "peak_rss_bytes": process.peak_rss_bytes})
        return value

    # ------------------------------------------------------------------ what to record
    def report(self) -> dict[str, Any]:
        """The timing and resources of the run, for the record."""
        out_of_process = [region for region in self.regions if region["kind"] != "in_process"]
        scope = "detector_process" if out_of_process else "harness_process_lifetime"
        peak = self.process_peak_rss_bytes if out_of_process else (self.harness_peak_rss_bytes or 0)
        return {
            "timing_scope": TIMING_SCOPE,
            "detection_seconds": self.seconds,
            "regions": [dict(region) for region in self.regions],
            "isolation_overhead_seconds": sum(r.get("isolation_overhead_seconds", 0.0) for r in self.regions),
            "detector_peak_rss_bytes": int(peak),
            "detector_peak_rss_scope": scope,
            "harness_peak_rss_at_stop_bytes": self.harness_peak_rss_bytes,
            "rss_before_detector_bytes": self.rss_before_bytes,
            "user_cpu_seconds": self.user_cpu_seconds,
            "system_cpu_seconds": self.system_cpu_seconds,
            "child_user_cpu_seconds": self.child_user_cpu_seconds,
            "child_system_cpu_seconds": self.child_system_cpu_seconds,
        }


def current() -> DetectorClock | None:
    """The clock of the detector run in progress, or None outside :func:`detector_run`."""
    return _CURRENT.get()


def active() -> DetectorClock:
    """The clock an adapter times its detector with: the run's own, or a detached one when called outside a run."""
    return _CURRENT.get() or DetectorClock()


@contextlib.contextmanager
def detector_run() -> Iterator[DetectorClock]:
    """Open the clock of one detector run; the adapters reach it through :func:`active`."""
    clock = DetectorClock()
    token = _CURRENT.set(clock)
    try:
        yield clock
    finally:
        _CURRENT.reset(token)


def read_report(path: str | os.PathLike) -> float | None:
    """The ``detector_seconds`` a program wrote to ``path`` (JSON), or None when it wrote none."""
    import json

    try:
        return float(json.loads(Path(path).read_text(encoding="utf-8"))[REPORT_KEY])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_report(path: str | os.PathLike, seconds: float) -> None:
    """The worker side of :func:`read_report`."""
    import json

    Path(path).write_text(json.dumps({REPORT_KEY: float(seconds), "clock": "perf_counter"}), encoding="utf-8")
