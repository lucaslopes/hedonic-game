"""A memory watchdog for one detector run: stop the run that drives the machine below its free-memory floor, alone.

The runs' other guards act on a whole run: ``parallel.run_forked`` stops all its workers, and the worker guard
(``runmanager.start_memory_guard``) kills the run's whole process tree and ends it. Both leave nothing on record, so a
queue started again runs the same detector again and meets the same wall. An evaluation of Clique Percolation on
LiveJournal/top5000 lists every maximal clique and grows to tens of GB; two of them beside each other restarted a 64 GB
Mac. This module is the per-run half: while one detector runs, the free memory is read with the measure the floor
uses (``parallel.free_memory_percent``, ``HEDONIC_MEMORY_FLOOR``, default 10%), and below the floor that run alone is
stopped and recorded as ``status: "memory"`` with the floor and the free memory observed, the way a time limit records
``status: "timeout"`` (``limits``).

* A child the experiments fork or exec is watched from its parent, never from inside: :func:`join` for a forked child
  the parent joins (the calibrations' probes, the controlled study), :class:`Watch` for a process the parent waits on
  (``timing.DetectorClock.process``, from a thread, as its time limit is). At the floor the child's whole tree is
  killed: its process group and every process below it, including programs that started a session of their own.
* A detector that runs in this process (Hedonic and the igraph methods in the benchmark, a start-quality condition)
  has no parent to watch it, and the native code holds the GIL, so no thread could look. :func:`in_process` reads the
  free memory from a profiling-timer signal every ``POLL_SECONDS`` of CPU time (a few microseconds each) and raises
  :class:`MemoryLimitReached` inside the detector, which polls for signals; the run's memory is freed as it unwinds.

Polling is at most every ``POLL_SECONDS`` and changes no result. A stop is policy, never identity: no key, digest or
unit folder names the floor. A ``"memory"`` record stands like a timeout (``limits.stands``): it answers every later
run whose floor is at least the one that stopped it, so a queue does not run it again in a loop; a lower floor, or
``HEDONIC_RERUN_MEMORY=1`` (``hedonic run ... --rerun-memory``), runs it again. A run can also be *declared* out of
memory without running (``--out-of-memory METHOD@PAIR``): its record says so (``memory_declared``) and cites why.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from typing import Any, Callable, Iterator

from hedonic.experiments.overlapping import parallel

MEMORY_STATUS = "memory"
RERUN_ENV = "HEDONIC_RERUN_MEMORY"
POLL_SECONDS = 0.5

# Runs known to exhaust this kind of machine, cited by ``--out-of-memory METHOD@PAIR`` (a pair not listed here takes the
# generic reason). Keyed by (method, network:cover); the text is the evidence, recorded with every declared run.
DECLARED: dict[tuple[str, str], str] = {
    ("clique_percolation", "livejournal:top5000"): (
        "Clique Percolation lists every maximal clique of the graph; on LiveJournal/top5000 (84,438 vertices, 1,521,988 "
        "edges) each evaluation was killed by the OS for memory after about 4 minutes (child exit -9): k = 3 on "
        "2026-10-17 (baseline-calibration-rc1), k = 4 and k = 5 on 2026-10-20 (calibration-clique-percolation-large), "
        "and the next evaluation, beside Friendster's, exhausted the 64 GB machine and restarted it"),
    ("clique_percolation", "friendster:top5000"): (
        "Clique Percolation lists every maximal clique of the graph; Friendster/top5000 (220,015 vertices, 4,031,793 "
        "edges) is larger than LiveJournal/top5000, where every evaluation was killed by the OS for memory (2026-10-17 "
        "and 2026-10-20), and its first evaluation was running when the 64 GB machine ran out of memory and restarted "
        "on 2026-10-20"),
}


class MemoryLimitReached(Exception):
    """The machine's free memory fell below the floor while this run's detector was running; the run was stopped."""

    def __init__(self, free: float, floor: float):
        super().__init__(reason(free, floor))
        self.free = float(free)
        self.floor = float(floor)

    def fields(self) -> dict[str, Any]:
        return stopped(self.free, self.floor)


def reason(free: float, floor: float) -> str:
    return (f"stopped for memory: the machine's free memory fell to {free:.0f}% while it ran (floor {floor:g}%, "
            "HEDONIC_MEMORY_FLOOR); its processes were killed")


def stopped(free: float, floor: float) -> dict[str, Any]:
    """The fields of a run the watchdog stopped (``status``, the floor and the free memory it saw, ``reason``)."""
    return {"status": MEMORY_STATUS, "memory_floor_percent": float(floor), "free_memory_percent": float(free),
            "reason": reason(free, floor)}


def declared(text: str) -> dict[str, Any]:
    """The fields of a run declared out of memory, never started (``--out-of-memory``)."""
    return {"status": MEMORY_STATUS, "memory_declared": True, "reason": f"out of memory (declared, not run): {text}"}


def is_memory(record: dict | None) -> bool:
    return isinstance(record, dict) and record.get("status") == MEMORY_STATUS


def rerun_requested() -> bool:
    """``HEDONIC_RERUN_MEMORY`` set (``--rerun-memory``): runs a memory stop recorded are run again."""
    return bool(os.environ.get(RERUN_ENV, "").strip())


def stands(record: dict, floor: float | None = None) -> bool:
    """Whether a ``"memory"`` record still answers a run now: unless a rerun is asked, a declared one always does, and
    a stopped one while the floor is at least the one that stopped it (a lower floor, or none, runs it again)."""
    if rerun_requested():
        return False
    if record.get("memory_declared"):
        return True
    floor = parallel.memory_floor() if floor is None else float(floor)
    was = record.get("memory_floor_percent")
    return floor > 0 and isinstance(was, (int, float)) and floor >= float(was) - 1e-9


# --------------------------------------------------------------------------- declarations
def parse_declared(text: str | None, canonical: Callable[[str], str] = str) -> dict[tuple[str, str], str]:
    """``METHOD@NETWORK:COVER,...`` -> {(method, network:cover): reason}. ``canonical`` spells a pair as the run spells
    its networks (``top5k`` -> ``top5000``); the reason is the evidence of ``DECLARED`` or a generic one."""
    out: dict[tuple[str, str], str] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        method, at, pair = part.partition("@")
        if not at or not method.strip() or not pair.strip():
            raise ValueError(f"--out-of-memory takes METHOD@NETWORK:COVER items (e.g. clique_percolation@livejournal:top5000), "
                             f"not {part!r}")
        key = (method.strip(), canonical(pair.strip()))
        out[key] = DECLARED.get(key, f"{key[0]} on {key[1]} was declared out of memory for this machine (--out-of-memory)")
    return out


def spell_declared(declared_runs: dict[tuple[str, str], str]) -> str:
    return ",".join(f"{method}@{pair}" for method, pair in sorted(declared_runs))


# --------------------------------------------------------------------------- killing a tree
def descendants(pid: int) -> list[int]:
    """Every process below ``pid`` (children, grandchildren, ...), from one ``ps`` listing."""
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found, stack = [], [pid]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


def kill_tree(pid: int) -> None:
    """SIGKILL ``pid``, its process group (when it leads one that is not ours) and everything below it. An executable
    the clock starts leads a session of its own, and a CDlib worker is a grandchild of a forked probe: killing only the
    child would leave them running. Twice, so a process forked while the first pass ran is caught too."""
    own = os.getpgrp()
    for _ in range(2):
        for target in reversed([pid, *descendants(pid)]):
            with contextlib.suppress(OSError):
                group = os.getpgid(target)
                if group == target and group != own:
                    os.killpg(group, signal.SIGKILL)
            with contextlib.suppress(OSError):
                os.kill(target, signal.SIGKILL)


# --------------------------------------------------------------------------- a child, watched from its parent
class Watch:
    """Free memory, read for one child process ``pid`` by its parent; :meth:`look` kills the child's tree at the floor
    and keeps the stop in ``tripped`` (the record's fields), else None."""

    def __init__(self, pid: int, floor: float | None = None):
        self.pid = int(pid)
        self.floor = parallel.memory_floor() if floor is None else float(floor)
        self.tripped: dict[str, Any] | None = None

    @property
    def armed(self) -> bool:
        return self.floor > 0

    def look(self) -> bool:
        """True once the child was stopped for memory (now or earlier)."""
        if self.tripped is not None:
            return True
        if not self.armed:
            return False
        free = parallel.free_memory_percent()
        if free is None or free >= self.floor:
            return False
        kill_tree(self.pid)
        self.tripped = stopped(free, self.floor)
        return True

    @contextlib.contextmanager
    def thread(self, poll: float = POLL_SECONDS) -> Iterator["Watch"]:
        """Look every ``poll`` seconds from a daemon thread while the body waits for the child (``os.wait4`` releases the
        GIL); the thread ends with the body."""
        if not self.armed or parallel.free_memory_percent() is None:
            yield self
            return
        done = threading.Event()

        def run() -> None:
            while not done.wait(poll):
                if self.look():
                    return

        watcher = threading.Thread(target=run, name="memory-watch", daemon=True)
        watcher.start()
        try:
            yield self
        finally:
            done.set()
            watcher.join()


def join(process: Any, timeout: float | None, floor: float | None = None, poll: float = POLL_SECONDS) -> Watch:
    """``process.join(timeout)`` for a ``multiprocessing`` child, looking at the free memory every ``poll`` seconds.

    Returns the :class:`Watch`: ``tripped`` says the child was stopped for memory (its tree is killed and it is joined);
    otherwise the child exited, or is still alive at ``timeout`` (the caller handles a timeout as before)."""
    watch = Watch(process.pid, floor)
    if not watch.armed or parallel.free_memory_percent() is None:
        process.join(timeout)
        return watch
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    while True:
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            return watch
        process.join(poll if left is None else min(poll, left))
        if process.exitcode is not None:
            return watch
        if watch.look():
            process.join()
            return watch


# --------------------------------------------------------------------------- a detector in this process
@contextlib.contextmanager
def in_process(floor: float | None = None, poll: float = POLL_SECONDS) -> Iterator[Watch | None]:
    """Stop the detector run in the body with :class:`MemoryLimitReached` once free memory falls below the floor.

    A profiling timer (``ITIMER_PROF``: process CPU time, so it costs nothing while the run waits) raises ``SIGPROF``
    every ``poll`` seconds; its handler reads the free memory, a few microseconds, and raises below the floor. The
    native core and igraph poll for signals, so the exception surfaces inside them. Needs the main thread (only it runs
    signal handlers); elsewhere, with the floor at 0, or where free memory cannot be read, the body runs unwatched and
    ``None`` is yielded. The previous handler and timer are restored afterwards."""
    limit = parallel.memory_floor() if floor is None else float(floor)
    if (limit <= 0 or threading.current_thread() is not threading.main_thread()
            or parallel.free_memory_percent() is None):
        yield None
        return
    watch = Watch(os.getpid(), limit)

    def look(_signum, _frame) -> None:
        free = parallel.free_memory_percent()
        if free is not None and free < limit and watch.tripped is None:
            signal.setitimer(signal.ITIMER_PROF, 0)
            watch.tripped = stopped(free, limit)
            raise MemoryLimitReached(free, limit)

    previous = signal.signal(signal.SIGPROF, look)
    signal.setitimer(signal.ITIMER_PROF, poll, poll)
    try:
        yield watch
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)
