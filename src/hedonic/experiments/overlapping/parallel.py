"""Process-level parallelism for the experiment runners: forked one-shot workers and a machine-wide core budget.

The detectors and the scoring code are single-threaded Python/C, so the cores of a machine are used by running
independent conditions in separate processes. Two small tools do that without threads (forking a process that has
threads is unsafe, and the metric code holds the GIL):

* :func:`run_forked` calls ``fn(item)`` for every item in a forked child, at most ``workers`` at a time, and yields
  ``(index, result)`` as each child finishes. The children inherit the parent's memory copy-on-write, so a loaded graph
  is shared and nothing is pickled on the way in; only the (picklable) result travels back, through a file. With
  ``workers == 1`` it calls ``fn`` in this process, so single-worker behaviour, tracebacks and memory are exactly those
  of a plain loop.
* :class:`Slots` is a cross-process counting semaphore made of ``flock`` files. Several independent processes that
  each fork their own children (the four searches of the Hedonic calibration) take a slot around every unit of work, so
  together they never run more than ``size`` units at once. The kernel drops a lock when its process dies, so a killed
  run cannot leave a slot taken.

Timing-sensitive code (the benchmark's detector runs, whose time and peak memory are results) must not use this: a
run measured next to seven others is not the run the paper reports.

Memory. Workers share the loaded graph copy-on-write, but each also builds its own start, cover and scores, so N workers
need more than one. :func:`run_forked` therefore watches the machine's free memory (``kern.memorystatus_level`` on macOS,
``MemAvailable`` on Linux) while it waits, and when it falls below a floor (default 10%, ``HEDONIC_MEMORY_FLOOR``, ``0``
turns it off) it kills its workers and raises :class:`MemoryLow`, instead of letting the machine swap or restart. The
items that had finished were already yielded, so a resumable run loses only what was in flight.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import fcntl
import functools
import multiprocessing as mp
import os
import pickle
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import connection
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

WORKERS_ENV = "HEDONIC_WORKERS"
MEMORY_FLOOR_ENV = "HEDONIC_MEMORY_FLOOR"
DEFAULT_MEMORY_FLOOR = 10.0  # percent of the machine's memory that must stay free


class WorkerFailed(RuntimeError):
    """A forked worker raised, or died without a result (killed, out of memory)."""


class MemoryLow(WorkerFailed):
    """The machine's free memory fell below the floor; the workers were stopped before it ran out."""


# --------------------------------------------------------------------------- how many workers
@functools.lru_cache(maxsize=1)
def performance_cores() -> int:
    """Performance cores of this machine (Apple silicon separates them from the efficiency cores), else all CPUs."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"], capture_output=True, text=True, timeout=5)
            count = int(out.stdout.strip())
            if count > 0:
                return count
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return os.cpu_count() or 1


def default_workers(requested: int | None = None) -> int:
    """``requested`` when given, else ``HEDONIC_WORKERS``, else the performance cores (8 on an M1 Max)."""
    if requested is not None:
        return max(1, int(requested))
    raw = os.environ.get(WORKERS_ENV, "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return performance_cores()


# --------------------------------------------------------------------------- free memory
@functools.lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    """The C library, opened once: ``find_library`` alone costs tens of microseconds, and the memory watchdog
    (``memwatch``) reads the free memory twice a second, inside a detector run."""
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def free_memory_percent() -> float | None:
    """Percentage of the machine's memory that is free or reclaimable, or ``None`` when this platform does not say."""
    if sys.platform == "darwin":
        try:
            libc = _libc()
            value = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(value))
            if libc.sysctlbyname(b"kern.memorystatus_level", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
                return float(value.value)
        except (OSError, AttributeError):
            pass
        return None
    try:
        info = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])
        return 100.0 * info["MemAvailable"] / info["MemTotal"]
    except (OSError, KeyError, ValueError, IndexError):
        return None


def memory_floor() -> float:
    """The free-memory floor in percent: ``HEDONIC_MEMORY_FLOOR``, else 10; 0 disables the check."""
    raw = os.environ.get(MEMORY_FLOOR_ENV, "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return DEFAULT_MEMORY_FLOOR


LAST_RESORT_SECONDS = 2.0  # how long a whole-run guard lets free memory stay below the floor before it acts


class FloorWatch:
    """When a guard of a whole run (:func:`run_forked`, the worker guard) stops everything: at once below half the
    floor, else once free memory has stayed below the floor for ``LAST_RESORT_SECONDS``.

    Such a guard records nothing for the runs it kills, so a queue started again meets the same wall. The per-run
    watchdog (``memwatch``) reads the same measure at least as often and stops only the run that is growing, which is
    recorded as out of memory; the grace lets it act first, and lets the memory it frees show, unless the machine is
    already close to running out."""

    def __init__(self, floor: float):
        self.floor = float(floor)
        self.since: float | None = None

    def breached(self, free: float | None) -> bool:
        if free is None or free >= self.floor:
            self.since = None
            return False
        if free < self.floor / 2:
            return True
        now = time.monotonic()
        if self.since is None:
            self.since = now
        return now - self.since >= LAST_RESORT_SECONDS


# --------------------------------------------------------------------------- forked one-shot workers
def _child(fn: Callable[[Any], Any], item: Any, path: str) -> None:
    try:
        payload: tuple[bool, Any] = (True, fn(item))
    except BaseException:  # the parent decides what a failure means; the child only reports it
        payload = (False, traceback.format_exc())
    part = path + ".part"
    with open(part, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(part, path)


def _outcome(path: str, exitcode: int | None) -> tuple[bool, Any]:
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return False, f"the worker exited with status {exitcode} without a result (killed or out of memory)"


def run_forked(fn: Callable[[Any], Any], items: Iterable[Any], workers: int,
               floor: float | None = None, admit: Callable[[int], bool] | None = None,
               stagger: float = 0.0, alone: Callable[[Any], bool] | None = None) -> Iterator[tuple[int, Any]]:
    """Yield ``(index, fn(item))`` as results arrive, running at most ``workers`` forked children at a time.

    Results come back in completion order; the index is the item's position, so a caller that needs the input order
    stores by index. When a worker fails, no new item is started, the running ones are allowed to finish (their results
    are still yielded, so finished work can be recorded), and :class:`WorkerFailed` is raised with the first traceback.
    Leaving the generator early, or an exception in the caller, kills the running children. While children run, the
    machine's free memory is read about once a second; below ``floor`` percent (default :func:`memory_floor`) the children
    are killed and :class:`MemoryLow` is raised, at once below half the floor and otherwise after
    ``LAST_RESORT_SECONDS`` below it (:class:`FloorWatch`), so a run's own watchdog can stop and record it first (the
    single-worker path runs in this process and has no such check).

    ``admit(running)`` and ``stagger`` decide *when* the next item may start, never how many run at most: a new child
    starts only while fewer than ``workers`` run, ``stagger`` seconds have passed since the last start, and ``admit`` (given
    the number running) says yes. One item always starts when none is running, so the queue cannot stall. A caller whose
    items grow slowly to their peak memory uses them to see one child's footprint before the next begins.

    ``alone(item)`` true marks an item that runs with nothing beside it (a pair too large to share the machine): it
    starts only when no child runs, and no other item starts while it runs. Items keep their order.
    """
    work = list(items)
    if not work:
        return
    workers = max(1, min(int(workers), len(work)))
    if workers == 1:
        for index, item in enumerate(work):
            yield index, fn(item)
        return
    context = mp.get_context("fork")
    folder = tempfile.mkdtemp(prefix="hedonic-forked-")
    running: dict[int, tuple[Any, str]] = {}
    queue = iter(enumerate(work))
    failure: tuple[int, str] | None = None
    limit = memory_floor() if floor is None else float(floor)
    guard = FloorWatch(limit)
    last_look = float("-inf")
    last_start = float("-inf")
    waiting: tuple[int, Any] | None = None  # the next item, held back until the running ones are done (``alone``)
    solo = False  # the running child is an item that runs alone
    try:
        while True:
            if limit > 0 and time.monotonic() - last_look >= 1.0:
                last_look = time.monotonic()
                free = free_memory_percent()
                if guard.breached(free):
                    raise MemoryLow(
                        f"free memory fell to {free:.0f}% (floor {limit:.0f}%): stopped {len(running)} running worker(s); "
                        f"rerun with fewer workers (the finished items are kept)")
            while failure is None and len(running) < workers:
                if running and (time.monotonic() - last_start < stagger or (admit is not None and not admit(len(running)))):
                    break
                if running and (solo or waiting is not None):
                    break  # an item that runs alone is running, or is next and waits for the others to finish
                nxt = waiting if waiting is not None else next(queue, None)
                waiting = None
                if nxt is None:
                    break
                index, item = nxt
                if running and alone is not None and alone(item):
                    waiting = nxt
                    break
                solo = alone is not None and alone(item)
                path = os.path.join(folder, f"{index}.pkl")
                process = context.Process(target=_child, args=(fn, item, path))
                process.start()
                last_start = time.monotonic()
                running[index] = (process, path)
            if not running:
                break
            ready = connection.wait([process.sentinel for process, _ in running.values()], timeout=1.0)
            for index in [i for i, (process, _) in running.items() if process.sentinel in ready]:
                process, path = running.pop(index)
                process.join()
                ok, value = _outcome(path, process.exitcode)
                if ok:
                    yield index, value
                elif failure is None:
                    failure = (index, str(value))
        if failure is not None:
            raise WorkerFailed(f"worker for item {failure[0]} failed:\n{failure[1]}")
    finally:
        for process, _ in running.values():
            if process.is_alive():
                process.kill()
            process.join()
        shutil.rmtree(folder, ignore_errors=True)


def map_forked(fn: Callable[[Any], Any], items: Sequence[Any], workers: int) -> list[Any]:
    """``[fn(item) for item in items]`` in input order, computed by :func:`run_forked`."""
    out: list[Any] = [None] * len(items)
    for index, value in run_forked(fn, items, workers):
        out[index] = value
    return out


# --------------------------------------------------------------------------- a machine-wide core budget
_HELD: set[int] = set()  # slot descriptors this process holds; a forked child closes its copies (see below)


def _close_inherited() -> None:
    for fd in list(_HELD):
        with contextlib.suppress(OSError):
            os.close(fd)  # closing a duplicate never releases the parent's lock (no LOCK_UN: that would)
    _HELD.clear()


os.register_at_fork(after_in_child=_close_inherited)


class Slots:
    """``size`` cross-process slots, each a lock file in ``directory``; hold one around a unit of work.

    A lock belongs to an open file description, which a fork shares, so a child must not keep a slot of its parent
    alive: forked children close the descriptors they inherited. The work a holder forks therefore ends (or is
    killed) while the holder still holds the slot, which is how the runners use it.
    """

    def __init__(self, directory: str | Path, size: int, poll: float = 0.2):
        self.directory = Path(directory)
        self.size = max(1, int(size))
        self.poll = poll
        self.directory.mkdir(parents=True, exist_ok=True)

    def _try(self) -> int | None:
        order = list(range(self.size))
        random.shuffle(order)  # spread the contenders over the files
        for number in order:
            fd = os.open(self.directory / f"slot-{number}.lock", os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                continue
            return fd
        return None

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        fd = self._try()
        while fd is None:
            time.sleep(self.poll)
            fd = self._try()
        _HELD.add(fd)
        try:
            yield
        finally:
            _HELD.discard(fd)
            os.close(fd)  # releases the lock
