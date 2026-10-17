"""How much memory a step may use, and what it is expected to need, decided before the step starts.

Two earlier failures came from steps that never asked: a spectrum audit whose eight workers each built a
32 GB state on a 64 GB machine (261 GB attributed to the terminal, a reboot), and a validation that leaked
377 GB. ``parallel.run_forked`` and the worker guard (``runmanager.start_memory_guard``) stop a run that is
already over the floor; this module is the other half: a step estimates its own footprint, compares it with the
capacity that is available *now*, runs with as many workers as fit, and fails closed with a precise message when
not even one fits.

* ``budget_bytes`` is the memory a run may add: what is free now, minus the floor the guard enforces and a margin
  so the guard stays a last resort. ``HEDONIC_MEMORY_BUDGET_GB`` replaces the measurement (a fixed allowance).
* ``Need`` is a step's footprint: bytes shared by every worker (copy-on-write or memory-mapped, counted once) and
  bytes each worker adds.
* ``plan_workers`` returns the largest worker count whose ``Need`` fits the budget, or raises ``CapacityError``.

The exact stability audit needs no plan of its own: the C audit's prepared cover is O(n + m + incidences) and forked
audit workers share it, so the per-worker state that once had to be priced here no longer exists.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass

from hedonic.experiments.overlapping import parallel

BUDGET_ENV = "HEDONIC_MEMORY_BUDGET_GB"
MARGIN_PERCENT = 10.0  # extra percentage points of the machine kept free above the guard's floor
SAFETY = 1.25  # estimates are priced from samples and formulas: leave a quarter of slack before comparing


class CapacityError(RuntimeError):
    """A step needs more memory than the machine can give it right now."""


def total_bytes() -> int | None:
    """The machine's physical memory, or ``None`` when the platform does not say."""
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            value = ctypes.c_uint64(0)
            size = ctypes.c_size_t(ctypes.sizeof(value))
            if libc.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
                return int(value.value)
        except (OSError, AttributeError):
            return None
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def budget_bytes() -> int | None:
    """Memory a step may add right now, or ``None`` when it cannot be measured (the step then runs unplanned)."""
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if raw:
        try:
            return int(float(raw) * 1024**3)
        except ValueError:
            pass
    total, free = total_bytes(), parallel.free_memory_percent()
    if total is None or free is None:
        return None
    reserve = parallel.memory_floor() + MARGIN_PERCENT
    return max(0, int(total * (free - reserve) / 100.0))


def gib(value: float) -> str:
    return f"{value / 1024**3:.1f} GiB"


@dataclass(frozen=True)
class Need:
    """What a step holds: ``shared`` once for all workers, ``per_worker`` for each."""

    label: str
    shared: float
    per_worker: float

    def total(self, workers: int) -> float:
        return SAFETY * (self.shared + workers * self.per_worker)


@dataclass(frozen=True)
class Plan:
    workers: int
    requested: int
    budget: int | None
    need: Need

    @property
    def reduced(self) -> bool:
        return self.workers < self.requested

    def line(self) -> str:
        budget = "unmeasured" if self.budget is None else gib(self.budget)
        text = (f"{self.need.label}: {self.workers} worker(s) · needs {gib(self.need.total(self.workers))} "
                f"({gib(self.need.shared * SAFETY)} shared + {gib(self.need.per_worker * SAFETY)} per worker) · "
                f"budget {budget}")
        return text + (f" · reduced from {self.requested}" if self.reduced else "")


def plan_workers(need: Need, requested: int, budget: int | None = None) -> Plan:
    """The most workers (at most ``requested``) whose footprint fits the budget; ``CapacityError`` if not even one.

    With no measurable budget the request is returned as it is, so a platform that cannot say never blocks a run;
    the worker guard remains the floor there.
    """
    requested = max(1, int(requested))
    budget = budget_bytes() if budget is None else budget
    if budget is None:
        return Plan(requested, requested, None, need)
    workers = requested
    while workers >= 1 and need.total(workers) > budget:
        workers -= 1
    if workers < 1:
        raise CapacityError(
            f"{need.label} needs about {gib(need.total(1))} with one worker ({gib(need.shared * SAFETY)} shared + "
            f"{gib(need.per_worker * SAFETY)} per worker), but only {gib(budget)} can be used now (free memory minus "
            f"the {parallel.memory_floor():.0f}% floor and a {MARGIN_PERCENT:.0f}% margin; "
            f"{BUDGET_ENV} sets a fixed allowance). Close other applications, or run it on a larger machine")
    return Plan(workers, requested, budget, need)
