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
* ``sample_support_entries`` and ``state_bytes`` price one ``robustness.FractionalState`` from a sample of the
  graph before it is built.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

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


# --------------------------------------------------------------------------- pricing a fractional state
def sample_support_entries(graph, memberships: Sequence[Sequence[int]], sample: int = 2000, seed: int = 0) -> float:
    """Estimated number of (vertex, label) support entries of the cover on this graph, from a vertex sample.

    A vertex's support holds one entry for every distinct label among its neighbours, so the sample counts those
    for random vertices and scales by the vertex count. The estimate is unbiased; its spread is a few percent
    on the graphs of this study (a heavy tail of hub vertices is why the sample is not small).
    """
    n = graph.vcount()
    if n == 0:
        return 0.0
    rng = random.Random(seed)
    picked = range(n) if n <= sample else rng.sample(range(n), sample)
    total = 0
    for vertex in picked:
        labels: set[int] = set()
        for neighbour in graph.neighbors(vertex):
            labels.update(memberships[neighbour])
        total += len(labels)
    return total / len(picked) * n


def state_bytes(n_vertices: int, n_edges: int, incidences: int, support_entries: float, n_labels: int) -> float:
    """Resident size of a compact ``robustness.FractionalState`` (what workers share), in bytes."""
    support = support_entries * (4 + 8)  # int32 label + float64 value, CSR
    per_vertex = n_vertices * (8 + 8 + 64)  # indptr, row sizes, membership tuple header
    per_incidence = incidences * 8 * 2  # membership tuples and the label array they are cut from
    per_edge = n_edges * 8  # edge weights
    per_label = n_labels * (8 + 8 + 32)  # masses, lightest-label order, boxed floats
    return support + per_vertex + per_incidence + per_edge + per_label


def state_build_transient(n_edges: int, block: int, support_entries: float | None = None) -> float:
    """Peak extra bytes while a state is built: the edge list, the sorted contributions and one block's arrays.

    Measured on the 16.9 M-edge LiveJournal cover (3.2e8 support entries): the build peaked 9.5 GB above the loaded
    graph, 3.9 GB of which is the finished support; 400 bytes per edge plus 64 per block entry prices the rest.
    """
    if support_entries is not None:  # a block never holds more contributions than the whole build produces
        block = min(block, 2 * support_entries)
    return n_edges * 400 + block * 64


GIB = 1024 ** 3
ENTRY_BYTES_PER_VERTEX = 20_000  # candidate entries of one vertex, as the Python lists and arrays they pass through


def exact_audit_workers(graph, memberships: Sequence[Sequence[int]], requested: int, *, label: str = "audit",
                        say=print) -> int:
    """Forked workers an exact stability audit of this cover may use, from the memory free now.

    Raises ``CapacityError`` when the state cannot even be built. The state is built once and shared copy-on-write
    (its arrays are not reference-counted, so forks do not copy them); each worker adds the dicts and lists of the
    vertex it is on, and the candidate entries are built block by block (``stability.ENTRY_BLOCK`` vertices).
    """
    from hedonic.experiments.overlapping import robustness as rb
    from hedonic.experiments.overlapping import stability as st

    n, edges = graph.vcount(), graph.ecount()
    labels = max((label for row in memberships for label in row), default=-1) + 1
    support = sample_support_entries(graph, memberships)
    state = state_bytes(n, edges, sum(map(len, memberships)), support, labels)
    build = state_build_transient(edges, rb.SUPPORT_BLOCK, support)
    plan_workers(Need(f"{label}: building the audit state", state + build, 0), 1)
    plan = plan_workers(Need(f"{label}: exact stability audit", state + ENTRY_BYTES_PER_VERTEX * st.ENTRY_BLOCK,
                             0.5 * GIB), requested)
    if say is not None:
        say(f"  [capacity] {plan.line()}", flush=True)
    return plan.workers
