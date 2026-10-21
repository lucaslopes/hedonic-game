"""Time limits of a queue: ``timeout`` (interrupt a run at that wall-clock) and ``max_eta`` (skip a run estimated longer).

Both apply to **one run**: one detector run of a benchmark unit, one evaluation of a calibration search, one condition
of a start-quality curve. Never to the sum of several: a Hedonic calibration that tries thirty resolutions, each well
inside the limit, is never interrupted because the thirty together took longer, and a pair is never skipped because all
its runs together would. Each runner measures its own runs (a forked child joined with a timeout, or a wall-clock timer
around the in-process detector), so a limit means the same thing wherever it applies.

A limit is a policy, never part of a record's identity: it only lowers the runner's own per-run timeout
(``min(own, limit)``), and a run it stopped is recorded as ``status: "timeout"`` with ``limit_seconds``. Such a record
stands for every later run whose limit is at most that one; a run with a larger limit (or none) runs it again.

In a queue file::

    [limits]                 # every item of this queue (and of the queues it runs)
    timeout = "2h"           # a single run still going after 2 hours is interrupted
    max_eta = "6h"           # a single run estimated at more than 6 hours is skipped

    [[items]]
    name = "benchmark-angel"
    limits = { timeout = "30m" }   # this item only; "none" removes an inherited limit

An item's own limits win over its queue's, and a queue's over the queue that runs it.
"""

from __future__ import annotations

import contextlib
import re
import signal
import threading
import time
from typing import Any, Iterator

KEYS = ("timeout", "max_eta")
TIMEOUT_STATUS = "timeout"
MEMORY_STATUS = "memory"  # a run the memory watchdog stopped, or one declared out of memory (``memwatch``)
_UNITS = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([smhd])")


def parse_duration(value: Any) -> float | None:
    """Seconds from ``90``, ``"90s"``, ``"30m"``, ``"2h"``, ``"1h30m"``, ``"1.5d"``; None from ``"none"``/``"off"``."""
    if isinstance(value, bool):
        raise ValueError(f"a duration is a number of seconds or text such as \"2h\", not {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip().lower().replace(" ", "")
        if text in ("none", "off", "no"):
            return None
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            seconds = float(text)
        else:
            parts = _PART.findall(text)
            if not parts or "".join(f"{n}{u}" for n, u in parts) != text:
                raise ValueError(f"cannot read the duration {value!r}: write seconds or e.g. \"90s\", \"30m\", \"2h\", \"1h30m\"")
            seconds = sum(float(n) * _UNITS[u] for n, u in parts)
    if not seconds > 0:
        raise ValueError(f"a duration must be positive, not {value!r}")
    return seconds


def fmt(seconds: float | None) -> str:
    """``2h``, ``1h30m``, ``45m``, ``90s``: the shortest exact spelling (``none`` for no limit)."""
    if seconds is None:
        return "none"
    whole = int(round(seconds))
    if abs(seconds - whole) > 1e-9:
        return f"{seconds:g}s"
    out = ""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        if whole >= size:
            count, whole = divmod(whole, size)
            out += f"{count}{unit}"
    return out or "0s"


def parse(table: Any, where: str) -> dict[str, float | None]:
    """A ``[limits]`` table (or an item's ``limits``): only the keys it sets, each in seconds (None: removed)."""
    if table is None:
        return {}
    if not isinstance(table, dict):
        raise SystemExit(f"{where}: limits must be a table, for example limits = {{ timeout = \"2h\" }}")
    unknown = sorted(set(table) - set(KEYS))
    if unknown:
        raise SystemExit(f"{where}: limits takes {', '.join(KEYS)}, not {', '.join(unknown)}")
    try:
        return {key: parse_duration(table[key]) for key in KEYS if key in table}
    except ValueError as exc:
        raise SystemExit(f"{where}: {exc}") from None


def merge(*layers: dict[str, float | None]) -> dict[str, float | None]:
    """Outermost first: each later layer overrides the keys it sets (None removes a limit). Unset keys are dropped."""
    out: dict[str, float | None] = {}
    for layer in layers:
        out.update(layer or {})
    return {key: value for key, value in out.items() if value is not None}


def effective(own: float | None, limit: float | None) -> float | None:
    """The per-run timeout a runner applies: its own, lowered (never raised) by the queue's limit."""
    if limit is None:
        return own
    return limit if own is None else min(float(own), float(limit))


def describe(limits: dict[str, float | None]) -> str:
    """``timeout 2h · max_eta 6h`` (empty without limits)."""
    return " · ".join(f"{key} {fmt(limits[key])}" for key in KEYS if limits.get(key) is not None)


class RunLimitReached(Exception):
    """Raised inside a run that reached its wall-clock limit (see :func:`wall_clock`)."""

    def __init__(self, seconds: float):
        super().__init__(f"stopped at its time limit of {fmt(seconds)} (one run)")
        self.seconds = seconds


@contextlib.contextmanager
def wall_clock(seconds: float | None) -> Iterator[bool]:
    """Interrupt the body with :class:`RunLimitReached` once ``seconds`` of wall-clock have passed.

    For a detector that runs in this process: the native core polls for signals, so the timer's handler raises inside
    it. Yields whether the limit is armed: it needs the main thread (no other thread receives signals); without it, or
    without a limit, the body runs unbounded. The previous handler and timer are restored afterwards.
    """
    if seconds is None or threading.current_thread() is not threading.main_thread():
        yield False
        return

    def stop(_signum, _frame):
        raise RunLimitReached(seconds)

    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def stands(record: dict, limit: float | None, default: float | None = None) -> bool:
    """Whether a record a time limit stopped still answers a run under ``limit``: only when that limit is at most the
    one that stopped it (``default``: the limit a record from before ``limit_seconds`` was written ran under).

    A run the memory watchdog stopped (``status: "memory"``, ``memwatch``) stands the same way, by its floor: while
    the floor is at least the one that stopped it, and unless ``--rerun-memory`` asks for it again."""
    if record.get("status") == MEMORY_STATUS:
        from hedonic.experiments.overlapping import memwatch

        return memwatch.stands(record)
    if record.get("status") != TIMEOUT_STATUS:
        return True
    stopped = record.get("limit_seconds", default)
    return stopped is not None and limit is not None and float(limit) <= float(stopped) + 1e-9


def timed_out(seconds: float, started: float | None = None) -> dict[str, Any]:
    """The fields of a run a limit stopped."""
    out: dict[str, Any] = {"status": TIMEOUT_STATUS, "limit_seconds": float(seconds),
                           "reason": f"stopped at its time limit of {fmt(seconds)} (one run)"}
    if started is not None:
        out["seconds"] = time.monotonic() - started
    return out
