"""Redirect what one thread prints, leaving every other thread's output where it was.

``contextlib.redirect_stdout`` swaps ``sys.stdout`` for the whole process. The queue view builds its tree on a thread of
its own, and the state checks it runs silence the planners they call; with ``redirect_stdout`` that silenced the main
thread too, so the "reopen / stop" lines printed after the view closed went into the sink while a rebuild was under way
(a rebuild of a large queue takes seconds and starts again every 1.5 s, so closing the view usually lost them).

``redirected(stdout=..., stderr=...)`` installs, once, a router on ``sys.stdout`` and ``sys.stderr`` that sends a thread's
writes to the sink it pushed (``DROP`` discards them, or any object with ``write``) and everything else to the stream
that was there.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any, Iterator

DROP = object()  # a sink that discards what is written
KEEP = object()  # leave this stream as it is for the thread

_LOCAL = threading.local()
_LOCK = threading.Lock()


class _Router:
    def __init__(self, name: str, target: Any):
        self._name, self._target = name, target

    def _sink(self) -> Any:
        stack = getattr(_LOCAL, self._name, None)
        return stack[-1] if stack else self._target

    def write(self, text: str) -> int:
        sink = self._sink()
        return len(text) if sink is DROP else sink.write(text)

    def flush(self) -> None:
        sink = self._sink()
        if sink is not DROP and hasattr(sink, "flush"):
            sink.flush()

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target, attr)


def _install() -> None:
    with _LOCK:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if stream is not None and not isinstance(stream, _Router):
                setattr(sys, name, _Router(name, stream))


@contextlib.contextmanager
def redirected(stdout: Any = DROP, stderr: Any = DROP) -> Iterator[None]:
    """This thread's ``stdout`` and ``stderr`` go to the given sinks (``DROP``: discarded, ``KEEP``: unchanged)."""
    _install()
    pushed = []
    for name, sink in (("stdout", stdout), ("stderr", stderr)):
        if sink is KEEP:
            continue
        stack = getattr(_LOCAL, name, None)
        if stack is None:
            stack = []
            setattr(_LOCAL, name, stack)
        stack.append(sink)
        pushed.append(stack)
    try:
        yield
    finally:
        for stack in pushed:
            stack.pop()
