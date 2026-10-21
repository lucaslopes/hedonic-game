"""Argument checks and runtime information for :class:`hedonic.Game` and
:mod:`hedonic.arrays`.

igraph is not imported with this module: a value can only be an igraph object
when igraph is already loaded, so the checks look it up in ``sys.modules``, and
the array path runs without it (only :func:`runtime_versions`, Game's, imports
it).
"""

from __future__ import annotations

import contextlib
import functools
import gc
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from igraph import Graph


# Every sum of payoff terms must stay below this (HIG_ARITHMETIC_BOUND in
# native/core/shared/inputs.c), so no payoff or error bound overflows.
ARITHMETIC_BOUND = sys.float_info.max / 4096.0

# Values that float() would read but that are not real numbers.
_NOT_REAL = (bool, np.bool_, str, bytes, complex, np.complexfloating, type(None))
_PLAIN_REALS = {float, int, np.float64, np.float32, np.int64, np.int32}


def _entry_error(where: str, index: int, value, problem: str) -> ValueError:
    shown = repr(value)
    if len(shown) > 40:
        shown = shown[:37] + "..."
    return ValueError(f"{where.format(index)} = {shown} {problem}")


def _loaded_igraph():
    """The igraph module if it is loaded, else None (nothing is imported)."""
    return sys.modules.get("igraph")


def _as_floats(values: list, where: str) -> np.ndarray:
    """A list of real numbers as float64, naming the first entry that is not one."""
    if set(map(type, values)) <= _PLAIN_REALS:
        try:
            return np.array(values, dtype=np.float64)
        except OverflowError:
            pass   # an int beyond the double range; named below
    floats = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if isinstance(value, _NOT_REAL):
            kind = "None" if value is None else type(value).__name__
            raise _entry_error(where, index, value, f"is not a real number ({kind})")
        try:
            floats[index] = float(value)
        except OverflowError:
            raise _entry_error(where, index, value, "is beyond the double range") from None
        except (TypeError, ValueError):
            igraph = _loaded_igraph()
            hint = (
                "; to use an attribute, pass its name"
                if igraph is not None and isinstance(value, (igraph.Edge, igraph.Vertex))
                else ""
            )
            raise _entry_error(
                where, index, value, f"is not a real number ({type(value).__name__}){hint}"
            ) from None
    return floats


def weight_vector(
    graph: Graph, values, name: str, *, per_vertex: bool, nonnegative: bool = False
) -> np.ndarray:
    """``values`` as a contiguous float64 array, one finite real per edge (vertex).

    ``values`` is a sequence (list, tuple, NumPy array, iterator) in edge (vertex)
    order, or the name of an edge (vertex) attribute. Booleans, strings, complex
    numbers, ``None`` and dicts are refused, naming the parameter and the entry.
    """
    what = "vertex" if per_vertex else "edge"
    count = graph.vcount() if per_vertex else graph.ecount()
    sequence = graph.vs if per_vertex else graph.es
    where = name + "[{}]"
    if isinstance(values, str):
        attributes = sequence.attributes()
        if values not in attributes:
            listed = ", ".join(repr(a) for a in attributes) or "none"
            raise ValueError(
                f"{name}: the graph has no {what} attribute {values!r} "
                f"(its {what} attributes: {listed})"
            )
        where = f"{name}: the {what} attribute {values!r} at {what} " + "{}"
        values = sequence[values]
    return real_vector(values, count, name, per_vertex=per_vertex,
                       nonnegative=nonnegative, where=where)


def real_vector(
    values, count: int, name: str, *, per_vertex: bool, nonnegative: bool = False,
    where: str | None = None,
) -> np.ndarray:
    """``values`` (a sequence, never an attribute name) as a contiguous float64
    array of ``count`` finite reals: the checks of :func:`weight_vector`, with
    its messages, for callers without an igraph graph."""
    what = "vertex" if per_vertex else "edge"
    if where is None:
        where = name + "[{}]"
    if isinstance(values, Mapping):
        raise ValueError(
            f"{name} must be a sequence with one number per {what}, in {what} "
            "order, not a dict (a dict would be read as its keys)"
        )
    igraph = _loaded_igraph()
    if igraph is not None and isinstance(values, (igraph.EdgeSeq, igraph.VertexSeq)):
        raise ValueError(
            f"{name} must be numbers, one per {what}, not an igraph "
            f"{type(values).__name__}; to use an attribute, pass its name"
        )
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional (one number per {what}), "
                f"got an array of shape {values.shape}"
            )
        kind = values.dtype.kind
        if kind in "iuf":
            array = values.astype(np.float64, copy=False)
        elif kind == "O":
            array = _as_floats(values.tolist(), where)
        else:
            raise ValueError(
                f"{name} must be real numbers, got an array of dtype {values.dtype}"
            )
    else:
        try:
            values = list(values)
        except TypeError:
            raise ValueError(
                f"{name} must be a sequence of numbers, one per {what}, "
                f"or the name of {'a vertex' if per_vertex else 'an edge'} attribute"
            ) from None
        array = _as_floats(values, where)
    if len(array) != count:
        raise ValueError(
            f"{name} must have one value per {what} ({count}), got {len(array)}"
        )
    finite = np.isfinite(array)
    if not finite.all():
        index = int(np.argmin(finite))
        raise _entry_error(where, index, float(array[index]), "is not finite")
    if nonnegative and (array < 0).any():
        index = int(np.argmax(array < 0))
        rule = (
            "vertex weights must be non-negative" if per_vertex
            else "a cover (max_memberships > 1) takes non-negative edge weights"
        )
        raise _entry_error(where, index, float(array[index]), f"is negative: {rule}")
    with np.errstate(over="ignore"):
        absolute = float(np.abs(array).sum())
    if not absolute <= ARITHMETIC_BOUND:
        raise ValueError(
            f"{name}: the sum of the absolute weights ({absolute:.3g}) exceeds "
            f"what the core computes with, about {ARITHMETIC_BOUND:.3g}; scale "
            "the weights down (the game is unchanged when every weight and the "
            "resolution are scaled by the same factor)"
        )
    return np.ascontiguousarray(array)


@contextlib.contextmanager
def gc_paused():
    """Hold the cyclic garbage collector while a block builds many small lists.

    One list per vertex triggers a collection every few hundred lists, and the
    older generations' passes traverse the whole heap: 93k one-element rows took
    24.7 ms with the collector and 2.4 ms without it (perflog). Nothing the
    block builds is cyclic garbage, and the collector runs again after it.
    """
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def sequential_sum(values: np.ndarray) -> float:
    """The left-to-right float sum of ``values`` (as the C core sums them)."""
    total = 0.0
    step = 1 << 20
    for start in range(0, len(values), step):
        chunk = np.concatenate(([total], values[start:start + step]))
        total = float(np.cumsum(chunk)[-1])
    return total


def total_edge_weight(graph: Graph, edge_weights=None) -> float:
    """Return the undirected total edge weight ``W`` used by ``Φ_γ = Φ̃_γ / W``.

    ``edge_weights is None`` means every edge has weight one, matching the
    native call.  An explicit sequence or edge-attribute name is summed in
    edge order.
    """
    if graph.ecount() == 0:
        return 0.0
    if edge_weights is None:
        return float(graph.ecount())
    if not isinstance(edge_weights, np.ndarray) or edge_weights.dtype != np.float64:
        edge_weights = weight_vector(graph, edge_weights, "edge_weights", per_vertex=False)
    return sequential_sum(edge_weights)


def node_weight_values(graph: Graph, node_weights) -> np.ndarray | None:
    """Validate vertex weights (finite, >= 0) as a float64 array."""
    if node_weights is None:
        return None
    return weight_vector(graph, node_weights, "node_weights", per_vertex=True, nonnegative=True)


def check_start_counts(
    rows: list[list[int]],
    source: str,
    max_total_communities: int | None,
    n_communities: int | None,
    projected: int | None = None,
) -> None:
    """Reject a start state that violates a community-count limit.

    The limits apply to the start the detector receives: after a projection
    to ``max_memberships`` (``projected`` vertices had to choose), which can
    leave fewer communities than the start as given. The native layer
    rejects such a start as well; checking here names the conflicting option
    instead of surfacing a generic error.
    """
    occupied = len({label for row in rows for label in row})
    state = (
        f"after projecting {projected} vertices to max_memberships it occupies"
        if projected else "it occupies"
    )
    if max_total_communities is not None and occupied > max_total_communities:
        raise ValueError(
            f"Invalid {source}: {state} {occupied} communities, more than "
            f"max_total_communities={max_total_communities}"
        )
    if n_communities is not None and occupied != n_communities:
        raise ValueError(
            f"Invalid {source}: {state} {occupied} communities, but "
            f"n_communities={n_communities} requires exactly that many"
        )


@functools.cache
def _core_versions() -> tuple[tuple[str, str | None], ...]:
    try:
        from importlib.metadata import version

        hedonic_version = version("hedonic")
    except Exception:  # pragma: no cover - metadata absent in a bare checkout
        hedonic_version = None
    from .. import _core

    return (
        ("hedonic", hedonic_version),
        ("hedonic_core", _core.core_version),
    )


@functools.cache
def _versions() -> tuple[tuple[str, str | None], ...]:
    import igraph

    return _core_versions() + (("python_igraph", getattr(igraph, "__version__", None)),)


def runtime_versions() -> dict[str, str | None]:
    """Versions of the packages that ran a detector call.

    Read once per process (the metadata lookup costs about 0.3 ms, a tenth of
    a small detector call) and returned as a fresh dict each time.
    """
    return dict(_versions())


def core_versions() -> dict[str, str | None]:
    """:func:`runtime_versions` without igraph: what ran an array call."""
    return dict(_core_versions())
