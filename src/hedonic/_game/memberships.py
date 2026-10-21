"""Membership-state normalization and validation for :class:`hedonic.Game`."""

from __future__ import annotations

import sys
from itertools import chain
from numbers import Integral

from .helpers import gc_paused


def memberships_as_cover(game) -> list[list[int]]:
    """Convert stored per-vertex memberships to community lists."""
    if not game.memberships:
        raise ValueError(
            "Game has no memberships; run community_hedonic or assign "
            "Game.memberships before evaluating it"
        )
    try:
        rows = game._normalize_membership(game.memberships)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Game.memberships is invalid: {exc}") from exc

    communities: dict[int, list[int]] = {}
    for vertex, labels in enumerate(rows):
        for label in labels:
            communities.setdefault(label, []).append(vertex)
    return [communities[label] for label in sorted(communities)]


def validate_membership(
    game,
    membership: list[int] | list[list[int]],
    *,
    overlapping: bool = False,
    max_memberships: int | None = None,
) -> bool:
    """Return whether a partition or overlapping cover is valid for ``game``."""
    try:
        rows = game._normalize_membership(
            membership, max_memberships=max_memberships
        )
    except (TypeError, ValueError):
        return False
    if not overlapping and any(len(row) != 1 for row in rows):
        return False
    return True


def _rows_of(membership):
    """Per-vertex entries of a membership given as a list, tuple, NumPy array or
    igraph clustering/cover (only when igraph is loaded: nothing else can make
    one, so the array path never imports it)."""
    igraph = sys.modules.get("igraph")
    if igraph is not None and isinstance(membership, igraph.VertexClustering):
        return list(membership.membership), membership.graph
    if igraph is not None and isinstance(membership, igraph.VertexCover):
        # Its rows as they are: normalize_rows copies every row it keeps.
        return membership.membership, membership.graph
    if hasattr(membership, "tolist") and not isinstance(membership, (list, tuple)):
        membership = membership.tolist()   # a NumPy array
    return membership, None


def normalize_membership(
    game,
    membership: list[int] | list[list[int]],
    *,
    max_memberships: int | None = None,
    collapse_duplicate_labels: bool = False,
    renumber_labels: bool = False,
    name: str = "memberships",
) -> list[list[int]]:
    """:func:`normalize_rows` for the vertices of ``game``."""
    return normalize_rows(
        membership, game.vcount(), max_memberships=max_memberships,
        collapse_duplicate_labels=collapse_duplicate_labels,
        renumber_labels=renumber_labels, name=name,
    )


def normalize_rows(
    membership: list[int] | list[list[int]],
    n: int,
    *,
    max_memberships: int | None = None,
    collapse_duplicate_labels: bool = False,
    renumber_labels: bool = False,
    name: str = "memberships",
) -> list[list[int]]:
    """Validate and normalize a partition or cover of ``n`` vertices to per-vertex rows.

    Labels are required to be non-negative integers and contiguous from zero,
    matching igraph's membership convention. Every row must be non-empty and
    duplicate labels within a vertex are rejected. The optional cap is checked
    here so persisted state cannot silently be accepted for an incompatible
    detector call. Detector initialization may opt to collapse repeated labels
    within a row and to renumber labels with gaps in their order (labels are
    names: 0, 2, 5 become 0, 1, 2); ordinary stored-state validation remains
    strict. An igraph ``VertexClustering`` or ``VertexCover`` of a graph with
    the same vertex count stands for its per-vertex memberships.
    """
    membership, source_graph = _rows_of(membership)
    if not isinstance(membership, (list, tuple)):
        raise TypeError(
            f"{name} must be a list (or tuple, NumPy array, or igraph "
            "VertexClustering/VertexCover) with one entry per vertex, got "
            f"{type(membership).__name__}"
        )
    if source_graph is not None and source_graph.vcount() != n:
        raise ValueError(
            f"{name} is a clustering of a graph with {source_graph.vcount()} "
            f"vertices; this graph has {n}"
        )
    if len(membership) != n:
        raise ValueError(
            f"{name} must contain one row per vertex ({n}), "
            f"got {len(membership)}: give each vertex its label (or its list "
            "of labels), not a list of communities"
        )
    if not membership:
        raise ValueError(f"{name} cannot be empty")
    if max_memberships is not None and (
        isinstance(max_memberships, bool)
        or not isinstance(max_memberships, Integral)
        or max_memberships < 1
    ):
        raise ValueError("max_memberships must be an integer >= 1")

    with gc_paused():
        plain = _plain_rows(membership, max_memberships, collapse_duplicate_labels)
    if plain is not None:
        rows, all_labels = plain
    else:
        rows, all_labels = _checked_rows(membership, name, max_memberships,
                                         collapse_duplicate_labels)

    if all_labels and max(all_labels) != len(all_labels) - 1 and renumber_labels:
        new = {label: index for index, label in enumerate(sorted(all_labels))}
        with gc_paused():
            return [[new[label] for label in row] for row in rows]
    if all_labels and max(all_labels) != len(all_labels) - 1:
        # Labels are non-negative and distinct, so they are 0 .. k-1
        # exactly when the largest is k - 1.
        missing = min(set(range(max(all_labels) + 1)) - all_labels) \
            if max(all_labels) < 10 * len(membership) else None
        raise ValueError(
            f"{name} labels must be contiguous from 0"
            + (f" (label {missing} is unused)" if missing is not None else "")
        )
    return rows


def _plain_rows(membership, max_memberships, collapse_duplicate_labels):
    """The rows and label set of a membership of plain ``int`` labels (flat, or
    rows that are all lists or tuples) that passes every check of
    :func:`_checked_rows`, without its per-label checks; None for anything
    else, which :func:`_checked_rows` then reads (and names what is wrong).

    ``type(x) is int`` admits exactly the labels the per-label check accepts
    as they are (``int(x)`` is ``x``; a bool's type is ``bool``), so the rows
    are the same lists of the same values.
    """
    kinds = set(map(type, membership))
    if kinds == {int}:
        if min(membership) < 0:
            return None
        return [[label] for label in membership], set(membership)
    if not kinds <= {list, tuple}:
        return None
    lengths = list(map(len, membership))
    if min(lengths) == 0:
        return None
    labels = list(chain.from_iterable(membership))
    if set(map(type, labels)) != {int} or min(labels) < 0:
        return None
    if max(lengths) == 1:
        rows = list(map(list, membership))
    else:
        rows = [list(dict.fromkeys(row)) for row in membership]
        if not collapse_duplicate_labels and any(map(int.__ne__, map(len, rows), lengths)):
            return None
        if max_memberships is not None and max(map(len, rows)) > max_memberships:
            return None
    return rows, set(labels)


def _checked_rows(membership, name, max_memberships, collapse_duplicate_labels):
    """The rows and label set of ``membership``, checked label by label: the
    first vertex that breaks a rule is named."""

    def row_of(entry):
        if isinstance(entry, (list, tuple)):
            return entry
        if hasattr(entry, "tolist") and getattr(entry, "ndim", 0) == 1:
            return entry.tolist()   # a NumPy row
        return None

    first_nested = row_of(membership[0]) is not None
    rows: list[list[int]] = []
    all_labels: set[int] = set()
    for vertex, entry in enumerate(membership):
        if first_nested:
            entry = row_of(entry)
            if entry is None:
                raise ValueError(
                    f"{name} cannot mix flat and nested rows (vertex {vertex})"
                )
            if len(entry) == 0:
                raise ValueError(
                    f"each {name} row must be non-empty (vertex {vertex} has none)"
                )
            labels = []
            for label in entry:
                if isinstance(label, bool) or not isinstance(label, Integral):
                    raise ValueError(
                        f"{name} labels must be integers (vertex {vertex} has {label!r})"
                    )
                value = int(label)
                if value < 0:
                    raise ValueError(
                        f"{name} labels must be non-negative (vertex {vertex} has {value})"
                    )
                labels.append(value)
            if len(labels) != len(set(labels)) and not collapse_duplicate_labels:
                raise ValueError(f"duplicate labels in the {name} row of vertex {vertex}")
            if collapse_duplicate_labels:
                labels = list(dict.fromkeys(labels))
        else:
            if row_of(entry) is not None:
                raise ValueError(
                    f"{name} cannot mix flat and nested rows (vertex {vertex})"
                )
            if isinstance(entry, bool) or not isinstance(entry, Integral):
                raise ValueError(
                    f"{name} labels must be integers (vertex {vertex} has {entry!r})"
                )
            value = int(entry)
            if value < 0:
                raise ValueError(
                    f"{name} labels must be non-negative (vertex {vertex} has {value})"
                )
            labels = [value]
        if max_memberships is not None and len(labels) > max_memberships:
            raise ValueError(
                f"vertex {vertex} exceeds max_memberships "
                f"({len(labels)} > {max_memberships})"
            )
        rows.append(labels)
        all_labels.update(labels)
    return rows, all_labels


def as_overlapping_init(
    membership: list[int] | list[list[int]],
) -> list[list[int]]:
    """Normalize flat or nested membership to per-vertex lists."""
    if membership and isinstance(membership[0], (list, tuple)):
        return [list(map(int, entry)) for entry in membership]
    return [[int(community)] for community in membership]
