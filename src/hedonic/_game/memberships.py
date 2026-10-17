"""Membership-state normalization and validation for :class:`hedonic.Game`."""

from __future__ import annotations

from numbers import Integral


def memberships_as_cover(game) -> list[list[int]]:
    """Convert stored per-vertex memberships to community lists."""
    if game.memberships is None or game.memberships == []:
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


def normalize_membership(
    game,
    membership: list[int] | list[list[int]],
    *,
    max_memberships: int | None = None,
    collapse_duplicate_labels: bool = False,
) -> list[list[int]]:
    """Validate and normalize a partition or cover to per-vertex rows.

    Labels are required to be non-negative integers and contiguous from zero,
    matching igraph's membership convention. Every row must be non-empty and
    duplicate labels within a vertex are rejected. The optional cap is checked
    here so persisted state cannot silently be accepted for an incompatible
    detector call. Detector initialization may opt to collapse repeated labels
    within a row; ordinary stored-state validation remains strict.
    """
    if not isinstance(membership, list):
        raise TypeError("memberships must be a list")
    if len(membership) != game.vcount():
        raise ValueError(
            f"memberships must contain one row per vertex ({game.vcount()})"
        )
    if not membership:
        raise ValueError("memberships cannot be empty")
    if max_memberships is not None and (
        not isinstance(max_memberships, int) or max_memberships < 1
    ):
        raise ValueError("max_memberships must be an integer >= 1")

    first_nested = isinstance(membership[0], (list, tuple))
    rows: list[list[int]] = []
    all_labels: set[int] = set()
    for entry in membership:
        if first_nested:
            if not isinstance(entry, (list, tuple)):
                raise ValueError("membership cannot mix flat and nested rows")
            if len(entry) == 0:
                raise ValueError("each membership row must be non-empty")
            labels = []
            for label in entry:
                if isinstance(label, bool) or not isinstance(label, Integral):
                    raise ValueError("membership labels must be integers")
                value = int(label)
                if value < 0:
                    raise ValueError("membership labels must be non-negative")
                labels.append(value)
            if len(labels) != len(set(labels)) and not collapse_duplicate_labels:
                raise ValueError("duplicate labels in a membership row")
            if collapse_duplicate_labels:
                labels = list(dict.fromkeys(labels))
        else:
            if isinstance(entry, (list, tuple)):
                raise ValueError("membership cannot mix flat and nested rows")
            if isinstance(entry, bool) or not isinstance(entry, Integral):
                raise ValueError("membership labels must be integers")
            value = int(entry)
            if value < 0:
                raise ValueError("membership labels must be non-negative")
            labels = [value]
        if max_memberships is not None and len(labels) > max_memberships:
            raise ValueError(
                "a vertex exceeds max_memberships "
                f"({len(labels)} > {max_memberships})"
            )
        rows.append(labels)
        all_labels.update(labels)

    if all_labels:
        max_label = max(all_labels)
        if all_labels != set(range(max_label + 1)):
            raise ValueError("membership labels must be contiguous from 0")
    return rows


def as_overlapping_init(
    membership: list[int] | list[list[int]],
) -> list[list[int]]:
    """Normalize flat or nested membership to per-vertex lists."""
    if membership and isinstance(membership[0], (list, tuple)):
        return [list(map(int, entry)) for entry in membership]
    return [[int(community)] for community in membership]
