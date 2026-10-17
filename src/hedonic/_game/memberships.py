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


def validate_membership(game, membership: list[int] | list[list[int]]) -> bool:
    """Return whether a partition is valid for ``game``."""
    try:
        game._normalize_membership(membership)
    except (TypeError, ValueError):
        return False
    return True


def normalize_membership(
    game,
    membership: list[int] | list[list[int]],
) -> list[list[int]]:
    """Validate and normalize a partition to per-vertex rows of one label.

    A partition is given as flat labels (one per vertex) or as one-label rows.
    Labels are required to be non-negative integers and contiguous from zero,
    matching igraph's membership convention. Every row must hold exactly one
    label; duplicate labels within a row are rejected.
    """
    if not isinstance(membership, list):
        raise TypeError("memberships must be a list")
    if len(membership) != game.vcount():
        raise ValueError(
            f"memberships must contain one row per vertex ({game.vcount()})"
        )
    if not membership:
        raise ValueError("memberships cannot be empty")

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
            if len(labels) != len(set(labels)):
                raise ValueError("duplicate labels in a membership row")
            if len(labels) > 1:
                raise ValueError(
                    f"a partition has one label per vertex (a row holds {len(labels)})"
                )
        else:
            if isinstance(entry, (list, tuple)):
                raise ValueError("membership cannot mix flat and nested rows")
            if isinstance(entry, bool) or not isinstance(entry, Integral):
                raise ValueError("membership labels must be integers")
            value = int(entry)
            if value < 0:
                raise ValueError("membership labels must be non-negative")
            labels = [value]
        rows.append(labels)
        all_labels.update(labels)

    if all_labels:
        max_label = max(all_labels)
        if all_labels != set(range(max_label + 1)):
            raise ValueError("membership labels must be contiguous from 0")
    return rows

