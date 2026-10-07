"""Tests for persistent, canonical Game membership state."""

from __future__ import annotations

from unittest.mock import patch

import igraph as ig
import pytest

from hedonic import Game


class _Result:
    def __init__(self, membership):
        self.membership = membership


def _game() -> Game:
    return Game(ig.Graph.Ring(3))


def test_memberships_are_canonical_list_of_lists_and_flat_assignment_normalizes():
    game = _game()
    assert game.memberships is None
    game.memberships = [0, 0, 1]
    assert game.memberships == [[0], [0], [1]]
    game.memberships = [[0], [0, 1], [1]]
    assert game.memberships == [[0], [0, 1], [1]]


def test_copying_a_game_preserves_membership_state_without_aliasing_rows():
    game = _game()
    game.memberships = [[0], [0, 1], [1]]
    copied = Game(game)
    assert copied.memberships == game.memberships
    assert copied.memberships is not game.memberships
    assert copied.memberships[1] is not game.memberships[1]


def test_omitted_initialization_reuses_loaded_state():
    game = _game()
    game.memberships = [[0], [0, 1], [1]]
    calls = []

    def fake_leiden(**kwargs):
        calls.append(kwargs["initial_membership"])
        return _Result(kwargs["initial_membership"])

    with patch.object(game, "community_leiden", side_effect=fake_leiden):
        game.community_hedonic(max_memberships=2, resolution=0.5)
    assert calls == [[[0], [0, 1], [1]]]
    assert game.memberships == [[0], [0, 1], [1]]


def test_explicit_none_overrides_loaded_state_with_singletons():
    game = _game()
    game.memberships = [[0], [0, 1], [1]]
    calls = []

    def fake_leiden(**kwargs):
        calls.append(kwargs["initial_membership"])
        return _Result(kwargs["initial_membership"])

    with patch.object(game, "community_leiden", side_effect=fake_leiden):
        game.community_hedonic(initial_membership=None, max_memberships=2, resolution=0.5)
    assert calls == [[[0], [1], [2]]]


def test_empty_or_unconfigured_state_uses_singleton_default():
    game = _game()
    game.memberships = []
    calls = []

    def fake_leiden(**kwargs):
        calls.append(kwargs["initial_membership"])
        return _Result(kwargs["initial_membership"])

    with patch.object(game, "community_leiden", side_effect=fake_leiden):
        game.community_hedonic(max_memberships=1, resolution=0.5)
    assert calls == [[0, 1, 2]]


@pytest.mark.parametrize(
    "value",
    [
        [[0], [0, 0], [1]],  # duplicate per-vertex label
        [[0], [], [1]],  # empty row
        [[0], [0, 2], [2]],  # non-contiguous labels
        [[0], [0, "x"], [1]],  # non-integer label
        [[0], [0, 1], [1]],  # exceeds cap when loaded
    ],
)
def test_invalid_loaded_state_never_falls_back_to_default(value):
    game = _game()
    # Bypass the validating setter to model stale/corrupt persisted state.
    game._memberships = value
    with patch.object(game, "community_leiden") as leiden:
        with pytest.raises(ValueError):
            game.community_hedonic(max_memberships=1, resolution=0.5)
    leiden.assert_not_called()


def test_explicit_initialization_has_precedence_over_loaded_state():
    game = _game()
    game.memberships = [[0], [0], [1]]
    calls = []

    def fake_leiden(**kwargs):
        calls.append(kwargs["initial_membership"])
        return _Result(kwargs["initial_membership"])

    with patch.object(game, "community_leiden", side_effect=fake_leiden):
        game.community_hedonic(
            initial_membership=[0, 1, 1], max_memberships=1, resolution=0.5
        )
    assert calls == [[0, 1, 1]]
    assert game.memberships == [[0], [1], [1]]


def test_evaluate_against_scores_disjoint_state_and_accepts_partition_vector():
    game = _game()
    game.memberships = [0, 0, 1]

    assert game.evaluate_against([0, 0, 1], method="f1") == 1.0
    assert game.evaluate_against([[0, 1], [2]], method="jaccard") == 1.0
    assert game.evaluate_accuracy([[0, 1], [2]], method="one_to_one_f1") == 1.0


def test_evaluate_against_scores_overlap_state_and_selects_metrics():
    game = _game()
    game.memberships = [[0], [0, 1], [1]]
    truth = [[0, 1], [1, 2]]

    assert game.evaluate_against(truth, method="f1") == 1.0
    assert game.evaluate_against(truth, method="node_micro_f1") == 1.0
    assert game.evaluate_against(truth, method="size_weighted_community_f1") == 1.0
    assert game.evaluate_against(truth, method="omega") == 1.0


def test_evaluate_against_requires_state_ground_truth_and_known_method():
    game = _game()
    with pytest.raises(ValueError, match="no memberships"):
        game.evaluate_against([[0, 1], [2]])

    game.memberships = [0, 0, 1]
    with pytest.raises(ValueError, match="ground_truth is required"):
        game.evaluate_against(None)
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        game.evaluate_against([[0, 1], [2]], method="rand")
