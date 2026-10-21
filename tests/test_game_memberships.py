"""Tests for the canonical Game membership state and the start it gives."""

from __future__ import annotations

from unittest.mock import patch

import igraph as ig
import pytest

from hedonic import Game


class _Result:
    def __init__(self, membership):
        self.membership = membership


def _fake_native(calls):
    """Stand-in for the C layer that records the start it receives and
    returns it unchanged (None, the default start, as the core's singletons)."""

    def fake(graph, **kwargs):
        start = kwargs["initial_membership"]
        calls.append(start)
        return _Result([[v] for v in range(graph.vcount())] if start is None else start)

    return fake


NATIVE = "hedonic._native.community_hedonic"


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


def test_a_call_ignores_the_stored_state_unless_it_is_passed():
    game = _game()
    game.memberships = [[0], [0, 1], [1]]
    calls = []

    with patch(NATIVE, side_effect=_fake_native(calls)):
        game.community_hedonic(max_memberships=2, resolution=0.5)
        game.community_hedonic(game.memberships, max_memberships=2, resolution=0.5)
    assert calls == [None, [[0], [1], [2]]]
    assert game.memberships == [[0], [1], [2]]


def test_a_seed_sweep_on_one_game_gives_independent_runs():
    game = Game(ig.Graph.Famous("Zachary"))
    results = {tuple(game.community_hedonic(seed=seed).membership) for seed in range(10)}
    assert len(results) > 1


@pytest.mark.parametrize(
    "value",
    [
        [[0], [], [1]],  # empty row
        [[0], [0, "x"], [1]],  # non-integer label
        [0, 1],  # one row short
    ],
)
def test_an_invalid_start_never_reaches_the_detector(value):
    with patch(NATIVE) as native:
        with pytest.raises(ValueError, match="Invalid initial_membership"):
            _game().community_hedonic(value, resolution=0.5)
    native.assert_not_called()


def test_a_start_with_label_gaps_is_renumbered_in_label_order():
    calls = []
    with patch(NATIVE, side_effect=_fake_native(calls)):
        _game().community_hedonic([[0], [0, 7], [7]], max_memberships=2, resolution=0.5)
        _game().community_hedonic([5, 9, 5], resolution=0.5)
    assert calls == [[[0], [0, 1], [1]], [0, 1, 0]]


def test_a_cover_start_keeps_each_vertex_preferred_labels():
    # Triangle 0-1-2 with a tail 2-3-4; vertex 2 holds both communities.
    game = Game(ig.Graph(5, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)]))
    rows = [[0], [0], [0, 1], [1], [1]]
    calls = []
    with patch(NATIVE, side_effect=_fake_native(calls)):
        result = game.community_hedonic(rows, resolution=0.5)
        # Weight 5 on the edge 2-3 makes the tail's community preferred.
        game.community_hedonic(rows, resolution=0.5, edge_weights=[1, 1, 1, 5, 1])
        game.community_hedonic([[0, 1, 2]] * 5, max_memberships=2, resolution=0.0)
    assert calls[:2] == [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
    # Equal payoffs and sizes: the smaller labels stay.
    assert calls[2] == [[0, 1]] * 5
    assert result._hedonic_provenance["projected_vertices"] == 1
    assert result._hedonic_provenance["initialization"] == "initial_membership, projected to max_memberships"


def test_a_partition_from_a_cover_is_an_equilibrium():
    game = Game(ig.Graph.Famous("Zachary"))
    game.community_hedonic(max_memberships=4, seed=0)
    cover = game.memberships
    assert any(len(row) > 1 for row in cover)
    result = game.community_hedonic(cover, seed=0)
    again = game.community_hedonic(list(result.membership), seed=1)
    assert again.membership == result.membership


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


def test_ground_truth_vertices_must_be_vertices_and_flat_labels_may_be_anything():
    game = _game()
    game.memberships = [0, 0, 1]
    for truth in ([[0, 1, 3]], [[0, -1]], [[0, 1.5]]):
        with pytest.raises(ValueError, match="not an integer|outside range"):
            game.evaluate_against(truth)
    assert game.evaluate_against(["a", "a", "b"]) == 1.0
    assert game.evaluate_against([7, 7, 3]) == 1.0


def test_scores_other_than_one_to_one_need_no_scipy():
    game = _game()
    game.memberships = [0, 0, 1]
    with patch.dict("sys.modules", {"scipy": None, "scipy.optimize": None}):
        for method in ("f1", "jaccard", "omega", "size_weighted_f1"):
            assert game.evaluate_against([0, 0, 1], method=method) == 1.0
        with pytest.raises(ImportError, match="hedonic\\[metrics\\]"):
            game.evaluate_against([0, 0, 1], method="one_to_one_f1")


def test_evaluate_against_requires_state_ground_truth_and_known_method():
    game = _game()
    with pytest.raises(ValueError, match="no memberships"):
        game.evaluate_against([[0, 1], [2]])

    game.memberships = [0, 0, 1]
    with pytest.raises(ValueError, match="ground_truth is required"):
        game.evaluate_against(None)
    with pytest.raises(ValueError, match="Unknown evaluation method"):
        game.evaluate_against([[0, 1], [2]], method="rand")


def test_a_cover_keeps_labels_with_identical_members():
    """Two labels with the same members are two coalitions of the game: the cover keeps both (documented)."""
    import igraph as ig
    from hedonic import Game

    for seed in range(50):
        g = Game(ig.Graph.Erdos_Renyi(30, 0.2))
        cover = g.community_hedonic(resolution=0.05, max_memberships=3, seed=seed)
        bodies = [frozenset(c) for c in cover]
        if len(set(bodies)) < len(bodies):
            occupied = {label for row in g.memberships for label in row}
            assert len(bodies) == len(occupied)  # one community per occupied label, duplicates included
            return
    raise AssertionError("no cover with duplicate bodies in 50 seeds")
