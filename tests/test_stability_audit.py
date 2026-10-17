"""Bit equality with the paper audit, independent of the detector pins."""
import copy
import itertools
import signal

import igraph as ig
import numpy as np
import pytest

from hedonic import Game, _core
from hedonic import _audit
from hedonic.stability import nash_window, stable_fraction, stable_measures
from tests._audit_instances import random_instance, many_label_instance
from tests import _audit_reference as ref

ATOL, RTOL = 1e-10, 1e-9
GRID = [0., 1e-4, .01, .05, .2, .5, .8, 1.]


def same_bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64))


def compare(graph, rows, *, vertices=None):
    game = Game(graph)
    state = ref.build_fractional_state(graph, rows)
    selected = list(range(game.vcount())) if vertices is None else vertices
    top = max(map(len, rows))
    for cap, allow, prune in itertools.product((top, top + 1, top + 2), (False, True), (False, True)):
        actual = _audit.stable_intervals(game, rows, max_memberships=cap,
                                         allow_isolation=allow, vertices=selected, prune=prune)
        expected = [ref.exact_stable_intervals(state, v, cap, allow, atol=ATOL, rtol=RTOL, prune=prune)
                    for v in selected]
        assert actual == expected, (cap, allow, prune)
        result = _audit.regrets(game, GRID, rows, max_memberships=cap,
                               allow_isolation=allow, vertices=selected, prune=prune)
        entries = ref.build_entries(state, cap, allow, selected, prune=prune)
        for i, gamma in enumerate(GRID):
            r, t = ref.compiled_regrets(entries, gamma)
            same_bits(result.regret[:, i], r)
            same_bits(result.tolerance[:, i], t)


@pytest.mark.parametrize("seed", range(40))
def test_random_exact_equality(seed):
    compare(*random_instance(seed))


@pytest.mark.parametrize("seed", range(3))
def test_many_labels(seed):
    graph, rows = many_label_instance(seed)
    compare(graph, rows, vertices=list(range(0, graph.vcount(), 11)))


def test_ties_weights_parallel_edges_and_private_labels():
    graph = ig.Graph.Full(6)
    rows = [[0, 1], [0, 1], [0, 1], [1, 2], [1, 2], [2]]
    compare(graph, rows)
    graph.add_edges([(0, 1), (0, 1), (2, 5)])
    graph.es["weight"] = [0., 1e-9, 1., .1, 2.5, 1e-300] * 3
    compare(graph, rows)
    graph.add_vertices(2)
    rows += [[3], [4]]  # explicit completion of uncovered vertices
    compare(graph, rows)
    compare(ig.Graph(n=3), [[0], [0], [1]])


@pytest.mark.parametrize("seed", range(10))
def test_grid_and_best_response_away_from_endpoints(seed):
    graph, rows = random_instance(seed)
    state = ref.build_fractional_state(graph, rows)
    game = Game(graph)
    for cap, allow in itertools.product((max(map(len, rows)), max(map(len, rows)) + 1), (False, True)):
        intervals = game.stable_intervals(rows, max_memberships=cap, allow_isolation=allow)
        for v, found in enumerate(intervals):
            for gamma in np.linspace(0, 1, 101):
                if any(abs(gamma - end) < 1e-6 for pair in found for end in pair):
                    continue
                response = ref.best_response(state, v, gamma, cap, allow)
                expected = max(0, response["regret"]) <= ref._tolerance(response["current_utility"], ATOL, RTOL)
                assert ref.in_intervals(found, gamma) == expected


def test_cliques_and_pendant_are_nash_at_point_three():
    edges = [(a, b) for block in (range(4), range(4, 8)) for a, b in itertools.combinations(block, 2)]
    game = Game(n=9, edges=edges + [(7, 8)])
    rows = [[0]] * 4 + [[1]] * 4 + [[2]]
    for allow in (False, True):
        intervals = game.stable_intervals(rows, max_memberships=2, allow_isolation=allow)
        result = game.regrets([.3], rows, max_memberships=2, allow_isolation=allow)
        assert np.all(result.regret <= result.tolerance)
        assert ref.in_intervals(nash_window(intervals), .3)


def test_labelled_detector_equilibrium():
    game = Game(ig.Graph.Famous("Zachary"))
    game.community_hedonic(max_memberships=3, resolution=game.density(), seed=0, allow_isolation=True)
    for allow in (False, True):
        result = game.regrets([game.density()], max_memberships=3, allow_isolation=allow)
        assert np.all(result.regret <= result.tolerance)
        assert result.regret.max() < 1e-12
        assert all(ref.in_intervals(row, game.density()) for row in
                   game.stable_intervals(max_memberships=3, allow_isolation=allow))


def test_normalization_defaults_selection_and_no_mutation():
    graph, original = random_instance(5)
    game = Game(graph)
    rows = [[3 + 7 * c for c in reversed(row)] + [3 + 7 * row[0]] for row in original]
    before = copy.deepcopy(rows)
    expected = game.stable_intervals(original)
    assert game.stable_intervals(rows) == expected
    assert rows == before
    game.memberships = original
    assert game.stable_intervals() == expected
    assert game.stable_intervals(vertices=[3, 0, 3]) == [expected[v] for v in (3, 0, 3)]
    assert game.stable_intervals(vertices=range(2, 5)) == expected[2:5]
    assert game.stable_intervals(vertices=[]) == []
    assert game.regrets([]).regret.shape == (game.vcount(), 0)
    assert game.regrets(GRID, vertices=[]).regret.shape == (0, len(GRID))
    full = game.regrets(GRID)
    selected = game.regrets(GRID, vertices=[3, 0, 3])
    same_bits(selected.regret, full.regret[[3, 0, 3]])
    same_bits(selected.tolerance, full.tolerance[[3, 0, 3]])
    empty = Game(n=0)
    assert empty.stable_intervals([]) == []
    assert empty.regrets([], []).regret.shape == (0, 0)


@pytest.mark.parametrize("kwargs", [
    {"memberships": [[], [0]]}, {"memberships": [[-1], [0]]},
    {"memberships": [[True], [0]]}, {"memberships": [[0.5], [0]]},
    {"memberships": [[0, 1], [1]], "max_memberships": 1},
    {"max_memberships": 0}, {"max_memberships": -1}, {"max_memberships": True},
    {"atol": -1}, {"rtol": np.inf}, {"atol": np.nan},
    {"vertices": [2]}, {"vertices": [-1]}, {"vertices": [True]}, {"vertices": [0.0]},
    {"edge_weights": [-1]}, {"edge_weights": [np.inf]}, {"edge_weights": [np.nan]},
])
def test_invalid_inputs(kwargs):
    game = Game(n=2, edges=[(0, 1)])
    game.memberships = [0, 1]
    with pytest.raises((ValueError, TypeError)):
        game.stable_intervals(**kwargs)
    with pytest.raises((ValueError, TypeError)):
        game.regrets([.5], **kwargs)


def test_invalid_graph_and_resolutions():
    for graph in (ig.Graph(n=2, edges=[(0, 0)]), ig.Graph(n=2, edges=[(0, 1)], directed=True)):
        with pytest.raises(ValueError):
            Game(graph).stable_intervals([0, 1])
    game = Game(n=2)
    with pytest.raises(ValueError, match="no memberships"):
        game.stable_intervals()
    for values in ([-1], [np.nan], [np.inf], [1.1], [[.5]], .5):
        with pytest.raises(ValueError):
            game.regrets(values, [0, 1])
    with pytest.raises(ValueError, match="unit node weights"):
        _core._audit_prepare(game, np.array([0, 1, 2], dtype=np.int64), np.array([0, 1], dtype=np.int64),
                             node_weights=np.array([1., 2.]))
    with pytest.raises(ValueError, match="offset"):
        _core._audit_prepare(game, np.array([0, 1, 9], dtype=np.int64), np.array([0, 1], dtype=np.int64))


def test_aggregate_helpers():
    intervals = [[(0, .5), (.75, 1)], [(.25, 1)]]
    np.testing.assert_array_equal(stable_fraction(intervals, [0, .25, .5, .6, .75, 1]), [.5, 1, 1, .5, 1, 1])
    assert nash_window(intervals) == [(.25, .5), (.75, 1)]
    linear, log = stable_measures(intervals)
    assert linear == .75
    expected = (np.log10(.5) + 4 + np.log10(1 / .75) + np.log10(4)) / 8
    assert abs(log - expected) < 1e-15
    assert nash_window([[], [(0, 1)]]) == []
    assert nash_window([]) == [(0, 1)]
    assert stable_measures([[(0, 1)]]) == (1, 1)


@pytest.mark.skipif(not hasattr(signal, "setitimer"), reason="POSIX interval timer")
def test_interrupt_a_long_single_vertex_audit():
    game = Game(n=20000)
    state, cap = _audit._prepare(game, [[v] for v in range(20000)])
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGALRM, interrupted)
    try:
        signal.setitimer(signal.ITIMER_REAL, .001)
        with pytest.raises(KeyboardInterrupt):
            _core._audit_intervals(state, np.array([0], dtype=np.int64), 20000, False, ATOL, RTOL, False)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert _core._audit_intervals(state, np.array([0], dtype=np.int64), cap) == [[(0., 1.)]]


@pytest.mark.parametrize("atol,rtol", [(0., 0.), (.02, 0.), (1e-10, .01), (0., 2.)])
def test_tolerance_pieces(atol, rtol):
    graph, rows = random_instance(9)
    state = ref.build_fractional_state(graph, rows)
    game = Game(graph)
    cap = max(map(len, rows)) + 2
    for allow in (False, True):
        expected = [ref.exact_stable_intervals(state, v, cap, allow, atol=atol, rtol=rtol)
                    for v in range(game.vcount())]
        assert game.stable_intervals(rows, max_memberships=cap, allow_isolation=allow,
                                     atol=atol, rtol=rtol) == expected
        result = game.regrets(GRID, rows, max_memberships=cap, allow_isolation=allow,
                              atol=atol, rtol=rtol)
        entries = ref.build_entries(state, cap, allow)
        for i, gamma in enumerate(GRID):
            regret, tolerance = ref.compiled_regrets(entries, gamma, atol, rtol)
            same_bits(result.regret[:, i], regret)
            same_bits(result.tolerance[:, i], tolerance)


def test_regret_epsilon_and_overflow():
    game = Game(n=2, edges=[(0, 1)])
    for weight in (1e-15, np.nextafter(1e-15, np.inf)):
        result = game.regrets([0.], [0, 1], edge_weights=[weight], atol=0., rtol=0.)
        assert result.regret[0, 0] == (weight if weight > 1e-15 else 0.)
        assert result.tolerance[0, 0] == 0.
    with pytest.raises(ValueError, match="exceeds"):
        game.stable_intervals([0, 1], edge_weights=[1e308])
    with pytest.raises(OverflowError):
        game.regrets([0.], [0, 1], rtol=1e308)


def test_aggregate_reductions_match_the_paper_bits():
    rng = np.random.default_rng(12)
    cuts = np.sort(rng.random((1001, 4)), axis=1)
    intervals = [[(a, b), (c, d)] for a, b, c, d in cuts]
    # The paper forms these arrays in vertex order, then reduces with NumPy.
    lo = np.array([lo for row in intervals for lo, hi in row])
    hi = np.array([hi for row in intervals for lo, hi in row])
    floor = np.maximum(lo, 1e-4)
    above = hi > floor
    expected = [float((hi-lo).sum()) / len(intervals),
                float((np.log10(hi[above])-np.log10(floor[above])).sum()) / (4 * len(intervals))]
    same_bits(stable_measures(intervals), expected)
    assert stable_measures([[], []]) == (0., 0.)
