"""``hedonic.arrays``: the detector and the audit over NumPy arrays, without igraph.

``detect(n, edges, ...)`` must return the bits of
``Game(igraph.Graph(n, edges)).community_hedonic(...)`` (membership or rows,
quality, slack, provenance) on any graph, weights, options and seed, and the
audit those of ``Game.stable_intervals`` / ``Game.regrets``: the differential
tests below draw many cases and compare every bit, on whichever core is loaded
(run them with ``HEDONIC_CORE=c`` too where the C core is installed). The
array path never imports igraph, and reads the edges without copying them
into Python objects.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import igraph as ig
import numpy as np

from hedonic import Game, arrays


def _bits(value):
    """A float's bits (so 0.0 and -0.0 differ, and NaN equals NaN)."""
    return value.hex() if isinstance(value, float) else value


def _plain(provenance: dict) -> dict:
    return {key: _bits(value) for key, value in provenance.items() if key != "versions"}


def _graph(rnd: random.Random):
    """A random multigraph without self-loops, its pairs in random orientation."""
    n = rnd.randint(2, 40) if rnd.random() < 0.9 else rnd.randint(100, 400)
    pairs = []
    for _ in range(rnd.randint(1, 3 * n)):
        u, v = rnd.sample(range(n), 2)
        pairs.append((u, v))
        if rnd.random() < 0.1:
            pairs.append((v, u))   # a parallel edge
    return n, pairs


def _start(rnd: random.Random, n: int, previous):
    choice = rnd.random()
    if choice < 0.4:
        return None
    if choice < 0.55 and previous is not None:
        return previous
    if choice < 0.75:   # flat labels with gaps
        return [rnd.choice([0, 2, 5, 9]) for _ in range(n)]
    labels = rnd.randint(1, 6)   # rows, possibly beyond the cap (projected)
    return [rnd.sample(range(labels), rnd.randint(1, min(4, labels))) for _ in range(n)]


def _options(rnd: random.Random, n: int, m: int):
    cap = rnd.choice([1, 1, 2, 3, -1])
    options = {"max_memberships": cap}
    if rnd.random() < 0.7:
        options["seed"] = rnd.choice([0, 1, 7, rnd.randrange(-2**40, 2**40)])
    if rnd.random() < 0.6:
        options["resolution"] = rnd.choice([0.0, 0.5, 1.0, rnd.uniform(0, 0.6)])
    if rnd.random() < 0.4:
        options["allow_isolation"] = True
    if rnd.random() < 0.4:
        low = -0.5 if cap == 1 else 0.0
        weights = [round(rnd.uniform(low, 3.0), 3) for _ in range(m)]
        if sum(weights) <= 0:
            weights = [abs(w) + 0.5 for w in weights]
        options["edge_weights"] = weights
    if rnd.random() < 0.3:
        options["node_weights"] = [rnd.choice([0.0, 0.5, 1.0, 2.0]) for _ in range(n)]
    if cap == 1 and rnd.random() < 0.25:
        options["max_communities"] = rnd.randint(1, n)
    if rnd.random() < 0.15:
        options["max_total_communities"] = rnd.randint(1, n)
    elif rnd.random() < 0.15:
        options["n_communities"] = rnd.randint(1, n)
    return options


def _game_call(n, pairs, start, options, module_seed):
    game = Game(ig.Graph(n, pairs))
    random.seed(module_seed)
    try:
        result = game.community_hedonic(start, **options)
    except ValueError as error:
        return ("error", str(error).replace("community_hedonic", "<caller>"))
    overlapping = result._hedonic_provenance["effective_max_memberships"] > 1
    return {
        "rows": game.memberships,
        "membership": None if overlapping else list(result.membership),
        "communities": [list(c) for c in result],
        "quality": _bits(result._hedonic_native_quality),
        "slack": _bits(result._hedonic_provenance["equilibrium_slack"]),
        "provenance": _plain(result._hedonic_provenance),
    }


def _array_call(n, edges, start, options, module_seed):
    random.seed(module_seed)
    try:
        result = arrays.detect(n, edges, initial_membership=start, **options)
    except ValueError as error:
        return ("error", str(error).replace("detect", "<caller>"))
    return {
        "rows": result.rows(),
        "membership": None if result.overlapping else result.membership.tolist(),
        "communities": [c.tolist() for c in result.communities()],
        "quality": _bits(result.quality),
        "slack": _bits(result.equilibrium_slack),
        "provenance": _plain(result.provenance),
    }


class TestDetectAgainstGame(unittest.TestCase):
    """Every bit of detect() is Game.community_hedonic()'s."""

    def test_random_graphs_options_and_starts(self):
        rnd = random.Random(20261020)
        outcomes = {"partition": 0, "cover": 0, "error": 0}
        previous = {}
        for case in range(400):
            n, pairs = _graph(rnd)
            options = _options(rnd, n, len(pairs))
            start = _start(rnd, n, previous.get(n))
            module_seed = rnd.randrange(2**32)
            with self.subTest(case=case, n=n, m=len(pairs), **{
                k: v for k, v in options.items() if k not in ("edge_weights", "node_weights")
            }):
                expected = _game_call(n, pairs, start, options, module_seed)
                actual = _array_call(n, np.array(pairs), start, options, module_seed)
                self.assertEqual(actual, expected)
                if isinstance(expected, dict):
                    previous[n] = expected["rows"]
                    outcomes["partition" if expected["membership"] is not None else "cover"] += 1
                else:
                    outcomes["error"] += 1
        # The draw reaches every kind of outcome.
        self.assertGreater(outcomes["partition"], 100)
        self.assertGreater(outcomes["cover"], 100)
        self.assertGreater(outcomes["error"], 0)

    def test_a_result_continues_as_game_memberships_do(self):
        pairs = ig.Graph.Famous("Zachary").get_edgelist()
        for cap in (1, 3):
            game = Game(ig.Graph(34, pairs))
            game.community_hedonic(max_memberships=4, resolution=0.1, seed=1)
            first = arrays.detect(34, pairs, max_memberships=4, resolution=0.1, seed=1)
            again = game.community_hedonic(game.memberships, max_memberships=cap,
                                           resolution=0.2, seed=2)
            mine = arrays.detect(34, pairs, initial_membership=first, max_memberships=cap,
                                 resolution=0.2, seed=2)
            self.assertEqual(mine.rows(), game.memberships)
            self.assertEqual(_bits(mine.quality), _bits(again._hedonic_native_quality))

    def test_unseeded_calls_draw_from_the_random_module(self):
        pairs = ig.Graph.Famous("Zachary").get_edgelist()
        state = random.getstate()
        a = arrays.detect(34, pairs, resolution=0.1)
        self.assertNotEqual(random.getstate(), state)
        self.assertEqual(a.provenance["rng"], "random module")
        random.setstate(state)
        b = Game(ig.Graph(34, pairs)).community_hedonic(resolution=0.1)
        self.assertEqual(a.membership.tolist(), b.membership)

    def test_csr_is_the_upper_triangle_in_row_major_order(self):
        rnd = random.Random(5)
        for case in range(30):
            n, pairs = _graph(rnd)
            weights = [rnd.choice([0.5, 1.0, 2.0]) for _ in pairs]
            matrix = {}
            for (u, v), w in zip(pairs, weights):
                key = (min(u, v), max(u, v))
                matrix[key] = matrix.get(key, 0.0) + w
            upper = sorted(matrix)
            entries = sorted(upper + [(j, i) for i, j in upper])
            values = {**matrix, **{(j, i): w for (i, j), w in matrix.items()}}
            indptr = np.searchsorted([i for i, _ in entries], np.arange(n + 1))
            indices = np.array([j for _, j in entries])
            data = np.array([values[e] for e in entries])
            for cap in (1, 2):
                with self.subTest(case=case, cap=cap):
                    expected = _array_call(n, upper, None, dict(
                        max_memberships=cap, seed=case,
                        edge_weights=[matrix[e] for e in upper]), 0)
                    self.assertEqual(_array_call(n, (indptr, indices, data), None,
                                                 dict(max_memberships=cap, seed=case), 0),
                                     expected)
                    self.assertEqual(_game_call(n, upper, None, dict(
                        max_memberships=cap, seed=case,
                        edge_weights=[matrix[e] for e in upper]), 0), expected)
                    unweighted = _array_call(n, (indptr, indices), None,
                                             dict(max_memberships=cap, seed=case), 0)
                    self.assertEqual(unweighted, _game_call(n, upper, None, dict(
                        max_memberships=cap, seed=case), 0))


class TestAuditAgainstGame(unittest.TestCase):
    def test_intervals_and_regrets_have_the_bits_of_game(self):
        rnd = random.Random(11)
        grid = np.linspace(0.0, 1.0, 17)
        for case in range(60):
            n, pairs = _graph(rnd)
            labels = rnd.randint(1, 5)
            cover = [rnd.sample(range(labels), rnd.randint(1, min(3, labels))) for _ in range(n)]
            options = {}
            if rnd.random() < 0.5:
                options["edge_weights"] = [rnd.choice([0.0, 0.5, 1.0, 3.0]) for _ in pairs]
            if rnd.random() < 0.5:
                options["allow_isolation"] = True
            if rnd.random() < 0.3:
                options["max_memberships"] = 3
            if rnd.random() < 0.3:
                options["vertices"] = [rnd.randrange(n) for _ in range(5)]
            game = Game(ig.Graph(n, pairs))
            with self.subTest(case=case, **{k: v for k, v in options.items() if k != "edge_weights"}):
                try:
                    expected = game.stable_intervals(cover, **options)
                except ValueError as error:
                    with self.assertRaisesRegex(ValueError, "^" + str(error) + "$"):
                        arrays.stable_intervals(n, pairs, cover, **options)
                    continue
                actual = arrays.stable_intervals(n, np.array(pairs), cover, **options)
                self.assertEqual([[tuple(map(_bits, pair)) for pair in row] for row in actual],
                                 [[tuple(map(_bits, pair)) for pair in row] for row in expected])
                a = arrays.regrets(n, pairs, grid, cover, **options)
                b = game.regrets(grid, cover, **options)
                for x, y in zip(a, b):
                    self.assertEqual(x.tobytes(), y.tobytes())

    def test_a_result_is_an_audited_cover(self):
        pairs = ig.Graph.Famous("Zachary").get_edgelist()
        result = arrays.detect(34, pairs, max_memberships=2, resolution=0.1, seed=0)
        game = Game(ig.Graph(34, pairs))
        game.community_hedonic(max_memberships=2, resolution=0.1, seed=0)
        self.assertEqual(arrays.stable_intervals(34, pairs, result), game.stable_intervals())


class TestEdges(unittest.TestCase):
    """The binding reads the endpoint arrays as buffers (``_core.detect_arrays``,
    ``_core._audit_prepare_arrays``), never an igraph graph or edge list."""

    PAIRS = ig.Graph.Famous("Zachary").get_edgelist()

    def test_the_edges_go_to_the_core_as_buffers(self):
        from hedonic import _core

        listed = AssertionError("the graph entry points were called")
        with patch.object(_core, "detect", side_effect=listed), \
                patch.object(_core, "_audit_prepare", side_effect=listed):
            arrays.detect(34, self.PAIRS, seed=0)
            arrays.detect(34, self.PAIRS, max_memberships=2, seed=0)
            arrays.stable_intervals(34, self.PAIRS, [[v % 3] for v in range(34)])

    def test_any_buffer_of_the_edges_gives_the_same_result(self):
        from array import array

        from hedonic import _core

        first, second = (array("q", side) for side in zip(*self.PAIRS))
        for cap in (1, 2):
            expected = arrays.detect(34, self.PAIRS, max_memberships=cap, seed=3)
            out = _core.detect_arrays(34, first, second, random.Random(3), expected.provenance["resolution"],
                                      max_memberships=cap)
            rows = [[label] for label in out[0]] if cap == 1 else out[0]
            self.assertEqual(rows, expected.rows())
            self.assertEqual(_bits(out[-2]), _bits(expected.quality))

    def test_a_core_without_the_buffer_entry_points_gives_the_same_result(self):
        """A second core (HEDONIC_CORE) built without them reads the edge list instead."""
        import types

        from hedonic import _core

        older = types.SimpleNamespace(**{name: getattr(_core, name) for name in dir(_core)
                                         if name not in ("detect_arrays", "_audit_prepare_arrays")})
        cover = [[v % 3] for v in range(34)]
        for cap in (1, 2):
            expected = arrays.detect(34, self.PAIRS, max_memberships=cap, seed=3, initial_membership=cover)
            intervals = arrays.stable_intervals(34, self.PAIRS, expected)
            with patch.object(arrays, "_core", older), patch.object(arrays._audit, "_core", older):
                fallback = arrays.detect(34, self.PAIRS, max_memberships=cap, seed=3, initial_membership=cover)
                self.assertEqual(arrays.stable_intervals(34, self.PAIRS, fallback), intervals)
            self.assertEqual(fallback.rows(), expected.rows())
            self.assertEqual(_bits(fallback.quality), _bits(expected.quality))
            self.assertEqual(_plain(fallback.provenance), _plain(expected.provenance))

    def test_the_binding_checks_its_buffers(self):
        from array import array

        from hedonic import _core

        q = lambda *values: array("q", values)  # noqa: E731
        call = lambda *edges, **options: _core.detect_arrays(  # noqa: E731
            3, *edges, random.Random(0), 0.5, **options)
        with self.assertRaisesRegex(ValueError, "second has 1 entries, expected 2"):
            call(q(0, 1), q(1))
        with self.assertRaisesRegex(TypeError, "first must be a contiguous buffer of 64-bit integers"):
            call(array("i", [0]), q(1))
        with self.assertRaisesRegex(ValueError, "n must be non-negative"):
            _core.detect_arrays(-1, q(), q(), random.Random(0), 0.5)
        with self.assertRaisesRegex(ValueError, "together, or neither"):
            call(q(0), q(1), start_offsets=q(0, 1, 2, 3))
        with self.assertRaisesRegex(ValueError, "start_offsets has 3 entries, expected 4"):
            call(q(0), q(1), start_offsets=q(0, 1, 2), start_labels=q(0, 0))
        with self.assertRaisesRegex(ValueError, "start_offsets must start at 0"):
            call(q(0), q(1), start_offsets=q(0, 2, 1, 3), start_labels=q(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "each membership row must be non-empty"):
            call(q(0), q(1), start_offsets=q(0, 1, 1, 2), start_labels=q(0, 0))
        with self.assertRaisesRegex(ValueError, "membership labels must be non-negative"):
            call(q(0), q(1), start_offsets=q(0, 1, 2, 3), start_labels=q(0, -1, 0))
        with self.assertRaisesRegex(ValueError, r"a vertex exceeds max_memberships \(2 > 1\)"):
            call(q(0), q(1), start_offsets=q(0, 2, 3, 4), start_labels=q(0, 1, 0, 1))
        with self.assertRaisesRegex(ValueError, "contiguous from 0"):
            call(q(0), q(1), start_offsets=q(0, 1, 2, 3), start_labels=q(0, 2, 0))
        with self.assertRaisesRegex(ValueError, "outside|vertex"):
            call(q(0), q(3))
        # A repeated label counts once, as in detect()'s rows.
        self.assertEqual(
            call(q(0), q(1), max_memberships=2, start_offsets=q(0, 2, 3, 4), start_labels=q(1, 1, 1, 0))[0],
            _core.detect(ig.Graph(3, [(0, 1)]), random.Random(0), 0.5, max_memberships=2,
                         start=[[1, 1], [1], [0]])[0],
        )

    def test_pairs_in_either_orientation_are_one_edge(self):
        flipped = [(v, u) for u, v in self.PAIRS]
        for cap in (1, 2):
            a = arrays.detect(34, self.PAIRS, max_memberships=cap, seed=0)
            b = arrays.detect(34, flipped, max_memberships=cap, seed=0)
            self.assertEqual(a.rows(), b.rows())


class TestResult(unittest.TestCase):
    def test_partition_and_cover_arrays(self):
        pairs = ig.Graph.Famous("Zachary").get_edgelist()
        partition = arrays.detect(34, pairs, seed=0)
        self.assertFalse(partition.overlapping)
        self.assertEqual(partition.membership.dtype, np.int64)
        self.assertIs(partition.labels, partition.membership)
        self.assertEqual(partition.offsets.tolist(), list(range(35)))
        self.assertEqual(partition.community_count, len(set(partition.membership.tolist())))
        cover = arrays.detect(34, pairs, max_memberships=3, resolution=0.1, seed=0)
        self.assertTrue(cover.overlapping)
        self.assertIsNone(cover.membership)
        self.assertEqual(len(cover.offsets), 35)
        self.assertEqual(cover.offsets[-1], len(cover.labels))
        self.assertEqual(sum(len(c) for c in cover.communities()), len(cover.labels))
        self.assertIn("cover of 34 vertices", repr(cover))


class TestValidation(unittest.TestCase):
    def check(self, message, n, edges, **options):
        with self.assertRaisesRegex(ValueError, message):
            arrays.detect(n, edges, **options)

    def test_the_graph(self):
        self.check("n must be a non-negative integer", 2.0, [(0, 1)])
        self.check("n must be a non-negative integer", True, [(0, 1)])
        self.check("detect requires a nonempty graph", 0, [])
        self.check(r"an \(m, 2\) array", 3, [0, 1, 2])
        self.check("must hold integers", 3, [(0.0, 1.0)])
        self.check("must hold integers", 3, [(True, False)])
        self.check(r"edge 1 = \(1, 3\) has a vertex outside 0 \.\. 2", 3, [(0, 1), (1, 3)])
        self.check(r"edge 0 = \(-1, 1\) has a vertex outside", 3, [(-1, 1)])
        self.check("edge 1 joins vertex 2 to itself", 3, [(0, 1), (2, 2)])
        self.check("detect needs a graph with edges", 3, np.empty((0, 2), dtype=np.int64))

    def test_csr(self):
        self.check("must be symmetric", 3, (np.array([0, 1, 1, 1]), np.array([1])))
        self.check("must be symmetric", 2, (np.array([0, 1, 2]), np.array([1, 0]),
                                            np.array([1.0, 2.0])))
        self.check("indptr must start at 0", 2, (np.array([1, 1, 2]), np.array([1, 0])))
        self.check("indices must be columns", 2, (np.array([0, 1, 2]), np.array([1, 5])))
        self.check("joins vertex 1 to itself", 2, (np.array([0, 1, 3]), np.array([1, 0, 1])))
        self.check("not both", 2, (np.array([0, 1, 2]), np.array([1, 0]), np.ones(2)),
                   edge_weights=[1.0])

    def test_the_options_are_game_s(self):
        pairs = [(0, 1), (1, 2), (2, 0)]
        self.check("max_memberships must be an integer >= 1, or -1", 3, pairs, max_memberships=0)
        self.check("seed must be an integer", 3, pairs, seed=1.5)
        self.check("allow_isolation must be True or False", 3, pairs, allow_isolation=1)
        self.check("resolution must be a finite real number", 3, pairs, resolution=math.inf)
        self.check("edge_weights must have one value per edge", 3, pairs, edge_weights=[1.0])
        self.check("attribute names need an igraph graph", 3, pairs, edge_weights="weight")
        self.check("attribute names need an igraph graph", 3, pairs, node_weights="w")
        self.check("non-negative edge weights", 3, pairs, max_memberships=2,
                   edge_weights=[1.0, -1.0, 1.0])
        self.check("Invalid initial_membership", 3, pairs, initial_membership=[0, 1])
        self.check("n_communities=4 exceeds", 3, pairs, n_communities=4)

    def test_the_audit_needs_a_cover(self):
        with self.assertRaisesRegex(ValueError, "needs a cover"):
            arrays.stable_intervals(3, [(0, 1)], None)
        with self.assertRaisesRegex(ValueError, "audit vertex outside the graph"):
            arrays.regrets(3, [(0, 1)], [0.5], [0, 0, 1], vertices=[3])


class TestWithoutIgraph(unittest.TestCase):
    """The array path imports only NumPy (and the standard library)."""

    SCRIPT = textwrap.dedent(
        """
        import sys

        before = set(sys.modules)

        class Blocked:
            def find_spec(self, name, path=None, target=None):
                if name == "igraph" or name.startswith("igraph."):
                    raise ImportError("igraph is blocked")

        if BLOCK:
            sys.meta_path.insert(0, Blocked())
        import hedonic.arrays as arrays
        pairs = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
        p = arrays.detect(6, pairs, seed=0)
        c = arrays.detect(6, pairs, max_memberships=2, seed=0, initial_membership=p)
        iv = arrays.stable_intervals(6, pairs, c)
        r = arrays.regrets(6, pairs, [0.1, 0.5], p)
        # (sysconfig's platform data, _sysconfigdata_*, is the standard library's too)
        third = sorted({name.split(".")[0] for name in set(sys.modules) - before
                        if not name.startswith("_sysconfigdata_")}
                       - set(sys.stdlib_module_names) - {"hedonic", "numpy"})
        print(p.membership.tolist(), len(iv), r.regret.shape, "igraph" in sys.modules, third)
        if BLOCK:
            try:
                from hedonic import Game
            except ImportError as error:
                print("Game:", error)
        """
    )

    def run_script(self, block: bool) -> list[str]:
        out = subprocess.run([sys.executable, "-c", f"BLOCK = {block}\n" + self.SCRIPT],
                             check=True, capture_output=True, text=True)
        return out.stdout.splitlines()

    def test_igraph_blocked(self):
        lines = self.run_script(True)
        self.assertEqual(lines[0], "[0, 0, 0, 1, 1, 1] 6 (6, 2) False []")
        self.assertEqual(lines[1], "Game: igraph is blocked")

    def test_igraph_installed_is_not_imported(self):
        self.assertEqual(self.run_script(False), ["[0, 0, 0, 1, 1, 1] 6 (6, 2) False []"])

    def test_importing_the_package_does_not_import_igraph(self):
        script = "import sys, hedonic; print('igraph' in sys.modules, hedonic.Game.__name__)"
        out = subprocess.run([sys.executable, "-c", script], check=True,
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.split(), ["False", "Game"])

    def test_the_game_submodule_does_not_replace_the_class(self):
        # Unpickling a Game imports the submodule hedonic.Game first.
        script = "import hedonic.Game, hedonic; print(isinstance(hedonic.Game, type))"
        out = subprocess.run([sys.executable, "-c", script], check=True,
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()
