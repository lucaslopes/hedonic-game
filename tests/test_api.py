"""The public surface of hedonic 2.0.

``Game.community_hedonic`` always runs local moving until no vertex can
improve, so it takes no iteration budget and no other detector's options;
seeded calls replay; importing the package runs no check on igraph's version;
results carry a fixed provenance record; ``hedonic.metrics`` scores covers.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import unittest

import igraph as ig

import hedonic
from hedonic import Game

REMOVED_KEYWORDS = {
    "n_iterations": -1,
    "local_move_only": True,
    "beta": 0.01,
    "debug_trace": True,
    "phase_policy": "direct",
}

PROVENANCE_KEYS = {
    "versions", "resolution", "max_memberships", "effective_max_memberships",
    "count_constraint", "max_total_communities", "n_communities", "max_communities",
    "max_communities_effect", "start_seed", "initialization", "occupied_communities",
    "projected_vertices",
    "allow_isolation", "edge_weighted", "node_weighted", "total_edge_weight", "seed",
    "rng", "equilibrium_notion", "equilibrium_slack",
}


def _karate() -> Game:
    return Game(ig.Graph.Famous("Zachary"))


def _two_cliques() -> Game:
    """Two 5-cliques joined by one edge; the cliques are the reference."""
    graph = ig.Graph.Full(5) + ig.Graph.Full(5)
    graph.add_edge(0, 5)
    return Game(graph)


TWO_CLIQUES_TRUTH = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


class TestRemovedKeywords(unittest.TestCase):
    def test_each_removed_keyword_raises_type_error(self):
        for max_memberships in (1, 2):
            for name, value in REMOVED_KEYWORDS.items():
                with self.subTest(name=name, max_memberships=max_memberships):
                    with self.assertRaisesRegex(TypeError, name):
                        _karate().community_hedonic(
                            max_memberships=max_memberships, **{name: value}
                        )

    def test_removed_names_are_gone(self):
        game_module = sys.modules["hedonic.Game"]
        for name in ("HEDONIC_ALGORITHM_IDENTITY", "HISTORICAL_ALGORITHM_IDENTITIES",
                     "seeded_igraph_rng"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(game_module, name))
                self.assertFalse(hasattr(hedonic, name))
        self.assertEqual(hedonic.__all__, ["Game", "arrays", "metrics", "stability"])

    def test_results_carry_no_removed_attributes(self):
        for max_memberships in (1, 3):
            result = _karate().community_hedonic(max_memberships=max_memberships, seed=0)
            for name in ("_hedonic_raw_memberships", "_hedonic_algorithm_identity",
                         "_hedonic_token_preflight", "_hedonic_disjoint_stage"):
                with self.subTest(name=name, max_memberships=max_memberships):
                    self.assertFalse(hasattr(result, name))
            self.assertNotIn("debug_trace", getattr(result, "_params", None) or {})


class TestSeedReplay(unittest.TestCase):
    def test_same_seed_same_result(self):
        cases = [
            dict(max_memberships=1),
            dict(max_memberships=1, resolution=0.05, allow_isolation=True),
            dict(max_memberships=1, max_communities=4),
            dict(max_memberships=1, n_communities=3, resolution=0.05),
            dict(max_memberships=3, resolution=0.1),
            dict(max_memberships=3, resolution=0.1, max_total_communities=5),
            dict(max_memberships=-1, resolution=0.05,
                 node_weights=[1.0 + v % 3 for v in range(34)]),
        ]
        for kwargs in cases:
            for seed in (0, 1, 12345):
                with self.subTest(seed=seed, **{k: v for k, v in kwargs.items()
                                                if k != "node_weights"}):
                    a = _karate().community_hedonic(initial_membership=None, seed=seed,
                                                    **kwargs)
                    b = _karate().community_hedonic(initial_membership=None, seed=seed,
                                                    **kwargs)
                    self.assertEqual(a.membership, b.membership)
                    quality = (lambda r: r.quality if isinstance(r, ig.VertexClustering)
                               else r._hedonic_native_quality)
                    self.assertEqual(quality(a), quality(b))

    def test_different_seeds_can_differ(self):
        results = {
            tuple(_karate().community_hedonic(initial_membership=None, seed=s,
                                              resolution=0.05).membership)
            for s in range(10)
        }
        self.assertGreater(len(results), 1)


class TestNoVersionCheckOnImport(unittest.TestCase):
    def test_there_is_no_runtime_pin_module(self):
        self.assertIsNone(importlib.util.find_spec("hedonic._runtime_pin"))
        self.assertFalse(hasattr(hedonic, "_runtime_pin"))

    def test_import_succeeds_whatever_igraph_reports(self):
        # An unexpected igraph version, and the variable that used to pin
        # one, must both leave the import alone.
        script = textwrap.dedent(
            """
            import igraph, igraph._igraph as core
            igraph.__version__ = "0.0.1"
            core.__igraph_version__ = "0.0.1"
            import hedonic
            from hedonic import Game
            g = Game(igraph.Graph.Famous("Zachary"))
            print(len(set(g.community_hedonic(seed=0).membership)))
            """
        )
        env = {"HEDONIC_REQUIRE_IGRAPH_C": "9.9.9-not-a-version", "PATH": ""}
        out = subprocess.run([sys.executable, "-c", script], check=True,
                             capture_output=True, text=True, env=env)
        self.assertGreater(int(out.stdout), 1)


class TestProvenance(unittest.TestCase):
    def test_keys(self):
        for kwargs in (dict(), dict(max_memberships=3, resolution=0.1),
                       dict(n_communities=3), dict(max_memberships=-1, seed=2)):
            with self.subTest(**kwargs):
                result = _karate().community_hedonic(**kwargs)
                provenance = result._hedonic_provenance
                self.assertEqual(set(provenance), PROVENANCE_KEYS)
                self.assertEqual(set(provenance["versions"]),
                                 {"hedonic", "hedonic_core", "python_igraph"})

    def test_values_follow_the_call(self):
        game = _karate()
        game.vs["w"] = [1.0] * 34
        cover = game.community_hedonic(
            initial_membership=None, max_memberships=-1, resolution=0.1, seed=4,
            allow_isolation=True, node_weights="w", max_total_communities=8,
        )
        provenance = cover._hedonic_provenance
        self.assertEqual(provenance["max_memberships"], -1)
        self.assertEqual(provenance["effective_max_memberships"], 34)
        self.assertEqual(provenance["count_constraint"], "at_most")
        self.assertEqual(provenance["max_total_communities"], 8)
        self.assertIsNone(provenance["n_communities"])
        self.assertIsNone(provenance["max_communities"])
        self.assertEqual(provenance["initialization"], "default")
        self.assertTrue(provenance["allow_isolation"])
        self.assertFalse(provenance["edge_weighted"])
        self.assertTrue(provenance["node_weighted"])
        self.assertEqual(provenance["total_edge_weight"], 78.0)
        self.assertEqual(provenance["seed"], 4)
        self.assertEqual(provenance["rng"], "seeded")
        self.assertEqual(provenance["equilibrium_notion"], "count_constrained")
        self.assertEqual(provenance["occupied_communities"],
                         len({c for row in cover.membership for c in row}))
        self.assertEqual(cover._hedonic_native_quality, cover._params["quality"])


class TestMetrics(unittest.TestCase):
    def test_detector_recovers_two_cliques(self):
        game = _two_cliques()
        part = game.community_hedonic(initial_membership=None, seed=0)
        self.assertEqual(part.membership, [0] * 5 + [1] * 5)
        for method in ("f1", "jaccard", "omega", "one_to_one_f1"):
            with self.subTest(method=method):
                self.assertEqual(game.evaluate_against(TWO_CLIQUES_TRUTH, method), 1.0)
        # A flat vector and an igraph clustering are accepted as the reference.
        self.assertEqual(game.evaluate_against([0] * 5 + [1] * 5, "f1"), 1.0)
        self.assertEqual(game.evaluate_against(part, "one_to_one_f1"), 1.0)

    def test_evaluate_cover_perfect_and_singleton(self):
        from hedonic.metrics import evaluate_cover

        perfect = evaluate_cover(TWO_CLIQUES_TRUTH, TWO_CLIQUES_TRUTH, 10)
        for key in ("f1", "jaccard", "omega", "matching_f1"):
            with self.subTest(key=key):
                self.assertEqual(perfect[key], 1.0)
        # Each singleton's best match is its clique (F1 = 2*1*0.2/1.2 = 1/3),
        # and each clique's best match is one of its singletons (also 1/3).
        singletons = evaluate_cover([[v] for v in range(10)], TWO_CLIQUES_TRUTH, 10)
        self.assertAlmostEqual(singletons["f1"], 1 / 3)
        self.assertAlmostEqual(singletons["jaccard"], 1 / 5)
        self.assertLess(singletons["matching_f1"], 1.0)

    def test_overlapping_state_against_an_overlapping_reference(self):
        game = _two_cliques()
        truth = [[0, 1, 2, 3, 4, 5], [5, 6, 7, 8, 9]]
        game.memberships = [[0]] * 5 + [[0, 1]] + [[1]] * 4
        for method in ("f1", "jaccard", "omega", "one_to_one_f1"):
            with self.subTest(method=method):
                self.assertEqual(game.evaluate_against(truth, method), 1.0)
        game.memberships = [0] * 5 + [1] * 5
        self.assertLess(game.evaluate_against(truth, "f1"), 1.0)

    def test_omega_is_exact_on_small_graphs(self):
        from itertools import combinations

        from hedonic.metrics import omega_index

        pred = [[0, 1, 2, 3], [3, 4, 5], [6, 7]]
        truth = [[0, 1, 2], [2, 3, 4, 5, 6], [7, 8]]
        n = 9

        def shared(cover):
            rows = [{c for c, members in enumerate(cover) if v in members} for v in range(n)]
            return [len(rows[u] & rows[v]) for u, v in combinations(range(n), 2)]

        a, b = shared(pred), shared(truth)
        pairs = len(a)
        observed = sum(x == y for x, y in zip(a, b)) / pairs
        expected = sum(a.count(k) * b.count(k) for k in set(a) | set(b)) / pairs**2
        exact = (observed - expected) / (1 - expected)
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                self.assertAlmostEqual(omega_index(pred, truth, n, seed=seed), exact, places=12)

    def test_restrict_to_reference_drops_unlabelled_vertices(self):
        from hedonic.metrics import restrict_cover

        truth = [[0, 1, 2], [3, 4]]
        self.assertEqual(restrict_cover([[0, 1, 9], [7, 8], [3, 4]], truth),
                         [[0, 1], [3, 4]])
        game = _two_cliques()
        game.memberships = [[0]] * 3 + [[1]] * 2 + [[2]] * 5
        self.assertLess(game.evaluate_against(truth, "f1"), 1.0)
        self.assertEqual(game.evaluate_against(truth, "f1", restrict_to_reference=True), 1.0)
        with self.assertRaises(TypeError):
            game.evaluate_against(truth, "f1", restrict_to_reference="yes")

    def test_metrics_is_reachable_from_the_package(self):
        script = ("import sys, hedonic; print('hedonic.metrics' in sys.modules); "
                  "print(hedonic.metrics.restrict_cover([[0, 5]], [[0]]))")
        out = subprocess.run([sys.executable, "-c", script], check=True,
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.split(), ["False", "[[0]]"])

    def test_metrics_import_does_not_load_scipy(self):
        script = "import sys, hedonic.metrics; print('scipy' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", script], check=True,
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


class TestLeanInstall(unittest.TestCase):
    """``pip install hedonic`` is the library alone: igraph and numpy."""

    CLI_ONLY = ("scipy", "pandas", "matplotlib", "networkx", "pyarrow", "duckdb", "tqdm")

    def test_the_base_dependencies_are_igraph_and_numpy(self):
        import re
        import tomllib
        from pathlib import Path

        project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())["project"]
        names = {re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0] for req in project["dependencies"]}
        self.assertEqual(names, {"igraph", "numpy"})
        cli = {re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0] for req in project["optional-dependencies"]["cli"]}
        self.assertEqual(cli, set(self.CLI_ONLY))

    def test_the_package_reports_its_version(self):
        from importlib.metadata import version

        import hedonic

        self.assertEqual(hedonic.__version__, version("hedonic"))

    def test_the_library_runs_without_the_cli_extra(self):
        script = f"""
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {self.CLI_ONLY!r}:
            raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Block())
import igraph as ig
from hedonic import Game, metrics, stability
g = Game(ig.Graph.Famous("Zachary"))
part = g.community_hedonic(max_memberships=1, seed=0)
cover = g.community_hedonic(max_memberships=3, seed=0)
intervals = g.stable_intervals()
regrets = g.regrets([0.1, g.density()])
print(len(part.membership), len(intervals), regrets.regret.shape,
      round(metrics.evaluate_cover([[0, 1]], [[0, 1]], 2, one_to_one=False)["f1"], 3))
"""
        out = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
        self.assertEqual(out.stdout.split(), ["34", "34", "(34,", "2)", "1.0"])


class TestMaxCommunitiesStart(unittest.TestCase):
    """``max_communities`` draws a random start; a draw may miss labels."""

    N, K = 11, 6

    def _starts(self, seed):
        from unittest.mock import patch

        from hedonic import _native

        seen = {}
        real = _native.community_hedonic

        def spy(game, **kwargs):
            seen["start"] = list(kwargs["initial_membership"])
            return real(game, **kwargs)

        with patch.object(_native, "community_hedonic", spy):
            result = Game.Ring(self.N).community_hedonic(
                initial_membership=None, max_communities=self.K, seed=seed
            )
        return seen["start"], result

    def _draws(self):
        from hedonic.utils import sample_uniform_ints

        return {s: sample_uniform_ints(self.N, self.K - 1, s).tolist() for s in range(200)}

    def test_a_draw_that_misses_labels_is_renumbered_in_order(self):
        gappy = {s: d for s, d in self._draws().items() if len(set(d)) < self.K}
        self.assertTrue(gappy)
        for seed, drawn in list(gappy.items())[:10]:
            start, result = self._starts(seed)
            labels = sorted(set(drawn))
            self.assertEqual(start, [labels.index(label) for label in drawn])
            self.assertEqual(
                result._hedonic_provenance["max_communities_effect"],
                "random disjoint start",
            )

    def test_a_complete_draw_is_used_as_drawn(self):
        complete = {s: d for s, d in self._draws().items() if len(set(d)) == self.K}
        self.assertTrue(complete)
        for seed, drawn in list(complete.items())[:10]:
            start, _ = self._starts(seed)
            self.assertEqual(start, drawn)


if __name__ == "__main__":
    unittest.main()
