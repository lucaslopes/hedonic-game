"""Mode parity of ``Game.community_hedonic`` (roadmap section 6.4).

Every option with a shared meaning is exercised for partitions and covers:
both values of ``allow_isolation`` and ``local_move_only``, zero, positive
and negative iteration budgets, default and supplied starts, unconstrained,
at-most and exact community counts, and each diagnostic level. A traced
call returns the untraced result. Invalid combinations raise errors that
name the conflicting option.
"""

from __future__ import annotations

import itertools
import unittest

import igraph as ig

from hedonic import Game


def _karate() -> Game:
    graph = ig.Graph.Famous("Zachary")
    graph.es["weight"] = [1.0 + (edge % 3) / 2 for edge in range(graph.ecount())]
    return Game(graph)


def _occupied(rows, overlapping: bool) -> int:
    if overlapping:
        return len({label for row in rows for label in row})
    return len(set(rows))


class TestModeParityMatrix(unittest.TestCase):
    def test_matrix(self):
        n = 34
        cells = 0
        for (
            max_memberships, allow_isolation, local_move_only, budget, supplied, count,
        ) in itertools.product(
            (1, 3), (False, True), (True, False), (0, 1, -1), (False, True),
            ("none", "at_most", "exact"),
        ):
            overlapping = max_memberships > 1
            limits = {}
            if count == "at_most":
                limits["max_total_communities"] = 5
            elif count == "exact":
                limits["n_communities"] = 4
            start = None
            if supplied:
                k = 4 if count != "none" else 6
                start = (
                    [sorted({v % k, (v + 1) % k}) if v % 2 == 0 else [v % k]
                     for v in range(n)]
                    if overlapping else [v % k for v in range(n)]
                )
            kwargs = dict(
                initial_membership=start, max_memberships=max_memberships,
                allow_isolation=allow_isolation, local_move_only=local_move_only,
                n_iterations=budget, resolution=0.08, edge_weights="weight",
                seed=100 + cells, **limits,
            )
            label = {k: v for k, v in kwargs.items() if k != "initial_membership"}
            with self.subTest(supplied=supplied, **label):
                plain = _karate().community_hedonic(**kwargs)
                full = _karate().community_hedonic(debug_trace=True, **kwargs)
                compact = _karate().community_hedonic(debug_trace="counters", **kwargs)
                self.assertEqual(plain.membership, full.membership)
                self.assertEqual(plain.membership, compact.membership)
                self.assertIsInstance(
                    plain, ig.VertexCover if overlapping else ig.VertexClustering
                )

                trace = full._params["debug_trace"]
                counters = trace["counters"]
                self.assertEqual(trace["mode"], "cover" if overlapping else "partition")
                self.assertEqual(trace["seed_context"]["seed"], 100 + cells)
                self.assertEqual(counters["max_memberships"], max_memberships)
                self.assertEqual(
                    counters["visits"],
                    counters["accepted_moves"] + counters["rejected_visits"],
                )
                self.assertEqual(len(trace["moves"]), counters["accepted_moves"])
                if budget == 0:
                    self.assertEqual(counters["visits"], 0)
                if budget < 0 and not local_move_only:
                    self.assertGreaterEqual(counters["certificate_sweeps"], 1)
                if not overlapping or local_move_only:
                    self.assertEqual(trace["projections"], [])
                self.assertEqual(
                    compact._params["debug_trace"]["counters"]["accepted_moves"],
                    counters["accepted_moves"],
                )

                provenance = plain._hedonic_provenance
                expected_mode = {"none": "none", "at_most": "at_most",
                                 "exact": "exact"}[count]
                self.assertEqual(provenance["count_constraint"], expected_mode)
                self.assertEqual(provenance["allow_isolation"], allow_isolation)
                self.assertEqual(provenance["local_move_only"], local_move_only)
                self.assertIsNone(provenance["debug_trace"])
                self.assertEqual(full._hedonic_provenance["debug_trace"], "full")

                occupied = _occupied(plain.membership, overlapping)
                if count == "at_most":
                    self.assertLessEqual(occupied, 5)
                elif count == "exact":
                    self.assertEqual(occupied, 4)
                for move in trace["moves"]:
                    self.assertLessEqual(move["abs_error"], move["tolerance"])
                    if count == "exact":
                        self.assertEqual(move["occupied_after"], 4)
                cells += 1
        self.assertEqual(cells, 2 * 2 * 2 * 3 * 2 * 3)


class TestMessagesNameTheConflictingOption(unittest.TestCase):
    def test_invalid_combinations(self):
        directed = Game(ig.Graph([(0, 1), (1, 2)], directed=True))
        game = _karate()
        cases = [
            (lambda: directed.community_hedonic(max_memberships=2), "undirected"),
            (lambda: game.community_hedonic(max_memberships=0), "max_memberships"),
            (lambda: game.community_hedonic(n_communities=35),
             "n_communities=35 exceeds the vertex count"),
            (lambda: game.community_hedonic(max_memberships=2, n_communities=69),
             "n_communities=69 exceeds the vertex count times max_memberships"),
            (lambda: game.community_hedonic(max_total_communities=2, n_communities=3),
             "n_communities must not exceed max_total_communities"),
            (lambda: game.community_hedonic(max_total_communities=0),
             "max_total_communities"),
            (lambda: game.community_hedonic(
                initial_membership=list(range(34)), max_total_communities=3),
             "initial_membership.*max_total_communities=3"),
            (lambda: game.community_hedonic(
                max_memberships=2, initial_membership=[[v] for v in range(34)],
                n_communities=3),
             "initial_membership.*n_communities=3"),
            (lambda: game.community_hedonic(
                initial_membership=[[0, 1]] + [[0]] * 33),
             "initial_membership.*max_memberships"),
            (lambda: game.community_hedonic(debug_trace="moves"), "debug_trace"),
            (lambda: game.community_hedonic(resolution=float("nan")), "resolution"),
            (lambda: game.community_hedonic(node_weights=[-1.0] * 34), "node_weights"),
        ]
        for call, pattern in cases:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, pattern):
                    call()

    def test_max_communities_effect_is_recorded(self):
        game = _karate()
        disjoint = game.community_hedonic(initial_membership=None, max_communities=3, seed=1)
        self.assertEqual(
            disjoint._hedonic_provenance["max_communities_effect"], "random disjoint start"
        )
        cover = game.community_hedonic(
            initial_membership=None, max_communities=3, max_memberships=2
        )
        self.assertEqual(
            cover._hedonic_provenance["max_communities_effect"],
            "none: overlapping singleton start",
        )
        supplied = game.community_hedonic(
            initial_membership=[v % 2 for v in range(34)], max_communities=3
        )
        self.assertEqual(
            supplied._hedonic_provenance["max_communities_effect"],
            "none: a start state was supplied",
        )


class TestPhasePolicy(unittest.TestCase):
    def test_disjoint_then_overlap_equals_the_manual_warm_start(self):
        for local_move_only in (True, False):
            for limits in ({}, {"max_total_communities": 6}, {"n_communities": 4}):
                with self.subTest(local_move_only=local_move_only, **limits):
                    common = dict(
                        resolution=0.08, edge_weights="weight", seed=7,
                        local_move_only=local_move_only, allow_isolation=False,
                        **limits,
                    )
                    two_stage = _karate().community_hedonic(
                        initial_membership=None, max_memberships=3,
                        phase_policy="disjoint_then_overlap", **common,
                    )
                    game = _karate()
                    partition = game.community_hedonic(
                        initial_membership=None, max_memberships=1, **common
                    )
                    manual = game.community_hedonic(
                        initial_membership=list(partition.membership),
                        max_memberships=3, **common,
                    )
                    self.assertEqual(two_stage.membership, manual.membership)
                    self.assertEqual(
                        two_stage._hedonic_disjoint_stage.membership,
                        partition.membership,
                    )
                    provenance = two_stage._hedonic_provenance
                    self.assertEqual(provenance["phase_policy"], "disjoint_then_overlap")
                    self.assertEqual(provenance["initialization"], "disjoint_stage")
                    self.assertEqual(
                        provenance["disjoint_stage"]["occupied_communities"],
                        len(set(partition.membership)),
                    )
                    if "n_communities" in limits:
                        self.assertEqual(_occupied(two_stage.membership, True), 4)

    def test_direct_is_the_default_and_records_its_start(self):
        cover = _karate().community_hedonic(
            initial_membership=None, max_memberships=2, resolution=0.08, seed=1
        )
        self.assertEqual(cover._hedonic_provenance["phase_policy"], "direct")
        self.assertEqual(cover._hedonic_provenance["initialization"], "default")
        self.assertFalse(hasattr(cover, "_hedonic_disjoint_stage"))

    def test_trace_describes_the_overlapping_stage(self):
        cover = _karate().community_hedonic(
            initial_membership=None, max_memberships=3, resolution=0.08, seed=2,
            phase_policy="disjoint_then_overlap", debug_trace=True,
        )
        trace = cover._params["debug_trace"]
        self.assertEqual(trace["mode"], "cover")
        self.assertEqual(len(trace["moves"]), trace["counters"]["accepted_moves"])

    def test_invalid_phase_policy_combinations_name_the_option(self):
        game = _karate()
        with self.assertRaisesRegex(ValueError, "phase_policy"):
            game.community_hedonic(phase_policy="two_stage")
        with self.assertRaisesRegex(ValueError, "phase_policy.*max_memberships > 1"):
            game.community_hedonic(max_memberships=1, phase_policy="disjoint_then_overlap")
        with self.assertRaisesRegex(ValueError, "phase_policy.*initial_membership"):
            game.community_hedonic(
                initial_membership=[0] * 34, max_memberships=2,
                phase_policy="disjoint_then_overlap",
            )


if __name__ == "__main__":
    unittest.main()
