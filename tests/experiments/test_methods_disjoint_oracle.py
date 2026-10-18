"""The ground-truth disjoint oracle: a partition built from the reference cover, its runner hook and front-door control."""

from __future__ import annotations

import argparse
import itertools
import unittest

import igraph as ig

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import disjoint_oracle as do
from hedonic.experiments.overlapping import extra_methods
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping.metrics import symmetric_best_match_f1

# Two overlapping communities, one nested in a larger one, a duplicate and a vertex shared by three.
REFERENCE = [[0, 1, 2, 3], [3, 4, 5], [5, 6, 7, 8, 9, 10], [6, 7, 8], [6, 7, 8], [3, 5, 11]]


class TestDisjointOracle(unittest.TestCase):
    def test_answer_is_a_partition_of_the_covered_vertices_into_reference_subsets(self):
        blocks, report = do.oracle_partition(REFERENCE)
        members = [v for block in blocks for v in block]
        self.assertEqual(len(members), len(set(members)))
        self.assertEqual(set(members), {v for c in REFERENCE for v in c})
        for block in blocks:
            self.assertTrue(any(set(block) <= set(c) for c in REFERENCE))
        self.assertEqual(report["reference_communities"], 5)  # the duplicate counts once
        self.assertEqual(len(report["candidates"]), 2 * len(do.RULES))
        self.assertAlmostEqual(report["f1"], symmetric_best_match_f1(blocks, REFERENCE))
        self.assertEqual(report["f1"], max(c["f1"] for c in report["candidates"]))

    def test_a_disjoint_reference_is_returned_exactly(self):
        reference = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        blocks, report = do.oracle_partition(reference)
        self.assertEqual(blocks, reference)
        self.assertEqual(report["f1"], 1.0)

    def test_it_is_the_best_disjoint_answer_on_a_small_reference(self):
        # Every assignment of each vertex to one of its own communities: the oracle reaches the best of them here.
        communities = do._communities(REFERENCE)
        member_of = do._memberships(communities)
        vertices = sorted(member_of)
        best = max(symmetric_best_match_f1(do._blocks(dict(zip(vertices, choice))), communities)
                   for choice in itertools.product(*(member_of[v] for v in vertices)))
        self.assertAlmostEqual(do.oracle_partition(REFERENCE)[1]["f1"], best)

    def test_moves_never_lower_the_surrogate_and_rules_are_checked(self):
        communities = do._communities(REFERENCE)
        member_of = do._memberships(communities)
        for rule in do.RULES:
            _, search = do.improve(do.start(rule, communities, member_of), communities, member_of)
            self.assertGreaterEqual(search["surrogate_after"], search["surrogate_before"])
        with self.assertRaises(ValueError):
            do.start("random", communities, member_of)
        self.assertEqual(do.oracle_partition([])[0], [])

    def test_runner_hook_reads_the_reference_and_adds_no_parameters(self):
        extra_methods.install()
        self.assertIn("disjoint_oracle", cr.METHOD_SPECS)
        self.assertIn("disjoint_oracle", cr.EXTENDED_METHODS)
        cover, meta = cr._run_method("disjoint_oracle", ig.Graph(12), REFERENCE, 0, {})
        self.assertTrue(meta["supervised"])
        self.assertEqual(meta["max_memberships"], 1)
        self.assertEqual(cover, do.oracle_partition(REFERENCE)[0])
        args = cr._build_parser().parse_args([])
        self.assertNotIn("disjoint_oracle", cr._method_parameters(args))  # existing records keep their digests
        self.assertEqual(cr._method_preflight("disjoint_oracle", {})["status"], "ready")

    def test_front_door_offers_it_as_a_control_outside_all(self):
        universe = [m.key for m in qs.METHODS]
        self.assertNotIn("disjoint_oracle", qs.parse_choice("all", universe, qs.ALIASES, "method", qs.CONTROL_KEYS))
        self.assertEqual(qs.parse_choice("hedonic,disjoint_oracle", universe, qs.ALIASES, "method", qs.CONTROL_KEYS),
                         ["hedonic", "disjoint_oracle"])
        with self.assertRaises(ValueError):
            qs.parse_choice("disjoint_oracle", universe, qs.ALIASES, "method")
        cfg = qs.config_from_args(argparse.Namespace(**{**vars(qs.build_parser().parse_args([])),
                                                        "config": "paper-oracle"}))
        self.assertEqual(cfg.methods, ["disjoint_oracle"])
        self.assertEqual((cfg.nodes, cfg.seeds, cfg.save_covers, cfg.robustness), (0, [0], True, True))
        self.assertEqual(qs.BY_KEY["disjoint_oracle"].runner, "disjoint_oracle")


if __name__ == "__main__":
    unittest.main()
