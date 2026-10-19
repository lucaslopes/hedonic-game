"""The exact robustness audit and the ground-truth helpers the spectrum and the start-quality curve share.

Ported from the 1.x ``TestGroundTruthRobustness`` (its v3 runner is not part of hedonic 2.0; these are the helpers that
remain): content-addressed artifacts, raw memberships, the prefix best response against the exhaustive oracle, the
endpoint certificate and its disjoint limit, incidence switches, the tiny nearest-equilibrium calibration, and the
explicit partial-cover policies.
"""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import igraph as ig

from hedonic import Game
from hedonic.experiments.disjoint import sbm_sweep
from hedonic.experiments.overlapping.ground_truth_data import (
    covered_induced_dataset,
    prepare_dataset,
    singleton_completed_dataset,
)
from hedonic.experiments.overlapping.ground_truth_robustness import (
    _load_cover_artifact,
    _load_membership_artifact,
    _normalize_raw_memberships,
    _persist_cover,
    _persist_memberships,
    _raw_cover_hash,
)
from hedonic.experiments.overlapping.robustness import (
    audit_cover,
    best_response,
    build_fractional_state,
    cover_hash,
    cover_to_vertex_memberships,
    exhaustive_best_response,
    fractional_phi,
    nearest_equilibrium_tiny,
    perturb_cover_incidence,
)
from hedonic.experiments.overlapping.snap import SnapDataset


class TestGroundTruthHelpers(unittest.TestCase):
    def test_rescore_artifact_loaders_reject_tampered_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cover_digest = _persist_cover(output, [[0, 1], [1, 2]])
            self.assertEqual(
                _load_cover_artifact(output, cover_digest), [[0, 1], [1, 2]]
            )
            with gzip.open(
                output / "covers" / f"{cover_digest}.json.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                json.dump([[0, 2]], stream)
            self.assertIsNone(_load_cover_artifact(output, cover_digest))

            memberships = [[0], [0, 1], [1]]
            membership_digest = _persist_memberships(output, memberships, 3)
            self.assertEqual(
                _load_membership_artifact(output, membership_digest), memberships
            )
            with gzip.open(
                output / "raw_memberships" / f"{membership_digest}.json.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                json.dump([[0], [0], [0]], stream)
            self.assertIsNone(
                _load_membership_artifact(output, membership_digest)
            )

    def test_raw_memberships_preserve_duplicate_communities(self):
        raw = _normalize_raw_memberships(
            [[0, 1], [0, 1], [0, 1], [2], [2], [2]],
            n_vertices=6,
            cap=2,
        )
        raw_cover = []
        for label in range(3):
            raw_cover.append(
                [vertex for vertex, labels in enumerate(raw) if label in labels]
            )
        self.assertEqual(raw_cover, [[0, 1, 2], [0, 1, 2], [3, 4, 5]])
        self.assertNotEqual(_raw_cover_hash(raw_cover), cover_hash(raw_cover))

    @staticmethod
    def _fixture():
        graph = ig.Graph(
            n=6,
            edges=[
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 2),
                (4, 5),
                (5, 0),
            ],
        )
        cover = [[0, 1, 2], [2, 3, 4], [0, 4, 5]]
        return graph, cover

    def test_prefix_best_response_matches_exhaustive_oracle(self):
        graph, cover = self._fixture()
        memberships = cover_to_vertex_memberships(cover, graph.vcount())
        state = build_fractional_state(graph, memberships)
        for gamma in (0.0, 0.2, 0.7, 1.0):
            for allow_isolation in (False, True):
                for vertex in range(graph.vcount()):
                    fast = best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                        dense=True,
                    )
                    slow = exhaustive_best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                    )
                    self.assertAlmostEqual(fast["best_utility"], slow["best_utility"])
                    self.assertAlmostEqual(fast["regret"], slow["regret"])
                    sparse = best_response(
                        state,
                        vertex,
                        gamma,
                        max_memberships=3,
                        allow_isolation=allow_isolation,
                        dense=False,
                    )
                    self.assertAlmostEqual(sparse["best_utility"], slow["best_utility"])

    def test_endpoint_certificate_and_disjoint_limit(self):
        graph, cover = self._fixture()
        memberships = cover_to_vertex_memberships(cover, graph.vcount())
        audit = audit_cover(
            graph,
            memberships,
            max_memberships=3,
            allow_isolation=False,
            gamma=0.5,
            dense=True,
        )
        direct = 0
        state = build_fractional_state(graph, memberships)
        for vertex in range(graph.vcount()):
            endpoints = [
                best_response(state, vertex, gamma, 3, False, dense=True)
                for gamma in (0.0, 1.0)
            ]
            direct += int(all(item["regret"] <= 1e-9 for item in endpoints))
        self.assertEqual(audit["robust_vertex_count_gamma_0_1"], direct)
        self.assertEqual(
            audit["robust_fraction_gamma_0_1"],
            direct / graph.vcount(),
        )

        # With one membership per vertex the fractional potential reduces to
        # the ordinary clique_percolation expression under the same 2m normalization.
        disjoint = [[0, 1, 2], [3, 4, 5]]
        disjoint_memberships = cover_to_vertex_memberships(disjoint, graph.vcount())
        gamma = 0.25
        internal = sum(
            1
            for first, second in graph.get_edgelist()
            if any(first in community and second in community for community in disjoint)
        )
        expected = (2.0 * internal - gamma * sum(len(c) ** 2 for c in disjoint)) / (
            2.0 * graph.ecount()
        )
        self.assertAlmostEqual(
            fractional_phi(graph, disjoint_memberships, gamma), expected
        )
        disjoint_audit = audit_cover(
            graph,
            disjoint_memberships,
            max_memberships=1,
            allow_isolation=False,
            gamma=graph.density(),
            dense=True,
        )
        self.assertAlmostEqual(
            disjoint_audit["robust_fraction_gamma_0_1"],
            sbm_sweep.robustness(Game(graph), [0, 0, 0, 1, 1, 1]),
        )

    def test_incidence_switch_preserves_both_degree_sequences(self):
        graph, cover = self._fixture()
        perturbed, metadata = perturb_cover_incidence(
            cover,
            graph.vcount(),
            swaps=1,
            seed=4,
        )
        original_memberships = cover_to_vertex_memberships(cover, graph.vcount())
        final_memberships = cover_to_vertex_memberships(perturbed, graph.vcount())
        self.assertEqual(
            list(map(len, original_memberships)), list(map(len, final_memberships))
        )
        self.assertEqual(sorted(map(len, cover)), sorted(map(len, perturbed)))
        self.assertTrue(metadata["vertex_membership_counts_preserved"])
        self.assertTrue(metadata["community_sizes_preserved"])
        self.assertEqual(metadata["successful_swaps"], 1)
        self.assertAlmostEqual(
            metadata["realized_incidence_distance"],
            2.0
            * metadata["successful_swaps"]
            / metadata["initial_incidence_count"],
        )

    def test_tiny_nearest_equilibrium_calibration(self):
        graph = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3), (3, 0)])
        result = nearest_equilibrium_tiny(
            graph,
            [[0, 1], [2, 3]],
            gamma=0.5,
            n_communities=2,
            max_memberships=1,
            allow_isolation=False,
        )
        self.assertEqual(result["status"], "completed")
        self.assertGreater(result["candidate_count"], 0)
        self.assertGreater(result["equilibrium_count"], 0)
        self.assertIsNotNone(result["nearest"])
        self.assertGreaterEqual(result["nearest"]["distance_to_ground_truth"], 0.0)

    def test_partial_cover_policies_are_explicit(self):
        graph = ig.Graph(n=5, edges=[(0, 1), (1, 2), (2, 3), (3, 4)])
        raw = SnapDataset(
            "fixture",
            "all",
            graph,
            [[0, 1], [1, 2]],
            {"id_mapping_strategy": "fixture"},
        )
        induced = covered_induced_dataset(raw)
        self.assertEqual(induced.graph.vcount(), 3)
        self.assertEqual(induced.cover, [[0, 1], [1, 2]])
        self.assertEqual(
            induced.report["ground_truth_completion"]["policy"], "covered-induced"
        )
        completed = singleton_completed_dataset(raw)
        self.assertEqual(completed.graph.vcount(), 5)
        self.assertEqual(completed.cover[-2:], [[3], [4]])
        self.assertEqual(
            completed.report["ground_truth_completion"]["synthetic_memberships_added"],
            2,
        )
        prepared = prepare_dataset(raw, policy="covered-induced")
        self.assertEqual(prepared.policy, "covered-induced")
        self.assertEqual(prepared.dataset.graph.vcount(), 3)
        self.assertTrue(prepared.graph_identity)


if __name__ == "__main__":
    unittest.main()
