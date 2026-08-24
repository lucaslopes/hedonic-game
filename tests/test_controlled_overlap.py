"""Focused tests for the LFR-derived controlled-overlap experiment."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import controlled_overlap


class TestControlledOverlap(unittest.TestCase):
    def assert_implementation_identity_current(self, identity):
        self.assertEqual(identity["algorithm"], "sha256")
        self.assertEqual(identity["path_scope"], "repository-relative")
        repository_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            set(identity["source_files"]),
            set(controlled_overlap.IMPLEMENTATION_SOURCE_FILES),
        )
        for relative, recorded_digest in identity["source_files"].items():
            self.assertEqual(
                recorded_digest,
                hashlib.sha256((repository_root / relative).read_bytes()).hexdigest(),
                relative,
            )
        descriptor = {
            key: value
            for key, value in identity.items()
            if key != "identity_sha256"
        }
        self.assertEqual(
            identity["identity_sha256"],
            controlled_overlap._sha256_json(descriptor),
        )
        self.assertEqual(
            identity["environment"], controlled_overlap._environment_identity()
        )

    def _instance(self):
        return controlled_overlap.generate_controlled_cover(
            n=80,
            mu=0.2,
            overlap_fraction=0.2,
            memberships_per_overlapping_vertex=2,
            secondary_edge_probability=0.2,
            seed=0,
            average_degree=5,
            min_community=15,
            max_community=30,
        )

    def test_construction_is_deterministic_and_realizes_overlap(self):
        first = self._instance()
        second = self._instance()
        self.assertEqual(first.graph.get_edgelist(), second.graph.get_edgelist())
        self.assertEqual(first.cover, second.cover)
        self.assertEqual(len(first.overlapping_vertices), 16)
        memberships = [
            sum(vertex in community for community in first.cover)
            for vertex in range(first.graph.vcount())
        ]
        self.assertTrue(all(memberships[v] == 2 for v in first.overlapping_vertices))
        self.assertTrue(
            all(
                memberships[v] == 1
                for v in range(first.graph.vcount())
                if v not in first.overlapping_vertices
            )
        )
        self.assertEqual(
            first.metadata["construction_label"],
            "LFR-derived controlled overlap",
        )
        self.assertIn("not canonical overlapping LFR", first.metadata["construction_caveat"])

    def test_one_real_detector_run_records_ablation_and_metrics(self):
        instance = self._instance()
        rows = controlled_overlap.run_ablations(
            instance,
            phases=["local"],
            cap_specs=["gt"],
            starts=["gt-primary"],
            resolution_multipliers=[1.0],
            seed=0,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["max_memberships"], 2)
        self.assertEqual(row["initialization_supervision"], "ground-truth primary labels")
        self.assertEqual(row["local_move_only"], True)
        self.assertEqual(row["allow_isolation"], True)
        self.assertIn("matching_f1", row)
        self.assertEqual(row["omega"], None)
        summary = controlled_overlap._summary(rows)
        self.assertEqual(summary[0]["sd_matching_f1"], 0.0)
        self.assertEqual(
            summary[0]["ci95_low_matching_f1"],
            summary[0]["mean_matching_f1"],
        )

    def test_known_native_offender_reaches_equilibrium_after_native_fix(self):
        instance = controlled_overlap.generate_controlled_cover(
            n=80,
            mu=0.2,
            overlap_fraction=0.1,
            memberships_per_overlapping_vertex=2,
            secondary_edge_probability=0.1,
            seed=0,
            average_degree=5,
            min_community=15,
            max_community=30,
            lfr_max_iters=500,
        )
        rows = controlled_overlap.run_ablations(
            instance,
            phases=["local"],
            cap_specs=["2"],
            starts=["singleton"],
            resolution_multipliers=[1.0],
            seed=0,
            timeout_seconds=0.1,
        )
        # The released lucas-igraph 1.0.0.2 native guard/termination fix
        # resolves the historical non-terminating cell within the watchdog.
        self.assertEqual(rows[0]["status"], "completed")
        summary = controlled_overlap._summary(rows)
        self.assertEqual(summary[0]["n_expected"], 1)
        self.assertEqual(summary[0]["n_completed"], 1)
        self.assertEqual(summary[0]["n_timeout"], 0)
        self.assertIsNotNone(summary[0]["mean_matching_f1"])

    def test_detector_seed_is_paired_across_ablation_cells(self):
        rows = controlled_overlap.run_ablations(
            self._instance(),
            phases=["local", "multiphase"],
            cap_specs=["1", "gt"],
            starts=["singleton", "gt-primary"],
            resolution_multipliers=[1.0, 10.0],
            seed=7,
        )
        self.assertEqual({row["seed"] for row in rows}, {7})
        self.assertEqual({row["allow_isolation"] for row in rows}, {True})

    def test_cli_registration_help_and_csv_output(self):
        command = CLI.COMMANDS["overlapping-controlled"]
        self.assertEqual(
            command.module,
            "hedonic.experiments.overlapping.controlled_overlap",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            code = CLI.main(["overlapping-controlled", "--help"])
        self.assertEqual(code, 0)

        # Restrict the grid to one run; this tests the shipped JSON/CSV path
        # without turning a unit test into the 16-run smoke experiment.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            status = controlled_overlap.main(
                [
                    "--n",
                    "80",
                    "--mus",
                    "0.2",
                    "--overlap-fractions",
                    "0.2",
                    "--overlap-memberships",
                    "2",
                    "--secondary-edge-probabilities",
                    "0.2",
                    "--graph-seeds",
                    "0",
                    "--average-degree",
                    "5",
                    "--min-community",
                    "15",
                    "--max-community",
                    "30",
                    "--phases",
                    "local",
                    "--max-memberships",
                    "gt",
                    "--starts",
                    "gt-primary",
                    "--resolution-multipliers",
                    "1",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".csv").is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assert_implementation_identity_current(
                payload["implementation_identity"]
            )
            self.assertEqual(
                payload["environment"],
                payload["implementation_identity"]["environment"],
            )
            self.assertEqual(
                payload["implementation_identity"],
                payload["protocol"]["implementation_identity"],
            )

    def test_tracked_v2_is_strict_and_bound_to_code_and_protocol(self):
        artifact = (
            Path(__file__).resolve().parents[1]
            / "artifacts/evidence/overlapping_communities/controlled_overlap_v2.json"
        )
        if not artifact.is_file():
            self.skipTest("private paper reference artifact is not present in this checkout")
        raw = artifact.read_text(encoding="utf-8")
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
        self.assertEqual(
            payload["implementation_sha256"],
            controlled_overlap._implementation_sha256(),
        )
        if payload["implementation_identity"]["environment"] != controlled_overlap._environment_identity():
            self.skipTest(
                "controlled_overlap_v2 is historical evidence recorded under "
                "lucas-igraph 1.0.0.2; rerun the ledger before asserting current "
                "implementation identity"
            )
        self.assert_implementation_identity_current(
            payload["implementation_identity"]
        )
        self.assertEqual(
            payload["implementation_identity"],
            payload["protocol"]["implementation_identity"],
        )
        self.assertEqual(
            payload["protocol_sha256"],
            controlled_overlap._sha256_json(payload["protocol"]),
        )
        self.assertEqual(len(payload["instances"]), 24)
        self.assertEqual(sum(len(item["runs"]) for item in payload["instances"]), 864)
        self.assertEqual(len(payload["summary"]), 36)
        self.assertTrue(all(row["n_expected"] == 24 for row in payload["summary"]))
        self.assertEqual(len(payload["condition_summary"]), 288)
        self.assertTrue(
            all(row["n_expected"] == 3 for row in payload["condition_summary"])
        )
        self.assertEqual(payload["protocol"]["detector"]["allow_isolation"], True)
        self.assertEqual(payload["protocol"]["construction"]["n"], 80)
        self.assertEqual(
            payload["protocol"]["detector"]["paired_detector_seed_rule"],
            "detector_seed = graph_seed",
        )
        self.assertEqual(
            payload["protocol"]["detector"]["hard_timeout_seconds_per_cell"],
            5.0,
        )
        for key in (
            "python",
            "networkx",
            "lucas_igraph_distribution",
            "igraph_distribution",
            "hedonic",
        ):
            self.assertTrue(payload["environment"][key], key)


if __name__ == "__main__":
    unittest.main()
