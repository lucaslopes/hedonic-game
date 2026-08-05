"""Tests for the locked Chapter 5 small-instance DNN diagnostic."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import dnn_certificate as dnn


EXPECTED_INSTANCE_HASHES = {
    "path4": "4dad1bc08c1e1a63389681f42278140f6999c75fc484e493214638c12f5e7f63",
    "bow_tie5": "71f9eeb2eccc24d0df1837c2a3ca3aa3361ab68831ca552d2b2085a4967c0308",
    "weighted_bridge5": "27ddec51fde8aec34c033dfe4d0b81216b16b9c89d1dd998e663de7b8e5bef3d",
}


class TestDnnCertificate(unittest.TestCase):
    def test_locked_instances_have_stable_content_hashes(self):
        self.assertEqual(
            {
                name: dnn.instance_sha256(instance)
                for name, instance in dnn.LOCKED_INSTANCES.items()
            },
            EXPECTED_INSTANCE_HASHES,
        )

    def test_exact_enumeration_path4(self):
        result = dnn.exact_valid_cover_optimum(dnn.LOCKED_INSTANCES["path4"])
        self.assertEqual(result["candidate_count"], 3**4)
        self.assertAlmostEqual(result["objective"], 0.2072504807090734, places=13)
        self.assertTrue(result["verification"]["valid"])
        factor = np.asarray(result["factor"])
        gram = np.asarray(result["gram"])
        np.testing.assert_allclose(gram, factor @ factor.T, atol=1e-14)

    def test_hedonic_result_is_a_valid_enumerated_cover(self):
        instance = dnn.LOCKED_INSTANCES["bow_tie5"]
        exact = dnn.exact_valid_cover_optimum(instance)
        result = dnn.run_hedonic(instance, seed=0)
        self.assertTrue(result["verification"]["valid"])
        self.assertLessEqual(
            result["objective_recomputed_eq_5_10"], exact["objective"] + 1e-12
        )
        self.assertAlmostEqual(
            result["objective_recomputed_eq_5_10"], result["native_quality"], places=12
        )

    @unittest.skipUnless(importlib.util.find_spec("cvxpy"), "experiments extra absent")
    def test_dnn_chain_and_raw_solver_evidence(self):
        result = dnn.run_instance(
            dnn.LOCKED_INSTANCES["path4"], seeds=[0], eps=1e-7, max_iters=100_000
        )
        chain = result["certificate_chain"]
        self.assertTrue(chain["algorithm_le_exact"])
        self.assertTrue(chain["exact_le_dnn_numerical"])
        self.assertTrue(chain["dnn_numerical_le_repaired_dual"])
        self.assertGreater(chain["valid_cover_to_dnn_numerical_outer_gap"], 0.0)
        solver = result["dnn"]["solver"]
        self.assertEqual(solver["name"], "SCS")
        self.assertEqual(solver["requested_tolerances"]["eps_abs"], 1e-7)
        self.assertIn("status", solver["raw_status"])
        self.assertGreaterEqual(
            result["dnn"]["dual_upper_checks"][
                "repaired_psd_residual_minimum_eigenvalue"
            ],
            -1e-12,
        )

    def test_manifest_is_strict_json_and_cli_is_registered(self):
        self.assertIn("overlapping-dnn", CLI.COMMANDS)
        manifest = dnn.build_manifest(["path4"], seeds=[0], exact_only=True)
        encoded = json.dumps(manifest, allow_nan=False)
        self.assertEqual(json.loads(encoded)["protocol"]["protocol_version"], dnn.PROTOCOL_VERSION)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = CLI.main(["overlapping-dnn", "--help"])
        self.assertEqual(code, 0)
        self.assertIn("doubly-nonnegative", output.getvalue())

    def test_tracked_reference_artifact_is_bound_to_current_protocol(self):
        artifact_path = (
            Path(__file__).parents[1]
            / "docs"
            / "papers"
            / "overlapping_communities"
            / "evidence"
            / "dnn_certificate_v1.json"
        )
        if not artifact_path.is_file():
            self.skipTest("private paper reference artifact is not present in this checkout")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(
            artifact["protocol"]["implementation_sha256"],
            dnn.implementation_sha256(),
        )
        self.assertEqual(
            artifact["protocol_sha256"], dnn.sha256_json(artifact["protocol"])
        )
        self.assertTrue(
            all(
                result["certificate_chain"]["algorithm_le_exact"]
                and result["certificate_chain"]["exact_le_dnn_numerical"]
                and result["certificate_chain"]["dnn_numerical_le_repaired_dual"]
                for result in artifact["results"]
            )
        )

    def test_cli_writes_requested_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            code = dnn.main(
                [
                    "--instances",
                    "path4",
                    "--seeds",
                    "0",
                    "--exact-only",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text())["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()
