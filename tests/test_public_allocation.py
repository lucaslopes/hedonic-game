"""Focused regression tests for the public-allocation-v0.1 prototype."""

from __future__ import annotations

import unittest

import numpy as np

from hedonic.experiments.overlapping import public_allocation as pa


class TestPublicAllocation(unittest.TestCase):
    def _problem(self, **overrides):
        values = {
            "adjacency": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "project_costs": [1.0, 2.0],
            "budget": 2.0,
            "capacities": [2.0, 2.0],
            "contribution_budgets": [1.0, 1.0, 1.0],
            "theta": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        }
        values.update(overrides)
        return pa.PublicAllocationProblem(**values)

    def test_constructor_accepts_equation_and_descriptive_names(self):
        problem = self._problem()
        self.assertEqual(problem.n_agents, 3)
        self.assertEqual(problem.n_projects, 2)
        np.testing.assert_allclose(problem.A, problem.adjacency)
        np.testing.assert_allclose(problem.p, problem.project_costs)
        np.testing.assert_allclose(problem.ybar, problem.capacities)
        np.testing.assert_allclose(problem.b, problem.contribution_budgets)
        self.assertEqual(problem.mode, pa.MODE_VIRTUAL_SHARE)
        self.assertEqual(problem.cost_model, "zero_direct_cost_virtual_share")

    def test_validation_rejects_non_loopless_or_asymmetric_network(self):
        with self.assertRaisesRegex(ValueError, "loopless"):
            self._problem(adjacency=[[1, 0], [0, 0]], project_costs=[1], capacities=[1], contribution_budgets=[1, 1])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            self._problem(adjacency=[[0, 1], [0, 0]], project_costs=[1], capacities=[1], contribution_budgets=[1, 1])

    def test_virtual_share_requires_exact_weighted_budget(self):
        problem = self._problem()
        valid = np.asarray([[1.0, 0.0], [0.0, 0.5], [1.0, 0.0]])
        np.testing.assert_allclose(problem.contribution_spend(valid), [1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "exactly|p @ z"):
            problem.validate_contributions([[1.0, 0.0], [0.0, 0.4], [1.0, 0.0]])

    def test_voluntary_payment_budget_and_declared_cost(self):
        problem = self._problem(
            mode="voluntary_payment",
            payment_rates=[2.0, 3.0, 5.0],
        )
        profile = np.asarray([[0.5, 0.0], [0.0, 0.25], [0.0, 0.0]])
        result = problem.aggregate(profile)
        np.testing.assert_allclose(result.contribution_costs, [1.0, 1.5, 0.0])
        np.testing.assert_allclose(result.preference_values, problem.theta @ result.outcome)
        np.testing.assert_allclose(result.utilities, result.preference_values - result.contribution_costs)
        with self.assertRaisesRegex(ValueError, "exceed"):
            problem.validate_contributions([[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def test_additive_aggregation_and_weighted_budget_projection(self):
        problem = self._problem(eta=0.0)
        profile = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 0.5]])
        result = problem.aggregate(profile)
        np.testing.assert_allclose(result.project_totals, [2.0, 0.5])
        np.testing.assert_allclose(result.synergy, [1.0, 0.0])
        np.testing.assert_allclose(result.raw_score, result.project_totals)
        # Weighted budget p=(1,2) binds: y=(2-lambda, .5-2lambda)
        # with lambda=.2, hence y=(1.8,.1).
        np.testing.assert_allclose(result.outcome, [1.8, 0.1])
        self.assertTrue(result.feasible)
        self.assertTrue(problem.is_feasible_outcome(result.outcome))
        self.assertFalse(problem.is_feasible_outcome([2.0, 0.5]))
        self.assertAlmostEqual(result.budget_slack, 0.0)
        self.assertGreater(result.projection_residual, 0.0)

    def test_network_synergy_is_sqrt_normalized_and_deterministic(self):
        problem = self._problem(eta=2.0, project_costs=[1.0, 1.0])
        profile = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.0, 1.0]])
        first = problem.aggregate(profile)
        second = problem.aggregate(profile.copy())
        # Edge 0--1 contributes sqrt(.5*.5)=.5 to both projects; edge 1--2
        # contributes zero to project A and sqrt(.5*1)=sqrt(.5) to B.
        np.testing.assert_allclose(first.synergy, [0.5, 0.5 + np.sqrt(0.5)])
        np.testing.assert_array_equal(first.outcome, second.outcome)
        np.testing.assert_array_equal(first.raw_score, second.raw_score)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_projection_matches_known_box_and_weighted_budget_cases(self):
        # Inactive budget: simple clipping is the Euclidean projection.
        inactive = pa.project_with_diagnostics([2.0, -1.0], [1.0, 1.0], 10.0, [1.0, 3.0])
        np.testing.assert_allclose(inactive.y, [1.0, 0.0])
        self.assertFalse(inactive.budget_active)
        # Active weighted budget: KKT solution satisfies y=(2-lambda,
        # 3-2lambda) and lambda=1.2, giving [0.8,.6].
        active = pa.project_with_diagnostics([2.0, 3.0], [1.0, 2.0], 2.0, [10.0, 10.0])
        np.testing.assert_allclose(active.y, [0.8, 0.6], atol=1e-9)
        self.assertTrue(active.budget_active)
        self.assertTrue(active.feasible)
        np.testing.assert_allclose(pa.project_to_feasible([2.0, 3.0], [1.0, 2.0], 2.0, [10.0, 10.0]), active.y)

    def test_linear_preference_helper_and_json_record(self):
        theta = [[1, 2], [-1, 1]]
        y = [0.25, 0.5]
        np.testing.assert_allclose(pa.linear_preferences(theta, y), [1.25, 0.25])
        result = self._problem().aggregate([[1, 0], [0, 0.5], [1, 0]])
        record = result.to_dict()
        self.assertEqual(record["protocol_version"], pa.PROTOCOL_VERSION)
        self.assertEqual(record["schema_version"], pa.SCHEMA_VERSION)
        self.assertIsInstance(record["outcome"], list)
        self.assertTrue(record["feasible"])

    def test_proposition_counterexamples_are_exhaustive_and_exact(self):
        report = pa.validate_counterexamples()
        self.assertTrue(report["ok"])
        self.assertEqual(report["profiles_checked"], 16)
        self.assertTrue(report["additive_totals_have_payoff_collision"])
        fixtures = pa.counterexample_fixtures()
        first = fixtures["proposition_1"]
        second = fixtures["proposition_2"]
        np.testing.assert_allclose(pa._binary_totals(first.state), [2, 2])
        np.testing.assert_allclose(pa.binary_network_payoffs(first.adjacency, first.state), [1, 1, 1, 1])
        np.testing.assert_allclose(pa.binary_network_payoffs(first.adjacency, first.alternate_state), [0, 0, 0, 0])
        np.testing.assert_allclose(pa.weighted_pair_synergy(second.adjacency, second.z), [0, 1])
        np.testing.assert_allclose(pa.weighted_pair_synergy(second.adjacency, second.z_prime), [0, 1])
        self.assertNotEqual(second.expected_payoffs, second.expected_alternate_payoffs)

    def test_no_claim_of_fairness_or_truthfulness_is_encoded(self):
        # The result records preferences, costs, and welfare only; it does not
        # label any equilibrium/fairness status that the design has not proved.
        result = self._problem().aggregate([[1, 0], [0, 0.5], [1, 0]])
        self.assertNotIn("fairness", result.to_dict())
        self.assertNotIn("truthful", result.to_dict())
        self.assertNotIn("equilibrium", result.to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
