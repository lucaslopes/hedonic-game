"""Steps look at the capacity available before they run, and refuse or shrink instead of exhausting the machine."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from hedonic.experiments.overlapping import capacity, parallel


GIB = 1024 ** 3


class TestBudget(unittest.TestCase):
    def test_the_budget_is_free_memory_minus_the_floor_and_the_margin(self):
        with patch.object(capacity, "total_bytes", return_value=64 * GIB), \
                patch.object(parallel, "free_memory_percent", return_value=80.0), \
                patch.object(parallel, "memory_floor", return_value=10.0), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop(capacity.BUDGET_ENV, None)
            self.assertEqual(capacity.budget_bytes(), int(64 * GIB * (80 - 10 - capacity.MARGIN_PERCENT) / 100))

    def test_no_room_means_a_zero_budget_and_an_unmeasurable_machine_means_none(self):
        with patch.object(capacity, "total_bytes", return_value=64 * GIB), \
                patch.object(parallel, "free_memory_percent", return_value=12.0), patch.object(parallel, "memory_floor", return_value=10.0):
            os.environ.pop(capacity.BUDGET_ENV, None)
            self.assertEqual(capacity.budget_bytes(), 0)
        with patch.object(capacity, "total_bytes", return_value=None):
            os.environ.pop(capacity.BUDGET_ENV, None)
            self.assertIsNone(capacity.budget_bytes())

    def test_an_explicit_allowance_replaces_the_measurement(self):
        with patch.dict(os.environ, {capacity.BUDGET_ENV: "8"}):
            self.assertEqual(capacity.budget_bytes(), 8 * GIB)


class TestPlanWorkers(unittest.TestCase):
    need = capacity.Need("test", shared=4 * GIB / capacity.SAFETY, per_worker=2 * GIB / capacity.SAFETY)

    def test_as_many_workers_as_fit_up_to_the_request(self):
        self.assertEqual(capacity.plan_workers(self.need, 8, budget=21 * GIB).workers, 8)  # 4 + 2*8 = 20
        self.assertEqual(capacity.plan_workers(self.need, 8, budget=12.5 * GIB).workers, 4)  # 4 + 2*4 = 12
        self.assertEqual(capacity.plan_workers(self.need, 2, budget=100 * GIB).workers, 2)  # never above the request

    def test_a_reduced_plan_says_so(self):
        plan = capacity.plan_workers(self.need, 8, budget=12.5 * GIB)
        self.assertTrue(plan.reduced)
        self.assertIn("reduced from 8", plan.line())

    def test_not_even_one_worker_is_refused_with_the_figures(self):
        with self.assertRaises(capacity.CapacityError) as caught:
            capacity.plan_workers(self.need, 8, budget=5 * GIB)
        message = str(caught.exception)
        self.assertIn("test needs about 6.0 GiB", message)
        self.assertIn("5.0 GiB can be used now", message)

    def test_an_unmeasurable_budget_never_blocks(self):
        with patch.object(capacity, "budget_bytes", return_value=None):
            self.assertEqual(capacity.plan_workers(self.need, 5).workers, 5)


if __name__ == "__main__":
    unittest.main()
