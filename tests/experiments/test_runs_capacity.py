"""Steps look at the capacity available before they run, and refuse or shrink instead of exhausting the machine."""
from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

import igraph as ig

from hedonic.experiments.overlapping import capacity, parallel
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import stability as st


def random_case(seed: int, n: int = 60, labels: int = 12, extra_edges: int = 150, loops: bool = True):
    """A random graph and a random cover with contiguous labels (as tests/test_compact_state.py of the 1.x tree)."""
    rng = random.Random(seed)
    edges = {(min(a, b), max(a, b)) for a, b in ((rng.randrange(n), rng.randrange(n)) for _ in range(extra_edges)) if a != b}
    graph = ig.Graph(n=n, edges=sorted(edges))
    if loops:
        graph.add_edges([(0, 0), (3, 3)])
    memberships = []
    for _ in range(n):
        memberships.append(rng.sample(range(labels), rng.randint(1, 4)))
    used = {label for row in memberships for label in row}
    remap = {label: i for i, label in enumerate(sorted(used))}
    return graph, [[remap[label] for label in row] for row in memberships]


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


class TestSupportEstimate(unittest.TestCase):
    def test_the_sampled_estimate_matches_the_exact_count(self):
        graph, memberships = random_case(4, n=300, extra_edges=2500)
        state = rb.build_fractional_state(graph, memberships)
        exact = state.support.entries
        self.assertAlmostEqual(capacity.sample_support_entries(graph, memberships, sample=10_000), exact, delta=1)  # whole graph
        estimate = capacity.sample_support_entries(graph, memberships, sample=100, seed=1)
        self.assertLess(abs(estimate - exact) / exact, 0.25)

    def test_the_priced_state_is_close_to_the_real_arrays(self):
        graph, memberships = random_case(9, n=500, extra_edges=4000)
        state = rb.build_fractional_state(graph, memberships)
        priced = capacity.state_bytes(graph.vcount(), graph.ecount(), sum(map(len, memberships)), state.support.entries,
                                      state.n_communities)
        self.assertGreaterEqual(priced, state.support.nbytes)


class TestChunkedAudit(unittest.TestCase):
    def rows(self, engine, block):
        graph, memberships = random_case(7, n=120, labels=14, extra_edges=500)
        state = rb.build_fractional_state(graph, memberships)
        cap = max(len(row) for row in memberships)
        gammas = [0.0, 0.01, 0.1, 0.5, 1.0]
        for allow in (False, True):
            whole = st.vector_spectrum_rows(st.build_entries(state, cap, allow), gammas, atol=1e-9, rtol=1e-9,
                                            allow_isolation=allow, engine=engine)
            chunked = st.chunked_spectrum_rows(state, cap, gammas, atol=1e-9, rtol=1e-9, allow_isolation=allow,
                                               engine=engine, block=block)
            yield whole, chunked

    def test_chunked_rows_equal_the_whole_ones(self):
        engines = ["numpy"] + (["numba"] if st.HAVE_NUMBA else [])
        for engine in engines:
            for block in (1, 7, 1000):
                for whole, chunked in self.rows(engine, block):
                    for a, b in zip(whole, chunked):
                        for key in a:
                            if isinstance(a[key], float):
                                self.assertAlmostEqual(a[key], b[key], places=9, msg=f"{engine} {block} {key}")
                            else:
                                self.assertEqual(a[key], b[key], f"{engine} {block} {key}")


class TestPolicyRefusals(unittest.TestCase):
    def test_exact_audit_workers_shrinks_then_refuses(self):
        graph, memberships = random_case(3, n=200, extra_edges=1500)
        with patch.object(capacity, "budget_bytes", return_value=None):
            self.assertEqual(capacity.exact_audit_workers(graph, memberships, 6, say=None), 6)
        with patch.object(capacity, "budget_bytes", return_value=1):  # one byte: nothing fits
            with self.assertRaises(capacity.CapacityError):
                capacity.exact_audit_workers(graph, memberships, 6, say=None)


if __name__ == "__main__":
    unittest.main()
