"""The loop nesting of a benchmark run (``order``) is a setting of its config file, not code."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from pathlib import Path

from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runqueue

from hedonic.experiments.resources import CONFIG_DIR


def config_file(folder: str, body: str) -> str:
    path = Path(folder) / "run.toml"
    path.write_text("[hedonic_run]\n" + body)
    return str(path)


def load(path: str) -> qs.Config:
    return qs.config_from_args(qs.build_parser().parse_args(["--config", path]))


class TestOrderKey(unittest.TestCase):
    def test_a_config_file_sets_the_order_as_a_string_or_a_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load(config_file(tmp, 'order = "seed,method,network"\n')).order, "seed,method,network")
            self.assertEqual(load(config_file(tmp, 'order = ["method", "seed", "network"]\n')).order,
                             "method,seed,network")

    def test_without_the_key_the_default_nesting_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load(config_file(tmp, 'networks = ["dblp"]\n')).order, "network,seed,method")

    def test_a_flag_overrides_the_config_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = config_file(tmp, 'order = ["seed", "method", "network"]\n')
            cfg = qs.config_from_args(qs.build_parser().parse_args(["--config", path, "--order", "network,method,seed"]))
            self.assertEqual(cfg.order, "network,method,seed")

    def test_a_bad_order_is_refused_naming_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in ('"seed,method"', '["seed", "seed", "method"]', '"seed,method,networks"', '[]'):
                path = config_file(tmp, f"order = {bad}\n")
                with self.assertRaises(ValueError) as caught:
                    load(path)
                self.assertIn("order in", str(caught.exception))
                self.assertIn("outermost loop first", str(caught.exception))
        with self.assertRaises(ValueError):
            qs.parse_order("seed,method")

    def test_every_nesting_walks_the_same_units_with_the_outermost_loop_slowest(self):
        base = qs.load_config("paper")
        base.networks, base.methods, base.seeds = ["dblp", "amazon"], ["codeseg", "fox", "clique_percolation"], [0, 1]
        units = None
        for order in itertools.permutations(("network", "seed", "method")):
            base.order = ",".join(order)
            plan = qs.build_plan(base, {})
            seen = sorted((n, s, m.key) for n, s, m in plan)
            units = units or seen
            self.assertEqual(seen, units)  # same 12 units whatever the nesting
            axis = {"network": lambda t: t[0], "seed": lambda t: t[1], "method": lambda t: t[2].key}[order[0]]
            values = [axis(t) for t in plan]
            self.assertEqual(values, sorted(values, key=values.index))  # the outermost axis changes only 2-3 times
            self.assertEqual(len({k for k, _ in itertools.groupby(values)}), len(set(values)))
        self.assertEqual(len(units), 12)

    def test_seed_major_runs_one_seed_on_every_method_and_network_before_the_next(self):
        base = qs.load_config("paper")
        base.networks, base.methods, base.seeds, base.order = ["dblp", "amazon"], ["clique_percolation", "angel"], [0, 1], "seed,method,network"
        plan = [(n, s, m.key) for n, s, m in qs.build_plan(base, {})]
        self.assertEqual(plan, [("dblp", 0, "clique_percolation"), ("amazon", 0, "clique_percolation"), ("dblp", 0, "angel"), ("amazon", 0, "angel"),
                                ("dblp", 1, "clique_percolation"), ("amazon", 1, "clique_percolation"), ("dblp", 1, "angel"), ("amazon", 1, "angel")])


class TestShippedBenchmark(unittest.TestCase):
    def test_the_paper_benchmark_nests_seed_then_pair_then_method(self):
        cfg = load(str(CONFIG_DIR / "paper-benchmark.toml"))
        self.assertEqual(cfg.order, "seed,network,method")
        self.assertEqual(cfg.seeds, list(range(10)))
        self.assertEqual(cfg.methods[0], "hedonic")
        self.assertEqual(len(cfg.methods), 9)
        self.assertTrue(cfg.robustness and cfg.save_covers)
        self.assertEqual(len(qs.build_plan(cfg, {})), 8 * 9 * 10)

    def test_the_queue_item_reads_that_file_and_the_queue_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = runqueue.load_queue("paper", root=tmp)
        row = next(item for item in queue["items"] if item["name"] == "benchmark")
        self.assertEqual(runqueue._option(row["run"], "--config"), str(CONFIG_DIR / "paper-benchmark.toml"))  # stored absolute
        self.assertNotIn("--order", row["run"])  # the nesting is not repeated on the command line

    def test_the_config_is_found_by_its_shipped_name(self):
        self.assertEqual(qs.load_config("paper-benchmark").order, load(str(CONFIG_DIR / "paper-benchmark.toml")).order)


if __name__ == "__main__":
    unittest.main()
