"""The Chapter 6 joint plots: records joined to the audit, seed aggregates, the figure and CLI registration."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.plots import benchmark_tradeoffs as bt


def write_record(folder: Path, network: str, seed: int, method: str, *, f1: float | None, seconds: float,
                 status: str = "completed", unit: str = "full") -> None:
    runner = qs.BY_KEY[method].runner
    base, cover = qs.split_target(network)
    tag = base if cover is None else f"{base}-{cover}"
    network = base
    directory = folder / f"{tag}-{unit}" / f"seed{seed}" / "runs" / network
    directory.mkdir(parents=True, exist_ok=True)
    record = {"dataset": network, "seed": seed, "status": status, "detection_seconds": seconds,
              "metrics": {"f1": f1, "onmi": 0.5} if f1 is not None else {}}
    (directory / f"{runner}.json").write_text(json.dumps(record))


def write_audit(folder: Path, network: str, seed: int, method: str, log_measure: float, profile: str = "cover") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    record = {"key": ca.key_of(network, seed, method, profile), "status": "completed", "network": network, "seed": seed,
              "method": method, "profile": profile,
              "policies": {policy: {"stable_measure_log": log_measure, "stable_measure_linear": log_measure / 2}
                           for policy, _ in ca.POLICIES}}
    with (folder / ca.RESULTS_NAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class TestData(unittest.TestCase):
    def fixture(self, tmp: str) -> tuple[Path, Path]:
        bench, audit = Path(tmp) / "bench", Path(tmp) / "audit"
        for seed, (acc, sec) in enumerate([(0.40, 10.0), (0.44, 12.0), (0.42, 11.0)]):
            write_record(bench, "dblp", seed, "hedonic", f1=acc, seconds=sec)
            write_audit(audit, "dblp", seed, "hedonic", 0.7 + 0.01 * seed)
            write_audit(audit, "dblp", seed, "hedonic", 0.1, profile="labelled")  # never used for comparison
        write_record(bench, "dblp", 0, "clique_percolation", f1=0.3, seconds=5.0)
        write_record(bench, "dblp", 1, "clique_percolation", f1=None, seconds=5.0)            # no metric: skipped
        write_record(bench, "dblp", 2, "clique_percolation", f1=0.3, seconds=5.0, status="timeout")  # not completed: skipped
        write_record(bench, "amazon", 0, "hedonic", f1=0.5, seconds=3.0)
        return bench, audit

    def test_rows_join_the_cover_profile_robustness_and_skip_unfinished_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench, audit = self.fixture(tmp)
            rows, skipped = bt.load_benchmark([bench], "f1")
            self.assertEqual(skipped, 2)
            self.assertEqual(len(rows), 5)
            bt.join(rows, bt.load_robustness([audit], "log", "open_labels"))
            by_unit = {(r["network"], r["seed"], r["method"]): r for r in rows}
            self.assertAlmostEqual(by_unit[("dblp", 2, "hedonic")]["robustness"], 0.72)  # the cover profile, not 0.1
            self.assertIsNone(by_unit[("dblp", 0, "clique_percolation")]["robustness"])
            linear = bt.load_robustness([audit], "linear", "fixed_labels")
            self.assertAlmostEqual(linear[("dblp", 0, "hedonic")], 0.35)

    def test_the_covers_of_one_network_stay_apart(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench, audit = Path(tmp) / "bench", Path(tmp) / "audit"
            write_record(bench, "amazon", 0, "hedonic", f1=0.5, seconds=3.0)
            write_record(bench, "amazon:top5000", 0, "hedonic", f1=0.7, seconds=0.2, unit="full-covered")
            write_audit(audit, "amazon", 0, "hedonic", 0.6)
            write_audit(audit, "amazon:top5000", 0, "hedonic", 0.8)
            rows, _ = bt.load_benchmark([bench], "f1")
            bt.join(rows, bt.load_robustness([audit], "log", "open_labels"))
            by_target = {r["network"]: r for r in rows}
            self.assertEqual(set(by_target), {"amazon", "amazon:top5000"})
            self.assertEqual((by_target["amazon"]["accuracy"], by_target["amazon"]["robustness"]), (0.5, 0.6))
            self.assertEqual((by_target["amazon:top5000"]["accuracy"], by_target["amazon:top5000"]["robustness"]),
                             (0.7, 0.8))
            only, _ = bt.load_benchmark([bench], "f1", networks=["amazon:top5k"])
            self.assertEqual([r["network"] for r in only], ["amazon:top5000"])
            self.assertEqual([r["network"] for r in bt.load_benchmark([bench], "f1", networks=["amazon:all"])[0]],
                             ["amazon"])
            self.assertEqual([a["network"] for a in bt.aggregate(rows)], ["amazon", "amazon:top5000"])
            written = bt.plot(rows, bt.aggregate(rows), metric="f1", measure="log", policy="open_labels",
                              output=Path(tmp) / "fig", formats=("png",))
            self.assertTrue(written[0].is_file())

    def test_filters_and_the_first_folder_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench, _ = self.fixture(tmp)
            other = Path(tmp) / "other"
            write_record(other, "dblp", 0, "hedonic", f1=0.99, seconds=1.0)
            rows, _ = bt.load_benchmark([bench, other], "f1", networks=["dblp"], methods=["hedonic"])
            self.assertEqual(len(rows), 3)
            first = next(r for r in rows if r["seed"] == 0)
            self.assertAlmostEqual(first["accuracy"], 0.40)

    def test_aggregate_gives_mean_and_student_interval_over_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            bench, audit = self.fixture(tmp)
            rows, _ = bt.load_benchmark([bench], "f1")
            bt.join(rows, bt.load_robustness([audit], "log", "open_labels"))
            agg = {(a["network"], a["method"]): a for a in bt.aggregate(rows)}
            hoc = agg[("dblp", "hedonic")]
            self.assertEqual((hoc["n"], hoc["n_robustness"]), (3, 3))
            self.assertAlmostEqual(hoc["accuracy_mean"], 0.42)
            # t(0.975, 2) = 4.3027, sd = 0.02, n = 3
            self.assertAlmostEqual(hoc["accuracy_ci95"], 4.302653 * 0.02 / math.sqrt(3), places=5)
            single = agg[("amazon", "hedonic")]
            self.assertIsNone(single["accuracy_ci95"])
            self.assertIsNone(single["robustness_mean"])
            order = [a["method"] for a in bt.aggregate(rows) if a["network"] == "dblp"]
            self.assertEqual(order, ["hedonic", "clique_percolation"])  # the catalogue's order


class TestFigureAndCLI(unittest.TestCase):
    def test_smoke_writes_the_figure_and_both_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = bt.main(["--smoke", "--output-dir", tmp, "--formats", "png"])
            self.assertEqual(code, 0)
            for name in ("benchmark_tradeoffs.png", "benchmark_tradeoffs.csv", "benchmark_tradeoffs_rows.csv"):
                self.assertTrue((Path(tmp) / name).stat().st_size > 0, name)
            self.assertIn("24 units", buffer.getvalue())  # 2 networks x (5 + 5 seeds + 1 run of each deterministic method)

    def test_a_missing_benchmark_is_a_usage_error_and_an_empty_one_fails(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            self.assertEqual(bt.main(["--output-dir", tmp]), 2)
            self.assertEqual(bt.main(["--benchmark", tmp, "--output-dir", tmp]), 1)

    def test_registered_in_the_cli_with_working_help(self):
        self.assertIn("overlapping-tradeoffs", CLI.COMMANDS)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(["overlapping-tradeoffs", "--help"]), 0)


if __name__ == "__main__":
    unittest.main()
