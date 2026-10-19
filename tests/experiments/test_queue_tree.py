"""The interactive queue view: the tree model (counts, formulas, ETAs), the run builders, the input and drawing of the
viewer, and a pty run of the real loop."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import queuebuild as qb
from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import queuetui as qt
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import tui

ANSI = re.compile(r"\033\[[0-9;?<]*[A-Za-z]")
G = {**tui.glyphs(True), "open": "▾", "closed": "▸"}


def _family(done_pairs=(), cost=None) -> qm.Family:
    """2 pairs x 2 policies x 3 resolutions x 2 seeds = 24 units; the units of ``done_pairs`` are done."""
    dims = [qm.Dim("pair", "pairs"), qm.Dim("action policy", "action policies", meaning={"fixed": "stays"}),
            qm.Dim("resolution", "resolutions", "γ {}"), qm.Dim("seed", "seeds", "seed {}")]
    units = [qm.Unit((pair, policy, gamma, seed), pair in done_pairs, 1.5 if pair in done_pairs else None)
             for pair in ("A", "B") for policy in ("fixed", "open") for gamma in ("0", "0.1", "1") for seed in ("0", "1")]
    return qm.Family("detector", "detector runs", "one seeded run", dims, units, "pair > policy > resolution > seed", cost)


class TestModel(unittest.TestCase):
    def test_a_count_is_the_sum_of_the_rows_under_it_and_its_formula_is_written_out(self):
        nodes = qm.family_nodes(_family(done_pairs=("A",)), _family(done_pairs=("A",)).units)
        by = {n.label: n for n in nodes}
        self.assertEqual((by["A"].done, by["A"].total), (12, 12))
        self.assertEqual((by["B"].done, by["B"].total), (0, 12))
        policy = by["B"].children[0]
        self.assertEqual(policy.total, 6)
        self.assertEqual(sum(c.total for c in policy.children), 6)
        text = "\n".join(by["B"].detail)
        self.assertIn("12 detector runs = 2 action policies x 3 resolutions x 2 seeds", text)
        self.assertIn("runner order, outermost first: pair > policy > resolution > seed", text)
        resolution = policy.children[0]
        self.assertEqual(resolution.label, "γ 0")
        self.assertIn("2 detector runs = 2 seeds", "\n".join(resolution.detail))
        self.assertEqual(resolution.strip, [False, False])  # its children are units: one mark each
        self.assertIn("stays", "\n".join(policy.detail))  # a value's meaning is shown on its own row

    def test_a_plan_that_is_not_a_full_grid_says_so_instead_of_inventing_a_product(self):
        family = _family()
        family.units = family.units[:-1]
        lines = qm.explain(family, 0, family.units)
        self.assertIn("23 detector runs over 2 pairs x 2 action policies x 3 resolutions x 2 seeds", lines[0])
        self.assertIn("not every combination", lines[0])

    def test_etas_add_up_and_a_unit_without_an_estimate_makes_the_total_a_lower_bound(self):
        def cost(unit):
            return (2.0, "measured") if unit.coords[0] == "A" else (None, "")
        nodes = qm.family_nodes(_family(cost=cost), _family().units)
        a, b = nodes
        self.assertEqual((a.left, a.basis, a.unknown), (24.0, "measured", 0))
        self.assertEqual(qm.eta_text(a), "0:24")
        self.assertEqual((b.left, b.unknown), (0.0, 12))
        self.assertEqual(qm.eta_text(b), "?")
        run = qm.run_node(1, "r", "", "running", qm.RunModel([_family(cost=cost)]), None)
        self.assertEqual((run.left, run.unknown), (24.0, 12))
        self.assertEqual(qm.eta_text(run), "≥0:24")
        root = qm.queue_node("q", [run])
        self.assertEqual(qm.eta_text(root), "≥0:24")

    def test_estimates_and_bounds_are_marked_and_a_finished_node_says_done(self):
        node = qm.Node(("x",), "x", total=4, done=1, left=90.0, basis="estimate")
        self.assertEqual(qm.eta_text(node), "~1:30")
        node.basis = "bound"
        self.assertEqual(qm.eta_text(node), "≥1:30")
        node.done = 4
        self.assertEqual(qm.eta_text(node), "done")
        self.assertEqual(qm.fmt_seconds(3 * 86400 + 7200), "3d 02h")
        self.assertEqual(qm.fmt_seconds(3725), "1:02:05")

    def test_two_kinds_of_unit_under_one_pair_are_two_children_and_the_pair_sums_them(self):
        audit = qm.Family("audit", "audit steps", "a check", [qm.Dim("pair", "pairs"), qm.Dim("resolution", "resolutions")],
                          [qm.Unit(("A", str(i)), i < 2) for i in range(5)] + [qm.Unit(("B", str(i)), False) for i in range(5)])
        run = qm.run_node(1, "r", "t", "running", qm.RunModel([audit, _family()]), None)
        pair = run.children[0]
        self.assertEqual([c.label for c in pair.children], ["audit steps", "detector runs"])
        self.assertEqual((pair.done, pair.total), (2, 5 + 12))
        self.assertEqual((run.done, run.total), (2, 34))
        self.assertIn("5 audit steps + 12 detector runs = 17 steps", pair.detail[0])
        self.assertEqual(run.noun, "steps")

    def test_a_run_without_a_model_uses_its_own_counts_and_eta(self):
        run = qm.run_node(2, "r", "", "running", None, (3, 10, "searches"), eta=(120.0, "measured"))
        self.assertEqual((run.done, run.total, run.left), (3, 10, 120.0))
        lost = qm.run_node(2, "r", "", "pending", None, (0, 10, "searches"))
        self.assertEqual(lost.unknown, 1)
        self.assertEqual(qm.eta_text(lost), "?")
        done = qm.run_node(2, "r", "", "done", None, (10, 10, "searches"))
        self.assertEqual(done.status, "done")


class TestBuilders(unittest.TestCase):
    ROWS = [{"job": job, "dataset": job.split("-")[0], "cover": job.split("-")[1], "policy": policy, "gamma": gamma,
             "seed": seed, "status": status, "seconds": 8.0 if status == "completed" else None, "f1": 0.5}
            for job, status in (("amazon-all", "completed"), ("dblp-all", "pending"))
            for policy in ("fixed_labels", "open_labels") for gamma in (0.0, 0.1, 1.0) for seed in (0, 1)]
    OPTIONS = {"isolation_policies": ["fixed_labels", "open_labels"], "audit_resolutions": [0.0, 0.5, 1.0, 2.0], "workers": 8}

    def _spectrum(self, stage="detector", audit_done=None, audit_seconds=None, log=""):
        folder = Path(tempfile.mkdtemp())
        (folder / "worker.log").write_text(log)
        return qb.spectrum(self.ROWS, folder=folder, live={"stage": stage}, options=self.OPTIONS,
                           audit_done=audit_done or {"amazon-all": 8, "dblp-all": 0},
                           audit_seconds=audit_seconds if audit_seconds is not None else {"amazon-all": 400.0},
                           unit_finished=lambda r: r.get("status") == "completed")

    def test_the_spectrum_run_counts_audit_steps_and_detector_runs_from_the_loops(self):
        model = self._spectrum()
        audit, detector = model.families
        self.assertEqual((len(audit.units), len(detector.units)), (16, 24))  # 2 pairs x 2 policies x 4 gammas / x 3 x 2 seeds
        run = qm.run_node(1, "spec", "", "running", model, None)
        self.assertEqual((run.done, run.total), (8 + 12, 40))
        text = "\n".join(run.detail)
        self.assertIn("16 audit steps = 2 pairs x 2 action policies x 4 resolutions", text)
        self.assertIn("24 detector runs = 2 pairs x 2 action policies x 3 resolutions x 2 seeds", text)
        policy = qm.family_nodes(detector, detector.units)[0].children[0]
        self.assertIn("action policy: fixed labels", policy.detail[0])
        self.assertIn("a vertex may only join a community that already exists", "\n".join(policy.detail))

    def test_detector_runs_cost_what_the_pair_measured_divided_by_its_workers_and_others_are_scaled(self):
        log = "[capacity] amazon-all: detector runs: 4 worker(s) · needs 1 GiB\n[capacity] dblp-all: detector runs: 2 worker(s)\n"
        model = self._spectrum(log=log)
        cost = model.families[1].cost
        self.assertEqual(cost(qm.Unit(("Amazon · all", "fixed labels", "0", "0"), False)), (2.0, "measured"))
        seconds, basis = cost(qm.Unit(("DBLP · all", "fixed labels", "0", "0"), False))
        self.assertEqual(basis, "estimate")  # scaled by edges from Amazon, over DBLP's 2 workers
        self.assertAlmostEqual(seconds, 8.0 * 1_049_866 / 925_872 / 2, places=6)

    def test_the_running_audit_costs_its_ETA_spread_over_its_steps_and_the_next_one_is_scaled(self):
        stage = "auditing dblp/all · fixed_labels · γ=0.5  4/8 resolutions · ETA ~1:00:00"
        model = self._spectrum(stage=stage, audit_done={"amazon-all": 8, "dblp-all": 4})
        cost = model.families[0].cost
        self.assertEqual(cost(qm.Unit(("DBLP · all", "open labels", "0"), False)), (900.0, "measured"))  # 3600 s over 4 steps left
        none = self._spectrum(audit_seconds={})  # nothing audited yet and nothing running: no basis
        self.assertEqual(none.families[0].cost(qm.Unit(("DBLP · all", "open labels", "0"), False)), (None, ""))

    def test_the_stage_line_eta_is_read_back_to_seconds(self):
        self.assertEqual(qb.stage_eta("x ETA ~17:33:22"), 17 * 3600 + 33 * 60 + 22)
        self.assertEqual(qb.stage_eta("x ETA ~5:07"), 5 * 60 + 7)
        self.assertIsNone(qb.stage_eta("detector"))

    def test_the_startcurve_run_costs_come_from_records_of_the_pair_even_at_earlier_settings(self):
        plan = {"workers": 4, "settings": {"dblp": {"cap": 26, "gamma": 0.15}}}
        keys = [{"network": "dblp", "level": level, "seed": seed, "key": f"k-{level}-{seed}"}
                for level in (0.0, "random") for seed in (0, 1)]
        records = {"k-0.0-0": {"network": "dblp", "level": "0", "seconds": 8.0, "workers": 4},
                   "old": {"network": "dblp", "level": "random", "seconds": 40.0, "workers": 4}}
        model = qb.startcurve(plan, records, keys)
        family = model.families[0]
        self.assertEqual([u.done for u in family.units], [True, False, False, False])
        self.assertEqual(family.cost(family.units[1]), (2.0, "measured"))  # level 0 measured in this run: 8 s / 4
        self.assertEqual(family.cost(family.units[2]), (10.0, "estimate"))  # random: only an earlier setting's record
        self.assertIn("cap 26", "\n".join(model.notes))

    def test_a_record_is_filed_under_its_pair_with_the_cover_it_was_scored_against(self):
        self.assertEqual(qb.record_target({"network": "livejournal", "cover": "all"}), "livejournal")
        self.assertEqual(qb.record_target({"network": "livejournal"}), "livejournal")  # written before covers existed
        self.assertEqual(qb.record_target({"network": "livejournal", "cover": "top5000"}), "livejournal:top5000")
        self.assertEqual(qb.record_target({"network": "livejournal:top5000", "cover": "top5000"}), "livejournal:top5000")

    def test_the_cheap_top5000_records_do_not_price_the_all_cover_of_the_same_network(self):
        """LiveJournal/all conditions took 83 s each and LiveJournal/top5000 ones 1.4 s. Filed by network alone, the
        unfinished corruption levels of /all were priced from the /top5000 records (0.17 s a unit, 9 s for the pair)."""
        plan = {"workers": 8, "settings": {"livejournal": {"cap": 20, "gamma": 0.1}, "livejournal:top5000": {"cap": 9, "gamma": 0.2}}}
        keys = [{"network": net, "level": level, "seed": seed, "key": f"{net}-{level}-{seed}"}
                for net in ("livejournal:top5000", "livejournal") for level in ("0", "0.1") for seed in (0, 1)]
        records = {f"livejournal:top5000-{level}-{seed}": {"network": "livejournal", "cover": "top5000", "level": level,
                                                          "seconds": 1.4, "workers": 8}
                   for level in ("0", "0.1") for seed in (0, 1)}
        records["livejournal-0-0"] = {"network": "livejournal", "cover": "all", "level": "0", "seconds": 83.0, "workers": 8}
        family = qb.startcurve(plan, records, keys).families[0]
        unit = lambda pair, level, seed: next(u for u in family.units if u.coords == (pair, level, seed))  # noqa: E731
        seconds, basis = family.cost(unit("LiveJournal · all", "0", "1"))
        self.assertEqual((round(seconds, 3), basis), (round(83.0 / 8, 3), "measured"))
        seconds, basis = family.cost(unit("LiveJournal · all", "0.1", "0"))  # no /all record at this level: the pair's mean
        self.assertEqual((round(seconds, 3), basis), (round(83.0 / 8, 3), "estimate"))
        self.assertEqual(family.cost(unit("LiveJournal · top5k", "0.1", "0"))[1], "measured")  # its own cheap records

    def test_the_notes_on_earlier_settings_name_the_pair_with_its_cover(self):
        record = {"network": "youtube", "cover": "top5000", "cap": 9, "gamma": 0.2}
        self.assertEqual(runqueue._startcurve_earlier({"settings": {"youtube:top5000": {"cap": 7, "gamma": 0.3}}},
                                                      {"old": record}, set()),
                         ["YouTube · top5k: 1 conditions ran at cap 9, gamma 0.2 (now cap 7, gamma 0.3); "
                          "other settings and code (1.x); kept, not reused"])
        # The same settings on other code: the note says the code, not the settings, is why.
        self.assertEqual(runqueue._startcurve_earlier({"settings": {"youtube:top5000": {"cap": 9, "gamma": 0.2}}},
                                                      {"old": record}, set()),
                         ["YouTube · top5k: 1 conditions ran at cap 9, gamma 0.2; same settings, other code (1.x); "
                          "kept, not reused"])

    # ---- the benchmark: a unit's time includes its audit, and a unit past its estimate has no ETA
    NOW = 1_000_000.0

    def _benchmark(self, plan, finished_labels=(), seeds=(0, 1, 2), history=None):
        from types import SimpleNamespace

        methods = {label: SimpleNamespace(label=label, key=label.lower()) for label in ("Hedonic", "BigCLAM")}
        units = [("youtube:top5000", seed, methods[label]) for label in methods for seed in seeds]
        finished = [(label, seed) in finished_labels for label in methods for seed in seeds]
        cfg = SimpleNamespace(robustness=True, order="seed,network,method")
        with patch.object(qb.qs, "expected_seconds", lambda m, c, n: {"Hedonic": 100.0, "BigCLAM": 1658.0}[m.label]):
            return qb.benchmark(cfg, units, finished, {"plan": plan}, now=self.NOW, history=history).families[0]

    def _row(self, label, seed, status, **fields):
        return {"network": "youtube:top5000", "label": label, "seed": seed, "status": status, **fields}

    def _cost(self, family, label, seed):
        unit = next(u for u in family.units if u.coords == ("YouTube · top5k", label, str(seed)))
        return family.cost(unit), unit

    def test_a_cell_with_finished_units_is_priced_by_their_whole_time_audit_included(self):
        plan = [self._row("Hedonic", 0, "completed", seconds=83.0, detection_seconds=2.2)]
        family = self._benchmark(plan, {("Hedonic", 0)})
        self.assertEqual(self._cost(family, "Hedonic", 1)[0], (83.0, "measured"))  # not the 2.2 s of detection
        self.assertEqual(self._cost(family, "BigCLAM", 0)[0], (1658.0, "bound"))  # untimed here: the constant is detection alone

    def test_a_running_unit_has_its_estimate_less_the_time_it_has_run(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 600.0)]
        self.assertEqual(self._cost(self._benchmark(plan), "BigCLAM", 0)[0], (1058.0, "bound"))

    def test_a_unit_that_has_run_past_its_estimate_has_no_eta_and_says_why(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 18_000.0, detection_seconds=993.0)]
        family = self._benchmark(plan)
        (seconds, basis), unit = self._cost(family, "BigCLAM", 0)
        self.assertEqual((seconds, basis), (None, ""))  # it used to say 27:38 left after five hours
        self.assertTrue(unit.running)
        self.assertIn("detection took 16:33", unit.note)
        self.assertIn("exact audit of its saved cover has run", unit.note)
        self.assertIn("unknown", unit.note)
        # the unit shows ?, and everything above it >= (part of the work has no estimate)
        node = next(n for n in qm.family_nodes(family, family.units) if n.label == "YouTube · top5k")
        self.assertGreaterEqual(node.unknown, 1)
        self.assertTrue(qm.eta_text(node).startswith("≥"))

    def test_the_other_seeds_of_an_overrun_cell_are_priced_at_least_as_long_as_the_first_took(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 18_000.0, detection_seconds=993.0)]
        family = self._benchmark(plan)
        self.assertEqual(self._cost(family, "BigCLAM", 1)[0], (18_000.0, "bound"))
        self.assertEqual(self._cost(family, "Hedonic", 1)[0], (100.0, "bound"))  # another method is not affected

    def test_a_unit_inside_its_estimate_does_not_change_its_siblings(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 600.0)]
        self.assertEqual(self._cost(self._benchmark(plan), "BigCLAM", 1)[0], (1658.0, "bound"))

    def _audit(self, eta, age=10.0, **extra):
        return {"policy": "fixed_labels", "number": 1, "policies": 2, "done": 5000, "total": 39841, "elapsed": 600.0,
                "eta_seconds": eta, "updated": self.NOW - age, **extra}

    def test_a_running_unit_whose_audit_reports_progress_has_the_time_that_progress_implies(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 18_000.0, detection_seconds=993.0, audit=self._audit(7200.0))]
        family = self._benchmark(plan)
        (seconds, basis), unit = self._cost(family, "BigCLAM", 0)
        self.assertEqual((round(seconds), basis), (7190, "estimate"))  # the report's 7200 s, ten seconds old
        self.assertIn("audited 5,000 of 39,841 vertices under policy 1 of 2", unit.note)
        self.assertNotIn("unknown", unit.note)
        node = next(n for n in qm.family_nodes(family, family.units) if n.label == "YouTube · top5k")
        self.assertEqual(node.unknown, 0)  # a number, not a ?

    def test_the_other_seeds_of_that_cell_are_priced_at_its_whole_length(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 18_000.0, detection_seconds=993.0, audit=self._audit(7200.0))]
        family = self._benchmark(plan)
        self.assertEqual(self._cost(family, "BigCLAM", 1)[0], (18_000.0 + 7190.0, "estimate"))
        self.assertEqual(self._cost(family, "Hedonic", 1)[0], (100.0, "bound"))

    def test_progress_inside_the_estimate_changes_no_sibling(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 600.0, audit=self._audit(300.0))]
        family = self._benchmark(plan)
        self.assertEqual(round(self._cost(family, "BigCLAM", 0)[0][0]), 290)
        self.assertEqual(self._cost(family, "BigCLAM", 1)[0], (1658.0, "bound"))

    def test_a_stale_report_is_not_trusted(self):
        plan = [self._row("BigCLAM", 0, "running", started=self.NOW - 18_000.0, detection_seconds=993.0,
                          audit=self._audit(7200.0, age=qb.AUDIT_FRESH_SECONDS + 1))]
        (seconds, basis), unit = self._cost(self._benchmark(plan), "BigCLAM", 0)
        self.assertEqual((seconds, basis), (None, ""))  # the audit or its worker is gone: back to "no ETA"
        self.assertIn("unknown", unit.note)

    # ---- a finished unit took its whole span, and a running one has every audit profile still to come
    def test_a_finished_unit_took_its_audits_too_not_its_detection_alone(self):
        plan = [self._row("Hedonic", 0, "completed", seconds=6600.0, detection_seconds=2.9)]
        family = self._benchmark(plan, {("Hedonic", 0), ("Hedonic", 1)},
                                 history={"units": {("youtube:top5000", 1, "hedonic"): 6500.0}})
        took = {u.coords[2]: u.took for u in family.units if u.coords[1] == "Hedonic"}
        self.assertEqual(took["0"], 6600.0)  # the tree said "done 2.9s" for a unit of 1h50
        self.assertEqual(took["1"], 6500.0)  # skipped by the scan (no row seconds): from its records on disk

    def _hedonic_audit(self, profile, eta=1000.0):
        return self._audit(eta, key=f"youtube:top5000/seed1/hedonic/{profile}")

    def test_a_running_unit_in_its_cover_audit_also_has_its_labelled_audit_to_go(self):
        plan = [self._row("Hedonic", 1, "running", started=self.NOW - 3000.0, detection_seconds=2.9,
                          audit=self._hedonic_audit("cover"))]
        history = {"audits": {("youtube:top5000", "hedonic", "labelled"): [3700.0, 3900.0]}}
        left = self._cost(self._benchmark(plan, history=history), "Hedonic", 1)[0][0]
        self.assertEqual(round(left), 990 + 3800)  # the report's rest of the cover audit, then a labelled audit
        # no finished labelled audit to go by: priced as long as the profile under way (2 policies of (600 + 1000) / 2)
        self.assertEqual(round(self._cost(self._benchmark(plan), "Hedonic", 1)[0][0]), 990 + 1600)

    def test_a_running_unit_in_its_last_profile_has_only_what_its_audit_reports(self):
        plan = [self._row("Hedonic", 1, "running", started=self.NOW - 3000.0, detection_seconds=2.9,
                          audit=self._hedonic_audit("labelled"))]
        history = {"audits": {("youtube:top5000", "hedonic", "labelled"): [3800.0]}}
        self.assertEqual(round(self._cost(self._benchmark(plan, history=history), "Hedonic", 1)[0][0]), 990)

    def test_a_report_without_its_profile_adds_no_profile(self):
        plan = [self._row("Hedonic", 1, "running", started=self.NOW - 3000.0, detection_seconds=2.9, audit=self._audit(1000.0))]
        history = {"audits": {("youtube:top5000", "hedonic", "labelled"): [3800.0]}}  # a worker from before the key
        self.assertEqual(round(self._cost(self._benchmark(plan, history=history), "Hedonic", 1)[0][0]), 990)

    def test_the_history_counts_only_units_whose_audits_are_all_recorded(self):
        from types import SimpleNamespace

        method = SimpleNamespace(label="Hedonic", key="hedonic")
        units = [("youtube:all", 0, method), ("youtube:all", 3, method)]
        audited = {"youtube:all/seed0/hedonic/cover": {"audit_seconds": 2900.0},
                   "youtube:all/seed0/hedonic/labelled": {"audit_seconds": 3785.0},
                   "youtube:all/seed3/hedonic/cover": {"audit_seconds": 2950.0}}  # seed 3 is in its labelled audit
        record = {"unit_seconds": 11.6, "runtime_seconds": 2.9, "dataset_report": {"m": 318432}}
        with patch.object(runqueue.qs, "unit_record", lambda cfg, net, seed, m: record):
            history = runqueue._benchmark_history(SimpleNamespace(induce=None), units, [True, True], audited)
        self.assertEqual(history["seconds"], {("youtube:all", "hedonic"): [11.6 + 2900.0 + 3785.0]})
        self.assertEqual(history["units"], {("youtube:all", 0, "hedonic"): 11.6 + 2900.0 + 3785.0})
        self.assertEqual(history["audits"][("youtube:all", "hedonic", "cover")], [2900.0, 2950.0])

    def test_controlled_instances_follow_the_loops_of_the_grid(self):
        grid = {"graph-seeds": ["0", "1"], "mus": ["0.1", "0.2"], "overlap-fractions": ["0.05", "0.1", "0.15"]}
        model = qb.controlled({"instances": 12, "groups": 4, "workers": 2}, 5, grid)
        units = model.families[0].units
        self.assertEqual(len(units), 12)
        self.assertEqual(sum(u.done for u in units), 5)
        self.assertEqual(units[0].coords, ("0", "0.1", "0.05"))


def _tree() -> qm.Node:
    """paper-like: a done run, a running run with a running pair, a pending run."""
    running = qm.run_node(2, "spec", "", "running", qm.RunModel([_family(done_pairs=("A",), cost=lambda u: (1.0, "measured"))]), None)
    running.children[1].status = "running"  # pair B is the one being worked on
    done = qm.run_node(1, "calib", "", "done", None, (8, 8, "searches"))
    pending = qm.run_node(3, "bench", "", "pending", None, (0, 20, "units"))
    return qm.queue_node("q", [done, running, pending], "running")


class TestViewer(unittest.TestCase):
    def test_input_is_parsed_into_keys_wheel_and_clicks(self):
        events, rest = qt.parse("j\x1b[A\x1b[B\x1b[5~\x1b[<64;10;5M\x1b[<65;10;5M\x1b[<0;12;7M\x1b[<0;12;7m \r")
        self.assertEqual(events, [("key", "j"), ("key", "up"), ("key", "down"), ("key", "pgup"), ("wheel", -1, 10, 5),
                                  ("wheel", 1, 10, 5), ("click", 12, 7), ("key", "space"), ("key", "enter")])
        self.assertEqual(rest, "")

    def test_an_unfinished_escape_sequence_waits_for_the_rest(self):
        events, rest = qt.parse("j\x1b[<65;1")
        self.assertEqual((events, rest), ([("key", "j")], "\x1b[<65;1"))
        events, rest = qt.parse(rest + "0;5M")
        self.assertEqual(events, [("wheel", 1, 10, 5)])

    def test_only_the_live_path_is_open_at_first_and_a_later_node_is_not_opened_for_you(self):
        root, view = _tree(), qt.View()
        rows = qt.flatten(root, view)
        path = qt.live_path(root)
        self.assertEqual(view.expanded, {node.key for node in path})
        self.assertEqual([node.depth for node in path], [0, 1, 2, 3, 4, 5])  # queue > run > pair > policy > resolution > seed
        self.assertEqual([r.depth for r in rows][:5], [0, 1, 1, 2, 2])
        view.expanded.clear()
        view.expanded.add(root.key)
        self.assertEqual(len(qt.flatten(root, view)), 4)  # the user closed the run; it stays closed on the next refresh

    def test_keys_open_close_and_walk_the_tree(self):
        root, view = _tree(), qt.View()
        qt.flatten(root, view)
        qt.render(root, view, (100, 30), G)
        run2 = root.children[1]

        def press(key):
            self.assertFalse(qt.apply(view, root, ("key", key)))
            qt.render(root, view, (100, 30), G)
        press("2")  # jump to run 2
        self.assertEqual(view.cursor, run2.key)
        press("left")
        self.assertNotIn(run2.key, view.expanded)
        press("left")  # a closed row: go to its parent
        self.assertEqual(view.cursor, root.key)
        press("down")
        press("down")
        self.assertEqual(view.cursor, run2.key)
        press("right")
        self.assertIn(run2.key, view.expanded)
        press("right")  # already open: step into it
        self.assertEqual(view.cursor, run2.children[0].key)
        press("e")
        self.assertIn(run2.children[0].children[0].key, view.expanded)
        press("c")
        self.assertNotIn(run2.children[0].key, view.expanded)
        press("E")
        self.assertTrue(all(n.key in view.expanded for n in qt.descendants(root) if n.children))
        press("C")
        self.assertEqual(view.expanded, {root.key})
        press("G")
        self.assertEqual(view.cursor, view.rows[-1].key)
        press("g")
        self.assertEqual(view.cursor, root.key)
        self.assertTrue(qt.apply(view, root, ("key", "q")))

    def test_the_wheel_scrolls_the_list_without_moving_the_selection_and_stays_in_range(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "E"))
        qt.render(root, view, (100, 14), G)
        cursor = view.cursor
        qt.apply(view, root, ("wheel", 1, 5, 5))
        qt.render(root, view, (100, 14), G)
        self.assertEqual((view.offset, view.cursor), (qt.WHEEL_STEP, cursor))
        for _ in range(100):
            qt.apply(view, root, ("wheel", 1, 5, 5))
        qt.render(root, view, (100, 14), G)
        self.assertEqual(view.offset, len(view.rows) - view.body[1])
        qt.apply(view, root, ("wheel", -1, 5, 5))
        self.assertEqual(view.offset, len(view.rows) - view.body[1] - qt.WHEEL_STEP)

    def test_a_click_selects_a_row_and_a_click_on_its_caret_or_a_second_click_opens_it(self):
        root, view = _tree(), qt.View()
        qt.render(root, view, (100, 30), G)
        top = view.body[0]
        run2 = root.children[1]
        row = top + qt.index_of(view.rows, run2.key) + 1  # the 1-based screen row
        caret_x = 3 + 2 * run2.depth + 1  # 1-based column of its ▾
        self.assertIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", 60, row), now=100.0)  # a first click only selects
        self.assertEqual(view.cursor, run2.key)
        self.assertIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", caret_x, row), now=200.0)  # on the caret: closes
        self.assertNotIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", 60, row), now=300.0)  # the row is selected: another click opens
        self.assertIn(run2.key, view.expanded)

    def test_every_drawn_line_fits_the_terminal_and_the_selection_is_marked(self):
        root = _tree()
        for width, height in ((100, 30), (70, 18), (120, 40)):
            view = qt.View()
            lines = qt.render(root, view, (width, height), G, qt.header_line(root, "running", 65.0, G))
            self.assertLessEqual(len(lines), height)
            for line in lines:
                self.assertLessEqual(len(ANSI.sub("", line)), width, line)
            self.assertTrue(any(line.lstrip().startswith("›") for line in lines))
        text = "\n".join(ANSI.sub("", line) for line in qt.render(root, qt.View(), (110, 30), G, qt.header_line(root, "running", 65.0, G)))
        self.assertIn("ETA", text.splitlines()[0])
        self.assertIn("1/3 runs", text.splitlines()[0])

    def test_the_detail_pane_explains_the_selected_row_and_where_its_eta_comes_from(self):
        root, view = _tree(), qt.View()
        qt.flatten(root, view)
        view.cursor = root.children[1].key
        screen = "\n".join(ANSI.sub("", line) for line in qt.render(root, view, (110, 40), G))
        self.assertIn("24 detector runs = 2 pairs x 2 action policies x 3 resolutions x 2 seeds", screen)
        self.assertIn("from this run's own timings", screen)
        view.cursor = root.key
        self.assertIn("finishes about", "\n".join(qt.detail_lines(root, 100, time.time())))

    def test_help_replaces_the_tree_until_a_key_is_pressed(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "?"))
        self.assertTrue(view.help)
        self.assertIn("mouse wheel", "\n".join(qt.render(root, view, (110, 30), G)))
        qt.apply(view, root, ("key", "x"))
        self.assertFalse(view.help)

    def test_the_cursor_moves_to_the_nearest_open_ancestor_when_its_row_disappears(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "E"))
        qt.render(root, view, (100, 30), G)
        deep = qt.descendants(root.children[1])[5]
        view.cursor = deep.key
        view.expanded.discard(root.children[1].key)  # close the run it is in
        qt.flatten(root, view)
        self.assertEqual(view.cursor, root.children[1].key)


class TestRunqueueTree(unittest.TestCase):
    def test_the_tree_of_a_queue_takes_each_run_from_its_files_and_keeps_the_report_numbers(self):
        folder = "/nowhere/spec"
        model = qm.RunModel([_family(done_pairs=("A",), cost=lambda u: (1.0, "measured"))])
        items = [{"name": "calib", "run": ["calibrate"]}, {"name": "spec", "run": ["spectrum", "--name", "spec"], "title": "T"}]
        entry = {"name": "q", "items": items, "file": ""}
        states = {"calib": ("completed", None), "spec": ("running", None)}
        data = {"calibrate": ("complete", "/nowhere/cal"), "spectrum": ("partial", folder)}
        counts = {"calibrate": (8, 8, "searches"), "spectrum": (12, 24, "conditions")}
        with patch.object(runqueue, "_status_of", lambda item, fresh=False: states[item["name"]]), \
                patch.object(runqueue, "data_state", lambda run, fresh=False: data[run[0]]), \
                patch.object(runqueue, "unit_counts", lambda run: counts[run[0]]), \
                patch.dict(runqueue._MODEL, {folder: model}), patch.object(runqueue.rm, "effective_status", return_value="running"):
            root = runqueue.tree(entry, {})
        calib, spec = root.children
        self.assertEqual((calib.status, calib.done, calib.total), ("done", 8, 8))
        self.assertEqual((spec.status, spec.done, spec.total, spec.noun), ("running", 12, 24, "detector runs"))
        self.assertEqual(spec.left, 12.0)
        self.assertEqual((root.done, root.total, root.left), (1, 2, 12.0))
        self.assertIn("whole run: 24 steps", spec.detail[0])  # what the row's number is made of comes first
        self.assertIn("T", spec.detail)  # then the item's title

    def _by_pair_entry(self):
        items = [{"name": "a", "run": ["spectrum", "--name", "a"], "by_pair": True, "pairs": ["dblp:top5000", "amazon:top5000"]},
                 {"name": "b", "run": ["calibrate"]}]
        return {"name": "q", "items": items, "file": "", "pairs": ["dblp:top5000", "amazon:top5000"]}

    def test_a_run_between_its_pairs_is_part_done_not_stopped_and_the_queue_names_its_order(self):
        entry = self._by_pair_entry()
        states = {"a": ("stopped", None), "b": ("new", None)}  # a: files say part done, no run of it going
        with patch.object(runqueue, "_status_of", lambda item, fresh=False: states[item["name"]]), \
                patch.object(runqueue, "data_state", lambda run, fresh=False: ("partial", "/nowhere/a")), \
                patch.object(runqueue, "unit_counts", lambda run: (3, 9, "conditions")), \
                patch.object(runqueue.rm, "effective_status", return_value="running"):
            root = runqueue.tree(entry, {"current": 0, "pair": "amazon:top5000", "step": 3, "steps": [{}] * 26})
        a, b = root.children
        self.assertEqual((a.status, a.done, a.total), ("partial", 3, 9))
        self.assertIn("dblp:top5k > amazon:top5k", "\n".join(root.detail))
        self.assertTrue(any("pair by pair" in line and "2 pairs" in line for line in a.detail))
        self.assertFalse(any("now on" in line for line in a.detail))  # it is not the run being worked on
        self.assertFalse(any("pair by pair" in line for line in b.detail))

    def test_the_run_being_worked_on_says_which_pair_and_which_step(self):
        entry = self._by_pair_entry()
        states = {"a": ("running", None), "b": ("new", None)}
        with patch.object(runqueue, "_status_of", lambda item, fresh=False: states[item["name"]]), \
                patch.object(runqueue, "data_state", lambda run, fresh=False: ("partial", "/nowhere/a")), \
                patch.object(runqueue, "unit_counts", lambda run: (3, 9, "conditions")), \
                patch.object(runqueue.rm, "effective_status", return_value="running"):
            root = runqueue.tree(entry, {"current": 0, "pair": "amazon:top5000", "step": 3, "steps": [{}] * 26})
        self.assertTrue(any("now on amazon:top5000 (step 4 of 26)" in line for line in root.children[0].detail))

    def test_runs_sit_under_the_method_they_study_in_the_order_the_methods_first_appear(self):
        items = [{"name": "cal", "run": ["calibrate"]}, {"name": "spec", "run": ["spectrum-full"]},
                 {"name": "cal-codeseg", "run": ["calibrate-baselines", "--methods", "codeseg"]},
                 {"name": "bench-codeseg", "run": ["exp", "--methods", "codeseg"]},
                 {"name": "bench-fox", "run": ["exp", "--methods", "fox"]}]
        states = {"cal": ("completed", None)}
        with patch.object(runqueue, "_status_of", lambda item, fresh=False: states.get(item["name"], ("new", None))), \
                patch.object(runqueue, "data_state", lambda run, fresh=False: ("missing", "/nowhere/" + run[0])), \
                patch.object(runqueue, "unit_counts", lambda run: (0, 4, "units")), \
                patch.object(runqueue.rm, "effective_status", return_value="running"):
            root = runqueue.tree({"name": "q", "items": items, "file": ""}, {})
        self.assertEqual([g.label for g in root.children], ["Hedonic", "CoDeSEG", "FOX"])
        hedonic, codeseg, _ = root.children
        self.assertEqual([r.label for r in codeseg.children], ["3. cal-codeseg", "4. bench-codeseg"])
        self.assertEqual((hedonic.done, hedonic.total, hedonic.status), (1, 2, "partial"))
        self.assertEqual((root.done, root.total), (1, 5))  # the root still counts runs
        self.assertEqual((hedonic.depth, hedonic.children[0].depth), (1, 2))

    def test_a_methods_bar_is_how_far_its_runs_are_not_how_many_finished(self):
        runs = [qm.Node(key=("run", "cal"), label="cal", status="running", done=7, total=8),
                qm.Node(key=("run", "bench"), label="bench", status="running", done=6, total=8),
                qm.Node(key=("run", "h"), label="h", status="done", done=3, total=3)]
        root = qm.queue_node("q", runs, groups=["NcGame", "NcGame", "Hedonic"])
        ncgame = root.children[0]
        self.assertEqual((ncgame.done, ncgame.total, ncgame.share), (0, 2, (7 / 8 + 6 / 8) / 2))
        self.assertIn(" 81% ", qt.right_columns(ncgame, 120, runqueue.tui.glyphs(True)))  # 0/2 runs, yet 81% of the work
        self.assertAlmostEqual(root.share, (7 / 8 + 6 / 8 + 1) / 3)

    def test_one_method_keeps_the_runs_at_the_top(self):
        runs = [qm.Node(key=("run", "a"), label="a"), qm.Node(key=("run", "b"), label="b")]
        root = qm.queue_node("q", runs, groups=["Hedonic", "Hedonic"])
        self.assertEqual([c.key for c in root.children], [("run", "a"), ("run", "b")])


class TestAdoptedPreview(unittest.TestCase):
    """Calibrations an earlier run finished are copied only when the item starts; the view counts them now, and a
    benchmark waiting for the file is counted with them while it stays awaiting."""

    def test_adopted_calibrations_count_and_the_waiting_benchmark_is_previewed_with_them(self):
        import contextlib
        import io

        from hedonic.experiments.overlapping import baseline_run as br

        with tempfile.TemporaryDirectory() as tmp:
            source = br.resolve(["--networks", "dblp:top5000", "--methods", "codeseg", "--output-dir", tmp + "/rc1"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(br.run_calibration(source), 0)  # a fixed setting: nothing is loaded
            cal = ["calibrate-baselines", "--networks", "dblp:top5000,amazon:top5000", "--methods", "codeseg",
                   "--output-dir", tmp + "/cal", "--adopt-from", tmp + "/rc1"]
            state, folder = runqueue._baselines_state(cal[1:])
            self.assertEqual(state, "missing")  # its own file decides whether the item runs: nothing is copied yet
            self.assertEqual(runqueue._COUNTS[folder], (2, 2, "calibrations"))  # one adopted, one a published setting
            self.assertTrue(any("authors' published setting" in line for line in runqueue._EARLIER[folder]))
            mine = str(Path(tmp).resolve() / "cal" / br.TUNED_NAME)
            self.assertEqual(runqueue._TUNED_PREVIEW[mine], [str(Path(tmp).resolve() / "rc1" / br.TUNED_NAME)])
            seen = []

            def judged(run):
                seen.append(runqueue._option(run, "--tuned"))
                return "partial", tmp + "/bench"

            bench = ["exp", "--methods", "codeseg", "--tuned", mine]
            with patch.object(runqueue, "_data_state", judged):
                self.assertEqual(runqueue._preview(bench), tmp + "/bench")
                self.assertEqual(runqueue._preview(["exp", "--tuned", tmp + "/nothing.json"]), "")
            self.assertEqual(seen, [str(Path(tmp).resolve() / "rc1" / br.TUNED_NAME)])
            self.assertIn("adopt from rc1", runqueue._PREVIEWED[tmp + "/bench"])


class TestCalibrationProbes(unittest.TestCase):
    """A Hedonic search counts its evaluations and lists every setting it probed, read from the calibration's ledger."""

    def _ledger(self, folder: Path) -> None:
        def record(network, gamma, seed, f1, minute):
            return {"instance": network, "config": {"phase": "local", "gamma": gamma, "M": -1, "seed": seed},
                    "status": "completed", "finished": f"2026-10-19 03:{minute:02d}:00", "detection_seconds": 2.0,
                    "scoring_seconds": 3.0, "audit_seconds": 1.0,
                    "metrics": {"f1": f1, "onmi": 0.5, "predicted_community_count": 100}}
        (folder / "ledger").mkdir(parents=True)
        rows = [record("dblp:top5000", 0.5, 0, 0.4, 0), record("dblp:top5000", 1e-3, 0, 0.6, 2),
                record("dblp:top5000", 1e-3, 1, 0.8, 4), record("amazon:top5000", 0.1, 0, 0.7, 0),
                record("amazon:top5000", 0.01, 0, 0.75, 10)]
        (folder / "ledger" / "evals-x.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\ntorn{")

    def test_each_search_counts_its_evaluations_with_a_row_per_setting_and_the_runs_eta_is_its_longest_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._ledger(folder)
            plan = {"output": str(folder), "networks": ["dblp:top5000", "amazon:top5000"], "cap": "unlimited"}
            searches = [{"key": f"hedonic/{n}", "network": n, "phase": "local"} for n in plan["networks"]]
            model = qb.calibrate(searches, {"hedonic/dblp:top5000"}, {"hedonic/dblp:top5000": 60.0})
            node = qm.run_node(1, "cal", "", "running", model, None)
            tuned = {"hedonic/dblp:top5000": {"absolute_resolution": 1e-3, "max_memberships": -1, "f1_median": 0.7}}
            qb.decorate_calibration(node, plan, tuned)
        dblp, amazon = (pair.children[0] for pair in node.children)
        self.assertEqual((dblp.status, dblp.done, dblp.total, dblp.noun), ("done", 3, 3, "evaluations"))
        self.assertEqual([(row.label, row.done, row.total) for row in dblp.children],
                         [("γ 0.001 · F1 0.600 seed 0, 0.700 mean · 100 communities · 2 seeds  ★ chosen", 2, 2),
                          ("γ 0.5 · F1 0.400 · 100 communities · seed 0", 1, 1)])
        self.assertEqual(dblp.children[0].took, 12.0)
        self.assertEqual(dblp.took, 18.0)  # its clock (60 s / 4 side by side) is less than its evaluations cost: reused
        # the unfinished search is under way, priced from its own pace (one evaluation every 10 minutes); the rest of
        # it is a row for the evaluation in flight and one for those to come, so every row has a bar
        total = qb.SEARCH_EVALS["unlimited"]
        self.assertEqual((amazon.status, amazon.done, amazon.total), ("running", 2, total))
        self.assertEqual((amazon.left, amazon.basis), ((total - 2) * 600.0, "estimate"))
        now, later = amazon.children[-2:]
        self.assertEqual((now.status, now.done, now.total), ("running", 0, 1))
        self.assertEqual((later.status, later.done, later.total, later.left), ("pending", 0, total - 3, (total - 3) * 600.0))
        self.assertTrue(later.label.startswith(f"evaluations 4-{total}"))
        self.assertEqual(sum(row.total for row in amazon.children), total)
        self.assertEqual((node.done, node.total, node.status), (5, 3 + total, "running"))
        self.assertEqual(node.left, amazon.left)  # side by side: the longest search, not the sum


def _timed(done_pairs=(), parallel=1, seconds=2.0) -> qm.Family:
    """Like ``_family`` but the finished units carry the time they took."""
    family = _family(done_pairs, cost=lambda u: (1.0, "measured"))
    for unit in family.units:
        if unit.done:
            unit.took = seconds
    family.parallel = parallel
    return family


class TestLiveMarks(unittest.TestCase):
    """The spinner goes down every level to the units being worked on, and only there."""

    def test_the_running_marks_reach_the_unit_being_worked_on_through_every_level(self):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        pair_a, pair_b = node.children
        self.assertEqual((node.status, pair_a.status, pair_b.status), ("running", "done", "running"))
        walk, depth = pair_b, 0
        while walk.children:
            running = [c for c in walk.children if c.status == "running"]
            self.assertEqual(len(running), 1, walk.label)  # one unit at a time with parallel 1: one path down
            walk, depth = running[0], depth + 1
        self.assertEqual((walk.label, depth), ("seed 0", 3))
        self.assertEqual([c.status for c in pair_b.children], ["running", "pending"])

    def test_the_next_parallel_units_of_the_pair_in_work_are_running_and_never_those_of_the_next_pair(self):
        family = _timed(done_pairs=("A",), parallel=3)
        qm.mark_running([family])
        running = [u.coords for u in family.units if u.running]
        self.assertEqual(running, [("B", "fixed", "0", "0"), ("B", "fixed", "0", "1"), ("B", "fixed", "0.1", "0")])
        late = _timed(done_pairs=(), parallel=50)  # more workers than units left in the pair: the pair's units only
        qm.mark_running([late])
        self.assertEqual({u.coords[0] for u in late.units if u.running}, {"A"})
        self.assertEqual(sum(u.running for u in late.units), 12)

    def test_a_builder_that_flags_its_own_running_unit_is_not_second_guessed(self):
        family = _timed(done_pairs=("A",), parallel=4)
        family.units[-1].running = True
        qm.mark_running([family])
        self.assertEqual(sum(u.running for u in family.units), 1)

    def test_between_families_the_one_of_the_earliest_pair_works_and_an_audit_goes_before_its_detector_runs(self):
        audit = qm.Family("audit", "audit steps", "", [qm.Dim("pair", "pairs"), qm.Dim("resolution", "resolutions")],
                          [qm.Unit((pair, g), pair == "A") for pair in ("A", "B") for g in ("0", "1")], parallel=2)
        detector = qm.Family("detector", "detector runs", "", [qm.Dim("pair", "pairs"), qm.Dim("seed", "seeds")],
                             [qm.Unit((pair, seed), False) for pair in ("A", "B") for seed in ("0", "1")], parallel=2)
        qm.mark_running([audit, detector])
        self.assertEqual([u.coords for u in audit.units if u.running], [])
        self.assertEqual([u.coords for u in detector.units if u.running], [("A", "0"), ("A", "1")])  # pair A is audited: its runs
        audit.units[0].done = audit.units[1].done = False  # audit steps of pair A are still to do: the runs wait for them
        detector.units[0].running = detector.units[1].running = False
        qm.mark_running([audit, detector])
        self.assertEqual([u.coords for u in audit.units if u.running], [("A", "0"), ("A", "1")])
        self.assertEqual([u.coords for u in detector.units if u.running], [])

    def test_a_stopped_run_shows_no_spinner_and_a_pair_with_some_work_done_is_partial_not_running(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A",))]), None)
        self.assertEqual(node.status, "stopped")
        self.assertFalse(any(u.running for u in qm.RunModel([_timed()]).families[0].units))
        pair_b = node.children[1]
        self.assertEqual(pair_b.status, "pending")
        part = qm.family_nodes(_timed(done_pairs=("A",)), [u for u in _timed(done_pairs=("A",)).units if u.coords[0] == "A"][:6] +
                               [qm.Unit(("A", "open", "0", "0"), False)])
        self.assertIn(part[0].status, ("partial", "done"))

    def test_a_partly_done_node_that_nothing_is_working_on_is_partial(self):
        units = [qm.Unit(("A", "x"), True, 1.0, "", False, 1.0), qm.Unit(("A", "y"), False)]
        family = qm.Family("f", "things", "", [qm.Dim("pair", "pairs"), qm.Dim("thing", "things")], units)
        (pair,) = qm.family_nodes(family, family.units)
        self.assertEqual(pair.status, "partial")
        units[1].running = True
        self.assertEqual(qm.family_nodes(family, family.units)[0].status, "running")

    def test_the_viewer_draws_a_spinner_on_each_level_down_to_the_unit(self):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        root = qm.queue_node("q", [node], "running")
        view = qt.View()
        view.expanded = {n.key for n in qt.live_path(root)}
        view.fresh = False
        screen = [ANSI.sub("", line) for line in qt.render(root, view, (110, 40), G, "", "@")]
        spinning = [line for line in screen if "@" in line]
        self.assertEqual(len(spinning), 6)  # queue, run, pair B, policy, resolution, seed
        self.assertTrue(any("seed 0" in line for line in spinning))


class TestTook(unittest.TestCase):
    def test_a_finished_node_says_done_and_how_long_it_took(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A", "B"), seconds=2.0)]), None)
        self.assertEqual(qm.eta_text(node), "done 48s")  # 24 units x 2 s
        self.assertEqual(qm.eta_text(node.children[0]), "done 24s")
        leaf = node.children[0].children[0].children[0].children[0]
        self.assertEqual(qm.eta_text(leaf), "done 2.0s")
        self.assertEqual(node.took_missing, 0)

    def test_finished_units_without_a_recorded_time_make_it_a_lower_bound_or_just_done(self):
        family = _timed(done_pairs=("A",), seconds=2.0)
        family.units[0].took = None  # one finished unit of pair A has no time
        done = qm.run_node(1, "spec", "", "stopped", qm.RunModel([family]), None).children[0]
        self.assertEqual(qm.eta_text(done), "done ≥22s")
        self.assertEqual(done.took_missing, 1)
        for unit in family.units:
            unit.took = None
        self.assertEqual(qm.eta_text(qm.run_node(1, "spec", "", "stopped", qm.RunModel([family]), None).children[0]), "done")

    def test_a_run_with_no_unit_times_falls_back_to_its_own_clock_and_a_queue_adds_its_runs_up(self):
        a = qm.run_node(1, "a", "", "done", None, (8, 8, "searches"), took=125.0)
        b = qm.run_node(2, "b", "", "done", None, (4, 4, "units"))
        self.assertEqual(qm.eta_text(a), "done 2:05")
        self.assertEqual(qm.eta_text(b), "done")
        root = qm.queue_node("q", [a, b], "running")
        self.assertEqual((root.took, root.took_missing), (125.0, 1))
        self.assertIn("took 2:05", "\n".join(root.detail))

    def test_the_detail_pane_says_what_the_took_figure_is(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A", "B"))]), None)
        text = "\n".join(qt.detail_lines(node, 120, time.time()))
        self.assertIn("took: 48s", text)
        self.assertIn("skipped", text)


class TestLayout(unittest.TestCase):
    def _screen(self, width, height=40, cursor=None):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        root = qm.queue_node("q", [node], "running")
        view = qt.View()
        qt.flatten(root, view)
        if cursor is not None:
            view.cursor = cursor(root).key
        return root, view, qt.render(root, view, (width, height), G, "", "@")

    def test_the_label_column_is_only_as_wide_as_its_longest_label_so_the_table_sits_next_to_it(self):
        root, view, lines = self._screen(180)
        heads = ANSI.sub("", lines[1])
        bar_at = heads.index("progress")
        rows = qt.flatten(root, view)
        longest = max(qt.label_need(n) for n in rows)
        self.assertLessEqual(bar_at, longest + 6)  # not pushed out to the terminal's right edge
        for line in lines[2:2 + len(rows)]:
            self.assertEqual(ANSI.sub("", line).count("%"), 1)
            self.assertEqual(ANSI.sub("", line).index("%") - 5 - 1 >= 0, True)
        # the same columns on a narrow terminal: labels give way, the table keeps its place at the right
        _, _, narrow = self._screen(70)
        for line in narrow:
            self.assertLessEqual(len(ANSI.sub("", line)), 70)

    def test_every_row_has_its_columns_at_the_same_place(self):
        root, view, lines = self._screen(150)
        marks = {ANSI.sub("", line).index("%") for line in lines[2:] if "%" in ANSI.sub("", line)}
        self.assertEqual(len(marks), 1)

    def test_the_selected_row_is_highlighted_from_its_first_column_to_the_terminal_edge(self):
        with patch("hedonic.experiments.overlapping.tui.color", lambda: True):
            root, view, lines = self._screen(120, cursor=lambda r: r.children[0].children[0])
        selected = next(line for line in lines if line.startswith("\033[7m"))
        self.assertEqual(selected.count("\033["), 2)  # one reverse-video start and the final reset: nothing resets it midway
        self.assertTrue(selected.endswith("\033[0m"))
        self.assertEqual(len(ANSI.sub("", selected)), 120)
        self.assertTrue(ANSI.sub("", selected).startswith("›"))
        self.assertEqual(sum(line.startswith("\033[7m") for line in lines), 1)


TRACEBACK = ("WorkerFailed: worker for item 80 failed:\nTraceback (most recent call last):\n  File \"x.py\", line 1, in f\n"
             "ValueError: Invalid initial_membership: membership labels must be contiguous from 0\n")


class TestFailureIsExplained(unittest.TestCase):
    """A red mark alone left the user asking why and where: the viewer says which run, which error and which step."""

    def _patched(self):
        run_entry = {"name": "leveraging-gt"}
        return (patch.object(runqueue, "_state", lambda name: ("failed", run_entry) if name == "leveraging-gt" else ("new", None)),
                patch.object(runqueue.rm, "read_progress", lambda entry: {"reason": TRACEBACK}))

    def test_a_failed_run_gives_its_error_the_step_it_was_in_and_where_the_traceback_is(self):
        first, second = self._patched()
        with first, second:
            lines = runqueue.failure_lines("leveraging-gt")
            self.assertEqual(runqueue.failure_lines("benchmark"), [])  # a run that never failed says nothing
        self.assertEqual(lines[0], "why: ValueError: Invalid initial_membership: membership labels must be contiguous from 0")
        self.assertEqual(lines[1], "while: WorkerFailed: worker for item 80 failed")
        self.assertEqual(lines[2], "whole traceback: hedonic run logs leveraging-gt")

    def test_a_failure_without_a_recorded_reason_says_nothing_instead_of_guessing(self):
        with patch.object(runqueue, "_state", lambda name: ("failed", {"name": name})), \
                patch.object(runqueue.rm, "read_progress", lambda entry: {}):
            self.assertEqual(runqueue.failure_lines("x"), [])

    def test_the_stop_lines_of_the_queue_carry_the_runs_error(self):
        entry = {"name": "overlap-queue", "items": [{"name": "calib", "run": ["calibrate"]}, {"name": "leveraging-gt", "run": ["startcurve"]}]}
        first, second = self._patched()
        with first, second, patch.object(runqueue, "_status_of", lambda item, fresh=False: ("stopped", None)):
            text = ANSI.sub("", "\n".join(runqueue.stop_lines(entry, {"current": 1, "reason": "leveraging-gt ended as failed"}, "failed")))
        self.assertIn("failed at leveraging-gt (item 2 of 2)", text)
        self.assertIn("ValueError: Invalid initial_membership", text)
        self.assertIn("hedonic run resume overlap-queue", text)

    def test_the_tree_view_shows_a_notice_under_the_header_and_still_fits_every_size(self):
        root = _tree()
        notice = ["  failed at spec (item 2 of 3)", "    why: ValueError: boom", "  continue with: hedonic run resume q"]
        for width, height in ((100, 30), (70, 18), (120, 40), (60, 8)):
            lines = qt.render(root, qt.View(), (width, height), G, qt.header_line(root, "failed", 65.0, G), notice=notice)
            self.assertLessEqual(len(lines), height)
            for line in lines:
                self.assertLessEqual(len(ANSI.sub("", line)), width, line)
        text = [ANSI.sub("", line) for line in qt.render(root, qt.View(), (110, 30), G, qt.header_line(root, "failed", 65.0, G), notice=notice)]
        self.assertIn("why: ValueError: boom", text[2])
        self.assertIn("progress", text[4])  # the column heads follow the notice, then the rows


class TestModelFailure(unittest.TestCase):
    def test_a_model_that_cannot_be_built_never_changes_a_runs_counts(self):
        def broken(*_args, **_kwargs):
            raise KeyError("job")
        with patch.dict(runqueue._MODEL, {}, clear=True):
            runqueue._set_model("/x", broken)
            self.assertNotIn("/x", runqueue._MODEL)
            runqueue._MODEL["/x"] = qm.RunModel()
            runqueue._set_model("/x", broken)  # and a stale model is dropped, not kept
            self.assertNotIn("/x", runqueue._MODEL)


@unittest.skipUnless(os.name == "posix", "needs a pty")
def _emulate(data: str, rows: int, cols: int) -> list[str]:
    """The screen a terminal shows after ``data``: cursor positioning, erase in line and display, text (enough for
    the viewer's output; other escape sequences are ignored)."""
    screen = [[" "] * cols for _ in range(rows)]
    r = c = i = 0
    while i < len(data):
        if data[i] == "\x1b":
            match = re.match(r"\x1b\[([0-9;?<]*)([A-Za-z])", data[i:])
            if match:
                params, command = match.groups()
                if command == "H" and not params.startswith("?"):
                    numbers = [int(x) for x in params.split(";") if x]
                    r = min(rows - 1, numbers[0] - 1) if numbers else 0
                    c = numbers[1] - 1 if len(numbers) > 1 else 0
                elif command == "K":
                    screen[r][c:] = [" "] * (cols - c)
                elif command == "J" and not params.startswith("?"):
                    screen[r][c:] = [" "] * (cols - c)
                    for below in range(r + 1, rows):
                        screen[below] = [" "] * cols
                i += len(match.group(0))
            else:
                i += 1
            continue
        ch = data[i]
        if ch == "\r":
            c = 0
        elif ch == "\n":
            r = min(rows - 1, r + 1)
        elif c < cols:
            screen[r][c] = ch
            c += 1
        i += 1
    return ["".join(row).rstrip() for row in screen]


class TestPaint(unittest.TestCase):
    def test_the_first_frame_draws_every_row_and_clears_below(self):
        text = qt.paint(None, ["a", "b"])
        self.assertTrue(text.startswith(qt.SYNC_ON) and text.endswith(qt.SYNC_OFF))
        self.assertEqual(_emulate("x" * 5 + "\r\n" + "y" * 5 + "\r\nzzz" + text, 4, 10)[:3], ["a", "b", ""])

    def test_an_unchanged_frame_writes_nothing(self):
        self.assertEqual(qt.paint(["a", "› selected", "c"], ["a", "› selected", "c"]), "")

    def test_only_the_rows_that_changed_are_written(self):
        text = qt.paint(["a", "› selected", "c ⠦"], ["a", "› selected", "c ⠧"])
        self.assertNotIn("selected", text)  # the highlighted row is left alone
        self.assertIn("\x1b[3;1Hc ⠧\x1b[K", text)
        self.assertEqual(_emulate(qt.paint(None, ["a", "› selected", "c ⠦"]) + text, 3, 20), ["a", "› selected", "c ⠧"])

    def test_a_shorter_frame_clears_the_rows_below_it(self):
        screen = _emulate(qt.paint(None, ["a", "b", "c"]) + qt.paint(["a", "b", "c"], ["a"]), 3, 5)
        self.assertEqual(screen, ["a", "", ""])


class TestPty(unittest.TestCase):
    SCRIPT = textwrap.dedent("""
        import sys
        sys.path.insert(0, {tests!r})
        from test_queue_tree import _tree
        from hedonic.experiments.overlapping import queuetui as qt
        code = qt.run(_tree, lambda: ("running", 5.0))
        sys.exit(code)
    """)

    def test_the_real_loop_draws_the_tree_answers_keys_wheel_and_clicks_and_restores_the_terminal(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        script = self.SCRIPT.format(tests=str(Path(__file__).resolve().parent))
        pid, fd = pty.fork()
        if pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execv(sys.executable, [sys.executable, "-c", script])
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
        out = b""

        def pump(seconds: float) -> None:
            nonlocal out
            end = time.time() + seconds
            while time.time() < end:
                if select.select([fd], [], [], 0.05)[0]:
                    try:
                        out += os.read(fd, 65536)
                    except OSError:
                        return

        def screen() -> str:  # the rows only change where they differ, so replay the whole output onto a screen
            text = out.decode(errors="ignore")
            return "\n".join(_emulate(text[text.rfind("\x1b[?1049h"):], 30, 110))
        try:
            pump(2.0)
            first = screen()
            self.assertIn("queue q", first)
            self.assertIn("2. spec", first)
            for data in ("j", "j", "l", "\x1b[<65;20;10M", "E"):
                os.write(fd, data.encode())
                pump(0.5)
            self.assertIn("fixed", screen().split("──")[0])  # E opened everything, down to the action policies
            os.write(fd, b"C")
            pump(0.5)
            self.assertNotIn("fixed", screen().split("──")[0])  # C closed all but the runs
            os.write(fd, b"q")
            pump(1.5)
        finally:
            try:
                os.kill(pid, 9)  # never leave the child waiting if an assertion above failed
            except ProcessLookupError:
                pass
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                status = 0
            os.close(fd)
        text = out.decode(errors="ignore")
        self.assertIn("\x1b[?1049h", text)  # alternate screen
        self.assertIn("\x1b[?1000h\x1b[?1006h", text)  # mouse reporting on
        self.assertLess(text.rfind("\x1b[?1000l"), text.rfind("\x1b[?1049l"))  # and off again before leaving it
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, "the viewer exits cleanly on q")


if __name__ == "__main__":
    unittest.main()
