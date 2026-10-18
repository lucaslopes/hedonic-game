"""`hedonic run queue`: what counts as finished on disk, the checks of a queue file, and the paper's queue file."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import start_quality as sq

from hedonic.experiments.resources import CONFIG_DIR

QUEUE = str(CONFIG_DIR / "queue-paper.toml")
PAPER_ITEMS = [
    "hedonic-calibration", "baseline-calibration", "synthetic-networks", "leveraging-gt", "gt-robustness", "benchmark",
]


def _register(root: Path, name: str, status: str) -> None:
    """A registered run with the given status, standing in for the worker that a launch would start."""
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    entry = {"name": name, "kind": "spectrum", "output_dir": str(out), "created": time.time(), "session": "",
             "backend": "process", "log": str(out / "worker.log"), "argv": []}
    rm.write_json(rm.entry_path(name), entry)
    rm.write_json(rm.progress_path(entry), {"status": status, "plan": [], "setup": []})


class _Launcher:
    """Stands in for `hedonic run ... --detach` and `hedonic run resume`: the launched run ends with ``outcome``."""

    def __init__(self, root: Path, outcome: str = "completed"):
        self.root, self.outcome, self.calls = root, outcome, []

    def __call__(self, name: str, run: list[str], resuming: bool) -> int:
        self.calls.append((name, "resume" if resuming else "start"))
        _register(self.root, name, self.outcome)
        return 0


def _coverage(folder: Path, expected: int, recorded: int) -> None:
    """A spectrum's coverage report, as the study writes it."""
    folder.mkdir(parents=True, exist_ok=True)
    report = {"complete": recorded >= expected, "expected_detector_conditions": expected,
              "recorded_detector_conditions": recorded}
    (folder / "coverage_report.json").write_text(json.dumps(report))


def _queue(root: Path, items: list[tuple[str, list[str]]]) -> dict:
    out = root / "queues" / "check"
    out.mkdir(parents=True, exist_ok=True)
    return {"name": "check", "kind": "queue", "items": [{"name": n, "run": r} for n, r in items], "file": "",
            "output_dir": str(out), "created": time.time(), "session": "", "backend": "process",
            "log": str(out / "worker.log")}


class _Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.overlap = self.root / "overlap"  # the output folder of the user config (the benchmark's and spectrum's base)
        self._patches = [
            patch.dict(os.environ, {"HEDONIC_RUNS_DIR": str(self.root / "registry")}),
            patch("hedonic.experiments.overlapping.userconfig.defaults",
                  return_value={"output_dir": str(self.overlap), "cache_dir": str(self.root / "cache")}),
            patch.object(runqueue, "POLL_SECONDS", 0),
        ]
        for patcher in self._patches:
            patcher.start()
        runqueue._DATA_CACHE.clear()

    def tearDown(self):
        for patcher in reversed(self._patches):
            patcher.stop()
        runqueue._DATA_CACHE.clear()
        self._tmp.cleanup()

    def _state(self, run: list[str]) -> str:
        """What the data on disk says about one item, read afresh (the queue reuses a reading for a minute)."""
        runqueue._DATA_CACHE.clear()
        return runqueue.data_state(run)[0]


class TestWhatIsFinished(_Base):
    def test_a_spectrum_is_complete_only_when_its_report_accounts_for_every_condition(self):
        run = ["spectrum", "--name", "study"]
        self.assertEqual(self._state(run), "missing")
        _coverage(self.overlap / "study", expected=3, recorded=1)
        self.assertEqual(self._state(run), "partial")
        _coverage(self.overlap / "study", expected=3, recorded=3)
        self.assertEqual(self._state(run), "complete")

    def test_an_explicit_output_folder_is_the_one_that_is_checked(self):
        elsewhere = self.root / "elsewhere"
        _coverage(elsewhere, expected=2, recorded=2)
        run = ["spectrum", "--name", "study", "--output-dir", str(elsewhere)]
        self.assertEqual(self._state(run), "complete")
        self.assertEqual(Path(runqueue.data_state(run)[1]).resolve(), elsewhere.resolve())

    def test_a_start_curve_is_complete_when_every_condition_has_a_record(self):
        config = self.root / "start.toml"
        config.write_text('[start_quality]\nnetworks = ["dblp"]\nseeds = "0-1"\nlevels = [0, "random"]\n'
                          f'phase = "local"\noutput = "{self.root / "curve"}"\n')
        run = ["startcurve", "--config", str(config), "--name", "curve"]
        keys = [condition["key"] for condition in sq.conditions(sq.front_study(run[1:]))]
        self.assertEqual(len(keys), 4)  # one network, two levels, two seeds
        self.assertEqual(self._state(run), "missing")
        (self.root / "curve").mkdir()
        records = self.root / "curve" / sq.RESULTS_NAME
        records.write_text("".join(json.dumps({"key": key, "phase": "local"}) + "\n" for key in keys[:2]))
        self.assertEqual(self._state(run), "partial")
        records.write_text("".join(json.dumps({"key": key, "phase": "local"}) + "\n" for key in keys))
        self.assertEqual(self._state(run), "complete")

    def test_a_controlled_study_is_complete_when_its_json_exists(self):
        target = self.root / "cc" / "controlled_overlap.json"  # the study's JSON goes to the run folder
        run = ["controlled", "--config", "overlap-heatmap", "--name", "cc", "--output-dir", str(self.root / "cc")]
        self.assertEqual(self._state(run), "missing")
        (self.root / "cc").mkdir()
        (self.root / "cc" / "progress.json").write_text(json.dumps({"done": 3, "total": 640}))
        self.assertEqual(self._state(run), "partial")
        target.write_text("{}")  # a JSON of another build (a 1.x one names no core): run again, after moving it aside
        self.assertEqual(self._state(run), "partial")
        from hedonic.experiments.overlapping import controlled_overlap as co

        target.write_text(json.dumps({"implementation_identity": {"environment": co._environment_identity()}}))
        self.assertEqual(self._state(run), "complete")

    def test_units_are_counted_from_the_files(self):
        run = ["spectrum", "--name", "study"]
        _coverage(self.overlap / "study", expected=4, recorded=1)
        self._state(run)
        self.assertEqual(runqueue.unit_counts(run), (1, 4, "conditions"))
        # the study's own plan is fresher than the report, and wins when it lists the planned jobs
        plan = [{"status": "completed"}] * 3 + [{"status": "pending"}] * 2
        from hedonic.experiments.overlapping import gt_spectrum as gs

        progress = self.overlap / "study" / "progress.json"
        progress.write_text(json.dumps({"jobs": [], "plan": plan, "tracked_files_sha256": gs.tracked_digest()}))
        self._state(run)
        self.assertEqual(runqueue.unit_counts(run), (3, 5, "conditions"))
        # a plan written under other code says nothing about the records the current code would reuse
        progress.write_text(json.dumps({"jobs": [], "plan": plan, "tracked_files_sha256": "other-code"}))
        self._state(run)
        self.assertEqual(runqueue.unit_counts(run), (1, 4, "conditions"))

    def test_the_overview_lists_every_run_with_what_is_done_and_left(self):
        _coverage(self.overlap / "done", expected=2, recorded=2)
        _coverage(self.overlap / "half", expected=4, recorded=1)
        queue = _queue(self.root, [("done", ["spectrum", "--name", "done"]), ("half", ["spectrum", "--name", "half"]),
                                   ("none", ["spectrum", "--name", "none"])])
        runqueue._DATA_CACHE.clear()
        text = "\n".join(runqueue.overview(queue["items"], runqueue.tui.glyphs(False)))
        self.assertIn("100% 2/2 conditions", text)
        self.assertIn("25% 1/4 conditions", text)
        self.assertIn("1 of 3 runs complete · 2 still to do", text)
        self.assertEqual(sum(line.lstrip()[:2] in ("1.", "2.", "3.") for line in text.splitlines()), 3)

    def test_the_report_opens_each_run_into_its_pairs_and_marks_stale_work(self):
        _coverage(self.overlap / "done", expected=2, recorded=2)
        _coverage(self.overlap / "old", expected=4, recorded=4)
        (self.overlap / "old" / "manifest.json").write_text(json.dumps({"identity": {"tracked_files_sha256": "other-code"}}))
        queue = _queue(self.root, [("done", ["spectrum", "--name", "done"]), ("old", ["spectrum", "--name", "old"])])
        queue["items"][0].update(title="Chapter 4", about="what this run is for")
        runqueue._DATA_CACHE.clear()
        text = "\n".join(runqueue.report(queue["items"], runqueue.tui.glyphs(False)))
        self.assertIn("1. done · Chapter 4", text)
        self.assertIn("what this run is for", text)
        self.assertIn("2. old", text)
        self.assertIn("written by other code", text)  # stale records are named, and do not count as done
        self.assertIn("0/4 conditions", text)
        self.assertIn("total: 1 of 2 runs complete · 1 still to do", text)

    def test_the_matrix_puts_every_pair_against_every_run_in_a_few_lines(self):
        folder = self.overlap / "curve"
        runqueue._COUNTS[str(folder)] = (1, 3, "conditions")
        runqueue._PARTS[str(folder)] = [("DBLP · all", "level 0", [True, False]), ("DBLP · top5000", "level 0", [False])]
        item = {"name": "curve-1006", "run": ["startcurve", "--name", "curve"], "short": "Ch5 start"}
        with patch.object(runqueue, "data_state", return_value=("partial", str(folder))), \
                patch.object(runqueue, "_status_of", return_value=("stopped", None)):
            lines = runqueue.matrix([item], runqueue.tui.glyphs(False))
        text = "\n".join(lines)
        self.assertEqual(len(lines), 4)  # header, two pairs, the totals
        self.assertIn("Ch5 start", lines[0])
        self.assertIn("50%", text)  # DBLP · all: one of two
        self.assertIn("1/3", lines[-1])
        self.assertTrue(all(len(runqueue.tui.strip_ansi(line)) <= runqueue.WIDTH for line in lines))

    def test_the_view_reads_the_words_of_each_run_from_the_queue_file(self):
        file = self.root / "q.toml"
        file.write_text('name = "check"\n[[items]]\nname = "a"\nrun = ["spectrum"]\ntitle = "T"\nshort = "S"\n')
        items = runqueue.with_labels([{"name": "a", "run": ["spectrum"]}, {"name": "b", "run": ["spectrum"]}], str(file))
        self.assertEqual((items[0]["title"], items[0]["short"]), ("T", "S"))
        self.assertNotIn("title", items[1])
        self.assertEqual(runqueue.with_labels([{"name": "a", "run": []}], str(self.root / "gone.toml")), [{"name": "a", "run": []}])

    def test_a_strip_shades_buckets_when_there_are_more_units_than_marks(self):
        glyphs = runqueue.tui.glyphs(False)
        self.assertEqual(runqueue._strip([True, False, True], glyphs), "#-#")
        long = [True] * 24 + [False] * 24
        strip = runqueue._strip(long, glyphs, width=4)
        self.assertEqual(strip, "##--")
        self.assertEqual(len(runqueue._strip([True] * 100, glyphs)), 24)

    def _unit(self, run: list[str], seed: int) -> tuple[Path, str]:
        """The folder of one detector run of a benchmark, and the digest its runner stamps on the record."""
        cfg = qs.config_from_args(qs.build_parser().parse_args(run[1:]))
        cfg.output_dir = str(self.overlap / run[run.index("--name") + 1])
        argv, out = qs.unit_argv(cfg, "dblp", seed, qs.BY_KEY["hedonic"], None)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            cr.main([*argv, "--identity-only"])
        return out, json.loads(buffer.getvalue().strip().splitlines()[-1])["config_digest"]

    def _record(self, out: Path, status: str, digest: str, cover: str | None = None) -> None:
        """A record as the runner writes it under a detector run's folder."""
        record = {"status": status, "config_digest": digest, "dataset": "dblp", "method": "hedonic_local"}
        if cover:
            record["cover_path"] = cover
            (out / cover).parent.mkdir(parents=True, exist_ok=True)
            (out / cover).write_text("{}")
        path = out / "runs" / "dblp" / "hedonic_local.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))

    def test_a_benchmark_unit_is_reused_only_with_the_digest_of_its_own_arguments(self):
        run = ["exp", "--networks", "dblp", "--methods", "hedonic", "--seeds", "0,1", "--name", "bench"]
        out0, digest0 = self._unit(run, 0)
        out1, digest1 = self._unit(run, 1)
        self.assertEqual(self._state(run), "missing")
        self._record(out0, "completed", digest0)
        self.assertEqual(self._state(run), "partial")  # seed 1 has no record yet
        self._record(out1, "completed", "digest-of-other-arguments")
        self.assertEqual(self._state(run), "partial")  # its record is not the runner's to reuse
        self._record(out1, "completed", digest1)
        self.assertEqual(self._state(run), "complete")

    def test_under_save_covers_a_completed_unit_needs_its_cover_on_disk(self):
        run = ["exp", "--networks", "dblp", "--methods", "hedonic", "--seeds", "0", "--save-covers", "--name", "bench"]
        out, digest = self._unit(run, 0)
        self._record(out, "completed", digest)
        self.assertEqual(self._state(run), "missing")  # the record is there, its cover is not
        self._record(out, "completed", digest, cover="covers/dblp-hedonic-0.json")
        self.assertEqual(self._state(run), "complete")


class TestSpectrumAudit(_Base):
    def test_a_pairs_exact_audit_is_shown_under_it_and_not_counted_in_its_conditions(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        folder = self.root / "spec"
        (folder / "audit").mkdir(parents=True)
        current = gs.tracked_digest()
        done = {"rows": [{}] * 6, "identity": {"implementation": {"tracked_files_sha256": current}}}
        old = {"rows": [{}] * 6, "identity": {"implementation": {"tracked_files_sha256": "e" * 64}}}
        (folder / "audit" / "amazon-all.json").write_text(json.dumps(done))
        (folder / "audit" / "dblp-all.json").write_text(json.dumps(old))  # written by other code: not shown as done
        rows = [{"job": job, "policy": "fixed_labels", "gamma": 0.1, "seed": 0} for job in ("amazon-all", "dblp-all", "youtube-all")]
        live = {"stage": "auditing youtube/all · open_labels · γ=0.1  2/6 resolutions · ETA ~1:00"}
        parts: list = []
        runqueue._note_spectrum_audit(parts, rows, folder, live, current)
        got = {group: sum(flags) for group, label, flags in parts if label == runqueue.AUDIT_LABEL}
        self.assertEqual(got, {"Amazon · all": 6, "DBLP · all": 0, "YouTube · all": 2})
        self.assertTrue(all(len(flags) == 6 for *_x, flags in parts))

    def test_without_any_audit_information_no_row_is_invented(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        parts: list = []
        runqueue._note_spectrum_audit(parts, [{"job": "amazon-all"}], self.root / "none", {}, gs.tracked_digest())
        self.assertEqual(parts, [])

    def test_the_report_keeps_the_audit_out_of_the_pair_line_and_the_matrix(self):
        folder = str(self.root / "spec2")
        runqueue._PARTS[folder] = [("Amazon · all", "fixed labels", [True, False]), ("Amazon · all", runqueue.AUDIT_LABEL, [True] * 3 + [False] * 5)]
        runqueue._COUNTS[folder] = (1, 2, "conditions")
        items = [{"name": "s", "run": ["spectrum", "--name", "s", "--output-dir", folder], "short": "spec"}]
        with patch.object(runqueue, "data_state", return_value=("partial", folder)), \
                patch.object(runqueue, "_status_of", return_value=("running", None)):
            text = "\n".join(runqueue.report(items))
            matrix = "\n".join(runqueue.matrix(items))
        self.assertIn("exact audit", text)
        self.assertRegex(text, r"Amazon · all\s+1/2")  # the pair line counts conditions only
        self.assertRegex(text, r"exact audit\s+3/8")
        self.assertIn("50%", matrix)  # 1 of 2 conditions; the 8 audit flags are not in it


class TestCodeIdentity(_Base):
    """hedonic 2.0 keeps no equivalence file: a record is current only under the digest of the code that reads it."""

    def test_a_digest_is_its_own(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        self.assertEqual(gs.canonical_digest("abc"), "abc")
        self.assertIsNone(gs.canonical_digest(None))
        self.assertFalse(hasattr(gs, "EQUIVALENCE_PATH"))


class TestQueueWork(_Base):
    def test_finished_items_are_skipped_and_the_others_start_with_their_own_arguments(self):
        _coverage(self.root / "done", expected=2, recorded=2)
        _coverage(self.root / "half", expected=3, recorded=1)
        items = [("done", ["spectrum", "--name", "done", "--output-dir", str(self.root / "done")]),
                 ("half", ["spectrum", "--name", "half", "--output-dir", str(self.root / "half")]),
                 ("fresh", ["spectrum", "--name", "fresh", "--output-dir", str(self.root / "fresh")])]
        launcher = _Launcher(self.root)
        prog: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        # half's data says partial, so it is started with its own arguments, not resumed from a registry entry
        self.assertEqual(launcher.calls, [("half", "start"), ("fresh", "start")])
        self.assertEqual([row["status"] for row in prog["items"]], ["completed", "completed", "completed"])
        self.assertEqual(prog["items"][0]["note"], "finished earlier, skipped")

    def test_a_registered_run_without_a_data_rule_resumes_from_its_registry_entry(self):
        _register(self.root, "flaky", "failed")
        launcher = _Launcher(self.root)
        prog: dict = {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, [("flaky", ["x"])]), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("flaky", "resume")])

    def test_completed_with_failures_is_finished_and_the_queue_goes_on(self):
        _register(self.root, "noisy", "completed_with_failures")
        launcher = _Launcher(self.root)
        prog: dict = {}
        items = [("noisy", ["x"]), ("after", ["x"])]
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("after", "start")])  # the failures are results: noisy is not rerun

    def test_data_that_appears_after_the_first_look_is_found_at_the_items_turn(self):
        _coverage(self.root / "first", expected=1, recorded=0)
        items = [("first", ["spectrum", "--name", "first", "--output-dir", str(self.root / "first")]),
                 ("second", ["spectrum", "--name", "second", "--output-dir", str(self.root / "second")])]
        launcher = _Launcher(self.root)
        inner = launcher.__call__

        def another_agent_finishes_second(name, run, resuming):
            _coverage(self.root / "second", expected=2, recorded=2)  # produced while the first item ran
            return inner(name, run, resuming)

        prog: dict = {}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=another_agent_finishes_second)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("first", "start")])  # second was never launched
        second = prog["items"][1]
        self.assertEqual((second["first"], second["second"]), ("pending", "completed"))
        self.assertIn("another run produced it", second["note"])
        self.assertIn("the data has appeared since", out.getvalue())

    def test_data_that_vanishes_after_the_first_look_runs_again(self):
        done = self.root / "done"
        _coverage(done, expected=1, recorded=1)
        items = [("gone", ["spectrum", "--name", "gone", "--output-dir", str(done)]),
                 ("after", ["spectrum", "--name", "after", "--output-dir", str(self.root / "after")])]
        # the first look sees "gone" complete; before its turn the report is replaced by an unfinished one
        real = runqueue.verify

        def look_then_spoil(queue_items):
            rows = real(queue_items)
            _coverage(done, expected=3, recorded=1)
            return rows

        launcher = _Launcher(self.root)
        prog: dict = {}
        with patch.object(runqueue, "verify", look_then_spoil), contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", _queue(self.root, items), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual(launcher.calls, [("gone", "start"), ("after", "start")])
        self.assertEqual((prog["items"][0]["first"], prog["items"][0]["second"]), ("completed", "stopped"))

    def test_the_second_look_ignores_the_reading_kept_for_a_minute(self):
        _coverage(self.root / "x", expected=2, recorded=0)
        run = ["spectrum", "--name", "x", "--output-dir", str(self.root / "x")]
        before = runqueue.data_state(run)[0]
        _coverage(self.root / "x", expected=2, recorded=2)
        self.assertEqual(runqueue.data_state(run)[0], before)  # the kept reading
        self.assertEqual(runqueue.data_state(run, fresh=True)[0], "complete")

    def test_the_plan_says_skip_or_start_from_the_data_on_disk(self):
        _coverage(self.root / "done", expected=1, recorded=1)
        items = [("done", ["spectrum", "--name", "done", "--output-dir", str(self.root / "done")]),
                 ("fresh", ["spectrum", "--name", "fresh", "--output-dir", str(self.root / "fresh")])]
        actions = {name: action for name, _, action in runqueue.plan(_queue(self.root, items))}
        self.assertEqual(actions, {"done": "skip: already completed", "fresh": "start"})


class TestQueueFile(_Base):
    def _write(self, text: str) -> str:
        path = self.root / "q.toml"
        path.write_text(text)
        return str(path)

    def test_an_item_runs_under_its_own_name_and_a_benchmark_names_itself(self):
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\nrun = ["spectrum", "--name", "b"]\n'))
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\nrun = ["exp", "--networks", "dblp"]\n'))

    def test_arguments_the_front_door_would_refuse_are_refused_before_anything_starts(self):
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\nrun = ["list"]\n'))
        with self.assertRaises(SystemExit):
            runqueue.load_queue(self._write('name = "q"\n[[items]]\nname = "a"\n'
                                            'run = ["exp", "--methods", "nonsense", "--name", "a"]\n'))

    def test_a_config_named_relative_to_the_start_directory_is_stored_absolute(self):
        (self.root / "heatmap.toml").write_text((CONFIG_DIR / "overlap-heatmap.toml").read_text())
        path = self._write('name = "q"\n[[items]]\nname = "cc"\n'
                           'run = ["controlled", "--config", "heatmap.toml", "--name", "cc"]\n')
        cwd = os.getcwd()
        os.chdir(self.root)
        try:
            queue = runqueue.load_queue(path)
        finally:
            os.chdir(cwd)
        self.assertEqual(queue["items"][0]["run"][2], str((self.root / "heatmap.toml").resolve()))

    def test_the_paper_queue_lists_every_item_under_its_own_name_with_existing_configs(self):
        queue = runqueue.load_queue("paper", root=str(self.root / "q"))
        self.assertEqual(queue["name"], "paper-queue")
        self.assertEqual([item["name"] for item in queue["items"]], PAPER_ITEMS)
        for item in queue["items"]:
            self.assertEqual(runqueue._option(item["run"], "--name"), item["name"])
            config = runqueue._option(item["run"], "--config")
            if config and "/" in config:  # a file: stored absolute, and present
                self.assertTrue(Path(config).is_file(), config)

    def test_the_item_that_writes_the_calibration_runs_before_every_item_that_reads_it(self):
        queue = runqueue.load_queue(QUEUE, root=str(self.root / "q"))
        names = [item["name"] for item in queue["items"]]
        readers = [item["name"] for item in queue["items"] if runqueue._option(item["run"], "--tuned")]
        self.assertEqual(readers, ["leveraging-gt", "gt-robustness", "benchmark"])
        for reader in readers:
            tuned = runqueue._option(next(i for i in queue["items"] if i["name"] == reader)["run"], "--tuned")
            self.assertIn("/hedonic-calibration/hoc_tuned.json", tuned)
            self.assertLess(names.index("hedonic-calibration"), names.index(reader))

    def test_every_item_covers_the_pairs_in_the_same_order_top5000_first(self):
        # the manuscript's pairs: DBLP, Amazon and YouTube with both covers, Friendster and LiveJournal with top5000
        top5000 = ["dblp:top5000", "amazon:top5000", "youtube:top5000", "friendster:top5000", "livejournal:top5000"]
        complete = ["dblp:all", "amazon:all", "youtube:all"]
        queue = runqueue.load_queue(QUEUE, root=str(self.root / "q"))
        by_name = {item["name"]: item for item in queue["items"]}
        for name in ("hedonic-calibration", "baseline-calibration"):
            self.assertEqual(runqueue._option(by_name[name]["run"], "--networks").split(","), top5000 + complete)
        plan = sq.resolve(["--config", runqueue._option(by_name["leveraging-gt"]["run"], "--config")])
        self.assertEqual(plan["networks"], top5000 + ["dblp", "amazon", "youtube"])
        cfg = qs.load_config(runqueue._option(by_name["benchmark"]["run"], "--config"))
        self.assertEqual(cfg.networks, top5000 + complete)
        spectrum = runqueue._spectrum_options("spectrum", by_name["gt-robustness"]["run"][1:])
        self.assertEqual([f"{d}:{c}" for d, c in spectrum["pairs"]], top5000 + complete)
        self.assertEqual(cfg.order, "seed,network,method")


PAPER_PAIRS = ["dblp:top5000", "amazon:top5000", "youtube:top5000", "friendster:top5000", "livejournal:top5000",
               "dblp:all", "amazon:all", "youtube:all"]


class TestPairByPair(_Base):
    """A queue with pairs goes pair by pair: the whole block of runs on one pair before the next pair starts."""

    ORDER = ["dblp:top5000", "amazon:top5000", "dblp:all", "amazon:all"]

    def _curve(self, name: str) -> list[str]:
        """A start-quality run over the four pairs of ORDER, in its own folder (spelled as the paper's config does)."""
        config = self.root / f"{name}.toml"
        config.write_text('[start_quality]\nnetworks = ["dblp:top5000", "amazon:top5000", "dblp", "amazon"]\nseeds = "0-1"\n'
                          f'levels = [0, "random"]\nphase = "local"\noutput = "{self.root / name}"\n')
        return ["startcurve", "--config", str(config), "--name", name]

    def _queue_file(self, names: tuple[str, ...] = ("first", "second"), pairs: str | None = None) -> dict:
        lines = ["name = \"check\"", pairs if pairs is not None else f"pairs = {json.dumps(self.ORDER)}"]
        for name in names:
            lines += ["", "[[items]]", f'name = "{name}"', "by_pair = true", f"run = {json.dumps(self._curve(name))}"]
        path = self.root / "pairs.toml"
        path.write_text("\n".join(lines) + "\n")
        return runqueue.load_queue(str(path))

    def _entry(self, queue: dict) -> dict:
        out = self.root / "queues" / "check"
        out.mkdir(parents=True, exist_ok=True)
        return {**queue, "kind": "queue", "output_dir": str(out), "created": time.time(), "session": "",
                "backend": "process", "log": str(out / "worker.log")}

    def _launcher(self, outcome: str = "completed") -> "_Launcher":
        """Like the real launch: the run ends with ``outcome`` and its records are on disk for the pair it was given."""
        test = self

        class Launcher(_Launcher):
            runs: list = []

            def __call__(self, name, run, resuming):
                code = super().__call__(name, run, resuming)
                self.runs.append((name, run))
                if outcome == "completed":
                    plan = sq.front_study(run[1:])
                    records = Path(plan["output"]) / sq.RESULTS_NAME
                    records.parent.mkdir(parents=True, exist_ok=True)
                    with records.open("a") as handle:
                        handle.writelines(json.dumps({"key": c["key"], "phase": "local"}) + "\n" for c in sq.conditions(plan))
                return code

        launcher = Launcher(test.root, outcome)
        launcher.runs = []
        return launcher

    def test_pair_spellings_read_alike(self):
        self.assertEqual([runqueue.pair_id(t) for t in ("dblp", "dblp:all", "dblp:top5000", "dblp/top5000", "dblp-top5000", "DBLP/All")],
                         ["dblp:all", "dblp:all", "dblp:top5000", "dblp:top5000", "dblp:top5000", "dblp:all"])

    def test_the_steps_go_pair_by_pair_in_the_queues_order(self):
        steps = runqueue.schedule(self._queue_file())
        self.assertEqual([(s["item"], s["pair"]) for s in steps],
                         [(name, pair) for pair in self.ORDER for name in ("first", "second")])

    def test_runs_that_are_not_by_pair_keep_their_place_around_the_block(self):
        queue = self._queue_file()
        queue["items"] = [{"name": "setup", "run": ["x"]}, *queue["items"], {"name": "closing", "run": ["x"]}]
        steps = runqueue.schedule(queue)
        self.assertEqual((steps[0]["item"], steps[0]["pair"]), ("setup", None))
        self.assertEqual((steps[-1]["item"], steps[-1]["pair"]), ("closing", None))
        self.assertEqual(len(steps), 2 + 2 * len(self.ORDER))
        self.assertEqual([s["index"] for s in steps[1:3]], [1, 2])  # indexes are the items' places in the queue

    def test_a_step_is_the_item_on_one_pair_spelled_as_its_own_config_spells_it(self):
        steps = {(s["item"], s["pair"]): s["run"] for s in runqueue.schedule(self._queue_file())}
        self.assertEqual(runqueue._option(steps["first", "dblp:top5000"], "--networks"), "dblp:top5000")
        self.assertEqual(runqueue._option(steps["first", "dblp:all"], "--networks"), "dblp")  # the config says "dblp"

    def test_the_steps_of_an_item_cut_its_conditions_by_pair_with_nothing_lost_or_repeated(self):
        queue = self._queue_file(("first",))
        whole = [c["key"] for c in sq.conditions(sq.front_study(queue["items"][0]["run"][1:]))]
        cut = [[c["key"] for c in sq.conditions(sq.front_study(s["run"][1:]))] for s in runqueue.schedule(queue)]
        self.assertEqual(len(whole), 16)
        self.assertEqual(sorted(key for part in cut for key in part), sorted(whole))
        self.assertEqual([len(part) for part in cut], [4, 4, 4, 4])

    def test_a_benchmark_unit_has_the_same_identity_in_its_pair_step_as_in_the_whole_item(self):
        run = ["exp", "--config", str(CONFIG_DIR / "paper-benchmark.toml"), "--name", "bench",
               "--output-dir", str(self.overlap / "bench"), "--yes"]
        step = runqueue.slice_run(run, "amazon:top5000", PAPER_PAIRS)
        whole = qs.config_from_args(qs.build_parser().parse_args(run[1:]))
        one = qs.config_from_args(qs.build_parser().parse_args(step[1:]))
        self.assertEqual(len(qs.build_plan(whole, {})), 8 * len(qs.build_plan(one, {})))  # 8 pairs, the same units each
        self.assertEqual(one.networks, ["amazon:top5000"])
        net, seed, method = qs.build_plan(one, {})[0]
        found = []
        for cfg in (whole, one):
            argv, out = qs.unit_argv(cfg, net, seed, method, None)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                cr.main([*argv, "--identity-only"])
            found.append((out, json.loads(buffer.getvalue().strip().splitlines()[-1])["config_digest"]))
        self.assertEqual(found[0], found[1])  # the same folder and the same digest: every record stays valid

    def test_a_spectrum_step_takes_the_pairs_up_to_its_own_and_its_records_do_not_depend_on_them(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        run = ["spectrum-full", "--name", "gt", "--output-dir", str(self.root / "gt")]
        step = runqueue.slice_run(run, "youtube:top5000", PAPER_PAIRS)
        self.assertEqual(runqueue._option(step, "--pairs"), "dblp/top5000,amazon/top5000,youtube/top5000")
        got = runqueue._spectrum_options("spectrum-full", step[1:])
        self.assertEqual([tuple(p) for p in got["pairs"]], [("dblp", "top5000"), ("amazon", "top5000"), ("youtube", "top5000")])
        whole = gs.implementation_identity(runqueue._spectrum_options("spectrum-full", run[1:]))
        self.assertEqual(gs._compact_identity(whole), gs._compact_identity(gs.implementation_identity(got)))

    def test_the_paper_queue_goes_pair_by_pair_with_the_top5000_covers_first(self):
        queue = runqueue.load_queue(QUEUE, root=str(self.root / "q"))
        self.assertEqual(queue["pairs"], PAPER_PAIRS)
        steps = runqueue.schedule(queue)
        self.assertEqual([s["item"] for s in steps[:3]], ["hedonic-calibration", "baseline-calibration", "synthetic-networks"])
        self.assertEqual([(s["item"], s["pair"]) for s in steps[3:]],
                         [(name, pair) for pair in PAPER_PAIRS for name in ("leveraging-gt", "gt-robustness", "benchmark")])
        # the first pair is finished in all three experiments before the second pair starts
        self.assertEqual([s["pair"] for s in steps[3:6]], ["dblp:top5000"] * 3)
        self.assertEqual(steps[-1]["pair"], "youtube:all")

    def test_the_worker_takes_the_steps_in_order_and_skips_a_pair_whose_files_are_complete(self):
        queue = self._queue_file()
        done = sq.front_study(runqueue.slice_run(queue["items"][0]["run"], "dblp:top5000", self.ORDER)[1:])
        Path(done["output"]).mkdir(parents=True)
        (Path(done["output"]) / sq.RESULTS_NAME).write_text(
            "".join(json.dumps({"key": c["key"], "phase": "local"}) + "\n" for c in sq.conditions(done)))
        launcher, prog = self._launcher(), {}
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", self._entry(queue), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertEqual([(name, runqueue._option(run, "--networks")) for name, run in launcher.runs],
                         [("second", "dblp:top5000"), ("first", "amazon:top5000"), ("second", "amazon:top5000"),
                          ("first", "dblp"), ("second", "dblp"), ("first", "amazon"), ("second", "amazon")])
        self.assertTrue(all(how == "start" for _name, how in launcher.calls))  # never resumed from the registry's old arguments
        self.assertEqual(prog["steps"][0]["note"], "finished earlier, skipped")
        self.assertEqual([row["status"] for row in prog["items"]], ["completed", "completed"])
        for item in queue["items"]:  # the pairs together are the whole item
            self.assertEqual(self._state(item["run"]), "complete")

    def test_a_failed_step_stops_the_queue_at_its_pair_and_says_so(self):
        queue, prog = self._queue_file(), {}
        entry = self._entry(queue)
        with contextlib.redirect_stdout(io.StringIO()):
            code = runqueue.work("check", entry, prog, launch_item=self._launcher("failed"))
        self.assertEqual(code, 1)
        self.assertEqual((prog["status"], prog["pair"], prog["current"]), ("failed", "dblp:top5000", 0))
        self.assertIn("first on dblp:top5000", prog["reason"])
        text = runqueue.tui.strip_ansi("\n".join(runqueue.stop_lines(entry, prog, "failed")))
        self.assertIn("failed at first on dblp:top5k", text)

    def test_a_run_already_going_is_followed_and_then_each_pair_is_judged_by_its_files(self):
        queue, prog = self._queue_file(("first",)), {}
        launcher = self._launcher()
        _register(self.root, "first", "completed")  # the run that was going, now ended
        real, going = runqueue._state, []

        def state(name):  # at the first step the run is found going (started before this queue); afterwards it has ended
            if name == "first" and prog.get("current") == 0 and not going:
                going.append(name)
                return "running", real(name)[1]
            return real(name)

        out = io.StringIO()
        with patch.object(runqueue, "_state", state), contextlib.redirect_stdout(out):
            code = runqueue.work("check", self._entry(queue), prog, launch_item=launcher)
        self.assertEqual(code, 0)
        self.assertIn("already running — following it, then judging dblp:top5000", out.getvalue())
        # once it ended, the pair was judged by its files (missing), so the item was started on it, restricted to the pair
        self.assertEqual(runqueue._option(launcher.runs[0][1], "--networks"), "dblp:top5000")
        self.assertEqual(len(launcher.runs), 4)

    def test_the_check_of_a_step_leaves_the_notes_of_the_whole_item_alone(self):
        queue = self._queue_file(("first",))
        run = queue["items"][0]["run"]
        self._state(run)
        folder = runqueue.data_state(run)[1]
        before = runqueue._COUNTS[folder]
        self.assertEqual(before[:2], (0, 16))
        self.assertEqual(runqueue.slice_state(runqueue.schedule(queue)[0]["run"], fresh=True)[0], "missing")
        self.assertEqual(runqueue._COUNTS[folder], before)  # the step's 4 conditions did not replace the item's 16

    def test_the_plan_names_what_each_step_would_do(self):
        queue = self._queue_file()
        done = sq.front_study(runqueue.slice_run(queue["items"][1]["run"], "amazon:top5000", self.ORDER)[1:])
        Path(done["output"]).mkdir(parents=True)
        (Path(done["output"]) / sq.RESULTS_NAME).write_text(
            "".join(json.dumps({"key": c["key"], "phase": "local"}) + "\n" for c in sq.conditions(done)))
        rows = runqueue.plan_steps(queue)
        self.assertEqual(rows[0], ("first", "dblp:top5000", "start"))
        self.assertIn(("second", "amazon:top5000", "skip"), rows)

    def test_a_spectrum_that_has_run_its_first_pairs_is_counted_over_all_of_the_items_pairs(self):
        from hedonic.experiments.overlapping import gt_spectrum as gs

        folder = self.root / "gt"
        folder.mkdir()
        run = ["spectrum-full", "--name", "gt", "--output-dir", str(folder)]
        options = runqueue._spectrum_options("spectrum-full", run[1:])
        per_pair = gs.conditions_per_job(options, "dblp", "top5000")
        plan = [{"job": "dblp-top5000", "dataset": "dblp", "cover": "top5000", "policy": policy, "gamma": setting["gamma"],
                 "seed": seed, "status": "completed"}
                for policy in options["isolation_policies"] for setting in gs.job_settings(options, "dblp", "top5000")
                for seed in options["detector_seeds"]]
        self.assertEqual(len(plan), per_pair)
        (folder / "progress.json").write_text(json.dumps({"jobs": ["dblp/top5000"], "plan": plan,
                                                          "tracked_files_sha256": gs.tracked_digest()}))
        self._state(run)
        self.assertEqual(runqueue.unit_counts(run), (per_pair, 8 * per_pair, "conditions"))
        self.assertIn(runqueue.data_state(run)[1], runqueue._MODEL)  # the view keeps its breakdown with only the first pair run
        # a pair the last coverage report lists as complete is finished at once, before the study has reloaded it
        rows = runqueue._stepped_plan("spectrum-full", run[1:], [{**row, "status": "pending"} for row in plan],
                                      [f"{d}-{c}" for d, c in options["pairs"]], {"dblp-top5000"})
        self.assertEqual((len(rows), sum(rm.unit_finished(row) for row in rows)), (8 * per_pair, per_pair))

    def test_the_queue_file_is_checked_for_its_pairs(self):
        with self.assertRaises(SystemExit):  # by pair, but no pairs
            self._queue_file(pairs="")
        with self.assertRaises(SystemExit):  # the items cover pairs the queue does not list
            self._queue_file(pairs='pairs = ["dblp:top5000", "amazon:top5000"]')
        with self.assertRaises(SystemExit):  # a pair listed twice
            self._queue_file(pairs='pairs = ["dblp:top5000", "dblp:top5000", "amazon:top5000", "dblp", "amazon"]')
        with self.assertRaises(SystemExit):  # not a pair
            self._queue_file(pairs='pairs = ["dblp:nonsense"]')

    def test_the_by_pair_items_must_stand_next_to_each_other(self):
        text = "\n".join(['name = "check"', f"pairs = {json.dumps(self.ORDER)}"]
                         + [line for name, by in (("a", True), ("b", False), ("c", True))
                            for line in ("", "[[items]]", f'name = "{name}"', *(["by_pair = true"] if by else []),
                                         f"run = {json.dumps(self._curve(name))}")]) + "\n"
        path = self.root / "gap.toml"
        path.write_text(text)
        with self.assertRaises(SystemExit):
            runqueue.load_queue(str(path))


if __name__ == "__main__":
    unittest.main()
