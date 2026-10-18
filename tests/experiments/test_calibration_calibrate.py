"""The Hedonic calibration run (``hedonic run calibrate``): plan, results file, data provisioning, the lazy runner, the
core budget, the packaged search end to end and the queue's view of a calibration.

Data-free: the searches are stand-in scripts (``--script``), or the packaged search on a planted instance through a
stand-in interpreter that registers it.
"""

from __future__ import annotations

import io
import json
import re
import os
import stat
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import calibrate as cal
from hedonic.experiments.overlapping import parallel
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runqueue

# A stand-in for the search: it logs two probes and appends the search record the real search writes.
FAKE_SCRIPT = textwrap.dedent('''
    import json, sys
    from pathlib import Path
    a = sys.argv
    get = lambda flag: a[a.index(flag) + 1]
    net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"  [{net} {phase} M=7 gamma=0.5 (c=1, t=0.1) seed=0] completed det=1.0s F1=0.4 ONMI=0.5", flush=True)
    print(f"  [{net} {phase} M=7 gamma=0.25 (c=1, t=0.1) seed=0] completed det=1.0s F1=0.5 ONMI=0.5", flush=True)
    record = {"kind": "tune", "instance": {"spec": net}, "phase": phase,
              "snap": {"fraction": "1/8", "kept": True},
              "final": {"gamma": 0.125, "M": 6, "f1_median": 0.51, "onmi_median": 0.6, "f1_sd": 0.001,
                        "confirm_seeds": [0, 1]},
              "detector_evaluations": 9, "wall_hours": 0.01, "argv": a[1:]}
    with (out / "searches.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\\n")
''')

# The same, slower, leaving the span of its run so a test can see which searches overlapped.
FAKE_SLOW_SCRIPT = textwrap.dedent('''
    import json, sys, time
    from pathlib import Path
    a = sys.argv
    get = lambda flag: a[a.index(flag) + 1]
    net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))
    out.mkdir(parents=True, exist_ok=True)
    mark = out / f"{net}-{phase}.span"
    start = time.time()
    time.sleep(0.8)
    mark.write_text(json.dumps([start, time.time(), a[1:]]))
    record = {"kind": "tune", "instance": {"spec": net}, "phase": phase,
              "snap": {"fraction": "1/8", "kept": True},
              "final": {"gamma": 0.125, "M": 6, "f1_median": 0.5, "onmi_median": 0.6, "f1_sd": 0.0,
                        "confirm_seeds": [0]},
              "detector_evaluations": 1, "wall_hours": 0.0}
    with (out / "searches.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\\n")
''')

# A stand-in with the parts of the search that lazy_search touches: a memoizing Probe, the records' fields and a main().
LAZY_SCRIPT = textwrap.dedent('''
    import json, sys, math
    from pathlib import Path
    CACHE, INST, SEARCHES, EVALS = {}, {}, None, None
    def _normal(cfg): return {**cfg, "gamma": float(f"{cfg['gamma']:.6g}")}
    def cfg_key(spec, cfg): return json.dumps({"instance": spec, **_normal(cfg)}, sort_keys=True)
    def append_line(path, line):
        with Path(path).open("a") as fh: fh.write(line + "\\n")
    def instance_summary(inst): return {k: inst[k] for k in ("spec", "rho", "rho_in")}
    class Probe:
        def __init__(self, spec, base, seeds): self.spec, self.base, self.seeds = spec, base, seeds
        def __call__(self, **over):
            for s in self.seeds:
                cfg = _normal({**self.base, **over, "seed": s}); key = cfg_key(self.spec, cfg)
                if key not in CACHE:  # f1 peaks at gamma 0.25; each evaluation "costs" 60 s
                    f1 = 0.6 - abs(cfg["gamma"] - 0.25)
                    CACHE[key] = {"status": "completed", "config": cfg, "metrics": {"f1": f1, "onmi": f1 / 2},
                                  "detection_seconds": 10.0, "scoring_seconds": 30.0, "audit_seconds": 20.0}
            return 0.0
        def prefetch(self, overs): pass
    def main():
        out = Path(sys.argv[sys.argv.index("--out") + 1]); global SEARCHES; SEARCHES = out / "searches.jsonl"
        out.mkdir(parents=True, exist_ok=True)
        spec = sys.argv[sys.argv.index("--instance") + 1]
        INST[spec] = {"spec": spec, "rho": 1e-4, "rho_in": 0.5}
        probe = Probe(spec, {"phase": "local", "M": 4}, [0])
        for g in (0.5, 0.125, 0.25, 0.2, 0.3, 0.1, 0.4, 0.05):
            probe(gamma=g); probe(gamma=g)  # a repeat is free
        append_line(SEARCHES, json.dumps({"kind": "tune", "instance": {"spec": spec}, "phase": "local", "final": {"gamma": 0.25, "M": 4}}))
        return 0
''')

# A stand-in interpreter for the packaged search: it registers a planted instance under the requested name (no SNAP
# data is read) and runs the module it is asked to run, exactly as ``python -m`` would.
FAKE_PYTHON = textwrap.dedent('''\
    #!{python}
    import random, runpy, sys
    import igraph as ig
    from hedonic.experiments.overlapping import calibration_search as cs

    def planted(n_comm=6, size=12, overlap=2, p_in=0.7, p_out=0.02, seed=0):
        rng = random.Random(seed)
        step = size - overlap
        n = n_comm * step
        comms = [[(i * step + j) % n for j in range(size)] for i in range(n_comm)]
        member = [set() for _ in range(n)]
        for c, vs in enumerate(comms):
            for v in vs:
                member[v].add(c)
        edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < (p_in if member[u] & member[v] else p_out)]
        return ig.Graph(n=n, edges=edges), comms

    assert sys.argv[1] == "-m", sys.argv
    module, rest = sys.argv[2], sys.argv[3:]
    graph, gt = planted()
    cs.register_instance(rest[rest.index("--instance") + 1], graph, gt)
    if module == "hedonic.experiments.overlapping.lazy_search":
        from hedonic.experiments.overlapping import lazy_search
        sys.exit(lazy_search.main(rest))
    assert module == cs.__name__, module
    sys.exit(cs.main(rest))
''')


def plan_for(tmp: str, *extra: str) -> dict:
    script = Path(tmp) / "fake_tune.py"
    script.write_text(FAKE_SCRIPT)
    plan = cal.resolve(["--script", str(script), "--output-dir", str(Path(tmp) / "cal"), "--name", "cal-test", *extra])
    plan["python"] = sys.executable
    return plan


def packaged_plan(tmp: str, *extra: str) -> dict:
    return cal.resolve(["--output-dir", str(Path(tmp) / "cal"), "--name", "cal-test", *extra])


def fake_python(tmp: str) -> str:
    path = Path(tmp) / "fakepython"
    path.write_text(FAKE_PYTHON.replace("{python}", sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class TestCoverTargets(unittest.TestCase):
    """Each (network, ground-truth cover) pair is calibrated on its covered-induced graph and keeps its own setting."""

    def plan(self, tmp: str, *extra: str) -> dict:
        return plan_for(tmp, "--networks", "dblp,amazon,youtube", "--covers", "all,top5000", "--phases", "local",
                        "--mode", "full", *extra)

    def test_networks_crossed_with_covers_are_searched_one_target_each(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp)
            self.assertEqual([s["key"] for s in cal.searches(plan)],
                             [f"hedonic/{net}:{cover}" for net in ("dblp", "amazon", "youtube")
                              for cover in ("all", "top5000")])
            argv = cal.search_argv(plan, cal.searches(plan)[1])
            self.assertEqual(argv[argv.index("--instance") + 1], "dblp:top5000")
            self.assertNotIn(":", argv[argv.index("--tag") + 1])  # the ledger file name

    def test_a_pair_starts_from_its_networks_setting_and_a_network_without_one_from_dblps(self):
        self.assertEqual(cal.start_for("hedonic", "amazon:top5000")["max_memberships"],
                         qs.HOC_TUNED["hedonic"]["amazon"]["max_memberships"])
        youtube = cal.start_for("hedonic", "youtube:all")
        self.assertEqual(youtube["max_memberships"], qs.HOC_TUNED["hedonic"]["dblp"]["max_memberships"])
        self.assertIn("transfer", youtube["source"])
        self.assertEqual(cal.start_for("hedonic", "dblp")["source"], "dblp")

    def test_a_cover_the_network_lacks_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                plan_for(tmp, "--networks", "wikipedia", "--covers", "top5000")

    def test_the_calibration_writes_one_setting_per_target_and_tuned_lays_them_over_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "1")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cal.run_calibration(plan, emit=lambda _: None), 0)
            data = json.loads((plan["output"] / cal.TUNED_NAME).read_text())
            self.assertTrue(data["complete"])
            self.assertEqual(set(data["settings"]["hedonic"]),
                             {f"{n}:{c}" for n in ("dblp", "amazon", "youtube") for c in ("all", "top5000")})
            self.assertEqual(data["searches"]["hedonic/youtube:all"]["start_source"], "dblp (transfer)")
            cfg = qs.config_from_args(qs.build_parser().parse_args(
                ["--networks", "dblp,amazon", "--covers", "all,top5000", "--induce", "covered", "--methods", "hedonic",
                 "--tuned", str(plan["output"] / cal.TUNED_NAME)]))
            for target in cfg.networks:
                params = qs.method_params(qs.BY_KEY["hedonic"], cfg, target)
                self.assertEqual(params["max_memberships"], 6)
                self.assertEqual(params["absolute_resolution"], 0.125)
            self.assertEqual(plan_for(tmp, "--networks", "dblp:top5000")["networks"], ["dblp:top5000"])


class TestPlan(unittest.TestCase):
    def test_searches_run_local_and_a_warm_start_brackets_the_recorded_optimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp)
            self.assertEqual([s["key"] for s in cal.searches(plan)], ["hedonic/dblp", "hedonic/amazon"])
            argv = cal.search_argv(plan, cal.searches(plan)[1])
            value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
            gamma = qs.HOC_TUNED["hedonic"]["amazon"]["absolute_resolution"]
            lo, hi = map(float, value("--bracket").split(","))
            self.assertAlmostEqual(lo, gamma / 1.5, places=8)
            self.assertAlmostEqual(hi, gamma * 1.5, places=8)
            self.assertEqual(value("--M-start"), str(qs.HOC_TUNED["hedonic"]["amazon"]["max_memberships"]))
            self.assertIn("--no-extremal", argv)
            self.assertEqual(value("--out"), str(plan["output"] / "ledger"))  # this run's own ledger
            self.assertEqual(value("--phase"), "local")

    def test_a_full_calibration_uses_the_searchs_own_bracket_and_multilevel_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--mode", "full")
            local = cal.search_argv(plan, cal.searches(plan)[0])
            self.assertNotIn("--bracket", local)
            self.assertNotIn("--M-start", local)
            self.assertNotIn("--no-extremal", local)
            for phases in ("multilevel", "local,multilevel"):
                with self.assertRaisesRegex(ValueError, "not available in hedonic 2.0"):
                    plan_for(tmp, "--phases", phases)

    def test_bad_arguments_are_refused_and_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (["--phases", "weird"], ["--networks", "nosuchnet"], ["--factor", "1"], ["--workers", "0"],
                        ["--budget", "0"], ["--cap-minutes", "60"]):
                with self.assertRaises(ValueError, msg=bad):
                    plan_for(tmp, *bad)
            out = Path(tmp) / "cal"
            buffer = io.StringIO()
            with patch.object(cal, "run_calibration", side_effect=AssertionError("dry run ran")), \
                    patch.object(cal, "provision", side_effect=AssertionError("dry run downloaded")), redirect_stdout(buffer):
                code = cal.main(["--output-dir", str(out), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            self.assertIn("hedonic/dblp", buffer.getvalue())
            self.assertIn(f"-m {cal.SEARCH_MODULE} tune", buffer.getvalue())

    def test_launch_argv_rebuilds_the_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            for plan in (plan_for(tmp, "--factor", "2", "--confirm-seeds", "0-2"),
                         packaged_plan(tmp, "--network-root", str(Path(tmp) / "nets"), "--snap-cache-dir",
                                       str(Path(tmp) / "cache"), "--no-download", "--budget", "5", "--cap-minutes", "7"),
                         packaged_plan(tmp)):
                again = cal.resolve(cal.launch_argv(plan))
                self.assertEqual(again, plan)

    def test_the_packaged_search_is_the_default_and_gets_the_data_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = packaged_plan(tmp, "--networks", "dblp:top5000", "--mode", "full", "--workers", "1",
                                 "--network-root", str(Path(tmp) / "nets"), "--no-download")
            self.assertIsNone(plan["script"])
            self.assertEqual(plan["python"], sys.executable)
            argv = cal.search_argv(plan, cal.searches(plan)[0])
            self.assertEqual(argv[:4], [sys.executable, "-m", cal.SEARCH_MODULE, "tune"])
            self.assertEqual(argv[argv.index("--network-root") + 1], str((Path(tmp) / "nets").resolve()))
            self.assertIn("--no-download", argv)
            self.assertNotIn("--snap-cache-dir", argv)
            scripted = plan_for(tmp, "--network-root", str(Path(tmp) / "nets"))  # a script has no such flags
            self.assertNotIn("--network-root", cal.search_argv(scripted, cal.searches(scripted)[0]))
            text = "\n".join(cal.plan_lines(plan))
            self.assertIn(f"search   {cal.SEARCH_MODULE} (packaged)", text)
            self.assertIn("(no downloads)", text)

    def test_the_default_workers_are_the_performance_cores(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(packaged_plan(tmp)["workers"], parallel.default_workers())
            with patch.dict(os.environ, {"HEDONIC_WORKERS": "3"}):
                self.assertEqual(packaged_plan(tmp)["workers"], 3)
            self.assertEqual(packaged_plan(tmp, "--workers", "2")["workers"], 2)

    def test_the_plan_says_which_pairs_are_on_disk_and_what_would_be_downloaded(self):
        from hedonic.experiments.overlapping import snap

        def fake_downloads(network, cover=None, root=None):
            return [] if network == "dblp" else [("https://snap.example/x", Path("/r") / f"com-{network}.ungraph.txt.gz")]

        with tempfile.TemporaryDirectory() as tmp, patch.object(snap, "pair_downloads", fake_downloads):
            plan = packaged_plan(tmp, "--networks", "dblp:top5000,amazon:top5000", "--mode", "full")
            self.assertEqual(cal.data_status(plan), {"dblp:top5000": "on disk",
                                                     "amazon:top5000": "to download: com-amazon.ungraph.txt.gz"})
            self.assertIn("[data to download: com-amazon.ungraph.txt.gz]", "\n".join(cal.plan_lines(plan)))
            plan["download"] = False
            self.assertTrue(cal.data_status(plan)["amazon:top5000"].startswith("MISSING (--no-download)"))


class TestProvision(unittest.TestCase):
    def test_every_target_is_put_on_disk_once_before_the_packaged_searches(self):
        from hedonic.experiments.overlapping import snap

        calls: list[tuple] = []

        def fake_ensure(network, cover=None, root=None, *, download=True, say=None, **_):
            calls.append((network, cover, root, download))
            if say:
                say(f"{network}:{cover} ok")

        with tempfile.TemporaryDirectory() as tmp, patch.object(snap, "ensure_pair", fake_ensure):
            plan = packaged_plan(tmp, "--networks", "dblp:top5000,amazon,dblp:top5000", "--network-root", tmp,
                                 "--no-download")
            lines: list[str] = []
            cal.provision(plan, lines.append)
            self.assertEqual(calls, [("dblp", "top5000", Path(tmp).resolve(), False), ("amazon", "all", Path(tmp).resolve(), False)])
            self.assertTrue(all(line.startswith("hoc calibration: ") for line in lines))

            def refuse(*_a, **_k):
                raise snap.SnapLoadError("not on disk")

            with patch.object(snap, "ensure_pair", refuse), self.assertRaisesRegex(RuntimeError, "dblp:top5000"):
                cal.provision(plan, lines.append)

    def test_the_run_provisions_only_pending_targets_and_only_for_the_packaged_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            provisioned: list[list[str]] = []
            summary = {"final": {"gamma": 0.125, "M": 6, "f1_median": 0.5, "onmi_median": 0.5, "confirm_seeds": [0]},
                       "snap": {"fraction": "1/8", "kept": True}, "detector_evaluations": 3, "wall_hours": 0.0}
            plan = packaged_plan(tmp, "--networks", "dblp:top5000,amazon:top5000", "--mode", "full", "--workers", "1")
            with patch.object(cal, "provision", lambda p, emit: provisioned.append(list(p["networks"]))), \
                    patch.object(cal, "run_search", lambda *a, **k: None), patch.object(cal, "final_of", lambda *a: summary):
                self.assertEqual(cal.run_calibration(plan, emit=lambda _: None), 0)
                self.assertEqual(provisioned, [["dblp:top5000", "amazon:top5000"]])
                self.assertEqual(cal.run_calibration(plan, emit=lambda _: None), 0)  # nothing pending: nothing fetched
                self.assertEqual(len(provisioned), 1)
            scripted = plan_for(tmp, "--networks", "dblp:top5000", "--workers", "1")
            with patch.object(cal, "provision", side_effect=AssertionError("a script loads its own data")):
                self.assertEqual(cal.run_calibration(scripted, emit=lambda _: None), 0)


class TestResults(unittest.TestCase):
    def test_results_resume_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--workers", "1")
            events: list[dict] = []
            lines: list[str] = []
            self.assertEqual(cal.state_of(plan), "missing")
            self.assertEqual(cal.run_calibration(plan, progress=events.append, emit=lines.append), 0)
            data = json.loads((plan["output"] / cal.TUNED_NAME).read_text())
            self.assertTrue(data["complete"])
            self.assertEqual(data["settings"]["hedonic"]["dblp"], {"max_memberships": 6, "absolute_resolution": 0.125})
            entry = data["searches"]["hedonic/dblp"]
            self.assertTrue(entry["changed"])
            self.assertEqual(entry["start"]["max_memberships"], qs.HOC_TUNED["hedonic"]["dblp"]["max_memberships"])
            self.assertEqual(cal.state_of(plan), "complete")
            self.assertEqual([e["done"] for e in events if e["type"] == "unit"], [1, 2])
            self.assertTrue(any(e["type"] == "stage" and "probe 2" in e["text"] for e in events))

            # a rerun runs no search; a partial file resumes only the missing searches
            lines.clear()
            self.assertEqual(cal.run_calibration(plan, emit=lines.append), 0)
            self.assertFalse(any(line.lstrip().startswith("$") for line in lines))
            data["searches"].pop("hedonic/amazon")
            data["complete"] = False
            (plan["output"] / cal.TUNED_NAME).write_text(json.dumps(data))
            self.assertEqual(cal.state_of(plan), "partial")
            lines.clear()
            cal.run_calibration(plan, emit=lines.append)
            self.assertEqual(sum(line.lstrip().startswith("$") for line in lines), 1)
            self.assertEqual(cal.state_of(plan), "complete")

    def test_the_file_names_this_build_and_no_pinned_igraph(self):
        from hedonic.experiments import runtime

        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--networks", "dblp", "--workers", "1")
            summary = {"final": {"gamma": 0.125, "M": 6, "f1_median": 0.5, "onmi_median": 0.5, "confirm_seeds": [0]},
                       "snap": None, "detector_evaluations": 3, "wall_hours": 0.0,
                       "build": {"hedonic": runtime.hedonic_version(), "hedonic_core_sha256": runtime.core_sha256()}}
            data = cal.record_search(plan, cal.searches(plan)[0], summary)
            self.assertEqual(data["algorithm_identity"], runtime.ALGORITHM_IDENTITY)
            self.assertEqual(data["environment"]["hedonic_core_sha256"], runtime.core_sha256())
            self.assertEqual(data["searches"]["hedonic/dblp"]["build"], summary["build"])
            self.assertEqual(data["search"], str(plan["script"]))
            text = (plan["output"] / cal.TUNED_NAME).read_text()
            for stale in ("lucas-igraph", "lucas_igraph", "required_igraph_c", "from_1_0_0_5"):
                self.assertNotIn(stale, text)
            self.assertEqual(cal.record_search(packaged_plan(tmp, "--networks", "dblp"), cal.searches(plan)[0],
                                               summary)["search"], cal.SEARCH_MODULE)

    def test_a_file_written_before_the_rename_is_read_as_hedonics(self):
        for build in (True, False):  # a search of this build is kept; one of 1.x (no build) is searched again
            with self.subTest(build=build), tempfile.TemporaryDirectory() as tmp:
                plan = plan_for(tmp, "--networks", "dblp,amazon")
                plan["output"].mkdir(parents=True)
                stamp = {"build": cal.current_build()} if build else {}
                (plan["output"] / cal.TUNED_NAME).write_text(json.dumps({
                    "complete": False,
                    "settings": {"hoc_local": {"dblp": {"max_memberships": 9, "absolute_resolution": 0.1}},
                                 "hoc_multilevel": {"dblp": {"max_memberships": 3, "absolute_resolution": 0.5}}},
                    "searches": {"hoc_local/dblp": {"max_memberships": 9, "absolute_resolution": 0.1, **stamp},
                                 "hoc_multilevel/dblp": {"max_memberships": 3, "absolute_resolution": 0.5, **stamp}}}))
                self.assertEqual(cal.recorded(plan), {"hedonic/dblp"} if build else set())
                self.assertEqual(cal.other_build(plan), set() if build else {"hedonic/dblp"})
                self.assertEqual(cal.read_tuned(plan)["settings"], {"hedonic": {"dblp": {"max_memberships": 9,
                                                                                        "absolute_resolution": 0.1}}})
                self.assertEqual(cal.state_of(plan), "partial" if build else "missing")
                lines: list[str] = []
                self.assertEqual(cal.run_calibration(plan, emit=lines.append), 0)
                # One command line per search started (side by side, each line carries its search's prefix).
                commands = [line for line in lines if re.match(r"\s*(\[[^\]]+\] )?\s*\$", line)]
                self.assertEqual(len(commands), 1 if build else 2)
                data = json.loads((plan["output"] / cal.TUNED_NAME).read_text())
                self.assertEqual(set(data["settings"]), {"hedonic"})
                self.assertEqual(set(data["searches"]), {"hedonic/dblp", "hedonic/amazon"})
                self.assertTrue(data["complete"])
                self.assertEqual(cal.recorded(plan), {"hedonic/dblp", "hedonic/amazon"})

    def test_an_unchanged_optimum_keeps_the_exact_start_value_and_an_unkept_fraction_is_not_reported(self):
        # A search that comes back to its start reports the probes' 6-digit value (0.142857 for 1/7); that is the same
        # configuration, so the setting must stay exactly the start (records made at it keep their identity).
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--networks", "dblp")
            search = cal.searches(plan)[0]
            start = qs.HOC_TUNED["hedonic"]["dblp"]
            summary = {"snap": {"fraction": "1/5", "kept": False},
                       "final": {"gamma": float(f"{start['absolute_resolution']:.6g}"), "M": int(start["max_memberships"]),
                                 "f1_median": 0.42, "onmi_median": 0.5, "f1_sd": 0.0, "confirm_seeds": [0]},
                       "detector_evaluations": 5, "wall_hours": 0.1}
            entry = cal.record_search(plan, search, summary)["searches"]["hedonic/dblp"]
            self.assertEqual(entry["absolute_resolution"], float(start["absolute_resolution"]))
            self.assertIsNone(entry["fraction"])
            self.assertFalse(entry["changed"])
            summary["final"]["gamma"] = 0.1612345678  # a different optimum: kept at the digits the search evaluated
            entry = cal.record_search(plan, search, summary)["searches"]["hedonic/dblp"]
            self.assertEqual(entry["absolute_resolution"], 0.161235)
            self.assertTrue(entry["changed"])
            summary["snap"] = {"fraction": "1/6", "kept": True}  # a kept fraction is the setting, exactly
            entry = cal.record_search(plan, search, summary)["searches"]["hedonic/dblp"]
            self.assertEqual((entry["absolute_resolution"], entry["fraction"]), (1 / 6, "1/6"))

    def test_a_failing_search_stops_the_run_and_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--workers", "1")
            Path(plan["script"]).write_text("import sys; sys.exit(3)")
            with self.assertRaises(RuntimeError):
                cal.run_calibration(plan, emit=lambda _: None)
            self.assertEqual(cal.state_of(plan), "missing")
            plan = plan_for(tmp)
            Path(plan["script"]).unlink()
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                cal.run_calibration(plan, emit=lambda _: None)


class TestPackagedSearchEndToEnd(unittest.TestCase):
    """The real search, run as ``python -m`` by the calibration, on a planted instance standing in for dblp:top5000."""

    def test_a_calibration_with_the_packaged_search_writes_a_complete_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = packaged_plan(tmp, "--networks", "dblp:top5000", "--mode", "full", "--workers", "2",
                                 "--confirm-seeds", "0,1")
            plan["python"] = fake_python(tmp)
            lines: list[str] = []
            with patch.object(cal, "provision", lambda *a: None):
                self.assertEqual(cal.run_calibration(plan, emit=lines.append), 0)
            data = cal.read_tuned(plan)
            self.assertTrue(data["complete"])
            entry = data["searches"]["hedonic/dblp:top5000"]
            self.assertGreater(entry["f1_median"], 0.8)
            self.assertEqual(entry["confirm_seeds"], [0, 1])
            self.assertGreater(entry["detector_evaluations"], 5)
            self.assertEqual(entry["mode"], "full")
            self.assertEqual(entry["build"]["hedonic_core_source_sha256"], data["environment"]["hedonic_core_source_sha256"])
            self.assertEqual(data["settings"]["hedonic"]["dblp:top5000"],
                             {"max_memberships": entry["max_memberships"], "absolute_resolution": entry["absolute_resolution"]})
            self.assertTrue(any("[dblp:top5000 local M=" in line for line in lines))  # the probe lines reach the log
            ledger = plan["output"] / "ledger"
            self.assertTrue((ledger / "evals-dblp_top5000-local.jsonl").is_file())
            self.assertTrue((ledger / "slots").is_dir())

    def test_a_lazy_calibration_with_the_packaged_search_keeps_its_budget_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = packaged_plan(tmp, "--networks", "dblp:top5000", "--mode", "full", "--workers", "1", "--budget", "3")
            plan["python"] = fake_python(tmp)
            with patch.object(cal, "provision", lambda *a: None):
                self.assertEqual(cal.run_calibration(plan, emit=lambda _: None), 0)
            entry = cal.read_tuned(plan)["searches"]["hedonic/dblp:top5000"]
            self.assertEqual((entry["lazy"]["stopped_by"], entry["lazy"]["evaluations"]), ("budget", 3))
            self.assertEqual(entry["detector_evaluations"], 3)


class TestLazy(unittest.TestCase):
    """A budgeted search stops before it asks for one more evaluation and keeps the best observed setting."""

    def run_lazy(self, tmp: str, *flags: str) -> dict:
        from hedonic.experiments.overlapping import lazy_search

        script = Path(tmp) / "fake_bt.py"
        script.write_text(LAZY_SCRIPT)
        out = Path(tmp) / "ledger"
        argv = ["--script", str(script), *flags, "tune", "--instance", "x:top5000", "--phase", "local", "--out", str(out)]
        saved = list(sys.argv)
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(lazy_search.main(argv), 0)
        finally:
            sys.argv[:] = saved
            sys.modules.pop("bisect_tuning", None)
            if str(script.parent) in sys.path:
                sys.path.remove(str(script.parent))
        return json.loads((out / "searches.jsonl").read_text().splitlines()[-1])

    def test_the_evaluation_budget_stops_the_search_and_keeps_the_best_observed_setting(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.run_lazy(tmp, "--budget", "4")
            self.assertEqual([a["gamma"] for a in record["lazy"]["asked"]], [0.5, 0.125, 0.25, 0.2])
            self.assertEqual(record["lazy"]["stopped_by"], "budget")
            self.assertEqual(record["final"]["gamma"], 0.25)  # F1 peaks there: best observed, not the last
            self.assertEqual(record["detector_evaluations"], 4)
            self.assertAlmostEqual(record["lazy"]["probe_seconds"], 240.0)

    def test_the_cost_cap_stops_before_a_new_evaluation_once_it_is_spent(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.run_lazy(tmp, "--budget", "8", "--cap-minutes", "3")  # 60 s per evaluation: three, then stop
            self.assertEqual(record["lazy"]["evaluations"], 3)
            self.assertEqual(record["lazy"]["stopped_by"], "cap")

    def test_a_search_that_ends_inside_the_budget_is_the_scripts_own_full_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertNotIn("lazy", self.run_lazy(tmp, "--budget", "20"))

    def test_the_plan_runs_the_lazy_runner_and_records_the_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--networks", "dblp:top5000", "--mode", "full", "--budget", "28", "--cap-minutes", "60",
                            "--workers", "1")
            argv = cal.search_argv(plan, cal.searches(plan)[0])
            self.assertEqual(argv[1:3], ["-m", cal.LAZY_MODULE])
            self.assertEqual(argv[argv.index("--script") + 1], str(plan["script"]))
            self.assertEqual(argv[argv.index("--budget") + 1], "28")
            self.assertEqual(argv[argv.index("--cap-minutes") + 1], "60")
            self.assertEqual(argv[argv.index("tune") + 1:argv.index("tune") + 3], ["--instance", "dblp:top5000"])
            self.assertIn("lazy", "\n".join(cal.plan_lines(plan)))
            again = cal.resolve(cal.launch_argv(plan))  # a resume rebuilds the same plan
            self.assertEqual((again["budget"], again["cap_minutes"]), (28, 60.0))
            packaged = packaged_plan(tmp, "--budget", "5")
            argv = cal.search_argv(packaged, cal.searches(packaged)[0])
            self.assertEqual(argv[1:5], ["-m", cal.LAZY_MODULE, "--budget", "5"])
            self.assertNotIn("--script", argv)
            self.assertNotIn(cal.LAZY_MODULE, cal.search_argv(plan_for(tmp), cal.searches(plan_for(tmp))[0]))

    def test_a_lazy_result_is_written_with_its_budget_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = plan_for(tmp, "--networks", "dblp:top5000", "--mode", "full", "--budget", "4", "--workers", "1")
            summary = {"final": {"gamma": 0.25, "M": 4, "f1_median": 0.6, "onmi_median": 0.3, "confirm_seeds": [0]},
                       "snap": None, "detector_evaluations": 4, "wall_hours": 0.1,
                       "lazy": {"budget": 4, "cap_minutes": None, "stopped_by": "budget", "stopped_after": "4 evaluations",
                                "evaluations": 4, "probe_seconds": 240.0, "detection_seconds": 40.0, "selection": "x"}}
            data = cal.record_search(plan, cal.searches(plan)[0], summary)
            self.assertEqual(data["searches"]["hedonic/dblp:top5000"]["lazy"]["stopped_by"], "budget")
            self.assertEqual(data["settings"]["hedonic"]["dblp:top5000"]["absolute_resolution"], 0.25)


class TestAutoMode(unittest.TestCase):
    def plan(self, mode: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            return plan_for(tmp, "--networks", "dblp,amazon,youtube,livejournal", "--covers", "all,top5000", "--mode", mode,
                            "--workers", "2")

    def test_auto_warms_only_the_all_covers_with_their_own_optimum(self):
        plan = self.plan("auto")
        modes = {s["network"]: cal.search_mode(plan, s["network"], s["method"]) for s in cal.searches(plan)}
        self.assertEqual(modes, {"dblp:all": "warm", "dblp:top5000": "full", "amazon:all": "warm", "amazon:top5000": "full",
                                 "youtube:all": "full", "youtube:top5000": "full", "livejournal:all": "full",
                                 "livejournal:top5000": "full"})
        self.assertEqual(cal.search_mode(plan, "dblp", "hedonic"), "warm")  # the whole graph

    def test_the_command_of_a_warm_search_has_a_bracket_and_a_full_one_has_none(self):
        plan = self.plan("auto")
        by = {s["network"]: cal.search_argv(plan, s) for s in cal.searches(plan)}
        self.assertIn("--bracket", by["dblp:all"])
        self.assertIn("--M-start", by["amazon:all"])
        self.assertNotIn("--bracket", by["dblp:top5000"])
        self.assertNotIn("--bracket", by["youtube:all"])

    def test_the_other_modes_are_unchanged(self):
        for mode in ("warm", "full"):
            plan = self.plan(mode)
            self.assertEqual({cal.search_mode(plan, s["network"], s["method"]) for s in cal.searches(plan)}, {mode})


class TestParallelCalibration(unittest.TestCase):
    def plan(self, tmp: str, *extra: str) -> dict:
        script = Path(tmp) / "slow_tune.py"
        script.write_text(FAKE_SLOW_SCRIPT)
        plan = cal.resolve(["--script", str(script), "--output-dir", str(Path(tmp) / "cal"), "--name", "cal-par", *extra])
        plan["python"] = sys.executable
        return plan

    def test_workers_share_one_budget_and_are_replayed_by_a_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4")
            self.assertEqual(plan["workers"], 4)
            for search in cal.searches(plan):
                argv = cal.search_argv(plan, search)
                self.assertEqual(argv[argv.index("--jobs") + 1], "4")
                self.assertEqual(argv[argv.index("--slot-count") + 1], "4")
                self.assertEqual(argv[argv.index("--slot-dir") + 1], str(plan["output"] / "ledger" / "slots"))
            self.assertEqual(cal.resolve(cal.launch_argv(plan))["workers"], 4)
            one = self.plan(tmp, "--workers", "1")
            self.assertNotIn("--jobs", cal.search_argv(one, cal.searches(one)[0]))

    def test_the_searches_run_side_by_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4", "--networks", "dblp,amazon,youtube,livejournal")
            lines: list[str] = []
            events: list[dict] = []
            started = time.time()
            self.assertEqual(cal.run_calibration(plan, emit=lines.append, progress=events.append), 0)
            elapsed = time.time() - started
            self.assertEqual(cal.state_of(plan), "complete")
            spans = [json.loads(path.read_text()) for path in (plan["output"] / "ledger").glob("*.span")]
            self.assertEqual(len(spans), 4)
            self.assertLess(max(s[0] for s in spans), min(s[1] for s in spans))  # all four were alive at the same moment
            self.assertLess(elapsed, 4 * 0.8)  # one after another would take at least 3.2 s
            self.assertTrue(any(line.startswith("[hedonic/dblp] ") for line in lines))  # interleaved lines say whose
            self.assertEqual(sorted(e["key"] for e in events if e["type"] == "unit"),
                             sorted(s["key"] for s in cal.searches(plan)))
            self.assertEqual(max(e["done"] for e in events if e["type"] == "unit"), 4)

    def test_one_worker_is_the_one_at_a_time_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "1")
            lines: list[str] = []
            self.assertEqual(cal.run_calibration(plan, emit=lines.append), 0)
            spans = sorted(json.loads(path.read_text())[:2] for path in (plan["output"] / "ledger").glob("*.span"))
            self.assertLessEqual(spans[0][1], spans[1][0] + 0.05)  # the second search started after the first ended
            self.assertFalse(any(line.startswith("[") for line in lines))

    def test_a_failing_search_does_not_discard_the_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--workers", "4")
            Path(plan["script"]).write_text(FAKE_SLOW_SCRIPT.replace(
                'net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))',
                'net, phase, out = get("--instance"), get("--phase"), Path(get("--out"))\n'
                'if net == "amazon": sys.exit(3)'))
            with self.assertRaises(RuntimeError):
                cal.run_calibration(plan, emit=lambda _: None)
            self.assertEqual(cal.recorded(plan), {"hedonic/dblp"})
            self.assertEqual(cal.state_of(plan), "partial")

    def test_the_stage_lists_only_running_searches_and_counts_the_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(tmp, "--networks", "dblp,amazon", "--workers", "2")
            stages: list[str] = []
            gate = threading.Event()

            def fake_run_search(plan_, search, progress, emit, prefix=""):
                progress({"type": "probe", "key": search["key"], "probes": 3, "text": ""})
                if search["key"].endswith("amazon"):
                    gate.wait(5)  # the second search is still running when the first has finished

            def fake_record(plan_, search, summary):
                if search["key"].endswith("dblp"):
                    gate.set()
                return {"searches": {search["key"]: {"absolute_resolution": 0.1, "max_memberships": 2,
                                                     "start": {"absolute_resolution": 0.1, "max_memberships": 2},
                                                     "f1_median": 0.5}}}

            def progress(item):
                if item["type"] == "stage":
                    stages.append(item["text"])

            with patch.object(cal, "run_search", fake_run_search), patch.object(cal, "final_of", lambda *a: {}), \
                    patch.object(cal, "record_search", fake_record), \
                    patch.object(cal, "read_tuned", lambda p: {"complete": True}):
                cal.run_calibration(plan, progress=progress, emit=lambda line: None)
            self.assertTrue(any(t.startswith("running 2 of 2") for t in stages), stages)
            last = [t for t in stages if t.startswith("running 1 of 2")]
            self.assertTrue(last, stages)
            self.assertNotIn("dblp", last[-1])
            self.assertIn("finished 1", last[-1])


class TestQueueAndCLI(unittest.TestCase):
    def test_the_queue_judges_a_calibration_by_its_results_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.py"
            script.write_text(FAKE_SCRIPT)
            run = ["calibrate", "--script", str(script), "--name", "cal-test", "--output-dir", str(Path(tmp) / "cal"),
                   "--workers", "1"]
            runqueue._check_arguments(run)
            self.assertEqual(runqueue._data_state(run)[0], "missing")
            plan = cal.front_study(run[1:])
            plan["python"] = sys.executable
            cal.run_calibration(plan, emit=lambda _: None)
            self.assertEqual(runqueue._data_state(run)[0], "complete")
            runqueue._check_arguments(["calibrate", "--networks", "dblp:top5000", "--name", "x"])  # no --script needed

    def test_registration_help_and_the_run_verb(self):
        self.assertIn("overlapping-calibrate", CLI.COMMANDS)
        self.assertIn("calibrate", CLI.RUN_ALIASES)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(["overlapping-calibrate", "--help"]), 0)
            self.assertEqual(CLI.hedonic_main(["run", "calibrate", "--help"]), 0)
        with patch("hedonic.experiments.overlapping.calibrate.front_main", return_value=0) as front:
            self.assertEqual(CLI.hedonic_main(["run", "calibrate", "--dry-run"]), 0)
        front.assert_called_once_with(["--dry-run"])

    def test_foreground_runs_in_this_process_and_detach_is_not_part_of_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "s.py"
            script.write_text(FAKE_SCRIPT)
            argv = ["--script", str(script), "--networks", "dblp:top5000", "--workers", "1", "--python", sys.executable,
                    "--output-dir", str(Path(tmp) / "cal")]
            self.assertEqual(cal.front_study([*argv, "--detach"]), cal.front_study(argv))
            with patch("hedonic.experiments.overlapping.runmanager.launch_calibrate",
                       side_effect=AssertionError("launched a durable run")), redirect_stdout(io.StringIO()):
                self.assertEqual(cal.front_main([*argv, "--foreground"]), 0)
            self.assertEqual(cal.state_of(cal.front_study(argv)), "complete")
            self.assertNotIn("--foreground", cal.launch_argv(cal.front_study([*argv, "--foreground"])))

    def test_usage_errors_are_status_two(self):
        with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(cal.main(["--phases", "multilevel"]), 2)
            self.assertEqual(cal.main(["--networks", "nosuchnet"]), 2)
            self.assertEqual(cal.main(["--no-such-flag"]), 2)
            self.assertEqual(cal.front_main(["--phases", "multilevel"]), 2)

    def test_the_run_view_names_the_build_not_a_pinned_igraph(self):
        entry = {"name": "c", "created": 0, "backend": "process", "session": "", "output_dir": "/x"}
        prog = {"status": "completed", "done": 1, "total": 1, "started": 0, "finished": 1, "updated": 1,
                "environment": qs.environment()}
        text = cal.snapshot(entry, prog, "●")
        self.assertIn(f"hedonic {qs.environment()['hedonic']}", text)
        self.assertNotIn("lucas-igraph", text)
        self.assertNotIn("igraph C", text)


if __name__ == "__main__":
    unittest.main()
