"""The cover audit: plan, profiles, exact stability against the grid audit, resumable records, queue state and CLI routing."""

from __future__ import annotations

import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import CLI
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import gt_posthoc as gp
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import stability as st
from tests.experiments.test_studies_stability_exact import random_instance

# Two disjoint 4-cliques (vertices 0-3 and 4-7) and a vertex 8 that no community contains.
CLIQUES = [[0, 1, 2, 3], [4, 5, 6, 7]]
N = 9


def two_cliques() -> ig.Graph:
    edges = [(a, b) for block in CLIQUES for i, a in enumerate(block) for b in block[i + 1:]]
    edges.append((7, 8))
    return ig.Graph(n=N, edges=edges)


def write_unit(source: Path, network: str, seed: int, method: str, graph: ig.Graph, *, cover=None, labelled=None,
               status: str = "completed", gamma: float = 0.3, cap: int = 2) -> Path:
    """One benchmark unit as ``codeseg_reproduction`` leaves it: record under runs/, cover under covers/."""
    runner = qs.BY_KEY[method].runner
    unit = source / f"{network}-full" / f"seed{seed}"
    (unit / "runs" / network).mkdir(parents=True, exist_ok=True)
    record = {"status": status, "config_digest": "digest", "detection_seconds": 1.5, "method": runner,
              "dataset_report": {"n": graph.vcount(), "m": graph.ecount(), "cover_variant": "all"}}
    if status == "completed":
        record["detector"] = {"effective_resolution": gamma, "max_memberships": cap,
                              "final_memberships": labelled if labelled is not None else []}
        if cover is not None:
            relative = f"covers/{network}/{runner}.json.gz"
            (unit / "covers" / network).mkdir(parents=True, exist_ok=True)
            with gzip.open(unit / relative, "wt", encoding="utf-8") as handle:
                json.dump(cover, handle)
            record["cover_path"] = relative
    path = unit / "runs" / network / f"{runner}.json"
    path.write_text(json.dumps(record))
    return path


def plan_for(source: Path, output: Path, *extra: str) -> dict:
    return ca.resolve(["--source", str(source), "--networks", "dblp", "--seeds", "0",
                       "--workers", "1", "--output-dir", str(output), "--name", "audit-test", *extra])


class TestPlan(unittest.TestCase):
    def test_conditions_go_method_by_method_and_only_hoc_has_a_labelled_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = ca.resolve(["--source", tmp, "--networks", "dblp,amazon", "--methods", "hedonic,clique_percolation",
                               "--seeds", "0-1", "--output-dir", str(Path(tmp) / "out")])
            keys = [c["key"] for c in ca.conditions(plan)]
            # networks x (hoc: 2 seeds x cover+labelled; clique_percolation, deterministic: its lowest seed's cover)
            self.assertEqual(len(keys), 2 * (2 * 2 + 1))
            self.assertEqual(keys[:2], ["dblp/seed0/hedonic/cover", "dblp/seed0/hedonic/labelled"])
            methods = [c["method"] for c in ca.conditions(plan)]
            self.assertEqual(methods, sorted(methods, key=["hedonic", "clique_percolation"].index))  # all of one method, then the next
            self.assertNotIn("dblp/seed0/clique_percolation/labelled", keys)

    def test_methods_are_found_in_the_sources_when_not_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            write_unit(source, "dblp", 0, "clique_percolation", two_cliques(), cover=CLIQUES)
            plan = plan_for(source, Path(tmp) / "out")
            self.assertEqual(plan["methods"], ["clique_percolation"])

    def test_bad_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for bad in (["--networks", "nowhere"], ["--profiles", "bogus"], ["--cap", "1"], ["--cap", "x"]):
                with self.assertRaises(ValueError, msg=bad):
                    ca.resolve(["--source", tmp, "--methods", "clique_percolation", *bad])

    def test_dry_run_prints_the_plan_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            buffer = io.StringIO()
            with patch.object(ca, "run_audit", side_effect=AssertionError("dry run audited")), redirect_stdout(buffer):
                code = ca.main(["--source", tmp, "--methods", "hedonic", "--seeds", "0", "--output-dir", str(out),
                                "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            self.assertIn("dry run", buffer.getvalue())
            self.assertIn("hedonic", buffer.getvalue())


class TestProfiles(unittest.TestCase):
    def test_uncovered_vertices_get_private_singleton_labels(self):
        memberships, info = ca.cover_profile(CLIQUES, N)
        self.assertEqual(info, {"communities": 2, "completed_vertices": 1})
        self.assertEqual(sorted(len(row) for row in memberships), [1] * N)
        self.assertEqual(len({row[0] for row in memberships}), 3)  # two communities and vertex 8 alone

    def test_labelled_profile_keeps_duplicate_bodies_apart(self):
        # Vertex 0 carries two labels whose bodies are identical ({0,1,2}); a merged cover would count them once.
        rows = [[5, 9], [5, 9], [5, 9], [7], [7], [7], [7], [7], [7]]
        memberships, info = ca.labelled_profile(rows, N)
        self.assertEqual(info["communities"], 3)
        self.assertEqual(len(memberships[0]), 2)
        with self.assertRaises(ValueError):
            ca.labelled_profile(rows[:-1], N)

    def test_cap_rules(self):
        self.assertEqual(ca.effective_cap("record", 4, 2), 4)
        self.assertEqual(ca.effective_cap("record", None, 3), 3)
        self.assertEqual(ca.effective_cap("record", None, 1), 2)  # never below two
        self.assertEqual(ca.effective_cap("own", 8, 3), 3)
        self.assertEqual(ca.effective_cap(6, 4, 3), 6)
        self.assertEqual(ca.effective_cap(2, 4, 5), 5)  # a cap below the cover's own multiplicity is raised


class TestOwnResolution(unittest.TestCase):
    def test_the_parameter_is_preferred_and_a_disagreement_is_noted(self):
        self.assertEqual(ca.own_resolution({"effective_resolution": 0.2}), (0.2, None))
        self.assertEqual(ca.own_resolution({"effective_resolution": 0.2, "parameters": {"absolute_resolution": 0.2}}),
                         (0.2, None))
        gamma, note = ca.own_resolution({"effective_resolution": 2e-5, "parameters": {"absolute_resolution": 0.1462}})
        self.assertEqual(gamma, 0.1462)
        self.assertIn("absolute_resolution", note)
        self.assertEqual(ca.own_resolution({}), (None, None))


class TestExactAudit(unittest.TestCase):
    def test_stable_fractions_agree_with_the_grid_audit_and_the_regret_table(self):
        grid = [0.0, 0.1, 0.35, 0.6, 1.0]
        for seed in range(6):
            graph, memberships = random_instance(seed)
            n = graph.vcount()
            cover = st.prepare(graph, memberships)
            cap = max(2, max(len(row) for row in memberships))
            audited = ca.audit_state(cover, cap=cap, grid=grid, regret_gammas=grid, workers=1,
                                     atol=gp.ATOL, rtol=gp.RTOL)
            reference = gp.analyse_state(st.prepare(graph, memberships, engine="python"), cap=cap, grid=grid, workers=1)
            for policy, block in audited.items():
                self.assertEqual([round(f * n) for f in block["stable_fraction"]],
                                 [row["exact_stable_vertex_count"] for row in reference[policy]["per_gamma"]],
                                 (seed, policy))
                self.assertEqual([row["gamma"] for row in block["regret"]], grid)
                for fraction, row in zip(block["stable_fraction"], block["regret"]):
                    self.assertAlmostEqual(fraction, row["stable_fraction"], msg=(seed, policy, row["gamma"]))
                self.assertTrue(0.0 <= block["stable_measure_linear"] <= 1.0 + 1e-12)
                self.assertTrue(0.0 <= block["stable_measure_log"] <= 1.0 + 1e-12)

    def test_cliques_are_a_nash_equilibrium_at_their_own_resolution(self):
        graph = two_cliques()
        labelled = [[0]] * 4 + [[1]] * 4 + [[2]]
        memberships, _ = ca.labelled_profile(labelled, N)
        cover = st.prepare(graph, memberships)
        out = ca.audit_state(cover, cap=2, grid=[0.0, 0.3], regret_gammas=[0.3], workers=1,
                             atol=gp.ATOL, rtol=gp.RTOL)
        for block in out.values():
            self.assertEqual(block["regret"][0]["stable_fraction"], 1.0)
            self.assertTrue(block["regret"][0]["is_nash_equilibrium"])
            self.assertTrue(any(lo <= 0.3 <= hi for lo, hi in block["nash_window"]))
            self.assertEqual(block["vertices_never_stable"], 0)


class TestAuditProgress(unittest.TestCase):
    """The exact stage reports the vertices it has done; the audit process hands them to the worker through a file."""

    def test_the_eta_is_the_rest_of_this_policy_plus_as_long_again_for_each_policy_to_come(self):
        event = ca.vertices_event("k · ", "fixed_labels", 1, 2, 100, 1000, 10.0)
        self.assertEqual((event["done"], event["total"], event["number"], event["policies"]), (100, 1000, 1, 2))
        self.assertAlmostEqual(event["eta_seconds"], 90.0 + 100.0)  # 100 s a policy: 90 s left of this one, 100 s of the next
        self.assertEqual(event["text"], "k · " + event["detail"])
        self.assertIn("fixed_labels (1/2): exact stability 100/1,000 vertices · about 3:10 left", event["detail"])
        last = ca.vertices_event("", "open_labels", 2, 2, 1000, 1000, 95.0)
        self.assertEqual(last["eta_seconds"], 0.0)
        self.assertNotIn("left", last["detail"])
        self.assertIsNone(ca.vertices_event("", "fixed_labels", 1, 2, 0, 1000, 0.0)["eta_seconds"])
        early = ca.vertices_event("", "fixed_labels", 1, 2, 49, 1000, 224.0)  # the first chunk lands in a burst: no rate yet
        self.assertIsNone(early["eta_seconds"])
        self.assertNotIn("left", early["detail"])
        self.assertIsNotNone(ca.vertices_event("", "fixed_labels", 1, 2, 50, 1000, 224.0)["eta_seconds"])

    def test_audit_state_reports_every_vertex_and_times_each_stage(self):
        graph, memberships = random_instance(3)
        n = graph.vcount()
        cover = st.prepare(graph, memberships)
        events = []
        out = ca.audit_state(cover, cap=2, grid=[0.0, 0.5], regret_gammas=[0.0, 1.0], workers=1,
                             atol=gp.ATOL, rtol=gp.RTOL, progress=events.append, label="u · ")
        for number, policy in enumerate(out, 1):
            seen = [e["done"] for e in events if e["type"] == "vertices" and e["policy"] == policy]
            self.assertEqual(seen, list(range(1, n + 1)), policy)  # fewer than PROGRESS_STEPS vertices: one event each
            self.assertTrue(all(e["number"] == number and e["total"] == n for e in events if e.get("policy") == policy))
            self.assertGreaterEqual(out[policy]["regret_seconds"], 0.0)
            self.assertGreaterEqual(out[policy]["exact_seconds"], 0.0)
        self.assertTrue(any(e["type"] == "stage" and "regret" in e["text"] for e in events))

    def test_events_are_thinned_to_progress_steps_per_policy(self):
        graph = ig.Graph.Ring(2000)
        memberships = [[v // 100] for v in range(2000)]
        cover = st.prepare(graph, memberships)
        events = []
        with patch.object(ca, "PROGRESS_STEPS", 20):
            ca.audit_state(cover, cap=2, grid=[0.0], regret_gammas=[0.0], workers=1,
                           atol=gp.ATOL, rtol=gp.RTOL, progress=events.append)
        counts = [e["done"] for e in events if e["type"] == "vertices" and e["policy"] == "fixed_labels"]
        self.assertEqual(counts, list(range(100, 2001, 100)))  # every 2000 // 20 vertices, ending on the last

    def test_the_sink_writes_the_latest_event_atomically_and_the_reader_survives_a_torn_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertIsNone(ca.read_audit_progress(folder))
            sink = ca.progress_sink(folder / ca.PROGRESS_NAME, "dblp/seed0/clique_percolation/cover")
            sink(ca.vertices_event("", "fixed_labels", 1, 2, 10, 100, 1.0))
            sink(ca.vertices_event("", "fixed_labels", 1, 2, 11, 100, 1.1))  # within a second: not written
            info = ca.read_audit_progress(folder)
            self.assertEqual((info["key"], info["done"], info["total"]), ("dblp/seed0/clique_percolation/cover", 10, 100))
            self.assertIn("updated", info)
            sink(ca.vertices_event("", "open_labels", 2, 2, 100, 100, 9.0))  # the last vertex of a policy is always written
            self.assertEqual(ca.read_audit_progress(folder)["done"], 100)
            (folder / ca.PROGRESS_NAME).write_text('{"key": "x", "do')
            self.assertIsNone(ca.read_audit_progress(folder))

    def test_a_unit_record_carries_its_stage_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            write_unit(run, "dblp", 0, "clique_percolation", two_cliques(), cover=CLIQUES)
            plan = ca.resolve(["--source", str(run), "--networks", "dblp", "--methods", "clique_percolation", "--seeds", "0",
                               "--output-dir", str(run / ca.STAGE_DIR)])
            events, graphs = [], {}
            with patch.object(ca, "load_graph", lambda network, variant, root, induce="full": two_cliques()):
                record = ca.audit_unit(ca.conditions(plan)[0], plan, graphs, events.append)
                again = ca.audit_unit(ca.conditions(plan)[0], plan, graphs, events.append)
            self.assertGreaterEqual(record["load_seconds"], 0.0)
            self.assertIsNone(again["load_seconds"])  # the graph was already loaded: this unit paid nothing for it
            self.assertGreaterEqual(record["state_seconds"], 0.0)
            for block in record["policies"].values():
                self.assertIn("regret_seconds", block)
            self.assertTrue(any(e["type"] == "vertices" for e in events))


class TestRun(unittest.TestCase):
    def build(self, tmp: str) -> tuple[Path, ig.Graph]:
        graph = two_cliques()
        source = Path(tmp) / "bench"
        labelled = [[0]] * 4 + [[1]] * 4 + [[2]]
        write_unit(source, "dblp", 0, "hedonic", graph, cover=CLIQUES, labelled=labelled)
        write_unit(source, "dblp", 0, "clique_percolation", graph, cover=CLIQUES)
        return source, graph

    def test_records_resume_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, graph = self.build(tmp)
            out = Path(tmp) / "out"
            plan = plan_for(source, out, "--methods", "hedonic,clique_percolation")
            loads: list[str] = []

            def loader(network, variant, root, induce="full"):
                loads.append(network)
                return graph

            with patch.object(ca, "load_graph", loader):
                self.assertEqual(ca.run_audit(plan, emit=lambda _: None), 0)
                rows = [json.loads(line) for line in (out / ca.RESULTS_NAME).read_text().splitlines()]
                self.assertEqual([r["key"] for r in rows], ["dblp/seed0/hedonic/cover", "dblp/seed0/hedonic/labelled",
                                                           "dblp/seed0/clique_percolation/cover"])
                self.assertEqual({r["status"] for r in rows}, {"completed"})
                self.assertEqual(loads, ["dblp"])  # one graph load serves every audit of the network
                hoc_cover, hoc_labelled, clique_percolation = rows
                self.assertEqual(hoc_cover["completed_vertices"], 1)
                self.assertEqual(hoc_cover["cap"], 2)
                self.assertIsNone(clique_percolation["gamma_own"])
                self.assertEqual(clique_percolation["cap"], 2)  # no detector cap: the cover's own multiplicity, at least 2
                self.assertEqual(ca.own_gamma_value(hoc_labelled, "open_labels", "stable_fraction"), 1.0)
                self.assertIsNone(ca.own_gamma_value(clique_percolation, "open_labels", "stable_fraction"))
                self.assertIn(0.3, hoc_labelled["grid"])
                self.assertEqual(len(hoc_labelled["grid"]), len(hoc_labelled["policies"]["open_labels"]["stable_fraction"]))
                self.assertTrue((out / ca.TABLE_NAME).is_file())
                summary = json.loads((out / ca.SUMMARY_NAME).read_text())
                self.assertEqual(summary["status"], {"completed": 3})
                self.assertEqual(ca.state_of(plan), "complete")

                # A rerun finds every audit recorded and loads nothing.
                loads.clear()
                self.assertEqual(ca.run_audit(plan, emit=lambda _: None), 0)
                self.assertEqual(loads, [])
                self.assertEqual(len((out / ca.RESULTS_NAME).read_text().splitlines()), 3)

                # --overwrite deletes the records first.
                self.assertEqual(ca.run_audit(plan_for(source, out, "--methods", "hedonic,clique_percolation", "--overwrite"),
                                              emit=lambda _: None), 0)
                self.assertEqual(loads, ["dblp"])
                self.assertEqual(len((out / ca.RESULTS_NAME).read_text().splitlines()), 3)

    def test_a_labelled_profile_that_is_not_an_equilibrium_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = two_cliques()
            source = Path(tmp) / "bench"
            # Every vertex alone: at gamma 0 each clique member gains by joining a neighbour, so this is no equilibrium.
            write_unit(source, "dblp", 0, "hedonic", graph, cover=CLIQUES, labelled=[[i] for i in range(N)], gamma=0.0)
            out = Path(tmp) / "out"
            plan = plan_for(source, out, "--methods", "hedonic", "--profiles", "labelled")
            lines: list[str] = []
            with patch.object(ca, "load_graph", lambda *a: graph):
                ca.run_audit(plan, emit=lines.append)
            row = json.loads((out / ca.RESULTS_NAME).read_text())
            self.assertIn("not an equilibrium", row["warnings"][0])
            self.assertTrue(any("WARNING" in line for line in lines))
            self.assertIn("not an equilibrium", json.loads((out / ca.SUMMARY_NAME).read_text())["warnings"][row["key"]][0])

    def test_a_missing_or_failed_source_record_is_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, graph = self.build(tmp)
            write_unit(source, "dblp", 0, "fox", graph, status="failed")  # a detector run that ended without a cover
            out = Path(tmp) / "out"
            plan = plan_for(source, out, "--methods", "clique_percolation,fox,angel", "--seeds", "0")
            with patch.object(ca, "load_graph", lambda *a: graph):
                with self.assertRaises(RuntimeError) as caught:
                    ca.run_audit(plan, emit=lambda _: None)
            self.assertIn("dblp/seed0/angel/cover", str(caught.exception))  # angel has no record yet
            rows = {json.loads(line)["key"]: json.loads(line)
                    for line in (out / ca.RESULTS_NAME).read_text().splitlines()}
            self.assertEqual(rows["dblp/seed0/clique_percolation/cover"]["status"], "completed")
            self.assertEqual(rows["dblp/seed0/fox/cover"]["status"], "no_cover")  # finished, so not retried
            self.assertNotIn("dblp/seed0/angel/cover", rows)
            self.assertEqual(ca.state_of(plan), "partial")

    def test_a_record_from_another_graph_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, graph = self.build(tmp)
            other = ig.Graph(n=graph.vcount() + 1, edges=graph.get_edgelist())
            plan = plan_for(source, Path(tmp) / "out", "--methods", "clique_percolation")
            with patch.object(ca, "load_graph", lambda *a: other):
                with self.assertRaises(ValueError):
                    ca.run_audit(plan, emit=lambda _: None)

    def test_a_run_without_saved_covers_is_marked_not_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph = two_cliques()
            source = Path(tmp) / "bench"
            write_unit(source, "dblp", 0, "clique_percolation", graph, cover=None)  # completed, but --save-covers was off
            plan = plan_for(source, Path(tmp) / "out", "--methods", "clique_percolation")
            with patch.object(ca, "load_graph", lambda *a: graph):
                ca.run_audit(plan, emit=lambda _: None)
            row = json.loads((Path(tmp) / "out" / ca.RESULTS_NAME).read_text())
            self.assertEqual(row["status"], "no_cover")
            self.assertIn("save-covers", row["reason"])


class TestQueueAndViewer(unittest.TestCase):
    def test_queue_judges_an_audit_by_its_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            source.mkdir()
            out = Path(tmp) / "out"
            run = ["audit", "--source", str(source), "--networks", "dblp", "--methods", "hedonic,clique_percolation", "--seeds", "0",
                   "--name", "audit-test", "--output-dir", str(out)]
            runqueue._check_arguments(run)
            self.assertEqual(runqueue._data_state(run), ("missing", f"{out.resolve()}#audit-test"))  # a key per item
            out.mkdir()
            keys = [c["key"] for c in ca.conditions(ca.front_study(run[1:]))]
            (out / ca.RESULTS_NAME).write_text(json.dumps({"key": keys[0], "status": "completed"}) + "\n")
            self.assertEqual(runqueue._data_state(run)[0], "partial")
            (out / ca.RESULTS_NAME).write_text(
                "".join(json.dumps({"key": key, "status": "completed"}) + "\n" for key in keys))
            self.assertEqual(runqueue._data_state(run)[0], "complete")

    def test_launch_argv_rebuilds_the_same_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = ca.resolve(["--source", tmp, "--methods", "hedonic,clique_percolation", "--seeds", "0-2", "--networks", "dblp",
                               "--cap", "own", "--output-dir", str(Path(tmp) / "o"), "--name", "n"])
            again = ca.resolve(ca.launch_argv(plan))
            for key in ("sources", "networks", "methods", "seeds", "profiles", "cap", "name", "output", "workers"):
                self.assertEqual(again[key], plan[key], key)

    def test_snapshot_renders_a_completed_run(self):
        entry = {"name": "audit-test", "created": 0.0, "backend": "process", "session": "", "output_dir": "/tmp/none"}
        prog = {"status": "completed", "done": 240, "total": 240, "started": 0.0, "finished": 60.0,
                "describe": "cover audit", "output": "/tmp/none/cover_audit.jsonl"}
        with patch("hedonic.experiments.overlapping.runmanager.effective_status", return_value="completed"):
            text = ca.snapshot(entry, prog, "●")
        self.assertIn("completed", text)
        self.assertIn("240/240", text)

    def test_the_manager_counts_and_estimates_an_audit_run(self):
        from hedonic.experiments.overlapping import runmanager as rm

        entry, prog = {"kind": "audit"}, {"done": 3, "total": 10, "eta_seconds": 42.0}
        self.assertEqual(rm.progress_counts(entry, prog), (3, 10))
        self.assertEqual(rm.remaining_seconds(entry, prog, "running"), 42.0)
        self.assertIsNone(rm.remaining_seconds(entry, prog, "completed"))


class TestCLI(unittest.TestCase):
    def test_registration_help_and_the_run_verb_routes_to_the_front_door(self):
        self.assertIn("overlapping-cover-audit", CLI.COMMANDS)
        self.assertIn("audit", CLI.RUN_ALIASES)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(CLI.main(["overlapping-cover-audit", "--help"]), 0)
            self.assertEqual(CLI.hedonic_main(["run", "audit", "--help"]), 0)
        with patch("hedonic.experiments.overlapping.cover_audit.front_main", return_value=0) as front:
            self.assertEqual(CLI.hedonic_main(["run", "audit", "--source", "x", "--dry-run"]), 0)
        front.assert_called_once_with(["--source", "x", "--dry-run"])

    def test_front_dry_run_starts_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer = io.StringIO()
            with patch("hedonic.experiments.overlapping.runmanager.launch_audit",
                       side_effect=AssertionError("dry run launched a run")), redirect_stdout(buffer):
                code = ca.front_main(["--source", tmp, "--methods", "clique_percolation", "--output-dir", str(Path(tmp) / "run"),
                                      "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse((Path(tmp) / "run").exists())

    def test_a_bad_argument_is_a_usage_error_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()):
            self.assertEqual(ca.front_main(["--source", tmp, "--networks", "nowhere"]), 2)
            self.assertEqual(ca.main(["--source", tmp, "--networks", "nowhere"]), 2)


if __name__ == "__main__":
    unittest.main()
