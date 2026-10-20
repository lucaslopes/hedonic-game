"""What a benchmark unit keeps of its answer (cover, labelled rows, vertex ids, graph hash), and the two readers that use
it without running a detector: the cover audit (stale audits run again) and the rescoring (``--check`` reproduces the
record)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import igraph as ig
import numpy as np

from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import cover_rescore as rs
from hedonic.experiments.overlapping import cover_store as cs
from hedonic.experiments.overlapping import unit_integrity as ui


def old_identity(graph: ig.Graph) -> str:
    """The hash ``cover_audit.graph_identity`` computed before ``cover_store`` (Python-sorted edge tuples)."""
    edges = np.asarray(sorted(tuple(sorted(edge)) for edge in graph.get_edgelist()), dtype=np.int64)
    return hashlib.sha256(edges.tobytes() + str(graph.vcount()).encode()).hexdigest()


def smoke_unit(out: Path) -> dict:
    """One Hedonic unit of the runner on its smoke graph, covers saved, as a benchmark leaves it."""
    with redirect_stdout(io.StringIO()):
        cr.main(["--smoke", "--datasets", "dblp", "--methods", "hedonic_local", "--output-dir", str(out), "--seed", "0",
                 "--save-covers"])
    return json.loads((out / "runs" / "dblp" / "hedonic_local.json").read_text())


class TestGraphHash(unittest.TestCase):
    def test_the_fast_hash_is_the_bytes_the_audit_always_hashed(self):
        graphs = [ig.Graph(4), ig.Graph.Famous("Petersen"), ig.Graph([(3, 1), (1, 0), (2, 0)]),
                  ig.Graph.Erdos_Renyi(200, 0.05, directed=False)]
        for graph in graphs:
            self.assertEqual(cs.graph_sha256(graph), old_identity(graph))
            self.assertEqual(ca.graph_identity(graph), old_identity(graph))
        self.assertNotEqual(cs.graph_sha256(ig.Graph(4)), cs.graph_sha256(ig.Graph(5)))  # isolated vertices count

    def test_a_gzip_has_no_time_stamp_so_the_same_answer_is_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json.gz", Path(tmp) / "b.json.gz"
            cs.write_json_gz(a, [[0, 1], [2]])
            with patch("time.time", return_value=4e9):
                cs.write_json_gz(b, [[0, 1], [2]])
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(cs.read_json_gz(a), [[0, 1], [2]])


class TestVertexIds(unittest.TestCase):
    def dataset(self, graph, strategy, kept=None):
        report = {"id_mapping_strategy": strategy}
        if kept is not None:
            report["ground_truth_completion"] = {"policy": "covered-induced", "original_vertex_ids": kept,
                                                 "source_graph_vertices": 100}
        return SimpleNamespace(name="dblp", cover_variant="top5000", graph=graph, report=report)

    def test_each_loader_numbering_gives_snap_ids(self):
        graph = ig.Graph(3)
        graph.vs["label"] = [10, 20, 30]
        self.assertEqual(cs.snap_vertex_ids(self.dataset(graph, "cached_vertex_label_attribute+ground_truth_covered_induced",
                                                         [0, 5, 9]))[0], [10, 20, 30])  # DBLP: the label, not the index
        plain = ig.Graph(3)
        self.assertEqual(cs.snap_vertex_ids(self.dataset(plain, "cached_identity_vertex_index+ground_truth_covered_induced",
                                                         [1, 2, 5]))[0], [1, 2, 5])  # the kept indices are the ids
        self.assertEqual(cs.snap_vertex_ids(self.dataset(plain, "cached_identity_vertex_index"))[0], [0, 1, 2])
        ids, why = cs.snap_vertex_ids(self.dataset(plain, "raw_sorted_original_id_to_contiguous_index"))
        self.assertIsNone(ids)
        self.assertIn("keeps no SNAP ids", why)

    def test_the_id_map_is_written_once_per_graph_and_the_record_keeps_a_pointer(self):
        graph = ig.Graph(3)
        data = self.dataset(graph, "cached_identity_vertex_index+ground_truth_covered_induced", [4, 7, 8])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path = cs.write_vertex_ids(folder, data, "abc")
            first = path.stat().st_mtime_ns
            self.assertEqual(cs.write_vertex_ids(folder, data, "abc"), path)
            self.assertEqual(path.stat().st_mtime_ns, first)  # the same graph: not written again
            stored = cs.read_vertex_ids(folder, "dblp")
            self.assertEqual((stored["snap_ids"], stored["graph_sha256"], stored["n"]), ([4, 7, 8], "abc", 3))
            report = cs.record_report(data.report, "abc", str(path.relative_to(folder)))
            self.assertNotIn("original_vertex_ids", report["ground_truth_completion"])
            self.assertEqual(report["ground_truth_completion"]["original_vertex_ids_count"], 3)
            self.assertEqual((report["graph_sha256"], report["vertex_ids_path"]), ("abc", "covers/dblp/vertex_ids.json.gz"))
            self.assertIn("original_vertex_ids", data.report["ground_truth_completion"])  # the dataset is untouched


class TestRunnerAnswer(unittest.TestCase):
    def test_a_unit_saves_its_cover_labels_and_graph_and_seals_all_of_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            record = smoke_unit(out)
            detector = record["detector"]
            self.assertNotIn("final_memberships", detector)  # the rows live in their own file now
            labels = out / detector["final_memberships_path"]
            self.assertTrue(labels.name.endswith(cs.LABELS_SUFFIX))
            rows = cs.read_labels(record, out)
            self.assertEqual(len(rows), record["dataset_report"]["n"])
            self.assertEqual(detector["final_memberships_rows"], len(rows))
            self.assertEqual(len(record["dataset_report"]["graph_sha256"]), 64)
            self.assertTrue((out / record["dataset_report"]["vertex_ids_path"]).is_file())
            seal = ui.seal(out, "dblp", "hedonic_local", by="run")
            self.assertEqual(seal["labels_sha256"], ui.sha256_file(labels))
            self.assertEqual(ui.check(out, "dblp", "hedonic_local", record["config_digest"], True).state, "verified")
            labels.write_bytes(labels.read_bytes() + b"x")  # the labels are certified like the cover
            self.assertEqual(ui.check(out, "dblp", "hedonic_local", record["config_digest"], True).state, "corrupt")
            labels.unlink()
            self.assertEqual(ui.check(out, "dblp", "hedonic_local", record["config_digest"], True).state, "missing")

    def test_a_record_with_inline_labels_still_reads(self):
        record = {"detector": {"final_memberships": [[0], [0, 1]]}}
        self.assertTrue(cs.has_labels(record))
        self.assertEqual(cs.read_labels(record, Path("/nowhere")), [[0], [0, 1]])
        self.assertFalse(cs.has_labels({"detector": {"final_memberships": []}}))


class TestRescore(unittest.TestCase):
    def test_check_reproduces_the_record_and_another_setting_is_its_own_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            unit = source / "dblp-full" / "seed0"
            record = smoke_unit(unit)
            smoke = cr._load_dataset("dblp", cover_variant="all", network_root=Path(tmp), max_nodes=None, smoke=True)
            out = Path(tmp) / "rescore"
            with patch.object(rs, "load_pair", lambda *a, **k: smoke), redirect_stdout(io.StringIO()):
                self.assertEqual(rs.main(["--source", str(source), "--networks", "dblp", "--methods", "hedonic",
                                          "--seeds", "0", "--check", "--output-dir", str(out)]), 0)
                checked = list(ca.read_records(out / rs.RESULTS_NAME).values())
                self.assertEqual(len(checked), 1)
                self.assertEqual(checked[0]["differs"], [])
                self.assertEqual(checked[0]["metrics"]["f1"], record["metrics"]["f1"])
                self.assertEqual(rs.main(["--source", str(source), "--networks", "dblp", "--methods", "hedonic",
                                          "--seeds", "0", "--no-paper-filter", "--no-omega", "--nf1",
                                          "--output-dir", str(out)]), 0)
            records = ca.read_records(out / rs.RESULTS_NAME)
            self.assertEqual(len(records), 2)  # keyed by the setting's digest
            other = next(r for r in records.values() if not r["spec"]["paper_filter"])
            self.assertIn("nf1", other["metrics"])
            self.assertIsNone(other["metrics"]["omega"])
            self.assertTrue((out / rs.TABLE_NAME).is_file() and (out / rs.SUMMARY_NAME).is_file())

    def test_check_fails_when_a_metric_differs(self):
        self.assertEqual(rs.differences({"f1": 0.5, "onmi": 0.2}, {"f1": 0.5, "onmi": 0.3, "n": 3}), ["onmi", "n"])

    def test_a_graph_with_other_edges_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            smoke_unit(source / "dblp-full" / "seed0")
            smoke = cr._load_dataset("dblp", cover_variant="all", network_root=Path(tmp), max_nodes=None, smoke=True)
            rewired = smoke.graph.copy()
            a, b = rewired.get_edgelist()[0]
            pair = next((u, v) for u in range(rewired.vcount()) for v in range(u + 1, rewired.vcount())
                        if not rewired.are_adjacent(u, v))
            rewired.delete_edges([(a, b)])
            rewired.add_edges([pair])  # same n and m, other edges
            other = SimpleNamespace(**{**smoke.__dict__, "graph": rewired}) if hasattr(smoke, "__dict__") else smoke
            with patch.object(rs, "load_pair", lambda *a, **k: other), redirect_stdout(io.StringIO()), \
                    patch("sys.stderr", io.StringIO()) as err:
                self.assertEqual(rs.main(["--source", str(source), "--networks", "dblp", "--methods", "hedonic",
                                          "--seeds", "0", "--check", "--output-dir", str(Path(tmp) / "o")]), 1)
            self.assertIn("other edges", err.getvalue())


class TestStaleAudits(unittest.TestCase):
    def test_an_audit_of_an_earlier_cover_is_not_done_and_its_new_record_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            unit = source / "dblp-full" / "seed0"
            smoke_unit(unit)
            ui.seal(unit, "dblp", "hedonic_local", by="run")
            out = Path(tmp) / "audit"
            plan = ca.resolve(["--source", str(source), "--networks", "dblp", "--methods", "hedonic", "--seeds", "0",
                               "--workers", "1", "--output-dir", str(out)])
            seal = ui.read_seal(unit, "dblp", "hedonic_local")
            cover, labelled = ca.conditions(plan)
            now = {"key": cover["key"], "status": "completed", "profile": "cover", "config_digest": seal["config_digest"],
                   "cover_file_sha256": seal["cover_sha256"]}
            self.assertTrue(ca.done(plan, {cover["key"]: now}, cover))
            earlier = {**now, "cover_file_sha256": "0" * 64}  # the unit ran again since: a new cover
            self.assertFalse(ca.done(plan, {cover["key"]: earlier}, cover))
            label_now = {"key": labelled["key"], "status": "completed", "profile": "labelled",
                         "config_digest": seal["config_digest"], "labels_file_sha256": seal["labels_sha256"]}
            self.assertTrue(ca.done(plan, {labelled["key"]: label_now}, labelled))
            self.assertFalse(ca.done(plan, {labelled["key"]: {**label_now, "config_digest": "other"}}, labelled))
            out.mkdir()
            (out / ca.RESULTS_NAME).write_text(json.dumps({**earlier, "n": 1}) + "\n" + json.dumps({**now, "n": 2}) + "\n")
            self.assertEqual(ca.read_records(out / ca.RESULTS_NAME)[cover["key"]]["n"], 2)  # the last record wins

    def test_the_labelled_audit_reads_the_rows_from_their_file_and_checks_the_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            unit = source / "dblp-full" / "seed0"
            record = smoke_unit(unit)
            smoke = cr._load_dataset("dblp", cover_variant="all", network_root=Path(tmp), max_nodes=None, smoke=True)
            plan = ca.resolve(["--source", str(source), "--networks", "dblp", "--methods", "hedonic", "--seeds", "0",
                               "--workers", "1", "--output-dir", str(Path(tmp) / "a")])
            labelled = ca.conditions(plan)[1]
            with patch.object(ca, "load_graph", lambda *a, **k: smoke.graph):
                result = ca.audit_unit(labelled, plan, {})
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["graph_sha256"], record["dataset_report"]["graph_sha256"])
            self.assertEqual(result["labels_file_sha256"], ui.sha256_file(unit / record["detector"]["final_memberships_path"]))
            rewired = smoke.graph.copy()
            a, b = rewired.get_edgelist()[0]
            pair = next((u, v) for u in range(rewired.vcount()) for v in range(u + 1, rewired.vcount())
                        if not rewired.are_adjacent(u, v))
            rewired.delete_edges([(a, b)])
            rewired.add_edges([pair])
            with patch.object(ca, "load_graph", lambda *a, **k: rewired), self.assertRaisesRegex(ValueError, "other edges"):
                ca.audit_unit(labelled, plan, {})

    def test_skip_missing_finishes_with_the_covers_it_could_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench"
            source.mkdir()
            out = Path(tmp) / "audit"
            argv = ["--source", str(source), "--networks", "dblp", "--methods", "hedonic", "--seeds", "0",
                    "--workers", "1", "--output-dir", str(out)]
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "no benchmark record"):
                    ca.run_audit(ca.resolve(argv), emit=lambda *_: None)
                self.assertEqual(ca.run_audit(ca.resolve([*argv, "--skip-missing"]), emit=lambda *_: None), 0)
            self.assertIn("--skip-missing", ca.launch_argv(ca.resolve([*argv, "--skip-missing"])))


class TestAuditView(unittest.TestCase):
    def test_the_audit_item_counts_current_audits_and_prices_the_rest(self):
        from hedonic.experiments.overlapping import queuebuild as qb

        with tempfile.TemporaryDirectory() as tmp:
            plan = ca.resolve(["--source", tmp, "--networks", "dblp:top5000,amazon:top5000", "--methods", "hedonic",
                               "--seeds", "0-1", "--workers", "2", "--output-dir", str(Path(tmp) / "a")])
        keys = [c["key"] for c in ca.conditions(plan)]  # hedonic: cover + labelled, 2 pairs x 2 seeds
        self.assertEqual(len(keys), 8)
        done = {keys[0]: {"status": "completed", "audit_seconds": 100.0, "state_seconds": 5.0, "load_seconds": 20.0},
                keys[1]: {"status": "completed", "audit_seconds": 40.0}}
        stale = {keys[2]: {"status": "completed", "audit_seconds": 999.0}}
        now = 1000.0
        progress = {"status": "running", "current": {"key": keys[3], "eta_seconds": 30.0, "updated": now - 10.0}}
        model = qb.audit(plan, {**done, **stale}, progress, now=now, finished=set(done))
        family = model.families[0]
        self.assertEqual(sum(u.done for u in family.units), 2)  # the stale record is not done
        self.assertEqual(family.units[0].seconds, 125.0)  # load + state + audit
        cost = {u.coords: family.cost(u) for u in family.units if not u.done}
        running = next(u for u in family.units if u.running)
        self.assertEqual(cost[running.coords], (20.0, "estimate"))  # what its report says is left, aged
        stale_unit = family.units[2]  # same method, pair and profile as keys[0]: measured
        self.assertEqual(cost[stale_unit.coords], (125.0, "measured"))
        self.assertEqual(qb.audit(plan, {}, None).families[0].units[0].done, False)


if __name__ == "__main__":
    unittest.main()
