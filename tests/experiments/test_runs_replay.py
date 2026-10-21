"""``hedonic run check --replay N``: a run's Hedonic units and cover audits replayed with the loaded core, bit for bit,
on small synthetic runs laid out as the benchmark and the audit leave them."""

from __future__ import annotations

import gzip
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import igraph as ig

from hedonic.experiments import runtime, timing
from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import cover_audit as ca
from hedonic.experiments.overlapping import cover_store
from hedonic.experiments.overlapping import replay
from hedonic.experiments.overlapping import runmanager


def planted(n: int, seed: int) -> ig.Graph:
    rng = random.Random(seed)
    blocks = [{v % 3} | ({rng.randrange(3)} if rng.random() < 0.3 else set()) for v in range(n)]
    edges = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < (0.3 if blocks[a] & blocks[b] else 0.03)]
    return ig.Graph(n=n, edges=edges)


def write_hedonic_unit(source: Path, network: str, seed: int, graph: ig.Graph, *, gamma: float = 0.05,
                       cap: int = -1, inline_labels: bool = False) -> Path:
    """A Hedonic unit as ``codeseg_reproduction._run_dataset`` leaves it under --save-covers."""
    unit = source / f"{network}-full-covered" / f"seed{seed}"
    parameters = {"absolute_resolution": gamma, "allow_isolation": True, "max_memberships": cap, "dataset": network}
    with timing.detector_run():
        cover, detector = cr._run_method("hedonic_local", graph, [], seed, parameters)
    report = {"n": graph.vcount(), "m": graph.ecount(), "cover_variant": "all",
              "graph_sha256": cover_store.graph_sha256(graph)}
    record = {"status": "completed", "method": "hedonic_local", "dataset": network, "seed": seed,
              "config_digest": "digest", "detection_seconds": 0.01, "dataset_report": report,
              "runtime_versions": runtime.runtime_versions()}
    if inline_labels:  # an older record: the labelled rows inline, no cover file
        record["detector"] = detector
    else:
        record["cover_path"] = cover_store.save_answer(unit, network, "hedonic_local", cover, detector)
        record["detector"] = detector
    path = unit / "runs" / network / "hedonic_local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


class TestReplay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "queue-root"
        self.bench = self.root / "benchmark"
        self.graphs = {"dblp": planted(60, 1), "amazon": planted(45, 2)}
        write_hedonic_unit(self.bench, "dblp", 0, self.graphs["dblp"])
        write_hedonic_unit(self.bench, "dblp", 1, self.graphs["dblp"], cap=2)
        write_hedonic_unit(self.bench, "amazon", 0, self.graphs["amazon"], inline_labels=True)
        self.loads = []

    def tearDown(self):
        self.tmp.cleanup()

    def loader(self, network, variant, induce):
        self.loads.append((network, variant, induce))
        return self.graphs[network]

    def check(self, count=5, **kwargs):
        lines = []
        code = replay.check(self.root, count, load_graph=self.loader, emit=lines.append, **kwargs)
        return code, "\n".join(lines)

    def audit(self):
        """Audit the dblp units' covers and labelled rows into the robustness ledger, as the queue's audit item does."""
        out = self.bench / "robustness"
        plan = ca.resolve(["--source", str(self.bench), "--networks", "dblp", "--seeds", "0-1", "--methods", "hedonic",
                           "--workers", "1", "--output-dir", str(out), "--induce", "covered"])
        with patch.object(ca, "load_graph", lambda network, variant, root, induce="full": self.graphs[network]):
            self.assertEqual(ca.run_audit(plan, emit=lambda _: None), 0)
        return out / ca.RESULTS_NAME

    def test_units_are_found_cheapest_first(self):
        units = replay.find_units(self.root)
        self.assertEqual([u["record"]["dataset"] for u in units], ["amazon", "dblp", "dblp"])
        self.assertTrue(all(u["source"] == runtime.core_built_source_sha256() for u in units))

    def test_every_unit_reproduces_and_the_pass_is_recorded(self):
        code, text = self.check()
        self.assertEqual(code, 0, text)
        self.assertIn("3 identical, 0 different, 0 skipped", text)
        manifest = json.loads((self.root / replay.MANIFEST_NAME).read_text())
        entry = manifest["adopted"][-1]
        self.assertEqual(entry["core"]["hedonic_core_fingerprints"], list(runtime.core_fingerprints()))
        self.assertEqual(entry["core"]["hedonic_core_source_sha256"], runtime.core_source_sha256())
        self.assertEqual(entry["sources"], {runtime.core_built_source_sha256(): 3})
        self.assertEqual(len(entry["replayed"]), 3)
        self.assertEqual(self.check()[0], 0)
        self.assertEqual(len(json.loads((self.root / replay.MANIFEST_NAME).read_text())["adopted"]), 2)  # appended

    def test_a_changed_answer_is_reported_and_nothing_is_recorded(self):
        labels = next(self.bench.rglob(f"hedonic_local{cover_store.LABELS_SUFFIX}"))
        rows = cover_store.read_json_gz(labels)
        rows[0] = rows[0] + [max(map(max, rows)) + 1]  # a label the detector never gave
        cover_store.write_json_gz(labels, rows)
        code, text = self.check()
        self.assertEqual(code, 1)
        self.assertIn("DIFFERS: labelled rows", text)
        self.assertFalse((self.root / replay.MANIFEST_NAME).exists())

    def test_a_different_core_is_caught(self):
        real = cr._run_method

        def other_core(name, graph, ground_truth, seed, parameters):
            return real(name, graph, ground_truth, seed + 1, parameters)  # another trajectory

        with patch.object(cr, "_run_method", other_core):
            code, text = self.check()
        self.assertEqual(code, 1)
        self.assertIn("DIFFERS: cover, labelled rows", text)

    def test_a_record_of_another_graph_is_skipped_not_compared(self):
        self.graphs["dblp"] = planted(60, 99)
        code, text = self.check()
        self.assertIn("skipped: the loaded dblp graph is not the one the record ran on", text)
        self.assertEqual(code, 0)  # the amazon unit still reproduced

    def test_cover_audits_are_replayed_on_a_vertex_sample_against_the_reference(self):
        self.audit()
        audits = replay.find_audits(self.root)
        self.assertEqual(len(audits), 4)  # two seeds x (cover, labelled)
        code, text = self.check(count=0)
        self.assertEqual(code, 1, "nothing replayed is not a pass")
        code, text = self.check(count=4, vertices=10)
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("✓ audit"), 4)
        entry = json.loads((self.root / replay.MANIFEST_NAME).read_text())["adopted"][-1]
        self.assertEqual(sum(item["kind"] == "audit" for item in entry["replayed"]), 4)
        self.assertTrue(all(item["vertices"] == 10 for item in entry["replayed"] if item["kind"] == "audit"))

    def test_an_audit_whose_core_differs_from_the_reference_fails(self):
        from hedonic.experiments.overlapping import stability as st

        self.audit()
        real = st._CCover.intervals

        def shifted(self, vertices, cap, allow_isolation, *, atol, rtol):
            return [[(lo, hi * (1 - 1e-12)) for lo, hi in row] for row in real(self, vertices, cap, allow_isolation,
                                                                             atol=atol, rtol=rtol)]

        with patch.object(st._CCover, "intervals", shifted):
            code, text = self.check(count=1)
        self.assertEqual(code, 1)
        self.assertIn("DIFFERS: fixed_labels, open_labels", text)

    def test_an_audit_of_a_cover_that_ran_again_is_skipped(self):
        self.audit()
        cover = next(self.bench.rglob("seed0/covers/dblp/hedonic_local.json.gz"))
        with gzip.open(cover, "wt") as handle:
            json.dump([[0, 1]], handle)
        lines = []
        replay.check(self.root, 4, load_graph=self.loader, emit=lines.append, record=False)
        self.assertIn("skipped: the cover on disk is not the one audited", "\n".join(lines))

    def test_dry_run_lists_and_touches_nothing(self):
        before = sorted(p.stat().st_mtime_ns for p in self.root.rglob("*"))
        code, text = self.check(dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.loads, [])
        self.assertIn("dry run: nothing loaded, run or written", text)
        self.assertIn(runtime.core_built_source_sha256()[:12], text)
        self.assertEqual(sorted(p.stat().st_mtime_ns for p in self.root.rglob("*")), before)
        self.assertFalse((self.root / replay.MANIFEST_NAME).exists())

    def test_the_run_check_verb_takes_replay_and_root(self):
        buffer = io.StringIO()
        with patch.object(replay, "_default_loader", lambda root: self.loader), redirect_stdout(buffer):
            code = runmanager.dispatch(["check", "--replay", "1", "--root", str(self.root)])
        self.assertEqual(code, 0, buffer.getvalue())
        self.assertIn("1 identical", buffer.getvalue())
        with self.assertRaises(SystemExit):
            runmanager.dispatch(["check", "--root", str(self.root)])  # --root alone is not a replay


if __name__ == "__main__":
    unittest.main()
