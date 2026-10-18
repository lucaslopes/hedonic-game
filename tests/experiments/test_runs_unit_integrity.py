"""A benchmark unit is done only when its record is present, final, of the same settings and hash-certified."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import unit_integrity as ui

DIGEST = "d" * 16


def write_unit(out: Path, net="dblp", runner="codeseg", *, status="completed", digest=DIGEST, cover=True) -> Path:
    path = ui.record_path(out, net, runner)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": status, "config_digest": digest, "metrics": {"f1": 0.5}}
    if cover:
        cover_file = out / "covers" / net / f"{runner}.json.gz"
        cover_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cover_file, "wt", encoding="utf-8") as handle:
            json.dump([[0, 1], [2, 3]], handle)
        record["cover_path"] = str(cover_file.relative_to(out))
    path.write_text(json.dumps(record))
    return path


class TestCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def check(self, **kw):
        return ui.check(self.out, "dblp", "codeseg", kw.pop("digest", DIGEST), kw.pop("save_covers", True))

    def test_no_record_is_missing(self):
        self.assertEqual(self.check().state, "missing")

    def test_a_valid_record_without_a_seal_is_unsealed_and_reusable(self):
        write_unit(self.out)
        result = self.check()
        self.assertEqual(result.state, "unsealed")
        self.assertTrue(result.ok)

    def test_adopt_seals_it_and_a_later_check_is_verified(self):
        write_unit(self.out)
        self.assertEqual(ui.adopt(self.out, "dblp", "codeseg", DIGEST, True).state, "verified")
        self.assertTrue(ui.seal_path(self.out, "dblp", "codeseg").is_file())
        self.assertEqual(self.check().state, "verified")
        seal = ui.read_seal(self.out, "dblp", "codeseg")
        self.assertEqual(seal["record_sha256"], ui.sha256_file(ui.record_path(self.out, "dblp", "codeseg")))
        self.assertEqual(seal["sealed_by"], "adopted")

    def test_a_plain_check_writes_nothing(self):
        write_unit(self.out)
        self.check()
        self.assertFalse((self.out / ui.SEAL_DIR).exists())

    def test_a_changed_record_is_corrupt(self):
        path = write_unit(self.out)
        ui.adopt(self.out, "dblp", "codeseg", DIGEST, True)
        path.write_text(path.read_text().replace("0.5", "0.9"))  # still valid JSON, still the same digest
        result = self.check()
        self.assertEqual(result.state, "corrupt")
        self.assertFalse(result.ok)

    def test_a_changed_cover_is_corrupt(self):
        write_unit(self.out)
        ui.adopt(self.out, "dblp", "codeseg", DIGEST, True)
        with gzip.open(self.out / "covers" / "dblp" / "codeseg.json.gz", "wt", encoding="utf-8") as handle:
            json.dump([[0], [1]], handle)
        self.assertEqual(self.check().state, "corrupt")

    def test_a_truncated_record_is_corrupt(self):
        path = write_unit(self.out)
        path.write_text(path.read_text()[:10])
        self.assertEqual(self.check().state, "corrupt")

    def test_other_settings_are_stale_not_corrupt(self):
        write_unit(self.out, digest="e" * 16)
        self.assertEqual(self.check().state, "stale")

    def test_a_running_record_is_missing(self):
        write_unit(self.out, status="running")
        self.assertEqual(self.check().state, "missing")

    def test_a_missing_cover_means_the_unit_has_to_run(self):
        write_unit(self.out)
        (self.out / "covers" / "dblp" / "codeseg.json.gz").unlink()
        self.assertEqual(self.check().state, "missing")
        self.assertEqual(self.check(save_covers=False).state, "unsealed")

    def test_a_failed_record_is_a_result(self):
        write_unit(self.out, status="failed", cover=False)
        self.assertTrue(self.check().ok)

    def test_adopt_refuses_an_unreadable_cover(self):
        write_unit(self.out)
        (self.out / "covers" / "dblp" / "codeseg.json.gz").write_bytes(b"not gzip")
        result = ui.adopt(self.out, "dblp", "codeseg", DIGEST, True)
        self.assertEqual(result.state, "corrupt")
        self.assertFalse(ui.seal_path(self.out, "dblp", "codeseg").exists())

    def test_quarantine_renames_and_keeps_every_file(self):
        write_unit(self.out)
        ui.adopt(self.out, "dblp", "codeseg", DIGEST, True)
        moved = ui.quarantine(self.out, "dblp", "codeseg", "record hash differs")
        self.assertEqual(len(moved), 3)
        self.assertTrue(all(Path(m).is_file() for m in moved))
        self.assertFalse(ui.record_path(self.out, "dblp", "codeseg").exists())
        self.assertEqual(self.check().state, "missing")
        self.assertEqual(list((self.out / "runs" / "dblp").glob("*.json")), [])  # nothing a record reader would list


class TestScan(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = qs.Config(networks=["dblp"], methods=["codeseg", "fox"], seeds=[0, 1], replication="all",  # every seed of FOX/CliquePerc too
                              save_covers=True,
                             output_dir=self._tmp.name, cache_dir=self._tmp.name)
        self.plan = qs.build_plan(self.cfg, {})
        patcher = mock.patch.object(qs, "unit_digest", return_value=DIGEST)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def out(self, seed):
        return qs.unit_argv(self.cfg, "dblp", seed, qs.BY_KEY["codeseg"], None)[1]

    def test_scan_marks_what_is_certified_and_what_remains(self):
        write_unit(self.out(0), runner="codeseg")
        write_unit(self.out(1), runner="codeseg", digest="x" * 16)
        scan = qs.scan_plan(self.cfg, self.plan, None)
        states = {(e["seed"], e["method"]): e["state"] for e in scan}
        self.assertEqual(states[(0, "codeseg")], "unsealed")
        self.assertEqual(states[(1, "codeseg")], "stale")
        self.assertEqual(states[(0, "fox")], "missing")
        self.assertEqual(qs.summarize_scan(scan), {"units": 4, "done": 1, "stale": 1, "missing": 2})

    def test_the_scan_follows_the_files_not_the_plan_order(self):
        write_unit(self.out(1), runner="codeseg")
        first = {(e["seed"], e["method"]): e["done"] for e in qs.scan_plan(self.cfg, self.plan, None)}
        self.cfg.order = "seed,method,network"
        again = {(e["seed"], e["method"]): e["done"]
                 for e in qs.scan_plan(self.cfg, qs.build_plan(self.cfg, {}), None)}
        self.assertEqual(first, again)

    def test_an_unaudited_unit_is_certified_but_not_done(self):
        write_unit(self.out(0), runner="codeseg")

        class Ledger:
            records: dict = {}

            def audited(self, net, seed, method):
                return False

        scan = qs.scan_plan(self.cfg, self.plan, None, stage=Ledger())
        entry = next(e for e in scan if (e["seed"], e["method"]) == (0, "codeseg"))
        self.assertTrue(entry["audit_pending"])
        self.assertFalse(entry["done"])

    def test_a_tampered_record_is_found_by_the_scan(self):
        path = write_unit(self.out(0), runner="codeseg")
        qs.unit_check(self.cfg, "dblp", 0, qs.BY_KEY["codeseg"], None, adopt=True)
        path.write_text(path.read_text().replace("0.5", "0.6"))
        scan = qs.scan_plan(self.cfg, self.plan, None)
        self.assertEqual(next(e for e in scan if (e["seed"], e["method"]) == (0, "codeseg"))["state"], "corrupt")

    def test_the_queue_judges_units_with_the_same_check(self):
        path = write_unit(self.out(0), runner="codeseg")
        method = qs.BY_KEY["codeseg"]
        self.assertTrue(runqueue._unit_reusable(self.cfg, "dblp", 0, method, None))
        qs.unit_check(self.cfg, "dblp", 0, method, None, adopt=True)
        path.write_text(path.read_text().replace("0.5", "0.6"))
        self.assertFalse(runqueue._unit_reusable(self.cfg, "dblp", 0, method, None))

    def test_a_record_the_run_wrote_is_sealed(self):
        method = qs.BY_KEY["codeseg"]

        def fake_runner(argv):
            write_unit(self.out(0), runner="codeseg")

        with mock.patch("hedonic.experiments.overlapping.codeseg_reproduction.main", side_effect=fake_runner):
            rec = qs.run_one(self.cfg, "dblp", 0, method, None)
        self.assertEqual(rec["status"], "completed")
        seal = ui.read_seal(self.out(0), "dblp", "codeseg")
        self.assertEqual(seal["sealed_by"], "run")
        self.assertEqual(qs.unit_check(self.cfg, "dblp", 0, method, None).state, "verified")


class TestWorkerScan(unittest.TestCase):
    """The worker's scan marks certified units finished before anything runs, and sets corrupt ones aside."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = qs.Config(networks=["dblp"], methods=["codeseg", "fox"], seeds=[0, 1], replication="all",  # every seed of FOX/CliquePerc too
                              save_covers=True,
                             output_dir=self._tmp.name, cache_dir=self._tmp.name)
        self.plan = qs.build_plan(self.cfg, {})
        self.prog = {"plan": [{"network": n, "seed": s, "method": m.key, "label": m.label, "status": "pending",
                               "expected": 1.0} for n, s, m in self.plan]}
        patcher = mock.patch.object(qs, "unit_digest", return_value=DIGEST)
        patcher.start()
        self.addCleanup(patcher.stop)

    def out(self, seed, key="codeseg"):
        return qs.unit_argv(self.cfg, "dblp", seed, qs.BY_KEY[key], None)[1]

    def scan(self):
        return rm._scan(self.cfg, self.plan, None, None, self.prog, Path(self._tmp.name) / "progress.json")

    def test_certified_units_are_finished_before_anything_runs_and_carry_no_wall_time(self):
        write_unit(self.out(0), runner="codeseg")
        write_unit(self.out(1), runner="codeseg")
        self.scan()
        done = [u for u in self.prog["plan"] if u["status"] == "completed"]
        self.assertEqual(len(done), 2)
        self.assertTrue(all(u["origin"] == "scan" and "seconds" not in u for u in done))
        self.assertEqual(self.prog["scan"]["done"], 2)
        self.assertEqual(self.prog["scan"]["missing"], 2)
        self.assertEqual(self.prog["status"], "running")
        self.assertTrue(ui.seal_path(self.out(0), "dblp", "codeseg").is_file())  # adopted by the scan

    def test_a_corrupt_unit_is_moved_aside_and_runs_again(self):
        path = write_unit(self.out(0), runner="codeseg")
        qs.unit_check(self.cfg, "dblp", 0, qs.BY_KEY["codeseg"], None, adopt=True)
        path.write_text(path.read_text().replace("0.5", "0.6"))
        scan = self.scan()
        entry = next(e for e in scan if (e["seed"], e["method"]) == (0, "codeseg"))
        self.assertFalse(entry["done"])
        self.assertEqual(entry["state"], "missing")
        self.assertFalse(path.exists())
        self.assertTrue(list(path.parent.glob("codeseg.json.corrupt-*")))

    def test_the_final_check_flags_a_record_changed_during_the_run(self):
        path = write_unit(self.out(0), runner="codeseg")
        self.scan()
        self.assertEqual(rm._final_check(self.cfg, self.plan, None, self.prog, Path(self._tmp.name) / "progress.json"), [])
        path.write_text(path.read_text().replace("0.5", "0.6"))
        bad = rm._final_check(self.cfg, self.plan, None, self.prog, Path(self._tmp.name) / "progress.json")
        self.assertEqual(len(bad), 1)
        self.assertIn("hash", bad[0])
        self.assertEqual(self.prog["plan"][0]["status"], "failed")


class TestWorkerRun(unittest.TestCase):
    """The whole worker, detector mocked: it scans, runs only what is missing, and a second start runs nothing."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = qs.Config(networks=["dblp"], methods=["codeseg", "fox"], seeds=[0, 1], replication="all",  # every seed of FOX/CliquePerc too
                              save_covers=True,
                             output_dir=self._tmp.name, cache_dir=self._tmp.name, order="seed,method,network")
        self.entry = {"name": "t", "config": self.cfg.__dict__, "output_dir": self._tmp.name}
        for patcher in (mock.patch.object(qs, "unit_digest", return_value=DIGEST),
                        mock.patch.object(qs, "prepare", return_value=(None, {})),
                        mock.patch.object(qs, "finish", return_value=[]),
                        mock.patch.object(qs, "render", return_value="")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ran = []

    def out(self, seed, key):
        return qs.unit_argv(self.cfg, "dblp", seed, qs.BY_KEY[key], None)[1]

    def fake_run_one(self, cfg, net, seed, m, binary):
        self.ran.append((seed, m.key))
        write_unit(self.out(seed, m.key), runner=m.runner)
        rec = json.loads(ui.record_path(self.out(seed, m.key), net, m.runner).read_text())
        ui.seal(self.out(seed, m.key), net, m.runner, by="run")
        rec.update(network=net, method_key=m.key, seed=seed)
        return rec

    def start(self):
        prog: dict = {}
        with mock.patch.object(qs, "run_one", side_effect=self.fake_run_one), mock.patch("builtins.print"):
            status = rm._work("t", self.entry, prog)
        return status, json.loads((Path(self._tmp.name) / "progress.json").read_text())

    def test_only_the_missing_units_run_and_a_restart_runs_nothing(self):
        write_unit(self.out(0, "codeseg"), runner="codeseg")  # finished by an earlier run, never sealed
        write_unit(self.out(1, "fox"), runner="fox")
        status, prog = self.start()
        self.assertEqual(status, 0)
        self.assertEqual(self.ran, [(0, "fox"), (1, "codeseg")])  # seed-major order, only what had no record
        self.assertEqual(prog["status"], "completed")
        self.assertEqual(prog["scan"]["done"], 2)
        self.assertEqual({u["origin"] for u in prog["plan"]}, {"scan", "run"})
        self.assertEqual(prog["verified"]["uncertified"], [])
        self.ran.clear()
        status, prog = self.start()
        self.assertEqual(self.ran, [])
        self.assertEqual(prog["scan"]["done"], 4)
        self.assertEqual(prog["status"], "completed")

    def test_a_record_that_appears_after_the_scan_is_used_not_rerun(self):
        original = qs.scan_plan

        def scan_then_someone_finishes_a_unit(*args, **kwargs):
            result = original(*args, **kwargs)
            write_unit(self.out(0, "fox"), runner="fox")  # another process finishes it after the scan
            return result

        with mock.patch.object(qs, "scan_plan", side_effect=scan_then_someone_finishes_a_unit):
            status, prog = self.start()
        self.assertNotIn((0, "fox"), self.ran)
        self.assertEqual(next(u for u in prog["plan"] if (u["seed"], u["method"]) == (0, "fox"))["origin"], "reach")
        self.assertEqual(prog["status"], "completed")


class TestWorkerRunWithCovers(unittest.TestCase):
    """The worker on targets (network:cover) and the induced graph: scan, run what is missing, skip on restart."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = qs.Config(networks=["dblp:all", "dblp:top5000"], methods=["hedonic", "clique_percolation"], seeds=[0, 1], replication="all",  # every seed of FOX/CliquePerc too
                              nodes=0,
                             save_covers=True, induce="covered", output_dir=self._tmp.name, cache_dir=self._tmp.name,
                             order="seed,network,method")
        self.entry = {"name": "t", "config": self.cfg.__dict__, "output_dir": self._tmp.name}
        for patcher in (mock.patch.object(qs, "unit_digest", return_value=DIGEST),
                        mock.patch.object(qs, "prepare", return_value=(None, {})),
                        mock.patch.object(qs, "finish", return_value=[]),
                        mock.patch.object(qs, "render", return_value="")):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.ran = []

    def out(self, net, seed, key):
        return qs.unit_argv(self.cfg, net, seed, qs.BY_KEY[key], None)[1]

    def fake_run_one(self, cfg, net, seed, m, binary):
        self.ran.append((seed, net, m.key))
        out = self.out(net, seed, m.key)
        write_unit(out, runner=m.runner)  # the record sits under runs/<dataset>/, not under the cover's name
        rec = json.loads(ui.record_path(out, "dblp", m.runner).read_text())
        ui.seal(out, "dblp", m.runner, by="run")
        rec.update(network=net, method_key=m.key, seed=seed)
        return rec

    def start(self):
        with mock.patch.object(qs, "run_one", side_effect=self.fake_run_one), mock.patch("builtins.print"):
            status = rm._work("t", self.entry, {})
        return status, json.loads((Path(self._tmp.name) / "progress.json").read_text())

    def test_each_target_runs_in_its_own_folder_and_a_restart_runs_nothing(self):
        write_unit(self.out("dblp:top5000", 0, "clique_percolation"), runner="clique_percolation")  # finished by an earlier run
        status, prog = self.start()
        self.assertEqual(status, 0)
        self.assertEqual(len(self.ran), 8 - 1)
        self.assertEqual(self.ran[0], (0, "dblp:all", "hedonic"))  # seed, then target, then method
        self.assertEqual(self.ran[2], (0, "dblp:top5000", "hedonic"))
        self.assertNotIn((0, "dblp:top5000", "clique_percolation"), self.ran)
        self.assertEqual({u["network"] for u in prog["plan"]}, {"dblp:all", "dblp:top5000"})
        self.assertEqual(prog["status"], "completed")
        folders = {self.out(n, 0, "clique_percolation").parent.name for n in self.cfg.networks}
        self.assertEqual(folders, {"dblp-all-full-covered", "dblp-top5000-full-covered"})
        self.ran.clear()
        status, prog = self.start()
        self.assertEqual(self.ran, [])
        self.assertEqual(prog["scan"]["done"], 8)


class TestUnitDigest(unittest.TestCase):
    """The identity of an opt-in method is readable in a process that never provisioned runtimes (a scan, the queue view)."""

    def setUp(self):
        from hedonic.experiments.overlapping import extra_methods

        extra_methods.uninstall()  # as if nothing had been provisioned yet
        self.addCleanup(extra_methods.uninstall)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cfg = qs.Config(networks=["dblp:top5000"], methods=["ego_splitting"], seeds=[0], nodes=0,
                             induce="covered", output_dir=self._tmp.name, cache_dir=self._tmp.name)

    def test_ego_splitting_has_an_identity_without_provisioning(self):
        digest = qs.unit_digest(self.cfg, "dblp:top5000", 0, qs.BY_KEY["ego_splitting"], None)
        self.assertRegex(digest or "", r"^[0-9a-f]{64}$")
        self.assertEqual(digest, qs.unit_digest(self.cfg, "dblp:top5000", 0, qs.BY_KEY["ego_splitting"], None))


if __name__ == "__main__":
    unittest.main()
