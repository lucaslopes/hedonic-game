"""The spectrum's equivalence table (``hedonic.experiments.spectrum_code``): code proven to compute what earlier code
computed files its records under the earlier digest, so editing a tracked file for identity or plumbing does not rerun
the ground-truth spectrum; any other edit does, and the table never lies about which is which."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import spectrum_code as sc
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import runqueue

RC11 = "18ca62d99ead6c879cdb1cb67e45b3f28eaad9d9d912a1b36f343d5f09efad13"


def raw() -> tuple[dict, str]:
    files = gs.tracked_file_hashes()
    return files, gs._implementation_digest(files, gs._runtime_identity())


def digest_of(files: dict) -> str:
    return gs._implementation_digest(files, gs._runtime_identity())


class TestTheLoadedCode(unittest.TestCase):
    def test_the_loaded_code_is_in_the_table(self):
        files, digest = raw()
        if digest in sc.ALIASES or digest in sc.ROOTS:
            return
        root = min(sc.ROOTS, key=lambda d: sum(sc.ROOTS[d]["files"].get(k) != v for k, v in files.items()))
        changed = sorted(k for k in set(files) | set(sc.ROOTS[root]["files"]) if files.get(k) != sc.ROOTS[root]["files"].get(k))
        self.fail(
            f"a file the ground-truth spectrum tracks changed ({', '.join(changed)}), so its records' digest is now "
            f"{digest}: every gt-robustness record would run again.\n"
            "If the change is identity or plumbing only (names, help text, docstrings, the identity machinery), add "
            "this entry to ALIASES in src/hedonic/experiments/spectrum_code.py, with the reason:\n\n"
            + sc.suggest(files, sc.ROOTS[root]["files"], digest, root)
            + "\n\nIf it can change a result, add the digest to ROOTS instead (its records start again).")

    def test_its_records_name_rc11s_digest(self):
        self.assertEqual(gs.tracked_digest(), RC11)
        identity = gs.implementation_identity({})
        self.assertEqual(identity["tracked_files_sha256"], RC11)
        self.assertEqual(identity["tracked_files"], sc.ROOTS[RC11]["files"])
        self.assertEqual(digest_of(identity["tracked_files"]), RC11, "the stored files reproduce the stored digest")
        files, digest = raw()  # the loaded files' own hashes are provenance
        self.assertEqual((identity["tracked_files_built"], identity["tracked_files_sha256_built"]), (files, digest))
        self.assertEqual(gs._compact_identity(identity)["tracked_files_sha256"], RC11)

    def test_no_drift_without_an_edit(self):
        self.assertEqual(gs.implementation_drift(gs.implementation_identity({})), [])


class TestTheTable(unittest.TestCase):
    def test_roots_reproduce_their_digests(self):
        for digest, root in sc.ROOTS.items():
            self.assertEqual(gs._implementation_digest(root["files"], gs._runtime_identity()), digest)

    def test_every_alias_ends_at_a_root_and_names_its_files(self):
        for later, entry in sc.ALIASES.items():
            self.assertIn(sc.canonical(later), sc.ROOTS, f"{later} does not lead to a root")
            self.assertNotEqual(later, entry["earlier"])
            self.assertTrue(entry["files"] and entry["why"] and "<" not in entry["why"])
            for path, (earlier, newer) in entry["files"].items():
                self.assertNotEqual(earlier, newer, path)
        self.assertFalse(set(sc.ALIASES) & set(sc.ROOTS))

    def test_the_table_is_not_a_tracked_file(self):
        self.assertNotIn("hedonic/experiments/spectrum_code.py", gs.tracked_file_hashes())

    def test_canonical_follows_the_chain_and_stops_on_a_cycle(self):
        table = {"c": {"earlier": "b"}, "b": {"earlier": "a"}, "x": {"earlier": "y"}, "y": {"earlier": "x"}}
        with patch.dict(sc.ALIASES, table, clear=True):
            self.assertEqual(sc.canonical("c"), "a")
            self.assertEqual(sc.canonical("a"), "a")
            self.assertIn(sc.canonical("x"), {"x", "y"})
        self.assertIsNone(sc.canonical(None))


class TestLaterCode(unittest.TestCase):
    """What 2.1 does: ``hedonic/__init__.py`` changes and a module joins ``hedonic/*.py``."""

    def later_files(self) -> dict:
        files = dict(gs.tracked_file_hashes())
        files["hedonic/__init__.py"] = "1" * 64
        files["hedonic/_rust.py"] = "2" * 64
        return dict(sorted(files.items()))

    def entry(self, files: dict) -> dict:
        _, now = raw()
        current = gs._tracked(gs._runtime_identity())[0]
        return {"earlier": RC11,
                "files": {path: [sc.ROOTS[RC11]["files"].get(path), files.get(path)]
                          for path in set(files) | set(current) if files.get(path) != sc.ROOTS[RC11]["files"].get(path)},
                "why": "test"}

    def test_an_entry_maps_later_code_to_the_root(self):
        files = self.later_files()
        later = digest_of(files)
        self.assertNotEqual(later, RC11)
        with patch.object(gs, "tracked_file_hashes", lambda: files), patch.dict(sc.ALIASES, {later: self.entry(files)}):
            self.assertEqual(gs.tracked_digest(), RC11)
            identity = gs.implementation_identity({})
            self.assertEqual(identity["tracked_files"], sc.ROOTS[RC11]["files"])  # the added module is not filed
            self.assertEqual(identity["tracked_files_sha256_built"], later)
            self.assertEqual(gs.canonical_digest(later), RC11)
        with patch.object(gs, "tracked_file_hashes", lambda: files):  # without its entry it is new code
            self.assertEqual(gs.tracked_digest(), later)

    def test_a_chain_of_entries_is_followed(self):
        files = self.later_files()
        current, mid = raw()
        later = digest_of(files)
        step = {"earlier": mid, "why": "test",
                "files": {"hedonic/__init__.py": [current["hedonic/__init__.py"], "1" * 64],
                          "hedonic/_rust.py": [None, "2" * 64]}}
        with patch.object(gs, "tracked_file_hashes", lambda: files), patch.dict(sc.ALIASES, {later: step}):
            self.assertEqual(gs.tracked_digest(), RC11)  # 2.1 -> this code -> rc11

    def test_an_entry_that_does_not_reproduce_is_ignored(self):
        files = self.later_files()
        later = digest_of(files)
        wrong = self.entry(files)
        wrong["files"]["hedonic/__init__.py"] = [wrong["files"]["hedonic/__init__.py"][0], "3" * 64]  # other bytes
        with patch.object(gs, "tracked_file_hashes", lambda: files), patch.dict(sc.ALIASES, {later: wrong}):
            self.assertEqual(gs.tracked_digest(), later)
        lying = self.entry(files)
        lying["earlier"] = "e" * 64  # undoing the files does not give this digest
        with patch.object(gs, "tracked_file_hashes", lambda: files), patch.dict(sc.ALIASES, {later: lying}):
            self.assertEqual(gs.tracked_digest(), later)

    def test_drift_during_a_run(self):
        identity = gs.implementation_identity({})
        files = self.later_files()
        later = digest_of(files)
        with patch.object(gs, "tracked_file_hashes", lambda: files):
            self.assertEqual(gs.implementation_drift(identity), ["hedonic/__init__.py", "hedonic/_rust.py"])
            with patch.dict(sc.ALIASES, {later: self.entry(files)}):
                self.assertEqual(gs.implementation_drift(identity), [])  # an equivalent edit is no drift


class TestTheQueue(unittest.TestCase):
    def test_records_of_aliased_code_count_as_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            job = folder / "runs" / "amazon-all"
            job.mkdir(parents=True)
            for i, digest in enumerate((RC11, *sc.ALIASES, "e" * 64)):
                (job / f"{i}.json").write_text(json.dumps({"protocol_identity": {"tracked_files_sha256": digest}}))
            lines = runqueue._spectrum_earlier(folder, gs.tracked_digest())
        self.assertEqual(len(lines), 1)
        self.assertIn("1 records on disk from earlier code", lines[0])


if __name__ == "__main__":
    unittest.main()
