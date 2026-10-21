"""Records name the core by its results class: a core with the C core's three fingerprints, whatever its source, keeps
every identity 2.0.0rc11 wrote, byte for byte; a core with other fingerprints is a new class.

The rc11 values below were computed with the 2.0.0rc11 wheel from PyPI (its C core, c8684664..., and its Python
layer): every consumer of the identity must still produce them for the current build.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments import fingerprint, runtime
from hedonic.experiments.disjoint import sbm_sweep
from hedonic.experiments.disjoint import store
from hedonic.experiments.overlapping import calibration_search
from hedonic.experiments.overlapping import codeseg_reproduction as cr
from hedonic.experiments.overlapping import controlled_overlap as co
from hedonic.experiments.overlapping import controlled_overlap_config as coc
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import start_quality

ROOT = Path(__file__).resolve().parents[2]
RC11_CORE = "ead786fe4cede5140154d802b12e5d8efbf39278280b29f5a3d93a4e7a67b2a4"
RC11 = {
    "build_identity": {"hedonic": "2.0", "hedonic_core_source_sha256": RC11_CORE},
    "algorithm_identity": "community_hedonic/hedonic-2.0+core-ead786fe4cede514",
    "start_quality_code": "hedonic-2.0+core-ead786fe4cede514",
    # codeseg_reproduction --identity-only with BENCHMARK_ARGV and python-igraph 1.0.0
    "benchmark_config_digest": "76f904a013c60dc12fde4c36e4ba44877352b5fcaadf9530dddd99bc9f5aa3e2",
    "spectrum_tracked_digest": "18ca62d99ead6c879cdb1cb67e45b3f28eaad9d9d912a1b36f343d5f09efad13",
    # the disjoint store's Hedonic identity with igraph 1.0.0 (extension 62d309cb...), n_runs 4; and the legacy alias
    "store_hedonic_identity": "ed4a4d495714698607487542d36142e3c3e1887b01759bbf68213efd99c56c1c",
    "store_hedonic_legacy": "e95b906dfdde1b64cf22f4377ad65a4c0a637972ccb92d7a4e8512b269530b5a",
}
RC11_IGRAPH = {"igraph_c": "1.0.0", "igraph_extension_sha256": "62d309cb3d8459e572ffae5a5267adb4b2bc87d6f0cd6c8d7d8917d8c62f3857"}
BENCHMARK_ARGV = [
    "--datasets", "amazon", "--methods", "hedonic_local", "--seed", "0", "--timeout", "3600", "--resume",
    "--auto-download", "--snap-cache-dir", "/nonexistent/snap", "--output-dir", "/nonexistent/out",
    "--method-params", json.dumps({"hedonic_local": {"absolute_resolution": 0.001, "allow_isolation": True,
                                                     "max_memberships": -1}}),
    "--cover", "top5000", "--induce", "covered", "--network-root", "/nonexistent/net", "--save-covers", "--identity-only"]


def _clear_caches() -> None:
    for cached in (runtime.core_source_sha256, runtime.core_fingerprints, runtime.core_built_source_sha256,
                   runtime.core_sha256, start_quality.code_identity):
        cached.cache_clear()
    fingerprint._CACHE.clear()


@contextlib.contextmanager
def fake_core(built: str, fingerprints: dict[str, str] | None = None):
    """The loaded core as another build: another source digest and compiled bytes, and (optionally) other
    fingerprints. Every cache that holds the loaded core's identity is cleared before and after."""
    real = dict(zip(fingerprint.GRIDS, runtime.core_fingerprints()))
    values = {**real, **(fingerprints or {})}
    _clear_caches()
    try:
        with patch.object(runtime, "core_built_source_sha256", lambda: built), \
                patch.object(runtime, "core_sha256", lambda: "f" * 64), \
                patch.object(fingerprint, "fingerprint", lambda grid=fingerprint.GRID: values[grid]):
            yield
    finally:
        _clear_caches()


def identities() -> dict:
    """Every consumer's identity of the loaded core, as each compares it."""
    buffer = io.StringIO()
    with patch.object(cr.ig, "__version__", "1.0.0"), contextlib.redirect_stdout(buffer):
        cr.main(list(BENCHMARK_ARGV))
    rt = {**store.runtime_identity(), **RC11_IGRAPH}
    info = sbm_sweep.METHODS["Hedonic"]
    environment = co._environment_identity()
    return {
        "build_identity": runtime.build_identity(),
        "algorithm_identity": runtime.algorithm_identity(),
        "calibration_build": calibration_search.build(),
        "start_quality_code": start_quality.code_identity(),
        "benchmark_runtime": {k: v for k, v in cr.method_runtime_identity(["hedonic_local"]).items()
                              if k != "python_igraph"},
        "benchmark_config_digest": json.loads(buffer.getvalue().strip().splitlines()[-1])["config_digest"],
        "spectrum_tracked_digest": gs.tracked_digest(),
        "spectrum_compact": gs._compact_identity(gs.implementation_identity({"config_sha256": "c" * 64})),
        "controlled_core": environment["hedonic_core_source_sha256"],
        "store_hedonic_identity": store.spec_hash(store.method_identity("Hedonic", info, n_runs=4, runtime=rt)),
        "store_hedonic_legacy": store.spec_hash(store.method_identity("Hedonic", info, n_runs=4, runtime=rt,
                                                                      legacy=True)),
        "store_source": store.method_source("Hedonic", rt),
    }


class TestCoreClasses(unittest.TestCase):
    def test_the_loaded_core_is_a_listed_member_of_its_class(self):
        results = runtime.core_class()
        self.assertIsNotNone(results, f"no class has the loaded core's fingerprints {runtime.core_fingerprints()}: "
                                      "it computes other results (add a class only if that is intended)")
        built = runtime.core_built_source_sha256()
        self.assertIn(built, results["members"],
                      f"the loaded core's source {built} computes the class's results: add it to the class's members "
                      "in runtime.CORE_CLASSES with the release and the evidence")
        self.assertEqual(runtime.core_source_sha256(), results["records"])

    def test_the_table_is_consistent(self):
        for key, results in runtime.CORE_CLASSES.items():
            self.assertEqual(tuple(value.split(":")[0] for value in key), fingerprint.GRIDS)
            self.assertIn(results["records"], results["members"], "the digest records name is a member")
            for digest, member in results["members"].items():
                self.assertEqual(len(digest), 64)
                self.assertTrue(member["tag"].startswith("v") and member["evidence"])
        members = [digest for results in runtime.CORE_CLASSES.values() for digest in results["members"]]
        self.assertEqual(len(members), len(set(members)), "a source is in one class only")

    def test_every_equivalent_core_is_in_its_targets_class(self):
        for later, earlier in runtime.EQUIVALENT_CORES.items():
            classes = [results for results in runtime.CORE_CLASSES.values() if later in results["members"]]
            self.assertEqual(len(classes), 1, f"{later} is in no class")
            self.assertIn(earlier, classes[0]["members"])
            self.assertEqual(classes[0]["records"], earlier, "records name the alias's target")
            self.assertNotIn(earlier, runtime.EQUIVALENT_CORES)


class TestRc11Identities(unittest.TestCase):
    def test_every_consumer_names_what_rc11_named(self):
        got = identities()
        for key in ("build_identity", "algorithm_identity", "start_quality_code", "benchmark_config_digest",
                    "spectrum_tracked_digest", "store_hedonic_identity", "store_hedonic_legacy"):
            with self.subTest(key=key):
                self.assertEqual(got[key], RC11[key])
        self.assertEqual(got["calibration_build"], RC11["build_identity"])
        self.assertEqual(got["benchmark_runtime"], RC11["build_identity"])
        self.assertEqual(got["controlled_core"], RC11_CORE)
        self.assertEqual(got["store_source"], RC11_CORE)
        self.assertEqual(got["spectrum_compact"], {
            "schema_version": 1, "protocol_name": "overlapping-gt-spectrum-v1",
            "tracked_files_sha256": RC11["spectrum_tracked_digest"],
            "runtime": {**RC11["build_identity"], "algorithm_identity": RC11["algorithm_identity"]},
            "analysis_graph_pipeline": "bounded_then_common_undirected_simple_then_covered_induced_v1"})


class TestFakeCores(unittest.TestCase):
    """Another source with the same fingerprints keeps every identity; other fingerprints change it."""

    def test_same_fingerprints_other_source_same_identity_everywhere(self):
        real = identities()
        with fake_core("1" * 64):
            self.assertEqual(runtime.core_built_source_sha256(), "1" * 64)
            same = identities()
            versions = runtime.runtime_versions()
        self.assertEqual(same, real)
        self.assertEqual(versions["hedonic_core_built_source_sha256"], "1" * 64)  # provenance keeps the real one
        self.assertEqual(versions["hedonic_core_source_sha256"], RC11_CORE)
        self.assertEqual(versions["hedonic_core_fingerprints"], list(runtime.core_fingerprints()))

    def test_other_cover_or_audit_fingerprints_are_a_new_class(self):
        real = identities()
        for grid in (fingerprint.COVER_GRID, fingerprint.AUDIT_GRID):
            with self.subTest(grid=grid), fake_core("1" * 64, {grid: f"{grid}:" + "0" * 64}):
                other = identities()
                self.assertIsNone(runtime.core_class())
                self.assertEqual(other["build_identity"]["hedonic_core_source_sha256"], "1" * 64)
                for key in ("build_identity", "algorithm_identity", "calibration_build", "start_quality_code",
                            "benchmark_runtime", "benchmark_config_digest", "spectrum_tracked_digest",
                            "spectrum_compact", "controlled_core", "store_source"):
                    self.assertNotEqual(other[key], real[key], key)
                # The disjoint store holds partitions only: its rows stay keyed on the partition fingerprint.
                self.assertEqual(other["store_hedonic_identity"], real["store_hedonic_identity"])

    def test_other_partition_fingerprint_changes_the_store_too(self):
        real = identities()
        with fake_core("1" * 64, {fingerprint.GRID: f"{fingerprint.GRID}:" + "0" * 64}):
            other = identities()
        self.assertNotEqual(other["store_hedonic_identity"], real["store_hedonic_identity"])
        self.assertNotEqual(other["build_identity"], real["build_identity"])

    def test_the_identity_is_computed_on_first_use_not_at_import(self):
        script = ("import hedonic.experiments.runtime as r, hedonic.experiments.fingerprint as f\n"
                  "import hedonic.experiments.overlapping.calibrate, hedonic.experiments.overlapping.quickstart\n"
                  "import hedonic.experiments.overlapping.gt_spectrum, hedonic.experiments.overlapping.cover_audit\n"
                  "assert not f._CACHE and r.core_fingerprints.cache_info().currsize == 0, 'computed at import'\n"
                  "print(r.ALGORITHM_IDENTITY)")
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), RC11["algorithm_identity"])


class TestResultsSeries(unittest.TestCase):
    def test_a_later_package_version_keeps_the_series(self):
        real = identities()
        for version in ("2.0.0rc12", "2.0.1", "2.1.0", "2.1.0rc1"):
            with self.subTest(version=version), patch.object(runtime, "hedonic_version", return_value=version):
                _clear_caches()
                try:
                    self.assertEqual(runtime.release_series(), version[:3])  # provenance follows the version
                    self.assertEqual(identities(), real)
                    stored = {"implementation_identity": {"environment": co._environment_identity()}}
                finally:
                    _clear_caches()
            # The controlled study records the full version as provenance; whether its JSON is current never reads it.
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "controlled.json"
                path.write_text(json.dumps(stored))
                self.assertEqual(coc.json_build(path), "current")
        self.assertEqual(runtime.RESULTS_SERIES, "2.0")

    def test_the_disjoint_store_never_keys_rows_on_the_version(self):
        info = sbm_sweep.METHODS["Hedonic"]
        rt = store.runtime_identity()
        base = store.spec_hash(store.method_identity("Hedonic", info, n_runs=2, runtime=rt))
        self.assertEqual(store.spec_hash(store.method_identity("Hedonic", info, n_runs=2,
                                                               runtime={**rt, "hedonic": "2.1.0"})), base)


@unittest.skipIf(os.environ.get("HEDONIC_REBUILD_CORES") == "0", "HEDONIC_REBUILD_CORES=0")
class TestRebuildMembers(unittest.TestCase):
    """Each member of a class, rebuilt from its release tag, has the class's three fingerprints (about 8 s a member:
    git archive, setup.py build_ext, the three grids). It needs a git checkout with the tags and setuptools: an
    installed wheel skips it."""

    def test_every_member_rebuilt_from_its_tag_is_in_its_class(self):
        if shutil.which("git") is None or not (ROOT / ".git").exists() or importlib.util.find_spec("setuptools") is None:
            self.skipTest("needs a git checkout with the release tags and setuptools")
        grids = Path(fingerprint.__file__)
        for key, results in runtime.CORE_CLASSES.items():
            for digest, member in results["members"].items():
                tagged = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet",
                                         f"{member['tag']}^{{commit}}"], capture_output=True)
                if tagged.returncode:
                    self.skipTest(f"the tag {member['tag']} is not in this checkout")
                with self.subTest(member=member["tag"]), tempfile.TemporaryDirectory() as tmp:
                    archive = subprocess.run(["git", "-C", str(ROOT), "archive", member["tag"], "src", "native",
                                              "setup.py", "pyproject.toml"], capture_output=True, check=True)
                    subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
                    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace", "-q"], cwd=tmp,
                                   capture_output=True, check=True)
                    script = ("import importlib.util, json, sys; import hedonic._core as c\n"
                              "spec = importlib.util.spec_from_file_location('grids', sys.argv[1])\n"
                              "g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)\n"
                              "print(json.dumps([c.source_sha256, g.compute(), g.compute_covers(), g.compute_audit()]))")
                    out = subprocess.run([sys.executable, "-s", "-c", script, str(grids)], capture_output=True,
                                         text=True, check=True, cwd=tmp, env={**os.environ, "PYTHONPATH": f"{tmp}/src"})
                    built, *values = json.loads(out.stdout.strip().splitlines()[-1])
                    self.assertEqual(built, digest, "the tag's sources are the member's")
                    self.assertEqual(tuple(values), key)


if __name__ == "__main__":
    unittest.main()
