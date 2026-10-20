from __future__ import annotations

import json
import gzip
import sys
import io
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from unittest import mock

from hedonic.experiments.overlapping import codeseg_setup
from hedonic.experiments.overlapping.snap import load_snap_dataset
from hedonic.experiments.overlapping import snap as snap_module


class TestBenchmarkRuntimes(unittest.TestCase):
    def test_a_runtime_the_benchmark_prepares_is_written_where_it_reads_it(self):
        from hedonic.experiments.overlapping import quickstart

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HEDONIC_CODESEG_CACHE", None)
            os.environ.pop("HEDONIC_CODESEG_MANIFEST", None)
            cache = Path(tmp) / "codeseg"  # the runtime folder itself, as config.codeseg_cache_dir() gives it
            missing = quickstart.ensure_runtimes(["clique_percolation"], cache, lambda *a: None)
            self.assertEqual(missing, {})  # it went to cache/codeseg/setup_manifest.json and was reported unavailable
            self.assertFalse((cache / "codeseg").exists())
            self.assertEqual(json.loads((cache / "setup_manifest.json").read_text())["methods"]["clique_percolation"]
                             ["status"], "ready")
            self.assertNotIn("HEDONIC_CODESEG_CACHE", os.environ)


class TestCoDeSEGSetup(unittest.TestCase):
    def test_default_setup_scope_is_full_snap_dblp_only(self):
        self.assertEqual(codeseg_setup.DEFAULT_DATASETS, ("dblp",))
        self.assertIn("friendster", codeseg_setup.SNAP_DATASETS)
        for method in ("nise", "sse", "qoce", "svi", "essc"):
            self.assertIn(method, codeseg_setup.DEFAULT_METHODS)

    def test_snap_source_defaults_to_upstream_master_branch(self):
        parser = codeseg_setup._setup_parser()
        args = parser.parse_args([])
        self.assertEqual(args.snap_ref, "master")

    def test_fox_compiler_override_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            compiler = Path(temporary) / "g++-test"
            compiler.write_text("#!/bin/sh\nexit 0\n")
            compiler.chmod(0o755)
            self.assertEqual(
                codeseg_setup._fox_openmp_compiler(compiler), compiler.resolve()
            )

    def test_native_cmake_builds_use_release_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            codeseg = cache / "upstream" / "CoDeSEG"
            snap = cache / "upstream" / "snap"
            (codeseg / ".git").mkdir(parents=True)
            (codeseg / "code_c++" / "CoDeSEG").mkdir(parents=True)
            (codeseg / "code_c++" / "fox").mkdir(parents=True)
            (snap / ".git").mkdir(parents=True)
            (snap / "examples" / "bigclam").mkdir(parents=True)
            commands = []

            def fake_build(command, **_kwargs):
                commands.append(command)
                return {"status": "ready", "returncode": 0}

            with mock.patch.object(codeseg_setup, "_run_build", side_effect=fake_build), mock.patch.object(
                codeseg_setup, "_fox_openmp_compiler", return_value=None
            ), mock.patch.object(codeseg_setup.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"):
                result = codeseg_setup._native_runtime(
                    cache_dir=cache,
                    offline=True,
                    dry_run=False,
                    build_native=True,
                    codeseg_bin=None,
                    bigclam_bin=None,
                    fox_bin=None,
                    fox_cxx=None,
                    upstream_url="codeseg",
                    upstream_ref="main",
                    snap_url="snap",
                    snap_ref="master",
                    timeout=1,
                )

            cmake_configures = [command for command in commands if command[:2] == ["/usr/bin/cmake", "-S"]]
            self.assertEqual(len(cmake_configures), 2)
            self.assertTrue(
                all("-DCMAKE_BUILD_TYPE=Release" in command for command in cmake_configures)
            )
            self.assertEqual(result["upstream_codeseg"]["build_codeseg"]["build_type"], "Release")
            self.assertEqual(result["upstream_codeseg"]["build_fox"]["build_type"], "Release")

    def test_setup_dry_run_plans_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            root = Path(temporary)
            manifest = root / "setup.json"
            out = io.StringIO()
            with redirect_stdout(out):
                code = codeseg_setup.setup_main(
                    ["--all", "--dry-run", "--offline", "--json", "--cache-dir", str(root / "cache"),
                     "--network-root", str(root / "nets"), "--manifest", str(manifest)]
                )
            self.assertEqual(code, 0)
            self.assertFalse(manifest.exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), [])
            plan = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertTrue(plan["dry_run"])
            self.assertEqual(plan["pairs"], list(codeseg_setup.PAPER_PAIRS))
            self.assertEqual(plan["methods"], list(codeseg_setup.PAPER_BASELINES))

    def test_a_bare_setup_prepares_the_paper_baselines(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            out = io.StringIO()
            with redirect_stdout(out):
                codeseg_setup.setup_main(["--dry-run", "--json", "--cache-dir", str(Path(temporary) / "cache")])
            plan = json.loads(out.getvalue().strip().splitlines()[-1])
            self.assertEqual((plan["pairs"], plan["methods"]), ([], list(codeseg_setup.PAPER_BASELINES)))

    def test_setup_and_the_runs_read_one_cache_dir_the_same_way(self):
        from hedonic.experiments.config import codeseg_cache_dir

        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ):
            os.environ.pop("HEDONIC_CODESEG_CACHE", None)
            out = io.StringIO()
            with redirect_stdout(out):
                codeseg_setup.setup_main(["--dry-run", "--cache-dir", str(Path(temporary) / "cache")])
            expected = Path(temporary) / "cache" / "codeseg"
            self.assertIn(f"cache {expected}", out.getvalue())
            self.assertEqual(codeseg_cache_dir(Path(temporary) / "cache"), expected)  # `hedonic run exp --cache-dir`
            config = Path(temporary) / "config.toml"
            config.write_text(f'[defaults]\ncache_dir = "{Path(temporary) / "cache"}"\n')
            os.environ["HEDONIC_CONFIG"] = str(config)
            self.assertEqual(codeseg_cache_dir().resolve(), expected.resolve())  # the user config's cache_dir

    def test_required_tools_name_what_each_method_needs(self):
        rows = {row["tool"]: row for row in codeseg_setup.required_tools(
            ["fox", "bigclam", "angel", "codeseg", "hedonic"], build=True, fox_cxx=None, manifest={})}
        self.assertIn("uv", rows)
        self.assertIn("angel", rows["uv"]["for"])
        self.assertIn("openmp-c++", rows)
        self.assertEqual(rows["openmp-c++"]["for"], ["fox"])
        self.assertTrue(all(row["hint"] for row in rows.values()))
        self.assertFalse(any("hedonic" in row["for"] for row in rows.values()))
        self.assertFalse(codeseg_setup.needs_setup("hedonic"))
        self.assertTrue(codeseg_setup.needs_setup("fox"))

    def test_a_missing_tool_stops_a_real_setup_with_its_hint(self):
        missing = [{"tool": "cmake", "for": ["fox"], "path": None, "hint": "install CMake"}]
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ), \
                patch.object(codeseg_setup, "required_tools", return_value=missing):
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = codeseg_setup.setup_main(["--methods", "fox", "--cache-dir", str(Path(temporary) / "cache")])
            self.assertEqual(code, 2)
            self.assertIn("install CMake", err.getvalue())
            self.assertFalse((Path(temporary) / "cache" / "setup_manifest.json").exists())

    def test_catalogue_fallback_is_lazy_and_cached(self):
        class FakeNetwork:
            def edges(self, cast=int):
                return [(10, 11), (11, 12)]

            def nodes(self):
                return [10, 11, 12]

        class FakeGroundTruth:
            def communities(self):
                return [[10, 11], [11, 12]]

        snap_module = types.ModuleType("networks.snap")
        calls: list[tuple[str, str]] = []

        def load(name: str, **_kwargs):
            calls.append(("load", name))
            return FakeNetwork()

        def load_ground_truth(name: str, variant: str = "all"):
            calls.append(("ground_truth", f"{name}:{variant}"))
            return FakeGroundTruth()

        snap_module.load = load  # type: ignore[attr-defined]
        snap_module.load_ground_truth = load_ground_truth  # type: ignore[attr-defined]
        networks_module = types.ModuleType("networks")
        networks_module.snap = snap_module  # type: ignore[attr-defined]
        previous_networks = sys.modules.get("networks")
        previous_snap = sys.modules.get("networks.snap")
        sys.modules["networks"] = networks_module
        sys.modules["networks.snap"] = snap_module
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = load_snap_dataset(
                    "dblp",
                    cover_variant="all",
                    data_root=root / "missing-networks",
                    cache_dir=root / "cache",
                )
                self.assertEqual(first.report["source_kind"], "snap_catalog")
                self.assertEqual(first.graph.vcount(), 3)
                self.assertEqual(first.cover, [[0, 1], [1, 2]])
                calls_after_first = len(calls)
                second = load_snap_dataset(
                    "dblp",
                    cover_variant="all",
                    data_root=root / "missing-networks",
                    cache_dir=root / "cache",
                )
                self.assertEqual(second.report["source_kind"], "validated_normalized_cache")
                self.assertEqual(len(calls), calls_after_first)
        finally:
            if previous_networks is None:
                sys.modules.pop("networks", None)
            else:
                sys.modules["networks"] = previous_networks
            if previous_snap is None:
                sys.modules.pop("networks.snap", None)
            else:
                sys.modules["networks.snap"] = previous_snap

    def test_official_http_fallback_works_without_catalogue_package(self):
        previous_networks = sys.modules.pop("networks", None)
        previous_snap = sys.modules.pop("networks.snap", None)
        original_downloader = snap_module._download_catalog_file

        def fake_download(url: str, destination: Path, **_kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = b"10 11\n11 12\n" if "ungraph" in url else b"10 11\n11 12\n"
            if "cmty" in url:
                payload = b"10 11\n11 12\n"
            with gzip.open(destination, "wb") as stream:
                stream.write(payload)

        snap_module._download_catalog_file = fake_download
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                loaded = load_snap_dataset(
                    "dblp",
                    cover_variant="all",
                    data_root=root / "missing-networks",
                    cache_dir=root / "cache",
                )
                self.assertEqual(loaded.report["source_kind"], "snap_catalog")
                self.assertEqual(loaded.report["catalog_backend"], "official_snap_http")
                self.assertEqual(loaded.graph.vcount(), 3)
        finally:
            snap_module._download_catalog_file = original_downloader
            if previous_networks is not None:
                sys.modules["networks"] = previous_networks
            if previous_snap is not None:
                sys.modules["networks.snap"] = previous_snap


if __name__ == "__main__":
    unittest.main()
