"""Records name the core by the source digest of its results class and hedonic by its results series, so a rebuilt
wheel, another platform, the next release or another source that computes the same results keeps every finished
record valid (test_core_identity.py checks the classes)."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic import _core
from hedonic.experiments import runtime

ROOT = Path(__file__).resolve().parents[2]


def rust_source_digest(crate: Path) -> str:
    """native/rust/build.rs's digest of the Rust core: src/**/*.rs, build.rs and Cargo.toml, relative to the crate,
    each as path, length and bytes (CRLF as LF), in path order."""
    paths = sorted([str(p.relative_to(crate)).replace("\\", "/") for p in (crate / "src").rglob("*.rs")]
                   + ["build.rs", "Cargo.toml"])
    digest = hashlib.sha256()
    for rel in paths:
        data = (crate / rel).read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest()


class TestBuildIdentity(unittest.TestCase):
    def test_the_core_carries_the_digest_of_the_sources_it_was_built_from(self):
        if getattr(_core, "backend", None) != "rust":
            self.skipTest(f"the loaded core is not this checkout's Rust core: {_core.__name__} (HEDONIC_CORE)")
        crate = ROOT / "native" / "rust"
        if not (crate / "build.rs").is_file():
            self.skipTest("no source tree (an installed wheel)")
        digest = rust_source_digest(crate)
        self.assertEqual(_core.source_sha256, digest, "rebuild the extension: the Rust sources changed since")
        self.assertEqual(runtime.core_built_source_sha256(), digest)

    def test_an_equivalent_core_takes_the_earlier_identity(self):
        built = runtime.core_built_source_sha256()
        expected = runtime.EQUIVALENT_CORES.get(built, built)
        self.assertEqual(runtime.core_class()["records"], expected)
        self.assertEqual(runtime.core_source_sha256(), expected)
        self.assertEqual(runtime.build_identity()["hedonic_core_source_sha256"], expected)
        self.assertIn(expected[:16], runtime.algorithm_identity())
        versions = runtime.runtime_versions()
        self.assertEqual((versions["hedonic_core_source_sha256"], versions["hedonic_core_built_source_sha256"]),
                         (expected, built), "the record names the identity and keeps the built source as provenance")
        for later, earlier in runtime.EQUIVALENT_CORES.items():
            self.assertNotIn(earlier, runtime.EQUIVALENT_CORES, "an alias names a core that is not itself an alias")
            self.assertNotEqual(later, earlier)

    def test_the_release_series_drops_dev_rc_and_patch(self):
        for version, series in (("2.0.0.dev9", "2.0"), ("2.0.0rc1", "2.0"), ("2.0.3", "2.0"), ("2.1.0", "2.1")):
            with patch.object(runtime, "hedonic_version", return_value=version):
                self.assertEqual(runtime.release_series(), series)
                self.assertEqual(runtime.build_identity()["hedonic"], runtime.RESULTS_SERIES)  # provenance only

    def test_no_identity_names_the_full_version_or_the_compiled_bytes(self):
        identity = runtime.build_identity()
        self.assertEqual(identity, {"hedonic": runtime.RESULTS_SERIES,
                                    "hedonic_core_source_sha256": runtime.core_source_sha256()})
        self.assertNotIn("dev", runtime.algorithm_identity())
        self.assertNotIn(runtime.core_sha256()[:16], runtime.algorithm_identity())


if __name__ == "__main__":
    unittest.main()
