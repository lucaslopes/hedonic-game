"""Records name the C core by the digest of its sources and hedonic by its release series, so a rebuilt wheel, another
platform or the next dev release keeps every finished record valid while the core's source is unchanged."""

from __future__ import annotations

import hashlib
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic import _core
from hedonic.experiments import runtime

ROOT = Path(__file__).resolve().parents[2]


def rust_source_digest(crate: Path) -> str:
    """rust/build.rs's digest of the Rust core: src/**/*.rs, build.rs, Cargo.toml and Cargo.lock (when present),
    relative to the crate, in the format of setup.py's source_digest."""
    lock = ["Cargo.lock"] if (crate / "Cargo.lock").is_file() else []
    paths = sorted([str(p.relative_to(crate)).replace("\\", "/") for p in (crate / "src").rglob("*.rs")]
                   + ["build.rs", "Cargo.toml"] + lock)
    digest = hashlib.sha256()
    for rel in paths:
        data = (crate / rel).read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return digest.hexdigest()


class TestBuildIdentity(unittest.TestCase):
    def test_the_core_carries_the_digest_of_the_sources_it_was_built_from(self):
        if getattr(_core, "backend", "c") == "rust":
            crate = ROOT / "rust"
            if not (crate / "build.rs").is_file():
                self.skipTest("no source tree (an installed wheel)")
            digest = rust_source_digest(crate)
            self.assertEqual(_core.source_sha256, digest, "rebuild the extension: the Rust sources changed since")
            self.assertEqual(runtime.core_built_source_sha256(), digest)
            return
        setup = ROOT / "setup.py"
        if not setup.is_file():
            self.skipTest("no source tree (an installed wheel)")
        try:
            import setuptools  # noqa: F401
        except ImportError:
            self.skipTest("setuptools is not installed (an installed wheel)")
        with patch("setuptools.setup"):
            build = runpy.run_path(str(setup))
        digest = build["source_digest"](str(ROOT))
        compiled = {str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "native").rglob("*")
                    if p.suffix in (".c", ".h") and "tests" not in p.relative_to(ROOT / "native").parts}
        self.assertEqual(set(build["source_paths"](str(ROOT))), compiled, "every C source of the core is in its digest")
        self.assertEqual(_core.source_sha256, digest, "rebuild the extension: the C sources changed since it was built")
        self.assertEqual(runtime.core_built_source_sha256(), digest)

    def test_an_equivalent_core_takes_the_earlier_identity(self):
        built = runtime.core_built_source_sha256()
        expected = runtime.EQUIVALENT_CORES.get(built, built)
        self.assertEqual(runtime.core_source_sha256(), expected)
        self.assertEqual(runtime.build_identity()["hedonic_core_source_sha256"], expected)
        self.assertIn(expected[:16], runtime.algorithm_identity())
        versions = runtime.runtime_versions()
        self.assertEqual((versions["hedonic_core_source_sha256"], versions["hedonic_core_built_source_sha256"]),
                         (expected, built), "the record names the identity and keeps the built source as provenance")
        for later, earlier in runtime.EQUIVALENT_CORES.items():
            self.assertNotIn(earlier, runtime.EQUIVALENT_CORES, "an alias names a core that is not itself an alias")
            self.assertNotEqual(later, earlier)

    def test_the_series_drops_dev_rc_and_patch(self):
        for version, series in (("2.0.0.dev9", "2.0"), ("2.0.0rc1", "2.0"), ("2.0.3", "2.0"), ("2.1.0", "2.1")):
            with patch.object(runtime, "hedonic_version", return_value=version):
                self.assertEqual(runtime.release_series(), series)

    def test_no_identity_names_the_full_version_or_the_compiled_bytes(self):
        identity = runtime.build_identity()
        self.assertEqual(identity, {"hedonic": runtime.release_series(),
                                    "hedonic_core_source_sha256": runtime.core_source_sha256()})
        self.assertNotIn("dev", runtime.algorithm_identity())
        self.assertNotIn(runtime.core_sha256()[:16], runtime.algorithm_identity())


if __name__ == "__main__":
    unittest.main()
