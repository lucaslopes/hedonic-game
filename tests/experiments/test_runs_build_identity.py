"""Records name the C core by the digest of its sources and hedonic by its release series, so a rebuilt wheel, another
platform or the next dev release keeps every finished record valid while the core's source is unchanged."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic import _core
from hedonic.experiments import runtime

ROOT = Path(__file__).resolve().parents[2]


class TestBuildIdentity(unittest.TestCase):
    def test_the_core_carries_the_digest_of_the_sources_it_was_built_from(self):
        setup = ROOT / "setup.py"
        if not setup.is_file():
            self.skipTest("no source tree (an installed wheel)")
        with patch("setuptools.setup"):
            digest = runpy.run_path(str(setup))["source_digest"](str(ROOT))
        self.assertEqual(_core.source_sha256, digest, "rebuild the extension: the C sources changed since it was built")
        self.assertEqual(runtime.core_source_sha256(), digest)

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
