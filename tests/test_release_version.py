"""tools/release_version.py: which tags publish, where, and as which version."""

import importlib.util
import unittest
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "tools" / "release_version.py"
_SPEC = importlib.util.spec_from_file_location("release_version", _PATH)
release_version = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_version)


class TestReleaseTags(unittest.TestCase):
    def test_release_and_candidates_go_to_pypi(self):
        self.assertEqual(release_version.target("v2.0.0", "2.0.0"), ("2.0.0", "pypi"))
        self.assertEqual(release_version.target("v2.0.0rc1", "2.0.0"), ("2.0.0rc1", "pypi"))

    def test_dev_tags_go_to_testpypi(self):
        self.assertEqual(
            release_version.target("v2.0.0.dev3", "2.0.0"), ("2.0.0.dev3", "testpypi")
        )

    def test_invalid_tags_are_refused(self):
        for tag in ("v2.0.1", "v2.0.0-rc1", "v2.0.0a1", "2.0.0", "v2.0", "v2.0.0rc"):
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                release_version.target(tag, "2.0.0")

    def test_the_project_version_is_a_release(self):
        import tomllib

        pyproject = _PATH.parent.parent / "pyproject.toml"
        version = tomllib.loads(pyproject.read_text())["project"]["version"]
        self.assertEqual(release_version.target(f"v{version}", version)[0], version)


if __name__ == "__main__":
    unittest.main()
