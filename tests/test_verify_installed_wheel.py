"""Regression tests for the hosted uploaded-wheel verifier."""

from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_installed_wheel.py"
SPEC = spec_from_file_location("verify_installed_wheel", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFY = module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class _Distribution:
    def __init__(self, package_init: Path) -> None:
        self.package_init = package_init

    def locate_file(self, path: str) -> Path:
        assert path == "hedonic/__init__.py"
        return self.package_init


class TestVerifyInstalledWheel(unittest.TestCase):
    def test_main_checks_import_origin_before_native_behavior(self):
        with (
            patch.object(VERIFY.importlib.metadata, "version", return_value="0.1.1"),
            patch.object(
                VERIFY,
                "verify_import_origin",
                side_effect=AssertionError("checkout shadow"),
            ) as verify_origin,
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(AssertionError, "checkout shadow"):
                VERIFY.main()
        verify_origin.assert_called_once_with()

    def test_import_origin_accepts_installed_package(self):
        package_init = Path("/tmp/site-packages/hedonic/__init__.py")
        distribution = _Distribution(package_init)
        imported = SimpleNamespace(__file__=str(package_init))

        with (
            patch.object(VERIFY.importlib.metadata, "distribution", return_value=distribution),
            patch.object(VERIFY, "hedonic", imported),
        ):
            VERIFY.verify_import_origin()

    def test_import_origin_rejects_checkout_shadow(self):
        distribution = _Distribution(Path("/tmp/site-packages/hedonic/__init__.py"))
        imported = SimpleNamespace(
            __file__="/tmp/repository/src/hedonic/__init__.py"
        )

        with (
            patch.object(VERIFY.importlib.metadata, "distribution", return_value=distribution),
            patch.object(VERIFY, "hedonic", imported),
        ):
            with self.assertRaisesRegex(
                AssertionError, "outside the installed distribution"
            ):
                VERIFY.verify_import_origin()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
