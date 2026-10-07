"""The release-test expectation follows the lock, never the ambient runtime."""

from pathlib import Path
import unittest
from unittest.mock import patch

from tests._version_contract import locked_lucas_igraph_version


class TestLockedLucasIgraphVersion(unittest.TestCase):
    def test_reads_changed_version_from_the_lock(self):
        for version in ("1.0.0.4", "1.0.0.5", "2.3.4.5"):
            with self.subTest(version=version):
                encoded = (
                    '[[package]]\nname = "numpy"\nversion = "99.0"\n'
                    f'[[package]]\nname = "lucas-igraph"\nversion = "{version}"\n'
                ).encode()
                with patch.object(Path, "read_bytes", return_value=encoded) as reader:
                    self.assertEqual(locked_lucas_igraph_version(), version)
                reader.assert_called_once_with()

    def test_missing_ambiguous_or_unversioned_package_fails_closed(self):
        for encoded in (
            b'[[package]]\nname = "igraph"\nversion = "1.0.0.3"\n',
            b'[[package]]\nname = "lucas-igraph"\n',
            b'[[package]]\nname = "lucas-igraph"\nversion = ""\n',
            (b'[[package]]\nname = "lucas-igraph"\nversion = "1.0.0.4"\n'
             b'[[package]]\nname = "lucas-igraph"\nversion = "1.0.0.5"\n'),
        ):
            with self.subTest(encoded=encoded):
                with patch.object(Path, "read_bytes", return_value=encoded):
                    with self.assertRaises(ValueError):
                        locked_lucas_igraph_version()

    def test_new_runtime_does_not_satisfy_an_old_lock(self):
        encoded = b'[[package]]\nname = "lucas-igraph"\nversion = "1.0.0.4"\n'
        with patch.object(Path, "read_bytes", return_value=encoded):
            with self.assertRaises(AssertionError):
                self.assertEqual("1.0.0.5", locked_lucas_igraph_version())

    def test_missing_lock_is_not_silently_replaced(self):
        with patch.object(Path, "read_bytes", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                locked_lucas_igraph_version()
