"""Release failures must stop before credentials or upload are used."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'release.sh'
SHA = '1' * 40


class TestReleaseHelper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = dict(os.environ)
        for key in list(self.env):
            if key.startswith(('PYPI_', 'UV_PUBLISH_')):
                self.env.pop(key)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.env['PATH'] = str(self.bin) + os.pathsep + self.env['PATH']
        self.env['MOCK_SHA'] = SHA
        self.env['MOCK_CONCLUSION'] = 'success'
        self.mock('gh', '''#!/usr/bin/env python3
import json, os, sys
if sys.argv[1:3] == ['run', 'view']:
    print(json.dumps(dict(headSha=os.environ['MOCK_SHA'], conclusion=os.environ['MOCK_CONCLUSION'], workflowName='Build', url='https://example.invalid/run')))
else:
    raise SystemExit('unexpected network operation')
''')
        self.mock('uv', '#!/bin/sh\necho "UPLOAD MUST NOT RUN" >&2\nexit 90\n')

    def mock(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def function(self, command, *args):
        # Keep the actual function definitions; omit only the CLI dispatch.
        source = SCRIPT.read_text().rsplit('main "$@"', 1)[0]
        harness = self.root / 'functions.sh'
        harness.write_text(source + '\n' + command + '\n')
        return subprocess.run(['bash', str(harness), *map(str, args)],
                              env=self.env, text=True, capture_output=True)

    def cli(self, *args):
        return subprocess.run(['bash', str(SCRIPT), 'publish-pypi', *args],
                              env=self.env, text=True, capture_output=True)

    def wheel(self, metadata):
        path = self.root / 'hedonic-0.1.1-py3-none-any.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('hedonic-0.1.1.dist-info/METADATA', metadata)
        return path

    def test_metadata_uses_headers_not_readme_body(self):
        wheel = self.wheel('Name: hedonic\nVersion: 0.1.1\n\nName: other\nVersion: 9\n')
        result = self.function('distribution_metadata "$1"', wheel)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'hedonic\t0.1.1')

    def test_duplicate_metadata_headers_are_rejected(self):
        wheel = self.wheel('Name: hedonic\nName: other\nVersion: 0.1.1\n')
        self.assertNotEqual(self.function('distribution_metadata "$1"', wheel).returncode, 0)

    def test_existing_index_file_requires_matching_bytes(self):
        artifact = self.root / 'artifact.whl'
        artifact.write_bytes(b'release bytes')
        manifest = self.root / 'index.tsv'
        manifest.write_text('artifact.whl\t' + hashlib.sha256(artifact.read_bytes()).hexdigest() + '\n')
        self.assertEqual(self.function('verify_index_hash "$1" "$2"', artifact, manifest).returncode, 0)
        artifact.write_bytes(b'different bytes')
        result = self.function('verify_index_hash "$1" "$2"', artifact, manifest)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('SHA-256 mismatch', result.stderr)

    def test_run_id_does_not_substitute_for_expected_commit(self):
        result = self.cli('--repo', 'owner/repo', '--run-id', '42', '--publish')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('pass --commit', result.stderr)

    def test_mismatched_commit_stops_before_download_and_upload(self):
        result = self.cli('--repo', 'owner/repo', '--run-id', '42',
                          '--commit', '2' * 40, '--publish')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match expected', result.stderr)
        self.assertNotIn('UPLOAD MUST NOT RUN', result.stderr)

    def test_failed_run_stops_before_download_and_upload(self):
        self.env['MOCK_CONCLUSION'] = 'failure'
        result = self.cli('--repo', 'owner/repo', '--run-id', '42', '--commit', SHA, '--publish')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not success', result.stderr)
        self.assertNotIn('UPLOAD MUST NOT RUN', result.stderr)

    def test_unproven_saved_artifacts_are_rejected(self):
        self.mock('gh', self.bin.joinpath('gh').read_text().replace(
            "else:\n    raise SystemExit('unexpected network operation')",
            "elif sys.argv[1:3] == ['run', 'download']:\n    pass\nelse:\n    raise SystemExit('unexpected network operation')"))
        self.wheel('Name: hedonic\nVersion: 0.1.1\n')
        result = self.cli('--repo', 'owner/repo', '--run-id', '42', '--commit', SHA,
                          '--artifacts-dir', str(self.root), '--publish')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('do not match the exact Actions run', result.stderr)
        self.assertNotIn('UPLOAD MUST NOT RUN', result.stderr)
