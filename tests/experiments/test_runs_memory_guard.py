"""A run's worker stops itself, and every process below it, when the machine runs out of memory."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from hedonic.experiments.overlapping import runmanager as rm

CHILD = textwrap.dedent('''
    import json, os, subprocess, sys, time
    from unittest.mock import patch
    from hedonic.experiments.overlapping import parallel, runmanager as rm

    out = sys.argv[1]
    entry = {"name": "guarded", "output_dir": out, "kind": "controlled"}
    prog = {"plan": [{"status": "running"}], "status": "running"}
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    open(os.path.join(out, "child.pid"), "w").write(str(sleeper.pid))
    with patch.object(parallel, "free_memory_percent", return_value=3.0), patch.object(parallel, "memory_floor", return_value=10.0):
        rm.start_memory_guard("guarded", entry, prog)
        time.sleep(30)  # the guard exits the process long before this
    sys.exit(0)
''')


class TestMemoryGuard(unittest.TestCase):
    def test_descendants_lists_children_and_grandchildren(self):
        before = set(rm.descendants(os.getpid()))  # other tests of the same process may have left children
        child = subprocess.Popen([sys.executable, "-c",
                                  "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"])
        try:
            deadline = time.time() + 10
            found: set[int] = set()
            while time.time() < deadline and len(found) < 2:
                found = set(rm.descendants(os.getpid())) - before
                time.sleep(0.1)
            self.assertIn(child.pid, found)
            self.assertEqual(len(found), 2)  # the child and its own child
        finally:
            for pid in set(rm.descendants(os.getpid())) - before:
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
            child.wait()

    def test_below_the_floor_the_worker_kills_its_tree_and_records_why(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = subprocess.run([sys.executable, "-c", CHILD, tmp], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 75, run.stderr)
            progress = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(progress["status"], "failed")
            self.assertIn("memory floor", progress["reason"])
            self.assertEqual(progress["plan"][0]["status"], "pending")  # the unit that was running is to be run again
            pid = int((Path(tmp) / "child.pid").read_text())
            for _ in range(50):  # the kill is delivered before the worker exits; allow the system a moment to reap
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(0.1)
            else:
                self.fail("the child of the stopped worker is still alive")

    def test_a_floor_of_zero_disables_the_guard(self):
        from unittest.mock import patch

        from hedonic.experiments.overlapping import parallel

        with patch.object(parallel, "memory_floor", return_value=0.0), patch.object(parallel, "free_memory_percent", return_value=0.0):
            self.assertIsNone(rm.start_memory_guard("x", {"output_dir": "."}, {}))  # no watcher, so nothing can exit

    def test_the_watch_ends_when_its_worker_returns(self):
        import threading

        from unittest.mock import patch

        from hedonic.experiments.overlapping import parallel

        with patch.object(parallel, "memory_floor", return_value=10.0), patch.object(parallel, "free_memory_percent", return_value=90.0), \
                patch.object(rm, "MEMORY_GUARD_SECONDS", 0.01):
            done = rm.start_memory_guard("x", {"output_dir": "."}, {})
            self.assertTrue(any(t.name == "memory-guard" for t in threading.enumerate()))
            done.set()
            for _ in range(100):
                if not any(t.name == "memory-guard" for t in threading.enumerate()):
                    break
                time.sleep(0.02)
            self.assertFalse(any(t.name == "memory-guard" for t in threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
