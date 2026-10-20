"""What one thread prints can be silenced or captured without touching another thread's output (``threadio``)."""

from __future__ import annotations

import io
import sys
import threading
import unittest
from contextlib import redirect_stdout

from hedonic.experiments.overlapping import threadio


class TestRedirected(unittest.TestCase):
    def test_a_thread_silenced_while_the_main_thread_prints_does_not_swallow_the_main_threads_lines(self):
        seen = io.StringIO()
        inside, done = threading.Event(), threading.Event()

        def quiet_worker():
            with threadio.redirected():
                print("a planner's own error")
                inside.set()
                done.wait(5)

        with redirect_stdout(seen):
            worker = threading.Thread(target=quiet_worker)
            worker.start()
            inside.wait(5)
            print("reopen: hedonic run attach q")  # the view closed while the builder thread was checking state
            done.set()
            worker.join()
        self.assertEqual(seen.getvalue(), "reopen: hedonic run attach q\n")

    def test_a_thread_can_capture_its_output_into_a_buffer(self):
        seen, captured = io.StringIO(), io.StringIO()

        def worker():
            with threadio.redirected(stdout=captured):
                print('{"config_digest": "abc"}')
                print("to stderr", file=sys.stderr)

        with redirect_stdout(seen):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            print("main")
        self.assertEqual(captured.getvalue(), '{"config_digest": "abc"}\n')
        self.assertEqual(seen.getvalue(), "main\n")

    def test_nested_redirects_restore_the_outer_one(self):
        outer, inner, seen = io.StringIO(), io.StringIO(), io.StringIO()
        with redirect_stdout(seen):
            with threadio.redirected(stdout=outer):
                print("outer 1")
                with threadio.redirected(stdout=inner):
                    print("inner")
                print("outer 2")
            print("plain")
        self.assertEqual((outer.getvalue(), inner.getvalue(), seen.getvalue()), ("outer 1\nouter 2\n", "inner\n", "plain\n"))


if __name__ == "__main__":
    unittest.main()
