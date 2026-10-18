"""The home screen of a bare ``hedonic``: only in a terminal, a header from local state, a menu that runs the
command it shows, views that come back, and a pty run of the real loop."""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from hedonic.experiments import CLI, home
from hedonic.experiments.overlapping import tui

ANSI = re.compile(r"\033\[[0-9;?<]*[A-Za-z]")


class TestWhenItOpens(unittest.TestCase):
    def test_outside_a_terminal_a_bare_hedonic_prints_the_help(self):
        out = io.StringIO()
        with patch.object(home, "main") as screen, redirect_stdout(out):
            self.assertEqual(CLI.hedonic_main([]), 0)
        screen.assert_not_called()
        self.assertIn("usage: hedonic COMMAND", out.getvalue())

    def test_in_a_terminal_it_opens_unless_the_plain_view_is_asked_for(self):
        with patch.object(tui, "interactive", return_value=True):
            with patch.dict(os.environ, {"HEDONIC_PLAIN_VIEW": ""}):
                with patch.object(home, "main", return_value=0) as screen:
                    self.assertEqual(CLI.hedonic_main([]), 0)
                screen.assert_called_once()
            with patch.dict(os.environ, {"HEDONIC_PLAIN_VIEW": "1"}), patch.object(home, "main") as screen, \
                    redirect_stdout(io.StringIO()):
                CLI.hedonic_main([])
            screen.assert_not_called()
            with patch.object(home, "main") as screen, redirect_stdout(io.StringIO()):
                CLI.hedonic_main(["--help"])  # --help is always text
            screen.assert_not_called()


class TestMenu(unittest.TestCase):
    def test_runs_come_first_and_every_item_is_a_hedonic_command(self):
        state = home.State(active=[("q", "running")], unfinished=[("old", "interrupted")], favourite="paper")
        items = home.menu(state)
        self.assertEqual(items[0].argv, ["run", "attach", "q"])
        self.assertEqual(items[1].argv, ["run", "resume", "old"])
        self.assertTrue(items[1].confirm)
        self.assertEqual(items[2].argv, ["run", "exp"])  # the quick default comes before a favourite
        self.assertEqual(items[3].argv, ["run"])
        self.assertIsNone(items[-1].argv)
        first_words = {i.argv[0] for i in items if i.argv}
        self.assertLessEqual(first_words, set(CLI.TOP_LEVEL))
        # Nothing a view runs starts work: the queue is only planned, setup is a dry run, update only compares.
        views = {tuple(i.argv) for i in items if i.back}
        self.assertIn(("run", "queue", "paper", "--dry-run"), views)
        self.assertIn(("setup", "--dry-run"), views)
        self.assertIn(("update",), views)

    def test_without_runs_the_menu_starts_with_a_benchmark(self):
        items = home.menu(home.State())
        self.assertEqual(items[0].argv, ["run", "exp"])

    def test_the_header_reads_local_state_only(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
                "HEDONIC_RUNS_DIR": d + "/runs", "HEDONIC_NETWORKS_DIR": d + "/nets",
                "HEDONIC_CONFIG": d + "/config.toml"}):
            state = home.gather()
            text = ANSI.sub("", "\n".join(home.header(state)))
        self.assertEqual(state.pairs_present, 0)
        self.assertEqual(state.pairs_known, 13)
        self.assertIn("0 of 13 SNAP graph/cover pairs", text)
        self.assertIn("none yet", text)


class TestLoop(unittest.TestCase):
    def _run(self, picks, state=None):
        """Drive ``home.main`` with the given menu answers (labels); returns (exit code, commands run, output)."""
        ran, answers = [], list(picks)

        def select(title, options, default=0, help_text=""):
            want = answers.pop(0)
            if want == "q":
                raise tui.Cancelled
            return [o[0] for o in options].index(want)

        out = io.StringIO()
        with patch.object(home, "gather", return_value=state or home.State()), \
                patch.object(tui, "select", side_effect=select), redirect_stdout(out):
            code = home.main(run=lambda argv: ran.append(argv) or 3)
        return code, ran, out.getvalue()

    def test_a_view_comes_back_to_the_menu_and_shows_its_command(self):
        code, ran, out = self._run(["Runs", "Back to the menu", "Guide", "Quit"])
        self.assertEqual(code, 0)
        self.assertEqual(ran, [["run", "list"], ["guide"]])
        self.assertIn("→ hedonic run list", out)

    def test_a_run_hands_over_and_returns_its_code(self):
        code, ran, _ = self._run(["Start a benchmark"])
        self.assertEqual((code, ran), (3, [["run", "exp"]]))

    def test_resume_asks_first_and_no_runs_nothing(self):
        state = home.State(unfinished=[("old", "failed")])
        code, ran, _ = self._run(["Resume old", "No", "q"], state)
        self.assertEqual((code, ran), (0, []))
        code, ran, _ = self._run(["Resume old", "Yes"], state)
        self.assertEqual(ran, [["run", "resume", "old"]])

    def test_q_quits(self):
        self.assertEqual(self._run(["q"])[:2], (0, []))


class TestWizardsAskBeforeStarting(unittest.TestCase):
    """Every wizard ends with a yes/no; the quick default runs are the first answers, never the paper's."""

    def _select(self, answers):
        answers = list(answers)

        def select(title, options, default=0, help_text=""):
            return answers.pop(0)
        return select

    def test_the_benchmark_default_is_the_quick_run_and_no_goes_back(self):
        from hedonic.experiments.overlapping import quickstart as qs

        cfg = qs.Config()
        self.assertEqual((cfg.networks, cfg.nodes, cfg.seeds, cfg.methods), (["dblp"], 5000, [0], ["hedonic", "codeseg"]))
        # default -> No -> back at the start menu -> default -> Yes
        with patch.object(tui, "select", side_effect=self._select([0, 1, 0, 0])) as select, \
                redirect_stdout(io.StringIO()) as out:
            self.assertIs(qs.wizard(cfg), cfg)
        self.assertEqual(select.call_count, 4)
        self.assertIn("Start it?", [c.args[0] for c in select.call_args_list])
        self.assertIn("DBLP", out.getvalue())
        with patch.object(tui, "select", side_effect=tui.Cancelled):
            self.assertFalse(qs.confirm_start())

    def test_the_disjoint_wizard_defaults_to_smoke_and_no_starts_nothing(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import runmanager

        self.assertEqual(list(front.PRESETS)[1], "smoke")  # the wizard's pre-selected answer (default=1)
        with tempfile.TemporaryDirectory() as d, patch.object(tui, "interactive", return_value=True), \
                patch.object(tui, "select", side_effect=self._select([1, 1])), \
                patch.object(tui, "text", side_effect=lambda title, default="", validate=None: default), \
                patch.object(runmanager, "launch_disjoint") as launch, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(front.front_main(["-i", "--output-dir", d + "/out"]), 0)
        launch.assert_not_called()
        self.assertIn("not started", out.getvalue())


class TestNothingStartsUnasked(unittest.TestCase):
    """Fixes from the CLI/TUI swarm: no terminal and nothing typed starts nothing; Enter alone never starts; a
    backed-out wizard is a choice (exit 0) and brings the home screen back; the menu keeps its place."""

    def test_no_terminal_and_no_arguments_starts_nothing(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import quickstart as qs
        from hedonic.experiments.overlapping import runmanager

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"HEDONIC_CONFIG": d + "/c.toml",
                                                                        "HEDONIC_RUNS_DIR": d + "/runs"}), \
                patch.object(tui, "interactive", return_value=False), \
                patch.object(runmanager, "launch") as launch, patch.object(runmanager, "launch_disjoint") as dlaunch:
            for call in (lambda: qs.main([]), lambda: CLI.hedonic_main(["run"]), lambda: front.front_main([])):
                with redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(call(), 0)
                self.assertIn("not started: no terminal to confirm in", out.getvalue())
        launch.assert_not_called()
        dlaunch.assert_not_called()

    def test_the_last_question_has_no_preselected(self):
        seen = {}

        def select(title, options, default=0, help_text=""):
            seen["default"] = default
            return default
        with patch.object(tui, "select", side_effect=select):
            self.assertFalse(tui.confirm())  # Enter alone answers No
        self.assertEqual(seen["default"], 1)

    def test_a_wizard_backed_out_of_returns_to_the_menu(self):
        from hedonic.experiments.overlapping import quickstart as qs

        with patch.object(tui, "select", side_effect=[4]), patch.object(tui, "interactive", return_value=True), \
                redirect_stdout(io.StringIO()) as out:  # "Quit" on the wizard's first menu
            self.assertEqual(qs.main(["-i"]), 0)
        self.assertIn("nothing started", out.getvalue())
        self.assertTrue(tui.take_cancelled())
        self.assertFalse(tui.take_cancelled())  # read once

        ran, picks = [], ["Start a benchmark", "Runs", "Back to the menu", "q"]
        defaults = []

        def select(title, options, default=0, help_text=""):
            defaults.append(default)
            want = picks.pop(0)
            if want == "q":
                raise tui.Cancelled
            return [o[0] for o in options].index(want)

        def run(argv):
            ran.append(argv)
            if argv == ["run", "exp"]:
                tui.note_cancelled()
            return 0
        with patch.object(home, "gather", return_value=home.State()), patch.object(tui, "select", side_effect=select), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(home.main(run=run), 0)
        self.assertEqual(ran, [["run", "exp"], ["run", "list"]])  # the menu came back after the wizard
        labels = [i.label for i in home.menu(home.State())]
        self.assertEqual(defaults[1], labels.index("Start a benchmark"))  # the cursor stayed where it was
        self.assertEqual(defaults[3], labels.index("Runs"))

    def test_wizard_values_are_checked_at_the_prompt(self):
        from hedonic.experiments.disjoint import front
        from hedonic.experiments.overlapping import quickstart as qs

        self.assertEqual(qs._positive_seconds("-5"), "enter a positive number of seconds")
        self.assertEqual(qs._positive_seconds("abc"), "enter a positive number of seconds")
        self.assertIsNone(qs._positive_seconds("600"))
        typed = iter(["abc", "-5", "120", "2", "0.3", "0.5", "0", "2", "0.5", "8"])
        errors = []

        def text(title, default="", validate=None):
            while True:
                value = next(typed)
                problem = validate(value) if validate else None
                if not problem:
                    return value
                errors.append(problem)
        with patch.object(tui, "select", side_effect=[len(front.PRESETS)]), patch.object(tui, "text", side_effect=text), \
                patch.object(tui, "multiselect", side_effect=lambda title, options, selected=(): list(selected)), \
                redirect_stdout(io.StringIO()):
            settings = front.wizard(front.resolve(front.build_parser().parse_args([])))
        self.assertEqual((settings["nodes"], settings["communities"], settings["workers"]), (120, [2], 8))
        self.assertEqual(len(errors), 3)  # "abc" and "-5" vertices, noise 2: each asked again
        self.assertTrue(front.is_custom(settings))
        self.assertIn("custom grid", front.render_plan(settings))
        with self.assertRaisesRegex(ValueError, "invalid communities"):
            front.parse_ints("abc", "communities")
        with self.assertRaisesRegex(ValueError, r"--noises"):
            front.resolve(front.build_parser().parse_args(["--noises", "2"]))
        with self.assertRaisesRegex(ValueError, r"--nodes"):
            front.resolve(front.build_parser().parse_args(["--nodes", "120", "--communities", "7"]))

    def test_resume_leaves_a_completed_run_alone(self):
        from hedonic.experiments.overlapping import runmanager

        entry = {"name": "done-run", "kind": "benchmark", "output_dir": "/x"}
        with patch.object(runmanager, "load_entry", return_value=entry), \
                patch.object(runmanager, "read_progress", return_value={}), \
                patch.object(runmanager, "effective_status", return_value="completed"), \
                patch.object(runmanager, "cmd_resume", return_value=0) as resume, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(runmanager.dispatch(["resume", "done-run"]), 0)
            resume.assert_not_called()
            self.assertIn("is completed", out.getvalue())
            self.assertEqual(runmanager.dispatch(["resume", "done-run", "--again", "--detach"]), 0)
            resume.assert_called_once()

    def test_text_fits_by_words_and_durations_read_as_durations(self):
        from hedonic.experiments.overlapping import quickstart as qs

        hint = tui.strip_ansi(tui._hint("❯ Start a benchmark", "a quick default run (about a minute)", columns=40))
        self.assertTrue(hint.endswith("…"))
        self.assertNotIn("(a…", hint)  # cut at a word
        self.assertEqual(tui._hint("x" * 30, "words", columns=32), "")
        self.assertEqual([qs.fmt_duration(s) for s in (45, 720, 70921, 3 * 86400)], ["45 s", "12 min", "19 h 42 min", "3 d"])
        self.assertEqual(qs.seeds_text([0]), "seed 0")
        self.assertEqual(home._home_path("/private/tmp/a-very-long-folder-name-for-a-sandbox/hedonic/config.toml"),
                         "…/hedonic/config.toml")


@unittest.skipUnless(os.name == "posix", "needs a pty")
class TestPty(unittest.TestCase):
    def test_a_bare_hedonic_in_a_terminal_draws_the_home_screen_and_q_restores_the_terminal(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        with tempfile.TemporaryDirectory() as d:
            pid, fd = pty.fork()
            if pid == 0:
                os.environ.update(TERM="xterm-256color", HEDONIC_RUNS_DIR=d + "/runs", HEDONIC_PLAIN_VIEW="",
                                  HEDONIC_NETWORKS_DIR=d + "/nets", HEDONIC_CONFIG=d + "/config.toml")
                os.execv(sys.executable, [sys.executable, "-c",
                                          "import sys; from hedonic.experiments.CLI import hedonic_main; "
                                          "sys.exit(hedonic_main([]))"])
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
            out = b""

            def pump(until: str, seconds: float = 20.0) -> None:
                nonlocal out
                end = time.time() + seconds
                while time.time() < end and until not in (out.decode(errors="ignore") if until.startswith("\x1b")
                                                          else ANSI.sub("", out.decode(errors="ignore"))):
                    if select.select([fd], [], [], 0.05)[0]:
                        try:
                            out += os.read(fd, 65536)
                        except OSError:
                            return
            try:
                pump("Quit")
                text = ANSI.sub("", out.decode(errors="ignore"))
                self.assertIn("What would you like to do?", text)
                self.assertIn("SNAP graph/cover pairs on disk", text)
                self.assertIn("Start a benchmark", text)
                os.write(fd, b"q")
                pump("\x1b[?25h", 5.0)  # read before reaping: a closed pty drops what was not read
                _, status = os.waitpid(pid, 0)
            finally:
                try:
                    os.kill(pid, 9)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
                os.close(fd)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        self.assertIn(b"\x1b[?25h", out)  # the cursor is shown again


@unittest.skipUnless(os.name == "posix", "needs a pty")
class TestTypeAhead(unittest.TestCase):
    def test_keys_typed_ahead_in_one_burst_all_reach_the_prompt(self):
        import pty
        import select

        pid, fd = pty.fork()
        if pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execv(sys.executable, [sys.executable, "-c",
                                      "import sys, time; from hedonic.experiments.overlapping import tui; "
                                      "time.sleep(0.5); sys.exit(10 + tui.select('pick', ['a', 'b', 'c', 'd']))"])
        os.write(fd, b"\x1b[B\x1b[B\x1b[B\r")  # before the prompt is even drawn
        end, done, status = time.time() + 10, 0, None
        while not done and time.time() < end:
            # Drain the output so the child never blocks on a full pty. On Linux a read raises EIO once the child
            # has exited, before it is reaped: keep waiting for its status rather than leaving the loop.
            if select.select([fd], [], [], 0.05)[0]:
                try:
                    os.read(fd, 65536)
                except OSError:
                    time.sleep(0.05)
            done, status = os.waitpid(pid, os.WNOHANG)
        if not done:
            os.kill(pid, 9)
            _, status = os.waitpid(pid, 0)
        os.close(fd)
        self.assertEqual(os.waitstatus_to_exitcode(status), 13)  # the fourth option


if __name__ == "__main__":
    unittest.main()
