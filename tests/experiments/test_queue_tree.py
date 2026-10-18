"""The interactive queue view: the tree model (counts, formulas, ETAs), the queue's tree, the input and drawing of the
viewer, and a pty run of the real loop.

The model is generic (loops, units, costs); the toy family below has an outer loop of two values, A and B."""

from __future__ import annotations

import os
import re
import sys
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import queuetui as qt
from hedonic.experiments.overlapping import runqueue
from hedonic.experiments.overlapping import tui

ANSI = re.compile(r"\033\[[0-9;?<]*[A-Za-z]")
G = {**tui.glyphs(True), "open": "▾", "closed": "▸"}


def _family(done_pairs=(), cost=None) -> qm.Family:
    """2 pairs x 2 policies x 3 resolutions x 2 seeds = 24 units; the units of ``done_pairs`` are done."""
    dims = [qm.Dim("pair", "pairs"), qm.Dim("action policy", "action policies", meaning={"fixed": "stays"}),
            qm.Dim("resolution", "resolutions", "γ {}"), qm.Dim("seed", "seeds", "seed {}")]
    units = [qm.Unit((pair, policy, gamma, seed), pair in done_pairs, 1.5 if pair in done_pairs else None)
             for pair in ("A", "B") for policy in ("fixed", "open") for gamma in ("0", "0.1", "1") for seed in ("0", "1")]
    return qm.Family("detector", "detector runs", "one seeded run", dims, units, "pair > policy > resolution > seed", cost)


class TestModel(unittest.TestCase):
    def test_a_count_is_the_sum_of_the_rows_under_it_and_its_formula_is_written_out(self):
        nodes = qm.family_nodes(_family(done_pairs=("A",)), _family(done_pairs=("A",)).units)
        by = {n.label: n for n in nodes}
        self.assertEqual((by["A"].done, by["A"].total), (12, 12))
        self.assertEqual((by["B"].done, by["B"].total), (0, 12))
        policy = by["B"].children[0]
        self.assertEqual(policy.total, 6)
        self.assertEqual(sum(c.total for c in policy.children), 6)
        text = "\n".join(by["B"].detail)
        self.assertIn("12 detector runs = 2 action policies x 3 resolutions x 2 seeds", text)
        self.assertIn("runner order, outermost first: pair > policy > resolution > seed", text)
        resolution = policy.children[0]
        self.assertEqual(resolution.label, "γ 0")
        self.assertIn("2 detector runs = 2 seeds", "\n".join(resolution.detail))
        self.assertEqual(resolution.strip, [False, False])  # its children are units: one mark each
        self.assertIn("stays", "\n".join(policy.detail))  # a value's meaning is shown on its own row

    def test_a_plan_that_is_not_a_full_grid_says_so_instead_of_inventing_a_product(self):
        family = _family()
        family.units = family.units[:-1]
        lines = qm.explain(family, 0, family.units)
        self.assertIn("23 detector runs over 2 pairs x 2 action policies x 3 resolutions x 2 seeds", lines[0])
        self.assertIn("not every combination", lines[0])

    def test_etas_add_up_and_a_unit_without_an_estimate_makes_the_total_a_lower_bound(self):
        def cost(unit):
            return (2.0, "measured") if unit.coords[0] == "A" else (None, "")
        nodes = qm.family_nodes(_family(cost=cost), _family().units)
        a, b = nodes
        self.assertEqual((a.left, a.basis, a.unknown), (24.0, "measured", 0))
        self.assertEqual(qm.eta_text(a), "0:24")
        self.assertEqual((b.left, b.unknown), (0.0, 12))
        self.assertEqual(qm.eta_text(b), "?")
        run = qm.run_node(1, "r", "", "running", qm.RunModel([_family(cost=cost)]), None)
        self.assertEqual((run.left, run.unknown), (24.0, 12))
        self.assertEqual(qm.eta_text(run), "≥0:24")
        root = qm.queue_node("q", [run])
        self.assertEqual(qm.eta_text(root), "≥0:24")

    def test_estimates_and_bounds_are_marked_and_a_finished_node_says_done(self):
        node = qm.Node(("x",), "x", total=4, done=1, left=90.0, basis="estimate")
        self.assertEqual(qm.eta_text(node), "~1:30")
        node.basis = "bound"
        self.assertEqual(qm.eta_text(node), "≥1:30")
        node.done = 4
        self.assertEqual(qm.eta_text(node), "done")
        self.assertEqual(qm.fmt_seconds(3 * 86400 + 7200), "3d 02h")
        self.assertEqual(qm.fmt_seconds(3725), "1:02:05")

    def test_two_kinds_of_unit_under_one_pair_are_two_children_and_the_pair_sums_them(self):
        audit = qm.Family("audit", "audit steps", "a check", [qm.Dim("pair", "pairs"), qm.Dim("resolution", "resolutions")],
                          [qm.Unit(("A", str(i)), i < 2) for i in range(5)] + [qm.Unit(("B", str(i)), False) for i in range(5)])
        run = qm.run_node(1, "r", "t", "running", qm.RunModel([audit, _family()]), None)
        pair = run.children[0]
        self.assertEqual([c.label for c in pair.children], ["audit steps", "detector runs"])
        self.assertEqual((pair.done, pair.total), (2, 5 + 12))
        self.assertEqual((run.done, run.total), (2, 34))
        self.assertIn("5 audit steps + 12 detector runs = 17 steps", pair.detail[0])
        self.assertEqual(run.noun, "steps")

    def test_a_run_without_a_model_uses_its_own_counts_and_eta(self):
        run = qm.run_node(2, "r", "", "running", None, (3, 10, "searches"), eta=(120.0, "measured"))
        self.assertEqual((run.done, run.total, run.left), (3, 10, 120.0))
        lost = qm.run_node(2, "r", "", "pending", None, (0, 10, "searches"))
        self.assertEqual(lost.unknown, 1)
        self.assertEqual(qm.eta_text(lost), "?")
        done = qm.run_node(2, "r", "", "done", None, (10, 10, "searches"))
        self.assertEqual(done.status, "done")


def _tree() -> qm.Node:
    """queue-like: a done run, a running run with a running group, a pending run."""
    running = qm.run_node(2, "spec", "", "running", qm.RunModel([_family(done_pairs=("A",), cost=lambda u: (1.0, "measured"))]), None)
    running.children[1].status = "running"  # B is the one being worked on
    done = qm.run_node(1, "calib", "", "done", None, (8, 8, "searches"))
    pending = qm.run_node(3, "bench", "", "pending", None, (0, 20, "units"))
    return qm.queue_node("q", [done, running, pending], "running")


class TestViewer(unittest.TestCase):
    def test_input_is_parsed_into_keys_wheel_and_clicks(self):
        events, rest = qt.parse("j\x1b[A\x1b[B\x1b[5~\x1b[<64;10;5M\x1b[<65;10;5M\x1b[<0;12;7M\x1b[<0;12;7m \r")
        self.assertEqual(events, [("key", "j"), ("key", "up"), ("key", "down"), ("key", "pgup"), ("wheel", -1, 10, 5),
                                  ("wheel", 1, 10, 5), ("click", 12, 7), ("key", "space"), ("key", "enter")])
        self.assertEqual(rest, "")

    def test_an_unfinished_escape_sequence_waits_for_the_rest(self):
        events, rest = qt.parse("j\x1b[<65;1")
        self.assertEqual((events, rest), ([("key", "j")], "\x1b[<65;1"))
        events, rest = qt.parse(rest + "0;5M")
        self.assertEqual(events, [("wheel", 1, 10, 5)])

    def test_only_the_live_path_is_open_at_first_and_a_later_node_is_not_opened_for_you(self):
        root, view = _tree(), qt.View()
        rows = qt.flatten(root, view)
        path = qt.live_path(root)
        self.assertEqual(view.expanded, {node.key for node in path})
        self.assertEqual([node.depth for node in path], [0, 1, 2, 3, 4, 5])  # queue > run > pair > policy > resolution > seed
        self.assertEqual([r.depth for r in rows][:5], [0, 1, 1, 2, 2])
        view.expanded.clear()
        view.expanded.add(root.key)
        self.assertEqual(len(qt.flatten(root, view)), 4)  # the user closed the run; it stays closed on the next refresh

    def test_keys_open_close_and_walk_the_tree(self):
        root, view = _tree(), qt.View()
        qt.flatten(root, view)
        qt.render(root, view, (100, 30), G)
        run2 = root.children[1]

        def press(key):
            self.assertFalse(qt.apply(view, root, ("key", key)))
            qt.render(root, view, (100, 30), G)
        press("2")  # jump to run 2
        self.assertEqual(view.cursor, run2.key)
        press("left")
        self.assertNotIn(run2.key, view.expanded)
        press("left")  # a closed row: go to its parent
        self.assertEqual(view.cursor, root.key)
        press("down")
        press("down")
        self.assertEqual(view.cursor, run2.key)
        press("right")
        self.assertIn(run2.key, view.expanded)
        press("right")  # already open: step into it
        self.assertEqual(view.cursor, run2.children[0].key)
        press("e")
        self.assertIn(run2.children[0].children[0].key, view.expanded)
        press("c")
        self.assertNotIn(run2.children[0].key, view.expanded)
        press("E")
        self.assertTrue(all(n.key in view.expanded for n in qt.descendants(root) if n.children))
        press("C")
        self.assertEqual(view.expanded, {root.key})
        press("G")
        self.assertEqual(view.cursor, view.rows[-1].key)
        press("g")
        self.assertEqual(view.cursor, root.key)
        self.assertTrue(qt.apply(view, root, ("key", "q")))

    def test_the_wheel_scrolls_the_list_without_moving_the_selection_and_stays_in_range(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "E"))
        qt.render(root, view, (100, 14), G)
        cursor = view.cursor
        qt.apply(view, root, ("wheel", 1, 5, 5))
        qt.render(root, view, (100, 14), G)
        self.assertEqual((view.offset, view.cursor), (qt.WHEEL_STEP, cursor))
        for _ in range(100):
            qt.apply(view, root, ("wheel", 1, 5, 5))
        qt.render(root, view, (100, 14), G)
        self.assertEqual(view.offset, len(view.rows) - view.body[1])
        qt.apply(view, root, ("wheel", -1, 5, 5))
        self.assertEqual(view.offset, len(view.rows) - view.body[1] - qt.WHEEL_STEP)

    def test_a_click_selects_a_row_and_a_click_on_its_caret_or_a_second_click_opens_it(self):
        root, view = _tree(), qt.View()
        qt.render(root, view, (100, 30), G)
        top = view.body[0]
        run2 = root.children[1]
        row = top + qt.index_of(view.rows, run2.key) + 1  # the 1-based screen row
        caret_x = 3 + 2 * run2.depth + 1  # 1-based column of its ▾
        self.assertIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", 60, row), now=100.0)  # a first click only selects
        self.assertEqual(view.cursor, run2.key)
        self.assertIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", caret_x, row), now=200.0)  # on the caret: closes
        self.assertNotIn(run2.key, view.expanded)
        qt.apply(view, root, ("click", 60, row), now=300.0)  # the row is selected: another click opens
        self.assertIn(run2.key, view.expanded)

    def test_every_drawn_line_fits_the_terminal_and_the_selection_is_marked(self):
        root = _tree()
        for width, height in ((100, 30), (70, 18), (120, 40)):
            view = qt.View()
            lines = qt.render(root, view, (width, height), G, qt.header_line(root, "running", 65.0, G))
            self.assertLessEqual(len(lines), height)
            for line in lines:
                self.assertLessEqual(len(ANSI.sub("", line)), width, line)
            self.assertTrue(any(line.lstrip().startswith("›") for line in lines))
        text = "\n".join(ANSI.sub("", line) for line in qt.render(root, qt.View(), (110, 30), G, qt.header_line(root, "running", 65.0, G)))
        self.assertIn("ETA", text.splitlines()[0])
        self.assertIn("1/3 runs", text.splitlines()[0])

    def test_the_detail_pane_explains_the_selected_row_and_where_its_eta_comes_from(self):
        root, view = _tree(), qt.View()
        qt.flatten(root, view)
        view.cursor = root.children[1].key
        screen = "\n".join(ANSI.sub("", line) for line in qt.render(root, view, (110, 40), G))
        self.assertIn("24 detector runs = 2 pairs x 2 action policies x 3 resolutions x 2 seeds", screen)
        self.assertIn("from this run's own timings", screen)
        view.cursor = root.key
        self.assertIn("finishes about", "\n".join(qt.detail_lines(root, 100, time.time())))

    def test_help_replaces_the_tree_until_a_key_is_pressed(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "?"))
        self.assertTrue(view.help)
        self.assertIn("mouse wheel", "\n".join(qt.render(root, view, (110, 30), G)))
        qt.apply(view, root, ("key", "x"))
        self.assertFalse(view.help)

    def test_the_cursor_moves_to_the_nearest_open_ancestor_when_its_row_disappears(self):
        root, view = _tree(), qt.View()
        qt.apply(view, root, ("key", "E"))
        qt.render(root, view, (100, 30), G)
        deep = qt.descendants(root.children[1])[5]
        view.cursor = deep.key
        view.expanded.discard(root.children[1].key)  # close the run it is in
        qt.flatten(root, view)
        self.assertEqual(view.cursor, root.children[1].key)


class TestRunqueueTree(unittest.TestCase):
    def test_the_tree_of_a_queue_takes_each_run_from_its_files_and_keeps_the_report_numbers(self):
        items = [{"name": "smoke", "run": ["disjoint", "--name", "smoke"]},
                 {"name": "paper", "run": ["disjoint", "--name", "paper"], "title": "T"}]
        entry = {"name": "q", "items": items, "file": ""}
        states = {"smoke": ("completed", None), "paper": ("running", None)}
        data = {"smoke": ("complete", "/nowhere/smoke"), "paper": ("partial", "/nowhere/paper")}
        counts = {"smoke": (16, 16, "graphs"), "paper": (12, 24, "graphs")}
        with patch.object(runqueue, "_status_of", lambda item, fresh=False: states[item["name"]]), \
                patch.object(runqueue, "data_state", lambda run, fresh=False: data[run[2]]), \
                patch.object(runqueue, "unit_counts", lambda run: counts[run[2]]), \
                patch.object(runqueue.rm, "effective_status", return_value="running"):
            root = runqueue.tree(entry, {})
        smoke, paper = root.children
        self.assertEqual((smoke.status, smoke.done, smoke.total), ("done", 16, 16))
        self.assertEqual((paper.status, paper.done, paper.total, paper.noun), ("running", 12, 24, "graphs"))
        self.assertEqual(paper.children, [])  # hedonic 1 has no unit model of a disjoint run: its counts only
        self.assertEqual(qm.eta_text(paper), "?")  # no registry entry, so no ETA from its runner
        self.assertEqual((root.done, root.total), (1, 2))
        self.assertIn("T", paper.detail)  # the item's title
        self.assertIn("2 runs execute one after another, each with every worker.", root.detail)


def _timed(done_pairs=(), parallel=1, seconds=2.0) -> qm.Family:
    """Like ``_family`` but the finished units carry the time they took."""
    family = _family(done_pairs, cost=lambda u: (1.0, "measured"))
    for unit in family.units:
        if unit.done:
            unit.took = seconds
    family.parallel = parallel
    return family


class TestLiveMarks(unittest.TestCase):
    """The spinner goes down every level to the units being worked on, and only there."""

    def test_the_running_marks_reach_the_unit_being_worked_on_through_every_level(self):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        pair_a, pair_b = node.children
        self.assertEqual((node.status, pair_a.status, pair_b.status), ("running", "done", "running"))
        walk, depth = pair_b, 0
        while walk.children:
            running = [c for c in walk.children if c.status == "running"]
            self.assertEqual(len(running), 1, walk.label)  # one unit at a time with parallel 1: one path down
            walk, depth = running[0], depth + 1
        self.assertEqual((walk.label, depth), ("seed 0", 3))
        self.assertEqual([c.status for c in pair_b.children], ["running", "pending"])

    def test_the_next_parallel_units_of_the_pair_in_work_are_running_and_never_those_of_the_next_pair(self):
        family = _timed(done_pairs=("A",), parallel=3)
        qm.mark_running([family])
        running = [u.coords for u in family.units if u.running]
        self.assertEqual(running, [("B", "fixed", "0", "0"), ("B", "fixed", "0", "1"), ("B", "fixed", "0.1", "0")])
        late = _timed(done_pairs=(), parallel=50)  # more workers than units left in the pair: the pair's units only
        qm.mark_running([late])
        self.assertEqual({u.coords[0] for u in late.units if u.running}, {"A"})
        self.assertEqual(sum(u.running for u in late.units), 12)

    def test_a_builder_that_flags_its_own_running_unit_is_not_second_guessed(self):
        family = _timed(done_pairs=("A",), parallel=4)
        family.units[-1].running = True
        qm.mark_running([family])
        self.assertEqual(sum(u.running for u in family.units), 1)

    def test_between_families_the_one_of_the_earliest_pair_works_and_an_audit_goes_before_its_detector_runs(self):
        audit = qm.Family("audit", "audit steps", "", [qm.Dim("pair", "pairs"), qm.Dim("resolution", "resolutions")],
                          [qm.Unit((pair, g), pair == "A") for pair in ("A", "B") for g in ("0", "1")], parallel=2)
        detector = qm.Family("detector", "detector runs", "", [qm.Dim("pair", "pairs"), qm.Dim("seed", "seeds")],
                             [qm.Unit((pair, seed), False) for pair in ("A", "B") for seed in ("0", "1")], parallel=2)
        qm.mark_running([audit, detector])
        self.assertEqual([u.coords for u in audit.units if u.running], [])
        self.assertEqual([u.coords for u in detector.units if u.running], [("A", "0"), ("A", "1")])  # pair A is audited: its runs
        audit.units[0].done = audit.units[1].done = False  # audit steps of pair A are still to do: the runs wait for them
        detector.units[0].running = detector.units[1].running = False
        qm.mark_running([audit, detector])
        self.assertEqual([u.coords for u in audit.units if u.running], [("A", "0"), ("A", "1")])
        self.assertEqual([u.coords for u in detector.units if u.running], [])

    def test_a_stopped_run_shows_no_spinner_and_a_pair_with_some_work_done_is_partial_not_running(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A",))]), None)
        self.assertEqual(node.status, "stopped")
        self.assertFalse(any(u.running for u in qm.RunModel([_timed()]).families[0].units))
        pair_b = node.children[1]
        self.assertEqual(pair_b.status, "pending")
        part = qm.family_nodes(_timed(done_pairs=("A",)), [u for u in _timed(done_pairs=("A",)).units if u.coords[0] == "A"][:6] +
                               [qm.Unit(("A", "open", "0", "0"), False)])
        self.assertIn(part[0].status, ("partial", "done"))

    def test_a_partly_done_node_that_nothing_is_working_on_is_partial(self):
        units = [qm.Unit(("A", "x"), True, 1.0, "", False, 1.0), qm.Unit(("A", "y"), False)]
        family = qm.Family("f", "things", "", [qm.Dim("pair", "pairs"), qm.Dim("thing", "things")], units)
        (pair,) = qm.family_nodes(family, family.units)
        self.assertEqual(pair.status, "partial")
        units[1].running = True
        self.assertEqual(qm.family_nodes(family, family.units)[0].status, "running")

    def test_the_viewer_draws_a_spinner_on_each_level_down_to_the_unit(self):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        root = qm.queue_node("q", [node], "running")
        view = qt.View()
        view.expanded = {n.key for n in qt.live_path(root)}
        view.fresh = False
        screen = [ANSI.sub("", line) for line in qt.render(root, view, (110, 40), G, "", "@")]
        spinning = [line for line in screen if "@" in line]
        self.assertEqual(len(spinning), 6)  # queue, run, pair B, policy, resolution, seed
        self.assertTrue(any("seed 0" in line for line in spinning))


class TestTook(unittest.TestCase):
    def test_a_finished_node_says_done_and_how_long_it_took(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A", "B"), seconds=2.0)]), None)
        self.assertEqual(qm.eta_text(node), "done 48s")  # 24 units x 2 s
        self.assertEqual(qm.eta_text(node.children[0]), "done 24s")
        leaf = node.children[0].children[0].children[0].children[0]
        self.assertEqual(qm.eta_text(leaf), "done 2.0s")
        self.assertEqual(node.took_missing, 0)

    def test_finished_units_without_a_recorded_time_make_it_a_lower_bound_or_just_done(self):
        family = _timed(done_pairs=("A",), seconds=2.0)
        family.units[0].took = None  # one finished unit of pair A has no time
        done = qm.run_node(1, "spec", "", "stopped", qm.RunModel([family]), None).children[0]
        self.assertEqual(qm.eta_text(done), "done ≥22s")
        self.assertEqual(done.took_missing, 1)
        for unit in family.units:
            unit.took = None
        self.assertEqual(qm.eta_text(qm.run_node(1, "spec", "", "stopped", qm.RunModel([family]), None).children[0]), "done")

    def test_a_run_with_no_unit_times_falls_back_to_its_own_clock_and_a_queue_adds_its_runs_up(self):
        a = qm.run_node(1, "a", "", "done", None, (8, 8, "searches"), took=125.0)
        b = qm.run_node(2, "b", "", "done", None, (4, 4, "units"))
        self.assertEqual(qm.eta_text(a), "done 2:05")
        self.assertEqual(qm.eta_text(b), "done")
        root = qm.queue_node("q", [a, b], "running")
        self.assertEqual((root.took, root.took_missing), (125.0, 1))
        self.assertIn("took 2:05", "\n".join(root.detail))

    def test_the_detail_pane_says_what_the_took_figure_is(self):
        node = qm.run_node(1, "spec", "", "stopped", qm.RunModel([_timed(done_pairs=("A", "B"))]), None)
        text = "\n".join(qt.detail_lines(node, 120, time.time()))
        self.assertIn("took: 48s", text)
        self.assertIn("skipped", text)


class TestLayout(unittest.TestCase):
    def _screen(self, width, height=40, cursor=None):
        node = qm.run_node(1, "spec", "", "running", qm.RunModel([_timed(done_pairs=("A",))]), None)
        root = qm.queue_node("q", [node], "running")
        view = qt.View()
        qt.flatten(root, view)
        if cursor is not None:
            view.cursor = cursor(root).key
        return root, view, qt.render(root, view, (width, height), G, "", "@")

    def test_the_label_column_is_only_as_wide_as_its_longest_label_so_the_table_sits_next_to_it(self):
        root, view, lines = self._screen(180)
        heads = ANSI.sub("", lines[1])
        bar_at = heads.index("progress")
        rows = qt.flatten(root, view)
        longest = max(qt.label_need(n) for n in rows)
        self.assertLessEqual(bar_at, longest + 6)  # not pushed out to the terminal's right edge
        for line in lines[2:2 + len(rows)]:
            self.assertEqual(ANSI.sub("", line).count("%"), 1)
            self.assertEqual(ANSI.sub("", line).index("%") - 5 - 1 >= 0, True)
        # the same columns on a narrow terminal: labels give way, the table keeps its place at the right
        _, _, narrow = self._screen(70)
        for line in narrow:
            self.assertLessEqual(len(ANSI.sub("", line)), 70)

    def test_every_row_has_its_columns_at_the_same_place(self):
        root, view, lines = self._screen(150)
        marks = {ANSI.sub("", line).index("%") for line in lines[2:] if "%" in ANSI.sub("", line)}
        self.assertEqual(len(marks), 1)

    def test_the_selected_row_is_highlighted_from_its_first_column_to_the_terminal_edge(self):
        with patch("hedonic.experiments.overlapping.tui.color", lambda: True):
            root, view, lines = self._screen(120, cursor=lambda r: r.children[0].children[0])
        selected = next(line for line in lines if line.startswith("\033[7m"))
        self.assertEqual(selected.count("\033["), 2)  # one reverse-video start and the final reset: nothing resets it midway
        self.assertTrue(selected.endswith("\033[0m"))
        self.assertEqual(len(ANSI.sub("", selected)), 120)
        self.assertTrue(ANSI.sub("", selected).startswith("›"))
        self.assertEqual(sum(line.startswith("\033[7m") for line in lines), 1)


TRACEBACK = ("WorkerFailed: worker for item 80 failed:\nTraceback (most recent call last):\n  File \"x.py\", line 1, in f\n"
             "ValueError: Invalid initial_membership: membership labels must be contiguous from 0\n")


class TestFailureIsExplained(unittest.TestCase):
    """A red mark alone left the user asking why and where: the viewer says which run, which error and which step."""

    def _patched(self):
        run_entry = {"name": "disjoint-paper"}
        return (patch.object(runqueue, "_state", lambda name: ("failed", run_entry) if name == "disjoint-paper" else ("new", None)),
                patch.object(runqueue.rm, "read_progress", lambda entry: {"reason": TRACEBACK}))

    def test_a_failed_run_gives_its_error_the_step_it_was_in_and_where_the_traceback_is(self):
        first, second = self._patched()
        with first, second:
            lines = runqueue.failure_lines("disjoint-paper")
            self.assertEqual(runqueue.failure_lines("other"), [])  # a run that never failed says nothing
        self.assertEqual(lines[0], "why: ValueError: Invalid initial_membership: membership labels must be contiguous from 0")
        self.assertEqual(lines[1], "while: WorkerFailed: worker for item 80 failed")
        self.assertEqual(lines[2], "whole traceback: hedonic run logs disjoint-paper")

    def test_a_failure_without_a_recorded_reason_says_nothing_instead_of_guessing(self):
        with patch.object(runqueue, "_state", lambda name: ("failed", {"name": name})), \
                patch.object(runqueue.rm, "read_progress", lambda entry: {}):
            self.assertEqual(runqueue.failure_lines("x"), [])

    def test_the_stop_lines_of_the_queue_carry_the_runs_error(self):
        entry = {"name": "disjoint-queue", "items": [{"name": "calib", "run": ["disjoint"]}, {"name": "disjoint-paper", "run": ["disjoint"]}]}
        first, second = self._patched()
        with first, second, patch.object(runqueue, "_status_of", lambda item, fresh=False: ("stopped", None)):
            text = ANSI.sub("", "\n".join(runqueue.stop_lines(entry, {"current": 1, "reason": "disjoint-paper ended as failed"}, "failed")))
        self.assertIn("failed at disjoint-paper (item 2 of 2)", text)
        self.assertIn("ValueError: Invalid initial_membership", text)
        self.assertIn("hedonic run resume disjoint-queue", text)

    def test_the_tree_view_shows_a_notice_under_the_header_and_still_fits_every_size(self):
        root = _tree()
        notice = ["  failed at spec (item 2 of 3)", "    why: ValueError: boom", "  continue with: hedonic run resume q"]
        for width, height in ((100, 30), (70, 18), (120, 40), (60, 8)):
            lines = qt.render(root, qt.View(), (width, height), G, qt.header_line(root, "failed", 65.0, G), notice=notice)
            self.assertLessEqual(len(lines), height)
            for line in lines:
                self.assertLessEqual(len(ANSI.sub("", line)), width, line)
        text = [ANSI.sub("", line) for line in qt.render(root, qt.View(), (110, 30), G, qt.header_line(root, "failed", 65.0, G), notice=notice)]
        self.assertIn("why: ValueError: boom", text[2])
        self.assertIn("progress", text[4])  # the column heads follow the notice, then the rows


@unittest.skipUnless(os.name == "posix", "needs a pty")
class TestPty(unittest.TestCase):
    SCRIPT = textwrap.dedent("""
        import sys
        sys.path.insert(0, {tests!r})
        from test_queue_tree import _tree
        from hedonic.experiments.overlapping import queuetui as qt
        code = qt.run(_tree, lambda: ("running", 5.0))
        sys.exit(code)
    """)

    def test_the_real_loop_draws_the_tree_answers_keys_wheel_and_clicks_and_restores_the_terminal(self):
        import fcntl
        import pty
        import select
        import struct
        import termios

        script = self.SCRIPT.format(tests=str(Path(__file__).resolve().parent))
        pid, fd = pty.fork()
        if pid == 0:
            os.environ["TERM"] = "xterm-256color"
            os.execv(sys.executable, [sys.executable, "-c", script])
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
        out = b""

        def pump(seconds: float) -> None:
            nonlocal out
            end = time.time() + seconds
            while time.time() < end:
                if select.select([fd], [], [], 0.05)[0]:
                    try:
                        out += os.read(fd, 65536)
                    except OSError:
                        return

        def screen() -> str:
            text = out.decode(errors="ignore")
            return ANSI.sub("", text[text.rfind("\x1b[H"):])
        try:
            pump(2.0)
            first = screen()
            self.assertIn("queue q", first)
            self.assertIn("2. spec", first)
            for data in ("j", "j", "l", "\x1b[<65;20;10M", "E"):
                os.write(fd, data.encode())
                pump(0.5)
            self.assertIn("fixed", screen().split("──")[0])  # E opened everything, down to the action policies
            os.write(fd, b"C")
            pump(0.5)
            self.assertNotIn("fixed", screen().split("──")[0])  # C closed all but the runs
            os.write(fd, b"q")
            pump(1.5)
        finally:
            try:
                os.kill(pid, 9)  # never leave the child waiting if an assertion above failed
            except ProcessLookupError:
                pass
            try:
                _, status = os.waitpid(pid, 0)
            except ChildProcessError:
                status = 0
            os.close(fd)
        text = out.decode(errors="ignore")
        self.assertIn("\x1b[?1049h", text)  # alternate screen
        self.assertIn("\x1b[?1000h\x1b[?1006h", text)  # mouse reporting on
        self.assertLess(text.rfind("\x1b[?1000l"), text.rfind("\x1b[?1049l"))  # and off again before leaving it
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, "the viewer exits cleanly on q")


if __name__ == "__main__":
    unittest.main()
