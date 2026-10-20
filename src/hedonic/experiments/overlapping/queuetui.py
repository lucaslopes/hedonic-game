"""The interactive view of a queue: a tree you open and close, scroll with the wheel, and read the numbers of.

The queue is the root; its runs are the first level (the macro overview, each with its progress and ETA); a run opens
into its pairs, a pair into the loops that make its count (action policy, resolution, seed...), down to single units.
Every row carries its own progress and ETA, and the pane under the tree explains the selected row: what its number
counts, which loops multiply into it, in what order the runner executes them, and where its ETA comes from.

Keys: arrows or j/k move, left/right (or h/l) close/open (left on a closed row goes to its parent), Enter/Space
toggles, e/c open/close the whole subtree, E/C open everything / close to the runs, 1-9 jump to a run, PgUp/PgDn/
Home/End, d resizes the detail pane, m turns the mouse off (so text can be selected), ? help, q or Ctrl-C closes the
view only (the queue keeps going). The mouse wheel scrolls; a click selects, a click on the caret (or a second click)
toggles.

The drawing and the input handling are plain functions of a tree and a ``View`` (tests call them directly); ``run`` is
the loop around them: alternate screen, SGR mouse, a thread that rebuilds the tree while the keyboard stays live.
"""

from __future__ import annotations

import os
import re
import select
import shutil
import sys
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import fit, style

REFRESH_SECONDS = 1.5  # how often the tree is rebuilt from the files while the view is open
TICK = 0.2  # keyboard poll and spinner step
WHEEL_STEP = 3
COLOUR = {"done": "green", "running": "cyan", "partial": "yellow", "pending": "dim", "failed": "red", "stopped": "yellow",
          "awaiting": "dim"}
BAR_WIDTH, PERCENT_WIDTH, UNIT_WIDTH, ETA_WIDTH = 8, 4, 14, 13  # the columns right of the labels
MIN_COUNT_WIDTH = 11


@dataclass
class View:
    expanded: set = field(default_factory=set)
    cursor: tuple | None = None
    offset: int = 0
    detail_mode: int = 1  # 0 off, 1 small, 2 large
    mouse: bool = True
    help: bool = False
    seen: set = field(default_factory=set)  # node keys met so far: a node that appears later is not opened for you
    fresh: bool = True  # nothing drawn yet: open the live path once
    message: str = ""
    rows: list = field(default_factory=list)  # the nodes drawn, top to bottom (set by ``flatten``)
    parents: dict = field(default_factory=dict)  # key -> parent node
    body: tuple[int, int] = (0, 0)  # first screen row of the tree (0-based) and how many rows it has
    last_click: tuple[float, tuple | None] = (0.0, None)


# --------------------------------------------------------------------------- the tree as rows
def live_path(root: qm.Node) -> list[qm.Node]:
    """The queue and, below it, the running run, pair, loops... down to the unit being worked on: the part of the tree
    worth opening at first sight."""
    path, node = [root], root
    while True:
        running = [c for c in node.children if c.status == "running"]
        if not running:
            return path
        node = running[0]
        path.append(node)


def flatten(root: qm.Node, view: View) -> list[qm.Node]:
    """The nodes in drawing order (a closed node hides its children); keeps the cursor on a node that exists."""
    if view.fresh:
        view.expanded |= {node.key for node in live_path(root)}
        view.fresh = False
    rows, parents = [], {}

    def walk(node: qm.Node, parent: qm.Node | None) -> None:
        view.seen.add(node.key)
        rows.append(node)
        if parent is not None:
            parents[node.key] = parent
        if node.key in view.expanded:
            for child in node.children:
                walk(child, node)

    walk(root, None)
    view.rows, view.parents = rows, parents
    keys = [n.key for n in rows]
    if view.cursor not in keys:  # the row went away (a unit finished, a run collapsed): its nearest visible ancestor
        view.cursor = _ancestor(view, rows) or keys[0]
    return rows


def _ancestor(view: View, rows: list[qm.Node]) -> tuple | None:
    keys = {n.key for n in rows}
    key = view.cursor
    while key is not None:
        parent = view.parents.get(key)
        if parent is None:
            break
        if parent.key in keys:
            return parent.key
        key = parent.key
    for size in range(len(view.cursor or ()) - 1, 0, -1):  # keys are hierarchical tuples: try shorter prefixes
        if (view.cursor or ())[:size] in keys:
            return (view.cursor or ())[:size]
    return None


def index_of(rows: list[qm.Node], key: tuple | None) -> int:
    return next((i for i, n in enumerate(rows) if n.key == key), 0)


def descendants(node: qm.Node) -> list[qm.Node]:
    out = [node]
    for child in node.children:
        out += descendants(child)
    return out


# --------------------------------------------------------------------------- drawing
def bar(done: int, total: int, width: int, G: dict) -> str:
    filled = round(width * done / total) if total else 0
    return G["full"] * filled + G["empty"] * (width - filled)


def glyph_of(node: qm.Node, G: dict, frame: str) -> str:
    if node.log:  # a record, not a unit: a bullet says "this happened", never "this is finished"
        return "•" if G["full"] == "█" else "-"
    return {"done": G["done"], "running": frame, "partial": G.get("part", G["stopped"]), "pending": G["pending"],
            "failed": G["failed"], "stopped": G["stopped"], "awaiting": G["pending"]}.get(node.status, G["pending"])


def mark(node: qm.Node, G: dict, frame: str) -> str:
    return style(glyph_of(node, G, frame), COLOUR.get(node.status, "dim"))


def strip_text(flags: list[bool], width: int, G: dict) -> str:
    """One mark per unit; more units than ``width`` are bucketed, each bucket shaded by the share of it done."""
    if len(flags) <= width:
        return "".join(G["full"] if f else G["empty"] for f in flags)
    shades = "░▒▓█" if G["full"] == "█" else "-=+#"
    out = []
    for i in range(width):
        chunk = flags[i * len(flags) // width:(i + 1) * len(flags) // width]
        share = sum(chunk) / len(chunk)
        out.append(shades[-1] if share == 1 else shades[0] if share == 0 else shades[1 + min(1, int(share * 2))])
    return "".join(out)


@dataclass
class Layout:
    """Where the columns of a screen fall: the label column is only as wide as the longest label shown, and the counts
    as wide as the longest count, so the table sits next to the labels instead of at the far edge of the terminal."""

    left: int  # width of the label column (marker, indent, caret, state mark and label), after the two-column margin
    count: int = MIN_COUNT_WIDTH
    unit: bool = True


def label_need(node: qm.Node) -> int:
    """Columns a row's label side takes: indent, caret and space, state mark and space, then the label."""
    return 2 * min(node.depth, 8) + 2 + 2 + len(node.label)


def right_columns(node: qm.Node, width: int, G: dict, count_width: int = MIN_COUNT_WIDTH) -> str:
    """The fixed columns on the right of a row: progress bar, percent, count, unit (or the units as marks) and ETA."""
    if node.log:
        blank = " " * BAR_WIDTH + " " + " " * PERCENT_WIDTH + " " + " " * count_width
        unit = f" {node.note[:UNIT_WIDTH]:<{UNIT_WIDTH}}" if width >= 84 else ""
        return f" {blank}{unit} {qm.took_text(node):>{ETA_WIDTH}}"
    inner = node.sub if node.sub and node.done < node.total and node.sub[1] > 0 else None
    shown, of = (inner[0], inner[1]) if inner else (node.done, node.total)  # a unit under way fills its bar by its own steps
    if node.share is not None and not inner and node.total:  # a method or the queue: how far its runs are
        shown, of = (1.0 if node.status == "done" else node.share), 1.0
    percent = f"{min(100, int(100 * shown / of)):>3}%" if of else "  -"
    count = f"{node.done:,}/{node.total:,}" if node.total else ""
    text = f" {bar(shown, of, BAR_WIDTH, G)} {percent} {count:>{count_width}}"
    if width >= 84:
        unit = (f"{inner[0]:.1f}/{inner[1]:g} {inner[2]}".replace(".0/","/") if inner
                else strip_text(node.strip, UNIT_WIDTH, G) if len(node.strip) > 1 else node.noun)
        text += f" {unit[:UNIT_WIDTH]:<{UNIT_WIDTH}}"
    return text + f" {qm.eta_text(node):>{ETA_WIDTH}}"


def right_width(width: int, count_width: int = MIN_COUNT_WIDTH) -> int:
    return 1 + BAR_WIDTH + 1 + PERCENT_WIDTH + 1 + count_width + ((1 + UNIT_WIDTH) if width >= 84 else 0) + 1 + ETA_WIDTH


def layout_for(rows: list[qm.Node], width: int) -> Layout:
    """The columns for these rows on a terminal ``width`` wide."""
    count = max([MIN_COUNT_WIDTH] + [len(f"{n.done:,}/{n.total:,}") for n in rows if n.total])
    room = max(12, width - 2 - right_width(width, count))
    need = max((label_need(n) for n in rows), default=12) + 2  # two spaces between the labels and the bar
    return Layout(left=min(room, need), count=count, unit=width >= 84)


def column_heads(width: int, lay: Layout | None = None) -> str:
    count = lay.count if lay else MIN_COUNT_WIDTH
    text = (f" {'progress':<{BAR_WIDTH}} {'':>{PERCENT_WIDTH}} {'done/total':>{count}}"
            + (f" {'unit':<{UNIT_WIDTH}}" if width >= 84 else "") + f" {'ETA / took':>{ETA_WIDTH}}")
    left = lay.left if lay else max(0, width - 2 - len(text))
    return style("  " + " " * left + text, "dim")


def row_text(node: qm.Node, view: View, width: int, G: dict, frame: str, lay: Layout | None = None) -> str:
    """One tree row: selection mark, indent, caret, state mark, label, then the columns. The selected row is
    highlighted from the first column to the last of the terminal (its parts carry no colour of their own, because
    the reset that ends a coloured glyph would end the highlight with it)."""
    selected = node.key == view.cursor
    caret = (G["open"] if node.key in view.expanded else G["closed"]) if node.children else " "
    count_width = lay.count if lay else MIN_COUNT_WIDTH
    right = right_columns(node, width, G, count_width)
    head = "  " * min(node.depth, 8) + caret + " "
    left_room = lay.left if lay else max(4, width - 2 - len(right))
    name_room = max(4, left_room - len(head) - 2)
    name = node.label if len(node.label) <= name_room else node.label[:name_room - 1] + "…"
    state = glyph_of(node, G, frame) if selected else mark(node, G, frame)
    left = f"{head}{state} {name}"
    pad = " " * max(0, left_room - (len(head) + 2 + len(name)))
    if selected:
        plain = "› " + left + pad + tui.strip_ansi(right)
        return style(plain.ljust(width), "reverse")
    return "  " + left + pad + right


def eta_lines(node: qm.Node, now: float) -> list[str]:
    """What the ETA of a node rests on, in words."""
    if node.total and node.done >= node.total:
        return ["ETA: nothing left", *took_lines(node)]
    head = f"ETA: {qm.eta_text(node)}"
    if node.left:
        head += f"  (finishes about {qm.clock(node.left, now)})"
    lines = [head]
    basis = {"measured": "from this run's own timings", "estimate": "estimated: from other work of the run, earlier records or a "
             "planning constant, scaled by edges where needed", "bound": "a lower bound: some of the work under it is "
             "not in the estimate (the run's notes say which)"}.get(node.basis)
    if basis:
        lines.append(f"  {basis}")
    if node.unknown:
        lines.append(f"  {node.unknown:,} unfinished unit(s) under it have no estimate at all, so the real wait is longer")
    return lines + took_lines(node)


def took_lines(node: qm.Node) -> list[str]:
    """How long the finished work under a node took, and what that number is."""
    if not node.took:
        return ["took: no time was recorded for the finished work under it"] if node.done else []
    lines = [f"took: {qm.took_text(node)}",
             "  the sum of the finished units' recorded seconds, read from the files, so a unit the run found already done "
             "and skipped counts too; units that ran side by side count their share of the wall-clock"]
    if node.took_missing:
        lines.append(f"  {node.took_missing:,} finished unit(s) recorded no time, so this is a lower bound (≥)")
    return lines


def detail_lines(node: qm.Node, width: int, now: float) -> list[str]:
    head = f"{node.label} · {node.status}" + (f" · {node.done:,} of {node.total:,} {node.noun} done" if node.total else "")
    lines = [head, *eta_lines(node, now), ""] + list(node.detail)
    out = []
    for line in lines:
        lead = len(line) - len(line.lstrip())
        out += textwrap.wrap(line.strip(), max(20, width - 2), initial_indent=" " * lead,
                             subsequent_indent=" " * (lead + 2)) or [""]
    return out


HELP = [
    "move             ↑ ↓  or  j k          PgUp PgDn  Home End  (or g G)",
    "open / close     → ←  or  l h          ← on a closed row goes to its parent",
    "toggle           Enter  Space          click on the ▸ ▾ (or click a selected row)",
    "subtree          e opens it  c closes it",
    "whole queue      E opens everything  C closes all but the runs   1-9 jump to run N",
    "scroll           mouse wheel (or trackpad) scrolls the list; the selection stays",
    "detail pane      d cycles small / large / off",
    "mouse            m turns mouse capture off, so you can select text with it (press again to turn it on)",
    "close            q or Ctrl-C closes this view only; the queue keeps running",
    "ETA marks        17:33 measured · ~17:33 estimated · ≥17:33 part of the work has no estimate · ? none",
    "took             done 1:23:45 = finished, and how long it took (its units' recorded seconds, read from the files, so work the",
    "                 run skipped as already done counts too; ≥ when some finished unit recorded no time)",
    "state marks      ✓ done · spinner: being worked on now (every level down to the unit) · ◐ part done, nothing working on it ·",
    "                 ■ stopped · ✗ failed · · not started",
    "counts           a row's count is the sum of the rows under it; select a row to see which loops multiply into it",
]


def render(root: qm.Node, view: View, size: tuple[int, int], G: dict, header: str = "", frame: str = "●",
           now: float | None = None, notice: list[str] | None = None) -> list[str]:
    """The whole screen as ``size[1]`` lines (columns, rows). ``notice`` is shown under the header: where and why a
    queue stopped (never more than a quarter of the screen)."""
    columns, lines = size
    now = time.time() if now is None else now
    rows = flatten(root, view)
    lay = layout_for(rows, columns)
    out = [fit(header, columns) if header else style(root.label, "bold")]
    out += [fit(line, columns) for line in (notice or [])[:max(0, lines // 4)]]
    out.append(column_heads(columns, lay))
    if view.help:
        body = [*[style(l, "dim") for l in HELP]]
        out += [fit(line, columns) for line in body]
        out += [""] * max(0, lines - 1 - len(out))
        out.append(style("  any key closes the help", "dim"))
        view.body = (2, 0)
        return out[:lines]
    selected = next((n for n in rows if n.key == view.cursor), rows[0])
    detail = detail_lines(selected, columns, now) if view.detail_mode else []
    detail_room = {0: 0, 1: min(len(detail), max(6, lines // 3)), 2: min(len(detail), max(8, lines * 3 // 5))}[view.detail_mode]
    detail_room = min(detail_room, max(0, lines - 8))
    room = max(3, lines - len(out) - (detail_room + 1 if detail_room else 0) - 1)
    view.offset = max(0, min(view.offset, max(0, len(rows) - room)))
    shown = rows[view.offset:view.offset + room]
    out += [fit(row_text(node, view, columns, G, frame, lay), columns) for node in shown]
    out += [""] * (room - len(shown))
    view.body = (len(out) - room, room)
    if detail_room:
        title = f" {selected.label} "
        out.append(style(f"──{title}{'─' * max(0, columns - len(title) - 3)}", "dim"))
        extra = len(detail) - detail_room
        out += [fit(line, columns) for line in detail[:detail_room - 1 if extra > 0 else detail_room]]
        if extra > 0:
            out.append(style(f"  … {extra + 1} more lines (d enlarges)", "dim"))
    pos = f"{index_of(rows, view.cursor) + 1}/{len(rows)}"
    hint = (view.message or "↑↓ move · ←→ close/open · wheel scrolls · click selects · e/c subtree · E/C all · d detail · "
            "m mouse · ? help · q close (the queue keeps running)")
    out.append(style(fit(f"  {pos}  {hint}", columns), "dim"))
    return out[:lines]


# --------------------------------------------------------------------------- input
KEYS = {"\x1b[A": "up", "\x1bOA": "up", "\x1b[B": "down", "\x1bOB": "down", "\x1b[C": "right", "\x1bOC": "right",
        "\x1b[D": "left", "\x1bOD": "left", "\x1b[5~": "pgup", "\x1b[6~": "pgdn", "\x1b[H": "home", "\x1bOH": "home",
        "\x1b[1~": "home", "\x1b[F": "end", "\x1bOF": "end", "\x1b[4~": "end", "\x1b[3~": "delete"}
_MOUSE = re.compile(r"\x1b\[<(\d+);(\d+);(\d+)([Mm])")
_PARTIAL = re.compile(r"\x1b(\[[0-9;<]*|O)?$")


def parse(data: str) -> tuple[list[tuple], str]:
    """Events in a chunk of terminal input, and the unfinished escape sequence at its end (to prepend to the next chunk).

    Events: ``("key", name)``, ``("wheel", +1|-1, x, y)`` (+1 = towards later rows), ``("click", x, y)``.
    """
    events: list[tuple] = []
    i = 0
    while i < len(data):
        ch = data[i]
        if ch == "\x1b":
            match = _MOUSE.match(data, i)
            if match:
                button, x, y, kind = int(match[1]), int(match[2]), int(match[3]), match[4]
                if button & 64 and kind == "M":
                    events.append(("wheel", 1 if button & 1 else -1, x, y))
                elif button & 3 == 0 and not button & 32 and kind == "M":
                    events.append(("click", x, y))
                i = match.end()
                continue
            for length in (4, 3):
                if data[i:i + length] in KEYS:
                    events.append(("key", KEYS[data[i:i + length]]))
                    i += length
                    break
            else:
                if _PARTIAL.match(data, i):
                    return events, data[i:]
                i += 1
            continue
        if ch in "\r\n":
            events.append(("key", "enter"))
        elif ch == " ":
            events.append(("key", "space"))
        elif ch == "\x03":
            events.append(("key", "quit"))
        elif ch.isprintable():
            events.append(("key", ch))
        i += 1
    return events, ""


def _move(view: View, rows: list[qm.Node], to: int) -> None:
    to = max(0, min(len(rows) - 1, to))
    view.cursor = rows[to].key
    top, room = view.offset, max(1, view.body[1])
    if to < top:
        view.offset = to
    elif to >= top + room:
        view.offset = to - room + 1


def _toggle(view: View, node: qm.Node) -> None:
    if node.children:
        view.expanded ^= {node.key}


def apply(view: View, root: qm.Node, event: tuple, now: float | None = None) -> bool:
    """Change the view for one event; returns True when the viewer should close."""
    now = time.time() if now is None else now
    rows = view.rows or flatten(root, view)
    here = index_of(rows, view.cursor)
    node = rows[here]
    view.message = ""
    if view.help:
        view.help = False
        return False
    kind = event[0]
    if kind == "wheel":
        view.offset = max(0, view.offset + event[1] * WHEEL_STEP)
        return False
    if kind == "click":
        _x, y = event[1], event[2]
        top, room = view.body
        row = y - 1 - top
        if 0 <= row < room and view.offset + row < len(rows):
            clicked = rows[view.offset + row]
            caret_x = 3 + 2 * min(clicked.depth, 8)  # the ▸ ▾ sits after the selection mark and the indent
            again = view.last_click[1] == clicked.key and now - view.last_click[0] < 0.6
            if clicked.key == view.cursor or again or abs(event[1] - 1 - caret_x) <= 1:
                _toggle(view, clicked)
            view.cursor = clicked.key
            view.last_click = (now, clicked.key)
        return False
    key = event[1]
    if key in ("q", "quit"):
        return True
    if key in ("up", "k"):
        _move(view, rows, here - 1)
    elif key in ("down", "j"):
        _move(view, rows, here + 1)
    elif key == "pgup":
        _move(view, rows, here - max(1, view.body[1] - 1))
    elif key == "pgdn":
        _move(view, rows, here + max(1, view.body[1] - 1))
    elif key in ("home", "g"):
        _move(view, rows, 0)
    elif key in ("end", "G"):
        _move(view, rows, len(rows) - 1)
    elif key in ("right", "l"):
        if node.children and node.key not in view.expanded:
            view.expanded.add(node.key)
        elif node.children:
            _move(view, rows, here + 1)
    elif key in ("left", "h"):
        if node.key in view.expanded and node.children:
            view.expanded.discard(node.key)
        elif node.key in view.parents:
            _move(view, rows, index_of(rows, view.parents[node.key].key))
    elif key in ("enter", "space"):
        _toggle(view, node)
    elif key == "e":
        view.expanded |= {n.key for n in descendants(node) if n.children}
    elif key == "c":
        view.expanded -= {n.key for n in descendants(node)}
        view.cursor = node.key
    elif key == "E":
        view.expanded |= {n.key for n in descendants(root) if n.children}
    elif key == "C":
        view.expanded = {root.key}
        view.cursor = node.key if node.key == root.key or node.key in view.parents and view.parents[node.key].key == root.key else root.key
        view.offset = 0
    elif key.isdigit() and key != "0":
        runs = root.children
        if int(key) <= len(runs):
            view.expanded.add(root.key)
            flatten(root, view)
            _move(view, view.rows, index_of(view.rows, runs[int(key) - 1].key))
    elif key == "d":
        view.detail_mode = (view.detail_mode + 1) % 3
    elif key == "m":
        view.mouse = not view.mouse
        view.message = "mouse capture " + ("on: wheel and clicks work" if view.mouse else "off: select text with the mouse")
    elif key == "?":
        view.help = True
    return False


# --------------------------------------------------------------------------- the loop
ALT_ON = "\033[?1049h\033[?25l"
ALT_OFF = "\033[?25h\033[?1049l"
MOUSE_ON = "\033[?1000h\033[?1006h"
MOUSE_OFF = "\033[?1006l\033[?1000l"
SYNC_ON, SYNC_OFF = "\033[?2026h", "\033[?2026l"  # synchronized update: the terminal shows the frame at once (others ignore it)


def paint(previous: list[str] | None, lines: list[str]) -> str:
    """What to write to turn a screen showing ``previous`` into ``lines``: only the rows that changed, each at its own
    position, inside a synchronized update; "" when nothing changed (``previous`` None: every row).

    Rewriting the whole screen five times a second sent ~30 kB/s to a terminal that showed nothing new, and over SSH a
    client paints what has arrived, so frames showed half-drawn: the selected row's highlight flickered in and out."""
    changed = [i for i, line in enumerate(lines) if previous is None or i >= len(previous) or previous[i] != line]
    clear = previous is not None and len(previous) > len(lines)
    if not changed and not clear:
        return ""
    parts = [SYNC_ON] + [f"\033[{i + 1};1H{lines[i]}\033[K" for i in changed]
    if previous is None or clear:
        parts.append(f"\033[{len(lines) + 1};1H\033[J")
    return "".join(parts) + SYNC_OFF


def header_line(root: qm.Node, status: str, elapsed: float | None, G: dict, now: float | None = None) -> str:
    """The pinned line: the queue, its state, and the ETA of the whole queue."""
    now = time.time() if now is None else now
    colour = {"running": "cyan", "completed": "green", "failed": "red", "stopped": "yellow", "interrupted": "red"}.get(status, "dim")
    parts = [style(f"queue {root.label.removeprefix('queue ')}", "bold"), style(f"{G['dot']} {status}", colour)]
    if root.total and root.done >= root.total:
        parts.append(style(f"all {root.total} runs complete", "green"))
    else:
        eta = f"ETA {qm.eta_text(root)}"
        if root.left:
            eta += f" (≈ {qm.clock(root.left, now)})"
        parts += [f"{root.done}/{root.total} runs", style(eta, "bold")]
        unestimated = sum(1 for r in root.children if r.done < r.total and r.unknown and not r.left)
        if unestimated:
            parts.append(f"{unestimated} without estimate")
    if elapsed is not None:
        parts.append(f"elapsed {qm.fmt_seconds(elapsed)}")
    return " · ".join(parts)


class Builder(threading.Thread):
    """Rebuilds the tree from the files on its own thread, so reading a big run never stalls the keyboard."""

    def __init__(self, build: Callable[[], qm.Node], every: float = REFRESH_SECONDS):
        super().__init__(daemon=True)
        self.build, self.every = build, every
        self.root: qm.Node | None = None
        self.error = ""
        self.stamp = 0
        self.stop = threading.Event()
        self.wake = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                self.root, self.error = self.build(), ""
                self.stamp += 1
            except Exception as exc:  # noqa: BLE001 - the view stays up and says what failed
                self.error = f"{type(exc).__name__}: {exc}"
            self.wake.wait(self.every)
            self.wake.clear()


def run(build: Callable[[], qm.Node], status: Callable[[], tuple[str, float | None]],
        notice: Callable[[], list[str]] | None = None) -> int:
    """Show the queue until the user closes the view. ``build`` makes the tree, ``status`` gives (state, elapsed) and
    ``notice`` the lines that say where and why a stopped queue stopped."""
    G = tui.glyphs(True) if tui.unicode_output() else tui.glyphs(False)
    G = {**G, "open": "▾" if G["full"] == "█" else "v", "closed": "▸" if G["full"] == "█" else ">"}
    view = View()
    fd = sys.stdin.fileno()
    saved = tui.termios.tcgetattr(fd)
    builder = Builder(build)
    builder.start()
    out = sys.stdout
    tui.tty.setcbreak(fd)
    out.write(ALT_ON + MOUSE_ON)
    mouse_on, spin, carry = True, 0, ""
    shown: list[str] | None = None  # what the screen shows now (None: unknown, so the next frame draws every row)
    shown_size = None
    try:
        while builder.root is None and not builder.error:
            time.sleep(0.05)
        while True:
            root = builder.root
            if root is not None:
                size = shutil.get_terminal_size((100, 30))
                columns = max(20, size.columns - 1)  # a line that fills the last column loses its last character to the erase
                state, elapsed = status()
                frames = G["spin"]
                lines = render(root, view, (columns, size.lines), G, header_line(root, state, elapsed, G),
                               frames[spin % len(frames)], notice=notice() if notice else None)
                if builder.error:
                    lines[-1] = style(fit(f"  could not refresh: {builder.error}", columns), "red")
                lines = [fit(line, columns) for line in lines]
                if (size.columns, size.lines) != shown_size:  # a resized terminal: draw every row again
                    shown, shown_size = None, (size.columns, size.lines)
                text = paint(shown, lines)
                if text:
                    out.write(text)
                    out.flush()
                shown = lines
                if view.mouse != mouse_on:
                    out.write(MOUSE_ON if view.mouse else MOUSE_OFF)
                    out.flush()
                    mouse_on = view.mouse
            spin += 1
            if select.select([fd], [], [], TICK)[0]:
                data = carry + os.read(fd, 4096).decode(errors="ignore")
                events, carry = parse(data)
                for event in events:
                    if root is not None and apply(view, root, event):
                        return 0
                if events:
                    builder.wake.set()
    except KeyboardInterrupt:
        return 0
    finally:
        builder.stop.set()
        builder.wake.set()
        out.write(MOUSE_OFF + ALT_OFF)
        out.flush()
        tui.termios.tcsetattr(fd, tui.termios.TCSADRAIN, saved)
