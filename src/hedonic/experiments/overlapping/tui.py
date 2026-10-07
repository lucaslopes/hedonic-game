"""Minimal dependency-free terminal prompts (arrow keys, space, enter).

Falls back to numbered line input when stdin/stdout is not a terminal or the
platform has no ``termios`` (e.g. Windows), so every prompt also works in pipes
and notebooks.
"""

from __future__ import annotations

import os
import re
import select as _select
import shutil
import sys
from collections.abc import Sequence

try:  # POSIX only
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


class Cancelled(Exception):
    """The user pressed q / Ctrl-C / Esc."""


def interactive() -> bool:
    return termios is not None and sys.stdin.isatty() and sys.stdout.isatty()


def color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def style(text: str, *codes: str) -> str:
    if not color():
        return text
    table = {"bold": "1", "dim": "2", "cyan": "36", "green": "32", "yellow": "33", "red": "31", "magenta": "35",
             "reverse": "7"}
    return f"\033[{';'.join(table[c] for c in codes)}m{text}\033[0m"


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1).decode(errors="ignore")
        if ch == "\x1b":
            # A bare Esc sends only this byte; arrows send "[A" etc. straight after it.
            if not _select.select([fd], [], [], 0.05)[0]:
                return "esc"
            seq = os.read(fd, 2).decode(errors="ignore")
            return {"[A": "up", "[B": "down", "[C": "right", "[D": "left", "OA": "up", "OB": "down"}.get(seq, "esc")
        if ch in ("\r", "\n"):
            return "enter"
        if ch in ("\x03", "\x04"):
            return "ctrl-c"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


_ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def fit(text: str, columns: int | None = None) -> str:
    """Truncate to the terminal width by visible characters (ANSI-aware), so a line never wraps."""
    columns = columns or shutil.get_terminal_size((100, 24)).columns
    if visible_len(text) <= columns:
        return text
    out, seen, i = [], 0, 0
    while i < len(text) and seen < columns - 1:
        match = _ANSI.match(text, i)
        if match:
            out.append(match.group())
            i = match.end()
        else:
            out.append(text[i])
            seen += 1
            i += 1
    return "".join(out) + "…" + ("\033[0m" if "\033" in text else "")


def _render(lines: list[str], previous: int) -> int:
    """Redraw a prompt in place. Every line is fitted to one terminal row, so the cursor maths holds."""
    if previous:
        sys.stdout.write(f"\033[{previous}F\033[J")
    lines = [fit(line) for line in lines]
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    return len(lines)


class _hidden_cursor:
    """Hide the text cursor while an arrow-key prompt is open; always restore it (even on Ctrl-C)."""

    def __enter__(self):
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def __exit__(self, *exc):
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


def _ask_number(prompt: str, default: str, low: int, high: int) -> int:
    """Line input for the numbered fallback; re-asks on garbage, raises Cancelled on EOF/Ctrl-C."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise Cancelled from exc
        raw = raw or default
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(style(f"  enter a number from {low} to {high}", "yellow"))


def _labels(options: Sequence) -> list[tuple[str, str]]:
    """Options may be strings or (label, hint) tuples."""
    return [(o, "") if isinstance(o, str) else (o[0], o[1]) for o in options]


def _column(label: str, opts: list[tuple[str, str]]) -> str:
    """Pad labels to a common width so the dim hints line up in a column."""
    width = max((len(l) for l, hint in opts if hint), default=0)
    return label.ljust(width)


def select(title: str, options: Sequence, default: int = 0, help_text: str = "") -> int:
    """Single choice; returns the index."""
    opts = _labels(options)
    if not interactive():
        print(title)
        for i, (label, hint) in enumerate(opts, 1):
            print(f"  {i}) {label}" + (f"  — {hint}" if hint else ""))
        return _ask_number(f"choice [{default + 1}]: ", str(default + 1), 1, len(opts)) - 1
    index, shown = default, 0
    with _hidden_cursor():
        while True:
            lines = [style(title, "bold"), style(help_text or "↑/↓ move · enter select · q quit", "dim")]
            for i, (label, hint) in enumerate(opts):
                pointer = style("❯ ", "cyan") if i == index else "  "
                shown_label = _column(label, opts) if hint else label
                text = style(shown_label, "cyan", "bold") if i == index else shown_label
                lines.append(pointer + text + (style(f"  {hint}", "dim") if hint else ""))
            shown = _render(lines, shown)
            key = _read_key()
            if key in ("up", "k"):
                index = (index - 1) % len(opts)
            elif key in ("down", "j"):
                index = (index + 1) % len(opts)
            elif key == "enter":
                _render([style(title, "bold") + "  " + style(opts[index][0], "green")], shown)
                return index
            elif key in ("q", "esc", "ctrl-c"):
                raise Cancelled


def multiselect(title: str, options: Sequence, selected: Sequence[int] = (), help_text: str = "") -> list[int]:
    """Multiple choice; returns sorted indices (at least one)."""
    opts = _labels(options)
    chosen = set(selected)
    if not interactive():
        print(title)
        for i, (label, hint) in enumerate(opts, 1):
            print(f"  {i}) [{'x' if i - 1 in chosen else ' '}] {label}" + (f"  — {hint}" if hint else ""))
        while True:
            try:
                raw = input("numbers, comma-separated [keep marked]: ").strip()
            except (EOFError, KeyboardInterrupt) as exc:
                raise Cancelled from exc
            if not raw:
                if chosen:
                    return sorted(chosen)
            else:
                parts = [x.strip() for x in raw.split(",") if x.strip()]
                if parts and all(x.isdigit() and 1 <= int(x) <= len(opts) for x in parts):
                    return sorted({int(x) - 1 for x in parts})
            print(style(f"  enter numbers from 1 to {len(opts)}, comma-separated (at least one)", "yellow"))
    index, shown, warn = 0, 0, ""
    with _hidden_cursor():
        while True:
            lines = [style(title, "bold"),
                     style(help_text or "↑/↓ move · space toggle · a all/none · enter confirm · q quit", "dim")]
            for i, (label, hint) in enumerate(opts):
                box = style("◉", "green") if i in chosen else "○"
                pointer = style("❯ ", "cyan") if i == index else "  "
                shown_label = _column(label, opts) if hint else label
                text = style(shown_label, "cyan", "bold") if i == index else shown_label
                lines.append(f"{pointer}{box} {text}" + (style(f"  {hint}", "dim") if hint else ""))
            if warn:
                lines.append(style(warn, "yellow"))
            shown = _render(lines, shown)
            key = _read_key()
            warn = ""
            if key in ("up", "k"):
                index = (index - 1) % len(opts)
            elif key in ("down", "j"):
                index = (index + 1) % len(opts)
            elif key == " ":
                chosen ^= {index}
            elif key == "a":
                chosen = set() if len(chosen) == len(opts) else set(range(len(opts)))
            elif key == "enter":
                if not chosen:
                    warn = "select at least one (space)"
                    continue
                summary = "all " + str(len(opts)) if len(chosen) == len(opts) and len(opts) > 3 else \
                    ", ".join(opts[i][0] for i in sorted(chosen))
                _render([style(title, "bold") + "  " + style(summary, "green")], shown)
                return sorted(chosen)
            elif key in ("q", "esc", "ctrl-c"):
                raise Cancelled


def text(title: str, default: str = "", validate=None) -> str:
    """Free text with a default (line editing by the terminal)."""
    while True:
        try:
            raw = input(f"{style(title, 'bold')} {style(f'[{default}]', 'dim')}: ").strip()
        except (EOFError, KeyboardInterrupt) as exc:
            raise Cancelled from exc
        value = raw or default
        error = validate(value) if validate else None
        if not error:
            return value
        print(style(f"  {error}", "yellow"))
