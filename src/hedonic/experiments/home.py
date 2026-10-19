"""``hedonic`` with no arguments in a terminal: a home screen.

A header says where things stand (version, local SNAP pairs, ready methods,
runs) and a menu starts the command that does each thing. Every choice prints
the ``hedonic ...`` command it runs before running it, so the menu teaches the
command line and can never do anything the command line cannot. Nothing starts
without a choice: every wizard ends with a yes/no before a run starts, and
resuming a run asks first. Views (lists, ``show``, the guide, settings) come back to the menu;
a choice that starts or watches a run hands the terminal over to it.

Outside a terminal (a pipe, a script, CI, ``HEDONIC_PLAIN_VIEW=1``) a bare
``hedonic`` prints the help text instead (``CLI.hedonic_main``).
"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass, field

from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import style

ACTIVE = ("running", "setup", "starting")
UNFINISHED = ("interrupted", "failed")


def wanted() -> bool:
    """Open the home screen for a bare ``hedonic``: only in an interactive terminal."""
    return tui.interactive() and not os.environ.get("HEDONIC_PLAIN_VIEW")


@dataclass
class State:
    version: str = "hedonic"
    network_root: str = ""
    pairs_present: int = 0
    pairs_known: int = 0
    methods_ready: int = 0
    methods_known: int = 0
    active: list[tuple[str, str]] = field(default_factory=list)      # (name, "running 12/40")
    unfinished: list[tuple[str, str]] = field(default_factory=list)  # (name, status)
    favourite: str | None = None
    config_path: str = "~/.config/hedonic/config.toml"


def _home_path(path: str, longest: int = 48) -> str:
    """``~/...`` under the home folder; a long path keeps its last parts (``…/hedonic/config.toml``)."""
    home = os.path.expanduser("~")
    path = "~" + path[len(home):] if path.startswith(home) else path
    if len(path) > longest:
        parts = path.split("/")
        tail = parts[-1]
        for part in reversed(parts[:-1]):
            if len(part) + len(tail) + 3 > longest:
                break
            tail = f"{part}/{tail}"
        path = "…/" + tail
    return path


def gather() -> State:
    """The header's facts from local state only: file existence, the run registry, the setup manifest.

    Nothing is parsed, downloaded or scanned (a queue's progress is not counted here: that reads its records).
    Each part that fails is left at its default, so the menu always opens.
    """
    from hedonic.experiments.CLI import _package_version

    state = State(version=_package_version())
    try:
        from hedonic.experiments.overlapping import snap

        root = snap.networks_root()
        state.network_root = _home_path(str(root))
        for network, spec in snap.SPECS.items():
            for variant in spec.raw_cover_files:
                state.pairs_known += 1
                state.pairs_present += snap.pair_files(network, variant, root) is not None
    except Exception:
        pass
    try:
        from hedonic.experiments.config import codeseg_cache_dir
        from hedonic.experiments.overlapping import info

        rows = info.method_status(codeseg_cache_dir())
        state.methods_known = len(rows)
        state.methods_ready = sum(r["status"] == "ready" for r in rows)
    except Exception:
        pass
    try:
        from hedonic.experiments.overlapping import runmanager

        for entry in runmanager.all_entries():
            prog = runmanager.read_progress(entry)
            status = runmanager.effective_status(entry, prog)
            if status in ACTIVE:
                detail = status
                if entry.get("kind") != "queue":
                    done, total = runmanager.progress_counts(entry, prog)
                    detail += f" {done}/{total}" if total else ""
                state.active.append((entry["name"], detail))
            elif status in UNFINISHED:
                state.unfinished.append((entry["name"], status))
    except Exception:
        pass
    try:
        from hedonic.experiments.overlapping import userconfig

        state.favourite = userconfig.favourite()
        state.config_path = _home_path(str(userconfig.path()))
    except Exception:
        pass
    return state


def header(state: State) -> list[str]:
    lines = [style(f"  {state.version}", "bold", "magenta") + style("  · hedonic games for community detection", "dim")]
    if state.network_root:
        lines.append(f"  {style('data', 'bold')}     {state.network_root} · "
                     f"{state.pairs_present} of {state.pairs_known} SNAP graph/cover pairs on disk")
    if state.methods_known:
        lines.append(f"  {style('methods', 'bold')}  {state.methods_ready} of {state.methods_known} overlapping methods "
                     "ready" + ("" if state.methods_ready == state.methods_known else style("  (hedonic setup)", "dim")))
    runs = [f"{name} {style(detail, 'cyan')}" for name, detail in state.active[-2:]]
    if state.unfinished:
        runs.append(style(f"{len(state.unfinished)} unfinished", "yellow"))
    lines.append(f"  {style('runs', 'bold')}     " + (" · ".join(runs) if runs else style("none yet", "dim")))
    return lines


@dataclass
class Item:
    label: str
    hint: str
    argv: list[str] | None   # None: quit
    back: bool = False       # a view: come back to the menu when it ends
    confirm: str = ""        # ask this before running it


def _commands() -> tuple[set[str], set[str], set[str]]:
    """What this line has: top-level commands, ``hedonic run`` verbs, bundled queues (hedonic 1 has the disjoint
    ones; hedonic 2 adds the overlap ones). The menu offers only those, so one file serves both lines."""
    from hedonic.experiments import CLI
    from hedonic.experiments.resources import CONFIG_DIR

    queues = {p.stem.removeprefix("queue-") for p in CONFIG_DIR.glob("queue-*.toml")}
    return set(CLI.TOP_LEVEL), {*CLI.RUN_ALIASES, *CLI.RUN_VERBS}, queues


def menu(state: State, commands: tuple[set[str], set[str], set[str]] | None = None) -> list[Item]:
    top, run, queues = commands or _commands()
    items: list[Item] = []
    for name, detail in reversed(state.active[-2:]):
        items.append(Item(f"Watch {name}", f"{detail} · Ctrl-C closes the view only", ["run", "attach", name]))
    if state.unfinished:
        name, status = state.unfinished[-1]
        items.append(Item(f"Resume {name}", f"{status}; finished work is reused", ["run", "resume", name],
                          confirm=f"Resume {name}?"))
    if "exp" in run:
        items.append(Item("Start a benchmark", "a quick default run (about a minute), the paper's, or your own",
                          ["run", "exp"]))
        if state.favourite:
            items.append(Item(f"Run {state.favourite}", "your favourite config (hedonic config set run NAME; asks first)",
                              ["run"]))
    if "detect" in top:
        networks = "SBM, LFR, a SNAP pair, a file..." if "setup" in top else "SBM, LFR, a famous graph, a file"
        items.append(Item("Detect communities", f"one network ({networks}) x one method x one seed, here; set every "
                          "parameter", ["detect"], back=True))
    if "overlap" in queues:
        items.append(Item("Plan the overlap paper queue", "the overlap paper's experiments: plan and time, nothing starts",
                          ["run", "queue", "overlap", "--dry-run"], back=True))
    if "disjoint" in queues:
        items.append(Item("Plan the disjoint paper queue", "the smoke check, then the 100-seed grid: plan, nothing starts",
                          ["run", "queue", "disjoint", "--dry-run"], back=True))
    items += [
        Item("Disjoint SBM benchmark", "planted partitions: a seconds-long smoke run, or the paper's grid",
             ["run", "disjoint"]),
        Item("Runs", "every run and its progress", ["run", "list"], back=True),
    ]
    if "setup" in top:
        items += [
            Item("Networks & covers", "the SNAP graph/cover pairs on disk", ["show", "covers"], back=True),
            Item("Methods", "which methods are ready, which need hedonic setup", ["show", "methods"], back=True),
            Item("Set up data and methods", "what hedonic setup would download and build (a dry run)",
                 ["setup", "--dry-run"], back=True),
        ]
    else:
        items.append(Item("Disjoint presets", "grids, methods and estimates", ["show", "disjoint"], back=True))
    items += [
        Item("Guide", "a short tour and suggested next steps", ["guide"], back=True),
        Item("Settings", state.config_path, ["config", "show"], back=True),
        Item("Check for updates", "compare with PyPI; never installs from here", ["update"], back=True),
        Item("Quit", "", None),
    ]
    return items


def command_line(argv: list[str]) -> str:
    return "hedonic " + " ".join(shlex.quote(a) for a in argv) if argv else "hedonic"


def _pause() -> bool:
    """After a view: True to go back to the menu."""
    try:
        choice = tui.select("", [("Back to the menu", ""), ("Quit", "")], help_text="enter select · q quit")
    except tui.Cancelled:
        return False
    return choice == 0


def main(argv: list[str] | None = None, *, run=None) -> int:
    """The home screen loop. ``run(argv) -> int`` executes a command (default: ``CLI.hedonic_main``)."""
    if run is None:
        from hedonic.experiments.CLI import hedonic_main as run
    last = ""  # the label chosen last: the cursor comes back to it
    while True:
        state = gather()
        items = menu(state)
        print("\n".join([""] + header(state) + [""]))
        labels = [i.label for i in items]
        try:
            picked = items[tui.select("What would you like to do?", [(i.label, i.hint) for i in items],
                                      default=labels.index(last) if last in labels else 0)]
        except tui.Cancelled:
            return 0
        last = picked.label
        if picked.argv is None:
            return 0
        if picked.confirm:
            try:
                if tui.select(picked.confirm, [("Yes", command_line(picked.argv)), ("No", "back to the menu")], default=1) != 0:
                    continue
            except tui.Cancelled:
                continue
        print(style(f"\n  → {command_line(picked.argv)}\n", "dim"))
        sys.stdout.flush()
        tui.take_cancelled()
        try:
            code = run(list(picked.argv))
        except KeyboardInterrupt:
            code = 130
        except SystemExit as exc:  # argparse inside a command
            code = exc.code if isinstance(exc.code, int) else 1
        if tui.take_cancelled():  # the user backed out of a wizard: nothing started, so the menu again
            continue
        if not picked.back:
            return int(code or 0)
        print()
        if not _pause():
            return 0
