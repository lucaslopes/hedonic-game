"""Command-line front doors of the hedonic experiments (hedonic 1: the disjoint paper).

Two console scripts share this module:

* ``hedonic`` (:func:`hedonic_main`) is the task-oriented front door: ``hedonic
  run`` starts and manages durable runs (``hedonic run disjoint --preset
  paper`` is the disjoint paper's SBM benchmark, ``hedonic run queue
  disjoint`` its queue), and ``hedonic show``, ``guide``, ``config`` and
  ``update`` inspect and configure;
* ``hedonic-exp`` (:func:`main`) keeps every experiment module's own command
  line (``hedonic-exp list``). A command that a ``hedonic`` verb performs
  durably prints a one-line notice naming that verb.

The overlapping-communities experiments (SNAP benchmark, calibrations,
robustness spectrum, setup of external methods) are hedonic 2; their verbs
fail here with a message that says so.

Examples::

    hedonic --help
    hedonic run disjoint --preset smoke --dry-run
    hedonic run queue disjoint --dry-run
    hedonic-exp list
    hedonic-exp smoke

**Agents:** when you add, remove, rename, or change the CLI of an experiment
module under ``hedonic.experiments``, update this file (registry and help text)
in the same change.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Command:
    """Registered experiment subcommand."""

    name: str
    module: str
    attr: str
    summary: str
    needs_data: str | None = None  # "synthetic" | None
    has_argparse_help: bool = True


# ---------------------------------------------------------------------------
# Registry: the single source of truth for `hedonic-exp` subcommands.
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Command] = {
    "smoke": Command(
        name="smoke",
        module="",  # handled specially
        attr="",
        summary="Isolated local check (no data): a tiny disjoint SBM sweep",
        has_argparse_help=False,
    ),
    "disjoint": Command(
        name="disjoint",
        module="hedonic.experiments.disjoint.sbm_sweep",
        attr="main",
        summary=(
            "SBM parameter sweep for disjoint community detection "
            "(presets: sbm, sbm-smoke -> Parquet store; v1020, v1020-smoke -> JSON)"
        ),
    ),
    "disjoint-store": Command(
        name="disjoint-store",
        module="hedonic.experiments.disjoint.store",
        attr="main",
        summary="Summarize, query (DuckDB SQL), export and replay-verify a disjoint Parquet result store",
    ),
    "disjoint-stats": Command(
        name="disjoint-stats",
        module="hedonic.experiments.disjoint.paper_stats",
        attr="main",
        summary=(
            "Every statistic of the disjoint paper (grid counts, SNR threshold, "
            "GT robustness, tracking means, SNR-band tables) from a result store"
        ),
    ),
    "disjoint-load": Command(
        name="disjoint-load",
        module="hedonic.experiments.disjoint.data_loader",
        attr="main",
        summary="Combine disjoint JSON results into a gzipped CSV",
        needs_data="synthetic",
    ),
    "plots": Command(
        name="plots",
        module="hedonic.experiments.plots.paper_figures",
        attr="main",
        summary="Disjoint paper figures from a result CSV (gt_robustness, noise, n_communities, acc_*)",
        needs_data="synthetic",
    ),
    "reproduce-disjoint": Command(
        name="reproduce-disjoint",
        module="hedonic.experiments.disjoint.reproduce",
        attr="main",
        summary=(
            "End-to-end disjoint reproduction: sweep -> store/CSV -> paper figures "
            "(presets: sbm, sbm-smoke, v1020, v1020-smoke)"
        ),
    ),
}

SUBCOMMANDS: tuple[str, ...] = tuple(COMMANDS.keys())



def _posix_only() -> int | None:
    """The experiment command line runs on macOS and Linux (tmux, fork, flock); on Windows only the library works."""
    if sys.platform == "win32":
        print("hedonic: the experiment command line runs on macOS and Linux; on Windows use the library "
              "(from hedonic import Game).", file=sys.stderr)
        return 2
    return None


# Distributions the `cli` extra installs (pyproject.toml), by import name.
CLI_MODULES = ("scipy", "pandas", "matplotlib", "pyarrow", "duckdb", "tqdm")


def _cli_extra() -> int | None:
    """``pip install hedonic`` is the library alone; the command line needs ``hedonic[cli]``."""
    import importlib.util

    missing = [name for name in CLI_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print(f"hedonic: the command line needs the cli extra ({', '.join(missing)} not installed):\n"
              f'  pip install "hedonic[cli]"', file=sys.stderr)
        return 2
    return None


def main(argv: list[str] | None = None) -> int:
    if (refused := _posix_only() or _cli_extra()) is not None:
        return refused
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return 0

    if argv[0] in ("-V", "--version"):
        print(_package_version())
        return 0

    if argv[0] == "list":
        _print_command_list()
        return 0

    command = argv[0]
    rest = argv[1:]

    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run `hedonic-exp list` or `hedonic-exp --help`.", file=sys.stderr)
        return 2

    equivalent = _run_equivalent(command, rest)
    if equivalent:
        print(f"hedonic-exp {command} is deprecated; use: {equivalent}", file=sys.stderr)

    if rest and rest[0] in ("-h", "--help"):
        return _dispatch_help(command)

    return _dispatch(command, rest)


# ``hedonic run VERB``: the experiment each run verb starts (its hedonic-exp command).
RUN_ALIASES: dict[str, str] = {"disjoint": "disjoint"}
# ``hedonic run VERB`` for the durable runs themselves (runmanager.VERBS).
RUN_VERBS = ("list", "status", "attach", "logs", "stop", "resume", "check", "queue")
# hedonic-exp commands that a `hedonic` command performs durably; the deprecation notice names it. In hedonic 1
# only the disjoint sweep has one, and it depends on the preset (``_run_equivalent``).
RUN_EQUIVALENTS: dict[str, str] = {}
# Top-level `hedonic` commands.
TOP_LEVEL = ("run", "show", "guide", "config", "update")
# Verbs of the overlapping experiments, which hedonic 1 does not have: they fail with a pointer to hedonic 2.
HEDONIC_2_RUN_VERBS = ("exp", "paper", "paper-hoc", "spectrum", "spectrum-full", "posthoc", "controlled",
                       "startcurve", "audit", "calibrate", "calibrate-baselines")
HEDONIC_2_COMMANDS = ("setup", "pickle")
HEDONIC_2 = ("hedonic 1 reproduces the disjoint paper (the SBM benchmark) only; the overlapping-communities "
             "experiments are hedonic 2. See `hedonic --help`.")


def _preset_option(rest: list[str]) -> str | None:
    for index, arg in enumerate(rest):
        if arg == "--preset" and index + 1 < len(rest):
            return rest[index + 1]
        if arg.startswith("--preset="):
            return arg.partition("=")[2]
    return None


def _run_equivalent(command: str, rest: list[str]) -> str | None:
    """The `hedonic` command that performs what ``hedonic-exp COMMAND REST`` does durably, else None."""
    if command in RUN_EQUIVALENTS:
        return RUN_EQUIVALENTS[command]
    if command in ("disjoint", "reproduce-disjoint") and "--plots-only" not in rest:
        preset = _preset_option(rest) or ("sbm-smoke" if command == "reproduce-disjoint" else None)
        if preset in ("sbm", "sbm-smoke"):  # v1020 and the custom grid have no run equivalent
            return "hedonic run disjoint"
    return None


HEDONIC_HELP = """\
usage: hedonic COMMAND [options]

Disjoint communities (planted-partition SBM: the disjoint paper)
  hedonic run disjoint [options] the SBM benchmark: store, figures and paper statistics (wizard without options)
  hedonic run disjoint --preset smoke   the whole pipeline in seconds
  hedonic run disjoint --preset paper   the paper's 100-seed grid (50,000 graphs)
  hedonic run queue disjoint     the disjoint paper's queue: the smoke check, then the paper grid (--dry-run plans)
Every run (durable: survives Ctrl-C, closed terminals and, on macOS, idle sleep)
  hedonic run list               runs and their progress
  hedonic run status [NAME]      snapshot of a run (default: latest)
  hedonic run attach [NAME]      live view of a running run
  hedonic run logs [NAME]        raw output (tmux pane or log file)
  hedonic run stop NAME          stop a run (explicit; never automatic)
  hedonic run resume [NAME]      continue an interrupted run; finished work is reused
  hedonic run check [NAME]       morning check: replay a sample of a disjoint run's rows from their seeds
  hedonic run queue FILE|NAME    run the experiments of a queue file one after another (overnight)
Settings and help
  hedonic show disjoint          disjoint presets, grids, methods and estimates
  hedonic guide [TOPIC]          a short tour of the CLI and suggested next steps
  hedonic config [set KEY VALUE] per-user machine settings (~/.config/hedonic/config.toml)
  hedonic update [--yes]         compare with the latest release on PyPI; upgrade only when asked

`hedonic run disjoint --help` and `hedonic run queue --help` list every option.
`hedonic-exp list` lists the experiment modules' own commands.
The overlapping-communities experiments (SNAP networks) are hedonic 2."""


def hedonic_main(argv: list[str] | None = None) -> int:
    if (refused := _posix_only() or _cli_extra()) is not None:
        return refused
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[:2] in (["run", "-h"], ["run", "--help"]):
        print(HEDONIC_HELP)
        return 0
    if argv[0] in ("-V", "--version"):
        print(_package_version())
        return 0
    if argv[0] in HEDONIC_2_COMMANDS:
        print(f"hedonic {argv[0]}: {HEDONIC_2}", file=sys.stderr)
        return 2
    if argv[0] == "guide":
        from hedonic.experiments.overlapping import guide

        return guide.main(argv[1:])
    if argv[0] == "update":
        from hedonic.experiments.overlapping import selfupdate

        return selfupdate.main(argv[1:])
    if argv[0] == "config":
        from hedonic.experiments.overlapping import userconfig

        return userconfig.main(argv[1:])
    if argv == ["run"] or (argv[0] == "run" and len(argv) > 1 and argv[1].startswith("-")):
        print("hedonic run: name what to run: hedonic run disjoint, hedonic run queue FILE|NAME, or a run verb "
              f"({', '.join(v for v in RUN_VERBS if v != 'queue')}). See `hedonic --help`.", file=sys.stderr)
        return 2
    if argv[0] == "show":
        from hedonic.experiments.overlapping import info

        return info.main(argv[1:])
    if argv[0] == "run" and argv[1] in HEDONIC_2_RUN_VERBS:
        print(f"hedonic run {argv[1]}: {HEDONIC_2}", file=sys.stderr)
        return 2
    if argv[0] != "run" or (argv[1] not in RUN_ALIASES and argv[1] not in RUN_VERBS):
        print(f"Unknown command: {' '.join(argv)}. Try `hedonic --help`.", file=sys.stderr)
        return 2
    if argv[1] in RUN_VERBS:
        from hedonic.experiments.overlapping import runmanager

        return runmanager.dispatch(argv[1:])
    # argv[1] == "disjoint": the disjoint SBM benchmark through the durable front door
    from hedonic.experiments.disjoint import front

    return front.front_main(argv[2:])


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return f"hedonic {version('hedonic')}"
    except Exception:
        return "hedonic (version unknown)"


def _print_root_help() -> None:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp",
        description=(
            "The hedonic experiment modules' own command lines: disjoint SBM sweeps, their result store, statistics "
            "and figures, and an isolated smoke check. Durable runs and queues are under the `hedonic` command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_examples_epilog(),
    )
    parser.add_argument(
        "command",
        choices=(*SUBCOMMANDS, "list"),
        help="Experiment or meta-command to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected experiment",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=_package_version(),
    )
    parser.print_help()
    print()
    _print_command_list()


def _examples_epilog() -> str:
    return """\
examples:
  # Isolated local check (no data)
  hedonic-exp smoke

  # Disjoint SBM benchmark (seeded graphs, Parquet store); durable form: hedonic run disjoint
  hedonic-exp reproduce-disjoint --preset sbm-smoke --output_root ~/Databases/Hedonic/Disjoint
  hedonic-exp disjoint-store ~/Databases/Hedonic/Disjoint/sbm-smoke --verify --all
  hedonic-exp disjoint-stats --data ~/Databases/Hedonic/Disjoint/sbm-smoke
  hedonic-exp disjoint --preset sbm --output_root ~/Databases/Hedonic/Disjoint --preflight
"""


def _print_command_list() -> None:
    print("Subcommands:")
    width = max(len(name) for name in COMMANDS)
    for name, cmd in COMMANDS.items():
        data = f"  [{cmd.needs_data}]" if cmd.needs_data else ""
        print(f"  {name:<{width}}  {cmd.summary}{data}")
    print()
    print("Meta:")
    print("  list       List subcommands")
    print("  -h/--help  This help")
    print("  -V/--version  Package version")
    print()
    print("Data paths (environment overrides; see also ~/.config/hedonic/config.toml):")
    print("  HEDONIC_DATA_DIR       data root (default ~/Databases/Hedonic)")
    print("  HEDONIC_SYNTHETIC_DIR  disjoint synthetic results")
    print("  HEDONIC_OUTPUT_DIR     hedonic-exp outputs (default <data root>/Overlap; disjoint under artifacts/disjoint)")
    print("  HEDONIC_DISJOINT_OUTPUT_DIR  disjoint runs (default <data root>/Disjoint)")
    print("  HEDONIC_RUNS_DIR       registry of durable runs (default ~/.cache/hedonic/runs)")


def _load_main(command: str) -> Callable:
    cmd = COMMANDS[command]
    if not cmd.module:
        raise ValueError(f"Command {command!r} has no module (special handler)")
    mod = importlib.import_module(cmd.module)
    return getattr(mod, cmd.attr)


def _dispatch_help(command: str) -> int:
    cmd = COMMANDS[command]
    if command == "smoke":
        print(
            "usage: hedonic-exp smoke [--keep-output]\n\n"
            "Isolated local check that needs no data:\n"
            "  disjoint --smoke into a temp directory\n\n"
            "Options:\n"
            "  --keep-output  Print and keep the temp output root "
            "(default: delete after run)\n"
        )
        return 0
    if not cmd.has_argparse_help:
        print(
            f"usage: hedonic-exp {command}\n\n"
            f"{cmd.summary}.\n"
            "No additional flags.\n"
        )
        return 0
    try:
        _load_main(command)(["--help"])
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    return 0


def _call_experiment(fn: Callable, rest: list[str]) -> int:
    """Invoke experiment main(); normalize return / SystemExit to exit code."""
    try:
        result = fn(rest)
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        # argparse / explicit SystemExit("message") — surface the text
        print(exc.code, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if result is None or result is True:
        return 0
    if result is False:
        return 1
    if isinstance(result, int):
        return result
    return 0


def _run_smoke(rest: list[str]) -> int:
    """Isolated reproduction check without external data."""
    keep = False
    unknown: list[str] = []
    for arg in rest:
        if arg == "--keep-output":
            keep = True
        elif arg in ("-h", "--help"):
            return _dispatch_help("smoke")
        else:
            unknown.append(arg)
    if unknown:
        print(f"Unknown smoke args: {unknown}", file=sys.stderr)
        return 2

    print("=" * 55)
    print("  hedonic-exp smoke  (isolated, no data required)")
    print("=" * 55)

    from hedonic.experiments.disjoint.sbm_sweep import main as sweep_main

    if keep:
        tmp = tempfile.mkdtemp(prefix="hedonic-smoke-")
        print(f"\n[smoke] disjoint SBM -> {tmp} (kept)")
        code = _call_experiment(
            sweep_main,
            ["--smoke", "--folder_name", "smoke", "--output_root", tmp],
        )
    else:
        with tempfile.TemporaryDirectory(prefix="hedonic-smoke-") as tmp:
            print(f"\n[smoke] disjoint SBM -> {tmp}")
            code = _call_experiment(
                sweep_main,
                ["--smoke", "--folder_name", "smoke", "--output_root", tmp],
            )

    if code != 0:
        return code
    print("\n[smoke] All isolated checks passed.")
    return 0


def _dispatch(command: str, rest: list[str]) -> int:
    if command == "smoke":
        return _run_smoke(rest)
    return _call_experiment(_load_main(command), rest)


if __name__ == "__main__":
    raise SystemExit(main())
