"""Command-line front doors of the hedonic experiments.

Two console scripts share this module:

* ``hedonic`` (:func:`hedonic_main`) is the task-oriented front door: ``hedonic
  run`` starts and manages durable runs (``hedonic run queue paper`` is the
  overlap paper's experiment queue, ``hedonic run disjoint --preset paper`` the
  disjoint paper's benchmark), ``hedonic setup`` downloads data and prepares the
  external methods, and ``hedonic show``, ``guide``, ``config``, ``pickle`` and
  ``update`` inspect and configure;
* ``hedonic-exp`` (:func:`main`) keeps every experiment module's own command
  line (``hedonic-exp list``). A command that a ``hedonic`` verb performs
  durably prints a one-line notice naming that verb.

Examples::

    hedonic --help
    hedonic setup --dry-run --all
    hedonic run queue paper --dry-run
    hedonic run disjoint --preset smoke --foreground
    hedonic run exp --yes
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
    needs_data: str | None = None  # "snap-networks" | "synthetic" | None
    has_argparse_help: bool = True


# ---------------------------------------------------------------------------
# Registry: the single source of truth for `hedonic-exp` subcommands.
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Command] = {
    "smoke": Command(
        name="smoke",
        module="",  # handled specially
        attr="",
        summary="Isolated local check (no data): small overlapping graphs + a tiny SBM sweep",
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
    "overlapping-small": Command(
        name="overlapping-small",
        module="hedonic.experiments.overlapping.small_graphs",
        attr="main",
        summary="Quick overlapping smoke tests on small graphs (no data)",
        has_argparse_help=False,
    ),
    "overlapping-controlled": Command(
        name="overlapping-controlled",
        module="hedonic.experiments.overlapping.controlled_overlap_config",
        attr="main",
        summary=(
            "LFR-derived controlled overlap: cap, initialization and resolution ablations "
            "(not canonical overlapping LFR; durable form: hedonic run controlled)"
        ),
    ),
    "start-quality": Command(
        name="start-quality",
        module="hedonic.experiments.overlapping.start_quality",
        attr="main",
        summary=(
            "Start-quality curve: Hedonic local moving from the SNAP ground-truth cover corrupted towards a random "
            "start (durable form: hedonic run startcurve)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-cover-audit": Command(
        name="overlapping-cover-audit",
        module="hedonic.experiments.overlapping.cover_audit",
        attr="main",
        summary=(
            "Robustness audit of the covers a benchmark saved: exact stable-vertex fraction over resolution, "
            "both action spaces (durable form: hedonic run audit)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-calibrate": Command(
        name="overlapping-calibrate",
        module="hedonic.experiments.overlapping.calibrate",
        attr="main",
        summary=(
            "Calibrate Hedonic's resolution and membership cap per network:cover pair and write hoc_tuned.json "
            "for runs that take --tuned (durable form: hedonic run calibrate)"
        ),
        needs_data="snap-networks",
    ),
    "baseline-calibrate": Command(
        name="baseline-calibrate",
        module="hedonic.experiments.overlapping.baseline_calibration",
        attr="main",
        summary=(
            "Label-free calibration of the baselines on G[S] only (ledger, budget, baseline_tuned.json; "
            "durable multi-pair form: hedonic run calibrate-baselines)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-tradeoffs": Command(
        name="overlapping-tradeoffs",
        module="hedonic.experiments.plots.benchmark_tradeoffs",
        attr="main",
        summary=(
            "Joint plots: accuracy vs detection time and accuracy vs cover robustness, from benchmark records and "
            "cover-audit ledgers (--smoke needs no data)"
        ),
    ),
    "participation-budget": Command(
        name="participation-budget",
        module="hedonic.experiments.plots.participation_budget",
        attr="main",
        summary="Model illustration: the unit-l2 participation 1/sqrt(k) of a vertex with k labels (no data)",
    ),
    "overlapping-codeseg": Command(
        name="overlapping-codeseg",
        module="hedonic.experiments.overlapping.codeseg_reproduction",
        attr="main",
        summary=(
            "The benchmark runner: one SNAP network x method x seed, scored with the shared cover metrics and "
            "LFK-ONMI (what hedonic run exp calls for every unit)"
        ),
        needs_data="snap-networks",
    ),
    "codeseg-setup": Command(
        name="codeseg-setup",
        module="hedonic.experiments.overlapping.codeseg_setup",
        attr="setup_main",
        summary="Download SNAP pairs and prepare the external method runtimes (= hedonic setup)",
        needs_data="snap-networks",
    ),
    "codeseg-doctor": Command(
        name="codeseg-doctor",
        module="hedonic.experiments.overlapping.codeseg_setup",
        attr="doctor_main",
        summary="Read-only readiness audit of the SNAP data and the method runtimes",
        needs_data="snap-networks",
    ),
    "snap-pickle": Command(
        name="snap-pickle",
        module="hedonic.experiments.overlapping.snap_pickle",
        attr="main",
        summary=(
            "Write the hedonic.Game of a SNAP network:cover pair (covered-induced, reference cover attached) to a "
            "pickle (front door: hedonic pickle)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-gt-spectrum": Command(
        name="overlapping-gt-spectrum",
        module="hedonic.experiments.overlapping.gt_spectrum",
        attr="main",
        summary=(
            "SNAP ground-truth robustness spectrum: stable-vertex fraction vs resolution of every local cover, plus "
            "local moving started from it (durable form: hedonic run spectrum)"
        ),
        needs_data="snap-networks",
    ),
    "overlapping-gt-posthoc": Command(
        name="overlapping-gt-posthoc",
        module="hedonic.experiments.overlapping.gt_posthoc",
        attr="main",
        summary=(
            "Exact post-hoc stability of a SNAP ground-truth cover: per-vertex resolution intervals "
            "(durable form: hedonic run posthoc)"
        ),
        needs_data="snap-networks",
    ),
    "info": Command(
        name="info",
        module="hedonic.experiments.overlapping.info",
        attr="main",
        summary=(
            "Show the methods (ready / installable / missing tools), the SNAP networks and covers, and the "
            "accuracy metrics (= hedonic show)"
        ),
    ),
    "quickstart": Command(
        name="quickstart",
        module="hedonic.experiments.overlapping.quickstart",
        attr="main",
        summary="Overlapping benchmark front door: networks x methods x seeds, durable (= hedonic run exp)",
        needs_data="snap-networks",
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
CLI_MODULES = ("scipy", "pandas", "matplotlib", "networkx", "pyarrow", "duckdb", "tqdm")


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
RUN_ALIASES: dict[str, str] = {"exp": "quickstart", "paper": "quickstart", "paper-hoc": "quickstart",
                               "spectrum": "overlapping-gt-spectrum", "spectrum-full": "overlapping-gt-spectrum",
                               "posthoc": "overlapping-gt-posthoc",
                               "controlled": "overlapping-controlled",
                               "startcurve": "start-quality",
                               "audit": "overlapping-cover-audit",
                               "calibrate": "overlapping-calibrate",
                               "calibrate-baselines": "baseline-calibrate",
                               "disjoint": "disjoint"}
# ``hedonic run VERB`` for the durable runs themselves (runmanager.VERBS).
RUN_VERBS = ("list", "status", "attach", "logs", "stop", "resume", "check", "queue")
# hedonic-exp commands that a `hedonic` command performs durably; the deprecation notice names it.
RUN_EQUIVALENTS: dict[str, str] = {"quickstart": "hedonic run exp", "overlapping-gt-spectrum": "hedonic run spectrum",
                                   "overlapping-gt-posthoc": "hedonic run posthoc",
                                   "overlapping-controlled": "hedonic run controlled",
                                   "codeseg-setup": "hedonic setup"}
# Top-level `hedonic` commands.
TOP_LEVEL = ("run", "setup", "show", "guide", "config", "pickle", "update")


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

  hedonic                        in a terminal: the home screen (status, then a menu of what to do); elsewhere this help

Overlapping communities (SNAP networks)
  hedonic run queue paper        the overlap paper's experiments, one after another (--dry-run plans, --root DIR)
  hedonic run exp [options]      a benchmark: networks x methods x seeds (interactive wizard without options)
  hedonic run paper              the whole-graph benchmark profile (= run exp --config paper)
  hedonic run paper-hoc          only its Hedonic rows, at the tuned per-network settings
  hedonic run calibrate          calibrate Hedonic's resolution and cap per pair; writes hoc_tuned.json for --tuned
  hedonic run calibrate-baselines --networks LIST   calibrate the eight baselines on each pair without ground truth
  hedonic run spectrum           ground-truth robustness spectrum of the local covers
  hedonic run spectrum-full      the same on complete graphs (= run spectrum --config full)
  hedonic run startcurve --config NAME   start-quality curve from corrupted ground-truth starts
  hedonic run controlled --config NAME   LFR-derived controlled-overlap grid
  hedonic run posthoc --config FILE      exact post-hoc stability of a ground-truth cover
  hedonic run audit --source DIR...      robustness of the covers a benchmark saved
Disjoint communities (planted-partition SBM)
  hedonic run disjoint [options] the SBM benchmark: store, figures and paper statistics (wizard without options)
  hedonic run disjoint --preset paper   the paper's 100-seed grid (50,000 graphs)
Every run (durable: survives Ctrl-C, closed terminals and, on macOS, idle sleep)
  hedonic run list               runs and their progress
  hedonic run status [NAME]      snapshot of a run (default: latest)
  hedonic run attach [NAME]      live view of a running run
  hedonic run logs [NAME]        raw output (tmux pane or log file)
  hedonic run stop NAME          stop a run (explicit; never automatic)
  hedonic run resume [NAME]      continue an interrupted run; finished work is reused
  hedonic run check [NAME]       morning check: replay a sample of the results and compare
  hedonic run queue FILE|NAME    run the experiments of a queue file one after another (overnight)
  hedonic run                    your favourite config (hedonic config set run NAME), else the wizard
Data, runtimes and settings
  hedonic setup [--all]          download SNAP pairs and build the external methods (--dry-run checks the tools)
  hedonic show methods           overlapping methods: ready / installable / missing tools
  hedonic show networks          SNAP networks: vertices, edges, on disk, --remote sizes
  hedonic show covers            graph/cover pairs found locally
  hedonic show table             local network/cover statistics as a CSV
  hedonic show metrics           what each accuracy metric measures
  hedonic show disjoint          disjoint presets, grids, methods and estimates
  hedonic pickle PAIR... | --all write the Game of network:cover pairs to pickles (--dry-run plans)
  hedonic guide [TOPIC]          a short tour of the CLI and suggested next steps
  hedonic config [set KEY VALUE] per-user settings (~/.config/hedonic/config.toml)
  hedonic update [--yes]         compare with the latest release on PyPI; upgrade only when asked

`hedonic run exp --help`, `hedonic run disjoint --help` and `hedonic setup --help` list every option.
`hedonic-exp list` lists the experiment modules' own commands."""


def hedonic_main(argv: list[str] | None = None) -> int:
    if (refused := _posix_only() or _cli_extra()) is not None:
        return refused
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        from hedonic.experiments import home

        if home.wanted():  # a terminal; a pipe, a script or HEDONIC_PLAIN_VIEW=1 gets the help text
            return home.main()
    if not argv or argv[0] in ("-h", "--help") or argv[:2] in (["run", "-h"], ["run", "--help"]):
        print(HEDONIC_HELP)
        return 0
    if argv[0] in ("-V", "--version"):
        print(_package_version())
        return 0
    if argv[0] == "setup":  # data and external runtimes
        from hedonic.experiments.overlapping import codeseg_setup

        return _call_experiment(codeseg_setup.setup_main, argv[1:])
    if argv[0] == "guide":
        from hedonic.experiments.overlapping import guide

        return guide.main(argv[1:])
    if argv[0] == "pickle":  # the Game of a network:cover pair as a pickle
        from hedonic.experiments.overlapping import snap_pickle

        return snap_pickle.main(argv[1:])
    if argv[0] == "update":
        from hedonic.experiments.overlapping import selfupdate

        return selfupdate.main(argv[1:])
    if argv[0] == "config":
        from hedonic.experiments.overlapping import userconfig

        return userconfig.main(argv[1:])
    if argv == ["run"] or (argv[0] == "run" and len(argv) > 1 and argv[1].startswith("-")):
        # bare `hedonic run`: the favourite config from ~/.config/hedonic/config.toml, else the wizard
        from hedonic.experiments.overlapping import userconfig

        favourite = userconfig.favourite()
        extra = argv[1:]
        if favourite:
            return _dispatch("quickstart", ["--config", favourite, "--confirm", *extra])
        return _dispatch("quickstart", extra)
    if argv[0] == "show":
        from hedonic.experiments.overlapping import info

        return info.main(argv[1:])
    if argv[0] != "run" or len(argv) < 2 or (argv[1] not in RUN_ALIASES and argv[1] not in RUN_VERBS):
        print(f"Unknown command: {' '.join(argv)}. Try `hedonic --help`.", file=sys.stderr)
        return 2
    if argv[1] in RUN_VERBS:
        from hedonic.experiments.overlapping import runmanager

        return runmanager.dispatch(argv[1:])
    command, rest = RUN_ALIASES[argv[1]], argv[2:]
    if argv[1] == "disjoint":  # disjoint SBM benchmark through the durable front door
        from hedonic.experiments.disjoint import front

        return front.front_main(rest)
    if argv[1] == "posthoc":  # exact post-hoc stability of a ground-truth cover, durable (`--config FILE`)
        from hedonic.experiments.overlapping import gt_posthoc

        return gt_posthoc.front_main(rest)
    if argv[1] == "controlled":  # LFR-derived controlled-overlap study from a TOML table, durable (`--config NAME`)
        from hedonic.experiments.overlapping import controlled_overlap_config

        return controlled_overlap_config.front_main(rest)
    if argv[1] == "startcurve":  # start-quality curve from a TOML table, durable (`--config NAME`)
        from hedonic.experiments.overlapping import start_quality

        return start_quality.front_main(rest)
    if argv[1] == "calibrate":  # Hedonic calibration, durable
        from hedonic.experiments.overlapping import calibrate

        return calibrate.front_main(rest)
    if argv[1] == "calibrate-baselines":  # label-free calibration of the baselines per network:cover pair, durable
        from hedonic.experiments.overlapping import baseline_run

        return baseline_run.front_main(rest)
    if argv[1] == "audit":  # robustness audit of the covers a benchmark saved, durable (`--source RUN_DIR ...`)
        from hedonic.experiments.overlapping import cover_audit

        return cover_audit.front_main(rest)
    if argv[1] in ("spectrum", "spectrum-full"):  # SNAP ground-truth robustness spectrum, durable
        from hedonic.experiments.overlapping import gt_spectrum

        if argv[1] != "spectrum":  # named config: `hedonic run spectrum --config full`
            rest = ["--config", argv[1].removeprefix("spectrum-"), *rest]
        return gt_spectrum.front_main(rest)
    if argv[1] in ("paper", "paper-hoc"):  # named configs: `hedonic run exp --config paper[-hoc]`
        rest = ["--config", argv[1], *rest]
    if rest and rest[0] in ("-h", "--help"):
        return _dispatch_help(command)
    return _dispatch(command, rest)


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
            "The hedonic experiment modules' own command lines: disjoint SBM sweeps, overlapping SNAP studies and "
            "isolated smoke checks. Durable runs, queues and setup are under the `hedonic` command."
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
  # Isolated local checks (no data)
  hedonic-exp smoke
  hedonic-exp overlapping-small
  hedonic-exp overlapping-controlled --smoke --output ~/Databases/Hedonic/Overlap/controlled/smoke.json
  hedonic-exp overlapping-tradeoffs --smoke --output-dir ~/Databases/Hedonic/Overlap/tradeoffs/smoke
  hedonic-exp participation-budget --output ~/Databases/Hedonic/Overlap/participation_budget.pdf

  # Disjoint SBM benchmark (seeded graphs, Parquet store); durable form: hedonic run disjoint
  hedonic-exp reproduce-disjoint --preset sbm-smoke --output_root ~/Databases/Hedonic/Disjoint
  hedonic-exp disjoint-store ~/Databases/Hedonic/Disjoint/sbm-smoke --verify --all
  hedonic-exp disjoint-stats --data ~/Databases/Hedonic/Disjoint/sbm-smoke

  # Overlapping SNAP studies (data: hedonic setup --pairs dblp:top5000,...)
  hedonic-exp start-quality --config start-quality-hoc-local --dry-run
  hedonic-exp overlapping-calibrate --networks dblp:top5000 --dry-run
  hedonic-exp baseline-calibrate --dry-run --methods all
  hedonic-exp baseline-calibrate --smoke --methods clique_percolation,codeseg --max-evals 6
  hedonic-exp overlapping-gt-spectrum --discover
  hedonic-exp overlapping-codeseg --datasets dblp --max-nodes 5000 --methods hedonic_local \\
      --output-dir ~/Databases/Hedonic/Overlap/codeseg/dblp-local
  hedonic-exp codeseg-doctor --methods all
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
    print("  HEDONIC_NETWORKS_DIR   SNAP networks (default <data root>/networks)")
    print("  HEDONIC_SYNTHETIC_DIR  disjoint synthetic results")
    print("  HEDONIC_OUTPUT_DIR     run outputs (default <data root>/Overlap)")
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
            "  1. overlapping-small (Petersen + tiny SBM metrics)\n"
            "  2. disjoint --smoke into a temp directory\n\n"
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

    from hedonic.experiments.overlapping.small_graphs import main as small_main

    code = _call_experiment(small_main, [])
    if code != 0:
        return code

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
