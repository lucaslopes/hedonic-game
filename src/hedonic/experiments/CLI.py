"""CLI entrypoint for hedonic experiments.

Install with experiments extra, then::

    hedonic-exp --help
    hedonic-exp list
    hedonic-exp smoke
    hedonic-exp disjoint --smoke --output_root /tmp/hedonic-smoke
    hedonic-exp overlapping-small
    hedonic-exp overlapping-subgraph --levels 1 --n_communities 5
    hedonic-exp overlapping-full --n_iterations -1
    hedonic-exp overlapping-scale --smoke --output_dir /tmp/scale
    hedonic-exp overlapping-resolution --smoke --output_dir /tmp/res-f1
    hedonic-exp disjoint-load --results_folder /path/to/jsons
    hedonic-exp plots --smoke --output_dir /tmp/figs
    hedonic-exp reproduce-disjoint --preset v1020-smoke --output_root /tmp/V1020_CLI

**Agents:** when you add, remove, rename, or change the CLI of any experiment
module under ``hedonic.experiments``, you **must** update this file (registry +
help text) and the CLI section in ``AGENTS.md`` in the same change.
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
    needs_data: str | None = None  # "dblp" | "synthetic" | None
    has_argparse_help: bool = True


# ---------------------------------------------------------------------------
# Registry — single source of truth for subcommands.
# Add new experiments here AND document them in AGENTS.md.
# ---------------------------------------------------------------------------

COMMANDS: dict[str, Command] = {
    "smoke": Command(
        name="smoke",
        module="",  # handled specially
        attr="",
        summary="Isolated local check (no DBLP): small graphs + tiny SBM sweep",
        needs_data=None,
        has_argparse_help=False,
    ),
    "disjoint": Command(
        name="disjoint",
        module="hedonic.experiments.disjoint.sbm_sweep",
        attr="main",
        summary=(
            "SBM parameter sweep for disjoint community detection "
            "(presets: v1020, v1020-smoke)"
        ),
        needs_data="synthetic",
    ),
    "disjoint-load": Command(
        name="disjoint-load",
        module="hedonic.experiments.disjoint.data_loader",
        attr="main",
        summary="Combine disjoint JSON results into a gzipped CSV",
        needs_data="synthetic",
    ),
    "overlapping-small": Command(
        name="overlapping-small",
        module="hedonic.experiments.overlapping.small_graphs",
        attr="main",
        summary="Quick overlapping smoke tests on small graphs (no data dirs)",
        needs_data=None,
        has_argparse_help=False,
    ),
    "overlapping-subgraph": Command(
        name="overlapping-subgraph",
        module="hedonic.experiments.overlapping.dblp_subgraph",
        attr="main",
        summary="DBLP L-hop subgraph overlapping experiment",
        needs_data="dblp",
    ),
    "overlapping-full": Command(
        name="overlapping-full",
        module="hedonic.experiments.overlapping.dblp_full",
        attr="main",
        summary="Full DBLP overlapping experiment (+ optional resolution sweep)",
        needs_data="dblp",
    ),
    "overlapping-scale": Command(
        name="overlapping-scale",
        module="hedonic.experiments.overlapping.complexity_scale",
        attr="main",
        summary=(
            "Wallclock scaling of overlapping hedonic as subnetworks grow "
            "(only_local_moving T/F vs size; timeout stop + plot)"
        ),
        needs_data="dblp",
    ),
    "overlapping-resolution": Command(
        name="overlapping-resolution",
        module="hedonic.experiments.overlapping.resolution_f1",
        attr="main",
        summary=(
            "Full-DBLP F1 vs resolution (0→1) for multi-phase overlapping "
            "hedonic; multi-seed CI band + line plot"
        ),
        needs_data="dblp",
    ),
    "plots": Command(
        name="plots",
        module="hedonic.experiments.plots.paper_figures",
        attr="main",
        summary=(
            "PHYSA V1020 paper figures from disjoint CSV "
            "(gt_robustness, noise, n_communities, acc_*)"
        ),
        needs_data="synthetic",
    ),
    "reproduce-disjoint": Command(
        name="reproduce-disjoint",
        module="hedonic.experiments.disjoint.reproduce",
        attr="main",
        summary=(
            "End-to-end disjoint reproduction: sweep → CSV → paper figures "
            "(presets: v1020, v1020-smoke)"
        ),
        needs_data="synthetic",
    ),
}

SUBCOMMANDS: tuple[str, ...] = tuple(COMMANDS.keys())


def main(argv: list[str] | None = None) -> int:
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
            "Reproduce hedonic experiments: disjoint SBM sweeps, "
            "overlapping DBLP runs, and isolated smoke checks."
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
  # Isolated local check (no databases required)
  hedonic-exp smoke
  hedonic-exp overlapping-small
  hedonic-exp disjoint --smoke --output_root /tmp/hedonic-smoke

  # Tiny V1020-compatible structural smoke (safe output root)
  hedonic-exp disjoint --preset v1020-smoke \\
      --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

  # Full PHYSA V1020 grid (very large — do not write into archived V1020)
  hedonic-exp disjoint --preset v1020 \\
      --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI

  # Ad-hoc disjoint SBM sweep
  hedonic-exp disjoint --folder_name exp --max_n_nodes 60 \\
      --n_communities 2 --seeds 42 --p_in 0.1 --difficulty 0.5

  # Load disjoint JSON results → CSV
  hedonic-exp disjoint-load --results_folder /path/to/resultados --simple
  hedonic-exp disjoint-load \\
      --results_folder .../V1020_CLI/resultados \\
      --output .../V1020_CLI/resultados.csv.gzip --simple

  # Paper figures (same stems as archived V1020/figures/)
  hedonic-exp plots --smoke --output_dir /tmp/hedonic-figs
  hedonic-exp plots --data .../V1020/resultados_ari.csv.gzip \\
      --output_dir .../V1020_CLI/figures --format pdf

  # Complete disjoint pipeline (sweep → CSV → figures)
  hedonic-exp reproduce-disjoint --preset v1020-smoke \\
      --output_root .../V1020_CLI
  hedonic-exp reproduce-disjoint --plots-only \\
      --data .../V1020/resultados_ari.csv.gzip \\
      --output_root .../V1020_CLI --max_rows 50000

  # Overlapping on DBLP (set HEDONIC_DBLP_DIR if needed)
  # Defaults: n_iterations=-1 (to equilibrium), max_memberships=n_GT
  hedonic-exp overlapping-subgraph --levels 1 --n_communities 5 \\
      --methods leiden,hedonic_v1 --output /tmp/subgraph_smoke.json
  hedonic-exp overlapping-full --resolution 1e-4 --output /tmp/dblp_full.json

  # Complexity scale: size vs wallclock (local-moving vs full multi-phase)
  # Stops each line when --timeout is hit; does not require full DBLP finish
  hedonic-exp overlapping-scale --smoke --output_dir /tmp/hedonic-scale
  hedonic-exp overlapping-scale --timeout 30 --max-levels 6 \\
      --output_dir /tmp/hedonic-scale-dblp
  # Full multi-phase only, 10 min budget per size point
  hedonic-exp overlapping-scale --variant full --timeout 600 --max-levels 6 \\
      --community_idx 1004 --output_dir /tmp/hedonic-scale-dblp-full

  # Full-DBLP F1 vs resolution (multi-phase, multi-seed CI); --smoke skips DBLP
  hedonic-exp overlapping-resolution --smoke --output_dir /tmp/hedonic-res-f1
  hedonic-exp overlapping-resolution --resolutions 0:1:11 --seeds 0-4 \\
      --output_dir /tmp/hedonic-res-f1-dblp
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
    print("Data paths (env overrides):")
    print("  HEDONIC_DBLP_DIR       DBLP network directory")
    print("  HEDONIC_SYNTHETIC_DIR  Synthetic/SBM results root")


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
            "Isolated local check that needs no DBLP or large data dirs:\n"
            "  1. overlapping-small (Petersen + tiny SBM metrics)\n"
            "  2. disjoint --smoke into a temp directory\n\n"
            "Options:\n"
            "  --keep-output  Print and keep the temp output root "
            "(default: delete after run)\n"
        )
        return 0
    if command == "overlapping-small" or not cmd.has_argparse_help:
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
    """Isolated reproduction check without external databases."""
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
    print("  hedonic-exp smoke  (isolated, no DBLP required)")
    print("=" * 55)

    from hedonic.experiments.overlapping.small_graphs import main as small_main

    code = _call_experiment(small_main, [])
    if code != 0:
        return code

    from hedonic.experiments.disjoint.sbm_sweep import main as sweep_main

    if keep:
        tmp = tempfile.mkdtemp(prefix="hedonic-smoke-")
        print(f"\n[smoke] disjoint SBM → {tmp} (kept)")
        code = _call_experiment(
            sweep_main,
            ["--smoke", "--folder_name", "smoke", "--output_root", tmp],
        )
    else:
        with tempfile.TemporaryDirectory(prefix="hedonic-smoke-") as tmp:
            print(f"\n[smoke] disjoint SBM → {tmp}")
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
