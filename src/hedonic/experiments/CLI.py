"""CLI entrypoint for hedonic experiments.

Examples:
    hedonic-exp disjoint --folder_name exp_name --max_n_nodes 120
    hedonic-exp overlapping-small
    hedonic-exp overlapping-subgraph --levels 1 --n_communities 20
    hedonic-exp overlapping-full --resolution_sweep
"""

from __future__ import annotations

import argparse
import sys

SUBCOMMANDS = (
    "disjoint",
    "overlapping-small",
    "overlapping-subgraph",
    "overlapping-full",
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_root_help()
        return 0

    command = argv[0]
    rest = argv[1:]

    if command not in SUBCOMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_root_help()
        return 2

    if rest and rest[0] in ("-h", "--help"):
        return _dispatch_help(command)

    return _dispatch(command, rest)


def _print_root_help() -> None:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp",
        description=(
            "Run hedonic library experiments "
            "(disjoint SBM and overlapping DBLP)."
        ),
    )
    parser.add_argument(
        "command",
        choices=SUBCOMMANDS,
        help="Experiment to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected experiment",
    )
    parser.print_help()
    print(
        "\nSubcommands:\n"
        "  disjoint              SBM sweep for disjoint detection\n"
        "  overlapping-small     Quick tests on small graphs\n"
        "  overlapping-subgraph  DBLP L-hop subgraph experiment\n"
        "  overlapping-full      Full DBLP overlapping experiment\n"
        "\nExamples:\n"
        "  hedonic-exp disjoint --folder_name exp --max_n_nodes 60\n"
        "  hedonic-exp overlapping-small\n"
        "  hedonic-exp overlapping-subgraph --levels 1 --n_communities 5\n"
        "  hedonic-exp overlapping-full --resolution_sweep\n"
    )


def _dispatch_help(command: str) -> int:
    try:
        if command == "disjoint":
            from hedonic.experiments.disjoint.sbm_sweep import main as m

            m(["--help"])
        elif command == "overlapping-small":
            print(
                "usage: hedonic-exp overlapping-small\n\n"
                "Quick small-graph tests via community_hedonic(max_memberships>1)."
            )
        elif command == "overlapping-subgraph":
            from hedonic.experiments.overlapping.dblp_subgraph import main as m

            m(["--help"])
        elif command == "overlapping-full":
            from hedonic.experiments.overlapping.dblp_full import main as m

            m(["--help"])
        else:
            print(f"Unknown command: {command}", file=sys.stderr)
            return 2
    except SystemExit as exc:
        # argparse --help calls sys.exit(0)
        return int(exc.code) if exc.code is not None else 0
    return 0


def _dispatch(command: str, rest: list[str]) -> int:
    if command == "disjoint":
        from hedonic.experiments.disjoint.sbm_sweep import main as m

        m(rest)
    elif command == "overlapping-small":
        from hedonic.experiments.overlapping.small_graphs import main as m

        m(rest)
    elif command == "overlapping-subgraph":
        from hedonic.experiments.overlapping.dblp_subgraph import main as m

        m(rest)
    elif command == "overlapping-full":
        from hedonic.experiments.overlapping.dblp_full import main as m

        m(rest)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
