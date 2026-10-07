"""``hedonic guide``: a short, current tour of what the CLI can do and what to do next.

    hedonic guide             where you are (data, runs) and the topics
    hedonic guide TOPIC       one topic: start, benchmark, paper, spectrum, runs, config, data, update
    hedonic guide --list      topic names only

In a terminal, the index is an arrow-key menu. Everything is read from the local
machine; nothing is downloaded or started.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import textwrap

from hedonic.experiments.overlapping.tui import style

TOPICS: dict[str, tuple[str, str]] = {
    "start": ("First steps", """
        hedonic show methods            which of the ten methods can run on this machine
        hedonic show covers             which SNAP graph/cover pairs are on disk
        hedonic run exp --dry-run       the plan and time estimate of the default run
        hedonic run exp                 a wizard (arrow keys, space, enter); the first choice is the default run

        The default run takes about a minute: DBLP, a 5,000-vertex subgraph, seed 0, HOC-local and CoDeSEG.
        Everything runs detached, so closing the terminal or pressing Ctrl-C only closes the view.
    """),
    "benchmark": ("Benchmark runs (`hedonic run exp`)", """
        hedonic run exp --networks dblp,amazon --methods all --seeds 0-2 --nodes 20000
        hedonic run exp --networks youtube --full --methods hoc_local,fox,cpm
        hedonic run exp --config my.toml         a saved configuration (see `hedonic guide config`)

        Every accuracy metric is defined by `hedonic show metrics`. Data and external tools are fetched on
        first use into ~/.cache/hedonic; a method that cannot be prepared is reported as unavailable while
        the others still run. `--dry-run` prints the plan first; `--save-config` writes it to a file.
    """),
    "paper": ("Reproduce the paper (`hedonic run paper`)", """
        hedonic run paper --dry-run     the 200-run plan (DBLP and Amazon, 10 methods, seeds 0-9) and its estimate
        hedonic run paper               start it: durable and resumable, many hours

        The order is seed -> method -> network. Finished runs are cached, so `hedonic run resume NAME` continues
        after a reboot. Changing the paper configuration changes the experiment: record why in the manuscript.
    """),
    "spectrum": ("Ground-truth robustness spectrum (`hedonic run spectrum`)", """
        hedonic show covers                       present / missing / unsupported graph/cover pairs
        hedonic run spectrum --dry-run --inspect  cohort, plan, sizes and estimates; runs nothing
        hedonic run spectrum                      wizard, then a durable run
        hedonic-exp overlapping-gt-spectrum --replot   rebuild figures and tables from a finished ledger

        For every locally available overlapping cover it plots the fraction of vertices with no profitable
        unilateral action against the resolution (fixed and open labels), and runs local moving from the exact
        ground-truth cover, auditing and scoring every return. Nothing is downloaded without --provision.
        Full guide: docs/overlapping_gt_spectrum.md
    """),
    "runs": ("Managing runs", """
        hedonic run list                all runs, status and progress
        hedonic run attach [NAME]       live view again (Ctrl-C closes only the view)
        hedonic run status [NAME]       one snapshot          (--json for scripts)
        hedonic run logs [NAME]         raw tmux pane or log file
        hedonic run stop NAME           explicit; nothing stops a run implicitly
        hedonic run resume [NAME]       continue an interrupted, stopped or failed run

        Statuses: running, completed, completed_with_failures, stopped, interrupted (the worker vanished, for
        example after a reboot) and failed (an error, with its reason). NAME may be a unique fragment.
    """),
    "config": ("Your settings (`hedonic config`)", """
        hedonic config                                show ~/.config/hedonic/config.toml
        hedonic config set output_dir ~/hedonic-runs  where every run is saved
        hedonic config set run paper                  what a bare `hedonic run` starts (it asks first)
        hedonic config set profiles.quick.nodes 20000 your own named configuration

        Machine settings (output_dir, cache_dir, network_root, threads) apply to every run unless a flag
        overrides them. Keep machine paths out of repositories; the shareable template is
        configs/hedonic-run.example.toml.
    """),
    "data": ("Data and caches", """
        SNAP archives are looked up in the network root (HEDONIC_NETWORKS_DIR or `hedonic config set network_root`)
        and otherwise downloaded on first use into ~/.cache/hedonic. `hedonic show networks --remote` shows the
        download sizes; the spectrum study never downloads unless you add --provision.
        External method runtimes are prepared by `hedonic-exp codeseg-setup --all --build-native` and audited by
        `hedonic-exp codeseg-doctor --require-all`.
    """),
    "update": ("Keeping hedonic current (`hedonic update`)", """
        hedonic update            compare the installed version with PyPI (one HTTPS GET; installs nothing)
        hedonic update --yes      upgrade the package with pip
        A development checkout is never modified: update it with git and `uv sync`. Runs already in flight keep
        their old code until you `hedonic run resume` them.
    """),
}


def _body(topic: str) -> str:
    title, text = TOPICS[topic]
    return style(f"\n  {title}", "bold", "magenta") + "\n" + textwrap.indent(textwrap.dedent(text).strip("\n"), "  ") + "\n"


def next_steps() -> list[str]:
    """Suggestions from local state only (no network, no downloads)."""
    from hedonic.experiments.overlapping import gt_spectrum, runmanager, userconfig
    from hedonic.experiments.config import NETWORKS_DIR

    steps = []
    try:
        entries = runmanager.all_entries()
    except OSError:
        entries = []
    unfinished = [e["name"] for e in entries
                  if runmanager.effective_status(e, runmanager.read_progress(e)) in ("interrupted", "failed")]
    running = [e["name"] for e in entries
               if runmanager.effective_status(e, runmanager.read_progress(e)) in ("running", "setup", "starting")]
    if running:
        steps.append(f"a run is in progress: hedonic run attach {running[-1]}")
    if unfinished:
        steps.append(f"an unfinished run can continue: hedonic run resume {unfinished[-1]}")
    root = userconfig.defaults().get("network_root") or str(NETWORKS_DIR)
    present = [r for r in gt_spectrum.discover(root) if r["status"] == "present"]
    if present:
        steps.append(f"{len(present)} graph/cover pairs are on disk: hedonic run spectrum --dry-run --inspect")
    if not entries:
        steps.append("no runs yet: hedonic run exp   (a minute for the default experiment)")
    if not userconfig.path().is_file():
        steps.append("set where results go: hedonic config set output_dir ~/hedonic-runs")
    return steps


def render_index() -> str:
    lines = [style("\n  hedonic guide", "bold", "magenta")]
    steps = next_steps()
    if steps:
        lines += ["", style("  suggested next steps", "bold")] + [f"    · {s}" for s in steps]
    lines += ["", style("  topics", "bold")] + [f"    hedonic guide {name:<10} {title}" for name, (title, _) in TOPICS.items()]
    lines.append(style("\n  every command: hedonic --help · every experiment: hedonic-exp list", "dim"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hedonic guide", description="A short tour of the hedonic CLI.")
    parser.add_argument("topic", nargs="?", help="one of: " + ", ".join(TOPICS))
    parser.add_argument("--list", action="store_true", help="print the topic names and exit")
    parser.add_argument("--json", action="store_true", help="machine-readable topics")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps({name: {"title": t, "text": textwrap.dedent(x).strip()} for name, (t, x) in TOPICS.items()}, indent=2))
        return 0
    if args.list:
        print("\n".join(TOPICS))
        return 0
    if args.topic:
        if args.topic not in TOPICS:
            print(f"hedonic guide: unknown topic {args.topic!r}; choose from: {', '.join(TOPICS)}", file=sys.stderr)
            return 2
        print(_body(args.topic))
        return 0
    print(render_index())
    from hedonic.experiments.overlapping import tui

    if tui.interactive():
        try:
            names = list(TOPICS)
            picked = tui.select("\nOpen a topic", [(TOPICS[n][0], "") for n in names] + [("Quit", "")])
        except tui.Cancelled:
            return 0
        if picked < len(names):
            print(_body(names[picked]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
