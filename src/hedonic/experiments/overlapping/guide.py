"""``hedonic guide``: a short, current tour of what the CLI can do and what to do next.

    hedonic guide             where you are (data, runs) and the topics
    hedonic guide TOPIC       one topic: start, disjoint, runs, config, update
    hedonic guide --list      topic names only

In a terminal, the index is an arrow-key menu. Everything is read from the local
machine; nothing is downloaded or started.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap

from hedonic.experiments.overlapping.tui import style

TOPICS: dict[str, tuple[str, str]] = {
    "start": ("First steps", """
        hedonic show disjoint                    the disjoint presets, grids, methods and estimates
        hedonic run disjoint --preset smoke --dry-run   the plan of the smoke run
        hedonic run disjoint --preset smoke      the whole pipeline in seconds (store, table, figures, statistics)
        hedonic run queue disjoint --dry-run     the disjoint paper's queue and what is already done
        hedonic detect                           one network x one method x one seed, here (`hedonic guide detect`)

        This release (1.x) reproduces the disjoint paper (the Physica A SBM benchmark). Everything runs
        detached, so closing the terminal or pressing Ctrl-C only closes the view. The overlapping-communities
        experiments come with release 2.
    """),
    "detect": ("One network, one method (`hedonic detect`)", """
        hedonic detect                                    a wizard: network, its parameters, method, its parameters, seed
        hedonic detect --list                             every network source and method, with its parameters
        hedonic detect --graph sbm:k=4,p_in=0.3 --method hedonic
        hedonic detect --graph lfr:n=1000,mu=0.3 --method leiden:resolution=0.05 --seed 3
        hedonic detect --graph famous:name=Zachary --method louvain
        hedonic detect --graph file:path=g.txt,truth=labels.txt --method infomap

        A network is SOURCE:name=value,... and a method METHOD:name=value,...; --seed seeds both. The answer is
        scored against the network's ground truth (ARI, NMI, overlap) and saved as result.json under the data
        folder's Detect/ (--output DIR, or --no-save). The run prints its own command line, so a wizard answer can
        be repeated or scripted. It runs here, in this terminal: for many seeds or networks, use
        `hedonic run disjoint` or a queue.
    """),
    "disjoint": ("Disjoint SBM benchmark (`hedonic run disjoint`, `hedonic run queue disjoint`)", """
        hedonic run queue disjoint --dry-run         the disjoint paper's queue: a smoke check, then the paper grid
        hedonic run queue disjoint                   start it (durable, resumable; nothing to download)
        hedonic show disjoint                        presets, grids, methods and estimates
        hedonic run disjoint --preset smoke          the whole pipeline in seconds
        hedonic run disjoint --preset paper --dry-run   the paper's 50,000-graph plan, time and disk
        hedonic run disjoint --preset paper          start it: durable, resumable, about a day on 8 workers
        hedonic run disjoint                         a wizard (presets or a custom grid)

        Every grid value is a flag (--nodes, --communities, --p-in, --difficulty, --seeds, --noises,
        --partition-seeds, --methods, --restarts, --workers, --partitions). Graphs are regenerated from their
        seeds and never stored; every detector run has its own seed. The run writes a Parquet store, the
        logical-cell table, the five figures and the paper statistics (stats/summary.md). Where it goes:
        `hedonic config set disjoint_output_root PATH`. Query it: `hedonic-exp disjoint-store STORE --sql ...`.

        A store keeps each method's rows while that method's identity (call, parameters, restarts and the
        binaries it runs in) is unchanged; a run computes only the missing methods. `--dry-run` reads the store
        and says, per method, how many graphs are current, from another build, or missing. Rows of another
        build are refused unless recomputed (`--rerun hedonic,leiden`) or carried after a replay matches
        (`--reuse-from STORE --reuse-methods mirror,spectral`, never across timing rules).
    """),
    "runs": ("Managing runs", """
        hedonic run list                all runs, status and progress
        hedonic run attach [NAME]       live view again (Ctrl-C closes only the view)
        hedonic run status [NAME]       one snapshot          (--json for scripts)
        hedonic run logs [NAME]         raw tmux pane or log file
        hedonic run stop NAME           explicit; nothing stops a run implicitly
        hedonic run resume [NAME]       continue an interrupted, stopped or failed run
        hedonic run check [NAME]        morning check: replay sampled rows of a disjoint run from their seeds

        Statuses: running, completed, completed_with_failures, stopped, interrupted (the worker vanished, for
        example after a reboot) and failed (an error, with its reason). NAME may be a unique fragment.

        A queue opens as a tree you can open and close: the queue, then its runs with their progress and ETA.
        The mouse wheel scrolls, a click selects or opens a row, the arrows move, and the pane under the tree
        explains the selected row. Press ? for the keys, q or Ctrl-C to close the view (the queue keeps
        running). HEDONIC_PLAIN_VIEW=1 prints the text view instead; `hedonic run queue FILE --report` prints
        it once without starting anything. `schedule = "shortest"` in a queue file (or --schedule shortest) still
        runs one item at a time, but takes at each turn the ready item with the smallest estimate.
    """),
    "config": ("Your settings (`hedonic config`)", """
        hedonic config                                show ~/.config/hedonic/config.toml
        hedonic config set disjoint_output_root ~/Databases/Hedonic/Disjoint   where disjoint runs and queues go
        hedonic config set threads 8                  a machine setting kept for your own scripts
        hedonic config unset KEY                      back to the default
        hedonic config example                        a commented template of every setting

        Machine settings apply to every run unless a flag overrides them. Keep machine paths out of
        repositories. HEDONIC_CONFIG points at another file.
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
    from hedonic.experiments.overlapping import runmanager, userconfig

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
    if not entries:
        steps.append("no runs yet: hedonic run disjoint --preset smoke   (seconds; the whole pipeline)")
    if not userconfig.path().is_file():
        steps.append("set where results go: hedonic config set disjoint_output_root ~/Databases/Hedonic/Disjoint")
    return steps


def render_index() -> str:
    lines = [style("\n  hedonic guide", "bold", "magenta")]
    steps = next_steps()
    if steps:
        lines += ["", style("  suggested next steps", "bold")] + [f"    · {s}" for s in steps]
    lines += ["", style("  topics", "bold")] + [f"    hedonic guide {name:<10} {title}" for name, (title, _) in TOPICS.items()]
    lines.append(style("\n  every command: hedonic --help · analysis and utility commands: hedonic-exp list", "dim"))
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
