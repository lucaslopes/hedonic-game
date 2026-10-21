"""``hedonic guide``: a short, current tour of what the CLI can do and what to do next.

    hedonic guide             where you are (data, runs) and the topics
    hedonic guide TOPIC       one topic: start, benchmark, overlap, spectrum, disjoint, runs, config, data, update
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
        hedonic                         in a terminal: the home screen (where things stand, then a menu)
        hedonic show methods            which of the nine methods (Hedonic and eight baselines) can run here
        hedonic show covers             which SNAP graph/cover pairs are on disk
        hedonic setup --dry-run         what `hedonic setup` would download and build, and the system tools it needs
        hedonic run exp --dry-run       the plan and time estimate of the default run
        hedonic run exp                 a wizard (arrow keys, space, enter); the first choice is the quick default run, then a yes/no
        hedonic detect                  one network x one method x one seed, in this terminal (see `hedonic guide detect`)

        The default run takes about a minute: DBLP, a 5,000-vertex subgraph, seed 0, Hedonic and CoDeSEG.
        Everything runs detached, so closing the terminal or pressing Ctrl-C only closes the view.
        To reproduce the overlapping-communities paper, see `hedonic guide overlap`.
    """),
    "detect": ("One network, one method (`hedonic detect`)", """
        hedonic detect                                    a wizard: network, its parameters, method, its parameters, seed
        hedonic detect --list                             every network source and method, with its parameters
        hedonic detect --graph sbm:k=4,p_in=0.3 --method hedonic
        hedonic detect --graph lfr:n=1000,mu=0.3 --method leiden:resolution=0.05 --seed 3
        hedonic detect --graph famous:name=Zachary --method louvain
        hedonic detect --graph file:path=g.txt,truth=labels.txt --method infomap
        hedonic detect --graph lfr_overlap:n=500 --method hedonic:max_memberships=2
        hedonic detect --graph snap:pair=amazon:top5000 --method codeseg   (a pair on disk; download=yes fetches it)

        A network is SOURCE:name=value,... and a method METHOD:name=value,...; --seed seeds both. The answer is
        scored against the network's ground truth (ARI, NMI, overlap for partitions; F1, ONMI, Omega for covers) and
        saved as result.json under the data folder's Detect/ (--output DIR, or --no-save). The run prints its own
        command line, so a wizard answer can be repeated or scripted. It runs here, in this terminal: for many
        seeds or networks, use `hedonic run exp`, `hedonic run disjoint` or a queue.
    """),
    "benchmark": ("Benchmark runs (`hedonic run exp`)", """
        hedonic run exp --networks dblp,amazon --methods all --seeds 0-2 --nodes 20000
        hedonic run exp --networks youtube --full --methods hedonic,fox,clique_percolation
        hedonic run exp --config my.toml         a saved configuration (see `hedonic guide config`)

        hedonic run exp --networks dblp,amazon --covers all,top5000 --induce covered --seeds 0-9
                                                 every cover of each network, each method on the subgraph the cover
                                                 induces (a network can name one: --networks dblp:top5000)

        Every accuracy metric is defined by `hedonic show metrics`. Data and external tools are fetched on
        first use; a method that cannot be prepared is reported as unavailable while the others still run.
        `--dry-run` prints the plan first; `--save-config` writes it to a file.

        How the loops nest is a setting: `--order seed,method,network` (outermost first) or, in a config file,
        `order = ["seed", "method", "network"]`. It never invalidates a finished run.

        A deterministic method (CoDeSEG at one thread, CliquePerc, FOX, BigCLAM, NcGame, NEO-K-Means) runs once
        per network, at the lowest seed, since more seeds repeat its cover; Hedonic, ANGEL and Ego-Splitting run
        at every seed. `--replication all` runs every method at every seed; both settings share their records.

        A started or resumed run first scans the plan against the files on disk: a unit is done only when its record
        is present, final, of the same settings, and its record and cover match the SHA-256 seal in RUN/integrity.
        A record whose bytes changed is moved aside (renamed, never deleted) and rerun. To see the scan only:
        hedonic run exp --config FILE --name NAME --scan
    """),
    "overlap": ("Reproduce the overlapping-communities paper (`hedonic run queue overlap`)", """
        hedonic setup --all --dry-run           the eight graph/cover pairs and eight baselines it needs
        hedonic setup --all                     download the pairs, build and install the baselines
        hedonic run queue overlap --dry-run       its runs, the order of their steps, with what is already done
        hedonic run queue overlap                 start it (durable, resumable; days on a laptop)
        hedonic run queue overlap --root DIR      the same, writing under DIR instead of OUTPUT_DIR/overlap-queue
        hedonic run attach overlap-queue          the live tree view (q closes the view only)

        The queue goes method by method, in two phases. Phase 1, on DBLP, Amazon and YouTube: Hedonic first (the
        synthetic networks, its calibration of the resolution only with no membership cap, then the robustness
        spectrum, the ground-truth start curve and its benchmark rows one pair at a time), then each baseline in
        turn, cheapest first: its calibration, then its benchmark. Phase 2 repeats every method, Hedonic first, on
        LiveJournal/top5000 and Friendster/top5000, the largest graphs, so they cannot hold up the rest. All
        benchmark items write into one folder so a finished unit is reused. The benchmark items only save each
        answer (cover, labels, vertex ids, graph hash); each phase ends with an audit item that audits every saved
        cover, so all detector runs of a phase finish before any audit.
        Every run decides from its files what is done, so starting the queue again continues it. The queue checks
        that the data and methods are prepared and prints the `hedonic setup` command when they are not.

        The runs can also be started one by one:
        hedonic run calibrate --networks dblp:top5000 --mode auto   re-tune Hedonic's (gamma, cap); writes hedonic_tuned.json
        hedonic run calibrate --networks dblp:top5000 --cap unlimited   the resolution only, no cap (the paper's)
        hedonic run calibrate-baselines --networks PAIRS --methods all   label-free baseline calibration
            (--method-budget angel=5:240, --transfer PAIRS, --adopt-from DIR: the queue's budget, unsearched pairs
            that take the nearest calibrated pair's setting, and searches copied from an earlier calibration)
        hedonic run exp --config paper-benchmark --tuned FILE[,FILE]     the benchmark at the calibrated settings
        hedonic run audit --source RUN          exact robustness of the covers a run saved without --robustness
        hedonic-exp overlapping-rescore --source RUN --networks PAIRS   score the saved covers again (--check: reproduce)
    """),
    "spectrum": ("Ground-truth robustness spectrum (`hedonic run spectrum`)", """
        hedonic show covers                       present / missing / unsupported graph/cover pairs
        hedonic run spectrum --dry-run            cohort and plan; runs nothing
        hedonic run spectrum --dry-run --inspect  also sizes and estimates: loads every pair (many GB for the large ones)
        hedonic run spectrum                      wizard, then a durable run
        hedonic run spectrum-full                 the eight paper pairs on their covered graphs (= --config full)
        hedonic run spectrum --replot             rebuild figures and tables from a finished ledger

        For every locally available overlapping cover it plots the fraction of vertices with no profitable
        unilateral action against the resolution (fixed and open labels), and runs local moving from the exact
        ground-truth cover, auditing and scoring every return. Nothing is downloaded without --provision.
        --config NAME reads a shipped config (overlapping-gt-spectrum-NAME) or a TOML file of your own.
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
        hedonic run check [NAME]        morning check: replay sampled results from their seeds and compare
        hedonic run check --replay N [NAME | --root DIR]
                                        after a change of core: replay N Hedonic units and N cover audits with the
                                        loaded core, bit for bit (--dry-run lists them; a pass goes to adopted.json)

        Statuses: running, completed, completed_with_failures, stopped, interrupted (the worker vanished, for
        example after a reboot) and failed (an error, with its reason). NAME may be a unique fragment.

        A queue opens as a tree you can open and close: its runs, then each run's pairs, then the loops behind
        every count (action policy, resolution, seed...). The mouse wheel scrolls, a click selects or opens a row,
        the arrows move, and the pane under the tree explains the selected row: what its number counts, the
        loops that multiply into it, and where its ETA comes from. Press ? for the keys, q or Ctrl-C to close
        the view (the queue keeps running).

        A queue file that lists `pairs` and marks runs `by_pair = true` goes pair by pair: the whole block of those
        runs on the first pair, then on the next, in the order of `pairs`. A run between its pairs reads as part done
        (the half-filled mark), not as stopped. `hedonic run queue FILE --dry-run` prints the steps in order.
        `schedule = "shortest"` in the file (or --schedule shortest) still runs one step at a time, but takes at each
        turn the ready step with the smallest estimate; a run that reads another's folder (--tuned) or names it in
        `after = [...]` waits for it.

        An item can be a whole queue: `run = ["queue", "overlap"]` (a shipped name or a file, optionally with
        "--root", DIR). It starts that queue, waits until it completes and goes on, so `hedonic run queue all` runs
        the overlap paper's queue and then the disjoint paper's. The tree shows the inner queue's runs under it.

        Time limits, in the queue file (or `hedonic run queue FILE --timeout 2h --max-eta 6h`):
            [limits]                       every item of this queue and of the queues it runs
            timeout = "2h"                 a single run still going after 2 hours is stopped (recorded as a timeout)
            max_eta = "6h"                 a pair whose single run is estimated over 6 hours is skipped
            limits = { timeout = "30m" }   under an [[items]] entry: that item only ("none" removes a limit)
        Both are about one run (a detector run, a calibration probe, a start-quality condition), never a total: a
        calibration whose thirty probes each finish inside the limit is never stopped because together they take
        longer. A limit never changes what a record is: a run it stopped runs again under a larger limit or none.
        The spectrum keeps its own per-run timeout (part of each condition's identity); --dry-run shows, per run,
        what one run is and where a limit does not apply.

        Memory: while a detector runs (a benchmark run, a calibration probe, a start-quality condition), the
        machine's free memory is read twice a second; below the floor (HEDONIC_MEMORY_FLOOR, default 10% free)
        that run's processes are killed and it is recorded as out of memory (status memory, with the floor and
        the free memory seen). Like a timeout it stands: the queue does not run it again unless the floor is lower
        or you add --rerun-memory to `hedonic run ...`. In a queue file, `--alone PAIRS` runs those pairs of a
        calibration one at a time, never beside another, and `--out-of-memory METHOD@PAIR,...` records a run as
        out of memory without starting it (the overlap queue does so for Clique Percolation on LiveJournal and
        Friendster). Tables and `hedonic run check` say "out of memory".
    """),
    "config": ("Your settings (`hedonic config`)", """
        hedonic config                                show ~/.config/hedonic/config.toml
        hedonic config set output_dir ~/Databases/Hedonic/Overlap  where every run is saved
        hedonic config set network_root ~/Databases/Hedonic/Networks   where SNAP files are looked up and downloaded
        hedonic config set run paper                  what a bare `hedonic run` starts (it asks first)
        hedonic config set profiles.quick.nodes 20000 your own named configuration

        hedonic config example                        a commented template of every setting

        Machine settings (output_dir, cache_dir, network_root, threads) apply to every run unless a flag
        overrides them. Keep machine paths out of repositories.
    """),
    "data": ("Data and caches", """
        Data lives under ~/Databases/Hedonic (HEDONIC_DATA_DIR) and runs under ~/Databases/Hedonic/Overlap (HEDONIC_OUTPUT_DIR), unless
        the user config says otherwise. SNAP archives are looked up in the network root (HEDONIC_NETWORKS_DIR or
        `hedonic config set network_root`) and downloaded there on first use; `hedonic show networks --remote` shows
        the download sizes. `hedonic show table` reads only pairs already on disk and writes a CSV analysis, a LaTeX
        table and three figure panels next to it; `hedonic show figure` redraws the panels. Neither downloads. The
        spectrum study never downloads unless you add --provision.
        External method runtimes are prepared by `hedonic setup` (`--dry-run` first; `--methods M,...`,
        `--pairs NETWORK:COVER,...` or `--all`) into ~/.cache/hedonic/codeseg and audited by
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
        steps.append(f"{len(present)} graph/cover pairs are on disk: hedonic run spectrum --dry-run")
    if not entries:
        steps.append("no runs yet: hedonic run exp   (a minute for the default experiment)")
    if not userconfig.path().is_file():
        steps.append("set where results go: hedonic config set output_dir ~/Databases/Hedonic/Overlap")
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
