"""Calibration of Hedonic's settings as a durable run (``hedonic run calibrate``).

The Hedonic method has two settings that depend on the network: the resolution gamma and the membership cap M.
``quickstart.HEDONIC_TUNED`` holds the optimum the theory-guided search found on the whole DBLP and Amazon graphs (on the
1.x build). This run repeats that search (``calibration_search``) on the build that is loaded now and writes
``hedonic_tuned.json``, which the other runs read through ``--tuned`` (``hedonic run exp --tuned``, ``hedonic run startcurve
--tuned``), so every Hedonic row uses the calibration of the build it runs on.

A network may be given as a *target* ``network:cover`` (``--networks dblp,amazon --covers all,top5k`` crosses them): the
search then runs on the graph a ``--induce covered`` benchmark run hands to every method (the subgraph induced by that
cover's vertices), scored against that cover, so each (network, ground-truth cover) pair gets its own gamma and cap. Their
settings are written under the target's name and ``--tuned`` lays them over the run's ``method_params``. A pair without a
recorded optimum starts from its network's whole-graph value (DBLP's for a network that has none): use ``--mode full``
for the covered graphs, whose density differs from the whole graph's, or ``--mode auto``, which does that only where it
matters. A pair that is not on disk is downloaded from SNAP before the searches start (``--no-download`` refuses instead).

A **warm** calibration (the default) starts each search at the recorded optimum: the resolution is searched in
[gamma/f, gamma*f] around it (the search still gallops outward if the best probe sits on an edge), the cap search starts
at the recorded cap, and the clique-regime probe is skipped. A **full** calibration starts from the search's own bracket.
``auto`` chooses per search (``search_mode``): warm when the target is the whole graph or the ``all`` cover of a network
that has a recorded optimum (its covered subgraph is the same graph minus the uncovered vertices, so the known optimum is
a good start), full otherwise. All end with the same confirmation on seeds 0-4. The search keeps its evaluations in
``<run>/ledger/``, a folder of this run only; it memoizes by configuration and never reuses an evaluation of another
build.

    hedonic run calibrate --networks dblp:top5000,amazon:top5000 --mode auto --name calibration-hedonic
    hedonic run exp --config paper-benchmark --tuned ~/.../calibration-hedonic/hedonic_tuned.json

The search is the packaged ``hedonic.experiments.overlapping.calibration_search``, run as ``python -m``; ``--script FILE``
runs a search script with the same command line instead (the 1.x ``bisect_tuning.py``).

The run folder holds ``hedonic_tuned.json`` (settings, the recorded values they replace, scores, cost, the build; ``complete``
is true once every search has finished), ``ledger/``, ``progress.json`` and ``worker.log``. A folder calibrated before
the method's rename holds ``hoc_tuned.json`` instead: it is read and continued under that name (``tuned_path``).

A **lazy** calibration (``--budget N``, ``--cap-minutes M``) runs the same search and stops it before it asks for a new
selection evaluation once N have been asked for or the probes already run cost M single-core minutes
(``lazy_search``). The setting is the best-observed (gamma, cap) by selection-seed F1: no rational snap, no confirmation,
no branch check, so ``f1_median`` is one seed's score and the entry carries a ``lazy`` block. Use it where the full search
is out of reach, not for a final-quality claim.

Cores. A probe is one detector run (seconds) followed by its scoring and audit (a minute of single-threaded Python on a
large graph), and a search takes its probes one at a time because each probe's result chooses the next. Two things are
independent and run side by side: the searches, and the evaluations inside a search that do not depend on one another
(the seeds of the confirmation, the points of a scan, the candidates of the branch check). ``--workers N`` (default: the
performance cores) is one budget for all of it: every evaluation takes a slot of ``<run>/ledger/slots`` for as long as it
runs, so the searches together never use more than N cores. Every record is a deterministic function of its
configuration, so the number of workers changes the wall-clock only, never what a search finds. ``--workers 1`` runs one
evaluation at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from hedonic.experiments.config import RUNS_ROOT, expand_path
from hedonic.experiments.overlapping import parallel
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.runtime import algorithm_identity, runtime_versions

PROTOCOL = "hedonic-calibration-v2"
TUNED_NAME = qs.TUNED_NAME  # hedonic_tuned.json; quickstart.LEGACY_TUNED_NAMES are read too
PHASE_OF = {"hedonic": "local"}
OLD_METHOD_NAMES = {"hoc_local": "hedonic"}  # files written before the rename call the method hoc_local
DROPPED_METHODS = ("hoc_multilevel",)  # its rows in an older file are ignored: the multilevel phase is gone
SEARCH_MODULE = "hedonic.experiments.overlapping.calibration_search"
LAZY_MODULE = "hedonic.experiments.overlapping.lazy_search"
DEFAULT_NETWORKS = "dblp,amazon"
DEFAULT_PHASES = "local"
DEFAULT_CONFIRM = "0-4"
DEFAULT_FACTOR = 1.5
CAP_MODES = ("search", "unlimited")
UNLIMITED = -1  # max_memberships of an unlimited cap
PROBE_LINE = re.compile(r"^\s*\[(\S+) (local|multilevel) M=(\d+) gamma=(\S+) .*?\] (\w+)")
MULTILEVEL_GONE = "--phases takes local only: the multilevel phase is not available in hedonic 2.0"

Progress = Callable[[dict[str, Any]], None]


def _silent(event: dict[str, Any]) -> None:
    return None


# --------------------------------------------------------------------------- plan
def build_parser(prog: str = "hedonic run calibrate") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Calibrate Hedonic's resolution and cap on the loaded build (warm start from the recorded optimum "
                     "by default) and write hedonic_tuned.json for the runs that take --tuned."))
    parser.add_argument("--script", metavar="FILE",
                        help="run this search script instead of the packaged search (a script with the command line of "
                             f"{SEARCH_MODULE}, such as the 1.x bisect_tuning.py)")
    parser.add_argument("--mode", choices=("warm", "full", "auto"), default="warm",
                        help="warm: start at the recorded optimum (default); full: the search's own bracket; auto: warm for "
                             "a target that is a network's whole graph or its `all` cover and has its own recorded optimum, "
                             "full for the rest (other covers, networks without an optimum)")
    parser.add_argument("--cap", choices=CAP_MODES, default="search",
                        help="search: search the resolution and the membership cap (default); unlimited: search the "
                             "resolution only, with no cap (max_memberships -1), on a wide log grid")
    parser.add_argument("--networks", default=DEFAULT_NETWORKS,
                        help=f"comma list; a name alone is the whole graph, network:cover (dblp:top5k) the graph induced "
                             f"by that cover's vertices (default {DEFAULT_NETWORKS})")
    parser.add_argument("--covers", metavar="LIST",
                        help="ground-truth covers to cross with every network, e.g. all,top5k: each pair is calibrated on "
                             "the covered-induced graph of that cover")
    parser.add_argument("--phases", default=DEFAULT_PHASES, help=f"the phase to calibrate; only local remains (default {DEFAULT_PHASES})")
    parser.add_argument("--factor", type=float, default=DEFAULT_FACTOR,
                        help=f"warm start: resolution bracket [gamma/F, gamma*F] (default {DEFAULT_FACTOR})")
    parser.add_argument("--confirm-seeds", default=DEFAULT_CONFIRM, help=f"seeds of the confirmation (default {DEFAULT_CONFIRM})")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="cores for the whole calibration: the searches run side by side and share this budget "
                             "(default: the performance cores, or HEDONIC_WORKERS); 1 runs one evaluation at a time")
    parser.add_argument("--budget", type=int, metavar="N",
                        help="lazy calibration: stop each search before its Nth+1 selection evaluation and keep the "
                             "best-observed setting by selection-seed F1 (no snap, no confirmation); see lazy_search")
    parser.add_argument("--cap-minutes", type=float, metavar="M",
                        help="lazy calibration: also stop a search once its selection evaluations have cost M "
                             "single-core minutes (detection + scoring + audit); needs --budget")
    parser.add_argument("--timeout", type=float, metavar="S",
                        help="seconds one detector run may take before it is killed (default 1800); each run of a search "
                             "is timed on its own, never the search's total")
    parser.add_argument("--limit-seconds", type=float, metavar="S",
                        help="a queue's time limit for one detector run: lowers --timeout (a probe it stopped is a "
                             "timeout, rerun under a larger limit)")
    parser.add_argument("--network-root", metavar="DIR",
                        help="SNAP networks folder (default: HEDONIC_NETWORKS_DIR, the user config's network_root, else "
                             "the package default)")
    parser.add_argument("--snap-cache-dir", metavar="DIR", help="normalized SNAP cache (default: the package's)")
    parser.add_argument("--no-download", action="store_true",
                        help="refuse a pair that is not on disk instead of downloading it from SNAP")
    parser.add_argument("--python", help="interpreter that runs the searches (default: this one)")
    parser.add_argument("--name", help="run name (default hedonic-calibration)")
    parser.add_argument("--output-dir", help="run folder (default: the user config's output folder / NAME)")
    parser.add_argument("--overwrite", action="store_true", help="delete this calibration's results and ledger first")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the commands; run nothing")
    parser.add_argument("--foreground", action="store_true",
                        help="hedonic run calibrate: run in this terminal instead of a durable background run")
    parser.add_argument("--detach", action="store_true", help="hedonic run calibrate: start the durable run and return")
    return parser


def resolve(argv: list[str], prog: str = "hedonic run calibrate") -> dict[str, Any]:
    """The plan of ``hedonic run calibrate ARGV``. Reads nothing but the arguments; writes nothing."""
    from hedonic.experiments.overlapping import userconfig

    args = build_parser(prog).parse_args(argv)
    networks = qs.parse_targets(args.networks)
    if args.covers:
        networks = qs.expand_covers(networks, args.covers)
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    for phase in phases:
        if phase not in PHASE_OF.values():
            raise ValueError(MULTILEVEL_GONE if phase == "multilevel" else f"{MULTILEVEL_GONE}, not {phase!r}")
    methods = [m for m, phase in PHASE_OF.items() if phase in phases]
    for network in networks:
        for method in methods:
            if start_for(method, network) is None:
                raise ValueError(f"no setting to start from for {method} on {network!r}; "
                                 f"choose from {', '.join(qs.HEDONIC_TUNED[method])} (or a cover of one of them)")
    if not networks or not methods:
        raise ValueError("nothing to calibrate")
    if not args.factor > 1.0:
        raise ValueError("--factor must be greater than 1")
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.budget is not None and args.budget < 1:
        raise ValueError("--budget must be at least 1")
    if args.cap_minutes is not None and (args.budget is None or not args.cap_minutes > 0):
        raise ValueError("--cap-minutes must be positive and goes with --budget")
    script = expand_path(args.script).resolve() if args.script else None
    if script is not None and not script.is_file():
        raise ValueError(f"--script {args.script}: no such file (leave --script out to use the packaged search)")
    name = qs.parse_name(args.name or "hedonic-calibration")
    root = Path(userconfig.defaults().get("output_dir") or RUNS_ROOT)
    output = expand_path(args.output_dir).resolve() if args.output_dir else expand_path(root / name).resolve()
    for flag, value in (("--timeout", args.timeout), ("--limit-seconds", args.limit_seconds)):
        if value is not None and not value > 0:
            raise ValueError(f"{flag} must be a positive number of seconds")
    if args.cap == "unlimited" and args.budget is not None:
        raise ValueError("--cap unlimited searches the resolution only; it does not go with --budget")
    return {"script": script, "mode": args.mode, "cap": args.cap, "networks": networks, "methods": methods,
            "factor": float(args.factor), "confirm": qs.parse_seeds(args.confirm_seeds), "python": args.python or sys.executable,
            "workers": parallel.default_workers(args.workers),
            "budget": args.budget, "cap_minutes": args.cap_minutes,
            "timeout": args.timeout, "limit": args.limit_seconds,
            "network_root": expand_path(args.network_root).resolve() if args.network_root else None,
            "snap_cache_dir": expand_path(args.snap_cache_dir).resolve() if args.snap_cache_dir else None,
            "download": not args.no_download,
            "name": name, "output": output, "overwrite": bool(args.overwrite), "dry_run": bool(args.dry_run)}


def start_for(method: str, network: str) -> dict[str, Any] | None:
    """The setting a search of ``network`` (a network or a ``network:cover`` target) starts from: its own recorded
    optimum, else its network's whole-graph one, else DBLP's (the transfer YouTube and LiveJournal have used)."""
    known = qs.HEDONIC_TUNED[method]
    base = qs.split_target(network)[0]
    for key in (network, base, "dblp"):
        if key in known:
            return {**known[key], "source": key if key in (network, base) else f"{key} (transfer)"}
    return None


def search_mode(plan: dict[str, Any], network: str, method: str) -> str:
    """``warm`` or ``full`` for one search: the plan's mode, or under ``auto`` warm exactly when the target is a network's
    whole graph or its ``all`` cover and the network has its own recorded optimum (not a transfer from another network)."""
    if plan["mode"] != "auto":
        return plan["mode"]
    base, cover = qs.split_target(network)
    start = start_for(method, network)
    own = start is not None and start["source"] in (network, base)
    return "warm" if own and cover in (None, "all") else "full"


def searches(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """The searches in run order, method by method (Hedonic is the only method calibrated here)."""
    out = []
    for method in plan["methods"]:
        for network in plan["networks"]:
            out.append({"key": f"{method}/{network}", "method": method, "network": network, "phase": PHASE_OF[method]})
    return out


def search_argv(plan: dict[str, Any], search: dict[str, Any]) -> list[str]:
    """The command of one search: the packaged search run with ``python -m`` (or ``--script``), wrapped by the lazy
    runner under ``--budget``. A warm start passes the recorded optimum as the bracket centre and the start cap."""
    start = start_for(search["method"], search["network"])
    script = plan.get("script")
    if plan.get("budget") is not None:
        head = [plan["python"], "-m", LAZY_MODULE, *(["--script", str(script)] if script else []),
                "--budget", str(plan["budget"]), *(["--cap-minutes", f"{plan['cap_minutes']:g}"] if plan.get("cap_minutes") else [])]
    else:
        head = [plan["python"], str(script)] if script else [plan["python"], "-m", SEARCH_MODULE]
    argv = [*head, "tune", "--instance", search["network"], "--phase", search["phase"],
            "--tag", f"{search['network'].replace(':', '_')}-{search['phase']}", "--out", str(plan["output"] / "ledger"),
            "--confirm-seeds", ",".join(map(str, plan["confirm"]))]
    if plan["workers"] > 1:
        argv += ["--jobs", str(plan["workers"]), "--slot-dir", str(plan["output"] / "ledger" / "slots"),
                 "--slot-count", str(plan["workers"])]
    timeout = run_timeout(plan)
    if timeout is not None:  # one detector run's timeout (the search's default when neither is given)
        argv += ["--timeout", f"{timeout:g}"]
    if plan.get("cap") == "unlimited":  # the resolution only, no cap: the search's wide scan, no warm bracket
        argv += ["--fixed-M", str(UNLIMITED)]
    elif search_mode(plan, search["network"], search["method"]) == "warm":
        gamma, cap = float(start["absolute_resolution"]), int(start["max_memberships"])
        argv += ["--bracket", f"{gamma / plan['factor']:.9g},{gamma * plan['factor']:.9g}", "--M-start", str(cap),
                 "--no-extremal"]
    if not script:  # where the packaged search finds the data (a 1.x script has no such flags)
        if plan.get("network_root"):
            argv += ["--network-root", str(plan["network_root"])]
        if plan.get("snap_cache_dir"):
            argv += ["--snap-cache-dir", str(plan["snap_cache_dir"])]
        if not plan.get("download", True):
            argv.append("--no-download")
    return argv


def run_timeout(plan: dict[str, Any]) -> float | None:
    """The timeout of one detector run: ``--timeout`` (else the search's own default), lowered by ``--limit-seconds``;
    None when neither is given, so the search keeps its default and its command line stays as it was."""
    from hedonic.experiments.overlapping import limits
    from hedonic.experiments.overlapping.calibration_search import DEFAULT_TIMEOUT

    if plan.get("timeout") is None and plan.get("limit") is None:
        return None
    return limits.effective(plan.get("timeout") or DEFAULT_TIMEOUT, plan.get("limit"))


def data_status(plan: dict[str, Any]) -> dict[str, str]:
    """For every target, ``on disk`` or what a run would download first. Reads only the file listing."""
    from hedonic.experiments.overlapping import snap

    out = {}
    for network in dict.fromkeys(s["network"] for s in searches(plan)):
        base, cover = qs.split_target(network)
        try:
            missing = snap.pair_downloads(base, cover or "all", plan.get("network_root"))
        except Exception as exc:  # noqa: BLE001 - a plan line, never a failure
            out[network] = f"unknown ({exc})"
            continue
        out[network] = ("on disk" if not missing else
                        ("to download: " if plan.get("download", True) else "MISSING (--no-download): ")
                        + ", ".join(Path(dest).name for _url, dest in missing))
    return out


def provision(plan: dict[str, Any], emit: Callable[[str], Any]) -> None:
    """Put every target's graph and cover on disk before the searches start (``snap.ensure_pair``, which downloads the
    official SNAP files when they are absent and ``download`` is on). One target at a time in this process: two searches
    of one network running side by side must never download the same file at once."""
    from hedonic.experiments.overlapping import snap

    for network in dict.fromkeys(s["network"] for s in searches(plan)):
        base, cover = qs.split_target(network)
        try:
            snap.ensure_pair(base, cover or "all", plan.get("network_root"), download=plan.get("download", True),
                             say=lambda line: emit(f"hedonic calibration: {line}"))
        except snap.SnapLoadError as exc:
            raise RuntimeError(f"the data of {network} is not available: {exc}") from exc


def describe(plan: dict[str, Any]) -> str:
    lazy = (f", lazy: {plan['budget']} evaluations" + (f" or {plan['cap_minutes']:g} probe min" if plan.get("cap_minutes") else "")
            if plan.get("budget") is not None else "")
    count = len(searches(plan))
    start = ("resolution only, no cap (-1), wide log scan" if plan.get("cap") == "unlimited"
             else f"{plan['mode']} start{lazy}")
    return (f"Hedonic calibration ({start}) · {count} search{'' if count == 1 else 'es'}: "
            + ", ".join(s["key"] for s in searches(plan)) + f" · {plan['workers']} worker{'s' if plan['workers'] != 1 else ''}")


def plan_lines(plan: dict[str, Any]) -> list[str]:
    done = recorded(plan)
    script = plan.get("script")
    lines = [f"hedonic run calibrate · {describe(plan)} ({PROTOCOL})",
             (f"  script   {script}" + ("" if script.is_file() else "   (NOT FOUND)")) if script
             else f"  search   {SEARCH_MODULE} (packaged)",
             f"  build    {qs.environment_line()}",
             f"  folder   {plan['output']} ({tuned_path(plan['output']).name}, ledger/, progress.json, worker.log)",
             (f"  budget   {plan['budget']} selection evaluations"
              + (f" or {plan['cap_minutes']:g} single-core probe minutes" if plan.get("cap_minutes") else "")
              + ", best-observed F1 setting, no confirmation (lazy)" if plan.get("budget") is not None
              else f"  confirm  seeds {', '.join(map(str, plan['confirm']))}"),
             f"  workers  {plan['workers']} "
             + ("(the searches run side by side and share this many cores)" if plan["workers"] > 1
                else "(one evaluation at a time)")]
    data = data_status(plan) if not script else {}
    if not script:
        from hedonic.experiments.overlapping import snap

        lines.append(f"  data     {plan.get('network_root') or snap.networks_root()}"
                     + ("" if plan.get("download", True) else " (no downloads)"))
    for search in searches(plan):
        start = start_for(search["method"], search["network"])
        mode = search_mode(plan, search["network"], search["method"])
        what = ("gamma on a log grid over [1e-6, 0.5], then refined · no cap" if plan.get("cap") == "unlimited" else
                f"{mode:<4} · reference gamma {start['absolute_resolution']:.6g}, cap {start['max_memberships']}"
                f" ({start['source']})")
        lines.append(f"  {search['key']:<28} {what}"
                     + ("   [recorded]" if search["key"] in done else "")
                     + (f"   [data {data[search['network']]}]" if search["network"] in data and search["key"] not in done
                        else ""))
    lines.append(f"  present  {len(done)} of {len(searches(plan))} search{'' if len(searches(plan)) == 1 else 'es'} already recorded "
                 + ("(--overwrite: deleted first)" if plan["overwrite"] else "(a run resumes the rest)"))
    return lines


# --------------------------------------------------------------------------- results
def tuned_path(folder: Path) -> Path:
    """The calibration file of a run folder: ``hedonic_tuned.json``, or the ``hoc_tuned.json`` a release before the
    method's rename wrote there (then read and continued under that name, so a folder never holds two of them)."""
    return qs.tuned_file(Path(folder) / TUNED_NAME)


def read_tuned(plan: dict[str, Any]) -> dict[str, Any]:
    """The calibration file of this run ({} when absent; ``tuned_path``). A file written before the method's rename names
    it ``hoc_local``: its rows and searches are read as ``hedonic``'s, and the rows of the dropped ``hoc_multilevel`` are
    left out."""
    path = tuned_path(plan["output"])
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return normalize_tuned(data) if isinstance(data, dict) else {}


def normalize_tuned(data: dict[str, Any]) -> dict[str, Any]:
    def method_of(name: str) -> str:
        return OLD_METHOD_NAMES.get(name, name)

    settings = data.get("settings")
    if isinstance(settings, dict):
        merged: dict[str, Any] = {}
        for method, rows in settings.items():
            if method in DROPPED_METHODS:
                continue
            merged.setdefault(method_of(method), {}).update(rows if isinstance(rows, dict) else {})
        data["settings"] = merged
    found = data.get("searches")
    if isinstance(found, dict):
        renamed = {}
        for key, entry in found.items():
            method, slash, target = str(key).partition("/")
            if method in DROPPED_METHODS:
                continue
            renamed[f"{method_of(method)}{slash}{target}"] = entry
        data["searches"] = renamed
    return data


def current_build() -> dict[str, str]:
    """The build a search's evaluations are made on (``calibration_search.build``)."""
    from hedonic.experiments.overlapping.calibration_search import build

    return build()


def recorded(plan: dict[str, Any]) -> set[str]:
    """The searches of the calibration file made on this build. A search of another build (or of 1.x, which recorded none)
    is not recorded: a run searches it again and replaces its entry, and the queue does not count it."""
    mine = current_build()
    return {key for key, entry in (read_tuned(plan).get("searches") or {}).items()
            if isinstance(entry, dict) and entry.get("build") == mine and same_cap_mode(plan, entry)}


def same_cap_mode(plan: dict[str, Any], entry: dict[str, Any]) -> bool:
    """A recorded search answers this plan only under the same cap mode: an unlimited calibration's entries have cap
    -1, a cap search's a positive cap."""
    unlimited = int(entry.get("max_memberships") or 0) == UNLIMITED
    return unlimited == (plan.get("cap") == "unlimited")


def other_build(plan: dict[str, Any]) -> set[str]:
    """The searches of the calibration file that another build made (kept until this build replaces them)."""
    return set(read_tuned(plan).get("searches") or {}) - recorded(plan)


def state_of(plan: dict[str, Any]) -> str:
    """complete | partial | missing, from the result file alone."""
    done = recorded(plan)
    wanted = {s["key"] for s in searches(plan)}
    if wanted <= done:
        return "complete"
    return "partial" if done & wanted else "missing"


def final_of(ledger: Path, search: dict[str, Any]) -> dict[str, Any]:
    """The last finished search of this instance and phase in the script's search ledger."""
    last = None
    path = ledger / "searches.jsonl"
    for line in (path.read_text().splitlines() if path.is_file() else []):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        instance = record.get("instance") or {}
        name = instance.get("spec") if isinstance(instance, dict) else instance
        if record.get("kind") == "tune" and record.get("phase") == search["phase"] and name in (search["network"], None):
            last = record
    if last is None:
        raise RuntimeError(f"the search for {search['key']} finished without a record in {path}")
    return last


def chosen_gamma(summary: dict[str, Any], start: float | None = None) -> float:
    """The resolution the search chose, exactly: the snapped fraction when it was kept, else the located optimum.

    The probes are evaluated at 6 significant digits, so a search that comes back to its starting point reports that
    point rounded (0.142857 for 1/7). That is the same configuration, not a new optimum: it returns ``start`` itself, so
    an unchanged optimum keeps the exact value and every record made at it stays valid under ``--tuned``."""
    snap = summary.get("snap") or {}
    if snap.get("kept") and snap.get("fraction"):
        return float(Fraction(snap["fraction"]))
    located = float(f"{float(summary['final']['gamma']):.6g}")  # the value the search evaluated and confirmed
    if start is not None and located == float(f"{float(start):.6g}"):
        return float(start)
    return located


def record_search(plan: dict[str, Any], search: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Add one finished search to the calibration file (atomically; ``tuned_path``) and return the file's content."""
    start = start_for(search["method"], search["network"])
    final = summary["final"]
    gamma, cap = chosen_gamma(summary, start["absolute_resolution"]), int(final["M"])
    snap = summary.get("snap") or {}
    entry = {"max_memberships": cap, "absolute_resolution": gamma,
             "fraction": snap.get("fraction") if snap.get("kept") else None,  # a fraction the search did not keep is not the setting
             "f1_median": final.get("f1_median"), "onmi_median": final.get("onmi_median"),
             "f1_sd": final.get("f1_sd"), "confirm_seeds": final.get("confirm_seeds"),
             "detector_evaluations": summary.get("detector_evaluations"), "wall_hours": summary.get("wall_hours"),
             "start_source": start["source"], "mode": search_mode(plan, search["network"], search["method"]),
             "start": {"max_memberships": int(start["max_memberships"]),
                       "absolute_resolution": float(start["absolute_resolution"])},
             "changed": cap != int(start["max_memberships"]) or gamma != float(start["absolute_resolution"])}
    # The build the search's evaluations were made on: the search's own record, else this run's (an external --script
    # reports none, and runs on the run's interpreter).
    entry["build"] = summary.get("build") or current_build()
    if summary.get("lazy"):  # a stopped search: its setting is the best observed, scored on one selection seed
        entry["lazy"] = {k: summary["lazy"][k] for k in ("budget", "cap_minutes", "stopped_by", "stopped_after", "evaluations",
                                                        "probe_seconds", "detection_seconds", "selection")}
    data = read_tuned(plan) or {}
    data.update({"protocol": PROTOCOL, "mode": plan["mode"], "cap": plan.get("cap", "search"),
                 "algorithm_identity": algorithm_identity(),
                 "environment": runtime_versions(),
                 "search": str(plan["script"]) if plan.get("script") else SEARCH_MODULE})
    data.setdefault("searches", {})[search["key"]] = entry
    data.setdefault("settings", {}).setdefault(search["method"], {})[search["network"]] = {
        "max_memberships": cap, "absolute_resolution": gamma}
    mine = current_build()
    data["complete"] = {s["key"] for s in searches(plan)} <= {
        key for key, value in data["searches"].items()
        if isinstance(value, dict) and value.get("build") == mine and same_cap_mode(plan, value)}
    path = tuned_path(plan["output"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)
    return data


# --------------------------------------------------------------------------- run
def run_search(plan: dict[str, Any], search: dict[str, Any], progress: Progress, emit: Callable[[str], Any],
               prefix: str = "") -> None:
    argv = search_argv(plan, search)
    emit(prefix + "  $ " + " ".join(argv))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    probes = 0
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            emit(prefix + line.rstrip())
            found = PROBE_LINE.match(line)
            if found:
                probes += 1
                progress({"type": "probe", "key": search["key"], "probes": probes,
                          "text": f"{search['key']}: probe {probes} (gamma {found[4]}, cap {found[3]}) {found[5]}"})
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"the search for {search['key']} exited with status {code}; see the worker log")


def run_calibration(plan: dict[str, Any], *, progress: Progress = _silent, emit: Callable[[str], Any] = print) -> int:
    out = plan["output"]
    out.mkdir(parents=True, exist_ok=True)
    if plan["overwrite"]:
        for name in (TUNED_NAME, *qs.LEGACY_TUNED_NAMES):
            (out / name).unlink(missing_ok=True)
        ledger = out / "ledger"
        for path in ledger.glob("*") if ledger.is_dir() else []:
            path.unlink()
    script = plan.get("script")
    if script and not script.is_file():
        raise RuntimeError(f"the search script {script} does not exist")
    todo = searches(plan)
    done = recorded(plan)
    wanted = {s["key"] for s in todo}
    pending = [s for s in todo if s["key"] not in done]
    emit(f"hedonic calibration: {len(todo)} searches ({plan['mode']} start); {len(done & wanted)} already recorded; "
         f"{plan['workers']} worker{'s' if plan['workers'] != 1 else ''}")
    if pending and not script:  # the packaged search: every pair on disk first (a script loads its own data)
        provision({**plan, "networks": [s["network"] for s in pending]}, emit)
    together = plan["workers"] > 1 and len(pending) > 1  # searches side by side; their lines carry a prefix
    lock = threading.RLock()
    finished = [len(done & wanted)]
    probes: dict[str, int] = {}
    active = {s["key"] for s in pending}  # the searches still running: a finished one is not shown as if it were
    failures: list[str] = []

    def say(line: str) -> None:
        with lock:
            emit(line)

    def stage_text() -> str:
        """Which searches are running and how many probes each has made (finished ones are counted, not listed)."""
        running = ", ".join(f"{key} probe {probes.get(key, 0)}" for key in sorted(active))
        return f"running {len(active)} of {len(todo)}: {running or 'none'} · finished {len(todo) - len(active)}"

    def event(item: dict[str, Any]) -> None:
        with lock:
            if item["type"] == "probe":
                probes[item["key"]] = item["probes"]
                progress({"type": "stage", "text": stage_text() if together else item["text"]})
            else:
                progress(item)

    def one(search: dict[str, Any]) -> None:
        prefix = f"[{search['key']}] " if together else ""
        try:
            run_search(plan, search, event, say, prefix)
            summary = final_of(out / "ledger", search)
            with lock:  # the calibration file is read-modify-written: one search at a time
                data = record_search(plan, search, summary)
                finished[0] += 1
                entry = data["searches"][search["key"]]
                was = entry.get("start") or {}
                emit(f"hedonic calibration: {search['key']} → gamma {entry['absolute_resolution']:.6g}, cap {entry['max_memberships']} "
                     f"(started at {was.get('absolute_resolution', float('nan')):.6g}, cap {was.get('max_memberships')}); "
                     f"F1 {entry['f1_median']}")
                active.discard(search["key"])
                progress({"type": "unit", "done": finished[0], "total": len(todo), "key": search["key"]})
                if together:
                    progress({"type": "stage", "text": stage_text()})
        except Exception as exc:  # noqa: BLE001 - a thread must never end silently; the run reports every failure
            with lock:
                failures.append(search["key"])
                active.discard(search["key"])
                emit(f"hedonic calibration: {search['key']} FAILED: {exc}")

    if together:
        threads = [threading.Thread(target=one, args=(search,), name=search["key"]) for search in pending]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for search in pending:
            one(search)
            if failures:
                break
    if failures:
        raise RuntimeError("the search for " + ", ".join(failures) + " failed; the finished searches are kept, resume to retry")
    final = read_tuned(plan)
    emit(f"hedonic calibration: {'complete' if final.get('complete') else 'NOT complete'}; wrote {tuned_path(out)}")
    return 0 if final.get("complete") else 1


# --------------------------------------------------------------------------- CLI entry points
def main(argv: list[str] | None = None) -> int:
    """``hedonic-exp overlapping-calibrate``: foreground run; use ``hedonic run calibrate`` for a durable one."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        plan = resolve(argv, prog="hedonic-exp overlapping-calibrate")
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic-exp overlapping-calibrate: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        for search in searches(plan):
            print("  $ " + " ".join(search_argv(plan, search)))
        print("  dry run: nothing started, nothing written")
        return 0
    try:
        return run_calibration(plan)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"hedonic-exp overlapping-calibrate: {exc}", file=sys.stderr)
        return 1


def front_study(argv: list[str]) -> dict[str, Any]:
    """The plan of ``hedonic run calibrate ARGV``. Loads nothing and writes nothing."""
    return resolve(argv)


def launch_argv(plan: dict[str, Any]) -> list[str]:
    """Every setting spelled out, so a resume rebuilds the same plan."""
    argv = (["--script", str(plan["script"])] if plan.get("script") else []) + [
        "--mode", plan["mode"], "--cap", plan.get("cap", "search"), "--networks", ",".join(plan["networks"]),
        "--phases", ",".join(sorted({PHASE_OF[m] for m in plan["methods"]})),
        "--factor", f"{plan['factor']:g}", "--confirm-seeds", ",".join(map(str, plan["confirm"])),
        "--workers", str(plan["workers"]), "--python", plan["python"], "--name", plan["name"],
        "--output-dir", str(plan["output"])]
    if plan.get("budget") is not None:
        argv += ["--budget", str(plan["budget"])]
    if plan.get("cap_minutes"):
        argv += ["--cap-minutes", f"{plan['cap_minutes']:g}"]
    if plan.get("timeout") is not None:
        argv += ["--timeout", f"{plan['timeout']:g}"]
    if plan.get("limit") is not None:
        argv += ["--limit-seconds", f"{plan['limit']:g}"]
    if plan.get("network_root"):
        argv += ["--network-root", str(plan["network_root"])]
    if plan.get("snap_cache_dir"):
        argv += ["--snap-cache-dir", str(plan["snap_cache_dir"])]
    if not plan.get("download", True):
        argv.append("--no-download")
    return argv


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run calibrate``: a durable run; ``--dry-run`` only prints the plan."""
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    detach = "--detach" in argv
    try:
        plan = resolve(argv)
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic run calibrate: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        print(style("  dry run: nothing started, nothing written", "dim"))
        return 0
    if "--foreground" in argv:  # in this terminal, like hedonic-exp overlapping-calibrate
        try:
            return run_calibration(plan)
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"hedonic run calibrate: {exc}", file=sys.stderr)
            return 1
    if plan["overwrite"]:
        try:
            runmanager._refuse_if_running(plan["name"])  # never delete results under a live worker
        except SystemExit as exc:
            print(f"hedonic run calibrate: {exc.code}", file=sys.stderr)
            return 2
        for name in (TUNED_NAME, *qs.LEGACY_TUNED_NAMES):
            (plan["output"] / name).unlink(missing_ok=True)
        ledger = plan["output"] / "ledger"
        for path in ledger.glob("*") if ledger.is_dir() else []:
            path.unlink()
    entry = runmanager.launch_calibrate(launch_argv(plan), plan["name"], str(plan["output"]))
    where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
             else f"background process (log {entry['log']})")
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


# --------------------------------------------------------------------------- durable run
def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body of a calibration run: the searches, with their progress mirrored into progress.json."""
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    plan = resolve(entry["argv"])
    todo = searches(plan)
    already = len(recorded(plan) & {s["key"] for s in todo})
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "calibrate", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": describe(plan), "command": entry["command"],
                 "environment": qs.environment(), "done": already, "total": len(todo), "resumed_from": already,
                 "eta_seconds": None, "stage": "starting the calibration", "output": str(tuned_path(plan["output"])),
                 "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(plan))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    started = time.time()  # the clock of the ETA: this launch, not the first finished search (that one arrives hours in)

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] == "stage":
            prog.update(status="running", stage=event["text"], updated=time.time())
        elif event["type"] == "unit":
            fresh = event["done"] - prog["resumed_from"]
            eta = (time.time() - started) / fresh * (event["total"] - event["done"]) if fresh else None
            prog.update(status="running", done=event["done"], total=event["total"], last=event["key"], eta_seconds=eta,
                        updated=time.time())
        rm.write_json(ppath, prog)

    prog.update(status="running", updated=time.time())
    rm.write_json(ppath, prog)
    code = run_calibration(plan, progress=on_event)
    if code != 0:
        raise RuntimeError("the calibration did not complete")
    prog.update(status="completed", done=len(todo), eta_seconds=0, stage=None, finished=time.time(), updated=time.time())
    rm.write_json(ppath, prog)
    print(style(f"  wrote {tuned_path(plan['output'])}", "green"), flush=True)
    return 0


def snapshot(entry: dict[str, Any], prog: dict[str, Any], frame: str) -> str:
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    status = rm.effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "interrupted": "red",
              "stopped": "yellow", "failed": "red"}.get(status, "dim")
    done, total = int(prog.get("done", 0)), int(prog.get("total", 0))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting") else prog.get("updated", time.time()))
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "Hedonic calibration"), "dim")]
    env = prog.get("environment") or {}
    if env:
        core = env.get("hedonic_core_sha256")
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {core[:12] if core else '?'} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = f"  {done}/{total} searches · elapsed {qs.fmt_time(end - started)}"
        if status in ("running", "setup") and prog.get("eta_seconds") is not None:
            tail += f" · ETA ~{qs.fmt_time(prog['eta_seconds'])} (rough: searches differ in length)"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    if status in ("running", "setup") and prog.get("stage"):
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    if status in ("running", "setup") and prog.get("last"):
        lines.append(style(f"  last finished: {prog['last']}", "dim"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); restart with: hedonic run resume {entry['name']}", "red"))
    elif status == "stopped":
        lines.append(style(f"\n  stopped by you; resume keeps the finished searches and the ledger: "
                           f"hedonic run resume {entry['name']}", "yellow"))
    if status == "completed":
        lines.append(style(f"\n  settings  {prog.get('output', entry['output_dir'])}", "dim"))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
