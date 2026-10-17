"""Sequences of durable runs: ``hedonic run queue FILE.toml``.

A queue file lists the runs in the order they should execute::

    name = "paper-queue"

    [[items]]
    name = "gt-robustness"          # the run's name: the queue looks the run up by it
    run = ["spectrum", "--config", "full", "--name", "gt-robustness", "--output-dir", "{root}/gt-robustness"]

``run`` holds the arguments after ``hedonic run``; ``--name`` repeats the item's name. ``{root}`` in an argument is the
queue's root folder: ``hedonic run queue FILE --root DIR``, by default ``<output_dir>/<queue name>`` with the user
config's ``[defaults] output_dir`` (else ``~/hedonic-runs``). A shipped queue is named without its file:
``hedonic run queue paper`` reads ``queue-paper.toml`` from the package's configs.

Each item is an ordinary durable run (its own tmux session, its own row in ``hedonic run list``, its own
``attach``/``stop``/``resume``). The queue starts an item with ``--detach`` only after the previous one has
completed, so every item gets all configured workers. The queue is a run too: ``hedonic run status NAME``
shows every item.

Running the same queue again continues it. An item whose results are already on disk is skipped; the test is the
one its runner uses to reuse work (see ``data_state``). A stopped or interrupted item is relaunched with its own
arguments and resumes from its finished records.

Pair by pair. A queue may name its pairs (``pairs = ["dblp:top5000", ...]``, in the order they run) and mark items
``by_pair = true``. The by-pair items then run pair by pair: the whole block of them on the first pair, then on the
second, and so on, instead of each item on all pairs before the next item starts. Each step is the item's own run
restricted to one pair (``slice_run``), started under the item's name and writing to the item's folder, so its records
are the records the item would have written with all its pairs at once and nothing already on disk is orphaned. The
files decide each step as they decide an item: a pair whose files are complete is skipped.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable
import io
import json
import os
import re
import shutil
import sys
import textwrap
import threading
import time
import tomllib
from pathlib import Path

from hedonic.experiments.overlapping import queuebuild as qb
from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import fit, style

POLL_SECONDS = 30  # how often the queue checks the run it is waiting for
ACTIVE = ("running", "setup", "starting")
WIDTH = 80  # the queue view is read in a terminal this wide; longer lines are cut, never wrapped
COLOUR = {"awaiting": "dim", "running": "cyan", "setup": "magenta", "starting": "magenta", "completed": "green",
          "completed_with_failures": "yellow", "interrupted": "red", "stopped": "yellow", "failed": "red"}


BENCHMARK_VERBS = ("exp", "paper", "paper-hoc")  # their run name is generated unless --name is given


def _check_arguments(run: list[str]) -> None:
    """Parse one item's arguments with the parser of its own front door. Nothing is loaded, computed or written."""
    from hedonic.experiments.CLI import RUN_ALIASES

    verb, args = run[0], run[1:]
    if verb not in RUN_ALIASES:
        raise ValueError(f"{verb!r} is not a run that a queue can start")
    try:
        _check_verb_arguments(verb, args, run)
    except qs.TunedMissing:
        pass  # the calibration that an earlier item of the queue writes does not exist yet


def _check_verb_arguments(verb: str, args: list[str], run: list[str]) -> None:
    if verb in BENCHMARK_VERBS:
        qs.config_from_args(qs.build_parser().parse_args(args if verb == "exp" else ["--config", verb, *args]))
    elif verb == "controlled":
        from hedonic.experiments.overlapping import controlled_overlap_config

        controlled_overlap_config.resolve(args)
    elif verb == "startcurve":
        from hedonic.experiments.overlapping import start_quality

        start_quality.front_study(args)
    elif verb == "audit":
        from hedonic.experiments.overlapping import cover_audit

        cover_audit.front_study(args)
    elif verb == "calibrate":
        from hedonic.experiments.overlapping import calibrate

        calibrate.front_study(args)
    elif verb == "calibrate-baselines":
        from hedonic.experiments.overlapping import baseline_run

        baseline_run.front_study(args)


def _absolute_config(run: list[str]) -> list[str]:
    """A ``--config`` that names a file (relative to the directory the queue starts in, or a config shipped with the
    package: ``configs/start-quality-hoc-local.toml``, ``overlap-heatmap``) is stored as its absolute path, so a resume
    from another directory still finds it. A name that is no file (``--config full`` of a spectrum, a profile such as
    ``paper``) is left to its front door."""
    from hedonic.experiments import resources

    run = list(run)
    if "--config" in run and run.index("--config") + 1 < len(run):
        index = run.index("--config") + 1
        if run[0] in BENCHMARK_VERBS and run[index] in qs.PROFILES:
            return run
        found = resources.config_path(run[index])
        if found.is_file():
            run[index] = str(found.resolve())
    return run


SPECTRUM_VERBS = ("spectrum", "spectrum-full")


def pair_id(text: str) -> str:
    """A canonical queue target; public ``top5k`` labels map to SNAP's stable ``top5000`` ID."""
    text = str(text).strip().lower()
    for place, char in enumerate(text):
        if char in ":/-":
            return f"{text[:place]}:{qs.snap.canonical_cover_variant(text[place + 1:] or 'all')}"
    return f"{text}:all"


def _with_option(run: list[str], flag: str, value: str) -> list[str]:
    """``run`` with ``flag`` set to ``value`` (replacing its value when it is already there)."""
    run = list(run)
    if flag in run and run.index(flag) + 1 < len(run):
        run[run.index(flag) + 1] = value
        return run
    return [*run, flag, value]


def _without_option(run: list[str], flag: str) -> list[str]:
    run = list(run)
    if flag in run:
        place = run.index(flag)
        del run[place:place + 2]
    return run


def item_targets(run: list[str]) -> list[str]:
    """The pairs an item covers, each spelled as the item's own runner takes it, in the order the item lists them.

    Empty for an item without pairs (the calibration, whose searches are one budget, or a synthetic study). Nothing is
    loaded or written; a ``--tuned`` file that does not exist yet is left out, since the pairs do not depend on it.
    """
    run = _without_option(run, "--tuned")
    verb, args = run[0], run[1:]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if verb in BENCHMARK_VERBS:
            return list(qs.config_from_args(qs.build_parser().parse_args(args if verb == "exp" else ["--config", verb, *args])).networks)
        if verb == "startcurve":
            from hedonic.experiments.overlapping import start_quality

            return list(start_quality.front_study(args)["networks"])
        if verb in SPECTRUM_VERBS:
            return [f"{dataset}/{cover}" for dataset, cover in _spectrum_options(verb, args).get("pairs") or []]
    return []


def slice_run(run: list[str], pair: str, order: list[str]) -> list[str]:
    """The item's arguments restricted to what the queue runs at the turn of ``pair`` (``order`` is the queue's pairs).

    A benchmark or a start-quality curve takes exactly that pair (``--networks``): its records are one file per unit and
    the units of a pair are the same units whatever else the run lists. A spectrum takes the pair together with the pairs
    before it (``--pairs``, in the queue's order): the study writes its coverage report, summary and figures for the pairs
    of the call it ends, so after the first k pairs they describe those k and after the last they describe the item; a
    pair already done costs only its load, since its audits and conditions are files with their own identities.
    """
    targets = item_targets(run)
    place = order.index(pair)
    if run[0] in SPECTRUM_VERBS:
        upto = sorted((t for t in targets if order.index(pair_id(t)) <= place), key=lambda t: order.index(pair_id(t)))
        return _with_option(run, "--pairs", ",".join(upto))
    mine = [t for t in targets if pair_id(t) == pair]
    return _with_option(_without_option(run, "--covers"), "--networks", ",".join(mine))


def schedule(queue: dict) -> list[dict]:
    """The steps the worker takes, in order: ``{"index": item number, "item": name, "pair": pair or None, "run": argv}``.

    An item that is not by pair is one step, where it stands in the file. The by-pair items form one block, placed where
    the first of them stands: it goes pair by pair in the queue's pair order, and on each pair its items run in the order
    of the file (an item with nothing on a pair has no step there). A queue without pairs is one step per item.
    """
    items, order = queue["items"], queue.get("pairs") or []
    steps: list[dict] = []
    placed = False
    for index, item in enumerate(items):
        if not item.get("by_pair"):
            steps.append({"index": index, "item": item["name"], "pair": None, "run": item["run"]})
        elif not placed:
            placed = True
            members = [(i, it) for i, it in enumerate(items) if it.get("by_pair")]
            for pair in order:
                steps += [{"index": i, "item": it["name"], "pair": pair, "run": slice_run(it["run"], pair, order)}
                          for i, it in members if pair in (it.get("pairs") or [])]
    return steps


def shipped_queues() -> list[str]:
    """The names of the queues shipped with the package (``paper`` for ``queue-paper.toml``)."""
    from hedonic.experiments import resources

    return [p.stem.removeprefix("queue-") for p in resources.shipped_configs()
            if p.name.startswith("queue-") and p.suffix == ".toml"]


def queue_file(name_or_path: str) -> Path:
    """The queue file a ``hedonic run queue`` argument names: a file as given, else a shipped queue by its name
    (``paper`` -> ``queue-paper.toml``) or by its file name."""
    from hedonic.experiments import resources

    raw = Path(str(name_or_path)).expanduser()
    if raw.is_file():
        return raw
    for candidate in (resources.config_path(str(name_or_path)), resources.config_path(f"queue-{name_or_path}")):
        if candidate.is_file():
            return candidate
    shipped = ", ".join(shipped_queues()) or "none"
    raise SystemExit(f"no queue file {name_or_path}; shipped queues: {shipped} (for example `hedonic run queue paper`)")


def default_root(name: str) -> Path:
    """Where a queue's items write by default: ``<output_dir>/<queue name>``, with the user config's ``[defaults]
    output_dir`` (``hedonic config set output_dir DIR``), else ``~/hedonic-runs`` (``HEDONIC_OUTPUT_DIR``)."""
    from hedonic.experiments import config
    from hedonic.experiments.overlapping import userconfig

    base = userconfig.defaults().get("output_dir")
    return (Path(os.path.expanduser(str(base))) if base else Path(config.RUNS_ROOT)) / name


def load_queue(path: str, root: str | Path | None = None) -> dict:
    """Read and check a queue file; every problem is reported before anything starts.

    ``{root}`` in an item's arguments becomes ``root`` (default :func:`default_root`), made absolute."""
    file = queue_file(path)
    path = str(file)
    data = tomllib.loads(file.read_text())
    try:
        name = qs.parse_name(str(data.get("name") or file.stem))
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    root_dir = (Path(os.path.expanduser(str(root))) if root else default_root(name)).resolve()
    pairs: list[str] = []
    if data.get("pairs") is not None:
        listed = data["pairs"]
        if not isinstance(listed, list) or not listed or not all(isinstance(p, str) for p in listed):
            raise SystemExit(f"{path}: pairs must be a list of network:cover targets, for example [\"dblp:top5k\"]")
        try:
            qs.parse_targets(",".join(listed))
        except (ValueError, SystemExit) as exc:
            raise SystemExit(f"{path}: pairs: {exc}") from None
        pairs = [pair_id(p) for p in listed]
        if len(set(pairs)) != len(pairs):
            raise SystemExit(f"{path}: a pair is listed twice in pairs")
    items: list[dict] = []
    seen: set[str] = set()
    for number, item in enumerate(data.get("items") or [], 1):
        run = item.get("run")
        if not isinstance(run, list) or not run or not all(isinstance(arg, str) for arg in run):
            raise SystemExit(f"{path}: item {number} needs run = [\"VERB\", ...], for example run = [\"spectrum\"]")
        try:
            item_name = qs.parse_name(str(item.get("name", "")))
        except ValueError as exc:
            raise SystemExit(f"{path}: item {number}: {exc}") from None
        if item_name in seen or item_name == name:
            raise SystemExit(f"{path}: the run name {item_name!r} is used more than once")
        run = _absolute_config([arg.replace("{root}", str(root_dir)) for arg in run])
        given = _option(run, "--name")
        if given is not None and given != item_name:
            raise SystemExit(f"{path}: item {item_name!r} runs as {given!r}; its --name must be the item's own name")
        if run[0] in BENCHMARK_VERBS and given is None:
            raise SystemExit(f"{path}: item {item_name!r} needs --name {item_name}; a benchmark's name is otherwise generated")
        problem = io.StringIO()
        try:
            with contextlib.redirect_stderr(problem):
                _check_arguments(run)
        except (SystemExit, ValueError) as exc:  # argparse has printed its usage error; its last line says why
            reason = str(exc) if isinstance(exc, ValueError) else (problem.getvalue().strip().splitlines() or ["invalid arguments"])[-1]
            raise SystemExit(f"{path}: item {item_name!r}: {reason}") from None
        seen.add(item_name)
        row = {"name": item_name, "run": run,
               **{key: str(item[key]) for key in ("title", "about", "short", "note") if item.get(key)}}
        if item.get("by_pair"):
            row.update(by_pair=True, pairs=_covered_pairs(path, item_name, run, pairs))
        items.append(row)
    if not items:
        raise SystemExit(f"{path} lists no [[items]]")
    marks = [bool(item.get("by_pair")) for item in items]
    if any(marks) and not all(marks[marks.index(True):len(marks) - marks[::-1].index(True)]):
        raise SystemExit(f"{path}: the by_pair items must stand next to each other: they run as one block, pair by pair")
    return {"name": name, "items": items, "file": str(file.resolve()), "root": str(root_dir),
            **({"pairs": pairs} if pairs else {})}


def _covered_pairs(path: str, item_name: str, run: list[str], order: list[str]) -> list[str]:
    """The queue's pairs that a by-pair item covers, in the queue's order. Every pair the item names must be one of the
    queue's (it would never run otherwise), and the restriction to each pair must be an argument list its runner accepts."""
    if not order:
        raise SystemExit(f"{path}: item {item_name!r} is by_pair, but the queue lists no pairs")
    try:
        targets = item_targets(run)
    except (SystemExit, ValueError) as exc:
        raise SystemExit(f"{path}: item {item_name!r}: its pairs cannot be read ({exc})") from None
    covered = [pair_id(t) for t in targets]
    if not covered:
        raise SystemExit(f"{path}: item {item_name!r} is by_pair, but its arguments name no pairs")
    unknown = [t for t in targets if pair_id(t) not in order]
    if unknown:
        raise SystemExit(f"{path}: item {item_name!r} covers {', '.join(unknown)}, which the queue's pairs do not list: "
                         "it would never run there")
    mine = [pair for pair in order if pair in covered]
    for pair in mine:
        sliced = slice_run(run, pair, order)
        if run[0] in SPECTRUM_VERBS and pair_id(item_targets(sliced)[-1]) != pair:
            raise SystemExit(f"{path}: item {item_name!r} restricted to {pair}: the study did not take --pairs")
        problem = io.StringIO()
        try:
            with contextlib.redirect_stderr(problem):
                _check_arguments(sliced)
        except (SystemExit, ValueError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else (problem.getvalue().strip().splitlines() or ["invalid arguments"])[-1]
            raise SystemExit(f"{path}: item {item_name!r} restricted to {pair}: {reason}") from None
    return mine


def _state(name: str) -> tuple[str, dict | None]:
    """Effective status of a registered run, or ``("new", None)`` when it has never been started."""
    path = rm.entry_path(name)
    if not path.is_file():
        return "new", None
    try:
        entry = json.loads(path.read_text())
    except ValueError:
        return "new", None
    return rm.effective_status(entry, rm.read_progress(entry)), entry


# --------------------------------------------------------------------------- what is already on disk
# A run is finished when its own files say so, by the rule its runner uses to reuse work:
#   benchmark (exp, paper, paper-hoc): every unit has a record stamped with this item's arguments
#   spectrum (spectrum, spectrum-full): its coverage report accounts for every condition
#   startcurve: every condition has a record in its records file
#   controlled: its study JSON exists (the study writes it once, at the end, and restarts otherwise)
#   calibrate: hoc_tuned.json holds every search of its plan and says complete
#   calibrate-baselines: baseline_tuned.json holds every (method, pair) calibration of its plan and says complete
#   audit: every cover of its plan has a record in its records file (name --methods, so the plan does not depend on disk)
# Other kinds (disjoint, posthoc) have no such rule; their registry entry decides.
DATA_TTL = 60.0  # seconds a data check is reused; the queue view is redrawn every second
# Checks that only read the run's own progress and record files are cheap, so the interactive view sees them nearly live;
# a benchmark check certifies every unit's files (hashes) and keeps the long reuse.
LIVE_VERBS = (*SPECTRUM_VERBS, "startcurve", "controlled", "calibrate", "calibrate-baselines")
LIVE_TTL = 2.0
_DATA_CACHE: dict[str, tuple[float, tuple[str, str]]] = {}
# What each state check counted on its way: folder -> (finished, total, noun). Filled by the checks below, read by
# ``unit_counts`` after ``data_state`` (so the same TTL applies and nothing is scanned twice).
_COUNTS: dict[str, tuple[int, int, str]] = {}


_pair_label = qb.pair_label  # ``dblp`` -> "DBLP · all", ``dblp:top5000`` -> "DBLP · top5k": covers always named


# The breakdown behind those counts: folder -> [(group, label, flags)], one flag per unit (finished or not). A group is
# a pair (``dblp:top5000``) and a label what is counted inside it (a method, a policy, a corruption level); ``report``
# draws the groups as subsections and the flags as marks.
_PARTS: dict[str, list[tuple[str, str, list[bool]]]] = {}
# Work that exists on disk but is not reused, and why: folder -> lines. Shown so that finished-looking work is never
# silently ignored by the report.
_EARLIER: dict[str, list[str]] = {}
# Why a run's recorded work does not count as done (its code changed since): folder -> reason.
_STALE: dict[str, str] = {}
# The loops behind a run's counts, for the interactive view: folder -> RunModel (units with coordinates and costs).
_MODEL: dict[str, qm.RunModel] = {}

# A step of a pair-by-pair queue asks the same state checks about one pair of an item. The notes above are the item's, kept
# per folder for the view, so a check made for a step must not write them: inside ``_quiet`` they stay as they were.
_SCRATCH = threading.local()


@contextlib.contextmanager
def _quiet():
    before = getattr(_SCRATCH, "quiet", False)
    _SCRATCH.quiet = True
    try:
        yield
    finally:
        _SCRATCH.quiet = before


def _is_quiet() -> bool:
    return bool(getattr(_SCRATCH, "quiet", False))


_DECORATE: dict[str, Callable] = {}  # folder -> adds rows to a run's node after it is built


def _set_model(key: str | Path, build, *args, **kwargs) -> None:
    """Register the loops behind a run's counts. The model only feeds the interactive view: if it cannot be built from
    what is on disk, the run keeps its counts and the view shows it without a breakdown."""
    if _is_quiet():
        return
    try:
        _MODEL[str(key)] = build(*args, **kwargs)
    except Exception:  # noqa: BLE001
        _MODEL.pop(str(key), None)


def _note(folder: str | Path, done: int, total: int, noun: str,
          parts: list[tuple[str, str, list[bool]]] | None = None, stale: str = "", earlier: list[str] | None = None) -> None:
    if _is_quiet():
        return
    _COUNTS[str(folder)] = (int(done), int(total), noun)
    _PARTS[str(folder)] = parts or []
    _EARLIER[str(folder)] = earlier or []
    _MODEL.pop(str(folder), None)
    if stale:
        _STALE[str(folder)] = stale
    else:
        _STALE.pop(str(folder), None)


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _option(args: list[str], flag: str) -> str | None:
    """The value given after ``flag`` in an item's arguments, or None."""
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def _unit_reusable(cfg: qs.Config, net: str, seed: int, method: qs.Method, binary: Path | None) -> bool:
    """Would the runner reuse this unit's record? Certified on the files (identity, final status, cover, seal), with no detector work."""
    return qs.unit_check(cfg, net, seed, method, binary).ok


def _benchmark_state(run: list[str]) -> tuple[str, str]:
    """A quickstart run: its folder, and whether the runner would reuse every unit in it."""
    rest = run[1:] if run[0] == "exp" else ["--config", run[0], *run[1:]]
    args = qs.build_parser().parse_args(rest)
    if not args.name:  # without a name the folder is only chosen when the run starts
        return "unknown", ""
    cfg = qs.config_from_args(args)
    folder = qs.expand_path(cfg.output_dir)
    if not str(cfg.output_dir).endswith(args.name):  # the folder runmanager.launch gives this run
        folder = folder / args.name
    cfg.output_dir = str(folder)
    progress = _read(folder / "progress.json") or {}
    unavailable = {method: "" for method in progress.get("unavailable") or {}}  # methods this machine cannot run
    binary = qs.codeseg_binary(qs.codeseg_tools(cfg)) if "codeseg" in cfg.methods else None
    if binary is not None and not binary.is_file():
        binary = None
    units = qs.build_plan(cfg, unavailable)
    audited = None
    if cfg.robustness:  # the run also audits every cover: a unit is finished only once its audits are recorded
        from hedonic.experiments.overlapping import cover_audit as ca

        audited = set(ca.read_records(folder / ca.STAGE_DIR / ca.RESULTS_NAME))
    reusable = [_unit_reusable(cfg, net, seed, method, binary) for net, seed, method in units]
    if audited is None:
        finished = reusable
    else:
        finished = [ok and all(key in audited for key in ca.unit_keys(net, seed, method.key))
                    for ok, (net, seed, method) in zip(reusable, units)]
    by_cell: dict[tuple[str, str], list[bool]] = {}
    for done, (net, _seed, method) in zip(finished, units):
        by_cell.setdefault((_pair_label(net), method.label), []).append(bool(done))
    _note(folder, sum(finished), len(units), "units", [(group, label, flags) for (group, label), flags in by_cell.items()])
    _set_model(folder, qb.benchmark, cfg, units, finished, progress)
    if not any(reusable):
        return "missing", str(folder)
    return ("complete" if all(finished) else "partial"), str(folder)


def _spectrum_options(verb: str, args: list[str]) -> dict:
    """The study's resolved options for an item (its config read, nothing loaded), or {} when they cannot be resolved."""
    from hedonic.experiments.overlapping import gt_spectrum

    study = gt_spectrum._strip_front(list(args))
    if verb != "spectrum":  # `hedonic run spectrum-full` is `--config full`
        study = ["--config", verb.removeprefix("spectrum-"), *study]
    try:
        return dict(gt_spectrum.resolve_options(gt_spectrum.build_parser().parse_args(study)))
    except (SystemExit, Exception):  # noqa: BLE001
        return {}


def _spectrum_pairs(verb: str, args: list[str]) -> list[str] | None:
    """The jobs (``dataset-cover``) the item's config names, or None when it leaves the pairs to discovery."""
    pairs = _spectrum_options(verb, args).get("pairs")
    return [f"{dataset}-{cover}" for dataset, cover in pairs] if pairs else None


def _spectrum_state(args: list[str], verb: str = "spectrum") -> tuple[str, str]:
    """A spectrum study: its folder, and whether its coverage report accounts for every condition of every planned pair.

    A pair added to the item's config after the study finished is a pair the report does not list: the item is then
    partial, and the study (which resumes) runs only that pair's conditions.
    """
    from hedonic.experiments.overlapping import userconfig

    explicit, name = _option(args, "--output-dir"), _option(args, "--name")
    if explicit:
        folder = qs.expand_path(explicit)
    elif name:  # without --output-dir the front door puts the study under its name, in the user's output folder
        from hedonic.experiments.config import RUNS_ROOT

        folder = qs.expand_path(userconfig.defaults().get("output_dir") or RUNS_ROOT) / name
    else:
        return "unknown", ""
    from hedonic.experiments.overlapping import gt_spectrum as gs

    report = _read(folder / "coverage_report.json") or {}
    expected = int(report.get("expected_detector_conditions") or 0)
    recorded = int(report.get("recorded_detector_conditions") or 0)
    planned = _spectrum_pairs(verb, args)
    listed = set(report.get("jobs") or [])
    live = _read(folder / "progress.json") or {}  # the study's own plan, kept up to date while it runs
    live_jobs = {str(j).replace("/", "-") for j in live.get("jobs") or []}
    current = gs.tracked_digest()
    # Records carry the hash of the code that wrote them and are reused only when it equals the current one. The study's
    # plan says which code it ran under; a coverage report is judged by the manifest written with it.
    manifest = _read(folder / "manifest.json") or {}
    report_digest = gs.canonical_digest((manifest.get("identity") or {}).get("tracked_files_sha256"))
    plan_current = gs.canonical_digest(live.get("tracked_files_sha256")) == current
    report_current = report_digest == current if manifest else True
    stale = ""
    if not report_current and (recorded or live.get("plan")):
        stale = (f"{recorded or sum(rm.unit_finished(r) for r in live.get('plan') or [])} recorded conditions were "
                 "written by other code (a tracked file changed since): they run again")
    if live.get("plan") and not plan_current:
        live = {}
    parts: list[tuple[str, str, list[bool]]] = []
    rows = None
    if live.get("plan"):
        if planned is None or set(planned) <= live_jobs:
            rows = live["plan"]
        elif live_jobs and live_jobs < set(planned):  # a step of a pair-by-pair queue: the study has the first pairs only
            done_jobs = set(listed) if report_current and report.get("complete") is True else set()
            rows = _stepped_plan(verb, args, live["plan"], planned, done_jobs)
    if rows:
        _note_spectrum_parts(parts, rows)
        _note_spectrum_audit(parts, rows, folder, live, current)
        _note(folder, sum(rm.unit_finished(row) for row in rows), len(rows), "conditions", parts)
        total, done, seconds = _spectrum_audit_progress(folder, rows, live, current)
        _set_model(folder, qb.spectrum, rows, folder=folder, live=live, options=_spectrum_options(verb, args),
                   audit_done=done, audit_seconds=seconds, unit_finished=rm.unit_finished)
    elif planned and expected and listed:  # the report lags a widened study: scale it to the planned pairs
        per_pair = expected / len(listed)
        valid = report_current
        for pair in planned:
            parts.append((_spectrum_label(pair), "conditions", [bool(valid and pair in listed)] * int(per_pair)))
        done = min(recorded, per_pair * len(planned)) if valid else 0
        _note(folder, done, per_pair * len(planned), "conditions", parts, stale)
    elif expected:
        _note(folder, recorded if report_current else 0, expected, "conditions", [], stale)
    if not _is_quiet():
        _EARLIER[str(folder)] = _spectrum_earlier(folder, current)
    if stale and not (live.get("plan") and plan_current):
        return "missing", str(folder)
    if planned is not None and not set(planned) <= listed:
        return ("partial" if recorded else "missing"), str(folder)
    if report.get("complete") is True and expected and recorded >= expected:
        return "complete", str(folder)
    return ("partial" if recorded else "missing"), str(folder)


def _stepped_plan(verb: str, args: list[str], plan: list[dict], planned: list[str], done_jobs: set[str]) -> list[dict]:
    """The rows of a whole item while the study runs only its first pairs (a step of a pair-by-pair queue).

    The study's own rows stand for the pairs it has; for each other pair of the item there is one row per condition, as the
    study would plan it. A pair the last coverage report lists as complete is finished: its conditions are files the study
    only reads again, so its rows read as finished at once instead of after the study has reloaded each of those pairs.
    """
    from hedonic.experiments.overlapping import gt_spectrum as gs

    options = _spectrum_options(verb, args)
    rows = [{**row, "status": "completed"} if str(row.get("job")) in done_jobs and not rm.unit_finished(row) else row
            for row in plan]
    seen = {str(row.get("job")) for row in plan}
    for dataset, cover in options.get("pairs") or []:
        job = f"{dataset}-{cover}"
        if job in seen or job not in planned:
            continue
        for policy in options["isolation_policies"]:
            for setting in gs.job_settings(options, dataset, cover):
                for seed in options["detector_seeds"]:
                    rows.append({"job": job, "dataset": dataset, "cover": cover, "policy": policy, "gamma": setting["gamma"],
                                 "seed": seed, **({"setting": setting["label"]} if setting["label"] is not None else {}),
                                 "status": "completed" if job in done_jobs else "pending"})
    return rows


def _spectrum_earlier(folder: Path, current: str) -> list[str]:
    """Per pair, the stored condition records written by other code than the current one (they are replaced as they rerun)."""
    import re

    from hedonic.experiments.overlapping import gt_spectrum as gs

    lines = []
    pattern = re.compile(r'"tracked_files_sha256":\s*"([0-9a-f]{64})"')
    for job in sorted((folder / "runs").glob("*")) if (folder / "runs").is_dir() else []:
        n = 0
        for record in job.glob("*.json"):
            try:
                match = pattern.search(record.read_text())
            except OSError:
                continue
            n += bool(match and gs.canonical_digest(match.group(1)) != current)
        if n:
            lines.append(f"{_spectrum_label(job.name)}: {n} records on disk from earlier code (kept, not reused)")
    return lines


def _spectrum_label(job: str) -> str:
    dataset, _, cover = job.partition("-")
    return _pair_label(f"{dataset}:{cover}" if cover and cover != "all" else dataset)


def _note_spectrum_parts(parts: list, rows: list[dict]) -> None:
    """Per pair and action policy, one flag per resolution: finished when every seed at that resolution has a result."""
    cells: dict[tuple[str, str, float], list[bool]] = {}
    for row in rows:
        cells.setdefault((str(row.get("job", "")), str(row.get("policy", "")), float(row.get("gamma", 0.0))),
                         []).append(rm.unit_finished(row))
    flags: dict[tuple[str, str], list[bool]] = {}
    for (job, policy, _gamma), done in cells.items():
        flags.setdefault((job, policy), []).append(all(done))
    for (job, policy), marks in flags.items():
        parts.append((_spectrum_label(job), policy.replace("_", " "), marks))


AUDIT_LABEL = "exact audit"  # a spectrum pair's exact stability audit: shown under its pair, never counted in its conditions
_AUDIT_STAGE = re.compile(r"^auditing (\S+)/(\S+) · .*?(\d+)/(\d+) resolutions")


def _spectrum_audit_progress(folder: Path, rows: list[dict], live: dict, current: str) -> tuple[int, dict[str, int], dict[str, float]]:
    """Per pair (job ``dataset-cover``), how many audit steps are finished, and the seconds the finished audits took.

    The audit is hours long on the large pairs and finishes before any of the pair's detector conditions start, so the
    condition rows alone read 0 for as long as it runs. A running audit is read from the worker's ``stage`` line; a
    finished one from its file in ``audit/`` when it was written by the current code (``canonical_digest``). Returns
    the steps one audit has (268 on the registered grid) as well.
    """
    from hedonic.experiments.overlapping import gt_spectrum as gs

    total, finished, seconds = 0, {}, {}
    jobs = sorted({str(r.get("job", "")) for r in rows} - {""})
    for job in jobs:
        data = _read(folder / "audit" / f"{job}.json") or {}
        stored = ((data.get("identity") or {}).get("implementation") or {}).get("tracked_files_sha256")
        if data.get("rows") and gs.canonical_digest(stored) == current:
            finished[job] = len(data["rows"])
            total = max(total, len(data["rows"]))
            if isinstance(data.get("audit_runtime_seconds"), (int, float)):
                seconds[job] = float(data["audit_runtime_seconds"])
    running = {}
    match = _AUDIT_STAGE.match(str(live.get("stage") or ""))
    if match:
        running = {f"{match[1]}-{match[2]}": int(match[3])}
        total = max(total, int(match[4]))
    done = {job: (total if job in finished and job not in running else running.get(job, 0)) for job in jobs}
    return total, done, seconds


def _note_spectrum_audit(parts: list, rows: list[dict], folder: Path, live: dict, current: str) -> None:
    """Per pair, how far its exact audit is: one flag per (policy, resolution), 268 on the registered grid."""
    total, done, _seconds = _spectrum_audit_progress(folder, rows, live, current)
    if not total:
        return
    for job in sorted(done):
        parts.append((_spectrum_label(job), AUDIT_LABEL, [True] * done[job] + [False] * (total - done[job])))


def _startcurve_state(args: list[str]) -> tuple[str, str]:
    """A start-quality curve: its folder, and whether every condition already has a record."""
    from hedonic.experiments.overlapping import start_quality as sq

    plan = sq.front_study(args)
    records = sq.read_records(Path(plan["output"]) / sq.RESULTS_NAME)
    keys = [condition["key"] for condition in sq.conditions(plan)]
    done = sum(key in records for key in keys)
    cells: dict[tuple[str, float], list[bool]] = {}
    for condition in sq.conditions(plan):
        cells.setdefault((condition["network"], condition["level"]), []).append(condition["key"] in records)
    _note(plan["output"], done, len(keys), "conditions",
          [(_pair_label(network), f"corruption {level if isinstance(level, str) else format(level, 'g')}", flags) for (network, level), flags in cells.items()],
          earlier=_startcurve_earlier(plan, records, set(keys)))
    _set_model(plan["output"], qb.startcurve, plan, records, sq.conditions(plan))
    state = "complete" if keys and done == len(keys) else "partial" if done else "missing"
    return state, str(plan["output"])


def _startcurve_earlier(plan: dict, records: dict, keys: set[str]) -> list[str]:
    """Records on disk that the current plan does not reuse: per pair, the settings they ran at against the ones now."""
    seen: dict[str, dict[tuple, int]] = {}
    for key, record in records.items():
        if key in keys:
            continue
        target = qb.record_target(record)  # the pair, cover included: top5000 and all are different pairs
        seen.setdefault(target, {}).setdefault((record.get("cap"), record.get("gamma")), 0)
        seen[target][(record.get("cap"), record.get("gamma"))] += 1
    lines = []
    for network, found in seen.items():
        now = (plan.get("settings") or {}).get(network)
        for (cap, gamma), n in found.items():
            at = f"cap {cap}, gamma {gamma:.4g}" if isinstance(gamma, (int, float)) else f"cap {cap}"
            want = f" (now cap {now['cap']}, gamma {now['gamma']:.4g})" if now else ""
            lines.append(f"{_pair_label(network)}: {n} conditions ran at {at}{want}; other settings (kept, not reused)")
    return lines


def _audit_state(args: list[str]) -> tuple[str, str]:
    """A cover audit: its folder, and whether every cover of its plan already has a record."""
    from hedonic.experiments.overlapping import cover_audit as ca

    plan = ca.front_study(args)
    records = ca.read_records(plan["output"] / ca.RESULTS_NAME)
    todo = ca.conditions(plan)
    _note(plan["output"], sum(c["key"] in records for c in todo), len(todo), "audits")  # (no breakdown: not queued)
    return ca.state_of(plan), str(plan["output"])


def _calibrate_state(args: list[str]) -> tuple[str, str]:
    """A HOC calibration: its folder, and whether hoc_tuned.json holds every search of its plan."""
    from hedonic.experiments.overlapping import calibrate as cal

    plan = cal.front_study(args)
    wanted = {search["key"] for search in cal.searches(plan)}
    have = cal.recorded(plan)
    _note(plan["output"], len(have & wanted), len(wanted), "searches",
          [(_pair_label(search["network"]), f"{search['phase']} search", [search["key"] in have])
           for search in cal.searches(plan)])
    searches = cal.read_tuned(plan).get("searches") or {}
    seconds = {key: float(detail["wall_hours"]) * 3600 for key, detail in searches.items()
               if isinstance(detail, dict) and isinstance(detail.get("wall_hours"), (int, float))}
    _set_model(plan["output"], qb.calibrate, cal.searches(plan), have, seconds)
    return cal.state_of(plan), str(plan["output"])


def _baselines_state(args: list[str]) -> tuple[str, str]:
    """The baselines' calibration: its folder, and whether baseline_tuned.json holds every (method, pair) of its plan."""
    from hedonic.experiments.overlapping import baseline_run as br

    plan = br.front_study(args)
    todo = br.searches(plan)
    have = br.recorded(plan)
    _note(plan["output"], len(have & {s["key"] for s in todo}), len(todo), "calibrations",
          [(_pair_label(network), "calibrations", [f"{method}/{network}" in have for method in plan["methods"]])
           for network in plan["networks"]])
    results = br.read_tuned(plan).get("searches") or {}
    _set_model(plan["output"], qb.baselines, plan, todo, results)
    _DECORATE[str(plan["output"])] = lambda node, plan=plan, results=results: qb.decorate_baselines(node, plan, results)
    return br.state_of(plan), str(plan["output"])


def _controlled_state(args: list[str]) -> tuple[str, str]:
    """A controlled-overlap study: complete once its JSON exists; otherwise it starts again from its first instance."""
    from hedonic.experiments.overlapping import controlled_overlap_config as cc

    plan = cc.resolve(args)
    total = int(plan["plan"]["instances"])
    grid = _controlled_grid(cc, plan["config"])
    if Path(plan["json"]).is_file():
        _note(plan["folder"], total, total, "instances", [("no network (synthetic)", "instances", [True] * total)])
        _set_model(plan["folder"], qb.controlled, plan["plan"], total, grid)
        return "complete", plan["folder"]
    progress = _read(Path(plan["folder"]) / "progress.json") or {}
    got = min(int(progress.get("done") or 0), total)
    _note(plan["folder"], got, total, "instances",
          [("no network (synthetic)", "instances", [True] * got + [False] * (total - got))])
    _set_model(plan["folder"], qb.controlled, plan["plan"], got, grid)
    return ("partial" if int(progress.get("done") or 0) else "missing"), plan["folder"]


def _controlled_grid(cc, config: str) -> dict[str, list[str]]:
    """The values of the study's three loops, read from its config's arguments (empty when it does not name them)."""
    try:
        arguments = cc.config_arguments(config)
    except (OSError, ValueError, SystemExit):
        return {}
    grid = {}
    for flag in ("graph-seeds", "mus", "overlap-fractions"):
        value = _option(arguments, f"--{flag}")
        if value:
            grid[flag] = _expand_list(value)
    return grid


def _expand_list(text: str) -> list[str]:
    """``0,1,2`` -> three values; ``0-9`` -> ten."""
    values: list[str] = []
    for part in text.split(","):
        low, dash, high = part.partition("-")
        if dash and low.strip().isdigit() and high.strip().isdigit():
            values += [str(i) for i in range(int(low), int(high) + 1)]
        elif part.strip():
            values.append(part.strip())
    return values


def _data_state(run: list[str]) -> tuple[str, str]:
    verb, args = run[0], run[1:]
    if verb in BENCHMARK_VERBS:
        return _benchmark_state(run)
    if verb in SPECTRUM_VERBS:
        return _spectrum_state(args, verb)
    if verb == "startcurve":
        return _startcurve_state(args)
    if verb == "controlled":
        return _controlled_state(args)
    if verb == "audit":
        return _audit_state(args)
    if verb == "calibrate":
        return _calibrate_state(args)
    if verb == "calibrate-baselines":
        return _baselines_state(args)
    return "unknown", ""  # no data check for this kind: the registry decides


def data_state(run: list[str], fresh: bool = False) -> tuple[str, str]:
    """What the data on disk says about one item: ("complete" | "partial" | "missing" | "awaiting" | "unknown", its folder).

    ``fresh`` ignores the reading kept for ``DATA_TTL`` and reads the files again (the queue's two verifications).

    "awaiting" is an item that reads a calibration file (--tuned) which an earlier item has yet to write: whether its
    records are still valid depends on the calibrated settings, so it is judged when its turn comes, never before.
    """
    key = json.dumps(run)
    cached = None if fresh else _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < (LIVE_TTL if run[0] in LIVE_VERBS else DATA_TTL):
        return cached[1]
    try:  # the planners print their own errors; a bad argument is reported when the item starts, not while viewing
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = _data_state(run)
    except qs.TunedMissing:
        result = ("awaiting", "")
    except (SystemExit, Exception):  # noqa: BLE001
        result = ("unknown", "")
    _DATA_CACHE[key] = (time.time(), result)
    return result


def unit_counts(run: list[str]) -> tuple[int, int, str] | None:
    """(finished, total, noun) for one item from its files, or None when its kind has no such count or is not judged yet."""
    state, folder = data_state(run)
    if state in ("unknown", "awaiting"):
        return None
    return _COUNTS.get(folder)


def _status_of(item: dict, fresh: bool = False) -> tuple[str, dict | None]:
    """An item's status, and its registry entry when that entry (not the data on disk) decides its state.

    Work on disk decides wherever it can. Finished work is completed. Partial work is stopped and is relaunched with the
    item's own arguments, since its runner reuses the finished records. Missing work is new. Only a kind without a data
    rule keeps its registry status, where a run that finished with recorded failures counts as finished.
    """
    status, entry = _state(item["name"])
    if status in ACTIVE:
        return status, entry
    data = data_state(item["run"], fresh)[0]
    if data == "complete":
        return "completed", entry
    if data == "partial":
        return "stopped", None
    if data == "missing":
        return "new", None
    if data == "awaiting":
        return "awaiting", None
    if status == "completed_with_failures":
        return "completed", entry
    return status, entry


def plan(queue: dict) -> list[tuple[str, str, str]]:
    """For each item: its current status and what a run of the queue would do with it."""
    rows = []
    for item in queue["items"]:
        status, _ = _status_of(item, fresh=True)
        if status == "completed":
            action = "skip: already completed"
        elif status == "new":
            action = "start"
        elif status == "awaiting":
            action = "decided when its turn comes: it reads the calibration, which an earlier item writes"
        elif status in ACTIVE:
            action = "wait: already running, the queue follows it"
        else:
            action = "resume from its finished records"
        rows.append((item["name"], "pending" if status == "new" else status, action))
    return rows


_STEP_ACTION = {"complete": "skip", "partial": "resume", "missing": "start", "awaiting": "judged at its turn", "unknown": "start"}


def plan_steps(queue: dict) -> list[tuple[str, str | None, str]]:
    """For each step of a pair-by-pair queue, in the order the worker takes them: (item, pair, what the queue would do)."""
    rows = []
    for step in schedule(queue):
        if step["pair"] is None:
            status = _status_of(queue["items"][step["index"]], fresh=True)[0]
            action = ("skip" if status == "completed" else "start" if status == "new" else
                      "judged at its turn" if status == "awaiting" else "follow" if status in ACTIVE else "resume")
        else:
            action = _STEP_ACTION.get(slice_state(step["run"], fresh=True)[0], "start")
        rows.append((step["item"], step["pair"], action))
    return rows


def counts(entry: dict) -> tuple[int, int]:
    """Completed items and total items of a queue."""
    done = sum(_status_of(item)[0] == "completed" for item in entry["items"])
    return done, len(entry["items"])


def _shown(status: str) -> str:
    return "pending" if status == "new" else status


def _tally(item: dict) -> str:
    """'done/total noun' of one item read from its files (empty when its kind has no count)."""
    count = unit_counts(item["run"])
    return f"{count[0]:,}/{count[1]:,} {count[2]}" if count else ""


def verify(items: list[dict]) -> list[tuple[str, str, str]]:
    """Pass one: read every item's files afresh. (name, status, count) per item; nothing is started or changed."""
    rows = []
    for item in items:
        status, _ = _status_of(item, fresh=True)
        rows.append((item["name"], _shown(status), _tally(item)))
    return rows


def _decision(first: str, second: str) -> str:
    """What the second look at an item's turn decided, against what the first said."""
    if second == "completed":
        return ("complete as pass 1 found it: skipped" if first == "completed"
                else f"pass 1 said {first}, but the data has appeared since: skipped")
    if first == "completed":
        return f"pass 1 said complete, but the files no longer certify it ({second}): it runs again"
    return {"pending": "still missing: starting", "awaiting": "calibration present now: judged",
            "stopped": "partly done: resuming"}.get(second, f"{second}: continuing")


def _launch_item(name: str, run: list[str], resuming: bool) -> int:
    """Start one item detached, or resume it when it already has a registry entry."""
    if resuming:
        return rm.cmd_resume(name, attach=False)
    from hedonic.experiments.CLI import hedonic_main

    return hedonic_main(["run", *run, "--detach"])


def _run_item(row: dict, prog: dict, ppath: Path, launch_item) -> str:
    """Start (or resume) one item and wait until it ends. Returns the item's final status."""
    name = row["name"]
    status, entry = _status_of(row, fresh=True)  # pass two: read its files again now, whatever the first look found
    first = row.get("first") or "pending"
    row.update(second=_shown(status), second_count=_tally(row), checked=time.time())
    print(style(f"  verify (2nd) {name}: {row['second']}{' · ' + row['second_count'] if row['second_count'] else ''}"
                f" (1st: {first}) — {_decision(first, status if status == 'completed' else row['second'])}", "dim"), flush=True)
    if status == "completed":
        row.update(status="completed", note="finished earlier, skipped" if first == "completed"
                   else "found complete at its turn (another run produced it): skipped")
        rm.write_json(ppath, prog)
        print(style(f"  ✓ {name} finished earlier — skipped", "green"), flush=True)
        return "completed"
    if status in ACTIVE:
        note = "already running, following it"
    else:
        resuming = status in ("stopped", "failed", "interrupted")
        note = "resuming from its finished records" if resuming else "starting"
        if launch_item(name, row["run"], resuming and entry is not None) != 0:
            row.update(status="failed", note="could not be started")
            rm.write_json(ppath, prog)
            return "failed"
    print(style(f"  ● {name} — {note}", "cyan"), flush=True)
    return _wait(name, row, prog, ppath)


def _wait(name: str, row: dict, prog: dict, ppath: Path) -> str:
    """Follow a started run until it ends; its final status (recorded failures count as completed)."""
    while True:
        status, entry = _state(name)
        if entry is None:
            row.update(status="failed", note="did not register under this name; check its config")
            rm.write_json(ppath, prog)
            return "failed"
        done, total = rm.progress_counts(entry, rm.read_progress(entry))
        row.update(status=status, done=done, total=total)
        rm.write_json(ppath, prog)
        if status in rm.TERMINAL or status == "interrupted":
            if status == "completed_with_failures":  # recorded failures are results: the runner never reruns them
                print(style(f"  completed with recorded failures: {name} (kept as results)", "yellow"), flush=True)
                return "completed"
            print(style(f"  {status}: {name}", COLOUR.get(status, "dim")), flush=True)
            return status
        time.sleep(POLL_SECONDS)


def slice_state(run: list[str], fresh: bool = False) -> tuple[str, str]:
    """What the files say about one step of an item (its run restricted to a pair): ``data_state`` without writing the
    notes the view keeps for the whole item."""
    with _quiet():
        return data_state(run, fresh)


def _run_step(row: dict, step: dict, mark: dict, prog: dict, ppath: Path, launch_item) -> str:
    """One step of a pair-by-pair queue: the item ``row`` on ``step['pair']``, with the arguments ``step['run']``. ``mark``
    is the step's row in the progress file. Returns the step's final status.

    The step's own files decide it at its turn, as an item's do: a pair that is complete is skipped, the rest runs. The run
    is always started fresh with the restricted arguments (never resumed from the registry, whose entry holds the arguments
    of whatever pair the item last ran on); its runner reuses every record already on disk.
    """
    name, pair, run = row["name"], step["pair"], step["run"]
    status, _entry = _state(name)
    if status in ACTIVE:  # the item is running already (started before this queue, or on the whole item): wait for it
        print(style(f"  ● {name} is already running — following it, then judging {pair}", "cyan"), flush=True)
        ended = _wait(name, row, prog, ppath)
        if ended != "completed":
            return ended
    state = slice_state(run, fresh=True)[0]
    row.update(pair=pair, second=f"{pair}: {state}", second_count=_tally(row), checked=time.time())
    print(style(f"  verify (2nd) {name} · {pair}: {state}", "dim"), flush=True)
    if state == "complete":
        mark.update(status="completed", note="finished earlier, skipped")
        rm.write_json(ppath, prog)
        print(style(f"  ✓ {name} · {pair} finished earlier — skipped", "green"), flush=True)
        return "completed"
    if launch_item(name, run, False) != 0:
        row.update(status="failed", note=f"could not be started on {pair}")
        mark.update(status="failed", note="could not be started")
        rm.write_json(ppath, prog)
        return "failed"
    print(style(f"  ● {name} · {pair} — {'resuming from its finished records' if state == 'partial' else 'starting'}", "cyan"),
          flush=True)
    mark.update(status="running", started=time.time())
    rm.write_json(ppath, prog)
    ended = _wait(name, row, prog, ppath)
    mark.update(status=ended, finished=time.time())
    return ended


def work(name: str, entry: dict, prog: dict, launch_item=_launch_item) -> int:
    """The queue worker: starts each step once the previous one has completed.

    A step is an item, or, in a queue with pairs, an item on one pair (``schedule``): the by-pair items go pair by pair.
    """
    ppath = rm.progress_path(entry)
    steps = schedule(entry)
    by_pair = any(step["pair"] for step in steps)
    prog.update(name=name, kind="queue", status="running", pid=os.getpid(), started=time.time(), updated=time.time(),
                current=None, pair=None, file=entry.get("file", ""),
                items=[{"name": i["name"], "run": i["run"], "status": "pending"} for i in entry["items"]],
                steps=[{"item": s["item"], "pair": s["pair"], "status": "pending"} for s in steps] if by_pair else [])
    rm.write_json(ppath, prog)
    how = (f"{len(entry['items'])} runs, by pair ({len(steps)} steps): the whole block on one pair before the next pair"
           if by_pair else f"{len(entry['items'])} runs, one after another")
    print(style(f"hedonic run queue {name}", "bold") + f" · {how}")
    print(style("Ctrl-C does not stop this queue. stop: hedonic run stop " + name, "dim"), flush=True)
    print(style("verify (1st): every run, from its files on disk, before anything starts", "bold"), flush=True)
    for row, (_, first, tally) in zip(prog["items"], verify(entry["items"])):
        row.update(first=first, first_count=tally)
        print(f"  {row['name']:<30} {first:<10} {tally}", flush=True)
    prog.update(verified=time.time())
    rm.write_json(ppath, prog)
    for place, step in enumerate(steps):
        row = prog["items"][step["index"]]
        prog.update(current=step["index"], pair=step["pair"], step=place, updated=time.time())
        if step["pair"] is None:
            ended = _run_item(row, prog, ppath, launch_item)
        else:
            ended = _run_step(row, step, prog["steps"][place], prog, ppath, launch_item)
            more = any(later["index"] == step["index"] for later in steps[place + 1:])
            if ended == "completed":  # the item as a whole is finished only with its last pair
                row.update(status="pending" if more else "completed",
                           note=f"{step['pair']} done; the next pair follows" if more else "all its pairs done")
        if ended != "completed":
            where = row["name"] + (f" on {step['pair']}" if step["pair"] else "")
            reason = f"{where} ended as {ended}; fix it, then continue with: hedonic run resume {name}"
            prog.update(status="stopped" if ended == "stopped" else "failed", reason=reason,
                        finished=time.time(), updated=time.time())
            rm.write_json(ppath, prog)
            print(style(f"\n  queue stopped: {reason}", "red"), flush=True)
            return 1
    prog.update(status="completed", finished=time.time(), updated=time.time(), current=None, pair=None)
    rm.write_json(ppath, prog)
    print(style(f"\n  queue completed: {len(prog['items'])} runs", "green"), flush=True)
    return 0


def launch(entry: dict, *, backend: str = "auto") -> dict:
    """Register the queue and start its worker detached (used for a new queue and for resume)."""
    name = entry["name"]
    rm._refuse_if_running(name)
    output = str(rm.registry_dir() / "queues" / name)
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "progress.json").unlink(missing_ok=True)
    use_tmux = backend == "tmux" or (backend == "auto" and shutil.which("tmux"))
    record = {"name": name, "kind": "queue", "items": entry["items"], "file": entry.get("file", ""),
              "pairs": entry.get("pairs") or [], "root": entry.get("root", ""), "output_dir": output,
              "created": time.time(),
              "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
              "log": str(Path(output) / "worker.log"),
              "command": f"hedonic run queue {entry.get('file', '')}" + (f" --root {entry['root']}" if entry.get("root") else "")}
    return rm._spawn(record)


def stop_items(entry: dict) -> None:
    """Stop the item that is running; the caller then stops the queue that waits for it."""
    for item in entry["items"]:
        if _state(item["name"])[0] in ACTIVE:
            rm.cmd_stop(item["name"])


_STOP_WORDS = {"stopped": "stopped", "failed": "failed", "interrupted": "interrupted"}
_NO_REASON = {"stopped": "stopped with hedonic run stop; the queue recorded no reason",
              "failed": "failed; the queue recorded no reason",
              "interrupted": "the queue worker is gone (reboot or kill)"}


def _culprit(entry: dict, prog: dict) -> tuple[int, str, str] | None:
    """The item that stopped the queue: its position, its name and its state. None when the queue stopped between runs."""
    current = prog.get("current")
    if isinstance(current, int) and 0 <= current < len(entry["items"]):
        name = entry["items"][current]["name"]
        state = _status_of(entry["items"][current])[0]
        if state != "completed":
            return current, name, state
    return None


def failure_lines(name: str) -> list[str]:
    """What a run that ended as failed recorded about why: the error (the last line of its reason), the step it was in
    (the first line) and where the whole traceback is. Empty for a run in any other state or without a recorded reason."""
    try:
        status, run_entry = _state(name)
        if status not in ("failed", "interrupted") or run_entry is None:
            return []
        reason = [line.strip() for line in str(rm.read_progress(run_entry).get("reason") or "").splitlines() if line.strip()]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    if not reason:
        return []
    lines = [f"why: {reason[-1]}"]
    if reason[0] != reason[-1]:
        lines.append(f"while: {reason[0].rstrip(':')}")
    lines.append(f"whole traceback: hedonic run logs {name}")
    return lines


def stop_lines(entry: dict, prog: dict, status: str) -> list[str]:
    """Where a stopped, failed or interrupted queue ended, why, and the command that continues it."""
    lines = []
    culprit = _culprit(entry, prog)
    if culprit is not None:
        position, name, _ = culprit
        pair = f" on {qm._public(prog['pair'])}" if prog.get("pair") else ""
        lines.append(style(f"  {_STOP_WORDS[status]} at {name}{pair} (item {position + 1} of {len(entry['items'])})",
                           COLOUR.get(status, "dim")))
        for line in failure_lines(name):  # long error lines wrap: the end of a message is the part that says what is wrong
            lines += [style(text, "red") for text in textwrap.wrap(line, width=WIDTH - 2, initial_indent="    ",
                                                                   subsequent_indent="        ")]
    resume = f"hedonic run resume {entry['name']}"
    # the worker's reason ends with its own copy of the resume command; the command is printed on a line of its own
    reason = str(prog.get("reason") or _NO_REASON[status]).replace(f"; fix it, then continue with: {resume}", "")
    lines += [style(line, "yellow" if status == "stopped" else "red")
              for line in textwrap.wrap(f"reason: {reason}", width=WIDTH - 2, initial_indent="  ",
                                        subsequent_indent="          ")]
    lines.append(style(f"  continue with: {resume}", "bold"))
    return lines


def _bar(done: int, total: int, width: int = 12, full: str = "█", empty: str = "░") -> str:
    filled = round(width * done / total) if total else 0
    return full * filled + empty * (width - filled)


def overview(items: list[dict], G: dict | None = None) -> list[str]:
    """One table for the whole queue: every run, what it has finished, what is left, and the queue's totals.

    The counts are those of the runs' own files (``unit_counts``), so a run that has not started shows 0 of its total
    and a finished one its full total; a unit is whatever that run counts (searches, conditions, benchmark units).
    """
    G = G or tui.glyphs(None)
    full, empty = G["full"], G["empty"]
    rows, finished_runs, runs_with_counts = [], 0, 0
    for number, item in enumerate(items, 1):
        status, entry = _status_of(item)
        shown = "pending" if status == "new" else status
        count = unit_counts(item["run"])
        finished_runs += status == "completed"
        if count is None and entry is not None:
            item_done, item_total = rm.progress_counts(entry, rm.read_progress(entry))
            count = (item_done, item_total, "units") if item_total else None
        rows.append((number, item["name"], shown, count))
    name_w = min(max(len(r[1]) for r in rows), 28)
    lines = [style("overview", "bold")]
    for number, name, shown, count in rows:
        if count is None:
            lines.append(f"  {number}. {name:<{name_w}} " + style(f"{shown:<10}", COLOUR.get(shown, "dim"))
                         + " " + style("not counted until its turn" if shown == "awaiting" else "no count", "dim"))
            continue
        done, total, noun = count
        runs_with_counts += 1
        percent = 100 * done // total if total else 0
        lines.append(f"  {number}. {name:<{name_w}} " + style(f"{shown:<9}", COLOUR.get(shown, "dim"))
                     + f" {_bar(done, total, 8, full, empty)} {percent:>3}% {done:,}/{total:,} {noun}")
    left_runs = len(items) - finished_runs
    lines.append(style(f"  total: {finished_runs} of {len(items)} runs complete · {left_runs} still to do", "bold"))
    return lines


_LABELS: dict[str, tuple[float, dict]] = {}


def with_labels(items: list[dict], file: str = "") -> list[dict]:
    """The items with the ``title``/``about``/``short`` their queue file gives them now.

    A queue's registry entry keeps the items as they were when it started; the words that describe them live in the
    file and may be added later, so the view reads them from the file (by item name) every time it is drawn.
    """
    try:
        path = Path(file)
        stamp = path.stat().st_mtime
        cached = _LABELS.get(file)
        if cached is None or cached[0] != stamp:
            data = tomllib.loads(path.read_text())
            _LABELS[file] = cached = (stamp, {str(i.get("name")): i for i in data.get("items") or []})
    except (OSError, ValueError):
        return items
    return [{**{k: str(cached[1][item["name"]][k]) for k in ("title", "about", "short", "note")
                if cached[1].get(item["name"], {}).get(k)}, **item} for item in items]


def matrix(items: list[dict], G: dict | None = None) -> list[str]:
    """The whole queue on one screen: a row per pair, a column per run, each cell the share of that pair's work done.

    It is the part of the live view that survives a short terminal: ten lines say, for every experiment and every
    network/cover pair, whether it is finished, going, or still to do, with the run totals underneath.
    """
    G = G or tui.glyphs(None)
    columns = []
    for item in items:
        status, _ = _status_of(item)
        count = unit_counts(item["run"])
        groups: dict[str, list[bool]] = {}
        for group, label, flags in _PARTS.get(data_state(item["run"])[1], []):
            if label != AUDIT_LABEL:
                groups.setdefault(group, []).extend(flags)
        columns.append((item.get("short") or item["name"][:9], status, count, groups))
    rows: list[str] = []
    for *_rest, groups in columns:
        rows += [group for group in groups if group not in rows]
    cell_w = 11
    label_w = min(max([len(row) for row in rows] + [9]), 22)

    def cell(status: str, done: int, total: int, text: str | None = None) -> str:
        mark = G["done"] if total and done == total else G["running"] if status in ACTIVE or done else G["pending"]
        colour = "green" if total and done == total else "cyan" if status in ACTIVE else "yellow" if done else "dim"
        shown = text if text is not None else (G["done"] if total and done == total else f"{100 * done // total}%" if total else "-")
        return style(f"{mark} {shown:<{cell_w - 2}}", colour)

    lines = [style(f"{'':<{label_w}} ", "bold") + "".join(style(f"{name[:cell_w - 1]:<{cell_w}}", "bold") for name, *_ in columns)]
    for row in rows:
        out = f"{row[:label_w]:<{label_w}} "
        for _name, status, _count, groups in columns:
            flags = groups.get(row)
            out += cell(status, sum(flags), len(flags)) if flags else style(f"{'':<{cell_w}}", "dim")
        lines.append(out)
    out = style(f"{'all pairs':<{label_w}} ", "bold")
    for _name, status, count, _groups in columns:
        out += cell(status, count[0], count[1], f"{count[0]}/{count[1]}") if count else style(f"{'?':<{cell_w}}", "dim")
    lines.append(out)
    return lines


# --------------------------------------------------------------------------- the whole queue, section by section
def _strip(flags: list[bool], G: dict, width: int = 24) -> str:
    """One mark per unit; more units than ``width`` are bucketed, each bucket shaded by the share of it finished."""
    full, empty = G["full"], G["empty"]
    if len(flags) <= width:
        return "".join(full if flag else empty for flag in flags)
    shades = ("░▒▓█" if full == "█" else "-=+#")
    out = []
    for bucket in range(width):
        chunk = flags[bucket * len(flags) // width:(bucket + 1) * len(flags) // width]
        share = sum(chunk) / len(chunk)
        out.append(shades[-1] if share == 1 else shades[0] if share == 0 else shades[1 + min(1, int(share * 2))])
    return "".join(out)


def _line(label: str, flags: list[bool], G: dict, indent: str, name_w: int, marks: bool = True) -> str:
    done, total = sum(flags), len(flags)
    mark = G["done"] if total and done == total else G["running"] if done else G["pending"]
    colour = "green" if total and done == total else "yellow" if done else "dim"
    text = f"{indent}{style(mark, colour)} {label:<{name_w}} {done:>4}/{total:<4}"
    return text + (" " + _strip(flags, G) if marks and total > 1 else "")


def _check_line(item: dict) -> str:
    """The two verifications of a run: what the first look found at the queue's start and the second at its turn."""
    if not item.get("first"):
        return ""
    text = f"checked: 1st (queue start) {item['first']}"
    if item.get("first_count"):
        text += f" · {item['first_count']}"
    if item.get("second"):
        text += f"  ->  2nd (its turn) {item['second']}"
        if item.get("second_count"):
            text += f" · {item['second_count']}"
        if item["second"] != item["first"] and not item.get("pair"):  # a step's look is about one pair, not the whole run
            text += "  (differs)"
    else:
        text += "  ->  2nd: when its turn comes"
    return text


def report(items: list[dict], G: dict | None = None, full: bool = False, frame: str | None = None) -> list[str]:
    """Everything the queue has to do, as sections: a run, then its pairs, then what is counted inside each pair.

    A finished run is folded to one line per pair. A run with work done or in progress opens up one more level
    (methods, policies or corruption levels, one mark per seed or resolution). ``full`` opens every run.
    """
    G = G or tui.glyphs(None)
    lines: list[str] = []
    totals = [0, 0]
    for number, item in enumerate(items, 1):
        status, entry = _status_of(item)
        shown = "pending" if status == "new" else status
        count = unit_counts(item["run"])
        folder = data_state(item["run"])[1]
        parts = _PARTS.get(folder, [])
        stale = _STALE.get(folder, "")
        if count is None and entry is not None:
            item_done, item_total = rm.progress_counts(entry, rm.read_progress(entry))
            count = (item_done, item_total, "units") if item_total else None
        colour = COLOUR.get(shown, "dim")
        title = f" · {item['title']}" if item.get("title") else ""
        lines.append("")
        lines.append(style(f"{number}. {item['name']}", "bold") + style(title, "dim"))
        if item.get("about"):
            lines += [style(line, "dim") for line in textwrap.wrap(item["about"], width=WIDTH - 5, initial_indent="   ",
                                                                  subsequent_indent="   ")]
        if count is None:
            lines.append("   " + style(shown, colour) + style("  (counted when its turn comes)" if shown == "awaiting" else "", "dim"))
            if _check_line(item):
                lines.append("   " + style(_check_line(item), "dim"))
            if frame is not None and status in ACTIVE and entry is not None:
                lines += rm.progress_block(entry, rm.read_progress(entry), status, frame, indent="   ", G=G)
            continue
        done, total, noun = count
        totals[0] += done
        totals[1] += total
        percent = 100 * done // total if total else 0
        left = f"{total - done:,} left" if done < total else "nothing left"
        lines.append("   " + style(f"{shown:<10}", colour) + f" {_bar(done, total, 14, G['full'], G['empty'])} {percent:>3}% · "
                     f"{done:,}/{total:,} {noun} · {left}")
        if _check_line(item):
            lines.append("   " + style(_check_line(item), "dim"))
        if stale:
            lines.append("   " + style(f"{G['stopped']} {stale}", "yellow"))
        for text in _EARLIER.get(folder, []):
            lines += [style(line, "yellow") for line in textwrap.wrap(
                f"{G['stopped']} earlier, not reused · {text}", width=WIDTH - 4, initial_indent="   ", subsequent_indent="     ")]
        if item.get("note"):
            lines += [style(line, "dim") for line in textwrap.wrap("note: " + item["note"], width=WIDTH - 5, initial_indent="   ",
                                                                  subsequent_indent="         ")]
        if status in ACTIVE and entry is not None:
            progress = rm.read_progress(entry)
            if progress.get("stage"):
                lines.append("   " + style(f"{G['running']} now: {progress['stage']}", "cyan"))
            if frame is not None:  # the live view: the run's own clock, ETA and unit in progress, inside its section
                live = rm.progress_block(entry, progress, status, frame, indent="   ", G=G)
                lines += live
        groups: dict[str, list[tuple[str, list[bool]]]] = {}
        for group, label, flags in parts:
            groups.setdefault(group, []).append((label, flags))
        if not groups:
            continue
        opened = full or status in ACTIVE or 0 < done < total
        group_w = min(max(len(group) for group in groups), 30)
        for group, rows in groups.items():
            counted = [(label, flags) for label, flags in rows if label != AUDIT_LABEL]
            merged = [flag for _label, flags in counted for flag in flags]
            single = len(counted) == 1 and len(rows) == 1
            started = 0 < sum(merged) < len(merged)
            lines.append(_line(group, merged, G, "   ", group_w, marks=single or (started and not opened)))
            if opened and not single:
                label_w = min(max(len(label) for label, _ in rows), 24)
                lines += [_line(label, flags, G, "      ", label_w) for label, flags in rows]
    lines.append("")
    left_runs = sum(_status_of(item)[0] != "completed" for item in items)
    lines.append(style(f"total: {len(items) - left_runs} of {len(items)} runs complete · {left_runs} still to do · "
                       f"{totals[0]:,}/{totals[1]:,} counted units", "bold"))
    return lines


def with_checks(items: list[dict], prog: dict) -> list[dict]:
    """The items with what the queue's two verifications found (kept in the worker's progress rows)."""
    rows = {str(r.get("name")): r for r in prog.get("items") or []}
    keys = ("first", "first_count", "second", "second_count", "pair")
    return [{**{k: rows[i["name"]][k] for k in keys if rows.get(i["name"], {}).get(k)}, **i} for i in items]


def body(items: list[dict], G: dict, full: bool = False, frame: str | None = None) -> list[str]:
    """The queue itself: the matrix, the overview, then every run as a section. The same lines serve the live view,
    ``status`` and ``--report``; ``frame`` (a spinner glyph) adds the running run's live progress to its section."""
    return matrix(items, G) + [""] + overview(items, G) + report(items, G, full, frame)


def snapshot(entry: dict, prog: dict, frame: str | None = None, table: bool = True,
             unicode_glyphs: bool | None = None) -> str:
    """What ``attach`` and ``status`` show: the whole queue, section by section, exactly what ``--report`` prints, with
    the running run's progress inside its section. ``table`` is accepted for the viewer's signature and ignored: the
    view is never shortened to the terminal (the viewer scrolls it)."""
    G = tui.glyphs(unicode_glyphs)
    frame = frame or G["dot"]
    status = rm.effective_status(entry, prog)
    done, total = counts(entry)
    lines = [style(f"queue {entry['name']} · {len(entry['items'])} runs", "bold") + "  "
             + style(f"{rm.status_mark(status, frame, G)} {status}", COLOUR.get(status, "dim"))
             + f"  · {done}/{total} runs completed"]
    if entry.get("pairs"):
        place = f" · now on {qm._public(prog['pair'])}" if status in ACTIVE and prog.get("pair") else ""
        lines.append(style(f"  by pair: {len(entry['pairs'])} pairs, the whole block of runs on one before the next{place}", "dim"))
    if status in _STOP_WORDS:
        lines += stop_lines(entry, prog, status)
    items = with_checks(with_labels(entry["items"], entry.get("file", "")), prog)
    lines += body(items, G, frame=frame)
    return "\n".join(fit(line, WIDTH) for line in lines)


_NODE_STATUS = {"completed": "done", "completed_with_failures": "done", "new": "pending", "stopped": "stopped",
                "interrupted": "stopped", "failed": "failed", "awaiting": "awaiting", "running": "running",
                "setup": "running", "starting": "running", "partial": "partial"}


def tree(entry: dict, prog: dict) -> qm.Node:
    """The queue as nodes for the interactive view: every run with the loops behind its counts and an ETA at each level.

    The runs are read as ``report`` reads them (the files on disk decide; the registry only where a kind has no rule),
    so the numbers of the tree are those of ``--report``, broken down.
    """
    items = with_checks(with_labels(entry["items"], entry.get("file", "")), prog)
    runs = []
    for number, item in enumerate(items, 1):
        status, run_entry = _status_of(item)
        if status == "stopped" and run_entry is None:  # work on disk that nothing is doing now: part done, not a stopped run
            status = "partial"
        state, folder = data_state(item["run"])
        count = unit_counts(item["run"])
        model = None if state in ("unknown", "awaiting") else _MODEL.get(folder)
        eta: qm.Cost = (None, "")
        if run_entry is not None:
            progress = rm.read_progress(run_entry)
            if count is None:
                done, total = rm.progress_counts(run_entry, progress)
                count = (done, total, "units") if total else None
            seconds = rm.remaining_seconds(run_entry, progress, status) if status in ACTIVE else None
            if seconds is not None:
                eta = (seconds, "measured")
        took = None
        if status == "completed" and run_entry is not None:  # the run's own clock, for a run whose units recorded no time
            took = rm.elapsed_seconds(run_entry, rm.read_progress(run_entry), "completed")
        node = qm.run_node(number, item["name"], item.get("title", ""), _NODE_STATUS.get(status, status), model, count,
                           item.get("about", ""), eta, took)
        if model is not None and folder in _DECORATE:
            try:
                _DECORATE[folder](node)  # rows below a leaf that the model cannot hold (the settings a search tried)
            except Exception:  # noqa: BLE001 - the view only decorates; it never changes a count
                pass
        for line in (_check_line(item), _STALE.get(folder, ""), *_EARLIER.get(folder, []), item.get("note", ""),
                     *failure_lines(item["name"])):
            if line:
                node.detail.append(line)
        if item.get("by_pair"):
            now = prog.get("pair") if status in ACTIVE and prog.get("current") == number - 1 else None
            node.detail.append(f"runs pair by pair, in the queue's order ({len(item.get('pairs') or [])} pairs)"
                               + (f"; now on {now} (step {int(prog.get('step') or 0) + 1} of {len(prog.get('steps') or [])})" if now else ""))
        runs.append(node)
    return qm.queue_node(entry["name"], runs, rm.effective_status(entry, prog), entry.get("file", ""),
                         order=list(entry.get("pairs") or []))


def _queue_item_named(name: str) -> dict | None:
    """The item of a registered queue that has this run's name: its arguments are the front door's, which the planners read."""
    for path in sorted(rm.registry_dir().glob("*.json")):
        try:
            queue = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for item in queue.get("items") or []:
            if queue.get("kind") == "queue" and item.get("name") == name and item.get("run"):
                return item
    return None


def single_tree(entry: dict, prog: dict) -> qm.Node:
    """One durable run as the same tree a queue shows: the run is read as a one-item queue of its own arguments.

    The arguments are those of the queue item that started it (same name) or, for a run started by hand, the
    ``hedonic run VERB ...`` it records. A run whose kind has no planner for them (``unknown``) has no tree: the
    caller keeps the panel. The tree is rooted at the run, not at a queue."""
    import shlex

    item = _queue_item_named(entry["name"])
    if item is None:
        words = shlex.split(entry.get("command") or "")
        if words[:2] != ["hedonic", "run"] or len(words) < 3:
            raise ValueError("this run was not started through `hedonic run`; it has no tree")
        item = {"name": entry["name"], "run": words[2:]}
    if data_state(item["run"], fresh=True)[0] in ("unknown", "awaiting"):
        raise ValueError("this kind of run has no planner to break its counts down")
    item = {"name": entry["name"], "run": item["run"]}
    root = tree({"name": entry["name"], "items": [item], "file": ""}, prog)
    root.label = f"run {entry['name']}"
    root.detail[:1] = [f"run {entry['name']}", f"hedonic run {' '.join(item['run'])}"[:300]]
    if root.children:
        root.children[0].label = root.children[0].label.split(". ", 1)[-1]
    return root


# --------------------------------------------------------------------------- what the queue needs on this machine
def _listed(value: str | None) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def requirements(queue: dict) -> tuple[list[str], list[str]]:
    """The network:cover pairs and the methods (``hedonic run exp`` keys) the queue's items use, in queue order.

    Read from the items' arguments only (nothing is loaded): the queue's ``pairs``, each item's targets and the
    ``--networks`` of a calibration; the methods of each benchmark and of a baseline calibration."""
    pairs: list[str] = list(queue.get("pairs") or [])
    methods: list[str] = []
    for item in queue["items"]:
        run = item["run"]
        verb = run[0]
        try:
            targets = item_targets(run)
        except (SystemExit, ValueError):
            targets = []
        if verb in ("calibrate", "calibrate-baselines"):
            targets = targets or _listed(_option(run, "--networks"))
        pairs += [pair_id(t) for t in targets]
        if verb in BENCHMARK_VERBS:
            args = _without_option(run, "--tuned")[1:]
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    cfg = qs.config_from_args(qs.build_parser().parse_args(
                        args if verb == "exp" else ["--config", verb, *args]))
                methods += list(cfg.methods)
            except (SystemExit, ValueError):
                pass
        elif verb == "calibrate-baselines":
            methods += [qs.ALIASES.get(m, m) for m in _listed(_option(run, "--methods"))]
    return list(dict.fromkeys(pairs)), list(dict.fromkeys(methods))


def setup_check(queue: dict) -> dict:
    """What ``hedonic setup`` still has to prepare for this queue: the pairs whose files are not under the network root,
    the methods whose runtime the setup manifest does not list as ready, and the command that prepares them."""
    from hedonic.experiments.overlapping import codeseg_setup as cs
    from hedonic.experiments.overlapping import snap

    pairs, methods = requirements(queue)
    root = snap.networks_root()
    missing_pairs = []
    for target in pairs:
        try:
            found = snap.pair_files(*snap.split_pair(target), root=root)
        except Exception:  # noqa: BLE001 - an unknown pair is the queue's own error, reported elsewhere
            found = None
        if found is None:
            missing_pairs.append(target)
    manifest = cs.load_setup_manifest()
    missing_methods = [m for m in methods if cs.needs_setup(m) and not cs._ready_before(manifest, m)]
    command = None
    if missing_pairs or missing_methods:
        parts = ["hedonic setup"]
        if missing_pairs:
            parts += ["--pairs", ",".join(missing_pairs)]
        if missing_methods:
            parts += ["--methods", ",".join(missing_methods)]
        command = " ".join(parts)
    return {"network_root": str(root), "pairs": pairs, "methods": methods, "missing_pairs": missing_pairs,
            "missing_methods": missing_methods, "command": command}


def _print_setup_check(check: dict) -> None:
    if not check["command"]:
        print(style(f"setup: the {len(check['pairs'])} pairs are under {check['network_root']} and the "
                    f"{len(check['methods'])} methods are ready", "dim"))
        return
    if check["missing_pairs"]:
        print(style(f"setup: {len(check['missing_pairs'])} pairs are not under {check['network_root']}: "
                    f"{', '.join(check['missing_pairs'])}", "yellow"))
    if check["missing_methods"]:
        print(style(f"setup: {len(check['missing_methods'])} method runtimes are not prepared: "
                    f"{', '.join(check['missing_methods'])}", "yellow"))
    print(f"  prepare them with: {check['command']}")
    print(style("  (data kept elsewhere: hedonic config set network_root DIR, or HEDONIC_NETWORKS_DIR)", "dim"))


ROOT_STAMP = ".hedonic-queue.json"


def _major(version: str) -> str:
    return str(version or "").split(".")[0]


def root_check(queue: dict) -> str | None:
    """Why the queue must not write into its root, or None when it may.

    The queue writes every item under ``root``. A root that this major version of hedonic stamped (``ROOT_STAMP``,
    written when a queue starts) is its own; an empty or absent root is new. A root that already holds folders and
    no stamp, or a stamp of another major version, holds another installation's runs (a 1.x queue's records would be
    rerun and overwritten in place), so the queue refuses it unless told to adopt it."""
    from hedonic.experiments import runtime

    root = Path(queue.get("root") or "")
    if not str(root) or not root.is_dir():
        return None
    current = runtime.hedonic_version()
    stamp = _read(root / ROOT_STAMP)
    if stamp:
        if _major(stamp.get("hedonic", "")) == _major(current):
            return None
        return f"{root} holds the runs of a hedonic {stamp.get('hedonic')} queue (this is {current})"
    entries = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    if not entries:
        return None
    shown = ", ".join(entries[:4]) + (", ..." if len(entries) > 4 else "")
    return (f"{root} already holds {len(entries)} folder(s) ({shown}) that no hedonic {_major(current)}.x queue "
            "started (an earlier installation's runs?)")


def stamp_root(queue: dict) -> None:
    """Mark the root as this hedonic's queue folder (kept if it is already marked)."""
    from hedonic.experiments import runtime

    root = Path(queue["root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / ROOT_STAMP
    if _read(path):
        return
    path.write_text(json.dumps({"queue": queue["name"], "hedonic": runtime.hedonic_version(),
                                "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2) + "\n")


def front_main(argv: list[str]) -> int:
    """``hedonic run queue FILE|NAME``: start the runs of a queue file one after another."""
    parser = argparse.ArgumentParser(
        prog="hedonic run queue",
        description="Run the experiments of a queue file one after another, each with every worker. Re-running "
                    "the same file skips finished runs (judged by their files on disk) and resumes the first "
                    "unfinished one.")
    parser.add_argument("file", metavar="FILE|NAME",
                        help="queue file, or the name of a shipped queue: " + (", ".join(shipped_queues()) or "none"))
    parser.add_argument("--root", default=None,
                        help="folder the items write into ({root} in the queue file); default "
                             "<output_dir>/<queue name>, output_dir from `hedonic config` else ~/hedonic-runs")
    parser.add_argument("--dry-run", action="store_true", help="print what would run and what is skipped, then exit")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    parser.add_argument("--report", action="store_true",
                        help="print the whole queue as sections (run > pair > method) from the files on disk, then exit")
    parser.add_argument("--full", action="store_true", help="with --report: open every run, finished ones too")
    parser.add_argument("--setup", action="store_true",
                        help="run `hedonic setup` for the pairs and methods that are missing before starting")
    parser.add_argument("--no-setup-check", action="store_true",
                        help="start even when pairs or method runtimes look missing (data kept elsewhere)")
    parser.add_argument("--adopt-root", action="store_true",
                        help="write into a --root that already holds runs this hedonic's queues did not start "
                             "(a record certifies only for the build that reads it; others are rerun in place)")
    args = parser.parse_args(argv)
    queue = load_queue(args.file, root=args.root)
    print(style(f"queue {queue['name']} · {len(queue['items'])} runs", "bold") + style(f" · root {queue['root']}", "dim"))
    items = with_labels(queue["items"], queue.get("file", ""))
    if args.report:
        print("\n".join(fit(line, WIDTH) for line in body(items, tui.glyphs(None), full=args.full)))
        return 0
    print(style("verify (1st): every run, from its files on disk", "bold"))
    for number, (name, status, action) in enumerate(plan(queue), 1):
        print(f"  {number}. {name:<28} {status:<12} → {action}")
    print(style("  each run is read again at its turn (2nd), so data that appeared or vanished since is caught", "dim"))
    if queue.get("pairs"):
        print(style("order: pair by pair — the by-pair runs take turns on each pair, in this order", "bold"))
        lines: list[tuple[str | None, list[str]]] = []
        for item, pair, action in plan_steps(queue):  # one line per run of steps on the same pair
            if not lines or lines[-1][0] != pair:
                lines.append((pair, []))
            lines[-1][1].append(f"{item} {action}")
        for pair, parts in lines:
            print(f"  {pair or 'whole queue':<18} " + " · ".join(parts))
    check = setup_check(queue)
    _print_setup_check(check)
    problem = root_check(queue)
    if problem:
        print(style(f"root: {problem}; a start needs --root NEWDIR or --adopt-root", "yellow"))
    if args.dry_run:
        print("\n".join(fit(line, WIDTH) for line in body(items, tui.glyphs(None))))
        return 0
    if check["command"] and args.setup:
        from hedonic.experiments.overlapping import codeseg_setup

        setup_args = (["--pairs", ",".join(check["missing_pairs"])] if check["missing_pairs"] else []) + \
                     (["--methods", ",".join(check["missing_methods"])] if check["missing_methods"] else [])
        if codeseg_setup.setup_main(setup_args) != 0:
            raise SystemExit("hedonic setup did not prepare everything; the queue was not started")
        check = setup_check(queue)
    if check["command"] and not args.no_setup_check:
        raise SystemExit(f"not started: run `{check['command']}` first (or pass --setup to run it now; "
                         "--no-setup-check starts anyway)")
    if problem and not args.adopt_root:
        raise SystemExit(f"not started: {problem}. Give the queue a fresh folder with --root NEWDIR, or pass "
                         "--adopt-root to write into this one.")
    rm._refuse_if_running(queue["name"])
    stamp_root(queue)
    entry = launch(queue)
    where = f"tmux session {entry['session']}" if entry["backend"] == "tmux" else f"background process (log {entry['log']})"
    print(style(f"  started queue {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  all runs: hedonic run list · this queue: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return rm.view(entry)
