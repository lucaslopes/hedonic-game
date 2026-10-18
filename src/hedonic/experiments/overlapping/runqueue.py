"""Sequences of durable runs: ``hedonic run queue FILE.toml``.

A queue file lists the runs in the order they should execute::

    name = "disjoint-queue"

    [[items]]
    name = "disjoint-paper"          # the run's name: the queue looks the run up by it
    run = ["disjoint", "--preset", "paper", "--name", "disjoint-paper", "--output-dir", "{root}/disjoint-paper", "--yes"]

``run`` holds the arguments after ``hedonic run``; ``--name`` repeats the item's name. ``{root}`` in an argument is the
queue's root folder: ``hedonic run queue FILE --root DIR``, by default ``<disjoint_output_root>/<queue name>`` with the
user config's ``[defaults] disjoint_output_root`` (else ``~/Databases/Hedonic/Disjoint``). A shipped queue is named
without its file: ``hedonic run queue disjoint`` reads ``queue-disjoint.toml`` from the package's configs.

Each item is an ordinary durable run (its own tmux session, its own row in ``hedonic run list``, its own
``attach``/``stop``/``resume``). The queue starts an item with ``--detach`` only after the previous one has
completed, so every item gets all configured workers. The queue is a run too: ``hedonic run status NAME``
shows every item.

Running the same queue again continues it. An item whose results are already on disk is skipped; the test is the
one its runner uses to reuse work (see ``data_state``). A stopped or interrupted item is relaunched with its own
arguments and resumes from its finished records.

Shortest first. ``schedule = "shortest"`` in the file (or ``hedonic run queue FILE --schedule shortest``) keeps one run
going at a time, as always, but chooses each next run at its turn: of the runs that are ready, the one with the smallest
estimate of what it has left (a running run's own ETA; a disjoint run's planned time for the graphs it still lacks), so
a long run no longer holds back the short ones behind it. A run is ready when every run it reads is finished: an item
waits for an earlier item whose output folder one of its arguments names, and for the items listed in its
``after = [...]``. A run with no estimate goes after the runs that have one. ``schedule = "file"`` (the default) is the
order of the file.
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Callable
import io
import json
import os
import shutil
import sys
import textwrap
import time
import tomllib
from pathlib import Path

from hedonic.experiments import config
from hedonic.experiments.overlapping import common
from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import runmanager as rm
from hedonic.experiments.overlapping import tui
from hedonic.experiments.overlapping.tui import fit, style

POLL_SECONDS = 30  # how often the queue checks the run it is waiting for
ACTIVE = ("running", "setup", "starting")
WIDTH = 80  # the queue view is read in a terminal this wide; longer lines are cut, never wrapped
COLOUR = {"running": "cyan", "setup": "magenta", "starting": "magenta", "completed": "green",
          "completed_with_failures": "yellow", "interrupted": "red", "stopped": "yellow", "failed": "red"}


def _check_arguments(run: list[str]) -> None:
    """Parse one item's arguments with the parser of its own front door. Nothing is loaded, computed or written."""
    from hedonic.experiments.CLI import RUN_ALIASES

    verb, args = run[0], run[1:]
    if verb not in RUN_ALIASES:
        raise ValueError(f"{verb!r} is not a run that a queue can start (hedonic 1 queues `disjoint` runs)")
    if verb == "disjoint":
        from hedonic.experiments.disjoint import front

        front.front_study(args)


def _absolute_config(run: list[str]) -> list[str]:
    """A ``--config`` that names a file (relative to the directory the queue starts in, or a config shipped with the
    package) is stored as its absolute path, so a resume from another directory still finds it. A name that is no
    file is left to its front door."""
    from hedonic.experiments import resources

    run = list(run)
    if "--config" in run and run.index("--config") + 1 < len(run):
        index = run.index("--config") + 1
        found = resources.config_path(run[index])
        if found.is_file():
            run[index] = str(found.resolve())
    return run


SCHEDULES = ("file", "shortest")


def dependencies(queue: dict) -> dict[str, set[str]]:
    """Item name -> the earlier items it must wait for: those named in its ``after``, and those whose output folder one
    of its arguments names."""
    root = queue.get("root") or ""
    deps: dict[str, set[str]] = {}
    earlier: list[tuple[str, list[str]]] = []
    for item in queue["items"]:
        mine = set(item.get("after") or [])
        for name, folders in earlier:
            if any(any(arg == f or f + os.sep in arg for f in folders) for arg in item["run"]):
                mine.add(name)
        deps[item["name"]] = mine
        folders = [f for f in (_option(item["run"], "--output-dir"), str(Path(root) / item["name"]) if root else None) if f]
        earlier.append((item["name"], [os.path.normpath(os.path.expanduser(f)) for f in folders]))
    return deps


def ready_items(items: list[dict], pending: list[int], deps: dict[str, set[str]]) -> list[int]:
    """The pending items that may start now: no item they depend on is still pending."""
    waiting = {items[i]["name"] for i in pending}
    return [i for i in pending if not deps.get(items[i]["name"], set()) & waiting]


def shortest_order(items: list[dict], pending: list[int], deps: dict[str, set[str]],
                   cost: Callable[[int], float | None]) -> list[int]:
    """The pending items in the order a shortest-first queue would take them if no estimate changed: each turn the ready
    item with the smallest ``cost`` (None, no estimate, goes after every item that has one; ties keep the file's order)."""
    left, order = list(pending), []
    while left:
        ready = ready_items(items, left, deps) or left[:1]
        index = min(ready, key=lambda i: (cost(i) is None, cost(i) or 0.0, i))
        order.append(index)
        left.remove(index)
    return order


def item_cost(item: dict) -> float | None:
    """Seconds an item should still take: 0 when its files are complete, a running run's own ETA, a disjoint run's
    planned time scaled to the graphs it still lacks; None when there is no basis."""
    status, run_entry = _status_of(item)
    if status == "completed":
        return 0.0
    if status in ACTIVE and run_entry is not None:
        seconds = rm.remaining_seconds(run_entry, rm.read_progress(run_entry), status)
        if seconds is not None:
            return float(seconds)
    if item["run"][0] == "disjoint":
        try:
            from hedonic.experiments.disjoint import front

            settings = front.front_study(item["run"][1:])["settings"]
            seconds = float(front.plan(settings)["seconds"])
        except Exception:  # noqa: BLE001 - without an estimate the item keeps its place in the file
            return None
        count = unit_counts(item["run"])
        if count and count[1]:
            seconds *= 1.0 - min(count[0], count[1]) / count[1]
        return seconds
    return None


def shipped_queues() -> list[str]:
    """The names of the queues shipped with the package (``disjoint`` for ``queue-disjoint.toml``)."""
    from hedonic.experiments import resources

    return [p.stem.removeprefix("queue-") for p in resources.shipped_configs()
            if p.name.startswith("queue-") and p.suffix == ".toml"]


def queue_file(name_or_path: str) -> Path:
    """The queue file a ``hedonic run queue`` argument names: a file as given, else a shipped queue by its name
    (``disjoint`` -> ``queue-disjoint.toml``) or by its file name."""
    from hedonic.experiments import resources

    raw = Path(str(name_or_path)).expanduser()
    if raw.is_file():
        return raw
    for candidate in (resources.config_path(str(name_or_path)), resources.config_path(f"queue-{name_or_path}")):
        if candidate.is_file():
            return candidate
    shipped = ", ".join(shipped_queues()) or "none"
    raise SystemExit(f"no queue file {name_or_path}; shipped queues: {shipped} (for example `hedonic run queue disjoint`)")


def default_root(name: str) -> Path:
    """Where a queue's items write by default: ``<disjoint_output_root>/<queue name>`` (``hedonic config set
    disjoint_output_root DIR``), else ``~/Databases/Hedonic/Disjoint`` (``HEDONIC_DISJOINT_OUTPUT_DIR``)."""
    from hedonic.experiments.overlapping import userconfig

    base = userconfig.defaults().get("disjoint_output_root")
    return (Path(os.path.expanduser(str(base))) if base else Path(config.DISJOINT_RUNS_ROOT)) / name


def load_queue(path: str, root: str | Path | None = None) -> dict:
    """Read and check a queue file; every problem is reported before anything starts.

    ``{root}`` in an item's arguments becomes ``root`` (default :func:`default_root`), made absolute."""
    file = queue_file(path)
    path = str(file)
    data = tomllib.loads(file.read_text())
    try:
        name = common.parse_name(str(data.get("name") or file.stem))
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from None
    if data.get("pairs") is not None or any(item.get("by_pair") for item in data.get("items") or []):
        raise SystemExit(f"{path}: pairs and by_pair belong to the overlap queues of hedonic 2; "
                         "hedonic 1 runs the items one after another")
    root_dir = Path(os.path.expanduser(str(root))).resolve() if root else default_root(name).resolve()
    items: list[dict] = []
    seen: set[str] = set()
    for number, item in enumerate(data.get("items") or [], 1):
        run = item.get("run")
        if not isinstance(run, list) or not run or not all(isinstance(arg, str) for arg in run):
            raise SystemExit(f"{path}: item {number} needs run = [\"VERB\", ...], for example run = [\"disjoint\"]")
        try:
            item_name = common.parse_name(str(item.get("name", "")))
        except ValueError as exc:
            raise SystemExit(f"{path}: item {number}: {exc}") from None
        if item_name in seen or item_name == name:
            raise SystemExit(f"{path}: the run name {item_name!r} is used more than once")
        run = _absolute_config([arg.replace("{root}", str(root_dir)) for arg in run])
        given = _option(run, "--name")
        if given is not None and given != item_name:
            raise SystemExit(f"{path}: item {item_name!r} runs as {given!r}; its --name must be the item's own name")
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
        after = item.get("after") or []
        if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
            raise SystemExit(f"{path}: item {item_name!r}: after must be a list of earlier item names")
        unknown = [a for a in after if a not in seen - {item_name}]
        if unknown:
            raise SystemExit(f"{path}: item {item_name!r} is after {', '.join(unknown)}, which no earlier item is named")
        if after:
            row["after"] = list(after)
        items.append(row)
    if not items:
        raise SystemExit(f"{path} lists no [[items]]")
    plan = str(data.get("schedule") or "file")
    if plan not in SCHEDULES:
        raise SystemExit(f"{path}: schedule must be one of {', '.join(SCHEDULES)}, not {plan!r}")
    return {"name": name, "items": items, "file": str(file.resolve()), "root": str(root_dir), "schedule": plan}


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
#   disjoint: every graph of its grid holds current rows of every method, and the table, figures and statistics were
#   written after them (``_disjoint_state``)
# A kind without such a rule keeps its registry status.
DATA_TTL = 60.0  # seconds a data check is reused; the queue view is redrawn every second
_DATA_CACHE: dict[str, tuple[float, tuple[str, str]]] = {}
# What each state check counted on its way: folder -> (finished, total, noun). Filled by the checks below, read by
# ``unit_counts`` after ``data_state`` (so the same TTL applies and nothing is scanned twice).
_COUNTS: dict[str, tuple[int, int, str]] = {}
# The breakdown behind those counts: folder -> [(group, label, flags)], one flag per unit (finished or not). For a
# disjoint run a group is a method and its flags one per number of communities K; ``report`` draws the groups as
# subsections and the flags as marks.
_PARTS: dict[str, list[tuple[str, str, list[bool]]]] = {}
# Why a run's recorded work does not count as done (rows of another build): folder -> reason.
_STALE: dict[str, str] = {}


def _note(folder: str | Path, done: int, total: int, noun: str,
          parts: list[tuple[str, str, list[bool]]] | None = None, stale: str = "") -> None:
    _COUNTS[str(folder)] = (int(done), int(total), noun)
    _PARTS[str(folder)] = parts or []
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


DISJOINT_OUTPUTS = ("plot_table.parquet", "figures", "stats/summary.md")


def _disjoint_state(args: list[str]) -> tuple[str, str]:
    """A disjoint SBM run: complete when every graph holds current rows of every method (each method's identity,
    `disjoint.store`) and the table, figures and statistics were written after them; partial when some rows are
    current (the run keeps them and computes the rest), missing otherwise. Rows of another build count as missing."""
    from hedonic.experiments.disjoint import front

    study = front.front_study(args)
    settings, folder = study["settings"], study["output"]
    status = front.store_status(settings, folder)
    total = int(front.plan(settings)["graphs"])
    done = int(status.get("complete_graphs") or 0)
    per_k = total // max(1, len(settings["communities"]))
    parts = []
    for method in settings["methods"]:
        table = (status.get("graphs_by_k") or {}).get(method) or {}
        parts.append((front.METHOD_LABELS.get(method, method), "K values (all graphs current)",
                      [int((table.get(k) or {}).get("current", 0)) >= per_k for k in settings["communities"]]))
    _note(folder, done, total, "graphs", parts, stale=status["refused"] or "")
    if status["refused"]:
        return "partial", folder  # a start refuses it with the reason; never judged complete
    store = Path(folder)
    finished = done == total and all((store / item).exists() for item in DISJOINT_OUTPUTS)
    if finished and (store / "manifest.json").stat().st_mtime > (store / "plot_table.parquet").stat().st_mtime:
        finished = False  # rows changed after the table: the run writes the table, figures and statistics again
    if finished:
        return "complete", folder
    return ("partial" if done or any(c.get("current") for c in status["graphs"].values()) else "missing"), folder


def _data_state(run: list[str]) -> tuple[str, str]:
    verb, args = run[0], run[1:]
    if verb == "disjoint":
        return _disjoint_state(args)
    return "unknown", ""  # no data check for this kind: the registry decides


def data_state(run: list[str], fresh: bool = False) -> tuple[str, str]:
    """What the data on disk says about one item: ("complete" | "partial" | "missing" | "unknown", its folder).

    ``fresh`` ignores the reading kept for ``DATA_TTL`` and reads the files again (the queue's two verifications).
    """
    key = json.dumps(run)
    cached = None if fresh else _DATA_CACHE.get(key)
    if cached and time.time() - cached[0] < DATA_TTL:
        return cached[1]
    try:  # the planners print their own errors; a bad argument is reported when the item starts, not while viewing
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = _data_state(run)
    except (SystemExit, Exception):  # noqa: BLE001
        result = ("unknown", "")
    _DATA_CACHE[key] = (time.time(), result)
    return result


def unit_counts(run: list[str]) -> tuple[int, int, str] | None:
    """(finished, total, noun) for one item from its files, or None when its kind has no such count or is not judged yet."""
    state, folder = data_state(run)
    if state == "unknown":
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
        elif status in ACTIVE:
            action = "wait: already running, the queue follows it"
        else:
            action = "resume from its finished records"
        rows.append((item["name"], "pending" if status == "new" else status, action))
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
    return {"pending": "still missing: starting", "stopped": "partly done: resuming"}.get(second, f"{second}: continuing")


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


def work(name: str, entry: dict, prog: dict, launch_item=_launch_item) -> int:
    """The queue worker: starts each item once the previous one has completed."""
    ppath = rm.progress_path(entry)
    prog.update(name=name, kind="queue", status="running", pid=os.getpid(), started=time.time(), updated=time.time(),
                current=None, file=entry.get("file", ""),
                items=[{"name": i["name"], "run": i["run"], "status": "pending"} for i in entry["items"]])
    rm.write_json(ppath, prog)
    print(style(f"hedonic run queue {name}", "bold") + f" · {len(entry['items'])} runs, one after another")
    print(style("Ctrl-C does not stop this queue. stop: hedonic run stop " + name, "dim"), flush=True)
    print(style("verify (1st): every run, from its files on disk, before anything starts", "bold"), flush=True)
    for row, (_, first, tally) in zip(prog["items"], verify(entry["items"])):
        row.update(first=first, first_count=tally)
        print(f"  {row['name']:<30} {first:<10} {tally}", flush=True)
    prog.update(verified=time.time())
    rm.write_json(ppath, prog)
    shortest = entry.get("schedule") == "shortest"
    deps = dependencies(entry) if shortest else {}
    if shortest:
        print(style("order: shortest estimate first, chosen again at each turn among the runs that are ready", "bold"),
              flush=True)
    pending = list(range(len(entry["items"])))
    while pending:
        index = pending[0]
        if shortest:
            ready = ready_items(entry["items"], pending, deps) or pending[:1]
            costs = {i: item_cost(entry["items"][i]) for i in ready}
            index = min(ready, key=lambda i: (costs[i] is None, costs[i] or 0.0, i))
            print(style(f"  next: {entry['items'][index]['name']} ("
                        + (f"~{qm.fmt_seconds(costs[index])}" if costs[index] is not None else "no estimate")
                        + f"; the shortest of {len(ready)} ready, {len(pending)} left)", "dim"), flush=True)
        pending.remove(index)
        row = prog["items"][index]
        prog.update(current=index, updated=time.time())
        ended = _run_item(row, prog, ppath, launch_item)
        if ended != "completed":
            reason = f"{row['name']} ended as {ended}; fix it, then continue with: hedonic run resume {name}"
            prog.update(status="stopped" if ended == "stopped" else "failed", reason=reason,
                        finished=time.time(), updated=time.time())
            rm.write_json(ppath, prog)
            print(style(f"\n  queue stopped: {reason}", "red"), flush=True)
            return 1
    prog.update(status="completed", finished=time.time(), updated=time.time(), current=None)
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
              "root": entry.get("root", ""), "output_dir": output, "schedule": entry.get("schedule") or "file",
              "created": time.time(),
              "session": f"hedonic-{name}" if use_tmux else "", "backend": "tmux" if use_tmux else "process",
              "log": str(Path(output) / "worker.log"),
              "command": f"hedonic run queue {entry.get('file', '')}" + (f" --root {entry['root']}" if entry.get("root") else "")
              + (" --schedule shortest" if entry.get("schedule") == "shortest" else "")}
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
        lines.append(style(f"  {_STOP_WORDS[status]} at {name} (item {position + 1} of {len(entry['items'])})",
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
    and a finished one its full total; a unit is whatever that run counts (graphs of the SBM grid).
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
                         + " " + style("no count", "dim"))
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
    """The whole queue on one screen: a row per group of work, a column per run, each cell the share of it done.

    It is the part of the live view that survives a short terminal: a few lines say, for every run and every group
    of its work (a block size of the SBM grid), whether it is finished, going, or still to do, with the run totals
    underneath.
    """
    G = G or tui.glyphs(None)
    columns = []
    for item in items:
        status, _ = _status_of(item)
        count = unit_counts(item["run"])
        groups: dict[str, list[bool]] = {}
        for group, label, flags in _PARTS.get(data_state(item["run"])[1], []):
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
    out = style(f"{'all':<{label_w}} ", "bold")
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
        if item["second"] != item["first"]:
            text += "  (differs)"
    else:
        text += "  ->  2nd: when its turn comes"
    return text


def report(items: list[dict], G: dict | None = None, full: bool = False, frame: str | None = None) -> list[str]:
    """Everything the queue has to do, as sections: a run, then its groups, then what is counted inside each group.

    A finished run is folded to one line per group. A run with work done or in progress opens up one more level
    (one mark per graph). ``full`` opens every run.
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
            lines.append("   " + style(shown, colour))
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
            counted = rows
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
    keys = ("first", "first_count", "second", "second_count")
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
    if status in _STOP_WORDS:
        lines += stop_lines(entry, prog, status)
    items = with_checks(with_labels(entry["items"], entry.get("file", "")), prog)
    lines += body(items, G, frame=frame)
    return "\n".join(fit(line, WIDTH) for line in lines)


_NODE_STATUS = {"completed": "done", "completed_with_failures": "done", "new": "pending", "stopped": "stopped",
                "interrupted": "stopped", "failed": "failed", "running": "running",
                "setup": "running", "starting": "running", "partial": "partial"}


def tree(entry: dict, prog: dict) -> qm.Node:
    """The queue as nodes for the interactive view: every run with its counts and an ETA.

    The runs are read as ``report`` reads them (the files on disk decide; the registry only where a kind has no rule),
    so the numbers of the tree are those of ``--report``. hedonic 1 has no unit model of a disjoint run, so a run's
    node carries its counts only.
    """
    items = with_checks(with_labels(entry["items"], entry.get("file", "")), prog)
    runs = []
    for number, item in enumerate(items, 1):
        status, run_entry = _status_of(item)
        if status == "stopped" and run_entry is None:  # work on disk that nothing is doing now: part done, not a stopped run
            status = "partial"
        folder = data_state(item["run"])[1]
        count = unit_counts(item["run"])
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
        node = qm.run_node(number, item["name"], item.get("title", ""), _NODE_STATUS.get(status, status), None, count,
                           item.get("about", ""), eta, took)
        for line in (_check_line(item), _STALE.get(folder, ""), item.get("note", ""),
                     *failure_lines(item["name"])):
            if line:
                node.detail.append(line)
        runs.append(node)
    return qm.queue_node(entry["name"], runs, rm.effective_status(entry, prog), entry.get("file", ""))


# --------------------------------------------------------------------------- what the queue needs on this machine
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
                             "<disjoint_output_root>/<queue name>, disjoint_output_root from `hedonic config` else "
                             + str(config.DISJOINT_RUNS_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="print what would run and what is skipped, then exit")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    parser.add_argument("--report", action="store_true",
                        help="print the whole queue as sections (run > group > graphs) from the files on disk, then exit")
    parser.add_argument("--full", action="store_true", help="with --report: open every run, finished ones too")
    parser.add_argument("--schedule", choices=SCHEDULES, default=None,
                        help="file: the runs in the file's order (default); shortest: at each turn the ready run with "
                             "the smallest estimate, so short work is not held back by a long run (overrides the "
                             "file's schedule key)")
    parser.add_argument("--adopt-root", action="store_true",
                        help="write into a --root that already holds runs this hedonic's queues did not start "
                             "(a record certifies only for the build that reads it; others are rerun in place)")
    args = parser.parse_args(argv)
    queue = load_queue(args.file, root=args.root)
    if args.schedule:
        queue["schedule"] = args.schedule
    print(style(f"queue {queue['name']} · {len(queue['items'])} runs", "bold") + style(f" · root {queue['root']}", "dim"))
    items = with_labels(queue["items"], queue.get("file", ""))
    if args.report:
        print("\n".join(fit(line, WIDTH) for line in body(items, tui.glyphs(None), full=args.full)))
        return 0
    print(style("verify (1st): every run, from its files on disk", "bold"))
    for number, (name, status, action) in enumerate(plan(queue), 1):
        print(f"  {number}. {name:<28} {status:<12} → {action}")
    print(style("  each run is read again at its turn (2nd), so data that appeared or vanished since is caught", "dim"))
    if queue.get("schedule") == "shortest":
        costs = {i: item_cost(item) for i, item in enumerate(queue["items"])}
        print(style("order: shortest estimate first — chosen again at each turn; with today's estimates:", "bold"))
        for number, index in enumerate(shortest_order(queue["items"], list(costs), dependencies(queue), costs.get), 1):
            cost = costs[index]
            print(f"  {number}. {queue['items'][index]['name']}  "
                  + ("done" if cost == 0 else f"~{qm.fmt_seconds(cost)}" if cost is not None else "no estimate"))
    problem = root_check(queue)
    if problem:
        print(style(f"root: {problem}; a start needs --root NEWDIR or --adopt-root", "yellow"))
    if args.dry_run:
        print("\n".join(fit(line, WIDTH) for line in body(items, tui.glyphs(None))))
        return 0
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
