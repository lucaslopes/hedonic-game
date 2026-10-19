"""What a queue is made of, as a tree a viewer can open and close.

Every run of a queue counts *units* (a detector run, an audit step, a search). A unit has coordinates, one value per
nested loop that produced it (pair, action policy, resolution, seed...), and is done or not. ``Family`` holds the units
of one kind with the loops that generate them, in the order the loops nest; ``RunModel`` holds a run's families.
``run_node`` and ``queue_node`` turn them into ``Node`` rows (queue > run > pair > loop > loop > unit), each with its
counts, an ETA and a text that says what the number counts (``130 detector runs = 2 action policies x 13 resolutions x
5 seeds``), so a count is never a bare figure.

A parent's count is always the sum of its children's, and its ETA the sum of theirs (the runs of a queue, the pairs of
a run and the units of a pair execute one after another; side-by-side workers are inside a unit's cost).

Nothing here reads files or draws: the state checks of ``runqueue`` fill the models, ``queuetui`` draws the nodes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

# Where an ETA comes from, weakest last: "measured" (this run's own timings), "estimate" (scaled from other work or a
# planning constant), "bound" (an estimate that leaves out work it knows about). "" is nothing left to wait for.
RANK = {"": 0, "measured": 1, "estimate": 2, "bound": 3}
Cost = tuple[float | None, str]  # seconds one unit still needs (None: no basis for an estimate) and where that comes from


@dataclass
class Dim:
    """One nested loop: what it iterates, in the singular and plural, how a value is written and what it changes."""

    name: str
    plural: str
    label: str = "{}"  # how a value reads in a row: "seed {}", "γ {}"
    about: str = ""  # what changing this loop changes in the experiment
    meaning: dict[str, str] = field(default_factory=dict)  # value -> what that value means


@dataclass
class Unit:
    coords: tuple[str, ...]  # one value per Dim of the family, outermost first
    done: bool
    seconds: float | None = None  # wall-clock of the unit when it ran, if known
    note: str = ""  # e.g. the F1 it reached
    running: bool = False
    took: float | None = None  # the wall-clock this unit added: its recorded seconds, divided by the units that ran beside it


@dataclass
class Factor:
    """A multiplier of a unit count that is not a loop of its own (rarely needed)."""

    plural: str
    count: int


@dataclass
class Family:
    key: str
    noun: str  # plural: "detector runs"
    what: str  # one sentence: what one unit is
    dims: list[Dim]
    units: list[Unit]
    order: str = ""  # how the loops nest when the runner executes them, outermost first
    cost: Callable[[Unit], Cost] | None = None  # seconds a not-yet-done unit still needs, for the ETA
    notes: list[str] = field(default_factory=list)  # lines for the detail of every node of this family's run
    parallel: int = 1  # units of this family the runner works on at once (the next ones to do are the ones running)


@dataclass
class RunModel:
    families: list[Family] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # lines for the run's detail text (why work is not reused, ...)
    stage: str = ""  # what the run is doing right now
    run_left: Cost = (None, "")  # the run's own estimate, when its runner publishes one


@dataclass
class Node:
    key: tuple
    label: str
    status: str = ""  # "done" | "running" | "partial" | "pending" | "failed" | "stopped" | "awaiting"
    done: int = 0
    total: int = 0
    noun: str = ""
    left: float = 0.0  # seconds left under this node, summed over the parts that have an estimate
    basis: str = ""  # the weakest basis among them (RANK)
    unknown: int = 0  # unfinished units under this node with no estimate at all
    took: float = 0.0  # seconds the finished units under this node took (their recorded time, read from the files)
    took_missing: int = 0  # finished units under this node with no recorded time
    detail: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)
    strip: list[bool] = field(default_factory=list)  # one mark per unit, for the node whose children are units
    sub: tuple[float, float, str] | None = None  # progress inside one unfinished unit: (done, total, what), drawn in its bar
    depth: int = 0
    log: bool = False  # a record under a unit (a setting tried), not a unit: no bar, no count, never marked done or pending
    note: str = ""  # for a log row, the text of its unit column


# --------------------------------------------------------------------------- words
def _public(pair: str) -> str:
    """A pair as users read it: the stable ``top5000`` ID is shown as ``top5k``."""
    return f"{pair[:-7]}top5k" if str(pair).endswith(":top5000") else str(pair)


def span(values: list[str], limit: int = 8) -> str:
    """``a, b, c`` for a few values, ``a, b, c ... z`` for many."""
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:4])} ... {values[-1]}"


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(max(0.0, seconds)))
    if seconds >= 2 * 86400:
        return f"{seconds // 86400}d {seconds % 86400 // 3600:02d}h"
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60}:{seconds % 60:02d}"


def fmt_took(seconds: float) -> str:
    """A duration that already happened: tenths of a second under ten seconds, whole seconds under a minute."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    return fmt_seconds(seconds)


def took_text(node: Node) -> str:
    """How long a node's finished work took (``≥`` when some finished unit has no recorded time), or nothing if unknown."""
    if not node.took:
        return ""
    return ("≥" if node.took_missing else "") + fmt_took(node.took)


def eta_text(node: Node) -> str:
    """The ETA column: the time left, ``~`` for an estimate, ``>=`` when part of the work has no estimate, ``?`` for none.
    A finished node says ``done`` and how long it took, read from the files even when the run skipped it as already done."""
    if node.total and node.done >= node.total:
        took = took_text(node)
        return f"done {took}" if took else "done"
    if node.unknown and not node.left:
        return "?"
    mark = "≥" if node.unknown or node.basis == "bound" else "~" if node.basis == "estimate" else ""
    return f"{mark}{fmt_seconds(node.left)}"


def clock(seconds: float, now: float | None = None) -> str:
    """When ``seconds`` from now falls on the wall clock: ``Fri 04:12``, with the date once it is past tomorrow."""
    now = time.time() if now is None else now
    when = time.localtime(now + seconds)
    if seconds >= 5 * 86400:
        return time.strftime("%b %d %H:%M", when)
    return time.strftime("%a %H:%M", when)


def singular(noun: str) -> str:
    """``searches`` -> ``search``, ``detector runs`` -> ``detector run``."""
    return noun[:-2] if noun.endswith("ches") else noun[:-1] if noun.endswith("s") else noun


def distinct(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


# --------------------------------------------------------------------------- explaining a count
def explain(family: Family, level: int, units: list[Unit]) -> list[str]:
    """What the loops from ``level`` inward count, and how the node's total comes from them."""
    inner = list(range(level, len(family.dims)))
    seen = {i: distinct([u.coords[i] for u in units if len(u.coords) > i]) for i in inner}
    sizes = [len(seen[i]) for i in inner]
    product = 1
    for size in sizes:
        product *= size
    lines = []
    if len(units) == 1:
        lines.append(f"1 {singular(family.noun)}")
    elif sizes and product == len(units):
        lines.append(f"{len(units):,} {family.noun} = " + " x ".join(
            f"{len(seen[i])} {family.dims[i].plural if len(seen[i]) != 1 else family.dims[i].name}" for i in inner))
    elif sizes:
        lines.append(f"{len(units):,} {family.noun} over " + " x ".join(
            f"{len(seen[i])} {family.dims[i].plural}" for i in inner) + " (not every combination is planned)")
    else:
        lines.append(f"{len(units):,} {family.noun}")
    for i in inner:
        dim = family.dims[i]
        values = seen[i]
        head = f"  {dim.plural} ({len(values)}): {span(values)}"
        lines.append(head)
        if dim.about:
            lines.append(f"      {dim.about}")
        for value in values[:6]:
            if value in dim.meaning:
                lines.append(f"      {value}: {dim.meaning[value]}")
    return lines


def _describe(family: Family, level: int, value: str, group: list[Unit], done: int) -> list[str]:
    dim = family.dims[level]
    lines = [f"{dim.name}: {value}"]
    if value in dim.meaning:
        lines.append(f"  {dim.meaning[value]}")
    lines.append(f"{done:,} of {len(group):,} {family.noun} done")
    if level + 1 < len(family.dims):
        lines += ["inside it:", *["  " + line for line in explain(family, level + 1, group)]]
    else:
        unit = group[0]
        lines.append("  " + ("done" if unit.done else "running now" if unit.running else "not yet")
                     + (f" · {unit.seconds:.1f} s" if unit.seconds is not None else "") + (f" · {unit.note}" if unit.note else ""))
        if unit.done and unit.took is not None and unit.seconds is not None and abs(unit.took - unit.seconds) > 0.05:
            lines.append(f"  its share of the wall-clock: {fmt_took(unit.took)} (side-by-side workers split the time)")
    if family.order:
        lines.append(f"runner order, outermost first: {family.order}")
    return lines


# --------------------------------------------------------------------------- what is being worked on
def mark_running(families: list[Family]) -> None:
    """Flag the units a live run is working on: the next ``parallel`` unfinished units, in the order the runner takes them.

    A builder that knows better (the benchmark reads its running unit from the worker's progress file, the spectrum
    audit from the stage line) flags its own, and then nothing is added. Between families the one working is the one
    whose first unfinished unit belongs to the earliest pair (a pair is audited before its detector runs, and pairs go
    one after another); the units flagged are those of that pair only, since a pair's graph is loaded once.
    """
    if any(unit.running for family in families for unit in family.units):
        return
    pairs = {pair: place for place, pair in enumerate(dict.fromkeys(u.coords[0] for f in families for u in f.units))}
    best = None
    for index, family in enumerate(families):
        first = next((unit for unit in family.units if not unit.done), None)
        if first is not None and (best is None or (pairs[first.coords[0]], index) < best[0]):
            best = ((pairs[first.coords[0]], index), family, first)
    if best is None:
        return
    _, family, first = best
    taken = 0
    for unit in family.units:
        if not unit.done and unit.coords[0] == first.coords[0]:
            unit.running = True
            taken += 1
            if taken >= max(1, family.parallel):
                break


# --------------------------------------------------------------------------- the tree of one family
def _status(done: int, total: int, running: bool) -> str:
    """Done, running (a unit under it is being worked on right now), partial (some done, none being worked on) or pending."""
    return "done" if total and done == total else "running" if running else "partial" if done else "pending"


def _aggregate(node: Node, kids: list[Node]) -> None:
    node.done = sum(k.done for k in kids)
    node.total = sum(k.total for k in kids)
    node.left = sum(k.left for k in kids)
    node.unknown = sum(k.unknown for k in kids)
    node.took = sum(k.took for k in kids)
    node.took_missing = sum(k.took_missing for k in kids)
    unfinished = [k for k in kids if k.done < k.total]
    node.basis = max((k.basis for k in unfinished), key=RANK.__getitem__, default="")
    node.status = _status(node.done, node.total, any(k.status == "running" for k in kids))


def family_nodes(family: Family, units: list[Unit], level: int = 0, key: tuple = ()) -> list[Node]:
    """The nodes one level down, from the units that share every coordinate above ``level``."""
    groups: dict[str, list[Unit]] = {}
    for unit in units:
        groups.setdefault(unit.coords[level], []).append(unit)
    dim = family.dims[level]
    nodes = []
    for value, group in groups.items():
        node = Node(key=key + (value,), label=dim.label.format(value), noun=family.noun)
        if level == len(family.dims) - 1:  # the unit itself
            unit = group[0]
            node.done, node.total = sum(u.done for u in group), len(group)
            cost = (0.0, "") if node.done == node.total else (family.cost(unit) if family.cost else (None, ""))
            if cost[0] is None and node.done < node.total:
                node.unknown = 1
            else:
                node.left, node.basis = (cost[0] or 0.0), cost[1]
            node.status = _status(node.done, node.total, any(u.running for u in group))
            node.took = sum(u.took or 0.0 for u in group if u.done)
            node.took_missing = sum(1 for u in group if u.done and u.took is None)
        else:
            node.children = family_nodes(family, group, level + 1, node.key)
            _aggregate(node, node.children)
            if level + 1 == len(family.dims) - 1:
                node.strip = [u.done for u in group]
        node.detail = _describe(family, level, value, group, node.done)
        nodes.append(node)
    return nodes


def _stamp(node: Node, depth: int) -> None:
    node.depth = depth
    for child in node.children:
        _stamp(child, depth + 1)


# --------------------------------------------------------------------------- one run
def run_node(index: int, name: str, title: str, status: str, model: RunModel | None, count: tuple[int, int, str] | None,
             about: str = "", eta: Cost = (None, ""), took: float | None = None) -> Node:
    """The node of one run. With a model its counts and ETA come from the units under it; without one, from ``count``
    (the run's own totals) and ``eta`` (what its runner publishes). ``took`` is the run's own clock, used for a finished
    run when no unit under it recorded a time."""
    node = Node(key=("run", name), label=f"{index}. {name}", status=status)
    node.detail = [line for line in (title, about) if line]
    if model is None or not any(f.units for f in model.families):
        done, total, noun = count or (0, 0, "")
        node.done, node.total, node.noun = done, total, noun
        if status == "done" or (total and done >= total):
            node.status = "done"
            node.took = took or 0.0
            node.took_missing = 0 if took else 1
        elif eta[0] is not None:
            node.left, node.basis = eta
        else:
            node.unknown = 1
        if model is not None:
            node.detail += model.notes
        return node
    if status == "running":
        mark_running(model.families)
    by_pair: dict[str, list[tuple[Family, Node]]] = {}
    for family in model.families:
        if family.units:
            for child in family_nodes(family, family.units, 0, ('run', name, family.key)):
                by_pair.setdefault(child.label, []).append((family, child))
    for label, entries in by_pair.items():
        if len(entries) == 1:
            node.children.append(entries[0][1])
            continue
        pair = Node(key=("run", name, "pair", label), label=label, noun="steps")
        for family, child in entries:  # several kinds of unit under one pair: a node per kind, in the order they run
            child.label = family.noun
            child.key = pair.key + (family.key,)
            pair.children.append(child)
        _aggregate(pair, pair.children)
        pair.detail = [f"{label}: " + " + ".join(f"{c.total:,} {c.label}" for c in pair.children)
                       + f" = {pair.total:,} steps ({pair.done:,} done)"]
        for family, child in entries:
            pair.detail += [f"{family.noun}: {family.what}"]
            pair.detail += ["  " + line for line in explain(family, 1, [u for u in family.units if u.coords[0] == label])]
        node.children.append(pair)
    _aggregate(node, node.children)
    node.status = "done" if node.total and node.done >= node.total else status
    if node.status == "done" and not node.took and took:  # no unit recorded a time: the run's own clock is what there is
        node.took, node.took_missing = took, 0
    node.noun = "steps" if len(model.families) > 1 else model.families[0].noun
    for child in node.children:
        _stamp(child, 1)
    if model.run_left[0] is not None and node.done < node.total:  # the runner's own estimate beats a sum of guesses
        node.left, node.basis, node.unknown = model.run_left[0], model.run_left[1], 0
    # the run's detail: what its total is made of first (that is what the number on its row means), then the context
    kinds = [f"{len(family.units):,} {family.noun}" for family in model.families]
    node.detail = [f"whole run: {node.total:,} steps" + (" = " + " + ".join(kinds) if len(kinds) > 1 else "")]
    for family in model.families:
        node.detail.append(f"- {family.noun}: {family.what}")
        node.detail += ["    " + line for line in explain(family, 0, family.units)]
        if family.order:
            node.detail.append(f"    the runner nests them (outermost first): {family.order}")
        node.detail += ["    " + line for line in family.notes]
    node.detail += [""] + [line for line in (title, about) if line] + model.notes
    if model.stage:
        node.detail.append("now: " + model.stage)
    return node


# --------------------------------------------------------------------------- the queue
def group_node(label: str, runs: list[Node]) -> Node:
    """One method's runs (its calibration and its experiments): runs finished of runs, the sum of their ETAs."""
    group = Node(key=("group", label), label=label, noun="runs", children=runs)
    group.done = sum(r.status == "done" for r in runs)
    group.total = len(runs)
    group.left = sum(r.left for r in runs)
    group.unknown = sum(r.unknown for r in runs)
    group.took = sum(r.took for r in runs)
    group.took_missing = sum(r.took_missing for r in runs if r.status == "done")
    group.basis = max((r.basis for r in runs if r.status != "done"), key=RANK.__getitem__, default="")
    states = {r.status for r in runs}
    group.status = ("done" if states == {"done"} else "running" if "running" in states
                    else "failed" if "failed" in states else "partial" if states & {"done", "partial", "stopped"} else "pending")
    group.detail = [f"{label}: {len(runs)} run{'s' if len(runs) != 1 else ''}, one after another"]
    group.detail += [f"  {r.label}: {r.status} · " + ("took " + (took_text(r) or "(no time recorded)") if r.status == "done"
                                                       else f"ETA {eta_text(r)}") for r in runs]
    return group


def queue_node(name: str, runs: list[Node], status: str = "", file: str = "", order: list[str] | None = None,
               groups: list[str] | None = None) -> Node:
    """The root: its count is runs finished of runs, its ETA the sum of the runs' (they execute one after another).

    ``order`` is the queue's pairs when it runs pair by pair: the runs then take turns on each pair in that order.
    ``groups`` names each run's method; when there is more than one, the runs sit under a node per method, in the
    order the methods first appear (method > experiment > pair > ...)."""
    root = Node(key=("queue",), label=f"queue {name}", status=status, noun="runs", depth=0)
    root.children = runs
    if groups and len(set(groups)) > 1:
        by: dict[str, list[Node]] = {}
        for run, group in zip(runs, groups):
            by.setdefault(group, []).append(run)
        root.children = [group_node(label, members) for label, members in by.items()]
    for child in root.children:
        _stamp(child, 1)
    root.done = sum(r.status == "done" for r in runs)
    root.total = len(runs)
    root.left = sum(r.left for r in runs)
    root.unknown = sum(r.unknown for r in runs)
    root.took = sum(r.took for r in runs)
    root.took_missing = sum(r.took_missing for r in runs if r.status == "done")
    root.basis = max((r.basis for r in runs if r.status != "done"), key=RANK.__getitem__, default="")
    how = ([f"{len(runs)} runs, each with every worker. The runs that go by pair take turns on one pair at a time: the whole "
            "block on the first pair, then on the next, in this order:", "  " + " > ".join(_public(o) for o in order)]
           if order else [f"{len(runs)} runs execute one after another, each with every worker."])
    root.detail = [f"queue {name}" + (f" · {file}" if file else ""), *how,
                   "The queue's ETA is the sum of its runs' ETAs, a run's is the sum of its pairs', a pair's is the sum of"
                   " its units'.",
                   "ETA marks: 17:33 measured · ~17:33 estimated from other work or a planning constant · "
                   "≥17:33 only part of the work has an estimate · ? no estimate at all.", ""]
    for number, run in enumerate(runs, 1):
        root.detail.append(f"  {number}. {run.label.split('. ', 1)[-1]}: {run.status} · "
                           + ("took " + (took_text(run) or "(no time recorded)") if run.status == "done" else f"ETA {eta_text(run)}"))
    return root
