"""The run models of a queue, one builder per kind of run (see ``queuemodel``).

Each builder takes what the queue's state check has already read from the run's files and returns a ``RunModel``:
the units of the run with the loops that produced them, and the cost of a unit still to do, so every row of the
viewer can say what its number counts and how long what is left should take. Nothing here touches a graph.

Costs come from, in this order: the run's own finished units in the same cell ("measured"), other finished units of
the same run or earlier records of the same pair ("estimate"), the same work on other pairs scaled by the number of
edges ("estimate"), a planning constant ("estimate"). A unit with none of these has no cost and the viewer says so.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Callable

from hedonic.experiments.overlapping import queuemodel as qm
from hedonic.experiments.overlapping import quickstart as qs


def pair_label(target: str) -> str:
    """Return a readable pair label; the top-5,000 SNAP variant is displayed as ``top5k``."""
    known = qs.NETWORKS.get(target)
    if known is None:  # a target the catalogue does not know (a test, a new network): shown as it is named
        return target.replace(":", " · ") or "all runs"
    return known[0] if ":" in target else f"{known[0]} · all"


def whole_edges(target: str) -> int | None:
    """Edges of the whole network behind a pair (the covered subgraph is smaller; only ratios between pairs are used)."""
    known = qs.NETWORKS.get(target)
    return int(known[2]) if known else None


def scaled(target: str, measured: dict[str, float]) -> float | None:
    """What ``target`` should cost, from pairs that were measured: their seconds per edge (same cover when any) times its edges."""
    edges = whole_edges(target)
    if not edges:
        return None
    cover = target.partition(":")[2] or "all"
    rates = {t: seconds / whole_edges(t) for t, seconds in measured.items() if whole_edges(t) and seconds > 0}
    same = [rate for t, rate in rates.items() if (t.partition(":")[2] or "all") == cover]
    pool = same or list(rates.values())
    return statistics.fmean(pool) * edges if pool else None


# --------------------------------------------------------------------------- the spectrum
POLICY_MEANING = {
    "fixed labels": "a vertex may only join a community that already exists (the labels of the supplied cover)",
    "open labels": "a vertex may also leave for a new, empty label, so it can end up in a community of its own",
}
_CAPACITY = re.compile(r"\[capacity\] (\S+): (spectrum audit|detector runs): (\d+) worker")
_STAGE_ETA = re.compile(r"ETA ~(?:(\d+):)?(\d+):(\d{2})")
_AUDIT_STAGE = re.compile(r"^auditing (\S+)/(\S+) · .*?(\d+)/(\d+) resolutions")


def capacity_workers(folder: Path) -> dict[tuple[str, str], int]:
    """(pair job, "detector runs" | "spectrum audit") -> the workers the study's capacity planning gave it, last line wins."""
    found: dict[tuple[str, str], int] = {}
    try:
        text = (folder / "worker.log").read_text(errors="replace")
    except OSError:
        return found
    for match in _CAPACITY.finditer(text):
        found[match[1], match[2]] = int(match[3])
    return found


def stage_eta(stage: str) -> float | None:
    """Seconds in the ``ETA ~h:mm:ss`` of a worker stage line."""
    match = _STAGE_ETA.search(stage or "")
    return None if not match else int(match[1] or 0) * 3600 + int(match[2]) * 60 + int(match[3])


def spectrum(rows: list[dict], *, folder: Path, live: dict, options: dict, audit_done: dict[str, int],
             audit_seconds: dict[str, float], unit_finished: Callable[[dict], bool]) -> qm.RunModel:
    """The ground-truth spectrum: per pair, an exact audit of the supplied cover, then seeded detector runs from it.

    ``audit_done`` is how many audit steps each pair (job ``dataset-cover``) has finished and ``audit_seconds`` how
    long the audits that finished took (the basis for the pairs still to audit).
    """
    policies = list(options.get("isolation_policies") or sorted({str(r.get("policy")) for r in rows}))
    shown = {p: p.replace("_", " ") for p in policies}
    grid = [float(g) for g in options.get("audit_resolutions") or []]
    jobs = list(dict.fromkeys(str(r.get("job", "")) for r in rows if r.get("job")))
    targets = {job: (f"{r['dataset']}:{r['cover']}" if r.get("cover") != "all" else str(r["dataset"]))
               for r in rows for job in [str(r.get("job"))]}
    labels = {job: pair_label(target) for job, target in targets.items()}
    workers = capacity_workers(folder)
    default_workers = int(options.get("workers") or 1)

    pair_dim = qm.Dim("pair", "pairs", about="a SNAP network with one of its ground-truth covers ('all' communities, or the "
                      "'top5k' largest); the graph is the part of the network the cover mentions")
    policy_dim = qm.Dim("action policy", "action policies", meaning=POLICY_MEANING,
                        about="which moves a vertex may consider when it looks for its best response")
    gamma_dim = qm.Dim("resolution", "resolutions", "γ {}",
                       about="γ, the resolution of the Constant Potts Model: the higher it is, the more the game prefers small, "
                             "dense communities")
    seed_dim = qm.Dim("seed", "seeds", "seed {}", about="the detector's random seed (the order vertices are visited in)")

    # detector runs ------------------------------------------------------------------------------------------------
    by_label = {labels[j]: targets[j] for j in jobs}
    side_by_side = {labels[j]: max(1, workers.get((j, "detector runs"), default_workers)) for j in jobs}
    units, seen = [], {}
    for r in rows:
        job = str(r.get("job"))
        f1 = r.get("f1")
        finished = unit_finished(r)
        seconds = r.get("seconds") if finished and isinstance(r.get("seconds"), (int, float)) else None
        units.append(qm.Unit((labels[job], shown.get(str(r.get("policy")), str(r.get("policy"))), f"{float(r.get('gamma', 0)):g}",
                              str(r.get("seed"))), finished, seconds, f"F1 {f1:.3f}" if isinstance(f1, (int, float)) else "",
                             (r.get("status") == "running"),
                             seconds / side_by_side[labels[job]] if seconds is not None else None))
        if finished and seconds is not None:
            seen.setdefault(targets[job], []).append(float(seconds))
    # seconds one detector run takes (alone in its worker), and how many a pair runs side by side
    raw = {target: statistics.fmean(v) for target, v in seen.items()}

    def detector_cost(unit: qm.Unit) -> qm.Cost:
        target = by_label[unit.coords[0]]
        wide = side_by_side[unit.coords[0]]
        if target in raw:
            return raw[target] / wide, "measured"
        guess = scaled(target, raw)
        return (guess / wide, "estimate") if guess is not None else (None, "")

    detector = qm.Family(
        "detector", "detector runs",
        "one seeded run of local moving started from the pair's exact ground-truth cover, scored against it",
        [pair_dim, policy_dim, gamma_dim, seed_dim], units, "pair > action policy > resolution > seed", detector_cost,
        [f"{len(policies)} action policies x {len({u.coords[2] for u in units})} resolutions x "
         f"{len({u.coords[3] for u in units})} seeds per pair; the runs of a pair go side by side "
         "(" + ", ".join(f"{label}: {n}" for label, n in side_by_side.items()) + " workers)",
         "a finished run's time is its recorded seconds divided by the runs beside it: its share of the wall-clock"],
        parallel=max(side_by_side.values(), default=1))

    # exact audit ---------------------------------------------------------------------------------------------------
    tuned = options.get("tuned_resolutions") or {}  # a pair's calibrated resolution joins its grid (`--tuned`)
    pair_of = {str(r.get("job")): f"{r.get('dataset')}/{r.get('cover')}" for r in rows if r.get("job")}

    def job_steps(job: str) -> list[tuple[str, str]]:
        gamma = tuned.get(pair_of.get(job, ""))
        gammas = grid if gamma is None or gamma in grid else sorted([*grid, float(gamma)])
        return [(shown[p], f"{g:g}") for p in policies for g in gammas]

    steps = job_steps(jobs[0]) if jobs else [(shown[p], f"{g:g}") for p in policies for g in grid]
    stage = str(live.get("stage") or "")
    running = _AUDIT_STAGE.match(stage)
    running_job = f"{running[1]}-{running[2]}" if running else ""
    eta_left = stage_eta(stage)
    audit_units: list[qm.Unit] = []
    audit_steps = len(steps)
    pool = max(1, workers.get((running_job, "spectrum audit"), default_workers)) if running_job else 1
    per_job = {job: len(job_steps(job)) for job in jobs}
    for job in jobs:
        done = audit_done.get(job, 0)
        steps_j = per_job[job]
        # a finished audit's recorded runtime is spread evenly over its steps (the pool does not time them one by one)
        share = audit_seconds[job] / steps_j if job in audit_seconds and steps_j and done >= steps_j else None
        for index, (policy, gamma) in enumerate(job_steps(job)):
            audit_units.append(qm.Unit((labels[job], policy, gamma), index < done, None, "",
                                       job == running_job and done <= index < done + pool,
                                       share if index < done else None))
    measured_seconds = {targets[j]: s for j, s in audit_seconds.items() if j in targets and s > 0}

    def audit_cost(unit: qm.Unit) -> qm.Cost:
        target = by_label[unit.coords[0]]
        job = next(j for j in jobs if labels[j] == unit.coords[0])
        if job == running_job and eta_left is not None:
            todo = max(1, per_job[job] - audit_done.get(job, 0))
            return eta_left / todo, "measured"
        whole = scaled(target, measured_seconds)
        return (whole / per_job[job], "estimate") if whole is not None and per_job[job] else (None, "")

    audit = qm.Family(
        "audit", "audit steps",
        "an exact stability check of the supplied cover at one resolution under one action policy: for every vertex, is "
        "its current set of communities a best response?",
        [pair_dim, policy_dim, qm.Dim("resolution", "resolutions", "γ {}", about="the audit grid: every resolution at which the "
                                      "cover is checked")], audit_units, "pair > action policy > resolution (a pool of "
        f"{default_workers} workers; the pool finishes steps roughly in this order, so which steps are done is approximate)",
        audit_cost, ["the audit of a pair runs before that pair's detector runs",
                     "a finished step's time is its pair's recorded audit runtime divided by the steps (they are not timed one by one)"],
        parallel=default_workers) if audit_units else None

    notes = [f"A pair first gets its exact audit ({audit_steps} steps = {len(policies)} action policies x {len(grid)} "
             "resolutions" + (", plus the pair's calibrated resolution under each policy" if tuned else "")
             + "), then its detector runs; pairs go one after another.",
             "The audit measures how stable the supplied cover is; the detector runs start from it and measure how far "
             "local moving drifts away."] if audit_units else []
    return qm.RunModel([f for f in (audit, detector) if f is not None], notes, stage)


# --------------------------------------------------------------------------- the start-quality curve
LEVEL_MEANING = {"random": "a fully random start that keeps each vertex's number of labels",
                 "0": "the reference itself (its cap projection when a vertex is above the cap): no corruption"}


def record_target(record: dict) -> str:
    """The target a stored start-quality record belongs to, spelled as the plan names it: the ``all`` cover is the
    network alone (``livejournal``), any other cover is ``livejournal:top5000``. A record keeps the two apart in its
    ``network`` and ``cover`` fields; reading ``network`` alone filed the cheap top5000 records under the ``all`` pair."""
    network, cover = str(record.get("network")), record.get("cover")
    return network if cover in (None, "", "all") or ":" in network else f"{network}:{cover}"


def startcurve(plan: dict, records: dict[str, dict], keys: list[dict]) -> qm.RunModel:
    """The start-quality curve: Hedonic from the ground-truth cover corrupted by a share of incidence switches, per pair."""
    workers = int(plan.get("workers") or 1)
    settings = plan.get("settings") or {}
    pair_dim = qm.Dim("pair", "pairs", about="a network with one of its covers; the start is that cover, the graph its covered subgraph")
    level_dim = qm.Dim("corruption level", "corruption levels", "corruption {}",
                       about="the share of the reference's label incidences replaced by degree-preserving switches: "
                             "0 = the reference, 1 = fully switched, 'random' = a random start", meaning=LEVEL_MEANING)
    seed_dim = qm.Dim("seed", "seeds", "seed {}", about="detector seed; it also draws the corrupted start")
    units = []
    for c in keys:
        record = records.get(c["key"])
        level = c["level"] if isinstance(c["level"], str) else format(c["level"], "g")
        note = ""
        if record and isinstance(record.get("start_symmetric_f1"), (int, float)) and isinstance(record.get("final_symmetric_f1"), (int, float)):
            note = f"F1 {record['start_symmetric_f1']:.3f} -> {record['final_symmetric_f1']:.3f}"
        seconds = record.get("seconds") if record and isinstance(record.get("seconds"), (int, float)) else None
        wide = max(1, int((record or {}).get("workers") or workers))
        units.append(qm.Unit((pair_label(c["network"]), level, str(c["seed"])), record is not None, seconds, note, False,
                             seconds / wide if seconds is not None else None))
    # seconds per condition: this run's records of the pair and level, else the pair's other levels or earlier settings
    by_cell: dict[tuple[str, str], list[float]] = {}
    by_pair: dict[str, list[float]] = {}
    done_cells = {u.coords[:2] for u in units if u.done}
    for record in records.values():
        seconds = record.get("seconds")
        if not isinstance(seconds, (int, float)):
            continue
        label = pair_label(record_target(record))
        level = str(record.get("level"))
        wide = max(1, int(record.get("workers") or workers))
        by_cell.setdefault((label, level), []).append(seconds / wide)
        by_pair.setdefault(label, []).append(seconds / wide)
    pair_mean = {label: statistics.fmean(v) for label, v in by_pair.items()}
    target_of = {pair_label(c["network"]): c["network"] for c in keys}
    target_mean = {target_of[label]: m for label, m in pair_mean.items() if label in target_of}

    def cost(unit: qm.Unit) -> qm.Cost:
        pair, level = unit.coords[0], unit.coords[1]
        cell = by_cell.get((pair, level))
        if cell:
            return statistics.fmean(cell), "measured" if unit.coords[:2] in done_cells else "estimate"
        if pair in pair_mean:
            return pair_mean[pair], "estimate"
        guess = scaled(target_of.get(pair, ""), target_mean)
        return (guess, "estimate") if guess is not None else (None, "")

    family = qm.Family(
        "conditions", "conditions",
        "one Hedonic run (local moving) from the ground-truth cover corrupted by one level, scored against the reference",
        [pair_dim, level_dim, seed_dim], units, "pair > corruption level > seed", cost,
        [f"{workers} conditions run side by side; a condition's cost is its recorded seconds divided by that"],
        parallel=workers)
    notes = []
    for network, setting in settings.items():
        if isinstance(setting, dict) and setting.get("cap") is not None:
            notes.append(f"{pair_label(network)}: Hedonic at cap {setting['cap']} and gamma {setting['gamma']:.4g}"
                         + (f" ({setting['source']})" if setting.get("source") else ""))
    return qm.RunModel([family], notes)


# --------------------------------------------------------------------------- the benchmark
AUDIT_FRESH_SECONDS = 120.0  # a progress report older than this is not trusted (the audit or its worker is gone)


def benchmark(cfg, units: list[tuple[str, int, object]], finished: list[bool], progress: dict,
              now: float | None = None, history: dict | None = None) -> qm.RunModel:
    """The benchmark: one detector run per planned (pair, method, seed), executed in ``cfg.order``.

    ``units`` is the run's plan (``quickstart.build_plan``), so a deterministic method has one unit per pair (its lowest
    seed) unless the run asks for every seed, and nothing is priced for seeds the run will not execute.

    A finished unit's recorded ``seconds`` is the whole unit (detection, the robustness audit of its cover, scoring), so a
    cell with finished units is priced from them. A cell with none is priced by the planning constant, which is the
    detection alone: that stays a lower bound while the audit is on, because how long an audit takes depends on the cover
    (BigCLAM's cover of YouTube/top5k puts one vertex in 1,825 communities and its audit has run for hours, where every
    other cover of that pair took under a minute). A unit already running past its estimate has no basis for what is
    left, and the other seeds of its cell are priced at least as long as it has taken so far.

    ``history`` (``runqueue._benchmark_history``) holds what the finished units on disk took, for every pair of the run,
    including those a by-pair queue ran in earlier steps (their progress rows are gone). A cell this run has not timed is
    priced from its own units on disk, else from the same method on the other pairs, per edge of the graph the detectors
    saw (pairs with the same cover only: a top-5,000 subgraph says little about a whole network), and only then by the
    planning constant, which is for the whole graph and so far too long for a covered top-5,000 subgraph.
    """
    now = time.time() if now is None else now
    robust = bool(getattr(cfg, "robustness", False))
    pair_dim = qm.Dim("pair", "pairs", about="a network with one ground-truth cover; the detectors see the covered subgraph "
                      "and are scored against that cover")
    method_dim = qm.Dim("method", "methods", about="the community-detection method (Hedonic local moving at its calibrated settings)")
    seed_dim = qm.Dim("seed", "seeds", "seed {}", about="the method's random seed; a deterministic method runs once per pair, "
                      "at the lowest seed (the run's replication, see `hedonic run exp --help`)")
    measured: dict[tuple[str, str], list[float]] = {}
    whole: dict[tuple[str, str, str], float] = {}  # what a unit this run timed took: detection, audits and scoring
    for row in progress.get("plan") or []:
        if row.get("status") == "completed" and isinstance(row.get("seconds"), (int, float)):
            measured.setdefault((pair_label(str(row.get("network"))), str(row.get("label"))), []).append(float(row["seconds"]))
            whole[pair_label(str(row.get("network"))), str(row.get("label")), str(row.get("seed"))] = float(row["seconds"])
    running_rows = {(pair_label(str(r.get("network"))), str(r.get("label")), str(r.get("seed"))): r
                    for r in progress.get("plan") or [] if r.get("status") == "running"}
    # the detection time of every unit the worker knows (also those it found finished and skipped: its scan reads them)
    detection = {}
    for row in progress.get("plan") or []:
        seconds = row.get("detection_seconds")
        if isinstance(seconds, (int, float)):
            detection[pair_label(str(row.get("network"))), str(row.get("label")), str(row.get("seed"))] = float(seconds)
    expected: dict[tuple[str, str], float] = {}
    cells: dict[tuple[str, str], tuple[str, str]] = {}
    for net, _seed, method in units:
        expected[pair_label(net), method.label] = qs.expected_seconds(method, cfg, net)
        cells[pair_label(net), method.label] = (net, method.key)
    past = (history or {}).get("seconds") or {}
    sizes = (history or {}).get("edges") or {}
    unit_took = (history or {}).get("units") or {}
    audits = (history or {}).get("audits") or {}

    def from_disk(cell: tuple[str, str]) -> float | None:
        """What one unit of the cell should take from the finished units on disk: its own, else the same method's on the
        other pairs of the same cover per edge (edges of the graph the detectors saw when the pair has a record, else of
        the whole network)."""
        net, key = cells[cell]
        if past.get((net, key)):
            return statistics.fmean(past[net, key])
        cover = net.partition(":")[2] or "all"
        others = {t: statistics.fmean(v) for (t, k), v in past.items()
                  if k == key and v and t != net and (t.partition(":")[2] or "all") == cover}
        if not others:  # another cover is another kind of graph (a top-5,000 subgraph against a whole network)
            return None
        if net in sizes:
            rates = [seconds / sizes[t] for t, seconds in others.items() if sizes.get(t)]
            if rates:
                return statistics.fmean(rates) * sizes[net]
        return scaled(net, others)

    def estimate(cell: tuple[str, str]) -> qm.Cost:
        """What one unit of the cell takes: the mean of its finished units, else what the units on disk say (its own, or
        the same method on other pairs), else the planning constant (detection only)."""
        if measured.get(cell):
            return statistics.fmean(measured[cell]), "measured"
        seconds = from_disk(cell)
        if seconds is not None:
            return seconds, "estimate"
        return expected[cell], "bound" if robust else "estimate"

    def elapsed(key: tuple[str, str, str]) -> float | None:
        started = (running_rows.get(key) or {}).get("started")
        return max(0.0, now - float(started)) if isinstance(started, (int, float)) else None

    def profiles_left(key: tuple[str, str, str], info: dict) -> list[str]:
        """The audit profiles of a running unit after the one its audit reports (``cover``, then ``labelled`` for
        Hedonic). The report's key names the profile; a report without it (a worker from before it was kept) adds
        nothing, because the ledger cannot say: a unit's audits are written together once all of them are done."""
        from hedonic.experiments.overlapping import cover_audit as ca

        current = str(info.get("key") or "").rpartition("/")[2]
        if not current or key[:2] not in cells:
            return []
        net, method_key = cells[key[:2]]
        try:
            seed = int(key[2])
        except ValueError:
            return []
        profiles = [k.rpartition("/")[2] for k in ca.unit_keys(net, seed, method_key)]
        return profiles[profiles.index(current) + 1:] if current in profiles else []

    def audit_left(key: tuple[str, str, str]) -> float | None:
        """Seconds the exact audits of a running unit have left, from the progress it reported (None: unknown or stale).

        The report covers the profile under way (its policies included); each profile still to come is priced by what
        that profile took for the same method on this pair's finished seeds, else as long as the one under way."""
        info = (running_rows.get(key) or {}).get("audit")
        if not isinstance(info, dict) or not isinstance(info.get("eta_seconds"), (int, float)):
            return None
        updated = info.get("updated")
        age = max(0.0, now - float(updated)) if isinstance(updated, (int, float)) else 0.0
        if age > AUDIT_FRESH_SECONDS:
            return None
        eta = float(info["eta_seconds"])
        left = max(0.0, eta - age)
        later = profiles_left(key, info)
        if later:
            number, policies = int(info.get("number") or 1), max(1, int(info.get("policies") or 1))
            per_policy = (float(info.get("elapsed") or 0.0) + eta) / max(1, 1 + policies - number)
            net, method_key = cells[key[:2]]
            for profile in later:
                seen = audits.get((net, method_key, profile))
                left += statistics.fmean(seen) if seen else per_policy * policies
        return left

    def overdue(key: tuple[str, str, str]) -> float | None:
        """Seconds a running unit has taken when that is more than its cell's estimate and it can say no more, else None."""
        spent = elapsed(key)
        if audit_left(key) is not None:
            return None
        return spent if spent is not None and spent > (estimate(key[:2])[0] or 0.0) else None

    # a cell whose running unit has overrun: its other seeds are not expected to be quicker than that one; when the unit's
    # audit reports its progress the whole length is known and the other seeds are priced at it (an estimate: the covers differ)
    overran = {key[:2]: spent for key in running_rows if (spent := overdue(key)) is not None}
    projected = {}
    for key in running_rows:
        left, spent = audit_left(key), elapsed(key)
        if left is not None and spent is not None and spent + left > (estimate(key[:2])[0] or 0.0):
            projected[key[:2]] = spent + left
    made = []
    for (net, seed, method), done in zip(units, finished):
        key = (pair_label(net), method.label, str(seed))
        # a finished unit took its whole span (detection, the audits of its cover, scoring), not its detection alone
        seconds = (whole.get(key) or unit_took.get((net, seed, method.key)) or detection.get(key)) if done else None
        note = ""
        if key in running_rows and not done:
            spent, limit = elapsed(key), estimate(key[:2])[0]
            if spent is not None and key in detection:
                note = f"detection took {qm.fmt_seconds(detection[key])}; the exact audit of its saved cover has run {qm.fmt_seconds(spent - detection[key])}"
            elif spent is not None:
                note = f"running {qm.fmt_seconds(spent)}"
            info, left = running_rows[key].get("audit"), audit_left(key)
            if left is not None and isinstance(info, dict):
                note += (f"; it has audited {info.get('done', 0):,} of {info.get('total', 0):,} vertices under policy "
                         f"{info.get('number')} of {info.get('policies')}, about {qm.fmt_seconds(left)} left")
            if overdue(key) is not None:
                note += f", past its estimate of ~{qm.fmt_seconds(limit)}: how long the rest takes is unknown"
        made.append(qm.Unit(key, bool(done), seconds, note, key in running_rows, seconds))

    def cost(unit: qm.Unit) -> qm.Cost:
        cell = unit.coords[:2]
        seconds, basis = estimate(cell)
        if unit.coords in running_rows:
            left = audit_left(unit.coords)
            if left is not None:  # its audit reports its progress: the rest is extrapolated from the vertices done
                return left, "estimate"
            spent = elapsed(unit.coords)
            if spent is None:
                return seconds, basis
            return (seconds - spent, basis) if spent < seconds else (None, "")  # past its estimate: nothing to go on
        if cell in projected:  # a seed of a cell whose running seed knows its length: priced at it
            return max(seconds, projected[cell]), "estimate"
        if cell in overran:  # a seed of a cell whose first seed has overrun: at least as long as that one has taken
            return max(seconds, overran[cell]), "bound"
        return seconds, basis

    family = qm.Family(
        "units", "detector runs", "one run of one method on one pair's graph with one seed, scored against the cover",
        [pair_dim, method_dim, seed_dim], made, f"the runner nests {cfg.order.replace(',', ' > ')} (outermost first); this "
        "tree groups them pair > method > seed", cost,
        ["one run at a time (a run's time is a result)"]
        + (["a unit's time includes the exact robustness audit of its saved cover, which runs right after the detector; a "
            "method not yet timed on a pair is priced by its detection alone, so its ETA is a lower bound"]
           if robust else [])
        + ["a method not yet timed here is priced from its finished units on the other pairs of the same cover (per edge "
           "of the graph the detectors saw); with none, from its planning constant, an upper bound for covered subgraphs "
           "on detection",
           "a unit's exact audit reports the vertices it has done, so a running unit's remaining time is extrapolated from "
           "them (the vertices go through in a fixed pseudo-random order, so the rate so far is the rate of the rest) and the "
           "other seeds of its method are priced at its whole length",
           "a unit running past its estimate whose audit has reported nothing recently has no ETA (?); the other seeds of "
           "its method are priced at least as long as it has taken so far"])
    return qm.RunModel([family])


# --------------------------------------------------------------------------- the calibration
# --------------------------------------------------------------------------- the cover audit
PROFILE_MEANING = {"cover": "the saved scoring cover (every method)",
                   "labelled": "the exact labelled memberships Hedonic returned"}


def audit_seconds(record: dict) -> float | None:
    """What one audit record cost: loading its graph (when it did), building the state, and the exact audit."""
    parts = [record.get(key) for key in ("load_seconds", "state_seconds", "audit_seconds")]
    if not isinstance(parts[2], (int, float)):
        return None
    return float(sum(part for part in parts if isinstance(part, (int, float))))


def audit(plan: dict, records: dict[str, dict], progress: dict | None = None, now: float | None = None,
          finished: set[str] | None = None) -> qm.RunModel:
    """The cover audit (``hedonic run audit``): one exact audit per saved cover and profile, methods outermost.

    A finished audit took its recorded seconds. One still to do costs what the same method's audits of the same pair and
    profile took ("measured"), else the same method and profile on other pairs scaled by edges ("estimate"). The running
    audit costs what its progress report says is left while the report is fresh (``AUDIT_FRESH_SECONDS``). ``finished``
    is the keys the state check found done (a record of an earlier cover of a unit that ran again is not); by default
    every recorded key.
    """
    from hedonic.experiments.overlapping import cover_audit as ca

    now = time.time() if now is None else now
    current = (progress or {}).get("current") if (progress or {}).get("status") in ("running", "setup") else None
    fresh = bool(current) and now - float(current.get("updated") or 0.0) <= AUDIT_FRESH_SECONDS
    method_dim = qm.Dim("method", "methods", "{}", about="the detector whose saved covers are audited")
    pair_dim = qm.Dim("pair", "pairs", about="a network with one of its covers; the graph is its covered subgraph")
    seed_dim = qm.Dim("seed", "seeds", "seed {}", about="the detector seed of the audited cover")
    profile_dim = qm.Dim("profile", "profiles", "{}", about="which membership profile of the run is audited",
                         meaning=PROFILE_MEANING)
    units, cells, by_method = [], {}, {}
    for c in ca.conditions(plan):
        record = records.get(c["key"])
        if record is not None and finished is not None and c["key"] not in finished:
            record = None  # stale: it describes an earlier cover of its unit
        seconds = audit_seconds(record) if record else None
        note = ""
        if record and record.get("status") == "completed":
            block = (record.get("policies") or {}).get(ca.PRIMARY_POLICY) or {}
            if isinstance(block.get("stable_measure_log"), (int, float)):
                note = f"robust {block['stable_measure_log']:.3f}"
        elif record:
            note = str(record.get("status"))
        running = bool(current) and current.get("key") == c["key"] and record is None
        units.append(qm.Unit((method_label(c["method"]), pair_label(c["network"]), str(c["seed"]), c["profile"]),
                             record is not None, seconds, note, running, seconds))
        if seconds is not None:
            cells.setdefault((c["method"], c["network"], c["profile"]), []).append(seconds)
            by_method.setdefault((c["method"], c["profile"]), {}).setdefault(c["network"], []).append(seconds)
    key_of = {u.coords: c for u, c in zip(units, ca.conditions(plan))}

    def cost(unit: qm.Unit) -> qm.Cost:
        c = key_of[unit.coords]
        if unit.running and fresh and isinstance(current.get("eta_seconds"), (int, float)):
            age = now - float(current["updated"])
            return max(0.0, float(current["eta_seconds"]) - age), "estimate"
        cell = cells.get((c["method"], c["network"], c["profile"]))
        if cell:
            return statistics.fmean(cell), "measured"
        others = {net: statistics.fmean(v) for net, v in by_method.get((c["method"], c["profile"]), {}).items()}
        guess = scaled(c["network"], others)
        return (guess, "estimate") if guess is not None else (None, "")

    family = qm.Family(
        "audits", "audits",
        "the exact robustness of one saved cover under one profile: every vertex's stable resolutions, both action spaces",
        [method_dim, pair_dim, seed_dim, profile_dim], units, "method > pair > seed > profile", cost,
        [f"{plan.get('workers', 1)} forked vertex workers per audit; the audits run one at a time",
         "an audit costs what the same method's audits of the pair took, else of other pairs scaled by edges"])
    stage = str((progress or {}).get("stage") or "") if current else ""
    return qm.RunModel([family], [], stage)


SEARCHES_AT_ONCE = 4  # the calibration runs its four searches side by side


def calibrate(searches: list[dict], have: set[str], seconds: dict[str, float] | None = None) -> qm.RunModel:
    """``seconds`` is the wall-clock each finished search recorded (its key -> seconds)."""
    seconds = seconds or {}
    pair_dim = qm.Dim("pair", "pairs", about="a network with one cover; the setting found is used for that pair")
    phase_dim = qm.Dim("search", "searches", about="the detector phase whose resolution and cap are being tuned")
    units = [qm.Unit((pair_label(s["network"]), f"{s['phase']} search"), s["key"] in have, seconds.get(s["key"]), "", False,
                     seconds[s["key"]] / SEARCHES_AT_ONCE if s["key"] in have and s["key"] in seconds else None)
             for s in searches]
    family = qm.Family("searches", "searches",
                       "a bisection over the resolution and the membership cap, scored by F1 on seeds 0-4",
                       [pair_dim, phase_dim], units, "pair > search",
                       notes=[f"{SEARCHES_AT_ONCE} searches run side by side; a finished search's time is its recorded "
                              "wall-clock divided by that"], parallel=SEARCHES_AT_ONCE)
    return qm.RunModel([family])


# A search's evaluations are not known in advance (each result picks the next probe): these are the usual totals, so a
# search under way has a bar. Resolution only (--cap unlimited): the wide scan's 11 points and golden section up to 20,
# the clique-regime probe, the snap and four more confirmation seeds. A cap search alternates resolution and cap.
SEARCH_EVALS = {"unlimited": 26, "search": 40}


def calibration_evals(folder: Path, network: str) -> list[dict]:
    """Every evaluation of ``network`` in the calibration's ledger, oldest first (a search may reuse another's)."""
    out = []
    for path in sorted((folder / "ledger").glob("evals*.jsonl")):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("instance") == network and isinstance(record.get("config"), dict):
                out.append(record)
    return sorted(out, key=lambda r: str(r.get("finished", "")))


def decorate_calibration(node: qm.Node, plan: dict, searches: dict[str, dict]) -> None:
    """Count each Hedonic search in its evaluations (detector runs, each scored and audited) instead of as one unit,
    and list under it a row per setting it probed: resolution, cap when there is one, F1 over the seeds that ran it and
    the communities found; ★ marks the setting it chose. Parents count the evaluations under them."""
    folder = Path(plan["output"])
    labels = {pair_label(t): t for t in plan["networks"]}
    running = node.status == "running"
    expected = SEARCH_EVALS.get(plan.get("cap") or "search", 40)

    def visit(n: qm.Node) -> None:
        for child in n.children:
            visit(child)
        if n.children or len(n.key) < 2 or not str(n.key[-1]).endswith("search"):
            return
        target = labels.get(n.key[-2])
        if not target:
            return
        records = calibration_evals(folder, target)
        entry = searches.get(f"hedonic/{target}") or {}
        chosen = ((float(f"{float(entry['absolute_resolution']):.6g}"), int(entry["max_memberships"]))
                  if entry.get("absolute_resolution") is not None else None)
        groups: dict[tuple, list[dict]] = {}
        for record in records:
            config = record["config"]
            groups.setdefault((float(f"{float(config['gamma']):.6g}"), int(config["M"])), []).append(record)
        for (gamma, cap), group in sorted(groups.items()):
            ok = [r for r in group if r.get("status") == "completed" and isinstance(r.get("metrics"), dict)]
            f1 = statistics.fmean(r["metrics"]["f1"] for r in ok) if ok else None
            counts = [r["metrics"].get("predicted_community_count") for r in ok if r["metrics"].get("predicted_community_count")]
            first = next((r for r in ok if int(r["config"].get("seed", 0)) == 0), None)
            # the other probes ran seed 0 only: a confirmed setting shows its seed-0 F1 too, so the rows compare
            score = (f" · F1 {first['metrics']['f1']:.3f} seed 0, {f1:.3f} mean" if first is not None and len(ok) > 1
                     else f" · F1 {f1:.3f}" if f1 is not None else f" · {group[0].get('status', '?')}")
            label = (f"γ {gamma:.3g}" + (f" · cap {cap}" if cap > 0 else "") + score
                     + (f" · {int(statistics.median(counts)):,} communities" if counts else ""))
            seeds = sorted({int(r["config"].get("seed", 0)) for r in group})
            label += (f" · seed {seeds[0]}" if len(seeds) == 1 else f" · {len(seeds)} seeds") + (
                "  ★ chosen" if chosen == (gamma, cap) else "")
            row = qm.Node(key=n.key + ("probe", gamma, cap), label=label, depth=n.depth + 1, noun="evaluations",
                          done=len(group), total=len(group), status="done" if ok else "failed")
            row.took = sum(float(r.get(k) or 0.0) for r in group
                           for k in ("detection_seconds", "scoring_seconds", "audit_seconds"))
            row.detail = [f"γ = {gamma:.6g}" + (f", cap {cap}" if cap > 0 else ", no cap")]
            for r in group:
                m, lf = r.get("metrics") or {}, r.get("label_free") or {}
                row.detail.append(
                    f"  seed {r['config'].get('seed', 0)}: {r.get('status')}"
                    + (f" · F1 {m['f1']:.4f} · ONMI {m.get('onmi', 0):.4f} · {m.get('predicted_community_count', '?'):,} communities"
                       f" · max regret {lf.get('max_regret', float('nan')):.2g} · detection {float(r.get('detection_seconds') or 0):.1f} s"
                       if m else f" · {str(r.get('reason') or '')[:120]}"))
            n.children.append(row)
        done = len(records)
        spent = sum(row.took for row in n.children if row.done)
        if n.status == "done" and spent > n.took:  # its evaluations were reused from the ledger: show what they cost
            n.took = spent
            n.detail.append("its evaluations were reused from the ledger; the time shown is what they cost when they ran")
        n.noun = "evaluations"
        n.total = done if n.status == "done" else max(done + 1, expected)
        n.done = done if n.status != "done" else n.total
        if n.status != "done":
            if done and running:  # every unfinished search of a running calibration is under way
                n.status = "running"
            pace = search_pace(records)
            if pace is not None:
                n.left, n.basis, n.unknown = (n.total - n.done) * pace, "estimate", 0
            # what is left: the probe being evaluated (its gamma is known only once it is recorded) and the ones to come
            left = n.total - n.done
            under_way = n.status == "running"
            if under_way:
                row = qm.Node(key=n.key + ("probe", "now"), label=f"evaluation {done + 1}: under way (its γ is recorded when it ends)",
                              depth=n.depth + 1, noun="evaluations", done=0, total=1, status="running")
                if pace is not None:
                    since = time.time() - (search_last(records) or time.time())
                    row.left, row.basis = max(pace - since, 0.0), "estimate"
                else:
                    row.unknown = 1
                n.children.append(row)
                left -= 1
            if left > 0:
                first = done + 1 + under_way
                row = qm.Node(key=n.key + ("probe", "later"),
                              label=(f"evaluations {first}-{n.total}" if left > 1 else f"evaluation {first}")
                              + ": each γ is chosen from the results before it",
                              depth=n.depth + 1, noun="evaluations", done=0, total=left, status="pending")
                if pace is not None:
                    row.left, row.basis = left * pace, "estimate"
                else:
                    row.unknown = 1
                n.children.append(row)
            n.detail.append(f"{done} evaluations so far; a search like this one usually takes about {expected} "
                            "(each result chooses the next probe, so the total is not known in advance)"
                            + (f"; one every {pace:.0f} s so far" if pace is not None else ""))
        if entry:
            n.detail.append(f"chosen: γ {float(entry['absolute_resolution']):.6g}"
                            + (f", cap {entry['max_memberships']}" if int(entry["max_memberships"]) > 0 else ", no cap")
                            + f" · F1 median {entry.get('f1_median', float('nan')):.4f} over seeds {entry.get('confirm_seeds')}")

    def regroup(n: qm.Node) -> None:
        kids = [c for c in n.children if not c.log]
        if not kids or str(n.key[-1]).endswith("search"):  # a search keeps its own count: its rows only break it down
            return
        for child in kids:
            regroup(child)
        status = n.status
        qm._aggregate(n, kids)
        n.noun, n.strip = "evaluations", []
        if len(n.key) <= 2:  # the run: keep its own state unless every search has finished
            n.status = status if status != "done" or n.done >= n.total else "running"
            n.left = max((k.left for k in kids), default=0.0)  # the searches run side by side: the longest decides

    visit(node)
    regroup(node)


def search_last(records: list[dict]) -> float | None:
    """When the search's last evaluation ended (seconds since the epoch), or None."""
    try:
        return time.mktime(time.strptime(str(records[-1].get("finished")), "%Y-%m-%d %H:%M:%S")) if records else None
    except ValueError:
        return None


def search_pace(records: list[dict]) -> float | None:
    """Wall-clock seconds per evaluation of one search so far (evaluations inside it overlap, so this is not their
    summed cost), or None with fewer than two finished evaluations."""
    stamps = []
    for record in records:
        try:
            stamps.append(time.mktime(time.strptime(str(record.get("finished")), "%Y-%m-%d %H:%M:%S")))
        except ValueError:
            continue
    if len(stamps) < 2 or max(stamps) <= min(stamps):
        return None
    return (max(stamps) - min(stamps)) / (len(stamps) - 1)


# --------------------------------------------------------------------------- the controlled-overlap study
def controlled(plan: dict, done: int, grid: dict[str, list[str]]) -> qm.RunModel:
    """LFR-derived controlled overlap: one generated graph per (graph seed, mu, overlap fraction); done is a count."""
    seeds, mus, fractions = grid.get("graph-seeds") or [], grid.get("mus") or [], grid.get("overlap-fractions") or []
    seed_dim = qm.Dim("graph seed", "graph seeds", "graph seed {}", about="seed of the generated graph")
    mu_dim = qm.Dim("mixing mu", "mixing values", "mu {}", about="share of each vertex's edges that leave its community")
    frac_dim = qm.Dim("overlap fraction", "overlap fractions", "overlap {}",
                      about="share of vertices that belong to a second community")
    if not (seeds and mus and fractions) or len(seeds) * len(mus) * len(fractions) != int(plan.get("instances") or 0):
        units = [qm.Unit((f"instance {i + 1}",), i < done) for i in range(int(plan["instances"]))]
        return qm.RunModel([qm.Family("instances", "instances", "one generated graph and its detector run",
                                      [qm.Dim("instance", "instances")], units, "instance")])
    units, index = [], 0
    for seed in seeds:
        for mu in mus:
            for fraction in fractions:
                units.append(qm.Unit((seed, mu, fraction), index < done))
                index += 1
    family = qm.Family(
        "instances", "instances", "one generated LFR-derived graph with planted overlap and one Hedonic run on it",
        [seed_dim, mu_dim, frac_dim], units, "graph seed > mixing mu > overlap fraction",
        notes=[f"{plan.get('groups')} independent (graph seed, mu) groups run {plan.get('workers')} side by side; "
               "the study keeps only a count while it runs, so which instances are done is shown in this order"],
        parallel=max(1, int(plan.get("workers") or 1)))
    return qm.RunModel([family])


# --------------------------------------------------------------------------- the baselines' calibration
FIXED_METHODS = ("codeseg", "ncgame", "ego_splitting")  # the authors' setting, no search
SEARCH_CAP = {"clique_percolation": 4, "fox": 2}  # methods whose search has fewer settings than the budget
PARAM_WORDS = (("wcc_threshold", "θ"), ("communities", "K"), ("clusters", "K"), ("clique_size", "k"), ("threshold", "φ"), ("tau", "τ"))


def method_label(method: str) -> str:
    known = qs.BY_KEY.get(method)
    return known.label if known else method


def params_label(params: dict) -> str:
    for key, word in PARAM_WORDS:
        if key in params:
            return f"{word} {params[key]:g}" if isinstance(params[key], float) else f"{word} {params[key]}"
    return ", ".join(f"{k}={v}" for k, v in params.items() if k != "threads") or "published setting"


def search_budget(plan: dict, method: str) -> tuple[int, float]:
    """The settings (0: unbounded) and seconds (0: unbounded) one search of ``method`` may spend on a pair: its
    ``--method-budget`` when it has one, else the run's ``--max-evals`` and ``--max-minutes`` (``baseline_run.budget_of``)."""
    evals, minutes = (plan.get("method_budgets") or {}).get(method, (plan.get("max_evals") or 0, plan.get("max_minutes") or 0))
    evals = int(evals or 0)
    if method in SEARCH_CAP:
        evals = min(evals, SEARCH_CAP[method]) if evals else SEARCH_CAP[method]
    return evals, float(minutes or 0) * 60


def read_ledger(folder: Path, target: str, method: str) -> list[dict]:
    path = folder / "ledger" / target.replace(":", "_") / f"evals-{method}.jsonl"
    records = []
    try:
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue  # a torn last line of a run that is still writing
    except OSError:
        pass
    return records


def settings_of(records: list[dict]) -> dict[str, list[dict]]:
    """Ledger records grouped by setting (the replicas of one setting together), in the order they were tried."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(json.dumps(rec.get("params"), sort_keys=True), []).append(rec)
    return groups


def baselines(plan: dict, searches: list[dict], results: dict[str, dict]) -> qm.RunModel:
    """The baselines' calibration: pair > method, one unit per calibration. ``results`` is ``searches`` of baseline_tuned.json.

    A finished unit's note is the setting it chose. A search unit's cost is the settings it may still try times what a
    setting cost on this pair (its own ledger); the methods of a pair run side by side, so each unfinished search is
    charged its share of the time (the pair waits for the longest, not for the sum)."""
    folder = Path(plan["output"])
    workers = max(1, int(plan.get("workers") or 0) or 8)
    pair_dim = qm.Dim("pair", "pairs", about="a network with one cover; the settings are calibrated on, and used for, that pair")
    method_dim = qm.Dim("method", "methods", about="a baseline; its parameters are chosen on G[S] alone, by the rule its row names")
    ledgers = {(s["network"], s["method"]): read_ledger(folder, s["network"], s["method"]) for s in searches}
    per_setting: dict[tuple[str, str], float] = {}
    for key, records in ledgers.items():
        groups = settings_of(records)
        spent = [sum(r.get("detect_seconds", 0.0) + r.get("score_seconds", 0.0) for r in g) for g in groups.values() if g]
        if spent:
            per_setting[key] = statistics.fmean(spent)
    share = {}
    for target in dict.fromkeys(s["network"] for s in searches):
        open_ = [s for s in searches if s["network"] == target and s["key"] not in results and s["method"] not in FIXED_METHODS]
        share[target] = max(1, min(workers, len(open_)))
    units, wall = [], {}
    for s in searches:
        res = results.get(s["key"])
        note = ""
        if res:
            chosen = (res.get("runner_params") or res.get("params") or {})
            note = (res.get("status", "") + (" · " + params_label(chosen) if chosen else "")
                    + (f" · {res.get('evaluations', 0)} settings" if res.get("evaluations") else ""))
            if res.get("bracketed") is False:
                note += " · not bracketed"
        else:
            tried = len(settings_of(ledgers[(s["network"], s["method"])]))
            cap = search_budget(plan, s["method"])[0]
            note = (f"{tried} of ≤{cap} settings tried" if cap else f"{tried} settings tried") if tried else ""
        units.append(qm.Unit((pair_label(s["network"]), method_label(s["method"])), s["key"] in results,
                             res.get("seconds") if res else None, note, False,
                             (res.get("seconds") if res else None)))
    by_coords = {(pair_label(s["network"]), method_label(s["method"])): (s["network"], s["method"]) for s in searches}

    def cost(unit: qm.Unit) -> qm.Cost:
        target, method = by_coords[unit.coords]
        if method in FIXED_METHODS:
            return 0.0, "estimate"  # the published setting: nothing is run
        each = per_setting.get((target, method))
        if each is None:
            measured = [v for (t, m), v in per_setting.items() if m == method]
            if not measured:
                return None, ""
            each = statistics.fmean(measured)  # the same method on another pair: a rough guide
        cap, seconds = search_budget(plan, method)
        groups = settings_of(ledgers[(target, method)])
        left = max(0, (cap or 8) - len(groups)) * each
        if seconds:  # the search stops starting settings once its time budget is spent
            spent = sum(r.get("detect_seconds", 0.0) + r.get("score_seconds", 0.0) for g in groups.values() for r in g)
            left = min(left, max(0.0, seconds - spent))
        return left / share.get(target, 1), "estimate"

    family = qm.Family(
        "calibrations", "calibrations", "the search that sets one baseline's parameters on one pair, scored without the ground truth",
        [pair_dim, method_dim], units, "pair > method",
        cost=cost, parallel=workers,
        notes=[f"the methods of a pair run side by side (up to {workers}); a pair takes as long as its slowest method, "
               "so each unfinished search is charged its share of the time",
               "a method with the authors' published setting (CoDeSEG, FOX, NcGame, Ego-Splitting) runs nothing here",
               "open a method to see every setting it tried: the held-out pair log-likelihood (higher is better), the "
               "coverage, the communities returned and the detector time of each replica"])
    return qm.RunModel([family])


def decorate_baselines(node: qm.Node, plan: dict, results: dict[str, dict]) -> None:
    """Add, under every method of the calibration tree, a row per setting it tried (read from the ledger)."""
    from hedonic.experiments.overlapping import replication

    folder = Path(plan["output"])
    replicas = max(1, int(plan.get("replicates") or 1))
    labels = {pair_label(t): t for t in plan["networks"]}
    names = {method_label(m): m for m in plan["methods"]}

    def visit(n: qm.Node) -> None:
        for child in n.children:
            visit(child)
        if len(n.key) < 2 or n.children:
            return
        working = n.status == "running"  # the model marks the units being worked on; the spinner belongs on a leaf of it
        target, method = labels.get(n.key[-2]), names.get(n.key[-1])
        if not target or not method:
            return
        records = read_ledger(folder, target, method)
        groups = settings_of(records)
        res = results.get(f"{method}/{target}") or {}
        chosen = res.get("runner_params") or res.get("params") or {}
        want = replicas if replication.calibration_kind(method) == "replicas" else 1
        best = max(((statistics.fmean([r["score"] for r in g if r.get("score") is not None] or [float("-inf")]), k)
                    for k, g in groups.items()), default=None)
        for index, (key, group) in enumerate(groups.items(), 1):
            params = group[0].get("params") or {}
            scores = [r["score"] for r in group if r.get("score") is not None]
            finished = [r for r in group if r.get("status") != "running"]
            complete = len(finished) >= max(want, len(group))
            row = qm.Node(key=n.key + ("setting", key), label=params_label(params), noun="settings", depth=n.depth + 1,
                          done=int(complete), total=1, status="done" if complete else "running")
            if not complete:  # a setting under way fills its bar by the replicas that have finished
                row.sub = (len(finished), max(want, len(group)), "replicas")
                row.unknown = 1
            row.basis = "" if complete else "estimate"
            row.took = sum(r.get("detect_seconds", 0.0) + r.get("score_seconds", 0.0) for r in group) if complete else 0.0
            if complete and group[0].get("status") in ("timeout", "failed", "unavailable"):
                row.label += f"  ({group[0].get('status')})"
            star = chosen and all(params.get(k) == v for k, v in chosen.items() if k in params)
            if scores:
                row.label += f"  score {statistics.fmean(scores):.4f}" + ("  ★ chosen" if star and res else "")
            elif group[0].get("status") in ("timeout", "failed", "unavailable"):
                row.label += "  no score"
            lines = [f"setting {index}: {', '.join(f'{k}={v}' for k, v in params.items())}"]
            for rec in group:
                s = rec.get("summary") or {}
                v = rec.get("criterion_values") or {}
                if rec.get("status") == "completed":
                    lines.append(f"  replica {rec.get('replica', 0)}: score {rec['score']:.4f} (edges {v.get('positive_loglik', 0):.3f}, "
                                 f"non-edges {v.get('negative_loglik', 0):.2e}) · {s.get('communities', '?')} communities, "
                                 f"coverage {s.get('coverage', 0):.1%}, {s.get('mean_memberships', 0):.2f} memberships/vertex, "
                                 f"largest {s.get('largest_community', '?')} · {rec.get('detect_seconds', 0):.1f} s")
                else:
                    lines.append(f"  replica {rec.get('replica', 0)}: {rec.get('status')} · {str(rec.get('reason') or '')[:160]}")
            row.detail = lines
            n.children.append(row)
        if res:
            n.detail += [f"rule: {res.get('rule', '')}", f"stop: {res.get('stop_reason', '')}"
                         + ("" if res.get("bracketed") is None else " · bracketed by worse neighbours" if res["bracketed"]
                            else " · NOT bracketed: the best setting is at the edge of what was tried"),
                         *[f"note: {x}" for x in (res.get("notes") or [])]]
            if best:
                n.detail.append(f"best by the criterion: {params_label(next(iter(groups[best[1]]))['params'])}")
        if not res and method not in FIXED_METHODS:
            cap = search_budget(plan, method)[0]
            more = max(0, cap - len(groups)) if cap else 0
            if more:  # the settings the search may still try: they keep the method's total honest, and carry its ETA
                rest = qm.Node(key=n.key + ("more",), label=f"up to {more} more settings", noun="settings", depth=n.depth + 1,
                               done=0, total=more, status="pending", left=n.left, basis=n.basis, unknown=n.unknown)
                if working and not any(c.status == "running" for c in n.children):
                    rest.status = "running"  # the next setting is being tried right now: spin on the row that stands for it
                n.children.append(rest)
            n.detail.append(f"{len(groups)} settings tried so far (budget {cap or 'unbounded'}); the search ends earlier "
                            "when the optimum is bracketed")
        if n.children:
            n.noun, n.left, n.unknown, n.basis = "settings", 0.0, 0, ""
            qm._aggregate(n, n.children)
            if res:
                n.status = "done"

    visit(node)

    def lift(n: qm.Node) -> None:
        """Parents count what is under them: settings tried and still to try, up to the run."""
        for child in n.children:
            lift(child)
        if n.children and len(n.key) <= 4 and n.children[0].noun == "settings" or (n.children and n.noun != "settings"
                                                                                  and all(c.noun == "settings" for c in n.children)):
            status = n.status
            qm._aggregate(n, n.children)
            n.noun, n.strip = "settings", []
            if len(n.key) <= 2:
                n.status = status if status != "done" or n.done >= n.total else "running"

    lift(node)
