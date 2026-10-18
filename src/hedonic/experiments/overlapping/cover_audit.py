"""Robustness audit of the covers a benchmark returns, as a durable run (``hedonic run audit``).

Chapter 6 of the overlapping-communities paper asks how robust each method's cover is, with the definition of the
ground-truth chapter: for every vertex, the exact set of resolutions in [0, 1] where its memberships are a best
response, under both action spaces (fixed and open labels). This module applies that analysis to the covers saved by
``hedonic run exp ... --save-covers``, one record per cover. The computation is the library's C audit (the code
behind ``Game.stable_intervals`` and ``Game.regrets``) through ``overlapping.stability``; ``--workers`` forked
processes split the vertices and share its read-only prepared state.

Two profiles are audited, and every record names the one it used:

``cover``     the saved scoring cover: unique communities, one label each, which every method has. A vertex that
              no community contains is completed with a private singleton label (it plays alone), and the count is
              recorded. This is the profile to compare methods on.
``labelled``  the exact labelled memberships a HOC detector returned (``detector.final_memberships``). The saved
              cover merges labels with identical members, which changes the vertices' intensities, so the cover of
              a HOC run is not the profile the detector reached. The labelled profile is a Nash equilibrium at the
              run's own resolution and cap, which makes it the correctness check of this audit.

The membership cap of the game is the cap the detector played under when the record has one (HOC), and the cover's
own largest multiplicity otherwise (``--cap`` overrides). The resolution grid is the locked grid of the ground-truth
spectrum plus the run's own resolution and the graph density.

    hedonic run audit --source RUN_DIR [RUN_DIR ...] --methods hedonic,codeseg --name my-audit
    hedonic-exp overlapping-cover-audit --source RUN_DIR --dry-run

The run folder holds ``cover_audit.jsonl`` (one JSON line per cover and profile, appended and fsynced: a rerun
skips what is recorded), ``cover_audit.csv`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from hedonic.experiments.config import NETWORKS_DIR, RUNS_ROOT, expand_path
from hedonic.experiments.runtime import runtime_versions
from hedonic.experiments.overlapping import gt_posthoc as ph
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import replication
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import snap
from hedonic.experiments.overlapping import stability as st
from hedonic.experiments.overlapping.start_quality import append_record, repair_tail

PROTOCOL = "cover-audit-v1"
RESULTS_NAME = "cover_audit.jsonl"
TABLE_NAME = "cover_audit.csv"
SUMMARY_NAME = "summary.json"
PROGRESS_NAME = "audit_progress.json"  # where the audit process of a benchmark run reports; the worker shows it
PROGRESS_STEPS = 400  # progress events per policy during the exact stage
PROGRESS_POLL_SECONDS = 2.0  # how often the worker reads that file while it waits for the audit process
ETA_MIN_SHARE = 0.05  # of a policy's vertices that must be in before the audit predicts its end

POLICIES = ph.POLICIES  # (("fixed_labels", False), ("open_labels", True))
PRIMARY_POLICY = "open_labels"  # the benchmark's HOC adapters allow isolation
PROFILES = ("cover", "labelled")
LABELLED_METHODS = ("hedonic",)  # the methods whose records keep their exact labels
LOG_FLOOR = 1e-4  # the log measure averages over log10(gamma) from here to 1, as the spectrum's log grid does
REGRET_GAMMAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)  # resolutions where the regret distribution is evaluated, with own and density
DEFAULT_NETWORKS = "dblp,amazon"
DEFAULT_SEEDS = "0-9"
DEFAULT_CAP = "record"

Progress = Callable[[dict[str, Any]], None]


def _silent(event: dict[str, Any]) -> None:
    return None


# --------------------------------------------------------------------------- plan
def parse_cap(value: str) -> str | int:
    if value in ("record", "own"):
        return value
    try:
        cap = int(value)
    except ValueError:
        raise ValueError(f"--cap must be record, own or an integer, not {value!r}") from None
    if cap < 2:
        raise ValueError("--cap must be at least 2")
    return cap


def build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Robustness audit of the covers a benchmark returned: the exact stable-vertex fraction over "
                     "resolution, both action spaces, one record per cover."))
    parser.add_argument("--source", nargs="+", required=True, metavar="RUN_DIR",
                        help="benchmark run folders holding runs/ and covers/ (the first folder with a unit's record wins)")
    parser.add_argument("--networks", default=DEFAULT_NETWORKS, help=f"comma list (default {DEFAULT_NETWORKS})")
    parser.add_argument("--methods", help="comma list of method keys (default: those found in the sources)")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS,
                        help=f"seeds, e.g. 0-9 (default {DEFAULT_SEEDS}); a deterministic method is audited at the lowest "
                             "only, as the benchmark runs it (see --replication)")
    parser.add_argument("--replication", choices=replication.MODES, default=replication.DEFAULT_MODE,
                        help="auto (default): expect a deterministic method's cover at the lowest seed only, as `hedonic "
                             "run exp` plans it; all: at every seed (a benchmark run with --replication all)")
    parser.add_argument("--profiles", default=",".join(PROFILES),
                        help="cover (every method), labelled (HOC only, exact labels) or both (default)")
    parser.add_argument("--cap", default=DEFAULT_CAP,
                        help="membership cap of the audited game: record (the detector's cap, else the cover's own), "
                             "own, or an integer (default record)")
    parser.add_argument("--workers", type=int, help="forked vertex workers (default min(8, cpus))")
    parser.add_argument("--network-root", help="folder of the SNAP networks (default HEDONIC_NETWORKS_DIR)")
    parser.add_argument("--induce", choices=("full", "covered"), default=None,
                        help="the graph the source runs detected on: covered = the subgraph induced by the cover's "
                             "vertices (a network may be named NETWORK:COVER). Default: whichever folder the unit has")
    parser.add_argument("--name", help="run name (default cover-audit)")
    parser.add_argument("--output-dir", help="run folder (default: <output_dir>/NAME)")
    parser.add_argument("--overwrite", action="store_true", help="delete this audit's records before starting")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load nothing and write nothing")
    return parser


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def unit_folders(source: Path, network: str, seed: int, induce: str | None = None) -> list[Path]:
    """The seed folders of ``source`` that hold ``network``'s units (``NETWORK[:COVER]``), without those of its covers.

    A benchmark names a unit's folder ``<base>[-<cover>]-<full|nN>[-covered]/seed<S>``; ``induce`` ("full" or
    "covered") keeps only the folders of that graph, and None keeps both.
    """
    base, cover = qs.split_target(network)
    tag = re.escape(base if cover is None else f"{base}-{cover}")
    pattern = re.compile(rf"{tag}-(full|n\d+)(-covered)?")
    out = []
    for folder in sorted(source.glob(f"{glob.escape(base)}-*/seed{seed}")):
        match = pattern.fullmatch(folder.parent.name)
        if match and (induce is None or bool(match.group(2)) == (induce == "covered")):
            out.append(folder)
    return out


def record_files(source: Path, network: str, seed: int, runner: str, induce: str | None = None) -> list[Path]:
    base = qs.split_target(network)[0]
    return [path for folder in unit_folders(source, network, seed, induce)
            if (path := folder / "runs" / base / f"{runner}.json").is_file()]


def discover_methods(sources: list[Path], networks: list[str], seeds: list[int], induce: str | None = None) -> list[str]:
    """Method keys that have a record in any source, in the catalogue's order."""
    found = set()
    for method in qs.METHODS + qs.CONTROLS:
        for source, network, seed in itertools.product(sources, networks, seeds):
            if record_files(source, network, seed, method.runner, induce):
                found.add(method.key)
                break
    return [method.key for method in qs.METHODS + qs.CONTROLS if method.key in found]


def resolve(argv: list[str], prog: str = "hedonic run audit") -> dict[str, Any]:
    """The plan of ``hedonic run audit ARGV``. Reads folder names only: no graph, no record, nothing written."""
    from hedonic.experiments.overlapping import userconfig

    args = build_parser(prog).parse_args(argv)
    networks = _split(args.networks)
    unknown = [net for net in networks if qs.split_target(net)[0] not in qs.NETWORKS]  # NETWORK[:COVER]
    if unknown:
        raise ValueError(f"unknown network(s) {', '.join(unknown)}; choose from {', '.join(qs.NETWORKS)}")
    seeds = qs.parse_seeds(args.seeds)
    sources = [expand_path(source).resolve() for source in args.source]
    methods = (qs.parse_choice(args.methods, [m.key for m in qs.METHODS], qs.ALIASES, "method", qs.CONTROL_KEYS) if args.methods
               else discover_methods(sources, networks, seeds, args.induce))
    profiles = _split(args.profiles)
    bad = [profile for profile in profiles if profile not in PROFILES]
    if bad or not profiles:
        raise ValueError(f"--profiles takes {' and '.join(PROFILES)}, not {args.profiles!r}")
    name = qs.parse_name(args.name or "cover-audit")
    root = Path(userconfig.defaults().get("output_dir") or RUNS_ROOT)
    output = expand_path(args.output_dir).resolve() if args.output_dir else expand_path(root / name).resolve()
    return {"sources": sources, "networks": networks, "methods": methods, "seeds": seeds, "profiles": profiles,
            "induce": args.induce, "replication": args.replication,
            "cap": parse_cap(args.cap), "name": name, "output": output, "overwrite": bool(args.overwrite),
            "dry_run": bool(args.dry_run),
            "workers": max(1, int(args.workers or min(8, os.cpu_count() or 1))),
            "network_root": expand_path(args.network_root or NETWORKS_DIR).resolve(),
            "atol": ph.ATOL, "rtol": ph.RTOL}


def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Valid records by key (a torn or corrupt line is skipped; the first record of a key wins)."""
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("key"), str):
            records.setdefault(record["key"], record)
    return records


def key_of(network: str, seed: int, method: str, profile: str) -> str:
    return f"{network}/seed{seed}/{method}/{profile}"


def conditions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Every audit of the plan, methods outermost in the order given (so an interruption costs the last ones).

    A method is audited at the seeds the benchmark runs it at (``replication.planned_seeds``): under ``replication =
    "auto"`` a deterministic method has one cover per network, at the lowest seed. Covers of other seeds that an older
    run left on disk are not expected, and not audited."""
    out = []
    mode = plan.get("replication", replication.DEFAULT_MODE)
    for method, network, seed in itertools.product(plan["methods"], plan["networks"], plan["seeds"]):
        if seed not in replication.planned_seeds(method, plan["seeds"], mode):
            continue
        for profile in plan["profiles"]:
            if profile == "labelled" and method not in LABELLED_METHODS:
                continue
            out.append({"key": key_of(network, seed, method, profile), "network": network, "seed": seed,
                        "method": method, "profile": profile})
    return out


def unit_keys(network: str, seed: int, method: str, profiles: tuple[str, ...] = PROFILES) -> list[str]:
    """The ledger keys of one detector run: its cover audit, and its labelled audit for a HOC method."""
    return [key_of(network, seed, method, profile) for profile in profiles
            if profile != "labelled" or method in LABELLED_METHODS]


def find_record(plan: dict[str, Any], network: str, seed: int, method: str) -> tuple[Path | None, Path | None]:
    """The source folder and record file of one detector run, or (None, None) when no source has it."""
    runner = qs.BY_KEY[method].runner
    for source in plan["sources"]:
        for path in record_files(source, network, seed, runner, plan.get("induce")):
            return source, path
    return None, None


def describe(plan: dict[str, Any]) -> str:
    once = [m for m in plan["methods"] if len(replication.planned_seeds(m, plan["seeds"], plan.get("replication", "auto"))) == 1]
    seeds = f"{len(plan['seeds'])} seeds" + (f" (deterministic methods: seed {min(plan['seeds'])} only)"
                                             if once and len(plan["seeds"]) > 1 else "")
    return (f"cover audit · {len(plan['networks'])} networks × {len(plan['methods'])} methods × {seeds} · "
            f"{len(conditions(plan)):,} audits ({'+'.join(plan['profiles'])})")


def plan_lines(plan: dict[str, Any]) -> list[str]:
    todo = conditions(plan)
    present = read_records(plan["output"] / RESULTS_NAME)
    have = sum(1 for condition in todo if condition["key"] in present)
    missing = [c for c in todo if find_record(plan, c["network"], c["seed"], c["method"])[1] is None]
    lines = [f"hedonic-exp overlapping-cover-audit · {describe(plan)} ({PROTOCOL})",
             "  sources  " + ", ".join(str(source) for source in plan["sources"]),
             f"  networks {', '.join(plan['networks'])} · seeds {plan['seeds'][0]}..{plan['seeds'][-1]}"
             if plan["seeds"] else "  seeds    none",
             "  methods  " + (", ".join(plan["methods"]) or "none found in the sources"),
             f"  game     cap {plan['cap']} · both action spaces ({PRIMARY_POLICY} is the primary one) · "
             f"grid = the locked spectrum grid + the run's own resolution + the graph density",
             f"  workers  {plan['workers']} forked vertex workers",
             f"  records  {plan['output'] / RESULTS_NAME} (one JSON line per cover and profile)",
             f"  present  {have:,} of {len(todo):,} audits already recorded"
             + (" (--overwrite: they are deleted first)" if plan["overwrite"] else " (a run resumes them)")]
    if missing:
        lines.append(f"  waiting  {len(missing):,} audits have no benchmark record in the sources yet, e.g. {missing[0]['key']}")
    return lines


# --------------------------------------------------------------------------- profiles
def read_cover(path: Path) -> list[list[int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def complete_uncovered(memberships: list[list[int]], next_label: int) -> tuple[list[list[int]], int]:
    """Give every vertex without a label a private one: it plays alone, which adds nothing to the potential."""
    added = 0
    for vertex, labels in enumerate(memberships):
        if not labels:
            memberships[vertex] = [next_label + added]
            added += 1
    return memberships, added


def cover_profile(cover: list[list[int]], n_vertices: int) -> tuple[list[list[int]], dict[str, Any]]:
    canonical = rb.canonicalize_cover(cover, n_vertices)
    memberships = rb.cover_to_vertex_memberships(canonical, n_vertices, require_covered=False)
    memberships, added = complete_uncovered(memberships, len(canonical))
    return memberships, {"communities": len(canonical), "completed_vertices": added}


def labelled_profile(rows: list[list[int]], n_vertices: int) -> tuple[list[list[int]], dict[str, Any]]:
    if len(rows) != n_vertices:
        raise ValueError(f"the record keeps {len(rows)} membership rows for a graph of {n_vertices} vertices")
    labels = sorted({int(label) for row in rows for label in row})
    remap = {label: index for index, label in enumerate(labels)}
    memberships = [[remap[int(label)] for label in row] for row in rows]
    memberships, added = complete_uncovered(memberships, len(labels))
    return memberships, {"communities": len(labels), "completed_vertices": added}


def effective_cap(mode: str | int, record_cap: int | None, own: int) -> int:
    requested = record_cap or 0 if mode == "record" else 0 if mode == "own" else int(mode)
    return max(2, requested, own)


def own_resolution(detector: dict[str, Any]) -> tuple[float | None, str | None]:
    """The resolution a HOC detector played at, and a note when the record names two different ones.

    The run's own ``parameters.absolute_resolution`` is the value passed to the game; ``effective_resolution`` is what
    the adapter reports. They agree on current records. An earlier benchmark recorded the density there while playing
    the absolute value, so the parameter is preferred and the disagreement is kept in the audit record.
    """
    absolute = (detector.get("parameters") or {}).get("absolute_resolution")
    reported = detector.get("effective_resolution")
    if absolute is None and reported is None:
        return None, None
    if absolute is None:
        return float(reported), None
    note = None
    if reported is not None and not math.isclose(float(absolute), float(reported), rel_tol=1e-9, abs_tol=0.0):
        note = (f"the record reports effective_resolution {float(reported):.6g} but its parameters name "
                f"absolute_resolution {float(absolute):.6g}; the parameter was used")
    return float(absolute), note


def graph_identity(graph) -> str:
    edges = np.asarray(sorted(tuple(sorted(edge)) for edge in graph.get_edgelist()), dtype=np.int64)
    return hashlib.sha256(edges.tobytes() + str(graph.vcount()).encode()).hexdigest()


def load_graph(network: str, cover_variant: str, root: Path, induce: str = "full"):
    """The graph the benchmark's detectors ran on, loaded as ``codeseg_reproduction`` loads it.

    ``induce="covered"``: the subgraph induced by the vertices the cover contains (vertices renumbered, as the detectors
    saw them, so the saved covers index it).
    """
    dataset = snap.load_snap_dataset(network, cover_variant=cover_variant, data_root=root, allow_catalog=False)
    if induce == "covered":
        from hedonic.experiments.overlapping.ground_truth_data import covered_induced_dataset

        dataset = covered_induced_dataset(dataset)
    return dataset.graph


# --------------------------------------------------------------------------- audit of one profile
def stable_counts(lo: np.ndarray, hi: np.ndarray, owner: np.ndarray, grid: list[float]) -> list[int]:
    return [int(np.unique(owner[(lo <= gamma) & (hi >= gamma)]).size) for gamma in grid]


def vertices_event(label: str, policy: str, number: int, policies: int, done: int, total: int, elapsed: float) -> dict[str, Any]:
    """A progress event of the exact stage: ``done`` of ``total`` vertices of policy ``number`` of ``policies``.

    The vertices go through in a fixed pseudo-random order, so the rate so far is the rate of what is left. The rest of
    the audit is this policy's remainder plus as long again for each policy still to come (the regret table that ends
    each policy is not counted: it is a small part of the whole). The pool hands its results over in order and in
    chunks, so the first ones arrive in a burst after a whole chunk's time: no ETA is given before ``ETA_MIN_SHARE`` of
    the vertices are in (the first reading of BigCLAM's cover said 50 h for what is a few).
    """
    whole = elapsed / done * total if done and done >= ETA_MIN_SHARE * total else None
    eta = None if whole is None else max(0.0, whole - elapsed) + (policies - number) * whole
    detail = f"{policy} ({number}/{policies}): exact stability {done:,}/{total:,} vertices"
    if eta is not None and (done < total or number < policies):
        detail += f" · about {qs.fmt_time(eta)} left"
    return {"type": "vertices", "policy": policy, "number": number, "policies": policies, "done": done, "total": total,
            "elapsed": elapsed, "eta_seconds": eta, "detail": detail, "text": f"{label}{detail}"}


def audit_state(cover: st.PreparedCover, *, cap: int, grid: list[float], regret_gammas: list[float], workers: int,
                atol: float, rtol: float, progress: Progress = _silent, label: str = "") -> dict[str, Any]:
    """Exact stable intervals of every vertex under both action spaces, and what they say about ``grid``.

    ``cover`` comes from ``stability.prepare`` (the C engine by default); ``workers`` forked processes split the
    vertices of the exact stage and of the regret table.
    """
    n_vertices = cover.n_vertices
    policies: dict[str, Any] = {}
    for number, (policy, allow) in enumerate(POLICIES, 1):
        progress({"type": "stage", "text": f"{label}{policy}: exact stability of {n_vertices:,} vertices"})
        per_vertex: list[list[list[float]] | None] = [None] * n_vertices
        started = time.time()
        count = 0
        chunk = max(1, min(st.MAX_CHUNK, -(-n_vertices // PROGRESS_STEPS)))
        for vertices, intervals in ph._vertex_stream(cover, cap, allow, atol=atol, rtol=rtol, workers=workers,
                                                     chunk=chunk):
            for vertex, found in zip(vertices, intervals):
                per_vertex[vertex] = found
            count += len(vertices)
            progress(vertices_event(label, policy, number, len(POLICIES), count, n_vertices, time.time() - started))
        exact_seconds = time.time() - started
        lo, hi, owner = ph._interval_arrays(per_vertex)
        counts = stable_counts(lo, hi, owner, grid)
        floor = np.maximum(lo, LOG_FLOOR)
        above = hi > floor
        decades = -math.log10(LOG_FLOOR)
        progress({"type": "stage", "text": f"{label}{policy} ({number}/{len(POLICIES)}): regret at {len(regret_gammas)} resolutions"})
        regret_started = time.time()
        regret = st.spectrum_rows(cover, cap, regret_gammas, atol=atol, rtol=rtol, allow_isolation=allow,
                                  workers=workers)
        policies[policy] = {
            "exact_seconds": round(exact_seconds, 2),
            "regret_seconds": round(time.time() - regret_started, 2),
            "stable_fraction": [count / n_vertices for count in counts],
            "stable_measure_linear": float((hi - lo).sum()) / n_vertices,
            "stable_measure_log": float((np.log10(hi[above]) - np.log10(floor[above])).sum()) / (n_vertices * decades),
            "nash_window": ph._nash_window(per_vertex),
            "vertices_with_several_intervals": int(sum(len(intervals) > 1 for intervals in per_vertex)),
            "vertices_never_stable": int(sum(not intervals for intervals in per_vertex)),
            "regret": [{key: row[key] for key in ("gamma", "stable_fraction", "mean_positive_regret",
                                                  "max_positive_regret", "p95_positive_regret", "is_nash_equilibrium")}
                       for row in regret],
        }
    return policies


# --------------------------------------------------------------------------- one record
def audit_unit(condition: dict[str, Any], plan: dict[str, Any], graphs: dict[tuple[str, str, str], Any], progress: Progress = _silent
               ) -> dict[str, Any]:
    """One cover and profile: the record to append. A detector run without a cover is a finished record too."""
    network, seed, method, profile = (condition[k] for k in ("network", "seed", "method", "profile"))
    base = {"key": condition["key"], "protocol": PROTOCOL, "network": network, "seed": seed, "method": method,
            "profile": profile}
    source, path = find_record(plan, network, seed, method)
    if path is None:
        raise FileNotFoundError(f"no benchmark record for {network} seed {seed} {method} in the sources")
    record = json.loads(path.read_text())
    unit_dir = path.parents[2]  # <source>/<network>-full/seed<S>: the folder the record's cover_path is relative to
    detector = record.get("detector") or {}
    base.update(source=str(source), record_file=str(path.relative_to(source)), config_digest=record.get("config_digest"),
                detection_seconds=record.get("detection_seconds"))
    if record.get("status") != "completed":
        return {**base, "status": "no_cover", "reason": f"the detector run ended {record.get('status')!r}"}
    if profile == "labelled" and not detector.get("final_memberships"):
        return {**base, "status": "no_labels", "reason": "the record keeps no exact labelled memberships"}
    if profile == "cover" and not record.get("cover_path"):
        return {**base, "status": "no_cover", "reason": "the run saved no cover (it needs --save-covers)"}
    report = record.get("dataset_report") or {}
    variant = str(report.get("cover_variant") or "all")
    base_network = qs.split_target(network)[0]
    induce = "covered" if unit_dir.parent.name.endswith("-covered") or plan.get("induce") == "covered" else "full"
    graph_key = (base_network, variant, induce)
    load_seconds = None  # set when this unit loaded the graph; a graph an earlier unit loaded costs it nothing
    if graph_key not in graphs:
        for stale in [key for key in graphs if key != graph_key]:  # one graph at a time: a covered LiveJournal is GBs
            del graphs[stale]
        progress({"type": "stage", "text": f"loading the {network} graph" + (" (covered subgraph)" if induce == "covered" else "")})
        loading = time.time()
        graph = load_graph(base_network, variant, plan["network_root"], induce)
        graphs[graph_key] = (graph, graph_identity(graph))
        load_seconds = round(time.time() - loading, 2)
    graph, identity = graphs[graph_key]
    n_vertices = graph.vcount()
    if (report.get("n"), report.get("m")) != (n_vertices, graph.ecount()):
        raise ValueError(f"{path}: the record was produced on a graph of n={report.get('n')}, m={report.get('m')}, "
                         f"but the loaded {network} graph has n={n_vertices}, m={graph.ecount()}")
    if profile == "cover":
        memberships, info = cover_profile(read_cover(unit_dir / record["cover_path"]), n_vertices)
    else:
        memberships, info = labelled_profile(detector["final_memberships"], n_vertices)
    own = max(len(labels) for labels in memberships)
    record_cap = detector.get("max_memberships") if method in LABELLED_METHODS else None
    cap = effective_cap(plan["cap"], int(record_cap) if record_cap else None, own)
    own_gamma, gamma_note = own_resolution(detector) if method in LABELLED_METHODS else (None, None)
    density = float(graph.density())
    extra = [density] + ([own_gamma] if own_gamma is not None else [])
    grid = sorted(set(ph.locked_grid()) | set(extra))
    regret_gammas = sorted(set(REGRET_GAMMAS) | set(extra))
    building = time.time()
    cover = st.prepare(graph, memberships)
    state_seconds = round(time.time() - building, 2)
    started = time.time()
    policies = audit_state(cover, cap=cap, grid=grid, regret_gammas=regret_gammas, workers=plan["workers"],
                           atol=plan["atol"], rtol=plan["rtol"], progress=progress, label=f"{condition['key']} · ")
    result = {**base, "status": "completed", "n_vertices": n_vertices, "n_edges": graph.ecount(),
              "graph_sha256": identity, "cover_variant": variant, "density": density, "gamma_own": own_gamma,
              "cap": cap, "cap_mode": plan["cap"], "cap_in_record": record_cap, "own_max_multiplicity": own,
              "atol": plan["atol"], "rtol": plan["rtol"], "engine": cover.engine, "grid": grid,
              "audit_seconds": round(time.time() - started, 2), "load_seconds": load_seconds, "state_seconds": state_seconds,
              "policies": policies, **info}
    warnings = [gamma_note] if gamma_note else []
    if profile == "labelled" and own_gamma is not None:
        at_own = own_gamma_value(result, PRIMARY_POLICY, "stable_fraction")
        if at_own is not None and at_own < 1.0:
            warnings.append(f"the exact labelled memberships are not an equilibrium at their own resolution "
                            f"({at_own:.6f} of vertices stable): the record does not describe the state the detector "
                            f"stopped in")
    if warnings:
        result["warnings"] = warnings
    if profile == "cover":
        result["cover_file_sha256"] = hashlib.sha256((unit_dir / record["cover_path"]).read_bytes()).hexdigest()
    return result


# --------------------------------------------------------------------------- outputs
def own_gamma_value(record: dict[str, Any], policy: str, field: str) -> float | None:
    """A regret-table field at the run's own resolution, or None for a method that has none."""
    gamma = record.get("gamma_own")
    if gamma is None:
        return None
    for row in record["policies"][policy]["regret"]:
        if row["gamma"] == gamma:
            return row[field]
    return None


def table_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(records):
        record = records[key]
        head = {field: record.get(field) for field in ("key", "network", "seed", "method", "profile", "status")}
        head["warnings"] = "; ".join(record.get("warnings") or [])
        if record.get("status") != "completed":
            rows.append({**head, "reason": record.get("reason")})
            continue
        for policy, block in record["policies"].items():
            window = block["nash_window"]
            rows.append({**head, "policy": policy, "n_vertices": record["n_vertices"], "cap": record["cap"],
                         "own_max_multiplicity": record["own_max_multiplicity"], "communities": record["communities"],
                         "completed_vertices": record["completed_vertices"], "gamma_own": record["gamma_own"],
                         "stable_fraction_at_own_gamma": own_gamma_value(record, policy, "stable_fraction"),
                         "max_regret_at_own_gamma": own_gamma_value(record, policy, "max_positive_regret"),
                         "stable_measure_linear": block["stable_measure_linear"],
                         "stable_measure_log": block["stable_measure_log"],
                         "nash_window_lo": window[0][0] if window else None,
                         "nash_window_hi": window[-1][1] if window else None,
                         "vertices_never_stable": block["vertices_never_stable"],
                         "vertices_with_several_intervals": block["vertices_with_several_intervals"],
                         "detection_seconds": record.get("detection_seconds"),
                         "audit_seconds": record["audit_seconds"], "exact_seconds": block["exact_seconds"]})
    return rows


def write_table(path: Path, records: dict[str, dict[str, Any]]) -> None:
    rows = table_rows(records)
    fields: list[str] = []
    for row in rows:
        fields += [field for field in row if field not in fields]
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "sd": statistics.stdev(values) if len(values) > 1 else None}


def summarise(plan: dict[str, Any], todo: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status: dict[str, int] = {}
    for condition in todo:
        state = records[condition["key"]]["status"] if condition["key"] in records else "pending"
        status[state] = status.get(state, 0) + 1
    groups: dict[str, dict[str, Any]] = {}
    for key, record in records.items():
        if record.get("status") != "completed":
            continue
        group = f"{record['network']}/{record['method']}/{record['profile']}"
        block = groups.setdefault(group, {"stable_measure_log": [], "stable_measure_linear": [], "stable_at_own_gamma": []})
        primary = record["policies"][PRIMARY_POLICY]
        block["stable_measure_log"].append(primary["stable_measure_log"])
        block["stable_measure_linear"].append(primary["stable_measure_linear"])
        own = own_gamma_value(record, PRIMARY_POLICY, "stable_fraction")
        if own is not None:
            block["stable_at_own_gamma"].append(own)
    warned = {key: record["warnings"] for key, record in sorted(records.items()) if record.get("warnings")}
    return {"name": plan["name"], "protocol": PROTOCOL, "primary_policy": PRIMARY_POLICY, "audits": len(todo),
            "status": status, "warnings": warned, "sources": [str(source) for source in plan["sources"]],
            "groups": {group: {measure: _mean_sd(values) for measure, values in block.items()}
                       for group, block in sorted(groups.items())},
            "environment": runtime_versions(), "records": str(plan["output"] / RESULTS_NAME),
            "table": str(plan["output"] / TABLE_NAME)}


def write_outputs(plan: dict[str, Any], todo: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = plan["output"]
    write_table(out / TABLE_NAME, records)
    summary = summarise(plan, todo, records)
    tmp = out / (SUMMARY_NAME + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, default=str))
    tmp.replace(out / SUMMARY_NAME)
    return summary


# --------------------------------------------------------------------------- run
def progress_line(done: int, total: int, record: dict[str, Any]) -> str:
    if record["status"] != "completed":
        return f"[{done}/{total}] {record['key']} {record['status']}: {record.get('reason', '')}"
    block = record["policies"][PRIMARY_POLICY]
    own = own_gamma_value(record, PRIMARY_POLICY, "stable_fraction")
    tail = f" · stable at own gamma {own:.4f}" if own is not None else ""
    if record.get("warnings"):
        tail += " · WARNING: " + "; ".join(record["warnings"])
    return (f"[{done}/{total}] {record['key']} · {PRIMARY_POLICY} log measure {block['stable_measure_log']:.4f}{tail}"
            f" · {record['audit_seconds']:.0f}s")


def run_audit(plan: dict[str, Any], *, progress: Progress = _silent, emit: Callable[[str], Any] = print) -> int:
    out = plan["output"]
    out.mkdir(parents=True, exist_ok=True)
    results = out / RESULTS_NAME
    if plan["overwrite"]:
        for name in (RESULTS_NAME, TABLE_NAME, SUMMARY_NAME):
            (out / name).unlink(missing_ok=True)
    repair_tail(results)
    records = read_records(results)
    todo = conditions(plan)
    if not todo:
        raise ValueError("nothing to audit: no method was named and none was found in the sources")
    done = sum(1 for condition in todo if condition["key"] in records)
    emit(f"cover audit: {len(todo)} audits; {done} already in {results}; {len(todo) - done} to run")
    graphs: dict[tuple[str, str, str], Any] = {}
    waiting = []
    for condition in todo:
        if condition["key"] in records:
            continue
        if find_record(plan, condition["network"], condition["seed"], condition["method"])[1] is None:
            waiting.append(condition["key"])
            continue
        record = audit_unit(condition, plan, graphs, progress)
        append_record(results, record)
        records[condition["key"]] = record
        done += 1
        progress({"type": "unit", "done": done, "total": len(todo), "key": condition["key"]})
        emit(progress_line(done, len(todo), record))
    write_outputs(plan, todo, records)
    if waiting:
        raise RuntimeError(f"{len(waiting)} audits have no benchmark record in the sources yet (first: {waiting[0]}); "
                           f"finish the benchmark, then run this audit again")
    emit(f"cover audit: wrote {results}, {out / TABLE_NAME} and {out / SUMMARY_NAME}")
    return 0


def state_of(plan: dict[str, Any]) -> str:
    """complete | partial | missing: whether every audit of the plan has a record."""
    records = read_records(plan["output"] / RESULTS_NAME)
    todo = conditions(plan)
    done = sum(1 for condition in todo if condition["key"] in records)
    return "complete" if todo and done == len(todo) else "partial" if done else "missing"


# --------------------------------------------------------------------------- the stage of a benchmark run
STAGE_DIR = "robustness"  # inside a benchmark run's folder: cover_audit.jsonl, cover_audit.csv, summary.json
_CHILD_GRAPHS: dict[tuple[str, str, str], Any] = {}  # the audit process keeps the graphs it has loaded between units


def progress_sink(path: Path, key: str) -> Callable[[dict[str, Any]], None]:
    """Write the audit's latest event to ``path`` (atomically, at most once a second): the worker is in another process."""
    since, last = time.time(), [0.0]

    def sink(event: dict[str, Any]) -> None:
        now = time.time()
        if event["type"] == "vertices" and now - last[0] < 1.0 and event["done"] < event["total"]:
            return
        last[0] = now
        try:
            ph._atomic_write(path, json.dumps({**event, "key": key, "since": since, "updated": now}).encode())
        except OSError:  # a progress line must never stop an audit
            pass

    return sink


def read_audit_progress(output: Path) -> dict[str, Any] | None:
    """The audit process's latest event in a stage folder, or None when there is none (or it is being replaced)."""
    try:
        info = json.loads((Path(output) / PROGRESS_NAME).read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def audit_in_child(plan: dict[str, Any], todo: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run in the audit process of a benchmark run: the records of these audits."""
    path = plan["output"] / PROGRESS_NAME
    records = []
    for condition in todo:
        try:
            records.append(audit_unit(condition, plan, _CHILD_GRAPHS, progress_sink(path, condition["key"])))
        finally:
            path.unlink(missing_ok=True)
    return records


class Stage:
    """The robustness stage of ``hedonic run exp --robustness``: each unit's saved cover is audited right after its
    detector run, so one run yields accuracy, time and robustness.

    The audit runs in one long-lived child process, never in the worker: the worker's lifetime peak RSS is what a
    benchmark record without ``--isolate`` reports as its memory, and the audit loads its own graph and cover. The
    worker waits for the child, so audits and detector runs never share the machine (the detector timings stay clean).
    Records go to ``<run>/robustness/cover_audit.jsonl``, the file ``hedonic run audit`` writes, so the two are
    interchangeable and a unit audited by either is skipped by the other.
    """

    def __init__(self, output_dir: str | Path, networks: list[str], methods: list[str], seeds: list[int],
                 network_root: str | Path | None = None, induce: str | None = None,
                 replication_mode: str = replication.DEFAULT_MODE):
        out = expand_path(output_dir).resolve()
        argv = ["--source", str(out), "--networks", ",".join(networks), "--methods", ",".join(methods),
                "--seeds", ",".join(str(seed) for seed in seeds), "--output-dir", str(out / STAGE_DIR),
                "--name", "robustness", "--replication", replication_mode]
        if network_root:
            argv += ["--network-root", str(network_root)]
        if induce:
            argv += ["--induce", induce]
        self.plan = resolve(argv, prog="hedonic run exp")
        self.results = self.plan["output"] / RESULTS_NAME
        self._pool = None
        self.on_progress: Callable[[dict[str, Any]], None] | None = None  # told the audit's latest event while the worker waits
        self.records = self.load()

    def load(self) -> dict[str, dict[str, Any]]:
        self.plan["output"].mkdir(parents=True, exist_ok=True)
        repair_tail(self.results)
        return read_records(self.results)

    def unit_conditions(self, network: str, seed: int, method: str) -> list[dict[str, Any]]:
        return [c for c in conditions(self.plan) if (c["network"], c["seed"], c["method"]) == (network, seed, method)]

    def audited(self, network: str, seed: int, method: str) -> bool:
        return all(c["key"] in self.records for c in self.unit_conditions(network, seed, method))

    def audit(self, network: str, seed: int, method: str) -> dict[str, float] | None:
        """Audit what this unit still lacks (every profile), and return its robustness measure (None without a cover)."""
        todo = [c for c in self.unit_conditions(network, seed, method) if c["key"] not in self.records]
        if todo:
            for record in self._in_child(todo):
                append_record(self.results, record)
                self.records[record["key"]] = record
        return self.measure(network, seed, method)

    def measure(self, network: str, seed: int, method: str) -> dict[str, float] | None:
        """The unit's saved-cover robustness under open labels: the log-averaged and the linear stable-vertex share."""
        record = self.records.get(key_of(network, seed, method, "cover"))
        if not record or record.get("status") != "completed":
            return None
        block = record["policies"][PRIMARY_POLICY]
        return {"log": block["stable_measure_log"], "linear": block["stable_measure_linear"]}

    def _in_child(self, todo: list[dict[str, Any]]) -> list[dict[str, Any]]:
        import concurrent.futures as futures
        import multiprocessing

        if self._pool is None:
            self._pool = futures.ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("fork"))
        try:
            future = self._pool.submit(audit_in_child, self.plan, todo)
            while not futures.wait([future], timeout=PROGRESS_POLL_SECONDS).done:
                info = read_audit_progress(self.plan["output"]) if self.on_progress else None
                if info is not None:
                    self.on_progress(info)
            return future.result()
        except futures.process.BrokenProcessPool:  # the audit process died (killed, out of memory): the next unit starts afresh
            self._pool = None
            raise

    def finish(self) -> None:
        """End the audit process and write the table and summary of what the ledger holds."""
        self.close()
        if self.records:
            write_outputs(self.plan, conditions(self.plan), self.records)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


# --------------------------------------------------------------------------- CLI entry points
def main(argv: list[str] | None = None) -> int:
    """``hedonic-exp overlapping-cover-audit``: foreground run; use ``hedonic run audit`` for a durable one."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        plan = resolve(argv, prog="hedonic-exp overlapping-cover-audit")
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic-exp overlapping-cover-audit: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        print("  dry run: no graph loaded, nothing written")
        return 0
    try:
        return run_audit(plan)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"hedonic-exp overlapping-cover-audit: {exc}", file=sys.stderr)
        return 1


def _front_parser() -> argparse.ArgumentParser:
    parser = build_parser("hedonic run audit")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    return parser


def front_study(argv: list[str]) -> dict[str, Any]:
    """The plan of ``hedonic run audit ARGV``, run name and folder included. Loads no graph and writes nothing."""
    argv = [arg for arg in argv if arg != "--detach"]
    return resolve(argv)


def launch_argv(plan: dict[str, Any]) -> list[str]:
    """Every setting spelled out, so a resume rebuilds the same plan."""
    return ["--source", *map(str, plan["sources"]), "--networks", ",".join(plan["networks"]),
            "--methods", ",".join(plan["methods"]), "--seeds", ",".join(map(str, plan["seeds"])),
            "--profiles", ",".join(plan["profiles"]), "--cap", str(plan["cap"]), "--workers", str(plan["workers"]),
            "--network-root", str(plan["network_root"]), "--name", plan["name"], "--output-dir", str(plan["output"]),
            "--replication", plan.get("replication", replication.DEFAULT_MODE),
            *(["--induce", plan["induce"]] if plan.get("induce") else [])]


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run audit``: a durable run; ``--dry-run`` only prints the plan."""
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _front_parser().parse_args(argv)
        plan = resolve([arg for arg in argv if arg != "--detach"])
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic run audit: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("\n".join(plan_lines(plan)))
        print(style("  dry run: nothing started, nothing written", "dim"))
        return 0
    if args.overwrite:
        try:
            runmanager._refuse_if_running(plan["name"])  # never delete records under a live worker
        except SystemExit as exc:
            print(f"hedonic run audit: {exc.code}", file=sys.stderr)
            return 2
        for name in (RESULTS_NAME, TABLE_NAME, SUMMARY_NAME):
            (plan["output"] / name).unlink(missing_ok=True)
    entry = runmanager.launch_audit(launch_argv(plan), plan["name"], str(plan["output"]))
    where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
             else f"background process (log {entry['log']})")
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


# --------------------------------------------------------------------------- durable run
def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body of an audit run: the audit with its progress mirrored into progress.json."""
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    plan = resolve(entry["argv"])
    todo = conditions(plan)
    results = plan["output"] / RESULTS_NAME
    already = sum(1 for condition in todo if condition["key"] in read_records(results))
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "audit", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": describe(plan), "command": entry["command"],
                 "environment": qs.environment(), "done": already, "total": len(todo), "resumed_from": already,
                 "eta_seconds": None, "stage": "starting the audit", "output": str(results), "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(plan))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    first_seen: dict[str, float] = {}

    def on_event(event: dict[str, Any]) -> None:
        if event["type"] in ("stage", "vertices"):
            prog.update(status="running", stage=event["text"], updated=time.time())
        elif event["type"] == "unit":
            first_seen.setdefault("at", time.time())
            fresh = event["done"] - prog["resumed_from"]
            elapsed = time.time() - first_seen["at"]
            eta = elapsed / (fresh - 1) * (event["total"] - event["done"]) if fresh > 1 else None
            prog.update(status="running", done=event["done"], total=event["total"], last=event["key"],
                        eta_seconds=eta, updated=time.time())
        rm.write_json(ppath, prog)

    prog.update(status="running", updated=time.time())
    rm.write_json(ppath, prog)
    code = run_audit(plan, progress=on_event)
    if code != 0:
        raise RuntimeError(f"the cover audit returned {code}")
    prog.update(status="completed", done=len(todo), eta_seconds=0, stage=None, finished=time.time(), updated=time.time())
    rm.write_json(ppath, prog)
    print(style(f"  wrote {results}", "green"), flush=True)
    return 0


def snapshot(entry: dict[str, Any], prog: dict[str, Any], frame: str) -> str:
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.tui import style

    status = rm.effective_status(entry, prog)
    colour = {"running": "cyan", "setup": "magenta", "completed": "green", "interrupted": "red",
              "stopped": "yellow", "failed": "red"}.get(status, "dim")
    done, total = int(prog.get("done", 0)), int(prog.get("total", 0))
    started = prog.get("started", entry["created"])
    end = prog.get("finished") or (time.time() if status in ("running", "setup", "starting")
                                   else prog.get("updated", time.time()))
    lines = [style(f"  hedonic · {entry['name']}", "bold") + "  " + style(f"● {status}", colour, "bold")
             + style(f"   ({entry['backend']}{': ' + entry['session'] if entry.get('session') else ''})", "dim"),
             style("  " + prog.get("describe", "cover audit"), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = f"  {done:,}/{total:,} audits · elapsed {qs.fmt_time(end - started)}"
        if status in ("running", "setup") and prog.get("eta_seconds") is not None:
            tail += f" · ETA ~{qs.fmt_time(prog['eta_seconds'])}"
        lines.append("  " + style("█" * filled, "green") + style("░" * (width - filled), "dim") + tail)
    else:
        lines.append(style(f"  preparing… elapsed {qs.fmt_time(end - started)}", "dim"))
    if status in ("running", "setup") and prog.get("stage"):
        lines.append(style(f"  {frame} {prog['stage']}", "magenta"))
    if status in ("running", "setup") and prog.get("last"):
        lines.append(style(f"  last finished: {prog['last']}", "dim"))
    if status == "failed":
        lines.append(style(f"\n  the run failed: {prog.get('reason', 'unknown error')}", "red"))
        lines.append(style(f"  details: hedonic run logs {entry['name']} · continue with: "
                           f"hedonic run resume {entry['name']}", "red"))
    elif status == "interrupted":
        lines.append(style(f"\n  the worker is gone (reboot or kill); restart with: hedonic run resume {entry['name']}",
                           "red"))
    elif status == "stopped":
        lines.append(style(f"\n  stopped by you; resume skips the audits already recorded: "
                           f"hedonic run resume {entry['name']}", "yellow"))
    if status == "completed":
        lines.append(style(f"\n  records  {prog.get('output', entry['output_dir'])}", "dim"))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
