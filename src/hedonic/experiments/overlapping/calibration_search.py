"""Theory-guided search of Hedonic's resolution and membership cap on one instance (the engine of ``hedonic run calibrate``).

The Hedonic method (local moving, isolation allowed, singleton start) has two settings that depend on the network: the
CPM resolution gamma and the per-vertex cap M. This module finds the F1-optimal pair on one instance with three
one-dimensional searches whose structure follows from the theory of the hedonic game:

1. **Resolution**, on ``u = log gamma``. Every label gain is affine in gamma, ``L_vc - gamma R_vc``, so each move switches
   sign at a Familiarity threshold (a local ratio in (0, 1)), and the accuracy profile is searched in log gamma: a scan at
   factor-2 spacing over ``[rho_in / 32, rho_in]`` (``rho_in`` the median internal density of the reference
   communities), a gallop outward while the best probe is the outermost one (within ``[rho, 0.9]``, stopping on a
   plateau), then a bracketed golden-section search until the bracket is narrower than ``--tol-u``. With ``--fixed-M`` (the
   resolution only, as the paper calibrates Hedonic at the unlimited cap -1) the scan covers ``[1e-6, 0.5]`` at
   factor-4 spacing instead, because the profile can peak far below the density. The clique regime
   ``gamma -> 1`` is probed once on its own (``EXTREMAL``), because the profile can have a second plateau there.
2. **Cap**, on ``log2 M``: gallop up while doubling gains at least ``--delta`` (or until the returned covers certify that
   the cap no longer binds: no vertex at the cap and every vertex at an uncapped best response), gallop down while
   halving stays within ``--delta`` of the best, then binary search for the smallest cap within ``--delta``. Cap and
   resolution are then re-centred alternately (cyclic coordinate ascent, at most ``--rounds`` rounds).
3. **Rational snap.** The final resolution is the simplest fraction ``b/c`` inside the final golden-section bracket
   (Stern-Brocot), kept when it scores within ``--delta`` of the best probe.

The setting is then confirmed on ``--confirm-seeds`` (medians reported), and when those seeds disagree by more than
``5 delta`` a branch check scores neighbouring simple fractions and a larger cap on every confirmation seed and keeps the
best median among the candidates whose seeds agree.

Every probe is one detector call, ``Game(graph).community_hedonic(None, max_memberships=M, resolution=gamma,
allow_isolation=True, seed=seed)``, run in a forked child (a crash or a ``--timeout`` cannot corrupt the search), scored
exactly as the benchmark scores the Hedonic method (the prediction intersected with the reference-covered vertices,
``evaluate_cover`` and LFK-ONMI), and audited without labels (``calibration_audit.CoverAudit``: maximum regret, cap
saturation, the uncapped certificate, stability on ``[gamma/2, 2 gamma]``).

Instances. ``network`` is the whole graph, scored against its ``all`` cover (the benchmark's default); ``network:cover``
(``dblp:top5000``) is the subgraph induced by that cover's vertices, scored against that cover: the graph a ``--induce
covered`` benchmark run hands to every method, loaded the same way (``codeseg_reproduction._load_dataset`` and the
runner's undirected simple projection), so the setting found is the setting for that run's units. A pair missing from
the networks folder is downloaded from SNAP first (``snap.ensure_pair``) unless ``--no-download`` is given.

Ledger. Every evaluation is appended to ``OUT/evals-<tag>.jsonl`` (one search per tag; a torn last line left by a kill is
cut before appending) and every finished search to ``OUT/searches.jsonl``. Evaluations are memoized by configuration
(instance, phase, gamma at 6 significant digits, cap, seed) across every ``evals*.jsonl`` of the folder, so a stopped
search resumes where it was and two searches of one instance share their probes. A record carries the build it was made
on (hedonic version and the SHA-256 of the compiled core); records of another build are never reused.

Cores. A search takes its steps one at a time, but the evaluations that do not depend on one another (the points of a
scan, the first cap probes, the confirmation seeds, the branch-check candidates) run side by side, at most ``--jobs`` of
them, each holding a slot of ``--slot-dir`` (a core budget shared with the other searches of a calibration) while it
runs. Every record is a deterministic function of its configuration, so ``--jobs`` changes the wall-clock only.

    python -m hedonic.experiments.overlapping.calibration_search tune --instance dblp:top5000 --out RUN/ledger
    python -m hedonic.experiments.overlapping.calibration_search probe --instance dblp:top5000 --gamma 0.1 --M 8 --out D
    python -m hedonic.experiments.overlapping.calibration_search grid --instance amazon --gammas 1/7,1/11 --M 4 --out D
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import multiprocessing as mp
import os
import pickle
import resource
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from hedonic import Game
from hedonic.experiments.overlapping import limits
from hedonic.experiments.overlapping import parallel as par
from hedonic.experiments.overlapping.calibration_audit import CoverAudit, active_density, internal_densities

PHI = (math.sqrt(5.0) - 1.0) / 2.0
EXTREMAL = 0.999  # clique regime: every frustrated move resolves against strangers
PHASE = "local"  # the only phase hedonic 2.0 has
# The resolution-only search (``--fixed-M``, the paper's unlimited cap -1): a log grid at factor-4 spacing over
# [1e-6, 0.5], then golden section around the best point. On the SNAP top5000 pairs the F1 profile has a peak near the
# reference density and another two to three decades below it (YouTube: 0.434 at 0.065, 0.508 at 3e-5), which the
# default bracket [rho_in/32, rho_in] misses.
WIDE_SCAN = (1e-6, 0.5)
WIDE_SCAN_FACTOR = 4.0
WIDE_SCAN_EVALS = 20
MULTILEVEL_GONE = ("the multilevel phase is not available in hedonic 2.0 (overlapping multilevel was removed; "
                   "only local moving remains): use --phase local")

# Module state of one search process: the ledger, the loaded instances, the evaluation cache and the core budget.
OUT: Path | None = None
EVALS: Path | None = None
SEARCHES: Path | None = None
INST: dict[str, dict] = {}
CACHE: dict[str, dict] = {}
JOBS = 1  # evaluations of this search that may run side by side (the seeds of a probe, the points of a scan)
SLOTS: "par.Slots | None" = None  # core budget shared with the other searches running at the same time
# Where SNAP data comes from: None means the package defaults (``snap.networks_root()``, ``snap.default_cache_dir()``).
NETWORK_ROOT: Path | None = None
SNAP_CACHE_DIR: Path | None = None
DOWNLOAD = True


def configure(out: str | Path, *, tag: str = "main", jobs: int = 1, slot_dir: str | Path | None = None,
              slot_count: int | None = None, network_root: str | Path | None = None,
              snap_cache_dir: str | Path | None = None, download: bool = True) -> None:
    """Point this process at a ledger folder and a data source (``main`` does this from its flags)."""
    global OUT, EVALS, SEARCHES, JOBS, SLOTS, NETWORK_ROOT, SNAP_CACHE_DIR, DOWNLOAD
    OUT = Path(out).expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    EVALS = OUT / f"evals-{tag}.jsonl"
    SEARCHES = OUT / "searches.jsonl"
    JOBS = max(1, int(jobs))
    SLOTS = par.Slots(Path(slot_dir).expanduser(), slot_count or JOBS) if slot_dir else None
    NETWORK_ROOT = Path(network_root).expanduser() if network_root else None
    SNAP_CACHE_DIR = Path(snap_cache_dir).expanduser() if snap_cache_dir else None
    DOWNLOAD = bool(download)
    CACHE.clear()


DEFAULT_TIMEOUT = 1800.0  # seconds one detector run may take (``--timeout``); a queue's limit lowers it


def build() -> dict[str, str]:
    """The build an evaluation is made on: two records are interchangeable only when this is equal."""
    from hedonic.experiments.runtime import build_identity

    return build_identity()


# ---------------------------------------------------------------- instances
def parse_instance(spec: str) -> tuple[str, str | None]:
    """``dblp`` -> (dblp, None): the whole graph; ``dblp:top5k`` -> (dblp, top5000): the covered-induced graph."""
    from hedonic.experiments.overlapping import snap

    text = str(spec).strip().lower()
    if "@" in text or not text or text.count(":") > 1:
        raise ValueError(f"bad instance {spec!r}: give a SNAP network (dblp) or a network:cover pair (dblp:top5000); "
                         "the derived research instances (snowball @NsS, thinning pP, disjoint union @NuK) are not part "
                         "of the packaged search")
    network, _, cover = text.partition(":")
    if network not in snap.SPECS:
        raise ValueError(f"unknown network {network!r} in instance {spec!r}; choose from {', '.join(snap.SPECS)}")
    if not cover:
        return network, None
    cover = snap.canonical_cover_variant(cover)
    if cover not in snap.SPECS[network].raw_cover_files:
        covers = ", ".join(snap.cover_variant_label(c) for c in snap.SPECS[network].raw_cover_files)
        raise ValueError(f"{network} has no cover {snap.cover_variant_label(cover)!r}; its covers: {covers}")
    return network, cover


def _snap_instance(spec: str) -> dict:
    """The graph and reference cover a benchmark run of this target hands to the Hedonic method, downloaded if absent."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr
    from hedonic.experiments.overlapping import snap

    network, cover = parse_instance(spec)
    root = Path(NETWORK_ROOT) if NETWORK_ROOT is not None else Path(snap.networks_root())
    snap.ensure_pair(network, cover or "all", root, download=DOWNLOAD, say=lambda line: print(f"  {line}", flush=True))
    dataset = cr._load_dataset(network, cover_variant=cover or "all", network_root=root, max_nodes=None, smoke=False,
                               snap_cache_dir=SNAP_CACHE_DIR, allow_catalog=False,
                               induce="covered" if cover else "full")
    gt, _gt_nodes, _report = cr._cover_for_paper_metrics(dataset)
    graph = cr._undirected_graph(dataset.graph)  # the projection the runner applies before every non-CoDeSEG method
    report = {k: dataset.report.get(k) for k in ("n", "m", "directed", "id_mapping_strategy") if k in dataset.report}
    del dataset
    return _finish_instance(spec, network, graph, [list(c) for c in gt], cover=cover,
                            induce="covered" if cover else "full", report=report)


def register_instance(spec: str, graph, gt, *, dataset: str | None = None) -> dict:
    """Use ``graph`` (undirected; loops and parallel edges are removed) and its reference cover ``gt`` as instance
    ``spec`` of this process, instead of loading a SNAP network: for tests and for calibrating a graph of one's own."""
    if graph.is_directed():
        raise ValueError("the instance graph must be undirected")
    graph = graph.copy()
    graph.simplify(multiple=True, loops=True, combine_edges=None)
    return _finish_instance(spec, dataset or spec, graph, [sorted({int(v) for v in c}) for c in gt if len(c)],
                            cover=None, induce="given")


def load_instance(spec: str) -> dict:
    if spec not in INST:
        _snap_instance(spec)
    return INST[spec]


def _finish_instance(spec, dataset, graph, gt, *, cover, induce, report=None) -> dict:
    if not gt:
        raise ValueError(f"instance {spec!r} has an empty reference cover")
    game = graph if isinstance(graph, Game) else Game(graph)  # built once; the forked detector children share it
    gt_nodes = {v for c in gt for v in c}
    rho, n_act = active_density(game)
    dens = internal_densities(game, gt)
    inst = dict(spec=spec, dataset=dataset, cover=cover, induce=induce, graph=game, gt=gt, gt_nodes=gt_nodes, rho=rho,
                n=game.vcount(), n_active=n_act, m=game.ecount(), ref_count=len(gt),
                rho_in=float(np.median(dens)), rho_in_mean=float(dens.mean()), report=report or {})
    INST[spec] = inst
    return inst


def instance_summary(inst: dict) -> dict:
    return {k: inst[k] for k in ("spec", "dataset", "cover", "induce", "rho", "n", "n_active", "m", "ref_count",
                                 "rho_in", "rho_in_mean")}


# --------------------------------------------------------------- evaluation
def _child(spec: str, cfg: dict, path: str) -> None:
    game = INST[spec]["graph"]
    try:
        t0 = time.perf_counter()
        game.community_hedonic(None, max_memberships=int(cfg["M"]), resolution=float(cfg["gamma"]),
                               allow_isolation=True, seed=int(cfg["seed"]))
        det = time.perf_counter() - t0
        rows = [list(map(int, r)) for r in game.memberships]  # the labelled profile the detector reached
        out = {"status": "completed", "rows": rows, "detection_seconds": det,
               "child_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except Exception as exc:  # recorded, never scored
        out = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    with open(path, "wb") as fh:
        pickle.dump(out, fh)


def _rows_to_cover(rows):
    comms: dict[int, list[int]] = {}
    for v, labels in enumerate(rows):
        for c in labels:
            comms.setdefault(c, []).append(v)
    return list(comms.values())


def score(inst: dict, rows) -> dict:
    """The benchmark's scores of this cover: filtered to the reference-covered vertices, then F1, ONMI and the rest."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr
    from hedonic.metrics import evaluate_cover, overlapping_normalized_mutual_information_lfk

    pred = cr._filter_prediction(_rows_to_cover(rows), gt_nodes=inst["gt_nodes"], n_vertices=inst["n"])
    acc = evaluate_cover(pred, inst["gt"], inst["n"], compute_omega=False)
    acc["onmi"] = overlapping_normalized_mutual_information_lfk(pred, inst["gt"])
    keep = ("f1", "onmi", "matching_f1", "node_micro_f1", "size_weighted_community_f1",
            "precision", "recall", "predicted_community_count",
            "predicted_overlapping_node_fraction",
            "predicted_average_memberships_per_covered_vertex",
            "predicted_median_community_size", "predicted_max_community_size")
    return {k: acc[k] for k in keep}


def label_free(inst: dict, rows, gamma: float, M: int) -> dict:
    if M < 1:  # -1: no cap, the game's cap is the vertex count
        M = len(rows)
    k = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
    act = np.asarray(inst["graph"].degree()) > 0
    au = CoverAudit(inst["graph"], rows)
    at = au.evaluate(gamma, M)
    unc = au.stable(gamma, None)
    sizes = np.array([len(c) for c in _rows_to_cover(rows)])
    return {
        "communities": int(len(sizes)),
        "median_size_all": float(np.median(sizes)),
        "mean_k": float(k[act].mean()), "max_k": int(k.max()),
        "saturated_fraction": float((k[act] >= M).mean()),
        "max_regret": float(at["regret"].max()),
        "uncapped_stable_fraction": float(unc[act].mean()),
        "robust_sqrt2": au.window_robust_fraction(gamma, math.sqrt(2.0), M, vertices=act),
        "robust_2": au.window_robust_fraction(gamma, 2.0, M, vertices=act),
    }


def _normal(cfg: dict) -> dict:
    return {**cfg, "gamma": float(f"{cfg['gamma']:.6g}")}


def cfg_key(spec: str, cfg: dict) -> str:
    return json.dumps({"instance": spec, **_normal(cfg)}, sort_keys=True)


def _cache() -> dict:
    """Every record of this build in the ledger folder (a record of another build, or of none, is skipped). A torn line (a write cut short by a kill) is skipped, not fatal:
    the other searches of a concurrent calibration are appending to their own files in the same folder."""
    cache, other = {}, 0
    mine = build()
    for path in sorted(OUT.glob("evals*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and "key" in r:
                if r.get("build") != mine:  # a record without a build was made by 1.x: never this build's
                    other += 1
                    continue
                cache[r["key"]] = r
    if other:
        print(f"  ledger: ignored {other} evaluations made by another build", flush=True)
    return cache


def _repair_tail(path: Path) -> None:
    """Cut a torn last line (a write interrupted by a kill) so the next record starts on a fresh line. Only the file
    this search owns: another search of the same folder may be in the middle of appending to its own."""
    if path.is_file():
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            with path.open("r+b") as fh:
                fh.truncate(raw.rfind(b"\n") + 1)


def _held():
    """One unit of the shared core budget (a detector run plus its scoring and audit), or nothing when unshared."""
    return SLOTS.hold() if SLOTS is not None else contextlib.nullcontext()


def _compute(spec: str, cfg: dict, timeout: float) -> dict:
    """One detector run plus scoring and label-free diagnostics. Touches neither the cache nor the ledger, so it can
    run in a forked worker."""
    key = cfg_key(spec, cfg)
    inst = load_instance(spec)
    fd, path = tempfile.mkstemp(suffix=".pkl"); os.close(fd)
    proc = mp.get_context("fork").Process(target=_child, args=(spec, cfg, path))
    t0 = time.perf_counter(); proc.start(); proc.join(timeout)
    if proc.is_alive():
        proc.kill(); proc.join()
        # ``limit_seconds``: a larger timeout later runs this configuration again (``limits.stands``)
        out = {"status": "timeout", "wall_seconds": time.perf_counter() - t0, "limit_seconds": float(timeout)}
    else:
        try:
            with open(path, "rb") as fh:
                out = pickle.load(fh)
        except Exception:
            out = {"status": "failed", "reason": f"child exit {proc.exitcode}"}
    os.unlink(path)
    rec = {"key": key, "instance": spec, "config": cfg, "rho": inst["rho"],
           "rho_in": inst["rho_in"], "c": cfg["gamma"] / inst["rho"],
           "t": 1.0 - math.log(cfg["gamma"]) / math.log(inst["rho"]),
           "gamma_over_rho_in": cfg["gamma"] / inst["rho_in"], "build": build(),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **out}
    if rec["status"] == "completed":
        rows = rec.pop("rows")
        t1 = time.perf_counter(); rec["metrics"] = score(inst, rows)
        rec["scoring_seconds"] = time.perf_counter() - t1
        t1 = time.perf_counter(); rec["label_free"] = label_free(inst, rows, cfg["gamma"], int(cfg["M"]))
        rec["audit_seconds"] = time.perf_counter() - t1
    return rec


def _record(cfg: dict, rec: dict) -> None:
    """Append one finished evaluation to this search's ledger and cache (the process that owns the ledger only)."""
    OUT.mkdir(parents=True, exist_ok=True)
    with EVALS.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    CACHE[rec["key"]] = rec
    spec = rec["instance"]
    m = rec.get("metrics", {}); lf = rec.get("label_free", {})
    print(f"  [{spec} {cfg['phase']} M={cfg['M']} gamma={cfg['gamma']:.4g} "
          f"(c={rec['c']:.3g}, t={rec['t']:.3f}) seed={cfg['seed']}] {rec['status']} "
          f"det={rec.get('detection_seconds', 0):.1f}s F1={m.get('f1', float('nan')):.4f} "
          f"ONMI={m.get('onmi', float('nan')):.4f} K={m.get('predicted_community_count')} "
          f"sat={lf.get('saturated_fraction', float('nan')):.2e} "
          f"rob2={lf.get('robust_2', float('nan')):.3f}", flush=True)


def evaluate_many(spec: str, cfgs: list[dict], *, timeout: float, objective: str) -> list[dict]:
    """The records of these configurations (memoized), in order. The ones not yet evaluated run side by side, at most
    JOBS of them (and, with a shared core budget, only while a slot is free). Each is a deterministic function of its
    configuration, so how many run together changes the wall-clock only, never a record's content."""
    if OUT is None:
        raise RuntimeError("no ledger folder: call configure(out) first")
    if not CACHE:
        _repair_tail(EVALS)
        CACHE.update(_cache())
    cfgs = [_normal(c) for c in cfgs]
    todo: dict[str, dict] = {}
    for cfg in cfgs:
        key = cfg_key(spec, cfg)
        # A run stopped at a timeout answers only under a timeout no larger than that one: one detector run each
        known = key in CACHE and limits.stands(CACHE[key], timeout, default=DEFAULT_TIMEOUT)
        if not known and key not in todo:
            todo[key] = cfg
    if todo:
        load_instance(spec)  # in this process, before any fork, so the workers share it
        pending = list(todo.values())
        if len(pending) > 1 and JOBS > 1:
            def task(cfg: dict) -> dict:
                with _held():
                    return _compute(spec, cfg, timeout)
            for index, rec in par.run_forked(task, pending, JOBS):
                _record(pending[index], rec)
        else:
            for cfg in pending:
                with _held():
                    rec = _compute(spec, cfg, timeout)
                _record(cfg, rec)
    return [CACHE[cfg_key(spec, cfg)] for cfg in cfgs]


def evaluate(spec: str, cfg: dict, *, timeout: float, objective: str) -> dict:
    """One detector run (memoized) plus scoring and label-free diagnostics."""
    return evaluate_many(spec, [cfg], timeout=timeout, objective=objective)[0]


def append_line(path: Path, text: str) -> None:
    """Append one line in a single write under an exclusive lock: the concurrent searches of a calibration share one
    searches.jsonl, and a record can be tens of kilobytes."""
    data = (text + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def objective_value(rec: dict, objective: str) -> float:
    if rec["status"] != "completed":
        return -math.inf
    m = rec["metrics"]
    return m["f1"] + m["onmi"] if objective == "f1+onmi" else m[objective]


class Probe:
    """Objective of a (possibly multi-seed) configuration, memoized."""

    def __init__(self, spec, base_cfg, seeds, *, timeout, objective):
        self.spec, self.base, self.seeds = spec, base_cfg, seeds
        self.timeout, self.objective = timeout, objective
        self.trace: list[dict] = []

    def __call__(self, **over) -> float:
        recs = evaluate_many(self.spec, [{**self.base, **over, "seed": s} for s in self.seeds],
                             timeout=self.timeout, objective=self.objective)
        vals = [objective_value(rec, self.objective) for rec in recs]
        val = float(np.mean(vals)) if all(np.isfinite(vals)) else -math.inf
        self.trace.append({**over, "J": val,
                           "f1": float(np.mean([r["metrics"]["f1"] for r in recs if "metrics" in r] or [np.nan])),
                           "onmi": float(np.mean([r["metrics"]["onmi"] for r in recs if "metrics" in r] or [np.nan])),
                           "status": [r["status"] for r in recs]})
        return val

    def prefetch(self, overs: list[dict]) -> None:
        """Evaluate these configurations side by side now, so the sequential search finds them in the cache. It only
        fills the cache: the search still asks for each point in its own order, so its decisions and trace are those of
        a one-at-a-time run."""
        evaluate_many(self.spec, [{**self.base, **over, "seed": s} for over in overs for s in self.seeds],
                      timeout=self.timeout, objective=self.objective)


# ------------------------------------------------------------------ searches
def golden_max_log(f, lo: float, hi: float, *, tol_u: float, max_evals: int,
                   floor: float, ceil: float, plateau: float = 0.0, prefetch=None, scan_factor: float = 2.0) -> dict:
    """Maximize f(gamma) over u = log gamma, allowing a multimodal profile.

    1. Scan: probes at factor-2 spacing (``scan_factor``) from lo to hi (log grid), because the profile has a Familiarity peak and a
       separate extremal plateau as gamma -> 1 (clique regime), which golden section alone can confuse.
    2. Gallop: while the best probe is the outermost one on either side, add a probe a factor 2 further out, within
       [floor, ceil]; stop when that step gains less than ``plateau`` (below the smallest Familiarity threshold, or above
       the largest, the dynamics no longer change).
    3. Bisection: bracketed golden-section search on (left neighbour, best, right neighbour) until the bracket is
       narrower than tol_u in log gamma.
    The extremal probe gamma = 0.999 is evaluated by the caller.

    ``prefetch(gammas)``, when given, evaluates the scan points side by side before the loop asks for them one by one
    (they are independent); it only warms the cache, so the search takes the same steps."""
    memo: dict[float, float] = {}

    def F(u):
        u = round(u, 9)
        if u not in memo:
            memo[u] = f(gamma=math.exp(u))
        return memo[u]

    uf, uc = math.log(floor), math.log(ceil)
    a0, b0 = math.log(lo), math.log(hi)
    k = max(2, math.ceil((b0 - a0) / math.log(scan_factor)))
    if prefetch is not None:
        prefetch([math.exp(round(a0 + (b0 - a0) * i / k, 9)) for i in range(k + 1)])
    for i in range(k + 1):
        F(a0 + (b0 - a0) * i / k)
    step = math.log(2.0)
    while len(memo) < max_evals:
        us = sorted(memo); best = max(us, key=lambda u: memo[u])
        if best == us[0] and us[0] > uf + 1e-9:
            prev = memo[round(us[0], 9)]; F(max(us[0] - step, uf))
            new = sorted(memo)[0]
        elif best == us[-1] and us[-1] < uc - 1e-9:
            prev = memo[round(us[-1], 9)]; F(min(us[-1] + step, uc))
            new = sorted(memo)[-1]
        else:
            break
        if memo[round(new, 9)] - prev < plateau:
            break
    us = sorted(memo); m = max(us, key=lambda u: memo[u]); i = us.index(m)
    a = us[i - 1] if i > 0 else m
    b = us[i + 1] if i + 1 < len(us) else m
    while (b - a) > tol_u and len(memo) < max_evals:
        if (m - a) >= (b - m):
            x = m - (1.0 - PHI) * (m - a)
        else:
            x = m + (1.0 - PHI) * (b - m)
        fx = F(x)
        if fx > memo[round(m, 9)]:
            if x < m:
                b = m
            else:
                a = m
            m = x
        elif x < m:
            a = x
        else:
            b = x
    best_u = max(memo, key=memo.get)
    return {"gamma": math.exp(best_u), "J": memo[best_u], "bracket": [math.exp(a), math.exp(b)],
            "evaluations": len(memo), "points": sorted((math.exp(u), v) for u, v in memo.items())}


def simplest_rational(lo: float, hi: float) -> Fraction:
    """Smallest-denominator fraction in [lo, hi] (0 < lo <= hi): the first Stern-Brocot node that a bisection toward the
    interval lands in."""
    if not 0 < lo <= hi:
        raise ValueError("simplest_rational needs 0 < lo <= hi")

    def rec(lo: Fraction, hi: Fraction) -> Fraction:
        fl = math.floor(lo)
        if lo == fl or fl + 1 <= hi:
            return Fraction(math.ceil(lo))
        return fl + 1 / rec(1 / (hi - fl), 1 / (lo - fl))
    return rec(Fraction(lo), Fraction(hi))


def cap_search(probe: Probe, gamma: float, M0: int, M_max: int, delta: float) -> dict:
    """Bidirectional galloping on log2 M, then binary search for the smallest cap whose score is within delta of the
    best cap seen.

    Upward galloping stops when doubling gains less than delta or when the returned covers carry the uncapped
    certificate (no vertex at the cap and every vertex at an uncapped best response: the state is an equilibrium for
    every larger cap). Downward galloping continues while halving stays within delta of the best, because the cap can
    also act as a regularizer."""
    J: dict[int, float] = {}

    def at(M):
        if M not in J:
            J[M] = probe(gamma=gamma, M=M)
        return J[M]

    def certified(M):  # the cap no longer binds for any seed of the probe
        recs = [CACHE.get(cfg_key(probe.spec, {**probe.base, "gamma": gamma, "M": M, "seed": s})) for s in probe.seeds]
        return all(r and r.get("label_free", {}).get("saturated_fraction", 1) == 0
                   and r["label_free"].get("uncapped_stable_fraction", 0) >= 1.0 for r in recs)

    reasons = []
    first = [M0] + ([2 * M0] if 2 * M0 <= M_max else []) + ([M0 // 2] if M0 // 2 >= 1 else [])
    probe.prefetch([{"gamma": gamma, "M": m} for m in first])  # the gallop asks for these first, in this order
    M = M0; at(M)
    while M * 2 <= M_max:  # gallop up while doubling still pays
        if certified(M):
            reasons.append(f"up: cap certificate at M={M}"); break
        at(2 * M)
        if J[2 * M] - J[M] < delta:
            reasons.append(f"up: J(2M)-J(M) < delta at M={M}"); break
        M *= 2
    else:
        reasons.append(f"up: reached M_max={M_max}")
    M = M0
    while M // 2 >= 1:  # gallop down while halving stays within delta of the best
        at(M // 2)
        if J[M // 2] < max(J.values()) - delta:
            reasons.append(f"down: J({M // 2}) below best - delta"); break
        M //= 2
    target = max(J.values()) - delta
    good = min(m for m in J if J[m] >= target)
    bad = max([m for m in J if m < good], default=None)
    if bad is not None:
        a, b = bad, good
        while b - a > max(1, a // 8):
            mid = (a + b) // 2
            if at(mid) >= target:
                b = mid
            else:
                a = mid
        good = b
    return {"M": good, "J": J[good], "gallop_stop": "; ".join(reasons), "evaluated": dict(sorted(J.items()))}


def tune(spec: str, phase: str = PHASE, *, M0: int, M_max: int, seeds: list[int], objective: str,
         timeout: float, delta: float, tol_u: float, max_evals: int, confirm_seeds: list[int],
         gamma_bracket: tuple[float, float] | None = None, fixed_M: int | None = None,
         extremal: bool = True, rounds: int = 3, M_start: int | None = None) -> dict:
    if phase != PHASE:
        raise ValueError(MULTILEVEL_GONE if phase == "multilevel" else f"unknown phase {phase!r}: use local")
    inst = load_instance(spec)
    print(f"== tune {spec} phase={phase}: rho={inst['rho']:.3e} rho_in={inst['rho_in']:.3f} "
          f"n={inst['n_active']} m={inst['m']}", flush=True)
    t0 = time.time()
    base = {"phase": phase}
    probe = Probe(spec, base, seeds, timeout=timeout, objective=objective)
    lo, hi = gamma_bracket or (inst["rho_in"] / 32.0, inst["rho_in"])
    floor, ceil = inst["rho"], 0.9
    scan_factor = 2.0
    if fixed_M is not None and gamma_bracket is None:
        # resolution only: the accuracy profile can have two separate peaks (one far below the density), so scan the
        # whole range on a coarse log grid first and refine around the best point
        lo, hi = WIDE_SCAN
        floor, scan_factor, max_evals = min(floor, lo / 10.0), WIDE_SCAN_FACTOR, max(max_evals, WIDE_SCAN_EVALS)
    M = fixed_M or min(M_start or M0 * 4, M_max)  # the search never chooses a cap above M_max
    # stage 1: resolution at a generous cap (interior regime), plus the extremal clique regime gamma -> 1 as one
    # separate probe
    J_ext = probe(gamma=EXTREMAL, M=M) if extremal else float("nan")
    g1 = golden_max_log(lambda gamma: probe(gamma=gamma, M=M), lo, hi, tol_u=tol_u,
                        max_evals=max_evals, floor=floor, ceil=ceil, plateau=delta / 2, scan_factor=scan_factor,
                        prefetch=lambda gammas: probe.prefetch([{"gamma": g, "M": M} for g in gammas]))
    stages = {"gamma_at_M": {"M": M, **g1}, "extremal": {"gamma": EXTREMAL, "M": M, "J": J_ext,
                                                        "beats_interior": bool(J_ext > g1["J"] + delta)}}
    gamma = g1["gamma"]
    if fixed_M is None:
        # stages 2-3, repeated (cyclic coordinate ascent on the (log gamma, log2 M) ridge): cap at the current
        # resolution, then re-centre the resolution at the chosen cap, until neither moves
        stages["rounds"] = []
        M_cur = M
        for rnd in range(rounds):
            # start at the current cap (stage 1's generous cap in round 0): the cap profile need not be monotone, so
            # never start below a known good cap
            cs = cap_search(probe, gamma, M_cur, M_max, delta)
            entry = {"round": rnd, "cap": cs}
            if rnd and cs["M"] == M_cur:
                stages["rounds"].append(entry); break
            M_cur = cs["M"]
            g3 = golden_max_log(lambda gamma: probe(gamma=gamma, M=M_cur), gamma / 1.5, gamma * 1.5,
                                tol_u=tol_u, max_evals=max_evals, floor=floor, ceil=ceil,
                                plateau=delta / 2,
                                prefetch=lambda gammas, Mc=M_cur: probe.prefetch(
                                    [{"gamma": g, "M": Mc} for g in gammas]))
            entry["gamma"] = g3
            stages["rounds"].append(entry)
            moved = abs(math.log(g3["gamma"] / gamma)) > tol_u
            gamma = g3["gamma"]
            stages["gamma_at_chosen_M"] = {"M": M_cur, **g3}
            if not moved and rnd:
                break
        stages["cap"] = stages["rounds"][-1]["cap"]
        M = M_cur
    # stage 4: rational snapping inside the final golden-section bracket (where the optimum is localized to tol_u), not
    # the whole delta-tie set, which is too wide on flat profiles
    last = stages.get("gamma_at_chosen_M", stages["gamma_at_M"])
    pts = last["points"]
    best = max(v for _, v in pts)
    snap_lo, snap_hi = last["bracket"]
    snap_lo = min(snap_lo, gamma * math.exp(-tol_u / 2)); snap_hi = max(snap_hi, gamma * math.exp(tol_u / 2))
    frac = simplest_rational(snap_lo, snap_hi)
    snapped = float(frac)
    J_snap = probe(gamma=snapped, M=M)
    final_gamma = snapped if J_snap >= best - delta else gamma
    # stage 5: seeded confirmation of the final setting
    conf = Probe(spec, base, confirm_seeds, timeout=timeout, objective=objective)
    conf(gamma=final_gamma, M=M)
    recs = [CACHE[cfg_key(spec, {**base, "gamma": final_gamma, "M": M, "seed": s})] for s in confirm_seeds]
    done = [r for r in recs if r["status"] == "completed"]
    # stage 6: branch check. When the confirmation seeds disagree by more than 5 delta (the output switches between
    # branches across seeds, so a single seed per probe overfits), evaluate the neighbouring simple fractions and a
    # larger cap with every confirmation seed and keep the best median among candidates whose seeds agree.
    branch = None
    f1s = [r["metrics"]["f1"] for r in done]
    if f1s and max(f1s) - min(f1s) > 5 * delta:
        cands = {(final_gamma, M)}
        for fac in (0.92, 1.0, 1.09):
            g = gamma * fac
            fr = simplest_rational(g * 0.97, g * 1.03)
            for Mc in (M, min(int(math.ceil(1.5 * M)), max(M, M_max))):
                cands.add((float(fr), Mc))
        scored = []
        conf.prefetch([{"gamma": g, "M": Mc} for g, Mc in sorted(cands)])  # independent: every candidate x seed together
        for g, Mc in sorted(cands):
            conf(gamma=g, M=Mc)
            rs = [CACHE[cfg_key(spec, {**base, "gamma": g, "M": Mc, "seed": sd})] for sd in confirm_seeds]
            v = [r["metrics"]["f1"] for r in rs if r["status"] == "completed"]
            if v:
                scored.append({"gamma": g, "M": Mc, "median": float(np.median(v)), "spread": max(v) - min(v)})
        if scored:
            agree = [c for c in scored if c["spread"] <= 5 * delta] or scored
            pick = max(agree, key=lambda c: c["median"])
            branch = {"triggered_spread": max(f1s) - min(f1s), "candidates": scored, "pick": pick}
            final_gamma, M = pick["gamma"], pick["M"]
            recs = [CACHE[cfg_key(spec, {**base, "gamma": final_gamma, "M": M, "seed": sd})] for sd in confirm_seeds]
            done = [r for r in recs if r["status"] == "completed"]
    med = lambda key: float(np.median([r["metrics"][key] for r in done])) if done else None  # noqa: E731
    summary = {
        "kind": "tune", "instance": instance_summary(inst), "phase": phase, "objective": objective,
        "seeds_per_probe": seeds, "delta": delta, "tol_u": tol_u, "stages": stages,
        "snap": {"bracket": [snap_lo, snap_hi], "fraction": f"{frac.numerator}/{frac.denominator}",
                 "J": J_snap, "kept": final_gamma == snapped},
        "branch_check": branch,
        "final": {"gamma": final_gamma, "M": M, "c": final_gamma / inst["rho"],
                  "t": 1 - math.log(final_gamma) / math.log(inst["rho"]),
                  "gamma_over_rho_in": final_gamma / inst["rho_in"],
                  "confirm_seeds": confirm_seeds, "completed": len(done),
                  "f1_median": med("f1"), "onmi_median": med("onmi"),
                  "matching_f1_median": med("matching_f1"),
                  "f1_sd": float(np.std([r["metrics"]["f1"] for r in done])) if done else None,
                  "detection_median": float(np.median([r["detection_seconds"] for r in done])) if done else None,
                  "K_median": med("predicted_community_count")},
        "detector_evaluations": sum(1 for r in CACHE.values() if r["instance"] == spec
                                    and r["config"]["phase"] == phase),
        "probe_trace": probe.trace, "wall_hours": (time.time() - t0) / 3600.0, "build": build(),
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    append_line(SEARCHES, json.dumps(summary, default=str))
    f = summary["final"]
    fr = Fraction(f["gamma"]).limit_denominator(100000)
    print(f"== {spec} {phase}: gamma*={f['gamma']:.4g} ({fr.numerator}/{fr.denominator}, c={f['c']:.0f}, "
          f"t={f['t']:.3f}, gamma/rho_in={f['gamma_over_rho_in']:.3f}) M*={M} "
          f"F1={f['f1_median']} ONMI={f['onmi_median']}", flush=True)
    return summary


# ------------------------------------------------------------------ command line
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m hedonic.experiments.overlapping.calibration_search",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["tune", "probe", "grid"],
                    help="tune: the whole search; probe: one evaluation (--gamma, --M); grid: --gammas at one --M")
    ap.add_argument("--instance", required=True,
                    help="a SNAP network (the whole graph, scored against its all cover) or network:cover (dblp:top5000: "
                         "the graph induced by that cover's vertices, scored against it)")
    ap.add_argument("--phase", choices=["local", "multilevel"], default="local",
                    help="local (the only phase of hedonic 2.0; multilevel is refused)")
    ap.add_argument("--out", required=True, help="ledger folder (evals-<tag>.jsonl, searches.jsonl); memoized by "
                                                 "configuration, records of another build are not reused")
    ap.add_argument("--objective", default="f1", help="f1 (the benchmark's rule for Hedonic) or f1+onmi")
    ap.add_argument("--seeds", default="0", help="seeds of a selection probe")
    ap.add_argument("--confirm-seeds", default="0,1,2,3,4")
    ap.add_argument("--M0", type=int, default=8)
    ap.add_argument("--M-max", type=int, default=128, help="the largest cap the search may choose")
    ap.add_argument("--fixed-M", type=int, help="search the resolution only, at this cap (-1: no cap); without "
                    "--bracket the scan covers [1e-6, 0.5] at factor-4 spacing")
    ap.add_argument("--delta", type=float, default=0.002, help="score tolerance of the cap search, snap and gallop stop")
    ap.add_argument("--tol-u", type=float, default=0.1, help="width of the final golden-section bracket in log gamma")
    ap.add_argument("--max-evals", type=int, default=14, help="evaluations of one resolution search at most")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help="seconds before one detector run is killed (each run on its own, never a search's total)")
    ap.add_argument("--bracket", help="lo,hi absolute gamma bracket of the first scan (default rho_in/32, rho_in)")
    ap.add_argument("--gamma", type=float, help="probe mode: the resolution")
    ap.add_argument("--gammas", help="grid mode: comma-separated resolutions (fractions allowed)")
    ap.add_argument("--M", type=int, help="probe and grid modes: the cap")
    ap.add_argument("--tag", default="main", help="ledger suffix, one per concurrent process")
    ap.add_argument("--rounds", type=int, default=3, help="cap/resolution coordinate-ascent rounds")
    ap.add_argument("--M-start", type=int, help="cap of the stage-1 resolution search and start of the cap search "
                    "(default 4 x M0); a warm start passes the previous optimum's cap")
    ap.add_argument("--jobs", type=int, default=1,
                    help="evaluations of this search that may run side by side (the seeds of a probe, the scan points, "
                         "the confirmation candidates); the search itself still takes its steps one at a time")
    ap.add_argument("--slot-dir", help="folder of a core budget shared with other searches running at the same time")
    ap.add_argument("--slot-count", type=int, help="size of that budget (default: --jobs)")
    ap.add_argument("--no-extremal", action="store_true", help="skip the gamma -> 1 clique-regime probe")
    ap.add_argument("--network-root", help="SNAP networks folder (default: HEDONIC_NETWORKS_DIR, the user config's "
                                           "network_root, else the package default)")
    ap.add_argument("--snap-cache-dir", help="normalized SNAP cache (default: the package's)")
    ap.add_argument("--no-download", action="store_true",
                    help="refuse a pair that is not on disk instead of downloading it from SNAP")
    return ap


def _seeds(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if args.phase == "multilevel":
        ap.error(MULTILEVEL_GONE)
    if args.instance not in INST:  # a registered instance needs no SNAP name
        try:
            parse_instance(args.instance)
        except ValueError as exc:
            ap.error(str(exc))
    if args.mode == "probe" and (args.gamma is None or args.M is None):
        ap.error("probe needs --gamma and --M")
    if args.mode == "grid" and (not args.gammas or args.M is None):
        ap.error("grid needs --gammas and --M")
    configure(args.out, tag=args.tag, jobs=args.jobs, slot_dir=args.slot_dir, slot_count=args.slot_count,
              network_root=args.network_root, snap_cache_dir=args.snap_cache_dir, download=not args.no_download)
    kw = dict(M0=args.M0, M_max=args.M_max, seeds=_seeds(args.seeds),
              objective=args.objective, timeout=args.timeout, delta=args.delta, tol_u=args.tol_u,
              max_evals=args.max_evals, confirm_seeds=_seeds(args.confirm_seeds),
              fixed_M=args.fixed_M, extremal=not args.no_extremal, rounds=args.rounds, M_start=args.M_start,
              gamma_bracket=tuple(float(x) for x in args.bracket.split(",")) if args.bracket else None)
    if args.mode == "tune":
        tune(args.instance, args.phase, **kw)
    elif args.mode == "grid":
        inst = load_instance(args.instance)
        probe = Probe(args.instance, {"phase": args.phase}, _seeds(args.seeds), timeout=args.timeout,
                      objective=args.objective)
        for g in args.gammas.split(","):
            gam = float(Fraction(g)) if "/" in g else float(g)
            val = probe(gamma=gam, M=args.M)
            print(f"GRID {args.instance} {args.phase} M={args.M} gamma={gam:.6g} "
                  f"c={gam / inst['rho']:.4g} J={val:.4f} F1={probe.trace[-1]['f1']:.4f} "
                  f"ONMI={probe.trace[-1]['onmi']:.4f}", flush=True)
    else:
        inst = load_instance(args.instance)
        print(json.dumps(instance_summary(inst)))
        rec = evaluate(args.instance, {"phase": args.phase, "gamma": args.gamma, "M": args.M,
                                       "seed": _seeds(args.seeds)[0]},
                       timeout=args.timeout, objective=args.objective)
        print(json.dumps({k: rec.get(k) for k in ("status", "detection_seconds", "scoring_seconds",
                                                  "audit_seconds", "metrics", "label_free")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
