"""Label-free calibration of the baseline methods, one function per method (``hedonic-exp baseline-calibrate``).

The protocol (Protocol C of the overlapping benchmark) gives every method the subgraph ``G[S]`` induced by the vertices of a
ground-truth cover, and scores its answer against that cover. This module chooses the parameters a method needs **using
only ``G[S]``**: no community of the ground truth is read, no score against it is computed, and no parameter is taken
from the number or the content of the true communities. A published value that came from the true number of communities
(e.g. 25,000 BigCLAM factors) is a historical reference, not a valid setting here.

Enforced, not just promised: a calibration receives a :class:`Context` that holds the graph and nothing else (no cover
attribute exists), every detector call goes through :func:`default_runner`, which hands the dispatcher
``ground_truth=[]``, and :func:`load_context` returns the graph only (the cover's vertex set defines ``G[S]``; the
communities are dropped on the spot). ``tests/test_baseline_calibration.py`` checks the empty reference at the call.

Two kinds of function, all returning a :class:`CalibrationResult` whose ``params`` is a ready ``method_params`` table:

``fixed``    the authors' published protocol, taken as is (no detector run unless a sensitivity check is asked for):
             ``calibrate_codeseg`` (tau_n = 0.3, overlap factor 1),
             ``calibrate_ncgame`` (Hub-Promoted similarity, epsilon 0.01) and ``calibrate_ego_splitting`` (label
             propagation with the Absolute Potts model, alpha 0.1 / 0; the adapter that exists implements Louvain, so the
             result says it is **not** a faithful reproduction and what is missing).
``search``   a parameter the authors leave to the user, or on which two published values disagree, chosen on ``G[S]``
             by a label-free rule: ``calibrate_fox`` (LazyFox's stopping threshold, 0.1 of the CoDeSEG paper or its own
             default 0.01),
             ``calibrate_clique_percolation`` (k from the percolation transition, a structural rule),
             ``calibrate_angel`` (phi from a pre-registered grid), ``calibrate_bigclam`` and ``calibrate_neo_kmeans``
             (the number of communities K, by the likelihood of held-out node pairs, which is the rule of the BigCLAM
             paper).

**Selection criterion.** Held-out pair likelihood, computed from the cover alone (:func:`pair_loglik`): 20% of the edges are
removed from ``G[S]`` and as many random non-adjacent pairs are drawn; the method runs on the remaining graph; a
community-affiliation model is fitted from the returned cover and the training graph (per-community edge density,
background density outside communities; ``P(u,v) = 1 - (1-eps) prod_{c shared}(1 - p_c)``); the score is the mean
log-likelihood of the held-out pairs (positives and negatives weighted equally). It sees no ground truth, needs no
method-specific output (the SNAP BigCLAM binary writes memberships, not factor strengths), and is used only to compare
settings *of one method*, never across methods. Limits: the SNAP binary cannot mask the held-out non-edges, so they are
treated as non-edges during fitting (a small optimistic bias, the same for every candidate); the score ignores membership
strengths.

**Budget and ledger** (the same budget rule as the Hedonic calibration):
every evaluation is one detector run plus its scoring, recorded in an append-only JSONL ledger keyed by method, parameters,
split, seed and graph fingerprint (memoised: a rerun or a larger budget reuses what was run); a :class:`Budget` stops the
search before a new evaluation when the evaluation or the single-core-time cap is reached, and one evaluation is killed
at ``eval_timeout`` and recorded as a failed probe. The result states the stop reason, the evaluations and seconds spent,
and whether the optimum was **bracketed** by worse neighbours or still moving at the stop.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import igraph as ig

from hedonic.experiments.config import codeseg_cache_dir
from hedonic.experiments.overlapping import limits, replication

PROTOCOL = "baseline-calibration-v1"
CRITERION = "heldout-pair-loglik-v1"
HOLDOUT_FRACTION = 0.2
HOLDOUT_MAX_PAIRS = 200_000  # positives plus negatives; keeps the scorer cheap on the large graphs
DEFAULT_EVAL_TIMEOUT = 3600.0  # seconds one evaluation's detector run may take (``--eval-timeout-minutes`` 60)

# Pre-registered grids (fixed before any calibration run; a change is a protocol change, record why).
CLIQUE_SIZES = (3, 4, 5, 6)
CLIQUE_GIANT_FRACTION = 0.5
ANGEL_THRESHOLDS = (0.2, 0.33, 0.5, 0.66, 0.8)  # 0.33 and 0.66 are the authors' examples
ANGEL_MIN_COMMUNITY_SIZE = 3
K_FRACTIONS = (0.02, 0.005, 0.08, 0.01, 0.04, 0.002)  # K / n, in the order they are tried (centre first, then spread)
K_FACTOR = 2.0  # one extension step when the best K sits on the edge of what was tried
K_MAX_EXTEND = 2
EGO_SPLITTING_PROTOCOL = {  # Epasto, Lattanzi & Paes Leme (KDD 2017), as the paper reports it
    "ego_net_partitioner": "label propagation, Absolute Potts model", "ego_alpha": 0.1,
    "persona_graph_partitioner": "label propagation, Absolute Potts model", "persona_alpha": 0.0,
    "neighbour_limit": 2000, "discard_communities_up_to_size": 4,
}
CODESEG_FIXED = {"tau": 0.3, "alpha": 1.0, "iterations": 10}  # tau_n = 0.3, overlap factor beta = 1
FOX_FIXED = {"wcc_threshold": 0.1, "queue_size": 1, "threads": 1}
# The two published stopping thresholds, in the order tried (a tie goes to the first): 0.1 is the value the CoDeSEG paper
# reports for FOX (Appendix F); 0.01 is LazyFox's own default, inherited from Lyu et al. (Fox, Table 7).
FOX_THRESHOLDS = (0.1, 0.01)
NCGAME_FIXED = {"similarity_type": "HP", "stop_criterion": 0.01}
NEO_FIXED = {"alpha": 40.0, "beta": 0.0001}  # the authors' DBLP/Amazon setting

# What the benchmark runner understands per method. A calibration may carry more (NcGame's flags go to its command
# template, threads belong to the benchmark's one CPU policy), so the settings handed to ``--tuned`` are filtered to these.
RUNNER_KEYS: dict[str, tuple[str, ...]] = {
    "clique_percolation": ("clique_size",), "angel": ("threshold", "min_community_size"),
    "ego_splitting": ("resolution",), "fox": ("wcc_threshold", "queue_size"),
    "bigclam": ("communities",), "ncgame": (), "neo_kmeans": ("clusters", "alpha", "beta"),
    "codeseg": ("tau", "alpha", "iterations"),
}

# How the implementation the benchmark calls behaves under a seed, and how many benchmark runs that makes: one table, in
# ``replication`` (the benchmark plans its runs from it too). Re-exported here, the calibration's own name for it.
REPLICATION = replication.REPLICATION
DEFAULT_REPLICATES = 3  # replicas per setting when a score selects a parameter of a "replicas" method


class BudgetExhausted(Exception):
    """The next evaluation would exceed the budget."""


@dataclass(frozen=True)
class Budget:
    """Evaluations and single-core seconds a calibration may spend (None = no cap on that axis)."""
    max_evals: int | None = 8
    max_seconds: float | None = None
    eval_timeout: float | None = DEFAULT_EVAL_TIMEOUT


@dataclass
class Context:
    """Everything a calibration may use: the graph ``G[S]`` and how to run a detector on it. No cover."""
    graph: ig.Graph
    seed: int = 0
    budget: Budget = field(default_factory=Budget)
    ledger_dir: Path | None = None
    runner: Callable[[str, ig.Graph, dict, int], list[list[int]]] | None = None
    runtime: dict[str, dict] = field(default_factory=dict)   # per-method binary / command (never written to the ledger)
    label: str = "graph"
    replicates: dict[str, int] = field(default_factory=dict)  # method -> independent runs averaged per setting (default 1)
    replica_workers: int = 1  # replicas of one setting run side by side in forked children (1 = one after another)
    evaluations: int = 0     # new (non-memoised) evaluations spent (a setting counts once, however many replicas)
    seconds: float = 0.0     # their single-core seconds
    _split: Any = field(default=None, repr=False)
    _fingerprint: str | None = field(default=None, repr=False)

    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = graph_fingerprint(self.graph)
        return self._fingerprint

    def split(self) -> "Holdout":
        if self._split is None:
            self._split = make_holdout(self.graph, self.seed)
        return self._split


# ---------------------------------------------------------------------------------------------- graph and split
def graph_fingerprint(graph: ig.Graph) -> str:
    """SHA-256 of the vertex count and the sorted undirected edge list."""
    edges = sorted((min(a, b), max(a, b)) for a, b in graph.get_edgelist())
    digest = hashlib.sha256(f"{graph.vcount()}|{len(edges)}|".encode())
    digest.update(repr(edges).encode() if len(edges) < 1000 else b"".join(
        a.to_bytes(8, "little") + b.to_bytes(8, "little") for a, b in edges))
    return digest.hexdigest()


@dataclass
class Holdout:
    train: ig.Graph
    positives: list[tuple[int, int]]
    negatives: list[tuple[int, int]]


def make_holdout(graph: ig.Graph, seed: int, fraction: float = HOLDOUT_FRACTION,
                 max_pairs: int = HOLDOUT_MAX_PAIRS) -> Holdout:
    """Remove ``fraction`` of the edges (capped) and draw as many non-adjacent pairs; deterministic in ``seed``."""
    simple = graph.copy()
    simple.simplify()
    rng = random.Random(f"{PROTOCOL}|holdout|{seed}")
    edges = simple.get_edgelist()
    k = min(max(1, round(fraction * len(edges))), max_pairs // 2, len(edges))
    chosen = sorted(rng.sample(range(len(edges)), k)) if edges else []
    positives = [edges[i] for i in chosen]
    train = simple.copy()
    train.delete_edges(chosen)
    n = simple.vcount()
    negatives: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while n > 1 and len(negatives) < len(positives) and attempts < 50 * (len(positives) + 1):
        attempts += 1
        u, v = rng.randrange(n), rng.randrange(n)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen or simple.are_adjacent(u, v):
            continue
        seen.add(key)
        negatives.append(key)
    return Holdout(train, positives, negatives)


# ---------------------------------------------------------------------------------------------- the criterion
def cover_memberships(cover: Sequence[Sequence[int]], n: int) -> list[list[int]]:
    """Per-vertex community indices of the non-trivial communities (size >= 2), duplicates removed."""
    memberships: list[list[int]] = [[] for _ in range(n)]
    seen: set[tuple[int, ...]] = set()
    index = 0
    for members in cover:
        body = tuple(sorted({int(v) for v in members if 0 <= int(v) < n}))
        if len(body) < 2 or body in seen:
            continue
        seen.add(body)
        for v in body:
            memberships[v].append(index)
        index += 1
    return memberships


def cover_summary(cover: Sequence[Sequence[int]], n: int) -> dict[str, Any]:
    """Label-free description of a cover: counts, coverage, overlap, size of the largest community."""
    memberships = cover_memberships(cover, n)
    covered = sum(1 for m in memberships if m)
    count = (max((c for m in memberships for c in m), default=-1) + 1)
    sizes = [0] * count
    for m in memberships:
        for c in m:
            sizes[c] += 1
    return {"communities": count, "coverage": covered / n if n else 0.0,
            "mean_memberships": (sum(len(m) for m in memberships) / covered) if covered else 0.0,
            "largest_community": max(sizes, default=0),
            "largest_fraction_of_covered": (max(sizes, default=0) / covered) if covered else 0.0}


def pair_loglik(cover: Sequence[Sequence[int]], train: ig.Graph, positives: Sequence[tuple[int, int]],
                negatives: Sequence[tuple[int, int]]) -> dict[str, float]:
    """Mean held-out log-likelihood of a cover under a community-affiliation model fitted on the training graph.

    ``P(u,v) = 1 - (1-eps) * prod_{c shared by u,v} (1 - p_c)``, ``p_c`` the training edge density inside community c,
    ``eps`` the training density of pairs sharing no community. Score = mean over positives of ``log P`` and mean over
    negatives of ``log(1-P)``, weighted equally. Uses the cover and the training graph only."""
    n = train.vcount()
    memberships = cover_memberships(cover, n)
    mem_sets = [frozenset(m) for m in memberships]
    count = max((c for m in memberships for c in m), default=-1) + 1
    sizes = [0] * count
    for m in memberships:
        for c in m:
            sizes[c] += 1
    inside = [0] * count
    outside = 0
    for u, v in train.get_edgelist():
        shared = mem_sets[u] & mem_sets[v]
        if shared:
            for c in shared:
                inside[c] += 1
        else:
            outside += 1
    total_pairs = n * (n - 1) / 2
    community_pairs = sum(s * (s - 1) / 2 for s in sizes)
    background_pairs = max(total_pairs - min(community_pairs, 0.9 * total_pairs), 0.1 * total_pairs, 1.0)
    eps = min(max(outside / background_pairs, 1e-12), 0.5)
    log1m_p = []
    for c in range(count):
        pairs_c = sizes[c] * (sizes[c] - 1) / 2
        p_c = min(max(inside[c] / pairs_c if pairs_c else 0.0, eps), 1.0 - 1e-6)
        log1m_p.append(math.log1p(-p_c))
    log_one_minus_eps = math.log1p(-eps)

    def log_not_linked(u: int, v: int) -> float:
        shared = mem_sets[u] & mem_sets[v]
        return log_one_minus_eps + sum(log1m_p[c] for c in shared)

    def log_linked(u: int, v: int) -> float:
        value = log_not_linked(u, v)
        return math.log(max(-math.expm1(value), 1e-300))

    pos = [log_linked(u, v) for u, v in positives]
    neg = [log_not_linked(u, v) for u, v in negatives]
    mean_pos = sum(pos) / len(pos) if pos else float("nan")
    mean_neg = sum(neg) / len(neg) if neg else float("nan")
    return {"score": 0.5 * (mean_pos + mean_neg), "positive_loglik": mean_pos, "negative_loglik": mean_neg,
            "background_density": eps, "pairs": len(pos) + len(neg)}


# ---------------------------------------------------------------------------------------------- running a detector
def default_runner(method: str, graph: ig.Graph, params: dict[str, Any], seed: int) -> list[list[int]]:
    """One detector run through the benchmark's dispatcher with an EMPTY reference: it can never see a community."""
    from hedonic.experiments.overlapping import codeseg_reproduction as cr
    from hedonic.experiments.overlapping import extra_methods

    extra_methods.install()
    cover, _meta = cr._run_method(method, graph, [], int(seed), dict(params))
    return [list(map(int, c)) for c in cover]


def _child(runner, method, graph, params, seed, path):
    # ``detect_seconds`` here is the probe's wall-clock (fork, pickle and the runner's conversions included): budget
    # accounting for the search, not a method comparison. Reported times come from ``timing`` in the benchmark.
    t0 = time.perf_counter()
    try:
        cover = runner(method, graph, params, seed)
        out: dict[str, Any] = {"status": "completed", "cover": cover}
    except BaseException as exc:  # noqa: BLE001 - recorded as a failed probe
        unavailable = type(exc).__name__ == "MethodUnavailable"
        out = {"status": "unavailable" if unavailable else "failed", "reason": f"{type(exc).__name__}: {exc}"[:500]}
    out["detect_seconds"] = time.perf_counter() - t0
    with open(path, "wb") as fh:
        pickle.dump(out, fh)


def _kill_tree(pid: int) -> None:
    """Kill a process and everything it started: an isolated CDlib worker (ANGEL, Ego-Splitting) is a grandchild of the
    probe, and killing only the probe leaves it running for hours as an orphan."""
    import signal
    import subprocess

    try:
        rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        rows = []
    kids: dict[int, list[int]] = {}
    for row in rows:
        parts = row.split()
        if len(parts) == 2 and all(x.isdigit() for x in parts):
            kids.setdefault(int(parts[1]), []).append(int(parts[0]))
    order, stack = [], [pid]
    while stack:
        cur = stack.pop()
        order.append(cur)
        stack.extend(kids.get(cur, []))
    for one in reversed(order):
        with contextlib.suppress(OSError):
            os.kill(one, signal.SIGKILL)


def run_isolated(runner, method, graph, params, seed, timeout: float | None) -> dict[str, Any]:
    """The detector in a forked child (its memory is freed on exit, a hang is killed at ``timeout``)."""
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    proc = mp.get_context("fork").Process(target=_child, args=(runner, method, graph, params, seed, path))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout)
    try:
        if proc.is_alive():
            _kill_tree(proc.pid)
            proc.kill()
            proc.join()
            return {"status": "timeout", "reason": f"killed after {timeout:.0f} s", "detect_seconds": time.perf_counter() - t0}
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:  # noqa: BLE001
            return {"status": "failed", "reason": f"child exit {proc.exitcode}", "detect_seconds": time.perf_counter() - t0}
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _ledger_path(ctx: Context, method: str) -> Path | None:
    if ctx.ledger_dir is None:
        return None
    ctx.ledger_dir.mkdir(parents=True, exist_ok=True)
    return ctx.ledger_dir / f"evals-{method}.jsonl"


def _key(ctx: Context, method: str, params: dict[str, Any], replica: int = 0) -> str:
    payload = {"protocol": PROTOCOL, "criterion": CRITERION, "method": method, "params": params, "seed": ctx.seed,
               "graph": ctx.fingerprint(), "holdout": [HOLDOUT_FRACTION, HOLDOUT_MAX_PAIRS]}
    if replica:
        payload["replica"] = int(replica)  # replica 0 keeps the key a single run always had
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _read_ledger(path: Path | None) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if path is not None and path.is_file():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:  # a torn last line
                continue
            if isinstance(rec, dict) and "key" in rec:
                records[rec["key"]] = rec
    return records


def _run_replica(ctx: Context, method: str, params: dict[str, Any], replica: int, key: str, path: Path | None) -> dict[str, Any]:
    split = ctx.split()  # one split for every setting and replica: scores differ by the setting and the run, not the data
    b = ctx.budget
    call_params = {**ctx.runtime.get(method, {}), **params}
    if b.eval_timeout:
        call_params.setdefault("timeout_seconds", b.eval_timeout)
    runner = ctx.runner or default_runner
    out = run_isolated(runner, method, split.train, call_params, ctx.seed + replica, b.eval_timeout)
    rec: dict[str, Any] = {"key": key, "method": method, "params": params, "seed": ctx.seed, "replica": replica,
                           "status": out["status"], "detect_seconds": out.get("detect_seconds", 0.0),
                           "graph": ctx.fingerprint(), "criterion": CRITERION,
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    if out["status"] == "completed":
        t1 = time.perf_counter()
        cover = out["cover"]
        rec["summary"] = cover_summary(cover, split.train.vcount())
        rec["criterion_values"] = pair_loglik(cover, split.train, split.positives, split.negatives)
        rec["score"] = rec["criterion_values"]["score"]
        rec["score_seconds"] = time.perf_counter() - t1
    else:
        rec["reason"] = out.get("reason")
        rec["score_seconds"] = 0.0
        if out["status"] == "timeout" and b.eval_timeout:
            rec["limit_seconds"] = float(b.eval_timeout)
    ctx.seconds += rec["detect_seconds"] + rec["score_seconds"]
    if path is not None:
        with path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def _merge_replicas(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """One record for a setting run several times: the mean score over the completed replicas, and their spread."""
    done = [r for r in recs if r["status"] == "completed"]
    if not done:
        return recs[0]
    mean = lambda values: sum(values) / len(values)  # noqa: E731
    scores = [r["score"] for r in done]
    spread = (sum((x - mean(scores)) ** 2 for x in scores) / (len(scores) - 1)) ** 0.5 if len(scores) > 1 else 0.0
    summary = {k: mean([r["summary"][k] for r in done]) for k in done[0]["summary"]}
    return {**done[0], "score": mean(scores), "score_sd": spread, "replicas": len(recs), "replicas_completed": len(done),
            "detect_seconds": mean([r["detect_seconds"] for r in done]), "summary": summary,
            "criterion_values": {**done[0]["criterion_values"], "score": mean(scores)}}


def evaluate(ctx: Context, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """One evaluation of ``params``: run on the training graph, score the held-out pairs. Memoised in the ledger.

    A method with ``ctx.replicates[method] = R > 1`` is run R times (replica r under seed ``ctx.seed + r``, which only
    matters to a method that receives it) and the setting's score is their mean, so a stochastic method's noise is not
    mistaken for a difference between settings. Each replica is its own ledger line; the budget counts a setting once.

    ``params`` are the knobs that identify the setting; machine paths (binary, command) come from ``ctx.runtime`` and are
    not part of the key. Raises :class:`BudgetExhausted` before a new evaluation would exceed the budget."""
    path = _ledger_path(ctx, method)
    known = _read_ledger(path)
    count = max(1, int(ctx.replicates.get(method, 1)))
    recs: list[dict[str, Any] | None] = [None] * count
    missing: list[int] = []
    fresh = False
    for replica in range(count):
        key = _key(ctx, method, params, replica)
        # a run stopped at a timeout answers only under a timeout no larger than that one (one evaluation each)
        if key in known and limits.stands(known[key], ctx.budget.eval_timeout, default=DEFAULT_EVAL_TIMEOUT):
            recs[replica] = known[key]
            continue
        if not fresh:
            b = ctx.budget
            if b.max_evals is not None and ctx.evaluations >= b.max_evals:
                raise BudgetExhausted(f"{ctx.evaluations} evaluations")
            if b.max_seconds is not None and ctx.seconds >= b.max_seconds:
                raise BudgetExhausted(f"{ctx.seconds:.0f} s")
            fresh = True
            ctx.evaluations += 1
        missing.append(replica)
    if len(missing) > 1 and ctx.replica_workers > 1:
        # The replicas of a setting do not depend on one another: run them side by side. Each child appends its own
        # ledger line; the seconds it spent are added here because a child's copy of the context is lost with it.
        from hedonic.experiments.overlapping import parallel

        def one(replica: int) -> dict[str, Any]:
            return _run_replica(ctx, method, params, replica, _key(ctx, method, params, replica), path)

        for index, rec in parallel.run_forked(one, missing, ctx.replica_workers):
            recs[missing[index]] = rec
            ctx.seconds += rec["detect_seconds"] + rec["score_seconds"]
    else:
        for replica in missing:
            recs[replica] = _run_replica(ctx, method, params, replica, _key(ctx, method, params, replica), path)
    return recs[0] if count == 1 else _merge_replicas(recs)


# ---------------------------------------------------------------------------------------------- results
@dataclass
class CalibrationResult:
    method: str
    status: str                       # "calibrated" | "fixed" | "unavailable" | "failed"
    params: dict[str, Any] | None     # a ready method_params table (None when nothing could be chosen)
    rule: str
    faithful_to_paper: bool | None    # True/False when the result is a claim about reproducing the authors; None = n/a
    uses_ground_truth: bool = False   # always False here; kept in the record so a reader does not have to assume
    criterion: str | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "complete"
    bracketed: bool | None = None
    evaluations: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    replication: dict[str, Any] = field(default_factory=dict)  # how many benchmark runs this method needs (REPLICATION)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _candidate(rec: dict) -> dict[str, Any]:
    out = {"params": rec["params"], "status": rec["status"], "detect_seconds": rec["detect_seconds"]}
    if rec["status"] == "completed":
        out.update(score=rec["score"], communities=rec["summary"]["communities"], coverage=rec["summary"]["coverage"],
                   mean_memberships=rec["summary"]["mean_memberships"],
                   largest_fraction_of_covered=rec["summary"]["largest_fraction_of_covered"])
        if "replicas" in rec:
            out.update(score_sd=rec["score_sd"], replicas=rec["replicas"], replicas_completed=rec["replicas_completed"])
    else:
        out["reason"] = rec.get("reason")
    return out


def _finish(ctx: Context, before: tuple[int, float], result: CalibrationResult) -> CalibrationResult:
    result.evaluations = ctx.evaluations - before[0]
    result.seconds = ctx.seconds - before[1]
    return result


def _best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    done = [c for c in candidates if c["status"] == "completed"]
    top = None
    for c in done:  # the first of equal scores wins (the order tried is the registered one)
        if top is None or c["score"] > top["score"]:
            top = c
    return top


def _all_failed(method: str, rule: str, cands: list[dict], stop: str) -> CalibrationResult:
    statuses = {c["status"] for c in cands}
    status = "unavailable" if statuses == {"unavailable"} else "failed"
    reasons = sorted({str(c.get("reason")) for c in cands if c.get("reason")})
    return CalibrationResult(method, status, None, rule, None, criterion=CRITERION, candidates=cands, stop_reason=stop,
                             notes=reasons[:3] or ["no evaluation completed"])


def k_grid(n: int, fractions: Sequence[float] = K_FRACTIONS) -> list[int]:
    """Candidate community counts, as fractions of the vertex count, in the order they are tried (distinct, >= 2)."""
    out: list[int] = []
    for f in fractions:
        k = max(2, min(int(round(f * n)), max(2, n // 2)))
        if k not in out:
            out.append(k)
    return out


def search_k(ctx: Context, method: str, make_params: Callable[[int], dict[str, Any]], *,
             fractions: Sequence[float] = K_FRACTIONS) -> tuple[list[dict], str, bool | None]:
    """Try the K grid centre-out, then extend by a factor 2 while the best K is on the edge of what was tried.

    Returns (candidates in the order tried, stop reason, bracketed)."""
    n = ctx.graph.vcount()
    tried: dict[int, dict] = {}
    stop = "complete"
    queue = k_grid(n, fractions)
    extensions = 0
    try:
        while True:
            for k in queue:
                if k not in tried:
                    tried[k] = _candidate(evaluate(ctx, method, make_params(k)))
                    if tried[k]["status"] == "unavailable":  # a missing runtime fails every K alike: stop at once
                        raise BudgetExhausted("runtime unavailable")
            done = {k: c for k, c in tried.items() if c["status"] == "completed"}
            if not done:
                break
            best_k = max(sorted(done), key=lambda k: done[k]["score"])
            ks = sorted(done)
            if extensions >= K_MAX_EXTEND:
                break
            if best_k == ks[0] and ks[0] > 2:
                queue = [max(2, int(round(ks[0] / K_FACTOR)))]
            elif best_k == ks[-1] and ks[-1] < max(2, n // 2):
                queue = [min(max(2, n // 2), int(round(ks[-1] * K_FACTOR)))]
            else:
                break
            if queue[0] in tried:
                break
            extensions += 1
    except BudgetExhausted as stop_exc:
        stop = "runtime unavailable" if str(stop_exc) == "runtime unavailable" else f"budget ({stop_exc})"
    cands = list(tried.values())
    done_ks = sorted(k for k, c in tried.items() if c["status"] == "completed")
    bracketed = None
    best = _best(cands)
    if best is not None:
        k_of = {id(c): k for k, c in tried.items()}[id(best)]
        bracketed = bool(done_ks and done_ks[0] < k_of < done_ks[-1])
    return cands, stop, bracketed


# ---------------------------------------------------------------------------------------------- the eight calibrations
def calibrate_clique_percolation(ctx: Context, *, ks: Sequence[int] = CLIQUE_SIZES,
                                 giant_fraction: float = CLIQUE_GIANT_FRACTION) -> CalibrationResult:
    """k from the percolation transition (structural rule, no score against anything).

    Below the transition one k-clique community absorbs most of the covered vertices; above it the communities
    fragment. The rule is: **the smallest k in ``ks`` whose largest community holds less than ``giant_fraction`` of the
    vertices that belong to any k-clique community**. If every k in the range still has a giant community the largest k
    is used and ``bracketed`` is False; if k = ks[0] already has none the smallest is used. The curve is in
    ``candidates``. It runs on the whole ``G[S]`` (no held-out split: the rule needs no score)."""
    before = (ctx.evaluations, ctx.seconds)
    rule = (f"smallest k in {list(ks)} whose largest k-clique community holds < {giant_fraction:.0%} of the vertices in "
            "any k-clique community (percolation transition); structural, no ground truth")
    cands: list[dict] = []
    stop = "complete"
    chosen = None
    runner = ctx.runner or default_runner
    for k in sorted(ks):
        if ctx.budget.max_evals is not None and ctx.evaluations - before[0] >= ctx.budget.max_evals:
            stop = f"budget ({ctx.evaluations - before[0]} evaluations)"
            break
        if ctx.budget.max_seconds is not None and ctx.seconds - before[1] >= ctx.budget.max_seconds:
            stop = f"budget ({ctx.seconds - before[1]:.0f} s)"
            break
        params = {"clique_size": int(k)}
        path = _ledger_path(ctx, "clique_percolation")
        key = _key(ctx, "clique_percolation", {**params, "split": "whole"})
        rec = _read_ledger(path).get(key)
        if rec is None:
            out = run_isolated(runner, "clique_percolation", ctx.graph,
                               {**ctx.runtime.get("clique_percolation", {}), **params,
                                "timeout_seconds": ctx.budget.eval_timeout},
                               ctx.seed, ctx.budget.eval_timeout)
            rec = {"key": key, "method": "clique_percolation", "params": params, "seed": ctx.seed,
                   "status": out["status"], "detect_seconds": out.get("detect_seconds", 0.0), "score_seconds": 0.0,
                   "graph": ctx.fingerprint(), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            if out["status"] == "completed":
                rec["summary"] = cover_summary(out["cover"], ctx.graph.vcount())
                rec["score"] = -rec["summary"]["largest_fraction_of_covered"]  # reported as a curve, not maximised
            else:
                rec["reason"] = out.get("reason")
            ctx.evaluations += 1
            ctx.seconds += rec["detect_seconds"]
            if path is not None:
                with path.open("a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
        cands.append(_candidate(rec))
        if rec["status"] == "completed" and chosen is None and rec["summary"]["largest_fraction_of_covered"] < giant_fraction:
            chosen = k
            break  # larger k only fragments further; the transition is bracketed from above
    done = [c for c in cands if c["status"] == "completed"]
    if not done:
        return _finish(ctx, before, _all_failed("clique_percolation", rule, cands, stop))
    notes = []
    if chosen is None:
        chosen = max(c["params"]["clique_size"] for c in done)
        bracketed = False
        notes.append(f"every k tried still has a giant community (>= {giant_fraction:.0%}); the largest k is used and "
                     "the transition is not bracketed: widen the range before relying on it")
    else:
        bracketed = chosen != min(sorted(ks)) or None
        if chosen == min(sorted(ks)):
            notes.append("no giant community already at the smallest k; the transition may lie below the range")
            bracketed = False
    notes.append("covers of nodes outside every k-clique stay uncovered (coverage is reported per k)")
    return _finish(ctx, before, CalibrationResult(
        "clique_percolation", "calibrated", {"clique_size": int(chosen)}, rule, None, candidates=cands,
        stop_reason=stop, bracketed=bracketed, notes=notes))


def calibrate_angel(ctx: Context, *, thresholds: Sequence[float] = ANGEL_THRESHOLDS,
                    min_community_size: int = ANGEL_MIN_COMMUNITY_SIZE) -> CalibrationResult:
    """phi (the threshold that merges local communities) from a pre-registered grid, by held-out pair likelihood.

    The authors give no universal value (0.33 and 0.66 are illustrations; larger phi merges less). The runner's
    ``threshold = 0.6`` is the benchmark's earlier choice, not an authors' recommendation. The grid is the same for every
    cover; ``min_community_size`` is fixed at the runner's 3."""
    before = (ctx.evaluations, ctx.seconds)
    rule = (f"phi in {list(thresholds)} (pre-registered), maximum held-out pair likelihood on G[S]; "
            f"min_community_size fixed at {min_community_size}; ties go to the first (smaller) phi")
    cands, stop = [], "complete"
    try:
        for phi in thresholds:
            cands.append(_candidate(evaluate(ctx, "angel", {"threshold": float(phi),
                                                            "min_community_size": int(min_community_size)})))
            if cands[-1]["status"] == "unavailable":
                stop = "runtime unavailable"
                break
    except BudgetExhausted as exc:
        stop = f"budget ({exc})"
    best = _best(cands)
    if best is None:
        return _finish(ctx, before, _all_failed("angel", rule, cands, stop))
    ordered = [c for c in cands if c["status"] == "completed"]
    phis = [c["params"]["threshold"] for c in ordered]
    bracketed = (min(phis) < best["params"]["threshold"] < max(phis)) if len(phis) >= 3 else None
    notes = ["the optimum is on the edge of the grid; phi is bounded in (0, 1), extend the grid only with a recorded reason"] \
        if bracketed is False else []
    return _finish(ctx, before, CalibrationResult(
        "angel", "calibrated", dict(best["params"]), rule, None, criterion=CRITERION, candidates=cands,
        stop_reason=stop, bracketed=bracketed, notes=notes))


def calibrate_ego_splitting(ctx: Context | None = None, *, adapter_sensitivity: Sequence[float] = ()) -> CalibrationResult:
    """The authors' Ego-Splitting protocol, and an honest statement that the adapter cannot run it.

    Paper (KDD 2017): label propagation with the Absolute Potts model on both the ego-nets (alpha = 0.1) and the persona
    graph (alpha = 0), at most 2,000 neighbours per vertex in the pre-step, communities up to size 4 dropped. The
    adapter in this repository (``extra_methods.ego_splitting`` in the pinned CDlib worker) uses **Louvain with
    resolution 1.0**, so the returned ``params`` is the adapter's own setting with ``faithful_to_paper = False``: a row
    produced with it must not be called a reproduction of the authors' setup. ``adapter_sensitivity`` (Louvain
    resolutions) optionally scores the adapter's only knob by the held-out criterion, labelled adapter-only."""
    rule = ("authors' protocol cannot be run by the adapter (Louvain, resolution 1.0, instead of Absolute Potts label "
            "propagation); the adapter's own setting is returned, flagged unfaithful")
    notes = [f"authors' protocol: {json.dumps(EGO_SPLITTING_PROTOCOL)}",
             "to claim a faithful reproduction the pinned CDlib worker's _ego_splitting must use Absolute Potts label "
             "propagation (alpha 0.1 ego-nets, 0 persona graph), the 2,000-neighbour cap and the size > 4 filter"]
    result = CalibrationResult("ego_splitting", "fixed", {"resolution": 1.0}, rule, False, notes=notes)
    if ctx is not None and adapter_sensitivity:
        before = (ctx.evaluations, ctx.seconds)
        try:
            for res in adapter_sensitivity:
                result.candidates.append(_candidate(evaluate(ctx, "ego_splitting", {"resolution": float(res)})))
        except BudgetExhausted as exc:
            result.stop_reason = f"budget ({exc})"
        result.criterion = CRITERION
        result.notes.append("candidates are the adapter's Louvain resolutions (adapter-only sensitivity, not the paper)")
        _finish(ctx, before, result)
    return result


def calibrate_fox(ctx: Context | None = None, *, thresholds: Sequence[float] = FOX_THRESHOLDS,
                  sensitivity: Sequence[float] = ()) -> CalibrationResult:
    """LazyFox at one of its two published stopping thresholds, chosen on ``G[S]`` by the held-out pair likelihood.

    The papers disagree: the CoDeSEG paper reports FOX with threshold 0.1, LazyFox's own default (from Lyu et al.) is
    0.01, and neither tunes it per network. Both values are tried (queue size 1, one thread: the benchmark's CPU cap,
    which the LazyFox paper says is the original Fox) and the better one by the label-free criterion is kept; a tie goes
    to 0.1. Without a context nothing is run and the setting of the CoDeSEG paper is returned. ``sensitivity``
    thresholds are scored too but are not candidates."""
    rule = (f"stopping threshold in {list(thresholds)} (the CoDeSEG paper's 0.1 and LazyFox's default 0.01), queue size 1, "
            "one thread; maximum held-out pair likelihood on G[S]; ties go to the first")
    if ctx is None:
        result = CalibrationResult("fox", "fixed", dict(FOX_FIXED), "published setting of the CoDeSEG paper: LazyFox "
                                   "stop threshold 0.1, queue size 1, one thread (not evaluated without a graph)", True,
                                   notes=["threads are fixed in the benchmark's timing runs; record the cap with the results"])
        return result
    before = (ctx.evaluations, ctx.seconds)
    cands, stop = [], "complete"
    try:
        for t in [*thresholds, *[x for x in sensitivity if x not in thresholds]]:
            cands.append(_candidate(evaluate(ctx, "fox", {**FOX_FIXED, "wcc_threshold": float(t)})))
            if cands[-1]["status"] == "unavailable":
                stop = "runtime unavailable"
                break
    except BudgetExhausted as exc:
        stop = f"budget ({exc})"
    best = _best([c for c in cands if c["params"]["wcc_threshold"] in thresholds])
    if best is None:
        return _finish(ctx, before, _all_failed("fox", rule, cands, stop))
    ok = [c for c in cands if c["status"] == "completed" and c["params"]["wcc_threshold"] in thresholds]
    notes = ["the two thresholds are the only candidates, so the choice is not a bracketed optimum"]
    if len(ok) < len(thresholds):
        notes.append("only one threshold completed: the setting is the one that did, not a comparison")
    return _finish(ctx, before, CalibrationResult(
        "fox", "calibrated", dict(best["params"]), rule, None, criterion=CRITERION, candidates=cands,
        stop_reason=stop, bracketed=None, notes=notes))


def calibrate_ncgame(ctx: Context | None = None, *, sensitivity: Sequence[float] = ()) -> CalibrationResult:
    """NcGame with Hub-Promoted similarity and stopping epsilon 0.01 (the setup of the reported results).

    ``sensitivity`` epsilons (0.005, the lower end of the authors' 0.005-0.01 range, is the one to ask for) are scored by
    the held-out criterion only when a context is given and the budget allows. The runner reaches NcGame through a command template;
    stop criterion and similarity are appended to it as flags of the packaged worker (its defaults are already HP / 0.01)."""
    rule = "Hub-Promoted similarity, stopping epsilon 0.01 (fixed); epsilon 0.005 only as a sensitivity check"
    result = CalibrationResult("ncgame", "fixed", dict(NCGAME_FIXED), rule, True,
                               notes=["the second phase's overlap criterion is computed per agent: no threshold to set"])
    if ctx is not None and sensitivity:
        _sensitivity(ctx, result, "ncgame", [{**NCGAME_FIXED, "stop_criterion": float(e)} for e in sensitivity])
        result.notes.append("a sensitivity run passes an empty reference file to the upstream script; verify it accepts it")
    return result


def calibrate_codeseg(ctx: Context | None = None, *, sensitivity: Sequence[float] = ()) -> CalibrationResult:
    """CoDeSEG at the authors' recommendation: tau_n = 0.3 and overlap factor 1.

    The paper reports 0.4 can stop too early and 0.2 costs time; those are the natural ``sensitivity`` values. The
    executable takes a reference file (``-t``) that it uses for its own printout; calibration runs pass an empty one."""
    rule = "tau_n = 0.3, overlap factor 1 (the authors' values, fixed)"
    result = CalibrationResult("codeseg", "fixed", dict(CODESEG_FIXED), rule, True,
                               notes=["the runner's earlier screened tau 0.5 / 20 iterations is not the authors' setting"])
    if ctx is not None and sensitivity:
        _sensitivity(ctx, result, "codeseg", [{**CODESEG_FIXED, "tau": float(t)} for t in sensitivity])
        result.notes.append("a sensitivity run passes an empty reference file to the executable; verify it accepts it")
    return result


def _sensitivity(ctx: Context, result: CalibrationResult, method: str, settings: list[dict[str, Any]]) -> None:
    before = (ctx.evaluations, ctx.seconds)
    result.criterion = CRITERION
    try:
        for params in [result.params, *settings]:
            result.candidates.append(_candidate(evaluate(ctx, method, dict(params))))
            if result.candidates[-1]["status"] == "unavailable":
                result.stop_reason = "runtime unavailable"
                break
    except BudgetExhausted as exc:
        result.stop_reason = f"budget ({exc})"
    _finish(ctx, before, result)


def _calibrate_k(ctx: Context, method: str, make_params: Callable[[int], dict[str, Any]], rule: str,
                 notes: list[str], fractions: Sequence[float]) -> CalibrationResult:
    before = (ctx.evaluations, ctx.seconds)
    cands, stop, bracketed = search_k(ctx, method, make_params, fractions=fractions)
    best = _best(cands)
    if best is None:
        return _finish(ctx, before, _all_failed(method, rule, cands, stop))
    notes = list(notes)
    if bracketed is False:
        notes.append("the best K is on the edge of the tried range and the budget or the K ceiling stopped the "
                     "extension: report this K as not bracketed")
    if best["coverage"] < 0.95:
        notes.append(f"coverage of the chosen setting is {best['coverage']:.1%}: vertices left unassigned")
    return _finish(ctx, before, CalibrationResult(
        method, "calibrated", dict(best["params"]), rule, None, criterion=CRITERION, candidates=cands,
        stop_reason=stop, bracketed=bracketed, notes=notes))


def calibrate_bigclam(ctx: Context, *, fractions: Sequence[float] = K_FRACTIONS, threads: int = 1) -> CalibrationResult:
    """K (``communities``) by the likelihood of held-out node pairs, the rule of the BigCLAM paper, on ``G[S]`` only.

    20% of the edges (plus as many non-edges) are held out, BigCLAM is fitted on the rest for K on a geometric grid
    (fractions of n, centre-out, extended by a factor 2 while the best sits on the edge), and the K with the highest
    held-out score is returned. The paper also uses the known number of communities in some experiments; that needs the
    ground truth and is not used. The 25,000 of the earlier catalogue was chosen for full DBLP and does not transfer."""
    rule = (f"K = fraction of n in {list(fractions)} (+ factor-{K_FACTOR:g} extension), maximum held-out pair likelihood on "
            "G[S] (BigCLAM paper's rule); no ground-truth count")
    return _calibrate_k(ctx, "bigclam", lambda k: {"communities": int(k), "threads": int(threads)}, rule,
                        ["the SNAP binary writes memberships only, so the score uses the cover, not the factor strengths"],
                        fractions)


def calibrate_neo_kmeans(ctx: Context, *, fractions: Sequence[float] = K_FRACTIONS, alpha: float = NEO_FIXED["alpha"],
                         beta: float = NEO_FIXED["beta"]) -> CalibrationResult:
    """K (``clusters``) by held-out pair likelihood with the authors' alpha = 40, beta = 0.0001 (DBLP / Amazon).

    alpha controls overlap and beta the non-exhaustiveness; the authors report low sensitivity near these values. K must be
    set and is not taken from the ground truth. With beta > 0 some vertices stay unassigned: coverage is reported and the
    held-out criterion treats them with the background density."""
    rule = (f"alpha = {alpha:g}, beta = {beta:g} (authors, DBLP/Amazon); K = fraction of n in {list(fractions)} "
            "(+ extension), maximum held-out pair likelihood on G[S]; no ground-truth count")
    return _calibrate_k(ctx, "neo_kmeans", lambda k: {"clusters": int(k), "alpha": float(alpha), "beta": float(beta)},
                        rule, ["beta > 0 allows unassigned vertices; coverage is in the candidate table"],
                        fractions)


CALIBRATORS: dict[str, Callable[..., CalibrationResult]] = {
    "clique_percolation": calibrate_clique_percolation,
    "angel": calibrate_angel,
    "ego_splitting": calibrate_ego_splitting,
    "fox": calibrate_fox,
    "bigclam": calibrate_bigclam,
    "ncgame": calibrate_ncgame,
    "neo_kmeans": calibrate_neo_kmeans,
    "codeseg": calibrate_codeseg,
}
FIXED_METHODS = ("codeseg", "ncgame", "ego_splitting")


def calibrate(method: str, ctx: Context, **options: Any) -> CalibrationResult:
    """Calibrate one method on ``ctx.graph`` (a fixed-protocol method returns its published setting, running nothing
    unless a sensitivity check is asked for in ``options``). The result carries the method's replication policy."""
    if method not in CALIBRATORS:
        raise KeyError(f"no calibration for {method!r}; known: {', '.join(CALIBRATORS)}")
    result = CALIBRATORS[method](ctx, **options)
    result.replication = dict(REPLICATION[method])
    return result


def runner_params(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    """The part of a calibrated setting that the benchmark runner reads (``method_params`` of ``--tuned``)."""
    keys = RUNNER_KEYS.get(method, ())
    return {key: value for key, value in (params or {}).items() if key in keys}


SENSITIVITY_OPTIONS: dict[str, dict[str, Any]] = {"ncgame": {"sensitivity": (0.005,)},
                                                  "codeseg": {"sensitivity": (0.2, 0.4)}}


def with_replicates(ctx: Context, replicates: int = DEFAULT_REPLICATES) -> Context:
    """Average ``replicates`` independent runs per setting for the methods whose runs are replicas (REPLICATION)."""
    ctx.replicates = {m: int(replicates) for m in REPLICATION if replication.calibration_kind(m) == "replicas"} \
        if replicates > 1 else {}
    return ctx


def calibrate_method(ctx: Context, method: str, *, sensitivity: bool = False) -> CalibrationResult:
    """``calibrate`` with the budget reset for this method; a fixed-protocol method scores its neighbours when asked."""
    ctx.evaluations, ctx.seconds = 0, 0.0  # the budget is per method and target
    if method == "ego_splitting":
        result = calibrate_ego_splitting(ctx)
        result.replication = dict(REPLICATION[method])
        return result
    options = SENSITIVITY_OPTIONS.get(method, {}) if sensitivity else {}
    return calibrate(method, ctx, **(options if method in FIXED_METHODS else {}))


# ---------------------------------------------------------------------------------------------- planning and output
def plan(n: int, m: int, methods: Sequence[str], budget: Budget) -> list[dict[str, Any]]:
    """What a calibration would try on a graph of n vertices (no detector run)."""
    rows = []
    for name in methods:
        if name == "clique_percolation":
            rows.append({"method": name, "kind": "search", "tries": [{"clique_size": k} for k in CLIQUE_SIZES],
                         "rule": "percolation transition (structural)"})
        elif name == "angel":
            rows.append({"method": name, "kind": "search", "tries": [{"threshold": t} for t in ANGEL_THRESHOLDS],
                         "rule": CRITERION})
        elif name == "bigclam":
            rows.append({"method": name, "kind": "search", "tries": [{"communities": k} for k in k_grid(n)],
                         "rule": CRITERION, "extension": f"x{K_FACTOR:g} up to {K_MAX_EXTEND} steps"})
        elif name == "neo_kmeans":
            rows.append({"method": name, "kind": "search", "tries": [{"clusters": k, **NEO_FIXED} for k in k_grid(n)],
                         "rule": CRITERION, "extension": f"x{K_FACTOR:g} up to {K_MAX_EXTEND} steps"})
        elif name == "fox":
            rows.append({"method": name, "kind": "search",
                         "tries": [{**FOX_FIXED, "wcc_threshold": t} for t in FOX_THRESHOLDS], "rule": CRITERION})
        else:
            fixed = {"codeseg": CODESEG_FIXED, "ncgame": NCGAME_FIXED,
                     "ego_splitting": {"resolution": 1.0}}[name]
            rows.append({"method": name, "kind": "fixed", "params": fixed})
    for row in rows:
        row["max_evals"] = budget.max_evals
    return rows


def toml_snippet(target: str, results: Sequence[CalibrationResult]) -> str:
    """``[hedonic_run.method_params.METHOD."TARGET"]`` tables for a benchmark config (a target key overrides its network's)."""
    lines = [f"# generated by hedonic-exp baseline-calibrate ({PROTOCOL}); label-free, chosen on G[S] only"]
    for r in results:
        if r.params is None:
            lines.append(f"# {r.method}: no setting ({r.status})")
            continue
        params = runner_params(r.method, r.params)
        if not params:
            lines.append(f"# {r.method}: the runner's own defaults are the calibrated setting")
            continue
        flag = "" if r.faithful_to_paper is not False else "  # NOT the authors' protocol, see notes"
        lines.append(f'[hedonic_run.method_params.{r.method}."{target}"]{flag}')
        for key, value in params.items():
            lines.append(f"{key} = {json.dumps(value)}")
        lines.append("")
    return "\n".join(lines)


def runtime_parameters(manifest_path: str | os.PathLike | None = None) -> dict[str, dict[str, Any]]:
    """Binaries and the NcGame command, from the environment and the setup manifest (as the benchmark runner reads them)."""
    manifest: dict[str, Any] = {}
    default = Path(os.environ.get("HEDONIC_CODESEG_MANIFEST")
                   or codeseg_cache_dir() / "setup_manifest.json"
                   ).expanduser()
    path = Path(manifest_path).expanduser() if manifest_path else default
    if path.is_file():
        with contextlib.suppress(ValueError, OSError):
            manifest = json.loads(path.read_text())
    methods = manifest.get("methods", {})

    def entry(name: str, key: str) -> Any:
        value = methods.get(name)
        return value.get(key) if isinstance(value, dict) else None

    runtime: dict[str, dict[str, Any]] = {}
    for method, env in (("codeseg", "HEDONIC_CODESEG_BIN"), ("bigclam", "HEDONIC_BIGCLAM_BIN"),
                        ("fox", "HEDONIC_FOX_BIN"), ("neo_kmeans", "HEDONIC_NEO_BIN")):
        binary = os.environ.get(env) or entry(method, "path")
        if binary:
            runtime[method] = {"binary": str(binary)}
    command = os.environ.get("HEDONIC_CODESEG_NCGAME_COMMAND") or entry("ncgame", "command")
    if command:
        runtime["ncgame"] = {"command": str(command)}
    return runtime


def ncgame_runner(inner: Callable) -> Callable:
    """Append the calibrated NcGame flags to the command template before the dispatcher formats it."""
    def run(method, graph, params, seed):
        if method == "ncgame" and params.get("command"):
            extra = []
            if "stop_criterion" in params:
                extra += ["--stop-criterion", str(float(params["stop_criterion"]))]
            if "similarity_type" in params:
                extra += ["--similarity-type", str(params["similarity_type"])]
            params = {**params, "command": f"{params['command']} {' '.join(extra)}".strip()}
        return inner(method, graph, params, seed)
    return run


def _pickled_graph(network: str, cover: str) -> ig.Graph | None:
    """The covered-induced graph ``hedonic pickle`` already wrote, as a plain graph (Friendster's raw edge list is 9.4 GB
    and streaming it again held 85 GB; the pickle is the same graph). ``None`` when there is no pickle."""
    from hedonic.experiments.overlapping import snap_pickle

    path = snap_pickle.pickle_path(snap_pickle.default_out_dir(), network, cover)
    if not path.is_file():
        return None
    game = snap_pickle.load_game(path)
    return ig.Graph(n=game.vcount(), edges=game.get_edgelist())


def load_context(target: str, *, network_root: str | None, induce: str = "covered", seed: int = 0,
                 budget: Budget | None = None, ledger_dir: Path | None = None,
                 runtime: dict | None = None) -> Context:
    """``G[S]`` of a ``network:cover`` target. Only the graph is kept: the communities are not read past the vertex set."""
    from hedonic.experiments.config import NETWORKS_DIR, expand_path
    from hedonic.experiments.overlapping import cover_audit, quickstart as qs

    base, variant = qs.split_target(target)
    root = expand_path(network_root or NETWORKS_DIR).resolve()
    graph = _pickled_graph(base, variant or "all") if induce == "covered" else None
    if graph is None:
        graph = cover_audit.load_graph(base, variant or "all", root, induce)
    return Context(graph=graph, seed=seed, budget=budget or Budget(), ledger_dir=ledger_dir,
                   runner=ncgame_runner(default_runner), runtime=runtime or {}, label=target)


def synthetic_graph(n: int = 400, seed: int = 0) -> ig.Graph:
    """A small planted-block graph for ``--smoke`` (blocks of 40; the planted labels are never given to a calibration)."""
    rng = random.Random(f"synthetic|{seed}")
    block = 40
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)
             if rng.random() < (0.25 if u // block == v // block else 0.01)]
    return ig.Graph(n=n, edges=edges)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", default="all", help=f"comma list or 'all' of: {', '.join(CALIBRATORS)}")
    ap.add_argument("--networks", default="dblp:all", help="comma list of network[:cover] targets (G[S] = covered-induced)")
    ap.add_argument("--network-root", help="folder of the SNAP networks (default HEDONIC_NETWORKS_DIR)")
    ap.add_argument("--induce", choices=("covered", "full"), default="covered")
    ap.add_argument("--output-dir", default="artifacts/overlapping/baseline_calibration")
    ap.add_argument("--seed", type=int, default=0, help="seed of the held-out split and of every detector run")
    ap.add_argument("--max-evals", type=int, default=8, help="new evaluations per method and target (0 = no cap)")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="single-core minutes per method and target (0 = no cap)")
    ap.add_argument("--eval-timeout-minutes", type=float, default=60.0)
    ap.add_argument("--runtime-manifest", help="the `hedonic setup` manifest (default ~/.cache/hedonic/codeseg/setup_manifest.json, HEDONIC_CODESEG_CACHE)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also score the neighbours of the fixed-protocol settings (FOX 0.01, NcGame 0.005, CoDeSEG 0.2/0.4)")
    ap.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES,
                    help="independent runs averaged per setting for BigCLAM, NEO-K-Means and FOX, whose runs are replicas "
                         f"(default {DEFAULT_REPLICATES}; 1 = a single run)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; load no graph, run nothing")
    ap.add_argument("--smoke", action="store_true", help="a 400-vertex synthetic graph instead of a SNAP network")
    args = ap.parse_args(argv)
    methods = list(CALIBRATORS) if args.methods == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in CALIBRATORS]
    if unknown:
        ap.error(f"unknown method(s): {', '.join(unknown)}")
    budget = Budget(max_evals=args.max_evals or None, max_seconds=args.max_minutes * 60 or None,
                    eval_timeout=args.eval_timeout_minutes * 60)
    targets = ["smoke"] if args.smoke else [t.strip() for t in args.networks.split(",") if t.strip()]
    out_root = Path(args.output_dir).expanduser()
    if args.dry_run:
        for target in targets:
            n = 400 if args.smoke else 0
            print(f"== {target}  (G[S] not loaded in a dry run{'' if n else ': K grids are shown for n = 300,000'})")
            for row in plan(n or 300_000, 0, methods, budget):
                print(json.dumps(row))
        return 0
    runtime = runtime_parameters(args.runtime_manifest)
    summary: dict[str, Any] = {"protocol": PROTOCOL, "criterion": CRITERION, "uses_ground_truth": False, "targets": {}}
    for target in targets:
        folder = out_root / target.replace(":", "_")
        if args.smoke:
            ctx = Context(graph=synthetic_graph(), seed=args.seed, budget=budget, ledger_dir=folder / "ledger",
                          runner=ncgame_runner(default_runner), runtime=runtime, label="smoke")
        else:
            ctx = load_context(target, network_root=args.network_root, induce=args.induce, seed=args.seed,
                               budget=budget, ledger_dir=folder / "ledger", runtime=runtime)
        print(f"== {target}: n={ctx.graph.vcount()} m={ctx.graph.ecount()}", flush=True)
        results = []
        with_replicates(ctx, args.replicates)
        for method in methods:
            result = calibrate_method(ctx, method, sensitivity=args.sensitivity)
            results.append(result)
            print(f"  {method:20s} {result.status:11s} {json.dumps(result.params)}  "
                  f"[{result.evaluations} evals, {result.seconds / 60:.1f} min, stop: {result.stop_reason}"
                  f"{'' if result.bracketed is None else ', bracketed' if result.bracketed else ', NOT bracketed'}]",
                  flush=True)
        folder.mkdir(parents=True, exist_ok=True)
        record = {"target": target, "graph": {"n": ctx.graph.vcount(), "m": ctx.graph.ecount(),
                                              "sha256": ctx.fingerprint()},
                  "seed": args.seed, "budget": budget.__dict__, "results": {r.method: r.to_dict() for r in results}}
        (folder / "baseline_tuned.json").write_text(json.dumps(record, indent=1, default=str))
        (folder / "baseline_tuned.toml").write_text(toml_snippet(target, results))
        summary["targets"][target] = str(folder / "baseline_tuned.json")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
