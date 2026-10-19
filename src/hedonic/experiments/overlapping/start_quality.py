"""The start-quality curve: Hedonic local moving from a corrupted ground-truth start (``hedonic-exp start-quality``).

Question: starting from the supplied SNAP ground-truth cover, how much does the final cover lose as the start is
corrupted, up to a fully random start? The graphs are the complete graphs (no ``max_nodes`` bound). They are loaded,
completed and cap-projected exactly as in the ground-truth spectrum (``gt_spectrum.load_job`` and ``Job.start``), and the
detector is the spectrum's worker (``ground_truth_robustness._detector_worker``): Hedonic local moving, the only phase
hedonic 2.0 has (``phase = "local"``; asking for the 1.x ``multilevel`` phase is an error). A pair missing locally is
downloaded first (``snap.ensure_pair``). The settings (gamma, cap) come from the calibration files given with
``--tuned`` (``hoc_tuned.json``, method key ``hedonic``; several files: a later one overrides an earlier one), else from
the frozen ``quickstart.HOC_TUNED``; a pair with neither takes the DBLP settings (the held-out transfer).

Levels. A *swap* level with incidence distance d starts from the projected reference and applies degree-preserving
incidence switches (``perturb_cover_incidence`` below: the output of ``robustness.perturb_cover_incidence``, computed
fast enough for the complete graphs) with ``swaps = round(d * I / 2)``, where I is the number of labelled incidences of
the reference. A swap removes two reference incidences, so the realized distance (the fraction
of reference incidences no longer present) is the requested d unless switches saturate. The *random* level keeps each
vertex's label count and draws its labels uniformly from the reference communities. Every start is scored against the
complete supplied ground truth, as the spectrum does.

Records. One JSON line per condition, appended and fsynced. A rerun reads the file and skips every condition already
present; the key contains the network, level, seed, cap, gamma, phase and the code that ran it (hedonic version and the
SHA-256 of the compiled core, :func:`code_identity`), so a changed setting or a rebuilt detector is a new condition and
records of the 1.x series are never reused. The summary CSV averages over seeds for each (network, level).

Cores. The conditions are independent and each costs a minute or more of single-threaded scoring around a seconds-long
detector run, so ``--workers N`` (default: the performance cores) runs N of them side by side in forked workers that
share the loaded graph. A condition is a deterministic function of its key (the detector and the starts are seeded), so
the records do not depend on N, except ``seconds``: the detector's wall-clock, which is longer when N runs share the
machine. Each record says how many workers ran with it (``workers``). Only the parent writes the records file.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import csv
import json
import math
import os
import random
import re
import statistics
import sys
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from hedonic.experiments import runtime
from hedonic.experiments.config import RUNS_ROOT, expand_path
from hedonic.experiments.resources import config_path as resolve_config
from hedonic.experiments.overlapping import gt_spectrum as gs
from hedonic.experiments.overlapping import ground_truth_robustness as v3
from hedonic.experiments.overlapping import parallel
from hedonic.experiments.overlapping import quickstart as qs
from hedonic.experiments.overlapping import robustness as rb
from hedonic.experiments.overlapping import snap
from hedonic.experiments.overlapping.metrics import evaluate_cover, overlapping_normalized_mutual_information_lfk

STUDY = "start-quality-v1"
SCHEMA_VERSION = 1
COVER = "all"
PHASES = ("local",)  # hedonic 2.0 runs local moving only
DEFAULT_PHASE = "local"
LEGACY_PHASE = "multilevel"  # a 1.x record without a phase field ran this phase; it is never reused
REMOVED_PHASES = ("multilevel", "multi-phase", "multiphase", "hoc_multilevel", "hoc-multilevel")
PHASE_TOKEN = "|phase="  # every resume key has it; a key without it was written before the phase option (1.x)
CODE_TOKEN = "|code="  # the code that ran the condition (hedonic version and core SHA-256)
METHOD = "hedonic"  # the quickstart.HOC_TUNED row and the method key of a calibration file
PHASE_NAME = {"local": "Hedonic local moving"}
PHASE_DESCRIPTION = {"local": "local moves to an equilibrium"}
DEFAULT_NETWORKS = ("dblp", "amazon", "youtube")
DEFAULT_SEEDS = "0-4"
DEFAULT_LEVELS = ("0", "0.005", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "random")
DEFAULT_OUTPUT = RUNS_ROOT / "start-quality"
SWAP_SEED_BASE = 100  # perturbation seed = 100 + detector seed
RANDOM_SEED_BASE = 200  # random-start seed = 200 + detector seed
RESULTS_NAME = "start_quality.jsonl"
SUMMARY_NAME = "start_quality_summary.csv"
CONFIG_KEYS = ("networks", "covers", "seeds", "levels", "output", "phase")
# The durable viewer reads the same line the foreground study prints for every finished condition.
PROGRESS_LINE = re.compile(r"\[(\d+)/(\d+)\] (.*)")
SUMMARY_FIELDS = (
    "network", "level", "level_kind", "requested_distance", "runs",
    "realized_distance_mean", "realized_distance_sd",
    "start_symmetric_f1_mean", "start_symmetric_f1_sd", "start_onmi_mean", "start_onmi_sd",
    "final_symmetric_f1_mean", "final_symmetric_f1_sd", "final_onmi_mean", "final_onmi_sd",
    "seconds_mean", "seconds_sd", "final_communities_mean", "cap", "gamma", "phase",
)


# --------------------------------------------------------------------------- settings and parsing
def parse_phase(value: str) -> str:
    """``local``; the 1.x multilevel phase (``multilevel``, ``multi-phase``, ``hoc_multilevel``) is a clear error."""
    phase = str(value).strip().lower()
    if phase in REMOVED_PHASES:
        raise ValueError(f"phase {value!r} is not available in hedonic 2.0: community_hedonic runs hedonic local "
                         "moving only (the overlapping multilevel phase was removed); use phase = \"local\"")
    if phase not in PHASES:
        raise ValueError(f"phase {value!r} must be one of {', '.join(PHASES)}")
    return phase


@lru_cache(maxsize=1)
def code_identity() -> str:
    """The code a condition ran on, as it enters the record key: ``hedonic-<version>+core-<first 16 hex of its SHA-256>``."""
    sha = runtime.core_source_sha256() or "unknown"
    return f"hedonic-{runtime.release_series()}+core-{sha[:16]}"


def load_tuned(files: str | list[str] | None) -> tuple[dict | None, list[str]]:
    """The Hedonic rows of one or more calibration files (``--tuned``: a path, a comma list, or repeated flags).

    Files are laid over one another in the order given (a later file overrides an earlier one for the same target);
    only the ``hedonic`` rows are kept (``qs.load_tuned_file`` reads ``hoc_local`` rows of older files as ``hedonic``).
    Returns ({"hedonic": {target: params}} or None, the resolved file paths).
    """
    names = qs.tuned_files(files) if files else []
    if not names:
        return None, []
    merged: dict[str, dict] = {}
    for name in names:
        for target, values in (qs.load_tuned_file(name).get(METHOD) or {}).items():
            merged.setdefault(target, {}).update(values)
    return {METHOD: merged}, [str(expand_path(name).resolve()) for name in names]


def tuned_setting(target: str, phase: str, calibrated: dict | None = None) -> dict[str, Any]:
    """The tuned (cap, gamma) of a network or of a ``network:cover`` target.

    A calibration file (``--tuned``) comes first and quickstart.HOC_TUNED second; in each the pair's own entry
    (``dblp:top5k``; a plain network is its ``all`` cover) beats its network's. A network with neither (YouTube,
    LiveJournal before they are calibrated) takes the DBLP settings, the held-out transfer the paper has always used.

    ``calibrated`` is the result of :func:`load_tuned` ({"hedonic": {target: params}}).
    """
    parse_phase(phase)
    frozen = qs.HOC_TUNED[METHOD]
    base, cover = qs.split_target(target)
    keys = [f"{base}:{cover or COVER}", target, base]
    for rows, source in (((calibrated or {}).get(METHOD) or {}, "calibration file"),
                         (frozen, "quickstart.HOC_TUNED")):
        for key in keys:
            if key in rows:
                values = rows[key]
                return {"cap": int(values["max_memberships"]), "gamma": float(values["absolute_resolution"]),
                        "source": source}
    if "dblp" not in frozen:
        raise ValueError(f"no tuned Hedonic setting for {target!r}")
    values = frozen["dblp"]
    return {"cap": int(values["max_memberships"]), "gamma": float(values["absolute_resolution"]),
            "source": "DBLP settings, transferred"}


def target_name(target: str) -> str:
    """A target's name in the records: the ``all`` cover is the benchmark default and is named by its network alone, so
    the records written before this study had covers keep their keys."""
    base, cover = qs.split_target(target)
    return base if cover in (None, COVER) else f"{base}:{cover}"


def parse_networks(value: str, phase: str = DEFAULT_PHASE, calibrated: dict | None = None,
                   covers: "str | list[str] | None" = None) -> list[str]:
    """Networks, each optionally with the cover to analyse (``dblp:top5k``); ``covers`` crosses every network with
    those covers, network by network."""
    names = [part.strip().lower() for part in str(value).split(",") if part.strip()]
    if not names:
        raise ValueError("no networks given")
    if covers:
        names = qs.expand_covers(names, covers)
    targets = [target_name(t) for t in qs.parse_targets(",".join(names))]
    if len(set(targets)) != len(targets):
        raise ValueError(f"duplicate network in {value!r}")
    unknown = [t for t in targets if qs.split_target(t)[0] not in snap.SPECS]
    if unknown:
        raise ValueError(f"unknown network(s) {', '.join(unknown)}; choose from {', '.join(sorted(snap.SPECS))}")
    for name in targets:
        tuned_setting(name, phase, calibrated)  # fails early if a network has no tuned setting for this phase
    return targets


def parse_levels(value: str) -> list[float | str]:
    levels: list[float | str] = []
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "random":
            token: float | str = "random"
        else:
            try:
                number = float(part)
            except ValueError as exc:
                raise ValueError(f"level {part!r} is neither a number nor 'random'") from exc
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"incidence distance {part} must lie in [0, 1]")
            token = number
        if token in levels:
            raise ValueError(f"duplicate level {part}")
        levels.append(token)
    if not levels:
        raise ValueError("no levels given")
    return levels


def label_of(level: float | str) -> str:
    return "random" if level == "random" else f"{float(level):g}"


def token_of(level: float | str) -> str:
    return "random" if level == "random" else repr(float(level))


def config_defaults(path: str) -> dict[str, str]:
    """The ``[start_quality]`` table as command-line defaults (lists joined into comma lists).

    ``path`` is a file, or the name of a shipped config (``start-quality-hoc-local``; ``resources.config_path``)."""
    file = resolve_config(path)
    if not file.is_file():
        raise ValueError(f"no such config: {file}")
    data = tomllib.loads(file.read_text())
    if "start_quality" not in data:
        raise ValueError(f"{file} has no [start_quality] table")
    table = dict(data["start_quality"])
    if "output_dir" in table:  # the same key as the flag
        table.setdefault("output", table.pop("output_dir"))
    unknown = sorted(set(table) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"{file}: unknown key(s) {', '.join(unknown)}; allowed: {', '.join(CONFIG_KEYS)}")
    return {key: ",".join(map(str, value)) if isinstance(value, list) else str(value) for key, value in table.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp start-quality",
        description=("Start-quality curve: Hedonic local moving from the SNAP ground-truth cover corrupted to a "
                     "random start, on the complete graphs at the calibrated settings (resumable; one JSON line per "
                     "condition)"),
    )
    parser.add_argument("--config", help="TOML file with a [start_quality] table, or the name of a shipped config "
                                         "(start-quality-hoc-local); explicit flags override it")
    parser.add_argument("--networks", help="comma list of dblp, amazon, youtube, livejournal (default: dblp, amazon, "
                                           "youtube); network:cover (dblp:top5k) names the cover to analyse")
    parser.add_argument("--covers", help="ground-truth covers to cross with every network, e.g. all,top5k: each pair is "
                                         "analysed on the graph its cover induces")
    parser.add_argument("--phase", help="detector phase: local (the only one in hedonic 2.0; the 1.x multilevel phase "
                                        "is refused)")
    parser.add_argument("--seeds", help="detector seeds, e.g. 0-4 (default 0-4)")
    parser.add_argument("--levels", help="comma list of incidence distances in [0, 1] and/or 'random' "
                                         "(default 0,0.005,0.02,0.05,0.1,0.2,0.5,1,random)")
    parser.add_argument("--output", "--output-dir", dest="output",
                        help="output folder for start_quality.jsonl and the summary CSV "
                             f"(default: the config's output, else {DEFAULT_OUTPUT})")
    parser.add_argument("--tuned", action="append", metavar="FILE[,FILE]",
                        help="calibrated settings (hoc_tuned.json of `hedonic run calibrate`, method key hedonic) "
                             "instead of quickstart.HOC_TUNED; a comma list or repeated flags, a later file overriding "
                             "an earlier one")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="conditions run side by side in forked workers (default: the performance cores, or "
                             "HEDONIC_WORKERS); 1 runs them one at a time in this process")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; load no graph and write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete this study's records first (a plain rerun resumes instead)")
    return parser


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, str | None]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, rest = pre.parse_known_args(argv)
    parser = build_parser()
    if known.config is not None:
        parser.set_defaults(**config_defaults(known.config))
    return parser.parse_args(rest), known.config


def resolve(argv: list[str]) -> dict[str, Any]:
    """The study's plan from its arguments (config, then flags). Loads no graph and writes nothing."""
    args, config = parse_args(argv)
    phase = parse_phase(args.phase or DEFAULT_PHASE)
    calibrated, tuned_paths = load_tuned(args.tuned)
    networks = parse_networks(args.networks or ",".join(DEFAULT_NETWORKS), phase, calibrated, args.covers)
    seeds = qs.parse_seeds(args.seeds or DEFAULT_SEEDS)
    if not seeds:
        raise ValueError("no seeds given")
    levels = parse_levels(args.levels or ",".join(DEFAULT_LEVELS))
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be at least 1")
    return {
        "config": str(resolve_config(config).resolve()) if config else None,
        "phase": phase,
        "networks": networks,
        "seeds": seeds,
        "levels": levels,
        "output": expand_path(args.output or DEFAULT_OUTPUT).resolve(),
        "tuned": ",".join(tuned_paths) or None,
        "settings": {network: tuned_setting(network, phase, calibrated) for network in networks},
        "workers": parallel.default_workers(args.workers),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
    }


def condition_key(network: str, level: float | str, seed: int, setting: dict[str, Any], phase: str) -> str:
    return (f"{network}|{label_of(level)}|{seed}|cap={setting['cap']}|gamma={setting['gamma']:.17g}"
            f"{PHASE_TOKEN}{phase}{CODE_TOKEN}{code_identity()}")


def resume_key(key: str, phase: str = LEGACY_PHASE) -> str:
    """The resume key of a stored record. A key written before the phase option has no phase token: it gets its phase.
    A 1.x key has no code token, so it never equals a key of this code."""
    return key if PHASE_TOKEN in key else f"{key}{PHASE_TOKEN}{phase}"


def conditions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Every (network, level, seed) condition of the plan, in run order, with its record key."""
    out = []
    for network in plan["networks"]:
        setting = plan["settings"][network]
        for level in plan["levels"]:
            for seed in plan["seeds"]:
                key = condition_key(network, level, seed, setting, plan["phase"])
                out.append({"network": network, "level": level, "seed": seed, "key": key})
    return out


# --------------------------------------------------------------------------- starts
class _SortedPairs:
    """A sorted list of distinct ``(vertex, label)`` pairs kept in blocks: insert, remove and k-th element without
    sorting the whole set again (a block is at most twice ``LOAD`` long, and k-th scans the block lengths)."""

    LOAD = 2048

    def __init__(self, items: list[tuple[int, int]]):
        self.blocks = [items[i:i + self.LOAD] for i in range(0, len(items), self.LOAD)]
        self.maxes = [block[-1] for block in self.blocks]
        self.size = len(items)

    def __len__(self) -> int:
        return self.size

    def kth(self, k: int) -> tuple[int, int]:
        for block in self.blocks:
            if k < len(block):
                return block[k]
            k -= len(block)
        raise IndexError("rank out of range")

    def remove(self, item: tuple[int, int]) -> None:
        index = bisect.bisect_left(self.maxes, item)
        block = self.blocks[index]
        del block[bisect.bisect_left(block, item)]
        self.size -= 1
        if block:
            self.maxes[index] = block[-1]
        else:
            del self.blocks[index], self.maxes[index]

    def add(self, item: tuple[int, int]) -> None:
        if not self.blocks:
            self.blocks, self.maxes, self.size = [[item]], [item], 1
            return
        index = min(bisect.bisect_left(self.maxes, item), len(self.blocks) - 1)
        block = self.blocks[index]
        bisect.insort(block, item)
        self.size += 1
        if len(block) > 2 * self.LOAD:
            half = len(block) // 2
            self.blocks[index:index + 1] = [block[:half], block[half:]]
            self.maxes[index:index + 1] = [block[half - 1], block[-1]]
        else:
            self.maxes[index] = block[-1]


def perturb_cover_incidence(cover: list[list[int]], n_vertices: int, *, swaps: int, seed: int,
                            max_attempts: int | None = None) -> tuple[list[list[int]], dict[str, Any]]:
    """``robustness.perturb_cover_incidence`` with the same output for the same inputs, fast enough for DBLP.

    The reference draws two incidences with ``rng.sample(sorted(incidence), 2)`` at every attempt and rebuilds the body
    of every other community at every accepted switch: on the complete DBLP graph (about a million incidences, five
    thousand communities) a 10% corruption takes days. This keeps the incidences in a sorted structure and the community
    bodies in a set (its size is the cover's, whatever the number of switches: a structure that kept every body ever
    formed used tens of gigabytes), and draws the same two ranks from the same random stream (``sample`` of a sequence depends on
    its length only), so every attempt makes the same decision and the cover, the metadata and the number of attempts are
    identical (``TestFastSwaps`` compares them with the reference)."""
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    canonical = rb.canonicalize_cover(cover, n_vertices=n_vertices)
    memberships = rb.cover_to_vertex_memberships(canonical, n_vertices)
    pairs = sorted((vertex, label) for vertex, labels in enumerate(memberships) for label in labels)
    incidence = set(pairs)
    order = _SortedPairs(pairs)
    community_sets = [frozenset(community) for community in canonical]
    bodies = set(community_sets)  # the current bodies, all distinct: the size of the cover, never more
    rng = random.Random(int(seed))
    attempts_limit = max_attempts if max_attempts is not None else max(100, swaps * 50)
    successful = 0
    attempts = 0
    while successful < swaps and attempts < attempts_limit:
        attempts += 1
        if len(order) < 2:
            break
        rank_a, rank_b = rng.sample(range(len(order)), 2)
        first, second = order.kth(rank_a), order.kth(rank_b)
        vertex_a, label_a = first
        vertex_b, label_b = second
        if vertex_a == vertex_b or label_a == label_b:
            continue
        cross_a = (vertex_a, label_b)
        cross_b = (vertex_b, label_a)
        if cross_a in incidence or cross_b in incidence:
            continue
        candidate_a = (community_sets[label_a] - {vertex_a}) | {vertex_b}
        candidate_b = (community_sets[label_b] - {vertex_b}) | {vertex_a}
        if candidate_a == candidate_b:
            continue
        body_a, body_b = community_sets[label_a], community_sets[label_b]
        bodies.discard(body_a)  # what is left is the bodies of every other label
        bodies.discard(body_b)
        if candidate_a in bodies or candidate_b in bodies:
            bodies.add(body_a)
            bodies.add(body_b)
            continue
        bodies.add(candidate_a)
        bodies.add(candidate_b)
        incidence.remove(first)
        incidence.remove(second)
        incidence.add(cross_a)
        incidence.add(cross_b)
        order.remove(first)
        order.remove(second)
        order.add(cross_a)
        order.add(cross_b)
        community_sets[label_a] = candidate_a
        community_sets[label_b] = candidate_b
        successful += 1

    perturbed_raw = [[] for _ in canonical]
    raw_memberships = [[] for _ in range(n_vertices)]
    for vertex, label in sorted(incidence):
        perturbed_raw[label].append(vertex)
        raw_memberships[vertex].append(label)
    perturbed = rb.canonicalize_cover(perturbed_raw, n_vertices=n_vertices)
    initial_pairs = set((vertex, label) for vertex, labels in enumerate(memberships) for label in labels)
    realized_vertex_f1 = rb._incidence_f1(memberships, raw_memberships)
    metadata = {
        "requested_swaps": int(swaps),
        "successful_swaps": int(successful),
        "attempts": int(attempts),
        "max_attempts": int(attempts_limit),
        "seed": int(seed),
        "realized_incidence_distance": float(1.0 - realized_vertex_f1),
        "realized_incidence_f1": float(realized_vertex_f1),
        "vertex_membership_counts_preserved": list(map(len, memberships)) == list(map(len, raw_memberships)),
        "community_sizes_preserved": sorted(map(len, canonical)) == sorted(map(len, perturbed)),
        "initial_incidence_count": len(initial_pairs),
        "final_incidence_count": sum(map(len, perturbed)),
    }
    return perturbed, metadata


def swap_start(reference_cover: list[list[int]], n_vertices: int, distance: float, *, seed: int) -> dict[str, Any]:
    """Degree-preserving incidence switches of the reference until about ``distance`` of its incidences are lost."""
    memberships = rb.cover_to_vertex_memberships(reference_cover, n_vertices)
    incidences = sum(map(len, memberships))
    swaps = round(distance * incidences / 2)
    cover, meta = perturb_cover_incidence(reference_cover, n_vertices, swaps=swaps, seed=seed)
    return {
        "memberships": rb.cover_to_vertex_memberships(cover, n_vertices),
        "cover": cover,
        "realized_distance": float(meta["realized_incidence_distance"]),
        "swaps_requested": swaps,
        "swaps_successful": int(meta["successful_swaps"]),
        "meta": meta,
        "reference_incidences": incidences,
    }


def random_start(reference: list[list[int]], n_labels: int, *, seed: int) -> list[list[int]]:
    """Each vertex keeps its label count and draws distinct labels uniformly from the ``n_labels`` communities."""
    rng = random.Random(seed)
    pool = range(n_labels)
    return [sorted(rng.sample(pool, len(labels))) for labels in reference]


def contiguous_labels(memberships: list[list[int]]) -> list[list[int]]:
    """The same communities with their labels renumbered 0..k-1 in increasing order of the labels in use.

    The detector refuses labels with gaps. A random start can leave a label of its range undrawn (on complete YouTube
    about three of 14,827), which is an empty community, not a different start. A start whose labels are already
    contiguous comes back unchanged, so nothing else (an earlier record, its seed, its trajectory) is affected."""
    used = sorted({label for labels in memberships for label in labels})
    if used == list(range(len(used))):
        return memberships
    renamed = {label: position for position, label in enumerate(used)}
    return [[renamed[label] for label in labels] for labels in memberships]


def label_loss(reference: list[list[int]], start: list[list[int]]) -> float:
    """Fraction of the reference incidences absent from ``start`` (labels fixed on both sides)."""
    total = sum(map(len, reference))
    lost = sum(len(set(ref) - set(got)) for ref, got in zip(reference, start))
    return lost / total if total else 0.0


# --------------------------------------------------------------------------- one condition
def score(cover: list[list[int]], reference: list[list[int]], n_vertices: int) -> tuple[float, float]:
    """Symmetric best-match F1 (the cover metric of the study) and LFK ONMI against the complete ground truth."""
    metrics = evaluate_cover(cover, reference, n_vertices, compute_omega=False)
    onmi = overlapping_normalized_mutual_information_lfk(cover, reference, n_vertices=n_vertices)
    return float(metrics["symmetric_best_match_f1"]), float(onmi)


def run_condition(job: Any, setting: dict[str, Any], level: float | str, seed: int, key: str,
                  phase: str) -> dict[str, Any]:
    graph = job.graph
    n = graph.vcount()
    cap, gamma = int(setting["cap"]), float(setting["gamma"])
    # the cap projection of the reference when the cap is below its multiplicity; -1 (no cap) keeps the reference
    start = job.start(cap if cap >= 1 else None)
    reference = start.memberships
    incidences = sum(map(len, reference))
    if level == "random":
        start_memberships = random_start(reference, len(start.cover), seed=RANDOM_SEED_BASE + seed)
        start_cover = rb.canonicalize_cover(rb.vertex_memberships_to_cover(start_memberships), n)
        realized = label_loss(reference, start_memberships)
        requested, swaps, successful, perturbation_seed = None, None, None, RANDOM_SEED_BASE + seed
    else:
        built = swap_start(start.cover, n, float(level), seed=SWAP_SEED_BASE + seed)
        start_memberships, start_cover = built["memberships"], built["cover"]
        realized, swaps, successful = built["realized_distance"], built["swaps_requested"], built["swaps_successful"]
        requested, perturbation_seed = float(level), SWAP_SEED_BASE + seed
    start_f1, start_onmi = score(start_cover, job.gt_cover, n)
    started = time.monotonic()
    # allow_isolation is True, as in the calibrated rows (methods._hedonic_local).
    outcome = v3._detector_worker(graph, contiguous_labels(start_memberships), cap, gamma, phase, True, seed)
    seconds = time.monotonic() - started
    final_raw = v3._normalize_raw_memberships(outcome["final_memberships"], n, cap if cap >= 1 else n)
    final_cover = rb.canonicalize_cover(rb.vertex_memberships_to_cover(final_raw), n)
    final_f1, final_onmi = score(final_cover, job.gt_cover, n)
    return {
        "study": STUDY,
        "schema_version": SCHEMA_VERSION,
        "key": key,
        "phase": phase,
        "network": job.dataset,
        "cover": job.cover,
        "level": label_of(level),
        "level_kind": "random" if level == "random" else "swap",
        "requested_distance": requested,
        "realized_distance": realized,
        "swaps_requested": swaps,
        "swaps_successful": successful,
        "detector_seed": seed,
        "perturbation_seed": perturbation_seed,
        "reference_incidences": incidences,
        "start_communities": len(start_cover),
        "start_is_cap_projected": start.projection is not None,
        "start_symmetric_f1": start_f1,
        "start_onmi": start_onmi,
        "final_symmetric_f1": final_f1,
        "final_onmi": final_onmi,
        "seconds": seconds,
        "final_communities": len(final_cover),
        "final_max_memberships": max(map(len, final_raw)),
        "cap": cap,
        "gamma": gamma,
        "n_vertices": n,
        "n_edges": graph.ecount(),
        "code": code_identity(),
        "runtime": runtime.runtime_versions(),
    }


# --------------------------------------------------------------------------- records
def read_records(path: Path) -> dict[str, dict[str, Any]]:
    """Valid records by resume key (a torn or corrupt line is skipped; the first record of a key wins). A record without
    a phase field was written before the phase option (1.x), so it is HOC-multilevel."""
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and isinstance(record.get("key"), str):
            phase = str(record.setdefault("phase", LEGACY_PHASE))
            records.setdefault(resume_key(record["key"], phase), record)
    return records


def repair_tail(path: Path) -> None:
    """Cut a torn last line (an interrupted write) so the next record starts on a fresh line."""
    if not path.is_file():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        with path.open("r+b") as handle:
            handle.truncate(raw.rfind(b"\n") + 1)


def append_record(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else None)


def write_summary(path: Path, todo: list[dict[str, Any]], records: dict[str, dict[str, Any]],
                  settings: dict[str, dict[str, Any]]) -> None:
    """Mean and sample standard deviation over seeds for each (network, level), from the current plan's records."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for condition in todo:
        if condition["key"] in records:
            groups.setdefault((condition["network"], label_of(condition["level"])), []).append(records[condition["key"]])
    rows = []
    for (network, level), group in groups.items():
        row: dict[str, Any] = {"network": network, "level": level, "level_kind": group[0]["level_kind"],
                               "requested_distance": group[0]["requested_distance"], "runs": len(group),
                               "cap": settings[network]["cap"], "gamma": settings[network]["gamma"]}
        for field, out in (("realized_distance", "realized_distance"), ("start_symmetric_f1", "start_symmetric_f1"),
                           ("start_onmi", "start_onmi"), ("final_symmetric_f1", "final_symmetric_f1"),
                           ("final_onmi", "final_onmi"), ("seconds", "seconds")):
            mean, sd = _mean_sd([float(item[field]) for item in group])
            row[f"{out}_mean"], row[f"{out}_sd"] = mean, sd
        row["final_communities_mean"] = _mean_sd([float(item["final_communities"]) for item in group])[0]
        row["phase"] = group[0]["phase"]
        rows.append(row)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


# --------------------------------------------------------------------------- study
def load_network_job(target: str) -> Any:
    """The complete graph and the cover of a target (``dblp``: the ``all`` cover, the benchmark default;
    ``dblp:top5k``: that cover), prepared as the ground-truth spectrum prepares them.

    A pair missing from the networks folder (``snap.networks_root()``) is downloaded first (``snap.ensure_pair``)."""
    network, cover = qs.split_target(target)
    files = snap.ensure_pair(network, cover or COVER, download=True, say=lambda line: print(f"  {line}", flush=True))
    from hedonic.experiments.overlapping import userconfig

    # A graph too large to build whole (Friendster) is read as a stream into the runner's cache (<cache_dir>/snap), the
    # one the benchmark and its cover audit use, so the stream is read once for all of them.
    cache = expand_path(userconfig.defaults().get("cache_dir") or "~/.cache/hedonic") / "snap"
    options = {"smoke": False, "data_root": str(files.root), "provision": False,
               "max_download_bytes": None, "policy": "covered-induced", "max_nodes": None,
               "snap_cache_dir": str(cache)}
    return gs.load_job(options, {"dataset": files.network, "cover": files.cover, "root_kind": None})


def progress_line(done: int, total: int, record: dict[str, Any]) -> str:
    return (f"[{done}/{total}] {record['network']} level={record['level']} seed={record['detector_seed']} "
            f"seconds={record['seconds']:.2f} start_f1={record['start_symmetric_f1']:.4f} "
            f"final_f1={record['final_symmetric_f1']:.4f}")


def run_study(plan: dict[str, Any], *, loader: Callable[[str], Any] = load_network_job,
              emit: Callable[[str], Any] = print) -> int:
    out = plan["output"]
    parse_phase(plan["phase"])
    out.mkdir(parents=True, exist_ok=True)
    results = out / RESULTS_NAME
    if plan["overwrite"]:
        for name in (RESULTS_NAME, SUMMARY_NAME):
            (out / name).unlink(missing_ok=True)
    repair_tail(results)
    records = read_records(results)
    todo = conditions(plan)
    count = sum(1 for condition in todo if condition["key"] in records)
    emit(f"start-quality: {len(todo)} conditions; {count} already in {results}; {len(todo) - count} to run")
    for network in plan["networks"]:
        pending = [condition for condition in todo if condition["network"] == network and condition["key"] not in records]
        if not pending:
            continue
        job = loader(network)  # loaded here, before any fork, so the workers share it
        workers = min(int(plan.get("workers") or 1), len(pending))
        setting = plan["settings"][network]
        job.start(int(setting["cap"]) if int(setting["cap"]) >= 1 else None)  # the cap projection is cached on the job: computed once here, not once per worker

        def task(condition: dict[str, Any], job: Any = job, setting: dict[str, Any] = setting) -> dict[str, Any]:
            return run_condition(job, setting, condition["level"], condition["seed"], condition["key"], plan["phase"])

        for index, record in parallel.run_forked(task, pending, workers):
            record["workers"] = workers  # how many conditions shared the machine with this one
            append_record(results, record)
            records[pending[index]["key"]] = record
            count += 1
            emit(progress_line(count, len(todo), record))
    write_summary(out / SUMMARY_NAME, todo, records, plan["settings"])
    emit(f"start-quality: wrote {results} and {out / SUMMARY_NAME}")
    return 0


def plan_lines(plan: dict[str, Any]) -> list[str]:
    todo = conditions(plan)
    present = sum(1 for condition in todo if condition["key"] in read_records(plan["output"] / RESULTS_NAME))
    levels = ", ".join(label_of(level) for level in plan["levels"])
    phase = plan["phase"]
    lines = [
        f"hedonic-exp start-quality · {PHASE_NAME[phase]} from corrupted ground-truth starts (start-quality-v1)",
        f"  graphs   {', '.join(plan['networks'])} · complete graphs · SNAP cover of each (a plain name is the {COVER} "
        f"cover) · covered-induced analysis graph",
        f"  phase    {phase} · {PHASE_DESCRIPTION[phase]} · code {code_identity()}",
        f"  levels   {levels}",
        f"  seeds    {', '.join(map(str, plan['seeds']))}",
        f"  grid     {len(todo):,} conditions = {len(plan['networks'])} networks × {len(plan['levels'])} levels "
        f"× {len(plan['seeds'])} seeds",
        f"  workers  {plan.get('workers', 1)} conditions side by side"
        + (" (seconds is the detector's wall-clock with that many running together)" if plan.get("workers", 1) > 1 else ""),
    ]
    for network in plan["networks"]:
        setting = plan["settings"][network]
        lines.append(f"  {network:<18} cap {setting['cap']} · gamma {setting['gamma']:.6g} ({setting['source']})")
    lines += [
        f"  records  {plan['output'] / RESULTS_NAME} (one JSON line per condition)",
        f"  summary  {plan['output'] / SUMMARY_NAME}",
        f"  present  {present:,} of {len(todo):,} conditions already recorded"
        + (" (--overwrite: they are deleted first)" if plan["overwrite"] else " (a run resumes them)"),
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        plan = resolve(argv)
    except SystemExit as exc:  # argparse printed the help or a usage error
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic-exp start-quality: {exc}", file=sys.stderr)
        return 2
    if plan["dry_run"]:
        print("\n".join(plan_lines(plan)))
        print("  dry run: no graph loaded, nothing written")
        return 0
    return run_study(plan)


# --------------------------------------------------------------------------- durable run (hedonic run startcurve)
def _front_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic run startcurve",
        description=("start-quality curve (Hedonic local moving from corrupted ground-truth starts) as a durable run "
                     "with a live viewer, ETA, stop and resume"),
    )
    parser.add_argument("--config", required=True,
                        help="TOML file with a [start_quality] table, or the name of a shipped config "
                             "(start-quality-hoc-local)")
    parser.add_argument("--networks", help="override the config's networks (comma list)")
    parser.add_argument("--covers", help="override the config's covers (comma list, e.g. all,top5k)")
    parser.add_argument("--tuned", action="append", metavar="FILE[,FILE]",
                        help="calibrated settings (hoc_tuned.json of `hedonic run calibrate`, method key hedonic) "
                             "instead of quickstart.HOC_TUNED; a comma list or repeated flags")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="conditions run side by side (default: the performance cores, or HEDONIC_WORKERS)")
    parser.add_argument("--name", help="run name (default: the config file's name)")
    parser.add_argument("--output-dir", help="run folder, which holds the records (default: the config's output, "
                                             "else <output_dir>/<name>)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete this study's records before starting (a plain start resumes them)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the paths; start nothing and write nothing")
    parser.add_argument("--detach", action="store_true", help="start in the background and return")
    return parser


def front_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = resolve_config(args.config).resolve()
    if not config.is_file():
        raise ValueError(f"no such config: {args.config} (neither a file nor a shipped config)")
    name = qs.parse_name(args.name or config.stem)
    study = ["--config", str(config)]
    if args.networks:
        study += ["--networks", args.networks]
    if args.covers:
        study += ["--covers", args.covers]
    for tuned in args.tuned or []:
        study += ["--tuned", tuned]
    if args.workers is not None:
        study += ["--workers", str(args.workers)]
    if args.output_dir:
        study += ["--output", str(expand_path(args.output_dir).resolve())]
    elif "output" not in config_defaults(str(config)):  # a named run: <output_dir>/<name>
        from hedonic.experiments.overlapping import userconfig

        study += ["--output", str((expand_path(userconfig.defaults().get("output_dir") or RUNS_ROOT) / name).resolve())]
    plan = resolve(study)
    plan["name"] = name
    return plan


def front_study(argv: list[str]) -> dict[str, Any]:
    """The plan of ``hedonic run startcurve ARGV``, run name and folder included. Loads no graph and writes nothing."""
    return front_plan(_front_parser().parse_args(argv))


def launch_argv(plan: dict[str, Any]) -> list[str]:
    """Every setting spelled out (no config file), so a resume rebuilds the same plan even if the file moves."""
    return ["--networks", ",".join(plan["networks"]), "--phase", plan["phase"],
            "--seeds", ",".join(map(str, plan["seeds"])), "--levels", ",".join(token_of(level) for level in plan["levels"]),
            "--output", str(plan["output"]), "--workers", str(plan.get("workers", 1))] + (["--tuned", plan["tuned"]] if plan.get("tuned") else [])


def describe(plan: dict[str, Any]) -> str:
    return (f"start-quality curve · {len(plan['networks'])} pairs × {len(plan['levels'])} levels × "
            f"{len(plan['seeds'])} seeds = {len(conditions(plan)):,} {PHASE_NAME[plan['phase']]} runs")


def front_lines(plan: dict[str, Any]) -> list[str]:
    return [f"hedonic run startcurve · {plan['name']} · {describe(plan)}",
            f"  config   {plan['config']}",
            f"  folder   {plan['output']} (records, summary, progress.json, worker.log)"] + plan_lines(plan)[1:]


def front_main(argv: list[str] | None = None) -> int:
    """``hedonic run startcurve --config FILE``: a durable run; ``--dry-run`` only prints the plan."""
    from hedonic.experiments.overlapping import runmanager
    from hedonic.experiments.overlapping.tui import style

    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = _front_parser().parse_args(argv)
        plan = front_plan(args)
    except SystemExit as exc:  # argparse printed the help or a usage error
        if isinstance(exc.code, str):
            print(f"hedonic run startcurve: {exc.code}", file=sys.stderr)
        return exc.code if isinstance(exc.code, int) else 2
    except ValueError as exc:
        print(f"hedonic run startcurve: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("\n".join(front_lines(plan)))
        print(style("  dry run: nothing started, nothing written", "dim"))
        return 0
    if args.overwrite:
        try:
            runmanager._refuse_if_running(plan["name"])  # never delete records under a live worker
        except SystemExit as exc:
            print(f"hedonic run startcurve: {exc.code}", file=sys.stderr)
            return 2
        for name in (RESULTS_NAME, SUMMARY_NAME):
            (plan["output"] / name).unlink(missing_ok=True)
    entry = runmanager.launch_startcurve(launch_argv(plan), plan["name"], str(plan["output"]))
    where = (f"tmux session {entry['session']}" if entry["backend"] == "tmux"
             else f"background process (log {entry['log']})")
    print(style(f"  started {entry['name']}", "green", "bold") + f" in a {where}")
    print(style(f"  it survives closing this terminal · reopen: hedonic run attach {entry['name']} · "
                f"stop: hedonic run stop {entry['name']}", "dim"))
    if args.detach or not sys.stdout.isatty():
        return 0
    time.sleep(0.5)
    return runmanager.view(entry)


def work(name: str, entry: dict[str, Any], prog: dict[str, Any]) -> int:
    """Worker body of a startcurve run: the study with its progress mirrored into progress.json."""
    from hedonic.experiments.overlapping import runmanager as rm
    from hedonic.experiments.overlapping.controlled_overlap_config import ProgressTap
    from hedonic.experiments.overlapping.tui import style

    plan = resolve(entry["argv"])
    todo = conditions(plan)
    results = plan["output"] / RESULTS_NAME
    already = sum(1 for condition in todo if condition["key"] in read_records(results))
    ppath = rm.progress_path(entry)
    prog.update({"name": name, "kind": "startcurve", "status": "setup", "pid": os.getpid(), "started": time.time(),
                 "updated": time.time(), "describe": describe(plan), "command": entry["command"],
                 "environment": qs.environment(), "done": already, "total": len(todo), "resumed_from": already,
                 "eta_seconds": None, "stage": "starting the study", "output": str(results), "setup": [], "plan": []})
    rm.write_json(ppath, prog)
    print(style(f"hedonic run {name}", "bold") + " · " + describe(plan))
    print(style(qs.environment_line(), "dim"))
    print(style("Ctrl-C does not stop this run. Detach: Ctrl-b d · stop: hedonic run stop " + name, "dim"), flush=True)
    first_seen: dict[str, float] = {}

    def observe(line: str) -> None:
        found = PROGRESS_LINE.match(line)
        if not found:
            return
        done, seen_total = int(found[1]), int(found[2])
        first_seen.setdefault("at", time.time())
        fresh = done - prog["resumed_from"]
        eta = (time.time() - first_seen["at"]) / (fresh - 1) * (seen_total - done) if fresh > 1 else None
        prog.update(status="running", done=done, total=seen_total, last=found[3], eta_seconds=eta,
                    stage=f"condition {done:,}/{seen_total:,}", updated=time.time())
        rm.write_json(ppath, prog)

    prog.update(status="running", stage="loading the first network", updated=time.time())
    rm.write_json(ppath, prog)
    with contextlib.redirect_stdout(ProgressTap(sys.stdout, observe)):
        code = run_study(plan)
    if code != 0:
        raise RuntimeError(f"the start-quality study returned {code}")
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
             style("  " + prog.get("describe", "start-quality curve"), "dim")]
    env = prog.get("environment") or {}
    if env:
        lines.append(style(f"  hedonic {env.get('hedonic')} · core {str(env.get('hedonic_core_sha256') or '?')[:12]} · "
                           f"python-igraph {env.get('python_igraph')} · Python {env.get('python')}", "dim"))
    if total:
        width = 36
        filled = int(width * min(done, total) / total)
        tail = f"  {done:,}/{total:,} conditions · elapsed {qs.fmt_time(end - started)}"
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
        lines.append(style(f"\n  stopped by you; resume skips the conditions already recorded: "
                           f"hedonic run resume {entry['name']}", "yellow"))
    if status == "completed":
        lines.append(style(f"\n  records  {prog.get('output', entry['output_dir'])}", "dim"))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
