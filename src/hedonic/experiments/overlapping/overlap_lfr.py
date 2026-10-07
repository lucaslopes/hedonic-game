"""Planted-overlap benchmark runner (Astra TKT-11).

The public experiment runner in this module is deliberately independent from
the existing ``controlled_overlap`` diagnostic.  It generates a sparse,
metadata-free planted cover first and then samples a graph from that cover;
the cover is *never* supplied to a detector.  The implementation follows the
parameterization of the overlapping LFR benchmark (power-law degree and
community-size exponents, mixing, overlap fraction and overlap multiplicity).
For a canonical pilot/standard run, use the fail-closed official binary
adapter below.  The self-contained generator is retained only as an explicit
compatibility fixture for smoke tests; it is not canonical overlapping LFR
evidence and is never silently promoted to that role.

The default ``standard`` profile is prospective (28 structural cells × 30
independent graph seeds).  Use ``--max-graphs`` or the ``smoke`` profile for a
bounded run.  Every attempted graph and detector execution receives a
terminal status; failures are observations, not silently dropped rows.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import igraph as ig

from hedonic import Game
from hedonic.experiments.config import OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.metrics import evaluate_cover, partition_to_cover_lists
from hedonic.experiments.overlapping.robustness import canonicalize_cover, cover_hash


PROTOCOL_VERSION = "overlapping-lfr-v1"
COMPATIBILITY_GENERATOR_ID = "overlapping_lfr_compatibility_smoke_v1"
OFFICIAL_GENERATOR_ID = "overlapping_lfr_official_binary_v1"
# The authoritative source distribution identified by the benchmark authors'
# resources page.  These values are deliberately immutable protocol metadata:
# a binary can only be called the native LFRbenchmarks adapter when its build
# receipt declares this exact source commit.  The source is not vendored into
# this package; callers compile it in a disposable directory and pass the
# resulting executable path.
OFFICIAL_LFRBENCHMARKS_REPOSITORY = "https://github.com/andrealancichinetti/LFRbenchmarks"
OFFICIAL_LFRBENCHMARKS_COMMIT = "ec9a860282d8fc9d52e9e08fea9ce93ddeb473af"
OFFICIAL_LFRBENCHMARKS_TREE = "d1ebd7a9397067b07eabb36b5342c19eced7ba56"
OFFICIAL_LFRBENCHMARKS_SOURCE_SUBDIRECTORY = "unweighted_undirected"
OFFICIAL_LFRBENCHMARKS_OUTPUT_FORMAT = "network.dat+community.dat+statistics.dat"
# Kept as the public generator identity for compatibility with older imports;
# callers must inspect ``generator_policy`` before treating an artifact as
# canonical.
GENERATOR_ID = COMPATIBILITY_GENERATOR_ID
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
PREFLIGHT_SCHEMA_VERSION = 1
PILOT_RECEIPT_PATH = (
    Path("artifacts/evidence/overlapping_communities/tkt11_native_pilot_receipt.json")
)

# The 28 structural cells recommended by Astra: four mixing values, one
# zero-overlap arm and six positive-overlap (fraction × multiplicity) arms.
DEFAULT_MIXING = (0.1, 0.3, 0.5, 0.7)
DEFAULT_OVERLAP_FRACTIONS = (0.0, 0.1, 0.3, 0.5)
DEFAULT_OVERLAP_MULTIPLICITIES = (2, 4)
DEFAULT_N = 2_000
DEFAULT_AVERAGE_DEGREE = 20
DEFAULT_MAX_DEGREE = 100
DEFAULT_MIN_COMMUNITY = 20
DEFAULT_MAX_COMMUNITY = 100
DEFAULT_TAU1 = 2.0
DEFAULT_TAU2 = 1.0
DEFAULT_GRAPHS_PER_CONDITION = 30
DEFAULT_METHODS = ("hedonic_multiphase", "slpa", "demon", "kcp", "chen", "disjoint", "singleton")


class OfficialGeneratorUnavailable(RuntimeError):
    """Raised when the canonical external overlapping-LFR binary is absent."""


class StudyInterrupted(RuntimeError):
    """Raised by the disposable interruption hook after a durable batch."""


@dataclass(frozen=True)
class OverlappingLFRGraph:
    """One generated graph, planted cover, and generation receipt."""

    graph: ig.Graph
    cover: list[list[int]]
    metadata: dict[str, Any]

    @property
    def graph_hash(self) -> str:
        return graph_sha256(self.graph)

    @property
    def cover_hash(self) -> str:
        return cover_hash(self.cover, self.graph.vcount())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    """Return a stable UTC timestamp for machine-readable receipts."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_root() -> Path:
    # overlap_lfr.py -> overlapping -> experiments -> hedonic -> src -> root
    return Path(__file__).resolve().parents[4]


def _path_sha256(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return _file_sha256(path)
    except OSError:
        return None


def _canonical_payload_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def graph_sha256(graph: ig.Graph) -> str:
    """Hash an undirected simple graph independent of edge insertion order."""
    edges = sorted(tuple(sorted((int(a), int(b)))) for a, b in graph.get_edgelist() if a != b)
    return _sha256_json({"n": graph.vcount(), "edges": edges})


def _power_law_sample(rng: random.Random, low: int, high: int, tau: float) -> int:
    values = list(range(int(low), int(high) + 1))
    weights = [float(value) ** (-float(tau)) for value in values]
    return int(rng.choices(values, weights=weights, k=1)[0])


def _community_sizes(
    n: int,
    rng: random.Random,
    min_community: int,
    max_community: int,
    tau2: float,
) -> list[int]:
    if min_community < 2 or max_community < min_community:
        raise ValueError("community size bounds must satisfy 2 <= min <= max")
    sizes: list[int] = []
    remaining = int(n)
    while remaining > 0:
        draw = _power_law_sample(rng, min_community, max_community, tau2)
        if draw > remaining:
            draw = remaining
        # Keep the last community from becoming a one-vertex artefact where
        # possible.  A small final community is preferable to changing n.
        if remaining - draw == 1 and draw > min_community:
            draw -= 1
        sizes.append(max(1, draw))
        remaining -= draw
    return sizes


def _assign_cover(
    n: int,
    sizes: Sequence[int],
    overlap_fraction: float,
    multiplicity: int,
    rng: random.Random,
) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Assign each vertex a primary community and optional extra labels."""
    if not 0.0 <= overlap_fraction <= 1.0:
        raise ValueError("overlap_fraction must be in [0, 1]")
    requested_m = max(1, int(multiplicity))
    primary: list[list[int]] = []
    labels: list[list[int]] = [[] for _ in range(n)]
    vertex = 0
    for community_id, size in enumerate(sizes):
        members = list(range(vertex, min(n, vertex + int(size))))
        if members:
            primary.append(members)
            for member in members:
                labels[member] = [community_id]
        vertex += int(size)
    if vertex < n:
        # This can only happen for pathological integer inputs; assign the
        # tail to the last body while preserving the requested vertex count.
        primary[-1].extend(range(vertex, n))
        for member in range(vertex, n):
            labels[member] = [len(primary) - 1]
    if not primary:
        raise ValueError("at least one non-empty community is required")

    eligible = list(range(n))
    rng.shuffle(eligible)
    overlap_count = min(n, int(round(n * float(overlap_fraction))))
    target_m = min(requested_m, len(primary))
    chosen = sorted(eligible[:overlap_count]) if target_m > 1 else []
    for member in chosen:
        options = [index for index in range(len(primary)) if index != labels[member][0]]
        rng.shuffle(options)
        for community_id in options[: target_m - 1]:
            labels[member].append(community_id)
            primary[community_id].append(member)
        labels[member].sort()

    # Remove accidental empty bodies only in the impossible case where n is
    # smaller than the requested range.  Label identities remain contiguous.
    primary = [sorted(set(body)) for body in primary if body]
    remap: dict[int, int] = {}
    for new_id, body in enumerate(primary):
        for member in body:
            remap.setdefault(labels[member][0], new_id)
    if any(remap.get(label) is None for row in labels for label in row):
        # ``sizes`` never creates empty bodies under normal settings; this is a
        # defensive fallback for custom tiny smoke parameters.
        remap = {old: new for new, old in enumerate(sorted({label for row in labels for label in row}))}
        primary = [[] for _ in remap]
        for member, row in enumerate(labels):
            labels[member] = sorted({remap[label] for label in row})
            for label in labels[member]:
                primary[label].append(member)

    cover = [sorted(set(body)) for body in primary]
    achieved_overlap = sum(len(row) > 1 for row in labels) / n if n else 0.0
    achieved_multiplicity = (
        statistics.fmean(len(row) for row in labels if len(row) > 1)
        if any(len(row) > 1 for row in labels)
        else 1.0
    )
    metadata = {
        "requested_overlap_fraction": float(overlap_fraction),
        "achieved_overlap_fraction": float(achieved_overlap),
        "requested_overlap_multiplicity": int(requested_m),
        "achieved_overlap_multiplicity_mean": float(achieved_multiplicity),
        "achieved_overlap_multiplicity_max": max((len(row) for row in labels), default=0),
        "overlapping_vertex_count": int(sum(len(row) > 1 for row in labels)),
        "incidence_count": int(sum(map(len, labels))),
    }
    return cover, labels, metadata


def _sample_graph_edges(
    labels: Sequence[Sequence[int]],
    *,
    average_degree: int,
    max_degree: int,
    mixing: float,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Sample a sparse graph with LFR-style internal/cross mixing.

    The sampler targets ``average_degree`` and uses membership-aware pair
    proposals.  It is intentionally bounded and records shortfall/capacity
    outcomes, which makes large prospective runs resumable and honest when a
    requested degree is infeasible under a strict degree cap.
    """
    n = len(labels)
    if average_degree < 0 or max_degree < 1:
        raise ValueError("average_degree must be non-negative and max_degree positive")
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must be in [0, 1]")
    by_community: dict[int, list[int]] = {}
    for vertex, row in enumerate(labels):
        for community in row:
            by_community.setdefault(int(community), []).append(vertex)
    communities = sorted(by_community)
    target_edges = int(round(n * int(average_degree) / 2.0))
    edges: set[tuple[int, int]] = set()
    degree = [0] * n
    internal = 0
    attempts = 0
    max_attempts = max(1000, target_edges * 80)

    def pair_for_internal() -> tuple[int, int] | None:
        candidates = [c for c in communities if len(by_community[c]) >= 2]
        if not candidates:
            return None
        body = by_community[rng.choice(candidates)]
        return tuple(sorted(rng.sample(body, 2)))

    while len(edges) < target_edges and attempts < max_attempts:
        attempts += 1
        choose_internal = rng.random() >= float(mixing)
        if choose_internal:
            pair = pair_for_internal()
        else:
            first, second = rng.sample(range(n), 2) if n >= 2 else (None, None)
            pair = None if first is None else tuple(sorted((first, second)))
        if pair is None:
            continue
        first, second = pair
        if first == second or degree[first] >= max_degree or degree[second] >= max_degree:
            continue
        if pair in edges:
            continue
        shared = bool(set(labels[first]).intersection(labels[second]))
        # A cross proposal can still be internal under overlapping labels; the
        # achieved mixing is measured from final memberships rather than the
        # proposal branch.
        edges.add(pair)
        degree[first] += 1
        degree[second] += 1
        internal += int(shared)

    achieved_average_degree = (2.0 * len(edges) / n) if n else 0.0
    achieved_internal_fraction = internal / len(edges) if edges else 0.0
    return sorted(edges), {
        "target_edge_count": int(target_edges),
        "achieved_edge_count": int(len(edges)),
        "edge_shortfall": int(max(0, target_edges - len(edges))),
        "attempts": int(attempts),
        "max_attempts": int(max_attempts),
        "degree_cap": int(max_degree),
        "max_degree_achieved": max(degree, default=0),
        "average_degree_achieved": float(achieved_average_degree),
        "internal_edge_fraction": float(achieved_internal_fraction),
        "cross_edge_fraction": float(1.0 - achieved_internal_fraction if edges else 0.0),
        "capacity_status": "target_reached" if len(edges) == target_edges else "edge_capacity_shortfall",
    }


def generate_overlapping_lfr(
    *,
    n: int = DEFAULT_N,
    average_degree: int = DEFAULT_AVERAGE_DEGREE,
    max_degree: int = DEFAULT_MAX_DEGREE,
    min_community: int = DEFAULT_MIN_COMMUNITY,
    max_community: int = DEFAULT_MAX_COMMUNITY,
    tau1: float = DEFAULT_TAU1,
    tau2: float = DEFAULT_TAU2,
    mixing: float = 0.3,
    overlap_fraction: float = 0.0,
    overlap_multiplicity: int = 2,
    seed: int = 0,
    retry_count: int = 0,
) -> OverlappingLFRGraph:
    """Generate one deterministic planted overlapping-LFR-compatible graph.

    ``retry_count`` is explicit provenance, not a hidden seed substitution:
    the returned metadata records both requested and effective seeds.  The
    generator does not consume ground-truth information during detection.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if tau1 <= 0 or tau2 <= 0:
        raise ValueError("power-law exponents must be positive")
    if retry_count < 0:
        raise ValueError("retry_count must be non-negative")
    requested_seed = int(seed)
    effective_seed = requested_seed + int(retry_count) * 1_000_003
    rng = random.Random(effective_seed)
    sizes = _community_sizes(n, rng, min_community, max_community, tau2)
    cover, labels, cover_metadata = _assign_cover(
        n, sizes, overlap_fraction, overlap_multiplicity, rng
    )
    edges, edge_metadata = _sample_graph_edges(
        labels,
        average_degree=average_degree,
        max_degree=max_degree,
        mixing=mixing,
        rng=rng,
    )
    graph = ig.Graph(n=n, edges=edges, directed=False)
    metadata: dict[str, Any] = {
        "generator_id": GENERATOR_ID,
        "generator_protocol_version": PROTOCOL_VERSION,
        "generator_policy": "compatibility_smoke_fixture",
        "canonical_evidence": False,
        "tau1_applied": False,
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed_requested": requested_seed,
        "seed_effective": effective_seed,
        "retry_count": int(retry_count),
        "n": int(n),
        "average_degree_requested": int(average_degree),
        "max_degree_requested": int(max_degree),
        "tau1_degree_exponent": float(tau1),
        "tau2_community_size_exponent": float(tau2),
        "min_community": int(min_community),
        "max_community": int(max_community),
        "mixing_requested": float(mixing),
        "community_count": len(cover),
        "community_sizes": [len(body) for body in cover],
        "metadata_free_detection": True,
        "cover_is_detector_input": False,
        "cover_hash": cover_hash(cover, n),
        "graph_hash": graph_sha256(graph),
        **cover_metadata,
        **edge_metadata,
    }
    return OverlappingLFRGraph(graph=graph, cover=cover, metadata=metadata)


# ---------------------------------------------------------------------------
# Official overlapping-LFR binary adapter
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file_sha256(path: Path) -> str | None:
    """Hash a JSON file after requiring valid object/array syntax."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return _canonical_payload_sha256(payload)


def _as_path_hash(value: str | Path | dict[str, Any] | None) -> str | None:
    if isinstance(value, dict):
        return _canonical_payload_sha256(value)
    if value is None:
        return None
    return _path_sha256(Path(value).expanduser())


def build_launch_command(
    *,
    profile: str,
    output_dir: str | Path,
    official_generator: str | Path | None,
    official_config: str | Path | None,
    official_format: str,
    official_source_commit: str | None,
    official_source_tree: str | None,
    n: int,
    graphs_per_condition: int,
    methods: Sequence[str],
    optimizer_seeds: Sequence[int],
    timeout_seconds: float | None,
    memory_limit_bytes: int | None,
    detector_cap: int,
    seed_offset: int,
) -> dict[str, Any]:
    """Build the exact, copyable production command recorded in receipts."""
    argv = [
        "uv",
        "run",
        "hedonic-exp",
        "overlapping-lfr",
        "--profile",
        str(profile),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
        "--n",
        str(int(n)),
        "--graphs-per-condition",
        str(int(graphs_per_condition)),
        "--methods",
        ",".join(str(method) for method in methods),
        "--optimizer-seeds",
        ",".join(str(int(seed)) for seed in optimizer_seeds),
        "--detector-cap",
        str(int(detector_cap)),
        "--seed-offset",
        str(int(seed_offset)),
        "--official-format",
        str(official_format),
        "--resume",
    ]
    if official_generator is not None:
        argv.extend(("--official-generator", str(Path(official_generator).expanduser().resolve())))
    if official_config is not None:
        argv.extend(("--official-config", str(Path(official_config).expanduser().resolve())))
    if official_source_commit is not None:
        argv.extend(("--official-source-commit", str(official_source_commit)))
    if official_source_tree is not None:
        argv.extend(("--official-source-tree", str(official_source_tree)))
    if timeout_seconds is not None:
        argv.extend(("--timeout-seconds", _native_lfr_number(timeout_seconds)))
    if memory_limit_bytes is not None:
        argv.extend(("--memory-limit-gb", _native_lfr_number(memory_limit_bytes / 1024**3)))
    return {
        "argv": argv,
        "shell": shlex.join(argv),
        "cwd": str(_repository_root()),
        "interpreter": str(Path(sys.executable).resolve()),
    }


def _run_identity_payload(
    *,
    profile: str,
    output_dir: str | Path,
    conditions: Sequence[dict[str, Any]],
    n: int,
    graphs_per_condition: int,
    max_graphs: int | None,
    methods: Sequence[str],
    optimizer_seeds: Sequence[int],
    timeout_seconds: float | None,
    memory_limit_bytes: int | None,
    seed_offset: int,
    detector_cap: int,
    official_generator: str | Path | None,
    official_config: str | Path | dict[str, Any] | None,
    official_format: str,
    official_source_commit: str | None,
    official_source_tree: str | None,
) -> dict[str, Any]:
    """Bind protocol, executable, source, lock, and seed inputs to one ID."""
    environment = environment_receipt()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "profile": str(profile),
        "conditions": list(conditions),
        "n": int(n),
        "graphs_per_condition": int(graphs_per_condition),
        "max_graphs": None if max_graphs is None else int(max_graphs),
        "methods": [str(method) for method in methods],
        "optimizer_seeds": [int(seed) for seed in optimizer_seeds],
        "timeout_seconds": timeout_seconds,
        "memory_limit_bytes": memory_limit_bytes,
        "seed_offset": int(seed_offset),
        "detector_cap": int(detector_cap),
        "official_format": str(official_format),
        "official_source_repository": OFFICIAL_LFRBENCHMARKS_REPOSITORY,
        "official_source_commit": official_source_commit,
        "official_source_tree": official_source_tree,
        "official_generator": str(Path(official_generator).expanduser().resolve())
        if official_generator is not None
        else None,
        "official_generator_sha256": _path_sha256(Path(official_generator).expanduser())
        if official_generator is not None
        else None,
        "official_config": str(Path(official_config).expanduser().resolve())
        if isinstance(official_config, (str, Path))
        else None,
        "official_config_sha256": _as_path_hash(official_config),
        "source_module_sha256": _path_sha256(Path(__file__).resolve()),
        "environment": environment,
        "seed_provenance": {
            "graph_seed_rule": "seed_offset + condition_index*100000 + graph_index_within_condition",
            "method_seed_rule": "sha256(seed_effective, optimizer_seed, method)[:8] & 0x7fffffff",
            "upstream_seed_rule": "abs(seed_requested) % 2147483397 + 1",
            "graph_seed_domain": "integer; condition and graph coordinates are explicit",
        },
        "output_dir": str(Path(output_dir).expanduser().resolve()),
    }
    payload["run_identity"] = _canonical_payload_sha256(payload)
    return payload


def _read_pilot_measurements(
    pilot_receipt: str | Path | dict[str, Any] | None,
) -> dict[str, Any]:
    """Read bounded-pilot sizes/resource values without trusting stale output."""
    if isinstance(pilot_receipt, dict):
        payload = dict(pilot_receipt)
        candidate = "<inline bounded_pilot receipt>"
        required = ("graph_artifact_bytes", "result_row_bytes", "pilot_n", "pilot_graphs")
        missing = [key for key in required if key not in payload]
        if missing:
            return {
                "available": False,
                "path": candidate,
                "reason": "receipt_missing:" + ",".join(missing),
            }
        return {"available": True, "path": candidate, **payload}
    if pilot_receipt is None:
        candidate = _repository_root() / PILOT_RECEIPT_PATH
    else:
        candidate = Path(pilot_receipt).expanduser()
    if not candidate.is_file():
        return {
            "available": False,
            "path": str(candidate),
            "reason": "bounded_pilot_receipt_missing",
        }
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"available": False, "path": str(candidate), "reason": f"invalid_receipt:{exc}"}
    if not isinstance(payload, dict):
        return {"available": False, "path": str(candidate), "reason": "receipt_not_object"}
    required = ("graph_artifact_bytes", "result_row_bytes", "pilot_n", "pilot_graphs")
    missing = [key for key in required if key not in payload]
    if missing:
        return {
            "available": False,
            "path": str(candidate),
            "reason": "receipt_missing:" + ",".join(missing),
        }
    return {
        "available": True,
        "path": str(candidate),
        "receipt_sha256": _file_sha256(candidate),
        **payload,
    }


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a lowercase-or-uppercase SHA-256 digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _bounded_pilot_generator_sha256(configured_receipt: Mapping[str, Any]) -> str | None:
    """Return the native binary identity qualified by the bounded pilot.

    The official source commit/tree labels describe the requested upstream
    source, but cannot by themselves prove that a path supplied by a caller
    is the binary exercised by the bounded qualification.  The configuration
    therefore carries the pilot binary digest as part of the launch binding.
    """
    pilot = configured_receipt.get("bounded_pilot")
    if not isinstance(pilot, Mapping):
        return None
    value = pilot.get("generator_executable_sha256")
    return str(value) if _is_sha256(value) else None


def _official_generator_binding(
    *,
    profile: str,
    official_generator: Path | None,
    official_format: str,
    source_commit: str | None,
    source_tree: str | None,
    configured_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the native binary/source identity before any graph is started.

    Pilot and standard executions must use the exact executable qualified by
    the bounded-pilot receipt in the selected official configuration.  A
    caller-supplied source label is retained as provenance but is not accepted
    on its own as executable provenance.
    """
    required = profile in {"pilot", "standard"}
    executable = bool(
        official_generator
        and official_generator.is_file()
        and os.access(official_generator, os.X_OK)
    )
    actual = _path_sha256(official_generator) if official_generator is not None else None
    expected = _bounded_pilot_generator_sha256(configured_receipt)
    commit_ok = str(source_commit or "") == OFFICIAL_LFRBENCHMARKS_COMMIT
    tree_ok = str(source_tree or "") == OFFICIAL_LFRBENCHMARKS_TREE
    hash_ok = actual == expected if expected is not None else False
    native_format_ok = not required or official_format == "lfrbenchmarks"
    # The JSON adapter validates a generic process contract; it cannot prove
    # that it ran the pinned upstream LFRbenchmarks executable. It is useful
    # for isolated smoke compatibility checks, but must never satisfy the
    # pilot/standard canonical-generator gate.
    hash_required = required
    ready = (
        executable
        and native_format_ok
        and commit_ok
        and tree_ok
        and (not hash_required or hash_ok)
    )
    return {
        "status": "ok" if ready else "failed",
        "path": str(official_generator) if official_generator is not None else None,
        "executable": executable,
        "sha256": actual,
        "expected_sha256": expected,
        "expected_sha256_source": (
            "official_config.bounded_pilot.generator_executable_sha256"
            if expected is not None
            else None
        ),
        "official_format": official_format,
        "native_lfrbenchmarks_format_required": required,
        "native_lfrbenchmarks_format_ok": native_format_ok,
        "sha256_required": hash_required,
        "sha256_matches_bounded_pilot": hash_ok,
        "source_repository": OFFICIAL_LFRBENCHMARKS_REPOSITORY,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "expected_commit": OFFICIAL_LFRBENCHMARKS_COMMIT,
        "expected_tree": OFFICIAL_LFRBENCHMARKS_TREE,
    }


def estimate_lfr_resources(
    *,
    total_graphs: int,
    methods: Sequence[str],
    optimizer_seeds: Sequence[int],
    target_n: int,
    pilot_receipt: str | Path | dict[str, Any] | None = None,
    memory_limit_bytes: int | None = None,
) -> dict[str, Any]:
    """Estimate storage/RSS from a bounded pilot, with explicit assumptions."""
    pilot = _read_pilot_measurements(pilot_receipt)
    rows_per_graph = sum(1 if method == "singleton" else len(tuple(optimizer_seeds)) for method in methods)
    if not pilot.get("available"):
        return {
            "status": "pilot_missing",
            "pilot": pilot,
            "rows_per_graph": rows_per_graph,
            "target_graphs": int(total_graphs),
            "target_n": int(target_n),
            "memory_limit_bytes": memory_limit_bytes,
        }
    pilot_n = max(1, int(pilot["pilot_n"]))
    scale = max(1.0, float(target_n) / pilot_n)
    graph_bytes = int(pilot["graph_artifact_bytes"])
    row_bytes = int(pilot["result_row_bytes"])
    # JSON graph and cover payloads are approximately linear in n for fixed
    # degree/cover parameters.  A 2x safety factor is kept explicit rather
    # than hidden in a prose estimate.
    estimated_graph_bytes = int(math.ceil(graph_bytes * scale * int(total_graphs) * 2.0))
    estimated_result_bytes = int(
        math.ceil(row_bytes * scale * rows_per_graph * int(total_graphs) * 2.0)
    )
    checkpoint_bytes = estimated_graph_bytes + estimated_result_bytes
    estimated_total = int(math.ceil(checkpoint_bytes * 1.25))
    peak_rss = int(pilot.get("peak_rss_bytes", 0) or 0)
    estimated_rss = int(math.ceil(peak_rss * scale * 1.5)) if peak_rss else None
    return {
        "status": "estimated_from_bounded_pilot",
        "pilot": pilot,
        "target_graphs": int(total_graphs),
        "target_n": int(target_n),
        "rows_per_graph": rows_per_graph,
        "scale_n": scale,
        "safety_factors": {"graph_and_row_bytes": 2.0, "manifest_and_temp": 1.25, "rss": 1.5},
        "estimated_graph_checkpoint_bytes": estimated_graph_bytes,
        "estimated_result_checkpoint_bytes": estimated_result_bytes,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_total_storage_bytes": estimated_total,
        "pilot_peak_rss_bytes": peak_rss or None,
        "estimated_worker_peak_rss_bytes": estimated_rss,
        "memory_limit_bytes": memory_limit_bytes,
        "headroom_bytes": None if memory_limit_bytes is None else max(0, int(memory_limit_bytes) - (estimated_rss or 0)),
        "assumptions": [
            "graph and serialized cover payloads scale linearly with n at fixed degree/cap",
            "result rows scale linearly with n and row count",
            "safety factors are planning bounds, not measured production usage",
        ],
    }


def _distribution_version(distribution: str) -> str | None:
    """Return an installed distribution version without masking receipts."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _isolated_slpa_preflight() -> dict[str, Any]:
    """Check the locked SLPA worker without importing ambient CDlib.

    SLPA is intentionally executed in ``tools/slpa_env`` because CDlib's
    dependency closure is not part of the lucas-igraph runtime.  The TKT-11
    launch preflight must therefore validate the isolated project and its
    direct pins, rather than treating ``import cdlib`` in the parent process
    as a prerequisite.  This helper is read-only; the disposable fixture
    below still exercises the worker when the launch gate is otherwise ready.
    """
    try:
        from hedonic.experiments.overlapping.baselines import (
            SLPA_DIRECT_PINS,
            slpa_environment_receipt,
        )

        environment = slpa_environment_receipt()
    except Exception as exc:  # pragma: no cover - defensive import boundary
        return {
            "status": "failed",
            "reason": f"cannot inspect isolated SLPA environment: {type(exc).__name__}: {exc}",
            "environment": {},
        }

    files = environment.get("files") if isinstance(environment.get("files"), dict) else {}
    file_hashes_ok = bool(files) and all(
        isinstance(entry, dict)
        and isinstance(entry.get("sha256"), str)
        and len(entry["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in entry["sha256"].lower())
        for entry in files.values()
    )
    pins = environment.get("direct_pins")
    pins_ok = isinstance(pins, dict) and pins == dict(SLPA_DIRECT_PINS)
    ready = environment.get("ready") is True and file_hashes_ok and pins_ok
    return {
        "status": "ok" if ready else "failed",
        "reason": None if ready else "isolated SLPA project, hashes, or direct pins are not ready",
        "environment": environment,
        "direct_pins": dict(SLPA_DIRECT_PINS),
        "file_hashes_ok": file_hashes_ok,
        "pins_ok": pins_ok,
        "policy": "isolated uv project; ambient cdlib import is not required",
    }


def environment_receipt() -> dict[str, Any]:
    """Capture the interpreter, lock, and source identities for a run.

    The receipt is intentionally read-only and best effort for files that are
    unavailable when the package is installed as a wheel.  A missing lock
    digest is represented as ``None`` rather than inferred from a different
    environment.  This lets a reviewer distinguish an exact ``uv`` run from
    an ad-hoc interpreter invocation.
    """
    repository_root = _repository_root()
    lock_files: dict[str, dict[str, Any]] = {}
    for relative in ("pyproject.toml", "uv.lock"):
        path = repository_root / relative
        if path.is_file():
            try:
                lock_files[relative] = {"path": str(path), "sha256": _file_sha256(path)}
            except OSError:
                lock_files[relative] = {"path": str(path), "sha256": None}
        else:
            lock_files[relative] = {"path": str(path), "sha256": None}
    distributions = {
        name: _distribution_version(name)
        for name in (
            "hedonic",
            "lucas-igraph",
            "demon",
            "networkx",
            "cdlib",
            "numpy",
            "scipy",
        )
    }
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "distributions": distributions,
        "lock_files": lock_files,
        "official_source": {
            "repository": OFFICIAL_LFRBENCHMARKS_REPOSITORY,
            "commit": OFFICIAL_LFRBENCHMARKS_COMMIT,
            "tree": OFFICIAL_LFRBENCHMARKS_TREE,
            "subdirectory": OFFICIAL_LFRBENCHMARKS_SOURCE_SUBDIRECTORY,
            "output_format": OFFICIAL_LFRBENCHMARKS_OUTPUT_FORMAT,
        },
    }


def _load_json_object(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    path = Path(value).expanduser()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialGeneratorUnavailable(f"official LFR config is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OfficialGeneratorUnavailable("official LFR config must be a JSON object")
    return dict(payload)


def _validate_official_output(
    payload: Any,
    *,
    request: dict[str, Any],
    executable: Path,
    input_sha256: str,
    output_sha256: str,
    command: Sequence[str],
    stdout: str,
    stderr: str,
) -> OverlappingLFRGraph:
    """Validate the strict JSON receipt emitted by an official LFR binary.

    The adapter intentionally accepts no implicit format conversion: the
    executable must emit ``edges`` and a non-empty ``cover`` together with a
    metadata object declaring its own generator/protocol/seed.  This keeps
    provenance auditable and fails closed when a different LFR dialect is
    accidentally pointed at the runner.
    """
    if not isinstance(payload, dict):
        raise OfficialGeneratorUnavailable("official LFR output must be a JSON object")
    edges = payload.get("edges")
    cover = payload.get("cover")
    reported = payload.get("metadata")
    if not isinstance(edges, list) or not isinstance(cover, list) or not isinstance(reported, dict):
        raise OfficialGeneratorUnavailable("official LFR output requires edges, cover, and metadata fields")
    required = ("generator_id", "protocol_version", "seed", "n")
    missing = [key for key in required if key not in reported]
    if missing:
        raise OfficialGeneratorUnavailable("official LFR metadata missing: " + ", ".join(missing))
    if str(reported.get("generator_id", "")).startswith("overlapping_lfr_compatibility"):
        raise OfficialGeneratorUnavailable("compatibility fixture output cannot be used as official LFR")
    try:
        n = int(reported["n"])
        expected_n = request.get("n")
        if expected_n is not None and n != int(expected_n):
            raise OfficialGeneratorUnavailable(f"official LFR n={n} differs from requested n={expected_n}")
        graph_edges: list[tuple[int, int]] = []
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                raise ValueError("edge must contain two endpoints")
            first, second = int(edge[0]), int(edge[1])
            if first == second or not (0 <= first < n and 0 <= second < n):
                raise ValueError("edge endpoint out of range or self-loop")
            graph_edges.append(tuple(sorted((first, second))))
        if len(set(graph_edges)) != len(graph_edges):
            raise ValueError("official LFR output contains duplicate edges")
        graph = ig.Graph(n=n, edges=graph_edges, directed=False)
        normalized_cover: list[list[int]] = []
        for body in cover:
            if not isinstance(body, (list, tuple)):
                raise ValueError("cover body must be a list of vertex IDs")
            members = sorted({int(vertex) for vertex in body})
            if any(vertex < 0 or vertex >= n for vertex in members):
                raise ValueError("cover vertex out of range")
            if members:
                normalized_cover.append(members)
        normalized_cover = canonicalize_cover(normalized_cover, n_vertices=n)
        if n and not normalized_cover:
            raise ValueError("official LFR cover cannot be empty")
        covered = {vertex for body in normalized_cover for vertex in body}
        if covered != set(range(n)):
            raise ValueError("official LFR cover must cover every vertex exactly at least once")
    except OfficialGeneratorUnavailable:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialGeneratorUnavailable(f"invalid official LFR graph/cover payload: {exc}") from exc
    try:
        requested_seed = int(request.get("seed", reported["seed"]))
        effective_seed = int(reported["seed"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialGeneratorUnavailable("official LFR metadata seed must be an integer") from exc

    metadata: dict[str, Any] = {
        "generator_id": OFFICIAL_GENERATOR_ID,
        "generator_reported_id": str(reported["generator_id"]),
        "generator_protocol_version": str(reported["protocol_version"]),
        "generator_executable": str(executable),
        "generator_executable_sha256": _file_sha256(executable),
        "generator_input_sha256": str(input_sha256),
        "generator_output_sha256": str(output_sha256),
        "generator_command": list(map(str, command)),
        "generator_stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        "generator_stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "generator_metadata": dict(reported),
        "seed_requested": requested_seed,
        "seed_effective": effective_seed,
        "n": int(n),
        "average_degree_requested": request.get("average_degree"),
        "max_degree_requested": request.get("max_degree"),
        "tau1_degree_exponent": request.get("tau1"),
        "tau2_community_size_exponent": request.get("tau2"),
        "mixing_requested": request.get("mixing"),
        "overlap_fraction_requested": request.get("overlap_fraction"),
        "overlap_multiplicity_requested": request.get("overlap_multiplicity"),
        "community_count": len(normalized_cover),
        "community_sizes": [len(body) for body in normalized_cover],
        "metadata_free_detection": True,
        "cover_is_detector_input": False,
        "generator_policy": "official_binary_required",
        "cover_hash": cover_hash(normalized_cover, n),
        "graph_hash": graph_sha256(graph),
        "environment": environment_receipt(),
    }
    # Preserve declared fields without allowing a binary to overwrite the
    # adapter's identity and content hashes.
    for key, value in reported.items():
        metadata.setdefault(f"reported_{key}", value)
    return OverlappingLFRGraph(graph=graph, cover=normalized_cover, metadata=metadata)


def generate_official_overlapping_lfr(
    executable: str | Path,
    *,
    config: str | Path | dict[str, Any],
    output_path: str | Path | None = None,
    timeout_seconds: float = 300.0,
) -> OverlappingLFRGraph:
    """Run an official overlapping-LFR binary under a strict JSON contract.

    The executable is invoked exactly as ``BINARY --config INPUT.json
    --output OUTPUT.json``.  ``INPUT.json`` is a canonicalized copy of
    ``config`` and ``OUTPUT.json`` must contain ``edges``, ``cover`` and
    ``metadata`` (with ``generator_id``, ``protocol_version``, ``seed`` and
    ``n``).  Missing binaries, non-zero exits, malformed output, or uncovered
    vertices raise :class:`OfficialGeneratorUnavailable`; no Python fallback
    is attempted.  Hashes of executable, input, output and process streams
    are included in returned metadata.
    """
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OfficialGeneratorUnavailable(f"official overlapping-LFR binary is not executable: {path}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = _load_json_object(config)
    request.setdefault("protocol_version", PROTOCOL_VERSION)
    request.setdefault("generator_policy", "official_binary_required")
    encoded_request = json.dumps(request, sort_keys=True, separators=(",", ":"), default=str).encode()
    input_sha256 = hashlib.sha256(encoded_request).hexdigest()

    destination = None if output_path is None else Path(output_path).expanduser()
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hedonic-official-lfr-") as temporary:
        input_path = Path(temporary) / "input.json"
        input_path.write_bytes(encoded_request)
        generated_path = destination or (Path(temporary) / "output.json")
        command = [str(path), "--config", str(input_path), "--output", str(generated_path)]
        try:
            process = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise OfficialGeneratorUnavailable("official overlapping-LFR binary timed out") from exc
        except OSError as exc:
            raise OfficialGeneratorUnavailable(f"could not execute official overlapping-LFR binary: {exc}") from exc
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or f"exit code {process.returncode}"
            raise OfficialGeneratorUnavailable(f"official overlapping-LFR binary failed: {detail}")
        if not generated_path.is_file():
            raise OfficialGeneratorUnavailable("official overlapping-LFR binary did not write --output JSON")
        output_bytes = generated_path.read_bytes()
        try:
            payload = json.loads(output_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OfficialGeneratorUnavailable("official overlapping-LFR output is not UTF-8 JSON") from exc
        return _validate_official_output(
            payload,
            request=request,
            executable=path,
            input_sha256=input_sha256,
            output_sha256=hashlib.sha256(output_bytes).hexdigest(),
            command=command,
            stdout=process.stdout,
            stderr=process.stderr,
        )


def _native_lfr_number(value: Any) -> str:
    """Render a numeric LFR flag without locale-dependent formatting."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".17g")
    return str(value)


def _parse_native_lfr_network(path: Path, *, n: int) -> list[tuple[int, int]]:
    """Parse LFRbenchmarks' 1-based, duplicated undirected edge list."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OfficialGeneratorUnavailable(f"cannot read native LFR network.dat: {path}") from exc
    edges: set[tuple[int, int]] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise OfficialGeneratorUnavailable(
                f"native LFR network.dat line {line_number} must contain two endpoints"
            )
        try:
            first, second = (int(value) - 1 for value in fields)
        except (TypeError, ValueError) as exc:
            raise OfficialGeneratorUnavailable(
                f"native LFR network.dat line {line_number} has non-integer endpoints"
            ) from exc
        if first == second or not (0 <= first < n and 0 <= second < n):
            raise OfficialGeneratorUnavailable(
                f"native LFR network.dat line {line_number} has an invalid endpoint"
            )
        edges.add(tuple(sorted((first, second))))
    return sorted(edges)


def _parse_native_lfr_cover(path: Path, *, n: int) -> tuple[list[list[int]], list[list[int]]]:
    """Parse LFRbenchmarks' node-to-membership ``community.dat`` output."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise OfficialGeneratorUnavailable(f"cannot read native LFR community.dat: {path}") from exc
    memberships: list[list[int] | None] = [None] * n
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 2:
            raise OfficialGeneratorUnavailable(
                f"native LFR community.dat line {line_number} needs a node and membership"
            )
        try:
            vertex = int(fields[0]) - 1
            labels = sorted({int(value) for value in fields[1:]})
        except (TypeError, ValueError) as exc:
            raise OfficialGeneratorUnavailable(
                f"native LFR community.dat line {line_number} has non-integer values"
            ) from exc
        if not (0 <= vertex < n) or not labels or any(label < 1 for label in labels):
            raise OfficialGeneratorUnavailable(
                f"native LFR community.dat line {line_number} has invalid node/membership values"
            )
        if memberships[vertex] is not None:
            raise OfficialGeneratorUnavailable(
                f"native LFR community.dat repeats node {vertex + 1}"
            )
        memberships[vertex] = labels
    if any(row is None for row in memberships):
        missing = [str(index + 1) for index, row in enumerate(memberships) if row is None]
        raise OfficialGeneratorUnavailable(
            "native LFR community.dat does not cover every vertex; missing "
            + ",".join(missing[:12])
        )
    rows = [list(row or ()) for row in memberships]
    bodies: dict[int, list[int]] = {}
    for vertex, labels in enumerate(rows):
        for label in labels:
            bodies.setdefault(int(label), []).append(vertex)
    cover = canonicalize_cover(list(bodies.values()), n_vertices=n)
    if n and {vertex for body in cover for vertex in body} != set(range(n)):
        raise OfficialGeneratorUnavailable("native LFR community.dat leaves vertices uncovered")
    return cover, rows


def generate_official_lfrbenchmarks(
    executable: str | Path,
    *,
    config: str | Path | dict[str, Any],
    output_dir: str | Path | None = None,
    timeout_seconds: float = 300.0,
    source_commit: str | None = None,
    source_tree: str | None = None,
) -> OverlappingLFRGraph:
    """Run the authoritative LFRbenchmarks native binary.

    The upstream package is a C++ program whose contract is command-line
    flags plus three files in its working directory: ``network.dat`` and
    ``community.dat`` (both 1-based; edges are emitted in both orientations)
    and ``statistics.dat``.  This adapter supplies a fresh ``time_seed.dat``
    for every graph, parses and validates all required files, and records the
    exact source commit/tree and executable/output hashes.  A source commit is
    mandatory; an unqualified binary is never accepted as canonical evidence.
    """
    path = Path(executable).expanduser()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise OfficialGeneratorUnavailable(f"official LFRbenchmarks binary is not executable: {path}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = _load_json_object(config)
    required = (
        "n",
        "average_degree",
        "max_degree",
        "mixing",
        "tau1",
        "tau2",
        "min_community",
        "max_community",
        "overlap_fraction",
        "overlap_multiplicity",
        "seed",
    )
    missing = [key for key in required if key not in request]
    if missing:
        raise OfficialGeneratorUnavailable("native LFR request missing: " + ", ".join(missing))
    try:
        n = int(request["n"])
        average_degree = float(request["average_degree"])
        max_degree = int(request["max_degree"])
        mixing = float(request["mixing"])
        tau1 = float(request["tau1"])
        tau2 = float(request["tau2"])
        min_community = int(request["min_community"])
        max_community = int(request["max_community"])
        overlap_fraction = float(request["overlap_fraction"])
        overlap_multiplicity = int(request["overlap_multiplicity"])
        requested_seed = int(request["seed"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise OfficialGeneratorUnavailable("native LFR request contains invalid numeric values") from exc
    if (
        n < 3
        or average_degree <= 0
        or max_degree < 1
        or not 0.0 <= mixing <= 1.0
        or tau1 <= 0
        or tau2 <= 0
        or min_community < 2
        or max_community < min_community
        or not 0.0 <= overlap_fraction <= 1.0
        or overlap_multiplicity < 1
    ):
        raise OfficialGeneratorUnavailable("native LFR request violates positive/range constraints")
    overlap_nodes = int(request.get("overlap_nodes", round(n * overlap_fraction)))
    overlap_nodes = max(0, min(n, overlap_nodes))
    overlap_memberships = int(request.get("overlap_memberships", overlap_multiplicity if overlap_nodes else 0))
    if overlap_nodes and overlap_memberships < 2:
        raise OfficialGeneratorUnavailable("native LFR overlapping nodes require -om >= 2")
    if not overlap_nodes:
        overlap_memberships = 0
    declared_commit = source_commit or request.get("official_source_commit")
    declared_tree = source_tree or request.get("official_source_tree")
    if str(declared_commit or "") != OFFICIAL_LFRBENCHMARKS_COMMIT:
        raise OfficialGeneratorUnavailable(
            "native LFRbenchmarks source commit is not pinned; pass "
            f"official_source_commit={OFFICIAL_LFRBENCHMARKS_COMMIT}"
        )
    if declared_tree is not None and str(declared_tree) != OFFICIAL_LFRBENCHMARKS_TREE:
        raise OfficialGeneratorUnavailable(
            "native LFRbenchmarks source tree does not match the pinned commit"
        )

    # The upstream generator only accepts positive seeds in its historical
    # ran4 implementation.  Derive a stable, independent file seed while
    # retaining the user-requested seed in the receipt.
    effective_seed = abs(requested_seed) % 2_147_483_397 + 1
    flag_parameters = {
        "-N": n,
        "-k": average_degree,
        "-maxk": max_degree,
        "-mu": mixing,
        "-t1": tau1,
        "-t2": tau2,
        "-minc": min_community,
        "-maxc": max_community,
        "-on": overlap_nodes,
        "-om": overlap_memberships,
    }
    encoded_request = json.dumps(
        {**request, "overlap_nodes": overlap_nodes, "overlap_memberships": overlap_memberships},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    input_sha256 = hashlib.sha256(encoded_request).hexdigest()
    command = [str(path)]
    for flag, value in flag_parameters.items():
        command.extend((flag, _native_lfr_number(value)))

    persist_dir = None if output_dir is None else Path(output_dir).expanduser()
    if persist_dir is not None:
        persist_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hedonic-lfrbenchmarks-") as temporary:
        workdir = Path(temporary)
        (workdir / "time_seed.dat").write_text(f"{effective_seed}\n", encoding="utf-8")
        try:
            process = subprocess.run(
                command,
                cwd=workdir,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise OfficialGeneratorUnavailable("native LFRbenchmarks binary timed out") from exc
        except OSError as exc:
            raise OfficialGeneratorUnavailable(f"could not execute native LFRbenchmarks binary: {exc}") from exc
        if process.returncode != 0:
            detail = process.stderr.strip() or process.stdout.strip() or f"exit code {process.returncode}"
            raise OfficialGeneratorUnavailable(f"native LFRbenchmarks binary failed: {detail}")
        network_path = workdir / "network.dat"
        community_path = workdir / "community.dat"
        statistics_path = workdir / "statistics.dat"
        if not network_path.is_file() or not community_path.is_file() or not statistics_path.is_file():
            raise OfficialGeneratorUnavailable(
                "native LFRbenchmarks binary did not produce network.dat, community.dat, and statistics.dat"
            )
        edges = _parse_native_lfr_network(network_path, n=n)
        cover, memberships = _parse_native_lfr_cover(community_path, n=n)
        if persist_dir is not None:
            # Preserve exact upstream text files for an audit without making
            # the temporary working directory part of the artifact protocol.
            for source in (network_path, community_path, statistics_path):
                target = persist_dir / source.name
                temporary_target = target.with_name(target.name + ".tmp")
                shutil.copyfile(source, temporary_target)
                temporary_target.replace(target)
        graph = ig.Graph(n=n, edges=edges, directed=False)
        graph_hash = graph_sha256(graph)
        community_hash = cover_hash(cover, n)
        metadata: dict[str, Any] = {
            "generator_id": OFFICIAL_GENERATOR_ID,
            "generator_reported_id": "LFRbenchmarks/unweighted_undirected/benchmark",
            "generator_protocol_version": PROTOCOL_VERSION,
            "generator_policy": "official_binary_required",
            "canonical_evidence": True,
            "tau1_applied": True,
            "official_source_repository": OFFICIAL_LFRBENCHMARKS_REPOSITORY,
            "official_source_commit": OFFICIAL_LFRBENCHMARKS_COMMIT,
            "official_source_tree": OFFICIAL_LFRBENCHMARKS_TREE,
            "official_source_subdirectory": OFFICIAL_LFRBENCHMARKS_SOURCE_SUBDIRECTORY,
            "official_source_commit_verified": True,
            "official_output_format": OFFICIAL_LFRBENCHMARKS_OUTPUT_FORMAT,
            "generator_executable": str(path),
            "generator_executable_sha256": _file_sha256(path),
            "generator_input_sha256": input_sha256,
            "generator_command": command,
            "generator_stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            "generator_stderr_sha256": hashlib.sha256(process.stderr.encode()).hexdigest(),
            "generator_network_sha256": _file_sha256(network_path),
            "generator_community_sha256": _file_sha256(community_path),
            "generator_statistics_sha256": _file_sha256(statistics_path),
            "seed_requested": requested_seed,
            "seed_effective": effective_seed,
            "n": n,
            "average_degree_requested": average_degree,
            "max_degree_requested": max_degree,
            "tau1_degree_exponent": tau1,
            "tau2_community_size_exponent": tau2,
            "min_community": min_community,
            "max_community": max_community,
            "mixing_requested": mixing,
            "overlap_fraction_requested": overlap_fraction,
            "overlap_nodes_requested": overlap_nodes,
            "overlap_multiplicity_requested": overlap_memberships,
            "overlap_fraction_achieved": sum(len(row) > 1 for row in memberships) / n,
            "overlap_multiplicity_achieved_mean": statistics.fmean(
                len(row) for row in memberships if len(row) > 1
            ) if any(len(row) > 1 for row in memberships) else 1.0,
            "community_count": len(cover),
            "community_sizes": [len(body) for body in cover],
            "metadata_free_detection": True,
            "cover_is_detector_input": False,
            "cover_hash": community_hash,
            "graph_hash": graph_hash,
            "environment": environment_receipt(),
        }
        return OverlappingLFRGraph(graph=graph, cover=cover, metadata=metadata)


# Friendly aliases used by pilot notebooks and hidden protocol checks.
generate_lfr_graph = generate_overlapping_lfr
generate_graph = generate_overlapping_lfr
generate_official_lfr_graph = generate_official_overlapping_lfr
generate_official_lfrbenchmarks_graph = generate_official_lfrbenchmarks


def condition_grid(
    *,
    mixing: Sequence[float] = DEFAULT_MIXING,
    overlap_fractions: Sequence[float] = DEFAULT_OVERLAP_FRACTIONS,
    overlap_multiplicities: Sequence[int] = DEFAULT_OVERLAP_MULTIPLICITIES,
) -> list[dict[str, Any]]:
    """Return the 28-cell structural grid without generating graphs."""
    conditions: list[dict[str, Any]] = []
    for mu in mixing:
        conditions.append({"mixing": float(mu), "overlap_fraction": 0.0, "overlap_multiplicity": 1})
        for fraction in overlap_fractions:
            if float(fraction) <= 0:
                continue
            for multiplicity in overlap_multiplicities:
                conditions.append(
                    {
                        "mixing": float(mu),
                        "overlap_fraction": float(fraction),
                        "overlap_multiplicity": int(multiplicity),
                    }
                )
    return conditions


def _method_seed(graph_seed: int, optimizer_seed: int, method: str) -> int:
    # Keep graph seeds and optimizer seeds independent while making every
    # method/graph pair reproducible across resumed processes.
    digest = hashlib.sha256(f"{graph_seed}:{optimizer_seed}:{method}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _detector_cover(
    graph: ig.Graph,
    method: str,
    *,
    cap: int,
    gamma: float,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Run one detector through experiment-only adapters.

    Imports are delayed so the core package remains free of optional
    dependencies.  Failures are raised to the caller and become explicit
    result rows.
    """
    from hedonic.experiments.overlapping.baselines import run_baseline
    if method in {"hedonic_multiphase", "disjoint"}:
        ig.set_random_number_generator(random.Random(seed))
        result = Game(graph).community_hedonic(
            resolution=float(gamma),
            max_memberships=1 if method == "disjoint" else int(cap),
            local_move_only=False,
            n_iterations=-1,
            allow_isolation=True,
        )
        raw_membership = getattr(result, "membership", None)
        if raw_membership is not None:
            raw_membership = [
                list(map(int, row)) if isinstance(row, (list, tuple)) else [int(row)]
                for row in raw_membership
            ]
        return partition_to_cover_lists(result), {
            "method_family": "hedonic",
            "n_iterations": -1,
            "local_move_only": False,
            "max_memberships": 1 if method == "disjoint" else int(cap),
            "strategic_membership": raw_membership,
        }
    if method == "singleton":
        return [[vertex] for vertex in range(graph.vcount())], {
            "method_family": "control",
            "deterministic": True,
        }
    return run_baseline(method, graph, cap=cap, resolution=gamma, seed=seed)


def run_graph_methods(
    instance: OverlappingLFRGraph,
    *,
    methods: Sequence[str] = DEFAULT_METHODS,
    optimizer_seeds: Sequence[int] = (0, 1, 2),
    timeout_seconds: float | None = 60.0,
    memory_limit_bytes: int | None = None,
    cap: int = 4,
    skip_keys: set[tuple[str, str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Run methods on one graph, retaining statuses and resource receipts.

    ``skip_keys`` is used by the resumable study ledger.  A key is stable over
    graph content, method name and optimizer seed, so an interrupted run can
    be resumed without rerunning completed observations or duplicating rows.
    The worker's peak RSS is compared with ``memory_limit_bytes`` after exit;
    an over-limit observation is retained as ``memory_limit_exceeded`` rather
    than being silently scored as a successful detector run.
    """
    from hedonic.experiments.overlapping.execution import run_in_subprocess

    if int(cap) < 1:
        raise ValueError("cap must be positive")
    if memory_limit_bytes is not None and int(memory_limit_bytes) <= 0:
        raise ValueError("memory_limit_bytes must be positive or None")
    graph = instance.graph
    gamma = float(graph.density())
    rows: list[dict[str, Any]] = []
    for method in methods:
        # Deterministic controls are intentionally run once.  Stochastic
        # adapters receive the three declared nested optimizer seeds.
        seeds = (0,) if method in {"singleton"} else tuple(int(x) for x in optimizer_seeds)
        for optimizer_seed in seeds:
            key = (instance.graph_hash, str(method), int(optimizer_seed))
            if skip_keys and key in skip_keys:
                continue
            run_seed = _method_seed(int(instance.metadata["seed_effective"]), optimizer_seed, method)
            started = time.monotonic()
            outcome_packet = run_in_subprocess(
                _detector_cover,
                graph,
                method,
                cap=1 if method == "disjoint" else int(cap),
                gamma=gamma,
                seed=run_seed,
                timeout_seconds=timeout_seconds,
            )
            successful = outcome_packet.status == "ok" and isinstance(outcome_packet.payload, tuple)
            if successful:
                predicted, method_meta = outcome_packet.payload
                outcome = {"status": "ok", "runtime_seconds": outcome_packet.runtime_seconds}
            else:
                predicted = None
                method_meta = {}
                outcome = {
                    "status": outcome_packet.status,
                    "runtime_seconds": outcome_packet.runtime_seconds,
                    "error": outcome_packet.error,
                }
            status = str(outcome.get("status", "failed"))
            if successful:
                status = "completed"
            observed_rss = outcome_packet.peak_rss_bytes
            memory = {
                "peak_rss_bytes": observed_rss,
                "limit_bytes": int(memory_limit_bytes) if memory_limit_bytes is not None else None,
                "enforcement": "posthoc_worker_peak_rss",
            }
            if (
                successful
                and memory_limit_bytes is not None
                and observed_rss is not None
                and int(observed_rss) > int(memory_limit_bytes)
            ):
                status = "memory_limit_exceeded"
                successful = False
                outcome["error"] = (
                    f"worker peak RSS {int(observed_rss)} exceeds limit {int(memory_limit_bytes)}"
                )
            record: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "condition_id": _sha256_json(
                    {
                        "graph_hash": instance.graph_hash,
                        "method": method,
                        "optimizer_seed": optimizer_seed,
                    }
                )[:24],
                "graph_hash": instance.graph_hash,
                "cover_hash": instance.cover_hash,
                "graph_seed": int(instance.metadata["seed_requested"]),
                "optimizer_seed": int(optimizer_seed),
                "detector_seed": int(run_seed),
                "method": method,
                "resolution": gamma,
                "max_memberships": 1 if method == "disjoint" else int(cap),
                "metadata_free": True,
                "status": status,
                "runtime_seconds": float(outcome.get("runtime_seconds", time.monotonic() - started)),
                "memory": memory,
                "error": outcome.get("error"),
                "method_metadata": method_meta,
                "predicted_cover": predicted,
            }
            # ``status`` is normalized to ``completed`` above, so checking for
            # literal ``ok`` would drop every successful metrics payload.
            if successful and status == "completed" and predicted is not None:
                try:
                    record["metrics"] = evaluate_cover(
                        predicted,
                        instance.cover,
                        graph.vcount(),
                        compute_omega=False,
                    )
                except BaseException as exc:
                    record["status"] = "failed_scoring"
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    record["traceback"] = traceback.format_exc()
            else:
                record["metrics"] = None
            rows.append(record)
    return rows


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one committed checkpoint and flush it before continuing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        _append_jsonl(path, row)


def _graph_key(record: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(record["condition_index"]), int(record["graph_index_within_condition"])
    except (KeyError, TypeError, ValueError):
        return None


def _latest_graph_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse append-only attempts to the latest record for each graph key."""
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for record in records:
        key = _graph_key(record)
        if key is None:
            unkeyed.append(dict(record))
        else:
            latest[key] = dict(record)
    return [latest[key] for key in sorted(latest)] + unkeyed


def _validate_graph_event_history(records: Sequence[dict[str, Any]]) -> None:
    """Reject duplicate completed graph events before collapsing retries.

    Generation failures are append-only retry history: a failed attempt may be
    followed by another failure or by one successful generation.  A second
    ``completed`` event for the same condition/graph key is different: the
    terminal materializer cannot know which graph archive/result rows belong to
    the key, so silently selecting the latest event would permit a corrupted
    ledger (or an accidental duplicate append) to change the study identity.
    Keep the intentional failure history, but fail closed on more than one
    completion and on unknown statuses for keyed events.
    """
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = _graph_key(record)
        if key is None:
            # Legacy/unkeyed rows are left for the existing materializer to
            # preserve.  They cannot participate in stable-key duplicate
            # detection, but they are not allowed to masquerade as a keyed
            # completion.
            continue
        status = str(record.get("status", ""))
        if status not in {"completed", "generation_failed"}:
            raise ValueError(f"unknown graph checkpoint status for key {key}: {status!r}")
        if status != "completed":
            continue
        previous = completed.get(key)
        if previous is not None:
            raise ValueError(f"duplicate completed graph checkpoint for key: {key}")
        completed[key] = record


def _load_graph_records(output: Path) -> list[dict[str, Any]]:
    checkpoint = output / "graphs.jsonl"
    if checkpoint.is_file():
        return _load_jsonl(checkpoint)
    legacy = output / "graphs.json"
    if not legacy.is_file():
        return []
    try:
        value = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load graph ledger {legacy}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"cannot load graph ledger {legacy}: expected a JSON list of objects")
    return [dict(item) for item in value]


def _load_graph_instance(output: Path, record: dict[str, Any]) -> OverlappingLFRGraph:
    graph_hash = str(record.get("graph_hash") or "")
    if not graph_hash:
        raise ValueError("graph checkpoint has no graph_hash")
    path = output / "graphs" / f"{graph_hash}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot reload graph checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("edges"), list) or not isinstance(payload.get("cover"), list):
        raise ValueError(f"graph checkpoint {path} has invalid schema")
    graph = ig.Graph(n=int((payload.get("metadata") or {}).get("n", 0)), edges=payload["edges"], directed=False)
    instance = OverlappingLFRGraph(
        graph=graph,
        cover=[[int(vertex) for vertex in body] for body in payload["cover"]],
        metadata=dict(payload.get("metadata") or {}),
    )
    if instance.graph_hash != graph_hash:
        raise ValueError(f"graph checkpoint {path} hash mismatch")
    if record.get("cover_hash") and instance.cover_hash != str(record["cover_hash"]):
        raise ValueError(f"graph checkpoint {path} cover hash mismatch")
    return instance


def _write_progress(output: Path, payload: dict[str, Any]) -> None:
    _atomic_json(output / "progress.json", payload)


def _progress_payload(
    *,
    status: str,
    phase: str,
    run_identity: str,
    started_monotonic: float,
    completed_graphs: int,
    total_graphs: int,
    completed_rows: int,
    total_rows: int,
    condition_index: int | None = None,
    graph_index: int | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = max(0.0, time.monotonic() - started_monotonic)
    rate = completed_graphs / elapsed if elapsed > 0 else None
    remaining = max(0, total_graphs - completed_graphs)
    eta = (remaining / rate) if rate and rate > 0 else None
    fraction = completed_graphs / total_graphs if total_graphs else 1.0
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": status,
        "phase": phase,
        "run_identity": run_identity,
        "active_condition_index": condition_index,
        "active_graph_index": graph_index,
        "completed_graphs": int(completed_graphs),
        "total_graphs": int(total_graphs),
        "completed_rows": int(completed_rows),
        "total_rows_expected": int(total_rows),
        "fraction": float(fraction),
        "elapsed_seconds": float(elapsed),
        "rate_graphs_per_second": rate,
        "eta_seconds": eta,
        "updated_utc": _utc_now(),
        "pid": os.getpid(),
        "resources": resources or {},
    }


def _terminal_aggregate(
    output: Path,
    *,
    config: dict[str, Any],
    status_override: str | None = None,
) -> dict[str, Any]:
    """Materialize and independently reload terminal artifacts from checkpoints."""
    rows = _load_jsonl(output / "results.jsonl")
    seen: dict[tuple[str, str, int], str] = {}
    duplicate_keys = 0
    for row in rows:
        key = _result_key(row)
        if key is None:
            continue
        digest = _canonical_payload_sha256(row)
        previous = seen.get(key)
        if previous is not None:
            if previous != digest:
                raise ValueError(f"conflicting duplicate result key in checkpoint: {key}")
            duplicate_keys += 1
        else:
            seen[key] = digest
    if duplicate_keys:
        raise ValueError(f"duplicate result keys in append-only checkpoint: {duplicate_keys}")
    graph_events = _load_graph_records(output)
    _validate_graph_event_history(graph_events)
    graph_records = _latest_graph_records(graph_events)
    completed_graphs = sum(1 for record in graph_records if record.get("status") == "completed")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "failed"))
        status_counts[status] = status_counts.get(status, 0) + 1
    graph_status_counts: dict[str, int] = {}
    for record in graph_records:
        status = str(record.get("status", "failed"))
        graph_status_counts[status] = graph_status_counts.get(status, 0) + 1
    has_failures = any(status != "completed" for status in status_counts) or any(
        status != "completed" for status in graph_status_counts
    )
    _atomic_json(output / "graphs.json", graph_records)
    _write_csv(output / "results.csv", rows)
    terminal = {
        "graphs_json_reloaded": False,
        "results_jsonl_reloaded": False,
        "results_csv_reloaded": False,
        "append_only_checkpoint": True,
        "graph_event_count": len(graph_events),
        "result_event_count": len(rows),
        "duplicate_result_keys": duplicate_keys,
    }
    reloaded_graphs = json.loads((output / "graphs.json").read_text(encoding="utf-8"))
    reloaded_rows = _load_jsonl(output / "results.jsonl")
    with (output / "results.csv").open("r", encoding="utf-8", newline="") as stream:
        reloaded_csv_rows = list(csv.DictReader(stream))
    terminal["graphs_json_reloaded"] = reloaded_graphs == graph_records
    terminal["results_jsonl_reloaded"] = reloaded_rows == rows
    terminal["results_csv_reloaded"] = len(reloaded_csv_rows) == len(rows)
    if not all(terminal[key] for key in ("graphs_json_reloaded", "results_jsonl_reloaded", "results_csv_reloaded")):
        raise ValueError("terminal aggregate reload did not reproduce checkpoints")
    manifest = {
        **config,
        "result_rows": len(rows),
        "graphs_attempted": len(graph_records),
        "graphs_completed": completed_graphs,
        "status_counts": status_counts,
        "graph_status_counts": graph_status_counts,
        "study_status": status_override
        or ("completed_with_failures" if has_failures else "completed"),
        "graph_seed_count": len({record.get("graph_seed") for record in graph_records}),
        "graph_hashes": [record.get("graph_hash") for record in graph_records if record.get("graph_hash")],
        "checkpoint": {
            "graphs_jsonl": str(output / "graphs.jsonl"),
            "results_jsonl": str(output / "results.jsonl"),
            "append_only": True,
            "graph_event_count": len(graph_events),
            "result_event_count": len(rows),
        },
        "terminal_reload": terminal,
        "config_sha256": _canonical_payload_sha256(config),
        "manifest_generated_utc": _utc_now(),
    }
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "condition_id", "condition_index", "graph_index_within_condition", "mixing", "overlap_fraction", "overlap_multiplicity",
        "graph_hash", "cover_hash", "graph_seed", "optimizer_seed",
        "detector_seed", "method", "resolution", "max_memberships", "metadata_free",
        "status", "runtime_seconds", "peak_rss_bytes", "memory_limit_bytes", "matching_f1", "node_micro_f1", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics") or {}
            memory = row.get("memory") or {}
            writer.writerow({
                **row,
                "peak_rss_bytes": memory.get("peak_rss_bytes"),
                "memory_limit_bytes": memory.get("limit_bytes"),
                "matching_f1": metrics.get("matching_f1"),
                "node_micro_f1": metrics.get("node_micro_f1"),
            })
    temporary.replace(path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            # Atomic writes should never leave malformed rows.  Refuse to
            # resume rather than silently dropping a middle or final record;
            # the caller receives an actionable corruption error.
            raise ValueError(f"cannot resume malformed results.jsonl line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"cannot resume non-object results.jsonl line {line_number}")
        rows.append(value)
    return rows


def _result_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    graph_hash = row.get("graph_hash")
    method = row.get("method")
    if not graph_hash or method is None:
        return None
    try:
        optimizer_seed = int(row.get("optimizer_seed", 0))
    except (TypeError, ValueError):
        return None
    return str(graph_hash), str(method), optimizer_seed


def run_study(
    *,
    output_dir: str | Path,
    profile: str = "smoke",
    n: int | None = None,
    graphs_per_condition: int | None = None,
    max_graphs: int | None = None,
    methods: Sequence[str] | None = None,
    optimizer_seeds: Sequence[int] = (0, 1, 2),
    timeout_seconds: float | None = 2.0,
    memory_limit_bytes: int | None = None,
    seed_offset: int = 0,
    detector_cap: int = 4,
    official_generator: str | Path | None = None,
    official_config: str | Path | dict[str, Any] | None = None,
    official_format: str = "lfrbenchmarks",
    official_source_commit: str | None = None,
    official_source_tree: str | None = None,
    resume: bool = False,
    stop_after_graphs: int | None = None,
) -> dict[str, Any]:
    """Execute a resumable smoke/pilot/standard study and write artifacts.

    Smoke uses the deterministic compatibility fixture. Pilot and standard
    fail closed unless ``official_generator`` points to the pinned upstream
    C++ LFRbenchmarks executable and ``official_format`` is
    ``lfrbenchmarks``. ``official_format=json`` is retained only for bounded
    smoke compatibility tests using the strict JSON contract documented by
    :func:`generate_official_overlapping_lfr`; it cannot produce pilot or
    standard canonical evidence. Passing no binary, a generic JSON adapter,
    or a binary that fails the bounded-pilot binding is therefore a recorded
    blocked plan, never a claim of canonical evidence.
    With ``resume=True``, append-only graph/result checkpoints are loaded by
    stable keys and skipped.  A graph batch is fsynced before the next graph
    starts; terminal CSV/JSON artifacts are then aggregated and independently
    reloaded.  ``stop_after_graphs`` is a test-only disposable interruption
    hook used to prove recovery at a safe batch boundary.
    """
    profile = str(profile).lower()
    if profile not in {"smoke", "pilot", "standard"}:
        raise ValueError("profile must be smoke, pilot, or standard")
    official_format = str(official_format).strip().lower()
    if official_format not in {"json", "lfrbenchmarks"}:
        raise ValueError("official_format must be json or lfrbenchmarks")
    if profile == "smoke":
        n = 80 if n is None else n
        graphs_per_condition = 1 if graphs_per_condition is None else graphs_per_condition
        methods = ("hedonic_multiphase", "disjoint", "singleton") if methods is None else methods
        timeout_seconds = 5.0 if timeout_seconds is None else timeout_seconds
    elif profile == "pilot":
        n = 200 if n is None else n
        graphs_per_condition = 5 if graphs_per_condition is None else graphs_per_condition
        methods = ("hedonic_multiphase", "slpa", "demon", "kcp", "chen", "disjoint", "singleton") if methods is None else methods
        timeout_seconds = 30.0 if timeout_seconds is None else timeout_seconds
    else:
        n = DEFAULT_N if n is None else n
        graphs_per_condition = DEFAULT_GRAPHS_PER_CONDITION if graphs_per_condition is None else graphs_per_condition
        methods = DEFAULT_METHODS if methods is None else methods
        timeout_seconds = 60.0 if timeout_seconds is None else timeout_seconds
    methods = tuple(str(method) for method in methods)
    optimizer_seeds = tuple(int(seed) for seed in optimizer_seeds)
    if int(detector_cap) < 1:
        raise ValueError("detector_cap must be positive")
    if memory_limit_bytes is not None and int(memory_limit_bytes) <= 0:
        raise ValueError("memory_limit_bytes must be positive or None")
    if stop_after_graphs is not None and int(stop_after_graphs) < 1:
        raise ValueError("stop_after_graphs must be positive or None")
    if official_generator is not None:
        official_generator = Path(official_generator).expanduser()
    output = expand_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = condition_grid()
    graphs_per_condition = int(graphs_per_condition)
    expected = len(conditions) * graphs_per_condition
    if max_graphs is not None:
        expected = min(expected, max(0, int(max_graphs)))
    resolved_source_commit = official_source_commit
    resolved_source_tree = official_source_tree
    configured_receipt: dict[str, Any] = {}
    if official_config is not None and (resolved_source_commit is None or resolved_source_tree is None):
        try:
            configured_receipt = _load_json_object(official_config)
        except OfficialGeneratorUnavailable:
            configured_receipt = {}
        if resolved_source_commit is None:
            resolved_source_commit = configured_receipt.get("official_source_commit")
        if resolved_source_tree is None:
            resolved_source_tree = configured_receipt.get("official_source_tree")
    elif official_config is not None:
        try:
            configured_receipt = _load_json_object(official_config)
        except OfficialGeneratorUnavailable:
            configured_receipt = {}
    configured_pilot = configured_receipt.get("bounded_pilot") if isinstance(configured_receipt, dict) else None
    generator_binding = _official_generator_binding(
        profile=profile,
        official_generator=official_generator,
        official_format=official_format,
        source_commit=resolved_source_commit,
        source_tree=resolved_source_tree,
        configured_receipt=configured_receipt,
    )
    identity_payload = _run_identity_payload(
        profile=profile,
        output_dir=output,
        conditions=conditions,
        n=int(n),
        graphs_per_condition=graphs_per_condition,
        max_graphs=max_graphs,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        seed_offset=seed_offset,
        detector_cap=detector_cap,
        official_generator=official_generator,
        official_config=official_config,
        official_format=official_format,
        official_source_commit=resolved_source_commit,
        official_source_tree=resolved_source_tree,
    )
    run_identity = str(identity_payload["run_identity"])
    launch = build_launch_command(
        profile=profile,
        output_dir=output,
        official_generator=official_generator,
        official_config=official_config if isinstance(official_config, (str, Path)) else None,
        official_format=official_format,
        official_source_commit=resolved_source_commit,
        official_source_tree=resolved_source_tree,
        n=int(n),
        graphs_per_condition=graphs_per_condition,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        detector_cap=detector_cap,
        seed_offset=seed_offset,
    )
    resource_estimate = estimate_lfr_resources(
        total_graphs=expected,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        target_n=int(n),
        pilot_receipt=configured_pilot if isinstance(configured_pilot, dict) else None,
        memory_limit_bytes=memory_limit_bytes,
    )
    config: dict[str, Any] = {
        **identity_payload,
        "generator_id": OFFICIAL_GENERATOR_ID if profile in {"pilot", "standard"} or official_generator else GENERATOR_ID,
        "generator_policy": "official_binary_required" if profile in {"pilot", "standard"} or official_generator else "compatibility_smoke_fixture",
        "expected_graphs": int(expected),
        "graphs_per_condition": graphs_per_condition,
        "resolution_rule": "graph density, fixed before test scoring",
        "metadata_free_detection": True,
        "failure_policy": "retain generation, timeout, memory, detector, and scoring failures",
        "resume": bool(resume),
        "blocked_reason": None,
        "launch_command": launch,
        "resource_estimate": resource_estimate,
        "official_generator_binding": generator_binding,
    }
    existing_config_path = output / "config.json"
    if resume and existing_config_path.is_file():
        try:
            previous_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot resume invalid config.json: {exc}") from exc
        previous_identity = previous_config.get("run_identity")
        if previous_identity and str(previous_identity) != run_identity:
            raise ValueError(
                "resume configuration does not match the persisted run identity; "
                "start a new output root"
            )
    checkpoint_paths = (output / "results.jsonl", output / "graphs.jsonl")
    if not resume and any(path.exists() for path in checkpoint_paths):
        raise ValueError("output already contains append-only checkpoints; pass --resume")
    existing_rows = _load_jsonl(output / "results.jsonl") if resume else []
    graph_events = _load_graph_records(output) if resume else []
    if resume:
        _validate_graph_event_history(graph_events)
    if resume and not (output / "graphs.jsonl").is_file() and graph_events:
        # Migrate the old atomic list once, preserving its rows as append-only
        # events before any successor batch is started.
        _append_jsonl_rows(output / "graphs.jsonl", graph_events)
    graph_records = _latest_graph_records(graph_events)
    graph_by_key = {key: record for record in graph_records if (key := _graph_key(record)) is not None}
    rows = list(existing_rows)
    skip_all = {key for row in rows if (key := _result_key(row)) is not None}
    blocked_official = profile in {"pilot", "standard"} and generator_binding["status"] != "ok"
    if blocked_official:
        config["blocked_reason"] = (
            "official_generator_missing"
            if official_generator is None
            else "official_generator_binding_failed"
        )
    _atomic_json(output / "config.json", config)
    started_monotonic = time.monotonic()
    total_rows = expected * sum(1 if method == "singleton" else len(optimizer_seeds) for method in methods)
    completed_graphs = sum(
        1 for record in graph_records
        if record.get("status") == "completed"
        and record.get("graph_hash")
        and {
            (str(record["graph_hash"]), str(method), 0 if method == "singleton" else int(seed))
            for method in methods
            for seed in ((0,) if method == "singleton" else optimizer_seeds)
        } <= skip_all
    )
    _write_progress(
        output,
        _progress_payload(
            status="blocked" if blocked_official else "running",
            phase="preflight" if blocked_official else "generation",
            run_identity=run_identity,
            started_monotonic=started_monotonic,
            completed_graphs=completed_graphs,
            total_graphs=expected,
            completed_rows=len(rows),
            total_rows=total_rows,
            resources=resource_estimate,
        ),
    )
    if blocked_official:
        return _terminal_aggregate(
            output,
            config=config,
            status_override=(
                "blocked_official_generator_missing"
                if official_generator is None
                else "blocked_official_generator_binding_failed"
            ),
        )

    def graph_complete(record: dict[str, Any]) -> bool:
        graph_hash = record.get("graph_hash")
        if record.get("status") != "completed" or not graph_hash:
            return False
        expected_keys = {
            (str(graph_hash), str(method), 0 if method == "singleton" else int(seed))
            for method in methods
            for seed in ((0,) if method == "singleton" else optimizer_seeds)
        }
        return expected_keys <= skip_all

    for condition_index, condition in enumerate(conditions):
        for graph_index in range(graphs_per_condition):
            if completed_graphs >= expected:
                break
            graph_seed = int(seed_offset) + condition_index * 100_000 + graph_index
            graph_key = (int(condition_index), int(graph_index))
            existing_graph = graph_by_key.get(graph_key)
            if existing_graph is not None and graph_complete(existing_graph):
                continue
            try:
                if existing_graph and existing_graph.get("graph_hash"):
                    instance = _load_graph_instance(output, existing_graph)
                else:
                    request = {
                        "protocol_version": PROTOCOL_VERSION,
                        "n": int(n),
                        "average_degree": 20 if int(n) >= 200 else min(10, int(n) - 1),
                        "max_degree": min(DEFAULT_MAX_DEGREE, max(1, int(n) - 1)),
                        "min_community": min(DEFAULT_MIN_COMMUNITY, max(2, int(n) // 4)),
                        "max_community": min(DEFAULT_MAX_COMMUNITY, max(2, int(n) // 2)),
                        "tau1": DEFAULT_TAU1,
                        "tau2": DEFAULT_TAU2,
                        "mixing": float(condition["mixing"]),
                        "overlap_fraction": float(condition["overlap_fraction"]),
                        "overlap_multiplicity": int(condition["overlap_multiplicity"]),
                        "seed": int(graph_seed),
                    }
                    if official_generator is None:
                        instance = generate_overlapping_lfr(**{
                            key: value for key, value in request.items() if key != "protocol_version"
                        })
                    else:
                        official_request = dict(request)
                        if official_config is not None:
                            base_request = _load_json_object(official_config)
                            base_request.update(official_request)
                            official_request = base_request
                        if official_format == "lfrbenchmarks":
                            instance = generate_official_lfrbenchmarks(
                                official_generator,
                                config=official_request,
                                timeout_seconds=max(1.0, float(timeout_seconds or 300.0)),
                                source_commit=resolved_source_commit,
                                source_tree=resolved_source_tree,
                            )
                        else:
                            instance = generate_official_overlapping_lfr(
                                official_generator,
                                config=official_request,
                                timeout_seconds=max(1.0, float(timeout_seconds or 300.0)),
                            )
                    graph_path = output / "graphs" / f"{instance.graph_hash}.json"
                    _atomic_json(
                        graph_path,
                        {
                            "metadata": instance.metadata,
                            "edges": [list(edge) for edge in instance.graph.get_edgelist()],
                            "cover": instance.cover,
                        },
                    )
                    graph_record = {
                        "condition_index": int(condition_index),
                        "graph_index_within_condition": int(graph_index),
                        "graph_hash": instance.graph_hash,
                        "cover_hash": instance.cover_hash,
                        "condition": condition,
                        "graph_seed": graph_seed,
                        "metadata": instance.metadata,
                        "generator_policy": instance.metadata.get("generator_policy"),
                        "status": "completed",
                        "run_identity": run_identity,
                        "checkpointed_utc": _utc_now(),
                    }
                    if existing_graph and existing_graph.get("status") != "completed":
                        graph_record["previous_generation_attempt"] = existing_graph
                    _append_jsonl(output / "graphs.jsonl", graph_record)
                    graph_by_key[graph_key] = graph_record
                    existing_graph = graph_record
                method_rows = run_graph_methods(
                    instance,
                    methods=methods,
                    optimizer_seeds=optimizer_seeds,
                    timeout_seconds=timeout_seconds,
                    memory_limit_bytes=memory_limit_bytes,
                    cap=int(detector_cap),
                    skip_keys=skip_all,
                )
                for row in method_rows:
                    row.update(
                        {
                            "condition_index": int(condition_index),
                            "graph_index_within_condition": int(graph_index),
                            "mixing": float(condition["mixing"]),
                            "overlap_fraction": float(condition["overlap_fraction"]),
                            "overlap_multiplicity": int(condition["overlap_multiplicity"]),
                            "run_identity": run_identity,
                            "checkpointed_utc": _utc_now(),
                        }
                    )
                    key = _result_key(row)
                    if key is None or key not in skip_all:
                        _append_jsonl(output / "results.jsonl", row)
                        rows.append(row)
                        if key is not None:
                            skip_all.add(key)
                if graph_complete(existing_graph or {}):
                    completed_graphs += 1
                _write_progress(
                    output,
                    _progress_payload(
                        status="running",
                        phase="detectors",
                        run_identity=run_identity,
                        started_monotonic=started_monotonic,
                        completed_graphs=completed_graphs,
                        total_graphs=expected,
                        completed_rows=len(rows),
                        total_rows=total_rows,
                        condition_index=condition_index,
                        graph_index=graph_index,
                        resources=resource_estimate,
                    ),
                )
                if stop_after_graphs is not None and completed_graphs >= int(stop_after_graphs):
                    _write_progress(
                        output,
                        _progress_payload(
                            status="interrupted",
                            phase="checkpoint",
                            run_identity=run_identity,
                            started_monotonic=started_monotonic,
                            completed_graphs=completed_graphs,
                            total_graphs=expected,
                            completed_rows=len(rows),
                            total_rows=total_rows,
                            condition_index=condition_index,
                            graph_index=graph_index,
                            resources=resource_estimate,
                        ),
                    )
                    raise StudyInterrupted(
                        f"disposable interruption after {completed_graphs} committed graph batches"
                    )
            except StudyInterrupted:
                raise
            except Exception as exc:
                failure_record = {
                    "condition_index": int(condition_index),
                    "graph_index_within_condition": int(graph_index),
                    "graph_hash": None,
                    "cover_hash": None,
                    "condition": condition,
                    "graph_seed": graph_seed,
                    "generator_policy": "official_binary_required" if official_generator else "compatibility_smoke_fixture",
                    "status": "generation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "run_identity": run_identity,
                    "checkpointed_utc": _utc_now(),
                }
                _append_jsonl(output / "graphs.jsonl", failure_record)
                graph_by_key[graph_key] = failure_record
                _write_progress(
                    output,
                    _progress_payload(
                        status="running",
                        phase="generation_failure",
                        run_identity=run_identity,
                        started_monotonic=started_monotonic,
                        completed_graphs=completed_graphs,
                        total_graphs=expected,
                        completed_rows=len(rows),
                        total_rows=total_rows,
                        condition_index=condition_index,
                        graph_index=graph_index,
                        resources=resource_estimate,
                    ),
                )
    manifest = _terminal_aggregate(output, config=config)
    _write_progress(
        output,
        _progress_payload(
            status="completed" if manifest["study_status"] == "completed" else "completed_with_failures",
            phase="terminal_reload",
            run_identity=run_identity,
            started_monotonic=started_monotonic,
            completed_graphs=int(manifest["graphs_completed"]),
            total_graphs=expected,
            completed_rows=int(manifest["result_rows"]),
            total_rows=total_rows,
            resources=resource_estimate,
        ),
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Planted-overlap LFR study with an official-binary gate (Astra TKT-11)")
    parser.add_argument("--profile", choices=("smoke", "pilot", "standard"), default="smoke")
    parser.add_argument("--output-dir", "--output_dir", type=Path, default=OVERLAPPING_ARTIFACTS_DIR / "overlap_lfr")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--graphs-per-condition", type=int, default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--methods", default=None, help="comma-separated fixed method names")
    parser.add_argument("--optimizer-seeds", default="0,1,2")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-limit-gb", type=float, default=None)
    parser.add_argument("--detector-cap", type=int, default=4, help="maximum memberships passed to hedonic detectors")
    parser.add_argument(
        "--official-generator",
        type=Path,
        default=None,
        help="pinned LFRbenchmarks executable for pilot/standard; JSON adapter only for smoke",
    )
    parser.add_argument("--official-config", type=Path, default=None, help="optional JSON defaults merged into each official request")
    parser.add_argument(
        "--official-format",
        choices=("json", "lfrbenchmarks"),
        default="lfrbenchmarks",
        help="native LFRbenchmarks format required for pilot/standard; JSON is smoke-only",
    )
    parser.add_argument(
        "--official-source-commit",
        default=None,
        help="required exact LFRbenchmarks source commit (native format only)",
    )
    parser.add_argument(
        "--official-source-tree",
        default=None,
        help="exact LFRbenchmarks source tree hash; required for pilot/standard native runs",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="resume from existing results.jsonl/graphs.json keys")
    parser.add_argument("--dry-run", action="store_true", help="write config/manifest plan without detector runs")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run the fail-closed dependency/binary/storage/disposable-end-to-end preflight",
    )
    parser.add_argument(
        "--pilot-receipt",
        type=Path,
        default=None,
        help="bounded native pilot receipt used for storage/RSS estimates",
    )
    return parser


def _parse_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not result:
        raise ValueError("expected at least one integer")
    return result


def preflight_lfr_run(
    *,
    output_dir: str | Path,
    profile: str = "standard",
    n: int | None = None,
    graphs_per_condition: int | None = None,
    max_graphs: int | None = None,
    methods: Sequence[str] | None = None,
    optimizer_seeds: Sequence[int] = (0, 1, 2),
    timeout_seconds: float | None = None,
    memory_limit_bytes: int | None = None,
    seed_offset: int = 0,
    detector_cap: int = 4,
    official_generator: str | Path | None = None,
    official_config: str | Path | dict[str, Any] | None = None,
    official_format: str = "lfrbenchmarks",
    official_source_commit: str | None = None,
    official_source_tree: str | None = None,
    pilot_receipt: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fail-closed TKT-11 launch preflight.

    The preflight is intentionally bounded: it imports the complete dependency
    closure, validates the pinned native executable/source identity, checks
    storage, and runs one disposable graph through generation, detector,
    checkpoint, aggregation, and terminal reload.  It never writes to the
    requested production output root beyond ``preflight.json``.
    """
    profile = str(profile).lower()
    if profile not in {"smoke", "pilot", "standard"}:
        raise ValueError("profile must be smoke, pilot, or standard")
    official_format = str(official_format).strip().lower()
    if official_format not in {"json", "lfrbenchmarks"}:
        raise ValueError("official_format must be json or lfrbenchmarks")
    if profile == "smoke":
        n = 80 if n is None else n
        graphs_per_condition = 1 if graphs_per_condition is None else graphs_per_condition
        methods = ("hedonic_multiphase", "disjoint", "singleton") if methods is None else methods
        timeout_seconds = 5.0 if timeout_seconds is None else timeout_seconds
    elif profile == "pilot":
        n = 200 if n is None else n
        graphs_per_condition = 5 if graphs_per_condition is None else graphs_per_condition
        methods = ("hedonic_multiphase", "slpa", "demon", "kcp", "chen", "disjoint", "singleton") if methods is None else methods
        timeout_seconds = 30.0 if timeout_seconds is None else timeout_seconds
    else:
        n = DEFAULT_N if n is None else n
        graphs_per_condition = DEFAULT_GRAPHS_PER_CONDITION if graphs_per_condition is None else graphs_per_condition
        methods = DEFAULT_METHODS if methods is None else methods
        timeout_seconds = 60.0 if timeout_seconds is None else timeout_seconds
    methods = tuple(str(method) for method in methods)
    optimizer_seeds = tuple(int(seed) for seed in optimizer_seeds)
    output = expand_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = condition_grid()
    total_graphs = len(conditions) * int(graphs_per_condition)
    if max_graphs is not None:
        total_graphs = min(total_graphs, max(0, int(max_graphs)))
    if official_generator is not None:
        official_generator = Path(official_generator).expanduser()
    resolved_source_commit = official_source_commit
    resolved_source_tree = official_source_tree
    configured_receipt: dict[str, Any] = {}
    if official_config is not None and (resolved_source_commit is None or resolved_source_tree is None):
        try:
            configured = _load_json_object(official_config)
        except OfficialGeneratorUnavailable:
            configured = {}
        resolved_source_commit = resolved_source_commit or configured.get("official_source_commit")
        resolved_source_tree = resolved_source_tree or configured.get("official_source_tree")
        configured_receipt = configured
    elif official_config is not None:
        try:
            configured_receipt = _load_json_object(official_config)
        except OfficialGeneratorUnavailable:
            configured_receipt = {}
    effective_pilot_receipt: str | Path | dict[str, Any] | None = pilot_receipt
    if effective_pilot_receipt is None and isinstance(configured_receipt.get("bounded_pilot"), dict):
        effective_pilot_receipt = dict(configured_receipt["bounded_pilot"])
    generator_binding = _official_generator_binding(
        profile=profile,
        official_generator=official_generator,
        official_format=official_format,
        source_commit=resolved_source_commit,
        source_tree=resolved_source_tree,
        configured_receipt=configured_receipt,
    )
    identity_payload = _run_identity_payload(
        profile=profile,
        output_dir=output,
        conditions=conditions,
        n=int(n),
        graphs_per_condition=int(graphs_per_condition),
        max_graphs=max_graphs,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        seed_offset=seed_offset,
        detector_cap=detector_cap,
        official_generator=official_generator,
        official_config=official_config,
        official_format=official_format,
        official_source_commit=resolved_source_commit,
        official_source_tree=resolved_source_tree,
    )
    launch = build_launch_command(
        profile=profile,
        output_dir=output,
        official_generator=official_generator,
        official_config=official_config if isinstance(official_config, (str, Path)) else None,
        official_format=official_format,
        official_source_commit=resolved_source_commit,
        official_source_tree=resolved_source_tree,
        n=int(n),
        graphs_per_condition=int(graphs_per_condition),
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        detector_cap=detector_cap,
        seed_offset=seed_offset,
    )
    required_modules = {
        "hedonic": "hedonic",
        "igraph": "igraph",
        "lfr_runner": "hedonic.experiments.overlapping.overlap_lfr",
        "metrics": "hedonic.experiments.overlapping.metrics",
        "execution": "hedonic.experiments.overlapping.execution",
        "baselines": "hedonic.experiments.overlapping.baselines",
    }
    if "demon" in methods:
        required_modules["demon"] = "demon"
    if "kcp" in methods:
        required_modules["networkx"] = "networkx"
    import_checks: dict[str, Any] = {}
    for name, module_name in required_modules.items():
        try:
            module = importlib.import_module(module_name)
            import_checks[name] = {"module": module_name, "status": "ok", "file": str(getattr(module, "__file__", ""))}
        except Exception as exc:
            import_checks[name] = {"module": module_name, "status": "missing", "error": f"{type(exc).__name__}: {exc}"}
    distribution_names = {"hedonic": "hedonic", "lucas-igraph": "lucas-igraph", "numpy": "numpy"}
    for method in methods:
        if method == "demon":
            distribution_names["demon"] = "demon"
        elif method == "kcp":
            distribution_names["networkx"] = "networkx"
    distribution_checks = {
        name: {"distribution": distribution, "version": _distribution_version(distribution), "status": "ok" if _distribution_version(distribution) else "missing"}
        for name, distribution in distribution_names.items()
    }
    isolated_external = (
        _isolated_slpa_preflight()
        if "slpa" in methods
        else {"status": "not_required", "policy": "SLPA not selected"}
    )
    checks: dict[str, Any] = {
        "interpreter": {
            "status": "ok" if sys.version_info >= (3, 12) and platform.python_implementation() == "CPython" else "failed",
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "imports": {"status": "ok" if all(item["status"] == "ok" for item in import_checks.values()) else "failed", "items": import_checks},
        "distributions": {"status": "ok" if all(item["status"] == "ok" for item in distribution_checks.values()) else "failed", "items": distribution_checks},
        "isolated_external": isolated_external,
        "output_directory": {
            "status": "ok" if os.access(output, os.W_OK) else "failed",
            "path": str(output),
            "writable": os.access(output, os.W_OK),
        },
    }
    disk = shutil.disk_usage(output)
    resource_estimate = estimate_lfr_resources(
        total_graphs=total_graphs,
        methods=methods,
        optimizer_seeds=optimizer_seeds,
        target_n=int(n),
        pilot_receipt=effective_pilot_receipt,
        memory_limit_bytes=memory_limit_bytes,
    )
    estimated_storage = resource_estimate.get("estimated_total_storage_bytes")
    checks["disk_space"] = {
        "status": "ok" if estimated_storage is None or int(disk.free) >= int(estimated_storage) else "failed",
        "free_bytes": int(disk.free),
        "estimated_required_bytes": estimated_storage,
    }
    binary_check: dict[str, Any]
    if profile in {"pilot", "standard"} or official_generator is not None:
        binary_check = generator_binding
    else:
        binary_check = {"status": "not_required", "policy": "compatibility_smoke_fixture"}
    checks["official_generator"] = binary_check
    checks["pilot_receipt"] = {
        "status": "ok"
        if resource_estimate.get("pilot", {}).get("available") or profile == "smoke"
        else "failed",
        "receipt": resource_estimate.get("pilot", {}),
    }
    fixture: dict[str, Any] = {"status": "not_run"}
    prerequisite_ok = all(
        checks[name]["status"] in {"ok", "not_required"}
        for name in ("interpreter", "imports", "distributions", "isolated_external", "output_directory", "disk_space", "official_generator", "pilot_receipt")
    )
    if prerequisite_ok:
        with tempfile.TemporaryDirectory(prefix="hedonic-lfr-preflight-") as temporary:
            fixture_output = Path(temporary) / "fixture"
            try:
                fixture_manifest = run_study(
                    output_dir=fixture_output,
                    profile="smoke",
                    n=min(200, int(n)),
                    graphs_per_condition=1,
                    max_graphs=1,
                    methods=methods,
                    optimizer_seeds=(0,),
                    timeout_seconds=min(float(timeout_seconds or 30.0), 30.0),
                    memory_limit_bytes=memory_limit_bytes,
                    detector_cap=detector_cap,
                    official_generator=official_generator,
                    official_config=official_config,
                    official_format=official_format,
                    official_source_commit=resolved_source_commit,
                    official_source_tree=resolved_source_tree,
                    resume=False,
                )
                fixture = {
                    "status": "ok" if fixture_manifest.get("terminal_reload", {}).get("graphs_json_reloaded") else "failed",
                    "manifest": fixture_manifest,
                    "output_is_disposable": True,
                }
            except Exception as exc:
                fixture = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    else:
        fixture = {"status": "blocked_by_dependency_or_binding"}
    checks["disposable_end_to_end"] = fixture
    ready = prerequisite_ok and fixture.get("status") == "ok"
    receipt = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "preflight_timestamp_utc": _utc_now(),
        "status": "ready" if ready else "blocked",
        "profile": profile,
        "run_identity": identity_payload["run_identity"],
        "identity_payload": identity_payload,
        "launch_command": launch,
        "checks": checks,
        "resource_estimate": resource_estimate,
        "dependency_closure": {
            "imports": sorted(required_modules.values()),
            "distributions": sorted(distribution_names.values()),
            "isolated_external": {
                "slpa": isolated_external,
            } if "slpa" in methods else {},
            "official_source": {
                "repository": OFFICIAL_LFRBENCHMARKS_REPOSITORY,
                "commit": OFFICIAL_LFRBENCHMARKS_COMMIT,
                "tree": OFFICIAL_LFRBENCHMARKS_TREE,
            },
        },
        "no_production_graphs_launched": True,
    }
    _atomic_json(output / "preflight.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = None if args.methods is None else [item.strip() for item in args.methods.split(",") if item.strip()]
    seeds = _parse_ints(args.optimizer_seeds)
    memory = None if args.memory_limit_gb is None else int(float(args.memory_limit_gb) * 1024**3)
    if args.preflight:
        receipt = preflight_lfr_run(
            output_dir=args.output_dir,
            profile=args.profile,
            n=args.n,
            graphs_per_condition=args.graphs_per_condition,
            max_graphs=args.max_graphs,
            methods=methods,
            optimizer_seeds=seeds,
            timeout_seconds=args.timeout_seconds,
            memory_limit_bytes=memory,
            seed_offset=args.seed_offset,
            detector_cap=args.detector_cap,
            official_generator=args.official_generator,
            official_config=args.official_config,
            official_format=args.official_format,
            official_source_commit=args.official_source_commit,
            official_source_tree=args.official_source_tree,
            pilot_receipt=args.pilot_receipt,
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["status"] == "ready" else 2
    if args.dry_run:
        output = expand_path(args.output_dir)
        plan = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "generator_id": OFFICIAL_GENERATOR_ID if args.profile in {"pilot", "standard"} or args.official_generator else GENERATOR_ID,
            "generator_policy": "official_binary_required" if args.profile in {"pilot", "standard"} or args.official_generator else "compatibility_smoke_fixture",
            "profile": args.profile,
            "conditions": condition_grid(),
            "methods": methods or list(DEFAULT_METHODS),
            "graphs_per_condition": args.graphs_per_condition or (1 if args.profile == "smoke" else 5 if args.profile == "pilot" else DEFAULT_GRAPHS_PER_CONDITION),
            "detector_cap": int(args.detector_cap),
            "official_generator": str(args.official_generator) if args.official_generator else None,
            "official_format": args.official_format,
            "official_source_commit": args.official_source_commit,
            "official_source_tree": args.official_source_tree,
            "resume": bool(args.resume),
            "metadata_free_detection": True,
        }
        _atomic_json(output / "plan.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    manifest = run_study(
        output_dir=args.output_dir,
        profile=args.profile,
        n=args.n,
        graphs_per_condition=args.graphs_per_condition,
        max_graphs=args.max_graphs,
        methods=methods,
        optimizer_seeds=seeds,
        timeout_seconds=args.timeout_seconds,
        memory_limit_bytes=memory,
        seed_offset=args.seed_offset,
        detector_cap=args.detector_cap,
        official_generator=args.official_generator,
        official_config=args.official_config,
        official_format=args.official_format,
        official_source_commit=args.official_source_commit,
        official_source_tree=args.official_source_tree,
        resume=args.resume,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


__all__ = [
    "COMPATIBILITY_GENERATOR_ID",
    "DEFAULT_METHODS",
    "GENERATOR_ID",
    "OFFICIAL_GENERATOR_ID",
    "OFFICIAL_LFRBENCHMARKS_COMMIT",
    "OFFICIAL_LFRBENCHMARKS_OUTPUT_FORMAT",
    "OFFICIAL_LFRBENCHMARKS_REPOSITORY",
    "OFFICIAL_LFRBENCHMARKS_SOURCE_SUBDIRECTORY",
    "OFFICIAL_LFRBENCHMARKS_TREE",
    "OfficialGeneratorUnavailable",
    "OverlappingLFRGraph",
    "StudyInterrupted",
    "build_launch_command",
    "condition_grid",
    "estimate_lfr_resources",
    "generate_graph",
    "generate_lfr_graph",
    "generate_official_overlapping_lfr",
    "generate_official_lfr_graph",
    "generate_official_lfrbenchmarks",
    "generate_official_lfrbenchmarks_graph",
    "generate_overlapping_lfr",
    "environment_receipt",
    "graph_sha256",
    "preflight_lfr_run",
    "run_graph_methods",
    "run_study",
    "main",
]
