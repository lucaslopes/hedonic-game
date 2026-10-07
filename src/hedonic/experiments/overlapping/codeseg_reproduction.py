"""Reproduce the CoDeSEG WWW'25 experiments and SNAP overlap protocol.

The paper's baseline catalogue contains nine method names, but the reported
experiments are split across two protocol families: CoDeSEG, SLPA, Bigclam,
NcGame, and Fox are evaluated on the seven unweighted SNAP overlap networks;
CoDeSEG, Louvain, DER, Leiden, and FLPA are evaluated on the two weighted
non-overlapping Tweet networks.  The runner keeps all nine names selectable
on SNAP so that the disjoint methods can be measured as explicit cross-task
controls; those DBLP rows are not paper-reported DBLP results.

It also exposes the repository's ``Game.community_hedonic`` and four native
hedonic resolution variants as explicit local overlap comparators; they are
not counted as paper methods:

``CoDeSEG, SLPA, Bigclam, NcGame, Fox, Louvain, DER, Leiden, FLPA`` plus
``community_hedonic``, four hedonic variants, ANGEL, Infomap, DEMON, CPM, and
Link Communities.

This module is deliberately separate from the historical five-method Hedonic
benchmark.  It preserves the paper's method names and parameters, uses the
existing ID-safe SNAP loader, reports explicit unavailable/failed methods, and
never substitutes a different detector under a paper method name.  External
executables are configured through CLI flags or environment variables; the
native igraph methods and the pinned CDlib workers are runnable without any
additional code checkout.

The default run uses the full ``all`` ground-truth cover and the paper's
``F1``/LFK-``ONMI`` definitions.  Its ledger also carries the shared
one-to-one, membership, Jaccard, size-weighted, sampled-Omega, coverage, and
structural-overlap metrics plus runtime/resource counters.  ``--max-nodes`` is
an explicit bounded local test mode, not a claim of full-dataset reproduction.
The raw SNAP archives are read-only; all temporary detector inputs and outputs
live below the experiment output directory or the system temporary directory.
The run also writes a Table-2-style Markdown and LaTeX report beside the
metrics ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import igraph as ig

from hedonic import Game

from hedonic.experiments.config import NETWORKS_DIR, OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping.baselines import (
    MethodUnavailable,
    run_slpa,
    slpa_environment_receipt,
)
from hedonic.experiments.overlapping.methods import (
    MethodUnavailable as AdapterMethodUnavailable,
    METHODS as OVERLAPPING_METHODS,
    canonical_method_name,
    run_method as run_overlapping_method,
)
from hedonic.experiments.overlapping.metrics import (
    evaluate_cover,
    overlapping_normalized_mutual_information_lfk,
)
from hedonic.experiments.overlapping.snap import (
    SPECS,
    SnapDataset,
    SnapLoadError,
    UnsupportedCoverVariant,
    bounded_induced_dataset,
    canonicalize_cover,
    load_snap_dataset,
    smoke_dataset,
    synthetic_agmfit_dataset,
)


PROTOCOL_VERSION = "codeseg-www25-snap-overlap-v3"
# Pre-registered, reference-free membership cap M_0 of the density-resolution
# Hedonic adapters.  Protocol-v3 records obtained this value from
# ``max(2, min(8, max(len(community) for community in reference)))``, which
# reads a community *size*, not a per-vertex multiplicity; it evaluates to 8
# on every SNAP network whose largest reference community has at least eight
# members (all saved networks).  The value is now a constant so that the
# unsupervised rows never read the reference cover.
HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS = 8
# Explicit value of ``--hedonic-max-memberships`` that requests the largest
# per-vertex multiplicity of the scored reference cover.  Such rows are
# reference-informed ablations and are labelled by ``max_memberships_source``.
HEDONIC_REFERENCE_MULTIPLICITY_CAP = "reference-multiplicity"
DATASETS: tuple[str, ...] = (
    "amazon",
    "youtube",
    "dblp",
    "livejournal",
    "orkut",
    "friendster",
    "wikipedia",
)
SMOKE_DATASET = "synthetic_agmfit"
PAPER_METHODS: tuple[str, ...] = (
    "codeseg",
    "slpa",
    "bigclam",
    "ncgame",
    "fox",
    "louvain",
    "der",
    "leiden",
    "flpa",
)
METHODS: tuple[str, ...] = PAPER_METHODS + ("community_hedonic",)
# Additional overlap-capable baselines with maintained local adapters.  The
# historical nine-name catalogue remains available as ``METHODS`` for paper
# compatibility; the default CLI run uses this extended set so a fresh setup
# produces a wider DBLP table without relabelling disjoint controls.
EXTENDED_METHODS: tuple[str, ...] = METHODS + (
    "hedonic_local",
    "hedonic_multiphase",
    "hedonic_multiphase_x10",
    "hedonic_multiphase_x100",
    "angel",
    "infomap",
    "demon",
    "cpm",
    "link_clustering",
    "oslom",
    "neo_kmeans",
    "nise",
    "sse",
    "qoce",
    "svi",
    "essc",
)
# The default DBLP protocol contains every method for which the literature
# review found a public implementation. OSLOM is retained as an explicit
# opt-in because the legacy binary can fail on an induced bounded graph; all
# other code-found methods are part of the fail-closed default.
DBLP_METHODS: tuple[str, ...] = tuple(
    method for method in EXTENDED_METHODS if method != "oslom"
)
PAPER_OVERLAP_PARAMS: dict[str, Any] = {
    "codeseg_tau": 0.3,
    "codeseg_alpha": 1.0,
    "codeseg_iterations": 10,
    "codeseg_threads": 1,
    "slpa_iterations": 21,
    "slpa_threshold": 0.01,
    "bigclam_communities": 25_000,
    "fox_wcc_threshold": 0.1,
    "der_walk_len": 3,
    "der_threshold": 1e-5,
    "der_iter_bound": 50,
    "ground_truth_min_size": 3,
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    paper_implementation: str
    graph_semantics: str
    parameters: dict[str, Any]


METHOD_SPECS: dict[str, MethodSpec] = {
    "codeseg": MethodSpec(
        "codeseg",
        "structural-entropy game / overlapping",
        "SELGroup/CoDeSEG C++ executable",
        "directed graph preserved; undirected graphs unchanged",
        {
            "tau": PAPER_OVERLAP_PARAMS["codeseg_tau"],
            "alpha": PAPER_OVERLAP_PARAMS["codeseg_alpha"],
            "max_iterations": PAPER_OVERLAP_PARAMS["codeseg_iterations"],
            "overlapping": True,
        },
    ),
    "slpa": MethodSpec(
        "slpa",
        "label propagation / overlapping",
        "cdlib.algorithms.slpa in the locked isolated worker",
        "undirected, unweighted NetworkX conversion",
        {
            "iterations": PAPER_OVERLAP_PARAMS["slpa_iterations"],
            "threshold": PAPER_OVERLAP_PARAMS["slpa_threshold"],
        },
    ),
    "bigclam": MethodSpec(
        "bigclam",
        "nonnegative matrix factorization / overlapping",
        "SNAP Bigclam executable",
        "undirected, unweighted edge list",
        {"communities": PAPER_OVERLAP_PARAMS["bigclam_communities"]},
    ),
    "ncgame": MethodSpec(
        "ncgame",
        "non-cooperative game / overlapping",
        "NcGame reference implementation from the CoDeSEG repository",
        "undirected, unweighted graph",
        {"stop_criterion": 0.01, "similarity": "HP"},
    ),
    "fox": MethodSpec(
        "fox",
        "triangle heuristic / overlapping",
        "LazyFox executable",
        "undirected, unweighted edge list",
        {"wcc_threshold": PAPER_OVERLAP_PARAMS["fox_wcc_threshold"]},
    ),
    "louvain": MethodSpec(
        "louvain",
        "modularity / disjoint",
        "igraph.Graph.community_multilevel",
        "undirected, unweighted graph",
        {"resolution": 1.0},
    ),
    "der": MethodSpec(
        "der",
        "diffusion entropy reduction / disjoint",
        "cdlib.algorithms.der in the locked isolated worker",
        "undirected, unweighted NetworkX conversion",
        {
            "walk_len": PAPER_OVERLAP_PARAMS["der_walk_len"],
            "threshold": PAPER_OVERLAP_PARAMS["der_threshold"],
            "iter_bound": PAPER_OVERLAP_PARAMS["der_iter_bound"],
        },
    ),
    "leiden": MethodSpec(
        "leiden",
        "modularity / disjoint",
        "leidenalg.ModularityVertexPartition (igraph binding)",
        "undirected, unweighted graph",
        {"objective_function": "modularity", "n_iterations": 2},
    ),
    "flpa": MethodSpec(
        "flpa",
        "fast label propagation / disjoint",
        "CoDeSEG repository FLPA_impt.py → igraph label propagation",
        "undirected, unweighted graph",
        {"upstream_function": "community_label_propagation"},
    ),
    "community_hedonic": MethodSpec(
        "community_hedonic",
        "hedonic game / overlapping local comparator",
        "hedonic.Game.community_hedonic",
        "undirected, unweighted graph with edge-density resolution",
        {
            "resolution": "graph.density()",
            "initialization": (
                "randomized ground-truth cover preserving community count and "
                "per-vertex multiplicities; uncovered vertices assigned once"
            ),
            "allow_isolation": False,
            "local_move_only": False,
            "n_iterations": -1,
            "max_communities": "ground_truth_max_memberships_per_vertex",
            "max_memberships": "ground_truth_max_memberships_per_vertex",
        },
    ),
    "hedonic_local": MethodSpec(
        "hedonic_local",
        "hedonic game / local-moving overlap",
        "hedonic.Game.community_hedonic (lucas-igraph)",
        "undirected, unweighted graph with edge-density resolution",
        {"resolution": "graph.density()", "local_move_only": True, "n_iterations": -1},
    ),
    "hedonic_multiphase": MethodSpec(
        "hedonic_multiphase",
        "hedonic game / Leiden multi-phase overlap",
        "hedonic.Game.community_hedonic (lucas-igraph)",
        "undirected, unweighted graph with edge-density resolution",
        {"resolution": "graph.density()", "local_move_only": False, "n_iterations": -1},
    ),
    "hedonic_multiphase_x10": MethodSpec(
        "hedonic_multiphase_x10",
        "hedonic game / multi-phase resolution ×10",
        "hedonic.Game.community_hedonic (lucas-igraph)",
        "undirected, unweighted graph with density-scaled resolution",
        {"resolution": "min(graph.density() * 10, 1)", "local_move_only": False, "n_iterations": -1},
    ),
    "hedonic_multiphase_x100": MethodSpec(
        "hedonic_multiphase_x100",
        "hedonic game / multi-phase resolution ×100",
        "hedonic.Game.community_hedonic (lucas-igraph)",
        "undirected, unweighted graph with density-scaled resolution",
        {"resolution": "min(graph.density() * 100, 1)", "local_move_only": False, "n_iterations": -1},
    ),
    "angel": MethodSpec(
        "angel",
        "ego-network / overlapping",
        "cdlib.algorithms.angel in the locked isolated worker",
        "undirected, unweighted NetworkX conversion",
        {"threshold": 0.6, "min_community_size": 3},
    ),
    "infomap": MethodSpec(
        "infomap",
        "flow-based / disjoint control",
        "lucas-igraph Graph.community_infomap",
        "undirected, unweighted graph",
        {"implementation": "native Infomap in lucas-igraph"},
    ),
    "demon": MethodSpec(
        "demon",
        "ego-network label propagation / overlapping",
        "demon.Demon 2.0.6",
        "undirected, unweighted NetworkX conversion",
        {"epsilon": 0.25, "min_community_size": 2},
    ),
    "cpm": MethodSpec(
        "cpm",
        "clique percolation / overlapping",
        "networkx.algorithms.community.k_clique_communities",
        "undirected, unweighted NetworkX conversion",
        {"clique_size": 3},
    ),
    "link_clustering": MethodSpec(
        "link_clustering",
        "edge-based / overlapping",
        "cdlib.algorithms.hierarchical_link_community in the locked isolated worker",
        "undirected, unweighted NetworkX conversion",
        {"output_conversion": "edge communities projected to node sets"},
    ),
    "oslom": MethodSpec(
        "oslom",
        "statistical significance / overlapping",
        "OSLOM2 oslom_undir executable",
        "undirected, unweighted edge list",
        {"fast": True, "min_community_size": 2},
    ),
    "neo_kmeans": MethodSpec(
        "neo_kmeans",
        "graph k-means / overlapping",
        "NEO-K-Means Graph Clustering executable",
        "undirected, unweighted METIS graph",
        {"alpha": 0.2, "beta": 0.0, "sigma": 0.0, "clusters": 64},
    ),
    "nise": MethodSpec(
        "nise",
        "seed-set expansion / overlapping",
        "official NISE MATLAB/Octave source",
        "undirected, unweighted sparse adjacency",
        {"seeding": "sphub", "ego": True, "expansion": "ppr", "workers": 1},
    ),
    "sse": MethodSpec(
        "sse",
        "seed-set expansion / overlapping",
        "official NISE/SSE seed-expansion source",
        "undirected, unweighted sparse adjacency",
        {"seeding": "sphub", "ego": True, "expansion": "ppr", "workers": 1},
    ),
    "qoce": MethodSpec(
        "qoce",
        "quadratic optimization clique expansion / overlapping",
        "PanShi2016/QOCE MATLAB/Octave source",
        "undirected, unweighted sparse adjacency plus maximal cliques",
        {"nthread": 1, "t0": 3, "mu": 0.0, "alpha": 0.2, "w": 5},
    ),
    "svi": MethodSpec(
        "svi",
        "stochastic variational inference / overlapping",
        "premgopalan/svinet executable",
        "undirected, unweighted edge list",
        {"communities": 64, "max_iterations": 10, "link_sampling": True},
    ),
    "essc": MethodSpec(
        "essc",
        "statistical significance extraction / overlapping",
        "jdwilson4/ESSC R package",
        "undirected, unweighted sparse Matrix adjacency",
        {"alpha": 0.1, "null": "Poisson", "samples": "all"},
    ),
}


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _resource_snapshot() -> dict[str, float | int]:
    """Capture process and child resource counters for one detector run.

    ``ru_maxrss`` is reported in bytes on macOS and KiB on Linux.  Converting
    here keeps the JSON/CSV ledger portable while avoiding a hard dependency on
    psutil.  The peak values are cumulative OS counters; the record stores both
    before/after snapshots so callers can distinguish a pre-existing peak from
    a detector's newly observed peak.
    """
    factor = 1 if sys.platform == "darwin" else 1024
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "user_cpu_seconds": float(own.ru_utime),
        "system_cpu_seconds": float(own.ru_stime),
        "child_user_cpu_seconds": float(children.ru_utime),
        "child_system_cpu_seconds": float(children.ru_stime),
        "peak_rss_bytes": int(own.ru_maxrss * factor),
        "peak_child_rss_bytes": int(children.ru_maxrss * factor),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_csv(value: str | None, allowed: Sequence[str], option: str) -> list[str] | None:
    if value is None:
        return None
    if value.strip().lower() == "all":
        return list(allowed)
    selected = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(
            f"unknown {option}: {', '.join(unknown)}; choose from {', '.join(allowed)}"
        )
    if not selected:
        raise ValueError(f"{option} must contain at least one value")
    return list(dict.fromkeys(selected))


def _cover_for_paper_metrics(dataset: SnapDataset) -> tuple[list[list[int]], set[int], dict[str, Any]]:
    minimum_size = 1 if dataset.name == "wikipedia" else int(PAPER_OVERLAP_PARAMS["ground_truth_min_size"])
    ground_truth, validation = canonicalize_cover(
        dataset.cover, n_vertices=dataset.graph.vcount(), minimum_size=minimum_size
    )
    nodes = {vertex for community in ground_truth for vertex in community}
    return ground_truth, nodes, {
        "minimum_community_size": minimum_size,
        "communities_before_filter": len(dataset.cover),
        "communities_after_filter": len(ground_truth),
        "covered_nodes": len(nodes),
        "canonicalization": validation,
        "prediction_filter": "intersect_each_predicted_community_with_gt_nodes",
    }


def _filter_prediction(
    cover: Sequence[Sequence[int]],
    *,
    gt_nodes: set[int],
    n_vertices: int,
) -> list[list[int]]:
    filtered = [
        [int(vertex) for vertex in community if int(vertex) in gt_nodes]
        for community in cover
    ]
    canonical, _ = canonicalize_cover(
        filtered, n_vertices=n_vertices, minimum_size=1
    )
    return canonical


def _undirected_graph(graph: ig.Graph) -> ig.Graph:
    result = graph.copy()
    if result.is_directed():
        result.to_undirected(mode="collapse")
    result.simplify(multiple=True, loops=True, combine_edges=None)
    return result


def _active_graph(graph: ig.Graph) -> tuple[ig.Graph, list[int]]:
    """Compact the edge-supported vertex universe for node-indexed detectors.

    SNAP edge lists identify only vertices that occur in an edge.  Cached
    igraph graphs may retain the gaps between those original IDs as isolated
    vertices (for example, Amazon has 548,552 cached slots but 334,863 edge
    endpoints).  The upstream NetworkX/igraph baselines never create those
    gaps, so compact them for detectors that consume contiguous node indices.
    The returned map converts detector-local IDs back to the cached graph IDs.
    """
    active = [
        vertex for vertex, degree in enumerate(graph.degree()) if int(degree) > 0
    ]
    if len(active) == graph.vcount():
        return graph, list(range(graph.vcount()))
    return graph.induced_subgraph(active), active


def _partition_cover(partition: Any) -> list[list[int]]:
    membership = getattr(partition, "membership", None)
    if membership is None:
        return [list(map(int, community)) for community in partition]
    communities: dict[int, list[int]] = {}
    for vertex, label in enumerate(membership):
        communities.setdefault(int(label), []).append(int(vertex))
    return list(communities.values())


def _cover_from_partition_or_cover(result: Any) -> list[list[int]]:
    """Convert either an igraph partition or a nested VertexCover result."""
    membership = getattr(result, "membership", None)
    if membership is None:
        return [list(map(int, community)) for community in result]
    if membership and isinstance(membership[0], (list, tuple)):
        communities: dict[int, list[int]] = {}
        for vertex, labels in enumerate(membership):
            for label in labels:
                communities.setdefault(int(label), []).append(int(vertex))
        return [communities[label] for label in sorted(communities)]
    return _partition_cover(result)


def _randomized_cover_initialization(
    ground_truth: Sequence[Sequence[int]],
    n_vertices: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Randomize a GT cover while retaining its community count and cap.

    The detector requires one non-empty row per vertex.  Ground-truth-covered
    vertices retain their original number of memberships, uncovered vertices
    receive one random label, and degree-preserving two-community swaps then
    randomize which vertices carry each label. This keeps the number of
    communities and the maximum per-vertex overlap explicit without passing a
    semantically misleading disjoint ``max_communities`` initialization.
    """
    if n_vertices < 1:
        raise ValueError("community_hedonic requires at least one vertex")
    if not ground_truth:
        raise ValueError("community_hedonic requires a nonempty ground-truth cover")
    community_sets = [
        {int(vertex) for vertex in community if 0 <= int(vertex) < n_vertices}
        for community in ground_truth
    ]
    if any(not members for members in community_sets):
        raise ValueError("ground-truth cover contains an empty community")
    n_communities = len(community_sets)
    rows = [set() for _ in range(n_vertices)]
    for label, members in enumerate(community_sets):
        for vertex in members:
            rows[vertex].add(label)

    rng = random.Random(int(seed))
    uncovered = [vertex for vertex, labels in enumerate(rows) if not labels]
    for vertex in uncovered:
        label = rng.randrange(n_communities)
        rows[vertex].add(label)
        community_sets[label].add(vertex)

    incidence_count = sum(len(labels) for labels in rows)
    target_swaps = min(200_000, max(1_000, 8 * incidence_count))
    accepted_swaps = 0
    if n_communities > 1:
        for _ in range(target_swaps):
            first, second = rng.sample(range(n_communities), 2)
            if not community_sets[first] or not community_sets[second]:
                continue
            first_vertex = rng.choice(tuple(community_sets[first]))
            second_vertex = rng.choice(tuple(community_sets[second]))
            if (
                first_vertex == second_vertex
                or second in rows[first_vertex]
                or first in rows[second_vertex]
            ):
                continue
            community_sets[first].remove(first_vertex)
            community_sets[first].add(second_vertex)
            community_sets[second].remove(second_vertex)
            community_sets[second].add(first_vertex)
            rows[first_vertex].remove(first)
            rows[first_vertex].add(second)
            rows[second_vertex].remove(second)
            rows[second_vertex].add(first)
            accepted_swaps += 1

    initialized = [sorted(labels) for labels in rows]
    return initialized, {
        "initial_community_count": n_communities,
        "ground_truth_community_count": len(ground_truth),
        "initial_incidence_count": incidence_count,
        "initial_uncovered_vertices_assigned": len(uncovered),
        "ground_truth_max_memberships_per_vertex": max(
            len({int(label) for label in labels})
            for labels in _cover_to_membership_rows(ground_truth, n_vertices)
        ),
        "initial_max_memberships_per_vertex": max(len(labels) for labels in initialized),
        "randomization_target_swaps": target_swaps,
        "randomization_accepted_swaps": accepted_swaps,
        "random_seed": int(seed),
    }


def _cover_to_membership_rows(
    cover: Sequence[Sequence[int]], n_vertices: int
) -> list[list[int]]:
    rows = [[] for _ in range(n_vertices)]
    for label, community in enumerate(cover):
        for vertex in community:
            vertex = int(vertex)
            if 0 <= vertex < n_vertices and label not in rows[vertex]:
                rows[vertex].append(label)
    return rows


def _hedonic_adapter_cap(
    parameters: dict[str, Any],
    ground_truth: Sequence[Sequence[int]],
    n_vertices: int,
) -> tuple[int, str]:
    """Return the Hedonic adapter cap and the provenance of its value.

    The default is the pre-registered constant
    :data:`HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS`.  An explicit integer is a
    user-selected cap.  :data:`HEDONIC_REFERENCE_MULTIPLICITY_CAP` computes the
    largest number of reference communities that contain one vertex (at least
    two, so the overlapping path is selected) and must be reported as a
    reference-informed ablation.
    """
    if "max_memberships" not in parameters:
        return HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS, "reference_free_registered_default"
    requested = parameters["max_memberships"]
    if requested == HEDONIC_REFERENCE_MULTIPLICITY_CAP:
        multiplicity = max(
            (len(row) for row in _cover_to_membership_rows(ground_truth, n_vertices)),
            default=0,
        )
        return max(2, int(multiplicity)), "reference_max_multiplicity_ablation"
    cap = int(requested)
    if cap < 2:
        raise ValueError("Hedonic overlapping adapters require max_memberships >= 2")
    return cap, "user_specified"


def _run_community_hedonic(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    initial_membership, initialization = _randomized_cover_initialization(
        ground_truth, graph.vcount(), seed
    )
    max_memberships = int(initialization["ground_truth_max_memberships_per_vertex"])
    resolution = float(graph.density())
    game = Game(graph)
    started = time.perf_counter()
    ig.set_random_number_generator(random.Random(int(seed)))
    result = game.community_hedonic(
        initial_membership=initial_membership,
        # ``max_communities`` is a disjoint-init parameter in Game.  Passing
        # the requested overlap cap keeps the call self-documenting, while
        # ``max_memberships`` is the argument that actually enforces it for a
        # supplied overlapping initialization.
        max_communities=max_memberships,
        max_memberships=max_memberships,
        n_iterations=int(parameters.get("n_iterations", -1)),
        resolution=resolution,
        allow_isolation=False,
        local_move_only=False,
        seed=int(seed),
        beta=float(parameters.get("beta", 0.01)),
    )
    cover = _cover_from_partition_or_cover(result)
    return cover, {
        "implementation": "hedonic.Game.community_hedonic",
        "seed": int(seed),
        "stochastic": True,
        "igraph_rng": "random.Random(seed)",
        "resolution": resolution,
        "resolution_source": "network edge density",
        "allow_isolation": False,
        "local_move_only": False,
        "n_iterations": int(parameters.get("n_iterations", -1)),
        "max_communities_argument": max_memberships,
        "max_memberships": max_memberships,
        "runtime_seconds": time.perf_counter() - started,
        "initialization": initialization,
        "returned_community_count": len(cover),
        "returned_max_memberships_per_vertex": max(
            (len(labels) for labels in getattr(result, "membership", [])),
            default=0,
        ),
    }


def _run_louvain(graph: ig.Graph, seed: int, _parameters: dict[str, Any]) -> tuple[list[list[int]], dict[str, Any]]:
    ig.set_random_number_generator(random.Random(int(seed)))
    result = graph.community_multilevel()
    return _partition_cover(result), {
        "implementation": "igraph.Graph.community_multilevel",
        "seed": int(seed),
        "stochastic": True,
    }


def _run_leiden(graph: ig.Graph, seed: int, parameters: dict[str, Any]) -> tuple[list[list[int]], dict[str, Any]]:
    ig.set_random_number_generator(random.Random(int(seed)))
    result = graph.community_leiden(
        objective_function="modularity",
        n_iterations=int(parameters.get("n_iterations", 2)),
        beta=0.01,
        resolution=1.0,
    )
    return _partition_cover(result), {
        "implementation": "igraph.Graph.community_leiden(objective_function='modularity')",
        "seed": int(seed),
        "n_iterations": int(parameters.get("n_iterations", 2)),
        "stochastic": True,
    }


def _run_flpa(graph: ig.Graph, seed: int, _parameters: dict[str, Any]) -> tuple[list[list[int]], dict[str, Any]]:
    ig.set_random_number_generator(random.Random(int(seed)))
    result = graph.community_label_propagation()
    return _partition_cover(result), {
        "implementation": "igraph.Graph.community_label_propagation",
        "seed": int(seed),
        "stochastic": True,
        "upstream_method_name": "FLPA_impt.py:LPA",
    }


def _last_json_line(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise MethodUnavailable("isolated baseline worker emitted no JSON result")


def _isolated_worker_python() -> tuple[Path, dict[str, Any]]:
    receipt = slpa_environment_receipt()
    if not receipt.get("ready"):
        raise MethodUnavailable("locked CDlib environment is unavailable")
    root = Path(receipt["root"])
    worker = root / "der_worker.py"
    if not worker.is_file():
        raise MethodUnavailable(f"DER worker is missing: {worker}")
    python = receipt.get("project_python")
    if not python:
        raise MethodUnavailable("materialized locked CDlib interpreter is unavailable")
    return Path(str(python)), receipt


def _run_der(
    graph: ig.Graph, seed: int, parameters: dict[str, Any]
) -> tuple[list[list[int]], dict[str, Any]]:
    python, receipt = _isolated_worker_python()
    # CDlib's reference DER implementation asserts that every detector node
    # has positive degree.  The full SNAP graphs satisfy that precondition, but
    # a GT-informed bounded induced subgraph can contain vertices whose edges
    # fall outside the bound.  Drop those vertices for DER only and map the
    # returned communities back to the original graph IDs; on a full graph
    # this is a no-op and preserves the paper protocol exactly.
    active_vertices = [
        vertex for vertex, degree in enumerate(graph.degree()) if int(degree) > 0
    ]
    active_index = {vertex: index for index, vertex in enumerate(active_vertices)}
    request = {
        "n": len(active_vertices),
        "edges": [
            [active_index[int(left)], active_index[int(right)]]
            for left, right in graph.get_edgelist()
            if int(left) in active_index and int(right) in active_index
        ],
        "walk_len": int(parameters.get("walk_len", 3)),
        "threshold": float(parameters.get("threshold", 1e-5)),
        "iter_bound": int(parameters.get("iter_bound", 50)),
    }
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "hedonic-codeseg-matplotlib"))
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [str(python), str(Path(receipt["root"]) / "der_worker.py")],
            input=json.dumps(request, separators=(",", ":")),
            text=True,
            capture_output=True,
            check=False,
            timeout=parameters.get("timeout_seconds"),
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MethodUnavailable(f"isolated DER worker could not complete: {exc}") from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().splitlines()
        raise MethodUnavailable(
            f"isolated DER worker exited {process.returncode}: "
            + (detail[-1] if detail else "no diagnostic")
        )
    payload = _last_json_line(process.stdout)
    communities = payload.get("communities")
    metadata = payload.get("metadata")
    if not isinstance(communities, list) or not isinstance(metadata, dict):
        raise MethodUnavailable("isolated DER worker returned an invalid payload")
    expected = {"cdlib": "0.4.0", "networkx": "3.6.1", "numpy": "2.3.3", "python-igraph": "1.0.0"}
    mismatches = {
        name: (metadata.get(f"{name.replace('-', '_')}_version"), version)
        for name, version in expected.items()
        if metadata.get(f"{name.replace('-', '_')}_version") != version
    }
    if mismatches:
        raise MethodUnavailable(f"DER worker dependency drift: {mismatches}")
    mapped_communities = [
        [active_vertices[int(vertex)] for vertex in community]
        for community in communities
    ]
    return mapped_communities, {
        **metadata,
        "implementation": "cdlib.algorithms.der",
        "worker_source": str(Path(receipt["root"]) / "der_worker.py"),
        "worker_source_sha256": _sha256_file(Path(receipt["root"]) / "der_worker.py"),
        "worker_seconds": time.perf_counter() - started,
        "seed": int(seed),
        "stochastic": False,
        "detector_graph_nodes": len(active_vertices),
        "isolated_vertices_dropped": int(graph.vcount() - len(active_vertices)),
    }


def _write_edge_list(graph: ig.Graph, path: Path, *, snap_header: bool = False) -> None:
    with path.open("w", encoding="utf-8") as stream:
        if snap_header:
            direction = "Directed" if graph.is_directed() else "Undirected"
            stream.write(f"# {direction} graph: CoDeSEG local reproduction\n")
            stream.write("# CoDeSEG local reproduction\n")
            stream.write(f"# Nodes: {graph.vcount()} Edges: {graph.ecount()}\n")
            stream.write("# FromNodeId\tToNodeId\n")
        for source, target in graph.get_edgelist():
            stream.write(f"{int(source)}\t{int(target)}\n")


def _write_metis_graph(graph: ig.Graph, path: Path) -> None:
    """Write the 1-based adjacency format consumed by NEO-K-Means."""
    undirected = _undirected_graph(graph)
    adjacency: list[list[int]] = [[] for _ in range(undirected.vcount())]
    for source, target in undirected.get_edgelist():
        if int(source) == int(target):
            continue
        adjacency[int(source)].append(int(target) + 1)
        adjacency[int(target)].append(int(source) + 1)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{undirected.vcount()} {undirected.ecount()}\n")
        for neighbors in adjacency:
            stream.write(" ".join(str(value) for value in sorted(set(neighbors))) + "\n")


def _run_oslom(
    graph: ig.Graph,
    _ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_OSLOM_BIN")
    if not binary:
        raise MethodUnavailable(
            "OSLOM executable is unavailable; pass --oslom-bin or set HEDONIC_OSLOM_BIN"
        )
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"OSLOM executable is not runnable: {binary_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-oslom-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        _write_edge_list(graph, edge_path)
        # OSLOM creates the output directory itself, then invokes a legacy
        # ``rm path/*`` cleanup.  On an empty directory that wildcard is
        # passed literally and the binary aborts.  Put a tiny ``rm -f`` shim
        # first on PATH so the cleanup remains harmless without pre-creating
        # OSLOM's directory (which would make its mkdir call fail instead).
        tool_dir = root / "tools"
        tool_dir.mkdir()
        rm_shim = tool_dir / "rm"
        rm_shim.write_text("#!/bin/sh\nexec /bin/rm -f \"$@\"\n", encoding="utf-8")
        rm_shim.chmod(0o755)
        command = [
            str(binary_path),
            "-f",
            str(edge_path),
            "-uw",
            "-seed",
            str(int(seed)),
        ]
        if bool(parameters.get("fast", True)):
            command.append("-fast")
        try:
            environment = os.environ.copy()
            environment["PATH"] = str(tool_dir) + os.pathsep + environment.get("PATH", "")
            result = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                env=environment,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"OSLOM command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"OSLOM exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        output_path = root / "graph.txt_oslo_files" / "tp"
        if not output_path.is_file():
            raise MethodUnavailable("OSLOM completed without graph.txt_oslo_files/tp")
        communities: list[list[int]] = []
        for line in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values = [int(token) for token in line.split()]
            except ValueError:
                continue
            if values:
                communities.append(values)
        if not communities:
            raise MethodUnavailable("OSLOM output contains no integer communities")
        return communities, {
            "implementation": "OSLOM2 oslom_undir executable",
            "binary": str(binary_path),
            "binary_sha256": _sha256_file(binary_path),
            "command": command,
            "stdout_tail": result.stdout[-2000:],
            "seed": int(seed),
            "stochastic": True,
            "output": "graph.txt_oslo_files/tp",
        }


def _run_neo_kmeans(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    _seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_NEO_BIN")
    if not binary:
        raise MethodUnavailable(
            "NEO-K-Means executable is unavailable; pass --neo-bin or set HEDONIC_NEO_BIN"
        )
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"NEO-K-Means executable is not runnable: {binary_path}")
    requested_clusters = int(parameters.get("clusters", 64))
    effective_clusters = max(2, min(requested_clusters, max(2, graph.vcount() - 1)))
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-neo-") as temporary:
        root = Path(temporary)
        graph_path = root / "graph.metis"
        _write_metis_graph(graph, graph_path)
        command = [
            str(binary_path),
            "-a",
            str(float(parameters.get("alpha", 0.2))),
            "-b",
            str(float(parameters.get("beta", 0.0))),
            "-s",
            str(float(parameters.get("sigma", 0.0))),
            str(graph_path),
            str(effective_clusters),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"NEO-K-Means command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"NEO-K-Means exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        # The legacy writer preserves the input suffix, producing
        # ``graph.metis_clust_...`` rather than ``graph_clust_...``.
        candidates = sorted(root.glob("graph*clust_*"))
        if not candidates:
            raise MethodUnavailable("NEO-K-Means completed without a graph_clust output")
        communities_by_node: list[list[int]] = []
        for line in candidates[-1].read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                communities_by_node.append([int(token) for token in line.split()])
            except ValueError:
                communities_by_node.append([])
        if len(communities_by_node) != graph.vcount():
            raise MethodUnavailable(
                f"NEO-K-Means returned {len(communities_by_node)} node rows for {graph.vcount()} nodes"
            )
        communities: dict[int, list[int]] = {}
        for node, labels in enumerate(communities_by_node):
            for label in labels:
                communities.setdefault(int(label), []).append(node)
        cover = [members for _, members in sorted(communities.items()) if members]
        if not cover:
            raise MethodUnavailable("NEO-K-Means output contains no memberships")
        return cover, {
            "implementation": "NEO-K-Means Graph Clustering executable",
            "binary": str(binary_path),
            "binary_sha256": _sha256_file(binary_path),
            "command": command,
            "requested_clusters": requested_clusters,
            "effective_clusters": effective_clusters,
            "stdout_tail": result.stdout[-2000:],
            "seed": None,
            "stochastic": True,
            "output": candidates[-1].name,
            "ground_truth_community_count": len(ground_truth),
        }


def _run_octave_script(
    script: str,
    *,
    root: Path,
    octave: str,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    script_path = root / "run_method.m"
    script_path.write_text(script, encoding="utf-8")
    try:
        return subprocess.run(
            [str(octave), "--no-gui", "--quiet", "--eval", f"run('{script_path.as_posix()}')"],
            cwd=str(root), capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MethodUnavailable(f"Octave method could not complete: {exc}") from exc


def _octave_output_or_error(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        useful = [line for line in detail if "ignoring const execution_exception" not in line]
        if useful:
            detail = useful
        errors = [line for line in detail if "error:" in line.lower()]
        if errors:
            detail = [errors[0]]
        raise MethodUnavailable(
            f"{label} Octave command exited {result.returncode}: "
            + (detail[-1] if detail else "no diagnostic")
        )


def _run_nise(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    _seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    octave = parameters.get("octave") or shutil.which("octave-cli") or shutil.which("octave")
    source = parameters.get("source")
    if not octave or not source:
        raise MethodUnavailable("official NISE/SSE source or octave-cli is unavailable")
    source_path = Path(str(source)).expanduser()
    if (source_path / "src" / "nise.m").is_file():
        source_path = source_path / "src"
    if not (source_path / "nise.m").is_file():
        raise MethodUnavailable(f"NISE source is missing nise.m: {source_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-nise-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        output_path = root / "communities.txt"
        _write_edge_list(graph, edge_path)
        # The official archive bundles Linux-only matlab_bgl MEX files for
        # connected/biconnected components.  These small Octave-compatible
        # fallbacks preserve NISE's seed-expansion path on macOS; the PPR
        # growth MEX files remain the authors' compiled code.
        (root / "components.m").write_text(
            "function [ci,sizes]=components(A,varargin)\n"
            "n=size(A,1); ci=zeros(n,1); sizes=[]; c=0;\n"
            "for s=1:n, if ci(s)==0, c=c+1; q=s; ci(s)=c; h=1;\n"
            "while h<=numel(q), v=q(h); h=h+1; nb=find(A(v,:)|A(:,v)');\n"
            "for u=nb(:)', if ci(u)==0, ci(u)=c; q(end+1)=u; end; end; end; sizes(c)=sum(ci==c); end; end\n",
            encoding="utf-8",
        )
        (root / "biconncore.m").write_text(
            "function [Abc,p,Af,fcc]=biconncore(A,varargin)\n"
            "Abc=sparse(A); Af=Abc; p=true(size(A,1),1); fcc=ones(size(A,1),1);\nend\n",
            encoding="utf-8",
        )
        requested_k = int(parameters.get("communities", max(2, min(64, len(ground_truth) or 64))))
        k = max(2, min(requested_k, max(2, graph.vcount() - 1)))
        script = f"""
addpath('{root.as_posix()}'); addpath(genpath('{source_path.as_posix()}'));
E = dlmread('{edge_path.as_posix()}');
n = {int(graph.vcount())};
if isempty(E), A = sparse(n,n); else, A = sparse([E(:,1);E(:,2)]+1,[E(:,2);E(:,1)]+1,1,n,n); end;
C = nise(A,{k},'{str(parameters.get('seeding', 'sphub'))}',{'true' if parameters.get('ego', True) else 'false'},'{str(parameters.get('expansion', 'ppr'))}',{int(parameters.get('nworkers', 1))});
fid = fopen('{output_path.as_posix()}','w');
for j = 1:size(C,2), I = find(C(:,j)>0)-1; if ~isempty(I), fprintf(fid,'%d ',I); fprintf(fid,'\\n'); end; end;
fclose(fid);
"""
        result = _run_octave_script(script, root=root, octave=str(octave), timeout=parameters.get("timeout_seconds"))
        _octave_output_or_error(result, "NISE")
        cover = _read_cover_lines(output_path)
        return cover, {
            "implementation": "official NISE MATLAB/Octave source",
            "source": str(source_path),
            "octave": str(octave),
            "communities": k,
            "seed_strategy": "sphub",
            "expansion": "ppr",
            "seed": int(_seed),
            "stochastic": False,
            "ground_truth_community_count": len(ground_truth),
        }


def _run_qoce(
    graph: ig.Graph,
    _ground_truth: Sequence[Sequence[int]],
    _seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    octave = parameters.get("octave") or shutil.which("octave-cli") or shutil.which("octave")
    source = parameters.get("source")
    clique_finder = parameters.get("clique_finder")
    if not octave or not source or not clique_finder:
        raise MethodUnavailable("official QOCE source, CliqueFinder, or octave-cli is unavailable")
    source_path = Path(str(source)).expanduser()
    qoce_codes = source_path / "QOCE_codes"
    if not (qoce_codes / "QOCE.m").is_file():
        raise MethodUnavailable(f"QOCE source is missing QOCE.m: {qoce_codes}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-qoce-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.pairs"
        seed_path = root / "seeds.txt"
        output_path = root / "communities.txt"
        # QOCE's loader expects 1-based MATLAB node identifiers.
        with edge_path.open("w", encoding="utf-8") as stream:
            for left, right in _undirected_graph(graph).get_edgelist():
                if int(left) != int(right):
                    stream.write(f"{int(left)+1} {int(right)+1}\n")
        try:
            clique_result = subprocess.run(
                [str(clique_finder), str(edge_path), str(int(parameters.get("min_clique_size", 3)))],
                cwd=str(root), capture_output=True, text=True,
                timeout=parameters.get("timeout_seconds"), check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"QOCE clique enumeration could not complete: {exc}") from exc
        if clique_result.returncode != 0:
            raise MethodUnavailable("QOCE CliqueFinder failed: " + (clique_result.stderr or clique_result.stdout)[-500:])
        seed_path.write_text(clique_result.stdout, encoding="utf-8")
        max_seeds = parameters.get("max_seeds")
        if max_seeds is not None:
            lines = [line for line in seed_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            seed_path.write_text("\n".join(lines[: int(max_seeds)]) + ("\n" if lines[: int(max_seeds)] else ""), encoding="utf-8")
        # Octave has no parpool/quadprog by default. These tiny compatibility
        # shims preserve the upstream QOCE control flow and use Octave's native
        # qp solver for the same constrained quadratic program.
        (root / "parpool.m").write_text("function p=parpool(varargin), p=[]; end\n", encoding="utf-8")
        (root / "gcp.m").write_text("function p=gcp(varargin), p=[]; end\n", encoding="utf-8")
        (root / "quadprog.m").write_text(
            "function x=quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options)\n"
            "if nargin<9 || isempty(x0), x0=max(lb(:),zeros(size(f(:)))); end\n"
            "[x,~,info]=qp(x0,H,f,Aeq,beq,lb,ub);\n"
            "if info.info>2, x=[]; end\nend\n", encoding="utf-8"
        )
        (root / "optimoptions.m").write_text(
            "function o=optimoptions(varargin), o=struct(); end\n", encoding="utf-8"
        )
        script = f"""
addpath('{root.as_posix()}'); addpath(genpath('{qoce_codes.as_posix()}'));
G = loadGraph('{edge_path.as_posix()}');
S = loadCommunities('{seed_path.as_posix()}');
C = QOCE(G,S,{int(parameters.get('nthread',1))},{int(parameters.get('t0',3))},{float(parameters.get('mu',0.0))},{float(parameters.get('alpha',0.2))},{int(parameters.get('w',5))});
saveCommunities('{output_path.as_posix()}',C);
"""
        result = _run_octave_script(script, root=root, octave=str(octave), timeout=parameters.get("timeout_seconds"))
        _octave_output_or_error(result, "QOCE")
        return _read_cover_lines(output_path), {
            "implementation": "PanShi2016/QOCE MATLAB/Octave source",
            "source": str(qoce_codes),
            "clique_finder": str(clique_finder),
            "octave": str(octave),
            "seed": int(_seed),
            "stochastic": False,
            "octave_compatibility": "parpool/gcp no-op; quadprog mapped to qp",
            "max_seeds": max_seeds,
        }


def _run_svi(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_SVI_BIN")
    if not binary:
        raise MethodUnavailable("SVI executable is unavailable; pass --svi-bin or set HEDONIC_SVI_BIN")
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"SVI executable is not runnable: {binary_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-svi-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        _write_edge_list(graph, edge_path)
        requested_k = int(parameters.get("communities", max(2, min(64, len(ground_truth) or 64))))
        k = max(2, min(requested_k, max(2, graph.vcount() - 1)))
        command = [str(binary_path), "-file", str(edge_path), "-n", str(graph.vcount()), "-k", str(k), "-label", "svi", "-seed", str(seed), "-max-iterations", str(int(parameters.get("max_iterations", 10))), "-nthreads", str(int(parameters.get("threads", 1)))]
        if parameters.get("link_sampling", True): command.append("-link-sampling")
        try:
            result = subprocess.run(command, cwd=str(root), capture_output=True, text=True, timeout=parameters.get("timeout_seconds"), check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"SVI command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable("SVI exited %s: %s" % (result.returncode, detail[-1] if detail else "no diagnostic"))
        candidates = sorted(root.glob("**/communities.txt"))
        if not candidates:
            raise MethodUnavailable("SVI completed without communities.txt")
        cover = _read_cover_lines(candidates[-1])
        return cover, {"implementation": "premgopalan/svinet", "binary": str(binary_path), "binary_sha256": _sha256_file(binary_path), "command": command, "requested_communities": requested_k, "communities": k, "seed": int(seed), "stochastic": True, "output": str(candidates[-1])}


def _run_essc(
    graph: ig.Graph,
    _ground_truth: Sequence[Sequence[int]],
    _seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    rscript = parameters.get("rscript") or shutil.which("Rscript")
    library = parameters.get("library")
    if not rscript or not library:
        raise MethodUnavailable("ESSC R package or Rscript is unavailable")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-essc-") as temporary:
        root = Path(temporary); edge_path = root / "graph.txt"; output_path = root / "communities.txt"
        _write_edge_list(graph, edge_path)
        script = root / "run.R"
        script.write_text(
            "args <- commandArgs(trailingOnly=TRUE); lib <- args[[1]]; edge <- args[[2]]; out <- args[[3]]; n <- as.integer(args[[4]]); "
            ".libPaths(c(lib, .libPaths())); suppressPackageStartupMessages(library(Matrix)); suppressPackageStartupMessages(library(ESSC)); "
            "E <- tryCatch(read.table(edge, header=FALSE), error=function(e) matrix(numeric(0), ncol=2)); "
            "if (nrow(E)>0) A <- sparseMatrix(i=c(E[,1],E[,2])+1,j=c(E[,2],E[,1])+1,x=1,dims=c(n,n)) else A <- sparseMatrix(n,n); "
            "R <- essc(A, alpha=0.1, Null='Poisson', Num.Samples=n); f <- file(out,'w'); "
            "for (cc in R$Communities) { if(length(cc)>0) { writeLines(paste(cc-1, collapse=' '), f) } }; close(f)",
            encoding="utf-8",
        )
        try:
            result = subprocess.run([str(rscript), str(script), str(library), str(edge_path), str(output_path), str(graph.vcount())], cwd=str(root), capture_output=True, text=True, timeout=parameters.get("timeout_seconds"), check=False, env={**os.environ, "R_LIBS_USER": str(library)})
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"ESSC R command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable("ESSC exited %s: %s" % (result.returncode, detail[-1] if detail else "no diagnostic"))
        return _read_cover_lines(output_path), {"implementation": "jdwilson4/ESSC R package", "rscript": str(rscript), "library": str(library), "seed": int(_seed), "stochastic": True, "null": "Poisson", "alpha": 0.1}


def _write_cover(cover: Sequence[Sequence[int]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for community in cover:
            stream.write("\t".join(str(int(vertex)) for vertex in community) + "\n")


def _read_cover_lines(path: Path) -> list[list[int]]:
    if not path.is_file():
        raise MethodUnavailable(f"detector did not create an output cover: {path}")
    communities: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            communities.append([int(token) for token in line.replace(",", " ").split()])
        except ValueError:
            # External tools often write a short diagnostic/header beside the
            # community file.  A line with no all-integer payload is ignored;
            # a completely empty parsed cover still fails closed below.
            continue
    if not communities:
        raise MethodUnavailable(f"detector output contains no integer communities: {path}")
    return communities


def _command_template(name: str, parameters: dict[str, Any]) -> str | None:
    configured = parameters.get("command")
    if configured:
        return str(configured)
    return os.environ.get(f"HEDONIC_CODESEG_{name.upper()}_COMMAND")


def _run_template_command(
    name: str,
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    template = _command_template(name, parameters)
    if not template:
        raise MethodUnavailable(
            f"{name} requires an explicit command via --{name}-command or "
            f"HEDONIC_CODESEG_{name.upper()}_COMMAND"
        )
    with tempfile.TemporaryDirectory(prefix=f"hedonic-codeseg-{name}-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        truth_path = root / "ground_truth.txt"
        output_path = root / "communities.txt"
        # The upstream NcGame script uses pandas.read_csv(..., skiprows=4),
        # so its command template receives a conventional four-line SNAP
        # header.  Other adapters use the dedicated plain edge-list writer.
        _write_edge_list(graph, edge_path, snap_header=name == "ncgame")
        _write_cover(ground_truth, truth_path)
        values = {
            "input": str(edge_path),
            "output": str(output_path),
            "ground_truth": str(truth_path),
            "dataset": parameters.get("dataset", "dataset"),
            "seed": int(seed),
            "nodes": int(graph.vcount()),
            "edges": int(graph.ecount()),
        }
        try:
            command = [part.format(**values) for part in shlex.split(template)]
        except (KeyError, ValueError) as exc:
            raise MethodUnavailable(f"invalid {name} command template: {exc}") from exc
        try:
            result = subprocess.run(
                command,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"{name} command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"{name} command exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        candidate = output_path if output_path.is_file() else None
        if candidate is None:
            output_file = parameters.get("output_file")
            if output_file:
                candidate = root / str(output_file)
        if candidate is None or not candidate.is_file():
            # A command may print communities to stdout when no output file is
            # part of its native API.  Keep that fallback explicit.
            candidate = root / "stdout.txt"
            candidate.write_text(result.stdout, encoding="utf-8")
        cover = _read_cover_lines(candidate)
        return cover, {
            "implementation": parameters.get("implementation", name),
            "command": command,
            "stdout_tail": result.stdout[-2000:],
            "seed": int(seed),
            "stochastic": True,
        }


def _run_codeseg(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_CODESEG_BIN")
    if not binary:
        detected = shutil.which("CoDeSEG") or shutil.which("codeseg")
        binary = detected
    if not binary:
        raise MethodUnavailable(
            "CoDeSEG executable is unavailable; pass --codeseg-bin or set HEDONIC_CODESEG_BIN"
        )
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"CoDeSEG executable is not runnable: {binary_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-codeseg-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        truth_path = root / "ground_truth.txt"
        output_path = root / "communities.txt"
        _write_edge_list(graph, edge_path)
        _write_cover(ground_truth, truth_path)
        command = [
            str(binary_path),
            "-i",
            str(edge_path),
            "-o",
            str(output_path),
            "-t",
            str(truth_path),
            "-n",
            str(int(parameters.get("iterations", PAPER_OVERLAP_PARAMS["codeseg_iterations"]))),
            "-m",
            str(int(parameters.get("minimum_nodes", 10))),
            "-e",
            str(float(parameters.get("tau", PAPER_OVERLAP_PARAMS["codeseg_tau"]))),
            "-a",
            str(float(parameters.get("alpha", PAPER_OVERLAP_PARAMS["codeseg_alpha"]))),
            "-p",
            str(int(parameters.get("threads", PAPER_OVERLAP_PARAMS["codeseg_threads"]))),
            "-x",
        ]
        if graph.is_directed():
            command.append("-d")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"CoDeSEG command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"CoDeSEG exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        return _read_cover_lines(output_path), {
            "implementation": "SELGroup/CoDeSEG C++ executable",
            "binary": str(binary_path),
            "binary_sha256": _sha256_file(binary_path),
            "command": command,
            "stdout_tail": result.stdout[-2000:],
            "seed": int(seed),
            "stochastic": False,
            "ground_truth_passed_to_upstream": True,
        }


def _run_bigclam(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_BIGCLAM_BIN")
    if not binary:
        raise MethodUnavailable(
            "Bigclam executable is unavailable; pass --bigclam-bin or set HEDONIC_BIGCLAM_BIN"
        )
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"Bigclam executable is not runnable: {binary_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-bigclam-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        truth_path = root / "ground_truth.txt"
        output_prefix = root / "result-"
        _write_edge_list(graph, edge_path)
        _write_cover(ground_truth, truth_path)
        # SNAP's published Bigclam example uses colon-prefixed options.  Keep
        # that exact interface as the default while allowing a command template
        # for forks with a different CLI.
        template = _command_template("bigclam", parameters)
        if template:
            return _run_template_command("bigclam", graph, ground_truth, seed, parameters)
        requested_communities = int(
            parameters.get("communities", PAPER_OVERLAP_PARAMS["bigclam_communities"])
        )
        effective_communities = requested_communities
        if parameters.get("smoke"):
            # The paper's 25,000-factor setting is faithful for SNAP runs but
            # needlessly non-terminating on the six-node smoke fixture.
            effective_communities = max(1, min(requested_communities, graph.vcount()))
        command = [
            str(binary_path),
            f"-i:{edge_path}",
            f"-o:{output_prefix}",
            f"-d:{parameters.get('dataset', 'dataset')}",
            f"-c:{effective_communities}",
            f"-nt:{int(parameters.get('threads', 1))}",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"Bigclam command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"Bigclam exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        candidates = sorted(root.glob("*cmtyvv.txt")) + sorted(root.glob("*.cmtyvv"))
        if not candidates:
            raise MethodUnavailable("Bigclam completed without a cmtyvv output file")
        return _read_cover_lines(candidates[-1]), {
            "implementation": "SNAP Bigclam executable",
            "binary": str(binary_path),
            "binary_sha256": _sha256_file(binary_path),
            "command": command,
            "requested_communities": requested_communities,
            "effective_communities": effective_communities,
            "smoke_override": bool(parameters.get("smoke")),
            "stdout_tail": result.stdout[-2000:],
            "seed": int(seed),
            "stochastic": True,
        }


def _run_fox(
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    binary = parameters.get("binary") or os.environ.get("HEDONIC_FOX_BIN")
    if not binary:
        raise MethodUnavailable(
            "LazyFox executable is unavailable; pass --fox-bin or set HEDONIC_FOX_BIN"
        )
    binary_path = Path(str(binary)).expanduser()
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise MethodUnavailable(f"LazyFox executable is not runnable: {binary_path}")
    with tempfile.TemporaryDirectory(prefix="hedonic-codeseg-fox-") as temporary:
        root = Path(temporary)
        edge_path = root / "graph.txt"
        output_dir = root / "output"
        _write_edge_list(graph, edge_path)
        command = [
            str(binary_path),
            "--input-graph",
            str(edge_path),
            "--output-dir",
            str(output_dir),
            "--queue-size",
            str(int(parameters.get("queue_size", 1))),
            "--thread-count",
            str(int(parameters.get("threads", 1))),
            "--wcc-threshold",
            str(float(parameters.get("wcc_threshold", PAPER_OVERLAP_PARAMS["fox_wcc_threshold"]))),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=parameters.get("timeout_seconds"),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise MethodUnavailable(f"LazyFox command could not complete: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise MethodUnavailable(
                f"LazyFox exited {result.returncode}: "
                + (detail[-1] if detail else "no diagnostic")
            )
        candidates = sorted(output_dir.glob("**/*clusters.txt"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise MethodUnavailable("LazyFox completed without a *clusters.txt output file")
        return _read_cover_lines(candidates[-1]), {
            "implementation": "LazyFox executable",
            "binary": str(binary_path),
            "binary_sha256": _sha256_file(binary_path),
            "command": command,
            "stdout_tail": result.stdout[-2000:],
            "seed": int(seed),
            "stochastic": True,
        }


def _run_method(
    name: str,
    graph: ig.Graph,
    ground_truth: Sequence[Sequence[int]],
    seed: int,
    parameters: dict[str, Any],
) -> tuple[list[list[int]], dict[str, Any]]:
    name = canonical_method_name(name)
    method_graph = graph if name == "codeseg" else _undirected_graph(graph)
    detector_map = list(range(method_graph.vcount()))
    if name in {"slpa", "louvain", "der", "leiden", "flpa", "oslom", "neo_kmeans", "nise", "sse", "qoce", "svi", "essc"}:
        method_graph, detector_map = _active_graph(method_graph)
    if name in {"nise", "sse"} and method_graph.vcount() > 0:
        components = method_graph.connected_components(mode="weak")
        if len(components) > 1:
            giant = max(components, key=len)
            detector_map = [detector_map[int(vertex)] for vertex in giant]
            method_graph = method_graph.induced_subgraph(giant)
    if name == "codeseg":
        cover, metadata = _run_codeseg(method_graph, ground_truth, seed, parameters)
    elif name == "slpa":
        cover, metadata = run_slpa(
            method_graph,
            iterations=int(parameters.get("iterations", PAPER_OVERLAP_PARAMS["slpa_iterations"])),
            threshold=float(parameters.get("threshold", PAPER_OVERLAP_PARAMS["slpa_threshold"])),
            seed=int(seed),
            allow_replica=False,
            timeout_seconds=float(parameters.get("timeout_seconds", 90.0)),
        )
    elif name == "bigclam":
        cover, metadata = _run_bigclam(method_graph, ground_truth, seed, parameters)
    elif name == "ncgame":
        cover, metadata = _run_template_command("ncgame", method_graph, ground_truth, seed, parameters)
    elif name == "fox":
        cover, metadata = _run_fox(method_graph, ground_truth, seed, parameters)
    elif name == "oslom":
        cover, metadata = _run_oslom(method_graph, ground_truth, seed, parameters)
    elif name == "neo_kmeans":
        cover, metadata = _run_neo_kmeans(method_graph, ground_truth, seed, parameters)
    elif name in {"nise", "sse"}:
        cover, metadata = _run_nise(method_graph, ground_truth, seed, parameters)
        metadata = {**metadata, "method_alias": name}
    elif name == "qoce":
        cover, metadata = _run_qoce(method_graph, ground_truth, seed, parameters)
    elif name == "svi":
        cover, metadata = _run_svi(method_graph, ground_truth, seed, parameters)
    elif name == "essc":
        cover, metadata = _run_essc(method_graph, ground_truth, seed, parameters)
    elif name == "louvain":
        cover, metadata = _run_louvain(method_graph, seed, parameters)
    elif name == "der":
        cover, metadata = _run_der(method_graph, seed, parameters)
    elif name == "leiden":
        cover, metadata = _run_leiden(method_graph, seed, parameters)
    elif name == "flpa":
        cover, metadata = _run_flpa(method_graph, seed, parameters)
    elif name == "community_hedonic":
        cover, metadata = _run_community_hedonic(
            method_graph, ground_truth, seed, parameters
        )
    elif name in {
        "hedonic_local",
        "hedonic_multiphase",
        "hedonic_multiphase_x10",
        "hedonic_multiphase_x100",
        "angel",
        "infomap",
        "demon",
        "cpm",
        "link_clustering",
    }:
        # Only the Hedonic adapters use the cap; the other adapters ignore it
        # and receive the reference-free constant so they never touch the
        # reference cover through this argument.
        if name.startswith("hedonic_"):
            cap, cap_source = _hedonic_adapter_cap(parameters, ground_truth, graph.vcount())
        else:
            cap, cap_source = HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS, None
        adapter_parameters = {
            key: value for key, value in parameters.items() if key != "max_memberships"
        }
        try:
            adapter = OVERLAPPING_METHODS[name]
            cover, metadata = run_overlapping_method(
                adapter,
                method_graph,
                max_memberships=cap,
                resolution=float(method_graph.density()),
                seed=int(seed),
                parameters=adapter_parameters,
            )
        except AdapterMethodUnavailable as exc:
            raise MethodUnavailable(str(exc)) from exc
        if cap_source is not None:
            metadata = {
                **metadata,
                "max_memberships": int(cap),
                "max_memberships_source": cap_source,
            }
    else:
        raise ValueError(f"unsupported CoDeSEG method: {name}")
    if (
        len(detector_map) != graph.vcount()
        or detector_map != list(range(len(detector_map)))
    ):
        cover = [
            [detector_map[int(vertex)] for vertex in community]
            for community in cover
        ]
        metadata = {
            **metadata,
            "detector_graph_nodes": int(len(detector_map)),
            "detector_graph_isolate_slots_dropped": int(graph.vcount() - len(detector_map)),
        }
    return cover, metadata


def _method_preflight(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    name = canonical_method_name(name)
    spec = METHOD_SPECS[name]
    if name in {
        "louvain",
        "leiden",
        "flpa",
        "community_hedonic",
        "hedonic_local",
        "hedonic_multiphase",
        "hedonic_multiphase_x10",
        "hedonic_multiphase_x100",
    }:
        return {"status": "ready", "reason": None, "spec": spec.__dict__}
    if name == "slpa" or name == "der":
        try:
            _isolated_worker_python()
        except MethodUnavailable as exc:
            return {"status": "unavailable", "reason": str(exc), "spec": spec.__dict__}
        return {"status": "ready", "reason": None, "spec": spec.__dict__}
    if name in {"angel", "link_clustering", "infomap", "demon", "cpm"}:
        try:
            from hedonic.experiments.overlapping.methods import method_availability

            availability = method_availability().get(name, {})
        except Exception as exc:
            return {"status": "unavailable", "reason": str(exc), "spec": spec.__dict__}
        return {
            "status": "ready" if availability.get("available") else "unavailable",
            "reason": availability.get("reason"),
            "dependency": availability.get("dependency"),
            "spec": spec.__dict__,
        }
    if name == "codeseg":
        binary = parameters.get("binary") or os.environ.get("HEDONIC_CODESEG_BIN")
        if not binary:
            binary = shutil.which("CoDeSEG") or shutil.which("codeseg")
        ready = bool(
            binary
            and Path(str(binary)).expanduser().is_file()
            and os.access(Path(str(binary)).expanduser(), os.X_OK)
        )
        return {
            "status": "ready" if ready else "unavailable",
            "reason": None if ready else "missing CoDeSEG binary",
            "binary": str(binary) if binary else None,
            "spec": spec.__dict__,
        }
    if name == "bigclam":
        binary = parameters.get("binary") or os.environ.get("HEDONIC_BIGCLAM_BIN")
        ready = bool(
            binary
            and Path(str(binary)).expanduser().is_file()
            and os.access(Path(str(binary)).expanduser(), os.X_OK)
        )
        return {
            "status": "ready" if ready else "unavailable",
            "reason": None if ready else "missing Bigclam binary",
            "binary": str(binary) if binary else None,
            "spec": spec.__dict__,
        }
    if name in {"fox", "oslom", "neo_kmeans"}:
        binary = parameters.get("binary") or os.environ.get("HEDONIC_FOX_BIN")
        if name == "oslom":
            binary = parameters.get("binary") or os.environ.get("HEDONIC_OSLOM_BIN")
        elif name == "neo_kmeans":
            binary = parameters.get("binary") or os.environ.get("HEDONIC_NEO_BIN")
        ready = bool(
            binary
            and Path(str(binary)).expanduser().is_file()
            and os.access(Path(str(binary)).expanduser(), os.X_OK)
        )
        return {
            "status": "ready" if ready else "unavailable",
            "reason": None if ready else f"missing {name} binary",
            "binary": str(binary) if binary else None,
            "spec": spec.__dict__,
        }
    if name in {"nise", "sse", "qoce"}:
        octave = parameters.get("octave") or shutil.which("octave-cli") or shutil.which("octave")
        source = parameters.get("source")
        ready = bool(octave and source and Path(str(source)).expanduser().exists())
        return {"status": "ready" if ready else "unavailable", "reason": None if ready else f"missing {name} source or Octave", "octave": octave, "source": source, "spec": spec.__dict__}
    if name == "svi":
        binary = parameters.get("binary") or os.environ.get("HEDONIC_SVI_BIN")
        ready = bool(binary and Path(str(binary)).expanduser().is_file() and os.access(Path(str(binary)).expanduser(), os.X_OK))
        return {"status": "ready" if ready else "unavailable", "reason": None if ready else "missing SVI binary", "binary": str(binary) if binary else None, "spec": spec.__dict__}
    if name == "essc":
        rscript = parameters.get("rscript") or shutil.which("Rscript")
        library = parameters.get("library")
        ready = bool(rscript and library and Path(str(library)).expanduser().exists())
        return {"status": "ready" if ready else "unavailable", "reason": None if ready else "missing ESSC R library or Rscript", "rscript": rscript, "library": library, "spec": spec.__dict__}
    template = _command_template(name, parameters)
    return {
        "status": "configured" if template else "unavailable",
        "reason": None if template else f"missing {name} command template",
        "command": template,
        "spec": spec.__dict__,
    }


def _load_dataset(
    name: str,
    *,
    cover_variant: str,
    network_root: Path,
    max_nodes: int | None,
    smoke: bool,
    smoke_nodes: int = 6,
    smoke_seed: int = 0,
    snap_cache_dir: Path | None = None,
    allow_catalog: bool = True,
    max_download_bytes: int | None = None,
) -> SnapDataset:
    if smoke:
        if name == SMOKE_DATASET:
            dataset = synthetic_agmfit_dataset(
                n_nodes=smoke_nodes,
                seed=smoke_seed,
            )
        else:
            dataset = smoke_dataset(name, cover_variant="all")
    else:
        dataset = load_snap_dataset(
            name,
            cover_variant=cover_variant,
            data_root=network_root,
            cache_dir=snap_cache_dir,
            allow_catalog=allow_catalog,
            max_download_bytes=max_download_bytes,
        )
    if max_nodes is not None and max_nodes > 0:
        dataset = bounded_induced_dataset(dataset, max_nodes)
    return dataset


def _run_dataset(
    dataset: SnapDataset,
    *,
    methods: Sequence[str],
    seed: int,
    method_parameters: dict[str, dict[str, Any]],
    timeout_seconds: float | None,
    paper_filter: bool,
    output_dir: Path,
    resume: bool,
    config_digest: str,
    compute_omega: bool = True,
    omega_sample_size: int = 100_000,
    save_covers: bool = False,
) -> list[dict[str, Any]]:
    ground_truth, gt_nodes, gt_report = _cover_for_paper_metrics(dataset)
    records: list[dict[str, Any]] = []
    for name in methods:
        run_path = output_dir / "runs" / dataset.name / f"{name}.json"
        if resume and run_path.is_file():
            try:
                cached = json.loads(run_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                cached = None
            if (
                isinstance(cached, dict)
                and cached.get("config_digest") == config_digest
                and cached.get("status") in {"completed", "unavailable", "failed"}
            ):
                records.append(cached)
                print(f"[resume] {dataset.name}/{name}: {cached.get('status', 'unknown')}")
                continue
        parameters = dict(method_parameters.get(name, {}))
        parameters["dataset"] = dataset.name
        if timeout_seconds is not None:
            parameters["timeout_seconds"] = float(timeout_seconds)
        started = time.perf_counter()
        resource_before = _resource_snapshot()
        record: dict[str, Any] = {
            "protocol_version": PROTOCOL_VERSION,
            "config_digest": config_digest,
            "dataset": dataset.name,
            "method": name,
            "seed": int(seed),
            "status": "running",
            "dataset_report": dataset.report,
            "ground_truth_report": gt_report,
            "method_spec": METHOD_SPECS[name].__dict__,
            "paper_filter_enabled": bool(paper_filter),
        }
        try:
            detection_started = time.perf_counter()
            cover, detector_metadata = _run_method(
                name, dataset.graph, ground_truth, int(seed), parameters
            )
            record["detection_seconds"] = time.perf_counter() - detection_started
            scoring_started = time.perf_counter()
            scored_prediction = (
                _filter_prediction(cover, gt_nodes=gt_nodes, n_vertices=dataset.graph.vcount())
                if paper_filter
                else canonicalize_cover(cover, n_vertices=dataset.graph.vcount(), minimum_size=1)[0]
            )
            scored_ground_truth = ground_truth if paper_filter else dataset.cover
            accuracy = evaluate_cover(
                scored_prediction,
                scored_ground_truth,
                dataset.graph.vcount(),
                compute_omega=compute_omega,
                omega_sample_size=omega_sample_size,
                omega_seed=seed,
            )
            f1 = float(accuracy["f1"])
            onmi = overlapping_normalized_mutual_information_lfk(
                scored_prediction, scored_ground_truth
            )
            accuracy.update(
                {
                    "onmi": float(onmi),
                    "onmi_percent": 100.0 * float(onmi),
                    "f1_percent": 100.0 * float(f1),
                }
            )
            record["scoring_seconds"] = time.perf_counter() - scoring_started
            if save_covers:  # the detector's cover (analysis-graph vertex ids), for later rescoring
                cover_path = output_dir / "covers" / dataset.name / f"{name}.json.gz"
                cover_path.parent.mkdir(parents=True, exist_ok=True)
                import gzip

                with gzip.open(cover_path, "wt", encoding="utf-8") as handle:
                    json.dump([sorted(int(v) for v in c) for c in cover], handle)
                record["cover_path"] = str(cover_path.relative_to(output_dir))
            record.update(
                {
                    "status": "completed",
                    "detector": detector_metadata,
                    "prediction_community_count": len(scored_prediction),
                    "metrics": accuracy,
                    "scored_ground_truth_community_count": len(scored_ground_truth),
                    "scored_ground_truth_node_count": len(gt_nodes),
                }
            )
        except MethodUnavailable as exc:
            reason = str(exc)
            lowered_reason = reason.lower()
            execution_failure = any(
                marker in lowered_reason
                for marker in (
                    "timed out",
                    "exited",
                    "could not complete",
                    "returned non-zero",
                    "failed",
                )
            )
            record.update(
                {
                    "status": "failed" if execution_failure else "unavailable",
                    "reason": reason,
                }
            )
        except Exception as exc:  # preserve the failure in the resumable ledger
            record.update(
                {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        resource_after = _resource_snapshot()
        record["runtime_seconds"] = time.perf_counter() - started
        record["resources"] = {
            "before": resource_before,
            "after": resource_after,
            "user_cpu_seconds": max(
                0.0,
                float(resource_after["user_cpu_seconds"])
                - float(resource_before["user_cpu_seconds"]),
            ),
            "system_cpu_seconds": max(
                0.0,
                float(resource_after["system_cpu_seconds"])
                - float(resource_before["system_cpu_seconds"]),
            ),
            "child_user_cpu_seconds": max(
                0.0,
                float(resource_after["child_user_cpu_seconds"])
                - float(resource_before["child_user_cpu_seconds"]),
            ),
            "child_system_cpu_seconds": max(
                0.0,
                float(resource_after["child_system_cpu_seconds"])
                - float(resource_before["child_system_cpu_seconds"]),
            ),
            "peak_rss_bytes": int(resource_after["peak_rss_bytes"]),
            "peak_child_rss_bytes": int(resource_after["peak_child_rss_bytes"]),
            "vertices_per_second": (
                float(dataset.graph.vcount()) / record["runtime_seconds"]
                if record["runtime_seconds"] > 0
                else None
            ),
            "edges_per_second": (
                float(dataset.graph.ecount()) / record["runtime_seconds"]
                if record["runtime_seconds"] > 0
                else None
            ),
        }
        _atomic_json(run_path, record)
        records.append(record)
        status = record["status"]
        metric_text = ""
        if status == "completed":
            metric_text = " " + json.dumps(record["metrics"], sort_keys=True)
        print(f"[{status}] {dataset.name}/{name}{metric_text}")
    return records


def _write_csv(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "method",
        "status",
        "seed",
        "runtime_seconds",
        "f1",
        "f1_percent",
        "onmi",
        "onmi_percent",
        "jaccard",
        "matching_precision",
        "matching_recall",
        "matching_f1",
        "matching_mean_weight",
        "node_micro_precision",
        "node_micro_recall",
        "node_micro_f1",
        "node_macro_f1",
        "size_weighted_community_f1",
        "omega",
        "predicted_vertices_covered_fraction",
        "gt_vertices_covered_fraction",
        "inclusion_rate",
        "coverage_rate",
        "overlapping_rate",
        "distribution_rate",
        "prediction_community_count",
        "scored_ground_truth_community_count",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "child_user_cpu_seconds",
        "child_system_cpu_seconds",
        "peak_rss_bytes",
        "peak_child_rss_bytes",
        "vertices_per_second",
        "edges_per_second",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics") or {}
            resources = record.get("resources") or {}
            writer.writerow(
                {
                    column: record.get(
                        column,
                        metrics.get(column, resources.get(column, "")),
                    )
                    for column in columns
                }
            )


def _table_marks(records: Sequence[dict[str, Any]], key: str) -> dict[str, str]:
    """Return paper-style marks for the best and second-best completed rows."""
    values: dict[str, float] = {}
    for record in records:
        metrics = record.get("metrics") or {}
        value = record.get(key, metrics.get(key))
        if record.get("status") == "completed" and value is not None:
            values[str(record.get("method"))] = float(value)
    distinct = sorted(set(values.values()), reverse=True)
    marks: dict[str, str] = {}
    for method, value in values.items():
        if distinct and value == distinct[0]:
            marks[method] = "best"
        elif len(distinct) > 1 and value == distinct[1]:
            marks[method] = "second"
    return marks


def _write_table2(path: Path, records: Sequence[dict[str, Any]], methods: Sequence[str]) -> None:
    """Write a compact Table-2-style DBLP report in Markdown and LaTeX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(str(record.get("dataset", "dataset")), []).append(record)
    markdown: list[str] = [
        "# Table 2-style overlapping-community results",
        "",
        "Percentages; bold is best and underlining is second-best among completed methods.",
        "",
    ]
    latex: list[str] = [
        r"% Generated by hedonic-exp overlapping-codeseg",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & ONMI (\%) & F1 (\%) \\",
        r"\midrule",
    ]
    for dataset, dataset_records in by_dataset.items():
        indexed = {str(record.get("method")): record for record in dataset_records}
        onmi_marks = _table_marks(dataset_records, "onmi")
        f1_marks = _table_marks(dataset_records, "f1")
        markdown.extend([f"## {dataset}", "", "| Method | ONMI (%) | F1 (%) |", "|---|---:|---:|"])
        latex.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{dataset}}}}} \\")
        for method in methods:
            record = indexed.get(method, {})
            status = record.get("status")
            if status != "completed":
                markdown.append(f"| {method} | N/A | N/A |")
                latex.append(rf"{method.replace('_', r'\_')} & N/A & N/A \\")
                continue
            metrics = record.get("metrics") or {}
            onmi_value = record.get("onmi")
            f1_value = record.get("f1")
            onmi = 100.0 * float(metrics["onmi"] if onmi_value is None else onmi_value)
            f1 = 100.0 * float(metrics["f1"] if f1_value is None else f1_value)
            onmi_text = f"{onmi:.2f}"
            f1_text = f"{f1:.2f}"
            if onmi_marks.get(method) == "best":
                onmi_text = f"**{onmi_text}**"
            elif onmi_marks.get(method) == "second":
                onmi_text = f"<u>{onmi_text}</u>"
            if f1_marks.get(method) == "best":
                f1_text = f"**{f1_text}**"
            elif f1_marks.get(method) == "second":
                f1_text = f"<u>{f1_text}</u>"
            markdown.append(f"| {method} | {onmi_text} | {f1_text} |")
            onmi_tex = rf"\textbf{{{onmi:.2f}}}" if onmi_marks.get(method) == "best" else rf"\underline{{{onmi:.2f}}}" if onmi_marks.get(method) == "second" else f"{onmi:.2f}"
            f1_tex = rf"\textbf{{{f1:.2f}}}" if f1_marks.get(method) == "best" else rf"\underline{{{f1:.2f}}}" if f1_marks.get(method) == "second" else f"{f1:.2f}"
            latex.append(rf"{method.replace('_', r'\_')} & {onmi_tex} & {f1_tex} \\")
        markdown.append("")
        latex.append(r"\midrule")
    if latex[-1] == r"\midrule":
        latex.pop()
    latex.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.with_suffix(".md").write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")
    path.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic-exp overlapping-codeseg",
        description=(
            "Reproduce the CoDeSEG WWW'25 nine-method SNAP overlap experiment "
            "plus native hedonic variants and local overlap baselines."
        ),
    )
    parser.add_argument("--datasets", default=",".join(DATASETS), help="comma-separated SNAP datasets")
    parser.add_argument(
        "--methods",
        default=",".join(DBLP_METHODS),
        help="comma-separated methods; default is the DBLP benchmark set (OSLOM is explicit-only)",
    )
    parser.add_argument("--cover", choices=("all", "top5000"), default="all")
    parser.add_argument("--network-root", default=str(NETWORKS_DIR))
    parser.add_argument("--snap-cache-dir", default=None, help="normalized SNAP cache; the optional MapEquation catalogue is used when local archives are absent")
    parser.add_argument("--auto-download", dest="auto_download", action="store_true", help="allow a preflight to fetch missing SNAP data")
    parser.add_argument("--no-auto-download", dest="auto_download", action="store_false", help="fail instead of lazily downloading a missing SNAP dataset")
    parser.add_argument("--max-download-bytes", type=int, default=None, help="optional bound passed to the SNAP catalogue downloader")
    parser.add_argument("--setup-manifest", default=None, help="codeseg-setup manifest used to resolve external runtimes")
    parser.add_argument("--output-dir", default=str(OVERLAPPING_ARTIFACTS_DIR / "codeseg"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=None, help="GT-informed induced-subgraph bound for local tests")
    parser.add_argument("--timeout", type=float, default=None, help="per-detector wall-clock timeout in seconds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="load/report data and dependencies without running detectors")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a built-in fixture instead of SNAP archives (six nodes unless --smoke-nodes is set)",
    )
    parser.add_argument(
        "--smoke-nodes",
        type=int,
        default=6,
        help="number of vertices for --datasets synthetic_agmfit in smoke mode",
    )
    parser.add_argument(
        "--omega-sample-size",
        type=int,
        default=100_000,
        help="sample size for the memory-safe Omega estimate",
    )
    parser.add_argument(
        "--no-omega",
        dest="compute_omega",
        action="store_false",
        help="skip sampled Omega while retaining the other accuracy metrics",
    )
    parser.add_argument("--require-all", action="store_true", help="return failure if any selected method is unavailable or fails")
    parser.add_argument("--no-paper-filter", dest="paper_filter", action="store_false", help="score raw covers instead of the paper's GT-node filtering")
    parser.set_defaults(paper_filter=True)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--list-datasets", action="store_true")
    parser.add_argument("--codeseg-bin", default=None)
    parser.add_argument("--bigclam-bin", default=None)
    parser.add_argument("--fox-bin", default=None)
    parser.add_argument("--oslom-bin", default=None)
    parser.add_argument("--neo-bin", default=None)
    parser.add_argument("--svi-bin", default=None)
    parser.add_argument("--ncgame-command", default=None, help="template; supports {input} {output} {ground_truth} {dataset} {seed}")
    parser.add_argument(
        "--hedonic-max-memberships",
        default=None,
        help=(
            "membership cap of the hedonic_local/hedonic_multiphase* adapters: an "
            f"integer >= 2, or {HEDONIC_REFERENCE_MULTIPLICITY_CAP!r} for the largest "
            "per-vertex multiplicity of the reference cover (a reference-informed "
            "ablation); default: the pre-registered reference-free value "
            f"{HEDONIC_REFERENCE_FREE_MAX_MEMBERSHIPS}"
        ),
    )
    parser.add_argument("--codeseg-threads", type=int, default=PAPER_OVERLAP_PARAMS["codeseg_threads"])
    parser.add_argument("--codeseg-iterations", type=int, default=PAPER_OVERLAP_PARAMS["codeseg_iterations"])
    parser.add_argument("--codeseg-tau", type=float, default=PAPER_OVERLAP_PARAMS["codeseg_tau"])
    parser.add_argument("--codeseg-alpha", type=float, default=PAPER_OVERLAP_PARAMS["codeseg_alpha"])
    parser.add_argument("--slpa-iterations", type=int, default=PAPER_OVERLAP_PARAMS["slpa_iterations"])
    parser.add_argument("--slpa-threshold", type=float, default=PAPER_OVERLAP_PARAMS["slpa_threshold"])
    parser.add_argument("--bigclam-communities", type=int, default=PAPER_OVERLAP_PARAMS["bigclam_communities"])
    parser.add_argument("--fox-wcc-threshold", type=float, default=PAPER_OVERLAP_PARAMS["fox_wcc_threshold"])
    parser.add_argument("--save-covers", action="store_true",
                        help="also write each detector's cover to OUTPUT_DIR/covers/<dataset>/<method>.json.gz")
    parser.add_argument(
        "--method-params",
        default=None,
        help="JSON object {method: {parameter: value}} merged over the per-method "
        "parameters (e.g. '{\"fox\": {\"threads\": 10}}'); used by screening sweeps",
    )
    parser.add_argument("--neo-clusters", type=int, default=64)
    parser.add_argument("--neo-alpha", type=float, default=0.2, help="NEO-K-Means overlap parameter alpha (paper DBLP setting: 40)")
    parser.add_argument("--neo-beta", type=float, default=0.0, help="NEO-K-Means outlier parameter beta (paper DBLP setting: 0.0001)")
    parser.add_argument("--nise-communities", type=int, default=64, help="NISE seed count k (paper DBLP setting: 25000)")
    parser.add_argument("--sse-communities", type=int, default=64, help="SSE seed count k (paper DBLP setting: 25000)")
    parser.add_argument("--svi-communities", type=int, default=64)
    parser.add_argument("--svi-max-iterations", type=int, default=10)
    parser.add_argument("--octave-bin", default=None)
    parser.add_argument("--nise-source", default=None)
    parser.add_argument("--qoce-source", default=None)
    parser.add_argument("--qoce-clique-finder", default=None)
    parser.add_argument("--qoce-max-seeds", type=int, default=None, help="optional bounded QOCE seed cap; auto-set for --max-nodes validation")
    parser.add_argument("--essc-rscript", default=None)
    parser.add_argument("--essc-library", default=None)
    parser.add_argument("--der-walk-len", type=int, default=PAPER_OVERLAP_PARAMS["der_walk_len"])
    parser.add_argument("--der-threshold", type=float, default=PAPER_OVERLAP_PARAMS["der_threshold"])
    parser.add_argument("--der-iter-bound", type=int, default=PAPER_OVERLAP_PARAMS["der_iter_bound"])
    parser.set_defaults(auto_download=None, compute_omega=True)
    return parser


def _method_parameters(
    args: argparse.Namespace,
    runtime_manifest: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    manifest_methods = (runtime_manifest or {}).get("methods", {})

    def manifest_value(method: str, key: str) -> Any:
        value = manifest_methods.get(method)
        return value.get(key) if isinstance(value, dict) else None

    requested_cap = getattr(args, "hedonic_max_memberships", None)
    if requested_cap is None:
        hedonic_cap: dict[str, Any] = {}
    elif requested_cap == HEDONIC_REFERENCE_MULTIPLICITY_CAP:
        hedonic_cap = {"max_memberships": HEDONIC_REFERENCE_MULTIPLICITY_CAP}
    else:
        try:
            cap_value = int(requested_cap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--hedonic-max-memberships must be an integer >= 2 or "
                f"{HEDONIC_REFERENCE_MULTIPLICITY_CAP!r}"
            ) from exc
        if cap_value < 2:
            raise ValueError("--hedonic-max-memberships must be >= 2")
        hedonic_cap = {"max_memberships": cap_value}

    return {
        "codeseg": {
            "binary": args.codeseg_bin
            or os.environ.get("HEDONIC_CODESEG_BIN")
            or manifest_value("codeseg", "path"),
            "threads": args.codeseg_threads,
            "iterations": args.codeseg_iterations,
            "tau": args.codeseg_tau,
            "alpha": args.codeseg_alpha,
        },
        "slpa": {"iterations": args.slpa_iterations, "threshold": args.slpa_threshold},
        "bigclam": {
            "binary": args.bigclam_bin
            or os.environ.get("HEDONIC_BIGCLAM_BIN")
            or manifest_value("bigclam", "path"),
            "communities": args.bigclam_communities,
            "smoke": bool(args.smoke),
        },
        "ncgame": {
            "command": args.ncgame_command
            or os.environ.get("HEDONIC_CODESEG_NCGAME_COMMAND")
            or manifest_value("ncgame", "command"),
        },
        "fox": {
            "binary": args.fox_bin
            or os.environ.get("HEDONIC_FOX_BIN")
            or manifest_value("fox", "path"),
            "wcc_threshold": args.fox_wcc_threshold,
        },
        "oslom": {
            "binary": args.oslom_bin
            or os.environ.get("HEDONIC_OSLOM_BIN")
            or manifest_value("oslom", "path"),
            "fast": True,
        },
        "neo_kmeans": {
            "binary": args.neo_bin
            or os.environ.get("HEDONIC_NEO_BIN")
            or manifest_value("neo_kmeans", "path"),
            "clusters": args.neo_clusters,
            "alpha": args.neo_alpha,
            "beta": args.neo_beta,
            "sigma": 0.0,
        },
        "svi": {
            "binary": args.svi_bin
            or os.environ.get("HEDONIC_SVI_BIN")
            or manifest_value("svi", "path")
            or manifest_value("svi", "binary"),
            "communities": args.svi_communities,
            "max_iterations": args.svi_max_iterations,
            "link_sampling": True,
        },
        "nise": {
            "source": args.nise_source
            or os.environ.get("HEDONIC_NISE_SOURCE")
            or manifest_value("nise", "path"),
            "octave": args.octave_bin or os.environ.get("HEDONIC_OCTAVE_BIN") or manifest_value("nise", "octave"),
            "communities": args.nise_communities,
        },
        "sse": {
            "source": args.nise_source
            or os.environ.get("HEDONIC_NISE_SOURCE")
            or manifest_value("sse", "path")
            or manifest_value("nise", "path"),
            "octave": args.octave_bin or os.environ.get("HEDONIC_OCTAVE_BIN") or manifest_value("sse", "octave") or manifest_value("nise", "octave"),
            "communities": args.sse_communities,
        },
        "qoce": {
            "source": args.qoce_source
            or os.environ.get("HEDONIC_QOCE_SOURCE")
            or manifest_value("qoce", "path")
            or manifest_value("upstream_qoce", "path"),
            "octave": args.octave_bin or os.environ.get("HEDONIC_OCTAVE_BIN") or manifest_value("qoce", "octave"),
            "clique_finder": args.qoce_clique_finder or os.environ.get("HEDONIC_QOCE_CLIQUE_FINDER") or manifest_value("qoce", "clique_finder"),
            "max_seeds": args.qoce_max_seeds,
        },
        "essc": {
            "rscript": args.essc_rscript or os.environ.get("HEDONIC_ESSC_RSCRIPT") or manifest_value("essc", "rscript"),
            "library": args.essc_library or os.environ.get("HEDONIC_ESSC_LIBRARY") or manifest_value("essc", "library"),
        },
        "der": {
            "walk_len": args.der_walk_len,
            "threshold": args.der_threshold,
            "iter_bound": args.der_iter_bound,
        },
        "louvain": {},
        "leiden": {},
        "flpa": {},
        "community_hedonic": {
            "allow_isolation": False,
            "local_move_only": False,
            "n_iterations": -1,
            "beta": 0.01,
        },
        "hedonic_local": dict(hedonic_cap),
        "hedonic_multiphase": dict(hedonic_cap),
        "hedonic_multiphase_x10": dict(hedonic_cap),
        "hedonic_multiphase_x100": dict(hedonic_cap),
        "angel": {"threshold": 0.6, "min_community_size": 3},
        "infomap": {},
        "demon": {"epsilon": 0.25, "min_community_size": 2},
        "cpm": {"clique_size": 3, "max_communities": 50_000},
        "link_clustering": {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.list_methods:
        for name in EXTENDED_METHODS:
            print(f"{name}\t{METHOD_SPECS[name].family}\t{METHOD_SPECS[name].paper_implementation}")
        return 0
    if args.list_datasets:
        for name in DATASETS:
            print(name)
        print(SMOKE_DATASET)
        return 0
    try:
        dataset_names = DATASETS + ((SMOKE_DATASET,) if args.smoke else ())
        datasets = _parse_csv(args.datasets, dataset_names, "--datasets") or list(DATASETS)
        canonical_methods = ",".join(
            canonical_method_name(part)
            for part in str(args.methods).split(",")
            if part.strip()
        )
        methods = (
            list(EXTENDED_METHODS)
            if canonical_methods in {"all", "literature"}
            else _parse_csv(canonical_methods, EXTENDED_METHODS, "--methods")
        ) or list(EXTENDED_METHODS)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.smoke_nodes < 6:
        parser.error("--smoke-nodes must be at least 6")
    if args.omega_sample_size <= 0:
        parser.error("--omega-sample-size must be positive")
    if SMOKE_DATASET in datasets and not args.smoke:
        parser.error(f"{SMOKE_DATASET!r} is only available with --smoke")

    network_root = expand_path(args.network_root)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest = None
    if args.setup_manifest:
        try:
            from hedonic.experiments.overlapping.codeseg_setup import load_setup_manifest

            runtime_manifest = load_setup_manifest(args.setup_manifest)
        except Exception:
            runtime_manifest = None
    else:
        try:
            from hedonic.experiments.overlapping.codeseg_setup import load_setup_manifest

            runtime_manifest = load_setup_manifest()
        except Exception:
            runtime_manifest = None
    if runtime_manifest:
        isolated_receipt = (
            runtime_manifest.get("methods", {})
            .get("slpa", {})
            .get("receipt", {})
        )
        isolated_root = isolated_receipt.get("root") if isinstance(isolated_receipt, dict) else None
        if isolated_root:
            # The isolated CDlib worker is selected by the setup manifest, not
            # by an ambient import in the lucas-igraph process.
            os.environ.setdefault("HEDONIC_SLPA_ENV", str(isolated_root))
    method_parameters = _method_parameters(args, runtime_manifest=runtime_manifest)
    if args.max_nodes is not None and args.qoce_max_seeds is None:
        # QOCE's official exhaustive clique expansion is the only adapter
        # whose per-seed quadratic programs can dominate a bounded smoke
        # run. Keep the full method for full-graph runs; bounded validation
        # records this deterministic seed cap in detector metadata.
        method_parameters["qoce"]["max_seeds"] = 256
    if args.method_params:
        overrides = json.loads(args.method_params)
        if not isinstance(overrides, dict):
            raise ValueError("--method-params must be a JSON object keyed by method name")
        for method_name, values in overrides.items():
            method_parameters.setdefault(canonical_method_name(method_name), {}).update(values)
    manifest_cache_dir = None
    if runtime_manifest:
        configuration = runtime_manifest.get("configuration", {})
        if isinstance(configuration, dict) and configuration.get("cache_dir"):
            manifest_cache_dir = expand_path(str(configuration["cache_dir"]))
    external_binaries = {}
    for method_name in ("codeseg", "bigclam", "fox"):
        configured = method_parameters[method_name].get("binary")
        binary_path = Path(str(configured)).expanduser() if configured else None
        external_binaries[method_name] = {
            "path": str(binary_path) if binary_path else None,
            "sha256": (
                _sha256_file(binary_path)
                if binary_path and binary_path.is_file()
                else None
            ),
        }
    config_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "datasets": datasets,
        "methods": methods,
        "cover": args.cover,
        "network_root": str(network_root.resolve()),
        "snap_cache_dir": str(
            expand_path(args.snap_cache_dir).resolve()
            if args.snap_cache_dir
            else manifest_cache_dir.resolve()
            if manifest_cache_dir
            else None
        ),
        "seed": int(args.seed),
        "max_nodes": args.max_nodes,
        "timeout_seconds": args.timeout,
        "paper_filter": bool(args.paper_filter),
        "smoke": bool(args.smoke),
        "smoke_nodes": int(args.smoke_nodes),
        "compute_omega": bool(args.compute_omega),
        "omega_sample_size": int(args.omega_sample_size),
        "auto_download": (
            bool(args.auto_download)
            if args.auto_download is not None
            else not bool(args.preflight)
        ),
        "parameters": method_parameters,
        "external_binaries": external_binaries,
        "setup_manifest": (
            str(expand_path(args.setup_manifest)) if args.setup_manifest else None
        ),
        "setup_manifest_schema": (
            runtime_manifest.get("schema") if runtime_manifest else None
        ),
    }
    config_digest = _sha256_json(config_payload)
    preflight = {
        "protocol_version": PROTOCOL_VERSION,
        "config": config_payload,
        "config_digest": config_digest,
        "methods": {name: _method_preflight(name, method_parameters[name]) for name in methods},
        "datasets": [],
    }
    loaded: dict[str, SnapDataset] = {}
    for name in datasets:
        try:
            dataset = _load_dataset(
                name,
                cover_variant=args.cover,
                network_root=network_root,
                max_nodes=args.max_nodes,
                smoke=bool(args.smoke),
                smoke_nodes=int(args.smoke_nodes),
                smoke_seed=int(args.seed),
                snap_cache_dir=(
                    expand_path(args.snap_cache_dir)
                    if args.snap_cache_dir
                    else manifest_cache_dir
                ),
                allow_catalog=(
                    bool(args.auto_download)
                    if args.auto_download is not None
                    else not bool(args.preflight)
                ),
                max_download_bytes=args.max_download_bytes,
            )
        except (SnapLoadError, UnsupportedCoverVariant, OSError) as exc:
            preflight["datasets"].append({"dataset": name, "status": "unavailable", "reason": str(exc)})
            print(f"[unavailable] dataset {name}: {exc}")
            continue
        loaded[name] = dataset
        preflight["datasets"].append({"dataset": name, "status": "ready", "report": dataset.report})
        print(
            f"[dataset] {name}: n={dataset.graph.vcount():,} m={dataset.graph.ecount():,} "
            f"communities={len(dataset.cover):,} directed={dataset.graph.is_directed()}"
        )
    _atomic_json(output_dir / "preflight.json", preflight)
    if args.preflight:
        return 0 if loaded or args.smoke else 1

    records: list[dict[str, Any]] = []
    for name in datasets:
        dataset = loaded.get(name)
        if dataset is None:
            continue
        records.extend(
            _run_dataset(
                dataset,
                methods=methods,
                seed=int(args.seed),
                method_parameters=method_parameters,
                timeout_seconds=args.timeout,
                paper_filter=bool(args.paper_filter),
                output_dir=output_dir,
                resume=bool(args.resume),
                config_digest=config_digest,
                compute_omega=bool(args.compute_omega),
                omega_sample_size=int(args.omega_sample_size),
                save_covers=bool(args.save_covers),
            )
        )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "config": config_payload,
        "config_digest": config_digest,
        "preflight": preflight,
        "records": records,
        "status_counts": {
            status: sum(1 for record in records if record.get("status") == status)
            for status in sorted({str(record.get("status")) for record in records})
        },
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    _write_csv(output_dir / "metrics.csv", records)
    _write_table2(output_dir / "table2", records, methods)
    completed = sum(record.get("status") == "completed" for record in records)
    unavailable = sum(record.get("status") == "unavailable" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    print(f"[summary] completed={completed} unavailable={unavailable} failed={failed} output={output_dir}")
    if args.require_all and (unavailable or failed or len(records) != len(datasets) * len(methods)):
        return 1
    return 0


__all__ = [
    "DATASETS",
    "SMOKE_DATASET",
    "METHODS",
    "METHOD_SPECS",
    "DBLP_METHODS",
    "PAPER_METHODS",
    "PROTOCOL_VERSION",
    "main",
]
