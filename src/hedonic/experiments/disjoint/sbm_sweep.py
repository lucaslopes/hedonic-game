"""SBM parameter sweep for disjoint hedonic community detection.

Migrated from tmp/hedonic/scripts/experiment.py, exp.py, config.py, and
save_exp_data.py so the CLI can reproduce the PHYSA V1020 synthetic grid.

V1020 (original pipeline)
  - max_n_nodes=1020, communities ∈ {2,3,4,5,6}
  - p_in ∈ {0.01…0.10}, difficulty ∈ {0.10…0.75}
  - noises ∈ {0.10,0.25,0.50,0.75,1.00}, 5 network seeds, 10 partition seeds
  - methods: GroundTruth, Mirror, OnePass, Spectral, Leiden, Hedonic
  - Leiden/Hedonic: 10 stochastic runs / cell (unique partitions kept)
  - layout: resultados/{n}C_{size}N/.../Network (NNN)/partition_MMM.json

Use ``--preset v1020`` for the full grid (very large) or ``--preset v1020-smoke``
for a tiny structural check. Prefer writing under a *new* root such as
``artifacts/disjoint/v1020`` rather than overwriting the archived V1020 folder.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import igraph as ig
import numpy as np
from igraph import compare_communities
from igraph.clustering import VertexClustering
from tqdm import tqdm

from hedonic import Game
from hedonic.experiments.config import (
    DISJOINT_ARTIFACTS_DIR,
    ensure_not_archived_v1020,
    expand_path,
)
from hedonic.experiments.disjoint import store as store_mod
from hedonic.experiments.leiden import LEIDEN_IDENTITY, leiden, seeded_igraph_rng

# ---------------------------------------------------------------------------
# Default method map: Leiden = stock igraph CPM Leiden, Hedonic = community_hedonic
# ---------------------------------------------------------------------------

METHODS = {
    "GroundTruth": {
        "method_call_name": "community_groundtruth",
        "parameters": {
            "groundtruth": None,
        },
    },
    "Mirror": {
        "method_call_name": "community_mirror",
        "parameters": {
            "initial_membership": None,
        },
    },
    "OnePass": {
        "method_call_name": "community_onepass_improvement",
        "parameters": {
            "initial_membership": None,
        },
    },
    "Spectral": {
        "method_call_name": "community_leading_eigenvector",
        "parameters": {
            "clusters": None,
            "weights": None,
            "arpack_options": None,
        },
    },
    "Leiden": {
        # Full Leiden: stock igraph CPM through hedonic.experiments.leiden
        # (detector identity LEIDEN_IDENTITY). Old tmp/hedonic used
        # community_leiden(only_first_phase=False).
        "method_call_name": "community_leiden_cpm",
        "detector_identity": LEIDEN_IDENTITY,
        "parameters": {
            "resolution": 1,
            "beta": 0.01,
            "initial_membership": None,
            "n_iterations": -1,
        },
    },
    "Hedonic": {
        # Primary exploratory method: hedonic local-moving phase
        # Old tmp/hedonic used community_leiden(only_first_phase=True).
        "method_call_name": "community_hedonic",
        "parameters": {
            "resolution": 1,
            "initial_membership": None,
            "allow_isolation": False,
            "max_memberships": 1,
        },
    },
}

# Methods that re-sample / re-run stochastically (old exp.py: runs=10).
STOCHASTIC_METHODS = frozenset({"Leiden", "Hedonic"})

# ---------------------------------------------------------------------------
# Presets — V1020 full grid + tiny structural smoke
# ---------------------------------------------------------------------------

# Exact parameter grid used to produce PHYSA/Synthetic_Networks/V1020.
V1020_PRESET: dict[str, Any] = {
    "folder_name": "resultados",
    "max_n_nodes": 1020,
    "n_communities": [2, 3, 4, 5, 6],
    "seeds": [0, 1, 2, 3, 4],
    "p_in": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    "difficulty": [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "noises": [0.10, 0.25, 0.50, 0.75, 1.00],
    "partition_seeds": 10,  # → seeds 0..9
    "methods": list(METHODS.keys()),
    "n_runs": 10,
    "layout": "v1020",
}

# Tiny subset with the *same layout/methods/fields* as V1020 (for smoke tests).
V1020_SMOKE_PRESET: dict[str, Any] = {
    "folder_name": "resultados",
    "max_n_nodes": 40,
    "n_communities": [2],
    "seeds": [0],
    "p_in": [0.10],
    "difficulty": [0.30],
    "noises": [0.10, 1.00],
    "partition_seeds": 2,  # → seeds 0,1
    "methods": list(METHODS.keys()),
    "n_runs": 2,
    "layout": "v1020",
}

# Current benchmark: same parameter grid as V1020, 100 seeded networks per
# condition generated on demand (never stored), columnar result store.
SBM_PRESET: dict[str, Any] = {
    "folder_name": "sbm",
    "max_n_nodes": 1020,
    "n_communities": [2, 3, 4, 5, 6],
    "seeds": list(range(100)),
    "p_in": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    "difficulty": [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
    "noises": [0.10, 0.25, 0.50, 0.75, 1.00],
    "partition_seeds": 10,
    "methods": list(METHODS.keys()),
    "n_runs": 10,
    "layout": "store",
    # ~55M rows: hashes take ~3 GB, full partitions ~27 GB.  Every row replays
    # from its seed on the recorded build; pass --partitions full to keep them.
    "partitions": "hash",
}

# The whole SBM pipeline at small scale: every method, several networks,
# noises and partition seeds, stochastic restarts, store → query → figures.
SBM_SMOKE_PRESET: dict[str, Any] = {
    "folder_name": "sbm-smoke",
    "max_n_nodes": 120,
    "n_communities": [2, 3],
    "seeds": [0, 1],
    "p_in": [0.10, 0.30],
    "difficulty": [0.20, 0.60],
    "noises": [0.10, 0.50, 1.00],
    "partition_seeds": 2,
    "methods": list(METHODS.keys()),
    "n_runs": 3,
    "layout": "store",
    "partitions": "full",
}

PRESETS: dict[str, dict[str, Any]] = {
    "sbm": SBM_PRESET,
    "sbm-smoke": SBM_SMOKE_PRESET,
    "v1020": V1020_PRESET,
    "v1020-smoke": V1020_SMOKE_PRESET,
}

LAYOUTS = ("store", "v1020", "legacy")


# ---------------------------------------------------------------------------
# Graph generation & partitions
# ---------------------------------------------------------------------------

def expected_vertex_count(number_of_communities: int, community_size: int) -> int:
    """Return the exact vertex count implied by an SBM condition.

    The historical generator occasionally serialized a graph with trailing
    isolated vertices omitted.  Keeping this arithmetic explicit lets every
    new run fail before detection if a graph ever violates the experiment's
    fixed-size contract.
    """
    n_communities = int(number_of_communities)
    size = int(community_size)
    if n_communities <= 0 or size <= 0:
        raise ValueError(
            "number_of_communities and community_size must both be positive"
        )
    return n_communities * size


def validate_graph_shape(
    g: Game,
    number_of_communities: int,
    community_size: int,
    *,
    context: str = "generated graph",
) -> None:
    """Validate the fixed-size undirected graph contract for one condition."""
    expected = expected_vertex_count(number_of_communities, community_size)
    actual = int(g.vcount())
    if actual != expected:
        raise ValueError(
            f"{context} has {actual} vertices; expected {expected} "
            f"({number_of_communities} × {community_size})."
        )
    if g.is_directed():
        raise ValueError(f"{context} must be undirected")


def validate_grid(
    *,
    max_n_nodes: int,
    n_communities: Iterable[int],
    p_in: Iterable[float],
    difficulty: Iterable[float],
    noises: Iterable[float],
    partition_seeds,
    methods: dict[str, dict],
    n_runs: int,
) -> list[int]:
    """Validate a sweep specification and return normalized community counts."""
    max_n_nodes = int(max_n_nodes)
    if max_n_nodes <= 0:
        raise ValueError("max_n_nodes must be positive")

    normalized_communities = [int(value) for value in n_communities]
    if not normalized_communities:
        raise ValueError("n_communities must contain at least one value")
    for n_community in normalized_communities:
        if n_community <= 0:
            raise ValueError("n_communities values must be positive")
        if max_n_nodes % n_community:
            raise ValueError(
                f"max_n_nodes={max_n_nodes} is not divisible by "
                f"n_communities={n_community}; refusing to truncate the SBM."
            )

    p_values = [float(value) for value in p_in]
    difficulty_values = [float(value) for value in difficulty]
    noise_values = [float(value) for value in noises]
    if not p_values or not difficulty_values or not noise_values:
        raise ValueError("p_in, difficulty, and noises must each be non-empty")
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values):
        raise ValueError("p_in values must be finite probabilities in [0, 1]")
    if any(
        not math.isfinite(value) or value < 0 for value in difficulty_values
    ):
        raise ValueError("difficulty values must be finite and non-negative")
    if any(not math.isfinite(value) or value < 0 for value in noise_values):
        raise ValueError("noise values must be finite and non-negative")
    if any(p * multiplier > 1 for p in p_values for multiplier in difficulty_values):
        raise ValueError("p_out = p_in * difficulty must not exceed 1")

    normalized_partition_seeds = _normalize_partition_seeds(partition_seeds)
    if not normalized_partition_seeds:
        raise ValueError("partition_seeds must contain at least one seed")
    if int(n_runs) <= 0:
        raise ValueError("n_runs must be positive")
    if not methods:
        raise ValueError("methods must contain at least one method")
    unknown = [name for name in methods if name not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")
    return normalized_communities

def probs_matrix(n_communities: int, p: float, q: float) -> list[list[float]]:
    return [
        [p if i == j else q for j in range(n_communities)]
        for i in range(n_communities)
    ]


SBM_GENERATOR_ID = "hedonic-sbm-pcg64-threshold-v1"
"""Identity of the graph generator; bump it whenever the sampled graph for a
given (parameters, seed) could change.  It is part of the random-stream key,
so two generator versions never silently share a stream."""

_SBM_STREAM_TAG = int.from_bytes(
    hashlib.sha256(SBM_GENERATOR_ID.encode("ascii")).digest()[:8], "big"
)
_TWO_64 = 1 << 64


def exact_probability(value) -> Fraction:
    """Return ``value`` as an exact rational probability.

    Floats are read through their shortest decimal ``repr`` so that ``0.06``
    means 6/100, not the nearest binary double.  The sampled graph therefore
    depends only on the decimal parameters a person writes down.
    """
    if isinstance(value, bool):
        raise TypeError("probability must be a number, not bool")
    if isinstance(value, Fraction):
        result = value
    elif isinstance(value, int):
        result = Fraction(value)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"probability must be finite, got {value!r}")
        result = Fraction(value)
    else:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"probability must be finite, got {value!r}")
        result = Fraction(Decimal(repr(number)))
    return result


def _edge_threshold(probability: Fraction) -> int:
    """Integer cut ``floor(p * 2**64)``: a pair is an edge iff raw < cut."""
    if probability < 0 or probability > 1:
        raise ValueError(f"probability must lie in [0, 1], got {probability}")
    return math.floor(probability * _TWO_64)


def sbm_seed_sequence(
    n_communities: int,
    community_size: int,
    p_in,
    multiplier,
    seed: int,
) -> np.random.SeedSequence:
    """Random-stream key of one SBM graph: generator id, seed and parameters."""
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    p = exact_probability(p_in)
    d = exact_probability(multiplier)
    return np.random.SeedSequence(
        [
            _SBM_STREAM_TAG,
            int(seed),
            int(n_communities),
            int(community_size),
            p.numerator,
            p.denominator,
            d.numerator,
            d.denominator,
        ]
    )


def sbm_edges(
    n_communities: int,
    community_size: int,
    p_in,
    multiplier,
    seed: int,
) -> np.ndarray:
    """Sample the planted-partition SBM edge list, fully determined by inputs.

    Vertex ``v`` belongs to block ``v // community_size``.  Pairs ``(i, j)``
    with ``i < j`` are visited in row-major order and each consumes exactly
    one raw 64-bit PCG64 output ``r``; the pair is an edge iff
    ``r < floor(p * 2**64)`` with ``p = p_in`` inside a block and
    ``p = p_in * multiplier`` across blocks (both exact rationals).  Only the
    PCG64/SeedSequence bit streams and integer comparisons are involved, so
    the result does not depend on NumPy's distribution code, floating-point
    rounding, the platform, or any global random state.

    Returns an ``(m, 2)`` ``int64`` array of edges ``(i, j)``, ``i < j``,
    sorted lexicographically.
    """
    n_communities = int(n_communities)
    community_size = int(community_size)
    n = expected_vertex_count(n_communities, community_size)
    p = exact_probability(p_in)
    q = p * exact_probability(multiplier)
    cut_in = _edge_threshold(p)
    cut_out = _edge_threshold(q)
    bits = np.random.PCG64(
        sbm_seed_sequence(n_communities, community_size, p_in, multiplier, seed)
    )

    def accept(raw: np.ndarray, cut: int) -> np.ndarray:
        if cut <= 0:
            return np.zeros(raw.shape, dtype=bool)
        # cut - 1 fits in uint64 for every cut in [1, 2**64].
        return raw <= np.uint64(cut - 1)

    chunks: list[np.ndarray] = []
    for i in range(n - 1):
        block_end = (i // community_size + 1) * community_size
        raw = np.asarray(bits.random_raw(n - 1 - i), dtype=np.uint64)
        n_in = block_end - i - 1
        mask = np.concatenate((accept(raw[:n_in], cut_in), accept(raw[n_in:], cut_out)))
        targets = np.flatnonzero(mask) + (i + 1)
        if targets.size:
            chunks.append(
                np.column_stack((np.full(targets.size, i, dtype=np.int64), targets))
            )
    if not chunks:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(chunks)


def graph_fingerprint(g: ig.Graph) -> str:
    """SHA-256 of the vertex count and the ordered, normalized edge list."""
    digest = hashlib.sha256()
    digest.update(np.int64(g.vcount()).tobytes())
    edges = np.asarray(g.get_edgelist(), dtype="<i8").reshape(-1, 2)
    digest.update(np.sort(edges, axis=1).tobytes())
    return digest.hexdigest()


def generate_graph(
    n_communities: int,
    community_size: int,
    p_in: float,
    multiplier: float,
    seed: int,
) -> Game:
    """Generate the planted-partition SBM graph for ``(parameters, seed)``.

    The same arguments give the same graph (vertex count, edge list and edge
    order) on every call, process and platform; see :func:`sbm_edges` and
    :data:`SBM_GENERATOR_ID`.  No global random state (Python, NumPy or
    igraph) is read or modified, so graphs need not be stored on disk.
    """
    n = expected_vertex_count(n_communities, community_size)
    edges = sbm_edges(n_communities, community_size, p_in, multiplier, seed)
    game = Game(ig.Graph(n=n, edges=edges.tolist(), directed=False))
    validate_graph_shape(
        game,
        n_communities,
        community_size,
        context="generated SBM graph",
    )
    return game


def get_ground_truth(
    number_of_communities: int,
    community_size: int,
    g: Game | None = None,
):
    gt_membership = np.concatenate(
        [np.full(community_size, i) for i in range(number_of_communities)]
    ).tolist()
    if g is not None:
        validate_graph_shape(
            g,
            number_of_communities,
            community_size,
            context="graph supplied for ground truth",
        )
        return VertexClustering(g, gt_membership)
    return gt_membership


def _membership_rng(seed):
    """Legacy ``RandomState`` stream: NumPy guarantees it never changes."""
    return np.random if seed is None else np.random.RandomState(seed)


def shuffle_with_noise(membership, noise=1.0, seed=None, rng=None):
    """Swap the labels of ``int(noise * n)`` randomly chosen vertices.

    With a seed the draws come from a private ``RandomState(seed)``, which
    yields the same draws the former global ``np.random.seed(seed)`` did
    without touching NumPy's global state.
    """
    if rng is None:
        rng = _membership_rng(seed)

    community_dict = defaultdict(list)
    for node, community in enumerate(membership):
        community_dict[community].append(node)

    for community_nodes in community_dict.values():
        rng.shuffle(community_nodes)

    shuffled_membership = [None] * len(membership)
    for community, nodes in community_dict.items():
        for node in nodes:
            shuffled_membership[node] = community

    n = len(membership)
    num_to_shuffle = int(noise * n)
    if num_to_shuffle <= 0:
        return shuffled_membership

    indices_to_shuffle = rng.choice(range(n), size=num_to_shuffle, replace=False)
    shuffled_indices = rng.permutation(indices_to_shuffle)
    for i, j in zip(indices_to_shuffle, shuffled_indices):
        shuffled_membership[i], shuffled_membership[j] = (
            shuffled_membership[j],
            shuffled_membership[i],
        )
    return shuffled_membership


def get_initial_membership(ground_truth, noise=1, seed=None):
    """Noisy start: ground truth with ``noise`` × n label swaps (``> 1``: singletons).

    A pure function of ``(ground_truth, noise, seed)`` when ``seed`` is given.
    """
    membership = (
        ground_truth if isinstance(ground_truth, list) else ground_truth.membership
    )
    if noise > 1:
        membership = list(range(len(membership)))  # singleton partition
    else:
        membership = shuffle_with_noise(membership, noise=noise, seed=seed)
    return membership


# ---------------------------------------------------------------------------
# Metrics & method helpers (ported for Game instances without them)
# ---------------------------------------------------------------------------

def accuracy(g: Game, partition, ground_truth) -> float:
    partition = (
        VertexClustering(g, partition) if isinstance(partition, list) else partition
    )
    ground_truth = (
        VertexClustering(g, ground_truth)
        if isinstance(ground_truth, list)
        else ground_truth
    )
    return float(compare_communities(partition, ground_truth, method="adjusted_rand"))


def _get_nodes_info(g: Game, membership_list: list[int]) -> dict:
    community_counter = Counter(membership_list)
    nodes_subset = set(range(g.vcount()))
    friends_counts = {node: Counter() for node in nodes_subset}
    for u, v in g.get_edgelist():
        if u in nodes_subset:
            friends_counts[u][membership_list[v]] += 1
        if v in nodes_subset:
            friends_counts[v][membership_list[u]] += 1
    nodes_info = {}
    for node in friends_counts:
        node_membership = membership_list[node]
        node_info = {}
        for community, total in community_counter.items():
            friend_count = friends_counts[node].get(community, 0)
            stranger_count = total - friend_count - (
                1 if community == node_membership else 0
            )
            node_info[community] = {
                "friends": friend_count,
                "strangers": stranger_count,
            }
        nodes_info[node] = node_info
    return nodes_info


def _classify_node_satisfaction(node_info, node_membership) -> str:
    max_friends = max(info["friends"] for info in node_info.values())
    min_strangers = min(info["strangers"] for info in node_info.values())
    robust_communities = set()
    for community in node_info:
        satisfy_max = node_info[community]["friends"] == max_friends
        satisfy_min = node_info[community]["strangers"] == min_strangers
        if satisfy_max and satisfy_min:
            robust_communities.add(community)
    if robust_communities:
        if node_membership not in robust_communities:
            return "never_satisfied"
        return "always_satisfied"
    return "relatively_satisfied"


def edge_array(g: ig.Graph) -> np.ndarray:
    """``(m, 2)`` int64 edge endpoints (compute once per graph, reuse)."""
    return np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)


def partition_agreement(membership, ground_truth) -> dict[str, float]:
    """Agreement of a partition with the planted one, beyond the Rand indices.

    ``overlap`` is the best one-to-one matching of found to planted
    communities (Hungarian assignment on the contingency table) as a fraction
    of vertices; unmatched found communities count as errors.  This is the
    overlap of the weak-recovery theorems.  ``overlap_normalized`` rescales it
    so that 1/K (chance for K planted communities) is 0 and 1 is perfect.
    ``nmi`` is igraph's normalized mutual information and ``vi`` the variation
    of information (natural log; 0 for identical partitions).
    """
    from scipy.optimize import linear_sum_assignment

    found = np.unique(np.asarray(membership), return_inverse=True)[1]
    truth = np.unique(np.asarray(ground_truth), return_inverse=True)[1]
    n = len(truth)
    k = int(truth.max()) + 1
    table = np.zeros((k, int(found.max()) + 1), dtype=np.int64)
    np.add.at(table, (truth, found), 1)
    rows, cols = linear_sum_assignment(table, maximize=True)
    overlap = float(table[rows, cols].sum()) / n
    chance = 1.0 / k
    return {
        "overlap": overlap,
        "overlap_normalized": (overlap - chance) / (1.0 - chance) if k > 1 else 1.0,
        "nmi": float(compare_communities(list(found), list(truth), method="nmi")),
        "vi": float(compare_communities(list(found), list(truth), method="vi")),
    }


def robustness(g: Game, partition, *, edges: np.ndarray | None = None) -> float:
    """Fraction of vertices that are always satisfied (vectorized).

    A vertex is *always satisfied* when its own community both maximizes its
    neighbours (friends) and minimizes its non-neighbours (strangers) over the
    occupied communities.  Equal to :func:`robustness_reference`; falls back
    to it when the ``n × K`` count matrices would be too large.
    """
    membership = (
        partition if isinstance(partition, list) else list(partition.membership)
    )
    n = g.vcount()
    if not n:
        return 0.0
    _, labels = np.unique(np.asarray(membership), return_inverse=True)
    k = int(labels.max()) + 1
    if n * k > 50_000_000:
        return robustness_reference(g, membership)
    if edges is None:
        edges = edge_array(g)
    u, v = edges[:, 0], edges[:, 1]
    friends = np.bincount(
        np.concatenate((u * k + labels[v], v * k + labels[u])), minlength=n * k
    ).reshape(n, k)
    strangers = np.bincount(labels, minlength=k)[None, :] - friends
    rows = np.arange(n)
    strangers[rows, labels] -= 1
    best = (friends == friends.max(axis=1, keepdims=True)) & (
        strangers == strangers.min(axis=1, keepdims=True)
    )
    return float(np.count_nonzero(best[rows, labels])) / n


def robustness_reference(g: Game, partition) -> float:
    """Direct transcription of the original definition (used for testing)."""
    membership = (
        partition if isinstance(partition, list) else list(partition.membership)
    )
    nodes_info = _get_nodes_info(g, membership)
    robust = sum(
        1
        for node, info in nodes_info.items()
        if _classify_node_satisfaction(info, membership[node]) == "always_satisfied"
    )
    return robust / g.vcount() if g.vcount() else 0.0


def community_groundtruth(g: Game, groundtruth) -> VertexClustering:
    if isinstance(groundtruth, list):
        return VertexClustering(g, groundtruth)
    return groundtruth


def community_mirror(g: Game, initial_membership=None) -> VertexClustering:
    if initial_membership is None:
        initial_membership = [0] * g.vcount()
    return VertexClustering(g, initial_membership)


def community_onepass_improvement(
    g: Game, initial_membership=None
) -> VertexClustering:
    if initial_membership is None:
        initial_membership = [0] * g.vcount()
    for node, community in zip(g.vs, initial_membership):
        g.vs[node.index]["community"] = int(community)
    nodes_to_move = []
    for node in g.vs:
        neighbors_comms = [g.vs[n]["community"] for n in g.neighbors(node)]
        if neighbors_comms:
            pref_comm = max(set(neighbors_comms), key=neighbors_comms.count)
            if pref_comm != node["community"]:
                nodes_to_move.append((node.index, pref_comm))
    new_membership = [int(i) for i in initial_membership]
    while nodes_to_move:
        node, community = nodes_to_move.pop()
        new_membership[node] = community
    return VertexClustering(g, new_membership)


def community_leiden_cpm(g: Game, **params):
    """Full disjoint Leiden (CPM) through :func:`hedonic.experiments.leiden.leiden`."""
    return leiden(g, **params)


_LOCAL_METHODS = {
    "community_leiden_cpm": community_leiden_cpm,
    "community_groundtruth": community_groundtruth,
    "community_mirror": community_mirror,
    "community_onepass_improvement": community_onepass_improvement,
}


def _call_method(g: Game, method_name: str, method_params: dict):
    if method_name in _LOCAL_METHODS:
        return _LOCAL_METHODS[method_name](g, **method_params)
    method = getattr(g, method_name)
    return method(**method_params)


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def get_method_result(
    g: Game,
    method_name,
    method_params,
    p_in,
    multiplier,
    community_size,
    number_of_communities,
    ground_truth,
    display_method: str | None = None,
):
    # Timing rule ``store_mod.TIMING_SCOPE``: the clock covers the detector call and nothing else.
    # The error fallback and its report come after the clock stops.
    error_message: str | None = None
    failure: Exception | None = None
    partition = None
    started = time.perf_counter()
    try:
        partition = _call_method(g, method_name, method_params)
    except Exception as e:
        failure = e
    duration = time.perf_counter() - started
    if failure is not None:
        error_message = f"{type(failure).__name__}: {failure}"
        partition = VertexClustering(g, [0] * g.vcount())
        print(
            f"\nPARTITIONING ERROR:\n{failure}\n{method_name=}\n{p_in=}\n"
            f"{multiplier=}\n{community_size=}\n{number_of_communities=}"
        )
    acc = accuracy(g, partition, ground_truth)
    rob = robustness(g, partition)
    label = display_method
    if label is None:
        label = method_name.split("_")[1] if "_" in method_name else method_name
    result = {
        "method": label,
        "number_of_communities": number_of_communities,
        "community_size": community_size,
        "p_in": p_in,
        "p_out": p_in * multiplier,
        "multiplier": multiplier,
        "resolution": method_params.get("resolution"),
        "duration": duration,
        "accuracy": acc,
        "robustness": rob,
        "partition": list(partition.membership),
        # New runs distinguish a successful detector from the historical
        # fallback partition used when a method raised.  Older archive records
        # lack this field and remain readable for analysis.
        "status": "error" if error_message else "ok",
    }
    if error_message:
        result["error"] = error_message
    return result


def _normalize_partition_seeds(partition_seeds) -> list[int]:
    if partition_seeds is None:
        return [0]
    if isinstance(partition_seeds, int):
        return list(range(partition_seeds)) if partition_seeds > 0 else [0]
    return list(partition_seeds)


def _method_n_runs(method_name: str, n_runs: int) -> int:
    """Leiden/Hedonic re-run stochastically (V1020 exp.py used runs=10)."""
    if method_name in STOCHASTIC_METHODS:
        return max(1, int(n_runs))
    return 1


def _prepare_method_params(
    method_call_name: str,
    parameters: dict,
    *,
    gt,
    edge_density: float,
    number_of_communities: int,
) -> dict:
    params = dict(parameters)
    if method_call_name == "community_groundtruth":
        params["groundtruth"] = gt
    if method_call_name == "community_leading_eigenvector":
        # Old exp.py set clusters = number_of_communities.
        params["clusters"] = number_of_communities
    if method_call_name in {"community_leiden", "community_leiden_cpm", "community_hedonic"}:
        params["resolution"] = edge_density
    return params


# ---------------------------------------------------------------------------
# Columnar store layout (one Parquet shard per generated graph)
# ---------------------------------------------------------------------------


def _uses_initial_membership(method_info: dict) -> bool:
    return "initial_membership" in method_info["parameters"]


def _execute_store_run(
    g: Game,
    gt: VertexClustering,
    method_info: dict,
    *,
    number_of_communities: int,
    initial_membership: list[int] | None,
    run_seed: int,
    edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run one detector call under ``run_seed`` and score it.

    ``community_hedonic`` and the Leiden helper receive the seed directly; every other method runs
    with igraph's generator temporarily set to ``random.Random(run_seed)``.
    The returned metrics are deterministic functions of the partition.
    """
    method_call_name = method_info["method_call_name"]
    params = _prepare_method_params(
        method_call_name,
        method_info["parameters"],
        gt=gt,
        edge_density=g.density(),
        number_of_communities=number_of_communities,
    )
    if "initial_membership" in params:
        params["initial_membership"] = list(initial_membership)
    error = None
    failure: Exception | None = None
    partition = None
    # Timing rule ``store_mod.TIMING_SCOPE``: the clock covers the detector call and nothing else.
    # The generator is seeded before it starts, and the membership copy, the error
    # fallback and every metric come after it stops, the same for every method.
    seeded = method_call_name in {"community_hedonic", "community_leiden_cpm"}
    call_params = dict(params, seed=run_seed) if seeded else params
    with contextlib.nullcontext() if seeded else seeded_igraph_rng(run_seed):
        start = time.perf_counter()
        try:
            partition = _call_method(g, method_call_name, call_params)
        except Exception as exc:  # recorded as an error row, never silently dropped
            failure = exc
        duration = time.perf_counter() - start
    if failure is None:
        try:
            membership = list(partition.membership)
        except Exception as exc:
            failure = exc
    if failure is not None:
        error = f"{type(failure).__name__}: {failure}"
        membership = [0] * g.vcount()
    clustering = VertexClustering(g, membership)
    canonical = store_mod.canonical_partition(membership)
    return {
        "resolution": params.get("resolution"),
        "status": "error" if error else "ok",
        "error": error,
        "duration": float(duration),
        "rand_index": float(compare_communities(clustering, gt, method="rand")),
        "adjusted_rand": float(
            compare_communities(clustering, gt, method="adjusted_rand")
        ),
        "robustness": float(robustness(g, membership, edges=edges)),
        **partition_agreement(membership, gt.membership),
        "n_found_communities": int(canonical.max()) + 1 if len(canonical) else 0,
        "partition_hash": store_mod.partition_hash(canonical),
        "partition": canonical,
    }


SUMMARY_METRICS = ("adjusted_rand", "overlap_normalized", "nmi", "robustness", "duration")


def run_graph_store(store_root: str | Path, **kwargs) -> str:
    """Generate one graph, run every method, write its shard; returns the status.

    See :func:`_graph_store` for the arguments.
    """
    return _graph_store(store_root, **kwargs)[0]


def _graph_store(
    store_root: str | Path,
    *,
    spec: dict[str, Any],
    spec_digest: str,
    number_of_communities: int,
    community_size: int,
    p_in: float,
    difficulty: float,
    network_seed: int,
    noises: Iterable[float],
    partition_seeds: Iterable[int],
    methods: dict[str, dict],
    resume: bool = False,
    legacy: dict[str, dict[str, str]] | None = None,
    carry: dict[str, Any] | None = None,
) -> str:
    """Generate one graph, run the methods whose rows are missing, write its shard.

    Methods that use ``initial_membership`` run per (noise, partition seed)
    cell; the others (GroundTruth, Spectral) run once per graph and are
    stored with null ``noise``/``partition_seed``.  Every restart is kept;
    ``is_duplicate`` marks a partition already returned in the same cell, and
    such rows store only ``partition_hash`` (the partition is in the earlier
    row of that cell with the same hash).

    With ``resume``, a method whose rows in the shard carry its current identity
    (``spec["method_identities"]``) and cover the cells is kept as stored, and so are the rows of
    methods this run does not name; only the rest is computed.  ``carry`` (``root``, ``methods``,
    ``legacy``) takes a method's rows from another store instead, when that store's shard has them
    for the same graph fingerprint and cells; the caller has verified them by replay.
    Returns ``"skipped"``, ``"written"`` or ``"carried"`` (written with no detector run).
    """
    noises = [float(value) for value in noises]
    partition_seeds = [int(value) for value in partition_seeds]
    keys = {
        "number_of_communities": int(number_of_communities),
        "community_size": int(community_size),
        "p_in": float(p_in),
        "difficulty": float(difficulty),
        "network_seed": int(network_seed),
    }
    current = store_mod.method_digests(spec)
    path = Path(store_root) / store_mod.shard_relative_path(**keys)
    stored = (store_mod.shard_methods(path, legacy) if path.is_file() else None) or {}

    def complete(entry: dict | None, name: str, identity: str | None) -> bool:
        if not entry or (identity is not None and entry.get("identity") != identity):
            return False
        return not _uses_initial_membership(methods[name]) or store_mod.covers_cells(
            entry, noises, partition_seeds
        )

    todo = [
        name for name in methods if not (resume and complete(stored.get(name), name, current[name]))
    ]
    if not todo:
        return "skipped", None

    g = generate_graph(
        number_of_communities, community_size, p_in, difficulty, network_seed
    )
    gt = get_ground_truth(number_of_communities, community_size, g)
    graph_sha = graph_fingerprint(g)
    edges = edge_array(g)
    p_out = float(exact_probability(p_in) * exact_probability(difficulty))
    base = dict(
        keys,
        p_out=p_out,
        graph_sha256=graph_sha,
        n_vertices=g.vcount(),
        n_edges=g.ecount(),
    )
    keep_partitions = spec["partitions"] == "full"
    cells_meta = {"noises": noises, "partition_seeds": partition_seeds}
    # Rows kept as stored: methods not recomputed (current ones, and those this run does not name).
    kept = [name for name in stored if name not in todo]
    rows: list[dict[str, Any]] = store_mod.read_shard_rows(path, kept) if kept else []
    methods_meta: dict[str, dict[str, Any]] = {name: stored[name] for name in kept}
    if any(row.get("graph_sha256") != graph_sha for row in rows):
        raise RuntimeError(f"{path}: stored rows belong to another graph (fingerprint differs)")
    carried_any = False
    if carry:
        source = Path(carry["root"]) / store_mod.shard_relative_path(**keys)
        offered = (store_mod.shard_methods(source, carry.get("legacy")) if source.is_file() else None) or {}
        take = [name for name in todo if name in carry["methods"] and complete(offered.get(name), name, None)]
        if take:
            taken = store_mod.read_shard_rows(source, take)
            if any(row.get("graph_sha256") != graph_sha for row in taken):
                raise RuntimeError(f"{source}: carried rows belong to another graph (fingerprint differs)")
            for row in taken:
                if row.get("noise") is not None and (
                    store_mod.decimal_text(row["noise"]) not in {store_mod.decimal_text(n) for n in noises}
                    or int(row["partition_seed"]) not in set(partition_seeds)
                ):
                    continue  # a cell outside this grid
                if not keep_partitions:
                    row["partition"] = None
                rows.append(row)
            for name in take:
                methods_meta[name] = dict(cells_meta, identity=current[name], carried_from=str(carry["root"]))
            todo = [name for name in todo if name not in take]
            carried_any = True
    computed = bool(todo)
    for method_name in todo:
        method_info = methods[method_name]
        methods_meta[method_name] = dict(cells_meta, identity=current[method_name])
        runs = _method_n_runs(method_name, spec["n_runs"])
        if _uses_initial_membership(method_info):
            cells = [(noise, seed) for noise in noises for seed in partition_seeds]
        else:
            cells = [(None, None)]
        for noise, partition_seed in cells:
            initial = (
                None
                if noise is None
                else get_initial_membership(gt, noise, partition_seed)
            )
            seen: set[bytes] = set()
            for run in range(runs):
                seed = store_mod.run_seed(
                    method=method_name,
                    noise=noise,
                    partition_seed=partition_seed,
                    run=run,
                    **keys,
                )
                result = _execute_store_run(
                    g,
                    gt,
                    method_info,
                    number_of_communities=number_of_communities,
                    initial_membership=initial,
                    run_seed=seed,
                    edges=edges,
                )
                digest = result["partition_hash"]
                row = dict(
                    base,
                    method=method_name,
                    noise=noise,
                    partition_seed=partition_seed,
                    run=run,
                    run_seed=seed,
                    is_duplicate=digest in seen,
                    **result,
                )
                seen.add(digest)
                if not keep_partitions or row["is_duplicate"]:
                    # A duplicate's partition is the earlier row of the same
                    # cell with the same partition_hash.
                    row["partition"] = None
                rows.append(row)

    store_mod.write_shard(
        store_root,
        rows=rows,
        shard_info=dict(
            keys,
            graph_generator=SBM_GENERATOR_ID,
            graph_sha256=graph_sha,
            n_vertices=g.vcount(),
            n_edges=g.ecount(),
            noises=noises,
            partition_seeds=partition_seeds,
        ),
        spec_digest=spec_digest,
        partitions=spec["partitions"],
        methods_meta=methods_meta,
    )
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        if row["status"] != "ok" or row["method"] not in methods:
            continue
        item = summary.setdefault(row["method"], {"n": 0, **{m: 0.0 for m in SUMMARY_METRICS}})
        item["n"] += 1
        for metric in SUMMARY_METRICS:
            item[metric] += float(row[metric])
    return ("carried" if carried_any and not computed else "written"), summary


def _run_graph_store_task(kwargs: dict[str, Any]) -> tuple[dict[str, Any], str, dict | None]:
    """Process-pool entry point (top level so it pickles under spawn).

    Returns the graph's keys, its status (written / skipped) and, for a
    written graph, per-method sums of the summary metrics.
    """
    status, summary = _graph_store(**kwargs)
    label = {
        key: kwargs[key]
        for key in ("number_of_communities", "p_in", "difficulty", "network_seed")
    }
    return label, status, summary


def replay_store_row(g: Game, row: dict[str, Any], spec: dict[str, Any]) -> dict:
    """Recompute one stored row from its keys and seed on the given graph."""
    method_name = row["method"]
    if method_name not in METHODS:
        raise ValueError(f"Unknown method in store: {method_name!r}")
    current = store_mod.build_spec(
        methods={method_name: METHODS[method_name]},
        n_runs=spec["n_runs"],
        partitions=spec["partitions"],
        generator_id=SBM_GENERATOR_ID,
    )["methods"][method_name]
    if current != spec["methods"].get(method_name):
        raise ValueError(
            f"{method_name}: METHODS changed since the store was written; "
            "replay would not run the stored configuration"
        )
    gt = get_ground_truth(row["number_of_communities"], row["community_size"], g)
    initial = (
        None
        if row["noise"] is None
        else get_initial_membership(gt, row["noise"], row["partition_seed"])
    )
    return _execute_store_run(
        g,
        gt,
        METHODS[method_name],
        number_of_communities=row["number_of_communities"],
        initial_membership=initial,
        run_seed=row["run_seed"],
    )


def _v1020_relative_dir(
    folder_name: str,
    number_of_communities: int,
    community_size: int,
    noise: float,
    p_in: float,
    difficulty: float,
    seed: int,
) -> Path:
    """Path layout matching archived V1020/resultados/.../Network (NNN)/."""
    return (
        Path(folder_name)
        / f"{number_of_communities}C_{community_size}N"
        / f"Noise = {noise:.2f}"
        / f"P_in = {p_in:.2f}"
        / f"Difficulty = {difficulty:.2f}"
        / f"Network ({seed:03d})"
    )


def _legacy_relative_dir(
    folder_name: str,
    number_of_communities: int,
    community_size: int,
    noise: float,
    p_in: float,
    difficulty: float,
    seed: int,
    partition_seed: int,
) -> Path:
    """One JSON per method under Partition (MMM)/ (experiment.py style)."""
    return (
        Path(folder_name)
        / f"{number_of_communities} Communities of {community_size} nodes"
        / f"Noise = {noise:.2f}"
        / f"P_in = {p_in:.2f}"
        / f"Difficulty = {difficulty:.2f}"
        / f"Network ({seed:03d})"
        / f"Partition ({partition_seed:03d})"
    )


def _atomic_json_dump(payload: Any, path: Path) -> None:
    """Write one result file atomically in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, path)
    finally:
        # If serialization or replace failed, do not leave a file that could
        # be mistaken for a completed cell by --resume.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _record_file_is_valid(
    path: Path,
    *,
    expected_methods: set[str],
    expected_vertices: int,
) -> bool:
    """Check the minimum schema needed to safely resume one output cell."""
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError, TypeError):
        return False
    records = payload if isinstance(payload, list) else [payload]
    methods_seen: set[str] = set()
    if not records:
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        method = record.get("method")
        partition = record.get("partition")
        if not isinstance(method, str) or method not in expected_methods:
            continue
        if record.get("status") not in (None, "ok"):
            return False
        if not isinstance(partition, list) or len(partition) != expected_vertices:
            return False
        methods_seen.add(method)
    return methods_seen >= expected_methods


def _v1020_cell_path(
    output_root: Path,
    folder_name: str,
    number_of_communities: int,
    community_size: int,
    noise: float,
    p_in: float,
    difficulty: float,
    seed: int,
    partition_seed: int,
) -> Path:
    return (
        output_root
        / _v1020_relative_dir(
            folder_name,
            number_of_communities,
            community_size,
            noise,
            p_in,
            difficulty,
            seed,
        )
        / f"partition_{partition_seed:03d}.json"
    )


def _legacy_cell_is_complete(
    output_root: Path,
    folder_name: str,
    number_of_communities: int,
    community_size: int,
    noise: float,
    p_in: float,
    difficulty: float,
    seed: int,
    partition_seed: int,
    expected_methods: set[str],
    expected_vertices: int,
) -> bool:
    """Validate the base result for each method in a legacy-layout cell.

    Legacy stochastic runs can produce an unknown number of ``_NN`` files.
    Therefore resume is conservative: it is only considered complete when
    the base file for every requested method is valid; callers disable this
    optimization for multi-run legacy jobs below.
    """
    relative = _legacy_relative_dir(
        folder_name,
        number_of_communities,
        community_size,
        noise,
        p_in,
        difficulty,
        seed,
        partition_seed,
    )
    return all(
        _record_file_is_valid(
            output_root / relative / f"{method_name}.json",
            expected_methods={method_name},
            expected_vertices=expected_vertices,
        )
        for method_name in expected_methods
    )


def _cell_is_complete(
    *,
    output_root: Path,
    folder_name: str,
    number_of_communities: int,
    community_size: int,
    noise: float,
    p_in: float,
    difficulty: float,
    seed: int,
    partition_seed: int,
    expected_methods: set[str],
    expected_vertices: int,
    layout: str,
    n_runs: int,
) -> bool:
    if layout == "v1020":
        return _record_file_is_valid(
            _v1020_cell_path(
                output_root,
                folder_name,
                number_of_communities,
                community_size,
                noise,
                p_in,
                difficulty,
                seed,
                partition_seed,
            ),
            expected_methods=expected_methods,
            expected_vertices=expected_vertices,
        )
    # With multiple stochastic restarts a base file alone does not prove that
    # all unique runs finished.  Re-run the legacy cell instead of silently
    # accepting a partial directory.
    if n_runs > 1:
        return False
    return _legacy_cell_is_complete(
        output_root,
        folder_name,
        number_of_communities,
        community_size,
        noise,
        p_in,
        difficulty,
        seed,
        partition_seed,
        expected_methods,
        expected_vertices,
    )


def preflight_grid(
    *,
    folder_name: str,
    max_n_nodes: int,
    n_communities: Iterable[int],
    seeds: Iterable[int],
    p_in: Iterable[float],
    difficulty: Iterable[float],
    noises: Iterable[float],
    partition_seeds,
    methods: dict[str, dict],
    n_runs: int,
    layout: str,
    output_root: Path,
    check_graphs: bool = True,
    spec: dict[str, Any] | None = None,
    max_graph_checks: int | None = None,
) -> dict[str, Any]:
    """Validate a selected sweep and return a no-write execution plan.

    With ``spec`` (store layout) the plan reads the store: ``method_graphs`` counts, per method,
    the graphs whose rows are current, of another build or setting, or missing.

    ``check_graphs`` performs the same fixed-size generation used by the real
    run, but never invokes a detector or writes a result.  This catches the
    historical isolated-tail corruption before a long run is started.
    """
    layout = (layout or "legacy").lower()
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; use one of {LAYOUTS}")
    # Materialize iterables once so validation and the no-write plan inspect
    # the same values even when callers provide generators.
    n_communities = list(n_communities)
    seeds = list(seeds)
    p_in = list(p_in)
    difficulty = list(difficulty)
    noises = list(noises)
    communities = validate_grid(
        max_n_nodes=max_n_nodes,
        n_communities=n_communities,
        p_in=p_in,
        difficulty=difficulty,
        noises=noises,
        partition_seeds=partition_seeds,
        methods=methods,
        n_runs=n_runs,
    )
    network_seeds = [int(value) for value in seeds]
    if not network_seeds:
        raise ValueError("seeds must contain at least one network seed")
    noise_values = [float(value) for value in noises]
    partition_seed_values = _normalize_partition_seeds(partition_seeds)
    p_values = [float(value) for value in p_in]
    difficulty_values = [float(value) for value in difficulty]
    graph_tasks = len(communities) * len(p_values) * len(difficulty_values) * len(
        network_seeds
    )
    cells_per_graph = len(noise_values) * len(partition_seed_values)
    expected_cells = graph_tasks * cells_per_graph
    graph_checks = 0
    # Evenly spaced sample of graphs to generate (None: every graph).
    check_every = (
        1
        if not max_graph_checks or graph_tasks <= max_graph_checks
        else math.ceil(graph_tasks / max_graph_checks)
    )
    task_index = -1
    malformed: list[dict[str, Any]] = []
    complete_cells = 0
    expected_methods = set(methods)
    # Store layout: per method, how many graphs hold rows of its current identity ("current"),
    # rows of another build or setting ("other") or none covering the cells ("missing").
    store_root = output_root / folder_name
    method_state = {name: Counter() for name in methods}
    by_k = {name: {k: Counter() for k in communities} for name in methods}
    current = store_mod.method_digests(spec) if spec is not None else {}
    manifest = store_mod.read_manifest(store_root) if layout == "store" else None
    legacy = store_mod.legacy_shard_digests(manifest) if manifest else {}
    for n_community in communities:
        community_size = max_n_nodes // n_community
        expected_vertices = expected_vertex_count(n_community, community_size)
        for p_value in p_values:
            for difficulty_value in difficulty_values:
                for seed in network_seeds:
                    task_index += 1
                    if check_graphs and task_index % check_every == 0:
                        try:
                            graph = generate_graph(
                                n_community,
                                community_size,
                                p_value,
                                difficulty_value,
                                seed,
                            )
                            validate_graph_shape(
                                graph,
                                n_community,
                                community_size,
                                context="preflight SBM graph",
                            )
                            graph_checks += 1
                        except Exception as exc:  # report every bad graph
                            malformed.append(
                                {
                                    "number_of_communities": n_community,
                                    "community_size": community_size,
                                    "p_in": p_value,
                                    "difficulty": difficulty_value,
                                    "network_seed": seed,
                                    "error": str(exc),
                                }
                            )
                    if layout == "store":
                        if spec is None:
                            continue
                        path = store_root / store_mod.shard_relative_path(
                            n_community, community_size, p_value, difficulty_value, seed
                        )
                        stored = (store_mod.shard_methods(path, legacy) if path.is_file() else None) or {}
                        all_current = True
                        for name, info in methods.items():
                            entry = stored.get(name)
                            covered = entry is not None and (
                                not _uses_initial_membership(info)
                                or store_mod.covers_cells(entry, noise_values, partition_seed_values)
                            )
                            if not covered:
                                state = "missing"
                            elif entry.get("identity") == current[name]:
                                state = "current"
                            else:
                                state = "other"
                            method_state[name][state] += 1
                            by_k[name][n_community][state] += 1
                            all_current &= state == "current"
                        if all_current:
                            complete_cells += cells_per_graph
                        continue
                    for noise in noise_values:
                        for partition_seed in partition_seed_values:
                            if _cell_is_complete(
                                output_root=output_root,
                                folder_name=folder_name,
                                number_of_communities=n_community,
                                community_size=community_size,
                                noise=noise,
                                p_in=p_value,
                                difficulty=difficulty_value,
                                seed=seed,
                                partition_seed=partition_seed,
                                expected_methods=expected_methods,
                                expected_vertices=expected_vertices,
                                layout=layout,
                                n_runs=n_runs,
                            ):
                                complete_cells += 1
    rows_per_graph = sum(
        _method_n_runs(name, n_runs)
        * (cells_per_graph if _uses_initial_membership(info) else 1)
        for name, info in methods.items()
    )
    return {
        "folder_name": folder_name,
        "layout": layout,
        "rows_per_graph": rows_per_graph,
        "expected_rows": rows_per_graph * graph_tasks,
        "max_n_nodes": int(max_n_nodes),
        "n_communities": communities,
        "network_seeds": network_seeds,
        "p_in": p_values,
        "difficulty": difficulty_values,
        "noises": noise_values,
        "partition_seeds": partition_seed_values,
        "methods": list(methods),
        "n_runs": int(n_runs),
        "graph_tasks": graph_tasks,
        "cells_per_graph": cells_per_graph,
        "expected_cells": expected_cells,
        "complete_cells": complete_cells,
        "pending_cells": expected_cells - complete_cells,
        "graph_checks": graph_checks,
        "malformed_graphs": malformed,
        "output_root": str(output_root),
        "method_graphs": {
            name: {key: int(counter[key]) for key in ("current", "other", "missing")}
            for name, counter in method_state.items()
        }
        if layout == "store" and spec is not None
        else {},
        # The same per number of communities K: {method: {K: {"current", "other", "missing"}}}.
        "method_graphs_by_k": {
            name: {int(k): {key: int(c[key]) for key in ("current", "other", "missing")} for k, c in table.items()}
            for name, table in by_k.items()
        }
        if layout == "store" and spec is not None
        else {},
    }


def run_experiment(
    folder_name,
    number_of_communities,
    community_size,
    p_in,
    difficulty,
    methods=None,
    noises=None,
    partition_seeds=None,
    seed=42,
    output_root: Path | None = None,
    n_runs: int = 1,
    layout: str = "legacy",
    resume: bool = False,
):
    """Run one (n_communities, size, p_in, difficulty, network seed) cell.

    Parameters
    ----------
    layout :
        ``\"v1020\"`` — write ``partition_MMM.json`` lists under
        ``{folder}/{n}C_{size}N/.../Network (seed)/`` (matches archived V1020).
        ``\"legacy\"`` — write one ``{Method}.json`` per method under
        ``.../Partition (MMM)/`` (tmp experiment.py style).
    n_runs :
        Stochastic restarts for Leiden/Hedonic (V1020 used 10). Unique
        partitions are kept (same dedup as old exp.py).
    resume :
        Skip an existing, schema-valid cell in ``v1020`` layout.  Corrupt or
        incomplete files are rerun; atomic writes ensure an interrupted task
        cannot look complete.
    """
    if methods is None:
        methods = METHODS
    if noises is None:
        noises = [0]
    noises = list(noises)
    partition_seeds = _normalize_partition_seeds(partition_seeds)
    output_root = (
        expand_path(output_root) if output_root is not None else DISJOINT_ARTIFACTS_DIR
    ).resolve()
    # Check both the root and the effective folder path.  The latter closes a
    # path traversal-by-configuration loophole such as output_root=.../parent
    # with folder_name="V1020".
    ensure_not_archived_v1020(output_root, label="output root")
    folder_path = Path(str(folder_name))
    if folder_path.is_absolute():
        raise ValueError("folder_name must be relative to output_root")
    ensure_not_archived_v1020(output_root / folder_path, label="experiment output")
    layout = (layout or "legacy").lower()
    if layout not in {"v1020", "legacy"}:
        raise ValueError(f"Unknown layout {layout!r}; use 'v1020' or 'legacy'")

    methods = dict(methods)
    validate_grid(
        max_n_nodes=expected_vertex_count(number_of_communities, community_size),
        n_communities=[number_of_communities],
        p_in=[p_in],
        difficulty=[difficulty],
        noises=noises,
        partition_seeds=partition_seeds,
        methods=methods,
        n_runs=n_runs,
    )
    partition_seeds = _normalize_partition_seeds(partition_seeds)
    noises = [float(value) for value in noises]
    expected_vertices = expected_vertex_count(number_of_communities, community_size)
    expected_methods = set(methods)
    pending_cells = {
        (noise, partition_seed)
        for noise in noises
        for partition_seed in partition_seeds
        if not resume
        or not _cell_is_complete(
            output_root=output_root,
            folder_name=folder_name,
            number_of_communities=number_of_communities,
            community_size=community_size,
            noise=noise,
            p_in=p_in,
            difficulty=difficulty,
            seed=seed,
            partition_seed=partition_seed,
            expected_methods=expected_methods,
            expected_vertices=expected_vertices,
            layout=layout,
            n_runs=n_runs,
        )
    }
    if not pending_cells:
        return True

    g = generate_graph(number_of_communities, community_size, p_in, difficulty, seed)
    validate_graph_shape(
        g,
        number_of_communities,
        community_size,
        context="generated SBM graph",
    )
    # generate_graph no longer touches global random state.  Detector restarts
    # still draw from igraph's global generator, so seed it explicitly from
    # the network seed to keep a cell's runs repeatable.
    ig.set_random_number_generator(random.Random(seed))
    gt = get_ground_truth(number_of_communities, community_size, g)
    edge_density = g.density()

    # Collect list-of-results per (noise, partition_seed) for v1020 layout.
    cell_results: dict[tuple[float, int], list[dict]] = defaultdict(list)

    for method_name, method_info in tqdm(
        methods.items(), desc="method", leave=False, total=len(methods)
    ):
        method_call_name = method_info["method_call_name"]
        params = _prepare_method_params(
            method_call_name,
            method_info["parameters"],
            gt=gt,
            edge_density=edge_density,
            number_of_communities=number_of_communities,
        )
        uses_init = "initial_membership" in params
        # Cache for methods that ignore initial membership (GroundTruth, Spectral).
        cached_results: list[dict] | None = None
        runs = _method_n_runs(method_name, n_runs)

        for noise in tqdm(noises, desc="noise", leave=False, total=len(noises)):
            for partition_seed in tqdm(
                partition_seeds, desc="partition_seed", leave=False
            ):
                if (noise, partition_seed) not in pending_cells:
                    continue
                initial_membership = get_initial_membership(gt, noise, partition_seed)
                if uses_init:
                    params["initial_membership"] = initial_membership
                    batch: list[dict] = []
                    saved_partitions: set[tuple] = set()
                    for _ in range(runs):
                        result = get_method_result(
                            g,
                            method_call_name,
                            params,
                            p_in,
                            difficulty,
                            community_size,
                            number_of_communities,
                            gt,
                            display_method=method_name,
                        )
                        result["noise"] = noise
                        result["network_seed"] = seed
                        result["partition_seed"] = partition_seed
                        part_key = tuple(result["partition"])
                        if part_key not in saved_partitions or runs == 1:
                            saved_partitions.add(part_key)
                            batch.append(result)
                    results_for_cell = batch
                else:
                    # Run once per network (old exp.py cached across memberships).
                    if cached_results is None:
                        batch = []
                        saved_partitions = set()
                        for _ in range(runs):
                            result = get_method_result(
                                g,
                                method_call_name,
                                params,
                                p_in,
                                difficulty,
                                community_size,
                                number_of_communities,
                                gt,
                                display_method=method_name,
                            )
                            part_key = tuple(result["partition"])
                            if part_key not in saved_partitions or runs == 1:
                                saved_partitions.add(part_key)
                                batch.append(result)
                        cached_results = batch
                    # Stamp noise / seeds for this cell (matches result schema).
                    results_for_cell = []
                    for base in cached_results:
                        stamped = dict(base)
                        stamped["noise"] = noise
                        stamped["network_seed"] = seed
                        stamped["partition_seed"] = partition_seed
                        results_for_cell.append(stamped)

                if layout == "v1020":
                    cell_results[(noise, partition_seed)].extend(results_for_cell)
                else:
                    relative = _legacy_relative_dir(
                        folder_name,
                        number_of_communities,
                        community_size,
                        noise,
                        p_in,
                        difficulty,
                        seed,
                        partition_seed,
                    )
                    for idx, result in enumerate(results_for_cell):
                        # Multiple stochastic runs → Method.json, Method_01.json, …
                        fname = (
                            f"{method_name}.json"
                            if idx == 0
                            else f"{method_name}_{idx:02d}.json"
                        )
                        file_path = output_root / relative / fname
                        _atomic_json_dump(result, file_path)

    if layout == "v1020":
        for (noise, partition_seed), results in cell_results.items():
            relative = _v1020_relative_dir(
                folder_name,
                number_of_communities,
                community_size,
                noise,
                p_in,
                difficulty,
                seed,
            )
            file_path = (
                output_root / relative / f"partition_{partition_seed:03d}.json"
            )
            _atomic_json_dump(results, file_path)
    return True


def _main_store(
    store_root: Path,
    *,
    spec: dict[str, Any],
    spec_digest: str,
    grid: dict[str, Any],
    methods: dict[str, dict],
    workers: int,
    resume: bool,
    progress=None,
    rerun: Iterable[str] = (),
    reuse: tuple[str, list[str], int] | None = None,
) -> int:
    """Run a grid into a columnar store, one shard per graph.

    ``progress`` (optional) receives ``{"type": "plan", ...}`` once and
    ``{"type": "graph", "label", "status", "summary"}`` after every graph;
    without it a tqdm bar is shown.

    The store is checked before anything is written: a method whose stored rows have another
    identity is refused unless named in ``rerun`` or carried.  ``reuse`` = (source store,
    methods, sample graphs) carries those methods' rows from the source after a replay of a sample
    of them with this build matched exactly; a mismatch refuses the run.
    """
    rerun = list(rerun)
    carry = None
    carried = None
    try:
        store_mod.check_store(store_root, spec, rerun=set(rerun) | set(reuse[1] if reuse else ()))
        if reuse:
            source, names, sample = reuse
            source = Path(source).expanduser().resolve()
            # The source may be this store itself: its rows of another build are then adopted in place.
            print(f"[disjoint] replaying {sample} graph(s) of {source} for {', '.join(names)} ...")
            report = store_mod.verify_carry(source, names, sample_shards=sample)
            if report["clock_differs"]:
                report["problems"].append(
                    f"{source} timed its rows under another rule ({report['timing_scope']}; this build: "
                    f"{spec['timing_scope']}), so their durations do not compare with rows timed now"
                )
            if spec["partitions"] == "full" and report["partitions"] != "full":
                report["problems"].append(
                    f"{source} keeps partition hashes only; this store keeps partitions"
                )
            if report["problems"]:
                shown = "\n  ".join(report["problems"][:10])
                raise ValueError(
                    f"rows of {', '.join(names)} in {source} are not what this build computes:\n  {shown}"
                )
            print(f"[disjoint] {report['rows_checked']} replayed row(s) match; carrying them")
            carry = {"root": str(source), "methods": names, "legacy": report["legacy"]}
            carried = {
                "source": str(source),
                "methods": names,
                "graphs_replayed": report["graphs_checked"],
                "rows_replayed": report["rows_checked"],
            }
        # Held until this process exits (also on kill): one writer per store.
        lock, stale = store_mod.acquire_writer_lock(store_root)
        manifest = store_mod.prepare_store(
            store_root,
            spec,
            grid,
            store_mod.invocation_environment(workers),
            rerun=rerun,
            carried=carried,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    legacy = store_mod.legacy_shard_digests(manifest)
    if stale:
        print(f"[disjoint] removed {stale} temporary shard file(s) of an interrupted run")
    tasks = [
        dict(
            store_root=str(store_root),
            spec=spec,
            spec_digest=spec_digest,
            number_of_communities=k,
            community_size=grid["max_n_nodes"] // k,
            p_in=p_in,
            difficulty=difficulty,
            network_seed=seed,
            noises=grid["noises"],
            partition_seeds=grid["partition_seeds"],
            methods=methods,
            # Rows of another identity are never kept: a refused method never gets here, and
            # a carried or rerun one is recomputed or replaced.
            resume=resume or bool(carry),
            legacy=legacy,
            carry=carry,
        )
        for k in grid["n_communities"]
        for p_in in grid["p_in"]
        for difficulty in grid["difficulty"]
        for seed in grid["seeds"]
    ]
    counts: Counter = Counter()
    bar = None if progress else tqdm(total=len(tasks), desc="graphs", unit="graph")
    if progress:
        progress({"type": "plan", "graphs": len(tasks), "workers": workers,
                  "communities": list(grid["n_communities"]),
                  "per_community": len(tasks) // max(1, len(grid["n_communities"]))})

    def finished(label, status, summary) -> None:
        counts[status] += 1
        if bar is not None:
            bar.update()
        if progress:
            progress({"type": "graph", "label": label, "status": status, "summary": summary})

    if workers == 1:
        for task in tasks:
            finished(*_run_graph_store_task(task))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_graph_store_task, task) for task in tasks]
            for future in as_completed(futures):
                finished(*future.result())
    if bar is not None:
        bar.close()
    print(
        f"[disjoint] store {store_root}: {counts['written']} graph(s) written, "
        f"{counts['carried']} carried, "
        f"{counts['skipped']} already complete (spec {spec_digest[:12]})"
    )
    print(f"[disjoint] inspect: hedonic-exp disjoint-store {store_root}")
    lock.close()
    return 0


# Tiny defaults for isolated / CI-style runs (no large data dirs required).
SMOKE_DEFAULTS = {
    "folder_name": "smoke",
    "max_n_nodes": 24,
    "n_communities": [2],
    "seeds": [0],
    "p_in": [0.35],
    "difficulty": [0.2],
    "noises": [0.01],
    "partition_seeds": 0,
    "methods": ["Mirror", "Hedonic", "Leiden"],
    "n_runs": 1,
    "layout": "legacy",
}


def _parse_methods(spec: str | None | Iterable[str]) -> dict:
    """Resolve a comma-separated method list (or iterable) to a METHODS subset."""
    if not spec:
        return dict(METHODS)
    if isinstance(spec, str):
        names = [n.strip() for n in spec.split(",") if n.strip()]
    else:
        names = [str(n).strip() for n in spec if str(n).strip()]
    unknown = [n for n in names if n not in METHODS]
    if unknown:
        known = ", ".join(METHODS)
        raise SystemExit(f"Unknown --methods: {unknown}. Known: {known}")
    return {n: METHODS[n] for n in names}


def _cli_override(cli_value, base_value):
    """Prefer explicit CLI value when provided (not None)."""
    return base_value if cli_value is None else cli_value


def main(argv=None, progress=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run hedonic game SBM experiments (disjoint). "
            "Use --smoke or --preset v1020-smoke for isolated checks; "
            "--preset v1020 for the full PHYSA V1020 grid."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "presets:\n"
            "  v1020        Full archived grid (1020 nodes, 5× community counts,\n"
            "               10×p_in, 10×difficulty, 5 noises, 5 nets, 10 partitions,\n"
            "               all methods, 10 Leiden/Hedonic runs, v1020 layout).\n"
            "               Very large — write under the artifacts root of the output folder.\n"
            "  v1020-smoke  Tiny structural clone of the V1020 layout/methods.\n"
            "\n"
            "examples:\n"
            "  # Tiny V1020-compatible smoke into a safe root\n"
            "  hedonic-exp disjoint --preset v1020-smoke \\\n"
            "      --output_root artifacts/disjoint/v1020-smoke\n"
            "\n"
            "  # Full V1020 reproduction (do NOT overwrite archived V1020)\n"
            "  hedonic-exp disjoint --preset v1020 \\\n"
            "      --output_root artifacts/disjoint/v1020\n"
            "\n"
            "  # Combine JSON → CSV\n"
            "  hedonic-exp disjoint-load \\\n"
            "      --results_folder artifacts/disjoint/v1020/resultados \\\n"
            "      --output artifacts/disjoint/v1020/resultados.csv.gzip\n"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Isolated mini-run: small graph, few methods, one noise/seed, "
            "legacy layout. Explicit CLI flags still override preset values."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Named parameter grid (v1020 or v1020-smoke). Explicit flags override.",
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        required=False,
        help="Subfolder under output_root (V1020 uses 'resultados')",
        default=None,
    )
    parser.add_argument(
        "--max_n_nodes",
        type=int,
        required=False,
        help="Maximum number of nodes (community_size = max_n_nodes // n_communities)",
        default=None,
    )
    parser.add_argument(
        "--n_communities",
        type=int,
        nargs="+",
        required=False,
        help="Number of clusters",
        default=None,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=False,
        help="Network generation seeds",
        default=None,
    )
    parser.add_argument(
        "--p_in",
        type=float,
        nargs="+",
        required=False,
        help="Probability of edge within communities",
        default=None,
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        nargs="+",
        required=False,
        help="Difficulty of the problem (p_out = p_in * difficulty)",
        default=None,
    )
    parser.add_argument(
        "--noises",
        type=float,
        nargs="+",
        required=False,
        help="Noise levels applied to initial membership",
        default=None,
    )
    parser.add_argument(
        "--partition_seeds",
        type=int,
        default=None,
        help="Number of seeds for initial partition (int count, 0 → [0])",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of methods "
            f"(default: all). Known: {', '.join(METHODS)}"
        ),
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help=(
            "Stochastic restarts for Leiden/Hedonic (V1020 used 10). "
            "Other methods always run once."
        ),
    )
    parser.add_argument(
        "--layout",
        type=str,
        choices=list(LAYOUTS),
        default=None,
        help=(
            "Output layout: store (Parquet shard per graph + manifest; query "
            "with `hedonic-exp disjoint-store`), v1020 (partition_*.json lists "
            "under nC_mN/...) or legacy (Method.json under Partition folders)"
        ),
    )
    parser.add_argument(
        "--partitions",
        choices=list(store_mod.PARTITION_MODES),
        default=None,
        help=(
            "store layout: keep every partition (full, ~0.25 B/vertex) or only "
            "its 128-bit hash (hash; replay by seed when needed)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "store layout: graphs processed in parallel (default 1). Durations "
            "are measured per call and share the CPU with other workers."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Root directory for results (default: artifacts/disjoint under the output folder, HEDONIC_OUTPUT_DIR; "
            "explicit paths are supported for isolated runs)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip existing schema-valid cells (especially useful when recovering "
            "an interrupted v1020 run); corrupt/incomplete cells are rerun."
        ),
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate the selected grid and generated graph sizes, print the "
            "planned cell count, and exit without running detectors or writing files."
        ),
    )
    parser.add_argument(
        "--rerun",
        type=str,
        default=None,
        help=(
            "store layout: comma list of methods whose rows in the store were computed by another "
            "build or setting and are to be recomputed (a run refuses them otherwise)"
        ),
    )
    parser.add_argument(
        "--reuse-from",
        dest="reuse_from",
        type=str,
        default=None,
        help=(
            "store layout: another store to take the rows of --reuse-methods from, after replaying "
            "a sample of them with this build (they must match exactly)"
        ),
    )
    parser.add_argument(
        "--reuse-methods",
        dest="reuse_methods",
        type=str,
        default=None,
        help="store layout: comma list of methods to take from --reuse-from",
    )
    parser.add_argument(
        "--reuse-sample",
        dest="reuse_sample",
        type=int,
        default=5,
        help="graphs of --reuse-from replayed before carrying (default 5; 10 rows per method each)",
    )
    args = parser.parse_args(argv)

    # Resolve defaults: named preset > --smoke > small test defaults.
    if args.preset:
        base = dict(PRESETS[args.preset])
    elif args.smoke:
        base = dict(SMOKE_DEFAULTS)
    else:
        base = {
            "folder_name": "test",
            "max_n_nodes": 60,
            "n_communities": [2],
            "seeds": [42],
            "p_in": [0.1],
            "difficulty": [0.5],
            "noises": [
                0.01, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1,
            ],
            "partition_seeds": 0,
            "methods": list(METHODS.keys()),
            "n_runs": 1,
            "layout": "legacy",
        }

    folder_name = _cli_override(args.folder_name, base["folder_name"])
    max_n_nodes = _cli_override(args.max_n_nodes, base["max_n_nodes"])
    n_communities = _cli_override(args.n_communities, base["n_communities"])
    seeds = _cli_override(args.seeds, base["seeds"])
    p_in_list = _cli_override(args.p_in, base["p_in"])
    difficulty_list = _cli_override(args.difficulty, base["difficulty"])
    noises = _cli_override(args.noises, base["noises"])
    partition_seeds = _cli_override(args.partition_seeds, base["partition_seeds"])
    n_runs = _cli_override(args.n_runs, base.get("n_runs", 1))
    layout = _cli_override(args.layout, base.get("layout", "legacy"))
    partitions = _cli_override(args.partitions, base.get("partitions", "full"))
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    if args.methods is not None:
        methods = _parse_methods(args.methods)
    elif args.smoke and not args.preset:
        methods = _parse_methods(SMOKE_DEFAULTS["methods"])
    else:
        methods = _parse_methods(base.get("methods"))

    output_root = (
        expand_path(args.output_root) if args.output_root else DISJOINT_ARTIFACTS_DIR
    ).resolve()

    try:
        ensure_not_archived_v1020(output_root, label="output root")
        folder_path = Path(str(folder_name))
        if folder_path.is_absolute():
            raise ValueError("folder_name must be relative to output_root")
        ensure_not_archived_v1020(
            output_root / folder_path,
            label="experiment output",
        )
        n_communities = validate_grid(
            max_n_nodes=max_n_nodes,
            n_communities=n_communities,
            p_in=p_in_list,
            difficulty=difficulty_list,
            noises=noises,
            partition_seeds=partition_seeds,
            methods=methods,
            n_runs=n_runs,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    spec = store_mod.build_spec(
        methods=methods,
        n_runs=n_runs,
        partitions=partitions,
        generator_id=SBM_GENERATOR_ID,
    )
    spec_digest = store_mod.spec_hash(spec)
    by_lower = {name.lower(): name for name in METHODS}

    def named(text: str | None) -> list[str]:  # --rerun/--reuse-methods ignore case
        out = [by_lower.get(part.strip().lower(), part.strip()) for part in (text or "").split(",")]
        return list(_parse_methods([part for part in out if part]))

    rerun = named(args.rerun) if args.rerun else []
    reuse = named(args.reuse_methods) if args.reuse_methods else []
    if bool(args.reuse_from) != bool(reuse):
        raise SystemExit("--reuse-from and --reuse-methods go together")
    if layout != "store" and (rerun or reuse):
        raise SystemExit("--rerun and --reuse-from need the store layout")
    if reuse:
        unknown = sorted(set(reuse) - set(methods))
        if unknown:
            raise SystemExit(f"--reuse-methods names methods this run does not: {', '.join(unknown)}")
    store_state = None
    if layout == "store":
        try:
            store_state = store_mod.check_store(
                output_root / folder_name, spec, rerun=set(rerun) | set(reuse)
            )
        except store_mod.StoreConflict as exc:
            if not args.preflight:
                raise SystemExit(str(exc)) from exc
            store_state = {"refused": str(exc)}

    if args.preflight:
        try:
            plan = preflight_grid(
                folder_name=folder_name,
                max_n_nodes=max_n_nodes,
                n_communities=n_communities,
                seeds=seeds,
                p_in=p_in_list,
                difficulty=difficulty_list,
                noises=noises,
                partition_seeds=partition_seeds,
                methods=methods,
                n_runs=n_runs,
                layout=layout,
                output_root=output_root,
                check_graphs=True,
                spec=spec,
                # The seeded generator yields k × size vertices by
                # construction; sample instead of building 50,000 graphs.
                max_graph_checks=20 if layout == "store" else None,
            )
        except (ValueError, OSError) as exc:
            raise SystemExit(f"Preflight failed: {exc}") from exc
        print("[disjoint] preflight passed (no detectors run; no files written)")
        print(
            f"[disjoint] graph_tasks={plan['graph_tasks']} "
            f"cells={plan['expected_cells']} "
            f"complete={plan['complete_cells']} "
            f"pending={plan['pending_cells']} "
            f"graph_checks={plan['graph_checks']}"
        )
        print(
            f"[disjoint] rows_per_graph={plan['rows_per_graph']} "
            f"expected_rows={plan['expected_rows']}"
        )
        if plan.get("method_graphs"):
            print(f"[disjoint] store {output_root / folder_name}: graphs per method")
            for name, counts in plan["method_graphs"].items():
                print(
                    f"  {name:<12} current {counts['current']:>6}  "
                    f"other build/setting {counts['other']:>6}  missing {counts['missing']:>6}"
                )
        if store_state and store_state.get("refused"):
            print(f"[disjoint] a run would be refused: {store_state['refused']}")
        elif store_state:
            for key, label in (("changed", "recomputed (--rerun)"), ("new", "new to the store")):
                if store_state.get(key):
                    print(f"[disjoint] {label}: {', '.join(store_state[key])}")
        if plan["malformed_graphs"]:
            print("[disjoint] malformed graphs:")
            for item in plan["malformed_graphs"]:
                print(f"  {item}")
            return 1
        return 0

    print(
        f"[disjoint] preset={args.preset or ('smoke' if args.smoke else 'custom')} "
        f"layout={layout} nodes≤{max_n_nodes} n_comm={list(n_communities)} "
        f"methods={list(methods)} n_runs={n_runs} → {output_root / folder_name}"
    )
    if args.preset in {"v1020", "sbm"}:
        print(
            f"[disjoint] WARNING: the full {args.preset} grid is very large "
            "(run --preflight first; `hedonic show disjoint` gives its size and time)."
        )

    if layout == "store":
        return _main_store(
            output_root / folder_name,
            spec=spec,
            spec_digest=spec_digest,
            grid={
                "max_n_nodes": int(max_n_nodes),
                "n_communities": [int(k) for k in n_communities],
                "seeds": [int(seed) for seed in seeds],
                "p_in": [float(p) for p in p_in_list],
                "difficulty": [float(d) for d in difficulty_list],
                "noises": [float(n) for n in noises],
                "partition_seeds": _normalize_partition_seeds(partition_seeds),
            },
            methods=methods,
            workers=args.workers,
            resume=args.resume,
            progress=progress,
            rerun=rerun,
            reuse=(args.reuse_from, reuse, args.reuse_sample) if reuse else None,
        )

    for n_community in tqdm(n_communities, desc="n_community", leave=False):
        community_size = int(max_n_nodes / n_community)
        for p_in in tqdm(p_in_list, desc="p_in", leave=False):
            for difficulty in tqdm(difficulty_list, desc="difficulty", leave=False):
                for seed in tqdm(seeds, desc="seed", leave=False):
                    run_experiment(
                        folder_name,
                        n_community,
                        community_size,
                        p_in,
                        difficulty,
                        methods,
                        noises,
                        partition_seeds,
                        seed,
                        output_root=output_root,
                        n_runs=n_runs,
                        layout=layout,
                        resume=args.resume,
                    )
    print("Experiments completed successfully.")
    return True


if __name__ == "__main__":
    main()
