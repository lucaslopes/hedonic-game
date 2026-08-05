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
``.../V1020_CLI`` rather than overwriting the archived V1020 folder.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import igraph as ig
import numpy as np
from igraph import compare_communities
from igraph.clustering import VertexClustering
from stopwatch import Stopwatch
from tqdm import tqdm

from hedonic import Game
from hedonic.experiments.config import SYNTHETIC_DIR

# ---------------------------------------------------------------------------
# Default method map (adapted to current lucas-igraph Leiden kwargs)
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
        # Full Leiden via community_hedonic (local_move_only=False)
        # Old tmp/hedonic used community_leiden(only_first_phase=False).
        "method_call_name": "community_hedonic",
        "parameters": {
            "resolution": 1,
            "beta": 0.01,
            "initial_membership": None,
            "n_iterations": -1,
            "local_move_only": False,
            "allow_isolation": False,
            "max_memberships": 1,
        },
    },
    "Hedonic": {
        # Primary exploratory method: hedonic local-moving phase
        # Old tmp/hedonic used community_leiden(only_first_phase=True).
        "method_call_name": "community_hedonic",
        "parameters": {
            "resolution": 1,
            "beta": 0.01,
            "initial_membership": None,
            "n_iterations": -1,
            "local_move_only": True,
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

PRESETS: dict[str, dict[str, Any]] = {
    "v1020": V1020_PRESET,
    "v1020-smoke": V1020_SMOKE_PRESET,
}


# ---------------------------------------------------------------------------
# Graph generation & partitions
# ---------------------------------------------------------------------------

def probs_matrix(n_communities: int, p: float, q: float) -> list[list[float]]:
    return [
        [p if i == j else q for j in range(n_communities)]
        for i in range(n_communities)
    ]


def generate_graph(
    n_communities: int,
    community_size: int,
    p_in: float,
    multiplier: float,
    seed: int,
) -> Game:
    """Generate an SBM graph as a hedonic.Game (igraph SBM, no networkx)."""
    block_sizes = [community_size] * n_communities
    p_out = p_in * multiplier
    pref = probs_matrix(n_communities, p_in, p_out)
    # igraph RNG expects a random.Random-like object (with randint).
    rng = random.Random(seed)
    ig.set_random_number_generator(rng)
    np.random.seed(seed)
    # lucas-igraph / python-igraph ≥0.11: SBM(pref_matrix, block_sizes, ...)
    g = ig.Graph.SBM(pref, block_sizes, directed=False)
    return Game(g)


def get_ground_truth(
    number_of_communities: int,
    community_size: int,
    g: Game | None = None,
):
    gt_membership = np.concatenate(
        [np.full(community_size, i) for i in range(number_of_communities)]
    ).tolist()
    if g is not None:
        return VertexClustering(g, gt_membership)
    return gt_membership


def shuffle_with_noise(membership, noise=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    community_dict = defaultdict(list)
    for node, community in enumerate(membership):
        community_dict[community].append(node)

    for community_nodes in community_dict.values():
        np.random.shuffle(community_nodes)

    shuffled_membership = [None] * len(membership)
    for community, nodes in community_dict.items():
        for node in nodes:
            shuffled_membership[node] = community

    n = len(membership)
    num_to_shuffle = int(noise * n)
    if num_to_shuffle <= 0:
        return shuffled_membership

    indices_to_shuffle = np.random.choice(range(n), size=num_to_shuffle, replace=False)
    shuffled_indices = np.random.permutation(indices_to_shuffle)
    for i, j in zip(indices_to_shuffle, shuffled_indices):
        shuffled_membership[i], shuffled_membership[j] = (
            shuffled_membership[j],
            shuffled_membership[i],
        )
    return shuffled_membership


def get_initial_membership(ground_truth, noise=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
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


def robustness(g: Game, partition) -> float:
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


_LOCAL_METHODS = {
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
    stopwatch = Stopwatch()
    stopwatch.start()
    try:
        partition = _call_method(g, method_name, method_params)
    except Exception as e:
        partition = VertexClustering(g, [0] * g.vcount())
        print(
            f"\nPARTITIONING ERROR:\n{e}\n{method_name=}\n{p_in=}\n"
            f"{multiplier=}\n{community_size=}\n{number_of_communities=}"
        )
    stopwatch.stop()
    acc = accuracy(g, partition, ground_truth)
    rob = robustness(g, partition)
    label = display_method
    if label is None:
        label = method_name.split("_")[1] if "_" in method_name else method_name
    return {
        "method": label,
        "number_of_communities": number_of_communities,
        "community_size": community_size,
        "p_in": p_in,
        "p_out": p_in * multiplier,
        "multiplier": multiplier,
        "resolution": method_params.get("resolution"),
        "duration": stopwatch.duration,
        "accuracy": acc,
        "robustness": rob,
        "partition": list(partition.membership),
    }


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
    if method_call_name in {"community_leiden", "community_hedonic"}:
        params["resolution"] = edge_density
    return params


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
    """
    if methods is None:
        methods = METHODS
    if noises is None:
        noises = [0]
    partition_seeds = _normalize_partition_seeds(partition_seeds)
    output_root = Path(output_root) if output_root is not None else SYNTHETIC_DIR
    layout = (layout or "legacy").lower()
    if layout not in {"v1020", "legacy"}:
        raise ValueError(f"Unknown layout {layout!r}; use 'v1020' or 'legacy'")

    g = generate_graph(number_of_communities, community_size, p_in, difficulty, seed)
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
                        os.makedirs(file_path.parent, exist_ok=True)
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(json.dumps(result))

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
            os.makedirs(file_path.parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(results, file)
    return True


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


def main(argv=None):
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
            "               Very large — write to a NEW root (e.g. V1020_CLI).\n"
            "  v1020-smoke  Tiny structural clone of the V1020 layout/methods.\n"
            "\n"
            "examples:\n"
            "  # Tiny V1020-compatible smoke into a safe root\n"
            "  hedonic-exp disjoint --preset v1020-smoke \\\n"
            "      --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI\n"
            "\n"
            "  # Full V1020 reproduction (do NOT overwrite archived V1020)\n"
            "  hedonic-exp disjoint --preset v1020 \\\n"
            "      --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI\n"
            "\n"
            "  # Combine JSON → CSV\n"
            "  hedonic-exp disjoint-load \\\n"
            "      --results_folder .../V1020_CLI/resultados \\\n"
            "      --output .../V1020_CLI/resultados.csv.gzip\n"
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
        choices=["v1020", "legacy"],
        default=None,
        help=(
            "Output layout: v1020 (partition_*.json lists under nC_mN/...) "
            "or legacy (Method.json under Partition folders)"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for results (default: HEDONIC_SYNTHETIC_DIR / SYNTHETIC_DIR)",
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

    if args.methods is not None:
        methods = _parse_methods(args.methods)
    elif args.smoke and not args.preset:
        methods = _parse_methods(SMOKE_DEFAULTS["methods"])
    else:
        methods = _parse_methods(base.get("methods"))

    output_root = Path(args.output_root) if args.output_root else SYNTHETIC_DIR

    # Guard: refuse to write the full V1020 preset into the archived folder.
    archived = Path(
        str(Path("~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020").expanduser())
    ).resolve()
    if args.preset == "v1020" and output_root.resolve() == archived:
        raise SystemExit(
            "Refusing to write --preset v1020 into the archived V1020 folder.\n"
            "Pass a different --output_root, e.g.\n"
            "  --output_root ~/Databases/Hedonic/PHYSA/Synthetic_Networks/V1020_CLI"
        )

    print(
        f"[disjoint] preset={args.preset or ('smoke' if args.smoke else 'custom')} "
        f"layout={layout} nodes≤{max_n_nodes} n_comm={list(n_communities)} "
        f"methods={list(methods)} n_runs={n_runs} → {output_root / folder_name}"
    )
    if args.preset == "v1020":
        print(
            "[disjoint] WARNING: full V1020 grid is very large "
            "(thousands of cells × methods × stochastic runs)."
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
                    )
    print("Experiments completed successfully.")
    return True


if __name__ == "__main__":
    main()
