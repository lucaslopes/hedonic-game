"""SBM parameter sweep for disjoint hedonic community detection.

Migrated from tmp/hedonic/scripts/experiment.py and config.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

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
        # Full Leiden via community_hedonic (only_local_moving=False)
        "method_call_name": "community_hedonic",
        "parameters": {
            "resolution": 1,
            "beta": 0.01,
            "initial_membership": None,
            "n_iterations": -1,
            "only_local_moving": False,
            "allow_isolation": False,
            "max_memberships": 1,
        },
    },
    "Hedonic": {
        # Primary exploratory method: hedonic local-moving phase
        "method_call_name": "community_hedonic",
        "parameters": {
            "resolution": 1,
            "beta": 0.01,
            "initial_membership": None,
            "n_iterations": -1,
            "only_local_moving": True,
            "allow_isolation": False,
            "max_memberships": 1,
        },
    },
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
    return {
        "method": method_name.split("_")[1] if "_" in method_name else method_name,
        "number_of_communities": number_of_communities,
        "community_size": community_size,
        "p_in": p_in,
        "p_out": p_in * multiplier,
        "multiplier": multiplier,
        "resolution": method_params.get("resolution"),
        "duration": stopwatch.duration,
        "accuracy": acc,
        "robustness": rob,
        "partition": partition.membership,
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
):
    if methods is None:
        methods = METHODS
    if noises is None:
        noises = [0]
    if partition_seeds is None:
        partition_seeds = [0]
    if isinstance(partition_seeds, int):
        partition_seeds = list(range(partition_seeds)) if partition_seeds > 0 else [0]

    output_root = Path(output_root) if output_root is not None else SYNTHETIC_DIR

    g = generate_graph(number_of_communities, community_size, p_in, difficulty, seed)
    gt = get_ground_truth(number_of_communities, community_size, g)
    edge_density = g.density()

    for method_name, method_info in tqdm(
        methods.items(), desc="method", leave=False, total=len(methods)
    ):
        result = None
        method_call_name = method_info["method_call_name"]
        parameters = method_info["parameters"]
        params = dict(parameters)
        if method_call_name == "community_groundtruth":
            params["groundtruth"] = gt
        if method_call_name in {"community_leiden", "community_hedonic"}:
            params["resolution"] = edge_density
        for noise in tqdm(noises, desc="noise", leave=False, total=len(noises)):
            initial_memberships = [
                get_initial_membership(gt, noise, partition_seed)
                for partition_seed in partition_seeds
            ]
            for partition_idx, initial_membership in enumerate(
                tqdm(initial_memberships, desc="partition_seed", leave=False)
            ):
                if "initial_membership" in params:
                    params["initial_membership"] = initial_membership
                    result = None
                if result is None:
                    result = get_method_result(
                        g,
                        method_call_name,
                        params,
                        p_in,
                        difficulty,
                        community_size,
                        number_of_communities,
                        gt,
                    )
                result = dict(result)
                result["method"] = method_name
                result["noise"] = noise
                result["network_seed"] = seed
                result["partition_seed"] = partition_seeds[partition_idx]
                relative = (
                    Path(folder_name)
                    / f"{number_of_communities} Communities of {community_size} nodes"
                    / f"Noise = {noise:.2f}"
                    / f"P_in = {p_in:.2f}"
                    / f"Difficulty = {difficulty:.2f}"
                    / f"Network ({seed:03d})"
                    / f"Partition ({partition_seeds[partition_idx]:03d})"
                )
                file_path = output_root / relative / f"{method_name}.json"
                os.makedirs(file_path.parent, exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(json.dumps(result))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run hedonic game SBM experiments.")
    parser.add_argument(
        "--folder_name",
        type=str,
        required=False,
        help="Name of the folder to store results",
        default="test",
    )
    parser.add_argument(
        "--max_n_nodes",
        type=int,
        required=False,
        help="Maximum number of nodes",
        default=60,
    )
    parser.add_argument(
        "--n_communities",
        type=int,
        nargs="+",
        required=False,
        help="Number of clusters",
        default=[2],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=False,
        help="Seeds",
        default=[42],
    )
    parser.add_argument(
        "--p_in",
        type=float,
        nargs="+",
        required=False,
        help="Probability of edge within communities",
        default=[0.1],
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        nargs="+",
        required=False,
        help="Difficulty of the problem",
        default=[0.5],
    )
    parser.add_argument(
        "--noises",
        type=float,
        nargs="+",
        required=False,
        help="Noise levels",
        default=[
            0.01, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1,
        ],
    )
    parser.add_argument(
        "--partition_seeds",
        type=int,
        default=0,
        help="Number of seeds for initial partition (int count, 0 → [0])",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for results (default: HEDONIC_SYNTHETIC_DIR / SYNTHETIC_DIR)",
    )
    args = parser.parse_args(argv)

    output_root = Path(args.output_root) if args.output_root else SYNTHETIC_DIR
    for n_community in tqdm(args.n_communities, desc="n_community", leave=False):
        community_size = int(args.max_n_nodes / n_community)
        for p_in in tqdm(args.p_in, desc="p_in", leave=False):
            for difficulty in tqdm(args.difficulty, desc="difficulty", leave=False):
                for seed in tqdm(args.seeds, desc="seed", leave=False):
                    run_experiment(
                        args.folder_name,
                        n_community,
                        community_size,
                        p_in,
                        difficulty,
                        METHODS,
                        args.noises,
                        args.partition_seeds,
                        seed,
                        output_root=output_root,
                    )
    print("Experiments completed successfully.")
    return True


if __name__ == "__main__":
    main()
