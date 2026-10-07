"""Native overlapping Leiden versus the independent unit-ℓ₂ oracle.

This adapter is imported only by the overlapping-oracle CLI flag and by
tests.  It is not imported by :mod:`unit_l2_oracle`, so the mathematical
oracle stays free of igraph.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import igraph as ig
import numpy as np

from hedonic import Game
from hedonic.experiments.overlapping import unit_l2_oracle as oracle
from hedonic.experiments.overlapping.robustness import audit_cover


def _graph_from_named(edges: Sequence[Sequence[float]], n: int) -> tuple[Game, list[float]]:
    pairs = [(int(row[0]), int(row[1])) for row in edges]
    weights = [float(row[2]) for row in edges]
    graph = Game(ig.Graph(n=n, edges=pairs))
    if weights:
        graph.es["weight"] = weights
    return graph, weights


def _memberships_from_cover(cover) -> list[list[int]]:
    return [[int(label) for label in labels] for labels in cover.membership]


def _oracle_max_regret(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    cap: int,
    *,
    allow_isolation: bool,
) -> float:
    used = max((max(row) if row else -1) for row in state) + 1
    label_count = max(used, 1) + (int(cap) if allow_isolation else 0)
    label_count = max(label_count, cap)
    regrets = oracle.vertex_regrets(
        adjacency, weights, gamma, state, label_count, cap
    )
    return max(item["regret"] for item in regrets)


def _run_native(
    graph: Game,
    *,
    initial: list[list[int]],
    cap: int,
    gamma: float,
    allow_isolation: bool,
    local_move_only: bool,
    edge_weights: list[float] | None,
    seed: int,
) -> dict[str, Any]:
    ig.set_random_number_generator(random.Random(seed))
    cover = graph.community_hedonic(
        initial_membership=initial,
        max_memberships=cap,
        n_iterations=-1,
        resolution=gamma,
        allow_isolation=allow_isolation,
        local_move_only=local_move_only,
        edge_weights=edge_weights,
    )
    memberships = _memberships_from_cover(cover)
    audit = audit_cover(
        graph,
        memberships,
        max_memberships=cap,
        allow_isolation=allow_isolation,
        gamma=gamma,
        dense=True,
        edge_weights=edge_weights,
    )
    adjacency = np.zeros((graph.vcount(), graph.vcount()), dtype=float)
    weight_list = edge_weights if edge_weights is not None else [1.0] * graph.ecount()
    for (src, dst), weight in zip(graph.get_edgelist(), weight_list):
        adjacency[src, dst] = float(weight)
        adjacency[dst, src] = float(weight)
    vertex_weights = np.ones(graph.vcount())
    oracle_regret = _oracle_max_regret(
        adjacency,
        vertex_weights,
        gamma,
        memberships,
        cap,
        allow_isolation=allow_isolation,
    )
    audit_regret = float(audit["max_positive_regret_at_resolution"])
    return {
        "native_memberships": memberships,
        "native_quality": getattr(cover, "quality", None)
        or (cover._params or {}).get("quality"),
        "token_preflight": getattr(cover, "_hedonic_token_preflight", None),
        "raw_memberships": getattr(cover, "_hedonic_raw_memberships", None),
        "audit_equilibrium": bool(audit["is_local_equilibrium_at_resolution"]),
        "audit_max_regret": audit_regret,
        "oracle_max_regret": oracle_regret,
        "oracles_agree": abs(audit_regret - oracle_regret) < 1e-8
        or (
            audit_regret < 1e-10
            and oracle_regret < 1e-10
        ),
    }


def ce1_records(*, seed: int = 20260905) -> list[dict[str, Any]]:
    """CE1 weighted/unweighted under local and multi-phase native search."""
    records = []
    weighted = oracle.ce1_weighted()
    unweighted = oracle.ce1_unweighted()
    for example, name in ((weighted, "ce1_weighted"), (unweighted, "ce1_unweighted")):
        graph, weights = _graph_from_named(example["edges"], example["n"])
        trap = [list(row) for row in example["state"]]
        for local_move_only in (True, False):
            for allow_isolation in (False, True):
                native = _run_native(
                    graph,
                    initial=trap,
                    cap=int(example["cap"]),
                    gamma=float(example["gamma"]),
                    allow_isolation=allow_isolation,
                    local_move_only=local_move_only,
                    edge_weights=weights,
                    seed=seed,
                )
                records.append(
                    {
                        "case": name,
                        "local_move_only": local_move_only,
                        "allow_isolation": allow_isolation,
                        "trap": trap,
                        "escaped_trap": native["native_memberships"] != trap,
                        "native_certified_false_on_trap": not (
                            native["native_memberships"] == trap
                            and native["audit_equilibrium"]
                        ),
                        **native,
                    }
                )
    return records


def random_tiny_records(
    *,
    cases: int = 8,
    seed: int = 20260907,
) -> list[dict[str, Any]]:
    """Random tiny graphs: native return vs both independent auditors."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(cases):
        n_vertices = int(rng.integers(3, 6))
        cap = int(rng.integers(2, 4))
        adjacency = np.triu(rng.integers(0, 3, (n_vertices, n_vertices)), 1)
        adjacency = adjacency + adjacency.T
        edges = [
            (i, j, float(adjacency[i, j]))
            for i in range(n_vertices)
            for j in range(i + 1, n_vertices)
            if adjacency[i, j] != 0
        ]
        if not edges:
            continue
        graph, weights = _graph_from_named(edges, n_vertices)
        initial = [[int(v)] for v in range(n_vertices)]
        gamma = float(graph.density())
        local_move_only = bool(index % 2 == 0)
        allow_isolation = bool(index % 3 == 0)
        native = _run_native(
            graph,
            initial=initial,
            cap=cap,
            gamma=gamma,
            allow_isolation=allow_isolation,
            local_move_only=local_move_only,
            edge_weights=weights,
            seed=seed + index,
        )
        records.append(
            {
                "case": f"random_{index}",
                "n": n_vertices,
                "cap": cap,
                "gamma": gamma,
                "local_move_only": local_move_only,
                "allow_isolation": allow_isolation,
                **native,
            }
        )
    return records


def run_native_differential(
    *,
    random_cases: int = 8,
    seed: int = 20260907,
) -> list[dict[str, Any]]:
    """CE1 plus random tiny native/oracle comparisons."""
    records = ce1_records(seed=seed)
    records.extend(random_tiny_records(cases=random_cases, seed=seed))
    if any(
        item["case"].startswith("ce1_")
        and item["native_memberships"] == item["trap"]
        and item["audit_equilibrium"]
        for item in records
        if "trap" in item
    ):
        raise AssertionError("native/auditor issued a false CE1 certificate")
    if any(not item["oracles_agree"] for item in records):
        disagree = [item["case"] for item in records if not item["oracles_agree"]]
        raise AssertionError(f"paper auditor and NumPy oracle disagree: {disagree}")
    return records


def write_native_differential(path: Path, records: list[dict[str, Any]] | None = None) -> Path:
    payload = records if records is not None else run_native_differential()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in payload:
            handle.write(json.dumps(record, default=_json_default) + "\n")
    return path


def _json_default(value: Any):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))
