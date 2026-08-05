"""Locked small-instance diagnostic for the DNN certificate in thesis Eq. 5.11.

This module deliberately targets tiny graphs.  It completely enumerates the
chapter's equal-intensity valid covers, runs the shipped hedonic detector from
seeded disjoint warm starts, and solves the doubly-nonnegative relaxation.  It
does not treat the SDP matrix as a cover and does not claim that the relaxation
is tight.

CVXPY and SCS are optional experiment dependencies and are imported only when
the SDP is solved, keeping :mod:`hedonic` importable with its core dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_VERSION = "dnn-certificate-v1"
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_EPS = 1e-8
DEFAULT_MAX_ITERS = 200_000


@dataclass(frozen=True)
class LockedInstance:
    """An immutable, JSON-serializable small weighted graph specification."""

    name: str
    n_vertices: int
    edges: tuple[tuple[int, int], ...]
    edge_weights: tuple[float, ...]
    vertex_weights: tuple[float, ...]
    resolution: float
    max_labels: int
    max_memberships: int


# These instances are intentionally literal rather than RNG-generated.  Changing
# any edge, weight, resolution, or cap changes the recorded SHA-256 identity.
LOCKED_INSTANCES: dict[str, LockedInstance] = {
    "path4": LockedInstance(
        name="path4",
        n_vertices=4,
        edges=((0, 1), (1, 2), (2, 3)),
        edge_weights=(1.0, 1.0, 1.0),
        vertex_weights=(1.0, 1.0, 1.0, 1.0),
        resolution=0.35,
        max_labels=2,
        max_memberships=2,
    ),
    "bow_tie5": LockedInstance(
        name="bow_tie5",
        n_vertices=5,
        edges=((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)),
        edge_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        vertex_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        resolution=0.40,
        max_labels=2,
        max_memberships=2,
    ),
    "weighted_bridge5": LockedInstance(
        name="weighted_bridge5",
        n_vertices=5,
        edges=((0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)),
        edge_weights=(1.0, 0.8, 1.2, 0.7, 1.1, 1.0),
        vertex_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        resolution=0.32,
        max_labels=3,
        max_memberships=2,
    ),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def instance_record(instance: LockedInstance) -> dict[str, Any]:
    record = asdict(instance)
    record["edges"] = [list(edge) for edge in instance.edges]
    record["edge_weights"] = list(instance.edge_weights)
    record["vertex_weights"] = list(instance.vertex_weights)
    return record


def instance_sha256(instance: LockedInstance) -> str:
    return sha256_json(instance_record(instance))


def implementation_sha256() -> str:
    """Bind an output manifest to the exact experiment module bytes."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def adjacency_matrix(instance: LockedInstance) -> np.ndarray:
    adjacency = np.zeros((instance.n_vertices, instance.n_vertices), dtype=float)
    for (source, target), weight in zip(instance.edges, instance.edge_weights):
        if source == target:
            raise ValueError(f"{instance.name}: self loops are not supported")
        adjacency[source, target] += weight
        adjacency[target, source] += weight
    return adjacency


def membership_matrix(
    memberships: Iterable[Iterable[int]], max_labels: int
) -> np.ndarray:
    rows = [tuple(sorted(set(int(label) for label in labels))) for labels in memberships]
    matrix = np.zeros((len(rows), max_labels), dtype=float)
    for vertex, labels in enumerate(rows):
        if not labels:
            raise ValueError(f"vertex {vertex} has no memberships")
        if labels[0] < 0 or labels[-1] >= max_labels:
            raise ValueError(f"vertex {vertex} uses a label outside [0, {max_labels})")
        matrix[vertex, list(labels)] = 1.0 / math.sqrt(len(labels))
    return matrix


def gram_objective(instance: LockedInstance, gram: np.ndarray) -> float:
    """Evaluate Eq. 5.10 using the symmetric zero-diagonal adjacency matrix."""

    total_edge_weight = float(sum(instance.edge_weights))
    if total_edge_weight <= 0:
        raise ValueError(f"{instance.name}: positive total edge weight is required")
    vertex_weights = np.asarray(instance.vertex_weights, dtype=float)
    coefficient = adjacency_matrix(instance) - instance.resolution * np.outer(
        vertex_weights, vertex_weights
    )
    return float(np.sum(coefficient * gram) / (2.0 * total_edge_weight))


def cover_objective(
    instance: LockedInstance, memberships: Iterable[Iterable[int]]
) -> tuple[float, np.ndarray, np.ndarray]:
    factor = membership_matrix(memberships, instance.max_labels)
    gram = factor @ factor.T
    return gram_objective(instance, gram), factor, gram


def _membership_options(max_labels: int, max_memberships: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        labels
        for size in range(1, min(max_labels, max_memberships) + 1)
        for labels in itertools.combinations(range(max_labels), size)
    )


def exact_valid_cover_optimum(instance: LockedInstance) -> dict[str, Any]:
    """Completely enumerate every labelled cover within the locked caps."""

    options = _membership_options(instance.max_labels, instance.max_memberships)
    best_value = -math.inf
    best_memberships: tuple[tuple[int, ...], ...] | None = None
    best_factor: np.ndarray | None = None
    best_gram: np.ndarray | None = None
    ties = 0
    tolerance = 1e-13

    for candidate in itertools.product(options, repeat=instance.n_vertices):
        value, factor, gram = cover_objective(instance, candidate)
        if value > best_value + tolerance:
            best_value = value
            best_memberships = candidate
            best_factor = factor
            best_gram = gram
            ties = 1
        elif abs(value - best_value) <= tolerance:
            ties += 1
            if best_memberships is None or candidate < best_memberships:
                best_memberships = candidate
                best_factor = factor
                best_gram = gram

    assert best_memberships is not None and best_factor is not None and best_gram is not None
    return {
        "method": "complete_labelled_cover_enumeration",
        "candidate_count": len(options) ** instance.n_vertices,
        "membership_options_per_vertex": [list(option) for option in options],
        "tie_count_in_labelled_search": ties,
        "objective": best_value,
        "memberships_by_vertex": [list(labels) for labels in best_memberships],
        "factor": best_factor.tolist(),
        "gram": best_gram.tolist(),
        "verification": cover_feasibility(instance, best_memberships, best_gram),
    }


def cover_feasibility(
    instance: LockedInstance,
    memberships: Iterable[Iterable[int]],
    gram: np.ndarray | None = None,
) -> dict[str, Any]:
    rows = [tuple(labels) for labels in memberships]
    valid_nonempty = len(rows) == instance.n_vertices and all(rows)
    valid_cap = valid_nonempty and all(
        len(set(labels)) <= instance.max_memberships for labels in rows
    )
    valid_labels = valid_nonempty and all(
        all(0 <= int(label) < instance.max_labels for label in labels) for labels in rows
    )
    factor = membership_matrix(rows, instance.max_labels)
    expected_gram = factor @ factor.T
    if gram is None:
        gram = expected_gram
    return {
        "valid": bool(valid_nonempty and valid_cap and valid_labels),
        "nonempty_memberships": bool(valid_nonempty),
        "membership_cap": bool(valid_cap),
        "label_bound": bool(valid_labels),
        "factor_row_norm_max_error": float(
            np.max(np.abs(np.sum(factor * factor, axis=1) - 1.0))
        ),
        "gram_factorization_max_error": float(np.max(np.abs(gram - expected_gram))),
    }


def _seeded_disjoint_warm_start(instance: LockedInstance, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    labels = np.arange(instance.n_vertices, dtype=int) % instance.max_labels
    rng.shuffle(labels)
    return [[int(label)] for label in labels]


def run_hedonic(instance: LockedInstance, seed: int) -> dict[str, Any]:
    """Run the repository's one public detector API and recompute its objective."""

    import igraph as ig

    from hedonic import Game

    graph = ig.Graph(n=instance.n_vertices, edges=list(instance.edges), directed=False)
    graph.es["weight"] = list(instance.edge_weights)
    game = Game(graph)
    initial = _seeded_disjoint_warm_start(instance, seed)
    result = game.community_hedonic(
        resolution=instance.resolution,
        max_memberships=instance.max_memberships,
        initial_membership=initial,
        local_move_only=False,
        allow_isolation=True,
        n_iterations=-1,
        edge_weights="weight",
        seed=seed,
    )
    memberships = [list(map(int, labels)) for labels in result.membership]
    value, factor, gram = cover_objective(instance, memberships)
    native_quality = (getattr(result, "_params", None) or {}).get("quality")
    return {
        "seed": seed,
        "initial_memberships_by_vertex": initial,
        "parameters": {
            "resolution": instance.resolution,
            "max_memberships": instance.max_memberships,
            "local_move_only": False,
            "allow_isolation": True,
            "n_iterations": -1,
            "edge_weights": "weight",
        },
        "memberships_by_vertex": memberships,
        "cover_by_label": [list(map(int, community)) for community in result],
        "objective_recomputed_eq_5_10": value,
        "native_quality": None if native_quality is None else float(native_quality),
        "factor": factor.tolist(),
        "gram": gram.tolist(),
        "verification": cover_feasibility(instance, memberships, gram),
    }


def _finite_json(value: Any) -> Any:
    """Convert NumPy objects and non-finite solver diagnostics to strict JSON."""

    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


def solve_dnn(
    instance: LockedInstance,
    *,
    eps: float = DEFAULT_EPS,
    max_iters: int = DEFAULT_MAX_ITERS,
) -> dict[str, Any]:
    """Solve Eq. 5.11 with SCS and construct a checked feasible dual repair."""

    try:
        import cvxpy as cp
        import scs
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            'DNN solving requires the experiments extra: uv sync --extra experiments'
        ) from exc

    n = instance.n_vertices
    total_edge_weight = float(sum(instance.edge_weights))
    vertex_weights = np.asarray(instance.vertex_weights, dtype=float)
    coefficient = (
        adjacency_matrix(instance)
        - instance.resolution * np.outer(vertex_weights, vertex_weights)
    ) / (2.0 * total_edge_weight)

    gram = cp.Variable((n, n), symmetric=True, name="Q")
    psd_constraint = gram >> 0
    nonnegative_constraint = gram >= 0
    diagonal_constraint = cp.diag(gram) == 1
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(coefficient, gram))),
        [psd_constraint, nonnegative_constraint, diagonal_constraint],
    )
    returned_value = problem.solve(
        solver="SCS",
        eps=eps,
        max_iters=max_iters,
        acceleration_lookback=10,
        normalize=True,
        verbose=False,
    )
    if gram.value is None or problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"SCS did not return a DNN solution: {problem.status}")

    q_value = np.asarray(gram.value, dtype=float)
    eigenvalues = np.linalg.eigvalsh(q_value)
    diag_error = float(np.max(np.abs(np.diag(q_value) - 1.0)))
    entrywise_min = float(np.min(q_value))
    recomputed = float(np.sum(coefficient * q_value))

    # CVXPY's dual signs give Diag(y) - C = Z + S, with Z PSD and S >= 0.
    # Clip the numerical S and shift y uniformly until the residual Z is PSD.
    # The resulting objective sum(y) is a floating-point checked feasible-dual
    # upper bound, distinct from the raw solver estimate.  No interval-arithmetic
    # or formal certification claim is made.
    y_value = np.asarray(diagonal_constraint.dual_value, dtype=float)
    s_value = np.maximum(np.asarray(nonnegative_constraint.dual_value, dtype=float), 0.0)
    dual_psd_residual = np.diag(y_value) - coefficient - s_value
    raw_min_eigenvalue = float(np.min(np.linalg.eigvalsh(dual_psd_residual)))
    roundoff_guard = 64.0 * np.finfo(float).eps * max(
        1.0, float(np.linalg.norm(dual_psd_residual, ord=2))
    )
    repair_shift = max(0.0, -raw_min_eigenvalue) + roundoff_guard
    repaired_y = y_value + repair_shift
    repaired_z = dual_psd_residual + repair_shift * np.eye(n)
    repaired_min_eigenvalue = float(np.min(np.linalg.eigvalsh(repaired_z)))
    repaired_upper_bound = float(np.sum(repaired_y))

    spectral_upper_bound = float(
        n * max(0.0, np.max(np.linalg.eigvalsh(coefficient)))
    )
    raw_extra = problem.solver_stats.extra_stats or {}
    raw_info = raw_extra.get("info", raw_extra)
    return {
        "formulation": "Eq. 5.11: max <A-gamma*n*n^T,Q>/(2W), Q PSD, Q>=0, diag(Q)=1",
        "status": problem.status,
        "termination_condition": raw_info.get("status", problem.status),
        "objective_returned": float(returned_value),
        "objective_recomputed": recomputed,
        "gram": q_value.tolist(),
        "primal_feasibility": {
            "minimum_eigenvalue": float(eigenvalues[0]),
            "entrywise_minimum": entrywise_min,
            "diagonal_max_error": diag_error,
            "objective_recompute_error": abs(recomputed - float(returned_value)),
        },
        "solver": {
            "name": problem.solver_stats.solver_name,
            "version": scs.__version__,
            "cvxpy_version": cp.__version__,
            "requested_tolerances": {"eps_abs": eps, "eps_rel": eps},
            "requested_max_iters": max_iters,
            "num_iters": problem.solver_stats.num_iters,
            "solve_time_seconds": problem.solver_stats.solve_time,
            "raw_status": _finite_json(raw_info),
        },
        "dual_upper_checks": {
            "raw_diagonal_dual": y_value.tolist(),
            "raw_nonnegative_dual_clipped": s_value.tolist(),
            "raw_psd_residual_minimum_eigenvalue": raw_min_eigenvalue,
            "uniform_repair_shift": repair_shift,
            "repaired_psd_residual_minimum_eigenvalue": repaired_min_eigenvalue,
            "repaired_dual_upper_bound": repaired_upper_bound,
            "spectral_fallback_upper_bound": spectral_upper_bound,
            "qualification": (
                "floating-point feasible-dual check; not interval-arithmetic or a formal proof artifact"
            ),
        },
    }


def run_instance(
    instance: LockedInstance,
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eps: float = DEFAULT_EPS,
    max_iters: int = DEFAULT_MAX_ITERS,
    exact_only: bool = False,
) -> dict[str, Any]:
    exact = exact_valid_cover_optimum(instance)
    algorithms = [run_hedonic(instance, int(seed)) for seed in seeds]
    for run in algorithms:
        run["algorithm_to_exact_gap"] = (
            exact["objective"] - run["objective_recomputed_eq_5_10"]
        )

    record: dict[str, Any] = {
        "instance": instance_record(instance),
        "instance_sha256": instance_sha256(instance),
        "exact_valid_cover_optimum": exact,
        "algorithm_runs": algorithms,
    }
    if exact_only:
        record["dnn"] = None
        record["certificate_chain"] = None
        return record

    dnn = solve_dnn(instance, eps=eps, max_iters=max_iters)
    numerical_outer_gap = dnn["objective_returned"] - exact["objective"]
    repaired_upper = dnn["dual_upper_checks"]["repaired_dual_upper_bound"]
    tolerance = max(10.0 * eps, 1e-9)
    best_algorithm = max(run["objective_recomputed_eq_5_10"] for run in algorithms)
    record["dnn"] = dnn
    record["certificate_chain"] = {
        "best_algorithm_objective": best_algorithm,
        "exact_valid_cover_objective": exact["objective"],
        "dnn_numerical_objective": dnn["objective_returned"],
        "repaired_dual_upper_bound": repaired_upper,
        "algorithm_to_exact_gap": exact["objective"] - best_algorithm,
        "valid_cover_to_dnn_numerical_outer_gap": numerical_outer_gap,
        "valid_cover_to_repaired_dual_outer_gap": repaired_upper - exact["objective"],
        "verification_tolerance": tolerance,
        "algorithm_le_exact": best_algorithm <= exact["objective"] + tolerance,
        "exact_le_dnn_numerical": exact["objective"] <= dnn["objective_returned"] + tolerance,
        "dnn_numerical_le_repaired_dual": dnn["objective_returned"] <= repaired_upper + tolerance,
        "interpretation": (
            "The exact-to-DNN value is the combined representation/cone outer gap. "
            "It is not decomposed because no certified completely-positive optimum was solved."
        ),
    }
    return record


def build_manifest(
    names: Iterable[str],
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eps: float = DEFAULT_EPS,
    max_iters: int = DEFAULT_MAX_ITERS,
    exact_only: bool = False,
) -> dict[str, Any]:
    selected = [LOCKED_INSTANCES[name] for name in names]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": implementation_sha256(),
        "instance_names": [instance.name for instance in selected],
        "instance_sha256": {
            instance.name: instance_sha256(instance) for instance in selected
        },
        "seeds": [int(seed) for seed in seeds],
        "dnn_solver": "SCS",
        "solver_eps": eps,
        "solver_max_iters": max_iters,
        "exact_only": exact_only,
    }
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Locked small-instance calibration of the Chapter 5 DNN outer certificate; "
            "not evidence for large-network detector performance or general tightness."
        ),
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "hedonic": _package_version("hedonic"),
            "lucas_igraph_distribution": _package_version("lucas-igraph"),
            "igraph_python": _module_version("igraph"),
            "cvxpy": None if exact_only else _package_version("cvxpy"),
            "scs": None if exact_only else _package_version("scs"),
        },
        "results": [
            run_instance(
                instance,
                seeds=protocol["seeds"],
                eps=eps,
                max_iters=max_iters,
                exact_only=exact_only,
            )
            for instance in selected
        ],
    }
    return _finite_json(manifest)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", None))
    except ImportError:
        return None


def _parse_csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def _parse_instances(value: str) -> list[str]:
    if value == "all":
        return list(LOCKED_INSTANCES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(LOCKED_INSTANCES))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"choose all or comma-separated names from {sorted(LOCKED_INSTANCES)}; unknown={unknown}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate locked valid covers and solve the Chapter 5 doubly-nonnegative "
            "upper-certificate SDP on tiny deterministic graphs."
        )
    )
    parser.add_argument("--instances", type=_parse_instances, default=list(LOCKED_INSTANCES))
    parser.add_argument("--seeds", type=_parse_csv_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--exact-only", action="store_true", help="skip CVXPY/SCS")
    parser.add_argument("--list-instances", action="store_true")
    parser.add_argument("--output", type=Path, help="strict JSON result path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_instances:
        for name, instance in LOCKED_INSTANCES.items():
            print(
                f"{name}\tn={instance.n_vertices}\tK={instance.max_labels}\t"
                f"max_memberships={instance.max_memberships}\tsha256={instance_sha256(instance)}"
            )
        return 0
    if args.eps <= 0 or args.max_iters < 1:
        raise ValueError("--eps and --max-iters must be positive")
    manifest = build_manifest(
        args.instances,
        seeds=args.seeds,
        eps=args.eps,
        max_iters=args.max_iters,
        exact_only=args.exact_only,
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
