"""Exact rational DNN dual certificates in a versioned companion protocol.

The historical :mod:`dnn_certificate` module is byte-pinned by frozen paper
locks. This module reuses its locked instances, enumeration, detector adapter,
and optional SCS solver without changing that producer identity. It adds an
exact rational symmetric-diagonal-dominance certificate and two deliberate
boundary instances under a new protocol namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from hedonic.experiments.config import OVERLAPPING_ARTIFACTS_DIR, expand_path
from hedonic.experiments.overlapping import dnn_certificate as numerical
from hedonic.experiments.overlapping._dnn_rational import (
    exact_rational_sdd_dual_certificate,
    rational_coefficient_matrix,
    verify_exact_rational_dual_certificate,
)
from hedonic.experiments.overlapping._dnn_rational_cases import (
    INSTANCE_BOUNDARY_TAGS,
    RATIONAL_INSTANCES,
    run_projection_collision_boundary,
)
from hedonic.experiments.overlapping.dnn_rational_verify import (
    verify_manifest_payload,
)


PROTOCOL_VERSION = "dnn-rational-certificate-v1"
DEFAULT_SEEDS = numerical.DEFAULT_SEEDS
DEFAULT_EPS = numerical.DEFAULT_EPS
DEFAULT_MAX_ITERS = numerical.DEFAULT_MAX_ITERS
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
IMPLEMENTATION_SOURCE_PATHS = (
    "src/hedonic/experiments/overlapping/dnn_rational_certificate.py",
    "src/hedonic/experiments/overlapping/_dnn_rational.py",
    "src/hedonic/experiments/overlapping/_dnn_rational_cases.py",
    "src/hedonic/experiments/overlapping/dnn_rational_verify.py",
    "src/hedonic/experiments/overlapping/dnn_certificate.py",
    "src/hedonic/Game.py",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation_sha256() -> str:
    return _sha256_file(REPOSITORY_ROOT / IMPLEMENTATION_SOURCE_PATHS[0])


def implementation_identity(*, exact_only: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_sha256": {
            relative: _sha256_file(REPOSITORY_ROOT / relative)
            for relative in IMPLEMENTATION_SOURCE_PATHS
        },
        "runtime_versions": numerical._environment_versions(exact_only=exact_only),
        "historical_numerical_protocol": numerical.PROTOCOL_VERSION,
        "historical_module_bytes_preserved": True,
    }


def run_instance(
    instance: numerical.LockedInstance,
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eps: float = DEFAULT_EPS,
    max_iters: int = DEFAULT_MAX_ITERS,
    exact_only: bool = False,
) -> dict[str, Any]:
    exact = numerical.exact_valid_cover_optimum(instance)
    rational = exact_rational_sdd_dual_certificate(instance)
    algorithms = [numerical.run_hedonic(instance, int(seed)) for seed in seeds]
    for run in algorithms:
        run["algorithm_to_exact_gap"] = (
            exact["objective"] - run["objective_recomputed_eq_5_10"]
        )
    best_algorithm = max(run["objective_recomputed_eq_5_10"] for run in algorithms)
    rational_upper = float(rational["upper_bound"]["float"])
    tolerance = max(10.0 * eps, 1e-9)
    chain: dict[str, Any] = {
        "best_algorithm_objective": best_algorithm,
        "exact_valid_cover_objective": exact["objective"],
        "exact_rational_dual_upper_bound": rational_upper,
        "algorithm_to_exact_gap": exact["objective"] - best_algorithm,
        "valid_cover_to_exact_rational_outer_gap": rational_upper - exact["objective"],
        "verification_tolerance": tolerance,
        "algorithm_le_exact": best_algorithm <= exact["objective"] + tolerance,
        "exact_le_exact_rational_dual": exact["objective"]
        <= rational_upper + tolerance,
        "exact_rational_dual_verified": rational["verification"]["ok"],
    }
    dnn = None
    if not exact_only:
        dnn = numerical.solve_dnn(instance, eps=eps, max_iters=max_iters)
        repaired_upper = float(dnn["dual_upper_checks"]["repaired_dual_upper_bound"])
        chain.update(
            {
                "dnn_numerical_objective": dnn["objective_returned"],
                "repaired_dual_upper_bound": repaired_upper,
                "valid_cover_to_dnn_numerical_outer_gap": dnn["objective_returned"]
                - exact["objective"],
                "valid_cover_to_repaired_dual_outer_gap": repaired_upper
                - exact["objective"],
                "rational_minus_numerical_repair": rational_upper - repaired_upper,
                "rational_bound_conservatism_vs_numerical_repair": max(
                    0.0, rational_upper - repaired_upper
                ),
                "exact_le_dnn_numerical": exact["objective"]
                <= dnn["objective_returned"] + tolerance,
                "dnn_numerical_le_repaired_dual": dnn["objective_returned"]
                <= repaired_upper + tolerance,
            }
        )
    chain["interpretation"] = (
        "The exact-to-DNN gap combines representation and cone relaxation. "
        "The rational dual is exact for the locked decimal-literal instance "
        "and deliberately conservative; the SCS and spectral-repair values "
        "remain numerical. No intermediate completely-positive optimum was solved."
    )
    return {
        "instance": numerical.instance_record(instance),
        "instance_sha256": numerical.instance_sha256(instance),
        "boundary_tags": list(INSTANCE_BOUNDARY_TAGS[instance.name]),
        "exact_valid_cover_optimum": exact,
        "exact_rational_dnn_upper_certificate": rational,
        "native_projection_boundary": run_projection_collision_boundary(instance),
        "algorithm_runs": algorithms,
        "dnn": dnn,
        "certificate_chain": chain,
    }


def build_manifest(
    names: Iterable[str],
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    eps: float = DEFAULT_EPS,
    max_iters: int = DEFAULT_MAX_ITERS,
    exact_only: bool = False,
    require_debug_trace: bool = False,
) -> dict[str, Any]:
    selected = [RATIONAL_INSTANCES[name] for name in names]
    identity = implementation_identity(exact_only=exact_only)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_sha256": implementation_sha256(),
        "implementation_identity": identity,
        "instance_names": [instance.name for instance in selected],
        "instance_sha256": {
            instance.name: numerical.instance_sha256(instance) for instance in selected
        },
        "instance_boundary_tags": {
            instance.name: list(INSTANCE_BOUNDARY_TAGS[instance.name])
            for instance in selected
        },
        "seeds": [int(seed) for seed in seeds],
        "dnn_solver": None if exact_only else "SCS",
        "solver_eps": eps,
        "solver_max_iters": max_iters,
        "exact_rational_dual": "symmetric-diagonal-dominance-v1",
        "require_debug_trace": require_debug_trace,
        "exact_only": exact_only,
    }
    results = [
        run_instance(
            instance,
            seeds=protocol["seeds"],
            eps=eps,
            max_iters=max_iters,
            exact_only=exact_only,
        )
        for instance in selected
    ]
    rational_ok = all(
        result["exact_rational_dnn_upper_certificate"]["verification"]["ok"]
        and result["certificate_chain"]["exact_le_exact_rational_dual"]
        for result in results
    )
    collision_records = [
        result["native_projection_boundary"]
        for result in results
        if "projection_collision" in result["boundary_tags"]
    ]
    trace_ok = bool(collision_records) and all(
        record is not None and record.get("ok") for record in collision_records
    )
    checks = {
        "exact_rational_duals": rational_ok,
        "requested_instances_accounted": len(results) == len(selected),
        "required_collision_trace": trace_ok if require_debug_trace else True,
    }
    manifest = {
        "schema_version": 1,
        "purpose": (
            "Exact rational tiny-instance DNN upper certificates and deliberate "
            "duplicate/cap/collision boundaries; not a large-network performance "
            "claim or a claim that DNN is generally tight."
        ),
        "protocol": protocol,
        "protocol_sha256": numerical.sha256_json(protocol),
        "environment": identity["runtime_versions"],
        "checks": checks,
        "status": "complete" if all(checks.values()) else "incomplete",
        "results": results,
    }
    independent = verify_manifest_payload(manifest)
    manifest["independent_verification"] = independent
    manifest["checks"]["independent_exact_reader"] = independent["ok"]
    manifest["status"] = (
        "complete" if all(manifest["checks"].values()) else "incomplete"
    )
    return numerical._finite_json(manifest)


def _parse_csv_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return result


def _parse_instances(value: str) -> list[str]:
    if value == "all":
        return list(RATIONAL_INSTANCES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(RATIONAL_INSTANCES))
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            f"choose all or comma-separated names from {sorted(RATIONAL_INSTANCES)}; "
            f"unknown={unknown}"
        )
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact rational diagonally-dominant DNN dual certificates "
            "without changing the frozen numerical DNN producer."
        )
    )
    parser.add_argument(
        "--instances", type=_parse_instances, default=list(RATIONAL_INSTANCES)
    )
    parser.add_argument("--seeds", type=_parse_csv_ints, default=list(DEFAULT_SEEDS))
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="skip CVXPY/SCS while retaining exact enumeration and rational bounds",
    )
    parser.add_argument(
        "--require-debug-trace",
        action="store_true",
        help=("require the candidate binding's collision/projection trace "
              "(1.0.0.5+ also requires label counts and audits the tie budget)"),
    )
    parser.add_argument("--list-instances", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=OVERLAPPING_ARTIFACTS_DIR
        / "dnn_rational_certificate"
        / "dnn_rational_certificate.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_instances:
        for name, instance in RATIONAL_INSTANCES.items():
            print(
                f"{name}\tn={instance.n_vertices}\tK={instance.max_labels}\t"
                f"max_memberships={instance.max_memberships}\t"
                f"sha256={numerical.instance_sha256(instance)}\t"
                f"boundaries={','.join(INSTANCE_BOUNDARY_TAGS[name])}"
            )
        return 0
    if args.eps <= 0 or args.max_iters < 1:
        raise ValueError("--eps and --max-iters must be positive")
    output = expand_path(args.output)
    manifest = build_manifest(
        args.instances,
        seeds=args.seeds,
        eps=args.eps,
        max_iters=args.max_iters,
        exact_only=args.exact_only,
        require_debug_trace=args.require_debug_trace,
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
        print(f"Wrote {output}")
    else:
        sys.stdout.write(encoded)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
