"""Independent exact-arithmetic reader for rational DNN certificate manifests."""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from hedonic.experiments.overlapping import dnn_certificate as numerical
from hedonic.experiments.overlapping._dnn_rational_cases import (
    INSTANCE_BOUNDARY_TAGS,
    RATIONAL_INSTANCES,
)


def _fraction(value: Any) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("rational value must be an object")
    numerator = int(value["numerator"])
    denominator = int(value["denominator"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    return Fraction(numerator, denominator)


def _matrix(value: Any, n: int) -> list[list[Fraction]]:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("rational matrix row count mismatch")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("rational matrix column count mismatch")
        result.append([_fraction(item) for item in row])
    return result


def _literal(value: float) -> Fraction:
    if not math.isfinite(float(value)):
        raise ValueError("instance literal is not finite")
    return Fraction(str(float(value)))


def _coefficient(instance: numerical.LockedInstance) -> list[list[Fraction]]:
    n = instance.n_vertices
    adjacency = [[Fraction() for _ in range(n)] for _ in range(n)]
    edge_weights = [_literal(value) for value in instance.edge_weights]
    total_weight = sum(edge_weights, Fraction())
    if total_weight <= 0 or len(edge_weights) != len(instance.edges):
        raise ValueError("invalid edge-weight domain")
    for (source, target), weight in zip(instance.edges, edge_weights):
        if source == target or not (0 <= source < n and 0 <= target < n):
            raise ValueError("invalid edge domain")
        adjacency[source][target] += weight
        adjacency[target][source] += weight
    gamma = _literal(instance.resolution)
    vertex_weights = [_literal(value) for value in instance.vertex_weights]
    if len(vertex_weights) != n:
        raise ValueError("invalid vertex-weight domain")
    return [
        [
            (
                adjacency[row][column]
                - gamma * vertex_weights[row] * vertex_weights[column]
            )
            / (2 * total_weight)
            for column in range(n)
        ]
        for row in range(n)
    ]


def _symmetric(matrix: list[list[Fraction]]) -> bool:
    return all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(row)
    )


def verify_certificate(
    instance: numerical.LockedInstance,
    certificate: dict[str, Any],
) -> dict[str, Any]:
    """Verify dual equality, nonnegativity, and Gershgorin PSD exactly."""

    n = instance.n_vertices
    try:
        expected_c = _coefficient(instance)
        coefficient = _matrix(certificate["coefficient_matrix"], n)
        z_value = _matrix(certificate["nonnegative_dual_z"], n)
        s_value = _matrix(certificate["psd_residual_s"], n)
        y_raw = certificate["diagonal_dual_y"]
        if not isinstance(y_raw, list) or len(y_raw) != n:
            raise ValueError("diagonal dual length mismatch")
        y_value = [_fraction(item) for item in y_raw]
        bound = _fraction(certificate["upper_bound"]["rational"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return {"ok": False, "checks": {}, "error": f"{type(error).__name__}: {error}"}

    radii = [
        sum(abs(s_value[row][column]) for column in range(n) if column != row)
        for row in range(n)
    ]
    checks = {
        "coefficient_reconstructed": coefficient == expected_c,
        "coefficient_symmetric": _symmetric(coefficient),
        "z_symmetric": _symmetric(z_value),
        "z_entrywise_nonnegative": all(item >= 0 for row in z_value for item in row),
        "s_symmetric": _symmetric(s_value),
        "dual_equality": all(
            (y_value[row] if row == column else Fraction()) - coefficient[row][column]
            == s_value[row][column] + z_value[row][column]
            for row in range(n)
            for column in range(n)
        ),
        "s_nonnegative_diagonal": all(s_value[index][index] >= 0 for index in range(n)),
        "s_diagonally_dominant": all(
            s_value[index][index] >= radii[index] for index in range(n)
        ),
        "bound_equals_sum_y": bound == sum(y_value, Fraction()),
    }
    minimum_lower_bound = min(
        s_value[index][index] - radii[index] for index in range(n)
    )
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "bound_fraction": f"{bound.numerator}/{bound.denominator}",
        "minimum_gershgorin_lower_bound": (
            f"{minimum_lower_bound.numerator}/{minimum_lower_bound.denominator}"
        ),
    }


def verify_manifest_payload(
    payload: dict[str, Any],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a complete companion manifest without calling its constructor."""

    repository_root = repository_root or Path(__file__).resolve().parents[4]
    protocol = payload.get("protocol") or {}
    expected_names = [str(name) for name in protocol.get("instance_names") or []]
    results = payload.get("results") or []
    reports = []
    for result in results:
        name = str((result.get("instance") or {}).get("name") or "")
        instance = RATIONAL_INSTANCES.get(name)
        if instance is None:
            reports.append({"name": name, "ok": False, "error": "unknown instance"})
            continue
        certificate = verify_certificate(
            instance, result.get("exact_rational_dnn_upper_certificate") or {}
        )
        chain = result.get("certificate_chain") or {}
        boundary = result.get("native_projection_boundary")
        collision_required = bool(protocol.get("require_debug_trace")) and (
            "projection_collision" in INSTANCE_BOUNDARY_TAGS[name]
        )
        trace_ok = not collision_required or (
            isinstance(boundary, dict) and boundary.get("ok") is True
        )
        report_checks = {
            "instance_identity": result.get("instance_sha256")
            == numerical.instance_sha256(instance),
            "exact_dual": certificate["ok"],
            "chain_marks_exact_dual": chain.get("exact_rational_dual_verified") is True,
            "exact_value_below_bound": chain.get("exact_le_exact_rational_dual")
            is True,
            "collision_trace_if_registered": trace_ok,
        }
        reports.append(
            {
                "name": name,
                "ok": all(report_checks.values()),
                "checks": report_checks,
                "certificate": certificate,
            }
        )

    source_errors = []
    for relative, expected in (
        (protocol.get("implementation_identity") or {}).get("source_sha256") or {}
    ).items():
        path = repository_root / relative
        actual = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
        if actual != expected:
            source_errors.append(
                {"path": relative, "expected_sha256": expected, "actual_sha256": actual}
            )
    checks = {
        "protocol_namespace": protocol.get("protocol_version")
        == "dnn-rational-certificate-v1",
        "protocol_digest": payload.get("protocol_sha256")
        == numerical.sha256_json(protocol),
        "unique_requested_names": len(expected_names) == len(set(expected_names)),
        "requested_results_accounted": sorted(expected_names)
        == sorted(report["name"] for report in reports),
        "all_exact_certificates": bool(reports)
        and all(report["ok"] for report in reports),
        "source_hashes": not source_errors,
    }
    return {
        "schema_version": 1,
        "verifier": "independent-rational-dnn-reader-v1",
        "arithmetic": "fractions.Fraction exact integer arithmetic",
        "qualification": (
            "Exact dual equality, entrywise nonnegativity, and symmetric "
            "diagonal dominance are verified independently. The comparison "
            "to enumerated irrational cover values remains floating-point."
        ),
        "checks": checks,
        "instances": reports,
        "source_errors": source_errors,
        "ok": all(checks.values()),
    }
