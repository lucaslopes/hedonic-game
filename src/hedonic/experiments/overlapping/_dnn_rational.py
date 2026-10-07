"""Exact rational diagonally-dominant dual certificates for tiny DNN cases."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Iterable

from hedonic.experiments.overlapping.dnn_certificate import LockedInstance


RationalMatrix = list[list[Fraction]]


def _literal_fraction(value: float) -> Fraction:
    """Interpret the canonical JSON decimal spelling as an exact rational."""

    if not math.isfinite(float(value)):
        raise ValueError("rational certificates require finite instance literals")
    return Fraction(str(float(value)))


def _fraction_record(value: Fraction) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
    }


def _fraction_from_record(value: Any) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError("rational value must be a numerator/denominator object")
    numerator = int(value["numerator"])
    denominator = int(value["denominator"])
    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    return Fraction(numerator, denominator)


def _matrix_records(matrix: Iterable[Iterable[Fraction]]) -> list[list[dict[str, str]]]:
    return [[_fraction_record(value) for value in row] for row in matrix]


def _matrix_from_records(value: Any, n: int) -> RationalMatrix:
    if not isinstance(value, list) or len(value) != n:
        raise ValueError("rational matrix has the wrong row count")
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("rational matrix has the wrong column count")
        matrix.append([_fraction_from_record(item) for item in row])
    return matrix


def rational_coefficient_matrix(instance: LockedInstance) -> RationalMatrix:
    """Return C=(A-gamma*w*w^T)/(2W) in exact decimal-literal arithmetic."""

    n = instance.n_vertices
    if len(instance.vertex_weights) != n:
        raise ValueError("vertex weight count does not match n_vertices")
    if len(instance.edges) != len(instance.edge_weights):
        raise ValueError("edge weight count does not match edges")
    edge_weights = [_literal_fraction(value) for value in instance.edge_weights]
    total_weight = sum(edge_weights, Fraction())
    if total_weight <= 0:
        raise ValueError("positive total edge weight is required")
    adjacency = [[Fraction() for _ in range(n)] for _ in range(n)]
    for (source, target), weight in zip(instance.edges, edge_weights):
        if source == target or not (0 <= source < n and 0 <= target < n):
            raise ValueError("rational DNN instances require valid non-loop edges")
        adjacency[source][target] += weight
        adjacency[target][source] += weight
    gamma = _literal_fraction(instance.resolution)
    vertex_weights = [_literal_fraction(value) for value in instance.vertex_weights]
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


def _is_symmetric(matrix: RationalMatrix) -> bool:
    return all(
        matrix[row][column] == matrix[column][row]
        for row in range(len(matrix))
        for column in range(row)
    )


def verify_exact_rational_dual_certificate(
    instance: LockedInstance,
    certificate: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct and exactly verify a stored rational DNN dual witness."""

    n = instance.n_vertices
    try:
        expected_coefficient = rational_coefficient_matrix(instance)
        coefficient = _matrix_from_records(certificate["coefficient_matrix"], n)
        z_value = _matrix_from_records(certificate["nonnegative_dual_z"], n)
        s_value = _matrix_from_records(certificate["psd_residual_s"], n)
        y_payload = certificate["diagonal_dual_y"]
        if not isinstance(y_payload, list) or len(y_payload) != n:
            raise ValueError("diagonal dual has the wrong length")
        y_value = [_fraction_from_record(value) for value in y_payload]
        bound = _fraction_from_record(certificate["upper_bound"]["rational"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        return {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "checks": {},
        }

    identity_ok = all(
        (y_value[row] if row == column else Fraction()) - coefficient[row][column]
        == s_value[row][column] + z_value[row][column]
        for row in range(n)
        for column in range(n)
    )
    radii = [
        sum(abs(s_value[row][column]) for column in range(n) if column != row)
        for row in range(n)
    ]
    lower_bounds = [s_value[row][row] - radii[row] for row in range(n)]
    checks = {
        "coefficient_matches_instance_literals": coefficient == expected_coefficient,
        "coefficient_symmetric": _is_symmetric(coefficient),
        "nonnegative_dual_symmetric": _is_symmetric(z_value),
        "nonnegative_dual_entrywise": all(
            value >= 0 for row in z_value for value in row
        ),
        "psd_residual_symmetric": _is_symmetric(s_value),
        "dual_equality_exact": identity_ok,
        "psd_diagonal_nonnegative": all(
            s_value[index][index] >= 0 for index in range(n)
        ),
        "psd_symmetric_diagonal_dominance": all(value >= 0 for value in lower_bounds),
        "upper_bound_matches_diagonal_dual": bound == sum(y_value, Fraction()),
    }
    return {
        "ok": all(checks.values()),
        "arithmetic": "fractions.Fraction exact integer arithmetic",
        "checks": checks,
        "row_offdiagonal_abs_sums": [_fraction_record(value) for value in radii],
        "gershgorin_lower_bounds": [_fraction_record(value) for value in lower_bounds],
        "minimum_gershgorin_lower_bound": _fraction_record(min(lower_bounds)),
    }


def exact_rational_sdd_dual_certificate(
    instance: LockedInstance,
) -> dict[str, Any]:
    """Construct an exact DNN upper bound using symmetric diagonal dominance.

    For off-diagonal entries, choose Z_ij=max(-C_ij,0). Then
    S_ij=-C_ij-Z_ij is either zero or -C_ij. Set each S_ii to its exact
    off-diagonal absolute row sum and y_i=C_ii+S_ii. This gives
    Diag(y)-C=S+Z exactly, Z>=0, and a symmetric diagonally-dominant S with
    nonnegative diagonal. Gershgorin therefore proves S is PSD.
    """

    coefficient = rational_coefficient_matrix(instance)
    n = instance.n_vertices
    z_value = [[Fraction() for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for column in range(n):
            if row != column:
                z_value[row][column] = max(-coefficient[row][column], Fraction())
    offdiagonal_s = [
        [
            (
                -coefficient[row][column] - z_value[row][column]
                if row != column
                else Fraction()
            )
            for column in range(n)
        ]
        for row in range(n)
    ]
    radii = [
        sum(abs(offdiagonal_s[row][column]) for column in range(n) if column != row)
        for row in range(n)
    ]
    y_value = [coefficient[index][index] + radii[index] for index in range(n)]
    s_value = [
        [
            (y_value[row] if row == column else Fraction())
            - coefficient[row][column]
            - z_value[row][column]
            for column in range(n)
        ]
        for row in range(n)
    ]
    bound = sum(y_value, Fraction())
    certificate = {
        "schema_version": 1,
        "method": "exact_rational_symmetric_diagonal_dominance",
        "model": ("DNN dual Diag(y)-C=S+Z with S PSD and Z entrywise nonnegative"),
        "literal_arithmetic": (
            "Each finite float in the locked instance is interpreted through "
            "its canonical decimal string and then exactly as a rational."
        ),
        "coefficient_matrix": _matrix_records(coefficient),
        "diagonal_dual_y": [_fraction_record(value) for value in y_value],
        "nonnegative_dual_z": _matrix_records(z_value),
        "psd_residual_s": _matrix_records(s_value),
        "upper_bound": {
            "rational": _fraction_record(bound),
            "fraction": f"{bound.numerator}/{bound.denominator}",
            "float": float(bound),
        },
        "qualification": (
            "Exact rational DNN upper certificate for the locked decimal-literal "
            "instance. Symmetric diagonal dominance is sufficient, so this "
            "bound can be looser than a numerically repaired spectral witness."
        ),
    }
    certificate["verification"] = verify_exact_rational_dual_certificate(
        instance, certificate
    )
    if not certificate["verification"]["ok"]:
        raise AssertionError("constructed rational DNN dual failed exact verification")
    return certificate
