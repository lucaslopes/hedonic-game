"""Independent NumPy unit-ℓ₂ participation oracle.

This module does not import igraph, lucas-igraph, or
:mod:`hedonic.experiments.overlapping.robustness`.  It is the paper-side
mathematical certificate for complete bounded-set best responses: gains are
computed with the moving row removed, every prefix of the sorted gain vector
is evaluated, and an already-optimal current action is retained on ties.

The unnormalized potential here is ``Φ̃_γ`` from the overlapping manuscript:
``0.5 ⟨A, FFᵀ⟩ − (γ/2) ‖wᵀ F‖²``, which equals the off-diagonal pair potential
``P_γ`` minus the profile-independent self term ``(γ/2) ∑_v w_v²``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SEED = 20260905
TOKEN_SEED = 20260906
FEE_SEED = 20260917
FEE_OBJECTIVE_VERSION = "unit-l2-cpm-fee-v1"
MANUSCRIPT_OBJECTIVE_VERSION = "unit-l2-cpm"
NAMED_EXAMPLES = (
    "ce1_weighted",
    "ce1_unweighted",
    "ce3_balanced_failure",
    "ce4_duplicate_body",
    "ce5_token_normalization",
    "ce6_externality",
    "ce7_l1_isolate",
    "collision_projection",
    "restricted_balance_token_identity",
)


def admissible_actions(label_count: int, cap: int) -> list[tuple[int, ...]]:
    """Return every nonempty subset of ``{0,…,q-1}`` with size at most ``cap``."""
    if label_count < 1 or cap < 1:
        raise ValueError("label_count and cap must be positive")
    return [
        tuple(choice)
        for size in range(1, min(label_count, cap) + 1)
        for choice in combinations(range(label_count), size)
    ]


def membership_matrix(state: Sequence[Sequence[int]], label_count: int) -> np.ndarray:
    """Return the equal-intensity unit-ℓ₂ membership matrix ``F``."""
    rows = np.zeros((len(state), label_count), dtype=float)
    for vertex, labels in enumerate(state):
        unique = tuple(sorted(set(int(label) for label in labels)))
        if not unique or unique[0] < 0 or unique[-1] >= label_count:
            raise ValueError(f"invalid membership row {labels!r}")
        if len(unique) != len(labels):
            raise ValueError(f"duplicate labels in membership row {labels!r}")
        rows[vertex, list(unique)] = 1.0 / math.sqrt(len(unique))
    return rows


def unnormalized_potential(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
) -> float:
    """Return ``Φ̃_γ`` for a labelled profile."""
    factor = membership_matrix(state, label_count)
    return float(
        0.5 * np.sum(adjacency * (factor @ factor.T))
        - (gamma / 2.0) * np.sum((weights @ factor) ** 2)
    )


def off_diagonal_potential(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
) -> float:
    """Return ``P_γ = ∑_{u<v} (a_uv − γ w_u w_v) κ_uv``."""
    factor = membership_matrix(state, label_count)
    kappa = factor @ factor.T
    pair = adjacency - gamma * np.outer(weights, weights)
    n = adjacency.shape[0]
    total = 0.0
    for first in range(n):
        for second in range(first + 1, n):
            total += float(pair[first, second] * kappa[first, second])
    return total


def label_gains(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    vertex: int,
) -> np.ndarray:
    """Return ``g_{vc}`` after removing vertex ``v``'s entire membership row."""
    factor = membership_matrix(state, label_count)
    factor[vertex] = 0.0
    return (adjacency[vertex] - gamma * weights[vertex] * weights) @ factor


@dataclass
class OracleWorkspace:
    """Reusable scratch space for the independent oracle.

    ``label_gains`` above is intentionally kept as the small, rebuilding
    reference implementation.  A complete audit calls it once per visited
    vertex, however, and rebuilding the same membership matrix on every visit
    dominates that path for moderate ``n``.  This workspace caches the matrix
    and normalisation roots and reuses a single coefficient/gain buffer while
    preserving the reference arithmetic and tie policy.

    The workspace is sequential by design: :meth:`gains_for_vertex` temporarily
    clears one row in the cached ``factor`` and restores it before returning.
    It must not be shared by concurrent visits.  ``copy=True`` is the safe
    public default; the optimized sweep uses ``copy=False`` only while
    consuming the scratch result immediately.
    """

    adjacency: np.ndarray
    weights: np.ndarray
    gamma: float
    memberships: tuple[tuple[int, ...], ...]
    label_count: int
    factor: np.ndarray
    _row_backup: np.ndarray
    _coefficients: np.ndarray
    _gains: np.ndarray
    sqrt_roots: tuple[float, ...]
    inverse_roots: tuple[float, ...]

    @classmethod
    def build(
        cls,
        adjacency: np.ndarray,
        weights: np.ndarray,
        gamma: float,
        state: Sequence[Sequence[int]],
        label_count: int,
    ) -> "OracleWorkspace":
        """Build a workspace after exactly the reference membership checks."""
        factor = membership_matrix(state, int(label_count))
        adjacency_array = np.asarray(adjacency)
        weights_array = np.asarray(weights)
        if adjacency_array.ndim != 2 or adjacency_array.shape[0] != factor.shape[0]:
            raise ValueError("adjacency must be a square matrix matching state")
        if adjacency_array.shape[1] != factor.shape[0]:
            raise ValueError("adjacency must be square")
        if weights_array.ndim != 1 or weights_array.shape[0] != factor.shape[0]:
            raise ValueError("weights must have one value per state row")
        memberships = tuple(
            tuple(int(label) for label in labels) for labels in state
        )
        max_cardinality = max(1, int(label_count))
        # Keep both forms.  The optimized scorer uses ``sqrt_roots`` in a
        # division, matching ``prefix_best_response`` bit-for-bit; inverse
        # roots are available to callers that can safely use multiplication.
        sqrt_roots = tuple(
            [0.0]
            + [math.sqrt(cardinality) for cardinality in range(1, max_cardinality + 1)]
        )
        inverse_roots = tuple(
            [0.0]
            + [1.0 / root for root in sqrt_roots[1:]]
        )
        return cls(
            adjacency=adjacency_array,
            weights=weights_array,
            gamma=float(gamma),
            memberships=memberships,
            label_count=int(label_count),
            factor=factor,
            _row_backup=np.empty(int(label_count), dtype=float),
            _coefficients=np.empty(factor.shape[0], dtype=float),
            _gains=np.empty(int(label_count), dtype=float),
            sqrt_roots=sqrt_roots,
            inverse_roots=inverse_roots,
        )

    def gains_for_vertex(self, vertex: int, *, copy: bool = True) -> np.ndarray:
        """Return reference-equivalent gains, reusing the workspace buffers."""
        vertex = int(vertex)
        if not 0 <= vertex < self.factor.shape[0]:
            raise IndexError("vertex outside state")
        row = self.factor[vertex]
        # ``label_gains`` computes the coefficient row first and then performs
        # one matrix product against a factor matrix whose moving row is zero.
        # Repeating those exact operations here avoids a matrix allocation for
        # every vertex.  Restore the row even if NumPy raises during matmul.
        self._row_backup[...] = row
        row[...] = 0.0
        try:
            np.multiply(
                self.weights,
                self.gamma * self.weights[vertex],
                out=self._coefficients,
            )
            np.subtract(
                self.adjacency[vertex],
                self._coefficients,
                out=self._coefficients,
            )
            np.matmul(
                self._coefficients,
                self.factor,
                out=self._gains,
            )
        finally:
            row[...] = self._row_backup
        return self._gains.copy() if copy else self._gains


def prepare_oracle_workspace(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
) -> OracleWorkspace:
    """Return reusable state for an optimized, fixed-profile oracle sweep."""
    return OracleWorkspace.build(adjacency, weights, gamma, state, label_count)


def set_score(gains: np.ndarray, labels: Sequence[int]) -> float:
    """Return ``U_v(S) = |S|^{-1/2} ∑_{c ∈ S} g_c``."""
    if not labels:
        raise ValueError("a membership action must be non-empty")
    return float(sum(gains[int(label)] for label in labels) / math.sqrt(len(labels)))


def _validated_fee(tau: float) -> float:
    value = float(tau)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    return value


def cardinality_fee(cardinality: int, tau: float) -> float:
    """Return ``τ(k-1)``; zero for a singleton or a zero fee."""
    cardinality = int(cardinality)
    if cardinality < 1:
        raise ValueError("membership cardinality must be positive")
    return _validated_fee(tau) * (cardinality - 1)


def fee_set_score(gains: np.ndarray, labels: Sequence[int], tau: float) -> float:
    """Return ``U_v^τ(S) = U_v(S) − τ(|S|-1)`` from (R9)."""
    return set_score(gains, labels) - cardinality_fee(len(labels), tau)


def extra_memberships(state: Sequence[Sequence[int]]) -> int:
    """Return ``∑_v (k_v-1)`` for nonempty membership rows."""
    total = 0
    for labels in state:
        row = tuple(int(label) for label in labels)
        if not row:
            raise ValueError("membership rows must be nonempty")
        if len(row) != len(set(row)):
            raise ValueError("membership rows must not contain duplicate labels")
        total += len(row) - 1
    return total


def fee_pair_potential(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    tau: float,
) -> float:
    """Return the exact pair potential ``Ψ_τ`` from (R9)."""
    return off_diagonal_potential(
        adjacency, weights, gamma, state, label_count
    ) - _validated_fee(tau) * extra_memberships(state)


def fee_unnormalized_potential(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    tau: float,
) -> float:
    """Return ``Φ̃_γ − τ∑_v(k_v-1)``, matching (R9) up to the constant self term."""
    return unnormalized_potential(
        adjacency, weights, gamma, state, label_count
    ) - _validated_fee(tau) * extra_memberships(state)


def positive_pair_degree(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    vertex: int,
) -> float:
    """Return ``D_v^+ = ∑_{u≠v} max(B_uv, 0)`` used by (R10)."""
    total = 0.0
    for other in range(adjacency.shape[0]):
        if other == vertex:
            continue
        coefficient = float(
            adjacency[vertex, other] - gamma * weights[vertex] * weights[other]
        )
        total += max(coefficient, 0.0)
    return total


def prefix_best_response(
    gains: np.ndarray,
    current: Sequence[int],
    cap: int,
    tau: float = 0.0,
) -> tuple[tuple[int, ...], float, float]:
    """Return ``(best_action, best_utility, regret)`` from sorted prefixes.

    Ties keep ``current`` whenever it is already optimal. A positive ``tau``
    is the separate ``unit-l2-cpm-fee-v1`` objective, not the manuscript model.
    """
    label_count = int(gains.shape[0])
    cap = min(int(cap), label_count)
    order = sorted(range(label_count), key=lambda label: (-float(gains[label]), label))
    best_utility = -math.inf
    best_action: tuple[int, ...] = ()
    prefix = 0.0
    for cardinality in range(1, cap + 1):
        prefix += float(gains[order[cardinality - 1]])
        utility = prefix / math.sqrt(cardinality) - cardinality_fee(cardinality, tau)
        if utility > best_utility + 1e-15:
            best_utility = utility
            best_action = tuple(order[:cardinality])
    current_action = tuple(int(label) for label in current)
    current_utility = fee_set_score(gains, current_action, tau)
    if current_utility + 1e-15 >= best_utility:
        return current_action, current_utility, 0.0
    return best_action, best_utility, float(best_utility - current_utility)


def prefix_best_response_cached(
    gains: np.ndarray,
    current: Sequence[int],
    cap: int,
    tau: float = 0.0,
    *,
    sqrt_roots: Sequence[float] | None = None,
) -> tuple[tuple[int, ...], float, float]:
    """Reference-prefix response with caller-provided normalisation roots.

    The default path remains :func:`prefix_best_response`.  Supplying roots
    avoids repeated ``sqrt`` calls in a long sweep while retaining the exact
    division operation used by the reference scorer.  In particular, this is
    deliberately *not* an inverse-root multiplication: floating-point tie
    behaviour is part of the oracle contract.
    """
    label_count = int(gains.shape[0])
    cap = min(int(cap), label_count)
    if sqrt_roots is None:
        roots = tuple(
            [0.0] + [math.sqrt(cardinality) for cardinality in range(1, cap + 1)]
        )
    else:
        roots = tuple(float(value) for value in sqrt_roots)
        if len(roots) <= cap:
            raise ValueError("sqrt_roots must cover every admissible cardinality")
    order = sorted(range(label_count), key=lambda label: (-float(gains[label]), label))
    best_utility = -math.inf
    best_action: tuple[int, ...] = ()
    prefix = 0.0
    for cardinality in range(1, cap + 1):
        prefix += float(gains[order[cardinality - 1]])
        utility = prefix / roots[cardinality] - cardinality_fee(cardinality, tau)
        if utility > best_utility + 1e-15:
            best_utility = utility
            best_action = tuple(order[:cardinality])
    current_action = tuple(int(label) for label in current)
    current_utility = fee_set_score(gains, current_action, tau)
    if current_utility + 1e-15 >= best_utility:
        return current_action, current_utility, 0.0
    return best_action, best_utility, float(best_utility - current_utility)


def exhaustive_best_response(
    gains: np.ndarray,
    current: Sequence[int],
    cap: int,
    tau: float = 0.0,
) -> tuple[tuple[int, ...], float, float]:
    """Enumerate every admissible set; keep the current action on optimal ties."""
    actions = admissible_actions(int(gains.shape[0]), cap)
    best_action = max(
        actions, key=lambda labels: (fee_set_score(gains, labels, tau), labels)
    )
    best_utility = fee_set_score(gains, best_action, tau)
    current_action = tuple(int(label) for label in current)
    current_utility = fee_set_score(gains, current_action, tau)
    if current_utility + 1e-15 >= best_utility:
        return current_action, current_utility, 0.0
    return best_action, best_utility, float(best_utility - current_utility)


def primitive_neighbor(current: Sequence[int], candidate: Sequence[int]) -> bool:
    """True when the candidate differs by at most one label join/leave/switch."""
    current_set = set(int(label) for label in current)
    candidate_set = set(int(label) for label in candidate)
    return len(candidate_set - current_set) <= 1 and len(current_set - candidate_set) <= 1


def vertex_regrets(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    cap: int,
    *,
    primitive: bool = False,
) -> list[dict[str, Any]]:
    """Return per-vertex full or primitive regrets."""
    actions = admissible_actions(label_count, cap)
    records = []
    for vertex, current in enumerate(state):
        gains = label_gains(adjacency, weights, gamma, state, label_count, vertex)
        if primitive:
            candidates = [
                action for action in actions if primitive_neighbor(current, action)
            ]
            best = max(candidates, key=lambda labels: set_score(gains, labels))
            current_utility = set_score(gains, current)
            best_utility = set_score(gains, best)
            if current_utility + 1e-15 >= best_utility:
                best = tuple(int(label) for label in current)
                best_utility = current_utility
                regret = 0.0
            else:
                regret = float(best_utility - current_utility)
        else:
            best, best_utility, regret = prefix_best_response(gains, current, cap)
        records.append(
            {
                "vertex": vertex,
                "current": [int(label) for label in current],
                "best": [int(label) for label in best],
                "current_utility": float(set_score(gains, current)),
                "best_utility": float(best_utility),
                "regret": float(regret),
                "gains": [float(value) for value in gains],
            }
        )
    return records


def vertex_regrets_optimized(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    cap: int,
    *,
    primitive: bool = False,
) -> list[dict[str, Any]]:
    """Return :func:`vertex_regrets` records using reusable scoring buffers.

    This is an opt-in implementation of the same independent oracle.  The
    rebuilding :func:`vertex_regrets` path remains the audit reference and is
    intentionally untouched.  Equivalence tests compare actions, scores, and
    regrets on exhaustive small cases plus randomized profiles before this
    path is used in any benchmark.
    """
    workspace = prepare_oracle_workspace(
        adjacency, weights, gamma, state, label_count
    )
    actions = admissible_actions(label_count, cap)
    records = []
    for vertex, current in enumerate(state):
        gains = workspace.gains_for_vertex(vertex, copy=False)
        if primitive:
            candidates = [
                action for action in actions if primitive_neighbor(current, action)
            ]
            best = max(candidates, key=lambda labels: set_score(gains, labels))
            current_utility = set_score(gains, current)
            best_utility = set_score(gains, best)
            if current_utility + 1e-15 >= best_utility:
                best = tuple(int(label) for label in current)
                best_utility = current_utility
                regret = 0.0
            else:
                regret = float(best_utility - current_utility)
        else:
            best, best_utility, regret = prefix_best_response_cached(
                gains,
                current,
                cap,
                sqrt_roots=workspace.sqrt_roots,
            )
        records.append(
            {
                "vertex": vertex,
                "current": [int(label) for label in current],
                "best": [int(label) for label in best],
                "current_utility": float(set_score(gains, current)),
                "best_utility": float(best_utility),
                "regret": float(regret),
                "gains": [float(value) for value in gains],
            }
        )
    return records


# A descriptive alias keeps call sites readable while retaining one canonical
# implementation and making it easy to compare both paths in a debugger.
cached_vertex_regrets = vertex_regrets_optimized


def _record_equivalence(
    reference: Sequence[dict[str, Any]],
    optimized: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two response-record lists without hiding action differences."""
    if len(reference) != len(optimized):
        return {
            "ok": False,
            "record_count_reference": len(reference),
            "record_count_optimized": len(optimized),
            "action_mismatches": len(reference) + len(optimized),
            "max_gain_abs_error": math.inf,
            "max_score_abs_error": math.inf,
            "max_regret_abs_error": math.inf,
        }
    action_mismatches = 0
    max_gain_error = 0.0
    max_score_error = 0.0
    max_regret_error = 0.0
    for left, right in zip(reference, optimized):
        if left["current"] != right["current"] or left["best"] != right["best"]:
            action_mismatches += 1
        left_gains = np.asarray(left["gains"], dtype=float)
        right_gains = np.asarray(right["gains"], dtype=float)
        if left_gains.shape != right_gains.shape:
            max_gain_error = math.inf
        else:
            gain_error = (
                float(np.max(np.abs(left_gains - right_gains)))
                if left_gains.size
                else 0.0
            )
            max_gain_error = max(
                max_gain_error,
                gain_error,
            )
        for field in ("current_utility", "best_utility"):
            max_score_error = max(
                max_score_error,
                abs(float(left[field]) - float(right[field])),
            )
        max_regret_error = max(
            max_regret_error,
            abs(float(left["regret"]) - float(right["regret"])),
        )
    return {
        "ok": (
            action_mismatches == 0
            and max_gain_error <= 0.0
            and max_score_error <= 0.0
            and max_regret_error <= 0.0
        ),
        "record_count_reference": len(reference),
        "record_count_optimized": len(optimized),
        "action_mismatches": action_mismatches,
        "max_gain_abs_error": max_gain_error,
        "max_score_abs_error": max_score_error,
        "max_regret_abs_error": max_regret_error,
    }


def optimization_equivalence(
    *,
    random_cases: int = 300,
    seed: int = SEED + 101,
) -> dict[str, Any]:
    """Exhaustively and randomly compare the reference and cached sweeps.

    The exhaustive component enumerates every labelled two-vertex profile for
    a small weighted graph and both primitive/full policies.  Random cases
    vary graph size, label bank, cap, weights, resolution, and current action.
    A response is accepted only when all actions and floating-point fields are
    bit-for-bit equal; this is intentionally stricter than the oracle's public
    numerical tolerance.
    """
    if random_cases < 0:
        raise ValueError("random_cases must be nonnegative")
    fixed_adjacency = make_symmetric_graph(
        2, ((0, 1, 1.5),)
    )
    fixed_weights = np.array([0.75, 1.25], dtype=float)
    fixed_gamma = 0.35
    fixed_actions = admissible_actions(3, 2)
    exhaustive_rows: list[dict[str, Any]] = []
    for first in fixed_actions:
        for second in fixed_actions:
            state = (first, second)
            for primitive in (False, True):
                reference = vertex_regrets(
                    fixed_adjacency,
                    fixed_weights,
                    fixed_gamma,
                    state,
                    3,
                    2,
                    primitive=primitive,
                )
                optimized = vertex_regrets_optimized(
                    fixed_adjacency,
                    fixed_weights,
                    fixed_gamma,
                    state,
                    3,
                    2,
                    primitive=primitive,
                )
                exhaustive_rows.append(_record_equivalence(reference, optimized))

    rng = np.random.default_rng(seed)
    random_rows: list[dict[str, Any]] = []
    for _ in range(random_cases):
        n_vertices = int(rng.integers(2, 9))
        label_count = int(rng.integers(1, 8))
        cap = int(rng.integers(1, label_count + 1))
        actions = admissible_actions(label_count, cap)
        adjacency = np.triu(
            rng.integers(0, 8, (n_vertices, n_vertices)) / 4.0,
            1,
        )
        adjacency = adjacency + adjacency.T
        weights = rng.integers(0, 6, n_vertices) / 3.0
        gamma = float(rng.integers(0, 11) / 10.0)
        state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
        primitive = bool(rng.integers(2))
        reference = vertex_regrets(
            adjacency,
            weights,
            gamma,
            state,
            label_count,
            cap,
            primitive=primitive,
        )
        optimized = vertex_regrets_optimized(
            adjacency,
            weights,
            gamma,
            state,
            label_count,
            cap,
            primitive=primitive,
        )
        random_rows.append(_record_equivalence(reference, optimized))

    named = named_examples()
    # CE1/CE3 carry full response records.  CE5/CE6 exercise direct gain
    # scoring; CE7 and the projection fixtures are rerun to ensure the named
    # counterexample assertions remain intact while the optional path exists.
    named_rows: dict[str, Any] = {}
    ce1_weighted_state = ((0,), (0,), (1,), (2,))
    ce1_weighted_adjacency = make_symmetric_graph(
        4, ((0, 1, 1.5), (0, 2, 0.9), (0, 3, 0.9))
    )
    ce1_weighted_ref = vertex_regrets(
        ce1_weighted_adjacency,
        np.ones(4),
        0.5,
        ce1_weighted_state,
        3,
        3,
    )
    ce1_weighted_opt = vertex_regrets_optimized(
        ce1_weighted_adjacency,
        np.ones(4),
        0.5,
        ce1_weighted_state,
        3,
        3,
    )
    named_rows["ce1_weighted"] = _record_equivalence(
        ce1_weighted_ref, ce1_weighted_opt
    )
    ce1_unweighted_state = ((0, 1, 2), (0,), (1, 2), (0, 1, 2))
    ce1_unweighted_adjacency = make_symmetric_graph(
        4, ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0))
    )
    ce1_unweighted_ref = vertex_regrets(
        ce1_unweighted_adjacency,
        np.ones(4),
        0.25,
        ce1_unweighted_state,
        3,
        3,
    )
    ce1_unweighted_opt = vertex_regrets_optimized(
        ce1_unweighted_adjacency,
        np.ones(4),
        0.25,
        ce1_unweighted_state,
        3,
        3,
    )
    named_rows["ce1_unweighted"] = _record_equivalence(
        ce1_unweighted_ref, ce1_unweighted_opt
    )
    ce2_state = ((0,), (0,), (1,))
    ce2_adjacency = make_symmetric_graph(
        3, ((0, 1, 1.0), (0, 2, 1.0))
    )
    ce2_rows = []
    for cap_value in (1, 2):
        ce2_rows.append(
            _record_equivalence(
                vertex_regrets(
                    ce2_adjacency,
                    np.ones(3),
                    3.0 / 5.0,
                    ce2_state,
                    2,
                    cap_value,
                ),
                vertex_regrets_optimized(
                    ce2_adjacency,
                    np.ones(3),
                    3.0 / 5.0,
                    ce2_state,
                    2,
                    cap_value,
                ),
            )
        )
    named_rows["ce2_cap_monotonicity"] = {
        "ok": all(row["ok"] for row in ce2_rows),
        "cap_rows": ce2_rows,
    }
    ce3_state = ((0, 1), (0, 1), (1, 2), (1, 2), (0, 2), (0, 2))
    ce3_adjacency = make_symmetric_graph(
        6,
        (
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 0, 1.0),
        ),
    )
    ce3_rows = []
    for gamma in (0.0, 1.0):
        ce3_rows.append(
            _record_equivalence(
                vertex_regrets(ce3_adjacency, np.ones(6), gamma, ce3_state, 3, 2),
                vertex_regrets_optimized(
                    ce3_adjacency, np.ones(6), gamma, ce3_state, 3, 2
                ),
            )
        )
    named_rows["ce3_balanced_failure"] = {
        "ok": all(row["ok"] for row in ce3_rows),
        "gamma_rows": ce3_rows,
    }
    named_rows["ce4_duplicate_body_fixture"] = {
        "ok": abs(
            named["ce4_duplicate_body"]["potential_after"]
            - named["ce4_duplicate_body"]["potential_before"]
        )
        > 1e-8
    }
    ce5_adjacency = make_symmetric_graph(2, ((0, 1, 1.0),))
    ce5_state = ((0, 1), (0,))
    ce5_workspace = prepare_oracle_workspace(
        ce5_adjacency, np.ones(2), 0.0, ce5_state, 2
    )
    ce5_gains = _record_equivalence(
        [
            {
                "current": [],
                "best": [],
                "gains": label_gains(ce5_adjacency, np.ones(2), 0.0, ce5_state, 2, 0).tolist(),
                "current_utility": 0.0,
                "best_utility": 0.0,
                "regret": 0.0,
            }
        ],
        [
            {
                "current": [],
                "best": [],
                "gains": ce5_workspace.gains_for_vertex(0).tolist(),
                "current_utility": 0.0,
                "best_utility": 0.0,
                "regret": 0.0,
            }
        ],
    )
    named_rows["ce5_token_normalization_gain_path"] = ce5_gains
    ce6_adjacency = make_symmetric_graph(2, ((0, 1, 1.0),))
    ce6_state = ((0,), (0,))
    ce6_workspace = prepare_oracle_workspace(
        ce6_adjacency, np.ones(2), 0.0, ce6_state, 2
    )
    ce6_gains = _record_equivalence(
        [
            {
                "current": [],
                "best": [],
                "gains": label_gains(ce6_adjacency, np.ones(2), 0.0, ce6_state, 2, 1).tolist(),
                "current_utility": 0.0,
                "best_utility": 0.0,
                "regret": 0.0,
            }
        ],
        [
            {
                "current": [],
                "best": [],
                "gains": ce6_workspace.gains_for_vertex(1).tolist(),
                "current_utility": 0.0,
                "best_utility": 0.0,
                "regret": 0.0,
            }
        ],
    )
    named_rows["ce6_externality_gain_path"] = ce6_gains
    assert_named_examples(named)
    named_rows["ce5_token_normalization_fixture"] = {
        "ok": not named["ce5_token_normalization"]["same_normalized_value"]
    }
    named_rows["ce7_l1_isolate_fixture"] = {"ok": named["ce7_l1_isolate"]["gain"] > 0.12}
    named_rows["restricted_balance_token_identity_fixture"] = {
        "ok": named["restricted_balance_token_identity"]["abs_error"] < 1e-12
    }
    named_rows["collision_projection_fixture"] = {
        "ok": named["collision_projection"]["decreasing_projection_rejected"]
    }
    all_rows = exhaustive_rows + random_rows
    return {
        "schema_version": 1,
        "reference_path": "vertex_regrets",
        "optimized_path": "vertex_regrets_optimized",
        "strict_bitwise_fields": True,
        "seed": seed,
        "random_cases": random_cases,
        "exhaustive_profiles": len(exhaustive_rows),
        "exhaustive_failures": sum(not row["ok"] for row in exhaustive_rows),
        "random_failures": sum(not row["ok"] for row in random_rows),
        "max_gain_abs_error": max(
            [row["max_gain_abs_error"] for row in all_rows] or [0.0]
        ),
        "max_score_abs_error": max(
            [row["max_score_abs_error"] for row in all_rows] or [0.0]
        ),
        "max_regret_abs_error": max(
            [row["max_regret_abs_error"] for row in all_rows] or [0.0]
        ),
        "named_fixtures": named_rows,
        "ok": (
            all(row["ok"] for row in all_rows)
            and all(row.get("ok", False) for row in named_rows.values())
        ),
    }


def benchmark_optimization(
    *,
    n_vertices: int = 240,
    label_count: int = 64,
    cap: int = 4,
    repeats: int = 5,
    seed: int = SEED + 202,
) -> dict[str, Any]:
    """Measure reference versus cached sweeps after an equivalence preflight."""
    if n_vertices < 2 or label_count < 1 or cap < 1 or repeats < 1:
        raise ValueError("benchmark dimensions and repeats must be positive")
    cap = min(int(cap), int(label_count))
    rng = np.random.default_rng(seed)
    adjacency = np.triu(rng.random((n_vertices, n_vertices)), 1)
    adjacency = adjacency + adjacency.T
    weights = rng.random(n_vertices)
    actions = admissible_actions(label_count, cap)
    state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
    gamma = 0.35
    reference = vertex_regrets(
        adjacency, weights, gamma, state, label_count, cap
    )
    optimized = vertex_regrets_optimized(
        adjacency, weights, gamma, state, label_count, cap
    )
    equivalence = _record_equivalence(reference, optimized)
    import sys
    import time
    import tracemalloc

    def measure(callable_: Any) -> tuple[list[float], list[int], int]:
        for _ in range(1):
            callable_()
        durations: list[float] = []
        peaks: list[int] = []
        checksum = 0
        for _ in range(repeats):
            tracemalloc.start()
            started = time.perf_counter()
            result = callable_()
            durations.append(time.perf_counter() - started)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peaks.append(int(peak))
            checksum += len(result) + int(sum(row["regret"] > 0 for row in result))
        return durations, peaks, checksum

    reference_times, reference_peaks, reference_checksum = measure(
        lambda: vertex_regrets(adjacency, weights, gamma, state, label_count, cap)
    )
    optimized_times, optimized_peaks, optimized_checksum = measure(
        lambda: vertex_regrets_optimized(
            adjacency, weights, gamma, state, label_count, cap
        )
    )

    def summary(times: Sequence[float], peaks: Sequence[int], checksum: int) -> dict[str, Any]:
        ordered_times = sorted(float(value) for value in times)
        ordered_peaks = sorted(int(value) for value in peaks)
        middle = len(ordered_times) // 2
        return {
            "times_seconds": [float(value) for value in times],
            "peak_tracemalloc_bytes": [int(value) for value in peaks],
            "median_seconds": ordered_times[middle],
            "min_seconds": ordered_times[0],
            "median_peak_tracemalloc_bytes": ordered_peaks[middle],
            "checksum": checksum,
        }

    reference_summary = summary(reference_times, reference_peaks, reference_checksum)
    optimized_summary = summary(optimized_times, optimized_peaks, optimized_checksum)
    return {
        "schema_version": 1,
        "seed": seed,
        "n_vertices": n_vertices,
        "label_count": label_count,
        "cap": cap,
        "gamma": gamma,
        "repeats": repeats,
        "reference_path": "vertex_regrets",
        "optimized_path": "vertex_regrets_optimized",
        "equivalence_preflight": equivalence,
        "reference": reference_summary,
        "optimized": optimized_summary,
        "speedup_median": (
            reference_summary["median_seconds"] / optimized_summary["median_seconds"]
            if optimized_summary["median_seconds"] > 0
            else None
        ),
        "peak_memory_ratio": (
            reference_summary["median_peak_tracemalloc_bytes"]
            / optimized_summary["median_peak_tracemalloc_bytes"]
            if optimized_summary["median_peak_tracemalloc_bytes"] > 0
            else None
        ),
        "checks": {
            "equivalent": bool(equivalence["ok"]),
            "checksums_match": reference_checksum == optimized_checksum,
        },
    }


def optimization_check(
    *,
    random_cases: int = 300,
    benchmark_repeats: int = 5,
) -> dict[str, Any]:
    """Run the TKT-17 equivalence suite and before/after benchmark."""
    equivalence = optimization_equivalence(random_cases=random_cases)
    benchmark = benchmark_optimization(repeats=benchmark_repeats)
    return {
        "schema_version": 1,
        "equivalence": equivalence,
        "benchmark": benchmark,
        "ok": bool(equivalence["ok"] and all(benchmark["checks"].values())),
    }


def make_symmetric_graph(n_vertices: int, edges: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Build a loopless symmetric adjacency matrix from weighted undirected edges."""
    adjacency = np.zeros((n_vertices, n_vertices), dtype=float)
    for first, second, weight in edges:
        if first == second:
            raise ValueError("self-loops are outside the stated input model")
        adjacency[first, second] = adjacency[second, first] = float(weight)
    return adjacency


def ce1_weighted() -> dict[str, Any]:
    """Weighted four-vertex primitive trap (review T3 / CE1)."""
    adjacency = make_symmetric_graph(
        4, ((0, 1, 1.5), (0, 2, 0.9), (0, 3, 0.9))
    )
    weights = np.ones(4)
    state = ((0,), (0,), (1,), (2,))
    gamma = 0.5
    primitive = vertex_regrets(adjacency, weights, gamma, state, 3, 3, primitive=True)
    full = vertex_regrets(adjacency, weights, gamma, state, 3, 3, primitive=False)
    hub_gains = label_gains(adjacency, weights, gamma, state, 3, 0)
    return {
        "name": "ce1_weighted",
        "n": 4,
        "q": 3,
        "cap": 3,
        "gamma": gamma,
        "edges": [[0, 1, 1.5], [0, 2, 0.9], [0, 3, 0.9]],
        "state": [list(row) for row in state],
        "hub_gains": [float(value) for value in hub_gains],
        "prefix_utilities": [
            set_score(hub_gains, (0,)),
            set_score(hub_gains, (0, 1)),
            set_score(hub_gains, (0, 1, 2)),
        ],
        "primitive": primitive,
        "full": full,
        "max_primitive_regret": max(item["regret"] for item in primitive),
        "max_full_regret": max(item["regret"] for item in full),
        "potential_before": unnormalized_potential(adjacency, weights, gamma, state, 3),
        "potential_after_hub_triple": unnormalized_potential(
            adjacency, weights, gamma, ((0, 1, 2), *state[1:]), 3
        ),
    }


def ce1_unweighted() -> dict[str, Any]:
    """Unweighted four-vertex star primitive trap (review T3)."""
    adjacency = make_symmetric_graph(4, ((0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)))
    weights = np.ones(4)
    state = ((0, 1, 2), (0,), (1, 2), (0, 1, 2))
    gamma = 0.25
    primitive = vertex_regrets(adjacency, weights, gamma, state, 3, 3, primitive=True)
    full = vertex_regrets(adjacency, weights, gamma, state, 3, 3, primitive=False)
    return {
        "name": "ce1_unweighted",
        "n": 4,
        "q": 3,
        "cap": 3,
        "gamma": gamma,
        "edges": [[0, 1, 1.0], [0, 2, 1.0], [0, 3, 1.0]],
        "state": [list(row) for row in state],
        "primitive": primitive,
        "full": full,
        "max_primitive_regret": max(item["regret"] for item in primitive),
        "max_full_regret": max(item["regret"] for item in full),
    }


def ce3_balanced_failure() -> dict[str, Any]:
    """C6: distinct equal bodies/masses do not survive cardinality changes."""
    adjacency = make_symmetric_graph(
        6,
        (
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 0, 1.0),
        ),
    )
    weights = np.ones(6)
    state = ((0, 1), (0, 1), (1, 2), (1, 2), (0, 2), (0, 2))
    factor = membership_matrix(state, 3)
    bodies = [np.flatnonzero(factor[:, label]).tolist() for label in range(3)]
    return {
        "name": "ce3_balanced_failure",
        "edges": [
            [0, 1, 1.0],
            [1, 2, 1.0],
            [2, 3, 1.0],
            [3, 4, 1.0],
            [4, 5, 1.0],
            [5, 0, 1.0],
        ],
        "q": 3,
        "cap": 2,
        "state": [list(row) for row in state],
        "bodies": bodies,
        "distinct_bodies": len({tuple(body) for body in bodies}) == 3,
        "masses": (weights @ factor).tolist(),
        "gamma0": vertex_regrets(adjacency, weights, 0.0, state, 3, 2),
        "gamma1": vertex_regrets(adjacency, weights, 1.0, state, 3, 2),
    }


def ce4_duplicate_body() -> dict[str, Any]:
    """Canonical covers coincide while the Gram matrix and potential change."""
    adjacency = make_symmetric_graph(3, ((0, 1, 1.0), (0, 2, 1.0)))
    weights = np.ones(3)
    before = ((0,), (0, 1), (1,))
    after = ((0, 2), (0, 1, 2), (1,))
    return {
        "name": "ce4_duplicate_body",
        "gamma": 0.2,
        "before": [list(row) for row in before],
        "after": [list(row) for row in after],
        "Q_before": (membership_matrix(before, 3) @ membership_matrix(before, 3).T).tolist(),
        "Q_after": (membership_matrix(after, 3) @ membership_matrix(after, 3).T).tolist(),
        "potential_before": unnormalized_potential(adjacency, weights, 0.2, before, 3),
        "potential_after": unnormalized_potential(adjacency, weights, 0.2, after, 3),
    }


def ce6_externality() -> dict[str, Any]:
    """Payoff of v falls when u adds a private label; the body of A is unchanged."""
    adjacency = make_symmetric_graph(2, ((0, 1, 1.0),))
    weights = np.ones(2)
    before = ((0,), (0,))
    after = ((0, 1), (0,))
    before_utility = set_score(label_gains(adjacency, weights, 0.0, before, 2, 1), before[1])
    after_utility = set_score(label_gains(adjacency, weights, 0.0, after, 2, 1), after[1])
    return {
        "name": "ce6_externality",
        "gamma": 0.0,
        "before": [list(row) for row in before],
        "after": [list(row) for row in after],
        "U_before": before_utility,
        "U_after": after_utility,
        "shared_body_unchanged": True,
    }


def ce7_l1_isolate() -> dict[str, Any]:
    """Unit-ℓ₁ isolate prefers two private labels once the self term is kept."""
    adjacency = make_symmetric_graph(3, ((1, 2, 1.0),))
    weights = np.ones(3)

    def l1_value(state: Sequence[Sequence[int]]) -> float:
        factor = np.zeros((3, 3), dtype=float)
        for vertex, labels in enumerate(state):
            factor[vertex, list(labels)] = 1.0 / len(labels)
        return float(
            0.5 * np.sum(adjacency * (factor @ factor.T))
            - 0.25 * np.sum((weights @ factor) ** 2)
        )

    single = l1_value(((0,), (2,), (2,)))
    split = l1_value(((0, 1), (2,), (2,)))
    return {
        "name": "ce7_l1_isolate",
        "U_single_private": single,
        "U_two_private": split,
        "gain": split - single,
    }


def named_examples() -> dict[str, Any]:
    """Return every named counterexample used by the independent oracle tests."""
    return {
        "ce1_weighted": ce1_weighted(),
        "ce1_unweighted": ce1_unweighted(),
        "ce3_balanced_failure": ce3_balanced_failure(),
        "ce4_duplicate_body": ce4_duplicate_body(),
        "ce5_token_normalization": ce5_token_normalization(),
        "ce6_externality": ce6_externality(),
        "ce7_l1_isolate": ce7_l1_isolate(),
        "collision_projection": collision_projection(),
        "restricted_balance_token_identity": restricted_balance_token_identity(),
    }


def random_identity_errors(
    *,
    cases: int = 3000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Check potential/delta, prefix completeness, and the disjoint offset."""
    rng = np.random.default_rng(seed)
    delta_error = 0.0
    gram_error = 0.0
    prefix_error = 0.0
    pair_error = 0.0
    disjoint_error = 0.0
    for _ in range(cases):
        n_vertices = int(rng.integers(2, 8))
        label_count = int(rng.integers(1, 6))
        cap = int(rng.integers(1, label_count + 1))
        actions = admissible_actions(label_count, cap)
        adjacency = np.triu(rng.integers(0, 8, (n_vertices, n_vertices)) / 4.0, 1)
        adjacency = adjacency + adjacency.T
        weights = rng.integers(0, 6, n_vertices) / 3.0
        gamma = float(rng.integers(0, 11) / 10.0)
        state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
        vertex = int(rng.integers(n_vertices))
        candidate = actions[int(rng.integers(len(actions)))]
        gains = label_gains(adjacency, weights, gamma, state, label_count, vertex)
        nxt = list(state)
        nxt[vertex] = candidate
        delta = unnormalized_potential(
            adjacency, weights, gamma, nxt, label_count
        ) - unnormalized_potential(adjacency, weights, gamma, state, label_count)
        delta_error = max(
            delta_error,
            abs(delta - (set_score(gains, candidate) - set_score(gains, state[vertex]))),
        )
        factor = membership_matrix(state, label_count)
        gram = float(
            np.sum((adjacency - gamma * np.outer(weights, weights)) * (factor @ factor.T))
            / 2.0
        )
        gram_error = max(
            gram_error,
            abs(gram - unnormalized_potential(adjacency, weights, gamma, state, label_count)),
        )
        pair_error = max(
            pair_error,
            abs(
                off_diagonal_potential(adjacency, weights, gamma, state, label_count)
                - (
                    unnormalized_potential(adjacency, weights, gamma, state, label_count)
                    + (gamma / 2.0) * float(np.sum(weights * weights))
                )
            ),
        )
        _, prefix_utility, _ = prefix_best_response(gains, state[vertex], cap)
        _, exhaustive_utility, _ = exhaustive_best_response(gains, state[vertex], cap)
        prefix_error = max(prefix_error, abs(prefix_utility - exhaustive_utility))
        partition = [(int(rng.integers(label_count)),) for _ in range(n_vertices)]
        pair_sum = sum(
            float(adjacency[first, second] - gamma * weights[first] * weights[second])
            for first in range(n_vertices)
            for second in range(first + 1, n_vertices)
            if partition[first] == partition[second]
        )
        disjoint_error = max(
            disjoint_error,
            abs(
                unnormalized_potential(adjacency, weights, gamma, partition, label_count)
                - (pair_sum - gamma * float(np.sum(weights * weights)) / 2.0)
            ),
        )
    return {
        "cases": cases,
        "seed": seed,
        "numpy": np.__version__,
        "max_delta_error": delta_error,
        "max_gram_error": gram_error,
        "max_pair_identity_error": pair_error,
        "max_prefix_error": prefix_error,
        "max_disjoint_offset_error": disjoint_error,
    }


def nonnegative_tuples_summing_to(total: int, parts: int) -> int:
    """Count nonnegative integer solutions of ``x_1+…+x_parts = total``.

    Recursion only. This is the stars-and-bars count used by the
    coefficient bound, derived independently of :func:`math.comb`.
    """
    if parts < 0 or total < 0:
        return 0
    if parts == 0:
        return int(total == 0)
    if parts == 1:
        return 1
    return sum(
        nonnegative_tuples_summing_to(total - first, parts - 1)
        for first in range(total + 1)
    )


def independent_coefficient_value_bound(
    edge_weight_sum: int,
    pair_weight_sum: int,
    cap: int,
) -> int:
    """Coefficient bound by enumerating slack-augmented compositions.

    ``D_M`` multiplicity pairs plus one slack variable sum to ``M W``
    (respectively ``M H``). The product is ``N_val``. Does not call
    :func:`coefficient_value_bound` or :func:`math.comb`.
    """
    if edge_weight_sum < 0 or pair_weight_sum < 0 or cap < 1:
        raise ValueError("weight sums must be nonnegative and cap >= 1")
    multiplicity_pairs = cap * (cap + 1) // 2
    edge_count = nonnegative_tuples_summing_to(
        cap * edge_weight_sum, multiplicity_pairs + 1
    )
    pair_count = nonnegative_tuples_summing_to(
        cap * pair_weight_sum, multiplicity_pairs + 1
    )
    return edge_count * pair_count


def coefficient_value_bound(
    edge_weight_sum: int,
    pair_weight_sum: int,
    cap: int,
) -> int:
    """Stars-and-bars upper bound on distinct ``(E,C)`` coefficient vectors.

    Pair coefficients are grouped by multiplicity pairs, of which there are
    ``D_M = M(M+1)/2``. Each shared-label count is at most ``M``, so the
    totals are at most ``M W`` and ``M H``. This bounds accepted strict
    improvements, not running time.
    """
    if edge_weight_sum < 0 or pair_weight_sum < 0 or cap < 1:
        raise ValueError("weight sums must be nonnegative and cap >= 1")
    multiplicity_pairs = cap * (cap + 1) // 2
    edge_total = cap * edge_weight_sum
    pair_total = cap * pair_weight_sum
    return math.comb(edge_total + multiplicity_pairs, multiplicity_pairs) * math.comb(
        pair_total + multiplicity_pairs, multiplicity_pairs
    )


def stars_and_bars_prefix_sum(total: int, parts: int) -> int:
    """Count nonnegative ``x_1+…+x_parts = total`` by prefix-sum DP.

    Independent of :func:`math.comb` and of the first-coordinate recursion
    in :func:`nonnegative_tuples_summing_to`.
    """
    if parts < 0 or total < 0:
        return 0
    counts = [1] + [0] * total
    for _ in range(parts):
        running = 0
        nxt = [0] * (total + 1)
        for index in range(total + 1):
            running += counts[index]
            nxt[index] = running
        counts = nxt
    return counts[total]


def endpoint_affine_errors(
    *,
    cases: int = 200,
    seed: int = SEED + 11,
) -> dict[str, Any]:
    """Check that a fixed deviation's potential gap is affine in ``γ``."""

    rng = np.random.default_rng(seed)
    max_affine = 0.0
    interior_failures = 0
    interior_checks = 0
    for _ in range(cases):
        n_vertices = int(rng.integers(2, 7))
        label_count = int(rng.integers(1, 5))
        cap = int(rng.integers(1, label_count + 1))
        actions = admissible_actions(label_count, cap)
        adjacency = np.triu(rng.integers(0, 6, (n_vertices, n_vertices)) / 3.0, 1)
        adjacency = adjacency + adjacency.T
        weights = rng.integers(0, 5, n_vertices) / 2.0
        state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
        vertex = int(rng.integers(n_vertices))
        candidate = actions[int(rng.integers(len(actions)))]
        nxt = list(state)
        nxt[vertex] = candidate

        def gap(gamma: float) -> float:
            return unnormalized_potential(
                adjacency, weights, gamma, nxt, label_count
            ) - unnormalized_potential(
                adjacency, weights, gamma, state, label_count
            )

        gap0 = gap(0.0)
        gap1 = gap(1.0)
        for interior in (0.25, 0.5, 0.75):
            predicted = (1.0 - interior) * gap0 + interior * gap1
            max_affine = max(max_affine, abs(gap(interior) - predicted))
        if gap0 <= 1e-12 and gap1 <= 1e-12:
            interior_checks += 1
            for interior in (0.25, 0.5, 0.75):
                if gap(interior) > 1e-9:
                    interior_failures += 1
    return {
        "cases": cases,
        "seed": seed,
        "max_affine_error": max_affine,
        "interior_endpoint_checks": interior_checks,
        "interior_failures": interior_failures,
    }


def proof_reconstruction(*, affine_cases: int = 200) -> dict[str, Any]:
    """Independently reconstruct the numbered manuscript propositions.

    This is a machine check of the short proofs, not a named coauthor
    referee report.
    """
    examples = named_examples()
    assert_named_examples(examples)
    weighted = examples["ce1_weighted"]
    u1 = 1.0
    u2 = 7.0 / (5.0 * math.sqrt(2.0))
    u3 = 9.0 / (5.0 * math.sqrt(3.0))
    prefix = weighted["prefix_utilities"]
    ce1_ok = all(
        (
            abs(prefix[0] - u1) < 1e-12,
            abs(prefix[1] - u2) < 1e-12,
            abs(prefix[2] - u3) < 1e-12,
            prefix[1] < prefix[0],
            prefix[2] > prefix[0],
            weighted["max_primitive_regret"] < 1e-10,
            weighted["max_full_regret"] > 0.03,
        )
    )
    n_actions = len(admissible_actions(3, 2))
    n_profiles = n_actions**2
    finite_ok = n_actions == 6 and n_profiles == 36
    affine = endpoint_affine_errors(cases=affine_cases)
    affine_ok = affine["max_affine_error"] < 1e-10 and affine["interior_failures"] == 0
    random = random_identity_errors(cases=80, seed=SEED)
    prefix_ok = random["max_prefix_error"] < 1e-10
    expansion = token_coefficient_errors(cases=80, seed=TOKEN_SEED)
    expansion_ok = expansion["max_coefficient_expansion_error"] < 1e-10
    balance = examples["restricted_balance_token_identity"]
    balanced = examples["ce3_balanced_failure"]
    balance_ok = (
        balance["abs_error"] < 1e-12
        and balance["equal_masses"]
        and abs(
            balance["crowding_increment"]
            - balance["expected_crowding_increment"]
        )
        < 1e-12
        and balance["crowding_increment"] >= -1e-12
        and balance["support_gain_gamma0"] <= 1e-12
        and balanced["distinct_bodies"]
        and max(item["regret"] for item in balanced["gamma0"]) < 1e-10
        and max(item["regret"] for item in balanced["gamma1"]) > 1e-8
    )
    comb = coefficient_value_bound(3, 6, 1)
    slack = independent_coefficient_value_bound(3, 6, 1)
    dp = stars_and_bars_prefix_sum(3, 2) * stars_and_bars_prefix_sum(6, 2)
    m1_formula = (3 + 1) * (6 + 1)
    coefficient_ok = comb == slack == dp == m1_formula == 28
    comb2 = coefficient_value_bound(3, 6, 2)
    slack2 = independent_coefficient_value_bound(3, 6, 2)
    pairs = 2 * 3 // 2
    dp2 = stars_and_bars_prefix_sum(6, pairs + 1) * stars_and_bars_prefix_sum(
        12, pairs + 1
    )
    coefficient_m2_ok = comb2 == slack2 == dp2
    propositions = {
        "finite_improvement": {
            "ok": finite_ok,
            "actions_per_vertex_q3_M2": n_actions,
            "labelled_profiles_n2": n_profiles,
        },
        "prefix_completeness_ce1": {
            "ok": ce1_ok and prefix_ok,
            "hub_gains": weighted["hub_gains"],
            "prefix_utilities": prefix,
            "closed_form_prefixes": [u1, u2, u3],
            "max_prefix_vs_exhaustive_error": random["max_prefix_error"],
        },
        "endpoint_interval": {
            "ok": affine_ok,
            **affine,
        },
        "restricted_balance_ce3": {
            "ok": balance_ok,
            "token_identity_abs_error": balance["abs_error"],
            "crowding_increment": balance["crowding_increment"],
            "expected_crowding_increment": balance[
                "expected_crowding_increment"
            ],
            "c6_distinct_bodies": balanced["distinct_bodies"],
            "c6_max_gamma1_regret": max(
                item["regret"] for item in balanced["gamma1"]
            ),
        },
        "coefficient_counting": {
            "ok": coefficient_ok and coefficient_m2_ok and expansion_ok,
            "M1_W3_H6": {
                "closed_form_binomial": comb,
                "slack_recursion": slack,
                "prefix_sum_dp": dp,
                "special_M1_formula": m1_formula,
                "hand_check": "four vertices, three unweighted edges: (3+1)(6+1)=28",
            },
            "M2_W3_H6_three_methods_agree": coefficient_m2_ok,
            "grouped_expansion_r6": {
                "cases": expansion["cases"],
                "max_coefficient_expansion_error": expansion[
                    "max_coefficient_expansion_error"
                ],
            },
            "qualification": (
                "accepted-move bound, not wall-clock; not a named coauthor "
                "referee report"
            ),
        },
    }
    ok = all(item["ok"] for item in propositions.values())
    return {
        "schema_version": 1,
        "ok": ok,
        "named_coauthor_referee_report": False,
        "propositions": propositions,
    }


def fee_identity_errors(
    *,
    cases: int = 80,
    seed: int = FEE_SEED,
) -> dict[str, Any]:
    """Check (R9) potential/delta and prefix completeness for a positive fee."""
    rng = np.random.default_rng(seed)
    delta_error = 0.0
    pair_delta_error = 0.0
    prefix_error = 0.0
    for _ in range(cases):
        n_vertices = int(rng.integers(2, 7))
        label_count = int(rng.integers(2, 6))
        cap = int(rng.integers(1, label_count + 1))
        tau = float(rng.choice([0.25, 1.0, 2.5]))
        actions = admissible_actions(label_count, cap)
        adjacency = np.triu(rng.integers(0, 8, (n_vertices, n_vertices)) / 4.0, 1)
        adjacency = adjacency + adjacency.T
        weights = rng.integers(0, 6, n_vertices) / 3.0
        gamma = float(rng.integers(0, 11) / 10.0)
        state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
        vertex = int(rng.integers(n_vertices))
        candidate = actions[int(rng.integers(len(actions)))]
        gains = label_gains(adjacency, weights, gamma, state, label_count, vertex)
        nxt = list(state)
        nxt[vertex] = candidate
        delta = fee_unnormalized_potential(
            adjacency, weights, gamma, nxt, label_count, tau
        ) - fee_unnormalized_potential(
            adjacency, weights, gamma, state, label_count, tau
        )
        pair_delta = fee_pair_potential(
            adjacency, weights, gamma, nxt, label_count, tau
        ) - fee_pair_potential(
            adjacency, weights, gamma, state, label_count, tau
        )
        expected = fee_set_score(gains, candidate, tau) - fee_set_score(
            gains, state[vertex], tau
        )
        delta_error = max(delta_error, abs(delta - expected))
        pair_delta_error = max(pair_delta_error, abs(pair_delta - expected))
        _, prefix_utility, _ = prefix_best_response(
            gains, state[vertex], cap, tau=tau
        )
        _, exhaustive_utility, _ = exhaustive_best_response(
            gains, state[vertex], cap, tau=tau
        )
        prefix_error = max(prefix_error, abs(prefix_utility - exhaustive_utility))
    return {
        "cases": cases,
        "seed": seed,
        "max_delta_error": delta_error,
        "max_pair_delta_error": pair_delta_error,
        "max_prefix_error": prefix_error,
    }


def fee_units_homogeneity_check() -> dict[str, Any]:
    """Check that ``tau`` scales in the same units as every pair coefficient."""
    adjacency = make_symmetric_graph(
        4, ((0, 1, 1.5), (0, 2, 0.9), (0, 3, 0.9))
    )
    weights = np.ones(4)
    gamma = 0.5
    tau = 0.2
    scale = 7.0
    state = ((0,), (0,), (1,), (2,))
    gains = label_gains(adjacency, weights, gamma, state, 3, 0)
    scaled_gains = label_gains(
        scale * adjacency, weights, scale * gamma, state, 3, 0
    )
    action, utility, _ = prefix_best_response(gains, state[0], 3, tau=tau)
    scaled_action, scaled_utility, _ = prefix_best_response(
        scaled_gains, state[0], 3, tau=scale * tau
    )
    potential = fee_pair_potential(
        adjacency, weights, gamma, state, 3, tau
    )
    scaled_potential = fee_pair_potential(
        scale * adjacency, weights, scale * gamma, state, 3, scale * tau
    )
    checks = {
        "gains_scale": bool(np.allclose(scaled_gains, scale * gains, atol=1e-12)),
        "best_action_invariant": scaled_action == action,
        "utility_scales": abs(scaled_utility - scale * utility) < 1e-12,
        "potential_scales": abs(scaled_potential - scale * potential) < 1e-12,
    }
    return {
        "pair_coefficient_scale": scale,
        "tau_scale": scale,
        "tau_units": "same units as B_uv, U_v, and pair potential P_gamma",
        "checks": checks,
        "ok": all(checks.values()),
    }


def anonymous_label_signature(
    state: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    """Canonical multiset of nonempty label bodies; duplicate bodies survive."""
    bodies: dict[int, list[int]] = {}
    for vertex, labels in enumerate(state):
        row = tuple(int(label) for label in labels)
        if not row or len(row) != len(set(row)):
            raise ValueError("membership rows must be nonempty sets")
        for label in row:
            bodies.setdefault(label, []).append(vertex)
    return tuple(sorted(tuple(vertices) for vertices in bodies.values()))


def positive_pair_potential_upper_bound(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
) -> float:
    """Return ``P_max = sum_{u<v} max(B_uv, 0)`` from T16."""
    pair = adjacency - float(gamma) * np.outer(weights, weights)
    return float(sum(max(float(pair[u, v]), 0.0) for u, v in combinations(range(len(weights)), 2)))


def fee_improvement_path(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    cap: int,
    tau: float,
    *,
    sweeps: int = 64,
) -> dict[str, Any]:
    """Run sequential fee best responses and check T16's pathwise bound."""
    tau = _validated_fee(tau)
    if sweeps < 1:
        raise ValueError("sweeps must be positive")
    membership_matrix(state, label_count)
    current = [tuple(int(label) for label in row) for row in state]
    psi_start = fee_pair_potential(
        adjacency, weights, gamma, current, label_count, tau
    )
    p_max = positive_pair_potential_upper_bound(adjacency, weights, gamma)
    incidence_bound = (
        None
        if tau == 0.0
        else len(current) + (p_max - psi_start) / tau
    )
    seen = {anonymous_label_signature(current)}
    steps: list[dict[str, Any]] = []
    terminated = False
    for _ in range(sweeps):
        moved = False
        for vertex in range(len(current)):
            gains = label_gains(adjacency, weights, gamma, current, label_count, vertex)
            action, _, regret = prefix_best_response(
                gains, current[vertex], cap, tau=tau
            )
            if regret > 1e-12:
                before = fee_pair_potential(
                    adjacency, weights, gamma, current, label_count, tau
                )
                current[vertex] = action
                after = fee_pair_potential(
                    adjacency, weights, gamma, current, label_count, tau
                )
                signature = anonymous_label_signature(current)
                incidence = sum(len(row) for row in current)
                steps.append(
                    {
                        "vertex": vertex,
                        "regret": float(regret),
                        "potential_before": before,
                        "potential_after": after,
                        "delta_error": abs((after - before) - float(regret)),
                        "incidences": incidence,
                        "within_pathwise_bound": incidence_bound is None
                        or incidence <= incidence_bound + 1e-9,
                        "anonymous_signature_new": signature not in seen,
                    }
                )
                seen.add(signature)
                moved = True
        if not moved:
            terminated = True
            break
    final_regret = 0.0
    for vertex, labels in enumerate(current):
        gains = label_gains(adjacency, weights, gamma, current, label_count, vertex)
        _, _, regret = prefix_best_response(gains, labels, cap, tau=tau)
        final_regret = max(final_regret, float(regret))
    checks = {
        "strict_potential_increase": all(
            row["potential_after"] > row["potential_before"] + 1e-12
            for row in steps
        ),
        "delta_matches_regret": all(row["delta_error"] < 1e-10 for row in steps),
        "anonymous_quotient_no_repeat": all(
            row["anonymous_signature_new"] for row in steps
        ),
        "pathwise_incidence_bound": tau == 0.0
        or all(row["within_pathwise_bound"] for row in steps),
        "terminal_full_response": terminated and final_regret <= 1e-12,
    }
    return {
        "state": [list(row) for row in current],
        "terminated": terminated,
        "steps": steps,
        "accepted_moves": len(steps),
        "initial_fee_pair_potential": psi_start,
        "positive_pair_potential_upper_bound": p_max,
        "pathwise_incidence_bound": incidence_bound,
        "max_observed_incidences": max(
            [sum(len(row) for row in state)]
            + [int(row["incidences"]) for row in steps]
        ),
        "final_full_regret": final_regret,
        "checks": checks,
        "ok": all(checks.values()),
    }


def fee_improve_to_equilibrium(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    cap: int,
    tau: float,
    *,
    sweeps: int = 64,
) -> tuple[list[tuple[int, ...]], bool]:
    """Apply sequential prefix best responses of the fee objective."""
    report = fee_improvement_path(
        adjacency,
        weights,
        gamma,
        state,
        label_count,
        cap,
        tau,
        sweeps=sweeps,
    )
    return [tuple(row) for row in report["state"]], bool(report["terminated"])


def fee_r10_holds(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
    tau: float,
) -> dict[str, Any]:
    """Check (R10) on a labelled profile that has unused private labels."""
    tau = _validated_fee(tau)
    active_labels = {
        int(label) for labels in state for label in labels
    }
    unused_labels = sorted(set(range(label_count)) - active_labels)
    max_excess = 0.0
    max_degree_gap = 0.0
    max_bound_gap = 0.0
    max_regret = 0.0
    for vertex, labels in enumerate(state):
        gains = label_gains(adjacency, weights, gamma, state, label_count, vertex)
        utility = set_score(gains, labels)
        degree = positive_pair_degree(adjacency, weights, gamma, vertex)
        cardinality = len(tuple(labels))
        _, _, regret = prefix_best_response(
            gains, labels, label_count, tau=tau
        )
        max_regret = max(max_regret, float(regret))
        max_excess = max(max_excess, tau * (cardinality - 1) - utility)
        max_degree_gap = max(max_degree_gap, utility - degree)
        if tau > 0:
            allowed = 1 + math.floor(degree / tau + 1e-12)
            max_bound_gap = max(max_bound_gap, cardinality - allowed)
    return {
        "full_equilibrium": max_regret <= 1e-12,
        "fresh_private_singleton_available": bool(unused_labels),
        "unused_labels": unused_labels,
        "fee_le_utility": max_excess <= 1e-9,
        "utility_le_positive_degree": max_degree_gap <= 1e-9,
        "cardinality_bound": max_bound_gap <= 0,
        "max_full_regret": max_regret,
        "max_fee_minus_utility": max_excess,
        "max_utility_minus_degree": max_degree_gap,
        "max_cardinality_over_bound": max_bound_gap,
    }


def fee_complete_graph_ablation() -> dict[str, Any]:
    """Graph-only fee sweep on K4; independent of frozen manuscript ledgers."""
    n_vertices = 4
    adjacency = np.ones((n_vertices, n_vertices)) - np.eye(n_vertices)
    weights = np.ones(n_vertices)
    gamma = 0.0
    label_count = 6
    cap = label_count
    start = [(0, 1, 2) for _ in range(n_vertices)]
    gain_scale = max(
        positive_pair_degree(adjacency, weights, gamma, vertex)
        for vertex in range(n_vertices)
    )
    rows = []
    for tau in (0.0, 0.25, 1.0, 2.0, 8.0):
        path = fee_improvement_path(
            adjacency, weights, gamma, start, label_count, cap, tau
        )
        state = [tuple(row) for row in path["state"]]
        terminated = bool(path["terminated"])
        mean_k = float(np.mean([len(row) for row in state]))
        r10 = (
            fee_r10_holds(adjacency, weights, gamma, state, label_count, tau)
            if tau > 0
            else {"ok_skipped_zero_fee": True}
        )
        rows.append(
            {
                "tau": tau,
                "tau_over_positive_gain_scale": tau / gain_scale,
                "terminated": terminated,
                "mean_memberships": mean_k,
                "state": [list(row) for row in state],
                "r10": r10,
                "path": path,
            }
        )
    large = next(row for row in rows if row["tau"] == 8.0)
    small = next(row for row in rows if row["tau"] == 0.0)
    return {
        "graph": "K4_unweighted",
        "gamma": gamma,
        "label_count": label_count,
        "hard_cap_equals_label_bank": cap == label_count,
        "positive_gain_scale": gain_scale,
        "start": [list(row) for row in start],
        "rows": rows,
        "large_fee_mean_memberships": large["mean_memberships"],
        "zero_fee_mean_memberships": small["mean_memberships"],
        "disjointness_visible": large["mean_memberships"] == 1.0
        and small["mean_memberships"] > 1.0,
        "not_a_recovery_claim": True,
    }


def fee_v1_check(*, identity_cases: int = 80) -> dict[str, Any]:
    """Machine-check the separate membership-fee objective (TKT-16).

    This is not the manuscript model and must not reuse frozen certificates.
    """
    identities = fee_identity_errors(cases=identity_cases)
    units = fee_units_homogeneity_check()
    ablation = fee_complete_graph_ablation()
    tau0_gains = np.array([1.0, 0.4, 0.4])
    zero_fee_action, zero_fee_utility, _ = prefix_best_response(
        tau0_gains, (0,), 3, tau=0.0
    )
    original_action, original_utility, _ = prefix_best_response(tau0_gains, (0,), 3)
    unused_ok = all(
        row["r10"]["full_equilibrium"]
        and row["r10"]["fresh_private_singleton_available"]
        and row["r10"]["fee_le_utility"]
        and row["r10"]["utility_le_positive_degree"]
        and row["r10"]["cardinality_bound"]
        for row in ablation["rows"]
        if row["tau"] > 0
    )
    checks = {
        "delta_identity": max(
            identities["max_delta_error"], identities["max_pair_delta_error"]
        )
        < 1e-10,
        "prefix_completeness": identities["max_prefix_error"] < 1e-10,
        "fee_units_homogeneity": units["ok"],
        "zero_fee_matches_manuscript_oracle": (
            zero_fee_action == original_action
            and abs(zero_fee_utility - original_utility) < 1e-15
        ),
        "r10_on_k4_equilibria": unused_ok,
        "anonymous_path_bound_on_k4": all(
            row["path"]["ok"] for row in ablation["rows"] if row["tau"] > 0
        ),
        "fee_induced_disjointness_visible": ablation["disjointness_visible"],
        "ablation_terminated": all(row["terminated"] for row in ablation["rows"]),
    }
    ok = all(checks.values())
    return {
        "schema_version": 1,
        "objective_version": FEE_OBJECTIVE_VERSION,
        "manuscript_objective_version": MANUSCRIPT_OBJECTIVE_VERSION,
        "not_the_manuscript_model": True,
        "does_not_reuse_original_certificates": True,
        "regularization_does_not_imply_recovery": True,
        "ok": ok,
        "checks": checks,
        "identities": identities,
        "units": units,
        "ablation": ablation,
        "anonymous_label_termination": {
            "incidence_bound": "T <= n + (P_max - Psi_initial) / tau",
            "reason": (
                "strict improvement cannot repeat an anonymous-label quotient "
                "state, and the incidence bound makes that quotient finite"
            ),
            "fixture_paths_checked": sum(
                row["path"]["accepted_moves"]
                for row in ablation["rows"]
                if row["tau"] > 0
            ),
            "ok": checks["anonymous_path_bound_on_k4"],
        },
    }


def token_coefficient_errors(
    *,
    cases: int = 500,
    seed: int = TOKEN_SEED,
) -> dict[str, Any]:
    """Check frozen-multiplicity unnormalized token identity on random instances."""
    rng = np.random.default_rng(seed)
    token_errors: list[float] = []
    count_errors: list[float] = []
    for _ in range(cases):
        n_vertices = int(rng.integers(2, 8))
        label_count = int(rng.integers(1, 6))
        cap = int(rng.integers(1, label_count + 1))
        actions = admissible_actions(label_count, cap)
        state = [actions[int(rng.integers(len(actions)))] for _ in range(n_vertices)]
        multiplicity = np.array([len(labels) for labels in state])
        factor = membership_matrix(state, label_count)
        adjacency = np.triu(rng.integers(0, 4, (n_vertices, n_vertices)), 1)
        adjacency = adjacency + adjacency.T
        weights = rng.integers(0, 4, n_vertices).astype(float)
        gamma = math.sqrt(2.0) / 3.0
        tokens = [(vertex, label) for vertex, labels in enumerate(state) for label in labels]
        masses = np.zeros(label_count)
        token_edges = 0.0
        for vertex, label in tokens:
            masses[label] += weights[vertex] / math.sqrt(multiplicity[vertex])
        for first_index, (first, first_label) in enumerate(tokens):
            for second, second_label in tokens[first_index + 1 :]:
                if first == second:
                    continue
                contribution = adjacency[first, second] / math.sqrt(
                    multiplicity[first] * multiplicity[second]
                )
                if first_label == second_label:
                    token_edges += contribution
        direct = float(
            np.sum(np.triu(adjacency * (factor @ factor.T), 1))
            - 0.5 * gamma * np.sum((weights @ factor) ** 2)
        )
        token_value = float(token_edges - 0.5 * gamma * (masses @ masses))
        token_errors.append(abs(direct - token_value))
        edge_counts: dict[tuple[int, int], int] = {}
        crowding_counts: dict[tuple[int, int], int] = {}
        for first in range(n_vertices):
            for second in range(first + 1, n_vertices):
                key = tuple(sorted((int(multiplicity[first]), int(multiplicity[second]))))
                shared = len(set(state[first]) & set(state[second]))
                edge_counts[key] = edge_counts.get(key, 0) + int(adjacency[first, second]) * shared
                crowding_counts[key] = crowding_counts.get(key, 0) + int(
                    weights[first] * weights[second]
                ) * shared
        expansion = sum(
            (edge_counts[key] - gamma * crowding_counts[key]) / math.sqrt(key[0] * key[1])
            for key in edge_counts
        )
        count_errors.append(abs(expansion - (direct + 0.5 * gamma * float(weights @ weights))))
        edge_weight_sum = int(np.sum(np.triu(adjacency, 1)))
        pair_weight_sum = int(
            sum(
                weights[first] * weights[second]
                for first in range(n_vertices)
                for second in range(first + 1, n_vertices)
            )
        )
        if sum(edge_counts.values()) > cap * edge_weight_sum:
            count_errors.append(1.0)
        if sum(crowding_counts.values()) > cap * pair_weight_sum:
            count_errors.append(1.0)
    return {
        "cases": cases,
        "seed": seed,
        "max_unnormalized_token_error": max(token_errors),
        "max_coefficient_expansion_error": max(count_errors),
    }


def unnormalized_token_potential(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
) -> float:
    """Frozen-multiplicity token CPM matching ``Φ̃_γ`` when labels do not collide.

    Token edge ``(u,c)—(v,c)`` carries ``a_{uv}/√(k_u k_v)``. Vertex masses
    are ``w_v/√k_v`` on each of ``v``'s labels.
    """
    tokens = frozen_tokens(state)
    multiplicity = [len(labels) for labels in state]
    label_count = max((label for _vertex, label in tokens), default=-1) + 1
    masses = np.zeros(label_count)
    token_edges = 0.0
    for vertex, label in tokens:
        masses[label] += weights[vertex] / math.sqrt(multiplicity[vertex])
    for first_index, (first, first_label) in enumerate(tokens):
        for second, second_label in tokens[first_index + 1 :]:
            if first == second or first_label != second_label:
                continue
            token_edges += adjacency[first, second] / math.sqrt(
                multiplicity[first] * multiplicity[second]
            )
    return float(token_edges - 0.5 * gamma * float(masses @ masses))


def restricted_balance_token_identity() -> dict[str, Any]:
    """Equal-mass, fixed-k replacement: crowding cannot create a gain."""
    adjacency = make_symmetric_graph(
        6,
        (
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 0, 1.0),
        ),
    )
    weights = np.ones(6)
    gamma = 1.0
    before = ((0, 1), (0, 1), (1, 2), (1, 2), (0, 2), (0, 2))
    after = ((0, 2), *before[1:])
    label_count = 3
    factor = membership_matrix(before, label_count)
    masses = weights @ factor
    current = before[0]
    candidate = after[0]
    other_masses = masses - weights[0] * factor[0]
    current_exposure = weights[0] * sum(other_masses[list(current)]) / math.sqrt(
        len(current)
    )
    candidate_exposure = weights[0] * sum(
        other_masses[list(candidate)]
    ) / math.sqrt(len(candidate))
    overlap = len(set(current) & set(candidate))
    expected_increment = weights[0] ** 2 * (1.0 - overlap / len(current))
    gains0 = label_gains(adjacency, weights, 0.0, before, label_count, 0)
    support_gain = set_score(gains0, candidate) - set_score(gains0, current)
    delta_original = unnormalized_potential(
        adjacency, weights, gamma, after, label_count
    ) - unnormalized_potential(adjacency, weights, gamma, before, label_count)
    delta_token = unnormalized_token_potential(
        adjacency, weights, gamma, after
    ) - unnormalized_token_potential(adjacency, weights, gamma, before)
    return {
        "name": "restricted_balance_token_identity",
        "k": 2,
        "before": [list(row) for row in before],
        "after": [list(row) for row in after],
        "equal_masses": bool(np.allclose(masses, masses[0])),
        "current_exposure": current_exposure,
        "candidate_exposure": candidate_exposure,
        "crowding_increment": candidate_exposure - current_exposure,
        "expected_crowding_increment": expected_increment,
        "support_gain_gamma0": support_gain,
        "delta_original": delta_original,
        "delta_token": delta_token,
        "abs_error": abs(delta_original - delta_token),
    }


def frozen_tokens(state: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    """Return frozen-multiplicity tokens ``(vertex, label)`` in vertex order."""
    tokens: list[tuple[int, int]] = []
    for vertex, labels in enumerate(state):
        for label in labels:
            tokens.append((vertex, int(label)))
    return tokens


def token_offsets(state: Sequence[Sequence[int]]) -> list[int]:
    """Return prefix sums of per-vertex multiplicities, length ``n+1``."""
    offsets = [0]
    for labels in state:
        offsets.append(offsets[-1] + len(labels))
    return offsets


def token_total_edge_weight(
    adjacency: np.ndarray,
    state: Sequence[Sequence[int]],
) -> float:
    """Return ``W_tok = ∑_{u<v} a_{uv} √(k_u k_v)`` for the frozen token graph."""
    total = 0.0
    n_vertices = len(state)
    for first in range(n_vertices):
        k_first = len(state[first])
        if k_first < 1:
            raise ValueError("token expansion requires nonempty membership rows")
        for second in range(first + 1, n_vertices):
            weight = float(adjacency[first, second])
            if weight == 0.0:
                continue
            k_second = len(state[second])
            total += weight * math.sqrt(k_first * k_second)
    return total


def original_normalized_quality(
    adjacency: np.ndarray,
    weights: np.ndarray,
    gamma: float,
    state: Sequence[Sequence[int]],
    label_count: int,
) -> float:
    """Return ``Φ_γ = Φ̃_γ / W`` on the original graph."""
    original_weight = 0.5 * float(np.sum(adjacency))
    if original_weight <= 0:
        raise ValueError("normalized quality is undefined when W=0")
    return (
        unnormalized_potential(adjacency, weights, gamma, state, label_count)
        / original_weight
    )


def project_token_labels(
    token_labels: Sequence[int],
    offsets: Sequence[int],
) -> tuple[list[list[int]], list[dict[str, int]]]:
    """Collapse per-vertex token labels to a set and record collision lists.

    Native C reports collisions only as a boolean. This projector keeps every
    ``(vertex, label, token_count)`` event so a decreasing projection can be
    named rather than inferred.
    """
    if not offsets or offsets[0] != 0 or offsets[-1] != len(token_labels):
        raise ValueError("token offsets must cover token_labels")
    rows: list[list[int]] = []
    collisions: list[dict[str, int]] = []
    for vertex in range(len(offsets) - 1):
        chunk = [int(label) for label in token_labels[offsets[vertex] : offsets[vertex + 1]]]
        if not chunk:
            raise ValueError("a projected membership row must be nonempty")
        counts: dict[int, int] = {}
        for label in chunk:
            counts[label] = counts.get(label, 0) + 1
        for label, count in sorted(counts.items()):
            if count > 1:
                collisions.append(
                    {"vertex": vertex, "label": label, "token_count": count}
                )
        rows.append(sorted(counts))
    return rows, collisions


def original_potential_nondecreasing(
    before: float,
    after: float,
    *,
    atol: float = 1e-12,
) -> bool:
    """Return whether an original-unit potential is acceptable after projection."""
    return after + atol >= before


def ce5_token_normalization() -> dict[str, Any]:
    """Original ``Φ = 1/√2`` vs token-normalized quality ``1/2`` at ``γ=0``.

    One edge of weight one, memberships ``(AB, A)``. Frozen tokenization
    creates two token edges of total weight ``√2``; only the matching-A
    edge is internal.
    """
    adjacency = make_symmetric_graph(2, ((0, 1, 1.0),))
    weights = np.ones(2)
    state = ((0, 1), (0,))
    gamma = 0.0
    original_weight = 1.0
    phi_tilde = unnormalized_potential(adjacency, weights, gamma, state, 2)
    phi = original_normalized_quality(adjacency, weights, gamma, state, 2)
    token_weight = token_total_edge_weight(adjacency, state)
    internal_token_weight = 1.0 / math.sqrt(2.0)
    token_normalized = internal_token_weight / token_weight
    return {
        "name": "ce5_token_normalization",
        "gamma": gamma,
        "state": [list(row) for row in state],
        "W": original_weight,
        "W_tok": token_weight,
        "unnormalized_original": phi_tilde,
        "normalized_original": phi,
        "token_internal_weight": internal_token_weight,
        "token_normalized_quality": token_normalized,
        "same_normalized_value": abs(phi - token_normalized) < 1e-12,
    }


def collision_projection() -> dict[str, Any]:
    """Two tokens of one vertex mapped to the same community collapse ``k``."""
    offsets = token_offsets(((0, 1), (0,)))
    rows, collisions = project_token_labels((7, 7, 3), offsets)
    adjacency = make_symmetric_graph(2, ((0, 1, 1.0),))
    weights = np.ones(2)
    before = ((0, 1), (0, 1))
    after = ((0,), (0, 1))
    phi_before = unnormalized_potential(adjacency, weights, 0.0, before, 2)
    phi_after = unnormalized_potential(adjacency, weights, 0.0, after, 2)
    return {
        "name": "collision_projection",
        "projected_rows": rows,
        "collisions": collisions,
        "multiplicity_dropped": True,
        "decreasing_projection_rejected": not original_potential_nondecreasing(
            phi_before, phi_after
        ),
        "phi_before": phi_before,
        "phi_after": phi_after,
    }


def assert_named_examples(examples: dict[str, Any] | None = None) -> None:
    """Fail loudly if a named counterexample no longer holds."""
    payload = examples if examples is not None else named_examples()
    weighted = payload["ce1_weighted"]
    if max(weighted["max_primitive_regret"], 0.0) >= 1e-10:
        raise AssertionError("CE1 weighted is not primitive-stable")
    if weighted["max_full_regret"] <= 0.03:
        raise AssertionError("CE1 weighted lacks a profitable complete replacement")
    if abs(weighted["prefix_utilities"][0] - 1.0) > 1e-12:
        raise AssertionError("CE1 hub singleton utility is not 1")
    if weighted["prefix_utilities"][1] >= weighted["prefix_utilities"][0]:
        raise AssertionError("CE1 prefix utilities are unimodal; they must dip then rise")
    unweighted = payload["ce1_unweighted"]
    if unweighted["max_primitive_regret"] >= 1e-10:
        raise AssertionError("CE1 unweighted is not primitive-stable")
    if unweighted["max_full_regret"] <= 0.005:
        raise AssertionError("CE1 unweighted lacks a profitable complete replacement")
    balanced = payload["ce3_balanced_failure"]
    if not balanced["distinct_bodies"]:
        raise AssertionError("CE3 label bodies must be distinct")
    if max(balanced["masses"]) - min(balanced["masses"]) > 1e-12:
        raise AssertionError("CE3 label masses must be equal")
    if max(item["regret"] for item in balanced["gamma0"]) > 1e-10:
        raise AssertionError("CE3 should be stable at gamma=0")
    if max(item["regret"] for item in balanced["gamma1"]) <= 1e-8:
        raise AssertionError("CE3 should be unstable at gamma=1")
    duplicate = payload["ce4_duplicate_body"]
    if abs(duplicate["potential_after"] - duplicate["potential_before"]) <= 1e-8:
        raise AssertionError("CE4 duplicate bodies must change the potential")
    externality = payload["ce6_externality"]
    if not (externality["U_after"] < externality["U_before"] - 1e-12):
        raise AssertionError("CE6 should reduce v's payoff while keeping A's body")
    isolate = payload["ce7_l1_isolate"]
    if isolate["gain"] <= 0.12:
        raise AssertionError("CE7 unit-l1 isolate should strictly prefer splitting")
    token_norm = payload["ce5_token_normalization"]
    if abs(token_norm["normalized_original"] - 1.0 / math.sqrt(2.0)) > 1e-12:
        raise AssertionError("CE5 original normalized quality must be 1/sqrt(2)")
    if abs(token_norm["token_normalized_quality"] - 0.5) > 1e-12:
        raise AssertionError("CE5 token-normalized quality must be 1/2")
    if token_norm["same_normalized_value"]:
        raise AssertionError("CE5 shows original and token-normalized quality differ")
    collision = payload["collision_projection"]
    if collision["projected_rows"] != [[7], [3]]:
        raise AssertionError("collision projection must collapse duplicate token labels")
    if collision["collisions"] != [{"vertex": 0, "label": 7, "token_count": 2}]:
        raise AssertionError("collision lists must name vertex, label, and token_count")
    if not collision["decreasing_projection_rejected"]:
        raise AssertionError("original-unit guard must reject a decreasing projection")
    balance = payload["restricted_balance_token_identity"]
    if balance["abs_error"] >= 1e-12:
        raise AssertionError("restricted-balance token identity residual too large")
    if not balance["equal_masses"]:
        raise AssertionError("restricted-balance fixture must have equal masses")
    if abs(
        balance["crowding_increment"] - balance["expected_crowding_increment"]
    ) >= 1e-12:
        raise AssertionError("restricted-balance crowding increment is incorrect")
    if balance["support_gain_gamma0"] > 1e-12:
        raise AssertionError("restricted-balance fixture must be gamma-zero stable")
    closed_form = coefficient_value_bound(3, 6, 1)
    enumerated = independent_coefficient_value_bound(3, 6, 1)
    if closed_form != enumerated or closed_form != 28:
        raise AssertionError("independent coefficient enumeration must match C(MW+D,D)")
    overlapping_closed = coefficient_value_bound(3, 6, 2)
    overlapping_enumerated = independent_coefficient_value_bound(3, 6, 2)
    if overlapping_closed != overlapping_enumerated:
        raise AssertionError("M=2 coefficient bound must match slack-variable enumeration")


def run_oracle(
    *,
    random_cases: int = 3000,
    token_cases: int = 500,
) -> dict[str, Any]:
    """Run named examples plus randomized identity checks."""
    examples = named_examples()
    assert_named_examples(examples)
    random_report = random_identity_errors(cases=random_cases)
    token_report = token_coefficient_errors(cases=token_cases)
    max_identity = max(
        random_report["max_delta_error"],
        random_report["max_gram_error"],
        random_report["max_pair_identity_error"],
        random_report["max_prefix_error"],
        random_report["max_disjoint_offset_error"],
        token_report["max_unnormalized_token_error"],
        token_report["max_coefficient_expansion_error"],
    )
    if max_identity >= 1e-10:
        raise AssertionError(f"identity residual {max_identity} exceeds 1e-10")
    proofs = proof_reconstruction()
    if not proofs["ok"]:
        raise AssertionError(f"proof reconstruction failed: {proofs}")
    return {
        "native_tested": False,
        "named_examples": examples,
        "random_oracle": random_report,
        "token_oracle": token_report,
        "proofs": proofs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independent unit-ℓ₂ prefix oracle and named counterexamples "
            "(CE1–CE7, CE5 token normalization, collision lists). "
            "Does not invoke lucas-igraph unless --native-differential."
        )
    )
    parser.add_argument("--random-cases", type=int, default=300)
    parser.add_argument("--token-cases", type=int, default=100)
    parser.add_argument("--list-examples", action="store_true")
    parser.add_argument(
        "--proof-check",
        action="store_true",
        help=(
            "reconstruct the numbered manuscript propositions (prefix/CE1, "
            "endpoint affinity, restricted balance, coefficient bound) and "
            "exit 1 if any check fails"
        ),
    )
    parser.add_argument(
        "--fee-check",
        action="store_true",
        help=(
            "check the separate unit-l2-cpm-fee-v1 identities (R9)/(R10) and "
            "a graph-only K4 ablation; does not reuse manuscript certificates"
        ),
    )
    parser.add_argument(
        "--optimization-check",
        action="store_true",
        help=(
            "prove reference/cached oracle equivalence on exhaustive and random "
            "profiles, then run a small before/after benchmark"
        ),
    )
    parser.add_argument(
        "--native-differential",
        type=Path,
        nargs="?",
        const=Path("artifacts/evidence/overlapping_communities/native_differential.jsonl"),
        default=None,
        help=(
            "optional native vs oracle JSONL; omit the path to use the "
            "evidence default. Invokes lucas-igraph on tiny graphs only."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON path; prints a compact summary when omitted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_examples:
        for name in NAMED_EXAMPLES:
            print(name)
        return 0
    if args.proof_check:
        proofs = proof_reconstruction()
        print(json.dumps(proofs, indent=2))
        return 0 if proofs["ok"] else 1
    if args.fee_check:
        report = fee_v1_check()
        text = json.dumps(report, indent=2)
        print(text)
        output = args.output
        if output is None:
            default = Path(
                "artifacts/evidence/overlapping_communities/unit_l2_cpm_fee_v1.json"
            )
            if default.parent.is_dir():
                output = default
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        return 0 if report["ok"] else 1
    if args.optimization_check:
        report = optimization_check(random_cases=args.random_cases)
        text = json.dumps(report, indent=2)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0 if report["ok"] else 1
    report = run_oracle(random_cases=args.random_cases, token_cases=args.token_cases)
    if args.native_differential is not None:
        from hedonic.experiments.overlapping.native_differential import (
            run_native_differential,
            write_native_differential,
        )

        records = run_native_differential()
        write_native_differential(args.native_differential, records)
        report["native_tested"] = True
        report["native_differential"] = str(args.native_differential)
        report["native_differential_records"] = len(records)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(args.output)
    else:
        summary = {
            "ce1_weighted_full_regret": report["named_examples"]["ce1_weighted"][
                "max_full_regret"
            ],
            "ce1_unweighted_full_regret": report["named_examples"]["ce1_unweighted"][
                "max_full_regret"
            ],
            "random_oracle": report["random_oracle"],
            "token_oracle": report["token_oracle"],
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
