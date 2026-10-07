"""Executable prototype for the ``public-allocation-v0.1`` design.

This module intentionally lives in a new namespace beside (and not inside)
the overlapping-community implementation.  It implements the smallest
mechanism described in :mod:`docs.plans.public_allocation_extension`:

* one public vector of divisible project outcomes;
* weighted project costs, a public budget, and project capacities;
* virtual-share and voluntary-payment contribution semantics;
* deterministic project totals plus compressed pairwise synergy; and
* deterministic Euclidean projection onto the public feasible set.

The implementation is a mechanism prototype, not a theorem.  In particular,
it does not claim fairness, truthfulness, incentive compatibility, equilibrium
existence, or preservation of the original pairwise participation game.  The
two data-free counterexample fixtures at the bottom of the module are kept as
regression checks for the information-loss statements in the design note.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence

import numpy as np


PROTOCOL_VERSION = "public-allocation-v0.1"
SCHEMA_VERSION = 1
MODE_VIRTUAL_SHARE = "virtual-share"
MODE_VOLUNTARY_PAYMENT = "voluntary-payment"
DEFAULT_TOLERANCE = 1e-10


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Copy a numeric input to a finite floating-point array."""

    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _immutable(array: np.ndarray) -> np.ndarray:
    """Return an array that cannot be mutated through the public object."""

    array = np.asarray(array, dtype=float).copy()
    array.setflags(write=False)
    return array


def _resolve_alias(
    primary: Any,
    alias: Any,
    *,
    primary_name: str,
    alias_name: str,
    required: bool = True,
) -> Any:
    """Resolve a documented short name and a descriptive constructor alias."""

    if primary is not None and alias is not None:
        # Arrays need numerical comparison; scalar comparison is handled by
        # the same helper without assuming either input has a ``shape``.
        try:
            equal = bool(np.array_equal(primary, alias))
        except Exception:  # pragma: no cover - defensive for unusual objects
            equal = primary == alias
        if not equal:
            raise ValueError(f"pass only one of {primary_name} and {alias_name}")
    result = primary if primary is not None else alias
    if required and result is None:
        raise TypeError(f"missing required argument: {primary_name}")
    return result


def _canonical_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    aliases = {
        "virtual": MODE_VIRTUAL_SHARE,
        "share": MODE_VIRTUAL_SHARE,
        MODE_VIRTUAL_SHARE: MODE_VIRTUAL_SHARE,
        "voluntary": MODE_VOLUNTARY_PAYMENT,
        "payment": MODE_VOLUNTARY_PAYMENT,
        MODE_VOLUNTARY_PAYMENT: MODE_VOLUNTARY_PAYMENT,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "mode must be 'virtual-share' or 'voluntary-payment'"
        ) from exc


@dataclass(frozen=True, slots=True, init=False)
class PublicAllocationProblem:
    r"""A validated ``public-allocation-v0.1`` instance.

    The short constructor names follow the equations in the design document:
    ``A, p, B, ybar, b``.  Descriptive aliases are accepted as keyword
    arguments as well (``adjacency``, ``project_costs``, ``budget``,
    ``capacities``, and ``contribution_budgets``), which makes call sites less
    dependent on mathematical notation.

    ``theta[v, c]`` defines the linear preference
    :math:`h_v(y)=\theta_v^\top y`.  Omitting it means all preferences are
    zero, which is useful for aggregation-only checks.  In payment mode,
    ``payment_rates[v]`` declares the linear cost
    :math:`\tau_v(z_v)=payment_rates_v\,p^\top z_v`; virtual-share mode has
    zero direct contribution cost by definition.
    """

    A: np.ndarray
    p: np.ndarray
    B: float
    ybar: np.ndarray
    b: np.ndarray
    eta: float
    mode: str
    theta: np.ndarray
    payment_rates: np.ndarray
    tolerance: float

    def __init__(
        self,
        A: Any = None,
        p: Any = None,
        B: float | None = None,
        ybar: Any = None,
        b: Any = None,
        *,
        adjacency: Any = None,
        project_costs: Any = None,
        budget: float | None = None,
        capacities: Any = None,
        contribution_budgets: Any = None,
        eta: float = 0.0,
        mode: str = MODE_VIRTUAL_SHARE,
        theta: Any = None,
        payment_rates: Any = None,
        payment_costs: Any = None,
        tau: Any = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        A = _resolve_alias(A, adjacency, primary_name="A", alias_name="adjacency")
        p = _resolve_alias(
            p, project_costs, primary_name="p", alias_name="project_costs"
        )
        B = _resolve_alias(B, budget, primary_name="B", alias_name="budget")
        ybar = _resolve_alias(
            ybar, capacities, primary_name="ybar", alias_name="capacities"
        )
        b = _resolve_alias(
            b,
            contribution_budgets,
            primary_name="b",
            alias_name="contribution_budgets",
        )
        payment_rates = _resolve_alias(
            payment_rates,
            payment_costs,
            primary_name="payment_rates",
            alias_name="payment_costs",
            required=False,
        )
        payment_rates = _resolve_alias(
            payment_rates,
            tau,
            primary_name="payment_rates",
            alias_name="tau",
            required=False,
        )

        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be a finite nonnegative number")
        A_array = _array(A, name="A", ndim=2)
        p_array = _array(p, name="p", ndim=1)
        ybar_array = _array(ybar, name="ybar", ndim=1)
        b_array = _array(b, name="b", ndim=1)
        if A_array.shape[0] != A_array.shape[1]:
            raise ValueError("A must be square")
        n_agents = A_array.shape[0]
        n_projects = p_array.shape[0]
        if n_agents < 1:
            raise ValueError("A must contain at least one agent")
        if n_projects < 1:
            raise ValueError("p must contain at least one project")
        if ybar_array.shape != (n_projects,):
            raise ValueError("ybar must have one entry per project")
        if b_array.shape != (n_agents,):
            raise ValueError("b must have one entry per agent")
        if not np.allclose(A_array, A_array.T, rtol=0.0, atol=tolerance):
            raise ValueError("A must be symmetric (the network is undirected)")
        if not np.allclose(np.diag(A_array), 0.0, rtol=0.0, atol=tolerance):
            raise ValueError("A must be loopless (zero diagonal)")
        if np.any(A_array < -tolerance):
            raise ValueError("A must have nonnegative edge weights")
        if np.any(p_array <= 0.0):
            raise ValueError("p must have strictly positive project costs")
        if np.any(ybar_array < 0.0):
            raise ValueError("ybar must be nonnegative")
        if np.any(b_array <= 0.0):
            raise ValueError("b must have strictly positive contribution budgets")
        B_value = float(B)
        if not np.isfinite(B_value) or B_value <= 0.0:
            raise ValueError("B must be a finite positive public budget")
        eta_value = float(eta)
        if not np.isfinite(eta_value) or eta_value < 0.0:
            raise ValueError("eta must be a finite nonnegative number")
        mode_value = _canonical_mode(mode)

        if theta is None:
            theta_array = np.zeros((n_agents, n_projects), dtype=float)
        else:
            theta_array = _array(theta, name="theta", ndim=2)
            if theta_array.shape != (n_agents, n_projects):
                raise ValueError("theta must have shape (n_agents, n_projects)")

        if payment_rates is None:
            payment_array = np.zeros(n_agents, dtype=float)
        else:
            payment_array = _array(payment_rates, name="payment_rates", ndim=1)
            if payment_array.shape != (n_agents,):
                raise ValueError("payment_rates must have one entry per agent")
        if np.any(payment_array < 0.0):
            raise ValueError("payment_rates must be nonnegative")
        if mode_value == MODE_VIRTUAL_SHARE and np.any(payment_array > tolerance):
            raise ValueError(
                "virtual-share mode has zero direct cost; use voluntary-payment "
                "mode for payment rates"
            )

        object.__setattr__(self, "A", _immutable(A_array))
        object.__setattr__(self, "p", _immutable(p_array))
        object.__setattr__(self, "B", B_value)
        object.__setattr__(self, "ybar", _immutable(ybar_array))
        object.__setattr__(self, "b", _immutable(b_array))
        object.__setattr__(self, "eta", eta_value)
        object.__setattr__(self, "mode", mode_value)
        object.__setattr__(self, "theta", _immutable(theta_array))
        object.__setattr__(self, "payment_rates", _immutable(payment_array))
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def adjacency(self) -> np.ndarray:
        return self.A

    @property
    def project_costs(self) -> np.ndarray:
        return self.p

    @property
    def budget(self) -> float:
        return self.B

    @property
    def capacities(self) -> np.ndarray:
        return self.ybar

    @property
    def contribution_budgets(self) -> np.ndarray:
        return self.b

    @property
    def n_agents(self) -> int:
        return int(self.A.shape[0])

    @property
    def n_projects(self) -> int:
        return int(self.p.shape[0])

    @property
    def cost_model(self) -> str:
        return (
            "zero_direct_cost_virtual_share"
            if self.mode == MODE_VIRTUAL_SHARE
            else "linear_per_budget_unit"
        )

    def validate_contributions(self, z: Any) -> np.ndarray:
        """Validate and copy a contribution profile.

        Virtual-share rows must spend exactly their personal budget.  Payment
        rows may leave budget unspent.  The returned array is clipped only for
        tiny negative round-off (within ``tolerance``); substantive violations
        raise ``ValueError`` rather than being silently projected.
        """

        result = _array(z, name="z", ndim=2)
        if result.shape != (self.n_agents, self.n_projects):
            raise ValueError(
                "z must have shape (n_agents, n_projects)"
            )
        if np.any(result < -self.tolerance):
            raise ValueError("z must be nonnegative")
        result[result < 0.0] = 0.0
        spent = result @ self.p
        scale = np.maximum(1.0, self.b)
        if self.mode == MODE_VIRTUAL_SHARE:
            if np.any(np.abs(spent - self.b) > self.tolerance * scale):
                raise ValueError(
                    "virtual-share contributions must satisfy p @ z_v == b_v"
                )
        elif np.any(spent - self.b > self.tolerance * scale):
            raise ValueError("voluntary-payment contributions exceed b_v")
        return result

    def contribution_spend(self, z: Any) -> np.ndarray:
        """Return each agent's weighted contribution amount ``p @ z_v``."""

        return self.validate_contributions(z) @ self.p

    def contribution_cost(self, z: Any) -> np.ndarray:
        """Return direct contribution costs under the declared institution."""

        spend = self.contribution_spend(z)
        if self.mode == MODE_VIRTUAL_SHARE:
            return np.zeros(self.n_agents, dtype=float)
        return self.payment_rates * spend

    def project(self, raw_score: Any) -> "ProjectionResult":
        """Project one raw project-score vector onto ``mathcal Y``."""

        return project_with_diagnostics(
            raw_score,
            self.p,
            self.B,
            self.ybar,
            tolerance=self.tolerance,
        )

    def is_feasible_outcome(self, outcome: Any) -> bool:
        """Return whether ``outcome`` satisfies the public constraints."""

        return is_feasible_outcome(
            outcome,
            self.p,
            self.B,
            self.ybar,
            tolerance=self.tolerance,
        )

    def validate_outcome(self, outcome: Any) -> np.ndarray:
        """Return a copied feasible outcome or raise ``ValueError``."""

        candidate = _array(outcome, name="outcome", ndim=1)
        if candidate.shape != (self.n_projects,):
            raise ValueError("outcome must have one entry per project")
        if not self.is_feasible_outcome(candidate):
            raise ValueError("outcome is outside the public feasible set")
        return candidate

    def aggregate(self, z: Any) -> "PublicAllocationResult":
        """Aggregate contributions into one feasible public outcome.

        The public statistic is ``(x, s)`` compressed to
        ``r = x + eta * s``.  Endpoint identities are not included in the
        result; callers that need those identities should use the separate
        diagnostics in the design document rather than treating them as part
        of this public outcome.
        """

        contributions = self.validate_contributions(z)
        totals = contributions.sum(axis=0)
        normalized = contributions / self.b[:, None]
        synergy = weighted_pair_synergy(self.A, normalized)
        raw = totals + self.eta * synergy
        projection = self.project(raw)
        preference_values = self.theta @ projection.y
        costs = self.contribution_cost(contributions)
        utilities = preference_values - costs
        return PublicAllocationResult(
            protocol_version=PROTOCOL_VERSION,
            schema_version=SCHEMA_VERSION,
            mode=self.mode,
            eta=self.eta,
            contributions=_immutable(contributions),
            project_totals=_immutable(totals),
            synergy=_immutable(synergy),
            raw_score=_immutable(raw),
            outcome=_immutable(projection.y),
            projection_residual=float(projection.residual),
            projection_multiplier=float(projection.multiplier),
            budget_slack=float(projection.budget_slack),
            capacity_slack=_immutable(projection.capacity_slack),
            preference_values=_immutable(preference_values),
            contribution_costs=_immutable(costs),
            utilities=_immutable(utilities),
            welfare=float(np.sum(preference_values)),
        )


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Projection output and diagnostics for the public feasible set."""

    y: np.ndarray
    residual: float
    multiplier: float
    budget_slack: float
    capacity_slack: np.ndarray
    budget_active: bool

    @property
    def feasible(self) -> bool:
        return bool(
            np.all(self.y >= -1e-9)
            and np.all(self.capacity_slack >= -1e-9)
            and self.budget_slack >= -1e-9
        )

    @property
    def outcome(self) -> np.ndarray:
        return self.y


@dataclass(frozen=True, slots=True)
class PublicAllocationResult:
    """One deterministic public-allocation evaluation."""

    protocol_version: str
    schema_version: int
    mode: str
    eta: float
    contributions: np.ndarray
    project_totals: np.ndarray
    synergy: np.ndarray
    raw_score: np.ndarray
    outcome: np.ndarray
    projection_residual: float
    projection_multiplier: float
    budget_slack: float
    capacity_slack: np.ndarray
    preference_values: np.ndarray
    contribution_costs: np.ndarray
    utilities: np.ndarray
    welfare: float

    @property
    def y(self) -> np.ndarray:
        return self.outcome

    @property
    def public_outcome(self) -> np.ndarray:
        return self.outcome

    @property
    def x(self) -> np.ndarray:
        return self.project_totals

    @property
    def raw(self) -> np.ndarray:
        return self.raw_score

    @property
    def pair_synergy(self) -> np.ndarray:
        return self.synergy

    @property
    def residual(self) -> float:
        return self.projection_residual

    @property
    def feasible(self) -> bool:
        return bool(
            np.all(self.outcome >= -1e-9)
            and np.all(self.capacity_slack >= -1e-9)
            and self.budget_slack >= -1e-9
        )

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "projection_residual_l2": float(self.projection_residual),
            "projection_multiplier": float(self.projection_multiplier),
            "budget_slack": float(self.budget_slack),
            "capacity_slack": self.capacity_slack.tolist(),
            "budget_active": bool(abs(self.budget_slack) <= 1e-8),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record without hiding diagnostics."""

        return {
            "protocol_version": self.protocol_version,
            "schema_version": int(self.schema_version),
            "mode": self.mode,
            "eta": float(self.eta),
            "contributions": self.contributions.tolist(),
            "project_totals": self.project_totals.tolist(),
            "synergy": self.synergy.tolist(),
            "raw_score": self.raw_score.tolist(),
            "outcome": self.outcome.tolist(),
            "projection_residual": float(self.projection_residual),
            "projection_multiplier": float(self.projection_multiplier),
            "budget_slack": float(self.budget_slack),
            "capacity_slack": self.capacity_slack.tolist(),
            "preference_values": self.preference_values.tolist(),
            "contribution_costs": self.contribution_costs.tolist(),
            "utilities": self.utilities.tolist(),
            "welfare": float(self.welfare),
            "feasible": bool(self.feasible),
        }


def is_feasible_outcome(
    outcome: Any,
    project_costs: Any,
    budget: float,
    capacities: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Check ``0 <= y <= capacities`` and ``project_costs @ y <= budget``."""

    try:
        y = _array(outcome, name="outcome", ndim=1)
        _, p, budget_value, caps, tolerance_value = _validate_projection_inputs(
            y,
            project_costs,
            budget,
            capacities,
            tolerance=tolerance,
        )
    except (TypeError, ValueError):
        return False
    return bool(
        np.all(y >= -tolerance_value)
        and np.all(y <= caps + tolerance_value)
        and float(np.dot(p, y)) <= budget_value + tolerance_value
    )


def _validate_projection_inputs(
    raw_score: Any,
    project_costs: Any,
    budget: float,
    capacities: Any,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    raw = _array(raw_score, name="raw_score", ndim=1)
    p = _array(project_costs, name="project_costs", ndim=1)
    caps = _array(capacities, name="capacities", ndim=1)
    if raw.shape != p.shape or raw.shape != caps.shape:
        raise ValueError("raw_score, project_costs, and capacities must have the same shape")
    if np.any(p <= 0.0):
        raise ValueError("project_costs must be strictly positive")
    if np.any(caps < 0.0):
        raise ValueError("capacities must be nonnegative")
    budget_value = float(budget)
    if not np.isfinite(budget_value) or budget_value <= 0.0:
        raise ValueError("budget must be finite and positive")
    tolerance_value = float(tolerance)
    if not np.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    return raw, p, budget_value, caps, tolerance_value


def project_with_diagnostics(
    raw_score: Any,
    project_costs: Any,
    budget: float,
    capacities: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> ProjectionResult:
    """Project ``raw_score`` onto ``0 <= y <= capacities, p @ y <= budget``.

    The KKT form is ``y = clip(raw_score - lambda * p, 0, capacities)``.  A
    deterministic bisection over the single multiplier is sufficient because
    the budget constraint is one-dimensional.  No SciPy or graph package is
    required, and a fixed iteration count makes repeated calls bit-stable up
    to the platform's elementary floating-point operations.
    """

    raw, p, budget_value, caps, tolerance_value = _validate_projection_inputs(
        raw_score,
        project_costs,
        budget,
        capacities,
        tolerance=tolerance,
    )
    clipped = np.clip(raw, 0.0, caps)
    clipped_cost = float(np.dot(p, clipped))
    if clipped_cost <= budget_value + tolerance_value:
        # If the comparison accepted a round-off-sized excess, trim it in a
        # deterministic coordinate order so the returned point is genuinely
        # feasible rather than merely feasible under the tolerance.
        y = clipped.copy()
        if clipped_cost > budget_value:
            excess = clipped_cost - budget_value
            for index in range(y.shape[0]):
                removable = min(y[index], excess / p[index])
                y[index] -= removable
                excess -= removable * p[index]
                if excess <= 1e-15 * max(1.0, budget_value):
                    break
        multiplier = 0.0
        active = False
    else:
        positive_ratios = raw[raw > 0.0] / p[raw > 0.0]
        # At this upper endpoint every positive raw coordinate is at most zero
        # after the multiplier is applied, so the budget function is zero.
        upper = max(1.0, float(np.max(positive_ratios, initial=0.0)))
        lower = 0.0
        for _ in range(120):
            midpoint = (lower + upper) / 2.0
            candidate = np.clip(raw - midpoint * p, 0.0, caps)
            if float(np.dot(p, candidate)) > budget_value:
                lower = midpoint
            else:
                upper = midpoint
        multiplier = (lower + upper) / 2.0
        y = np.clip(raw - multiplier * p, 0.0, caps)
        # Bisection can leave a sub-ulp excess at a kink.  Trim that excess in
        # ascending project order while leaving the KKT solution unchanged at
        # practical precision.
        excess = float(np.dot(p, y) - budget_value)
        if excess > 0.0:
            for index in range(y.shape[0]):
                removable = min(y[index], excess / p[index])
                y[index] -= removable
                excess -= removable * p[index]
                if excess <= 1e-15 * max(1.0, budget_value):
                    break
        active = True
    budget_slack = float(budget_value - np.dot(p, y))
    if budget_slack < 0.0 and abs(budget_slack) <= 1e-9 * max(1.0, budget_value):
        budget_slack = 0.0
    capacity_slack = caps - y
    # A final guard catches implementation regressions rather than silently
    # returning an infeasible public outcome.
    if (
        np.any(y < -1e-9)
        or np.any(y - caps > 1e-9)
        or budget_slack < -1e-9 * max(1.0, budget_value)
    ):
        raise RuntimeError("deterministic projection returned an infeasible point")
    y = np.maximum(y, 0.0)
    return ProjectionResult(
        y=_immutable(y),
        residual=float(np.linalg.norm(y - raw)),
        multiplier=float(multiplier),
        budget_slack=float(budget_slack),
        capacity_slack=_immutable(capacity_slack),
        budget_active=bool(active),
    )


def project_to_feasible(
    raw_score: Any,
    project_costs: Any,
    budget: float,
    capacities: Any,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """Return only the deterministic Euclidean projection onto ``mathcal Y``."""

    return project_with_diagnostics(
        raw_score,
        project_costs,
        budget,
        capacities,
        tolerance=tolerance,
    ).y


# A concise alias useful in notebooks and hidden smoke tests.
project_feasible = project_to_feasible


def weighted_pair_synergy(adjacency: Any, normalized_shares: Any) -> np.ndarray:
    """Compute ``s_c = sum_{u<v} a_uv sqrt(q_uc q_vc)`` deterministically."""

    A = _array(adjacency, name="adjacency", ndim=2)
    q = _array(normalized_shares, name="normalized_shares", ndim=2)
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    if q.shape[0] != A.shape[0]:
        raise ValueError("normalized_shares must have one row per agent")
    if np.any(q < -DEFAULT_TOLERANCE):
        raise ValueError("normalized_shares must be nonnegative")
    q[q < 0.0] = 0.0
    if not np.allclose(A, A.T, rtol=0.0, atol=DEFAULT_TOLERANCE):
        raise ValueError("adjacency must be symmetric")
    if not np.allclose(np.diag(A), 0.0, rtol=0.0, atol=DEFAULT_TOLERANCE):
        raise ValueError("adjacency must be loopless")
    if np.any(A < -DEFAULT_TOLERANCE):
        raise ValueError("adjacency must be nonnegative")
    synergy = np.zeros(q.shape[1], dtype=float)
    for first in range(A.shape[0]):
        for second in range(first + 1, A.shape[0]):
            weight = float(A[first, second])
            if weight:
                synergy += weight * np.sqrt(q[first] * q[second])
    return synergy


def aggregate_public_outcome(
    problem: PublicAllocationProblem, z: Any
) -> PublicAllocationResult:
    """Functional wrapper around :meth:`PublicAllocationProblem.aggregate`."""

    if not isinstance(problem, PublicAllocationProblem):
        raise TypeError("problem must be a PublicAllocationProblem")
    return problem.aggregate(z)


def linear_preferences(theta: Any, outcome: Any) -> np.ndarray:
    """Evaluate one linear preference vector per agent at a public outcome."""

    values = _array(theta, name="theta", ndim=2)
    y = _array(outcome, name="outcome", ndim=1)
    if values.shape[1] != y.shape[0]:
        raise ValueError("theta and outcome have incompatible project dimensions")
    return values @ y


def binary_network_payoffs(adjacency: Any, state: Sequence[int]) -> np.ndarray:
    """Return the pairwise same-project payoff used by the two fixtures."""

    A = _array(adjacency, name="adjacency", ndim=2)
    labels = np.asarray(state, dtype=int)
    if labels.ndim != 1 or labels.shape[0] != A.shape[0]:
        raise ValueError("state must have one binary label per agent")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("state must contain only 0 (A) and 1 (B)")
    return np.asarray(
        [sum(float(A[v, u]) for u in range(A.shape[0]) if labels[u] == labels[v])
         for v in range(A.shape[0])],
        dtype=float,
    )


def _binary_totals(state: Sequence[int], n_projects: int = 2) -> np.ndarray:
    labels = np.asarray(state, dtype=int)
    if n_projects != 2 or np.any((labels != 0) & (labels != 1)):
        raise ValueError("the locked fixtures use two binary projects")
    return np.asarray([(labels == project).sum() for project in range(2)], dtype=float)


@dataclass(frozen=True, slots=True)
class CounterexampleFixture:
    """A data-free witness from Section 4 of the public-allocation design."""

    name: str
    proposition: str
    adjacency: np.ndarray
    state: tuple[int, ...]
    alternate_state: tuple[int, ...]
    expected_totals: tuple[float, float]
    expected_alternate_totals: tuple[float, float]
    expected_synergy: tuple[float, float] | None
    expected_alternate_synergy: tuple[float, float] | None
    expected_payoffs: tuple[float, ...]
    expected_alternate_payoffs: tuple[float, ...]

    @property
    def z(self) -> np.ndarray:
        return np.eye(2, dtype=float)[np.asarray(self.state, dtype=int)]

    @property
    def z_prime(self) -> np.ndarray:
        return np.eye(2, dtype=float)[np.asarray(self.alternate_state, dtype=int)]

    @property
    def public_totals_equal(self) -> bool:
        return self.expected_totals == self.expected_alternate_totals

    @property
    def public_statistics_equal(self) -> bool:
        return self.public_totals_equal and (
            self.expected_synergy == self.expected_alternate_synergy
        )


def additive_totals_counterexample() -> CounterexampleFixture:
    """Proposition 1: equal additive totals hide pairwise payoff changes."""

    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = A[1, 0] = 1.0
    A[2, 3] = A[3, 2] = 1.0
    return CounterexampleFixture(
        name="additive_totals",
        proposition="proposition-1",
        adjacency=_immutable(A),
        state=(0, 0, 1, 1),
        alternate_state=(0, 1, 0, 1),
        expected_totals=(2.0, 2.0),
        expected_alternate_totals=(2.0, 2.0),
        expected_synergy=None,
        expected_alternate_synergy=None,
        expected_payoffs=(1.0, 1.0, 1.0, 1.0),
        expected_alternate_payoffs=(0.0, 0.0, 0.0, 0.0),
    )


def project_synergy_counterexample() -> CounterexampleFixture:
    """Proposition 2: project synergy hides which endpoint receives a gain."""

    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = A[1, 0] = 1.0
    A[0, 2] = A[2, 0] = 1.0
    return CounterexampleFixture(
        name="project_synergy",
        proposition="proposition-2",
        adjacency=_immutable(A),
        state=(1, 0, 1, 1),  # A support {1}
        alternate_state=(1, 1, 0, 1),  # A support {2}
        expected_totals=(1.0, 3.0),
        expected_alternate_totals=(1.0, 3.0),
        expected_synergy=(0.0, 1.0),
        expected_alternate_synergy=(0.0, 1.0),
        expected_payoffs=(1.0, 0.0, 1.0, 0.0),
        expected_alternate_payoffs=(1.0, 1.0, 0.0, 0.0),
    )


# Descriptive aliases keep the names used in the design note discoverable.
counterexample_1 = additive_totals_counterexample
counterexample_2 = project_synergy_counterexample
additive_counterexample = additive_totals_counterexample
synergy_counterexample = project_synergy_counterexample


def counterexample_fixtures() -> dict[str, CounterexampleFixture]:
    """Return fresh copies of the two locked information-loss fixtures."""

    first = additive_totals_counterexample()
    second = project_synergy_counterexample()
    return {
        "proposition_1": first,
        "proposition_2": second,
        "additive_totals": first,
        "project_synergy": second,
    }


def check_counterexamples() -> dict[str, Any]:
    """Run exhaustive binary checks for both documented propositions."""

    first = additive_totals_counterexample()
    second = project_synergy_counterexample()
    profiles = list(product((0, 1), repeat=4))
    total_to_payoffs: dict[tuple[float, float], set[tuple[float, ...]]] = {}
    for state in profiles:
        total_key = tuple(_binary_totals(state))
        payoff_key = tuple(binary_network_payoffs(first.adjacency, state))
        total_to_payoffs.setdefault(total_key, set()).add(payoff_key)
    additive_witness = (
        np.array_equal(_binary_totals(first.state), _binary_totals(first.alternate_state))
        and np.allclose(
            binary_network_payoffs(first.adjacency, first.state),
            first.expected_payoffs,
        )
        and np.allclose(
            binary_network_payoffs(first.adjacency, first.alternate_state),
            first.expected_alternate_payoffs,
        )
    )

    second_totals = _binary_totals(second.state)
    second_alt_totals = _binary_totals(second.alternate_state)
    # ``s`` is defined on binary one-hot shares, so the same helper used by
    # the mechanism computes the fixture's project-level synergy exactly.
    second_synergy = weighted_pair_synergy(second.adjacency, second.z)
    second_alt_synergy = weighted_pair_synergy(second.adjacency, second.z_prime)
    synergy_witness = (
        np.array_equal(second_totals, second_alt_totals)
        and np.allclose(second_synergy, second.expected_synergy)
        and np.allclose(second_alt_synergy, second.expected_alternate_synergy)
        and np.allclose(
            binary_network_payoffs(second.adjacency, second.state),
            second.expected_payoffs,
        )
        and np.allclose(
            binary_network_payoffs(second.adjacency, second.alternate_state),
            second.expected_alternate_payoffs,
        )
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "profiles_checked": len(profiles),
        "additive_totals_have_payoff_collision": any(
            len(payoffs) > 1 for payoffs in total_to_payoffs.values()
        ),
        "additive_witness": bool(additive_witness),
        "synergy_witness": bool(synergy_witness),
        "ok": bool(
            additive_witness
            and synergy_witness
            and any(len(payoffs) > 1 for payoffs in total_to_payoffs.values())
        ),
        "fixtures": {
            "proposition_1": {
                "totals": list(first.expected_totals),
                "alternate_totals": list(first.expected_alternate_totals),
                "payoffs": list(first.expected_payoffs),
                "alternate_payoffs": list(first.expected_alternate_payoffs),
            },
            "proposition_2": {
                "totals": list(second.expected_totals),
                "alternate_totals": list(second.expected_alternate_totals),
                "synergy": list(second.expected_synergy or ()),
                "alternate_synergy": list(second.expected_alternate_synergy or ()),
                "payoffs": list(second.expected_payoffs),
                "alternate_payoffs": list(second.expected_alternate_payoffs),
            },
        },
    }


def validate_counterexamples() -> dict[str, Any]:
    """Return the report or raise if a locked witness no longer holds."""

    report = check_counterexamples()
    if not report["ok"]:
        raise AssertionError("public-allocation counterexample regression failed")
    return report


# Names used by small scripts and tests in the surrounding experiment
# namespace.  They intentionally do not register a CLI command or write an
# artifact: this first prototype is data-free and import-safe.
run_counterexample_checks = check_counterexamples
counterexample_report = check_counterexamples


__all__ = [
    "PROTOCOL_VERSION",
    "SCHEMA_VERSION",
    "MODE_VIRTUAL_SHARE",
    "MODE_VOLUNTARY_PAYMENT",
    "DEFAULT_TOLERANCE",
    "PublicAllocationProblem",
    "PublicAllocationResult",
    "ProjectionResult",
    "is_feasible_outcome",
    "aggregate_public_outcome",
    "project_to_feasible",
    "project_feasible",
    "project_with_diagnostics",
    "weighted_pair_synergy",
    "linear_preferences",
    "binary_network_payoffs",
    "CounterexampleFixture",
    "additive_totals_counterexample",
    "project_synergy_counterexample",
    "counterexample_1",
    "counterexample_2",
    "additive_counterexample",
    "synergy_counterexample",
    "counterexample_fixtures",
    "check_counterexamples",
    "run_counterexample_checks",
    "counterexample_report",
    "validate_counterexamples",
]
