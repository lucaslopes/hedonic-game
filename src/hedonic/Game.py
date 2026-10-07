from __future__ import annotations

# Keep these imports available from the historical ``hedonic.Game`` module.
# They were part of the module namespace before the implementation was split,
# even though only ``Graph`` is needed by the facade itself.
import math
from collections.abc import Sequence
from numbers import Integral

from igraph import Graph

from .utils import sample_uniform_ints
from ._game.detection import community_hedonic as _community_hedonic
from ._game.evaluation import (
    EVALUATION_METHODS as _EVALUATION_METHODS_TABLE,
    evaluate_against as _evaluate_against,
)
from ._game.helpers import (
    _IGRAPH_INTEGER_MAX,
    _UNSET,
    _UnsetType,
    membership_incidence_trace,
    membership_multiplicity_drops,
    seeded_igraph_rng,
    token_graph_preflight,
    total_edge_weight,
)
from ._game.memberships import (
    as_overlapping_init as _as_overlapping_init_impl,
    memberships_as_cover as _memberships_as_cover,
    normalize_membership as _normalize_membership_impl,
    validate_membership as _validate_membership,
)

# Identity of the detector implemented by ``Game.community_hedonic``.  It names
# the native release whose semantics the facade targets; every returned
# object additionally carries the versions that actually ran
# (``_hedonic_provenance``).  1.0.0.5 changes the native trajectory of some
# runs (reconciliation schedule, label index) and propagates interruption.
HEDONIC_ALGORITHM_IDENTITY = "community_hedonic/lucas-igraph-1.0.0.5"

# Producer identities recorded by frozen historical ledgers.  They are kept
# verbatim so that frozen evidence can be verified against the producer that
# wrote it; they do not describe the live wrapper.
HISTORICAL_ALGORITHM_IDENTITIES = {
    "lucas-igraph-1.0.0.3": (
        "community_hedonic/lucas-igraph-1.0.0.3/interrupt-unsupported"
    ),
}


class Game(Graph):
    """A hedonic game represented as an `igraph.Graph`."""

    # ``evaluate_against`` deliberately keeps this dispatch table next to the
    # facade rather than reimplementing any scoring logic.  The implementation
    # remains in ``experiments.overlapping.metrics`` and is imported lazily.
    _EVALUATION_METHODS = _EVALUATION_METHODS_TABLE

    def __init__(self, graph: Graph | None = None, *args, **kwargs) -> None:
        if isinstance(graph, Graph):
            if args or kwargs:
                raise TypeError(
                    "Game(graph) copies an existing graph; pass constructor "
                    "keywords only when the first argument is omitted"
                )
            super().__init__(n=graph.vcount(), directed=graph.is_directed())
            if graph.ecount():
                self.add_edges(graph.get_edgelist())
            for attr in graph.vertex_attributes():
                self.vs[attr] = graph.vs[attr]
            for attr in graph.edge_attributes():
                self.es[attr] = graph.es[attr]
            for attr in graph.attributes():
                self[attr] = graph[attr]
            loaded = getattr(graph, "_memberships", None)
            self._memberships: list[list[int]] | None = (
                None
                if loaded is None
                else [list(row) for row in loaded]
            )
            return
        super().__init__(*args, **kwargs)
        self._memberships: list[list[int]] | None = None

    @property
    def memberships(self) -> list[list[int]] | None:
        """Canonical per-vertex community memberships.

        The state is always represented as one (non-empty) list of integer
        community labels per vertex.  A ``None`` value means that no state has
        been configured yet.  The setter accepts the flat partition form as a
        convenience, but stores the normalized list-of-lists representation.
        Validation against a membership cap is performed by
        :meth:`community_hedonic`, where the cap is known.
        """
        return self._memberships

    @memberships.setter
    def memberships(self, value: list[int] | list[list[int]] | None) -> None:
        if value is None:
            self._memberships = None
            return
        # An empty value is retained as an explicit "unconfigured" state.  It
        # is treated like ``None`` only when the detector's argument is
        # omitted; an explicitly passed empty value still raises ValueError.
        if value == []:
            self._memberships = []
            return
        self._memberships = self._normalize_membership(value)


    def _memberships_as_cover(self) -> list[list[int]]:
        """Convert the stored per-vertex memberships to community lists."""
        return _memberships_as_cover(self)

    def evaluate_against(
        self,
        ground_truth,
        method: str = "f1",
        *,
        singleton_mode: str = "all",
        matching_weight: str = "f1",
        omega_sample_size: int = 100_000,
        omega_seed: int = 0,
    ) -> float:
        """Score the stored partition/cover against an explicit ground truth.

        ``community_hedonic`` persists its return value in ``memberships`` as
        one list of community labels per vertex.  This convenience facade
        converts that state to the community-list representation used by the
        experimental evaluator and returns the selected scalar score.

        Parameters
        ----------
        ground_truth:
            Community lists, or an igraph ``VertexClustering``/``VertexCover``.
            A flat per-vertex partition vector is also accepted.
        method:
            ``f1`` (symmetric best-match), ``one_to_one_f1``, ``jaccard``,
            ``omega``, ``node_micro_f1``, or ``size_weighted_community_f1``.
            A few explicit aliases are accepted for compatibility.
        singleton_mode, matching_weight, omega_sample_size, omega_seed:
            Options forwarded to :func:`hedonic.experiments.overlapping.metrics.evaluate_cover`.

        Returns
        -------
        float
            The selected metric value.

        Notes
        -----
        The metrics module is imported only when this method is called, so
        importing the core ``hedonic.Game`` does not require SciPy.
        """
        return _evaluate_against(
            self,
            ground_truth,
            method,
            singleton_mode=singleton_mode,
            matching_weight=matching_weight,
            omega_sample_size=omega_sample_size,
            omega_seed=omega_seed,
        )

    # A descriptive alias for callers who prefer the word "accuracy".  Keep
    # one implementation so aliases cannot drift in metric dispatch behavior.
    evaluate_accuracy = evaluate_against

    def to_igraph(self) -> Graph:
        """Convert the `Game` object to a standalone `igraph.Graph` instance."""
        g = Graph(n=self.vcount(), directed=self.is_directed())
        if self.ecount():
            g.add_edges(self.get_edgelist())
        for attr in self.vertex_attributes():
            g.vs[attr] = self.vs[attr]
        for attr in self.edge_attributes():
            g.es[attr] = self.es[attr]
        for attr in self.attributes():
            g[attr] = self[attr]
        return g
    def validate_membership(
        self,
        membership: list[int] | list[list[int]],
        *,
        overlapping: bool = False,
        max_memberships: int | None = None,
    ) -> bool:
        """Validate a community membership vector or overlapping cover init.

        Disjoint (``overlapping=False``):
          - length == n (flat labels, or nested singleton rows)
          - non-negative integer labels
          - labels contiguous from 0

        Overlapping (``overlapping=True``):
          - length == n
          - each entry is a non-empty list of non-negative community ids
            (one membership list per vertex), **or** a flat disjoint vector
            (accepted and later expanded to singleton lists). Duplicate labels
            in one row are rejected.
        """
        return _validate_membership(
            self,
            membership,
            overlapping=overlapping,
            max_memberships=max_memberships,
        )

    def _normalize_membership(
        self,
        membership: list[int] | list[list[int]],
        *,
        max_memberships: int | None = None,
        collapse_duplicate_labels: bool = False,
    ) -> list[list[int]]:
        """Validate and normalize a partition or cover to per-vertex rows.

        Labels are required to be non-negative integers and contiguous from
        zero, matching igraph's membership convention.  Every row must be
        non-empty and duplicate labels within a vertex are rejected unless
        ``collapse_duplicate_labels`` is set, as detector initialization does
        for overlapping starts.  The optional cap is checked here so persisted
        state cannot silently be accepted for an incompatible detector call.
        """
        return _normalize_membership_impl(
            self,
            membership,
            max_memberships=max_memberships,
            collapse_duplicate_labels=collapse_duplicate_labels,
        )

    @staticmethod
    def _as_overlapping_init(
        membership: list[int] | list[list[int]],
    ) -> list[list[int]]:
        """Normalize flat or nested membership to list-of-lists (per vertex)."""
        return _as_overlapping_init_impl(membership)

    def community_hedonic(
        self,
        initial_membership: list[int] | list[list[int]] | None | _UnsetType = _UNSET,
        max_communities: int | None = None,
        max_memberships: int = 1,
        n_iterations: int = -1,
        resolution: float | None = None,
        allow_isolation: bool = False,
        local_move_only: bool = True,
        edge_weights=None,
        seed: int | None = None,
        beta: float = 0.01,
        *,
        node_weights=None,
        max_total_communities: int | None = None,
        n_communities: int | None = None,
        debug_trace: bool | str = False,
        phase_policy: str = "direct",
    ):
        """Community detection with the hedonic-game / Leiden local-moving model.

        This is the primary exploratory method for both disjoint and overlapping
        experiments. It delegates to ``community_leiden`` from lucas-igraph.

        Tagged ``v0.1.0`` exposed ``ensure_equilibrium``. That keyword is
        rejected here: a negative ``n_iterations`` (the default ``-1``) is the
        native no-change stop plus the final overlapping local-moving Nash
        sweep. A positive budget may return before that certificate. Do not
        call ordinary ``Graph.community_leiden`` for overlapping work; this
        wrapper is the supported combination.

        Parameters
        ----------
        initial_membership :
            Disjoint: flat list of community ids (length n).
            Overlapping (``max_memberships > 1``): flat list or list-of-lists
            (one community-id list per vertex). ``None`` builds a default init.
            When omitted, a valid ``self.memberships`` state is reused;
            explicit ``None`` requests the default singleton init, or a
            feasible round-robin init when a community-count limit is set.
            Flat and nested overlapping inits with the same labels are
            equivalent after normalization.
        max_communities :
            Initialization only: when ``initial_membership`` is ``None`` and
            ``max_memberships == 1``, random init over ``[0, max_communities)``.
            It has no effect with a supplied start, with overlapping defaults,
            or when community-count limits select a feasible default start;
            ``_hedonic_provenance["max_communities_effect"]`` records which
            case applied. It is *not* an output constraint; see
            ``max_total_communities``/``n_communities``.
        max_memberships :
            Per-vertex cap. ``1`` (default) → disjoint clustering
            (``VertexClustering``); ``> 1`` → overlapping cover
            (``VertexCover``) in which each vertex belongs to at most this
            many communities. ``-1`` removes the practical per-vertex cap: the
            effective cap is the vertex count, the largest cap the native
            layer accepts, inside its finite label bank of ``n * n`` labels.
            The effective value is recorded in ``_hedonic_provenance``.
        n_iterations :
            Leiden outer iterations. A negative value continues until the
            native no-change stopping condition and final Nash certificate
            sweep complete. Positive values are finite budgets and may stop
            before equilibrium.
        resolution :
            CPM resolution γ. Defaults to graph density. Must be a finite
            real. Zero total edge weight is rejected: the published
            normalization ``Φ_γ = Φ̃_γ / W`` is undefined when ``W=0``.
        allow_isolation :
            Allow moves into empty communities (new clusters).
        local_move_only :
            If True (default), run only the local-moving phase — the hedonic
            best-response phase, with no token graph. If False, run full
            Leiden (refine + aggregate) on the frozen-multiplicity token
            graph, then a quality guard in original-graph units for every
            projected iteration, including positive iteration budgets.
        edge_weights :
            Optional edge weights (sequence or edge-attribute name); finite
            and non-negative in the overlapping domain.
        seed :
            When given, the whole call runs with ``random.Random(seed)`` as
            igraph's random generator (restored afterwards), and the same
            seed drives the random disjoint initialization. On a given build
            the returned labels are then a function of the graph, weights,
            parameters, start state and seed. ``None`` keeps the caller's
            igraph generator (historical behavior).
        beta :
            Leiden refinement randomness (used when ``local_move_only`` is False).
        node_weights :
            Optional vertex weights ``w_v`` (sequence or vertex-attribute
            name), finite and non-negative. They enter the crowding term
            ``γ w_u w_v``; unit weights are the default.
        max_total_communities :
            Optional upper bound on the number of occupied communities of the
            result and of every state visited by local moving (all aggregate
            and token levels included).
        n_communities :
            Optional exact number of occupied communities. Without a start
            state, a deterministic round-robin cover occupies all ``K`` labels
            (one label per vertex when ``K <= n``, with extra memberships when
            ``K > n``). A start state that violates either count constraint is
            rejected.
        debug_trace :
            Native diagnostics in ``result._params["debug_trace"]``, for
            partitions and covers, with or without count limits. ``True`` or
            ``"full"`` records every accepted local move with a direct
            recomputation of its potential change (expensive; for bounded
            fixtures), every overlapping multilevel proposal, and the
            counters; ``"counters"`` records only the counters (visits split
            into accepted moves and rejected visits, iterations, certificate
            sweeps, guard decisions). The envelope states the schema version,
            runtime versions, count policy, what is recorded and what is
            omitted, and ``seed_context`` (this call's ``seed``). Rejected
            candidates are never recorded individually.
        phase_policy :
            How an overlapping cover (``max_memberships > 1``) is reached.
            ``"direct"`` (default) runs overlapping detection from its own
            start. ``"disjoint_then_overlap"`` is an opt-in warm start: it
            first runs the disjoint game (``max_memberships=1``) with the same
            resolution, weights, phase, isolation, iteration budget, seed and
            count limits, then passes that partition as
            ``initial_membership`` to the overlapping run. The first stage
            imposes a partition structure that the overlap can only extend;
            on complete DBLP it was faster but less accurate (symmetric F1
            0.380 versus 0.425 direct), so keep it for ablations, warm starts
            and very large graphs rather than as a default. It requires
            ``max_memberships > 1`` and rejects an explicit
            ``initial_membership``. The disjoint result is attached as
            ``_hedonic_disjoint_stage`` and summarized in
            ``_hedonic_provenance["phase_policy"]`` /
            ``["initialization"]``.

        Returns
        -------
        VertexClustering if ``max_memberships == 1``, else VertexCover.

        The return object carries ``_hedonic_provenance`` (algorithm identity,
        runtime versions, effective parameters, count mode, and the stopping
        rule), and may carry ``_hedonic_raw_memberships`` (negative
        iterations), ``_hedonic_token_preflight`` (always),
        ``_hedonic_original_edge_weight`` (always), overlapping start/returned
        incidence traces with collision lists, ``_hedonic_multiplicity_drops``,
        and ``_hedonic_native_quality`` when the binding stores original-graph
        ``Φ``.

        Interruption (for example ``KeyboardInterrupt``) is propagated by the
        native layer through igraph's error path, which releases its
        temporaries; the call then raises and returns no partial state.
        """
        return _community_hedonic(
            self,
            initial_membership,
            max_communities,
            max_memberships,
            n_iterations,
            resolution,
            allow_isolation,
            local_move_only,
            edge_weights,
            seed,
            beta,
            node_weights=node_weights,
            max_total_communities=max_total_communities,
            n_communities=n_communities,
            debug_trace=debug_trace,
            phase_policy=phase_policy,
            algorithm_identity=HEDONIC_ALGORITHM_IDENTITY,
        )
