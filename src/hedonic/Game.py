from __future__ import annotations

from igraph import Graph

from ._game.detection import community_hedonic as _community_hedonic
from ._game.evaluation import (
    EVALUATION_METHODS as _EVALUATION_METHODS_TABLE,
    evaluate_against as _evaluate_against,
)
from ._game.helpers import _UNSET, _UnsetType
from ._game.memberships import (
    as_overlapping_init as _as_overlapping_init_impl,
    memberships_as_cover as _memberships_as_cover,
    normalize_membership as _normalize_membership_impl,
    validate_membership as _validate_membership,
)


class Game(Graph):
    """A hedonic game represented as an `igraph.Graph`."""

    # ``evaluate_against`` keeps this dispatch table next to the facade; the
    # scoring lives in ``hedonic.metrics`` and is imported lazily.
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
            Options forwarded to :func:`hedonic.metrics.evaluate_cover`.

        Returns
        -------
        float
            The selected metric value.

        Notes
        -----
        The metrics module is imported only when this method is called.
        ``one_to_one_f1`` needs SciPy (``pip install "hedonic[metrics]"``).
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
        resolution: float | None = None,
        allow_isolation: bool = False,
        *,
        edge_weights=None,
        seed: int | None = None,
        node_weights=None,
        max_total_communities: int | None = None,
        n_communities: int | None = None,
    ):
        """Community detection with the CPM hedonic game, disjoint or overlapping.

        Each vertex is an agent that moves to the community (or, for covers,
        the set of at most ``max_memberships`` communities) it prefers, one
        vertex at a time, until no vertex can improve: every vertex is
        visited, in a random order, and revisited after a neighbour moves.
        A move is taken only when its computed gain exceeds a rigorous bound
        of its rounding error, so every move raises the exact potential and
        every call terminates. At return no unilateral deviation raises a
        vertex's exact payoff by more than
        ``_hedonic_provenance["equilibrium_slack"]`` times the total edge
        weight; the slack is 0 when the arithmetic is exact (for instance
        unit weights at resolution 0, 1/2 or 1), and the result is then an
        exact Nash equilibrium. The computation runs in this package's C
        core (``hedonic._core``). Every argument after ``allow_isolation`` is
        keyword-only.

        Parameters
        ----------
        initial_membership :
            Disjoint: flat list of community ids (length n).
            Overlapping (``max_memberships > 1``): flat list or list-of-lists
            (one community-id list per vertex). ``None`` builds a default start.
            When omitted, a valid ``self.memberships`` state is reused;
            explicit ``None`` requests the default singleton start, or a
            feasible round-robin start when a community-count limit is set.
            Flat and nested overlapping starts with the same labels are
            equivalent after normalization.
        max_communities :
            Initialization only: when ``initial_membership`` is ``None`` and
            ``max_memberships == 1``, random start over ``[0, max_communities)``
            (labels a draw leaves unused are dropped and the rest renumbered
            in order).
            It has no effect with a supplied start, with overlapping defaults,
            or when community-count limits select a feasible default start;
            ``_hedonic_provenance["max_communities_effect"]`` records which
            case applied. It is *not* an output constraint; see
            ``max_total_communities``/``n_communities``.
        max_memberships :
            Per-vertex cap. ``1`` (default) gives a partition
            (``VertexClustering``); ``> 1`` gives an overlapping cover
            (``VertexCover``) in which each vertex belongs to at most this
            many communities. ``-1`` removes the practical per-vertex cap (the
            effective cap is the vertex count, recorded in
            ``_hedonic_provenance``).
        resolution :
            CPM resolution γ. Defaults to the graph density. Must be a finite
            real. Smaller values give larger communities. Zero total edge
            weight is rejected: the normalized potential ``Φ_γ = Φ̃_γ / W`` is
            undefined when ``W=0``.
        allow_isolation :
            Allow a vertex to open a new, empty community.
        edge_weights :
            Optional edge weights (sequence or edge-attribute name); finite
            and non-negative for covers.
        seed :
            When given, the visiting order is drawn from
            ``random.Random(seed)``, and the same seed drives the random
            disjoint start. The returned labels are then a function of the
            graph, weights, parameters, start state and seed. ``None`` draws
            from Python's ``random`` module (python-igraph's default
            generator), so ``random.seed(...)`` also makes a run repeatable.
        node_weights :
            Optional vertex weights ``w_v`` (sequence or vertex-attribute
            name), finite and non-negative. They enter the crowding term
            ``γ w_u w_v``; unit weights are the default.
        max_total_communities :
            Optional upper bound on the number of occupied communities of the
            result and of every state visited.
        n_communities :
            Optional exact number of occupied communities. Without a start
            state, a deterministic round-robin start occupies all ``K``
            labels. A start state that violates either count limit is
            rejected.

        Returns
        -------
        VertexClustering if ``max_memberships == 1``, else VertexCover.

        The result carries ``_hedonic_provenance`` (versions, effective
        parameters, count mode) and ``_hedonic_native_quality`` (the
        potential ``Φ`` of the returned state). The returned memberships are
        also stored in ``self.memberships``, so a later call without
        ``initial_membership`` continues from them.
        """
        return _community_hedonic(
            self,
            initial_membership,
            max_communities,
            max_memberships,
            resolution,
            allow_isolation,
            edge_weights=edge_weights,
            seed=seed,
            node_weights=node_weights,
            max_total_communities=max_total_communities,
            n_communities=n_communities,
        )
