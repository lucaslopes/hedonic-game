from __future__ import annotations

import inspect

from igraph import Graph

from ._game.detection import community_hedonic as _community_hedonic
from ._game.evaluation import (
    EVALUATION_METHODS as _EVALUATION_METHODS_TABLE,
    evaluate_against as _evaluate_against,
)
from ._game.memberships import (
    as_overlapping_init as _as_overlapping_init_impl,
    memberships_as_cover as _memberships_as_cover,
    normalize_membership as _normalize_membership_impl,
    validate_membership as _validate_membership,
)


# Keywords of hedonic 1.x that 2.0 removed, and what replaces them.
_REMOVED_ARGUMENTS = {
    "n_iterations": "every call runs until no vertex can improve (the 1.x n_iterations=-1)",
    "local_move_only": "every call is local moving; for full Leiden use igraph's Graph.community_leiden",
    "beta": "Leiden's refinement is not part of hedonic 2.0",
    "debug_trace": "traces are not part of hedonic 2.0",
    "phase_policy": "pass a partition as initial_membership to warm-start a cover",
    "ensure_equilibrium": "every call returns an equilibrium",
}


def _removed_arguments(removed: dict) -> str:
    names = sorted(removed)
    known = [name for name in names if name in _REMOVED_ARGUMENTS]
    unknown = [name for name in names if name not in _REMOVED_ARGUMENTS]
    parts = []
    if unknown:
        listed = ", ".join(repr(name) for name in unknown)
        plural = "s" if len(unknown) > 1 else ""
        parts.append(f"got an unexpected keyword argument{plural} {listed}")
    if known:
        notes = "; ".join(f"{name}: {_REMOVED_ARGUMENTS[name]}" for name in known)
        parts.append(f"no longer takes {', '.join(known)} (removed in 2.0; {notes})")
    return "community_hedonic() " + "; it also ".join(parts)


class Game(Graph):
    """A hedonic game represented as an `igraph.Graph`."""

    # ``evaluate_against`` keeps this dispatch table next to the facade; the
    # scoring lives in ``hedonic.metrics`` and is imported lazily.
    _EVALUATION_METHODS = _EVALUATION_METHODS_TABLE

    def __init__(self, graph: Graph | None = None, *args, **kwargs) -> None:
        if graph is not None and not isinstance(graph, Graph):
            # Graph's own arguments (n, edges, directed, ...), positionally.
            super().__init__(graph, *args, **kwargs)
            self._memberships = None
            return
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

    def copy(self) -> "Game":
        """A copy of the graph, its attributes and its stored memberships."""
        return Game(self)

    @property
    def memberships(self) -> list[list[int]] | None:
        """Canonical per-vertex community memberships.

        One (non-empty) list of integer community labels per vertex: the
        last result of :meth:`community_hedonic`, or what was assigned.
        ``None`` means no state. The setter accepts the flat partition form
        (and tuples or NumPy arrays) and stores the normalized
        list-of-lists. A call starts from this state only when it is passed
        as ``initial_membership``. Assign a new state rather than editing the
        returned lists in place (an edit is not validated), and note that
        editing the graph does not update it.
        """
        return self._memberships

    @memberships.setter
    def memberships(self, value: list[int] | list[list[int]] | None) -> None:
        if value is None or (
            (isinstance(value, (list, tuple)) or hasattr(value, "ndim")) and len(value) == 0
        ):
            self._memberships = None   # no state
            return
        self._memberships = self._normalize_membership(value)


    def stable_intervals(
        self, memberships=None, *, max_memberships=None, allow_isolation=False,
        edge_weights=None, atol=1e-10, rtol=1e-9, vertices=None,
    ):
        """Audit a fixed cover's stable resolutions on [0, 1].

        Parameters
        ----------
        memberships:
            Per-vertex label lists, a flat partition, or an igraph cover;
            defaults to ``self.memberships``. Repeated labels count once,
            gaps are renumbered in label order, and input rows are not
            mutated. Every vertex needs a label; complete uncovered vertices
            with private labels before calling.
        max_memberships:
            Maximum labels in a deviation. Defaults to the largest row and
            cannot be smaller than it.
        allow_isolation:
            Whether a deviation may use a fresh label with score zero.
        edge_weights:
            Non-negative finite weights or an edge attribute name (``"weight"``).
            None is ones, as in ``community_hedonic``: audit with the weights
            the detector was given.
        atol, rtol:
            Heuristic tolerance ``atol + rtol * max(1, abs(current_utility))``.
            This is independent of the detector's certified rounding bound.
        vertices:
            A range or sequence of vertex indices; defaults to all vertices.
            Output follows this order, including repeated indices, so ranges
            can be split across processes.

        Returns
        -------
        list[list[tuple[float, float]]]
            Sorted, disjoint, closed stable intervals for each selected
            vertex. Empty lists mean never stable. Only unit node weights
            and undirected graphs without self-loops are supported.

        See Also
        --------
        regrets, hedonic.stability.stable_fraction,
        hedonic.stability.stable_measures, hedonic.stability.nash_window
        """
        from ._audit import stable_intervals

        return stable_intervals(
            self, memberships, max_memberships=max_memberships,
            allow_isolation=allow_isolation, edge_weights=edge_weights,
            atol=atol, rtol=rtol, vertices=vertices,
        )

    def regrets(
        self, resolutions, memberships=None, *, max_memberships=None,
        allow_isolation=False, edge_weights=None, atol=1e-10, rtol=1e-9,
        vertices=None,
    ):
        """Audit the best-response regret of a fixed cover on a resolution grid.

        Parameters
        ----------
        resolutions:
            One-dimensional sequence of finite resolutions in [0, 1].
        memberships, max_memberships, allow_isolation, edge_weights, atol, rtol, vertices:
            As in :meth:`stable_intervals`. Memberships default to the stored
            labelled state; the cap defaults to its largest row. Node weights
            are unit only.

        Returns
        -------
        hedonic.stability.RegretResult
            ``regret`` and ``tolerance`` NumPy arrays, shaped
            ``(len(vertices), len(resolutions))``. Regret is the best utility
            minus current utility, clamped to zero when at most 1e-15.
            A vertex is stable where ``regret <= tolerance``. The grid sums
            evaluated scores, so rounding near interval endpoints can differ
            from :meth:`stable_intervals`' affine coefficient sums.
        """
        from ._audit import regrets

        return regrets(
            self, resolutions, memberships, max_memberships=max_memberships,
            allow_isolation=allow_isolation, edge_weights=edge_weights,
            atol=atol, rtol=rtol, vertices=vertices,
        )

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
        restrict_to_reference: bool = False,
    ) -> float:
        """Score the stored partition/cover against an explicit ground truth.

        ``community_hedonic`` persists its return value in ``memberships`` as
        one list of community labels per vertex.  This convenience facade
        converts that state to the community-list representation used by the
        experimental evaluator and returns the selected scalar score.

        Parameters
        ----------
        ground_truth:
            Community lists (each a list of vertices of this graph), an igraph
            ``VertexClustering``/``VertexCover`` of a graph with the same
            vertices, or a flat vector with one hashable label per vertex
            (``None`` and NaN are refused: every vertex needs a label). A
            list whose entries are lists is read as communities, not as one
            row of labels per vertex. Communities with the same members count
            once, on both sides.
        method:
            ``f1`` (symmetric best-match), ``one_to_one_f1``, ``jaccard``,
            ``omega``, ``node_micro_f1``, or ``size_weighted_community_f1``.
            A few explicit aliases are accepted for compatibility.
        singleton_mode, matching_weight, omega_sample_size, omega_seed:
            Options forwarded to :func:`hedonic.metrics.evaluate_cover`.
            ``omega`` is estimated from ``omega_sample_size`` vertex pairs
            drawn with ``omega_seed``.
        restrict_to_reference:
            Score only the vertices the reference covers: every predicted
            community loses the vertices that are in no reference community
            (and is dropped if none is left). This is the usual protocol for
            a reference that labels only part of the graph (the SNAP top-5000
            covers, for example): without it, communities found among the
            unlabelled vertices count as unmatched predictions.

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
            restrict_to_reference=restrict_to_reference,
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
        """Whether ``membership`` is a valid membership state of this graph.

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

        With ``max_memberships``, no row may hold more labels. This checks a
        state strictly. ``community_hedonic`` is more lenient with a start:
        it reads a repeated label in a row once, renumbers labels with gaps
        in their order, and projects a row with more than
        ``max_memberships`` labels to the ones its vertex prefers.
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
        renumber_labels: bool = False,
        name: str = "memberships",
    ) -> list[list[int]]:
        """Validate and normalize a partition or cover to per-vertex rows.

        Labels are required to be non-negative integers and contiguous from
        zero, matching igraph's membership convention.  Every row must be
        non-empty and duplicate labels within a vertex are rejected unless
        ``collapse_duplicate_labels`` is set, and labels with gaps are
        rejected unless ``renumber_labels`` is set, as detector initialization
        does for its starts.  The optional cap is checked here so persisted
        state cannot silently be accepted for an incompatible detector call.
        """
        return _normalize_membership_impl(
            self,
            membership,
            max_memberships=max_memberships,
            collapse_duplicate_labels=collapse_duplicate_labels,
            renumber_labels=renumber_labels,
            name=name,
        )

    @staticmethod
    def _as_overlapping_init(
        membership: list[int] | list[list[int]],
    ) -> list[list[int]]:
        """Normalize flat or nested membership to list-of-lists (per vertex)."""
        return _as_overlapping_init_impl(membership)

    def community_hedonic(
        self,
        initial_membership: list[int] | list[list[int]] | None = None,
        *positional,
        max_communities: int | None = None,
        max_memberships: int = 1,
        resolution: float | None = None,
        allow_isolation: bool = False,
        edge_weights=None,
        seed: int | None = None,
        node_weights=None,
        max_total_communities: int | None = None,
        n_communities: int | None = None,
        **removed,
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
        weight. For a partition the slack is 0 whenever no deviation's exact
        gain can be positive: always when the arithmetic is exact (for
        instance unit weights at resolution 0, 1/2 or 1), and often
        otherwise; the result is then an exact Nash equilibrium. Covers,
        whose payoffs divide by square roots, keep a tiny positive slack
        (about 1e-15 on small graphs, less on large ones). The result
        is an equilibrium reached from the start, not the best partition or
        cover: from singletons at resolution 0, for instance, a connected
        graph need not end as one community. The computation runs in this
        package's own core (``hedonic._core``). Every argument after
        ``initial_membership`` is keyword-only.

        The graph must be undirected and free of self-loops (a loop ties a
        vertex to no one else; ``g.simplify(multiple=False, loops=True)``
        removes them). Parallel edges count as one edge whose weight is
        their sum.

        Parameters
        ----------
        initial_membership :
            The start. ``None`` (the default) is the singleton start, a
            random start with ``max_communities``, or a feasible round-robin
            start when a community-count limit is set. Otherwise one entry
            per vertex: a community label (a partition) or a list of labels
            (a cover); labels are non-negative integers, and labels with
            gaps are renumbered in their order (0, 2, 5 become 0, 1, 2). A
            list, tuple or NumPy array is accepted, as is an igraph
            ``VertexClustering`` or ``VertexCover`` of a graph with the same
            vertices. A label repeated in a row counts once. To continue from
            the previous result, pass ``initial_membership=g.memberships``.
            A vertex with more labels than ``max_memberships`` keeps the
            ``max_memberships`` it prefers, ranked by the payoff of belonging
            to each community alone, evaluated on the start as given (ties
            to the larger community, then the smaller label); the detector
            then moves from there to an equilibrium. A cover passed to a
            partition call (``max_memberships=1``) thus becomes the partition
            in which every vertex keeps its preferred community.
            ``_hedonic_provenance["projected_vertices"]`` counts the vertices
            that had to choose.
        max_communities :
            Initialization only: when ``initial_membership`` is ``None`` and
            ``max_memberships == 1``, random start over ``[0, max_communities)``
            (labels a draw leaves unused are dropped and the rest renumbered
            in order).
            It has no effect with a supplied start, with overlapping defaults,
            or when community-count limits select a feasible default start;
            ``_hedonic_provenance["max_communities_effect"]`` records which
            case applied. It is not an output constraint (see
            ``max_total_communities``/``n_communities``), but without
            ``allow_isolation`` no vertex can open a new community, so the
            result then has at most ``max_communities`` communities.
        max_memberships :
            Per-vertex cap. ``1`` (default) gives a partition
            (``VertexClustering``); ``> 1`` gives an overlapping cover
            (``VertexCover``) in which each vertex belongs to at most this
            many communities. ``-1`` removes the practical per-vertex cap (the
            effective cap is the vertex count, recorded in
            ``_hedonic_provenance``).
        resolution :
            CPM resolution γ. Defaults to the weighted density, the total
            edge weight ``W`` over the number of vertex pairs (the density of
            a simple unweighted graph). Must be a finite real. Smaller values
            give larger communities. A total edge weight ``W <= 0`` is
            rejected: the quality, the potential divided by ``W``, is
            undefined.
        allow_isolation :
            Allow a vertex to open a new, empty community.
        edge_weights :
            Optional edge weights (a sequence of real numbers in edge order,
            or an edge-attribute name), finite; non-negative for covers. A
            partition accepts negative weights as long as ``W > 0``. An edge
            attribute named ``"weight"`` is not used unless named here.
            Booleans, strings, complex numbers, ``None`` and dicts are
            refused. The sum of their absolute values must stay below about
            4e304, and ``W`` must be a normal double (at least about 2.2e-308).
        seed :
            When given, the visiting order is drawn from
            ``random.Random(seed)``, and the same seed drives the random
            disjoint start. The returned labels are then a function of the
            graph, weights, parameters, start state and seed. ``None`` draws
            from Python's ``random`` module (python-igraph's default
            generator), so ``random.seed(...)`` also makes a run repeatable.
            Any integer is a seed, negative ones included, as for
            ``random.Random``. The visiting order follows the vertex ids and
            each vertex's edges in the graph's edge order, so a repeatable run
            also needs the same vertex numbering and edge order.
        node_weights :
            Optional vertex weights ``w_v`` (a sequence of real numbers in
            vertex order, or a vertex-attribute name), finite and
            non-negative. They enter the crowding term ``γ w_u w_v``; unit
            weights are the default. ``|resolution| x N² x max_memberships``,
            for the total vertex weight ``N``, must stay below about 4e304.
        max_total_communities :
            Optional upper bound on the number of occupied communities of the
            result and of every state visited.
        n_communities :
            Optional exact number of occupied communities. Without a start
            state, a deterministic round-robin start occupies all ``K``
            labels. A start state that violates either count limit (after
            its projection to ``max_memberships``) is rejected.

        Returns
        -------
        VertexClustering if ``max_memberships == 1``, else VertexCover.

        A cover has one community per occupied label, and two labels may
        have the same members: they are different coalitions of the game
        (each counts in every member's utility and in the potential), so
        they are kept. To compare with a reference cover, deduplicate the
        bodies first, e.g. ``set(map(frozenset, cover))``.

        The result carries ``_hedonic_provenance`` (versions, effective
        parameters, count mode) and ``_hedonic_native_quality``:
        ``(Φ - γ Σ_v n_v² / 2) / W`` for the exact potential ``Φ`` of the
        returned state (a sum over pairs of vertices), the vertex weights
        ``n_v`` (1 by default) and the total edge weight ``W``; the constant
        term is each vertex's own crowding, the same for every state. The returned
        memberships are also stored in ``self.memberships``; pass
        ``initial_membership=g.memberships`` to continue from them.

        Every vertex of a cover keeps at least one community: the actions are
        the non-empty sets of at most ``max_memberships`` labels (with
        ``allow_isolation`` one of them is a new community of its own, worth
        0). A cover can return two labels with the same members; they are
        kept apart, as the equilibrium is a statement about labels.

        The binding holds the GIL for the whole call, so calls from several
        threads are safe but run one at a time (use processes to run them in
        parallel); with ``seed=None`` they also share the ``random`` module's
        state.
        """
        if positional:
            raise TypeError(
                "community_hedonic() takes initial_membership as its only "
                f"positional argument ({len(positional)} more given); pass "
                "the others by keyword. In 2.0 they are keyword-only (in 1.x "
                "the fourth positional argument was n_iterations, which 2.0 "
                "removed)"
            )
        if removed:
            raise TypeError(_removed_arguments(removed))
        return _community_hedonic(
            self,
            initial_membership,
            max_communities=max_communities,
            max_memberships=max_memberships,
            resolution=resolution,
            allow_isolation=allow_isolation,
            edge_weights=edge_weights,
            seed=seed,
            node_weights=node_weights,
            max_total_communities=max_total_communities,
            n_communities=n_communities,
        )


def _documented_signature(function) -> inspect.Signature:
    """``function``'s signature without the catch-alls that only report misuse."""
    signature = inspect.signature(function)
    return signature.replace(parameters=[
        parameter for parameter in signature.parameters.values()
        if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
    ])


# help() and IDEs show the keyword-only options; *positional and **removed
# exist only to explain 1.x calls.
Game.community_hedonic.__signature__ = _documented_signature(Game.community_hedonic)
