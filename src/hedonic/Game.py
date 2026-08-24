from __future__ import annotations

from igraph import Graph

from .utils import sample_uniform_ints


class Game(Graph):
    """A hedonic game represented as an `igraph.Graph`."""

    def __init__(self, graph: Graph | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(graph, Graph):
            self.add_vertices(graph.vcount())
            self.add_edges(graph.get_edgelist())
            for attr in graph.vertex_attributes():
                self.vs[attr] = graph.vs[attr]
            for attr in graph.edge_attributes():
                self.es[attr] = graph.es[attr]
            for attr in graph.attributes():
                self[attr] = graph[attr]

    def to_igraph(self) -> Graph:
        """Convert the `Game` object to a standalone `igraph.Graph` instance."""
        g = Graph()
        g.add_vertices(self.vcount())
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
    ) -> bool:
        """Validate a community membership vector or overlapping cover init.

        Disjoint (``overlapping=False``):
          - length == n
          - non-negative integer labels
          - labels contiguous from 0

        Overlapping (``overlapping=True``):
          - length == n
          - each entry is a non-empty list of non-negative community ids
            (one membership list per vertex), **or** a flat disjoint vector
            (accepted and later expanded to singleton lists)
        """
        if not isinstance(membership, list):
            return False
        if len(membership) != self.vcount():
            return False
        if not membership:
            return False

        if overlapping:
            first = membership[0]
            if isinstance(first, (list, tuple)):
                labels: set[int] = set()
                for entry in membership:
                    if not isinstance(entry, (list, tuple)) or len(entry) == 0:
                        return False
                    for label in entry:
                        if not isinstance(label, int) or label < 0:
                            return False
                        labels.add(label)
                if not labels:
                    return False
                max_label = max(labels)
                return labels == set(range(0, max_label + 1))
            # flat vector allowed as overlapping init (expanded later)
            overlapping = False

        if not overlapping:
            for label in membership:
                if not isinstance(label, int) or label < 0:
                    return False
            unique_labels = set(membership)
            max_label = max(unique_labels)
            return unique_labels == set(range(0, max_label + 1))

        return False

    @staticmethod
    def _as_overlapping_init(
        membership: list[int] | list[list[int]],
    ) -> list[list[int]]:
        """Normalize flat or nested membership to list-of-lists (per vertex)."""
        if membership and isinstance(membership[0], (list, tuple)):
            return [list(map(int, entry)) for entry in membership]
        return [[int(c)] for c in membership]

    def community_hedonic(
        self,
        initial_membership: list[int] | list[list[int]] | None = None,
        max_communities: int | None = None,
        max_memberships: int = 1,
        n_iterations: int = -1,
        resolution: float | None = None,
        allow_isolation: bool = False,
        local_move_only: bool = True,
        edge_weights=None,
        seed: int | None = None,
        beta: float = 0.01,
        ensure_equilibrium: bool = False,
    ):
        """Community detection with the hedonic-game / Leiden local-moving model.

        This is the primary exploratory method for both disjoint and overlapping
        experiments. It delegates to ``community_leiden`` from lucas-igraph.

        Parameters
        ----------
        initial_membership :
            Disjoint: flat list of community ids (length n).
            Overlapping (``max_memberships > 1``): flat list or list-of-lists
            (one community-id list per vertex). ``None`` builds a default init.
        max_communities :
            When ``initial_membership`` is ``None`` and ``max_memberships == 1``,
            random init over ``[0, max_communities)``. Ignored for overlapping
            defaults (singleton cover).
        max_memberships :
            ``1`` (default) → disjoint clustering (``VertexClustering``).
            ``> 1`` → overlapping cover (``VertexCover``); each vertex may
            belong to at most this many communities.
        n_iterations :
            Leiden outer iterations. A negative value asks the native binding
            to repeat its outer cycle until it reports no changed clustering;
            this is the binding's queue/candidate stopping condition, not an
            independent proof that every admissible move has non-positive
            regret. Experiments that require a game-theoretic equilibrium
            should audit the returned partition/cover explicitly.
        resolution :
            CPM resolution γ. Defaults to graph density.
        allow_isolation :
            Allow moves into empty communities (new clusters).
        local_move_only :
            If True (default), run only the local-moving phase — the hedonic
            best-response phase. If False, run full Leiden (refine + aggregate).
        ensure_equilibrium :
            If True, require the native binding to run until its no-change
            stopping condition by forcing ``n_iterations=-1``. The released
            lucas-igraph 1.0.0.3 native mover includes the best omitted
            existing community when ``allow_isolation=False``, so no Python
            full-Leiden → local-only cleanup pass is needed. The returned
            state should still be independently audited when a mathematical
            equilibrium certificate is required.
        edge_weights :
            Optional edge weights.
        seed :
            RNG seed for random disjoint initialization.
        beta :
            Leiden refinement randomness (used when ``local_move_only`` is False).

        Returns
        -------
        VertexClustering if ``max_memberships == 1``, else VertexCover.
        """
        if not isinstance(max_memberships, int) or max_memberships < 1:
            raise ValueError("max_memberships must be an integer >= 1")

        overlapping = max_memberships > 1
        res = self.density() if resolution is None else resolution

        if initial_membership is not None:
            if not self.validate_membership(
                initial_membership, overlapping=overlapping
            ):
                raise ValueError(
                    "Invalid initial_membership: expected length-n labels "
                    "(disjoint: contiguous ints from 0; overlapping: flat ints "
                    "or non-empty community-id lists per vertex)"
                )
            if overlapping:
                membership_vector = self._as_overlapping_init(initial_membership)
            else:
                membership_vector = list(initial_membership)
        else:
            if overlapping:
                # Singleton cover: vertex v alone in community v
                membership_vector = [[v] for v in range(self.vcount())]
            elif max_communities is None:
                membership_vector = list(range(self.vcount()))
            else:
                if not isinstance(max_communities, int) or max_communities <= 0:
                    raise ValueError(
                        "max_communities must be a positive integer when provided"
                    )
                if seed is None:
                    seed = 42
                membership_vector = sample_uniform_ints(
                    self.vcount(), max_communities - 1, seed
                ).tolist()

        result = self.community_leiden(
            initial_membership=membership_vector,
            n_iterations=-1 if ensure_equilibrium else n_iterations,
            resolution=res,
            allow_isolation=allow_isolation,
            local_move_only=local_move_only,
            weights=edge_weights,
            max_memberships=max_memberships,
            beta=beta,
        )

        if ensure_equilibrium:
            # Keep the native membership vectors available to experiment
            # layers without changing the public return type.  This attribute
            # is now provenance for the single native call, not a pre-cleanup
            # snapshot from a Python two-pass protocol.
            if max_memberships > 1:
                native_memberships = [
                    [int(label) for label in labels]
                    for labels in result.membership
                ]
            else:
                native_memberships = [[int(label)] for label in result.membership]
            setattr(result, "_hedonic_raw_memberships", native_memberships)

        return result
