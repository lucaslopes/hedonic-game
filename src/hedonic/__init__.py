import random
from igraph import Graph


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
    
    def validate_membership(self, membership: list[int]) -> bool:
        """Validate a community membership vector.

        Rules:
        - Length must equal the number of vertices
        - Labels must be non-negative integers
        - Labels must be contiguous starting at 0 (i.e., {0, 1, ..., K-1})
        """
        if not isinstance(membership, list):
            return False
        if len(membership) != self.vcount():
            return False
        if not membership:
            return False
        # Ensure all entries are ints and non-negative
        for label in membership:
            if not isinstance(label, int):
                return False
            if label < 0:
                return False
        unique_labels = set(membership)
        # Contiguous labels starting from 0
        max_label = max(unique_labels)
        expected = set(range(0, max_label + 1))
        return unique_labels == expected
    
    def community_hedonic(self,
            initial_membership: list[int] = None,
            max_communities: int | None = None,
            n_iterations: int = -1,
            resolution: float = None,
            allow_isolation: bool = False,
            only_local_moving: bool = True,
            edge_weights = None,
        ):
        """
        Community detection using the hedonic game model.
        """
        # Determine initial membership according to the specified rules
        if initial_membership is not None:
            if not self.validate_membership(initial_membership):
                raise ValueError("Invalid initial_membership: expected list of contiguous non-negative integer labels starting at 0, length equal to number of vertices")
            membership_vector = initial_membership
        else:
            # No initial membership provided
            if max_communities is None:
                # Singleton partition: each node in its own community
                membership_vector = list(range(self.vcount()))
            else:
                if not isinstance(max_communities, int) or max_communities <= 0:
                    raise ValueError("max_communities must be a positive integer when provided")
                # Random initialization with K possible labels [0..K-1]
                membership_vector = [random.randrange(max_communities) for _ in range(self.vcount())]

        p = self.community_leiden(
            initial_membership=membership_vector,
            n_iterations=n_iterations,
            resolution=self.density() if resolution is None else resolution,
            allow_isolation=allow_isolation,
            only_local_moving=only_local_moving,
            edge_weights=edge_weights)
        return p


__all__ = ["Game"]
