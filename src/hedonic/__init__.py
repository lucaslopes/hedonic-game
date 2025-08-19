from igraph import Graph  # type: ignore


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


__all__ = ["Game"]
