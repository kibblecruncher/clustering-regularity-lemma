from typing import List, Tuple

import networkx as nx


class GraphManager(object):
    """Builds and queries the tripartite cover used by the algorithm."""

    def __init__(self, G: nx.Graph):
        self.H = self.makeTripartite(G)

    def makeTripartite(self, G: nx.Graph) -> nx.Graph:
        """Generates the tripartite cover of the graph."""
        H = nx.Graph()
        V1 = [(node, 0) for node in G]
        V2 = [(node, 1) for node in G]
        V3 = [(node, 2) for node in G]

        H.add_nodes_from(V1, part=0)
        H.add_nodes_from(V2, part=1)
        H.add_nodes_from(V3, part=2)
        for i, j in [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]:
            H.add_edges_from(((u, i), (v, j)) for u, v in G.edges())
        return H

    def getV2(self) -> List[int]:
        """Returns the vertices in the second part of the tripartite cover."""
        return [n[0] for n, d in self.H.nodes(data=True) if d.get("part") == 1]

    def makeLinkGraph(self, vertex: int) -> nx.Graph:
        """Generates the link graph of the given vertex."""
        neighbors = list(self.H.neighbors((vertex, 1)))
        return self.H.subgraph(neighbors).copy()

    def linkGraphIndex(self, vertex: int):
        """Maps neighbors of vertex to the edge index in the link graph."""
        N = self.makeLinkGraph(vertex)
        return {n: i for i, n in enumerate(N.nodes())}

    def makeLinkPartition(self, vertex: int) -> Tuple[List, List]:
        """Generates the link partition of the given vertex."""
        N = self.makeLinkGraph(vertex)
        A = [n for n, d in N.nodes(data=True) if d.get("part") == 0]
        B = [n for n, d in N.nodes(data=True) if d.get("part") == 2]
        return A, B
