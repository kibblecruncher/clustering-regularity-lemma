#!/usr/bin/env python3
"""
Detailed analysis of what assemble_partition should actually return.

KEY INSIGHT:
- For each direction, we have local partitions at each vertex v in V2
- These local partitions mark neighbors of v with True/False
- We need to convert these to GLOBAL EDGE PARTITIONS

The output should be:
- For E12: bitmask of length = number of E12 edges
- For E23: bitmask of length = number of E23 edges
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Set


class EdgePartitionAnalyzer:
    """Analyze how to correctly assemble edge partitions from local vertex partitions."""
    
    def __init__(self, G: nx.Graph):
        self.G = G
        self.make_tripartite()
    
    def make_tripartite(self):
        """Create the tripartite cover H."""
        self.H = nx.Graph()
        V1 = [(node, 0) for node in self.G]
        V2 = [(node, 1) for node in self.G]
        V3 = [(node, 2) for node in self.G]
        
        self.H.add_nodes_from(V1, part=0)
        self.H.add_nodes_from(V2, part=1)
        self.H.add_nodes_from(V3, part=2)
        
        for i, j in [(0, 1), (1, 2), (2, 0), (1, 0), (2, 1), (0, 2)]:
            self.H.add_edges_from(((u, i), (v, j)) for u, v in self.G.edges())
    
    def get_edge_mapping(self) -> Tuple[List, List]:
        """
        Get list of E12 and E23 edges with their indices.
        
        Returns:
            (E12_edges, E23_edges) where each is a list of edges with (u, v) tuples
        """
        E12 = []  # Edges between V1 (part 0) and V2 (part 1)
        E23 = []  # Edges between V2 (part 1) and V3 (part 2)
        
        for u, v in self.H.edges():
            u_part = self.H.nodes[u]['part']
            v_part = self.H.nodes[v]['part']
            
            # Ensure u_part < v_part for consistent ordering
            if u_part > v_part:
                u, v = v, u
                u_part, v_part = v_part, u_part
            
            if u_part == 0 and v_part == 1:
                E12.append((u, v))
            elif u_part == 1 and v_part == 2:
                E23.append((u, v))
        
        return E12, E23
    
    def get_vertex_neighbors_mapping(self, v: int) -> Dict[str, List]:
        """
        For vertex v in V2, get its neighbors in the tripartite graph.
        
        Returns dict with keys 'A' and 'B' for part 0 and part 2 neighbors.
        """
        neighbors = list(self.H.neighbors((v, 1)))
        
        A = [n for n in neighbors if self.H.nodes[n]['part'] == 0]
        B = [n for n in neighbors if self.H.nodes[n]['part'] == 2]
        
        return {'A': A, 'B': B}
    
    def local_partition_to_global_edges(self, 
                                       v: int,
                                       A_bitmask: np.ndarray,
                                       B_bitmask: np.ndarray,
                                       E12: List,
                                       E23: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a local partition at vertex v to contributions in global edge partitions.
        
        Args:
            v: Vertex index in V2
            A_bitmask: Bitmask of length len(A_neighbors), marking selected A neighbors
            B_bitmask: Bitmask of length len(B_neighbors), marking selected B neighbors
            E12: List of all E12 edges
            E23: List of all E23 edges
        
        Returns:
            (E12_contribution, E23_contribution) - bitmasks indicating which edges are marked
        """
        neighbors = self.get_vertex_neighbors_mapping(v)
        A_neighbors = neighbors['A']
        B_neighbors = neighbors['B']
        
        # Initialize contributions
        E12_contrib = np.zeros(len(E12), dtype=bool)
        E23_contrib = np.zeros(len(E23), dtype=bool)
        
        # For E12: edges are from A neighbors to (v, 1)
        # An edge ((u, 0), (v, 1)) is in E12
        # If the corresponding A neighbor is marked True, mark this edge
        for idx, a_neighbor in enumerate(A_neighbors):
            if A_bitmask[idx]:
                # Find this edge in E12 and mark it
                edge = (a_neighbor, (v, 1))
                if edge in E12:
                    e_idx = E12.index(edge)
                    E12_contrib[e_idx] = True
                else:
                    # Try reverse direction
                    edge = ((v, 1), a_neighbor)
                    if edge in E12:
                        e_idx = E12.index(edge)
                        E12_contrib[e_idx] = True
        
        # For E23: edges are from (v, 1) to B neighbors
        # An edge ((v, 1), (u, 2)) is in E23
        # If the corresponding B neighbor is marked True, mark this edge
        for idx, b_neighbor in enumerate(B_neighbors):
            if B_bitmask[idx]:
                # Find this edge in E23 and mark it
                edge = ((v, 1), b_neighbor)
                if edge in E23:
                    e_idx = E23.index(edge)
                    E23_contrib[e_idx] = True
                else:
                    # Try reverse direction
                    edge = (b_neighbor, (v, 1))
                    if edge in E23:
                        e_idx = E23.index(edge)
                        E23_contrib[e_idx] = True
        
        return E12_contrib, E23_contrib
    
    def analyze_K3(self):
        """Analyze complete graph K3 in detail."""
        print("\n" + "="*80)
        print("ANALYSIS: Complete Graph K3")
        print("="*80)
        
        print(f"\nOriginal graph edges: {list(self.G.edges())}")
        
        E12, E23 = self.get_edge_mapping()
        print(f"\nE12 edges (part 0 - part 1): {E12}")
        print(f"E23 edges (part 1 - part 2): {E23}")
        
        print(f"\nTotal E12 edges: {len(E12)}")
        print(f"Total E23 edges: {len(E23)}")
        
        # For each vertex in V2
        V2 = [0, 1, 2]
        for v in V2:
            print(f"\n--- Vertex {v} in V2 ---")
            neighbors = self.get_vertex_neighbors_mapping(v)
            
            print(f"A neighbors (part 0): {neighbors['A']}")
            print(f"B neighbors (part 2): {neighbors['B']}")
            
            # Create a simple partition: mark first half of A and B as True
            A_bitmask = np.zeros(len(neighbors['A']), dtype=bool)
            B_bitmask = np.zeros(len(neighbors['B']), dtype=bool)
            
            if len(neighbors['A']) > 0:
                A_bitmask[0] = True
            if len(neighbors['B']) > 0:
                B_bitmask[0] = True
            
            print(f"A_bitmask: {A_bitmask}")
            print(f"B_bitmask: {B_bitmask}")
            
            # Convert to global edge partitions
            E12_contrib, E23_contrib = self.local_partition_to_global_edges(
                v, A_bitmask, B_bitmask, E12, E23
            )
            
            print(f"E12 contribution: {E12_contrib}")
            print(f"E23 contribution: {E23_contrib}")
            
            # Show which edges are marked
            marked_E12 = [E12[i] for i in range(len(E12)) if E12_contrib[i]]
            marked_E23 = [E23[i] for i in range(len(E23)) if E23_contrib[i]]
            
            print(f"Marked E12 edges: {marked_E12}")
            print(f"Marked E23 edges: {marked_E23}")


if __name__ == "__main__":
    G = nx.complete_graph(3)
    analyzer = EdgePartitionAnalyzer(G)
    analyzer.analyze_K3()
    
    print("\n" + "="*80)
    print("KEY INSIGHT")
    print("="*80)
    print("""
assemble_partition should do the following:
1. For each direction string
2. For each vertex v in V2:
   a. Load the partition (A_neighbors, B_neighbors)
   b. These are node identifiers of neighbors in the tripartite graph
   c. NOT JUST THE NEIGHBORS LIST, but the actual bitmask data
   d. Convert each bitmask to a contribution to the global E12 and E23 edge partitions
3. Combine contributions from all vertices into final E12 and E23 partitions
4. Return E12 and E23 as global boolean arrays

CURRENT ISSUE:
The current assemble_partition is returning node identifiers, not bitmasks!
It's concatenating the A and B neighbor lists, not the actual partition masks.
    """)
