#!/usr/bin/env python3
"""
Test suite for assemble_partition to verify correct mapping of neighbor bitmasks to global edge partitions.

The key insight is that for each vertex v in V2:
- Partitions A, B are bitmasks of the neighbors of v
- These need to be mapped to edges in the overall graph
- If neighbor i of v is marked true in A, then edge (neighbor_i, v) should be marked true in output
"""

import pytest
import os
import shutil
import json
import networkx as nx
import numpy as np
from typing import Tuple, List, Dict

from manager import Manager, FileManager, GraphManager


class AssemblePartitionTester:
    """Helper to test assemble_partition with known inputs."""
    
    def __init__(self):
        self.partition_dir = "test_assemble_partitions"
        self.graph_dir = "test_assemble_graphs"
        self.fm = FileManager(self.partition_dir, self.graph_dir)
    
    def cleanup(self):
        """Clean up test directories."""
        for d in [self.partition_dir, self.graph_dir]:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
    
    def create_test_case_small_complete(self) -> Manager:
        """Create a simple test case with complete graph K3."""
        G = nx.complete_graph(3)
        
        eps = 0.5
        manager = Manager(
            G=G,
            eps=eps,
            irreg_vtx_threshold=eps**5 / 90,
            dev_vtx_threshold=eps**2 / 9,
            irreg_vtx_count_threshold=eps**(5/2) / 9,
            dev_threshold=2 * eps**2 / 5,
            irreg_threshold=2 * eps**(5/2) / 5,
            clustering_threshold=eps,
            max_depth=10
        )
        
        # Override file manager to use test directories
        manager.partition_manager = self.fm
        
        return manager, G
    
    def manually_create_partitions(self, manager: Manager, direction: str):
        """Manually create known partition files for testing."""
        manager.V2 = manager.graph_manager.getV2()
        
        # For complete graph K3: vertices 0, 1, 2 in original graph
        # In tripartite: (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)
        # V2 = [0, 1, 2] (vertices with part=1)
        
        # For each vertex v in V2, get its neighbors in the link graph
        for v in manager.V2:
            neighbors = list(manager.graph_manager.H.neighbors((v, 1)))
            link_graph = manager.graph_manager.makeLinkGraph(v)
            
            print(f"\nVertex {v} in V2:")
            print(f"  Neighbors in H: {neighbors}")
            print(f"  Link graph nodes: {list(link_graph.nodes())}")
            print(f"  Link graph edges: {list(link_graph.edges())}")
            
            # Get the partition subsets
            A_neighbors = [n for n in neighbors if manager.graph_manager.H.nodes[n]['part'] == 0]
            B_neighbors = [n for n in neighbors if manager.graph_manager.H.nodes[n]['part'] == 2]
            
            print(f"  A neighbors (part=0): {A_neighbors}")
            print(f"  B neighbors (part=2): {B_neighbors}")
            
            # Create simple test partitions: mark first half as True
            A_bitmask = np.zeros(len(A_neighbors), dtype=bool)
            B_bitmask = np.zeros(len(B_neighbors), dtype=bool)
            
            if len(A_neighbors) > 0:
                A_bitmask[0:max(1, len(A_neighbors)//2)] = True
            if len(B_neighbors) > 0:
                B_bitmask[0:max(1, len(B_neighbors)//2)] = True
            
            print(f"  A_bitmask: {A_bitmask}")
            print(f"  B_bitmask: {B_bitmask}")
            
            # Save as partition - convert to numpy arrays, then get neighbors as lists for JSON
            # But actually, we want to save the neighbors themselves (the node identifiers)
            A_neighbors_array = np.array(A_neighbors)
            B_neighbors_array = np.array(B_neighbors)
            self.fm.savePartition(v, direction, A_neighbors_array, B_neighbors_array)


def test_assemble_partition_k3_manual():
    """Test assemble_partition with manually created partitions for K3."""
    tester = AssemblePartitionTester()
    
    try:
        manager, G = tester.create_test_case_small_complete()
        direction = "test"
        
        print(f"\n{'='*80}")
        print("TEST: assemble_partition with K3 (Complete Graph on 3 vertices)")
        print(f"{'='*80}")
        
        print(f"\nOriginal Graph: {nx.to_dict_of_lists(G)}")
        
        # Get V2 and print structure
        manager.V2 = manager.graph_manager.getV2()
        print(f"\nV2 vertices: {manager.V2}")
        print(f"Tripartite graph H nodes: {dict(manager.graph_manager.H.nodes(data=True))}")
        print(f"Tripartite graph H edges: {list(manager.graph_manager.H.edges())}")
        
        # Manually create and save partitions
        tester.manually_create_partitions(manager, direction)
        
        # Now try to assemble
        print(f"\nAttempting to assemble partition for direction '{direction}'...")
        
        try:
            part_A, part_B = manager.assemble_partition(direction)
            print(f"Success!")
            print(f"  Assembled partition A: {part_A}")
            print(f"  Assembled partition B: {part_B}")
        except Exception as e:
            print(f"Failed with error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    finally:
        tester.cleanup()


def test_link_partition_understanding():
    """Test to understand what link partitions contain."""
    print(f"\n{'='*80}")
    print("TEST: Understanding link partitions")
    print(f"{'='*80}")
    
    G = nx.complete_graph(4)
    
    eps = 0.5
    manager = Manager(
        G=G,
        eps=eps,
        irreg_vtx_threshold=eps**5 / 90,
        dev_vtx_threshold=eps**2 / 9,
        irreg_vtx_count_threshold=eps**(5/2) / 9,
        dev_threshold=2 * eps**2 / 5,
        irreg_threshold=2 * eps**(5/2) / 5,
        clustering_threshold=eps,
        max_depth=10
    )
    
    manager.V2 = manager.graph_manager.getV2()
    
    print(f"\nGraph: K4 (complete graph on 4 vertices)")
    print(f"Vertices: {list(G.nodes())}")
    print(f"Edges: {list(G.edges())}")
    
    # Examine one vertex in detail
    v_test = manager.V2[0]
    print(f"\nExamining vertex {v_test} in V2:")
    
    # Get neighbors in tripartite graph
    neighbors_H = list(manager.graph_manager.H.neighbors((v_test, 1)))
    print(f"  Neighbors in H: {neighbors_H}")
    
    # Separate by part
    A_neighbors = [n for n in neighbors_H if manager.graph_manager.H.nodes[n]['part'] == 0]
    B_neighbors = [n for n in neighbors_H if manager.graph_manager.H.nodes[n]['part'] == 2]
    
    print(f"  A-neighbors (part 0): {A_neighbors}")
    print(f"  B-neighbors (part 2): {B_neighbors}")
    
    # Get link graph
    link_graph = manager.graph_manager.makeLinkGraph(v_test)
    print(f"  Link graph nodes: {list(link_graph.nodes())}")
    print(f"  Link graph edges: {list(link_graph.edges())}")
    
    # Get the partition arrays returned by makeLinkPartition
    A, B = manager.graph_manager.makeLinkPartition(v_test)
    print(f"  Partition A from makeLinkPartition: {A}")
    print(f"  Partition B from makeLinkPartition: {B}")
    
    # The key question: what do A and B represent?
    # A should be the nodes from part 0 in the link graph
    # B should be the nodes from part 2 in the link graph
    print(f"\nInterpretation:")
    print(f"  A are nodes from V1 that are neighbors of v: {A}")
    print(f"  B are nodes from V3 that are neighbors of v: {B}")
    print(f"  These represent edges: ")
    for node_a in A:
        print(f"    Edge {node_a} -- {(v_test, 1)}")
    for node_b in B:
        print(f"    Edge {(v_test, 1)} -- {node_b}")


def test_edge_partition_semantics():
    """Test to understand what the edge partition should represent."""
    print(f"\n{'='*80}")
    print("TEST: Edge partition semantics")
    print(f"{'='*80}")
    
    G = nx.complete_graph(3)
    
    print(f"\nGraph: K3")
    print(f"Vertices: {list(G.nodes())}")
    print(f"Edges in original graph: {list(G.edges())}")
    
    # Count edges
    n_edges = G.number_of_edges()
    print(f"Total edges: {n_edges}")
    
    # The partition should have one label per edge
    # If we partition edges into 2 classes, output should be 2 bitmasks of length n_edges
    # Each bitmask marks which edges belong to that class
    
    gm = GraphManager(G)
    print(f"\nTripartite structure:")
    print(f"V1 (part 0): {[n for n, d in gm.H.nodes(data=True) if d['part'] == 0]}")
    print(f"V2 (part 1): {[n for n, d in gm.H.nodes(data=True) if d['part'] == 1]}")
    print(f"V3 (part 2): {[n for n, d in gm.H.nodes(data=True) if d['part'] == 2]}")
    print(f"\nTripartite edges: {list(gm.H.edges())}")
    
    # The edges in tripartite correspond to:
    # - E12: edges between V1 and V2 (originally from G)
    # - E23: edges between V2 and V3 (originally from G)
    # - E31: edges between V3 and V1 (originally from G)
    
    print(f"\nEdges by type:")
    e12 = [(u, v) for u, v in gm.H.edges() if gm.H.nodes[u]['part'] == 0 and gm.H.nodes[v]['part'] == 1]
    e23 = [(u, v) for u, v in gm.H.edges() if gm.H.nodes[u]['part'] == 1 and gm.H.nodes[v]['part'] == 2]
    e31 = [(u, v) for u, v in gm.H.edges() if gm.H.nodes[u]['part'] == 2 and gm.H.nodes[v]['part'] == 0]
    
    print(f"E12: {e12}")
    print(f"E23: {e23}")
    print(f"E31: {e31}")
    
    print(f"\nSo the algorithm partitions E12 and E23 edges")
    print(f"Output should be two bitmasks, each of length {len(e12)} (for E12 edges)")


if __name__ == "__main__":
    test_link_partition_understanding()
    test_edge_partition_semantics()
    test_assemble_partition_k3_manual()
