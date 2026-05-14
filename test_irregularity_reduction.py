#!/usr/bin/env python3
"""
Test to investigate whether irregularity splitting actually reduces irregularity.

This test compares the irregularity weight before and after splitting.
"""

import os
import shutil
import networkx as nx
import json
import numpy as np
from manager import Manager, FileManager
from clustering_task import Task

def test_irregularity_reduction():
    """Test if irregularity splitting reduces irregularity weight."""
    
    print("=" * 80)
    print("Testing Irregularity Reduction After Splitting")
    print("=" * 80)
    
    # Create K8
    G = nx.complete_graph(8)
    eps = 0.1
    
    manager = Manager(
        G=G,
        eps=eps,
        irreg_vtx_threshold=eps**5 / 90,
        dev_vtx_threshold=eps**2 / 9,
        irreg_vtx_count_threshold=eps**(5/2) / 9,
        dev_threshold=2 * eps**2 / 5,
        irreg_threshold=2 * eps**(5/2) / 5,
        clustering_threshold=eps,
        max_depth=20  # Higher depth for this test
    )
    
    # Setup directories
    partition_dir = "temp_debug"
    graph_dir = "temp_graphs"
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    manager.partition_manager = FileManager(partition_dir, graph_dir)
    
    # Manual initialization (copied from run method)
    manager.V2 = manager.graph_manager.getV2()
    for v in manager.V2:
        A, B = manager.graph_manager.makeLinkPartition(v)
        # Initial partitions: mark all neighbors as in partition class 0
        mask_A = np.ones(len(A), dtype=bool)
        mask_B = np.ones(len(B), dtype=bool)
        manager.partition_manager.savePartition(v, "", mask_A, mask_B, A, B)
        N = manager.graph_manager.makeLinkGraph(v)
        manager.partition_manager.saveLinkGraph(v, N)
    
    # Function to compute irregularity for a direction
    def compute_irregularity_for_direction(direction):
        """Compute total irregularity weight for a direction."""
        gamma = 336 / 2016  # We know this from previous analysis
        irreg_threshold_vtx = manager.irreg_vtx_count_threshold * 2016  # pathweight
        
        total_irreg = 0.0
        irreg_vertices_count = 0
        
        for v in manager.V2:
            success_g, link = manager.partition_manager.loadLinkGraph(v)
            success_p, partition_str = manager.partition_manager.loadPartition(v, direction)
            
            if not success_g or not success_p:
                continue
            
            partition_dict = json.loads(partition_str)
            A = manager._convert_json_to_nodes(partition_dict["neighbors_A"])
            B = manager._convert_json_to_nodes(partition_dict["neighbors_B"])
            
            task = Task(link, (A, B), eps)
            irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
            
            if irreg_count > irreg_threshold_vtx:
                irreg_weight_v = np.sum(irreg_v * task.pathweight)
                total_irreg += irreg_weight_v
                irreg_vertices_count += 1
        
        return total_irreg, irreg_vertices_count
    
    print("\n" + "-" * 80)
    print("Comparing Parent and Child Partition Irregularities")
    print("-" * 80)
    print(f"{'Direction':<20} {'Irregularity':<20} {'Irreg Vertices':<20}")
    print("-" * 80)
    
    # Compute for initial partition
    irreg_initial, count_initial = compute_irregularity_for_direction("")
    print(f"{'(initial)':<20} {irreg_initial:<20.2e} {count_initial:<20}")
    
    # Now manually create child partitions and test
    # According to the algorithm, when irregularity is high, we split each vertex's
    # partition into two parts based on irregular vertices
    
    print("\nCreating 'i0' and 'i1' child partitions from initial...")
    
    # Compute irregular vertices and create splits
    gamma = 336 / 2016
    for v in manager.V2:
        success_g, link = manager.partition_manager.loadLinkGraph(v)
        success_p, partition_str = manager.partition_manager.loadPartition(v, "")
        
        if success_g and success_p:
            partition_dict = json.loads(partition_str)
            A = manager._convert_json_to_nodes(partition_dict["neighbors_A"])
            B = manager._convert_json_to_nodes(partition_dict["neighbors_B"])
            
            task = Task(link, (A, B), eps)
            irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
            
            # Create the i0 and i1 partitions
            manager.partition_manager.savePartition(v, "i0", np.array(irreg_v), np.ones(len(B), dtype=bool), A, B)
            manager.partition_manager.savePartition(v, "i1", ~np.array(irreg_v), np.ones(len(B), dtype=bool), A, B)
    
    # Now check irregularity of i0 and i1
    irreg_i0, count_i0 = compute_irregularity_for_direction("i0")
    irreg_i1, count_i1 = compute_irregularity_for_direction("i1")
    
    print(f"{'i0':<20} {irreg_i0:<20.2e} {count_i0:<20}")
    print(f"{'i1':<20} {irreg_i1:<20.2e} {count_i1:<20}")
    
    print("\n" + "-" * 80)
    print("ANALYSIS")
    print("-" * 80)
    print(f"Initial irregularity: {irreg_initial:.2e}")
    print(f"i0 irregularity: {irreg_i0:.2e}")
    print(f"i1 irregularity: {irreg_i1:.2e}")
    print(f"Average child irregularity: {(irreg_i0 + irreg_i1) / 2:.2e}")
    
    if irreg_i0 < irreg_initial and irreg_i1 < irreg_initial:
        print("\n✓ Splitting DOES reduce irregularity")
    else:
        print("\n✗ Splitting DOES NOT reduce irregularity!")
        print("  This explains the infinite loop - the algorithm keeps splitting")
        print("  but the irregularity never gets small enough to pass the threshold.")
    
    # Cleanup
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    test_irregularity_reduction()
