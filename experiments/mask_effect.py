#!/usr/bin/env python3
"""
Test to investigate if partition masks affect irregular vertex detection.

This test checks if the mask is actually being used when computing irregular vertices.
"""

import os
import shutil
import networkx as nx
import json
import numpy as np
from manager import Manager, FileManager
from clustering_task import Task

def test_mask_effect_on_irregularity():
    """Test if different masks result in different irregularity patterns."""
    
    print("=" * 80)
    print("Testing Mask Effect on Irregular Vertex Detection")
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
        max_depth=20
    )
    
    # Setup directories
    partition_dir = "temp_debug_mask"
    graph_dir = "temp_graphs_mask"
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    manager.partition_manager = FileManager(partition_dir, graph_dir)
    
    # Manual initialization
    manager.V2 = manager.graph_manager.getV2()
    for v in manager.V2:
        A, B = manager.graph_manager.makeLinkPartition(v)
        mask_A = np.ones(len(A), dtype=bool)
        mask_B = np.ones(len(B), dtype=bool)
        manager.partition_manager.savePartition(v, "", mask_A, mask_B, A, B)
        N = manager.graph_manager.makeLinkGraph(v)
        manager.partition_manager.saveLinkGraph(v, N)
    
    print("\n" + "-" * 80)
    print("Analyzing Irregular Vertices for First V2 Vertex")
    print("-" * 80)
    
    gamma = 336 / 2016  # From previous analysis
    
    # Get first V2 vertex
    v = manager.V2[0]
    
    # Load initial partition
    success_p, partition_str = manager.partition_manager.loadPartition(v, "")
    success_g, link = manager.partition_manager.loadLinkGraph(v)
    
    if success_p and success_g:
        partition_dict = json.loads(partition_str)
        A = manager._convert_json_to_nodes(partition_dict["neighbors_A"])
        B = manager._convert_json_to_nodes(partition_dict["neighbors_B"])
        
        print(f"\nVertex {v}:")
        print(f"  Neighbors in part A: {A}")
        print(f"  Neighbors in part B: {B}")
        print(f"  Link graph has {link.number_of_edges()} edges")
        print(f"  Expected pathweight sum: {len(A) * len(B)}")
        
        # Compute irregular vertices for initial partition
        task_initial = Task(link, (A, B), eps)
        irreg_initial, count_initial = task_initial.compute_irregular_vertices(gamma)
        
        print(f"\n  Initial partition:")
        print(f"    Gamma: {gamma:.6f}")
        print(f"    Irregular vertices: {np.where(irreg_initial)[0]}")
        print(f"    Count: {count_initial}")
        print(f"    Pathweight array: {task_initial.pathweight}")
        print(f"    Deg A: {task_initial.deg_A_v}")
        print(f"    Deg B: {task_initial.deg_B_v}")
        
        # Now create i0 and i1 partitions
        manager.partition_manager.savePartition(v, "i0", np.array(irreg_initial), 
                                              np.ones(len(B), dtype=bool), A, B)
        manager.partition_manager.savePartition(v, "i1", ~np.array(irreg_initial), 
                                              np.ones(len(B), dtype=bool), A, B)
        
        # Load i0 and check
        success_i0, partition_i0_str = manager.partition_manager.loadPartition(v, "i0")
        if success_i0:
            partition_i0_dict = json.loads(partition_i0_str)
            A_i0 = manager._convert_json_to_nodes(partition_i0_dict["neighbors_A"])
            B_i0 = manager._convert_json_to_nodes(partition_i0_dict["neighbors_B"])
            
            print(f"\n  i0 partition (irregular vertices from initial):")
            print(f"    Mask A for neighbors in A: {partition_i0_dict['mask_A']}")
            print(f"    A nodes selected: {np.sum(partition_i0_dict['mask_A'])}")
            print(f"    B nodes selected: {np.sum(partition_i0_dict['mask_B'])}")
            print(f"    New A size: {len(A_i0)}")
            print(f"    New B size: {len(B_i0)}")
            
            task_i0 = Task(link, (A_i0, B_i0), eps)
            irreg_i0, count_i0 = task_i0.compute_irregular_vertices(gamma)
            
            print(f"    Irregular vertices: {np.where(irreg_i0)[0]}")
            print(f"    Count: {count_i0}")
            print(f"    Pathweight array: {task_i0.pathweight}")
            print(f"    Deg A: {task_i0.deg_A_v}")
            print(f"    Deg B: {task_i0.deg_B_v}")
        
        # Load i1 and check
        success_i1, partition_i1_str = manager.partition_manager.loadPartition(v, "i1")
        if success_i1:
            partition_i1_dict = json.loads(partition_i1_str)
            A_i1 = manager._convert_json_to_nodes(partition_i1_dict["neighbors_A"])
            B_i1 = manager._convert_json_to_nodes(partition_i1_dict["neighbors_B"])
            
            print(f"\n  i1 partition (non-irregular vertices from initial):")
            print(f"    Mask A for neighbors in A: {partition_i1_dict['mask_A']}")
            print(f"    A nodes selected: {np.sum(partition_i1_dict['mask_A'])}")
            print(f"    B nodes selected: {np.sum(partition_i1_dict['mask_B'])}")
            print(f"    New A size: {len(A_i1)}")
            print(f"    New B size: {len(B_i1)}")
            
            task_i1 = Task(link, (A_i1, B_i1), eps)
            irreg_i1, count_i1 = task_i1.compute_irregular_vertices(gamma)
            
            print(f"    Irregular vertices: {np.where(irreg_i1)[0]}")
            print(f"    Count: {count_i1}")
            print(f"    Pathweight array: {task_i1.pathweight}")
            print(f"    Deg A: {task_i1.deg_A_v}")
            print(f"    Deg B: {task_i1.deg_B_v}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("If masks are working correctly:")
    print("  1. The sizes of A and B should differ between partitions")
    print("  2. The pathweight arrays should be different")
    print("  3. The irregular vertex counts should differ")
    print("\nIf the masks are NOT working (or don't matter):")
    print("  1. A and B sizes will be unchanged")
    print("  2. Pathweight arrays will be identical")
    print("  3. Irregular vertex counts will be identical")
    
    # Cleanup
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    test_mask_effect_on_irregularity()
