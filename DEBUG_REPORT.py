#!/usr/bin/env python3
"""
Debugging Report: K8 Infinite Loop Analysis

KEY FINDINGS:
=============

1. GAMMA (Clustering Coefficient) VALUES:
   - Gamma = 0.166667 (exactly 1/6) for ALL partitions
   - Gamma is NOT zero for complete graphs ✓ (as expected)
   - Gamma 0.166667 > clustering_threshold 0.1 → Algorithm doesn't pass early termination
   - This makes sense: triangle_count/pathweight = 336/2016 = 1/6

2. PARTITION BEHAVIOR:
   - ALL 63 partitions generated fail the IRREGULARITY CHECK
   - ZERO partitions fail the deviation check
   - ZERO partitions pass all checks
   - The irregularity weight is MASSIVE: ~2016 >> threshold of ~2.55

3. INFINITE LOOP PROBLEM:
   - Algorithm exceeds max_depth=10 at direction code 'i0i0i0i0i0i0' (length 12)
   - Every partition is subdivided via irregularity splitting (dir → dir+'i0', dir+'i1')
   - The irregularity weight never decreases despite subdivision
   - The algorithm just keeps subdividing until depth limit is hit

4. CRITICAL OBSERVATION - ZERO DEVIATION:
   - Deviation weight is 0.0 for all partitions
   - This means the deviation check is never triggered
   - The algorithm only ever uses irregularity-based splitting

HYPOTHESIS:
===========
The complete graph K8 with epsilon=0.1 and these hyperparameters may be:
1. Genuinely too "irregular" to partition below the irregularity threshold
2. The irregularity splitting strategy isn't actually reducing irregularity
3. The threshold epsilon=0.1 may be too strict for small complete graphs

NEXT STEPS FOR INVESTIGATION:
=============================
1. Check if irregularity weight actually decreases after splitting
2. Examine compute_irregular_vertices() output in detail
3. Check if the threshold values are appropriate for complete graphs
4. Try with larger epsilon or different hyperparameters
5. Verify the irregularity weight calculation is correct
"""

# Create a detailed test script to investigate irregularity computation
test_code = '''
import networkx as nx
import numpy as np
from clustering_task import Task
from manager import Manager, FileManager
import json
import os
import shutil

def test_irregularity_details():
    """Detailed test of irregularity computation."""
    
    # Create K8 and setup manager
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
        max_depth=10
    )
    
    # Setup temp directories
    partition_dir = "temp_debug"
    graph_dir = "temp_graphs"
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
    
    manager.partition_manager = FileManager(partition_dir, graph_dir)
    
    # Initialize manager (this sets up initial partitions)
    manager.initialize()
    
    print("=" * 80)
    print("DETAILED IRREGULARITY ANALYSIS")
    print("=" * 80)
    
    # Load initial partition and check irregularities
    print("\\nInitial Partition Analysis:")
    print("-" * 80)
    
    for i, v in enumerate(manager.V2[:2]):  # Just check first 2 vertices for brevity
        success, link = manager.partition_manager.loadLinkGraph(v)
        success_p, partition_str = manager.partition_manager.loadPartition(v, "")
        
        if success and success_p:
            partition_dict = json.loads(partition_str)
            A = manager._convert_json_to_nodes(partition_dict["neighbors_A"])
            B = manager._convert_json_to_nodes(partition_dict["neighbors_B"])
            
            task = Task(link, (A, B), eps)
            gamma = 336 / 2016  # We know this from the test
            
            print(f"\\nVertex {v} (v in V2[{i}]):")
            print(f"  Link graph edges: {task.edges}")
            print(f"  Part A size: {len(A)}")
            print(f"  Part B size: {len(B)}")
            print(f"  Pathweight array shape: {task.pathweight.shape}")
            print(f"  Pathweight sum: {np.sum(task.pathweight)}")
            print(f"  Deg A values: {task.deg_A_v}")
            print(f"  Deg B values: {task.deg_B_v}")
            print(f"  Gamma: {gamma:.6f}")
            
            # Compute irregular vertices
            irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
            print(f"  Irregular vertices count: {irreg_count}")
            print(f"  Irregular vertices mask: {irreg_v}")
            print(f"  Threshold for count: {manager.irreg_vtx_count_threshold * 2016}")
            print(f"  Count exceeds threshold: {irreg_count > manager.irreg_vtx_count_threshold * 2016}")
            
            # If irregular, compute the weight
            if irreg_count > manager.irreg_vtx_count_threshold * 2016:
                irreg_weight_v = np.sum(irreg_v * task.pathweight)
                print(f"  Irregularity weight contribution: {irreg_weight_v:.2e}")
    
    # Cleanup
    for d in [partition_dir, graph_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)

if __name__ == "__main__":
    test_irregularity_details()
'''

print(__doc__)
print("\n" + "=" * 80)
print("To investigate further, run this test code:")
print("=" * 80)
print(test_code)
