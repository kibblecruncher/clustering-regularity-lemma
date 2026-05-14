#!/usr/bin/env python3
"""
Focused debugging test for the manager class on complete graphs.

Records for each generated partition:
- The direction code
- The clustering coefficient (gamma) and whether it passed the gamma check
- The irregularity weight and threshold
- The deviation weight and threshold
- The failure reason (or success)
"""

import os
import shutil
import time
import networkx as nx
import json
import numpy as np
from pathlib import Path
from manager import Manager, FileManager

def setup_test_directories(partition_dir="partitions_debug", graph_dir="graphs_debug"):
    """Clean and create test directories."""
    # Remove existing directories
    for directory in [partition_dir, graph_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    return partition_dir, graph_dir

def cleanup_test_directories(partition_dir="partitions_debug", graph_dir="graphs_debug"):
    """Clean up test directories."""
    for directory in [partition_dir, graph_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def test_manager_debug(graph_size, partition_dir, graph_dir, max_depth=10):
    """
    Test the manager on a complete graph with detailed logging.
    
    Args:
        graph_size: Size of the complete graph
        partition_dir: Directory for partitions
        graph_dir: Directory for graphs
        max_depth: Maximum depth for the algorithm
    
    Returns:
        dict: Contains detailed partition logs
    """
    print(f"\n{'='*80}")
    print(f"DEBUGGING COMPLETE GRAPH K{graph_size}")
    print(f"{'='*80}")
    
    # Create complete graph
    G = nx.complete_graph(graph_size)
    print(f"Graph Properties:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    
    # For complete graphs, check theoretical clustering coefficient
    # In a complete graph, every pair of neighbors of a vertex is connected
    # So the clustering coefficient of each vertex should be 1.0
    print(f"  Local Clustering Coefficients: {[nx.clustering(G, node) for node in G.nodes()][:5]}...")
    print(f"  Average Clustering Coefficient: {nx.average_clustering(G):.4f}")
    
    # Set up hyperparameters
    eps = 0.1
    print(f"\nHyperparameters:")
    print(f"  eps: {eps}")
    print(f"  clustering_threshold: {eps}")
    print(f"  irreg_threshold: {2 * eps**(5/2) / 5:.6f}")
    print(f"  dev_threshold: {2 * eps**2 / 5:.6f}")
    print(f"  max_depth: {max_depth}")
    
    manager = Manager(
        G=G,
        eps=eps,
        irreg_vtx_threshold=eps**5 / 90,
        dev_vtx_threshold=eps**2 / 9,
        irreg_vtx_count_threshold=eps**(5/2) / 9,
        dev_threshold=2 * eps**2 / 5,
        irreg_threshold=2 * eps**(5/2) / 5,
        clustering_threshold=eps,
        max_depth=max_depth
    )
    
    # Override the partition and graph directories in the manager
    manager.partition_manager = FileManager(partition_dir, graph_dir)
    
    # Start timer
    start_time = time.time()
    
    # Run the manager
    try:
        partition_labels_A, partition_labels_B = manager.run()
        elapsed_time = time.time() - start_time
        success = True
        error_msg = None
    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        success = False
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"\nERROR: {error_msg}")
        print(f"Traceback:\n{traceback_str}")
        partition_labels_A = None
        partition_labels_B = None
    
    # Print detailed partition logs
    print(f"\n{'='*80}")
    print("PARTITION ITERATION LOG")
    print(f"{'='*80}")
    
    if hasattr(manager, 'partition_logs'):
        logs = manager.partition_logs
        print(f"\nTotal partitions processed: {len(logs)}\n")
        
        # Print header
        print(f"{'Dir':<10} {'Gamma':<12} {'Status':<25} {'Irreg Weight':<15} {'Dev Weight':<15}")
        print("-" * 80)
        
        for log in logs:
            direction = log['direction']
            gamma = log['gamma']
            
            # Determine status
            if log['failure_reason'] == 'PASSED_GAMMA_CHECK':
                status = "PASS (gamma<thr)"
            elif log['failure_reason'] == 'FAILED_IRREGULARITY_CHECK':
                status = "FAIL IRREGULARITY"
            elif log['failure_reason'] == 'FAILED_DEVIATION_CHECK':
                status = "FAIL DEVIATION"
            elif log['failure_reason'] == 'PASSED_ALL_CHECKS':
                status = "PASS (all checks)"
            else:
                status = "UNKNOWN"
            
            irreg_str = f"{log['irreg_weight']:.2e}" if log['irreg_weight'] is not None else "N/A"
            dev_str = f"{log['dev_weight']:.2e}" if log['dev_weight'] is not None else "N/A"
            
            print(f"{direction:<10} {gamma:<12.6f} {status:<25} {irreg_str:<15} {dev_str:<15}")
        
        # Print detailed analysis for each partition
        print(f"\n{'='*80}")
        print("DETAILED PARTITION ANALYSIS")
        print(f"{'='*80}\n")
        
        for i, log in enumerate(logs, 1):
            print(f"Partition {i}: Direction '{log['direction']}'")
            print(f"  Path weight: {log['pathweight']}")
            print(f"  Triangle count: {log['triangle_count']}")
            print(f"  Gamma (clustering coeff): {log['gamma']:.6f}")
            print(f"  Clustering threshold: {log['clustering_threshold']:.6f}")
            
            if log['failure_reason'] == 'PASSED_GAMMA_CHECK':
                print(f"  Status: ✓ PASSED gamma check (gamma < threshold)")
            else:
                print(f"  Irregularity weight: {log['irreg_weight']:.6e}")
                print(f"  Irregularity threshold: {log['irreg_threshold']:.6e}")
                print(f"  Irregularity check: {log['irreg_weight']:.6e} > {log['irreg_threshold']:.6e} ? {log['irreg_weight'] > log['irreg_threshold']}")
                
                print(f"  Deviation weight: {log['dev_weight']:.6e}")
                print(f"  Deviation threshold: {log['dev_threshold']:.6e}")
                print(f"  Deviation check: {log['dev_weight']:.6e} > {log['dev_threshold']:.6e} ? {log['dev_weight'] > log['dev_threshold']}")
                
                print(f"  Status: {log['failure_reason']}")
            print()
    else:
        print("No partition logs found in manager!")
    
    # Compile results
    result = {
        'graph_size': graph_size,
        'success': success,
        'error': error_msg,
        'time_taken': elapsed_time,
        'num_partitions_processed': len(logs) if hasattr(manager, 'partition_logs') else 0,
        'partition_logs': logs if hasattr(manager, 'partition_logs') else [],
        'directions_considered': manager.directions_considered if hasattr(manager, 'directions_considered') else [],
    }
    
    return result

def main():
    """Run debugging test on K8."""
    print("Manager Debugging Test")
    print("=" * 80)
    
    # Test K8 with depth 10
    partition_dir, graph_dir = setup_test_directories()
    
    try:
        result = test_manager_debug(8, partition_dir, graph_dir, max_depth=10)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Graph Size: K{result['graph_size']}")
        print(f"Success: {result['success']}")
        print(f"Time Taken: {result['time_taken']:.4f} seconds")
        print(f"Partitions Processed: {result['num_partitions_processed']}")
        if result['error']:
            print(f"Error: {result['error']}")
        
        # Count failures by type
        failure_counts = {}
        for log in result['partition_logs']:
            reason = log['failure_reason']
            if reason not in failure_counts:
                failure_counts[reason] = 0
            failure_counts[reason] += 1
        
        print(f"\nFailure Breakdown:")
        for reason, count in sorted(failure_counts.items()):
            print(f"  {reason}: {count}")
        
        # Check gamma values
        gamma_values = [log['gamma'] for log in result['partition_logs']]
        if gamma_values:
            print(f"\nGamma Statistics:")
            print(f"  Min: {min(gamma_values):.6f}")
            print(f"  Max: {max(gamma_values):.6f}")
            print(f"  Mean: {np.mean(gamma_values):.6f}")
            print(f"  Number of partitions with gamma=0: {sum(1 for g in gamma_values if g == 0.0)}")
            print(f"  Number of partitions with gamma>0: {sum(1 for g in gamma_values if g > 0.0)}")
            
            # Check if any gamma values are exactly 0
            if any(g == 0.0 for g in gamma_values):
                print(f"\n⚠️  WARNING: Found partitions with gamma=0")
                for log in result['partition_logs']:
                    if log['gamma'] == 0.0:
                        print(f"    Direction '{log['direction']}': gamma=0, triangle_count={log['triangle_count']}, pathweight={log['pathweight']}")
    
    except Exception as e:
        import traceback
        print(f"Test failed with error: {e}")
        print(traceback.format_exc())
    
    finally:
        cleanup_test_directories(partition_dir, graph_dir)

if __name__ == '__main__':
    main()
