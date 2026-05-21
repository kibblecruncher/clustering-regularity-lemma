#!/usr/bin/env python3
"""
Test the correctness of the manager class on complete graphs of varying sizes.

Records:
- List of directions considered by the algorithm
- Time taken for each run
- Number of files created (partitions and link graphs)
"""

import os
import shutil
import time
import networkx as nx
import json
from pathlib import Path
from manager import Manager, FileManager

def count_files_in_directory(directory):
    """Count the number of files in a directory."""
    if not os.path.exists(directory):
        return 0
    return sum(1 for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)))

def setup_test_directories(partition_dir="partitions_test", graph_dir="graphs_test"):
    """Clean and create test directories."""
    # Remove existing directories
    for directory in [partition_dir, graph_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    return partition_dir, graph_dir

def cleanup_test_directories(partition_dir="partitions_test", graph_dir="graphs_test"):
    """Clean up test directories."""
    for directory in [partition_dir, graph_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)

def test_manager_on_complete_graph(graph_size, partition_dir, graph_dir, max_depth=10):
    """
    Test the manager on a complete graph of the given size.
    
    Args:
        graph_size: Size of the complete graph
        partition_dir: Directory for partitions
        graph_dir: Directory for graphs
        max_depth: Maximum depth for the algorithm
    
    Returns:
        dict: Contains directions, time taken, and number of files created
    """
    print(f"\n{'='*70}")
    print(f"Testing on Complete Graph K{graph_size}")
    print(f"{'='*70}")
    
    # Create complete graph
    G = nx.complete_graph(graph_size)
    print(f"Created complete graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Set up hyperparameters
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
        print(f"Full Traceback:\n{traceback_str}")
        partition_labels_A = None
        partition_labels_B = None
    
    # Count files created (the manager should have cleaned them up)
    files_after = count_files_in_directory(partition_dir) + count_files_in_directory(graph_dir)
    
    # Get directions considered
    directions = manager.directions_considered if hasattr(manager, 'directions_considered') else []
    
    # Compile results
    result = {
        'graph_size': graph_size,
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'directions_considered': directions,
        'num_directions': len(directions),
        'time_taken_seconds': elapsed_time,
        'files_remaining_after_cleanup': files_after,
        'success': success,
        'error': error_msg,
        'num_V2_vertices': len(manager.V2) if hasattr(manager, 'V2') else None,
        'max_depth': max_depth,
    }
    
    return result

def print_results(results):
    """Print formatted results of the test runs."""
    print(f"\n{'='*70}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Test {i}: Complete Graph K{result['graph_size']} ---")
        print(f"Graph Properties:")
        print(f"  Nodes: {result['graph_nodes']}")
        print(f"  Edges: {result['graph_edges']}")
        print(f"  V2 Vertices: {result['num_V2_vertices']}")
        
        if result['success']:
            print(f"\nAlgorithm Results:")
            print(f"  Directions Considered: {result['num_directions']}")
            print(f"  Direction List: {result['directions_considered']}")
            print(f"  Time Taken: {result['time_taken_seconds']:.4f} seconds")
            print(f"  Files Remaining After Cleanup: {result['files_remaining_after_cleanup']}")
        else:
            print(f"\nERROR: {result['error']}")
            print(f"  Time Before Error: {result['time_taken_seconds']:.4f} seconds")
    
    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Graph Size':<15} {'Directions':<15} {'Time (s)':<15} {'Files Remaining':<15}")
    print("-" * 60)
    for result in results:
        if result['success']:
            print(f"K{result['graph_size']:<14} {result['num_directions']:<15} {result['time_taken_seconds']:<15.4f} {result['files_remaining_after_cleanup']:<15}")
        else:
            print(f"K{result['graph_size']:<14} {'ERROR':<15} {result['time_taken_seconds']:<15.4f} {'N/A':<15}")

def main():
    """Run tests on complete graphs of different sizes."""
    print("Manager Correctness Testing")
    print("Testing on Complete Graphs")
    
    # Graph sizes to test with their max_depth parameters
    test_configs = [
        {'size': 8, 'max_depth': 10},    # K8 may need more depth
    ]
    
    results = []
    
    for config in test_configs:
        size = config['size']
        max_depth = config['max_depth']
        
        # Set up test directories
        partition_dir, graph_dir = setup_test_directories()
        
        try:
            result = test_manager_on_complete_graph(size, partition_dir, graph_dir, max_depth=max_depth)
            results.append(result)
        except Exception as e:
            print(f"Error during test: {e}")
            results.append({
                'graph_size': size,
                'success': False,
                'error': str(e),
                'time_taken_seconds': 0,
                'directions_considered': [],
                'max_depth': max_depth,
            })
        finally:
            # Clean up
            cleanup_test_directories(partition_dir, graph_dir)
    
    # Print results
    print_results(results)
    
    # Save results to JSON file
    json_results = []
    for r in results:
        json_results.append({
            'graph_size': r['graph_size'],
            'graph_nodes': r.get('graph_nodes'),
            'graph_edges': r.get('graph_edges'),
            'num_V2_vertices': r.get('num_V2_vertices'),
            'directions_considered': r['directions_considered'],
            'num_directions': r['num_directions'],
            'time_taken_seconds': r['time_taken_seconds'],
            'files_remaining_after_cleanup': r.get('files_remaining_after_cleanup'),
            'success': r['success'],
            'error': r['error'],
            'max_depth': r.get('max_depth'),
        })
    
    with open('test_manager_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Results saved to 'test_manager_results.json'")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
