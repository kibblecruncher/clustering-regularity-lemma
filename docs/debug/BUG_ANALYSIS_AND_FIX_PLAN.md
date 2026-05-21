#!/usr/bin/env python3
"""
COMPREHENSIVE BUG ANALYSIS AND FIX PLAN FOR assemble_partition

The fundamental issue:
- Partitions being saved store node identifiers (A and B neighbors)
- But they should store BITMASKS that indicate which neighbors are in each partition class
- AND we need the neighbor identifiers to map edges back to the global graph

The storage structure needs to be:
{
    "neighbors_A": [(1,0), (2,0), ...],  # Which nodes are the A neighbors
    "neighbors_B": [(1,2), (2,2), ...],  # Which nodes are the B neighbors
    "mask_A": [True, False, True, ...],   # Which of those are in partition class 0
    "mask_B": [True, True, False, ...]    # Which of those are in partition class 0
}

Then assemble_partition needs to:
1. For each vertex v in V2
2. Load this expanded partition structure
3. For each marked neighbor, create the corresponding edge partition
4. Return global edge bitmasks over E12 and E23
"""

import networkx as nx
import numpy as np
import json
import os
from typing import List, Dict, Tuple


class FixedPartitionStorage:
    """Demonstrate the fixed partition storage structure."""
    
    @staticmethod
    def save_partition_fixed(filepath: str,
                             vertex: int,
                             direction: str,
                             neighbors_A: List,
                             neighbors_B: List,
                             mask_A: np.ndarray,
                             mask_B: np.ndarray):
        """
        Save partition with both neighbor identifiers AND bitmasks.
        """
        # Convert tuples to lists for JSON serialization
        neighbors_A_list = [list(n) if isinstance(n, tuple) else n for n in neighbors_A]
        neighbors_B_list = [list(n) if isinstance(n, tuple) else n for n in neighbors_B]
        
        data = {
            "vtx": vertex,
            "dir": direction,
            "neighbors_A": neighbors_A_list,
            "neighbors_B": neighbors_B_list,
            "mask_A": mask_A.tolist(),
            "mask_B": mask_B.tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    @staticmethod
    def load_partition_fixed(filepath: str) -> Dict:
        """
        Load partition with neighbor identifiers and bitmasks.
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert neighbor lists back to tuples
        neighbors_A = [tuple(n) if isinstance(n, list) else n for n in data["neighbors_A"]]
        neighbors_B = [tuple(n) if isinstance(n, list) else n for n in data["neighbors_B"]]
        
        return {
            "neighbors_A": neighbors_A,
            "neighbors_B": neighbors_B,
            "mask_A": np.array(data["mask_A"], dtype=bool),
            "mask_B": np.array(data["mask_B"], dtype=bool)
        }


class FixedAssemblePartition:
    """
    Demonstrate how assemble_partition should work with fixed storage.
    """
    
    def __init__(self, H: nx.Graph, V2: List[int]):
        self.H = H  # Tripartite graph
        self.V2 = V2
        self._build_edge_maps()
    
    def _build_edge_maps(self):
        """Build mappings of edges for quick lookup."""
        self.E12_list = []
        self.E12_set = set()
        self.E23_list = []
        self.E23_set = set()
        
        for u, v in self.H.edges():
            u_part = self.H.nodes[u]['part']
            v_part = self.H.nodes[v]['part']
            
            # Normalize edge direction
            if u_part > v_part:
                u, v, u_part, v_part = v, u, v_part, u_part
            
            if u_part == 0 and v_part == 1:
                self.E12_list.append((u, v))
                self.E12_set.add((u, v))
            elif u_part == 1 and v_part == 2:
                self.E23_list.append((u, v))
                self.E23_set.add((u, v))
    
    def assemble_partition_fixed(self, partition_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert local vertex partition to global edge partition.
        
        Args:
            partition_dict: {"neighbors_A": [...], "neighbors_B": [...], 
                           "mask_A": [...], "mask_B": [...]}
        
        Returns:
            (E12_partition, E23_partition) - global boolean arrays
        """
        neighbors_A = partition_dict["neighbors_A"]
        neighbors_B = partition_dict["neighbors_B"]
        mask_A = partition_dict["mask_A"]
        mask_B = partition_dict["mask_B"]
        
        # Initialize global partitions
        E12_partition = np.zeros(len(self.E12_list), dtype=bool)
        E23_partition = np.zeros(len(self.E23_list), dtype=bool)
        
        # For E12: edges from part 0 (V1) to part 1 (V2)
        for idx, neighbor in enumerate(neighbors_A):
            if mask_A[idx]:
                # This neighbor is marked - add corresponding E12 edge
                edge = (neighbor, (self.V2[0], 1))  # This needs the actual vertex!
                # Need to search in E12_list
                if edge in self.E12_set:
                    e_idx = self.E12_list.index(edge)
                    E12_partition[e_idx] = True
        
        # For E23: edges from part 1 (V2) to part 2 (V3)  
        for idx, neighbor in enumerate(neighbors_B):
            if mask_B[idx]:
                # This neighbor is marked - add corresponding E23 edge
                edge = ((self.V2[0], 1), neighbor)  # Again, need actual vertex
                if edge in self.E23_set:
                    e_idx = self.E23_list.index(edge)
                    E23_partition[e_idx] = True
        
        return E12_partition, E23_partition


print("""
COMPREHENSIVE FIX PLAN
======================

Problem 1: Partition Storage
----------------------------
Current code saves:
    partition_manager.savePartition(v, dir, A, B)
    
Where A and B are just the neighbor node lists.

Fix: Modify savePartition to accept and store bitmasks
    partition_manager.savePartition(v, dir, neighbors_A, neighbors_B, mask_A, mask_B)
    
OR create a separate method for saving bitmask partitions.

Problem 2: Data Flow Through Algorithm
---------------------------------------
The issue is that the iterative algorithm computes bitmasks at each step:
    
    irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
    # irreg_v is a BITMASK indicating which neighbors are irregular
    
But currently it's saved as:
    partition_manager.savePartition(v, dir + "i0", irreg_v, task2.B)
    
This is mixing bitmask with neighbor identifiers!

The partition file should contain:
- The neighbor identifiers (for mapping to edges)
- The bitmask (which neighbors are in which partition class)

Problem 3: assemble_partition Logic
------------------------------------
Currently:
    
    for v in self.V2:
        success, partition_str = partition_manager.loadPartition(v, dir)
        partition_dict = json.loads(partition_str)
        mask_A = np.array(partition_dict["mask_A"], dtype=bool)
        mask_B = np.array(partition_dict["mask_B"], dtype=bool)
        neighbors_A = partition_dict["neighbors_A"]
        neighbors_B = partition_dict["neighbors_B"]
        part_A_iter.append(neighbors_A)
        part_B_iter.append(neighbors_B)
    
    part_A = np.concatenate(part_A_iter)  # Concatenate node lists!
    part_B = np.concatenate(part_B_iter)
    return part_A, part_B

This is fundamentally wrong because:
1. It returns node identifiers, not edge bitmasks
2. The returned arrays have inconsistent semantics
3. They can't be used as partition labels by partitionLabels()

Should be:

    E12_partition = np.zeros(total_E12_edges, dtype=bool)
    E23_partition = np.zeros(total_E23_edges, dtype=bool)
    
    for v in self.V2:
        partition_dict = load_partition(v, dir)
        
        # For this vertex's partition:
        # - Find all E12 edges incident to (v,1) 
        # - For each marked neighbor in mask_A, mark the corresponding edge
        
        # - Find all E23 edges incident to (v,1)
        # - For each marked neighbor in mask_B, mark the corresponding edge
    
    return E12_partition, E23_partition

IMPLEMENTATION STEPS
====================

1. Modify FileManager.savePartition() to also accept bitmasks
2. Modify partition-saving calls in iterate() to save bitmasks
3. Create a new method get_edge_lists() to return all E12 and E23 edges
4. Rewrite assemble_partition() to:
   - Get all edge lists
   - For each vertex, map its bitmask to edge contributions
   - Return global edge partitions
5. Update partitionLabels() if needed (may need to handle different semantics)
""")
