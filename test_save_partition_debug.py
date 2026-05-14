#!/usr/bin/env python3
"""Debug script to test partition saving."""

import networkx as nx
import numpy as np
import json
from manager import Manager, GraphManager, FileManager

# Create a simple complete graph
G = nx.complete_graph(3)

# Create managers
graph_manager = GraphManager(G)
partition_manager = FileManager("partitions_debug", "graphs_debug")

# Test makeLinkPartition
V2 = graph_manager.getV2()
print(f"V2: {V2}")

v = V2[0]
A, B = graph_manager.makeLinkPartition(v)

print(f"\nA type: {type(A)}")
if hasattr(A, 'dtype'):
    print(f"A dtype: {A.dtype}")
print(f"A: {A}")
print(f"A[0] type: {type(A[0])}")

print(f"\nB type: {type(B)}")
if hasattr(B, 'dtype'):
    print(f"B dtype: {B.dtype}")
print(f"B: {B}")
print(f"B[0] type: {type(B[0])}")

# Test savePartition
mask_A = np.ones(len(A), dtype=bool)
mask_B = np.ones(len(B), dtype=bool)

print(f"\nmask_A type: {type(mask_A)}")
print(f"mask_A: {mask_A}")

try:
    print("\nAttempting to save partition...")
    partition_manager.savePartition(v, "test", mask_A, mask_B, A, B)
    print("SUCCESS: Partition saved")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Try loading it
try:
    print("\nAttempting to load partition...")
    success, partition_str = partition_manager.loadPartition(v, "test")
    if success:
        print("SUCCESS: Partition loaded")
        partition_dict = json.loads(partition_str)
        print(f"Loaded data keys: {partition_dict.keys()}")
        print(f"mask_A: {partition_dict['mask_A']}")
        print(f"neighbors_A: {partition_dict['neighbors_A']}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
