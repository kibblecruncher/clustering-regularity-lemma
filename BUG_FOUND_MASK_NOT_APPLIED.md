#!/usr/bin/env python3
"""
BUG ANALYSIS AND FIX

================================================================================
THE BUG: Partition Masks Are Ignored When Loading
================================================================================

LOCATION: manager.py, iterate() method, around line 490

CURRENT CODE (BUGGY):
    partition_dict = json.loads(partition)
    A = self._convert_json_to_nodes(partition_dict["neighbors_A"])  # FULL list!
    B = self._convert_json_to_nodes(partition_dict["neighbors_B"])  # FULL list!
    task = Task(link, (A, B), self.eps)

PROBLEM:
    The partition file contains:
    - "mask_A": [1, 1, 1, 1, 1, 1, 1] or [0, 0, 0, 0, 0, 0, 0] (the bitmask)
    - "neighbors_A": [list of ALL neighbors] (the full list)
    - "mask_B": [1, 1, 1, 1, 1, 1, 1] or [1, 1, 1, 1, 1, 1, 1] (the bitmask)
    - "neighbors_B": [list of ALL neighbors] (the full list)
    
    But the code IGNORES the masks and uses the full lists!
    
CONSEQUENCE:
    Every partition has the same Task object constructed with the same (A, B) sets.
    The masks have NO EFFECT on the computation.
    
    → compute_irregular_vertices() gets the same input every time
    → pathweight is the same every time
    → irregularity is the same every time
    → splitting doesn't reduce irregularity
    → infinite loop until max_depth exceeded

================================================================================
THE FIX
================================================================================

When loading a partition, apply the mask to select only the masked neighbors:

    partition_dict = json.loads(partition)
    mask_A = np.array(partition_dict["mask_A"], dtype=bool)
    mask_B = np.array(partition_dict["mask_B"], dtype=bool)
    
    neighbors_A_full = self._convert_json_to_nodes(partition_dict["neighbors_A"])
    neighbors_B_full = self._convert_json_to_nodes(partition_dict["neighbors_B"])
    
    # Apply masks to select subset of neighbors
    A = np.array(neighbors_A_full)[mask_A].tolist()
    B = np.array(neighbors_B_full)[mask_B].tolist()
    
    task = Task(link, (A, B), self.eps)

This needs to be applied in:
1. iterate() method - line 490-491
2. compute_path_data() method - line 414-415
3. Any other place where partitions are loaded

================================================================================
VERIFICATION: Expected Results After Fix
================================================================================

When masks are properly applied:
1. Initial partition: A has 7 neighbors, B has 7 neighbors
2. i0 partition (irregular vertices): A has 7 neighbors, B has 7 neighbors
3. i1 partition (complement): A has 0 neighbors, B has 7 neighbors
4. Task construction will differ between i0 and i1
5. Pathweight will differ: i0 has 7*7=49, i1 has 0*7=0
6. Irregularity weight will differ
7. Eventually some partitions will pass the threshold
8. Algorithm will converge

================================================================================
FILES THAT NEED FIXING
================================================================================

1. manager.py:
   - iterate() method around line 490-491
   - iterate() method around line 516 (for irregularity splitting)
   - iterate() method around line 547 (for deviation splitting)
   - compute_path_data() method around line 414-415

2. Test files will automatically work once the fix is applied
"""

print(__doc__)
