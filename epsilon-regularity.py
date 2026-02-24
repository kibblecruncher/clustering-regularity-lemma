#!/usr/bin/env python3
# written with help from chatgpt (disapprovingly)

import numpy as np
import networkx as nx
from networkx.utils import random_sequence as rs
#import matplotlib as plt
import itertools as itools
import random
#import logging
import time
from typing import List, Dict, Tuple
from math import ceil

def refine_pair(C, eps):
    """
    Corollary 3.3: Check whether (A,B) is eps-regular,
    or return subsets A', B' that witness irregularity.

    Parameters
    ----------
    C : np.ndarray
        n x n adjacency submatrix (A rows, B columns)
    eps : float
        Regularity parameter

    Returns
    -------
    tuple
        (regular: bool, A_p: bitmask, B_p: bitmask)
    """
    # # will drop this if equitability is no longer required
    # # was a sanity check
    # if C.shape[0] != C.shape[1]:
    #     raise ValueError(f"Submatrix C must be square: \
    #     got {C.shape[0]} rows and {C.shape[1]} columns")

    m = C.shape[0]
    n = C.shape[1]

    num_edges = C.sum()

    d_A = num_edges / m
    d_B = num_edges / n
    empty_A = np.zeros(m, dtype=int)
    empty_B = np.zeros(m, dtype=int)
    delta = eps

    # ---- Case 1: avg degree trivially low ----
    # check in-person to see if this can be (easily) improved
    if (d_B < (eps**3) * m) or (d_A < (eps**3) * n):
        return True, np.zeros(m, dtype=bool), np.zeros(n, dtype=bool)

    # ---- Case 2: outstanding subset on B----
    deg_B = C.sum(axis=0)
    # note that this is just deviation as in distance
    deviation_B = np.abs(deg_B - d_B)
    mask = deviation_B >= delta * m
    candidates = np.where(mask)[0]

    # can refine this quantity if need be. probably should be using the same delta
    if len(candidates) > (1.0 / 8.0) * (delta) * n:
        pos = deviation_B[candidates] > 0
        neg = ~pos
        if pos.sum() >= neg.sum():
            B_mask = np.zeros(n, dtype=bool)
            B_mask[candidates[pos]] = True
        else:
            B_mask = np.zeros(n, dtype=bool)
            B_mask[candidates[neg]] = True
        #print("case 2")
        return False, np.zeros(m, dtype=bool), B_mask

    # ---- Case 3: deviation across subsets ----
    # at this point, we might have an epsilon-regular pair.
    remaining = np.where(~mask)[0]
    M_B = C.T @ C
    # oh. this was very incorrect before lmao (d**2 * m)
    deviation_matrix = M_B - (d_B**2 / m)

    for y0 in remaining:
        # check the use of this delta.
        B_y0_mask = deviation_matrix[y0, :] >= 2 * delta * n
        if B_y0_mask.sum() >= (1.0 / 4.0) * delta * n:
            A_mask = C[:, y0].astype(bool)
            #print("case 3")
            return False, A_mask, B_y0_mask

    # since none of the cases of lemma 3.1 apply, lemma 3.2 implies that we
    # indeed have an epsilon-regular pair.
    return True, np.zeros(m, dtype=bool), np.zeros(n, dtype=bool)

def venn_refine(part_labels, part_masks):
    """
    Refine partitions by splitting each part according to provided bitmasks.

    Parameters
    ----------
    part_labels : np.ndarray of shape (n,)
        Current partition labels
    part_masks : list of list of np.ndarray
        part_masks[i] is a list of bitmask arrays (dtype=bool) for part i.
        Each mask has shape (n,) and indicates membership in a witness subset.

    Returns
    -------
    part_labels : np.ndarray of shape (n,)
        Refined partition labels.
    """
    #c = np.sum(part_labels)
    n = len(part_labels)
    new_labels = part_labels.copy()
    next_label = 0

    k = len(part_masks)  # number of parts

    # ---- venn diagram refinement ----
    for i in range(k):
        # vertices in current part
        idx = np.where(part_labels == i)[0]
        #print(idx)
        if idx.size == 0:
            continue

        masks = part_masks[i]
        if not masks:  # no refinement; just assign all the same label
            new_labels[idx] = next_label
            next_label += 1
            continue

        # build sublabels via bit encoding (masks are part-local)
        sublabels = np.zeros(idx.size, dtype=int)
        for bit, mask in enumerate(masks):
            #print(mask)
            sublabels += mask.astype(int) << bit

        labels, inverse = np.unique(sublabels, return_inverse=True)
        new_labels[idx] = next_label + inverse
        next_label += labels.size

    return new_labels

    # ---- equitable splitting ----
    # equitable_labels = np.full_like(new_labels, -1)
    # unique_tmp_labels = np.unique(new_labels[new_labels != -1])
    # next_eq_label = 0
    # for tmp_lbl in unique_tmp_labels:
    #     block_idx = np.where(new_labels == tmp_lbl)[0]
    #     #b = max(1, int(c) // (4 ** int(k)))
    #     b = max(1, c // 2)

    #     for i in range(0, len(block_idx), b):
    #         blk = block_idx[i:i+b]
    #         if b > 1 and len(blk) == b:
    #             equitable_labels[blk] = next_eq_label  # or assign a new sequential label if preferred
    #             next_eq_label += 1

    # ---- check all non-exceptional labels have same size ----
    # unique_labels, counts = np.unique(equitable_labels[equitable_labels != -1], return_counts=True)
    # if unique_labels.size > 0 and not np.all(counts == counts[0]):
    #     raise ValueError(f"All non-exceptional labels must have the same number of vertices, got counts {dict(zip(unique_labels, counts))}")

def epsilon_regular_partition(G, eps, max_iter = 100):
    """
    Computes an epsilon-regular partition of the vertices of graph G.

    Parameters:
        G : networkx.Graph
            Input graph
        eps : float

    Returns:
        part_labels : np.ndarray
            Array of length n mapping each vertex to its part index

    """
    A = nx.to_numpy_array(G, dtype=int)
    n = A.shape[0]
    total_edges = G.size()

    # b might not be large enough, see Alon et al. (1994)
    #b = ceil( np.log( 600 * (eps**4 / 16)**(-5))/ np.log(4) )
    b = 2

    # initial trivial partition; no more exceptional set yay..
    part_labels = np.zeros(n, dtype=int)
    vertices = np.arange(n)
    np.random.shuffle(vertices)
    for i in range(n):
        part_labels[vertices[i]] = (i % b)

    # unique_labels, counts = np.unique(part_labels, return_counts=True)
    # for lbl, cnt in zip(unique_labels, counts):
    #     print(f"Label {lbl}: {cnt} vertices")

    for iteration in range(max_iter):
        #iter_start = time.time()
        #print(f"\n--- Iteration {iteration + 1} ---")

        unique_parts = np.unique(part_labels)
        k = len(unique_parts)
        # i think we need to rewrite the definition for the non-equitable one but w/e
        # total_pairs = k * (k - 1) // 2
        # threshold = eps * total_pairs #this is also wrong!!!
        threshold = (1 - 1.5*eps) * total_edges

        part_masks = [[] for _ in range(k)]
        verified_edges = 0
        done = False

        # check epsilon-regularity of pairs of non-exceptional sets
        pair_start = time.time()
        for i, j in itools.combinations(range(k), 2):
            A_idx = np.where(part_labels == i)[0]
            B_idx = np.where(part_labels == j)[0]

            C = A[np.ix_(A_idx, B_idx)]

            #mask_start = time.time()
            regular, A_mask, B_mask = refine_pair(C, eps)
            #print(f"refine_pair({i},{j}) done in {time.time() - mask_start:.3f} s")
            #print(A_mask)
            #print(B_mask)
            #print(regular)

            part_masks[i].append(A_mask)
            part_masks[j].append(B_mask)

            if regular:
                verified_edges += C.sum()
            if verified_edges >= threshold:
                done = True
                break

        if done:
            #print(f"Partition verified early after {time.time() - iter_start:.3f} s")
            break

        #print(f"All pair checks done in {time.time() - pair_start:.3f} s")

        #refine_start = time.time()
        part_labels = venn_refine(part_labels, part_masks)
        #print(f"venn_refine done in {time.time() - refine_start:.3f} s")
        #print(f"Iteration {iteration + 1} total time: {time.time() - iter_start:.3f} s")
    else:
        #print(part_labels)
        raise RuntimeError(f"epsilon_regular_partition failed to converge after {max_iter} iterations")

    return part_labels

def test_refine_pair_shapes():
    m = 1000
    n = 500
    eps = 0.25
    # adjacency submatrix
    C = np.random.randint(0, 2, size=(m, n))
    regular, A_mask, B_mask = refine_pair(C, eps)

    assert isinstance(regular, bool), "regular should be a boolean"
    assert A_mask.shape == (m,), f"A_mask shape should be {(n,)}, got {A_mask.shape}"
    assert B_mask.shape == (n,), f"B_mask shape should be {(n,)}, got {B_mask.shape}"
    assert A_mask.dtype == np.bool_, "A_mask should be bool"
    assert B_mask.dtype == np.bool_, "B_mask should be bool"

    print("test_refine_pair_shapes passed.")

def test_refine_pair_hard():
    n = 100
    p = 0.2
    eps = 0.05
    G = nx.bipartite.gnmk_random_graph(n, n, p*n*n)

    nodes = list(G.nodes())
    random.shuffle(nodes)

    U = set(nodes[:30])
    V = set(nodes[30:70])

    C = nx.bipartite.biadjacency_matrix(G, row_order=U, column_order=V).toarray()
    regular, A_mask, B_mask = refine_pair(C, eps)

    print(regular)
    print(A_mask)
    print(B_mask)

def test_venn_refine_basic():
    n = 1000
    part_labels = np.array([0]*300 + [1]*350 + [2]*250 + [3]*100)
    # create trivial masks (no refinement)
    part_masks = [[], [], [], []]

    refined = venn_refine(part_labels, part_masks)

    assert len(refined) == n
    # exceptional vertices remain -1
    #assert np.all(refined[part_labels == -1] == -1)
    # non-exceptional labels are integers
    #non_exc = refined[refined != -1]
    #assert np.all([isinstance(x, int) for x in refined])

    print(refined)
    print("test_venn_refine_basic passed.")

def main():
    n = 50000
    p = 0.8
    eps = 0.25
    block_size = 200

    start = time.time()
    G = nx.stochastic_block_model([block_size, block_size, block_size, block_size, block_size], \
                                  [[0.8, 0.2, 0.2, 0.2, 0.2], \
                                   [0.2, 0.8, 0.2, 0.2, 0.2], \
                                   [0.2, 0.2, 0.8, 0.2, 0.2], \
                                   [0.2, 0.2, 0.2, 0.8, 0.2], \
                                   [0.2, 0.2, 0.2, 0.2, 0.8]])

    part_labels = epsilon_regular_partition(G, eps, max_iter=10)

    # print("Partition labels:")
    # print(part_labels)

    unique_parts = np.unique(part_labels)
    print(f"Number of parts: {len(unique_parts)}")

    # counts = {lbl: np.sum(part_labels == lbl) for lbl in unique_parts}
    # print("Sizes of each part:", counts)

if __name__ == "__main__":
    import sys
    func = sys.argv[1] if len(sys.argv) > 1 else "main"
    globals()[func]()  # calls the function by name
