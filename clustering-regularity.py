#!/usr/bin/env python3
from scipy.sparse import csr_matrix # depending on graphs used i suppose
import numpy as np

class TripartiteBlowup:
    # sorry if this notation sucks, i feel uneasy working with an adjacency matrix G
    # assume M is int64
    def __init__(self, M, eps=0.0625):
        self.M = M
        self.n = M.shape[0]

        # triangle count in tripartite blowup
        self.triangle_count = int(np.trace(M @ M @ M) )

        degrees = M.sum(axis=1)
        self.path_count = int(np.sum(degrees ** 2))

        clustering_coefficient = 0.0 if self.path_count == 0 else self.triangle_count / self.path_count
        self.gamma = clustering_coefficient
        self.eps = eps

class LinkGraph:
    def __init__(self, H, v, A_mask, B_mask):
        self.H = H
        self.v = v
        self.A_mask = A_mask
        self.B_mask = B_mask
        self.L = (H.M)[A_mask][:, B_mask]

        # for convenience
        self.gamma = H.gamma
        self.eps = H.eps

        # mostly sanity checks
        self.edges = self.L.sum()
        self.density = 0.0 if self.edges == 0 else self.edges / (A_mask.sum() * B_mask.sum())
        self.deg_A = self.L.sum(axis=1)
        self.deg_B = self.L.sum(axis=0)

        intersection_counts = self.L @ (self.L).T
        local_deviation = intersection_counts - (self.gamma ** 2) * self.B_mask.sum()
        np.fill_diagonal(local_deviation, 0)
        self.total_deviation = local_deviation.sum()

    def compute_irregular_vertices(self):
        # number of vertices in U whose degree deviates from expected by > epsilon
        expected_deg = self.gamma * self.B_mask.sum()
        return np.sum(np.abs(self.deg_A - expected_deg) > self.eps)

    def compute_deviation(self):
        intersection_counts = self.L @ (self.L).T
        local_deviation = intersection_counts - (self.gamma ** 2) * self.B_mask.sum()
        np.fill_diagonal(local_deviation, 0)
        return local_deviation.sum()

    def deviation_set(self, delta):
        

class Manager:
    # TODO local direction trees
    # TODO direction queue
    # TODO def read(): str -> Maybe(Link)
    # TODO def write: L -> File (named vtx_directions)
    # TODO 

    def 
