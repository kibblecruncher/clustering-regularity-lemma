#!/usr/bin/env python3
from scipy.sparse import csr_matrix # depending on graphs used i suppose
import networkx as nx
import numpy as np

class Task:
    def __init__(self, link, partition, eps: float):
        self.link = self.G = link
        self.A, self.B = partition
        self.eps = eps

        self.M = bipartite.biadjacency_matrix(
            self.G,
            row_order=self.A,
            column_order=self.B
        )

        self.edges = self.G.number_of_edges()

        self.density = 0.0 if len(self.A) * len(self.B) == 0 else (
            self.edges / (len(self.A) * len(self.B))
        )

        self.deg_A_v = np.array(self.M.sum(axis=1)).ravel()
        self.deg_B_v = np.array(self.M.sum(axis=0)).ravel()
        self.pathweight = deg_A_v * deg_B_v

        self.deg_N_A = self.L.sum(axis=1) # degrees of N_A(v)
        self.deg_N_B = self.L.sum(axis=0) # degrees of N_B(v)

    def compute_local_deviation(self, gamma):
        spec = self.M @ self.M.T - (gamma**2) * self.deg_B_v
        self.spec_dev_matrix = spec
        return np.sum(spec)

    def compute_irregular_vertices(self, gamma):
        expected = gamma * self.deg_B_v

        delta_1 = self.eps**5 / 90

        positive = (self.deg_A_v - expected) > delta_1 * self.deg_B_v
        negative = (self.deg_A_v - expected) < -delta_1 * self.deg_B_v

        if np.sum(positive) > np.sum(negative):
            return positive, np.sum(positive)
        else:
            return negative, np.sum(negative)

    def produce_new_masks(self, gamma):
        self.local_dev = self.compute_local_deviation(gamma)
        self.big_irreg_set, self.total_irreg = self.compute_irregular_vertices(gamma)

        delta = self.eps

        U_v = np.abs(self.deg_A_v - gamma * self.deg_B_v) > delta
        scores = self.spec_dev_matrix @ U_v.astype(int)
        u_star = np.argmax(scores)

        row = np.asarray(self.spec_dev_matrix[u_star, :]).ravel()
        L_v = row > delta

        mask_B = (self.M[u_star, :].toarray().ravel() > 0).astype(int)

        return (self.A, self.B), (self.big_irreg_set, self.B), (L_v, mask_B)
        
