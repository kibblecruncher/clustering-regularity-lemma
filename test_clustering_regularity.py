#!/usr/bin/env python3
"""Unit tests for clustering_regularity.py"""

import unittest
import numpy as np
import networkx as nx
import importlib.util
spec = importlib.util.spec_from_file_location("clustering_task", "clustering_task.py")
clustering_task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clustering_task)
Task = clustering_task.Task


class TestTaskInitialization(unittest.TestCase):
    """Test Task class initialization."""
    
    def setUp(self):
        """Create a simple bipartite graph for testing."""
        # Create a simple bipartite graph: A = {0, 1, 2}, B = {3, 4, 5}
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2], bipartite=0)
        self.G.add_nodes_from([3, 4, 5], bipartite=1)
        self.G.add_edges_from([(0, 3), (0, 4), (1, 3), (2, 5)])
        
        self.partition = (set([0, 1, 2]), set([3, 4, 5]))
        self.eps = 0.1

    def test_task_creation(self):
        """Test that Task can be created with valid inputs."""
        task = Task(self.G, self.partition, self.eps)
        
        self.assertIsNotNone(task)
        self.assertEqual(task.eps, self.eps)
        self.assertEqual(task.A, self.partition[0])
        self.assertEqual(task.B, self.partition[1])

    def test_density_calculation(self):
        """Test density calculation for non-empty partitions."""
        task = Task(self.G, self.partition, self.eps)
        
        # 4 edges, |A|=3, |B|=3, so density = 4/9
        expected_density = 4 / 9
        self.assertAlmostEqual(task.density, expected_density)

    def test_empty_density(self):
        """Test density calculation when one partition is empty."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2], bipartite=0)
        G.add_nodes_from([3, 4, 5], bipartite=1)
        
        partition = (set([0, 1, 2]), set())
        task = Task(G, partition, self.eps)
        
        self.assertEqual(task.density, 0.0)


class TestComputeLocalDeviation(unittest.TestCase):
    """Test compute_local_deviation method."""
    
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2], bipartite=0)
        self.G.add_nodes_from([3, 4, 5], bipartite=1)
        self.G.add_edges_from([(0, 3), (0, 4), (1, 3), (2, 5)])
        
        self.partition = (set([0, 1, 2]), set([3, 4, 5]))
        self.eps = 0.1
        self.task = Task(self.G, self.partition, self.eps)

    def test_local_deviation_returns_float(self):
        """Test that compute_local_deviation returns a float."""
        gamma = 0.5
        result = self.task.compute_local_deviation(gamma)
        
        self.assertIsInstance(result, (float, np.floating))

    def test_local_deviation_stores_matrix(self):
        """Test that spectral deviation matrix is stored."""
        gamma = 0.5
        self.task.compute_local_deviation(gamma)
        
        self.assertIsNotNone(self.task.spec_dev_matrix)


class TestComputeIrregularVertices(unittest.TestCase):
    """Test compute_irregular_vertices method."""
    
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2], bipartite=0)
        self.G.add_nodes_from([3, 4, 5], bipartite=1)
        self.G.add_edges_from([(0, 3), (0, 4), (1, 3), (2, 5)])
        
        self.partition = (set([0, 1, 2]), set([3, 4, 5]))
        self.eps = 0.1
        self.task = Task(self.G, self.partition, self.eps)

    def test_irregular_vertices_returns_tuple(self):
        """Test that irregular vertices returns correct tuple type."""
        gamma = 0.5
        result = self.task.compute_irregular_vertices(gamma)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        # First element should be numpy array
        self.assertIsInstance(result[0], np.ndarray)
        # Second element should be int
        self.assertIsInstance(result[1], (int, np.integer))

    def test_irregular_vertices_count_non_negative(self):
        """Test that irregular vertex count is non-negative."""
        gamma = 0.5
        _, count = self.task.compute_irregular_vertices(gamma)
        
        self.assertGreaterEqual(count, 0)


class TestProduceNewMasks(unittest.TestCase):
    """Test produce_new_masks method."""
    
    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([0, 1, 2], bipartite=0)
        self.G.add_nodes_from([3, 4, 5], bipartite=1)
        self.G.add_edges_from([(0, 3), (0, 4), (1, 3), (2, 5)])
        
        self.partition = (set([0, 1, 2]), set([3, 4, 5]))
        self.eps = 0.1
        self.task = Task(self.G, self.partition, self.eps)

    def test_produce_new_masks_returns_tuple(self):
        """Test that produce_new_masks returns local A and B masks."""
        gamma = 0.5
        result = self.task.produce_new_masks(gamma)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        L_v, mask_B = result
        self.assertIsInstance(L_v, np.ndarray)
        self.assertIsInstance(mask_B, np.ndarray)
        self.assertEqual(L_v.shape[0], len(self.task.A))
        self.assertEqual(mask_B.shape[0], len(self.task.B))

    def test_produce_new_masks_sets_attributes(self):
        """Test that produce_new_masks sets required attributes."""
        gamma = 0.5
        self.task.produce_new_masks(gamma)
        
        self.assertIsNotNone(self.task.local_dev)
        self.assertIsNotNone(self.task.spec_dev_matrix)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_single_vertex_partitions(self):
        """Test with single vertex in each partition."""
        G = nx.Graph()
        G.add_nodes_from([0], bipartite=0)
        G.add_nodes_from([1], bipartite=1)
        G.add_edge(0, 1)
        
        partition = (set([0]), set([1]))
        task = Task(G, partition, 0.1)
        
        self.assertEqual(task.edges, 1)
        self.assertEqual(task.density, 1.0)

    def test_no_edges(self):
        """Test with no edges between partitions."""
        G = nx.Graph()
        G.add_nodes_from([0, 1], bipartite=0)
        G.add_nodes_from([2, 3], bipartite=1)
        
        partition = (set([0, 1]), set([2, 3]))
        task = Task(G, partition, 0.1)
        
        self.assertEqual(task.edges, 0)
        self.assertEqual(task.density, 0.0)


if __name__ == '__main__':
    unittest.main()
