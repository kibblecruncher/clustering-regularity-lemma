from typing import Any, List, Tuple
import json
import queue as q

import networkx as nx
import numpy as np

from clustering_task import Task
from edge_partition import EdgePartitionAssembler, partition_labels
from iteration_log import PartitionStats
from parameters import AlgorithmParameters
from partition import PartitionRecord, convert_json_to_nodes
from storage import FileManager
from tripartite import GraphManager


class AlgorithmRunner(object):
    """Coordinates graph storage, local partition refinement, and final labels."""

    @staticmethod
    def _convert_json_to_nodes(json_list: List) -> List:
        return convert_json_to_nodes(json_list)

    def __init__(
        self,
        G: nx.Graph,
        eps: float = None,
        irreg_vtx_threshold: float = None,
        dev_vtx_threshold: float = None,
        irreg_vtx_count_threshold: float = None,
        dev_threshold: float = None,
        dev_split_threshold: float = None,
        irreg_threshold: float = None,
        clustering_threshold: float = None,
        max_depth: int = float("inf"),
        partition_dir: str = "partitions",
        graph_dir: str = "graphs",
        partition_manager=None,
        graph_manager=None,
        parameters: AlgorithmParameters = None,
        file_manager_cls=FileManager,
    ) -> None:
        if parameters is None:
            if eps is None:
                raise ValueError("eps is required when parameters is not provided")
            parameters = AlgorithmParameters(
                eps=eps,
                irreg_vtx_threshold=irreg_vtx_threshold,
                dev_vtx_threshold=dev_vtx_threshold,
                irreg_vtx_count_threshold=irreg_vtx_count_threshold,
                dev_threshold=dev_threshold,
                dev_split_threshold=dev_split_threshold,
                irreg_threshold=irreg_threshold,
                clustering_threshold=clustering_threshold,
                max_depth=max_depth,
            )

        self.parameters = parameters
        self.eps = parameters.eps
        self.irreg_vtx_threshold = parameters.irreg_vtx_threshold
        self.dev_vtx_threshold = parameters.dev_vtx_threshold
        self.irreg_vtx_count_threshold = parameters.irreg_vtx_count_threshold
        self.dev_threshold = parameters.dev_threshold
        self.dev_split_threshold = parameters.dev_split_threshold
        self.irreg_threshold = parameters.irreg_threshold
        self.clustering_threshold = parameters.clustering_threshold
        self.max_depth = parameters.max_depth

        self.graph_manager = graph_manager or GraphManager(G)
        self.partition_manager = partition_manager or file_manager_cls(partition_dir, graph_dir)
        self.edge_assembler = EdgePartitionAssembler(self.graph_manager)
        self.V2 = self.graph_manager.getV2()
        self.q = q.Queue()
        self.directions_considered = []
        self.partition_logs = []

    def _get_edge_lists(self) -> Tuple[List[Tuple], List[Tuple]]:
        return self.edge_assembler.get_edge_lists()

    def _map_vertex_partition_to_edges(
        self,
        v: int,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        neighbors_A: List,
        neighbors_B: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.edge_assembler.map_vertex_partition_to_edges(
            v, mask_A, mask_B, neighbors_A, neighbors_B
        )

    def _load_partition_record(self, vertex: int, direction: str) -> Tuple[bool, PartitionRecord]:
        if hasattr(self.partition_manager, "loadPartitionRecord"):
            return self.partition_manager.loadPartitionRecord(vertex, direction)

        success, partition_str = self.partition_manager.loadPartition(vertex, direction)
        if not success:
            return False, None
        return True, PartitionRecord.from_json_dict(json.loads(partition_str))

    def _load_partition_with_mask(self, vertex: int, direction: str) -> Tuple[bool, Any]:
        """Load a partition and apply its masks to select neighbors."""
        try:
            success, record = self._load_partition_record(vertex, direction)
            if not success:
                return False, ([], [])
            return True, record.masked_neighbors()
        except Exception:
            return False, ([], [])

    def _initialize_link_data(self) -> None:
        """Create initial full partitions and link graphs for all V2 vertices."""
        self.V2 = self.graph_manager.getV2()
        for v in self.V2:
            A, B = self.graph_manager.makeLinkPartition(v)
            mask_A = np.ones(len(A), dtype=bool)
            mask_B = np.ones(len(B), dtype=bool)
            self.partition_manager.savePartition(v, "", mask_A, mask_B, A, B)
            self.partition_manager.saveLinkGraph(v, self.graph_manager.makeLinkGraph(v))

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        """Execute the main algorithm and return final E12 and E23 partition labels."""
        self._initialize_link_data()
        self.q = q.Queue()
        self.q.put("")

        dirs = self.iterate()
        masks_A = []
        masks_B = []
        for direction in dirs:
            mask_A, mask_B = self.assemble_partition(direction)
            masks_A.append(mask_A)
            masks_B.append(mask_B)

        self.partition_labels_A = self.partitionLabels(masks_A)
        self.partition_labels_B = self.partitionLabels(masks_B)

        self.partition_manager.deleteAllPartitions()
        self.partition_manager.deleteAllLinkGraphs()
        return self.partition_labels_A, self.partition_labels_B

    def assemble_partition(self, dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Assembles global edge partitions for all vertices in V2 for one direction."""
        return self.edge_assembler.assemble_partition(self.V2, self.partition_manager, dir)

    def partitionLabels(self, bitmasks: List[np.ndarray]) -> np.ndarray:
        return partition_labels(bitmasks)

    def compute_direction_code_length(self, direction: str) -> int:
        """Computes the length of a direction code."""
        return len(direction)

    def compute_path_data(self, dir: str) -> Tuple[int, int, float]:
        """Computes pathweight, triangle count, and gamma for the current partition."""
        pathweight = 0
        triangle_count = 0
        for v in self.V2:
            success, link = self.partition_manager.loadLinkGraph(v)
            success_partition, (A, B) = self._load_partition_with_mask(v, dir)
            if not success or not success_partition:
                continue

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            pathweight += len(task.A) * len(task.B)
            triangle_count += task.edges

        gamma = 0.0 if pathweight == 0 else triangle_count / pathweight
        return pathweight, triangle_count, gamma

    def _base_stats(self, direction: str, pathweight: float, triangle_count: float, gamma: float):
        return PartitionStats(
            direction=direction,
            pathweight=pathweight,
            triangle_count=triangle_count,
            gamma=gamma,
            clustering_threshold=self.clustering_threshold,
        )

    def _save_irregular_splits(self, direction: str, gamma: float, irreg_vertices: np.ndarray) -> None:
        for i, v in enumerate(self.V2):
            if not irreg_vertices[i]:
                continue

            success_g, link = self.partition_manager.loadLinkGraph(v)
            success_p, (A, B) = self._load_partition_with_mask(v, direction)
            if not success_g or not success_p:
                raise ValueError(
                    f"Failed to load partition for vertex {v} and direction {direction} "
                    "when setting irregularity partition"
                )

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            irreg_v, _ = task.compute_irregular_vertices(gamma)
            self.partition_manager.savePartition(
                v, direction + "i0", np.array(irreg_v), np.ones(len(B), dtype=bool), A, B
            )
            self.partition_manager.savePartition(
                v, direction + "i1", ~np.array(irreg_v), np.ones(len(B), dtype=bool), A, B
            )
            self.partition_manager.deletePartition(v, direction)

    def _save_deviation_splits(self, direction: str, gamma: float, dev_vertices: np.ndarray) -> None:
        for i, v in enumerate(self.V2):
            if not dev_vertices[i]:
                continue

            success_g, link = self.partition_manager.loadLinkGraph(v)
            success_p, (A, B) = self._load_partition_with_mask(v, direction)
            if not success_g or not success_p:
                raise ValueError(
                    f"Failed to load partition for vertex {v} and direction {direction} "
                    "when setting deviation partition"
                )

            task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
            L, R = task.produce_new_masks(gamma)
            self.partition_manager.savePartition(v, direction + "d0", np.array(L), np.array(R), A, B)
            self.partition_manager.savePartition(v, direction + "d1", ~np.array(L), np.array(R), A, B)
            self.partition_manager.savePartition(v, direction + "d2", np.array(L), ~np.array(R), A, B)
            self.partition_manager.savePartition(v, direction + "d3", ~np.array(L), ~np.array(R), A, B)
            self.partition_manager.deletePartition(v, direction)

    def iterate(self) -> List[str]:
        """Main refinement loop."""
        out = []
        self.directions_considered = []
        self.partition_logs = []

        while not self.q.empty():
            direction = self.q.get()
            self.directions_considered.append(direction)

            if self.compute_direction_code_length(direction) > self.max_depth:
                self.partition_manager.deleteAllPartitions()
                self.partition_manager.deleteAllLinkGraphs()
                raise ValueError(
                    f"Direction code '{direction}' "
                    f"(length {self.compute_direction_code_length(direction)}) "
                    f"exceeds maximum depth {self.max_depth}"
                )

            pathweight, triangle_count, gamma = self.compute_path_data(direction)
            stats = self._base_stats(direction, pathweight, triangle_count, gamma)

            if gamma < self.clustering_threshold:
                stats.failure_reason = "PASSED_GAMMA_CHECK"
                out.append(direction)
                self.partition_logs.append(stats.to_dict())
                continue

            irreg_weight = 0.0
            dev_weight = 0.0
            dev_vertices = np.array([False] * len(self.V2))
            irreg_vertices = np.array([False] * len(self.V2))

            for i, v in enumerate(self.V2):
                success_g, link = self.partition_manager.loadLinkGraph(v)
                success_p, (A, B) = self._load_partition_with_mask(v, direction)
                if not success_g or not success_p:
                    continue

                task = Task(link, (A, B), self.eps, self.irreg_vtx_threshold, self.dev_vtx_threshold, self.dev_split_threshold)
                irreg_v, irreg_count = task.compute_irregular_vertices(gamma)
                dev = task.compute_local_deviation(gamma)
                if irreg_count > self.irreg_vtx_count_threshold * pathweight:
                    irreg_vertices[i] = True
                    irreg_weight += np.sum(irreg_v * task.pathweight)
                elif dev > self.dev_vtx_threshold:
                    dev_vertices[i] = True
                    dev_weight += np.sum(task.pathweight)

            stats.irreg_weight = irreg_weight
            stats.dev_weight = dev_weight
            stats.irreg_threshold = self.irreg_threshold * pathweight
            stats.dev_threshold = self.dev_threshold * pathweight

            if irreg_weight > self.irreg_threshold * pathweight:
                stats.failure_reason = "FAILED_IRREGULARITY_CHECK"
                self.partition_logs.append(stats.to_dict())
                self.q.put(direction + "i0")
                self.q.put(direction + "i1")
                self._save_irregular_splits(direction, gamma, irreg_vertices)
            elif dev_weight > self.dev_threshold * pathweight:
                stats.failure_reason = "FAILED_DEVIATION_CHECK"
                self.partition_logs.append(stats.to_dict())
                self.q.put(direction + "d0")
                self.q.put(direction + "d1")
                self.q.put(direction + "d2")
                self.q.put(direction + "d3")
                self._save_deviation_splits(direction, gamma, dev_vertices)
            else:
                stats.failure_reason = "PASSED_ALL_CHECKS"
                self.partition_logs.append(stats.to_dict())
                out.append(direction)

        return out
