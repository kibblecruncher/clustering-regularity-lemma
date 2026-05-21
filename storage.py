import hashlib
import json
import os
from typing import List, Tuple

import networkx as nx
import numpy as np

from partition import PartitionRecord


class FileManager(object):
    """Persistence for partition records and per-vertex link graphs."""

    def __init__(self, partition_dir: str, graph_dir: str):
        os.makedirs(partition_dir, exist_ok=True)
        self.target_dir = partition_dir
        os.makedirs(graph_dir, exist_ok=True)
        self.graph_dir = graph_dir

    @staticmethod
    def _direction_key(direction: str) -> str:
        """Return a bounded stable key for direction-dependent partition filenames."""
        return hashlib.sha1(direction.encode("utf-8")).hexdigest()[:16]

    def partitionFileName(self, vertex: int, direction: str) -> str:
        """Generates a bounded file name for the given vertex and direction."""
        direction_key = self._direction_key(direction)
        return os.path.join(self.target_dir, f"partition_{vertex}_{direction_key}.json")

    def savePartition(
        self,
        vertex: int,
        direction: str,
        mask_A: np.ndarray,
        mask_B: np.ndarray,
        neighbors_A: List = None,
        neighbors_B: List = None,
    ) -> None:
        """Saves the canonical partition record to disk."""
        record = PartitionRecord(vertex, direction, mask_A, mask_B, neighbors_A, neighbors_B)
        if not record.has_nonempty_masks():
            return

        with open(self.partitionFileName(vertex, direction), "w", encoding="utf-8") as f:
            json.dump(record.to_json_dict(), f)

    def loadPartition(self, vertex: int, direction: str) -> Tuple[bool, str]:
        """Loads a partition JSON string from disk."""
        try:
            with open(self.partitionFileName(vertex, direction), "r", encoding="utf-8") as f:
                data = f.read()
        except OSError:
            return False, None
        else:
            return True, data

    def loadPartitionRecord(self, vertex: int, direction: str) -> Tuple[bool, PartitionRecord]:
        """Loads and validates a canonical partition record from disk."""
        success, partition_str = self.loadPartition(vertex, direction)
        if not success:
            return False, None
        return True, PartitionRecord.from_json_dict(json.loads(partition_str))

    def deletePartition(self, vertex: int, direction: str) -> None:
        """Deletes the partition file from disk."""
        try:
            os.remove(self.partitionFileName(vertex, direction))
        except OSError:
            pass

    def deleteAllPartitions(self):
        """Deletes all partitions in the partition directory."""
        for filename in os.listdir(self.target_dir):
            file_path = os.path.join(self.target_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    def deleteAllLinkGraphs(self):
        """Deletes all link graphs to clean up space."""
        for filename in os.listdir(self.graph_dir):
            file_path = os.path.join(self.graph_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    def graphFileName(self, vertex: int) -> str:
        """Generates a file name for the given vertex."""
        return os.path.join(self.graph_dir, f"linkGraph_{vertex}.json")

    def saveLinkGraph(self, vertex: int, G: nx.Graph) -> None:
        """Saves the link graph to disk."""
        # Use compatible parameter name for node_link_data
        try:
            # Try newer version of NetworkX with 'edges' parameter
            json_data = nx.node_link_data(G, edges="links")
        except TypeError:
            # Fall back to older version without 'edges' parameter
            json_data = nx.node_link_data(G)
        with open(self.graphFileName(vertex), "w", encoding="utf-8") as f:
            json.dump(json_data, f)

    def loadLinkGraph(self, v: int) -> Tuple[bool, nx.Graph]:
        """Loads the link graph from disk."""
        try:
            with open(self.graphFileName(v), "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError:
            return False, None
        else:
            try:
                # Try newer version of NetworkX with 'edges' parameter
                return True, nx.node_link_graph(data, edges="links")
            except TypeError:
                # Fall back to older version without 'edges' parameter
                return True, nx.node_link_graph(data)
