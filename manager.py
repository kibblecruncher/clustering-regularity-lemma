from algorithm import AlgorithmRunner
from edge_partition import EdgePartitionAssembler, partition_labels
from parameters import AlgorithmParameters
from partition import PartitionRecord, convert_json_to_nodes
from storage import FileManager
from tripartite import GraphManager


class Manager(AlgorithmRunner):
    """Compatibility entrypoint for the clustering regularity algorithm."""

    def __init__(
        self,
        G,
        eps: float = None,
        irreg_vtx_threshold: float = None,
        dev_vtx_threshold: float = None,
        irreg_vtx_count_threshold: float = None,
        dev_threshold: float = None,
        irreg_threshold: float = None,
        clustering_threshold: float = None,
        max_depth: int = float("inf"),
        partition_dir: str = "partitions",
        graph_dir: str = "graphs",
        partition_manager=None,
        graph_manager=None,
        parameters: AlgorithmParameters = None,
    ) -> None:
        super().__init__(
            G=G,
            eps=eps,
            irreg_vtx_threshold=irreg_vtx_threshold,
            dev_vtx_threshold=dev_vtx_threshold,
            irreg_vtx_count_threshold=irreg_vtx_count_threshold,
            dev_threshold=dev_threshold,
            irreg_threshold=irreg_threshold,
            clustering_threshold=clustering_threshold,
            max_depth=max_depth,
            partition_dir=partition_dir,
            graph_dir=graph_dir,
            partition_manager=partition_manager,
            graph_manager=graph_manager,
            parameters=parameters,
            file_manager_cls=FileManager,
        )


__all__ = [
    "AlgorithmParameters",
    "EdgePartitionAssembler",
    "FileManager",
    "GraphManager",
    "Manager",
    "PartitionRecord",
    "convert_json_to_nodes",
    "partition_labels",
]
