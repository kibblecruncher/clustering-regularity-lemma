from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


CANONICAL_PARTITION_KEYS = {
    "vtx",
    "dir",
    "mask_A",
    "mask_B",
    "neighbors_A",
    "neighbors_B",
}


def convert_json_to_nodes(json_list: List[Any]) -> List[Any]:
    """Convert JSON lists back to node tuples where needed."""
    if json_list is None:
        raise ValueError("neighbor list cannot be None")

    result = []
    for item in json_list:
        if isinstance(item, list):
            result.append(tuple(item))
        else:
            result.append(item)
    return result


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _node_to_json(node: Any) -> Any:
    if isinstance(node, np.ndarray):
        node = node.tolist()
    if isinstance(node, (tuple, list)):
        return [_json_scalar(part) for part in node]
    return _json_scalar(node)


def _normalize_neighbors(neighbors: Any) -> List[Any]:
    if neighbors is None:
        raise ValueError("neighbor list cannot be None")
    if isinstance(neighbors, np.ndarray):
        neighbors = neighbors.tolist()
    return convert_json_to_nodes(list(neighbors))


@dataclass(frozen=True)
class PartitionRecord:
    """Canonical persisted partition: masks plus the neighbor identifiers they index."""

    vertex: int
    direction: str
    mask_A: np.ndarray
    mask_B: np.ndarray
    neighbors_A: List[Any]
    neighbors_B: List[Any]

    def __post_init__(self) -> None:
        mask_A = np.asarray(self.mask_A, dtype=bool)
        mask_B = np.asarray(self.mask_B, dtype=bool)
        if mask_A.ndim != 1 or mask_B.ndim != 1:
            raise ValueError("mask_A and mask_B must be one-dimensional bitmasks")

        neighbors_A = _normalize_neighbors(self.neighbors_A)
        neighbors_B = _normalize_neighbors(self.neighbors_B)
        if len(mask_A) != len(neighbors_A):
            raise ValueError("mask_A length must match neighbors_A length")
        if len(mask_B) != len(neighbors_B):
            raise ValueError("mask_B length must match neighbors_B length")

        object.__setattr__(self, "mask_A", mask_A)
        object.__setattr__(self, "mask_B", mask_B)
        object.__setattr__(self, "neighbors_A", neighbors_A)
        object.__setattr__(self, "neighbors_B", neighbors_B)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "PartitionRecord":
        missing = CANONICAL_PARTITION_KEYS - set(data)
        if missing:
            raise ValueError(f"partition JSON missing required keys: {sorted(missing)}")
        if "A" in data or "B" in data:
            raise ValueError("legacy A/B partition keys are not supported")
        return cls(
            vertex=int(data["vtx"]),
            direction=str(data["dir"]),
            mask_A=np.asarray(data["mask_A"], dtype=bool),
            mask_B=np.asarray(data["mask_B"], dtype=bool),
            neighbors_A=convert_json_to_nodes(data["neighbors_A"]),
            neighbors_B=convert_json_to_nodes(data["neighbors_B"]),
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "vtx": int(self.vertex),
            "dir": self.direction,
            "mask_A": [int(value) for value in self.mask_A],
            "mask_B": [int(value) for value in self.mask_B],
            "neighbors_A": [_node_to_json(node) for node in self.neighbors_A],
            "neighbors_B": [_node_to_json(node) for node in self.neighbors_B],
        }

    def has_nonempty_masks(self) -> bool:
        return bool(np.any(self.mask_A) and np.any(self.mask_B))

    def masked_neighbors(self) -> Tuple[List[Any], List[Any]]:
        if not np.any(self.mask_A):
            raise ValueError(
                f"mask_A is all False for vertex {self.vertex} and direction {self.direction}"
            )
        if not np.any(self.mask_B):
            raise ValueError(
                f"mask_B is all False for vertex {self.vertex} and direction {self.direction}"
            )

        neighbors_A = np.asarray(self.neighbors_A, dtype=object)
        neighbors_B = np.asarray(self.neighbors_B, dtype=object)
        return (
            [tuple(node) if isinstance(node, (list, np.ndarray)) else node for node in neighbors_A[self.mask_A]],
            [tuple(node) if isinstance(node, (list, np.ndarray)) else node for node in neighbors_B[self.mask_B]],
        )
