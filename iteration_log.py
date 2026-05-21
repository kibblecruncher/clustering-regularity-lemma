from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PartitionStats:
    """Structured log entry for one direction processed by the algorithm."""

    direction: str
    pathweight: float
    triangle_count: float
    gamma: float
    clustering_threshold: float
    failure_reason: Optional[str] = None
    irreg_weight: Optional[float] = None
    dev_weight: Optional[float] = None
    irreg_threshold: Optional[float] = None
    dev_threshold: Optional[float] = None

    def to_dict(self):
        return {
            "direction": self.direction,
            "pathweight": int(self.pathweight) if isinstance(self.pathweight, (int, np.integer)) else float(self.pathweight),
            "triangle_count": int(self.triangle_count) if isinstance(self.triangle_count, (int, np.integer)) else float(self.triangle_count),
            "gamma": float(self.gamma),
            "clustering_threshold": float(self.clustering_threshold),
            "failure_reason": self.failure_reason,
            "irreg_weight": None if self.irreg_weight is None else float(self.irreg_weight),
            "dev_weight": None if self.dev_weight is None else float(self.dev_weight),
            "irreg_threshold": None if self.irreg_threshold is None else float(self.irreg_threshold),
            "dev_threshold": None if self.dev_threshold is None else float(self.dev_threshold),
        }
