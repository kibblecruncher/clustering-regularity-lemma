from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AlgorithmParameters:
    """Thresholds and limits used by the clustering regularity algorithm."""

    eps: float
    irreg_vtx_threshold: Optional[float] = None
    dev_vtx_threshold: Optional[float] = None
    irreg_vtx_count_threshold: Optional[float] = None
    dev_threshold: Optional[float] = None
    irreg_threshold: Optional[float] = None
    clustering_threshold: Optional[float] = None
    max_depth: float = float("inf")

    def __post_init__(self) -> None:
        if self.eps <= 0:
            raise ValueError("eps must be positive")

        defaults = {
            "irreg_vtx_threshold": self.eps**5 / 90,
            "dev_vtx_threshold": self.eps,
            "irreg_vtx_count_threshold": 0.1,
            "dev_threshold": 0.1,
            "irreg_threshold": self.eps,
            "clustering_threshold": self.eps,
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
