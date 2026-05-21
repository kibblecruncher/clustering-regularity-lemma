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
    dev_split_threshold: Optional[float] = None
    irreg_threshold: Optional[float] = None
    clustering_threshold: Optional[float] = None
    max_depth: float = float("inf")

    def __post_init__(self) -> None:
        """
        Initializes the manager with the given graph and parameters.
        
        Standard hyperparameters for the algorithm include:
        - eps < 1/16 : parameter for the clustering task, which determines the approximation quality
        - irreg_vtx_threshold <= eps**5/90: threshold for local irregularity of a vertex within a link graph
        - irreg_vtx_count_threshold <= eps**(5/2)/9: threshold for the number of irregular vertices at a vertex
        - irreg_threshold  <= 2*eps**(5/2)/5: threshold for the total irregularity weight of a partition
        - dev_vtx_threshold  <= eps**2/9 : threshold for local deviation of a vertex
        - dev_threshold <= 2*eps**2/5 : threshold for the total deviation weight of a partition
        - clustering_threshold <= eps : threshold for smallest allowed clustering coefficient of a partition
        """
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        
        defaults = {
            "irreg_vtx_threshold": self.eps**5 / 90,
            "dev_vtx_threshold": self.eps**2 / 9,
            "irreg_vtx_count_threshold": self.eps**(5/2) / 9,
            "dev_threshold": 2 * self.eps**2 / 5,
            "dev_split_threshold": self.eps**5,
            "irreg_threshold": 2 * self.eps**(5/2) / 5,
            "clustering_threshold": self.eps,
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
