from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Correspondences:
    """Traffic matrix probably should always be with source and target nodes lists"""

    traffic_mat: Optional[np.ndarray]  # between zones. May be None in two-stage models
    node_traffic_mat: Optional[np.ndarray]  # between all nodes
    sources: np.ndarray
    targets: np.ndarray
