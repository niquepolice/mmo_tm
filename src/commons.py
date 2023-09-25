from dataclasses import dataclass

import numpy as np


@dataclass
class Correspondences:
    traffic_mat: np.ndarray  # between zones
    node_traffic_mat: np.ndarray  # between all nodes
    sources: np.ndarray
    targets: np.ndarray

