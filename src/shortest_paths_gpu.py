from dataclasses import dataclass
import time
from typing import Optional, Union

import graph_tool as gt
import numpy as np

from src.commons import Correspondences

import cupy as cp
import cudf
import cugraph


@dataclass
class CUGraphState:
    src_cp: object
    dst_cp: object
    edges_df: Optional[object] = None
    weighted_graph: Optional[object] = None


def build_cugraph_state(graph: gt.Graph) -> CUGraphState:
    edges = graph.get_edges()
    src = edges[:, 0].astype(np.int32)
    dst = edges[:, 1].astype(np.int32)
    src_cp = cp.asarray(src, dtype=cp.int64)
    dst_cp = cp.asarray(dst, dtype=cp.int64)
    return CUGraphState(
        src_cp=src_cp,
        dst_cp=dst_cp,
    )


def _build_weighted_graph(state: CUGraphState, weights):
    weights_cp = cp.asarray(weights, dtype=cp.float64)
    if state.edges_df is None:
        state.edges_df = cudf.DataFrame(
            {
                "src": cudf.Series(state.src_cp),
                "dst": cudf.Series(state.dst_cp),
                "weight": cudf.Series(weights_cp),
            }
        )
    else:
        state.edges_df["weight"] = cudf.Series(weights_cp)
    if hasattr(cugraph, "DiGraph"):
        graph = cugraph.DiGraph()
    elif hasattr(cugraph, "Graph"):
        graph = cugraph.Graph(directed=True)
    else:
        raise RuntimeError("Unsupported cuGraph API: neither DiGraph nor Graph is available")
    graph.from_cudf_edgelist(state.edges_df, source="src", destination="dst", edge_attr="weight", renumber=False)
    return graph


def benchmark_cugraph_sssp_runtime(
    state: CUGraphState,
    sources: np.ndarray,
    weights: np.ndarray,
) -> float:
    # Build and cache graph/dataframe once; update only weights between calls.
    if state.weighted_graph is None:
        state.weighted_graph = _build_weighted_graph(state, weights)
    else:
        weights_cp = cp.asarray(weights, dtype=cp.float64)
        state.edges_df["weight"] = cudf.Series(weights_cp)
        edf = state.weighted_graph.edgelist.edgelist_df
        edf["weight"] = state.edges_df["weight"]
    weighted_graph = state.weighted_graph
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for source in sources:
        cugraph.sssp(weighted_graph, source=int(source))
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start
