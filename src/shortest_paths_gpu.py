from dataclasses import dataclass
from typing import Optional, Union

import graph_tool as gt
import numpy as np

from src.commons import Correspondences

try:
    import cupy as cp
    import cudf
    import cugraph
except Exception:
    cp = None
    cudf = None
    cugraph = None


@dataclass
class CUGraphState:
    src: np.ndarray
    dst: np.ndarray
    num_vertices: int
    num_edges: int
    src_cp: object
    dst_cp: object
    edge_key_sorted_cp: object
    edge_id_sorted_cp: object


def cugraph_is_available() -> bool:
    return cp is not None and cudf is not None and cugraph is not None


def build_cugraph_state(graph: gt.Graph) -> CUGraphState:
    if not cugraph_is_available():
        raise RuntimeError("cuGraph backend is not available")
    edges = graph.get_edges()
    src = edges[:, 0].astype(np.int32)
    dst = edges[:, 1].astype(np.int32)
    src_cp = cp.asarray(src, dtype=cp.int64)
    dst_cp = cp.asarray(dst, dtype=cp.int64)
    num_vertices = int(graph.num_vertices())
    edge_key_cp = src_cp * num_vertices + dst_cp
    edge_id_cp = cp.arange(edge_key_cp.size, dtype=cp.int32)
    perm_cp = cp.argsort(edge_key_cp)
    edge_key_sorted_cp = edge_key_cp[perm_cp]
    edge_id_sorted_cp = edge_id_cp[perm_cp]
    return CUGraphState(
        src=src,
        dst=dst,
        num_vertices=num_vertices,
        num_edges=int(graph.num_edges()),
        src_cp=src_cp,
        dst_cp=dst_cp,
        edge_key_sorted_cp=edge_key_sorted_cp,
        edge_id_sorted_cp=edge_id_sorted_cp,
    )


def _build_weighted_graph(state: CUGraphState, weights):
    if not cugraph_is_available():
        raise RuntimeError("cuGraph backend is not available")
    weights_cp = cp.asarray(weights, dtype=cp.float64)
    edges_df = cudf.DataFrame(
        {
            "src": cudf.Series(state.src_cp),
            "dst": cudf.Series(state.dst_cp),
            "weight": cudf.Series(weights_cp),
        }
    )
    if hasattr(cugraph, "DiGraph"):
        graph = cugraph.DiGraph()
    elif hasattr(cugraph, "Graph"):
        try:
            graph = cugraph.Graph(directed=True)
        except TypeError:
            raise RuntimeError("cuGraph Graph() without directed=True is unsupported for this directed-graph workload")
    else:
        raise RuntimeError("Unsupported cuGraph API: neither DiGraph nor Graph is available")
    graph.from_cudf_edgelist(edges_df, source="src", destination="dst", edge_attr="weight", renumber=False)
    return graph


def _full_sssp_arrays_cp(sssp_df, num_vertices: int):
    vertices = cp.asarray(sssp_df["vertex"].values, dtype=cp.int64)
    distance_sparse = cp.asarray(sssp_df["distance"].values, dtype=cp.float64)
    predecessor_sparse = cp.asarray(sssp_df["predecessor"].fillna(-1).astype(np.int64).values, dtype=cp.int64)
    distance = cp.full(num_vertices, cp.inf, dtype=cp.float64)
    predecessor = cp.full(num_vertices, -1, dtype=cp.int64)
    distance[vertices] = distance_sparse
    predecessor[vertices] = predecessor_sparse
    max_float = np.finfo(np.float64).max / 2
    distance = cp.where(distance >= max_float, cp.inf, distance)
    return distance, predecessor


def _edge_id_for_vertex_cp(state: CUGraphState, predecessor):
    vertices = cp.arange(state.num_vertices, dtype=cp.int64)
    valid_mask = predecessor >= 0
    vertex_key = cp.full(state.num_vertices, -1, dtype=cp.int64)
    vertex_key[valid_mask] = predecessor[valid_mask] * state.num_vertices + vertices[valid_mask]

    edge_id_for_vertex = cp.full(state.num_vertices, -1, dtype=cp.int32)
    idx = cp.searchsorted(state.edge_key_sorted_cp, vertex_key)
    mask = valid_mask & (idx < state.edge_key_sorted_cp.size)
    if not bool(cp.any(mask)):
        return edge_id_for_vertex
    masked_vertices = cp.nonzero(mask)[0]
    cand_idx = idx[masked_vertices]
    matched = state.edge_key_sorted_cp[cand_idx] == vertex_key[masked_vertices]
    if bool(cp.any(matched)):
        good_vertices = masked_vertices[matched]
        good_idx = cand_idx[matched]
        edge_id_for_vertex[good_vertices] = state.edge_id_sorted_cp[good_idx]
    return edge_id_for_vertex


def _accumulate_flows_bincount_cp(
    state: CUGraphState,
    source: int,
    targets: np.ndarray,
    distances,
    predecessor,
    traffic_mat_row: np.ndarray,
):
    edge_id_for_vertex = _edge_id_for_vertex_cp(state, predecessor)
    demand = cp.zeros(state.num_vertices, dtype=cp.float64)
    targets_cp = cp.asarray(targets, dtype=cp.int64)
    traffic_cp = cp.asarray(traffic_mat_row, dtype=cp.float64)
    valid_targets = cp.isfinite(distances[targets_cp]) & (traffic_cp != 0)
    demand[targets_cp[valid_targets]] = traffic_cp[valid_targets]

    edge_flow = cp.zeros(state.num_edges, dtype=cp.float64)
    for _ in range(state.num_vertices):
        cur = cp.nonzero(demand != 0)[0]
        if cur.size == 0:
            break
        eids = edge_id_for_vertex[cur]
        valid_eids = eids >= 0
        if bool(cp.any(valid_eids)):
            edge_flow += cp.bincount(
                eids[valid_eids].astype(cp.int64),
                weights=demand[cur[valid_eids]],
                minlength=state.num_edges,
            )
        pred_cur = predecessor[cur]
        valid_pred = pred_cur >= 0
        if not bool(cp.any(valid_pred)):
            break
        next_demand = cp.zeros_like(demand)
        next_demand += cp.bincount(
            pred_cur[valid_pred].astype(cp.int64),
            weights=demand[cur[valid_pred]],
            minlength=state.num_vertices,
        )
        demand = next_demand
        demand[source] = 0
    return edge_flow


def distance_mat_gpu(
    state: CUGraphState,
    sources: np.ndarray,
    targets: np.ndarray,
    weights,
    return_cupy: bool = False,
):
    weighted_graph = _build_weighted_graph(state, weights)
    distance_mat_cp = cp.zeros((sources.size, targets.size), dtype=cp.float64)
    targets_cp = cp.asarray(targets, dtype=cp.int64)
    for i, source in enumerate(sources):
        sssp_df = cugraph.sssp(weighted_graph, source=int(source))
        distance, _ = _full_sssp_arrays_cp(sssp_df, state.num_vertices)
        distance_mat_cp[i] = distance[targets_cp]
    return distance_mat_cp if return_cupy else cp.asnumpy(distance_mat_cp)


def flows_on_shortest_gpu(
    state: CUGraphState,
    corrs: Correspondences,
    weights,
    return_distance_mat: bool = False,
    return_cupy: bool = False,
):
    weighted_graph = _build_weighted_graph(state, weights)
    traffic_mat, sources, targets = corrs.traffic_mat, corrs.sources, corrs.targets
    flows_on_shortest_e_cp = cp.zeros(state.num_edges, dtype=cp.float64)
    distance_mat_cp: Optional[object] = None
    if return_distance_mat:
        distance_mat_cp = cp.zeros((sources.size, targets.size), dtype=cp.float64)
    targets_cp = cp.asarray(targets, dtype=cp.int64)

    for i, source in enumerate(sources):
        sssp_df = cugraph.sssp(weighted_graph, source=int(source))
        distance, predecessor = _full_sssp_arrays_cp(sssp_df, state.num_vertices)
        flows_on_shortest_e_cp += _accumulate_flows_bincount_cp(
            source=int(source),
            targets=targets,
            distances=distance,
            predecessor=predecessor,
            traffic_mat_row=traffic_mat[i],
            state=state,
        )
        if return_distance_mat and distance_mat_cp is not None:
            distance_mat_cp[i] = distance[targets_cp]

    if return_cupy:
        return (flows_on_shortest_e_cp, distance_mat_cp) if return_distance_mat else flows_on_shortest_e_cp
    flows_np = cp.asnumpy(flows_on_shortest_e_cp)
    if return_distance_mat:
        return flows_np, cp.asnumpy(distance_mat_cp)
    return flows_np
