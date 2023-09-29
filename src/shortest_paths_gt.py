from typing import Union

import numpy as np
import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numba
from numba.core import types

from src.commons import Correspondences


@numba.njit
def sum_flows_from_tree(source: int, targets: np.ndarray, pred_map_arr: np.ndarray, traffic_mat_row: np.ndarray,
                        edge_to_ind: numba.typed.Dict) -> np.ndarray:
    num_edges = len(edge_to_ind)
    flows_e = np.zeros(num_edges)
    for j, v in enumerate(targets):  # j = index of target in traffic_mat
        corr = traffic_mat_row[j]
        while v != source:
            v_pred = pred_map_arr[v]
            flows_e[edge_to_ind[(v_pred, v)]] += corr
            v = v_pred
    return flows_e


def distance_mat_gt(graph: gt.Graph, sources: np.ndarray, targets: np.ndarray, weights: gt.EdgePropertyMap):
    distance_mat = np.zeros((sources.size, targets.size))
    for i, source in enumerate(sources):  # i = index of source in traffic_mat
        dist_map = shortest_distance(graph, source=source, target=targets, weights=weights, pred_map=False)
        distance_mat[i] = dist_map

    return distance_mat


def flows_on_shortest_gt(graph: gt.Graph, corrs: Correspondences, weights: gt.EdgePropertyMap,
                         return_distance_mat: bool = False) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Returns flows on edges for each ij-pair
    (obtained from flows on shortest paths w.r.t given weights(costs))
    Also may return distance matrix for given weights
    """
    num_nodes, num_edges = graph.num_vertices(), graph.num_edges()

    traffic_mat, sources, targets = corrs.traffic_mat, corrs.sources, corrs.targets

    edges_arr = graph.get_edges()
    edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64)
    for i, edge in enumerate(edges_arr):
        edge_to_ind[tuple(edge)] = i

    flows_on_shortest_e = np.zeros(num_edges)

    if return_distance_mat:
        distance_mat = np.zeros((sources.size, targets.size))
    for i, source in enumerate(sources):  # i = index of source in traffic_mat
        dist_map, pred_map = shortest_distance(graph, source=source, target=targets, weights=weights, pred_map=True)
        flows_on_shortest_e += sum_flows_from_tree(
            source=source,
            targets=targets,
            pred_map_arr=np.array(pred_map.a),
            traffic_mat_row=traffic_mat[i],
            edge_to_ind=edge_to_ind,
        )
        if return_distance_mat:
            distance_mat[i] = dist_map

    return (flows_on_shortest_e, distance_mat) if return_distance_mat else flows_on_shortest_e


def get_graphtool_graph(nx_graph: nx.Graph) -> gt.Graph:
    """Creates `gt_graph: graph_tool.Graph` from `nx_graph: nx.Graph`.
    Nodes in `gt_graph` are labeled by their indices in `nx_graph.edges()` instead of their labels
    (possibly of `str` type) in `nx_graph`"""

    def edge_dict_to_arr(d: dict, edge_to_ind: dict) -> np.ndarray:
        arr = np.zeros(len(d))
        for edge, value in d.items():
            arr[edge_to_ind[edge]] = value
        return arr

    nx_edges = nx_graph.edges()
    nx_edge_to_ind = dict(zip(nx_edges, list(range(len(nx_graph.edges())))))

    edge_attribute_names = list(list(nx_graph.edges(data=True))[0][-1].keys())
    edge_attributes = [edge_dict_to_arr(nx.get_edge_attributes(nx_graph, attr_name), nx_edge_to_ind)
                       for attr_name in edge_attribute_names]

    nx_nodes = list(nx_graph.nodes())
    edge_list = []
    for i, e in enumerate(nx_graph.edges()):
        edge_list.append((*[nx_nodes.index(v) for v in e],  *[attr[i] for attr in edge_attributes]))

    gt_graph = gt.Graph(edge_list, eprops=[(attr_name, "double") for attr_name in edge_attribute_names])

    return gt_graph


def get_graph_props(graph: gt.Graph) -> tuple:
    fft = graph.ep.free_flow_times.a
    mu = graph.ep.mu.a
    rho = graph.ep.rho.a
    caps = graph.ep.capacities.a

    return fft, mu, rho, caps

