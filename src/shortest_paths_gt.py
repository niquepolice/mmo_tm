import numpy as np
import networkx as nx
import graph_tool as gt
from graph_tool.topology import shortest_distance
import numba
from numba.core import types


@numba.njit
def sum_flows_from_tree(source: int, targets: np.ndarray, pred_map_arr: np.ndarray, traffic_mat: np.ndarray,
                        edge_to_ind: numba.typed.Dict) -> np.ndarray:
    num_edges = len(edge_to_ind)
    flows_e = np.zeros(num_edges)
    for v in targets:
        corr = traffic_mat[source, v]
        while v != source:
            v_pred = pred_map_arr[v]
            flows_e[edge_to_ind[(v_pred, v)]] += corr
            v = v_pred
    return flows_e


def flows_on_shortest_gt(graph: gt.Graph, traffic_mat: np.ndarray, weights: gt.EdgePropertyMap) -> np.ndarray:
    """Returns flows on edges for each ij-pair
    (obtained from flows on shortest paths w.r.t costs induced by dual_costs)"""
    num_nodes, num_edges = graph.num_vertices(), graph.num_edges()

    edges_arr = graph.get_edges()
    edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64)
    for i, edge in enumerate(edges_arr):
        edge_to_ind[tuple(edge)] = i

    flows_on_shortest_e = np.zeros(num_edges)
    targets = np.arange(num_nodes)
    for source in range(num_nodes):
        _, pred_map = shortest_distance(graph, source=source, target=targets, weights=weights, pred_map=True)
        flows_on_shortest_e += sum_flows_from_tree(
            source=source,
            targets=targets,
            pred_map_arr=np.array(pred_map.a),
            traffic_mat=traffic_mat,
            edge_to_ind=edge_to_ind,
        )

    return flows_on_shortest_e


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

    nx_bandwidths = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "bandwidth"), nx_edge_to_ind)
    nx_costs = edge_dict_to_arr(nx.get_edge_attributes(nx_graph, "cost"), nx_edge_to_ind)

    nx_nodes = list(nx_graph.nodes())
    edge_list = []
    for i, e in enumerate(nx_graph.edges()):
        edge_list.append((*[nx_nodes.index(v) for v in e],  nx_bandwidths[i], nx_costs[i]))

    gt_graph = gt.Graph(edge_list, eprops=[("bandwidths", "double"), ("costs", "double")])

    return gt_graph
