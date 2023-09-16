import networkx as nx
import numpy as np


def flows_on_shortest_nx(graph: nx.Graph, traffic_mat: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Returns flows on edges for each ij-pair
    (obtained from flows on shortest paths w.r.t costs induced by dual_costs)"""
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    edge_to_ind = dict(zip(graph.edges(), list(range(len(graph.edges())))))
    node_to_ind = dict(zip(graph.nodes(), list(range(len(graph.nodes())))))

    def costs_func(node_from, node_to, edge_attributes: dict) -> float:
        edge = (node_from, node_to)
        return costs[edge_to_ind[edge]]

    paths_ij = nx.all_pairs_dijkstra(graph, weight=costs_func)

    flows_subgd_ije = np.zeros((n, n, m))
    for vertex_from, (len_j, path_j) in paths_ij:
        for vertex_to, path in path_j.items():
            i, j = node_to_ind[vertex_from], node_to_ind[vertex_to]
            edges_in_path = nx.path_graph(path).edges()

            for edge in edges_in_path:
                flows_subgd_ije[i, j, edge_to_ind[edge]] += traffic_mat[i, j]

    return flows_subgd_ije
