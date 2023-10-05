import cvxpy as cp
import networkx as nx
import numpy as np
from typing import Optional


def solve_min_cost_concurrent_flow(
    graph: nx.DiGraph, traffic_mat: np.ndarray, **solver_kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    capacities = np.array(
        list(nx.get_edge_attributes(graph, "capacities").values()), dtype=np.float32
    )
    costs = np.array(
        list(nx.get_edge_attributes(graph, "free_flow_times").values()),
        dtype=np.float32,
    )

    flows = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    prob = cp.Problem(
        cp.Minimize(cp.sum(flows, axis=1) @ costs),
        [
            cp.sum(flows, axis=1) <= capacities,
            (incidence_mat @ flows).T == -traffic_lapl,
            flows >= 0,
        ],
    )
    prob.solve(**solver_kwargs)
    flows = flows.value if flows is not None else None
    costs, potentials, nonneg_duals = [cons.dual_value for cons in prob.constraints]
    return flows, costs, potentials, nonneg_duals


def get_max_traffic_mat_mul(
    graph: nx.Graph, traffic_mat: np.ndarray, **solver_kwargs
) -> Optional[float]:
    graph = nx.DiGraph(graph)
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    bandwidth = np.array(
        list(nx.get_edge_attributes(graph, "capacities").values()), dtype=np.float32
    )

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [
            cp.sum(flow, axis=1) <= bandwidth,
            (incidence_mat @ flow).T == -gamma * traffic_lapl,
            flow >= 0,
        ],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    gamma = gamma.value if flow is not None else None
    return gamma
