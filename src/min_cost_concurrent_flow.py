import cvxpy as cp
import networkx as nx
import numpy as np


def solve_min_cost_concurrent_flow(
    graph: nx.DiGraph, traffic_mat: np.ndarray, **solver_kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    bandwidth = np.array(list(nx.get_edge_attributes(graph, "bandwidth").values()), dtype=np.float32)
    cost = np.array(list(nx.get_edge_attributes(graph, "cost").values()), dtype=np.float32)

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    prob = cp.Problem(
        cp.Minimize(cp.sum(flow, axis=1) @ cost),
        [cp.sum(flow, axis=1) <= bandwidth, (incidence_mat @ flow).T == -traffic_lapl, flow >= 0],
    )
    prob.solve(**solver_kwargs)
    flow = flow.value if flow is not None else None
    costs, potentials, nonneg_duals = [cons.dual_value for cons in prob.constraints]
    return flow, costs, potentials, nonneg_duals
