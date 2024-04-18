from typing import Optional

import cvxpy as cp
import networkx as nx
import numpy as np


def solve_min_cost_concurrent_flow(
    graph: nx.DiGraph, traffic_mat: np.ndarray, **solver_kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    capacities = np.array(list(nx.get_edge_attributes(graph, "capacities").values()), dtype=np.float32)
    costs = np.array(
        list(nx.get_edge_attributes(graph, "free_flow_times").values()),
        dtype=np.float32,
    )

    flows_ei = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    prob = cp.Problem(
        cp.Minimize(cp.sum(flows_ei, axis=1) @ costs),
        [
            cp.sum(flows_ei, axis=1) <= capacities,
            (incidence_mat @ flows_ei).T == -traffic_lapl,
            flows_ei >= 0,
        ],
    )
    prob.solve(**solver_kwargs)
    flows_ei = flows_ei.value if flows_ei is not None else None
    costs, potentials, nonneg_duals = [cons.dual_value for cons in prob.constraints]
    return flows_ei, costs, potentials, nonneg_duals


def solve_entropy_model_cp(
    departures: np.ndarray, arrivals: np.ndarray, distance_mat: np.ndarray, gamma: float
) -> np.ndarray:
    gammaT_ij = distance_mat * gamma
    d_ij = cp.Variable(shape=distance_mat.shape)

    obj = cp.Minimize(cp.sum(cp.multiply(gammaT_ij, d_ij)) - cp.sum(cp.entr(d_ij)))

    constraints = [cp.sum(d_ij, axis=1) == departures, cp.sum(d_ij, axis=0) == arrivals]

    prob = cp.Problem(obj, constraints)
    prob.solve()

    return d_ij.value


def solve_beckmann_model_cp(traffic_mat: np.ndarray, graph: nx.DiGraph, **solver_kwargs) -> tuple:
    # TODO: test on networks where can_pass_through_zones=False

    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    capacities = np.array(list(nx.get_edge_attributes(graph, "capacities").values()), dtype=np.float32)
    ffts = np.array(
        list(nx.get_edge_attributes(graph, "free_flow_times").values()),
        dtype=np.float32,
    )
    rhos = np.array(list(nx.get_edge_attributes(graph, "rho").values()), dtype=np.float32)
    mus = np.array(list(nx.get_edge_attributes(graph, "mu").values()), dtype=np.float32)

    flows_ei = cp.Variable((len(graph.edges), traffic_mat.shape[0]), nonneg=True)
    flows_e = cp.sum(flows_ei, axis=1)
    # flows_e = cp.Variable(len(graph.edges), nonneg=True)

    # in cvxpy, power should be scalar, so use loop
    sigmas = [
        ffts[e]
        * (
            flows_e[e]
            + (rhos[e] / (1 + 1 / mus[e])) * (cp.pos(flows_e[e]) ** (1 + 1 / mus[e]) / capacities[e] ** (1 / mus[e]))
        )
        for e in range(len(graph.edges))
    ]
    objective = cp.Minimize(cp.sum(sigmas))
    # objective = cp.Minimize(0)

    prob = cp.Problem(
        objective,
        [
            (incidence_mat @ flows_ei).T == -traffic_lapl,
        ],
    )
    prob.solve(**solver_kwargs)
    flows_ei = flows_ei.value if flows_ei is not None else None
    potentials = [cons.dual_value for cons in prob.constraints]
    return flows_ei, potentials


def get_max_traffic_mat_mul(graph: nx.Graph, traffic_mat: np.ndarray, **solver_kwargs) -> Optional[float]:
    graph = nx.DiGraph(graph)
    traffic_lapl = np.diag(traffic_mat.sum(axis=1)) - traffic_mat
    incidence_mat = nx.incidence_matrix(graph, oriented=True).todense()

    capacities = np.array(list(nx.get_edge_attributes(graph, "capacities").values()), dtype=np.float32)

    flow = cp.Variable((len(graph.edges), traffic_mat.shape[0]))
    gamma = cp.Variable()
    prob = cp.Problem(
        cp.Maximize(gamma),
        [
            cp.sum(flow, axis=1) <= capacities,
            (incidence_mat @ flow).T == -gamma * traffic_lapl,
            flow >= 0,
        ],
    )
    prob.solve(**solver_kwargs)

    if prob.status != "optimal":
        gamma = None

    gamma = gamma.value if flow is not None else None
    return gamma
