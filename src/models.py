import networkx as nx
import numpy as np
from abc import ABC, abstractmethod

from src.commons import Correspondences
from src.cvxpy_solvers import solve_min_cost_concurrent_flow
from src.newton import newton

from src.shortest_paths_gt import flows_on_shortest_gt, get_graphtool_graph, get_graph_props


class TrafficModel(ABC):
    def __init__(self, nx_graph: nx.DiGraph, correspondences: Correspondences):
        self.nx_graph = nx_graph
        self.graph = get_graphtool_graph(nx_graph)
        self.correspondences = correspondences

    def flows_on_shortest(self, times_e: np.ndarray) -> np.ndarray:
        """Get edge flows distribution for given edge costs"""
        if "times" not in self.graph.edge_properties:
            times = self.graph.new_edge_property("double")
            self.graph.ep["times"] = times

        self.graph.ep.times.a = times_e

        return flows_on_shortest_gt(self.graph, self.correspondences, self.graph.ep.times)

    def capacity_violation(self, flows_e: np.ndarray) -> float:
        """Could be ignored for Backmann model"""
        return np.linalg.norm(np.maximum(0, flows_e - self.graph.ep.capacities.a))

    @abstractmethod
    def primal(self, flows_e: np.ndarray) -> float:
        ...

    @abstractmethod
    def dual(self, times_e: np.ndarray, flows_subgd_e: np.ndarray) -> float:
        ...

    @abstractmethod
    def dual_subgradient(self, times_e: np.ndarray, flows_subgd_e: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        ...

    # TODO ??
    # @abstractmethod
    # def log(self, history) -> np.ndarray:
        ...

    @abstractmethod
    def solve_cvxpy(self, **solver_kwargs):
        ...


class BeckmannModel(TrafficModel):
    # def __init__(self, nx_graph: nx.DiGraph, traffic_mat: np.ndarray):
    #     super().__init__(nx_graph, traffic_mat)
    #     fft = self.graph.ep.free_flow_times.a
    #
    #     assert np.all(fft > 0), "zero free flow times are not supported yet"
    #     # other checks ?

    """Dualized on the constraint that flows respect correspondences"""
    def tau(self, flows_e):
        fft, mu, rho, caps = get_graph_props(self.graph)

        return fft * (1 + rho * (flows_e / caps) ** (1 / mu))

    def tau_inv(self, times_e):
        fft, mu, rho, caps = get_graph_props(self.graph)

        return caps * ((times_e / fft - 1) / rho) ** mu

    def sigma(self, flows_e) -> np.ndarray:
        fft, mu, rho, caps = get_graph_props(self.graph)

        return fft * flows_e * (1 + (rho / (1 + 1 / mu)) * (flows_e / caps) ** (1 / mu))

    def sigma_star(self, times_e) -> np.ndarray:
        fft, mu, rho, caps = get_graph_props(self.graph)

        dt = np.maximum(0, times_e - fft)

        return caps * (dt / (fft * rho)) ** mu * dt / (1 + mu)

    def primal(self, flows_e: np.ndarray) -> float:
        return self.sigma(flows_e).sum() 

    def dual(self, times_e, flows_subgd) -> np.ndarray:
        return -self.sigma_star(times_e).sum() + times_e @ flows_subgd
    
    def dual_subgradient(self, times_e: np.ndarray, flows_subgd_e: np.ndarray) -> np.ndarray:
        return -self.tau_inv(times_e) + flows_subgd_e

    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        fft, mu, rho, caps = get_graph_props(self.graph)

        # rewrite t - t_0 + stepsize * tau_inv(t) = 0 as x - x_0 + a x^mu = 0
        x_0 = (times_e - fft) / (fft * rho)
        a = stepsize * caps / (fft * rho)

        x = newton(x_0_arr=x_0, a_arr=a, mu_arr=mu)

        return fft * (rho * x + 1)


    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""
        flows_ie, costs, potentials, nonneg_duals = solve_beckmann(self.nx_graph, self.correspondences.traffic_mat,
                                                                   **solver_kwargs)
        return flows_ie, costs, potentials, nonneg_duals 


class SDModel(TrafficModel):
    """Dualized on the capacity constraints"""
    def primal(self, flows_e: np.ndarray) -> float:
        return float(self.graph.ep.free_flow_times.a @ flows_e)
    
    def dual(self, times_e: np.ndarray, flows_subgd_e: np.ndarray) -> float:
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return float(times_e @ flows_subgd_e - (times_e - fft) @ caps)
    
    def dual_subgradient(self, times_e: np.ndarray, flows_subgd_e: np.ndarray) -> np.ndarray:
        return flows_subgd_e - self.graph.ep.capacities.a

    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        """prox_{eta h}(t - eta * Ф'(t)) = proj(t - eta * Q'(t))"""
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return np.maximum(fft, times_e - stepsize * caps)

    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""

        flows_ie, costs, potentials, nonneg_duals = solve_min_cost_concurrent_flow(self.nx_graph,
                                                                                   self.correspondences.node_traffic_mat,
                                                                                   **solver_kwargs)
        return flows_ie.sum(axis=0), self.graph.ep.free_flow_times.a + costs, potentials, nonneg_duals

