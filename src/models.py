from typing import Union, Optional

import graph_tool as gt
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod

from src.commons import Correspondences
from src.cvxpy_solvers import solve_min_cost_concurrent_flow
from src.newton import newton

from src.shortest_paths_gt import (
    flows_on_shortest_gt,
    get_graphtool_graph,
    get_graph_props,
    distance_mat_gt,
)
from src.sinkhorn import Sinkhorn
import src.sinkhorn as sinkhorn


def maybe_create_and_get_times_ep(
    graph: gt.Graph, times_e: np.ndarray
) -> gt.EdgePropertyMap:
    if "times" not in graph.edge_properties:
        times = graph.new_edge_property("double")
        graph.ep["times"] = times

    graph.ep.times.a = times_e
    return graph.ep.times


class Model(ABC):
    """Primal-dual model with composite step (USTM-compatible)"""

    @abstractmethod
    def primal(self, *args) -> float:
        ...

    @abstractmethod
    def dual(self, *args) -> float:
        ...

    @abstractmethod
    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        ...

    @abstractmethod
    def capacity_violation(self, *args) -> float:
        ...

    @abstractmethod
    def solve_cvxpy(self, **solver_kwargs):
        ...

    @abstractmethod
    def init_dual_point(self) -> np.ndarray:
        ...

    @abstractmethod
    def func_psigrad_primal(self, times) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns minus dual func value, gradient of \Psi (non-composite term),
        and corresponding primal variable value
           Needed for USTM
        """
        ...


class TrafficModel(Model, ABC):
    def __init__(self, nx_graph: nx.DiGraph, correspondences: Correspondences):
        self.nx_graph = nx_graph
        self.graph = get_graphtool_graph(nx_graph)
        self.correspondences = correspondences

    def flows_on_shortest(
        self, times_e: np.ndarray, return_distance_mat: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get edge flows distribution for given edge costs, if all agents use the shortest paths
        May also return distance matrix if the flag is set
        """

        return flows_on_shortest_gt(
            self.graph,
            self.correspondences,
            maybe_create_and_get_times_ep(self.graph, times_e),
            return_distance_mat,
        )

    def dual(self, times_e, flows_subgd) -> float:
        return float(times_e @ flows_subgd - self.composite(times_e))

    @abstractmethod
    def composite(self, times_e: np.ndarray) -> float:
        """Composite term in dual function, denoted by h"""
        ...

    def capacity_violation(self, flows_e: np.ndarray) -> float:
        """Could be ignored for Beckmann model"""
        return np.linalg.norm(np.maximum(0, flows_e - self.graph.ep.capacities.a))

    def set_traffic_mat(self, traffic_mat: np.ndarray):
        """For two-stage models"""
        self.correspondences.traffic_mat = traffic_mat
        self.correspondences.node_traffic_mat = None

    def init_dual_point(self) -> np.ndarray:
        return self.graph.ep.free_flow_times.a

    def func_psigrad_primal(self, times) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns minus dual func value, gradient of it, and corresponding primal variable value
        Needed for USTM
        """
        flows = self.flows_on_shortest(times)
        return -self.dual(times, flows), -flows, flows


class BeckmannModel(TrafficModel):
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

    def composite(self, times_e: np.ndarray) -> float:
        return self.sigma_star(times_e).sum()

    def dual_subgradient(
        self, times_e: np.ndarray, flows_subgd_e: np.ndarray
    ) -> np.ndarray:
        return flows_subgd_e - self.tau_inv(times_e)

    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        fft, mu, rho, caps = get_graph_props(self.graph)

        # rewrite t - t_0 + stepsize * tau_inv(t) = 0 as x - x_0 + a x^mu = 0
        x_0 = (times_e - fft) / (fft * rho)
        a = stepsize * caps / (fft * rho)

        x = newton(x_0_arr=x_0, a_arr=a, mu_arr=mu)

        return fft * (rho * x + 1)

    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""
        # TODO: implement
        return None
        # flows_ie, costs, potentials, nonneg_duals = solve_beckmann(self.nx_graph, self.correspondences.traffic_mat,
        #                                                            **solver_kwargs)
        # return flows_ie, costs, potentials, nonneg_duals


class SDModel(TrafficModel):
    """Dualized on the capacity constraints"""

    def primal(self, flows_e: np.ndarray) -> float:
        return float(self.graph.ep.free_flow_times.a @ flows_e)

    def composite(self, times_e: np.ndarray) -> float:
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return (times_e - fft) @ caps

    def dual_subgradient(
        self, times_e: np.ndarray, flows_subgd_e: np.ndarray
    ) -> np.ndarray:
        return flows_subgd_e - self.graph.ep.capacities.a

    def dual_composite_prox(self, times_e: np.ndarray, stepsize: float) -> np.ndarray:
        """prox_{stepsize * h}(t - stepsize * Ф'(t)) = proj(t - stepsize * Q'(t))"""
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return np.maximum(fft, times_e - stepsize * caps)

    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""

        flows_ie, costs, potentials, nonneg_duals = solve_min_cost_concurrent_flow(
            self.nx_graph, self.correspondences.node_traffic_mat, **solver_kwargs
        )
        return (
            flows_ie.sum(axis=0),
            self.graph.ep.free_flow_times.a + costs,
            potentials,
            nonneg_duals,
        )


class TwostageModel(Model):
    def __init__(
        self,
        traffic_model: TrafficModel,
        departures: np.ndarray,
        arrivals: np.ndarray,
        gamma: float,
    ):
        self.traffic_model = traffic_model
        self.departures = departures
        self.arrivals = arrivals
        self.gamma = gamma
        self.sinkhorn = Sinkhorn(
            self.departures,
            self.arrivals,
            max_iter=int(1e5),
            use_numba=len(departures) > 100,
        )

        # previous solution to reuse as starting point
        # save it here, because entropy model is hidden from solver-side in case of USTM
        self.lambda_l_prev, self.lambda_w_prev = None, None

    def solve_entropy_model(
        self, distance_mat
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        traffic_mat, self.lambda_l_prev, self.lambda_w_prev = self.sinkhorn.run(
            self.gamma * distance_mat, self.lambda_l_prev, self.lambda_w_prev
        )
        return traffic_mat, self.lambda_l_prev, self.lambda_w_prev

    def distance_mat(self, times_e) -> np.ndarray:
        sources, targets = (
            self.traffic_model.correspondences.sources,
            self.traffic_model.correspondences.targets,
        )

        return distance_mat_gt(
            self.traffic_model.graph,
            sources,
            targets,
            maybe_create_and_get_times_ep(self.traffic_model.graph, times_e),
        )

    def primal(self, flows_e: np.ndarray, traffic_mat: np.ndarray) -> float:
        return (
            self.gamma * self.traffic_model.primal(flows_e)
            + (traffic_mat * np.log(traffic_mat)).sum()
        )

    def dual(
        self,
        times: np.ndarray,
        lambda_l: np.ndarray,
        lambda_w: np.ndarray,
        distance_mat: Optional[np.ndarray],
    ) -> float:
        if distance_mat is None:
            distance_mat = self.distance_mat(times)

        return -(
            sinkhorn.d_ij(lambda_l, lambda_w, self.gamma * distance_mat).sum()
            + (lambda_l * self.departures).sum()
            + (lambda_w * self.arrivals).sum()
            + self.gamma * self.traffic_model.composite(times)
        )

    def init_dual_point(self) -> np.ndarray:
        return self.traffic_model.graph.ep.free_flow_times.a

    def func_psigrad_primal(self, times) -> tuple[float, np.ndarray, tuple]:
        """Returns minus dual func value, gradient of it, and corresponding primal variable value"""
        distance_mat = self.distance_mat(times)
        # TODO: reuse previous lambdas
        traffic_mat, lambda_l, lambda_w = self.solve_entropy_model(distance_mat)

        self.traffic_model.set_traffic_mat(traffic_mat)
        flows = self.traffic_model.flows_on_shortest(times)

        return (
            -self.dual(times, lambda_l, lambda_w, distance_mat),
            -flows,
            (flows, traffic_mat),
        )

    def dual_composite_prox(self, times: np.ndarray, stepsize: float) -> np.ndarray:
        """prox_{stepsize * h}(t - stepsize * Ф'(t)) = proj(t - stepsize * Q'(t))"""
        return self.traffic_model.dual_composite_prox(times, stepsize)

    def capacity_violation(self, flows: np.ndarray, traffic_mat: np.ndarray) -> float:
        return self.traffic_model.capacity_violation(flows)

    def solve_cvxpy(self, **solver_kwargs):
        pass
