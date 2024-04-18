from typing import Union, Optional

import graph_tool as gt
import networkx as nx
import numpy as np
from abc import ABC, abstractmethod

from src.commons import Correspondences
from src.cvxpy_solvers import solve_min_cost_concurrent_flow, solve_beckmann_model_cp
from src.newton import newton

from src.shortest_paths_gt import (
    flows_on_shortest_gt,
    get_graphtool_graph,
    get_graph_props,
    distance_mat_gt,
)

# import src.sinkhorn_gpu as sinkhorn
import src.sinkhorn as sinkhorn


def maybe_create_and_get_times_ep(graph: gt.Graph, times: np.ndarray) -> gt.EdgePropertyMap:
    if "times" not in graph.edge_properties:
        times_ep = graph.new_edge_property("double")
        graph.ep["times"] = times_ep

    graph.ep.times.a = times
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
    def dual_composite_prox(self, times: np.ndarray, stepsize: float) -> np.ndarray:
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
        """Returns minus dual func value, gradient of Psi (non-composite term),
        and corresponding primal variable value
           Needed for USTM
        """
        ...


class TrafficModel(Model, ABC):
    def __init__(self, nx_graph: nx.DiGraph, correspondences: Correspondences):
        self.nx_graph = nx_graph
        self.graph = get_graphtool_graph(nx_graph)
        self.correspondences = correspondences

        fft, mu, rho, caps = get_graph_props(self.graph)

        mu_mask = mu == np.inf
        rho_mask = rho == 0
        fft_mask = fft == 0
        # the degenerate case : cost does not idepend on the flow
        self.all_cases = mu_mask + rho_mask + fft_mask

    def flows_on_shortest(
        self, times: np.ndarray, return_distance_mat: bool = False
    ) -> Union[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Get edge flows distribution for given edge costs, if all agents use the shortest paths
        May also return distance matrix if the flag is set
        """

        return flows_on_shortest_gt(
            self.graph,
            self.correspondences,
            maybe_create_and_get_times_ep(self.graph, times),
            return_distance_mat,
        )

    def dual(self, times, flows_subgd) -> float:
        return float(times @ flows_subgd - self.composite(times))

    @abstractmethod
    def composite(self, times: np.ndarray) -> float:
        """Composite term in dual function, denoted by h"""
        ...

    def capacity_violation(self, flows: np.ndarray) -> float:
        """Could be ignored for Beckmann model"""
        return np.linalg.norm(np.maximum(0, flows - self.graph.ep.capacities.a))

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

    def __init__(self, nx_graph: nx.DiGraph, correspondences: Correspondences):
        super().__init__(nx_graph, correspondences)
        self.graph_props = get_graph_props(self.graph)  # for some reason, this call is slow, so cache results

    def tau(self, flows):
        fft, mu, rho, caps = self.graph_props

        result = np.empty(len(mu))
        result[self.all_cases] = fft[self.all_cases] * (1 + rho[self.all_cases])
        result[~self.all_cases] = fft[~self.all_cases] * (
            1 + rho[~self.all_cases] * (flows[~self.all_cases] / caps[~self.all_cases]) ** (1 / mu[~self.all_cases])
        )
        return result

    def diff_tau(self, flows):
        fft, mu, rho, caps = get_graph_props(self.graph)

        result = np.empty(len(mu))

        result[self.all_cases] = 0
        result[~self.all_cases] = (
            (1.0 / mu[~self.all_cases])
            * fft[~self.all_cases]
            * rho[~self.all_cases]
            * np.power(flows[~self.all_cases], (1.0 / mu[~self.all_cases]) - 1.0)
            / np.power(caps[~self.all_cases], 1.0 / mu[~self.all_cases])
        )
        return result

    def tau_inv(self, times):
        fft, mu, rho, caps = self.graph_props
        result = np.empty(len(mu))

        result[self.all_cases] = 0
        result[~self.all_cases] = (
            caps[~self.all_cases]
            * ((times[~self.all_cases] / fft[~self.all_cases] - 1) / rho[~self.all_cases]) ** mu[~self.all_cases]
        )
        return result

    def sigma(self, flows) -> np.ndarray:
        fft, mu, rho, caps = self.graph_props
        result = np.empty(len(mu))

        result[self.all_cases] = fft[self.all_cases] * flows[self.all_cases] * (1 + rho[self.all_cases])
        result[~self.all_cases] = (
            fft[~self.all_cases]
            * flows[~self.all_cases]
            * (
                1
                + (rho[~self.all_cases] / (1 + 1 / mu[~self.all_cases]))
                * (flows[~self.all_cases] / caps[~self.all_cases]) ** (1 / mu[~self.all_cases])
            )
        )
        return result

    def sigma_star(self, times) -> np.ndarray:
        fft, mu, rho, caps = self.graph_props

        dt = np.maximum(0, times - fft)

        result = np.empty(len(mu))

        result[self.all_cases] = 0
        result[~self.all_cases] = (
            caps[~self.all_cases]
            * (dt[~self.all_cases] / (fft[~self.all_cases] * rho[~self.all_cases])) ** mu[~self.all_cases]
            * dt[~self.all_cases]
            / (1 + mu[~self.all_cases])
        )
        return result

    def primal(self, flows: np.ndarray) -> float:
        return float(self.sigma(flows).sum())

    def composite(self, times: np.ndarray) -> float:
        return self.sigma_star(times).sum()

    def dual_subgradient(self, times: np.ndarray, flows_subgd: np.ndarray) -> np.ndarray:
        return flows_subgd - self.tau_inv(times)

    def dual_composite_prox(self, times: np.ndarray, stepsize: float) -> np.ndarray:
        fft, mu, rho, caps = get_graph_props(self.graph)

        # rewrite t - t_0 + stepsize * tau_inv(t) = 0 as x - x_0 + a x^mu = 0

        mask = ~self.all_cases  # non-degenerated edges
        x_0 = (times - fft)[mask] / (fft * rho)[mask]
        a = stepsize * caps[mask] / (fft * rho)[mask]

        x = newton(x_0_arr=x_0, a_arr=a, mu_arr=mu)

        result = fft.copy()
        result[mask] = fft[mask] * (rho[mask] * x + 1)

        return result

    def solve_cvxpy(self, **solver_kwargs) -> np.ndarray:
        """solver_kwargs: arguments for cvxpy's problem.solve()"""
        flows_ei, potentials = solve_beckmann_model_cp(
            self.correspondences.traffic_mat, self.nx_graph, **solver_kwargs
        )
        assert flows_ei is not None

        return flows_ei.sum(axis=1)


class SDModel(TrafficModel):
    """Dualized on the capacity constraints"""

    def primal(self, flows: np.ndarray) -> float:
        return float(self.graph.ep.free_flow_times.a @ flows)

    def composite(self, times: np.ndarray) -> float:
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return (times - fft) @ caps

    def dual_subgradient(self, times: np.ndarray, flows_subgd: np.ndarray) -> np.ndarray:
        return flows_subgd - self.graph.ep.capacities.a

    def dual_composite_prox(self, times: np.ndarray, stepsize: float) -> np.ndarray:
        """prox_{stepsize * h}(t - stepsize * Ф'(t)) = proj(t - stepsize * Q'(t))"""
        fft = self.graph.ep.free_flow_times.a
        caps = self.graph.ep.capacities.a
        return np.maximum(fft, times - stepsize * caps)

    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""

        flows_ei, costs, potentials, nonneg_duals = solve_min_cost_concurrent_flow(
            self.nx_graph, self.correspondences.node_traffic_mat, **solver_kwargs
        )
        return (
            flows_ei.sum(axis=1),
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

        # previous solution to reuse as starting point
        # save it here, because entropy model is hidden from solver-side in case of USTM
        self.lambda_l_prev, self.lambda_w_prev = None, None

    def solve_entropy_model(self, distance_mat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calls sinkhorn to solve linear-entropy problem. Saves previous solution to reuse as starting point"""
        snkh = sinkhorn.Sinkhorn(
            departures=self.departures,
            arrivals=self.arrivals,
            max_iter=int(1e5),
        )
        traffic_mat, self.lambda_l_prev, self.lambda_w_prev = snkh.run(
            self.gamma * distance_mat, self.lambda_l_prev, self.lambda_w_prev
        )
        return traffic_mat, self.lambda_l_prev, self.lambda_w_prev

    def distance_mat(self, times) -> np.ndarray:
        sources, targets = (
            self.traffic_model.correspondences.sources,
            self.traffic_model.correspondences.targets,
        )

        return distance_mat_gt(
            self.traffic_model.graph,
            sources,
            targets,
            maybe_create_and_get_times_ep(self.traffic_model.graph, times),
        )

    def primal(self, flows: np.ndarray, traffic_mat: np.ndarray) -> float:
        zero_mask = traffic_mat == 0  # to make x * ln(x) = 0
        tmp = traffic_mat.copy()
        tmp[zero_mask] = 1
        return self.gamma * self.traffic_model.primal(flows) + (traffic_mat * np.log(tmp)).sum()

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
