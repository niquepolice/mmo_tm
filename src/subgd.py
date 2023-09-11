import networkx as nx
import numpy as np
import time
from abc import ABC, abstractmethod
from src.min_cost_concurrent_flow import solve_min_cost_concurrent_flow

from tqdm import tqdm

from src.shortest_paths_gt import flows_on_shortest_gt, get_graphtool_graph


class TrafficModel(ABC):
    def __init__(self, nx_graph: nx.Graph, traffic_mat: np.ndarray):
        self.nx_graph = nx_graph
        self.graph = get_graphtool_graph(nx_graph)
        self.traffic_mat = traffic_mat

    def flows_on_shortest(self, dual_costs: np.ndarray) -> np.ndarray:
        weights = self.graph.new_edge_property("double")
        weights.a = self.graph.ep.costs.a + dual_costs

        return flows_on_shortest_gt(self.graph, self.traffic_mat, weights)

    @abstractmethod
    def primal(self, flows_e: np.ndarray) -> float:
        ...

    @abstractmethod
    def dual(self, dual_costs: np.ndarray, flows_subgd_e: np.ndarray) -> float:
        ...


class SDModel(TrafficModel):
    """Dualized on capacity constraints"""
    def primal(self, flows_e: np.ndarray) -> float:
        return float(self.graph.ep.costs.a @ flows_e)
    
    def dual(self, dual_costs: np.ndarray, flows_subgd_e: np.ndarray) -> float:
        return float((self.graph.ep.costs.a + dual_costs) @ flows_subgd_e - dual_costs @ self.graph.ep.bandwidths.a)
    
    def constraints_violation_l1(self, flows_e: np.ndarray) -> float:
        return np.maximum(0, flows_e - self.graph.ep.bandwidths.a).sum()

    def constraints_violation(self, flows_e: np.ndarray) -> float:
        return np.linalg.norm(np.maximum(0, flows_e - self.graph.ep.bandwidths.a))

    def dual_subgradient(self, flows_subgd_e: np.ndarray) -> np.ndarray:
        return flows_subgd_e - self.graph.ep.bandwidths.a

    def solve_cvxpy(self, **solver_kwargs):
        """solver_kwargs: arguments for cvxpy's problem.solve()"""
        flows_ie, costs, potentials, nonneg_duals = solve_min_cost_concurrent_flow(self.nx_graph, self.traffic_mat, **solver_kwargs)
        return flows_ie, costs, potentials, nonneg_duals 


def subgd_mincost_mcf(
    model: TrafficModel,
    R: float,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 1000000,
    max_time: float = 0,  # max execution time in seconds, 0 for no limit
) -> tuple:
    time_start = time.time()
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    flows_averaged_e = np.zeros((num_edges))

    dual_costs = np.zeros(num_edges)

    dgap_log = []
    cons_log = []

    S = 0  # sum of stepsizes

    optimal = False

    rng = range(1000000) if max_iter is None else tqdm(range(max_iter))
    for k in rng:
        if max_time > 0 and time.time() - time_start > max_time:
            break

        # inlined subgradient calculation with paths set saving
        flows_subgd_e = model.flows_on_shortest(dual_costs)
        # flows_subgd_e = flows_subgd_ije.sum(axis=(0, 1))
        subgd = -model.dual_subgradient(flows_subgd_e)  # grad of varphi = -dual

        h = R / (k + 1) ** 0.5 / np.linalg.norm(subgd)

        dual_val = model.dual(dual_costs, flows_subgd_e)

        flows_averaged_e = (S * flows_averaged_e + h * flows_subgd_e) / (S + h)
        S += h

        # flows_averaged_e = flows_averaged_e.sum(axis=(0, 1))

        dgap_log.append(model.primal(flows_averaged_e) - dual_val)
        cons_log.append(model.constraints_violation(flows_averaged_e))

        if dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        dual_costs = np.maximum(0, dual_costs - h * subgd)

    return dual_costs, flows_averaged_e, dgap_log, cons_log, optimal


def ustm_mincost_mcf(
    model: TrafficModel,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 10000,
    max_time: float = 0,  # max execution time in seconds, 0 for no limit
    stop_by_crit: bool = True,
) -> tuple:
    time_start = time.time()
    
    dgap_log = []
    cons_log = []
    A_log = []
    
    A_prev = 0.0
    t_start = np.zeros(model.graph.num_edges())  # dual costs w
    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization
    grad_sum_prev = np.zeros(len(t_start))
    
    def func_grad_flows(dual_costs: np.ndarray):
        """func = varphi = -dual"""
        flows_subgd_e = model.flows_on_shortest(dual_costs)
        # flows_subgd_e = flows_subgd_ije.sum(axis=(0, 1))
        dual_grad = model.dual_subgradient(flows_subgd_e)
        return -model.dual(dual_costs, flows_subgd_e), -dual_grad, flows_subgd_e
         
    # these flows will not be used in averaging (multiplied by 0)
    _, grad_y, flows_averaged_e = func_grad_flows(y_start)  
    L_value = np.linalg.norm(grad_y) / 10
    
    A = u = t = y = None
    inner_iters_num = 0

    optimal = False 
    rng = range(1000000) if max_iter is None else tqdm(range(max_iter))
    for k in rng:
        if max_time > 0 and time.time() - time_start > max_time:
            break
        while True:
            inner_iters_num += 1
    
            alpha = 0.5 / L_value + (0.25 / L_value**2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha
    
            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, flows_y = func_grad_flows(y)
            grad_sum = grad_sum_prev + alpha * grad_y
            
            u = np.maximum(0, y_start - grad_sum)
            
            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _ = func_grad_flows(t)
            
            lvalue = func_t
            rvalue = (func_y + np.dot(grad_y, t - y) + 0.5 * L_value * np.sum((t - y) ** 2) + 
    #                     0.5 * alpha / A * eps_abs )  # because, in theory, noise accumulates
                        0.5 * eps_abs)
    #                    0.1 * eps_abs)
            
            if lvalue <= rvalue:
                break
            else:
                L_value *= 2
    
            assert L_value < np.inf

        A_prev = A
        L_value /= 2
    
        t_prev = t
        u_prev = u
        grad_sum_prev = grad_sum
        
        gamma = alpha / A
        flows_averaged_e = flows_averaged_e * (1 - gamma) + flows_y * gamma
        # flows_averaged_e = flows_averaged_ije.sum(axis=(0, 1))

        dgap_log.append(model.primal(flows_averaged_e) + func_t)
        cons_log.append(model.constraints_violation(flows_averaged_e))
        A_log.append(A)
    
        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

    return t, flows_averaged_e, dgap_log, cons_log, A_log, optimal
