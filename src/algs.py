import numpy as np
import time
from tqdm import tqdm

from src.models import TrafficModel


def frank_wolfe():
    pass


def ustm(
    model: TrafficModel,
    eps_abs: float,
    eps_cons_abs: float,
    max_iter: int = 10000,
    max_time: float = 0,  # max execution time in seconds, 0 for no limit
    stop_by_crit: bool = True,
) -> tuple:
    """ for composite function Ф + h.
    subgrad Ф(t) = -flows_on_shortest(t) = -flows_subgd_e(t)"""
    time_start = time.time()
    
    dgap_log = []
    cons_log = []
    A_log = []
    
    A_prev = 0.0
    fft = model.graph.ep.free_flow_times.a
    t_start = fft.copy()  # times
    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization
    grad_sum_prev = np.zeros(len(t_start))
    
    # def func_grad(times_e: np.ndarray):
    # #     """func = -dual"""
    #     flows_subgd_e = model.flows_on_shortest(times_e)
    #     dual_grad = model.dual_subgradient(times_e, flows_subgd_e)
    #     return -model.dual(times_e, flows_subgd_e), -dual_grad, flows_subgd_e
         
    grad_y = -model.flows_on_shortest(y_start)  # Ф'(y)
    L_value = np.linalg.norm(grad_y) / 10
    
    flows_averaged_e = A = u = t = y = None
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
            flows_y = model.flows_on_shortest(y)
            func_y, grad_y = -model.dual(y, flows_y), -flows_y
            grad_sum = grad_sum_prev + alpha * grad_y
            
            # u = np.maximum(fft, y_start - grad_sum)
            u = model.dual_composite_prox(y_start - grad_sum, A)
            
            t = (alpha * u + A_prev * t_prev) / A
            # func_t = -func_grad_flows(t)
            func_t = -model.dual(t, model.flows_on_shortest(t))

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
        flows_averaged_e = flows_y if k == 0 else flows_averaged_e * (1 - gamma) + flows_y * gamma

        dgap_log.append(model.primal(flows_averaged_e) + func_t)
        cons_log.append(model.capacity_violation(flows_averaged_e))
        A_log.append(A)
    
        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

    return t, flows_averaged_e, dgap_log, cons_log, A_log, optimal


def subgd(
        model: TrafficModel,
        R: float,
        eps_abs: float,
        eps_cons_abs: float,
        max_iter: int = 1000000,
        max_time: float = 0,  # max execution time in seconds, 0 for no limit
) -> tuple:
    time_start = time.time()
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    flows_averaged_e = np.zeros(num_edges)

    fft = model.graph.ep.free_flow_times.a
    times_e = fft.copy()

    dgap_log = []
    cons_log = []

    S = 0  # sum of stepsizes

    optimal = False

    rng = range(1000000) if max_iter is None else tqdm(range(max_iter))
    for k in rng:
        if 0 < max_time < time.time() - time_start:
            break

        # inlined subgradient calculation with paths set saving
        flows_subgd_e = model.flows_on_shortest(times_e)

        h = R / (k + 1) ** 0.5 / np.linalg.norm(flows_subgd_e)

        dual_val = model.dual(times_e, flows_subgd_e)

        flows_averaged_e = (S * flows_averaged_e + h * flows_subgd_e) / (S + h)
        S += h

        dgap_log.append(model.primal(flows_averaged_e) - dual_val)
        cons_log.append(model.capacity_violation(flows_averaged_e))

        if dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        times_e = model.dual_composite_prox(times_e + h * flows_subgd_e, h)

    return times_e, flows_averaged_e, dgap_log, cons_log, optimal


