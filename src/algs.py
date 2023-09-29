from typing import Optional

import numpy as np
import time
from tqdm import tqdm

from src.models import TrafficModel, BeckmannModel, TwostageModel, SDModel, Model


def frank_wolfe(
        model: BeckmannModel,
        eps_abs: float,
        max_iter: int = 10000,  # 0 for no limit (some big number)
        times_start: Optional[np.ndarray] = None,
        stop_by_crit: bool = True,
        use_tqdm: bool = True,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows_averaged_e = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    for k in rng:
        stepsize = 2 / (k + 2)

        times_e = model.tau(flows_averaged_e)
        flows_e = model.flows_on_shortest(times_e)

        # dgap_log.append(times_e @ (flows_averaged_e - flows_e))  # FW gap
        flows_averaged_e = flows_e if k == 0 else stepsize * flows_e + (1 - stepsize) * flows_averaged_e

        dual_val = model.dual(times_e, flows_e)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows_averaged_e)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return times_e, flows_averaged_e, (dgap_log, np.array(time_log)-time_log[0]), optimal


def ustm(
    model: Model,
    eps_abs: float,
    eps_cons_abs: float = np.inf,
    max_iter: int = 10000,  # 0 for no limit (some big number)
    max_sp_calls: int = 10000,  # max shortest paths calls, dont count the first (preprocessing) call
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
) -> tuple:
    """ for primal-dual minimization of composite minus dual function -D(t) =  Ф(t) + h(t).
    subgrad Ф(t) = -flows_on_shortest(t) = -flows_subgd_e(t)"""

    dgap_log = []
    cons_log = []
    time_log = []

    A_prev = 0.0
    # fft = model.graph.ep.free_flow_times.a
    # t_start = fft.copy()  # times
    t_start = model.init_dual_point()
    y_start = u_prev = t_prev = np.copy(t_start)
    assert y_start is u_prev  # acceptable at first initialization
    grad_sum_prev = np.zeros(len(t_start))

    # grad_y = -model.flows_on_shortest(y_start)  # Ф'(y)
    _, grad_y, _ = model.func_psigrad_primal(y_start)

    L_value = np.linalg.norm(grad_y) / 10
    
    A = u = t = y = None
    optimal = False

    max_dual_func_val = -np.inf

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    for k in rng:
        inner_iters_num = 0
        while True:
            inner_iters_num += 1
    
            alpha = 0.5 / L_value + (0.25 / L_value**2 + A_prev / L_value) ** 0.5
            A = A_prev + alpha
    
            y = (alpha * u_prev + A_prev * t_prev) / A
            func_y, grad_y, primal_var_y = model.func_psigrad_primal(y)  # -model.dual(y, flows_y), -flows_y

            grad_sum = grad_sum_prev + alpha * grad_y
            
            u = model.dual_composite_prox(y_start - grad_sum, A)

            t = (alpha * u + A_prev * t_prev) / A
            func_t, _, _ = model.func_psigrad_primal(t)

            max_dual_func_val = max(max_dual_func_val, -func_t)

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

        # primal variable can be tuple
        if k == 0:
            primal_var_averaged = primal_var_y
        elif type(primal_var_averaged) == tuple:
            primal_var_averaged = tuple(primal_var_averaged[i] * (1 - gamma) + primal_var_y[i] * gamma for i in range(len(primal_var_averaged)))
        else:
            primal_var_averaged = primal_var_averaged * (1 - gamma) + primal_var_y * gamma


        # consider every model.flows_on_shortest() call for fair algs comparison
        if type(primal_var_averaged) == tuple:
            primal, cons = model.primal(*primal_var_averaged), model.capacity_violation(*primal_var_averaged)
        else:
            primal, cons = model.primal(primal_var_averaged), model.capacity_violation(primal_var_averaged)
        dgap_log += [primal - max_dual_func_val] * (inner_iters_num * 2)
        cons_log += [cons] * (inner_iters_num * 2)
        time_log += [time.time()] * (inner_iters_num * 2)

        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        if len(dgap_log) > max_sp_calls:
            break

    return t, primal_var_averaged, (dgap_log, cons_log, np.array(time_log)-time_log[0]), optimal


def subgd(
        model: TrafficModel,
        R: float,
        eps_abs: float,
        eps_cons_abs: float,
        max_iter: int = 1000000,  # 0 for no limit (some big number)
        stop_by_crit: bool = True,
        use_tqdm: bool = True,
) -> tuple:
    num_nodes, num_edges = model.graph.num_vertices(), model.graph.num_edges()
    flows_averaged_e = np.zeros(num_edges)

    fft = model.graph.ep.free_flow_times.a
    times_e = fft.copy()

    dgap_log = []
    cons_log = []

    S = 0  # sum of stepsizes

    optimal = False

    max_dual_func_val = -np.inf

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter), disable=not use_tqdm)
    for k in rng:
        # inlined subgradient calculation with paths set saving
        flows_subgd_e = model.flows_on_shortest(times_e)

        h = R / (k + 1) ** 0.5 / np.linalg.norm(flows_subgd_e)

        dual_val = model.dual(times_e, flows_subgd_e)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        flows_averaged_e = (S * flows_averaged_e + h * flows_subgd_e) / (S + h)
        S += h

        dgap_log.append(model.primal(flows_averaged_e) - max_dual_func_val)
        cons_log.append(model.capacity_violation(flows_averaged_e))

        if stop_by_crit and dgap_log[-1] <= eps_abs and cons_log[-1] <= eps_cons_abs:
            optimal = True
            break

        times_e = model.dual_composite_prox(times_e + h * flows_subgd_e, h)

    return times_e, flows_averaged_e, (dgap_log, cons_log), optimal


def cyclic(
        model: TwostageModel,
        eps_abs: float,  # dgap tolerance
        traffic_assigment_eps_abs: float,
        traffic_assigment_max_iter: int = 100,
        # entropy_eps: Union[float, None] = None,
        max_iter: int = 20,
        stop_by_crit: bool = True,
) -> tuple:
    """For twostage model"""

    dgap_log = []
    cons_log = []
    time_log = []

    rng = range(1_000_000) if max_iter == 0 else tqdm(range(max_iter))

    traffic_model = model.traffic_model
    distance_mat_averaged = model.distance_mat(traffic_model.graph.ep.free_flow_times.a.copy())

    optimal = False
    times_e = None
    for k in rng:
        traffic_mat, lambda_l, lambda_w = model.solve_entropy_model(distance_mat_averaged)

        traffic_model.set_traffic_mat(traffic_mat)
        # isinstance fails after autoreload
        if traffic_model.__class__.__name__ == "BeckmannModel":
            times_e, flows_e, inner_dgap_log, *_ = frank_wolfe(traffic_model, eps_abs=traffic_assigment_eps_abs,
                                                             max_iter=traffic_assigment_max_iter, times_start=times_e, use_tqdm=False)
        elif traffic_model.__class__.__name__ == "SDModel":
            times_e, flows_e, inner_dgap_log, *_ = ustm(traffic_model, eps_abs=traffic_assigment_eps_abs, max_iter=traffic_assigment_max_iter, use_tqdm=False)
        else:
            assert False, f"traffic_model has wrong class name : {type(traffic_model.__class__.__name__)}"

        # print(f"inner iters={len(inner_dgap_log)}")
        distance_mat = model.distance_mat(times_e)

        dgap_log.append(model.primal(flows_e, traffic_mat) - model.dual(times_e, lambda_l, lambda_w, distance_mat))
        cons_log.append(traffic_model.capacity_violation(flows_e))
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

        distance_mat_averaged = (distance_mat_averaged + distance_mat) / 2  # average to fix oscillations

    return times_e, flows_e, traffic_mat, (dgap_log, cons_log, np.array(time_log)-time_log[0]), optimal
