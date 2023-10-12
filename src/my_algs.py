from typing import Optional

import numpy as np
import time
from tqdm import tqdm
from scipy.optimize import minimize_scalar
from src.models import TrafficModel, BeckmannModel, TwostageModel, Model



def conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
    alpha_default: float = 0.6 ,
    linesearch : bool = False ,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = (
        range(1,1_000_000)
        if max_iter == 0
        else tqdm(range(1,max_iter+1), disable=not use_tqdm)
    )

    gamma = 1.0
    alpha = 1
    for k in rng:

        times = model.tau(flows)
        yk_FW = model.flows_on_shortest(times)

        if k > 1 :
            hessian = model.diff_tau(flows)
            denom = np.sum(( x_star - flows ) * hessian * ( yk_FW - flows_old ))
            if denom == 0 :
                alpha = 0
            else :
                alpha = np.sum(( x_star - flows ) * hessian * ( yk_FW- flows )) / np.sum(( x_star - flows ) * hessian * ( yk_FW- flows_old )) 
            
            if alpha < 0 :
                alpha = 0 
            if alpha > alpha_default :
                alpha = alpha_default

        if k == 1 :
            x_star = yk_FW
        else :
            x_star = x_star*alpha + (1-alpha)*yk_FW           

        if linesearch :
            res = minimize_scalar( lambda y : model.primal((1-y)*flows + y*x_star) , bounds = (0.0,1.0) , tol = 1e-12 )
            gamma = res.x
        else :
            gamma = 2.0/(k + 2)

        flows_old = flows
        flows = (1.0 - gamma) * flows + gamma * x_star

        dual_val = model.dual(times, flows)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows,
        (dgap_log, np.array(time_log) - time_log[0]),
        optimal,
    )
