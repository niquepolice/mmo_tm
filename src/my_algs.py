from typing import Optional

import numpy as np
import time
from tqdm import tqdm

from src.models import TrafficModel, BeckmannModel, TwostageModel, Model



def conjugate_frank_wolfe(
    model: BeckmannModel,
    eps_abs: float,
    max_iter: int = 100,  # 0 for no limit (some big number)
    times_start: Optional[np.ndarray] = None,
    stop_by_crit: bool = True,
    use_tqdm: bool = True,
) -> tuple:
    """One iteration == 1 shortest paths call"""

    optimal = False

    # init flows, not used in averaging
    if times_start is None:
        times_start = model.graph.ep.free_flow_times.a.copy()
    flows_averaged = model.flows_on_shortest(times_start)

    max_dual_func_val = -np.inf
    dgap_log = []
    time_log = []

    rng = (
        range(1_000_000)
        if max_iter == 0
        else tqdm(range(max_iter), disable=not use_tqdm)
    )
    for k in rng:
        stepsize = 2 / (k + 2)

        times = model.tau(flows_averaged)
        flows = model.flows_on_shortest(times)

        # dgap_log.append(times @ (flows_averaged - flows))  # FW gap
        flows_averaged = (
            flows if k == 0 else stepsize * flows + (1 - stepsize) * flows_averaged
        )

        dual_val = model.dual(times, flows)
        max_dual_func_val = max(max_dual_func_val, dual_val)

        # equal to FW gap if dual_val == max_dual_func_val
        primal = model.primal(flows_averaged)
        dgap_log.append(primal - max_dual_func_val)
        time_log.append(time.time())

        if stop_by_crit and dgap_log[-1] <= eps_abs:
            optimal = True
            break

    return (
        times,
        flows_averaged,
        (dgap_log, np.array(time_log) - time_log[0]),
        optimal,
    )
