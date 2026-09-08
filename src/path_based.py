import networkx as nx
import time
import numpy as np
from tqdm import tqdm
from typing import Optional

from src.models import BeckmannModel
from src.models import maybe_create_and_get_times_ep

from graph_tool.topology import shortest_distance

import numba
from numba.core import types

def extract_path_from_pred_path(
    source: int,
    target: int,
    pred_map_arr: np.ndarray,
    edge_to_ind: numba.typed.Dict,
) -> np.ndarray:
    path = []
    flow = 0
    v = target
    while v != source:
        v_pred = pred_map_arr[v]
        # print(v_pred, v)
        path.insert(0, edge_to_ind[(v_pred, v)])
        v = v_pred
    return path

def shift_flows(
    model: BeckmannModel,
    flows: np.ndarray,
    flow_shift: float,
    path_from: list,
    path_to: list
):
    if len(path_from)>0:
        for i in path_from:
            flows[i] -= flow_shift
            flows[i] = max(flows[i], 0)

    for i in path_to:
        flows[i] += flow_shift
        
    times = model.tau(flows)
    return flows, times

def update_travel_time_for_path_set(
    times: np.ndarray,
    paths: list,
    basic_path: list
):
    '''
    update travel time for a set of paths
    output: list of travel time for each path, and the basic path id (if has)
    '''
    used_flag = False
    used_path_cost = []
    basic_path_id = None
    basic_path_cost = 0

    i = 0
    for path in paths:
        if not used_flag:
            if set(path) == set(basic_path):
                used_flag = True
                basic_path_id = i
            i += 1
                
        cost = 0
        for j in path:
            cost += times[j]
        used_path_cost.append(cost)
        
    for j in basic_path:
        basic_path_cost += times[j]        
        
    return used_path_cost, basic_path_id, basic_path_cost

def get_sum_of_gradient(
    model: BeckmannModel,
    flows: np.ndarray,
    p1: list,
    p0: list,
    n_edges: int
):
    '''
    get the sum of derivatives of links.
    '''
    # get the links in sp1 but not in sp0
    link_sp1 = set(p1)
    link_sp0 = set(p0)
    # find the links that contribute to either sp1 or sp0
    link_contributed = link_sp1.symmetric_difference(link_sp0)

    # get a masked edge list
    masked_edge_cost = np.zeros(n_edges)
    for link in link_contributed:
        masked_edge_cost[link] = 1
    dev_sum=np.sum(masked_edge_cost * model.diff_tau(flows))

    return dev_sum

def get_limits(
    basic_path: list,
    non_basic_path: list,
    flows: np.ndarray,
    caps: np.ndarray
):
    link_basic = set(basic_path)
    link_nonbasic = set(non_basic_path)
    link_contributed = list(link_basic.difference(link_nonbasic))
    
    if len(link_contributed) == 0:
        return float("inf")
    else:
        return max(np.min(caps[link_contributed] - flows[link_contributed]) - 1e-8, 0)

def pb_gradproj_ta(
    beckmann_model: BeckmannModel,
    iters: int,
    zero_flow_eps: float = 1e-7,
    log_period: int=500,
    log_max_diff: bool = False,
    check_criterion: int = -1,
    solution_flows: Optional[np.ndarray] = None,
    use_capacity: bool = False,
    time_limit: int = 1_000_000
):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    
    corrs = beckmann_model.correspondences
    traffic_mat, sources, targets = corrs.traffic_mat, corrs.sources, corrs.targets
    pqs = [[[] for _ in range(targets.size)]  for _ in range(sources.size)]
    list_flows = [[[] for _ in range(targets.size)]  for _ in range(sources.size)]    
        
    edges_arr = beckmann_model.graph.get_edges()
    edge_to_ind = numba.typed.Dict.empty(key_type=types.UniTuple(types.int64, 2), value_type=numba.core.types.int64)
    for i, edge in enumerate(edges_arr):
        edge_to_ind[tuple(edge)] = i
        
    need_log = log_period > 0
    opt_time = []
    flows_dist_log = []
    primal_log = []

    times = beckmann_model.graph.ep.free_flow_times.a.copy()
    flows = np.zeros(A.shape[1])
    assert(times.shape == flows.shape)

    if use_capacity:
        caps = beckmann_model.graph_props[3]

    start = time.time()

    for si in range(sources.size):
        s = sources[si]               
        _, pred_map = shortest_distance(beckmann_model.graph, source=s, target=targets,\
                                weights=maybe_create_and_get_times_ep(beckmann_model.graph, times), pred_map=True)
        for ti in range(targets.size):
            t = targets[ti]

            # print(np.linalg.norm(flows))
            path = extract_path_from_pred_path(s, t, np.array(pred_map.a), edge_to_ind)
            pqs[si][ti].append(path)
            list_flows[si][ti].append(traffic_mat[si][ti])
            flows, times = shift_flows(beckmann_model, flows, traffic_mat[si][ti], [], path)

    pbar = tqdm(range(1, iters))
    for i in pbar:
        for si in range(sources.size):
            s = sources[si]
            _, pred_map = shortest_distance(beckmann_model.graph, source=s, target=targets,\
                                    weights=maybe_create_and_get_times_ep(beckmann_model.graph, times), pred_map=True)
            for ti in range(targets.size):
                t = targets[ti]

                basic_path = extract_path_from_pred_path(s, t, np.array(pred_map.a), edge_to_ind)
                used_path_cost, basic_path_id, basic_path_cost = update_travel_time_for_path_set(times, pqs[si][ti], basic_path)
                basic_path_flow = 0 if basic_path_id is None else list_flows[si][ti][basic_path_id]
                    
                for j in range(len(pqs[si][ti])):
                    if j != basic_path_id:
                        non_basic_path = pqs[si][ti][j]
                        non_basic_path_cost = used_path_cost[j]
                        path_cost_diff = max(non_basic_path_cost - basic_path_cost, 0) # non-negative
                        dev_sum = get_sum_of_gradient(beckmann_model, flows, basic_path, non_basic_path, A.shape[1])
                        flow_to_shift = min(list_flows[si][ti][j],path_cost_diff/dev_sum)
                        if use_capacity:
                            flow_to_shift = min(flow_to_shift, get_limits(basic_path, non_basic_path, flows, caps))
                        list_flows[si][ti][j] -= flow_to_shift
                        basic_path_flow += flow_to_shift

                        flows, times = shift_flows(beckmann_model, flows, flow_to_shift, non_basic_path, basic_path)
                        
                if basic_path_id is None: # it's a new path:
                    list_flows[si][ti].append(basic_path_flow)
                    pqs[si][ti].append(basic_path)
                else:
                    list_flows[si][ti][basic_path_id] = basic_path_flow
            
            beckmann_model.graph.ep.times.a = times

        for s in range(sources.size):
            for t in range(targets.size):
                indices_to_remove = [i for i, flow in enumerate(list_flows[s][t]) if flow < zero_flow_eps]
                for i in sorted(indices_to_remove, reverse=True):
                    del pqs[s][t][i]
                    del list_flows[s][t][i]
                
        if need_log and i % log_period == 0:
            opt_time.append(time.time() - start)
            primal_log.append(float(beckmann_model.primal(flows)))
            if solution_flows is not None:
                flow_dist = np.linalg.norm(flows - solution_flows) if not log_max_diff else np.max(np.abs(flows - solution_flows))
                flows_dist_log.append(float(flow_dist))    
                
        if check_criterion > 0:
            primal_val = float(beckmann_model.primal(flows))
            if primal_val < check_criterion:
                break

    return (flows, log_period) + ((opt_time, primal_log, start) + ((flows_dist_log,) if flows_dist_log else ()) if log_period > 0 else ())
