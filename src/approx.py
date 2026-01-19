import networkx as nx
import time
import numpy as np
from tqdm import tqdm
from typing import Optional

from src.models import BeckmannModel
from src.salim import SaddleOracle
import cvxpy as cp

def seq_quad(beckmann_model: BeckmannModel, iters: int, need_log=True, log_max_diff = False, return_full=False, solution_flows: Optional[np.ndarray] = None, x_0_start=None, start_time=None):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    Ld = SaddleOracle(beckmann_model, None, None, None).Bmul(beckmann_model.correspondences.traffic_mat).T

    _, n_edges = A.shape

    x_0 = np.zeros((n_edges, Ld.shape[1]))
    if x_0_start is not None:
        x_0 = x_0_start
        
    AATi = np.linalg.pinv(A @ A.T)

    x = cp.Variable((n_edges, Ld.shape[1]))
        
    times = []
    cons_log = []
    flows_dist_log = []
    primal_log = []

    if start_time is None:
        start = time.time()
    else:
        start = start_time
    
    pbar = tqdm(range(iters)) if need_log else range(iters)
    for i in pbar:
        grad = beckmann_model.tau(np.maximum(x_0.sum(axis=1), 0))
        hess = 0.5 * beckmann_model.diff_tau(np.maximum(x_0.sum(axis=1), 0))
        bt = -A @ x_0 - Ld

        x.value = np.maximum(-x_0, 0)
        prob = cp.Problem(cp.Minimize(hess @ (cp.sum(x, axis=1)**2) + grad @ cp.sum(x, axis=1)),
                        [x >= -x_0,
                        A @ x == bt])

        prob.solve(solver="CLARABEL", max_iter = 10 + 5*i, warm_start=False, equilibrate_enable=False, iterative_refinement_enable=False)
        x_0 += x.value

                
        if need_log:
            cons_log.append(float(np.linalg.norm(A.T @ (AATi @ (A @ x_0 + Ld)))))
            times.append(time.time() - start)
            primal_log.append(float(beckmann_model.primal(np.maximum(x_0.sum(axis=1), 0))))
            if solution_flows is not None:
                flow_dist = np.linalg.norm(x_0.sum(axis=1) - solution_flows) if not log_max_diff else np.max(np.abs(x_0.sum(axis=1) - solution_flows))
                flows_dist_log.append(float(flow_dist))

            postfix = {}
            if solution_flows is not None:
                postfix["sol dist"] = flows_dist_log[-1]
            postfix["primal"] = primal_log[-1]
            pbar.set_postfix(postfix)
    
    solution = list(np.astype(x_0.sum(axis=1), float)) if not return_full else x_0
    
    if not need_log:
        return solution
    else:
        return (solution, cons_log, times, primal_log) + ((flows_dist_log,) if flows_dist_log else ())


def seq_quad_chp(beckmann_model: BeckmannModel, iters: int, need_log=True, log_max_diff = False, return_full=False, solution_flows: Optional[np.ndarray] = None, x_0_start=None, start_time=None, iters_quad: int = 60000):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    Ld = SaddleOracle(beckmann_model, None, None, None).Bmul(beckmann_model.correspondences.traffic_mat).T

    _, n_edges = A.shape

    x_0 = np.zeros((n_edges, Ld.shape[1]))
    if x_0_start is not None:
        x_0 = x_0_start
        
    AATi = np.linalg.pinv(A @ A.T)

    svals = np.linalg.svdvals(A)
    lam1 = svals[0]
    nu = 0.99 * 32 / lam1
    gamma = 0.98 * 0.99 / (nu * lam1**2)

    x = np.zeros((n_edges, Ld.shape[1]))
        
    times = []
    cons_log = []
    flows_dist_log = []
    primal_log = []

    if start_time is None:
        start = time.time()
    else:
        start = start_time
    
    pbar = tqdm(range(iters)) if need_log else range(iters)
    for i in pbar:
        grad = beckmann_model.tau(np.maximum(x_0.sum(axis=1), 0))
        hess = beckmann_model.diff_tau(np.maximum(x_0.sum(axis=1), 0)) + 1e-9
        bt = -A @ x_0 - Ld

        f_bar = f = np.maximum(-x_0, 0)
        y = np.zeros(Ld.shape)
        z = np.zeros(f.shape[0])
        for j in range(iters_quad):
            y = y + gamma * (A @ f_bar - bt)
            z = (hess * (z + gamma * f_bar.sum(axis=1)) + gamma * grad) / (gamma + hess)
            f_prev = f
            f = np.maximum(-x_0, f - nu * (A.T @ y + z[:, np.newaxis]))
                        
            f_bar = 2*f - f_prev
            
        x_0 += f

                
        if need_log:
            cons_log.append(float(np.linalg.norm(A.T @ (AATi @ (A @ x_0 + Ld)))))
            times.append(time.time() - start)
            primal_log.append(float(beckmann_model.primal(np.maximum(x_0.sum(axis=1), 0))))
            if solution_flows is not None:
                flow_dist = np.linalg.norm(x_0.sum(axis=1) - solution_flows) if not log_max_diff else np.max(np.abs(x_0.sum(axis=1) - solution_flows))
                flows_dist_log.append(float(flow_dist))

            postfix = {}
            if solution_flows is not None:
                postfix["sol dist"] = flows_dist_log[-1]
            postfix["primal"] = primal_log[-1]
            pbar.set_postfix(postfix)
    
    solution = list(np.astype(x_0.sum(axis=1), float)) if not return_full else x_0
    
    if not need_log:
        return solution
    else:
        return (solution, cons_log, times, primal_log) + ((flows_dist_log,) if flows_dist_log else ())
