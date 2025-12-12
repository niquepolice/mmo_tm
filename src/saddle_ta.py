import networkx as nx
import time
import numpy as np
from tqdm import tqdm
from typing import Optional

from src.models import BeckmannModel
from src.salim import SaddleOracle
from scipy.optimize import minimize_scalar


def opt_metric(f, y, grad_f, A):
    lagrange_grad_f = A.T @ y + grad_f
    tol = 1e-4
    lagrange_grad_f[f < tol] = np.minimum(0, lagrange_grad_f[f < tol])
    return np.linalg.norm(lagrange_grad_f)


def salim_ta(beckmann_model: BeckmannModel, iters: int, mu: float, L: float, lam1: float, lam2: float, log_period=500, log_max_diff = False, solution_flows: Optional[np.ndarray] = None):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    Ld = SaddleOracle(beckmann_model, None, None, None).Bmul(beckmann_model.correspondences.traffic_mat).T
    b = -Ld

    n_nodes, n_edges = A.shape
    x_f = x = np.zeros((n_edges, Ld.shape[1]))
    y = np.zeros(Ld.shape)

    tau = min(1, 0.5 * (mu * lam1 / L / lam2) ** 0.5)
    eta = 1 / (4 * tau * L)
    theta = 1 / (eta * lam1)
    
    alpha = mu

    need_log = log_period > 0
    times = []
    cons_log = []
    opt_log = []
    flows_dist_log = []
    if solution_flows is not None:
        true_flow_norm = np.linalg.norm(solution_flows) if not log_max_diff else np.max(solution_flows)
    primal_log = []
    
    start = time.time()

    for i in tqdm(range(iters)):
        x_g = tau * x + (1 - tau) * x_f
        grad_x_g = beckmann_model.grad_fei(x_g)
        x_half = np.maximum(0, (x - eta * (grad_x_g - alpha * x_g + A.T @ y)) / (1 + eta * alpha))
        y += theta * (A @ x_half - b)
        x_prev = x
        x = np.maximum(0, (x - eta * (grad_x_g - alpha * x_g + A.T @ y)) / (1 + eta * alpha))
        x_f = x_g + (2 * tau) / (2 - tau) * (x - x_prev)

        if need_log and i % 500 == 0:
            cons_log.append(float(np.linalg.norm(A @ x - b) / lam1))
            opt_log.append(float(opt_metric(x_g, y, grad_x_g, A)))
            times.append(time.time() - start)
            primal_log.append(float(beckmann_model.primal(x.sum(axis=1))))
            if solution_flows is not None:
                flow_dist = np.linalg.norm(x.sum(axis=1) - solution_flows) if not log_max_diff else np.max(np.abs(x.sum(axis=1) - solution_flows))
                flow_dist /= true_flow_norm
                flows_dist_log.append(float(flow_dist))



    return (list(np.astype(x.sum(axis=1), float)), log_period) + ((cons_log, opt_log, times, primal_log) + ((flows_dist_log,) if flows_dist_log else ()) if log_period > 0 else ())                

def chambolle_pock_ta(beckmann_model: BeckmannModel, iters: int, beta_ts: float = 0.018, log_period=500, log_max_diff = False, return_full=False, solution_flows: Optional[np.ndarray] = None):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    Ld = SaddleOracle(beckmann_model, None, None, None).Bmul(beckmann_model.correspondences.traffic_mat).T

    AATi = np.linalg.pinv(A @ A.T)

    pr_matr = A.T @ (AATi @ A)
    pr_matr = np.eye(pr_matr.shape[0]) - pr_matr

    good_inds = np.where(np.linalg.norm(np.abs(pr_matr), axis=1) >= 1e-10)[0]
    bad_inds = np.where(np.linalg.norm(np.abs(pr_matr), axis=1) < 1e-10)[0]
    f_zeroed = -np.linalg.pinv(A) @ Ld
    f_zeroed[good_inds] = 0
    Ld_s = Ld + A @ f_zeroed
    A_s = A[np.ix_(list(range(A.shape[0])), good_inds)]
    f_sol = f_zeroed[bad_inds]
    
    orig_n_edges = A.shape[1]
    n_nodes, n_edges = A_s.shape
    r = Ld.shape[1]
    
    svals = np.linalg.svdvals(A_s)
    lam1 = svals[0]



    c_max = 0.99 * 64 / lam1
    c_min = 0.99 * 0.01 / lam1
    nu = c_max
    gamma = 0.98 * 0.99 / (nu * lam1**2)

    f_bar = f = np.zeros((n_edges, Ld.shape[1]))
    y = np.zeros(Ld.shape)
    z = np.zeros(f.shape[0])
        
    need_log = log_period > 0
    times = []
    cons_log = []
    opt_log = []
    flows_dist_log = []
    if solution_flows is not None:
        true_flow_norm = np.linalg.norm(solution_flows) if not log_max_diff else np.max(solution_flows)
    primal_log = []

    start = time.time()
    
    rm = np.random.random((A_s.shape[0], A_s.shape[0]))
    q_rm, r_rm = np.linalg.qr(rm)
    q_rm = q_rm @ np.diag(np.sign(np.diag(r_rm)))
    A_s = q_rm @ A_s
    Ld_s = q_rm @ Ld_s

    pbar = tqdm(range(iters))
    for i in pbar:
        theta = 1 # / np.sqrt(1 + 2 * beta_ts * nu)\
        # print(i)
        
        y = y + gamma * (A_s @ f_bar + Ld_s)
        z = beckmann_model.dual_composite_prox(z + gamma * f_bar.sum(axis=1), good_indices=good_inds, stepsize=gamma)
        f_prev = f
        f = np.maximum(0, f - nu * (A_s.T @ y + z[:, np.newaxis]))
                       
        f_bar = f + theta * (f - f_prev)
            
        if i > 50000 and i % 1000 == 0:
            if (i // 50000) % 2 == 1:
                nu /= (1 + beta_ts)
                gamma *= (1 + beta_ts)
            else:
                nu *= (1 + beta_ts)
                gamma /= (1 + beta_ts)
                
        if need_log and i % log_period == 0:
            f_out = np.zeros((orig_n_edges, f.shape[1]))
            f_out[good_inds] = f
            f_out[bad_inds] = f_sol

            cons_log.append(float(np.linalg.norm(A.T @ (AATi @ (A @ f_out + Ld)))))
            opt_log.append(float(opt_metric(f, y, 0, A_s)))
            times.append(time.time() - start)
            primal_log.append(float(beckmann_model.primal(f_out.sum(axis=1))))
            if solution_flows is not None:
                flow_dist = np.linalg.norm(f_out.sum(axis=1) - solution_flows) if not log_max_diff else np.max(np.abs(f_out.sum(axis=1) - solution_flows))
                flows_dist_log.append(float(flow_dist))
            # print(nu, gamma)
        if i % 1000 == 0 and need_log:
            postfix = {"nu": round(nu * lam1 / 0.99, 3), "gamma": round(gamma * lam1 / 0.99, 3), "theta": round(theta, 3)}
            if solution_flows is not None:
                postfix["sol dist"] = flows_dist_log[-1]
            postfix["primal"] = primal_log[-1]
            pbar.set_postfix(postfix)
    
    f_out = np.zeros((orig_n_edges, f.shape[1]))
    f_out[good_inds] = f
    f_out[bad_inds] = f_sol
    
    solution = list(np.astype(f_out.sum(axis=1), float)) if not return_full else f_out
    
    

    return (solution, log_period) + ((cons_log, opt_log, times, primal_log, start) + ((flows_dist_log,) if flows_dist_log else ()) if log_period > 0 else ())
