import networkx as nx
import numpy as np
from tqdm import tqdm

from src.models import BeckmannModel
from src.salim import SaddleOracle


def opt_metric(f, y, grad_f, A):
    lagrange_grad_f = grad_f + A.T @ y
    tol = 1e-4
    lagrange_grad_f[f < tol] = np.minimum(0, lagrange_grad_f[f < tol])
    return np.linalg.norm(lagrange_grad_f)


def salim_ta(beckmann_model: BeckmannModel, iters: int, mu: float, L: float, lam1: float, lam2: float):
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

    cons_log = []
    opt_log = []

    for _ in tqdm(range(iters)):
        x_g = tau * x + (1 - tau) * x_f
        grad_x_g = beckmann_model.grad_fei(x_g)
        x_half = np.maximum(0, (x - eta * (grad_x_g - alpha * x_g + A.T @ y)) / (1 + eta * alpha))
        y += theta * (A @ x_half - b)
        x_prev = x
        x = np.maximum(0, (x - eta * (grad_x_g - alpha * x_g + A.T @ y)) / (1 + eta * alpha))
        x_f = x_g + (2 * tau) / (2 - tau) * (x - x_prev)

        cons_log.append(np.linalg.norm(A @ x - b))
        opt_log.append(opt_metric(x_g, y, grad_x_g, A))

    return x, y, cons_log, opt_log


def chambolle_pock_ta(beckmann_model: BeckmannModel, iters: int, gamma: float, nu: float):
    A = nx.incidence_matrix(beckmann_model.nx_graph, oriented=True).todense()
    Ld = SaddleOracle(beckmann_model, None, None, None).Bmul(beckmann_model.correspondences.traffic_mat).T

    n_nodes, n_edges = A.shape

    theta = 1
    f_bar = f = np.zeros((n_edges, Ld.shape[1]))
    y = np.zeros(Ld.shape)
    z = np.zeros(f.shape[0])

    cons_log = []
    opt_log = []

    for i in tqdm(range(iters)):
        y = y + gamma * (A @ f_bar + Ld)
        z = beckmann_model.dual_composite_prox(z + gamma * f_bar.sum(axis=1), stepsize=gamma)
        f_prev = f
        f = np.maximum(0, f - nu * (A.T @ y + z[:, np.newaxis]))

        f_bar = f + theta * (f - f_prev)

        # if not i % 10:
        cons_log.append(np.linalg.norm(A @ f + Ld))
        opt_log.append(opt_metric(f, y, beckmann_model.grad_fei(f), A))

    return f, y, cons_log, opt_log
