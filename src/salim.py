import time
from typing import Optional

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.models import BeckmannModel


def entr(x):
    zero_mask = x == 0
    tmp = x.copy()
    tmp[zero_mask] = 1

    return (x * np.log(tmp)).sum()


class SaddleOracle:
    def __init__(self, traffic_model: BeckmannModel, gamma: float, l: np.ndarray, w: np.ndarray):
        self.traffic_model = traffic_model
        self.incidence_mat = nx.incidence_matrix(traffic_model.nx_graph, oriented=True).todense()
        self.gamma = gamma
        self.l, self.w = l, w
        self.n_nodes, self.n_edges = self.incidence_mat.shape
        self.n_centroids = self.traffic_model.correspondences.sources.size

    # All arguments are 2d

    # z = B.Ty, g = Af, Bd = traffic_lapl
    def Bmul(self, x: np.ndarray) -> np.ndarray:
        sources = self.traffic_model.correspondences.sources
        targets = self.traffic_model.correspondences.targets
        res = np.zeros((self.n_centroids, self.n_nodes))
        if sources[0] == targets[0]:  # no non-thru
            res[:, : self.n_centroids] = np.diag(x.sum(axis=1)) - x
        elif targets[-1] == self.n_nodes - 1:  # non-thru centroids
            res[sources, sources] = x.sum(axis=1)
            res[:, -self.n_centroids :] = -x
        else:
            assert False, "Unsupported node ordering"
        # res = np.diag(res.sum(axis=1)) - res
        return res

    def BTmul(self, y: np.ndarray) -> np.ndarray:
        # return np.diag(y)[:, np.newaxis] - y[:, :self.n_centroids]
        sources = self.traffic_model.correspondences.sources
        targets = self.traffic_model.correspondences.targets
        if sources[0] == targets[0]:  # no non-thru
            return np.diag(y)[:, np.newaxis] - y[:, : self.n_centroids]
        elif targets[-1] == self.n_nodes - 1:  # non-thru centroids
            return np.diag(y)[:, np.newaxis] - y[:, -self.n_centroids :]
        else:
            assert False, "Unsupported node ordering"

    @staticmethod
    def Kmul(x: np.ndarray) -> np.ndarray:
        return np.hstack((x.sum(axis=1), x.sum(axis=0)))

    @staticmethod
    def KTmul(y: np.ndarray) -> np.ndarray:
        y1, y2 = y[: y.size // 2], y[y.size // 2 :]
        return y1[:, np.newaxis] + y2[np.newaxis, :]

    def Amul(self, x: np.ndarray) -> np.ndarray:
        return (self.incidence_mat @ x).T

    def ATmul(self, y: np.ndarray) -> np.ndarray:
        return (y @ self.incidence_mat).T

    def grad_d(self, d: np.ndarray) -> np.ndarray:
        """of the objective in combined problem: gradient of entropy"""
        return (1 + np.log(d)) / self.gamma

    def grad_f(self, f_ei: np.ndarray) -> np.ndarray:
        """of the objective in combined problem: gradient of sum of sigmas"""
        return self.traffic_model.tau(f_ei.sum(axis=1))[:, np.newaxis]


def combined_salim(
    oracle: SaddleOracle,
    mu: float,
    L: float,
    lam1: float,
    lam2: float,
    solution_flows: Optional[np.ndarray] = None,
    solution_corrs: Optional[np.ndarray] = None,
    corrs0: Optional[np.ndarray] = None,
    flows0: Optional[np.ndarray] = None,
    eps_abs: Optional[float] = None,
    iters: int = 1000,
    plot_convergence: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Salim Kovalev 2022 without Cheb acceleration"""

    def ABmul(f, d):
        return (oracle.Amul(f) + oracle.Bmul(d)).flatten()

    def ABTmul(yAB):
        return np.vstack((oracle.ATmul(yAB), oracle.BTmul(yAB)))

    n_nodes, n_edges = oracle.incidence_mat.shape
    xf_corrs = x_corrs = np.ones((oracle.l.size, oracle.w.size)) if corrs0 is None else corrs0.copy()
    xf_flows = x_flows = np.zeros((n_edges, oracle.l.size)) if flows0 is None else flows0.copy()

    b = np.hstack((oracle.l, oracle.w))
    yK = np.zeros(b.size)
    yAB = np.zeros((oracle.l.size, n_nodes))

    # alg parameters
    n = oracle.l.size

    tau = min(1, 0.5 * (mu * lam1 / L / lam2) ** 0.5)
    eta = 1 / (4 * tau * L)
    theta = 1 / (eta * lam1)
    alpha = mu

    times = []
    log_Kcons, log_ABcons, log_opt, log_dist = [], [], [], []

    eps = 1e-6
    start = time.time()

    grad_full_prev = None
    x_full_prev = None

    Lmax = 0
    mumin = 1e10

    for i in tqdm(range(iters)):
        xg_corrs = tau * x_corrs + (1 - tau) * xf_corrs
        xg_flows = tau * x_flows + (1 - tau) * xf_flows

        dFxg_corrs = oracle.grad_d(xg_corrs)
        dFxg_flows = oracle.grad_f(xg_flows)

        grad_full = np.hstack(
            (dFxg_corrs.flatten(), (dFxg_flows[:, np.newaxis] @ np.ones((1, oracle.l.size))).flatten())
        )
        x_full = np.hstack((xg_corrs.flatten(), xg_flows.flatten()))
        if x_full_prev is not None:
            dgrad = grad_full - grad_full_prev
            dx = x_full - x_full_prev
            Lcur = np.linalg.norm(dgrad) / np.linalg.norm(dx)
            if Lcur > Lmax:
                Lmax = Lcur
            mucur = (dgrad @ dx) / (dx @ dx)
            assert mucur >= 0
            if mucur < mumin:
                mumin = mucur

        grad_full_prev = grad_full
        x_full_prev = x_full

        x_half_corrs = (
            1
            / (1 + eta * alpha)
            * (x_corrs - eta * (dFxg_corrs - alpha * xg_corrs + oracle.BTmul(yAB) + oracle.KTmul(yK)))
        )
        x_half_corrs = np.maximum(x_half_corrs, eps)
        x_half_flows = 1 / (1 + eta * alpha) * (x_flows - eta * (dFxg_flows - alpha * xg_flows + oracle.ATmul(yAB)))
        x_half_flows = np.maximum(x_half_flows, 0)

        yAB = yAB + theta * (oracle.Amul(x_half_flows) + oracle.Bmul(x_half_corrs))
        yK = yK + theta * (oracle.Kmul(x_half_corrs) - b)

        x_prev_corrs, x_prev_flows = x_corrs, x_flows
        x_corrs = (
            1
            / (1 + eta * alpha)
            * (x_corrs - eta * (dFxg_corrs - alpha * xg_corrs + oracle.BTmul(yAB) + oracle.KTmul(yK)))
        )
        x_corrs = np.maximum(x_corrs, eps)
        x_flows = 1 / (1 + eta * alpha) * (x_flows - eta * (dFxg_flows - alpha * xg_flows + oracle.ATmul(yAB)))
        x_flows = np.maximum(x_flows, 0)

        xf_corrs = xg_corrs + 2 * tau / (2 - tau) * (x_corrs - x_prev_corrs)
        xf_flows = xg_flows + 2 * tau / (2 - tau) * (x_flows - x_prev_flows)

        times.append(time.time() - start)
        Kcons = np.linalg.norm(oracle.Kmul(x_corrs) - b)
        ABcons = np.linalg.norm(ABmul(x_flows, x_corrs))
        opt = (
            ((dFxg_corrs + oracle.BTmul(yAB) + oracle.KTmul(yK)) ** 2).sum()
            + (np.minimum(dFxg_flows + oracle.ATmul(yAB), 0) ** 2).sum()
        ) ** 0.5
        metric = Kcons + ABcons + opt

        if solution_flows is not None:
            log_dist.append(
                (
                    np.linalg.norm(x_flows.sum(axis=1) - solution_flows) ** 2
                    + np.linalg.norm(x_corrs - solution_corrs) ** 2
                )
                ** 0.5
            )

        log_Kcons.append(Kcons)
        log_ABcons.append(ABcons)
        log_opt.append(opt)

    print("Lmax=", Lmax)
    print("mumin=", mumin)

    print(Kcons, ABcons, opt)

    if plot_convergence:
        plt.plot(log_Kcons, label="K")
        plt.plot(log_ABcons, label="AB")
        plt.plot(log_opt, label="opt")
        plt.plot(log_dist, label="dist")
        plt.legend()
        plt.yscale("log")
        plt.show()

    pri_res = list(((np.array(log_Kcons) ** 2) + (np.array(log_ABcons) ** 2)) ** 0.5)
    return x_corrs, x_flows, yK, yAB, log_dist, times, pri_res, log_opt


def combined_salim_with_cheb(
    oracle: SaddleOracle,
    mu: float,
    L: float,
    lam1: float,
    lam2: float,
    solution_flows: Optional[np.ndarray] = None,
    solution_corrs: Optional[np.ndarray] = None,
    corrs0: Optional[np.ndarray] = None,
    flows0: Optional[np.ndarray] = None,
    eps_abs: Optional[float] = None,
    iters: int = 1000,
    plot_convergence: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Salim Kovalev 2022 with Cheb acceleration"""

    _, n_edges = oracle.incidence_mat.shape
    xf_flows = x_flows = np.zeros((n_edges, oracle.l.size)) if flows0 is None else flows0.copy()
    xf_corrs = x_corrs = np.ones((oracle.l.size, oracle.w.size)) if corrs0 is None else corrs0.copy()

    b = np.hstack((oracle.l, oracle.w))
    u_flows = np.zeros_like(xf_flows)
    u_corrs = np.zeros_like(xf_corrs)

    # alg parameters
    kappa = L / mu

    rho = (lam1 - lam2) ** 2 / 16
    nu = (lam1 + lam2) / 2
    N = int((lam1 / lam2) ** 0.5 + 10)

    # chebyshev will bring lam1 and lam2 closer:
    lam1, lam2 = 20 / 15, 10 / 15
    tau = min(1, 0.5 * (mu * lam1 / L / lam2) ** 0.5)
    eta = 1 / (4 * tau * L)
    theta = 1 / (eta * lam1)
    alpha = mu

    times = []
    log_Kcons, log_ABcons, log_opt, log_dist = [], [], [], []

    eps = 1e-6
    start = time.time()

    print("N", N)

    def cheb_iter(z_flows: np.ndarray, z_corrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma = -nu / 2

        def compute_p(z_flows: np.ndarray, z_corrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Kz_flows = oracle.Amul(z_flows) + oracle.Bmul(z_corrs)
            Kz_corrs = oracle.Kmul(z_corrs) - b
            return oracle.ATmul(Kz_flows), (oracle.BTmul(Kz_flows) + oracle.KTmul(Kz_corrs))

        p_flows, p_corrs = compute_p(z_flows, z_corrs)
        p_flows, p_corrs = -p_flows / nu, -p_corrs / nu
        z_flows = z_flows + p_flows
        z_corrs = z_corrs + p_corrs

        for _ in range(N):
            beta = rho / gamma
            gamma = -(nu + beta)
            p_flows_tmp, p_corrs_tmp = compute_p(z_flows, z_corrs)
            p_flows = (p_flows_tmp + beta * p_flows) / gamma
            p_corrs = (p_corrs_tmp + beta * p_corrs) / gamma
            z_flows += p_flows
            z_corrs += p_corrs

        return z_flows, z_corrs

    for i in tqdm(range(iters)):
        xg_flows = tau * x_flows + (1 - tau) * xf_flows
        xg_corrs = tau * x_corrs + (1 - tau) * xf_corrs

        dFxg_flows = oracle.grad_f(xg_flows)
        dFxg_corrs = oracle.grad_d(xg_corrs)

        x_half_flows = 1 / (1 + eta * alpha) * (x_flows - eta * (dFxg_flows - alpha * xg_flows + u_flows))
        x_half_flows = np.maximum(x_half_flows, 0)
        x_half_corrs = 1 / (1 + eta * alpha) * (x_corrs - eta * (dFxg_corrs - alpha * xg_corrs + u_corrs))
        x_half_corrs = np.maximum(x_half_corrs, eps)

        z_flows, z_corrs = cheb_iter(x_half_flows, x_half_corrs)
        r_flows = theta * (x_half_flows - z_flows)
        r_corrs = theta * (x_half_corrs - z_corrs)

        u_flows += r_flows
        u_corrs += r_corrs

        x_prev_flows, x_prev_corrs = x_flows, x_corrs
        x_flows = x_half_flows - eta / (1 + eta * alpha) * r_flows
        x_flows = np.maximum(x_flows, 0)
        x_corrs = x_half_corrs - eta / (1 + eta * alpha) * r_corrs
        x_corrs = np.maximum(x_corrs, eps)

        xf_flows = xg_flows + 2 * tau / (2 - tau) * (x_flows - x_prev_flows)
        xf_corrs = xg_corrs + 2 * tau / (2 - tau) * (x_corrs - x_prev_corrs)

        times.append(time.time() - start)
        Kcons = np.linalg.norm(oracle.Kmul(x_corrs) - b)
        ABcons = np.linalg.norm((oracle.Amul(x_flows) + oracle.Bmul(x_corrs)).flatten())
        opt = (((dFxg_corrs + u_corrs) ** 2).sum() + (np.minimum(dFxg_flows + u_flows, 0) ** 2).sum()) ** 0.5
        metric = Kcons + ABcons + opt

        if solution_flows is not None:
            log_dist.append(
                (
                    np.linalg.norm(x_flows.sum(axis=1) - solution_flows) ** 2
                    + np.linalg.norm(x_corrs - solution_corrs) ** 2
                )
                ** 0.5
            )

        log_Kcons.append(Kcons)
        log_ABcons.append(ABcons)
        log_opt.append(opt)

    print(Kcons, ABcons, opt)

    if plot_convergence:
        plt.plot(log_Kcons, label="K")
        plt.plot(log_ABcons, label="AB")
        plt.plot(log_opt, label="opt")
        plt.plot(log_dist, label="dist")
        plt.legend()
        plt.yscale("log")
        plt.show()

    pri_res = list(((np.array(log_Kcons) ** 2) + (np.array(log_ABcons) ** 2)) ** 0.5)
    return x_corrs, x_flows, u_corrs, u_flows, log_dist, times, pri_res, log_opt
