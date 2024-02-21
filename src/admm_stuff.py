import time
import warnings
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


class AdmmOracle:
    def __init__(self, traffic_model: BeckmannModel, gamma: float, l: np.ndarray, w: np.ndarray):
        self.traffic_model = traffic_model
        self.incidence_mat = nx.incidence_matrix(traffic_model.nx_graph, oriented=True).todense()
        self.gamma = gamma
        self.l, self.w = l, w
        self.n_nodes, self.n_edges = self.incidence_mat.shape
        self.n_centroids = l.size

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

    def agd_argmin_flows(
        self,
        d_ij: np.ndarray,
        y: np.ndarray,
        rho: float,
        x0: Optional[np.ndarray] = None,
        eps_abs: Optional[float] = None,
        iters: int = 1000,
        M0: float = 1e1,
        plot_convergence: bool = False,
    ) -> np.ndarray:
        """Returns flows_ei"""

        # to reuse matrix multiplications
        ATy = self.ATmul(y)
        Bd = self.Bmul(d_ij)

        def grad_f_subprob(f_ei: np.ndarray) -> np.ndarray:
            """Gradient of the objective in the flows subproblem in ADMM"""
            return self.grad_f(f_ei) + ATy + rho * self.ATmul(self.Amul(f_ei) + Bd)

        def func_f_subprob(f_ei: np.ndarray) -> np.ndarray:
            """Objective value in the flows subproblem in ADMM"""
            return (
                self.traffic_model.primal(f_ei.sum(axis=1))
                # + (y * self.Amul(f_ei)).sum()
                + (ATy * f_ei).sum()
                + rho * ((self.Amul(f_ei) + Bd) ** 2).sum() / 2
            )

        log = []

        M = M0
        beta = 0

        zeta = eta = x = x0.copy() if x0 is not None else np.ones((self.incidence_mat.shape[1], d_ij.shape[0]))
        for i in range(iters):
            M /= 2

            while True:
                alpha = (1 + (1 + 4 * M * beta) ** 0.5) / (2 * M)
                beta += alpha
                tau = alpha / beta

                x = tau * zeta + (1 - tau) * eta
                grad_x = grad_f_subprob(f_ei=x)
                zeta = np.maximum(zeta - alpha * grad_x, 0)
                eta = tau * zeta + (1 - tau) * eta

                lhs = func_f_subprob(f_ei=eta) - func_f_subprob(f_ei=x)
                rhs = (grad_x * (eta - x)).sum() + M / 2 * ((eta - x) ** 2).sum()
                if lhs <= rhs + 1e-10:
                    break
                M *= 2

            metric = np.linalg.norm(np.minimum(grad_x, 0))
            log.append(metric)

            if eps_abs and metric < eps_abs:
                break

            if not i % (iters // 5):
                print("flows agd:", metric, f"{M= :.1e}")

        print("agd", i)
        if i == iters - 1:
            warnings.warn(f"Agd reached iter limit", category=RuntimeWarning)

        if plot_convergence:
            plt.plot(log, label="||(grad_x)_{-}||")
            plt.legend()
            plt.yscale("log")
            plt.show()

        return x

    def salim_argmin_corrs(
        self,
        y_admm: np.ndarray,
        flows_ei: np.ndarray,
        rho: float,
        mu: float,
        L: float,
        x0: Optional[np.ndarray] = None,
        y0: Optional[np.ndarray] = None,
        eps_abs: Optional[float] = None,
        iters: int = 1000,
        plot_convergence: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Salim Kovalev 2022
        without Cheb acceleration"""

        gamma = self.gamma

        # to reuse matrix multiplications
        z = self.BTmul(y_admm)
        g = self.Amul(flows_ei)

        def gradF(x):
            return (np.log(x) + 1) / gamma + z + rho * (self.BTmul(self.Bmul(x) + g))

        xf = x = np.ones((self.l.size, self.w.size)) if x0 is None else x0.copy()
        b = np.hstack((self.l, self.w))
        y = np.zeros(b.size) if y0 is None else y0.copy()

        # alg parameters
        n = self.l.size
        lam1, lam2 = 2 * n, n
        tau = min(1, 0.5 * (mu * lam1 / L / lam2) ** 0.5)
        eta = 1 / (4 * tau * L)
        theta = 1 / (eta * lam1)
        alpha = mu

        log_cons = []
        log_func = []

        eps = 1e-6
        for i in range(iters):
            xg = tau * x + (1 - tau) * xf
            dFxg = gradF(xg)
            x_half = 1 / (1 + eta * alpha) * (x - eta * (dFxg - alpha * xg + self.KTmul(y)))
            x_half = np.maximum(x_half, eps)
            y = y + theta * (self.Kmul(x_half) - b)
            x_prev = x
            x = 1 / (1 + eta * alpha) * (x - eta * (dFxg - alpha * xg + self.KTmul(y)))
            x = np.maximum(x, eps)
            xf = xg + 2 * tau / (2 - tau) * (x - x_prev)

            metric = np.linalg.norm(self.Kmul(x) - b) + np.linalg.norm(gradF(x) + self.KTmul(y))

            if not iters % 10 and eps_abs and metric < eps_abs:
                break

            # if not i % (iters // 5):
            #     print(
            #         f"Salim cons={np.linalg.norm(self.Kmul(x) - b):.3e}",
            #         f"func={np.linalg.norm(gradF(x) + self.KTmul(y)):.3e}"
            #     )

            if plot_convergence:
                log_cons.append(np.linalg.norm(self.Kmul(x) - b))
                log_func.append(np.linalg.norm(gradF(x) + self.KTmul(y)))

        if i == iters - 1:
            warnings.warn(f"Salim reached iter limit", category=RuntimeWarning)
        print("salim", i)
        if plot_convergence:
            plt.plot(log_cons, label="cons")
            plt.plot(log_func, label="func")
            plt.legend()
            plt.yscale("log")
            plt.show()

        return x, y


def combined_salim(
    oracle: AdmmOracle,
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

    gamma = oracle.gamma

    n_nodes, n_edges = oracle.incidence_mat.shape
    xf_corrs = x_corrs = np.ones((oracle.l.size, oracle.w.size)) if corrs0 is None else corrs0.copy()
    xf_flows = x_flows = np.zeros((n_edges, oracle.l.size)) if corrs0 is None else corrs0.copy()

    b = np.hstack((oracle.l, oracle.w))
    yK = np.zeros(b.size)
    yAB = np.zeros((oracle.l.size, n_nodes))

    # alg parameters
    n = oracle.l.size
    # lam1, lam2 = 2 * n, n

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
        # if not i % 1000:
        #     xf_corrs = x_corrs
        #     xf_flows = x_flows

        xg_corrs = tau * x_corrs + (1 - tau) * xf_corrs
        xg_flows = tau * x_flows + (1 - tau) * xf_flows

        # dFxg = gradF(xg)
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
            # print(dFxg_corrs.shape, xg_corrs.shape, dFxg_flows.shape, xg_flows.shape)
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
        # x = 1 / (1 + eta * alpha) * (x - eta * (dFxg - alpha * xg + oracle.KTmul(y)))
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

        # if not iters % 10 and eps_abs and metric < eps_abs:
        #     break

        # if not i % (iters // 5):
        #     print(
        #         f"Salim cons={np.linalg.norm(oracle.Kmul(x) - b):.3e}",
        #         f"func={np.linalg.norm(gradF(x) + oracle.KTmul(y)):.3e}"
        #     )

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

    if i == iters - 1:
        warnings.warn(f"Salim reached iter limit", category=RuntimeWarning)
    print("salim", i)
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
    oracle: AdmmOracle,
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

    _, n_edges = oracle.incidence_mat.shape
    xf_flows = x_flows = np.zeros((n_edges, oracle.l.size)) if corrs0 is None else corrs0.copy()
    xf_corrs = x_corrs = np.ones((oracle.l.size, oracle.w.size)) if corrs0 is None else corrs0.copy()

    b = np.hstack((oracle.l, oracle.w))
    u_flows = np.zeros_like(xf_flows)
    u_corrs = np.zeros_like(xf_corrs)

    # alg parameters
    kappa = L / mu

    tau = min(1, 0.5 * np.sqrt(19 / (15 * kappa)))
    eta = 1 / (4 * tau * L)
    theta = 15 / (19 * eta)
    alpha = mu

    times = []
    log_Kcons, log_ABcons, log_opt, log_dist = [], [], [], []

    eps = 1e-6
    start = time.time()

    rho = (lam1 - lam2) ** 2 / 16
    nu = (lam1 + lam2) / 2

    def cheb_iter(z_flows: np.ndarray, z_corrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gamma = -nu / 2

        def compute_p(z_flows: np.ndarray, z_corrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Kz_flows = oracle.Amul(z_flows) + oracle.Bmul(z_corrs)
            Kz_corrs = oracle.Kmul(z_corrs) - b
            return -oracle.ATmul(Kz_flows) / nu, -(oracle.BTmul(Kz_flows) + oracle.KTmul(Kz_corrs)) / nu

        p_flows, p_corrs = compute_p(z_flows, z_corrs)
        z_flows += p_flows
        z_corrs += p_corrs

        for _ in tqdm(range(250)):
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
        r_flows = theta * (x_flows - z_flows)
        r_corrs = theta * (x_corrs - z_corrs)

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

        # if not iters % 10 and eps_abs and metric < eps_abs:
        #     break

        # if not i % (iters // 5):
        #     print(
        #         f"Salim cons={np.linalg.norm(oracle.Kmul(x) - b):.3e}",
        #         f"func={np.linalg.norm(gradF(x) + oracle.KTmul(y)):.3e}"
        #     )

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

    if i == iters - 1:
        warnings.warn(f"Salim with Cheb reached iter limit", category=RuntimeWarning)
    print("salim with Cheb", i)
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


def combined_admm(
    admm_oracle,
    d0: np.ndarray,
    f0: np.ndarray,
    solution_flows,
    solution_corrs,
    agd_kwargs: dict,
    salim_kwargs: dict,
    rho: float = 0.2,
    iters: int = 100,
):

    d_ij = d0.copy()  # maybe init with zeros or 1-rank approx?
    # d_ij = np.ones(d0.shape)
    # d_ij = l[:, np.newaxis] @ w[np.newaxis, :] / l.sum()

    y = np.zeros(d0.shape)

    f_dists, d_dists, cons, dual_res = [], [], [], []
    f_dual_feas, d_dual_feas = [], []
    lam, mu = None, None

    flows_ei, y_salim = f0, None
    start = time.time()
    admm_times = []
    for k in tqdm(range(iters)):
        flows_ei_ = admm_oracle.agd_argmin_flows(d_ij, y, rho, x0=flows_ei, **agd_kwargs)
        flows_ei = flows_ei_

        d_ij_, y_salim = admm_oracle.salim_argmin_corrs(
            y_admm=y, flows_ei=flows_ei, rho=rho, x0=d_ij, y0=y_salim, **salim_kwargs
        )
        dual_res.append(rho * np.linalg.norm(admm_oracle.ATmul(admm_oracle.Bmul(d_ij - d_ij_))))
        d_ij = d_ij_

        Bd_plus_Af = admm_oracle.Amul(flows_ei) + admm_oracle.Bmul(d_ij)
        y += rho * Bd_plus_Af

        admm_times.append(time.time() - start)
        f_dists.append(np.linalg.norm(solution_flows - flows_ei.sum(axis=1)))
        d_dists.append(np.linalg.norm(solution_corrs - d_ij))

        cons.append(np.linalg.norm(Bd_plus_Af))  # primal residual

    return admm_times, f_dists, d_dists, cons, dual_res
