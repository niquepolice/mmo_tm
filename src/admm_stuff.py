import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

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

    # All arguments are 2d

    # z = B.Ty, g = Af, Bd = traffic_lapl
    @staticmethod
    def Bmul(x: np.ndarray) -> np.ndarray:
        return np.diag(x.sum(axis=1)) - x

    @staticmethod
    def BTmul(x: np.ndarray) -> np.ndarray:
        return np.diag(x)[:, np.newaxis] - x

    @staticmethod
    def Kmul(x: np.ndarray) -> np.ndarray:
        return np.hstack((x.sum(axis=1), x.sum(axis=0)))

    @staticmethod
    def KTmul(y: np.ndarray) -> np.ndarray:
        y1, y2 = y[: y.size // 2], y[y.size // 2 :]
        return y1[:, np.newaxis] + y2[np.newaxis, :]

    def Amul(self, x: np.ndarray) -> np.ndarray:
        return (self.incidence_mat @ x).T

    def ATmul(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.incidence_mat).T

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

        zeta = eta = x = np.ones((self.incidence_mat.shape[1], d_ij.shape[0]))
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

            if not i % (iters // 5):
                print("flows agd:", np.linalg.norm(np.minimum(grad_x, 0)), f"{M= :.1e}")

            log.append(np.linalg.norm(np.minimum(grad_x, 0)))

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

        x0 = np.ones((self.l.size, self.w.size))
        xf = x = x0
        b = np.hstack((self.l, self.w))
        y = np.zeros(b.size)

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

            if not i % (iters // 5):
                print(
                    f"Salim cons={np.linalg.norm(self.Kmul(x) - b):.3e}",
                    f"func={np.linalg.norm(gradF(x) + self.KTmul(y)):.3e}"
                )

            if plot_convergence:
                log_cons.append(np.linalg.norm(self.Kmul(x) - b))
                log_func.append(np.linalg.norm(gradF(x) + self.KTmul(y)))

        if plot_convergence:
            plt.plot(log_cons, label="cons")
            plt.plot(log_func, label="func")
            plt.legend()
            plt.yscale("log")
            plt.show()

        return x, y
