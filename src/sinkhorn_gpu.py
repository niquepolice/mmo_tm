import numpy as np
import torch
from torch import from_numpy
from typing import Tuple, Union


def d_ij(lambda_l_i: torch.Tensor, lambda_w_j: torch.Tensor, gammaT_ij: torch.Tensor) -> torch.Tensor:
    if isinstance(gammaT_ij, np.ndarray):
        gammaT_ij = torch.from_numpy(gammaT_ij)

    if isinstance(lambda_l_i, np.ndarray):
        lambda_l_i = torch.from_numpy(lambda_l_i)

    if isinstance(lambda_w_j, np.ndarray):
        lambda_w_j = torch.from_numpy(lambda_w_j)

    return Sinkhorn.d_ij(lambda_l_i, lambda_w_j, gammaT_ij).detach().cpu().numpy()


class Sinkhorn:
    def __init__(
        self,
        departures: torch.Tensor,
        arrivals: torch.Tensor,
        max_iter: int,
        eps=1e-6,
        crit_check_period=10,
        device="cuda",
    ):
        if isinstance(departures, np.ndarray):
            departures = torch.from_numpy(departures)
        if isinstance(arrivals, np.ndarray):
            arrivals = torch.from_numpy(arrivals)

        self.L_i = departures.to(device)
        self.W_j = arrivals.to(device)
        self.n_types = self.L_i.shape[0]
        self.max_iter = max_iter
        self.eps = eps
        self.crit_check_period = crit_check_period
        self.device = device

    @staticmethod
    def logsumexp(input: torch.Tensor, dim: int, keepdim: bool, b: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.sum(b * torch.exp(input), dim=dim, keepdim=keepdim))

    @staticmethod
    def d_ij(lambda_l_i: torch.Tensor, lambda_w_j: torch.Tensor, gammaT_ij: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(1 + gammaT_ij + lambda_w_j + lambda_l_i.unsqueeze(1)))

    def _iteration(
        self,
        k: int,
        gammaT_ij: torch.Tensor,
        lambda_w_j: torch.Tensor,
        lambda_l_i: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if k % 2 == 0:
            lambda_l_i = self.logsumexp(
                -lambda_w_j.unsqueeze(0) - 1.0 - gammaT_ij,
                dim=1,
                keepdim=False,
                b=1.0 / self.L_i.unsqueeze(1),
            )
        else:
            lambda_w_j = self.logsumexp(
                (-lambda_l_i.unsqueeze(1) - 1.0 - gammaT_ij),
                dim=0,
                keepdim=False,
                b=1.0 / self.W_j.unsqueeze(0),
            )

        return lambda_w_j, lambda_l_i

    def run(
        self,
        gammaT_ij: Union[torch.Tensor, np.ndarray],
        lambda_l_i: Union[torch.Tensor, np.ndarray, None] = None,
        lambda_w_j: Union[torch.Tensor, np.ndarray, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if lambda_l_i is None:
            lambda_l_i = torch.zeros_like(self.L_i)
        if lambda_w_j is None:
            lambda_w_j = torch.zeros_like(self.W_j)

        if isinstance(gammaT_ij, np.ndarray):
            gammaT_ij = torch.from_numpy(gammaT_ij)

        if isinstance(lambda_l_i, np.ndarray):
            lambda_l_i = torch.from_numpy(lambda_l_i)

        if isinstance(lambda_w_j, np.ndarray):
            lambda_w_j = torch.from_numpy(lambda_w_j)

        gammaT_ij = gammaT_ij.to(self.device)
        lambda_l_i = lambda_l_i.to(self.device)
        lambda_w_j = lambda_w_j.to(self.device)

        k = 0
        while True:
            if k > 0 and not k % self.crit_check_period:
                if self._criteria(lambda_l_i, lambda_w_j, gammaT_ij):
                    break

            lambda_w_j, lambda_l_i = self._iteration(k, gammaT_ij, lambda_w_j, lambda_l_i)

            k += 1
            if k == self.max_iter:
                raise RuntimeError("Max iter exceeded in SinkhornGPU")
        return (
            self.d_ij(lambda_l_i, lambda_w_j, gammaT_ij).detach().cpu().numpy(),
            lambda_l_i.detach().cpu().numpy(),
            lambda_w_j.detach().cpu().numpy(),
        )

    def _criteria(
        self,
        lambda_l_i: torch.Tensor,
        lambda_w_j: torch.Tensor,
        gammaT_ij: torch.Tensor,
    ) -> bool:
        traffic_mat = self.d_ij(lambda_l_i, lambda_w_j, gammaT_ij)
        grad_l = traffic_mat.sum(dim=1) - self.L_i
        grad_w = traffic_mat.sum(dim=0) - self.W_j
        dual_grad = torch.cat((grad_l, grad_w))

        dual_grad_norm = torch.norm(dual_grad)
        inner_prod = -torch.cat((lambda_l_i, lambda_w_j)) @ dual_grad

        return dual_grad_norm < self.eps and inner_prod < self.eps
