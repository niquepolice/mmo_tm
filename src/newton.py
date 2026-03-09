import numpy as np
from numba import njit


@njit
def newton(x_0_arr, a_arr, mu_arr, xd, tol=1e-7, max_iter=1000):
    r"""
    Newton method for equation: :math:`x - x_0 + a x^{\mu} = 0, x \geq 0`.
    """
    res = np.copy(xd).astype(np.float64)
    for i in range(len(x_0_arr)):
        x_0 = x_0_arr[i]
        a = a_arr[i]
        mu = mu_arr[i]
        assert mu <= 10000, f"{mu}"
        if x_0 <= 0:
            res[i] = 0
            continue
        x = min(x_0, (x_0 / a) ** (1 / mu))
        for it in range(max_iter):
            x_next = x - f(x, x_0, a, mu) / der_f(x, x_0, a, mu)
            if x_next <= 0:
                x_next = 0.1 * x
            x = x_next
            if np.abs(f(x, x_0, a, mu)) < tol:
                break

        if it == max_iter - 1:
            print("warning! max iter in newton: func_val=", np.abs(f(x, x_0, a, mu)), "tol=", tol)
        res[i] = x
    return res


@njit
def f(x, x_0, a, mu):
    return x - x_0 + a * x**mu

@njit
def der_f(x, x_0, a, mu):
    return 1.0 + a * mu * x ** (mu - 1)
