import torch


def newton_torch(x_0_arr, a_arr, mu_arr, xd, tol=1e-7, max_iter=1000):
    r"""
    Newton method for equation: :math:`x - x_0 + a x^{\mu} = 0, x \geq 0`.
    """
    res = torch.clone(xd).to(torch.float64)
    res[x_0_arr <= 0] = 0
    x = torch.min(x_0_arr, (x_0_arr / a_arr) ** (1 / mu_arr))
    for it in range(max_iter):
        x_next = x - f(x, x_0_arr, a_arr, mu_arr) / der_f(x, x_0_arr, a_arr, mu_arr)
        x_next[x_next <= 0] = 0.1 * x[x_next <= 0]
        x = x_next
        if torch.norm(f(x, x_0_arr, a_arr, mu_arr)) < tol:
            break

    if it == max_iter - 1:
        print("warning! max iter in newton: func_val=", torch.norm(f(x, x_0_arr, a_arr, mu_arr)), "tol=", tol)
    res[x_0_arr > 0] = x[x_0_arr > 0]
    return res


def f(x, x_0, a, mu):
    return x - x_0 + a * x**mu

def der_f(x, x_0, a, mu):
    return 1.0 + a * mu * x ** (mu - 1)
