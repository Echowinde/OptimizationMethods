from functions import *
from utils import draw_contour


def goldstein(func, grad, x_k, d, max_alpha=1, rho=1e-4, t=2):
    phi_0 = func(x_k)
    dphi_0 = np.dot(grad(x_k), d)
    a = 0
    b = max_alpha
    k = 0
    alpha = np.random.rand()*max_alpha
    max_iter = 1000
    while k < max_iter:
        phi = func(x_k + d*alpha)
        if phi_0 + rho*alpha*dphi_0 >= phi:
            if phi_0 + (1-rho)*alpha*dphi_0 <= phi:
                break
            else:
                a = alpha
                if b >= max_alpha:
                    alpha = t*alpha
                    k += 1
                    continue
        else:
            b = alpha
        alpha = 0.5*(a+b)
        k += 1
    return alpha


def wolfe(func, grad, x_k, d, max_alpha=1, rho=1e-4, sigma=0.1):
    alpha_1 = 0
    alpha_2 = max_alpha
    phi_1 = func(x_k)
    dphi_1 = np.dot(grad(x_k), d)
    alpha = np.random.rand() * alpha_2

    max_iter = 100
    epoch = 0
    while epoch < max_iter:
        phi = func(x_k + d * alpha)
        if (phi - phi_1) <= rho * alpha * dphi_1:
            dphi = np.dot(grad(x_k + d * alpha), d)
            if dphi >= sigma * dphi_1:
                break
            else:
                alpha_bar = alpha + (alpha - alpha_1) * dphi / (dphi_1 - dphi)
                alpha_1 = alpha
                phi_1 = phi
                dphi_1 = dphi
                alpha = alpha_bar
        else:
            alpha_bar = alpha_1 + 0.5 * (alpha - alpha_1) / (1 + (phi_1 - phi) / (alpha - alpha_1) / dphi_1)
            alpha_2 = alpha
            alpha = alpha_bar
        epoch += 1

    return alpha


def steepest_descent(x, function, search='wolfe', epsilon=1e-3, verbose=100):
    record = [x]
    epoch = 0
    max_iter = 10000

    assert function in ('rosenbrock', 'beale')
    assert search in ('goldstein', 'wolfe')
    if function == 'rosenbrock':
        func = rosenbrock
        grad = rosen_grad
    elif function == 'beale':
        func = beale
        grad = beale_grad

    while epoch < max_iter:
        k = -grad(x)
        if search == 'goldstein':
            alpha = goldstein(func, grad, x, k)
        elif search == 'wolfe':
            alpha = wolfe(func, grad, x, k)
        x = x + alpha * k
        record.append(x)
        if (epoch + 1) % verbose == 0:
            print("---Iteration: {}---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
        if np.linalg.norm(k) < epsilon:
            print("---Iteration: {} Ends---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
            break
        epoch += 1
    return np.array(record)


if __name__ == "__main__":
    x_0 = np.array([-1.2, 1])
    result = steepest_descent(x_0, 'rosenbrock', 'goldstein', verbose=100)
    draw_contour(rosenbrock, result)

