from functions import *
from utils import *


def wolfe(func, grad, x_k, d, max_alpha=np.inf, rho=0.1, sigma=0.7):
    alpha_1 = 0
    alpha_2 = max_alpha
    phi_1 = func(x_k)
    dphi_1 = np.dot(grad(x_k), d)
    alpha = 1

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


def newton(x, function, method='pure', epsilon=1e-5, verbose=10):
    record = [x]
    epoch = 0
    max_iter = 1000
    assert function in ('rosenbrock', 'beale')
    if function == 'rosenbrock':
        func = rosenbrock
        grad = rosen_grad
        hessian = rosen_hessian
    elif function == 'beale':
        func = beale
        grad = beale_grad
        hessian = beale_hessian
    alpha = 1
    while epoch < max_iter:
        k = -grad(x)
        p = -np.dot(np.linalg.inv(hessian(x)), -k)
        if method == 'line':
            alpha = wolfe(func, grad, x, p)
        elif method == 'GP':
            angle = np.dot(p, k)/(np.linalg.norm(p) * np.linalg.norm(k))
            if angle < 0.5:
                p = k
            alpha = wolfe(func, grad, x, p)
        elif method == 'goldfeld':
            tau = 0.0
            muk = pow(np.linalg.norm(k), 1+tau)
            p = -np.dot(np.linalg.inv(hessian(x) + muk*np.eye(len(hessian(x)))), -k)
            alpha = wolfe(func, grad, x, p)
        x = x + alpha*p
        record.append(x)
        if (epoch+1) % verbose == 0:
            print("---Iteration: {}---".format(epoch+1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
        if np.linalg.norm(p) < epsilon:
            print("---Iteration: {} Ends---".format(epoch+1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
            break
        epoch += 1
    return np.array(record)


def quasi_newton(x, func, grad, method='BFGS', epsilon=1e-5, verbose=10):
    record = [x]
    epoch = 0
    max_iter = 100

    dim = len(x)
    H_0 = np.eye(dim)
    g_0 = grad(x)

    while epoch < max_iter:
        d = -np.dot(H_0, g_0)
        alpha = wolfe(func, grad, x, d)
        x = x + alpha * d
        g = grad(x)
        s = alpha * d
        y = g - g_0

        # BFGS
        H = H_0 + (1 + np.dot(y, np.dot(H_0, y))/np.dot(s, y)) * np.dot(s.reshape(-1, 1), s.reshape(1, -1))/np.dot(s, y) \
            - (np.dot(np.dot(s.reshape(-1, 1), y.reshape(1, -1)), H_0)
            + np.dot(np.dot(H_0, y.reshape(-1, 1)), s.reshape(1, -1))) / np.dot(s, y)

        record.append(x)
        if (epoch + 1) % verbose == 0:
            print("---Iteration: {}---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
        if np.linalg.norm(g) < epsilon:
            print("---Iteration: {} Ends---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, f(x) = {:.5f}".format(x, alpha, func(x)))
            break
        g_0 = g
        H_0 = H
        epoch += 1
    return np.array(record)


if __name__ == "__main__":
    # newton
    x_0 = np.array([3, 0.2])
    result = newton(x_0, 'beale', method='line', verbose=1)
    draw_contour(beale, result)
    convergence_rate(beale, result)

    # quasi-newton
    x_0 = np.array([-1.2, 1, -1.2, 1])
    result = quasi_newton(x_0, rosenbrock, rosen_grad, method='BFGS', verbose=10)
    convergence_rate(rosenbrock, result)
