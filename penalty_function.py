import numpy as np


def func(x, mu, penalty):
    """
    Example constrained optimization problem
    min x1^2 + 2x2^2 -2x1 - 6x2 -2x1x2
    s.t. -0.5x1 - x2 + 1 >= 0
         x1 - 2x2 + 2 >= 0
         x1 >= 0
         x2 >= 0
    """
    origin = x[0]**2 + 2*x[1]**2 - 2*x[0] - 6*x[1] - 2*x[0]*x[1]
    if penalty == 'quadratic':
        constraint = mu/2 * (max(-x[0], 0)**2 + max(-x[1], 0)**2 + max(0.5*x[0]+x[1]-1, 0)**2 + max(-x[0]+2*x[1]-2, 0)**2)
    elif penalty == 'l1':
        constraint = mu * (max(-x[0], 0) + max(-x[1], 0) + max(0.5*x[0]+x[1]-1, 0) + max(-x[0]+2*x[1]-2, 0))
    else:
        constraint = 0
    return origin + constraint


def grad(x, mu, penalty):
    if penalty == 'quadratic':
        grad_constraint1 = mu/2 * np.array([0.5*x[0]+x[1]-1, x[0]+2*x[1]-2]) if (0.5*x[0]+x[1]-1) > 0 else np.array([0, 0])
        grad_constraint2 = mu/2 * np.array([2*x[0]-4*x[1]+4, -4*x[0]+8*x[1]-8]) if (-x[0]+2*x[1]-2) > 0 else np.array([0, 0])
        grad_constraint3 = mu/2 * np.array([2*x[0], 0]) if -x[0] > 0 else np.array([0, 0])
        grad_constraint4 = mu/2 * np.array([0, 2*x[1]]) if -x[1] > 0 else np.array([0, 0])
    elif penalty == 'l1':
        grad_constraint1 = mu * np.array([0.5, 1]) if (0.5*x[0]+x[1]-1) > 0 else np.array([0, 0])
        grad_constraint2 = mu * np.array([-1, 2]) if (-x[0]+2*x[1]-2) > 0 else np.array([0, 0])
        grad_constraint3 = mu * np.array([-1, 0]) if -x[0] > 0 else np.array([0, 0])
        grad_constraint4 = mu * np.array([0, -1]) if -x[1] > 0 else np.array([0, 0])
    else:
        grad_constraint1, grad_constraint2 = np.array([0, 0]), np.array([0, 0])
        grad_constraint3, grad_constraint4 = np.array([0, 0]), np.array([0, 0])
    grad_x1 = 2*x[0] - 2*x[1] - 2
    grad_x2 = -2*x[0] + 4*x[1] - 6
    return np.array([grad_x1, grad_x2]) + grad_constraint1 + grad_constraint2 + grad_constraint3 + grad_constraint4


def goldstein(func, grad, mu, penalty, x_k, d, max_alpha=1, rho=1e-4, t=2):
    phi_0 = func(x_k, mu, penalty)
    dphi_0 = np.dot(grad(x_k, mu, penalty), d)
    a = 0
    b = max_alpha
    k = 0
    alpha = np.random.rand()*max_alpha
    max_iter = 1000
    while k < max_iter:
        phi = func(x_k + d*alpha, mu, penalty)
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


# various unconstrained optimization methods can be applied here
def steepest_descent(x, func, grad, mu, penalty, epsilon=1e-3):
    record = [x]
    epoch = 0
    max_iter = 100

    while epoch < max_iter:
        k = -grad(x, mu, penalty)

        alpha = goldstein(func, grad, mu, penalty, x, k)
        x = x + alpha * k
        record.append(x)
        if np.linalg.norm(k) < epsilon:
            break
        epoch += 1
    return np.array(record)


def penalty_function(x, rho, penalty, tau=1e-5):
    mu = 1
    epoch = 0
    max_iter = 6

    while epoch < max_iter:
        x_estimate = steepest_descent(x, func, grad, mu, penalty, epsilon=1e-3)[-1]
        grad_estimate = grad(x_estimate, mu, penalty)
        print("---Iteration: {}---".format(epoch + 1))
        print("x_estimate = {}, f(x) = {:.5f}, mu={}".format(x_estimate, func(x_estimate, 0, None), mu))
        if np.linalg.norm(grad_estimate) < tau:
            return x_estimate
        mu = mu*rho
        x = x_estimate
        epoch += 1


if __name__ == "__main__":
    x_0 = np.array([0, 0])
    penalty_function(x_0, rho=10, penalty='quadratic')    # quadratic penalty method, recommended rho=10
    penalty_function(x_0, rho=1.5, penalty='l1')    # classical l1 penalty method, recommended rho=1.5
