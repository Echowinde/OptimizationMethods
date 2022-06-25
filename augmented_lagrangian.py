import numpy as np


def func(x, lam, mu):
    origin = x[0]**2 + 2*x[1]**2 - 2*x[0] - 6*x[1] - 2*x[0]*x[1]
    s = [max(0, (-0.5 * x[0] - x[1] + 1) + lam[0] / mu), max(0, (x[0] - 2 * x[1] + 2) + lam[1] / mu),
         max(0, x[0] + lam[2] / mu), max(0, x[1] + lam[3] / mu)]
    lagrange = lam[0] * (-0.5 * x[0] - x[1] + 1 - s[0]) + lam[1] * (x[0] - 2 * x[1] + 2 - s[1]) + lam[2] * (
                x[0] - s[2]) + lam[3] * (x[1] - s[3])
    penalty = mu/2* ((-0.5*x[0]-x[1]+1-s[0])**2 + (x[0]-2*x[1]+2-s[1])**2 + (x[0]-s[2])**2 + (x[1]-s[3])**2)
    return origin + lagrange + penalty


def grad(x, lam, mu):
    grad_x1 = 2 * x[0] - 2 * x[1] - 2
    grad_x2 = -2 * x[0] + 4 * x[1] - 6

    grad_constraint1 = np.array([0, 0]) if mu * (-0.5 * x[0] - x[1] + 1) + lam[0] >= 0 else (mu * (
                -0.5 * x[0] - x[1] + 1) + lam[0]) * np.array([-0.5, -1])
    grad_constraint2 = np.array([0, 0]) if mu * (x[0] - 2 * x[1] + 2) + lam[1] >= 0 else (mu * (x[0] - 2 * x[1] + 2) +
                                                                                          lam[1]) * np.array([1, -2])
    grad_constraint3 = np.array([0, 0]) if mu * x[0] + lam[2] >= 0 else (mu * x[0] + lam[2]) * np.array([1, 0])
    grad_constraint4 = np.array([0, 0]) if mu * x[1] + lam[3] >= 0 else (mu * x[1] + lam[3]) * np.array([0, 1])
    return np.array([grad_x1, grad_x2]) + grad_constraint1 + grad_constraint2 + grad_constraint3 + grad_constraint4


def goldstein(func, grad, mu, lam, x_k, d, max_alpha=1, rho=1e-4, t=2):
    phi_0 = func(x_k, lam, mu)
    dphi_0 = np.dot(grad(x_k, lam, mu), d)
    a = 0
    b = max_alpha
    k = 0
    alpha = np.random.rand()*max_alpha
    max_iter = 1000
    while k < max_iter:
        phi = func(x_k + d*alpha, lam, mu)
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
def steepest_descent(x, func, grad, mu, lam, epsilon=1e-3):
    record = [x]
    epoch = 0
    max_iter = 100

    while epoch < max_iter:
        k = -grad(x, lam, mu)

        alpha = goldstein(func, grad, mu, lam, x, k)
        x = x + alpha * k
        record.append(x)
        if np.linalg.norm(k) < epsilon:
            break
        epoch += 1
    return np.array(record)


def augmented_lagrangian(x, lam, eta_star=1e-3, omega_star=1e-3):
    mu = 10
    omega = pow(mu, -1)
    eta = pow(mu, -0.1)
    max_iter = 6
    epoch = 0
    while epoch < max_iter:
        x_estimate = steepest_descent(x, func, grad, mu, lam, epsilon=1e-3)[-1]
        s = [max(0, (-0.5*x[0]-x[1]+1)+lam[0]/mu), max(0, (x[0]-2*x[1]+2)+lam[1]/mu), max(0, x[0]+lam[2]/mu), max(0, x[1]+lam[3]/mu)]
        cx_k = np.array([-0.5*x_estimate[0]-x_estimate[1]+1-s[0], x_estimate[0]-2*x_estimate[1]+2-s[1], x_estimate[0]-s[2], x_estimate[1]-s[3]])
        print("---Iteration: {}---".format(epoch + 1))
        print("x_estimate = {}, lambda = {}, mu={}".format(x_estimate, lam, mu))
        p_1 = 0 if x[0] - grad(x, lam, mu)[0] <= 0 else x[0] - grad(x, lam, mu)[0]
        p_2 = 0 if x[1] - grad(x, lam, mu)[1] <= 0 else x[1] - grad(x, lam, mu)[1]
        p = np.array([p_1, p_2])
        if np.linalg.norm(cx_k) <= eta:
            if (np.linalg.norm(cx_k) <= eta_star) & (np.linalg.norm(x-p) <= omega_star):
                return x_estimate
            lam = lam - mu*cx_k
            eta = eta*pow(mu, -0.9)
            omega = omega*pow(mu, -1)
        else:
            mu = 100*mu
            eta = pow(mu, -0.1)
            omega = pow(mu, -1)
        epoch += 1


if __name__ == "__main__":
    x_0 = [0.5, 0.5]
    lam_0 = np.array([0.1, 0.1, 0.1, 0.1])
    augmented_lagrangian(x_0, lam_0)