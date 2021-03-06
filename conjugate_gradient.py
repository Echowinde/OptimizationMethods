from functions import *
from utils import convergence_rate


def conjugate_gradient(x, method='FR', epsilon=1e-6, verbose=1):
    record = [x]
    epoch = 0
    max_iter = 5000
    dim = len(x)
    g_0 = quadratic_grad(x, dim)
    d_0 = -g_0
    while epoch < max_iter:
        alpha = -np.dot(d_0, g_0) / (np.dot(np.dot(d_0, hilbert(dim)), d_0))
        x = x + alpha * d_0
        g = quadratic_grad(x, dim)
        if method == 'FR':
            beta = np.dot(g, g) / np.dot(g_0, g_0)   # FR
        elif method == 'PRP':
            beta = np.dot(g, g-g_0)/np.dot(g_0, g_0)  # PRP
        elif method == 'DY':
            beta = np.dot(g, g)/np.dot(d_0, g-g_0)  # Dai-Yuan

        d = -g + beta * d_0
        record.append(x)

        if (epoch + 1) % verbose == 0:
            print("---Iteration: {}---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, beta = {:.3f}, f(x) = {:.5f}".format(x, alpha, beta, quadratic(x, dim)))
        if np.linalg.norm(g) < epsilon:
            print("---Iteration: {} Ends---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, beta = {:.3f}, f(x) = {:.5f}".format(x, alpha, beta, quadratic(x, dim)))
            break

        g_0 = g
        d_0 = d
        epoch += 1

    return np.array(record)


def preconditioned_cg(x, method='jacobi', epsilon=1e-6, verbose=1):
    record = [x]
    epoch = 0
    max_iter = 5000
    dim = len(x)
    g_0 = quadratic_grad(x, dim)
    if method == 'jacobi':
        M = diag(hilbert(dim))
    Minv = np.linalg.inv(M)
    y_0 = np.dot(Minv, g_0)
    d_0 = -y_0
    while epoch < max_iter:
        alpha = np.dot(g_0, y_0) / (np.dot(np.dot(d_0, hilbert(dim)), d_0))
        x = x + alpha * d_0
        g = quadratic_grad(x, dim)
        y = np.dot(Minv, g)

        beta = np.dot(g, y) / np.dot(g_0, y_0)   # FR
        d = -y + beta * d_0
        record.append(x)

        if (epoch + 1) % verbose == 0:
            print("---Iteration: {}---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, beta = {:.3f}, f(x) = {:.5f}".format(x, alpha, beta, quadratic(x, dim)))
        if np.linalg.norm(g) < epsilon:
            print("---Iteration: {} Ends---".format(epoch + 1))
            print("x = {}, alpha = {:.3f}, beta = {:.3f}, f(x) = {:.5f}".format(x, alpha, beta, quadratic(x, dim)))
            break

        g_0 = g
        d_0 = d
        y_0 = y
        epoch += 1

    return np.array(record)

if __name__ == "__main__":
    x_0 = np.zeros(8)
    result = conjugate_gradient(x=x_0, method='FR', epsilon=1e-6, verbose=2)
    convergence_rate(quadratic, result)

    result = preconditioned_cg(x=x_0, method='jacobi', epsilon=1e-6, verbose=2)
    convergence_rate(quadratic, result)