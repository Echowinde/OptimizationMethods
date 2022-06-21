import numpy as np


def active_set(x, working_set, epsilon=1e-6):
    epoch = 0
    max_iter = 100
    record = [x]
    while epoch < max_iter:

        working_A = A[working_set]
        working_b = np.zeros_like(b[working_set])
        g = np.dot(G, x) + c
        p_k, lam = solve_equality_constraints(G, g, working_A, working_b)
        func_value = 0.5 * np.dot(np.dot(x, G), x) + np.dot(c, x)
        print("---Iteration: {}---".format(epoch+1))
        print("x={}, value={}, p={}, lambda={}, working set id: {}".format(x, func_value, p_k, lam, working_set))
        if np.linalg.norm(p_k) < epsilon:

            if min(lam) >= 0:
                print("Find optimal solution")
                break

            else:
                pop_idx = np.argmin(lam)
                print("Operation: Delete constraint, index={}".format(working_set[pop_idx]))
                working_set.pop(pop_idx)
        else:
            alpha, add_idx = calculate_alpha(x, p_k, A, b, working_set)
            x = x + alpha * p_k
            record.append(x)
            if alpha < 1:
                print("Operation: Add constraint, index={}".format(add_idx))
                working_set.append(add_idx)
        epoch += 1
    return record


def solve_equality_constraints(G, c, A, b):
    Z = np.zeros([A.shape[0], A.shape[0]])
    upper = np.concatenate([G, -A.T], axis=1)
    lower = np.concatenate([A, Z], axis=1)
    matrix = np.concatenate([upper, lower])
    vector = np.concatenate([-c, b])
    solution = np.dot(np.linalg.pinv(matrix), vector)
    x, lam = solution[:2], solution[2:]
    return x, lam


def calculate_alpha(x, p, A, b, working_set):
    alpha = 1
    add_idx = -1
    for i in range(A.shape[0]):
        if i in working_set:
            continue
        else:
            denominator = np.dot(A[i], p)
            if denominator < 0:
                if alpha > (b[i] - np.dot(A[i], x))/denominator:
                    alpha = (b[i] - np.dot(A[i], x))/denominator
                    add_idx = i
    return alpha, add_idx


if __name__ == "__main__":
    """
    min 1/2 x^T G x + c^T x
    s.t. Ax >= b
    """
    G = np.array([[2, -2], [-2, 4]])
    c = np.array([-2, -6])
    A = np.array([[-0.5, -1], [1, -2], [1, 0], [0, 1]])
    b = np.array([-1, -2, 0, 0])

    x_0 = np.array([1, 1/2])
    initial_set = []
    active_set(x_0, initial_set)
