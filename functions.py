import numpy as np


def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2


def rosen_grad(x):
    return np.array([400*pow(x[0],3) - 400*x[0]*x[1] + 2*(x[0]-1), -200*(x[0]**2-x[1])])


def rosen_hessian(x):
    # [[x1x1, x1x2], [x2x1, x2x2]]
    return np.array([[-400*x[1]+1200*x[0]**2+2, -400*x[0]], [-400*x[0], 200]])


def beale(x):
    t_1 = 1.5-x[0]+x[0]*x[1]
    t_2 = 2.25-x[0]+x[0]*x[1]**2
    t_3 = 2.625-x[0]+x[0]*pow(x[1],3)
    return t_1**2 + t_2**2 + t_3**2


def beale_grad(x):
    t_1 = 1.5-x[0]+x[0]*x[1]
    t_2 = 2.25-x[0]+x[0]*x[1]**2
    t_3 = 2.625-x[0]+x[0]*pow(x[1],3)
    grad_1 = 2*(t_1*(x[1]-1) + t_2*(x[1]**2-1) + t_3*(pow(x[1],3)-1))
    grad_2 = 2*x[0]*t_1 + 4*x[0]*x[1]*t_2 + 6*x[0]*(x[1]**2)*t_3
    return np.array([grad_1, grad_2])


def beale_hessian(x):
    t_1 = 1.5-x[0]+x[0]*x[1]
    t_2 = 2.25-x[0]+x[0]*x[1]**2
    t_3 = 2.625-x[0]+x[0]*pow(x[1],3)
    grad_1 = 2*((x[1]-1)**2 + (x[1]**2-1)**2 + (pow(x[1],3)-1)**2) # x1x1
    grad_2 = 2*x[0]*(x[1]-1) + 2*t_1 + 4*x[0]*x[1]*(x[1]**2-1) + 4*x[1]*t_2 + 6*x[0]*(x[1]**2)*(pow(x[1],3)-1) + 6*(x[1]**2)*t_3 # x1x2
    grad_3 = 2*(t_1+x[0]*x[1]-x[0]) + 4*x[1]*(t_2+x[0]*x[1]**2-x[0]) + 6*(x[1]**2)*(t_3+x[0]*pow(x[1],3)-x[0]) # x2x1
    grad_4 = 2*x[0]**2 + 4*x[0]*t_2 + 8*(x[0]**2)*(x[1]**2) + 12*x[0]*x[1]*t_3 + 18*(x[0]**2)*pow(x[1],4) # x2x2
    return np.array([[grad_1, grad_2], [grad_3, grad_4]])


def hilbert(dim):
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            matrix[i][j] = 1/((i+1)+(j+1)-1)
    return matrix


def diag(origin):
    dim = len(origin)
    matrix = np.zeros((dim,dim))
    for i in range(dim):
        matrix[i][i] = origin[i][i]
    return matrix


def quadratic(x, dim=8):
    A = hilbert(dim)
    b = np.ones(dim)
    return 0.5*np.dot(np.dot(x, A), x) - np.dot(b, x)


def quadratic_grad(x, dim=8):
    A = hilbert(dim)
    b = np.ones(dim)
    return np.dot(A, x) - b
