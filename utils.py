import numpy as np
import matplotlib.pyplot as plt


def draw_contour(func, data, dens=30, margin=0.2):
    axis_x_min, axis_y_min = data.min(0)
    axis_x_max, axis_y_max = data.max(0)

    x1 = np.linspace(axis_x_min-margin, axis_x_max+margin, 200)
    x2 = np.linspace(axis_y_min-margin, axis_y_max+margin, 200)
    x, y = np.meshgrid(x1, x2)
    value = func([x, y])
    plt.xlabel('x1')
    plt.ylabel('x2')

    contour = plt.contour(x, y, value, dens)
    plt.clabel(contour, fontsize=6, colors='k')
    plt.plot(data[:, 0], data[:, 1], c='r', lw=0.5, ls='--', marker='x', markersize=3)
    plt.title('{} function | {} Iterations'.format(func.__name__, len(data) - 1))
    plt.show()


def convergence_rate(func, data):
    value = [func(i) for i in data]

    plt.plot(value, label='value')
    plt.xlabel('Iterations')
    plt.ylabel('$f(x)$')

    plt.title('Convergence rate | {} Iterations'.format(len(value) - 1))
    plt.legend()
    plt.show()