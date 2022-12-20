import math

import matplotlib.pyplot as plt
import numpy as np
import benchmarks
from tqdm.auto import tqdm
from matplotlib import cm, ticker
from mpl_toolkits.mplot3d import Axes3D

def draw_3d_fig(X, Y, Z, name):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 100, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(name)

    plt.savefig(f"imgs/{name}.png")
    plt.clf()
    plt.close('all')

def draw_projections(X, Y, Z, lb, ub, name):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    num_of_lines = 500
    locator = ticker.LogLocator(base=2)
    ax.contourf(X, Y, Z, num_of_lines, zdir='x', offset=lb, cmap='viridis')
    # ax.contour(X, Y, Z, 20, zdir='x', locator=locator, offset=lb, cmap='binary')
    ax.contourf(X, Y, Z, num_of_lines, zdir='y', offset=ub, cmap='viridis')
    # ax.contour(X, Y, Z, 20, zdir='y', locator=locator, offset=ub, cmap='binary')
    ax.contourf(X, Y, Z, num_of_lines, locator=locator, zdir='z', offset=Z.min(), cmap='viridis')
    ax.contour(X, Y, Z, 5, zdir='z', locator=locator, offset=Z.min(), cmap='binary')
    # fig.colorbar(p)
    ax.set_title(name)
    plt.savefig(f"imgs/{name}-proj.png")
    plt.clf()
    plt.close('all')

def draw_scaled(X, Y, Z, f, lb, ub, name, zoom=300):
    min_x, min_y = np.unravel_index(Z.argmin(), Z.shape)

    delta = (ub - lb) / zoom
    n = 100
    lb_x, ub_x = X[min_x, min_y] - delta, X[min_x, min_y] + delta
    lb_y, ub_y = Y[min_x, min_y] - delta, Y[min_x, min_y] + delta
    x = np.linspace(lb_x, ub_x, n)
    y = np.linspace(lb_y, ub_y, n)
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    for i in tqdm(range(n)):
        for j in range(n):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    Z -= Z.min()
    Z += 1e-5
    # delta = 10

    # X = X[min_x - delta:min_x + delta, min_y - delta:min_y + delta]
    # Y = Y[min_x - delta:min_x + delta, min_y - delta:min_y + delta]
    # Z = Z[min_x - delta:min_x + delta, min_y - delta:min_y + delta]
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    num_of_lines = 20
    locator = ticker.LogLocator(base=1.5)
    ax.contourf(X, Y, Z, num_of_lines, zdir='x', offset=lb_x, cmap='viridis')
    # ax.contour(X, Y, Z, 20, zdir='x', locator=locator, offset=lb, cmap='binary')
    ax.contourf(X, Y, Z, num_of_lines, zdir='y', offset=ub_y, cmap='viridis')
    # ax.contour(X, Y, Z, 20, zdir='y', locator=locator, offset=ub, cmap='binary')
    ax.contourf(X, Y, Z, num_of_lines, locator=locator, zdir='z', offset=Z.min(), cmap='viridis')
    ax.contour(X, Y, Z, 10, zdir='z', locator=locator, offset=Z.min(), cmap='binary')
    # fig.colorbar(p)
    ax.set_title(name)
    plt.savefig(f"imgs/{name}-scaled-{zoom}-lvls.png")
    plt.clf()
    plt.tight_layout()
    plt.close('all')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 100, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(name)

    plt.savefig(f"imgs/{name}-scaled-{zoom}.png")
    plt.clf()
    plt.close('all')

def draw_plots(objf, n=200):
    name, lb, ub, dim, _ = objf
    f = getattr(benchmarks, name)
    x = np.linspace(lb, ub, n)
    y = np.linspace(lb, ub, n)
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    for i in tqdm(range(n)):
        for j in range(n):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    Z -= Z.min()
    Z += 1e-5
    draw_3d_fig(X, Y, Z, name)
    draw_projections(X, Y, Z, lb, ub, name)
    draw_scaled(X, Y, Z, f, lb, ub, name)
    draw_scaled(X, Y, Z, f, lb, ub, name, zoom=4)

def draw_all_plots(fig, objf, k, n=200):
    name, lb, ub, dim, fmin = objf
    f = getattr(benchmarks, name)
    x = np.linspace(lb, ub, n)
    y = np.linspace(lb, ub, n)
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    for i in tqdm(range(n)):
        for j in range(n):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    Z -= Z.min()
    Z += 1e-5
    ax = fig.add_subplot(n, 4, k + 1, projection='3d')
    ax.contour3D(X, Y, Z, 100, cmap='viridis', antialiased=False)

    num_of_lines = 500
    locator = ticker.LogLocator(base=2)
    ax.contourf(X, Y, Z, num_of_lines, locator=locator, zdir='z', offset=Z.min(), cmap='viridis', antialiased=False)
    ax.contour(X, Y, Z, 5, zdir='z', locator=locator, offset=Z.min(), cmap='binary', antialiased=False)


    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(name)




if __name__ == "__main__":
    # name, lb, ub, dim = benchmarks.getFunctionDetails(f'F18')
    # f = getattr(benchmarks, name)
    # print(f([0, 0]))
    funcs = 19
    fig = plt.figure(figsize=(20, 25))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n = math.ceil(funcs / 4)
    for i in range(1, 2):
    # for i in range(18, 19):
        if i in [15]: # not possible in 2d
            continue
        draw_all_plots(fig, benchmarks.getFunctionDetails(f'F{i}'), i)

    plt.savefig(f"imgs/all.png")
    plt.show()