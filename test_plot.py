import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

import matplotlib.pyplot as plt
import numpy as np
import benchmarks
from tqdm.auto import tqdm
from matplotlib import cm, ticker
from mpl_toolkits.mplot3d import Axes3D


# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(15, 20))

# =============
# First subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(4, 5, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
fig.colorbar(surf, shrink=0.5, aspect=10)


for i in range(1, 19):
    if i in [15]:  # not possible in 2d
        continue
    ax = fig.add_subplot(4, 5, i, projection='3d')

    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    f = benchmarks.getFunctionDetails(f'F{i}')
    n = 200
    name, lb, ub, dim, fmin = f
    f = getattr(benchmarks, name)
    x = np.linspace(lb, ub, n)
    y = np.linspace(lb, ub, n)
    X, Y = np.meshgrid(x, y)

    Z = np.empty_like(X)
    for i in tqdm(range(n)):
        for j in range(n):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    Z -= Z.min()
    Z += 10
    ax.contour3D(X, Y, Z, 100, cmap='viridis')

    num_of_lines = 500
    locator = ticker.LogLocator(base=2)
    ax.contourf(X, Y, Z, num_of_lines, locator=locator, zdir='z', offset=Z.min(), cmap='viridis')
    ax.contour(X, Y, Z, 5, zdir='z', locator=locator, offset=Z.min(), cmap='binary')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_title(name)
plt.show()