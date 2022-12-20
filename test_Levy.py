import math
import numpy as np
import matplotlib.pyplot as plt
def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

def Levy2(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.normal(0, sigma, dim)
    v = np.random.normal(0, 1, dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

point = np.zeros(2)
n_steps = 1000
xs, ys = np.zeros(n_steps), np.zeros(n_steps)
deltas = []
for i in range(1, n_steps):
    delta = Levy(2)
    deltas.append(delta)
    point += delta
    xs[i], ys[i] = point
plt.plot(xs, ys)
plt.show()
deltas = np.array(deltas)
print(deltas.mean(axis=0))