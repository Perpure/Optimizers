import numpy as np
import math
import time
from solution import solution


def FFA_(objf_old, lb, ub, dim, n, MaxGeneration):

    alpha = 0.5
    gamma = 1
    beta0 = 1
    betamin = 0.2

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    shift = np.ones(dim) * 5
    lb = np.asarray(lb) + shift
    ub = np.asarray(ub) + shift
    objf = lambda x: objf_old(x - shift)
     
    fireflies = np.zeros((n, dim))
    for i in range(dim):
        fireflies[:, i] = np.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]
    intensities = np.empty(n)

    convergence = []
    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    best = np.inf
    scale = []
    for b in range(dim):
        scale.append(abs(ub[b] - lb[b]))
    scale = np.array(scale)
    for i in range(0, n):
        intensities[i] = objf(fireflies[i, :])
        best = min(best, intensities[i])
    for k in range(0, MaxGeneration):
        alpha *= 0.95
        idxs = np.argsort(intensities)
        intensities = intensities[idxs]
        fireflies = fireflies[idxs]
        for i in range(1, n):
            for j in range(n):
                if intensities[i] > intensities[j]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = (beta0 - betamin) * np.exp(-gamma * r ** 2) + betamin
                    tmpf = alpha * (np.random.rand(dim) - 0.5) * scale
                    fireflies[i] = fireflies[i] * (1 - beta) + fireflies[j] * beta + tmpf
                    fireflies[i] = np.clip(fireflies[i], lb, ub)
                    intensities[i] = objf(fireflies[i])
                    best = min(best, intensities[i])

        convergence.append(best)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "FFA"
    s.objfname = objf.__name__

    return s
