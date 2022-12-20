import random
import numpy as np
import math
from solution import solution
import time


def WOA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    Leader_pos = np.zeros(dim)
    Leader_score = float("inf")  

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    convergence_curve = np.zeros(Max_iter)

    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    t = 0

    for i in range(SearchAgents_no):
        fitness = objf(Positions[i, :])

        if fitness < Leader_score:
            Leader_score = fitness
            Leader_pos = Positions[i, :].copy()

    while t < Max_iter:
        a = 2 - t * ((2) / (Max_iter - 1))

        for i in range(SearchAgents_no):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            A = 2 * a * r1 - a  
            C = 2 * r2  

            b = 1
            
            l = 2 * np.random.rand(dim) - 1

            p = random.random()

            if p < 0.5:
                if np.linalg.norm(A) >= 1:
                    rand_leader_index = random.randrange(SearchAgents_no)
                    X_rand = Positions[rand_leader_index, :]
                    D_X_rand = abs(C * X_rand - Positions[i])
                    Positions[i] = X_rand - A * D_X_rand

                else:
                    D_Leader = abs(C * Leader_pos - Positions[i])
                    Positions[i] = Leader_pos - A * D_Leader

            else:
                distance2Leader = abs(Leader_pos - Positions[i])

                Positions[i] = (
                    distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi)
                    + Leader_pos
                )

        for i in range(SearchAgents_no):
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            fitness = objf(Positions[i, :])

            if fitness < Leader_score:
                Leader_score = fitness
                Leader_pos = Positions[i, :].copy()

        convergence_curve[t] = Leader_score
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "WOA"
    s.objfname = objf.__name__
    s.best = Leader_score
    s.bestIndividual = Leader_pos

    return s
