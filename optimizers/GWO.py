import random
import numpy as np
import math
from solution import solution
import time

def GWO(objf, lb, ub, dim, SearchAgents_no, Max_iter):
    Alpha_pos = np.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = np.zeros(dim)
    Beta_score = float("inf")

    Delta_pos = np.zeros(dim)
    Delta_score = float("inf")

    prey = np.zeros(3)

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    
    Positions = np.empty((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    for i in range(SearchAgents_no):
        fitness = objf(Positions[i, :])

        if fitness < Alpha_score:
            Delta_score = Beta_score
            Delta_pos = Beta_pos.copy()
            Beta_score = Alpha_score
            Beta_pos = Alpha_pos.copy()
            Alpha_score = fitness
            Alpha_pos = Positions[i, :].copy()
        elif fitness < Beta_score:
            Delta_score = Beta_score
            Delta_pos = Beta_pos.copy()
            Beta_score = fitness
            Beta_pos = Positions[i, :].copy()
        elif fitness < Delta_score:
            Delta_score = fitness
            Delta_pos = Positions[i, :].copy()

    Convergence_curve = np.zeros(Max_iter)
    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    for l in range(Max_iter):

        a = 2 - l * (2 / (Max_iter - 1))

        for i in range(SearchAgents_no):
            for j in range(dim):

                for k, leader in enumerate([Alpha_pos, Beta_pos, Delta_pos]):

                    r1 = random.random()
                    r2 = random.random()

                    A = 2 * a * r1 - a
                    C = 2 * r2

                    D = abs(C * leader[j] - Positions[i, j])
                    prey[k] = leader[j] - A * D
                

                Positions[i, j] = prey.mean()

        for i in range(SearchAgents_no):

            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            fitness = objf(Positions[i, :])

            if fitness < Alpha_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = Alpha_score
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                Alpha_pos = Positions[i, :].copy()
            elif fitness < Beta_score:
                Delta_score = Beta_score
                Delta_pos = Beta_pos.copy()
                Beta_score = fitness
                Beta_pos = Positions[i, :].copy()
            elif fitness < Delta_score:
                Delta_score = fitness
                Delta_pos = Positions[i, :].copy()

        Convergence_curve[l] = Alpha_score

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "GWO"
    s.objfname = objf.__name__

    return s
