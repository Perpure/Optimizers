import random
import numpy as np
import math
from solution import solution
import time

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


def HHO_(objf_old, lb, ub, dim, SearchAgents_no, Max_iter):

    Rabbit_Energy = float("inf")
    Rabbit_Location = np.zeros(dim)

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    shift = np.ones(dim) * 5
    lb = np.asarray(lb) + shift
    ub = np.asarray(ub) + shift
    objf = lambda x: objf_old(x - shift)

    X = np.asarray(
        [x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    convergence_curve = np.zeros(Max_iter)

    s = solution()

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    

    t = 0  

    for i in range(SearchAgents_no):
        fitness = objf(X[i])
        if fitness < Rabbit_Energy:
            Rabbit_Energy = fitness
            Rabbit_Location = X[i].copy()
    
    while t < Max_iter:
        E1 = 2 * (1 - (t / Max_iter))  

        
        for i in range(0, SearchAgents_no):

            E0 = 2 * random.random() - 1  
            Escaping_Energy = E1 * (
                E0
            )  

            

            if abs(Escaping_Energy) >= 1:
                
                q = random.random()
                rand_Hawk_index = math.floor(SearchAgents_no * random.random())
                X_rand = X[rand_Hawk_index]
                if q < 0.5:
                    
                    X[i] = X_rand - random.random() * abs(
                        X_rand - 2 * random.random() * X[i]
                    )

                elif q >= 0.5:
                    
                    X[i] = (Rabbit_Location - X.mean(0)) - random.random() * (
                        (ub - lb) * random.random() + lb
                    )

            
            elif abs(Escaping_Energy) < 1:

                r = random.random()  
                if (r >= 0.5 and abs(Escaping_Energy) < 0.5):
                    X[i] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i]
                    )

                elif (r >= 0.5 and abs(Escaping_Energy) >= 0.5):
                    Jump_strength = 2 * random.random()
                    X[i] = (Rabbit_Location - X[i]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i]
                    )

                elif (r < 0.5 and abs(Escaping_Energy) >= 0.5):
                    
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i]
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  
                        X[i] = X1.copy()
                    else:  
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i])
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i] = X2.copy()
                else:
                    Jump_strength = 2 * (1 - random.random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  
                        X[i] = X1.copy()
                    else:  
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i] = X2.copy()

        for i in range(0, SearchAgents_no):
            X[i] = np.clip(X[i], lb, ub)
            fitness = objf(X[i])

            if fitness < Rabbit_Energy:  
                Rabbit_Energy = fitness
                Rabbit_Location = X[i].copy()
        convergence_curve[t] = Rabbit_Energy
        t += 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = Rabbit_Energy
    s.bestIndividual = Rabbit_Location
    
    
    
    return s


