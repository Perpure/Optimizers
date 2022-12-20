import matplotlib.pyplot as plt
import pandas as pd
import math


def run(results_directory, optimizers, objectivefuncs, Iterations):
    if len(objectivefuncs) == 0:
        return
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")
    fig = plt.figure(figsize=(16, 24), dpi=600)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n = math.ceil(len(objectivefuncs) / 4)

    for j in range(0, len(objectivefuncs)):
        ax = fig.add_subplot(n, 4, j + 1)
        objective_name = objectivefuncs[j]
        allGenerations = [x + 1 for x in range(Iterations)]
        for i in range(len(optimizers)):
            optimizer_name = optimizers[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            row = row.iloc[:, 25:]
            ax.plot(allGenerations[20:], row.values.tolist()[0], label=optimizer_name)
        ax.grid()
        ax.set_title(objective_name)
        ax.legend()
        ax.set_xlabel('Итерации')
        ax.set_ylabel('Значение функции')



    fig_name = results_directory + "/convergence-all.png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()
    # plt.show()

def run_all(results_directory):
    pass