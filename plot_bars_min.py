import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run(results_directory, optimizers, objectivefuncs):
    if len(objectivefuncs) == 0:
        return
    plt.ioff()
    fileResultsData = pd.read_csv(results_directory + "/experiment.csv")
    fig = plt.figure(figsize=(20, 30))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    n = math.ceil(len(objectivefuncs) / 4)

    for j in range(0, len(objectivefuncs)):
        ax = fig.add_subplot(n, 4, j + 1)
        objective_name = objectivefuncs[j]

        errors = []
        for i in range(len(optimizers)):
            optimizer_name = optimizers[i]

            row = fileResultsData[
                (fileResultsData["Optimizer"] == optimizer_name)
                & (fileResultsData["objfname"] == objective_name)
            ]
            err = row.iloc[:, 4].tolist()[0]
            errors.append(err)
        bars = ax.bar(optimizers, errors)
        ax.bar_label(bars, padding=10, label_type='center', fmt='%.1e')
        ax.set_title(objective_name)
        ax.set_ylabel('Ошибка')



    fig_name = results_directory + "/errors-all-mean.png"
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()
