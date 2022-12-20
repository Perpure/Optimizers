from tqdm import tqdm
from pathlib import Path
import importlib
import benchmarks
import csv
import numpy as np
import time
import warnings
import os

import plot_bars_min
import plot_convergence as conv_plot
import plot_boxplot as box_plot
import plot_bars

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]
    module = importlib.import_module(f'optimizers.{algo}')
    func = getattr(module, algo)
    return func(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)


def run(optimizers, objectivefuncs, NumOfRuns, params, export_flags):

    
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    
    Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    Export_convergence = export_flags["Export_convergence"]
    Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    
    CnvgHeader = []

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for i in range(len(optimizers)):
        print(optimizers[i])
        for j in tqdm(range(0, len(objectivefuncs))):
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            errors = np.zeros(NumOfRuns)
            for k in range(0, NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objectivefuncs[j])
                func_details, target = func_details[:-1], func_details[-1]
                x = selector(optimizers[i], func_details, PopulationSize, Iterations)
                convergence[k] = x.convergence
                errors[k] = x.convergence[-1] - target
                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  
                            header = np.concatenate(
                                [["Optimizer", "objfname", "ExecutionTime", "Error"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  
                        executionTime[k] = x.executionTime
                        a = np.concatenate(
                            [[x.optimizer, x.objfname, x.executionTime, errors[k]], x.convergence]
                        )
                        writer.writerow(a)
                    out.close()

            if Export == True:
                ExportToFile = results_directory + "experiment.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  
                        header = np.concatenate(
                            [["Optimizer", "objfname", "ExecutionTime", "MinError", "MeanError"], CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True

                    avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                    avgConvergence = np.around(
                        np.mean(convergence, axis=0, dtype=np.float64), decimals=2
                    ).tolist()
                    a = np.concatenate(
                        [[optimizers[i], objectivefuncs[j], avgExecutionTime, np.min(errors), errors.mean()], avgConvergence]
                    )
                    writer.writerow(a)
                out.close()

    if Export_convergence == True:
        conv_plot.run(results_directory, optimizers, objectivefuncs, Iterations)
    plot_bars.run(results_directory, optimizers, objectivefuncs)
    plot_bars_min.run(results_directory, optimizers, objectivefuncs)

    print("Execution completed")
