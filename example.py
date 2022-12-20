from optimizer import run

optimizer = ["GWO", "WOA", "HHO", "FFA"]
optimizer = ["GWO_", "WOA_", "HHO_", "FFA_"]

objectivefunc = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12","F13","F14","F15","F16","F17","F18","F19","F20","F21","F22","F23"]

NumOfRuns = 30

params = {"PopulationSize": 30, "Iterations": 100}

export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
