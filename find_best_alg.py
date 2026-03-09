import json
from pathlib import Path

dirnames = ["SiouxFalls", "Anaheim", "Barcelona", "Terrassa-Asymmetric", "Berlin-Mitte-Center", "Berlin-Tiergarten", "Eastern-Massachusetts"]

for dirname in dirnames:
    file = Path("./experiments_data/") / dirname / f"{dirname}.json"
    with open(file, "r") as fp:
        data: dict = json.load(fp)
        
    best_primal = 1e15
    best_alg = ""
    for model_name in data.keys():
        if model_name == "chp":
            continue
        pr = data[model_name]["primal"][-1]
        if pr <= best_primal:
            best_primal = pr
            best_alg = model_name
            
    print(f"{dirname}: {best_alg}")