import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import pandas as pd

from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg')

#  def main(results_dir):

#  if __name__ == "__main__":
    #  if len(sys.argv) != 2:
        #  sys.stderr.write(f"Usage: {sys.argv[0]} <results>\n")
        #  sys.stderr.flush()
        #  sys.exit(-1)
    #  else:
        #  sys.exit(main(sys.argv[1]))

def filter_exp(exp):
    d = {}
    dist_comps = 0
    d["dataset"] = exp["dataset"].split("/")[-1]
    d["epsilon"] = exp["epsilon"]
    d["num_ranks"] = exp["num_ranks"]
    d["num_sites"] = exp["num_sites"]
    d["total_time"] = exp["total_time"]
    d["density"] = exp["build_epsilon_graph"]["density"]
    d["num_edges"] = exp["build_epsilon_graph"]["num_edges"]
    d["build_epsilon_graph_time"] = exp["build_epsilon_graph"]["time"]
    d["build_epsilon_graph_dist_comps"] = exp["build_epsilon_graph"]["dist_comps"]
    d["build_ghost_trees_time"] = exp["build_ghost_trees"]["time"]
    d["build_ghost_trees_dist_comps"] = exp["build_ghost_trees"]["dist_comps"]
    d["build_greedy_net_time"] = exp["build_greedy_net"]["time"]
    d["build_greedy_net_dist_comps"] = exp["build_greedy_net"]["dist_comps"]
    d["build_replication_tree_time"] = exp["build_replication_tree"]["time"]
    d["build_replication_tree_dist_comps"] = exp["build_replication_tree"]["dist_comps"]
    d["compute_cell_assignments_time"] = exp["compute_cell_assignments"]["time"]
    d["compute_ghost_points_time"] = exp["compute_ghost_points"]["time"]
    d["compute_ghost_points_dist_comps"] = exp["compute_ghost_points"]["dist_comps"]
    d["num_ghost_points"] = exp["compute_ghost_points"]["num_ghost_points"]
    dist_comps += exp["build_epsilon_graph"]["dist_comps"]
    dist_comps += exp["build_ghost_trees"]["dist_comps"]
    dist_comps += exp["build_greedy_net"]["dist_comps"]
    dist_comps += exp["build_replication_tree"]["dist_comps"]
    dist_comps += exp["compute_ghost_points"]["dist_comps"]
    d["dist_comps"] = dist_comps
    return d

def read_experiments(results_dir):
    exps = []
    for result_json in Path(results_dir).glob("*.json"):
        exp = json.load(open(str(result_json), "r"))
        if "num_ranks" in exp:
            exps.append(filter_exp(exp))
    data = pd.DataFrame(exps)
    #  data = data.sort_values(by=["epsilon", "num_ranks", "dist_comps"], ascending=[True, True, True])
    return data

def heatmap(data, num_sites):
    df = data[data["num_sites"] == num_sites]
    df = df[["epsilon", "num_ranks", "density", "total_time"]]
    num_ranks = sorted(list(set(df["num_ranks"])))
    epsilons = sorted(list(set(df["epsilon"])))
    rows = len(num_ranks)
    cols = len(epsilons)
    runtimes = np.zeros((rows, cols), dtype=np.float64)
    densities = [0] * cols
    for i in range(rows):
        p = num_ranks[i]
        for j in range(cols):
            r = epsilons[j]
            item = df[(df["num_ranks"] == p) & (df["epsilon"] == r)]
            runtimes[i,j] = item["total_time"].values[0]
            if i == 0: densities[j] = round(item["density"].values[0], 2)
    im = ax.imshow(runtimes)
    ax.set_xticks(range(cols), labels=densities, rotation=45)
    ax.set_yticks(range(rows), labels=num_ranks)
    for i in range(rows):
        for j in range(cols):
            text = ax.text(j, i, round(runtimes[i,j], 2), ha="center", va="center", color="w")
    ax.set_ylabel("Processor count")
    ax.set_xlabel("Average neighborhood size")
    return im

results_dir = "results"
data = read_experiments(results_dir)

fig, ax = plt.subplots()

heatmap(data, 128)
plt.title(f"Runtime (s) (num_sites={128})")

plt.plot()
plt.show()

