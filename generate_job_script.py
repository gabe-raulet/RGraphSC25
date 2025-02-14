import sys
import json
import getopt

def main(experiments_json, results_dir, minutes, dataset_choices):
    experiments = json.load(open(experiments_json, "r"))
    dataset_choices = dataset_choices.split(",")

    partition = "debug" if minutes <= 30 else "regular"
    num_nodes = 1

    for dataset in dataset_choices:
        num_nodes = max(num_nodes, experiments[dataset]["num_nodes"])
        if num_nodes > 4: partition = "regular"

    script = "#!/bin/bash\n\n"
    script += f"#SBATCH -N {num_nodes}\n"
    script += f"#SBATCH -C cpu\n"
    script += f"#SBATCH -q {partition}\n"
    script += f"#SBATCH --mail-user=ghraulet@lbl.gov\n"
    script += f"#SBATCH --mail-type=ALL\n"
    script += f"#SBATCH -t {minutes}\n\n"

    script += f"rm -rf {results_dir}\nmkdir {results_dir}\n\n"

    for dataset in dataset_choices:
        exp = experiments[dataset]
        script += f"CC -o {exp['rgraph_mpi']} -DDIM_SIZE={exp['dim']} -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp\n"

    script += "\n"

    for dataset in dataset_choices:
        exp = experiments[dataset]
        epsilons = " ".join([str(r) for r in exp["epsilons"]])
        proc_counts = " ".join([str(p) for p in exp["num_ranks"]])
        cell_counts = " ".join([str(m) for m in exp["cell_counts"]])
        script += f"for EPSILON in {epsilons}\n"
        script += f"    do\n"
        script += f"    for PROC_COUNT in {proc_counts}\n"
        script += f"        do\n"
        script += f"        for CELL_COUNT in {cell_counts}\n"
        script += f"            do\n"
        script += f"            RESULT={results_dir}/mpi.{dataset}.r${{EPSILON}}.m${{CELL_COUNT}}.p${{PROC_COUNT}}.json\n"
        script += f"            srun -n $PROC_COUNT -N {exp['num_nodes']} --cpu_bind=cores ./{exp['rgraph_mpi']} -c {exp['covering_factor']} -o $RESULT {exp['dataset']} $CELL_COUNT $EPSILON\n"
        script += f"        done\n"
        script += f"    done\n"
        script += f"done\n\n"

    script += f"tar cvzf {results_dir}.tar.gz {results_dir}\n"

    sys.stdout.write(script)
    sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.stderr.write(f"Usage: {sys.argv[0]} <experiments.json> <results> <minutes> [dataset1,dataset2,...]\n")
        sys.stderr.flush()
        sys.exit(-1)
    else:
        sys.exit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]))
