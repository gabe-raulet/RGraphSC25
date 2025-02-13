#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH --mail-user=ghraulet@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 30

#CC -o rgraph_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
#CC -o rgraph_mpi_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

CC -o rgraph_faces -DDIM_SIZE=20 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
CC -o rgraph_mpi_faces -DDIM_SIZE=20 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

export OMP_NUM_THREADS=128

#DATASET=datasets/izbicki15/small/corel.fvecs
DATASET=datasets/izbicki15/small/faces.fvecs

rm -rf results
mkdir -p results

for EPSILON in 20 40 60 80
    do
    for CELL_COUNT in 16 32 64 128
        do
        ./rgraph_faces -o results/serial.r${EPSILON}.m${CELL_COUNT}.json $DATASET $CELL_COUNT $EPSILON
        for PROC_COUNT in 1 2 4 8 16
            do
            srun -N 1 -n $PROC_COUNT --cpu_bind=cores ./rgraph_mpi_faces -o results/mpi.r${EPSILON}.m${CELL_COUNT}.p${PROC_COUNT}.json $DATASET $CELL_COUNT $EPSILON
        done
    done
done
