#!/bin/bash

#SBATCH -N 8
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH --mail-user=ghraulet@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 120

DIM=784
RGRAPH=rgraph_mpi_mnist
DATASET=../datasets/izbicki15/large/mnist.fvecs

COVERING_FACTOR=1.1

rm -rf mnist_results
mkdir mnist_results

CC -o $RGRAPH -DDIM_SIZE=$DIM -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

for EPSILON in 750 800 850
    do
    for PROC_COUNT in 64 128 256 512 1024
        do
        for CELL_COUNT in 512 1024 2048
            do
            RESULT=mnist_results/mpi.mnist.r${EPSILON}.m${CELL_COUNT}.p${PROC_COUNT}.c$COVERING_FACTOR.json
            srun -n $PROC_COUNT -N 8 --cpu_bind=cores ./$RGRAPH -c $COVERING_FACTOR -o $RESULT $DATASET $CELL_COUNT $EPSILON
        done
    done
done
