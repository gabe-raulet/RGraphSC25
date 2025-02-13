#!/bin/bash

mpic++ -o rgraph_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
mpic++ -o rgraph_mpi_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

export OMP_NUM_THREADS=12

DATASET=datasets/izbicki15/small/corel.fvecs

rm -rf results
mkdir -p results

for EPSILON in 0.05 0.10
    do
    for CELL_COUNT in 8 16 32 64
        do
        ./rgraph_corel -o results/serial.r${EPSILON}.m${CELL_COUNT}.json $DATASET $CELL_COUNT $EPSILON
        for PROC_COUNT in 1 2 4 8
            do
            mpirun -np $PROC_COUNT ./rgraph_mpi_corel -o results/mpi.r${EPSILON}.m${CELL_COUNT}.p${PROC_COUNT}.json $DATASET $CELL_COUNT $EPSILON
        done
    done
done
