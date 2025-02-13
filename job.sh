#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH --mail-user=ghraulet@lbl.gov
#SBATCH --mail-type=ALL
#SBATCH -t 30

CC -o rgraph_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
CC -o rgraph_mpi_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

CC -o rgraph_faces -DDIM_SIZE=20 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
CC -o rgraph_mpi_faces -DDIM_SIZE=20 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

CC -o rgraph_artificial40 -DDIM_SIZE=40 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph.cpp
CC -o rgraph_mpi_artificial40 -DDIM_SIZE=40 -std=c++20 -fopenmp -O2 -I./ -I./include rgraph_mpi.cpp

export OMP_NUM_THREADS=128

FACES=datasets/izbicki15/small/faces.fvecs
COREL=datasets/izbicki15/small/corel.fvecs
ARTIFICIAL40=datasets/izbicki15/small/artificial40.fvecs

#rm -rf test_results
#mkdir -p test_results

for EPSILON in 20 40 60 80 100 120 140
    do
    srun -N 1 -n 64 --cpu_bind=cores ./rgraph_mpi_faces -o test_results/mpi.faces.r${EPSILON}.json $FACES 256 $EPSILON
done

for EPSILON in 0.1 0.2 0.3 0.4 0.5
    do
    srun -N 1 -n 64 --cpu_bind=cores ./rgraph_mpi_corel -o test_results/mpi.corel.r${EPSILON}.json $COREL 256 $EPSILON
done

for EPSILON in 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0
    do
    srun -N 1 -n 64 --cpu_bind=cores ./rgraph_mpi_artificial40 -o test_results/mpi.artificial40.r${EPSILON}.json $ARTIFICIAL40 256 $EPSILON
done

#for EPSILON in 20 40 60 80
#    do
#    for CELL_COUNT in 16 32 64 128
#        do
#        ./rgraph_faces -o test_results/serial.r${EPSILON}.m${CELL_COUNT}.json $FACES $CELL_COUNT $EPSILON
#        for PROC_COUNT in 1 2 4 8 16
#            do
#            srun -N 1 -n $PROC_COUNT --cpu_bind=cores ./rgraph_mpi_faces -o test_results/mpi.r${EPSILON}.m${CELL_COUNT}.p${PROC_COUNT}.json $FACES $CELL_COUNT $EPSILON
#        done
#    done
#done
