#!/bin/bash

mpic++ -o rgraph_corel -DVERIFY -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ rgraph.cpp
./rgraph_corel datasets/izbicki15/small/corel.fvecs 128 0.0515 results.json
