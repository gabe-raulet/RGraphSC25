#!/bin/bash

mpic++ -o rgraph_serial_artificial40 -DDIM_SIZE=40 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_corel -DDIM_SIZE=32 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_faces -DDIM_SIZE=20 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_covtype -DDIM_SIZE=55 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_mnist -DDIM_SIZE=784 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_tinyImages100k -DDIM_SIZE=384 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
mpic++ -o rgraph_serial_twitter -DDIM_SIZE=78 -std=c++20 -fopenmp -O2 -I./ rgraph_serial.cpp &
