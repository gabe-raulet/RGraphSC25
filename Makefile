DEBUG?=0
LOG?=0
D?=20
FLAGS=-std=c++20 -fopenmp

ifeq ($(shell uname -s),Linux)
COMPILER=CC
MPI_COMPILER=CC
else
COMPILER=clang++
MPI_COMPILER=mpic++
endif

ifeq ($(DEBUG),1)
FLAGS+=-O0 -g -fsanitize=address -fno-omit-frame-pointer -DDEBUG
else
FLAGS+=-O2
endif

ifeq ($(LOG),1)
FLAGS+=-DLOG
endif

all: rgraph_serial

rgraph_serial: rgraph_serial.cpp
	$(MPI_COMPILER) -o $@ -DDIM_SIZE=$(D) $(FLAGS) -I./ $<

clean:
	rm -rf rgraph_serial rgraph_serial_* *.out *.dSYM

