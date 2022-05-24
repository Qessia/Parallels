SHELL := /bin/bash
USECUBLAS := 1
.PHONY: all clean cuda cuda-gdb nsys mpi

all: blas

clean:
	rm -rf *.out
	rm -rf *.nsys-rep

SIZE=128

cpu: parallel2.cpp
	pgc++ parallel2.cpp -o parallel2
	time ./parallel2 0.000001 $(SIZE) 1000000

gpu: parallel2.cpp
	pgc++ -acc -Minfo=accel parallel2.cpp -o parallel2
	time ./parallel2 0.000001 $(SIZE) 1000000

cublas: parallel2.cpp
	pgc++ -acc -cudalib=cublas -Minfo=accel parallel2.cpp -o parallel2
	time ./parallel2 0.000001 $(SIZE) 1000000

FILE=mpi.cu

cuda: $(FILE)
	nvcc $(FILE) -arch sm_70 -o cuda
	time ./cuda 0.000001 $(SIZE) 1000000

cuda-gdb:
	cuda-gdb -q --cuda-use-lockfile=0 -x debug.txt ./cuda

nsys: $(FILE)
	nvcc $(FILE) -arch sm_70 -o cuda -g
	nsys profile -o cudanew -t openacc,cuda ./cuda 0.000001 128 1000000

NUM_RANKS=4

mpi: $(FILE)
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -arch sm_70 $(FILE) -o mpibin
	time UCX_WARN_UNUSED_ENV_VARS=n mpirun -mca pml ucx -x UCX_TLS=cuda,sm,posix -np $(NUM_RANKS) ./mpibin 0.000001 $(SIZE) 1000000

sanitizer: $(FILE)
	nvcc $(FILE) -arch sm_70 -o cuda -g
	compute-sanitizer --leak-check full ./cuda 0.000001 $(SIZE) 1000000