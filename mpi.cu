#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

#define NUM_DEVICES 4

using namespace std;

#define CUDACHKERR(cudaErr) if (cudaErr != cudaSuccess) { \
    cerr << "Failed (error code " << cudaGetErrorString(cudaErr) << ")!" << endl; \
    exit(EXIT_FAILURE); \
}

void setDevice(int rank){
    cudaError_t cudaErr;
    int num_devices;
    cudaErr = cudaGetDeviceCount(&num_devices);
    CUDACHKERR(cudaErr);

    if (NUM_DEVICES > num_devices){
        cerr << "Error: number of devices (" << num_devices << ") is less than " << NUM_DEVICES << endl;
        exit(-1);
    }
    cudaSetDevice(rank % num_devices); // distribute ranks to devs
}

void getArrayBoundaries(int* start, int* end, int rank, int size, int numRanks){
    int y_size = size / numRanks; // number of strings per rank
    int remains = size - y_size * numRanks; // remaining strings

    int *y_starts = (int*)calloc(numRanks, sizeof(int));

    // distributing & computing number of strings
    y_starts[0] = 0;
    for (int i = 1; i < numRanks; i++){
        y_starts[i] = y_starts[i - 1] + y_size;
        if (remains > 0){
            y_starts[i]++;
            remains--;
        }
    }
    
    int *y_ends = (int*)calloc(numRanks, sizeof(int));

    // [start, end)
    y_ends[numRanks - 1] = size;
    if (rank != numRanks - 1)
        y_ends[rank] = y_starts[rank + 1];

    // getting final indices in [size*i + j] matrix
    *start = y_starts[rank] * size;
    *end = y_ends[rank] * size;

    free(y_starts);
    free(y_ends);
}

void interpolateHorizontal(double* arr, double leftValue, double rightValue, int startPosition, int size){
    arr[startPosition] = leftValue;
    arr[startPosition + size - 1] = rightValue;

    double step = (rightValue - leftValue) / ((double)size - 1);
    for (int i = startPosition + 1; i < startPosition + size - 1; i++)
        arr[i] = arr[i - 1] + step;
}

void interpolateVertical(double* arr, double topValue, double bottomValue, int startPos, int yPos, int numRows, int size){
    for (int i = 0; i < numRows; i++)
        arr[i * size + startPos] = (topValue * (size - 1 - i - yPos) + bottomValue * (i + yPos)) / (size - 1);
}

double* getSetMatrix(double* dst, int numElems, int size){
    cudaError_t cudaErr;

    double *matrix;
    cudaErr = cudaMalloc((void **)&matrix, (numElems + 2 * size) * sizeof(double));
    CUDACHKERR(cudaErr);

    // filling with zeros cuda matrix
    double *zeroMx = (double*)calloc(numElems + 2 * size, sizeof(double));
    cudaErr = cudaMemcpy(matrix, zeroMx, (numElems + 2 * size) * sizeof(double), cudaMemcpyHostToDevice);
    CUDACHKERR(cudaErr);

    // copying the matrix from the CPU to the GPU, with the space for the boundary values
    cudaErr = cudaMemcpy(matrix + size, dst, numElems * sizeof(double), cudaMemcpyHostToDevice);
    CUDACHKERR(cudaErr);

    free(zeroMx);
    return matrix;
}

__global__ void compute(double *newA, const double *A, int size, int y_start, int y_end){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < idx && idx < size - 1) && (y_start < idy && idy < y_end))
        newA[idy * size + idx] = 0.25 * (A[(idy - 1) * size + idx] + A[(idy + 1) * size + idx] +
                                          A[idy * size + (idx - 1)] + A[idy * size + (idx + 1)]);
}

__global__ void vecNeg(const double *newA, const double *A, double* ans, int size, int numElems){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size && idx < numElems + size)
        ans[idx] = newA[idx] - A[idx];
}

int main(int argc, char *argv[]){

    MPI_Status status; // declare mpi status var
    int rank, numRanks; // declare rank & amount of ranks vars
    MPI_Init(&argc, &argv); // mpi lib init

    double accuracy = atof(argv[1]);
    int size = atoi(argv[2]);
    int iters = atoi(argv[3]);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks); // get ranks amount

    cudaError_t cudaErr; // declare cuda errors catching var

    setDevice(rank); // assing rank to according device

    int start, end; // start & end !element! indices
    getArrayBoundaries(&start, &end, rank, size, numRanks);

    int numElems = end - start;
    int numRows = numElems / size;

    // interpolation for different processes
    double* tmp = (double*)calloc(numElems, sizeof(double));

    // on first and last rank we compute edge horizontal strings
    if (rank == 0)
        interpolateHorizontal(tmp, 10.0, 20.0, 0, size);
    if (rank == numRanks - 1)
        interpolateHorizontal(tmp, 20.0, 30.0, numElems - size, size);
    // set left & right column values
    interpolateVertical(tmp, 10.0, 20.0, 0, start / size, numRows, size);
    interpolateVertical(tmp, 20.0, 30.0, size - 1, start / size, numRows, size);

    // copying to GPU (+ 2 edge strings)
    double* A_d = getSetMatrix(tmp, numElems, size);
    double* Anew_d = getSetMatrix(tmp, numElems, size);
    free(tmp);

    dim3 GS = dim3(16, 16);
    dim3 BS = dim3(ceil(size / (double)GS.x), ceil((numRows + 2) / (double)GS.y));

    double *tmp_d;
    double *max_d;
    cudaMalloc(&tmp_d, sizeof(double) * numElems);
    cudaMalloc(&max_d, sizeof(double));

    // device-wide reduction
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // calculating ranks-neighbours (for next boundary values exchange)
    int topProcess = rank != numRanks - 1 ? rank + 1 : 0;
    int bottomProcess = rank != 0 ? rank - 1 : numRanks - 1;

    // calculating rows indices (not including edge strings) according to rank
    int y_start = rank == 0 ? 1 : 0;
    int y_end = rank == numRanks - 1 ? numRows : numRows + 1;

    int GS_neg = size;
    int BS_neg = ceil(numElems / (double)GS_neg);

    int iter = 0;
    double error = 1;
    double max = 0; // error value on current rank

    while (error > accuracy && iter++ < iters){
        
        // edge strings exchange
        //if (rank < numRanks - 1)
            MPI_Sendrecv(
                A_d + size, // начальный адрес буфера-отправителя
                size, // число элементов в буфере отправителя
                MPI_DOUBLE, // тип элементов в буфере отправителя
                bottomProcess, // номер процесса-получателя
                rank, // тег процесса-отправителя
                A_d + numElems + size, // начальный адрес приемного буфера
                size, // число элементов в приемном буфере
                MPI_DOUBLE, // тип элементов в приемном буфере
                topProcess, // номер процесса-отправителя
                topProcess, // тэг процесса-получателя
                MPI_COMM_WORLD, // коммуникатор (дескриптор)
                &status // статус
            );

        //if (rank > 0)
            MPI_Sendrecv(
                A_d + numElems,
                size,
                MPI_DOUBLE,
                topProcess,
                rank,
                A_d,
                size,
                MPI_DOUBLE,
                bottomProcess,
                bottomProcess,
                MPI_COMM_WORLD,
                &status
            );

        // compute new values with 5-point template
        compute<<<BS, GS>>>(Anew_d, A_d, size, y_start, y_end);

        if (iter % 100 == 0){
            vecNeg<<<BS_neg, GS_neg>>>(Anew_d, A_d, tmp_d, size, numElems);

            cudaErr = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, numRows * size);
            CUDACHKERR(cudaErr);

            cudaErr = cudaMemcpy(&max, max_d, sizeof(double), cudaMemcpyDeviceToHost);
            CUDACHKERR(cudaErr);

            MPI_Allreduce(&max, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            if (rank == 0)
                cout << iter << " " << error << endl;
        }
        
        double *tmp = A_d;
        A_d = Anew_d;
        Anew_d = tmp;
    }

    cudaFree(A_d);
    cudaFree(Anew_d);
    cudaFree(tmp_d);
    cudaFree(max_d);

    MPI_Finalize();

    return 0;
}

// почему число итераций меняется в зависимости от числа ранков
// как между собой общаются первый и последний ранк
// 