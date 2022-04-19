#include <iostream>
#include <stdlib.h>
#include <cub/cub.cuh>

using namespace std;

// 5-point template cell computation (on device)
__global__ void compute(double* CAnew, double* CA, int n){
    // getting indices (1 thread = 1 cell)
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	
    // compute considering 1-wide "padding"
	if((i > 0) && (i < n - 1) && (j > 0) && (j < n - 1))
        CAnew[i*n + j] = 0.25 * (CA[(i+1)*n + j] + CA[(i-1)*n + j] + CA[i*n + j-1] + CA[i*n + j+1]);
}

// Grid-wide Max Reduction
__global__ void Max_Reduction(double* CA, double* CAnew, int n, double* BlockErr){
    // redefining BlockReduce for 16x16 block
    typedef cub::BlockReduce<double, 16, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 16> BlockReduce;
    // allocating shared memory
    __shared__ typename BlockReduce::TempStorage temp_storage;
    // initializing thread data
    double thread_data=0.0;

    // getting indices
    int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
    if ((i > 0) && (i < n - 1) && (j > 0) && (j < n - 1))
        // computing difference for every cell
        thread_data = CAnew[i*n + j] - CA[i*n + j];
    // Block-wide reduce
    double aggregate = BlockReduce(temp_storage).Reduce(thread_data, cub::Max());
    // synchronizing threads
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0)
        // put result to grid-sized matrix
        BlockErr[blockIdx.y*gridDim.x + blockIdx.x] = aggregate;
}

int main(int argc, char *argv[]){
    double accuracy = atof(argv[1]);
    int size = strtol(argv[2], NULL, 10);
    int iters = strtol(argv[3], NULL, 10);

    int Csize;
    cudaMemcpy(&Csize, &size, sizeof(int), cudaMemcpyHostToDevice);

    double Q11 = 10.;
    double Q12 = 20.;
    double Q21 = 20.;
    double Q22 = 30.;

    double step = 10. / size;

    double A[size*size];

    A[0] = Q11;
    A[size-1] = Q12;
    A[(size-1)*size] = Q21;
    A[(size-1)*size + size-1] = Q22;

    for (int i = 1; i < size - 1; i++){
        A[i] = A[i-1] + step;
        A[(size-1)*size + i] = A[(size-1)*size + i-1] + step;
    }

    for (int j = 1; j < size - 1; j++){
        A[j*size] = A[(j-1)*size] + step;
        A[j*size + size-1] = A[(j-1)*size + size-1] + step;
    }

    for (int i = 1; i < size - 1; i++)
        for (int j = 1; j < size - 1; j++)
            A[i*size + j] = 0;
    
    int iter = 0;
    double err = 1;

    // device matrices
    double* CAnew;
    double* CA;
    cudaMalloc(&CAnew, size*size*sizeof(double));
	cudaMalloc(&CA, size*size*sizeof(double));

    // defining Block and Grid sizes
    dim3 BS = dim3(16,16);
	dim3 GS = dim3(ceil(size/(float)BS.x),ceil(size/(float)BS.y));

    // grid-size device matrix
    double* CBlockErr;
    cudaMalloc(&CBlockErr, GS.x*GS.y*sizeof(double));
    // same for host
    double BlockErr[GS.x*GS.y];

    // copying matrices to device
    cudaMemcpy(CA, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(CAnew, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    while ((err > accuracy) && (iter < iters)){
        iter+=2;
        
        // compute + swap (count as 2 iterations)
        compute<<<GS,BS>>>(CAnew, CA, size);
        compute<<<GS,BS>>>(CA, CAnew, size);

        // getting error every 100 iterations
        if (iter % 100 == 0){
            err = 0;

            Max_Reduction<<<GS,BS>>>(CAnew, CA, size, CBlockErr);
            cudaDeviceSynchronize();
            // moving block results to CPU
            cudaMemcpy(BlockErr, CBlockErr, GS.x*GS.y*sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < GS.x; i++)
                for (int j = 0; j < GS.y; j++)
                    // counting final result
                    err = max(err, BlockErr[i*GS.x + j]);
            
            cout << iter << ' ' << err << endl;
        }
    }

    cout << iter << ' ' << err << endl;

	cudaFree(CA);
	cudaFree(CAnew);
    return 0;
}

// Computations result:

/*


+===============+======+=======+========+=======+
|       N       | 128  |  256  |  512   | 1024  |
+===============+======+=======+========+=======+
| GPU           | 1.3s | 4s    | 15.16s | 1m42s |
+---------------+------+-------+--------+-------+
| CPU           | 5.5s | 1m13s |        |       |
+---------------+------+-------+--------+-------+
| GPU optimized | 0.4s | 0.9s  | 3.9s   | 52s   |
+---------------+------+-------+--------+-------+
| CPU optimized | 4.4s | 1m    |        |       |
+---------------+------+-------+--------+-------+
| CPU Multicore | 2.3s | 14.8s | 2m21s  |       |
+---------------+------+-------+--------+-------+
| CuBLAS        | 0.9s | 2.4 s | 11.5s  | 1m37s |
+---------------+------+-------+--------+-------+
| CUDA          | 4.3s | 4.7 s | 9.0s   | 54.8s |
+---------------+------+-------+--------+-------+

*/