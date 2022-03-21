#include <iostream>
#include <stdlib.h>
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"
#include </opt/nvidia/hpc_sdk/Linux_x86_64/21.11/cuda/11.0/targets/x86_64-linux/include/cuda_runtime.h>

using namespace std;

int main(int argc, char *argv[]){
    double accuracy = atof(argv[1]); //argv[1] - accuracy
    int size = strtol(argv[2], NULL, 10) + 2; //argv[2] - size
    int iters = strtol(argv[3], NULL, 10); //argv[3] - iters

    // initializing matrix
    double Q11 = 10.;
    double Q12 = 20.;
    double Q21 = 20.;
    double Q22 = 30.;

    double step = 10. / size;

    double A[size][size];

    A[0][0] = Q11;
    A[0][size-1] = Q12;
    A[size-1][0] = Q21;
    A[size-1][size-1] = Q22;

    #pragma acc parallel // using pragma parallel while fulfilling initial matrix
    {
    for (int i = 1; i < size-1; i++){
        A[0][i] = A[0][i-1] + step;
        A[size - 1][i] = A[size - 1][i-1] + step;
    }

    for (int j = 1; j < size - 1; j++){
        A[j][0] = A[j-1][0] + step;
        A[j][size - 1] = A[j-1][size - 1] + step;
    }

    for (int i = 1; i < size - 1; i++)
        for (int j = 1; j < size - 1; j++)
            A[i][j] = 0;
    }
    

    double Anew[size][size];
    int iter = 0;
    double err = 1;

    //Initializing cublas context
    cublasHandle_t handle;
    cublasCreate(&handle);

    double temp[size*size];

    #pragma acc data copy(A) create(Anew, err, temp) // here we copy array A to GPU and create Anew, err on GPU
    {
    #pragma acc kernels
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            Anew[i][j] = A[i][j];
    while ((err > accuracy) && (iter < iters)){
        iter++;
        
        if ((iter % 100 == 0) || (iter == 1)){ // every 100 iterations we nullify error and compute it
            #pragma acc kernels async(1) // asynchronous computations on a new thread
            {
            for (int j = 1; j < size - 1; j++)
                for (int i = 1; i < size - 1; i++)
                    Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                    
            }
            #pragma acc wait(1)
            int idx;
            //making addresses of device data avaiable on the host
            #pragma acc host_data use_device(Anew, A, temp)
            {
            // here we will be implementing the following expression:
            //  err = max(err, Anew[i][j] - A[i][j]);
            double alpha = -1.;
            for (int i = 0; i < size; i++){
                // copying Anew matrix to a cublas-like array temp
                cublasDcopy(handle, size, Anew[i], 1, &temp[i*size], 1);
                // multiplying A matrix with -1 and adding it to temp
                cublasDaxpy(handle, size, &alpha, A[i], 1, &temp[i*size], 1);
            }
            
            // searching for a max error
            cublasIdamax(handle, size*size, temp, 1, &idx);
            }
            
            // updating temp on CPU
            #pragma acc update self(temp[idx-1:1])
            
            // getting the result
            err = fabs(temp[idx-1]);

        } else{
            #pragma acc kernels async(1)
            #pragma acc loop independent collapse(2)
            for (int j = 1; j < size - 1; j++)
                for (int i = 1; i < size - 1; i++){
                    Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                    
                }

        }
        
        #pragma acc kernels async(1) // updating matrix
        for (int i = 1; i < size - 1; i++)
            for (int j = 1; j < size - 1; j++)
                A[i][j] = Anew[i][j];

        if ((iter % 100 == 0) || (iter == 1)){ // every 100 iterations:
            cout << iter << ' ' << err << endl;
        }
    }
    }

    cout << iter << ' ' << err << endl;

    // destroying CuBLAS context
    cublasDestroy(handle);
    cout << "Made with CuBLAS";

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

*/
