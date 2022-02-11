#include <iostream>
#include <stdlib.h>

using namespace std;

int main(int argc, char *argv[]){
    double accuracy = atof(argv[1]); //argv[1] - accuracy
    int size = strtol(argv[2], NULL, 10); //argv[2] - size
    int iters = strtol(argv[3], NULL, 10); //argv[3] - iters

    // initializing matrix
    double Q11 = 10;
    double Q12 = 20;
    double Q21 = 20;
    double Q22 = 30;

    double A[size][size];
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++){
            double R1 = Q11 * (size - i) / size + Q21 * i / size;
            double R2 = Q12 * (size - i) / size + Q22 * i / size;
            A[i][j] = R1 * (size - j) / size + R2 * j / size;
        }

    //!$acc data copy(A) create(Anew)
    int iter = 0;
    double err = 1;
    double Anew[size][size];

    while ((err > accuracy) && (iter < iters)){
        iter++;
        err = 0;

        //!$acc kernels
        for (int j = 1; j < size; j++)
            for (int i = 1; i < size; i++){
                Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);
                err = max(err, Anew[i][j] - A[i][j]);
            }
        
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                A[i][j] = Anew[i][j];
        //!$acc end kernels

        if ((iter % 100 == 0) || (iter == 1))
            cout << iter << ' ' << err << endl;
    }
    //!$acc end data
    return 0;
}