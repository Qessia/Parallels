#include <iostream>
#include <stdlib.h>

using namespace std;

void print(double**& A, int size){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++)
            cout << A[i][j] << ' ';
        cout << endl;
    }
}

int main(int argc, char *argv[]){
    double accuracy = atof(argv[1]); //argv[1] - accuracy
    int size = strtol(argv[2], NULL, 10); //argv[2] - size
    int iters = strtol(argv[3], NULL, 10); //argv[3] - iters

    // initializing matrix
    double Q11 = 10.;
    double Q12 = 20.;
    double Q21 = 20.;
    double Q22 = 30.;

    double step = 10. / (size + 2);

    double A[size+2][size+2];

    A[0][0] = Q11;
    A[0][size+1] = Q12;
    A[size+1][0] = Q21;
    A[size+1][size+1] = Q22;

    
#pragma acc parallel
{
    for (int i = 1; i < size + 1; i++){
        A[0][i] = A[0][i-1] + step;
        A[size + 1][i] = A[size + 1][i-1] + step;
    }

    for (int j = 1; j < size + 1; j++){
        A[j][0] = A[j-1][0] + step;
        A[j][size + 1] = A[j-1][size + 1] + step;
    }
}

    for (int i = 1; i < size + 1; i++)
        for (int j = 1; j < size + 1; j++)
            A[i][j] = 0;

    double Anew[size+2][size+2];

#pragma acc loop independent collapse(2)
    for (int i = 0; i < size + 2; i++)
        for (int j = 0; j < size + 2; j++)
            Anew[i][j] = A[i][j];

    // end
    
    
    int iter = 0;
    double err = 1;

    #pragma acc data copy(Anew,A) create(err) 
    {
    while ((err > accuracy) && (iter < iters)){
        iter++;
        #pragma acc kernels
        {
        err = 0;
        #pragma acc loop independent collapse(2)
        for (int j = 1; j < size + 1; j++)
            for (int i = 1; i < size + 1; i++)
                Anew[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1]);

        #pragma acc loop independent collapse(2) reduction(max:err)
        for (int j = 1; j < size + 1; j++)
            for (int i = 1; i < size + 1; i++){
                A[i][j] = 0.25 * (Anew[i+1][j] + Anew[i-1][j] + Anew[i][j-1] + Anew[i][j+1]);
                err = max(err, Anew[i][j] - A[i][j]);
            }
        
        }

        if ((iter % 100 == 0) || (iter == 1))
            #pragma acc update self(err)
            cout << iter << ' ' << err << endl;
    }
    }

    cout << iter << ' ' << err << endl;

    return 0;
}
