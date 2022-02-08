#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(){
        int n = 10000000;
        double sum = 0;
        double* a = (double*)malloc(sizeof(double)*n);
#pragma acc kernels
{
        for (int i = 0; i < n; i++)
                a[i] = sin(i * 2 * M_PI / (double)n);

        for (int i = 0; i < n; i++)
                sum += a[i];
}
printf("%lf\n", sum);
        return 0;

}


/*

    1  nvidia-sml
    2  nvidia-smi
    3  nano
    4  nao
    5  nano
    6  joe
    7  vim
    8  ls
    9  gcc parallel1.c
   10  gcc -lm parallel1.c
   11  gcc parallel1.c -lm
   12  ls
   13  ./a.out
   14  vim parallel.c
   15  ls
   16  vim parallel1.c
   17  gcc parallel1.c -lm
   18  ./a.out
   19  vim parallel1.c
   20  pgcc -acc -Minfo=accel parallel1.c
   21  who
   22  wall "Nu kak tam s parallelizmom"
   23  pgcc -acc -Minfo=accel parallel1.c
   24  vim parallel1.c
   25  ./a.out
   26  vim parallel1.c
   27  pgcc -acc -Minfo=accel parallel1.c
   28  vim parallel1.c
   29  pgcc -acc -Minfo=accel parallel1.c
   30  ./a.out
   31  pgcc parallel1.c
   32  ./a.out
   33  gcc parallel1.c -lm
   34  ./a.out
   35  vim parallel1.c
   36  PGI_ACC_TIME=1 ./a.out
   37  PGI_ACC_TIME=1./a.out
   38  pgcc -acc -Minfo=accel parallel1.c
   39  PGI_ACC_TIME=1 ./a.out
   40  vim parallel1.c
   41  pgcc -acc -Minfo=accel parallel1.c
   42  vim parallel1.c
   43  history
   44  vim parallel1.c
   45  pgcc -acc -Minfo=accel parallel1.c
   46  PGI_ACC_TIME=1 ./a.out
   47  vim parallel1.c
   48  history
   
   time ./a.out

*/
