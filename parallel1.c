
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
