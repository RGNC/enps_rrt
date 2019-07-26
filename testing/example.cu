#include <cuda.h>
#include <stdio.h>
#include <include/cub/cub.cuh>

#define N 5



int main()
{
	float * val;
	float * d_val;
	float * d_val2;
	float * d_val3;
	int pos;

	val = new float[N];
	cudaMalloc(&d_val,N*sizeof(float));
	cudaMalloc(&d_val2,N*sizeof(float));
	cudaMalloc(&d_val3,N*sizeof(float));

	for (int i =0; i< N; i++)
		val[i]=0.1 * (i+12);	
	val[N-1] = 0.004;
	cudaMemcpy(d_val,val,N*sizeof(float),cudaMemcpyHostToDevice);
	for (int i =0; i< N; i++) printf("val[%i]=%f\n",i,val[i]);

	for (int i =0; i< N; i++)
		val[i]=0.1 * (i+2);
	cudaMemcpy(d_val2,val,N*sizeof(float),cudaMemcpyHostToDevice);
	for (int i =0; i< N; i++) printf("val[%i]=%f\n",i,val[i]);
	size_t bytes;
	cub::DeviceReduce::Min(d_val3, bytes, d_val, d_val2, N);
 
	float res;
	cudaMemcpy(&res,d_val2,sizeof(float),cudaMemcpyDeviceToHost);

	printf("\n\n Minval=%f\n",res);

	cudaFree(d_val);
	cudaFree(d_val2);
	cudaFree(d_val3);
	free(val);



}
