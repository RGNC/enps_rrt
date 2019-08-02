#include <cuda.h>
#include <stdio.h>
#include "../include/cub/cub.cuh"

#define N 5 // rows
#define M 10 // columns



int main()
{
	float * val, * seg, * res;
	float * d_val, * d_res, * d_seg, * d_temp=NULL;
	int num_segments = N;
	size_t temp_bytes;

	val = new float[N*M];
	res = new float[N];
	seg = new float[N+1];

	cudaMalloc(&d_val,N*M*sizeof(float));
	cudaMalloc(&d_res,N*sizeof(float));
	cudaMalloc(&d_seg,(N+1)*sizeof(float));

	for (int i =0; i< N; i++)
		for (int j=0; j<M; j++)
			val[i*M+j]=-(i+j);	
	cudaMemcpy(d_val,val,N*M*sizeof(float),cudaMemcpyHostToDevice);
	for (int i =0; i< N; i++) { for (int j =0; j< M; j++) printf("v[%i,%i]=%f; ",i,j,val[i*M+j]); printf("\n");}

	seg[0]=0.f;
	for (int i =1; i<= N; i++)
		seg[i]=seg[i-1]+M;
	cudaMemcpy(d_seg,seg,(N+1)*sizeof(float),cudaMemcpyHostToDevice);
	for (int i =0; i<=N; i++) printf("seg[%i]=%f; ",i,seg[i]);
	printf("\n");

	cub::DeviceSegmentedReduce::Min(d_temp, temp_bytes, d_val, d_res,
		num_segments, d_seg, d_seg + 1);
	printf("Reducing on GPU using %d bytes\n",temp_bytes);
	// Allocate temporary storage
	cudaMalloc(&d_temp, temp_bytes);
	// Run min-reduction
	cub::DeviceSegmentedReduce::Min(d_temp, temp_bytes, d_val, d_res,
		num_segments, d_seg, d_seg + 1);
	cudaDeviceSynchronize();

	cudaMemcpy(res,d_res,N*sizeof(float),cudaMemcpyDeviceToHost);
	for (int i =0; i<N; i++) printf("res[%i]=%f; ",i,res[i]);
	printf("\n");
	cudaFree(d_val);
	cudaFree(d_res);
	cudaFree(d_seg);
	cudaFree(d_temp);
	free(val);
	free(seg);
	free(res);
}
