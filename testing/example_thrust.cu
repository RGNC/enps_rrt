#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <stdio.h>

#define N 5



struct d_squared_dist : public thrust::binary_function<float,float,float>
{
	float xr, yr;
	__host__ __device__ __forceinline__ float operator()(float x, float y) { return ((x - xr) * (x - xr)) + ((y - yr) * (y - yr)); }
};

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

	struct d_squared_dist dsqdist_op;
    dsqdist_op.xr=0.1;
	dsqdist_op.yr=0.1;
	//thrust::transform(thrust::device, vars->x, vars->x + vars->index, vars->y, vars->d, dsqdist_op);
	thrust::transform(thrust::device, d_val, d_val + N, d_val2, d_val3, dsqdist_op);	

	float *m_pos = thrust::min_element(thrust::device, d_val, d_val + N);

	float res;
	cudaMemcpy(&res,m_pos,sizeof(float),cudaMemcpyDeviceToHost);
	pos = m_pos - d_val;

	printf("\n\n Minval[%d]=%f\n",pos,res);

	cudaMemcpy(val,d_val3,N*sizeof(float),cudaMemcpyDeviceToHost);
	for (int i =0; i< N; i++) printf(" val[%i]=%f\n",i,val[i]);

	float m = thrust::reduce(thrust::device, d_val3, d_val3 + N, 0.0/*999999.9*/, thrust::maximum<float>());
	printf("\n Maxval3=%f\n",m);

	cudaFree(d_val);
	cudaFree(d_val2);
	cudaFree(d_val3);
	free(val);



}
