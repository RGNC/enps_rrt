/*
 * Code to test different parallel reduction kernels. CPU vs GPU vs CUB
 * Miguel A. Martínez-del-Amor
 * Research Group on Natural Computing
 * Universidad de Sevilla (Spain)
 *
 * GPU kernels taken from reduction NVIDIA SDK examples
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// compile with: nvcc -O3 -Xcompiler -fopenmp reduction_kernel.cu -o reduce

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
#include <cooperative_groups.h>

#include <float.h>
#include <omp.h>
#include "../include/cub/cub.cuh"
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
struct SharedMemory
{
    __device__ inline operator       float *()
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }

    __device__ inline operator const float *() const
    {
        extern __shared__ int __smem[];
        return (float *)__smem;
    }
};

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

/*
    This version minimizes multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize, bool nIsPow2>
__global__ void
k_reduce_min(float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    float *sdata = SharedMemory();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    float myMin = FLT_MAX;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMin =  fminf(myMin,g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMin =  fminf(myMin,g_idata[i+blockSize]);

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = myMin;
    cg::sync(cta);


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myMin = fminf(myMin,sdata[tid + 256]);
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = myMin = fminf(myMin,sdata[tid + 128]);
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = myMin = fminf(myMin,sdata[tid + 64]);
    }

    cg::sync(cta);

    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    if (cta.thread_rank() < 32)
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) myMin = fminf(myMin,sdata[tid + 32]);
        // Reduce final warp using shuffle
        for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
        {
             myMin = fminf(myMin,tile32.shfl_down(myMin, offset));
        }
    }

    // write result for this block to global mem
    if (cta.thread_rank() == 0) g_odata[blockIdx.x] = myMin;
}

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
void
callReduceMin(int size, int threads, int blocks, float *d_idata, float *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (threads)
        {
            case 512:
                k_reduce_min<12, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                k_reduce_min<56, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                k_reduce_min<28, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                k_reduce_min<64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                k_reduce_min<32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                k_reduce_min<16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                k_reduce_min< 8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                k_reduce_min< 4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                k_reduce_min< 2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                k_reduce_min< 1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
    else
    {
        switch (threads)
        {
            case 512:
                k_reduce_min<12, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                k_reduce_min<56, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                k_reduce_min<28, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                k_reduce_min<64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                k_reduce_min<32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                k_reduce_min<16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                k_reduce_min< 8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                k_reduce_min< 4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                k_reduce_min< 2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                k_reduce_min< 1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel
// 6, we observe the maximum specified number of blocks, because each thread in
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size exceed the upper bound
    cudaDeviceProp prop;
    int device;
    (cudaGetDevice(&device));
    (cudaGetDeviceProperties(&prop, device));

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    
    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = MIN(maxBlocks, blocks);
}

float reduceMin(int  n, float *d_idata, float * d_odata)
{
    int maxBlocks = 64;
    int maxThreads = 256;
    int numThreads = 0;
    int numBlocks = 0;
    float gpu_result = 0;
    //float *d_odata = NULL;

    getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

    //(cudaMalloc((void **) &d_odata, numBlocks*sizeof(float)));

    // execute the kernel
    callReduceMin(n, numThreads, numBlocks, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=numBlocks;

    while (s >= 1)
    {
        cudaDeviceSynchronize();
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
        cudaMemcpy(d_idata, d_odata, s*sizeof(float), cudaMemcpyDeviceToDevice);
        callReduceMin(s, threads, blocks, d_idata, d_odata);

        s = (s + (threads*2-1)) / (threads*2);
        if (s == 1) break;
    }

    cudaDeviceSynchronize();
    // copy final sum from device to host
    (cudaMemcpy(&gpu_result, d_odata, sizeof(float), cudaMemcpyDeviceToHost));
  
    return gpu_result;
}

int main(int argc, char* argv[])
{
    printf("Welcome to the testing program of reduction implementations\n");
    int selection = 1;
    int pow = 15;
    int iter = 1;
    if (argc == 1) {
        printf("Usage: ./reduce I P T\n\tI = implementation: 1 (openMP), 2 (GPU kernels from SDK), 3 (GPU CUB library), 5 (GPU Thrust library)\n\tP = pow of 2 leading to input size (1 <= P <= 32)\n\tT = number of times to repeat the reduction.");
    }
    if (argc>1) {
		selection = atoi(argv[1]);	
    }
    if (argc>2) {
        pow = atoi(argv[2]);
    }
    if (argc>3) {
        iter = atoi(argv[3]);
    }

    const int N = 2<<pow;
    float * test = new float[N];
    for (int i=0; i< N; i++)
        test[i] = i*(-1.f);

    float * dtest, *daux, *dres;
    float m = FLT_MAX;

    // CPU
    if (selection == 1) {
        for (int j=0; j< iter; j++) {
            #pragma omp parallel for reduction(min:m)
            for (int i=0;i<N;i++) {
                if (test[i]<m) {
                    m = test[i];
                }
            }
        }
    } // GPU
    else {
        cudaMalloc(&dtest,N*sizeof(float));
        cudaMemcpy(dtest,test,N*sizeof(float),cudaMemcpyHostToDevice);

        //// GPU Kernel
        if (selection == 2) {
            cudaMalloc(&daux,N*sizeof(float));
            for (int j=0; j< iter; j++)
                m = reduceMin(N,dtest,daux);
            cudaFree(daux);
        }
        //// GPU CUB
        else if (selection == 3) {
            cudaMalloc(&dres,sizeof(float));
            size_t temp_bytes;
            CubDebugExit(cub::DeviceReduce::Min(NULL, temp_bytes, dtest, dres, N));
            CubDebugExit(cudaMalloc(&daux,temp_bytes));
            for (int j=0; j< iter; j++) {
                cub::DeviceReduce::Min(daux, temp_bytes, dtest, dres, N);
                CubDebugExit(cudaDeviceSynchronize());
                CubDebugExit(cudaMemcpy(&m,dres,sizeof(float),cudaMemcpyDeviceToHost));
            }
            cudaFree(dres);
            cudaFree(daux);
        }
	//// GPU Thrust
	else if (selection == 4) {
	    for (int j=0; j< iter; j++)
	    	m = thrust::reduce(thrust::device, dtest, dtest + N, FLT_MAX, thrust::minimum<float>());
	}
        else {
            printf("No mode\n");
        }
        cudaFree(dtest);
    }

    printf("Obtained %f\n",m);   

    delete [] test;
}

#endif // #ifndef _REDUCE_KERNEL_H_
