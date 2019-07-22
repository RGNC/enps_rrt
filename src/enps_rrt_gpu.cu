#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" { // nvcc compiles en C++
#include <enps_rrt.h>
#include <pgm.h>
}

#include <omp.h>
#include <cuda.h>
#include <cub/cub.cuh>

typedef unsigned int uint;

using namespace std;
using namespace cub;

struct _dev_pointers {
	float *da,*db;
	float *dx,*dy,*dd,*ddp,*dm;
	void *dcubtemp;
	size_t temp_bytes;
	int num_multiproc;
	cub::KeyValuePair<int,float> *d_argmin;
} devp;

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)
#define THRES_GPU 500

void init_params(const char* file, int n, float delta, int debug, int algorithm, RRT_PARAMS* params)
{
	PGM* map = load_pgm(file);
	params->map = load_pgm(file);

	remove_inner_obstacles(map);
	
	params->epsilon = ROBOT_RADIUS * ROBOT_RADIUS;
	params->delta =   delta;
	
	params->n = n;
	params->N = 1<<n; // Number of nodes in RRT
	
	params->debug = debug;
	
	params->p = map->width * RESOLUTION;
	params->q = map->height * RESOLUTION;
	
	params->algorithm = algorithm;
	
	float x=0,y=0;
	int c=0;
	params->a = (float*)malloc(map->width * map->height * sizeof(float));
	params->b = (float*)malloc(map->width * map->height * sizeof(float));
	
	for (int i=0;i<map->height;i++) {
		for (int j=0;j<map->width;j++) {
			x+=RESOLUTION;
			if (IS_OBSTACLE(map,i,j)){
				params->a[c] = x;
				params->b[c] = y;
				c++;
			}
		}
		y += RESOLUTION;
		x = 0;
	}
	params->m = 0;
	params->M = 1;
	
	while (params->M < c) {
		params->m++;
		params->M <<= 1;
	}
	if (debug) {
		printf("Map: %s\n",file);
		printf("Number of obstacles: %d\n",c);
		printf("Number of nodes: %d\n",params->M);
	}
	params->a = (float*)realloc(params->a, (params->M)*sizeof(float));
	params->b = (float*)realloc(params->b, (params->M)*sizeof(float));
	//free(params->a);
	//free(params->b);
	//cudaMallocManaged(&params->a,(params->M)*sizeof(float));
	//cudaMallocManaged(&params->b,(params->M)*sizeof(float));
	for (int i=c;i<params->M;i++) {
		params->a[i] = 3* params->p;
		params->b[i] = 3* params->q;
	}
	
	params->device = -1;
  	cudaGetDevice(&params->device);
	struct cudaDeviceProp devprop;
	cudaGetDeviceProperties(&devprop,params->device);
 	printf("Using GPU: %s\n",devprop.name);
	devp.num_multiproc = devprop.multiProcessorCount;

	cudaMalloc(&devp.da,(params->M)*sizeof(float));
	cudaMalloc(&devp.db,(params->M)*sizeof(float));
	//k_init_paramsab<<<params->M/256,256>>>(devp.da,devp.db,3,params->p,params->q,params->M);
	cudaMemcpy(devp.da,params->a,(params->M)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(devp.db,params->b,(params->M)*sizeof(float),cudaMemcpyHostToDevice);

	destroy_pgm(map);
}

// initializes a vector dv with values val, from ini to end. At position ini, the value valini is given.
__global__ void k_init_vec(float* dv, float val, float valini, uint ini, uint end) 
{
	for(uint i = threadIdx.x+blockDim.x*blockIdx.x; i<end; i+=gridDim.x*blockDim.x) {
		if (i == ini)
			dv[i] = valini;
		else if (i > ini)
			dv[i] = val;
	}
}

void init_vars(float x_init, float y_init, const RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x = (float*)malloc((params->N)*sizeof(float));
	vars->y = (float*)malloc((params->N)*sizeof(float));
	//cudaMallocManaged(&vars->x,(params->N)*sizeof(float));
	//cudaMallocManaged(&vars->y,(params->N)*sizeof(float));	
	
	vars->x[0] = x_init;
	vars->y[0] = y_init;
	
	for (int i=1; i< params->N; i++) {
		vars->x[i] = 3* params->p;
		vars->y[i] = 3* params->q;
	}

	cudaMalloc(&devp.dx,(params->N)*sizeof(float));
	cudaMalloc(&devp.dy,(params->N)*sizeof(float));
	//cudaMemcpy(devp.dx,vars->x,(params->N)*sizeof(float),cudaMemcpyHostToDevice);
	//cudaMemcpy(devp.dy,vars->y,(params->N)*sizeof(float),cudaMemcpyHostToDevice);
	k_init_vec<<<devp.num_multiproc*64,256>>>(devp.dx,3*params->p,x_init,0,params->N);
	k_init_vec<<<devp.num_multiproc*64,256>>>(devp.dy,3*params->q,y_init,0,params->N);
	
	vars->px = (float*)malloc((params->N)*sizeof(float));
	vars->py = (float*)malloc((params->N)*sizeof(float));
	//cudaMallocManaged(&vars->x,(params->N)*sizeof(float));
        //cudaMallocManaged(&vars->y,(params->N)*sizeof(float));
	
	vars->d = (float*)malloc((params->N)*sizeof(float));
	//cudaMallocManaged(&vars->d,(params->N)*sizeof(float));
	cudaMalloc(&devp.dd,(params->N)*sizeof(float));
			
	vars->dp = (float*)malloc((params->M)*sizeof(float));
	//cudaMallocManaged(&vars->dp,(params->M)*sizeof(float));
	cudaMalloc(&devp.ddp,(params->M)*sizeof(float));

	cudaMalloc(&devp.d_argmin,sizeof(KeyValuePair<int, float>));
	CubDebugExit(cub::DeviceReduce::ArgMin(NULL, devp.temp_bytes, devp.ddp, devp.d_argmin, MAX(params->M,params->N)));
	CubDebugExit(cudaMalloc(&devp.dcubtemp,devp.temp_bytes));
	CubDebugExit(cudaMalloc(&devp.dm,sizeof(float)));
	
	vars->x_rand = 0;
	vars->y_rand = 0;
	
	vars->x_new = 0;
	vars->y_new = 0;
	
	vars->x_nearest = 0;
	vars->y_nearest = 0;
	
	vars->collision = 0;
	
	vars->index = 1;
	vars->halt = 0;
	
	if (params->algorithm == RRT_STAR_ALGORITHM) {
		vars->dpp = (float*)malloc(params->M * params->N * sizeof(float));
		vars->c = (float*)malloc(params->N * sizeof(float));
		vars->cp = (float*)malloc(params->N * sizeof(float));
		vars->c[0] = 0;
	} 
}

void free_memory(RRT_PARAMS* params,RRT_VARS* vars)
{
	destroy_pgm(params->map);
	free(params->a);
	free(params->b);
	free(vars->x);
	free(vars->y);
	free(vars->px);
	free(vars->py);
	free(vars->d);
	free(vars->dp);

	//TODO: release GPU memory, see pointers devp
	cudaFree(devp.da);
	cudaFree(devp.db);
	cudaFree(devp.dx);
	cudaFree(devp.dy);
	cudaFree(devp.dd);
	cudaFree(devp.ddp);
	cudaFree(devp.dcubtemp);
	cudaFree(devp.dm);
	//cudaFree(params->a);
	//cudaFree(params->b);
	//cudaFree(vars->x);
	//cudaFree(vars->y);
	//cudaFree(vars->px);
	//cudaFree(vars->py);
	//cudaFree(vars->d);
	//cudaFree(vars->dp);
	
	if (params->algorithm==RRT_STAR_ALGORITHM) {
		free(vars->dpp);
		free(vars->c);
		free(vars->cp);
	}
}


float rnd()
{
	return (float)rand()/(float)(RAND_MAX);
}


// Squared distance from point (Cx,Cy) to segment [(Ax,Ay),(Bx,By)]
float p_dist(float Cx, float Cy, float Ax, float Ay, float Bx, float By)
{
	float u = (Cx-Ax)*(Bx-Ax) + (Cy-Ay)*(By-Ay);
	u /= (Bx-Ax)*(Bx-Ax) + (By-Ay)*(By-Ay);
	if (u<0) {
	 return (Ax-Cx)*(Ax-Cx) + (Ay-Cy)*(Ay-Cy);
	}
	if (u>1) {
	 return (Bx-Cx)*(Bx-Cx) + (By-Cy)*(By-Cy);
	}
	float Px = Ax + u*(Bx-Ax);
	float Py = Ay + u*(By-Ay);
	return (Px-Cx)*(Px-Cx) + (Py-Cy)*(Py-Cy);
}

__device__ inline float d_p_dist(float Cx, float Cy, float Ax, float Ay, float Bx, float By)
{
	float x = Bx-Ax, y = By-Ay;
        float u = (Cx-Ax)*x + (Cy-Ay)*y;
        u /= x*x + y*y;
	u = __saturatef (u); // 0.0 <= u <= 1.0
	float Px = Ax + u*x; // if u=0.0, Px= Ax, if u=1.0, Px = Bx, otherwise Px = Ax + u*(Bx-Ax)
	float Py = Ay + u*y;
	
	return (Px-Cx)*(Px-Cx) + (Py-Cy)*(Py-Cy);
}

__global__ void k_pdist(float* ddp, float* da, float*db, float x_nearest, float y_nearest, float x_new, float y_new, uint M) 
{
	for(uint i = threadIdx.x+blockDim.x*blockIdx.x; i<M; i+=gridDim.x*blockDim.x)
		ddp[i] = d_p_dist(da[i],db[i],x_nearest,y_nearest,x_new,y_new);
}


void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars)
{
	// compute distances from all obstacles to segment [(x_nearest,y_nearest),(x_new,y_new)]
	k_pdist<<<MIN(params->M/256,devp.num_multiproc*8),256>>>(devp.ddp, devp.da, devp.db, vars->x_nearest, vars->y_nearest, vars->x_new, vars->y_new, params->M);
	CubDebugExit(cudaDeviceSynchronize());
	// Compute minimun distance	
	CubDebugExit(cub::DeviceReduce::Min(devp.dcubtemp, devp.temp_bytes, devp.ddp, devp.dm, params->M));
	cudaDeviceSynchronize();
	float m;
	CubDebugExit(cudaMemcpy(&m,devp.dm,sizeof(float),cudaMemcpyDeviceToHost));
	// for some reason, the following line does not work
	//float m = thrust::reduce(thrust::device, devp.ddp, devp.ddp + params->M, INF, thrust::minimum<float>());
	/*float *m_pos = thrust::min_element(thrust::device, devp.ddp, devp.ddp + params->M);
	float m;
	cudaMemcpy(&m,m_pos,sizeof(float),cudaMemcpyDeviceToHost);	*/
	// collision if minimun distance is less than epsilon
	// variable collision has a value greater than 0 if collision
	vars->collision = params->epsilon - m;

/*
	// CPU version:
	//compute distances from all obstacles to segment [(x_nearest,y_nearest),(x_new,y_new)]
	#pragma omp parallel for
	for (int i=0;i<	params->M;i++) {
		vars->dp[i] = p_dist(params->a[i],params->b[i],vars->x_nearest,vars->y_nearest,vars->x_new,vars->y_new);
	}
	
	// Compute minimun distance
	float m = INF;
	#pragma omp parallel for reduction(min:m)
	for (int i=0;i<params->M;i++) {
		if (vars->dp[i]<m) {
			m = vars->dp[i];
		}
	}
	// collision if minimun distance is less than epsilon
	// variable collision has a value greater than 0 if collision
	vars->collision = params->epsilon - m;*/
}


XYD xyd_min2(XYD a, XYD b)
{
	return a.d < b.d ? a : b;
}


#pragma omp declare reduction(xyd_min : XYD : omp_out=xyd_min2(omp_out,omp_in))\
		initializer(omp_priv={0,0,INF})

__global__ void k_sqdist(float* dd, float* dx, float*dy, float x_rand, float y_rand, uint index)
{
        for(uint i = threadIdx.x+blockDim.x*blockIdx.x; i<index; i+=gridDim.x*blockDim.x)
		dd[i] = (dx[i] - x_rand) * (dx[i] - x_rand) + (dy[i] - y_rand) * (dy[i] - y_rand);
}

void nearest(RRT_PARAMS* params, RRT_VARS* vars)
{
	if (vars->index > THRES_GPU) {
		// compute squared distances from all points in RRT to (x_rand,y_rand)
		k_sqdist<<<vars->index/256,256>>>(devp.dd, devp.dx, devp.dy, vars->x_rand, vars->y_rand, vars->index);
		CubDebugExit(cudaDeviceSynchronize());
		
		cub::KeyValuePair<int, float> argmin;
		// compute minimun distance and nearest point	
		CubDebugExit(cub::DeviceReduce::ArgMin(devp.dcubtemp, devp.temp_bytes, devp.dd, devp.d_argmin, vars->index));
		cudaDeviceSynchronize();

		CubDebugExit(cudaMemcpy(&argmin,devp.d_argmin,sizeof(cub::KeyValuePair<int,float>),cudaMemcpyDeviceToHost));
		vars->d[0] = argmin.value;
		vars->x_nearest = vars->x[argmin.key];
		vars->y_nearest = vars->y[argmin.key];
		/*float *m_pos = thrust::min_element(thrust::device, devp.dd, devp.dd + vars->index);
		int pos = m_pos - devp.dd;//vars->d;
		vars->x_nearest = vars->x[pos];
		vars->y_nearest = vars->y[pos];
		cudaMemcpy(vars->d,m_pos,sizeof(float),cudaMemcpyDeviceToHost);	*/
	}
	else {
		// CPU version:
		// compute squared distances from all points in RRT to (x_rand,y_rand)
		#pragma omp parallel for
		for (int i=0;i<vars->index;i++) {
			vars->d[i] = (vars->x[i] - vars->x_rand) * (vars->x[i] - vars->x_rand) +
							(vars->y[i] - vars->y_rand) * (vars->y[i] - vars->y_rand);
		}

		// compute minimun distance and nearest point
		XYD value = {0,0,INF};
		#pragma omp parallel for reduction(xyd_min:value)
		for (int i=0;i<vars->index;i++) {
			XYD new_value = {vars->x[i],vars->y[i],vars->d[i]};
			value = xyd_min2(value,new_value);
		}
		vars->x_nearest = value.x;
		vars->y_nearest = value.y;
		vars->d[0] = value.d;
	}
}



void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x[vars->index] = vars->x_new;
	vars->y[vars->index] = vars->y_new;
	vars->px[vars->index] = vars->x_nearest;
	vars->py[vars->index] = vars->y_nearest;

	if (vars->index == THRES_GPU) {
		cudaMemcpy(devp.dx,vars->x,(vars->index+1)*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(devp.dy,vars->y,(vars->index+1)*sizeof(float),cudaMemcpyHostToDevice);
	}
	else if (vars->index > THRES_GPU) {
		cudaMemcpy(devp.dx+vars->index,vars->x+vars->index,sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(devp.dy+vars->index,vars->y+vars->index,sizeof(float),cudaMemcpyHostToDevice);
	}
}


void extend_rrt_star(RRT_PARAMS* params, RRT_VARS* vars)
{
	// compute squared distances from all obstacles to all segments [(x,y),(x_new,y_new)] where (x,y) are points in RRT
	#pragma omp parallel for collapse(2)
	for (int i=0;i<vars->index;i++) {
		for (int j=0;j<params->M;j++) {
			vars->dpp[i* params->M + j] = p_dist(params->a[j],params->b[j],vars->x[i],vars->y[i],vars->x_new,vars->y_new);
		}
	}
	
	// For each point (x,y) in RRT, compute the minimun distance 
	#pragma omp parallel for
	for (int i=0;i<vars->index;i++) {
		float m = INF;
		#pragma omp parallel for reduction(min:m)
		for (int j=0;j<params->M;j++) {
			if (vars->dpp[i*params->M+j] < m) {
				m = vars->dpp[i*params->M+j];
			}
		}
		// if the minimun distance is less than epsilon, set a flag to avoid this possible edge
		if (m< params->epsilon) {
			vars->dpp[i*params->M]=1;
		} else {
			vars->dpp[i*params->M]=0;
		}
	}
	
	// compute new cost for all points in RRT 
	#pragma omp parallel for
	for (int i=0;i<vars->index;i++) {
		vars->cp[i] =  vars->dpp[i*params->M]>0 ? INF : 
							vars->c[i] + 
								(vars->x_new - vars->x[i])*(vars->x_new - vars->x[i]) +
								(vars->y_new - vars->y[i])*(vars->y_new - vars->y[i]);
		
		
		//vars->cp[i] = vars->c[i] + 
		//				(vars->x_new - vars->x[i])*(vars->x_new - vars->x[i]) +
		//				(vars->y_new - vars->y[i])*(vars->y_new - vars->y[i]) +
		//				INF * vars->dpp[i*params->M]; 
	}
	
	// compute minimun cost	
	XYD value = {0,0,INF};
	#pragma omp parallel for reduction(xyd_min:value)
	for (int i=0;i<vars->index;i++) {
		XYD new_value = {vars->x[i],vars->y[i],vars->cp[i]};
		value = xyd_min2(value,new_value);
	}
	
	vars->x[vars->index] = vars->x_new;
	vars->y[vars->index] = vars->y_new;
	vars->px[vars->index] = value.x;
	vars->py[vars->index] = value.y;
	vars->c[vars->index] = value.d;

	// Fix edges
	#pragma omp parallel for
	for (int i=0;i<vars->index;i++) {
		float aux = vars->c[vars->index] + 
						(vars->x_new - vars->x[i])*(vars->x_new - vars->x[i]) +
						(vars->y_new - vars->y[i])*(vars->y_new - vars->y[i]);
						
		if (vars->c[i] > aux) {
			vars->px[i] = vars->x_new;
			vars->py[i] = vars->y_new;
			vars->c[i] = aux;
		}
	}
}

void enps_rrt_one_iteration(RRT_PARAMS* params, RRT_VARS* vars)
{
	// Exit if halting condition has been reached
	if (vars->halt) {		
		return;
	}
	
	// set collision = 0
	vars->collision = 0;
	
	// compute (x_rand, y_rand)
	vars->x_rand = params->p * rnd();
	vars->y_rand = params->q * rnd();
	
	// compute (x_nearest, y_nearest)
	nearest(params, vars);
	
	// compute (x_new, y_new)
	vars->x_new = vars->x_nearest + params->delta * (vars->x_rand - vars->x_nearest) / sqrt(vars->d[0]);
	vars->y_new = vars->y_nearest + params->delta * (vars->y_rand - vars->y_nearest) / sqrt(vars->d[0]);
	
	// compute obstacle collision
	obstacle_free(params, vars);
	
	// Exit if collision from (x_nearest, y_nearest) to (x_new, y_new)
	if (vars->collision > 0) {
		return;
	}	
	
	// Extend RRT tree
	switch(params->algorithm)
	{
		case RRT_ALGORITHM:
			extend_rrt(params,vars);
		break;
		case RRT_STAR_ALGORITHM:
			extend_rrt_star(params,vars);
		break;
		default:
		;
	}
	
	// Increment RRT node index
	vars->index++;
	
	// If node index is 2^n then halt
	if (vars->index == params->N) {
		vars->halt = 1;
	}
}


