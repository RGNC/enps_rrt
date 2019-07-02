#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" { // nvcc compiles en C++
#include <enps_rrt.h>
#include <pgm.h>
}

#include <omp.h>
#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

using namespace std;

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
/*  	cudaMemPrefetchAsync(params->a, params->M*sizeof(float), params->device, NULL);
  	cudaMemPrefetchAsync(params->b, params->M*sizeof(float), params->device, NULL);
*/
	destroy_pgm(map);
}


void init_vars(float x_init, float y_init, const RRT_PARAMS* params, RRT_VARS* vars)
{
	//vars->x = (float*)malloc((params->N)*sizeof(float));
	//vars->y = (float*)malloc((params->N)*sizeof(float));
	cudaMallocManaged(&vars->x,(params->N)*sizeof(float));
	cudaMallocManaged(&vars->y,(params->N)*sizeof(float));
	
	vars->x[0] = x_init;
	vars->y[0] = y_init;
	
	for (int i=1; i< params->N; i++) {
		vars->x[i] = 3* params->p;
		vars->y[i] = 3* params->q;
	}
	
	vars->px = (float*)malloc((params->N)*sizeof(float));
	vars->py = (float*)malloc((params->N)*sizeof(float));
	//cudaMallocManaged(&vars->x,(params->N)*sizeof(float));
    //cudaMallocManaged(&vars->y,(params->N)*sizeof(float));
	
	//vars->d = (float*)malloc((params->N)*sizeof(float));
	cudaMallocManaged(&vars->d,(params->N)*sizeof(float));
			
	vars->dp = (float*)malloc((params->M)*sizeof(float));
	//cudaMallocManaged(&vars->dp,(params->M)*sizeof(float));

	// consider using cudaMemPrefetchAsync() for better performance
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
	//free(vars->x);
	//free(vars->y);
	free(vars->px);
	free(vars->py);
	//free(vars->d);
	free(vars->dp);

	//cudaFree(params->a);
	//cudaFree(params->b);
	cudaFree(vars->x);
	cudaFree(vars->y);
	//cudaFree(vars->px);
	//cudaFree(vars->py);
	cudaFree(vars->d);
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
__host__ __device__ float p_dist(float Cx, float Cy, float Ax, float Ay, float Bx, float By)
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

struct d_p_dist : public thrust::binary_function<float,float,float>
{

	float ax,ay,bx,by;
	__host__ __device__ float operator()(float cx, float cy) { return p_dist(cx,cy,ax,ay,bx,by); }
};

void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars)
{
	//k_obstacle_free<<<256,params->M/256+1>>>(vars->dp,params->a,params->b,vars->x_nearest,vars->y_nearest,vars->x_new,vars->y_new);
/*
	cudaMemPrefetchAsync(params->a, params->M*sizeof(float), params->device, NULL);
  	cudaMemPrefetchAsync(params->b, params->M*sizeof(float), params->device, NULL);
  	cudaDeviceSynchronize();

	// compute distances from all obstacles to segment [(x_nearest,y_nearest),(x_new,y_new)]
	struct d_p_dist dpdist_op;
	dpdist_op.ax=vars->x_nearest;
	dpdist_op.ay=vars->y_nearest;
	dpdist_op.bx=vars->x_new;
	dpdist_op.by=vars->y_new;
	thrust::transform(thrust::device, params->a, params->a + params->M, params->b, vars->dp, dpdist_op); 

	float * deb = new float[params->M];	
	for (int i=0;i<	params->M;i++) {
		deb[i] = p_dist(params->a[i],params->b[i],vars->x_nearest,vars->y_nearest,vars->x_new,vars->y_new);
	}


	/*cudaDeviceSynchronize();
	cudaMemPrefetchAsync(vars->dp,params->M*sizeof(float),0);
	cudaDeviceSynchronize();
	for (int i =0; i< params->M; i++) {
		if (deb[i] != vars->dp[i]) {
			printf("GPU[%d]=%f != CPU[%d]=%f \n",i,vars->dp[i],i,deb[i]);
		}
	}
	float m,m1 = INF;	
	for (int i=0;i<params->M;i++) {
		if (deb[i]<m1) {
			m1 = deb[i];
		}
	}
	/*if (m!=m1){
		printf("mGPU=%f != mCPU=%f \n",m,m1);		
	}
	delete [] deb;

	// Compute minimun distance
	//float *m_pos = thrust::min_elem(thrust::device, vars->dp, vars->dp + params->M);
	//float m = *m_pos;
	//float m = thrust::reduce(thrust::device, vars->dp, vars->dp + params->M, INF, thrust::minimum<float>());
	m=m1;

	// collision if minimun distance is less than epsilon
	// variable collision has a value greater than 0 if collision
	vars->collision = params->epsilon - m;
*/

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
	vars->collision = params->epsilon - m;
}


XYD xyd_min2(XYD a, XYD b)
{
	return a.d < b.d ? a : b;
}


#pragma omp declare reduction(xyd_min : XYD : omp_out=xyd_min2(omp_out,omp_in))\
		initializer(omp_priv={0,0,INF})


struct d_squared_dist : public thrust::binary_function<float,float,float>
{
	float xr,yr;
	__host__ __device__ float operator()(float x, float y) { return (x - xr) * (x - xr) + (y - yr) * (y - yr); }
};

void nearest(RRT_PARAMS* params, RRT_VARS* vars)
{
	// compute squared distances from all points in RRT to (x_rand,y_rand)
	struct d_squared_dist dsqdist_op;
    dsqdist_op.xr=vars->x_rand;
	dsqdist_op.yr=vars->y_rand;
	thrust::transform(thrust::device, vars->x, vars->x + vars->index, vars->y, vars->d, dsqdist_op);

	float *m_pos = thrust::min_element(thrust::device, vars->d, vars->d + vars->index);
	int pos = m_pos - vars->d;
	vars->x_nearest = vars->x[pos];
	vars->y_nearest = vars->y[pos];
	vars->d[0] = *m_pos;	

	// compute squared distances from all points in RRT to (x_rand,y_rand)
	/*#pragma omp parallel for
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
	vars->d[0] = value.d;*/
}



void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x[vars->index] = vars->x_new;
	vars->y[vars->index] = vars->y_new;
	vars->px[vars->index] = vars->x_nearest;
	vars->py[vars->index] = vars->y_nearest;
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
	cout << "here" << endl;
	
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

