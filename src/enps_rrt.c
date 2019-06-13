#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <enps_rrt.h>


void init_params(const char* file, int n, int debug, int algorithm, RRT_PARAMS* params)
{
	PGM* map = load_pgm(file);
	params->map = load_pgm(file);

	remove_inner_obstacles(map);

	
	params->epsilon = ROBOT_RADIUS * ROBOT_RADIUS;
	params->delta =   ROBOT_RADIUS;
	
	params->n = n;
	params->N = 1<<n;
	
	params->debug = debug;
	
	params->p = map->width * RESOLUTION;
	params->q = map->height * RESOLUTION;
	
	params->algorithm = algorithm;
	
	params->max_squared_distance = 9*params->p*params->p + 9*params->q*params->q;
	
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
	
	params->a = (float*)realloc(params->a, (params->M)*sizeof(float));
	params->b = (float*)realloc(params->b, (params->M)*sizeof(float));
	
	for (int i=c;i<params->M;i++) {
		params->a[i] = 3* params->p;
		params->b[i] = 3* params->q;
	}
	
	
	
	destroy_pgm(map);
}


void init_vars(float x_init, float y_init, const RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x = (float*)malloc((params->N)*sizeof(float));
	vars->y = (float*)malloc((params->N)*sizeof(float));
	
	vars->x[0] = x_init;
	vars->y[0] = y_init;
	
	for (int i=1; i< params->N; i++) {
		vars->x[i] = 3* params->p;
		vars->y[i] = 3* params->q;
	}
	
	vars->px = (float*)malloc((params->N)*sizeof(float));
	vars->py = (float*)malloc((params->N)*sizeof(float));
	
	vars->d = (float*)malloc((params->N)*sizeof(float));
	
		
	vars->dp = (float*)malloc((params->M)*sizeof(float));
	
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
	} else {
		vars->dpp = NULL;
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
	if (params->algorithm==RRT_STAR_ALGORITHM) {
		free(vars->dpp);
	}
}



float rnd()
{
	return (float)rand()/(float)(RAND_MAX);
}



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


// TODO
void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars)
{
	
	// PARALLEL FOR
	for (int i=0;i<	params->M;i++) {
		vars->dp[i] = p_dist(params->a[i],params->b[i],vars->x_nearest,vars->y_nearest,vars->x_new,vars->y_new);
	}
	
	// REDUCTION OVER MIN:dp[i]
	float min = vars->dp[0];
	for (int i=1;i<params->M;i++) {
		if (vars->dp[i]<min) {
			min = vars->dp[i];
		}
	}
	
	vars->collision = params->epsilon - min;
}


void nearest(RRT_PARAMS* params, RRT_VARS* vars)
{
	// compute distances
	// compute nearest
	
	// PARALLEL FOR
	for (int i=0;i<vars->index;i++) {
		vars->d[i] = (vars->x[i] - vars->x_rand) * (vars->x[i] - vars->x_rand) +
						(vars->y[i] - vars->y_rand) * (vars->y[i] - vars->y_rand);
	}
	
	float min = vars->d[0];
	vars->x_nearest = vars->x[0];
	vars->y_nearest = vars->y[0];
	
	
	// REDUCTION OVER MIN:d[i]
	for (int i=1;i<vars->index;i++) {
		if (vars->d[i] < min) {
			min = vars->d[i];
			vars->x_nearest = vars->x[i];
			vars->y_nearest = vars->y[i];
		}
	}
	
	vars->d[0] = min;

}



void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x[vars->index] = vars->x_new;
	vars->y[vars->index] = vars->y_new;
	vars->px[vars->index] = vars->x_nearest;
	vars->py[vars->index] = vars->y_nearest;

	if (params->debug)
	{
		int x0 = (int)round(vars->x_nearest / RESOLUTION);
		int y0 = (int)round(vars->y_nearest / RESOLUTION);
		int x1 = (int)round(vars->x_new / RESOLUTION);
		int y1 = (int)round(vars->y_new / RESOLUTION);
		draw_line(params->map,x0,y0,x1,y1,0);
		
	}
}

// TODO
void extend_rrt_star(RRT_PARAMS* params, RRT_VARS* vars)
{
	for (int i=0;i<params->M;i++) {
		for (int j=0;j<vars->index;j++) {
			vars->dpp[i* params->N + j] = p_dist(params->a[i],params->b[i],vars->x[j],vars->y[j],vars->x_new,vars->y_new);
			
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

