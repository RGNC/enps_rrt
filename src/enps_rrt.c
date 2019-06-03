#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <enps_rrt.h>

float rnd()
{
	return (float)rand()/(float)(RAND_MAX);
}

// TODO
void init_obstacles(RRT_PARAMS* params)
{
	for (int i=1;i<params->M;i++) {
		params->a[i] = 3*params->p;
		params->b[i] = 3*params->q;
	}
}


void init_enps_rrt(int n, int m, float p, float q, float delta, float epsilon, float x_init, float y_init, int algorithm, RRT_PARAMS* params, RRT_VARS* vars)
{
	
	params->n = n;
	params->m = m;
	params->N = 1 << n; // Number of nodes in the RRT
	params->M = 1 << m; // Number of obstacles
	params->p = p;
	params->q = q;
	params->delta = delta;
	params->epsilon = epsilon;
	params->algorithm = algorithm;
	
	vars->x = (float*)malloc((params->N)*sizeof(float));
	vars->y = (float*)malloc((params->N)*sizeof(float));
	
	vars->px = (float*)malloc((params->N)*sizeof(float));
	vars->py = (float*)malloc((params->N)*sizeof(float));
	
	vars->d = (float*)malloc((params->N)*sizeof(float));
	
	vars->xp = (float*)malloc((params->N)*sizeof(float));
	vars->yp = (float*)malloc((params->N)*sizeof(float));
	
	vars->index=1;
	vars->halt=0;
	
	params->a = (float*)malloc((params->M)*sizeof(float));
	params->b = (float*)malloc((params->M)*sizeof(float));
	vars->dp = (float*)malloc((params->M)*sizeof(float));
	
	vars->x[0] = x_init;
	vars->y[0] = y_init;
	for (int i=1;i<params->N;i++) {
		vars->x[i] = 3*p;
		vars->y[i] = 3*q;
	}

}

// TODO
void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars)
{
		
	
	
	
}


void nearest(RRT_PARAMS* params, RRT_VARS* vars)
{
	
	// compute distances
	for (int i=0;i<params->N;i++) {
		vars->d[i] = (vars->x[i] - vars->x_rand) * (vars->x[i] - vars->x_rand) +
						(vars->y[i] - vars->y_rand) * (vars->y[i] - vars->y_rand);
		vars->xp[i] = vars->x[i];
		vars->yp[i] = vars->y[i];
	}
	
	// compute minimun distance and nearest point
	for (int j=0; j < params->n; j++) {
		int limit = 1 << (params->n - j - 1);
		for (int i=0; i< limit; i++) {
			if (vars->d[i] > vars->d[i+limit]) {
				vars->d[i] = vars->d[i+limit];
				vars->xp[i] = vars->xp[i+limit];
				vars->yp[i] = vars->yp[i+limit];
			}
		}
	}
	vars->x_nearest = vars->xp[0];
	vars->y_nearest = vars->yp[0];
}



void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars)
{
	vars->x[vars->index] = vars->x_new;
	vars->y[vars->index] = vars->y_new;
	vars->px[vars->index] = vars->x_nearest;
	vars->py[vars->index] = vars->y_nearest;
}

// TODO
void extend_rrt_star(RRT_PARAMS* params, RRT_VARS* vars)
{
	
	
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

// TODO
void free_memory(RRT_PARAMS* params, RRT_VARS* vars)
{
	
	
}
