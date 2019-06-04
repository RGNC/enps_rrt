#ifndef _ENPS_RRT_H_
#define _ENPS_RRT_H_

#include <pgm.h>

#define RRT_ALGORITHM      0
#define RRT_STAR_ALGORITHM 1


typedef struct
{
	int n;
	int m;
	
	// number of nodes
	int N; // N = 2^n
	
	// number of obstacles
	int M; // M = 2^m 
	
	// environment is size (p,q)
	float p;
	float q;
	
	float delta;
	float epsilon;
	
	// algorithm id (RRT_ALGORITHM, RRT_STAR_ALGORITHM)
	int algorithm;
	
	PGM* map;
	float resolution;
	
	// obstacles
	float *a;
	float *b;
	
} RRT_PARAMS;


typedef struct
{
	// RRT points (x,y)
	float *x;
	float *y;
	
	// RRT parents (px,py)
	float *px;
	float *py;
	
	
	float *d;
	float *xp;
	float *yp;
	float *dp;
	
	float x_rand;
	float y_rand;
	float x_new;
	float y_new;
	
	float x_nearest;
	float y_nearest;
	
	float collision;
	int index;
	int halt;
} RRT_VARS;


// Get a random float number in [0,1)
float rnd();

// TODO
void init_obstacles(RRT_PARAMS* params); 

void init_enps_rrt1(int n, int m, float p, float q, float delta, float epsilon, float x_init, float y_init, int algorithm, RRT_PARAMS* params, RRT_VARS* vars);

// TODO
void free_memory(RRT_PARAMS* params, RRT_VARS* vars);

void enps_rrt_one_iteration(RRT_PARAMS* params, RRT_VARS* vars);

void nearest(RRT_PARAMS* params, RRT_VARS* vars);

// TODO
void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars);

void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars);

// TODO
void extend_rrt_star(RRT_PARAMS* params, RRT_VARS* vars);

#endif
