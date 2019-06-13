#ifndef _ENPS_RRT_H_
#define _ENPS_RRT_H_

#include <pgm.h>


#define RRT_ALGORITHM      0
#define RRT_STAR_ALGORITHM 1

#define RESOLUTION   0.05        // Each pixel is 5x5 cm 
#define ROBOT_RADIUS 0.2         // This is the robot radius in meters, epsilon parameter

#define NO_DEBUG 0
#define DEBUG    1

#define INF 9999999

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
	
	// obstacles
	float *a;
	float *b;
	
	PGM* map;
	
	int debug;
	
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
	
	float *c;
	float *cp;
	
	float *dp;
	float *dpp;
	
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


typedef struct
{
	float x;
	float y;
	float d;
} XYD;

XYD xyd_min2(XYD a, XYD b);
void init_params(const char* file, int n, float delta,int debug, int algorithm, RRT_PARAMS* params);
void init_vars(float x_init, float y_init, const RRT_PARAMS* params, RRT_VARS* vars);
void free_memory(RRT_PARAMS* params, RRT_VARS* vars);
float rnd(); // Get a random float number in [0,1)
float p_dist(float Cx, float Cy, float Ax, float Ay, float Bx, float By);
void enps_rrt_one_iteration(RRT_PARAMS* params, RRT_VARS* vars);
void nearest(RRT_PARAMS* params, RRT_VARS* vars);
void obstacle_free(RRT_PARAMS* params, RRT_VARS* vars);
void extend_rrt(RRT_PARAMS* params, RRT_VARS* vars);
void extend_rrt_star(RRT_PARAMS* params, RRT_VARS* vars);

#endif
