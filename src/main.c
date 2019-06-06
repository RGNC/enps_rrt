#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <enps_rrt.h>
#include <pgm.h>


#define RESOLUTION   0.05        // Each pixel is 5x5 cm 
#define ROBOT_RADIUS 0.2         // This is the robot radius in meters, epsilon parameter


void init_enps_rrt(const char* file, int n, int algorithm, RRT_PARAMS* params, RRT_VARS* vars)
{
		
	PGM* map = load_pgm(file);
	remove_inner_obstacles(map);
	
	params->epsilon = ROBOT_RADIUS;
	params->delta = 2*ROBOT_RADIUS;
	
	params->n = n;
	params->N = 1<<n;
	
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
	
	params->a = (float*)realloc(params->a, (params->M)*sizeof(float));
	params->b = (float*)realloc(params->b, (params->M)*sizeof(float));
	
	for (int i=c;i<params->M;i++) {
		params->a[i] = 3* params->p;
		params->b[i] = 3* params->q;
	}
	
	
	destroy_pgm(map);
}



int main(int argc, char* argv[])
{
	srand(time(NULL));
	
	RRT_PARAMS params;
	RRT_VARS vars;
	
	//init_enps_rrt(3,3,10,10,0.2,0.05,5,5,RRT_ALGORITHM, &params,&vars);
	
	init_enps_rrt(argv[1],10,RRT_ALGORITHM,&params,&vars);
		
	//while (!vars.halt) {
	//	enps_rrt_one_iteration(&params,&vars);
	//}
	
	
	
	free_memory(&params,&vars);
	
	return 0;
}
