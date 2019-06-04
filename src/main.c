#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <enps_rrt.h>
#include <pgm.h>


#define RESOLUTION   0.05        // Each pixel is 5x5 cm 
#define ROBOT_RADIUS 0.25        // This is the robot radius in meters, epsilon parameter
#define OBSTACLE_THRESHOLD 250   // Gray level in PGM file for obstacle


void init_enps_rrt(RRT_PARAMS* params, RRT_VARS* vars)
{
	params->map = load_pgm("../maps/basic.pgm");
	params->resolution = RESOLUTION;
	params->epsilon = ROBOT_RADIUS;
	params->delta = 2*ROBOT_RADIUS;
	for (int i=0;i<params->map->height;i++) {
		for (int j=0;j<params->map->width;j++) {
			int x = params->map->raster[i*params->map->width+j];
			if (x<OBSTACLE_THRESHOLD) {
				printf("*");
			} else {
				printf(".");
			}
		
		}
		printf("\n");
		
	}
}



int main()
{
	srand(time(NULL));
	
	RRT_PARAMS params;
	RRT_VARS vars;
	
	//init_enps_rrt(3,3,10,10,0.2,0.05,5,5,RRT_ALGORITHM, &params,&vars);
	
	init_enps_rrt(&params,&vars);
		
	while (!vars.halt) {
		enps_rrt_one_iteration(&params,&vars);
	}
	
	free_memory(&params,&vars);
	
	return 0;
}
