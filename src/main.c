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
	#define IS_OBSTACLE(i,j) (i<0 || j<0 || i>=params->map->height || i>=params->map->width || params->map->raster[i*params->map->width+j]<OBSTACLE_THRESHOLD)
	params->map = load_pgm("../maps/ccia.pgm");
	params->resolution = RESOLUTION;
	params->epsilon = ROBOT_RADIUS;
	params->delta = 2*ROBOT_RADIUS;
	
	params->p = params->map->width * RESOLUTION;
	params->q = params->map->height * RESOLUTION;
	
	float x=0,y=0;
	int c=0;
	params->a = (float*)malloc(params->map->width * params->map->height * sizeof(float));
	params->b = (float*)malloc(params->map->width * params->map->height * sizeof(float));
	for (int i=0;i<params->map->height;i++) {
		
		for (int j=0;j<params->map->width;j++) {
			x+=RESOLUTION;
			if (IS_OBSTACLE(i,j)) {
					
			
			
				params->a[c] = x;
				params->b[c] = y;
				c++;
			}
		}
		y+=RESOLUTION;
		x = 0;
	}
	printf("%d\n",c);
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
