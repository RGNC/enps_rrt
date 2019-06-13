#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <enps_rrt.h>
#include <pgm.h>

#define MAP "../maps/map.pgm"

int main(int argc, char* argv[])
{
	srand(time(NULL));
	
	RRT_PARAMS params;
	RRT_VARS vars;
	
	init_params(MAP,12,DEBUG,RRT_ALGORITHM,&params);
	init_vars(8,10,&params,&vars);
		
	while (!vars.halt) {
		enps_rrt_one_iteration(&params,&vars);
	}
		
		
	if (params.debug) {
		strcpy(params.map->file,"output.pgm");
		save_pgm(params.map);
	}	
		
	free_memory(&params,&vars);
	
	return 0;
}
