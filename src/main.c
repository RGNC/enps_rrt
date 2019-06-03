#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <enps_rrt.h>

int main()
{
	srand(time(NULL));
	
	RRT_PARAMS params;
	RRT_VARS vars;
	
	init_enps_rrt(3,3,10,10,0.2,0.05,5,5,RRT_ALGORITHM, &params,&vars);
		
	while (!vars.halt) {
		enps_rrt_one_iteration(&params,&vars);
	}
	
	free_memory(&params,&vars);
	
	return 0;
}
