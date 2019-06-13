#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <enps_rrt.h>
#include <math.h>
#include <pgm.h>


#define MAP1 "../maps/map.pgm"
#define MAP2 "../maps/office.pgm"
#define MAP3 "../maps/labyrinth.pgm"
#define MAP4 "../maps/ccia_h.pgm"



int main(int argc, char* argv[])
{
	int selection=1;	
	if (argc>1) {
		selection = atoi(argv[1]);	
	}

	
	srand(time(NULL));
	
	RRT_PARAMS params;
	RRT_VARS vars;
	
	
	switch (selection)
	{
		case 1:
			init_params(MAP1,12,0.15,DEBUG,RRT_STAR_ALGORITHM,&params);
			init_vars(8,10,&params,&vars);
		break;
		case 2:
			init_params(MAP2,12,0.15,DEBUG,RRT_STAR_ALGORITHM,&params);
			init_vars(32,9.3,&params,&vars);
		break;
		case 3:
			init_params(MAP3,12,0.15,DEBUG,RRT_STAR_ALGORITHM,&params);
			init_vars(21.5,21.5,&params,&vars);
		break;
		case 4:
			init_params(MAP4,12,0.15,DEBUG,RRT_STAR_ALGORITHM,&params);
			init_vars(5.25,30.45,&params,&vars);
		break;
		default:
			printf("Invalid map\n");
			return 0;
	}	
	while (!vars.halt) {
		enps_rrt_one_iteration(&params,&vars);
	}
	
	if (params.debug) {
		
		for (int i=1;i<params.N;i++) {
			int x0 = (int)round(vars.x[i] / RESOLUTION);
			int y0 = (int)round(vars.y[i] / RESOLUTION);
			int x1 = (int)round(vars.px[i] / RESOLUTION);
			int y1 = (int)round(vars.py[i] / RESOLUTION);
			draw_line(params.map,x0,y0,x1,y1,0);
		}
		strcpy(params.map->file,"output.pgm");
		save_pgm(params.map);
	}
			
	free_memory(&params,&vars);
	
	return 0;
}
