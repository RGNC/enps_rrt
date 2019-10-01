#ifndef _PGM_H_
#define _PGM_H_


#define IS_OBSTACLE(pgm,i,j) (i<0 || j<0 || i>=pgm->height || j>=pgm->width || pgm->raster[(i)*pgm->width+(j)]<250)

#define IS_INNER_OBSTACLE(pgm,i,j) (IS_OBSTACLE(pgm,i-1,j-1) && IS_OBSTACLE(pgm,i-1,j) && IS_OBSTACLE(pgm,i-1,j+1) && IS_OBSTACLE(pgm,i,j-1) && IS_OBSTACLE(pgm,i,j) && IS_OBSTACLE(pgm,i,j+1) &&  IS_OBSTACLE(pgm,i+1,j-1) && IS_OBSTACLE(pgm,i+1,j) && IS_OBSTACLE(pgm,i+1,j+1))

typedef struct
{
	char file[64];
	int width;	
	int height;
	int maxval;
	unsigned char *raster;
} PGM;

int save_pgm(PGM* pgm);

PGM* load_pgm(const char* file);

void draw_line(PGM* pgm, int x0, int y0, int x1, int y1, unsigned char color);

int detect_obstacle(PGM* pgm, int x0, int y0, int x1, int y1, unsigned char threshold);
    
void destroy_pgm(PGM* pgm);

void remove_inner_obstacles(PGM* pgm);


#endif
