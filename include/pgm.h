#ifndef _PGM_H_
#define _PGM_H_

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

#endif
