

/*  ENPS RRT
    Copyright (C) 2019  Ignacio Perez-Hurtado
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.*/

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
