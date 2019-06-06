#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pgm.h"

char last_error[256];

int next_line(FILE* fp, char* buffer, PGM* pgm) {
	int i,error;
	buffer[0]=0;
	int counter=0;
	do{
		i=0;
		do{
			if ( (error = fscanf(fp,"%c",&buffer[i++])) != EOF) {
				if (buffer[i-1]=='\n') {
					buffer[i-1]=0;
				}
				counter++;
			} 
		} while(error!=EOF && buffer[i-1]!=0);
	} while(error!=EOF && (buffer[0]=='#' || buffer[0]==0));
	
	if (error==EOF) {
		counter=0;
		fclose(fp);
		sprintf(last_error,"Error reading file.");
		free(pgm);
	}
	return counter;
}

PGM* load_pgm(const char* file)
{
	PGM* pgm=NULL;
	char buffer[256];
	int bytes;
	int offset=0;
	int raster_size;
	FILE* fp = fopen(file,"r");
	if (fp==NULL) {
		sprintf(last_error,"Error: Cannot open %s.",file);
		return NULL;
	}
	pgm = (PGM*)malloc(sizeof(PGM));
	strcpy(pgm->file,file);
	
	if ((bytes= next_line(fp,buffer,pgm))==0) {
		return NULL;
	} else {
		offset+=bytes;
	}
	if (strcmp(buffer,"P5")!=0) {
		fclose(fp);
		sprintf(last_error,"Error: Invalid PGM file.");
		free(pgm);
		return NULL;
	}
	if ((bytes= next_line(fp,buffer,pgm))==0) {
		return NULL;
	} else {
		offset+=bytes;
	}	
	sscanf(buffer,"%d %d",&(pgm->width),&(pgm->height));
	
	if (pgm->width<=0 || pgm->width>=65536  || pgm->height<=0 || pgm->width>=65536) {
		fclose(fp);
		sprintf(last_error,"Error: Invalid raster size.");
		free(pgm);
		return NULL;
	}
	
	if ((bytes= next_line(fp,buffer,pgm))==0) {
		return NULL;
	} else {
		offset+=bytes;
	}	
	
	sscanf(buffer,"%d",&(pgm->maxval));
	
	if (pgm->maxval<=0 || pgm->maxval>=256) {
		fclose(fp);
		sprintf(last_error,"Error: Invalid maximum gray value.");
		free(pgm);
		return NULL;
	}
	
	fclose(fp);
	
	raster_size = pgm->width * pgm->height;
	
	pgm->raster = (unsigned char*)malloc(raster_size);
	
	fp=fopen(file,"rb");
	fseek(fp,offset,SEEK_SET);
	
	if (fread(pgm->raster,1,raster_size,fp) != raster_size) {
		fclose(fp);
		sprintf(last_error,"Error reading raster data.");
		free(pgm->raster);
		free(pgm);
		return NULL;
	}
	
	fclose(fp);
	return pgm;
}


int save_pgm(PGM* pgm)
{
	FILE* fp = fopen(pgm->file,"w");
	if (fp==NULL) {
		sprintf(last_error,"Cannot save file.");
		return 0;
	}
	if (fprintf(fp,"P5\n# PGM file\n%d %d\n255\n",pgm->width,pgm->height)<0) {
		sprintf(last_error,"Cannot save file.");
		fclose(fp);
		return 0;
	}
	int raster_size = pgm->width * pgm->height;
	if (fwrite(pgm->raster,1,raster_size,fp) != raster_size) {
		sprintf(last_error,"Cannot save file.");
		fclose(fp);
		return 0;
	}
	fclose(fp);
	return 1;
}


void destroy_pgm(PGM* pgm)
{
	if (pgm!=NULL) {
		free(pgm->raster);
		free(pgm);
	}
}

void remove_inner_obstacles(PGM* pgm)
{
	unsigned char* raster = (unsigned char*)malloc(pgm->width * pgm->height);
	for (int i=0;i<pgm->height;i++) {
		for (int j=0;j<pgm->width;j++) {
			
			if (IS_INNER_OBSTACLE(pgm,i,j)) {
				raster[i*pgm->width+j] = 255;
			} else {
				raster[i*pgm->width+j] = pgm->raster[i*pgm->width+j];
			}
		}
	}
	free(pgm->raster);
	pgm->raster = raster;
}


void draw_line(PGM* pgm, int x0, int y0, int x1, int y1, unsigned char color)
{
	double x = x0;
	double y = y0;
	
	double gx = x1;
	double gy = y1;

	double d1 = (gx-x)*(gx-x)+(gy-y)*(gy-y);
	double d2 = d1;
		
	double d = sqrt(d1);
	
	double ix = (gx-x)/d;
	double iy = (gy-y)/d;
	
	do {
		d1 = d2;
		pgm->raster[(int)y * pgm->width + (int)x] = color;
		x += ix;
		y += iy;
		d2 = (gx-x)*(gx-x)+(gy-y)*(gy-y);
	}while (d2<d1);
	
}

int detect_obstacle(PGM* pgm, int x0, int y0, int x1, int y1, unsigned char threshold)
{
	int obstacle = 0;
	double x = x0;
	double y = y0;
	
	double gx = x1;
	double gy = y1;

	double d1 = (gx-x)*(gx-x)+(gy-y)*(gy-y);
	double d2 = d1;
		
	double d = sqrt(d1);
	
	double ix = (gx-x)/d;
	double iy = (gy-y)/d;
	
	do {
		d1 = d2;
		if (pgm->raster[(int)y * pgm->width + (int)x] < threshold) {
			obstacle = 1;
		}
		x += ix;
		y += iy;
		d2 = (gx-x)*(gx-x)+(gy-y)*(gy-y);
	}while (d2<d1 && !obstacle);
	
	return obstacle;
}

