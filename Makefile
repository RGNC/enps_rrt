ODIR = obj
SDIR = src
BDIR = bin
IDIR = include

OBJ = main.o enps_rrt.o pgm.o
      
BIN = enps_rrt

CFlags=-c -O3 -Wall -fopenmp 
LDFlags= -lm -fopenmp 
CC=gcc
RM=rm

all: $(OBJ) $(BIN) 

$(BIN): $(patsubst %,$(ODIR)/%,$(OBJ))
	@mkdir -p $(BDIR)
	$(CC) $^ $(LDFlags) -o $(BDIR)/$@ 

%.o: $(SDIR)/%.c	
	@mkdir -p $(ODIR)
	$(CC) $(CFlags) -I$(IDIR) -o $(ODIR)/$@ $<
	
clean:
	$(RM) $(patsubst %,$(ODIR)/%,$(OBJ)) $(BDIR)/$(BIN) 
	
