ODIR = obj
SDIR = src
LDIR = lib
BDIR = bin
IDIR = include

OBJ = main.o enps_rrt.o pgm.o
OBJ_LIB = enps_rrt.o
      
BIN = test
LIB = genetic.a

CFlags=-c -O3 -Wall -fopenmp 
LDFlags= -lm -fopenmp 
CC=gcc
RM=rm

all: $(OBJ) $(BIN) $(LIB)

$(LIB): $(patsubst %,$(ODIR)/%,$(OBJ_LIB))
	@mkdir -p $(LDIR)
	ar rcs $(LDIR)/$@ $^ 

$(BIN): $(patsubst %,$(ODIR)/%,$(OBJ))
	@mkdir -p $(BDIR)
	$(CC) $^ $(LDFlags) -o $(BDIR)/$@ 

%.o: $(SDIR)/%.c	
	@mkdir -p $(ODIR)
	$(CC) $(CFlags) -I$(IDIR) -o $(ODIR)/$@ $<
	
clean:
	$(RM) $(patsubst %,$(ODIR)/%,$(OBJ)) $(BDIR)/$(BIN) $(LDIR)/$(LIB)
	
install:
	@mkdir -p /usr/local/enps_rrt/include/enps_rrt/
	@mkdir -p /usr/local/enps_rrt/lib/
	@cp $(LDIR)/$(LIB) /usr/local/enps_rrt/lib/
	@cp $(IDIR)/* /usr/local/enps_rrt/include/enps_rrt/
