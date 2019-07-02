ODIR = obj
SDIR = src
LDIR = lib
BDIR = bin
IDIR = include

CC=gcc
RM=rm

OBJ = main.o pgm.o
OBJ_LIB = enps_rrt.o
BIN = test
LIB = genetic.a

CFlags=-c -Wall -fopenmp #-O3
LDFlags= -lm -fopenmp

############ NVIDIA specifics
NCC=nvcc -ccbin=$(CC)
#GENCODE_SM20    := -gencode arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM60    := -gencode arch=compute_60,code=\"sm_60,compute_60\"
#GENCODE_SM61    := -gencode arch=compute_61,code=\"sm_61,compute_61\"
#GENCODE_SM75    := -gencode arch=compute_75,code=\"sm_75,compute_75\"
GENCODE_FLAGS   := $(GENCODE_SM20) $(GENCODE_SM35) $(GENCODE_SM60)\
                   $(GENCODE_SM61) $(GENCODE_SM75)
NCFlags=-c --compiler-options -Wall -Xcompiler -fopenmp $(GENCODE_FLAGS) #-O3
NLDFlags= -lm -Xcompiler -fopenmp 
############

############ Options for GPU and debugging
ifeq ($(gpu),1)
	OBJ += enps_rrt_gpu.o
	XCC=$(NCC) 	
	XLD=$(NLDFlags)
	OBJ_LIB = enps_rrt_gpu.o
else
	OBJ += enps_rrt.o
	XCC=$(CC)	
	XLD=$(LDFlags)
endif

ifeq ($(dbg),1)
	CFlags += -O0 -g
	NCFlags += -O0 -g
else	
	CFlags += -O3
	NCFlags += -O3
endif
############

BIN = test
LIB = genetic.a

all: $(OBJ) $(BIN) $(LIB)

$(LIB): $(patsubst %,$(ODIR)/%,$(OBJ_LIB))
	@mkdir -p $(LDIR)
	ar rcs $(LDIR)/$@ $^ 

$(BIN): $(patsubst %,$(ODIR)/%,$(OBJ))
	@mkdir -p $(BDIR)
	$(XCC) $^ $(XLD) -o $(BDIR)/$@ 

%.o: $(SDIR)/%.c
	@mkdir -p $(ODIR)
	$(CC) $(CFlags) -I$(IDIR) -o $(ODIR)/$@ $<

%.o: $(SDIR)/%.cu
	@mkdir -p $(ODIR)
	$(NCC) $(NCFlags) -I$(IDIR) -o $(ODIR)/$@ $<

clean:
	$(RM) $(patsubst %,$(ODIR)/%,$(OBJ)) $(BDIR)/$(BIN) $(LDIR)/$(LIB)
	
install:
	@mkdir -p /usr/local/enps_rrt/include/enps_rrt/
	@mkdir -p /usr/local/enps_rrt/lib/
	@cp $(LDIR)/$(LIB) /usr/local/enps_rrt/lib/
	@cp $(IDIR)/* /usr/local/enps_rrt/include/enps_rrt/
