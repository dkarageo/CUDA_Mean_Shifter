CC=gcc
CXX=g++
CFLAGS=-O3 -Wall -Wextra -std=c99
LDLIBS=-lm -fopenmp
OBJDIR=obj

NVCC=nvcc
CUFLAGS=-O3
CULDLIBS= -lcusolver -lm -lcudart -lcuda
CULINK=-L/usr/local/cuda/lib64
CUOBJDIR=cuobj

BINDIR=bin

all: MODS=-DINCLUDE_NOSM
cpu_only: MODS=-DCPU_ONLY
gpu_only: MODS=-DGPU_ONLY
gpu_with_nosm: MODS=-DGPU_ONLY -DINCLUDE_NOSM

vpath %.c source source/svd tests
vpath %.h source source/svd
vpath %.cu source


objects=$(addprefix $(OBJDIR)/, \
				algebra.o \
				matrix.o \
				mean_shift.o \
				svd_double.o \
				demo.o \
				utils.o )

cuda_objects=$(addprefix $(CUOBJDIR)/, \
				cuda_algebra.o \
				cuda_mean_shift.o \
				cuda_mean_shift_nosm.o)


all: $(objects) $(cuda_objects) | $(BINDIR)
	$(CC) $(objects) $(cuda_objects) -o $(BINDIR)/demo_mean_shifter $(CFLAGS) \
			 $(CULINK) $(CULDLIBS) $(LDLIBS) $(MODS)

cpu_only: $(objects) | $(BINDIR)
	$(CC) $(objects) -o $(BINDIR)/cpu_demo_mean_shifter $(CFLAGS) $(LDLIBS) $(MODS)

gpu_only: $(objects) $(cuda_objects) | $(BINDIR)
	$(CC) $(objects) $(cuda_objects) -o $(BINDIR)/gpu_demo_mean_shifter $(CFLAGS) \
			 $(CULINK) $(CULDLIBS) $(LDLIBS) $(MODS)

gpu_with_nosm: $(objects) $(cuda_objects) | $(BINDIR)
	$(CC) $(objects) $(cuda_objects) -o $(BINDIR)/gpu_demo_mean_shifter_nosm $(CFLAGS) \
			 $(CULINK) $(CULDLIBS) $(LDLIBS) $(MODS)

$(OBJDIR)/%.o : %.c | $(OBJDIR)
	$(CC) $< -c -o $@ $(CFLAGS) $(MODS)

$(OBJDIR)/demo.o : demo.c FORCE
	$(CC) $< -c -o $@ $(CFLAGS) $(MODS)

$(CUOBJDIR)/%.o : %.cu | $(CUOBJDIR)
	$(NVCC) $< -c -o $@ $(CUFLAGS) $(CULDLIBS)

$(OBJDIR):
	mkdir $(OBJDIR)

$(CUOBJDIR):
	mkdir $(CUOBJDIR)

$(BINDIR):
	mkdir $(BINDIR)

clean:
	rm $(OBJDIR)/*.o
	rm $(CUOBJDIR)/*.o

purge: clean
	rm bin/*

run:
	./$(BINDIR)/demo_mean_shifter datasets/r15.karas 1.0

run_cpu:
	./$(BINDIR)/cpu_demo_mean_shifter datasets/r15.karas 1.0

run_gpu:
	./$(BINDIR)/gpu_demo_mean_shifter datasets/r15.karas 1.0

run_gpu_all:
	./$(BINDIR)/gpu_demo_mean_shifter_nosm datasets/r15.karas 1.0

runp:
	./$(BINDIR)/demo_mean_shifter $(data) $(h)

runp_cpu:
	./$(BINDIR)/cpu_demo_mean_shifter $(data) $(h)

runp_gpu:
	./$(BINDIR)/gpu_demo_mean_shifter $(data) $(h)

runp_gpu_all:
	./$(BINDIR)/gpu_demo_mean_shifter_nosm $(data) $(h)

.PHONY: all cpu_only gpu_only gpu_with_nosm clean purge test FORCE
FORCE:
