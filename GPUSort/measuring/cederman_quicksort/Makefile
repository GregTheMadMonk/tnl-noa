include ../util/config.mk

CUDA_SOURCES := $(wildcard *.cu)
CUDA_TARGETS := $(CUDA_SOURCES:%.cu=%)

## targets definitions follow
.PHONY: all host cuda
all: cuda
cuda: $(CUDA_TARGETS)

run: cuda
	./$(CUDA_TARGETS)
	
measure: cuda
	./$(CUDA_TARGETS) ../results/cederman_quicksort.csv

.PHONY: clean
clean:
	rm -f *.d *.o *.cuo $(CUDA_TARGETS)

$(CUDA_TARGETS): $(CUDA_TARGETS).cu
	nvcc $(CUDA_TARGETS).cu -o $(CUDA_TARGETS)

