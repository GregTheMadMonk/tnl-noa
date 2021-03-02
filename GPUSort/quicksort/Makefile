include config.mk

CUDA_SOURCES := $(wildcard *.cu)
CUDA_TARGETS := $(CUDA_SOURCES:%.cu=%)

## targets definitions follow
.PHONY: all host cuda
all: cuda
cuda: $(CUDA_TARGETS)

run: cuda
	./$(CUDA_TARGETS)

.PHONY: clean
clean:
	rm -f *.d *.o *.cuo $(CUDA_TARGETS)

# use .cuo instead of .cu.o to avoid problems with the implicit rules: https://stackoverflow.com/q/62967939
# (and use the host compiler for linking CUDA, nvcc does not understand that .cuo is an object file)
$(CUDA_TARGETS): % : %.o
	$(CUDA_CXX) $(CUDA_LDFLAGS) -o $@ $< $(CUDA_LDLIBS) -lcudadevrt

$(CUDA_SOURCES:%.cu=%.o): %.o : %.cu
	$(CUDA_CXX) $(CUDA_CPPFLAGS) $(CUDA_CXXFLAGS) -gencode arch=compute_52,code=sm_52 -dc -c -o $@ $<