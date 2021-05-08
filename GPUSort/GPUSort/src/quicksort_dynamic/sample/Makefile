include ../../util/config.mk

TARGET := main
EXTRA_ARCH := -gencode arch=compute_52,code=sm_52
DEVICE_CODE := -dc

CUDA_LDLIBS += -lcudadevrt

SRC_FOLDER := ..

## targets definitions follow
.PHONY: all host cuda
all: cuda
cuda: $(TARGET)

run: cuda
	./$(TARGET)
	
.PHONY: clean
clean:
	rm -f *.d *.o *.cuo $(TARGET)

$(TARGET): quicksort.o quicksort_link.o $(TARGET).o
	$(CXX) $(TNL_INCLUDE_DIRS) $(CUDA_LDFLAGS) -o $@ $^ $(CUDA_LDLIBS)

$(TARGET).o: $(TARGET).cu
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c -o $@ $<

quicksort.o: $(SRC_FOLDER)/quicksort.cu
	$(CUDA_CXX) $(CUDA_CXXFLAGS) $(EXTRA_ARCH) $(DEVICE_CODE) -c -o $@ $<

quicksort_link.o: quicksort.o
	$(CUDA_CXX) $(CUDA_LDFLAGS) -dlink -o $@ $< $(CUDA_LDLIBS)
