TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = tnl-heat-equation
CONFIG_FILE = $(TARGET).cfg.desc
INSTALL_DIR = ${HOME}/local
CXX = g++
CUDA_CXX = nvcc
OMP_FLAGS = -DHAVE_OPENMP -fopenmp 
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR) -O3 $(OMP_FLAGS)
LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-0.1 -lgomp

SOURCES = tnl-heat-equation.cpp
CUDA_SOURCES = tnl-heat-equation.cu
HEADERS = 
OBJECTS = tnl-heat-equation.o
DIST = $(SOURCES) Makefile

all: $(TARGET)
clean: 
	rm -f $(OBJECTS)
	rm -f $(TARGET)-conf.h	

dist: $(DIST)
	tar zcvf $(TARGET).tgz $(DIST) 

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/bin

uninstall: $(TARGET)
	rm -f $(INSTALL_DIR)/bin/$(TARGET) 

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $(CXX_FLAGS) $<

