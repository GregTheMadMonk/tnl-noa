TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = mean-curvature-flow-eoc
INSTALL_DIR = ${HOME}/local
CXX = g++
CUDA_CXX = nvcc
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR)  -g -O0
LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-dbg-0.1

SOURCES = tnl-mean-curvature-flow-eoc.cpp
HEADERS = 
OBJECTS = tnl-mean-curvature-flow-eoc.o
DIST = $(SOURCES) Makefile

all: $(TARGET)
clean: 
	rm -f $(OBJECTS)	

dist: $(DIST)
	tar zcvf $(TARGET).tgz $(DIST) 

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/bin
	cp $(INSTALL_DIR)/share/tnl-0.1/examples/mean-curvature-flow-eoc

uninstall: $(TARGET)
	rm -f $(INSTALL_DIR)/bin/$(TARGET) 
	rm -f $(INSTALL_DIR)/share/tnl-0.1/examples/mean-curvature-flow-eoc

$(TARGET): $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $(CXX_FLAGS) $<
