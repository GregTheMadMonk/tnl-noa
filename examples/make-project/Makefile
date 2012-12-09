TNL_VERSION="0.1"
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = program-name
CXX = g++
CUDA_CXX = nvcc
CXX_FLAGS = -std=gnu++0x -I${TNL_INCLUDE_DIR}
LD_FLAGS = -L${TNL_INSTALL_DIR} -ltnl-0.1

OBJECTS = Finite_Elements.o MatrixStorage/CSRMatrix.o MatrixStorage/objVector.o ParameterContainer/parameterContainer.o
DIST = main.cpp Makefile

all: prog
clean: 
	rm -f $(OBJECTS)
dist: $(DIST)
	tar zcvf prog.tgz $(DIST) 

install: prog
	mv prog $(PROG)

uninstall: prog
	rm f /usr/local/bin 

prog: $(OBJECTS)
	$(CC) -o prog $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $(CXX_FLAGS) $< 

