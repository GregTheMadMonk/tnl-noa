TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}
#TNL_INCLUDE_DIR=${HOME}/workspace/tnl/src

TARGET = navier-stokes
CONFIG_FILE = $(TARGET).cfg.desc
INSTALL_DIR = ${HOME}/local
CXX = g++
CUDA_CXX = nvcc
OMP_FLAGS = -DHAVE_OPENMP -fopenmp 
#CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR) -O0 -g -DDEBUG $(OMP_FLAGS) -DTEMPLATE_EXPLICIT_INSTANTIATION
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR) -O3 $(OMP_FLAGS) -DTEMPLATE_EXPLICIT_INSTANTIATION
#CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR) -O3 $(OMP_FLAGS) -pg -DTEMPLATE_EXPLICIT_INSTANTIATION
#CXX_FLAGS = -DHAVE_NOT_CXX11 -I$(TNL_INCLUDE_DIR) -O3 $(OMP_FLAGS) -DTEMPLATE_EXPLICIT_INSTANTIATION


LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-0.1 -lgomp

SOURCES = main.cpp
HEADERS = navierStokesSetter.h \
          navierStokesSetter_impl.h \
          navierStokesSolver.h \
          navierStokesSolver_impl.h \
          navierStokesSolverMonitor.h \
          navierStokesSolverMonitor_impl.h          
OBJECTS = main.o
DIST = $(SOURCES) $(HEADERS) $(CONFIG_FILE) Makefile

all: $(TARGET)

clean: 
	rm -f $(OBJECTS)
	rm -f $(TARGET)-conf.h	

dist: $(DIST)
	tar zcvf $(TARGET).tgz $(DIST) 

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/bin
	cp $(CONFIG_FILE) $(INSTALL_DIR)/share
	mkdir -p $(INSTALL_DIR)/share/$(TARGET)
	cp -rv share/* $(INSTALL_DIR)/share/$(TARGET)
	cp make-png-from-gnuplot $(INSTALL_DIR)/bin
	cp merge-figures $(INSTALL_DIR)/bin
	chmod +x $(INSTALL_DIR)/bin/make-png-from-gnuplot
	chmod +x $(INSTALL_DIR)/bin/merge-figures

uninstall: $(TARGET)
	rm -f $(INSTALL_DIR)/bin/$(TARGET) 
	rm -f $(CONFIG_FILE) $(INSTALL_DIR)/share
	rm -rf $(INSTALL_DIR)/share/$(TARGET)
	rm -rf $(INSTALL_DIR)/bin/make-png-from-gnuplot
	rm -rf $(INSTALL_DIR)/bin/merge-figures	

$(TARGET): $(TARGET)-conf.h $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp  $(HEADERS)
	$(CXX) -c -o $@ $(CXX_FLAGS) $<

$(TARGET)-conf.h:
	echo "#define CONFIG_FILE \"${INSTALL_DIR}/share/${CONFIG_FILE}\" " > $(TARGET)-conf.h 

