TNL_VERSION=0.1
TNL_INSTALL_DIR=${HOME}/local/lib
TNL_INCLUDE_DIR=${HOME}/local/include/tnl-${TNL_VERSION}

TARGET = program-name
CONFIG_FILE = $(TARGET).cfg.desc
INSTALL_DIR = ${HOME}/local
CXX = g++
CUDA_CXX = nvcc
CXX_FLAGS = -std=gnu++0x -I$(TNL_INCLUDE_DIR)
LD_FLAGS = -L$(TNL_INSTALL_DIR) -ltnl-0.1

SOURCES = main.cpp
HEADERS = 
OBJECTS = main.o
DIST = $(SOURCES) Makefile

all: $(TARGET)
clean: 
	rm -f $(OBJECTS)
	rm -f $(TARGET)-conf.h	

dist: $(DIST)
	tar zcvf $(TARGET).tgz $(DIST) 

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)/bin
	cp $(CONFIG_FILE) $(INSTALL_DIR)/share

uninstall: $(TARGET)
	rm -f $(INSTALL_DIR)/bin/$(TARGET) 
	rm -f $(CONFIG_FILE) $(INSTALL_DIR)/share

$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGET) $(OBJECTS) $(LD_FLAGS)

%.o: %.cpp $(TARGET)-conf.h $(HEADERS)
	$(CC) -c -o $@ $(CXX_FLAGS) $<

$(TARGET)-conf.h:
	echo "#define CONFIG_FILE \"${INSTALL_DIR}/share/${CONFIG_FILE}\" " > $(TARGET)-conf.h 

