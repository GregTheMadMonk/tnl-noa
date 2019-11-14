# Replace the following with your TNL installation path
TNL_HEADERS = ${HOME}/.local/include/tnl
INSTALL_DIR = ${HOME}/.local
WITH_CUDA = yes
WITH_OPENMP = yes
WITH_DEBUG = no

CXX = g++
CUDA_CXX = nvcc
#CXXFLAGS = -DNDEBUG -O3 -funroll-loops -g -std=c++11 -Dlinux
CXXFLAGS = -DNDEBUG -O0 -g -Dlinux -std=c++11 -fPIC
CPPFLAGS = -MD -MP

LDFLAGS += -lm #-lgomp
