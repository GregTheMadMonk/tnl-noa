# Replace the following with your project name
PROJECT_NAME = tnl-make-example

# Replace the following with your TNL installation path
TNL_HEADERS = ${HOME}/.local/include
INSTALL_DIR = ${HOME}/.local
WITH_CUDA = yes
WITH_OPENMP = yes
WITH_DEBUG = no

# If TNL is installed on your system, CUDA arch can be detected automatically using
# a tool 'tnl-cuda-arch'. This is done by default if CUDA_ARCH is set to 'auto'. 
# Otherwise, if you set it manually by telling the CUDA architecture number
# i.e 50, 60 etc.
CUDA_ARCH = auto

# Set-up compilers
CXX = g++
CUDA_CXX = nvcc

# Set-up CXX_FLAGS
CXXFLAGS = -pthread -Wall -Wno-unused-local-typedefs -Wno-unused-variable -Wno-unknown-pragmas -std=c++14 -I$(TNL_HEADERS)
ifeq ( $(WITH_DEBUG), yes )
	CXXFLAGS += -O0 -g
else
   CXXFLAGS += -DNDEBUG -O3 -funroll-loops
endif

# Set-up CUDA_CXXFLAGS
CUDA_CXXFLAGS = -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda -Xcudafe --std c++14 -I$(TNL_HEADERS) 
ifeq ( $(WITH_CUDA), yes )
   CUDA_CXXFLAGS += -DHAVE_CUDA
   ifeq ( $(CUDA_ARCH), auto )
      CUDA_CXXFLAGS += `tnl-cuda-arch`
   else
      CUDA_CXXFLAGS += -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
   endif
endif

# Set-up CPPFLAGS
CPPFLAGS = -MD -MP

# Set-up LDFLAGS
LDFLAGS += -lm
ifeq ( $(WITH_OPENMP), yes )
   LDFLAGS += -lgomp
endif
