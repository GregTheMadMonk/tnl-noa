## Process this file with automake to produce Makefile.in

#SUBDIRS = core \
#          diff \
#          matrix \
#          solver

headers = core/tnl-cuda-kernels.h \
          core/tnlArrayTester.h \
	       core/tnlCUDAKernelsTester.h \
	       core/tnlCommunicatorTester.h \
	       core/tnlFileTester.h \
	       core/tnlGridTester.h \
 	       core/tnlLongVectorCUDATester.h \
	       core/tnlLongVectorHostTester.h \
      	 core/tnlRealTester.h \
	       core/tnlSharedMemoryTester.h \
	       core/tnlVectorTester.h \
          diff/tnlMPIMeshTester.h \
          diff/tnlMersonSolverCUDATester.cu.h \
          matrix/tnlCSRMatrixTester.h \
          matrix/tnlEllpackMatrixTester.h \
          matrix/tnlRgCSRMatrixTester.h \
          solver/tnlMersonSolverTester.h


tnl_unit_tests_sources = tnl-unit-tests.cpp
tnl_cuda_unit_tests_sources = tnl-cuda-unit-tests.cu

check_PROGRAMS = tnl-unit-tests
                  
tnl_unit_tests_SOURCES = $(tnl_unit_tests_sources) $(headers)
tnl_unit_tests_LDADD = ../../src/libtnl-0.1.la \
                       ../../src/core/libcore-tests.la \
                       ../../src/diff/libdiff-tests.la \
                       ../../src/debug/libtnldebug-0.1.la

if BUILD_CUDA
check_PROGRAMS += tnl-cuda-unit-tests
tnl_cuda_unit_tests_SOURCES = $(tnl_cuda_unit_tests_sources)
tnl_cuda_unit_tests_CXXFLAGS = "-DHAVE_CUDA"
tnl_cuda_unit_tests_LDADD = ../../src/libtnl-0.1.la \
                           ../../src/core/libcore-tests.la \
                           ../../src/diff/libdiff-tests.la \
                           ../../src/debug/libtnldebug-0.1.la
endif


if BUILD_DBG
check_PROGRAMS += tnl-unit-tests-dbg
tnl_unit_tests_dbg_SOURCES = $(tnl_unit_tests_sources)
tnl_unit_tests_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnl_unit_tests_dbg_LDADD =  ../../src/libtnl-dbg-0.1.la \
                            ../../src/core/libcore-tests-dbg.la \
                            ../../src/diff/libdiff-tests-dbg.la \
                            ../../src/debug/libtnldebug-dbg-0.1.la                            
endif

#if BUILD_CUDA && BUILD_DBG
#check_PROGRAMS += tnl-cuda-unit-tests-dbg
#tnl_cuda_unit_tests_dbg_SOURCES = $(tnl_cuda_unit_tests_sources)
#tnl_cuda_unit_tests_dbg_CXXFLAGS = $(DBGCXXFLAGS)
#tnl_cuda_unit_tests_dbg_LDADD =  ../../src/libtnl-dbg-0.1.la \
#                                 ../../src/core/libcore-tests-dbg.la \
#                                 ../../src/diff/libdiff-tests-dbg.la \
#                                 ../../src/debug/libtnldebug-dbg-0.1.la                            
#endif

TESTS = tnl-unit-tests

#if BUILD_CUDA
#TESTS += tnl-cuda-unit-tests
#endif

#        tnl-benchmarks

##tnl-unit-tests \
##        tnl-benchmarks
