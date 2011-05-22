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
          matrix/tnlMatrixTester.h \
          matrix/tnlCSRMatrixTester.h \
          matrix/tnlEllpackMatrixTester.h \
          matrix/tnlRgCSRMatrixTester.h \
          matrix/tnlAdaptiveRgCSRMatrixTester.h \
          solver/tnlMersonSolverTester.h


check_PROGRAMS = tnl-unit-tests
                  
#tnl_unit_tests_SOURCES = $(headers) tnl-unit-tests.cpp
#tnl_unit_tests_LDADD = ../../src/libtnl-0.1.la \
#                       ../../src/core/libcore-tests.la \
#                       ../../src/diff/libdiff-tests.la \
#                       ../../src/debug/libtnldebug-0.1.la

#if BUILD_CUDA
tnl_unit_tests_SOURCES = $(headers) tnl-unit-tests-cuda.cu
tnl_unit_tests_CXXFLAGS = "-DHAVE_CUDA"
tnl_unit_tests_LDADD = ../../src/libtnl-0.1.la \
                           ../../src/core/libcore-tests.la \
                           ../../src/diff/libdiff-tests.la \
                           ../../src/debug/libtnldebug-0.1.la
#endif


if BUILD_DBG
check_PROGRAMS += tnl-unit-tests-dbg
tnl_unit_tests_dbg_SOURCES = $(headers) tnl-unit-tests.cpp
tnl_unit_tests_dbg_CXXFLAGS = $(DBGCXXFLAGS)
tnl_unit_tests_dbg_LDADD =  ../../src/libtnl-dbg-0.1.la \
                            ../../src/core/libcore-tests-dbg.la \
                            ../../src/diff/libdiff-tests-dbg.la \
                            ../../src/debug/libtnldebug-dbg-0.1.la                            
endif

TESTS = tnl-unit-tests


#        tnl-benchmarks

##tnl-unit-tests \
##        tnl-benchmarks
