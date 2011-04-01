## Process this file with automake to produce Makefile.in

SUBDIRS = unit-tests

check_PROGRAMS = matrix-formats-test \
                 sparse-matrix-benchmark \
                 matrix-solvers-benchmark \
                 tnl-benchmarks \
                 reorder-csr

matrix_formats_test_sources = matrix-formats-test.h
sparse_matrix_benchmark_sources = sparse-matrix-benchmark.h
matrix_solvers_benchmark_sources = matrix-solvers-benchmark.h
tnl_benchmarks_sources = tnl-benchmarks.h

#if BUILD_CUDA
matrix_formats_test_sources += matrix-formats-test-cuda.cu
sparse_matrix_benchmark_sources += sparse-matrix-benchmark-cuda.cu \
                                   cusp-test.h
matrix_solvers_benchmark_sources += matrix-solvers-benchmark-cuda.cu                                   
tnl_benchmarks_sources += tnl-benchmarks-cuda.cu                               
#else
#matrix_formats_test_sources += matrix-formats-test.cpp
#sparse_matrix_benchmark_sources += sparse-matrix-benchmark.cpp
#matrix_solvers_benchmark_sources += matrix-solvers-benchmark.cpp
#tnl_benchmarks_sources += tnl-benchmarks.cpp                               
#endif

matrix_formats_test_SOURCES = $(matrix_formats_test_sources)
matrix_formats_test_LDADD = ../src/libtnl-0.1.la \
                            ../src/core/libtnlcore-0.1.la \
                            ../src/matrix/libtnlmatrix-0.1.la 
                                                                  
sparse_matrix_benchmark_CXXFLAGS = $(OPTIMISECXXFLAGS) $(CUSP_CXXFLAGS)
sparse_matrix_benchmark_SOURCES = $(sparse_matrix_benchmark_sources)
sparse_matrix_benchmark_LDADD = ../src/libtnl-0.1.la \
                                ../src/core/libtnlcore-0.1.la \
                                ../src/matrix/libtnlmatrix-0.1.la 

matrix_solvers_benchmark_CXXFLAGS = $(OPTIMISECXXFLAGS) $(CUSP_CXXFLAGS)
matrix_solvers_benchmark_SOURCES = $(matrix_solvers_benchmark_sources)
matrix_solvers_benchmark_LDADD = ../src/libtnl-0.1.la \
                                 ../src/core/libtnlcore-0.1.la \
                                 ../src/matrix/libtnlmatrix-0.1.la 

tnl_benchmarks_SOURCES = $(tnl_benchmarks_sources)
tnl_benchmarks_LDADD = ../src/libtnl-0.1.la

reorder_csr_SOURCES = ReorderCSR.cpp

if BUILD_DBG
check_PROGRAMS += matrix-formats-test-dbg \
                  matrix-solvers-benchmark-dbg \
                  sparse-matrix-benchmark-dbg
                  
matrix_formats_test_dbg_SOURCES = $(matrix_formats_test_sources)
matrix_formats_test_dbg_CXXFLAGS = $(DBGCXXFLAGS)
matrix_formats_test_dbg_LDADD = ../src/libtnl-dbg-0.1.la \
                                ../src/core/libtnlcore-dbg-0.1.la \
                                ../src/matrix/libtnlmatrix-dbg-0.1.la

sparse_matrix_benchmark_dbg_SOURCES = $(sparse_matrix_benchmark_sources)
sparse_matrix_benchmark_dbg_CXXFLAGS = $(DBGCXXFLAGS) $(CUSP_CXXFLAGS)                               
sparse_matrix_benchmark_dbg_LDADD = ../src/libtnl-0.1.la \
                                    ../src/core/libtnlcore-0.1.la \
                                    ../src/matrix/libtnlmatrix-0.1.la                                  
                  
matrix_solvers_benchmark_dbg_SOURCES = $(matrix_solvers_benchmark_sources)
matrix_solvers_benchmark_dbg_CXXFLAGS = $(DBGCXXFLAGS) $(CUSP_CXXFLAGS)                               
matrix_solvers_benchmark_dbg_LDADD = ../src/libtnl-0.1.la \
                                     ../src/core/libtnlcore-0.1.la \
                                     ../src/matrix/libtnlmatrix-0.1.la                                  
                  
endif

#if BUILD_CUDA
matrix_formats_test_CXXFLAGS = "-DHAVE_CUDA"
sparse_matrix_benchmark_CXXFLAGS += "-DHAVE_CUDA"
tnl_benchmarks_CXXFLAGS = "-DHAVE_CUDA"
#endif


check_SCRIPTS = get-matrices \
                convert-matrices \
                run-matrix-formats-test \
                run-sparse-matrix-benchmark \
                tnl-sparse-matrix-test 

check_DATA = tnl-sparse-matrix-check.cfg.desc \
             sparse-matrix-benchmark.cfg.desc \
             matrix-solvers-benchmark.cfg.desc \
             florida-matrix-market \
             matrix-market    
             
EXTRADIST = get-matrices \
            convert-matrices \
            run-matrix-formats-test \
            run-sparse-matrix-benchmark \
            tnl-sparse-matrix-test 

                    

#TESTS = get-matrices