ADD_SUBDIRECTORY( unit-tests )

FILE( COPY matrix-market DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )
file( COPY florida-matrix-market DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )

file( COPY get-matrices DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
file( COPY convert-matrices DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
file( COPY draw-matrices DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
file( COPY run-sparse-matrix-benchmark DESTINATION ${EXECUTABLE_OUTPUT_PATH} )
file( COPY run-matrix-solvers-benchmark DESTINATION ${EXECUTABLE_OUTPUT_PATH} )

file( COPY process-cuda-profile.pl DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )

file( COPY sparse-matrix-benchmark.cfg.desc DESTINATION ${EXECUTABLE_OUTPUT_PATH}/config )

set( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

#sparse_matrix_benchmark_CXXFLAGS = $(OPTIMISECXXFLAGS)

if( BUILD_CUDA )
cuda_add_executable( sparse-matrix-benchmark${mpiExt}${debugExt} sparse-matrix-benchmark.cu )
target_link_libraries( sparse-matrix-benchmark${mpiExt}${debugExt} tnl${mpiExt}${debugExt}-0.1
                                                                   libtnlcore${mpiExt}${debugExt}-0.1
                                                                   libtnlconfig${mpiExt}${debugExt}-0.1 )
else()
add_executable( sparse-matrix-benchmark${mpiExt}${debugExt} sparse-matrix-benchmark.cpp )
target_link_libraries( sparse-matrix-benchmark${mpiExt}${debugExt} tnl${mpiExt}${debugExt}-0.1
                                                                   libtnlcore${mpiExt}${debugExt}-0.1
                                                                   libtnlconfig${mpiExt}${debugExt}-0.1 )
endif()

                                                                   