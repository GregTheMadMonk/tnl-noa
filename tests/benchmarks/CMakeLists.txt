ADD_SUBDIRECTORY( share )

SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu )
    TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${mpiExt}${debugExt}-0.1
                                                                  tnlcore${mpiExt}${debugExt}-0.1
                                                                  tnlconfig${mpiExt}${debugExt}-0.1 )
    
ELSE()
    ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
    TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${mpiExt}${debugExt}-0.1
                                                                  tnlcore${mpiExt}${debugExt}-0.1
                                                                  tnlconfig${mpiExt}${debugExt}-0.1 )
ENDIF()

INSTALL( TARGETS tnl-sparse-matrix-benchmark${debugExt}
         RUNTIME DESTINATION bin )