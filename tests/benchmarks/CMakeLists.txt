ADD_SUBDIRECTORY( share )

SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu )
    TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )    
ELSE()
    ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
    TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF()

INSTALL( TARGETS tnl-sparse-matrix-benchmark${debugExt}
         RUNTIME DESTINATION bin )