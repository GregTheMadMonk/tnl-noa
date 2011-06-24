ADD_SUBDIRECTORY( share )

SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu )        
ELSE()
    ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
ENDIF()
TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )
SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )

INSTALL( TARGETS tnl-sparse-matrix-benchmark${debugExt}
         RUNTIME DESTINATION bin )