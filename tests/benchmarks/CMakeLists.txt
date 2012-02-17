ADD_SUBDIRECTORY( share )

SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkCusparseMarix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu )
    #SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )        
ELSE()
    ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
    #SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
ENDIF()
TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )


INSTALL( TARGETS tnl-sparse-matrix-benchmark${debugExt}
         RUNTIME DESTINATION bin )