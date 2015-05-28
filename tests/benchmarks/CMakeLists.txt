ADD_SUBDIRECTORY( share )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-cuda-benchmarks${debugExt} tnl-cuda-benchmarks.cu
                         OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
    SET_TARGET_PROPERTIES( tnl-cuda-benchmarks${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    TARGET_LINK_LIBRARIES( tnl-cuda-benchmarks${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )                        
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cu
                         OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cu
                         OPTIONS ${CUDA_ADD_EXECUTABLE_OPTIONS} )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    SET_TARGET_PROPERTIES( tnl-benchmark-linear-solvers${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )                        
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cpp )
    ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cpp )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    SET_TARGET_PROPERTIES( tnl-benchmark-linear-solvers${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF()

TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers${debugExt} tnl${debugExt}-${tnlVersion} )

if( BUILD_CUDA )                                                              
   INSTALL( TARGETS tnl-cuda-benchmarks${debugExt}
            RUNTIME DESTINATION bin )
endif()

INSTALL( TARGETS tnl-benchmark-spmv${debugExt}
                 tnl-benchmark-linear-solvers${debugExt}
         RUNTIME DESTINATION bin )


                                            