ADD_SUBDIRECTORY( share )
ADD_SUBDIRECTORY( heat-equation-benchmark )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-cuda-benchmarks${debugExt} tnl-cuda-benchmarks.cu )
    if( HAVE_CUBLAS STREQUAL "yes" )
        CUDA_ADD_CUBLAS_TO_TARGET( tnl-cuda-benchmarks${debugExt} )
    endif()
    TARGET_LINK_LIBRARIES( tnl-cuda-benchmarks${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )                        
    
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )                        
    
    CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cu )
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers${debugExt} tnl${debugExt}-${tnlVersion} ${CUSPARSE_LIBRARY} )                        
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cpp )
    TARGET_LINK_LIBRARIES( tnl-benchmark-spmv${debugExt} tnl${debugExt}-${tnlVersion} )

    ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cpp )    
    TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers${debugExt} tnl${debugExt}-${tnlVersion} )
ENDIF()

if( BUILD_CUDA )                                                              
   INSTALL( TARGETS
                tnl-cuda-benchmarks${debugExt}
            RUNTIME DESTINATION bin )
endif()

INSTALL( TARGETS 
            tnl-benchmark-spmv${debugExt}
            tnl-benchmark-linear-solvers${debugExt}                 
         RUNTIME DESTINATION bin )


                                            
