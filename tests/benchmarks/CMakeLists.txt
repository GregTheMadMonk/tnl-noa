ADD_SUBDIRECTORY( share )

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cu
                         OPTIONS -arch sm_20 -shared )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    CUDA_ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cu
                         OPTIONS -arch sm_20 -shared )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    SET_TARGET_PROPERTIES( tnl-benchmark-linear-solvers${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )                    
ELSE()
    ADD_EXECUTABLE( tnl-benchmark-spmv${debugExt} tnl-benchmark-spmv.cpp )
    ADD_EXECUTABLE( tnl-benchmark-linear-solvers${debugExt} tnl-benchmark-linear-solvers.cpp )
    SET_TARGET_PROPERTIES( tnl-benchmark-spmv${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
    SET_TARGET_PROPERTIES( tnl-benchmark-linear-solvers${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
ENDIF()
TARGET_LINK_LIBRARIES( tnl-benchmark-spmv${debugExt} tnl${debugExt}-${tnlVersion} )
TARGET_LINK_LIBRARIES( tnl-benchmark-linear-solvers${debugExt} tnl${debugExt}-${tnlVersion} )
                                                              
INSTALL( TARGETS tnl-benchmark-spmv${debugExt}
                 tnl-benchmark-linear-solvers${debugExt}
         RUNTIME DESTINATION bin )





####################################################
SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkCusparseMarix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu
                         OPTIONS -arch sm_20 -shared )
    SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )        
ELSE()
    ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
    SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
ENDIF()
TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion}
                                                              ${CUSPARSE_LIBRARY} )



#IF( BUILD_CUDA )
#   CUDA_ADD_EXECUTABLE( tnl-matrix-solvers-benchmark${debugExt} matrix-solvers-benchmark.cu )        
#ELSE()
#   ADD_EXECUTABLE( tnl-matrix-solvers-benchmark${debugExt} matrix-solvers-benchmark.cpp )
#ENDIF()
#TARGET_LINK_LIBRARIES( tnl-matrix-solvers-benchmark${debugExt} tnl${debugExt}-${tnlVersion} )                                                                   
#INSTALL( TARGETS tnl-matrix-solvers-benchmark${debugExt}
#         RUNTIME DESTINATION bin )
         
#if( BUILD_MPI )
#  IF( BUILD_CUDA )
#       CUDA_ADD_EXECUTABLE( tnl-matrix-solvers-benchmark-mpi${debugExt} matrix-solvers-benchmark.cu )
#       SET_TARGET_PROPERTIES( tnl-matrix-solvers-benchmark-mpi${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${PETSC_CXX_FLAGS}" )        
#   ELSE()
#       ADD_EXECUTABLE( tnl-matrix-solvers-benchmark-mpi${debugExt} matrix-solvers-benchmark.cpp )
#       SET_TARGET_PROPERTIES( tnl-matrix-solvers-benchmark-mpi${debugExt} PROPERTIES COMPILE_FLAGS "${PETSC_CXX_FLAGS}" )
#   ENDIF()
#   TARGET_LINK_LIBRARIES( tnl-matrix-solvers-benchmark-mpi${debugExt} tnl-mpi${debugExt}-${tnlVersion}
#                                                                      ${MPI_LIBRARIES}
#                                                                      ${PETSC_LIBRARY} )
#   INSTALL( TARGETS tnl-matrix-solvers-benchmark-mpi${debugExt}
#            RUNTIME DESTINATION bin )
#endif()                                                                  