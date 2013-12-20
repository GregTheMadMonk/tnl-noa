ADD_SUBDIRECTORY( share )

SET( tnlSpmvBenchmark_headers sparse-matrix-benchmark.h             
                              tnlSpmvBenchmark.h
                              tnlSpmvBenchmarkAdaptiveRgCSRMatrix.h
                              tnlSpmvBenchmarkCSRMatrix.h
                              tnlSpmvBenchmarkCusparseMarix.h
                              tnlSpmvBenchmarkHybridMatrix.h
                              tnlSpmvBenchmarkRgCSRMatrix.h )
     

IF( BUILD_CUDA )
    #CUDA_ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cu )
    #SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES CUDA_COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )        
ELSE()
    #ADD_EXECUTABLE( tnl-sparse-matrix-benchmark${debugExt} sparse-matrix-benchmark.cpp )
    #SET_TARGET_PROPERTIES( tnl-sparse-matrix-benchmark${debugExt} PROPERTIES COMPILE_FLAGS "${CXX_OPTIMIZE_FLAGS}" )
ENDIF()
#TARGET_LINK_LIBRARIES( tnl-sparse-matrix-benchmark${debugExt} tnl${debugExt}-${tnlVersion}
#                                                              ${CUSPARSE_LIBRARY} )

#INSTALL( TARGETS tnl-sparse-matrix-benchmark${debugExt}
#         RUNTIME DESTINATION bin )


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